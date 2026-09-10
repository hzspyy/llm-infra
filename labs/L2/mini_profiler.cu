// L2.6 lab · 没有 ncu 的时候，自己造一把尺子。
//
// 本机的容器读不到 SM 硬件性能计数器（见 profile_ladder.sh 的权限诊断），
// 所以 ncu 用不了。但 ncu 报告里最重要的三个数字，不用特权也能自己测出来：
//
//   Theoretical Occupancy  —— 驱动直接告诉你（cudaOccupancyMaxActiveBlocks...）
//   Achieved Occupancy     —— 每个 block 记下自己在哪个 SM、什么时候开始结束，
//                             回头重建每个 SM 上的并发曲线。这**就是**它的定义。
//   Tail / 负载不均         —— 同一份数据顺手就有
//
// 关键是两个特殊寄存器：
//   %smid        当前 block 跑在哪个 SM 上
//   %globaltimer 全局纳秒计时器。注意**不能用 clock64()**：那是每个 SM
//                自己的周期计数器，跨 SM 比较没有意义。
//
// 编译：nvcc -O3 -arch=sm_XX -o mini_profiler mini_profiler.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

struct BlockRec {
    unsigned smid;
    unsigned block;
    unsigned long long t_begin;   // ns，全局时钟
    unsigned long long t_end;
};

__device__ __forceinline__ unsigned sm_id() {
    unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r;
}

// 探针宏：只有 0 号线程记录，开销一次两条指令 + 一次 16 字节写。
#define PROBE_BEGIN() \
    unsigned long long _t0 = gtimer(); unsigned _sm = sm_id();
#define PROBE_END(rec) \
    if (threadIdx.x == 0) { \
        BlockRec r; r.smid = _sm; r.block = blockIdx.x; \
        r.t_begin = _t0; r.t_end = gtimer(); (rec)[blockIdx.x] = r; }

// --------------------------------------------------------------------------
// 三个被测 kernel：负载均衡的 / 负载不均的 / 共享内存吃满占用率的
// --------------------------------------------------------------------------
__global__ void k_balanced(float* __restrict__ p, int iters, BlockRec* rec) {
    PROBE_BEGIN();
    float acc = p[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, 1e-7f);
    if (acc == 12345.678f) p[0] = acc;          // 防止被优化掉
    PROBE_END(rec);
}

// 前 1/8 的 block 干 8 倍的活 —— 典型的"长尾"形状
__global__ void k_skewed(float* __restrict__ p, int iters, BlockRec* rec) {
    PROBE_BEGIN();
    int n = (blockIdx.x < gridDim.x / 8) ? iters * 8 : iters;
    float acc = p[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < n; ++i) acc = fmaf(acc, 1.0000001f, 1e-7f);
    if (acc == 12345.678f) p[0] = acc;
    PROBE_END(rec);
}

// 静态申请 32 KiB 共享内存 —— 每个 SM 最多只能放下 100KiB/32KiB = 3 个 block
__global__ void k_smem_heavy(float* __restrict__ p, int iters, BlockRec* rec) {
    __shared__ float buf[8192];
    PROBE_BEGIN();
    buf[threadIdx.x] = p[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    float acc = buf[threadIdx.x];
    for (int i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, 1e-7f);
    if (acc == 12345.678f) p[0] = acc;
    PROBE_END(rec);
}

// --------------------------------------------------------------------------
// 分析：从 BlockRec 重建每个 SM 上的并发曲线
// --------------------------------------------------------------------------
struct Analysis {
    double achieved_blocks_per_sm;   // 时间加权平均并发 block 数
    int    max_blocks_per_sm;        // 实际观测到的峰值
    int    sms_used;
    double span_ms;
    double tail_frac;                // 最后 10% 时间里并发度掉到多少
    double sm_imbalance;             // 最忙 SM / 最闲 SM 的忙碌时间比
};

static Analysis analyze(std::vector<BlockRec>& rec) {
    Analysis a{};
    if (rec.empty()) return a;

    unsigned long long t0 = ~0ull, t1 = 0;
    unsigned max_sm = 0;
    for (auto& r : rec) {
        t0 = std::min(t0, r.t_begin);
        t1 = std::max(t1, r.t_end);
        max_sm = std::max(max_sm, r.smid);
    }
    a.span_ms = double(t1 - t0) / 1e6;

    // 把每个 block 拆成 (时刻, ±1) 事件，按 SM 分组扫描
    std::vector<std::vector<std::pair<unsigned long long,int>>> ev(max_sm + 1);
    std::vector<unsigned long long> busy(max_sm + 1, 0);
    for (auto& r : rec) {
        ev[r.smid].push_back({r.t_begin, +1});
        ev[r.smid].push_back({r.t_end,   -1});
        busy[r.smid] += r.t_end - r.t_begin;
    }

    double area = 0;                 // ∫ 并发数 dt，跨所有 SM 累加
    int used = 0;
    for (unsigned s = 0; s <= max_sm; ++s) {
        if (ev[s].empty()) continue;
        ++used;
        std::sort(ev[s].begin(), ev[s].end());
        int cur = 0;
        unsigned long long prev = ev[s][0].first;
        for (auto& e : ev[s]) {
            area += double(cur) * double(e.first - prev);
            a.max_blocks_per_sm = std::max(a.max_blocks_per_sm, cur);
            prev = e.first;
            cur += e.second;
        }
    }
    a.sms_used = used;
    a.achieved_blocks_per_sm = area / double(t1 - t0) / used;

    // 尾巴：最后 10% 的时间窗里，平均并发度是整体的几成
    unsigned long long tail_start = t1 - (t1 - t0) / 10;
    double tail_area = 0;
    for (auto& r : rec) {
        unsigned long long s = std::max(r.t_begin, tail_start);
        unsigned long long e = std::max(r.t_end,   tail_start);
        if (e > s) tail_area += double(e - s);
    }
    double tail_avg = tail_area / double(t1 - tail_start) / used;
    a.tail_frac = tail_avg / a.achieved_blocks_per_sm;

    unsigned long long bmax = 0, bmin = ~0ull;
    for (unsigned s = 0; s <= max_sm; ++s) {
        if (ev[s].empty()) continue;
        bmax = std::max(bmax, busy[s]);
        bmin = std::min(bmin, busy[s]);
    }
    a.sm_imbalance = bmin ? double(bmax) / double(bmin) : 0.0;
    return a;
}

template <typename K>
static void run_one(const char* name, K kernel, int grid, int block, int iters,
                    int n_sm, int warps_per_sm_max) {
    BlockRec* d_rec;
    CK(cudaMalloc(&d_rec, sizeof(BlockRec) * grid));
    CK(cudaMemset(d_rec, 0, sizeof(BlockRec) * grid));
    float* d_p;
    CK(cudaMalloc(&d_p, sizeof(float) * size_t(grid) * block));
    CK(cudaMemset(d_p, 0, sizeof(float) * size_t(grid) * block));

    // 驱动给出的理论占用率：不需要任何特权，直接查
    int max_blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block, 0));
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, kernel));

    kernel<<<grid, block>>>(d_p, iters, d_rec);        // 预热
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    kernel<<<grid, block>>>(d_p, iters, d_rec);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

    std::vector<BlockRec> rec(grid);
    CK(cudaMemcpy(rec.data(), d_rec, sizeof(BlockRec) * grid, cudaMemcpyDeviceToHost));
    Analysis a = analyze(rec);

    double theo_occ = double(max_blocks * block / 32) / warps_per_sm_max;
    double ach_occ  = a.achieved_blocks_per_sm * (block / 32) / warps_per_sm_max;

    printf("  %-14s grid=%-5d block=%-4d 寄存器/线程=%-3d smem/block=%zuB\n",
           name, grid, block, attr.numRegs, attr.sharedSizeBytes);
    printf("    %-26s %d  (occupancy %.0f%%)\n",
           "理论 blocks/SM", max_blocks, theo_occ * 100);
    printf("    %-26s %.2f  (occupancy %.0f%%)   峰值观测 %d\n",
           "实测 blocks/SM(时间加权)", a.achieved_blocks_per_sm, ach_occ * 100,
           a.max_blocks_per_sm);
    printf("    %-26s %d / %d\n", "用到的 SM", a.sms_used, n_sm);
    printf("    %-26s %.3f ms (event) / %.3f ms (globaltimer 跨度)\n",
           "耗时", ms, a.span_ms);
    printf("    %-26s %.2f   %s\n", "SM 忙碌时间 最大/最小", a.sm_imbalance,
           a.sm_imbalance > 1.5 ? "← 明显不均" : "");
    printf("    %-26s %.0f%%  %s\n", "末 10% 时间的并发度", a.tail_frac * 100,
           a.tail_frac < 0.5 ? "← 长尾，SM 大半空转" : "");
    printf("\n");

    CK(cudaFree(d_rec)); CK(cudaFree(d_p));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("=== %s  sm_%d%d  %d SM  每SM最多 %d 线程(%d warp)  smem/SM %zu KiB\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerMultiProcessor / 32,
           prop.sharedMemPerMultiprocessor / 1024);

    const int n_sm = prop.multiProcessorCount;
    const int wmax = prop.maxThreadsPerMultiProcessor / 32;

    printf("[1] 负载均衡：实测占用率应当贴近理论值\n");
    run_one("balanced", k_balanced, n_sm * 8, 256, 20000, n_sm, wmax);

    printf("[2] 负载倾斜：1/8 的 block 干 8 倍的活\n");
    run_one("skewed", k_skewed, n_sm * 8, 256, 20000, n_sm, wmax);

    printf("[3] 共享内存吃满：32 KiB/block 把 blocks/SM 压下去\n");
    run_one("smem_heavy", k_smem_heavy, n_sm * 8, 256, 20000, n_sm, wmax);

    printf("[4] 尾效应：grid 刚好比一个整波多一个 block\n");
    int per_wave = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_wave, k_balanced, 256, 0));
    int wave = per_wave * n_sm;
    printf("    一个满波 = %d blocks/SM × %d SM = %d blocks\n", per_wave, n_sm, wave);
    run_one("wave", k_balanced, wave, 256, 20000, n_sm, wmax);
    run_one("wave+1", k_balanced, wave + 1, 256, 20000, n_sm, wmax);
    return 0;
}
