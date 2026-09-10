// L1.1 lab · 把 GPU 的内存层次一层一层量出来。
//
// 厂商只告诉你「L2 96 MB、显存 1792 GB/s」。但真正决定 kernel 快慢的是：
//   - 每一级的**延迟**是多少周期？（决定你需要多少并发去掩盖它）
//   - 每一级的**带宽**是多少？（决定 roofline 的斜率）
//   - 工作集多大时会掉出某一级？（决定你的 tile 该开多大）
// 这三个数字没有一个能从规格表读到，必须测。
//
// 四个实验：
//   A. 指针追逐：测各级的**延迟**（依赖链，无法并行，暴露纯延迟）
//   B. 带宽 vs 工作集：扫 buffer 大小，看 L2 → 显存 的悬崖在哪
//   C. 共享内存延迟与 bank conflict
//   D. 并发度 vs 带宽：需要多少 warp 才能把带宽打满（Little's Law）
//
// 编译（注意 -arch 要匹配你的卡）：
//   nvcc -O3 -arch=sm_120 -o mem_hierarchy mem_hierarchy.cu
//   ./mem_hierarchy

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------------------
// A. 指针追逐 —— 测延迟
//
// 核心思想：让每次访存**依赖上一次的结果**（idx = buf[idx]），
// 这样硬件无法把它们并行发射，测到的就是纯粹的往返延迟。
// 如果用独立的 load，硬件会同时发出几十个请求，你测到的是带宽而不是延迟。
// ---------------------------------------------------------------------------
__global__ void chase_kernel(const uint32_t* __restrict__ buf, int steps,
                             uint32_t* out, long long* cycles) {
    uint32_t idx = 0;
    // 预热：把这条链走一遍，让 TLB 和目标层级都热起来
    for (int i = 0; i < steps; ++i) idx = buf[idx];

    long long t0 = clock64();
    for (int i = 0; i < steps; ++i) idx = buf[idx];
    long long t1 = clock64();

    *out = idx;            // 阻止编译器把整个循环优化掉
    *cycles = t1 - t0;
}

// 返回 {周期/次, 有效SM时钟GHz}。同时用 clock64() 与 CUDA event 计时，
// 两者相除就得到**本次运行的真实 SM 频率**——比读 cudaDevAttrClockRate（基频）准，
// 因为 GPU 会 boost，而 clock64() 数的是实际时钟周期。
struct ChaseResult { double cycles_per_access; double effective_ghz; };

// ★ 方法学要点（第一版在这里错过一次，值得记住）：
//
// 指针链必须**走遍整个工作集**，测到的才是这个工作集大小对应的层级延迟。
// 第一版对所有 buffer 都只走 4096 步：对 1 GiB 的 buffer 而言，
// 4096 次随机访问只碰到 4096 条 cache line ≈ 512 KB，**全程留在 L2 里**，
// 于是 1 MiB 到 1 GiB 全都测出 360 周期，L2 与显存的差别凭空消失。
//
// 正确做法：
//   1. 链上每个节点占**一整条 cache line**（128 B），这样 "节点数" = "line 数"；
//   2. 步数取 line 数的若干倍，保证重用距离 = 整个 buffer；
//   3. 随机顺序，打掉硬件预取器。
static ChaseResult pointer_chase2(size_t bytes, int min_steps = 4096) {
    const size_t LINE = 128;                       // NVIDIA 的 L2 line 是 128 B
    size_t n_lines = bytes / LINE;
    if (n_lines < 2) return {-1, 0};
    size_t n = n_lines * (LINE / sizeof(uint32_t)); // 实际分配的 uint32 个数

    // 节点 i 的地址 = i * LINE，链在 line 粒度上随机
    std::vector<uint32_t> order(n_lines);
    for (size_t i = 0; i < n_lines; ++i) order[i] = (uint32_t)i;
    std::mt19937 rng(12345);
    std::shuffle(order.begin() + 1, order.end(), rng);

    std::vector<uint32_t> host(n, 0);
    const size_t STEP = LINE / sizeof(uint32_t);    // 32 个 uint32 一条 line
    for (size_t i = 0; i + 1 < n_lines; ++i)
        host[(size_t)order[i] * STEP] = (uint32_t)(order[i + 1] * STEP);
    host[(size_t)order[n_lines - 1] * STEP] = (uint32_t)(order[0] * STEP);

    // 走 3 遍整个链；小 buffer 至少走 min_steps 步以保证计时精度
    size_t steps = std::max((size_t)min_steps, n_lines * 3);
    steps = std::min(steps, (size_t)40'000'000);    // 封顶，免得 1 GiB 跑太久
    bool truncated = steps < n_lines * 3;

    uint32_t* d_buf; uint32_t* d_out; long long* d_cyc;
    CHECK(cudaMalloc(&d_buf, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_cyc, sizeof(long long)));
    CHECK(cudaMemcpy(d_buf, host.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cudaEvent_t ea, eb; CHECK(cudaEventCreate(&ea)); CHECK(cudaEventCreate(&eb));
    CHECK(cudaEventRecord(ea));
    chase_kernel<<<1, 1>>>(d_buf, (int)std::min(steps, (size_t)INT32_MAX), d_out, d_cyc);
    CHECK(cudaEventRecord(eb));
    CHECK(cudaEventSynchronize(eb));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, ea, eb));

    long long cyc = 0;
    CHECK(cudaMemcpy(&cyc, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_buf)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cyc));
    CHECK(cudaEventDestroy(ea)); CHECK(cudaEventDestroy(eb));
    if (truncated) printf("      [注意] %zu MiB 的链未走满 3 遍，延迟可能偏乐观\n", bytes >> 20);
    // kernel 内跑了两遍（预热 + 计时），event 覆盖整个 kernel
    double ghz = (double)(cyc * 2) / (ms * 1e6);
    return {(double)cyc / (double)steps, ghz};
}

// ---------------------------------------------------------------------------
// B. 带宽 vs 工作集
//
// 用足够多的线程把 buffer 反复读一遍。buffer 小的时候整个落在 L2 里，
// 测到的是 L2 带宽；大到超过 L2 就掉进显存。悬崖的位置 = L2 的有效容量。
// ---------------------------------------------------------------------------
__global__ void bw_read_kernel(const float4* __restrict__ buf, size_t n4,
                               int iters, float* sink) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int it = 0; it < iters; ++it) {
        for (size_t i = tid; i < n4; i += stride) {
            float4 v = buf[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (acc.x == 1234.5678f) *sink = acc.x + acc.y + acc.z + acc.w;  // 防优化
}

static double read_bandwidth(size_t bytes, int sm_count) {
    size_t n4 = bytes / sizeof(float4);
    float4* d_buf; float* d_sink;
    CHECK(cudaMalloc(&d_buf, n4 * sizeof(float4)));
    CHECK(cudaMemset(d_buf, 1, n4 * sizeof(float4)));
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    int block = 256;
    int grid = sm_count * 8;                       // 每 SM 8 个 block，占满
    // 小 buffer 要多迭代几次才够长，才测得准
    int iters = (int)std::max<size_t>(1, (size_t)(512ull * 1024 * 1024) / bytes);
    iters = std::min(iters, 2000);

    bw_read_kernel<<<grid, block>>>(d_buf, n4, 2, d_sink);   // 预热
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    bw_read_kernel<<<grid, block>>>(d_buf, n4, iters, d_sink);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));

    CHECK(cudaFree(d_buf)); CHECK(cudaFree(d_sink));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return (double)bytes * iters / (ms * 1e-3) / 1e9;          // GB/s
}

// ---------------------------------------------------------------------------
// C. 共享内存：延迟与 bank conflict
//
// shared memory 分成 32 个 bank，每 bank 4 字节宽。
// 同一个 warp 里的 32 个线程如果访问**不同** bank，一拍全部完成；
// 如果都落在同一个 bank（stride = 32 个 float），就要串行 32 次。
// ---------------------------------------------------------------------------
// 关键：必须用**独立**的访存。第一版写成了依赖链（idx = s[idx]），
// 结果 stride=1 与 stride=32 都是 44 周期，冲突倍数 1.0×——什么都没测到。
// 原因：bank conflict 损失的是**吞吐**，而依赖链本来就一次只有一个在途请求，
// 吞吐根本不是瓶颈。改成每线程连发 8 个互不依赖的 load，冲突才会显现。
template <int STRIDE>
__global__ void smem_kernel(int steps, uint32_t* out, long long* cycles) {
    __shared__ uint32_t s[4096];
    int t = threadIdx.x;
    for (int i = t; i < 4096; i += blockDim.x) s[i] = i;
    __syncthreads();

    uint32_t base = (uint32_t)(t * STRIDE) & 4095u;
    uint32_t acc = 0;
    // 预热
    for (int i = 0; i < 64; ++i) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += s[(base + k * 512u + i) & 4095u];
    }
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += s[(base + k * 512u + i) & 4095u];
    }
    long long t1 = clock64();

    if (t == 0) { *out = acc; *cycles = t1 - t0; }
}

// ---------------------------------------------------------------------------
// D. 并发度 vs 带宽（Little's Law）
//
// 达到的带宽 = 并发在途请求数 × 每请求字节 / 延迟。
// 所以想打满带宽，需要足够多的 warp 同时在等访存。
// 这个实验扫「每 SM 多少个 block」，看带宽什么时候饱和。
// ---------------------------------------------------------------------------
static void concurrency_sweep(int sm_count, double dram_ref_gbs) {
    // 第一版用 512 MB + iters=4，在 32 blocks/SM 时报出 3752 GB/s ——
    // **超过了硬件理论带宽**，而且和实验 B 在同样 512 MB 下测到的 1632 GB/s 矛盾。
    // 一个数字超过物理上界时，一定是测法错了，不是硬件变快了。
    // 根因：多遍扫描 + 大 grid 时，后几遍能从 L2 捞到相当一部分数据。
    // 改法：工作集放大到 1 GB（>10 倍 L2），并且**只扫一遍**。
    const size_t bytes = 1024ull * 1024 * 1024;
    size_t n4 = bytes / sizeof(float4);
    float4* d_buf; float* d_sink;
    CHECK(cudaMalloc(&d_buf, n4 * sizeof(float4)));
    CHECK(cudaMemset(d_buf, 1, n4 * sizeof(float4)));
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    printf("\n[D] 并发度 vs 带宽（1 GB 工作集，单遍扫描，block=256）\n");
    printf("    %-11s %-12s %-12s %-12s %s\n",
           "blocks/SM", "常驻 warp", "带宽 GB/s", "占已测峰值", "Little's Law");
    std::vector<std::pair<int, double>> rows;
    for (int bps : {1, 2, 4, 8, 16}) {
        int grid = sm_count * bps, block = 256;
        bw_read_kernel<<<grid, block>>>(d_buf, n4, 1, d_sink);      // 预热（也把 L2 冲掉）
        CHECK(cudaDeviceSynchronize());
        // 取三次的最小耗时（= 最大带宽），避免偶发抖动
        float best = 1e30f;
        for (int r = 0; r < 3; ++r) {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            bw_read_kernel<<<grid, block>>>(d_buf, n4, 1, d_sink);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, ms);
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        rows.push_back({bps, (double)bytes / (best * 1e-3) / 1e9});
    }
    for (auto& r : rows) {
        // 每 SM 最多常驻的 warp 受硬件上限约束
        int warps = std::min(r.first * 8, 48);
        char pct[16]; snprintf(pct, sizeof pct, "%.0f%%", r.second / dram_ref_gbs * 100);
        printf("    %-11d %-12d %-12.1f %-12s %s\n",
               r.first, warps * sm_count, r.second, pct,
               r.second / dram_ref_gbs > 0.9 ? "已饱和" : "并发不足，延迟没被掩盖");
    }
    printf("    参照：实验 B 在 1 GB 工作集下测得 %.0f GB/s\n", dram_ref_gbs);
    CHECK(cudaFree(d_buf)); CHECK(cudaFree(d_sink));
}

// ---------------------------------------------------------------------------

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int clk_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    double ghz = clk_khz / 1e6;

    printf("=== %s  sm_%d%d  SM=%d  L2=%.1f MiB  显存=%.1f GB  SM时钟=%.2f GHz\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.l2CacheSize / 1048576.0, p.totalGlobalMem / 1e9, ghz);
    printf("    每 SM 最大线程 %d，每 block 最大共享内存 %zu KB，warp 大小 %d\n",
           p.maxThreadsPerMultiProcessor, p.sharedMemPerBlockOptin / 1024, p.warpSize);

    // ---- A ----
    // 先用一次小工作集的追逐把**有效 SM 频率**标定出来。
    // cudaDevAttrClockRate 报的是基频，而 GPU 实际会 boost，
    // 用基频换算 ns 会系统性偏大。
    // 标定必须用**足够长**的 kernel：短 kernel 里 launch 开销（几微秒）
    // 会被算进 event 时间，把频率系统性压低。这里走 200 万步 L1 命中（约 0.1 秒）。
    double eff_ghz = pointer_chase2(64ull << 10, 2'000'000).effective_ghz;
    printf("    标定：基频 %.2f GHz，本次运行有效 %.2f GHz（clock64 与 event 交叉反推）\n",
           ghz, eff_ghz);
    printf("    注意：单线程追逐几乎不给 GPU 负载，实际频率低于满载 boost；\n");
    printf("          周期数是硬事实，纳秒列只在同一次运行内可比。\n");

    printf("\n[A] 指针追逐延迟（随机环，单线程，无法并行 ⇒ 纯延迟）\n");
    printf("    %-12s %-12s %-12s %s\n", "工作集", "周期/次", "纳秒/次", "推测命中层级");
    size_t sizes[] = {
        4ull<<10, 16ull<<10, 32ull<<10, 64ull<<10, 128ull<<10, 256ull<<10,
        512ull<<10, 1ull<<20, 4ull<<20, 16ull<<20, 32ull<<20, 64ull<<20,
        96ull<<20, 128ull<<20, 256ull<<20, 512ull<<20};
    for (size_t s : sizes) {
        if (s > p.totalGlobalMem / 3) break;
        double cyc = pointer_chase2(s).cycles_per_access;
        char buf[32];
        if (s >= (1ull<<20)) snprintf(buf, sizeof buf, "%zu MiB", s>>20);
        else snprintf(buf, sizeof buf, "%zu KiB", s>>10);
        // 层级判定用**实测拐点**，不用厂商标称：
        //   <= 128 KiB  在 L1（每 SM 128 KiB 统一 L1/smem）
        //   <= L2 容量   在 L2
        //   再往上       进显存
        const char* lvl = s <= (128ull << 10) ? "L1"
                        : (s < (size_t)p.l2CacheSize ? "L2"
                        : (s <= (size_t)p.l2CacheSize ? "L2（已到容量边界）" : "显存"));
        printf("    %-12s %-12.1f %-12.2f %s\n", buf, cyc, cyc / eff_ghz, lvl);
    }

    // ---- B ----
    // 注意：小工作集时数据会留在**各 SM 自己的 L1** 里，测到的是 L1 聚合带宽，
    // 不是 L2。全卡 L1 总量 = SM 数 × 每 SM L1 容量，先算出来当分界参考。
    double l1_total_mb = p.multiProcessorCount * 128.0 / 1024.0;   // Blackwell 每 SM 128 KB 统一 L1/smem
    printf("\n[B] 只读带宽 vs 工作集\n");
    printf("    参考边界：L1 聚合约 %.0f MiB（%d SM × 128 KiB），L2 = %.0f MiB\n",
           l1_total_mb, p.multiProcessorCount, p.l2CacheSize / 1048576.0);
    printf("    %-12s %-14s %s\n", "工作集", "带宽 GB/s", "预期主要命中");
    for (size_t s : {1ull<<20, 4ull<<20, 16ull<<20, 32ull<<20, 64ull<<20,
                     80ull<<20, 96ull<<20, 128ull<<20, 192ull<<20,
                     256ull<<20, 512ull<<20, 1024ull<<20}) {
        if (s > p.totalGlobalMem / 3) break;
        double gbs = read_bandwidth(s, p.multiProcessorCount);
        char buf[32]; snprintf(buf, sizeof buf, "%zu MiB", s>>20);
        double mb = s / 1048576.0;
        const char* lvl = mb <= l1_total_mb ? "L1（各 SM 私有）"
                        : (s <= (size_t)p.l2CacheSize ? "L2" : "显存");
        printf("    %-12s %-14.1f %s\n", buf, gbs, lvl);
    }

    // ---- C ----
    {
        printf("\n[C] 共享内存延迟与 bank conflict\n");
        uint32_t* d_out; long long* d_cyc;
        CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));
        CHECK(cudaMalloc(&d_cyc, sizeof(long long)));
        long long c1 = 0, c32 = 0;
        smem_kernel<1><<<1, 32>>>(4096, d_out, d_cyc);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&c1, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost));
        smem_kernel<32><<<1, 32>>>(4096, d_out, d_cyc);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&c32, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost));
        const double n_acc = 4096.0 * 8;   // steps × 每步 8 个独立 load
        printf("    stride=1  （32 线程分散在 32 个 bank）：%.2f 周期/次访存\n", (double)c1/n_acc);
        printf("    stride=32 （32 线程全撞同一个 bank）  ：%.2f 周期/次访存\n", (double)c32/n_acc);
        printf("    冲突倍数 %.1f×  —— 这就是 L2.3/L2.4 里 swizzle 要消灭的东西\n",
               (double)c32 / (double)c1);
        printf("    （32 路冲突的理论上界就是 32×：一次广播变成 32 次串行事务）\n");
        CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cyc));
    }

    // ---- D ----
    {
        double dram_ref = read_bandwidth(1024ull << 20, p.multiProcessorCount);
        concurrency_sweep(p.multiProcessorCount, dram_ref);
    }

    printf("\n");
    return 0;
}
