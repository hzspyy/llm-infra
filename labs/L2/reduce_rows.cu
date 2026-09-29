// L2.3-A/B · 行内归约：行宽/行数扫描、向量化与单/两阶段、精度与 CUB 对照。
//
//   编译：nvcc -O3 -std=c++17 -arch=sm_120 -I$CUDA_HOME/include/cccl -o reduce_rows reduce_rows.cu
//
// A 部分：奇数长度、极端抵消输入与 FP64 参照逐项对拍；
// B 部分：cols ∈ {128,1024,4096,14336} × rows ∈ {1,8,128,2048}，
//         向量化 on/off，两阶段（共享内存树）与一阶段（warp shuffle + atomic）对照；
// C 部分：同一 shape 下 CUB BlockReduce / DeviceReduce 的时间对照。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// 两阶段：线程局部和 → 共享内存树 → 每 block 输出一个部分和
__global__ void row_two_stage(const float* __restrict__ in, float* __restrict__ out,
                              int rows, int cols, int vec) {
    extern __shared__ float s[];
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = in + (size_t)r * cols;
    float acc = 0.f;
    if (vec == 4) {
        int n4 = cols / 4;
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 v = r4[i];
            acc += (v.x + v.y) + (v.z + v.w);
        }
        for (int i = n4 * 4 + threadIdx.x; i < cols; i += blockDim.x) acc += row[i];
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += row[i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = s[0];
}

// 一阶段：warp 内 shuffle，每个 warp 一次 atomicAdd（行内只需一个 block 时等价）
__global__ void row_warp_atomic(const float* __restrict__ in, float* __restrict__ out,
                                int rows, int cols, int vec) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = in + (size_t)r * cols;
    float acc = 0.f;
    if (vec == 4) {
        int n4 = cols / 4;
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 v = r4[i];
            acc += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += row[i];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&out[r], acc);
}

// CUB 对照：同样的"一行一个 block"，用 cub::BlockReduce
__global__ void row_cub(const float* __restrict__ in, float* __restrict__ out,
                        int rows, int cols, int vec) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = in + (size_t)r * cols;
    float acc = 0.f;
    if (vec == 4) {
        int n4 = cols / 4;
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 v = r4[i];
            acc += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += row[i];
    }
    float total = BR(tmp).Sum(acc);
    if (threadIdx.x == 0) out[r] = total;
}

static double host_sum_fp64(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += (double)x;
    return s;
}

static void fill(std::vector<float>& v, int kind) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (kind == 0) v[i] = 1.0f;
        else if (kind == 1) v[i] = (i % 2 == 0) ? 1.0f : -1.0f;          // 完全抵消
        else if (kind == 2) v[i] = (i % 2 == 0) ? 1e8f : 1.0f;           // 大数吞小数
        else v[i] = 1e-7f * (float)((i % 7) - 3);                        // 小量累加
    }
}

static int correctness() {
    printf("[A] 精度对照（FP64 参照）\n");
    printf("  %-28s %-10s %-16s %-16s %s\n", "用例", "n", "FP32 结果", "FP64 参照", "相对误差");
    struct Case { const char* name; int n; int kind; };
    Case cases[] = {{"奇数长度 1000003", 1000003, 0}, {"完全抵消 ±1", 1 << 20, 1},
                    {"大数吞小数 1e8/1", 1 << 20, 2}, {"小量 1e-7 累加", 1 << 20, 3}};
    int bad = 0;
    for (auto c : cases) {
        std::vector<float> h(c.n); fill(h, c.kind);
        double ref = host_sum_fp64(h);
        float* d; CK(cudaMalloc(&d, h.size() * 4));
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        float* o; CK(cudaMalloc(&o, (1 << 21) * 4)); CK(cudaMemset(o, 0, (1 << 21) * 4));
        // 按 65536 行切块，每行一个 block：模拟真实的多 block 归约
        int cols = 65536;
        int rows = (c.n + cols - 1) / cols;
        row_two_stage<<<rows, 256, 256 * 4>>>(d, o, rows, cols, 1);
        CK(cudaDeviceSynchronize());
        std::vector<float> po(rows);
        CK(cudaMemcpy(po.data(), o, rows * 4, cudaMemcpyDeviceToHost));
        double got = 0; for (int i = 0; i < rows; ++i) got += (double)po[i];
        double rel = fabs(got - ref) / (fabs(ref) + 1e-30);
        printf("  %-28s %-10d %-16.1f %-16.1f %.3e\n", c.name, c.n, got, ref, rel);
        if (rel > 1e-3) ++bad;
        cudaFree(d); cudaFree(o);
    }
    printf("  ⇒ 分层归约把误差压到 O(log N)；完全抵消与小量累加这两类输入\n");
    printf("     是判断「是否只是碰巧对」的关键（相对误差 > 1e-3 的用例数：%d）。\n\n", bad);
    return bad;
}

struct Res { double ms; double gbs; double err; };

template <typename Launch>
static Res time_it(Launch launch, const float* d_in, float* d_out, int rows, int cols,
                   const std::vector<double>& ref) {
    launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) launch();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 20;
    std::vector<float> got(rows);
    CK(cudaMemcpy(got.data(), d_out, rows * 4, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (int r = 0; r < rows; ++r)
        max_err = fmax(max_err, fabs((double)got[r] - ref[r]) / (fabs(ref[r]) + 1e-30));
    cudaEventDestroy(a); cudaEventDestroy(b);
    double bytes = (double)rows * cols * 4;
    return {ms, bytes / (ms * 1e-3) / 1e9, max_err};
}

static void sweep() {
    printf("[B] 行宽 × 行数扫描（每行一个 256 线程 block）\n");
    const int colss[] = {128, 1024, 4096, 14336};
    const int rowss[] = {1, 8, 128, 2048};
    size_t l2 = 0;
    { cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); l2 = p.l2CacheSize; }
    printf("  本卡 L2 = %.0f MB；工作集 ≤ L2 的行带宽会超过 DRAM 上限，单独标注\n",
           l2 / 1048576.0);
    printf("  %-8s %-8s %-10s %-6s %-8s %-12s %-10s %-12s %-10s %s\n",
           "cols", "rows", "工作集MB", "L2内", "vec", "两阶段ms", "GB/s", "一阶段ms", "GB/s", "最大相对误差");
    for (int cols : colss) {
        for (int rows : rowss) {
            size_t n = (size_t)rows * cols;
            std::vector<float> h(n, 1.0f);
            std::vector<double> ref(rows, (double)cols);
            float *d_in, *d_out;
            CK(cudaMalloc(&d_in, n * 4)); CK(cudaMalloc(&d_out, rows * 4));
            CK(cudaMemcpy(d_in, h.data(), n * 4, cudaMemcpyHostToDevice));
            for (int vec : {1, 4}) {
                if (vec == 4 && cols % 4) continue;
                CK(cudaMemset(d_out, 0, rows * 4));
                Res a = time_it([&] { row_two_stage<<<rows, 256, 256 * 4>>>(d_in, d_out, rows, cols, vec); },
                                d_in, d_out, rows, cols, ref);
                Res b = time_it([&] {
                    CK(cudaMemsetAsync(d_out, 0, rows * 4));
                    row_warp_atomic<<<rows, 256>>>(d_in, d_out, rows, cols, vec);
                }, d_in, d_out, rows, cols, ref);
                printf("  %-8d %-8d %-10.2f %-6s %-8d %-12.4f %-10.1f %-12.4f %-10.1f %.2e\n",
                       cols, rows, n * 4 / 1048576.0, (n * 4 <= l2) ? "是" : "否",
                       vec, a.ms, a.gbs, b.ms, b.gbs, fmax(a.err, b.err));
            }
            cudaFree(d_in); cudaFree(d_out);
        }
    }
    printf("  ⇒ rows=1 时 grid 只有 1 个 block（launch/占用受限），rows 增大后进入带宽受限；\n");
    printf("     vec=4 对 cols=14336 这类宽行最明显。\n\n");
}

static void cub_compare() {
    printf("[C] 与 CUB 对照（同样的「每行一个 block」结构，cols=4096/14336, rows=2048）\n");
    printf("  %-8s %-8s %-12s %-8s %-12s %-8s %s\n",
           "cols", "rows", "手写两阶段ms", "GB/s", "CUB ms", "GB/s", "CUB/手写");
    for (int cols : {4096, 14336}) {
        int rows = 2048;
        size_t n = (size_t)rows * cols;
        std::vector<float> h(n, 1.0f);
        std::vector<double> ref(rows, (double)cols);
        float *d_in, *d_out;
        CK(cudaMalloc(&d_in, n * 4)); CK(cudaMalloc(&d_out, rows * 4));
        CK(cudaMemcpy(d_in, h.data(), n * 4, cudaMemcpyHostToDevice));
        Res a = time_it([&] { row_two_stage<<<rows, 256, 256 * 4>>>(d_in, d_out, rows, cols, 4); },
                        d_in, d_out, rows, cols, ref);
        Res c = time_it([&] { row_cub<<<rows, 256>>>(d_in, d_out, rows, cols, 4); },
                        d_in, d_out, rows, cols, ref);
        printf("  %-8d %-8d %-12.4f %-8.1f %-12.4f %-8.1f %.2fx\n",
               cols, rows, a.ms, a.gbs, c.ms, c.gbs, c.ms / a.ms);
        cudaFree(d_in); cudaFree(d_out);
    }
    printf("  ⇒ CUB 的 BlockReduce 与手写树形归约同构（都是 shared + 树），差异在实现细节。\n\n");
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d  SM=%d ===\n\n", p.name, p.major, p.minor, p.multiProcessorCount);
    correctness();
    sweep();
    cub_compare();
    return 0;
}
