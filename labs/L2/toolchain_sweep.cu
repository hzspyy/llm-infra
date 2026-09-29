// L2.2-B · unroll × 寄存器预算的扫描：寄存器、spill、local、occupancy 与时间一起报。
//
// 只改两个旋钮：
//   UNROLL  每个线程在一次循环里处理多少个 float4（展开因子）
//   MINB    __launch_bounds__(256, MINB) 里的"每 SM 至少几个 block"
//           —— 它逼编译器把寄存器用量压到 65536/(256*MINB) 以内
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_120 -Xptxas -v -o toolchain_sweep toolchain_sweep.cu
// 说明：__launch_bounds__ 是"用 spill 换占用率"，本实验就是把这句话量化。

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); } } while(0)

template <int UNROLL, int MINB>
__global__ void __launch_bounds__(256, MINB)
sweep_kernel(const float4* __restrict__ in, size_t n4, float* __restrict__ out) {
    size_t base = (size_t)blockIdx.x * blockDim.x * UNROLL + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x * UNROLL;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = base; i < n4; i += stride) {
        float4 a[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            size_t j = i + (size_t)k * blockDim.x;
            a[k] = (j < n4) ? in[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            acc.x += a[k].x; acc.y += a[k].y; acc.z += a[k].z; acc.w += a[k].w;
        }
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1234.5f) *out = s;
}

struct Row { const char* name; int regs; int spill_stores; int spill_loads; int local; int bps; int smem; double gbs; double ms; };

template <int UNROLL, int MINB>
Row run(const float4* d_in, size_t n4, float* d_out, size_t bytes, double tflops_ignore) {
    (void)tflops_ignore;
    Row r{};
    r.name = "";
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, (const void*)sweep_kernel<UNROLL, MINB>));
    r.regs = attr.numRegs;
    r.local = (int)attr.localSizeBytes;
    r.smem = (int)attr.sharedSizeBytes;
    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, (const void*)sweep_kernel<UNROLL, MINB>, 256, 0));
    r.bps = bps;
    int grid = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid, (const void*)sweep_kernel<UNROLL, MINB>, 256, 0);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    grid = sms * bps;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    sweep_kernel<UNROLL, MINB><<<grid, 256>>>(d_in, n4, d_out);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) sweep_kernel<UNROLL, MINB><<<grid, 256>>>(d_in, n4, d_out);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 20;
    r.ms = ms;
    r.gbs = bytes / (ms * 1e-3) / 1e9;
    cudaEventDestroy(a); cudaEventDestroy(b);
    return r;
}

// 计算受限版本：CHAINS 条独立 FMA 链，MINB 决定寄存器预算。
// CHAINS 越大活跃寄存器越多；MINB 越大预算越小 —— 两者相撞就 spill。
template <int CHAINS, int MINB>
__global__ void __launch_bounds__(256, MINB)
compute_kernel(float* __restrict__ out, int iters) {
    float a[CHAINS];
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) a[k] = threadIdx.x * 0.001f + k;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < CHAINS; ++k) a[k] = fmaf(a[k], 1.0001f, 0.5f);
    }
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) s += a[k];
    if (s == 1234.5f) *out = s;
}

struct CRow { int regs; int local; int bps; double tflops; };

template <int CHAINS, int MINB>
CRow run_compute(float* d_out, int iters) {
    CRow r{};
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, (const void*)compute_kernel<CHAINS, MINB>));
    r.regs = attr.numRegs;
    r.local = (int)attr.localSizeBytes;
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, (const void*)compute_kernel<CHAINS, MINB>, 256, 0));
    r.bps = bps;
    int grid = sms * (bps > 0 ? bps : 1);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    compute_kernel<CHAINS, MINB><<<grid, 256>>>(d_out, iters);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) compute_kernel<CHAINS, MINB><<<grid, 256>>>(d_out, iters);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 5;
    r.tflops = (double)grid * 256 * iters * CHAINS * 2 / (ms * 1e-3) / 1e12;
    cudaEventDestroy(a); cudaEventDestroy(b);
    return r;
}

static const int UNROLLS[] = {1, 2, 4, 8};
static const int MINBS[] = {1, 2, 4, 8};

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d  SM=%d  每 SM 寄存器 65536 ===\n",
           p.name, p.major, p.minor, p.multiProcessorCount);

    const size_t bytes = 1024ull << 20;          // 1 GiB，远超 L2
    float4* d_in; CK(cudaMalloc(&d_in, bytes));
    CK(cudaMemset(d_in, 1, bytes));
    size_t n4 = bytes / sizeof(float4);
    float* d_out; CK(cudaMalloc(&d_out, 4));

    printf("\n%-10s %-6s %-8s %-10s %-9s %-8s %-12s %s\n",
           "unroll", "MINB", "regs", "local(B)", "occ blk", "spill", "GB/s", "ms");
    printf("%s\n", "------------------------------------------------------------------------------------------");

#define CASE(U, M) do { \
    Row r = run<U, M>(d_in, n4, d_out, bytes, 0); \
    printf("%-10d %-6d %-8d %-10d %-9d %-8s %-12.1f %.4f\n", U, M, r.regs, r.local, r.bps, "-", r.gbs, r.ms); \
  } while (0)

    CASE(1, 1); CASE(1, 2); CASE(1, 4); CASE(1, 8);
    CASE(2, 1); CASE(2, 2); CASE(2, 4); CASE(2, 8);
    CASE(4, 1); CASE(4, 2); CASE(4, 4); CASE(4, 8);
    CASE(8, 1); CASE(8, 2); CASE(8, 4); CASE(8, 8);
#undef CASE

    printf("\n[compute] 独立 FMA 链数 × 寄存器预算（iters=200000）\n");
    printf("%-10s %-6s %-8s %-10s %-9s %-12s %s\n",
           "chains", "MINB", "regs", "local(B)", "occ blk", "TFLOPS", "每 SM 寄存器预算");
    printf("%s\n", "------------------------------------------------------------------------------------------");
    const int iters = 200000;
#define CCASE(C, M) do { \
    CRow r = run_compute<C, M>(d_out, iters); \
    printf("%-10d %-6d %-8d %-10d %-9d %-12.1f %d\n", C, M, r.regs, r.local, r.bps, r.tflops, \
           65536 / (256 * M)); \
  } while (0)
    CCASE(4, 1);  CCASE(4, 2);  CCASE(4, 8);
    CCASE(16, 1); CCASE(16, 2); CCASE(16, 8);
    CCASE(32, 1); CCASE(32, 2); CCASE(32, 8);
    CCASE(64, 1); CCASE(64, 2); CCASE(64, 8);
#undef CCASE
    printf("\n注：spill 的字节数在 -Xptxas -v 的日志里（本程序只报 local memory 大小）；\n");
    printf("    local(B) > 0 说明有变量被放到了 local memory，延迟按显存算。\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
