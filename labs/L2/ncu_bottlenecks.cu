// L2.6-B lab · 四类瓶颈在 spark（GB10，唯一能读硬件计数器的机器）上的计数器复核。
//
// 2.6-A 在 crater 上用"墙钟 + profiler 事件"把瓶颈分成四类，判据是时间关系。
// 本程序把同样的四类瓶颈写成四个 case，用 ncu 采计数器、nsys 采时间线，
// 回答两件事：
//   1) 哪些瓶颈能用计数器直接量化（带宽、占用），哪些只能从时间线看出来（CPU 提交、同步）；
//   2) 计数器与 crater 的计时判断是否一致（不一致就保留差异）。
//
// 每个 case 都单独可跑：不带 ncu 时打印"未插桩"的墙钟与 GPU 忙时间，
// 供与插桩结果对照。
//
//   nvcc -O3 -arch=sm_121 -o ncu_bottlenecks ncu_bottlenecks.cu
//   ./ncu_bottlenecks cpu_submit|bandwidth|sync|occ96|occ8|all

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

constexpr int TINY_N = 64;
constexpr size_t BW_BYTES = 512ull << 20;          // 512 MB 输入 × 2 + 512 MB 输出
constexpr size_t SYNC_BYTES = 32ull << 20;
constexpr int TINY_LAUNCHES = 2000;
constexpr int SYNC_ITERS = 20;

__global__ void tiny_kernel(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < TINY_N) x[i] = x[i] * 1.000001f + 1.0f;
}

__global__ void add_kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                           float4* __restrict__ c, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w;
        c[i] = x;
    }
}

// 每个 block 先把 SMEM_ELEMS 个 float4 读进动态共享内存，再写回全局。
// 动态共享内存的大小由启动参数决定 —— 这就是"占用受限"的单因素变量。
__global__ void smem_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            long n4, int elems_per_block) {
    extern __shared__ float4 s[];
    long base = (long)blockIdx.x * elems_per_block;
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        long idx = base + i;
        if (idx < n4) s[i] = in[idx];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        long idx = base + i;
        if (idx < n4) {
            float4 v = s[i];
            v.x *= 2.0f;
            out[idx] = v;
        }
    }
}

struct Timing { double wall_ms; double gpu_ms; };

template <typename F>
static Timing measure(F f, int reps) {
    f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    float gpu = 0; cudaEventElapsedTime(&gpu, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    double wall = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return {wall / reps, gpu / reps};
}

static void report(const char* name, const Timing& t, const char* unit) {
    printf("CASE %-10s wall_ms %.4f gpu_ms %.4f %s\n", name, t.wall_ms, t.gpu_ms, unit);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "all";
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    if (argc > 1 && strcmp(which, "info") == 0) {
        printf("device %s sm_%d%d SM=%d smem/block(optin)=%zu\n", p.name,
               p.major, p.minor, sms, p.sharedMemPerBlockOptin);
        return 0;
    }

    bool all = strcmp(which, "all") == 0;

    if (all || strcmp(which, "cpu_submit") == 0) {
        float* x; CK(cudaMalloc(&x, TINY_N * sizeof(float)));
        auto t = measure([&]{ tiny_kernel<<<1, 64>>>(x); }, TINY_LAUNCHES);
        report("cpu_submit", t, "2000 个小 kernel");
        CK(cudaFree(x));
    }

    if (all || strcmp(which, "bandwidth") == 0) {
        size_t n = BW_BYTES / sizeof(float);
        size_t n4 = n / 4;
        float *a, *b, *c;
        CK(cudaMalloc(&a, BW_BYTES)); CK(cudaMalloc(&b, BW_BYTES)); CK(cudaMalloc(&c, BW_BYTES));
        CK(cudaMemset(a, 0, BW_BYTES)); CK(cudaMemset(b, 0, BW_BYTES));
        int threads = 256;
        long blocks = (long)((n4 + threads - 1) / threads);
        auto t = measure([&]{ add_kernel<<<(unsigned)blocks, threads>>>(
            (const float4*)a, (const float4*)b, (float4*)c, (long)n4); }, 5);
        double gb = 3.0 * BW_BYTES / 1e9;
        printf("CASE %-10s wall_ms %.4f gpu_ms %.4f GB/s %.1f（按 GPU 忙时间）\n",
               "bandwidth", t.wall_ms, t.gpu_ms, gb / (t.gpu_ms * 1e-3));
        CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(c));
    }

    if (all || strcmp(which, "sync") == 0) {
        size_t n4 = (SYNC_BYTES / sizeof(float)) / 4;
        float *a, *b, *c; float host = 0.f;
        CK(cudaMalloc(&a, SYNC_BYTES)); CK(cudaMalloc(&b, SYNC_BYTES)); CK(cudaMalloc(&c, SYNC_BYTES));
        int threads = 256;
        long blocks = (long)((n4 + threads - 1) / threads);
        auto t = measure([&]{
            add_kernel<<<(unsigned)blocks, threads>>>((const float4*)a, (const float4*)b,
                                                      (float4*)c, (long)n4);
            CK(cudaMemcpy(&host, c, 4, cudaMemcpyDeviceToHost));   // 每次回读，强制同步
        }, SYNC_ITERS);
        report("sync", t, "20 次 kernel + D2H 回读");
        CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(c));
    }

    auto occ_case = [&](const char* name, int smem_bytes) {
        size_t n = BW_BYTES / sizeof(float);
        size_t n4 = n / 4;
        float *in, *out;
        CK(cudaMalloc(&in, BW_BYTES)); CK(cudaMalloc(&out, BW_BYTES));
        CK(cudaMemset(in, 0, BW_BYTES));
        int elems_per_block = smem_bytes / sizeof(float4);
        long blocks = (long)((n4 + elems_per_block - 1) / elems_per_block);
        auto launch = [&]{
            CK(cudaFuncSetAttribute(smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes));
            smem_kernel<<<(unsigned)blocks, 256, smem_bytes>>>((const float4*)in, (float4*)out,
                                                              (long)n4, elems_per_block);
        };
        int occ = 0;
        CK(cudaFuncSetAttribute(smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, smem_kernel, 256, smem_bytes));
        auto t = measure(launch, 5);
        printf("CASE %-10s wall_ms %.4f gpu_ms %.4f smem_B %d blocks_total %ld block/SM %d "
               "（SM=%d）\n", name, t.wall_ms, t.gpu_ms, smem_bytes, blocks, occ, sms);
        CK(cudaFree(in)); CK(cudaFree(out));
    };
    if (all || strcmp(which, "occ96") == 0) occ_case("occ96", 96 * 1024);
    if (all || strcmp(which, "occ8") == 0) occ_case("occ8", 8 * 1024);
    return 0;
}
