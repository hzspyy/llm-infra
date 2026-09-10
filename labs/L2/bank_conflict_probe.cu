// L2.6 lab · 用硬件计数器直接读 bank conflict 的次数。
//
// L1.1 里我是这样"证明"bank conflict 存在的：写两个 kernel，一个跨步 32，
// 一个跨步 33，测出时间差 10.1 倍，然后说"这就是 bank conflict"。
// 那是**推断**——时间变慢有一万种可能的解释。
//
// ncu 的 l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld 直接给出
// 冲突次数。本 probe 让三种访问模式跑同样多的次数，只有 bank 分布不同：
//
//   stride1   s[tid]              相邻线程访问相邻 bank    → 应为 0 冲突
//   stride32  s[tid*32]           32 个线程全落在 bank 0   → 应为 31 路冲突
//   padded    s[tid*33]           跨步 33 错开一位         → 应回到 0 冲突
//
// 编译：nvcc -O3 -arch=sm_XX -lineinfo -o bank_conflict_probe bank_conflict_probe.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); return 1;} } while(0)

constexpr int BLOCK = 256;
constexpr int REPS  = 64;          // 每个线程做多少次共享内存读
constexpr int SMEM  = 256 * 33;    // 够 stride 33 用

// 三个 kernel 结构完全相同，只有下标算式不同。
// 用 8 个独立累加器，避免依赖链把吞吐差异藏起来（L1.1 陷阱 11）。
//
// 注意：三个实例必须有**不同的函数名**，否则 ncu 的 -k 正则分不开
// （模板实例在 --kernel-name-base function 下都显示成 probe<...>）。
template <int STRIDE>
__device__ __forceinline__ void probe_body(float* out) {
    __shared__ float s[SMEM];
    for (int i = threadIdx.x; i < SMEM; i += BLOCK) s[i] = float(i);
    __syncthreads();

    float a[8] = {0,0,0,0,0,0,0,0};
    const int base = threadIdx.x * STRIDE;
    #pragma unroll 1
    for (int r = 0; r < REPS; ++r) {
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            a[k] += s[(base + k * 4 + r) % SMEM];      // 8 次独立的共享内存读
    }
    float acc = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) acc += a[k];
    if (acc == -1.0f) out[blockIdx.x] = acc;           // 防优化
}

__global__ void probe_stride1 (float* o) { probe_body<1> (o); }   // 无冲突
__global__ void probe_stride32(float* o) { probe_body<32>(o); }   // 32 路全撞 bank 0
__global__ void probe_stride33(float* o) { probe_body<33>(o); }   // padding 错开

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d\n", p.name, p.major, p.minor);
    printf("    每个 kernel：%d block × %d 线程 × %d 轮 × 8 次共享内存读\n",
           p.multiProcessorCount, BLOCK, REPS);
    printf("    总共享内存读次数 = %lld\n\n",
           1LL * p.multiProcessorCount * BLOCK * REPS * 8);

    float* d;
    CK(cudaMalloc(&d, sizeof(float) * p.multiProcessorCount));
    const int grid = p.multiProcessorCount;

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    #define RUN(K, S, name) do {                                           \
        K<<<grid, BLOCK>>>(d); CK(cudaDeviceSynchronize());                \
        CK(cudaEventRecord(e0));                                           \
        for (int i = 0; i < 20; ++i) K<<<grid, BLOCK>>>(d);                \
        CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());              \
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));                   \
        printf("    %-10s stride=%-3d  %.4f ms/次\n", name, S, ms / 20);   \
    } while (0)

    RUN(probe_stride1,  1,  "stride1");
    RUN(probe_stride32, 32, "stride32");
    RUN(probe_stride33, 33, "padded");
    printf("\n（时间只是旁证。真正的判据是 ncu 报出的 bank conflict 次数。）\n");
    return 0;
}
