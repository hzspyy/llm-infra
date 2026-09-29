// L2.3 补 · v0 的 82.9 ms 归因：把"原子竞争"从"读内存"里分出来。
//
// 三个 kernel 读同一份 256 MB 数据：
//   A. 纯读（结果写入一个不冲突的位置）        —— 只测内存
//   B. 每线程一次全局 atomicAdd（就是 v0）     —— 内存 + 原子串行
//   C. 共享内存树 + 每 block 一次 atomicAdd    —— 内存 + 少量原子
// 归因 = B - A（原子串行），C - A（残余开销）。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s: %s\n", #x, cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void pure_read(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += in[i];
    if (acc == 1234.5f) out[blockIdx.x % 1024] = acc;    // 不冲突的写
}

__global__ void atomic_per_thread(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += in[i];
    atomicAdd(out, acc);                                  // 每线程一次
}

__global__ void atomic_per_block(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[256];
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += in[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);            // 每 block 一次
}

static float time_ms(void (*k)(const float*, float*, size_t), const float* in, float* out,
                     size_t n, int grid, int iters) {
    k<<<grid, 256>>>(in, out, n); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) k<<<grid, 256>>>(in, out, n);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const size_t bytes = 256ull << 20;                 // 与 reduce_ladder 同一规模
    size_t n = bytes / 4;
    float* d_in; CK(cudaMalloc(&d_in, bytes)); CK(cudaMemset(d_in, 1, bytes));
    float* d_out; CK(cudaMalloc(&d_out, bytes));
    const int grid = p.multiProcessorCount * 6;

    // 为了让每线程只跑一次循环，per-thread atomic 用"每元素一个线程"的 grid
    size_t grid_pt = (n + 255) / 256;
    float t_read = 0, t_pt = 0, t_pb = 0;
    CK(cudaMemset(d_out, 0, bytes));
    t_read = time_ms(pure_read, d_in, d_out, n, grid, 20);
    CK(cudaMemset(d_out, 0, bytes));
    t_pb = time_ms(atomic_per_block, d_in, d_out, n, grid, 20);
    CK(cudaMemset(d_out, 0, bytes));
    {
        // per-thread 版本重复计时会不断累加，但只影响数值不影响时间
        atomic_per_thread<<<grid_pt, 256>>>(d_in, d_out, n);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) atomic_per_thread<<<grid_pt, 256>>>(d_in, d_out, n);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b); t_pt = ms / 20;
    }

    printf("=== %s  SM=%d  数据 %.0f MB ===\n", p.name, p.multiProcessorCount, bytes / 1048576.0);
    printf("  A 纯读（不冲突写）              %8.3f ms   %7.1f GB/s\n", t_read, bytes / (t_read * 1e-3) / 1e9);
    printf("  C 共享内存树 + 每 block 一次原子 %8.3f ms   %7.1f GB/s\n", t_pb, bytes / (t_pb * 1e-3) / 1e9);
    printf("  B 每线程一次全局原子            %8.3f ms   %7.1f GB/s\n", t_pt, bytes / (t_pt * 1e-3) / 1e9);
    printf("\n  归因：\n");
    printf("    纯内存                                                        %8.3f ms\n", t_read);
    printf("    + 每 block 一次原子（%d 个 block × 每 block 1 次）           %8.3f ms  (+%.3f)\n",
           grid, t_pb, t_pb - t_read);
    printf("    + 每线程一次原子（%zu 次 atomicAdd 串行）                    %8.3f ms  (+%.3f)\n",
           n, t_pt, t_pt - t_read);
    printf("    ⇒ v0 的额外耗时中，原子串行占 %.1f%%，其余是访存本身。\n",
           100.0 * (t_pt - t_read) / t_pt);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
