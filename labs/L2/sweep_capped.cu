// L2.2-B 补：用 -maxrregcount 硬性压寄存器，看 spill 与性能的代价。
// 与 toolchain_sweep.cu 的 __launch_bounds__ 版本对照：
// 后者只是"目标"，这里是硬上限，编译器只能靠 spill 让路。
//   nvcc -O3 -arch=sm_120 -maxrregcount=32 -Xptxas -v -o cap32 sweep_capped.cu
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s: %s\n", #x, cudaGetErrorString(e)); } } while(0)

#ifndef CHAINS
#define CHAINS 64
#endif

__global__ void compute_kernel(float* __restrict__ out, int iters) {
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

int main() {
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, (const void*)compute_kernel));
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, (const void*)compute_kernel, 256, 0));
    int grid = sms * (bps > 0 ? bps : 1);
    float* d_out; CK(cudaMalloc(&d_out, 4));
    const int iters = 200000;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    compute_kernel<<<grid, 256>>>(d_out, iters);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) compute_kernel<<<grid, 256>>>(d_out, iters);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 5;
    double tflops = (double)grid * 256 * iters * CHAINS * 2 / (ms * 1e-3) / 1e12;
    printf("regs=%d local=%d occ=%d tflops=%.1f ms=%.3f\n",
           attr.numRegs, (int)attr.localSizeBytes, bps, tflops, ms);
    cudaFree(d_out);
    return 0;
}
