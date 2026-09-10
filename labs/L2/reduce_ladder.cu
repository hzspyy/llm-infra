// L2.3 lab · memory-bound 算子的优化阶梯：以求和归约为例。
//
// 归约是最简单的 memory-bound 算子：读 N 个数，输出 1 个数。
// 理论下界很清楚 —— 时间 = N × 4 字节 / 显存带宽。
// L1.1 实测本机只读带宽 1674 GB/s，所以 256 MB 的下界是 0.16 ms。
//
// 但一个 naive 实现可能慢 100 倍。本实验一步步推上去，每步量出增量：
//
//   v0  每元素一次 atomicAdd 到全局           —— 灾难
//   v1  block 内共享内存树形归约              —— 消除全局原子
//   v2  + 最后一个 warp 用 shuffle            —— 消除 barrier
//   v3  全程 warp shuffle，共享内存只存 warp 结果
//   v4  + float4 向量化访存                   —— 每 warp 在途字节 ×4
//   v5  + grid-stride，grid 按波长对齐         —— 消灭尾效应
//   v6  持久化 kernel + 每线程多累加器          —— ILP
//
// 每一步只改一件事，这样增量才可归因。
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -o reduce_ladder reduce_ladder.cu

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// v0：每个元素一次全局 atomicAdd
// 所有线程争抢同一个地址 —— 硬件必须把它们串行化。
// ---------------------------------------------------------------------------
__global__ void red_v0(const float* __restrict__ in, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}

// ---------------------------------------------------------------------------
// v1：block 内共享内存树形归约，每 block 只做一次全局 atomicAdd
// 全局原子操作从 N 次降到 N/blockDim 次。
// ---------------------------------------------------------------------------
__global__ void red_v1(const float* __restrict__ in, float* out, size_t n) {
    extern __shared__ float s[];
    int t = threadIdx.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();                 // 每一轮都要同步整个 block
    }
    if (t == 0) atomicAdd(out, s[0]);
}

// ---------------------------------------------------------------------------
// v2：最后 32 个元素改用 warp shuffle
// 一个 warp 内部天然同步，不需要 __syncthreads。
// __shfl_down_sync 直接在寄存器之间搬数据，连共享内存都不碰。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void red_v2(const float* __restrict__ in, float* out, size_t n) {
    extern __shared__ float s[];
    int t = threadIdx.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        float v = s[t] + s[t + 32];
        v = warp_reduce(v);
        if (t == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// v3：全程 shuffle。共享内存只用来存每个 warp 的部分和。
// barrier 次数从 log2(blockDim) 降到 1。
// ---------------------------------------------------------------------------
__global__ void red_v3(const float* __restrict__ in, float* out, size_t n) {
    __shared__ float warp_sums[32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    size_t i = blockIdx.x * (size_t)blockDim.x + t;
    float v = (i < n) ? in[i] : 0.0f;
    v = warp_reduce(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();                                  // 只有这一次
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// v4：float4 向量化访存
// 每线程一次读 16 字节而不是 4 字节 —— 每 warp 的在途字节 ×4。
// 对照 L1.1：带宽-延迟积 536 KB，向量化直接把所需 warp 数降到 1/4。
// ---------------------------------------------------------------------------
__global__ void red_v4(const float4* __restrict__ in, float* out, size_t n4) {
    __shared__ float warp_sums[32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    size_t i = blockIdx.x * (size_t)blockDim.x + t;
    float v = 0.0f;
    if (i < n4) { float4 x = in[i]; v = x.x + x.y + x.z + x.w; }
    v = warp_reduce(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// v5：grid-stride 循环，grid 固定为一个"波"
// 每个 block 处理多个 tile，消灭尾效应（L2.1 实测：多 1 个 block 慢 17%）。
// ---------------------------------------------------------------------------
__global__ void red_v5(const float4* __restrict__ in, float* out, size_t n4) {
    __shared__ float warp_sums[32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float v = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + t; i < n4; i += stride) {
        float4 x = in[i];
        v += x.x + x.y + x.z + x.w;
    }
    v = warp_reduce(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// v6：每线程 4 个独立累加器
// 打破累加的依赖链，让 FADD 流水线填满（对照 L2.1 的 ILP 实验）。
// ---------------------------------------------------------------------------
__global__ void red_v6(const float4* __restrict__ in, float* out, size_t n4) {
    __shared__ float warp_sums[32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = blockIdx.x * (size_t)blockDim.x + t;
    for (; i + 3 * stride < n4; i += 4 * stride) {
        float4 x0 = in[i], x1 = in[i + stride],
               x2 = in[i + 2 * stride], x3 = in[i + 3 * stride];
        a0 += x0.x + x0.y + x0.z + x0.w;
        a1 += x1.x + x1.y + x1.z + x1.w;
        a2 += x2.x + x2.y + x2.z + x2.w;
        a3 += x3.x + x3.y + x3.z + x3.w;
    }
    for (; i < n4; i += stride) { float4 x = in[i]; a0 += x.x + x.y + x.z + x.w; }
    float v = (a0 + a1) + (a2 + a3);
    v = warp_reduce(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const size_t MB = 256, bytes = MB << 20, n = bytes / sizeof(float), n4 = n / 4;
    const int SMS = p.multiProcessorCount;

    printf("=== %s  SM=%d   数据 %zu MB（%zu 个 float）\n", p.name, SMS, MB, n);

    // L1.1 实测的只读带宽，用作分母
    const double PEAK_GBS = 1674.0;
    double lower_ms = bytes / (PEAK_GBS * 1e9) * 1e3;
    printf("    分母用 L1.1 实测的只读带宽 %.0f GB/s ⇒ 理论下界 %.3f ms\n\n", PEAK_GBS, lower_ms);

    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = 1.0f;      // 和应当 = n
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, bytes)); CK(cudaMalloc(&d_out, sizeof(float)));
    CK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    auto run = [&](const char* name, auto launcher, int iters = 20) {
        CK(cudaMemset(d_out, 0, sizeof(float)));
        launcher();                                   // 预热 + 验证
        CK(cudaDeviceSynchronize());
        float got = 0; CK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        // fp32 累加 2^26 个 1.0 会有舍入，放宽到 1%
        bool ok = std::fabs(got - (float)n) / (float)n < 0.01f;

        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        CK(cudaMemset(d_out, 0, sizeof(float)));
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launcher();
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        cudaEventDestroy(a); cudaEventDestroy(b);

        double gbs = bytes / (ms * 1e-3) / 1e9;
        char note[128] = "";
        if (!ok) snprintf(note, sizeof note,
                          "  ✗ 结果 %.0f，应为 %zu（相对误差 %.1f%%）",
                          (double)got, n, 100.0 * fabs(got - (double)n) / (double)n);
        printf("    %-46s %9.3f ms %9.1f GB/s %6.1f%%%s\n",
               name, ms, gbs, gbs / PEAK_GBS * 100, note);
        return ms;
    };

    printf("    %-46s %9s %9s %7s\n", "版本", "耗时", "有效带宽", "占上限");

    int blk = 256;
    int grid_n  = (int)((n + blk - 1) / blk);
    int grid_n4 = (int)((n4 + blk - 1) / blk);

    run("v0  每元素一次全局 atomicAdd", [&]{ red_v0<<<grid_n, blk>>>(d_in, d_out, n); }, 3);
    run("v1  共享内存树形归约", [&]{ red_v1<<<grid_n, blk, blk*4>>>(d_in, d_out, n); });
    run("v2  + 末 warp 用 shuffle", [&]{ red_v2<<<grid_n, blk, blk*4>>>(d_in, d_out, n); });
    run("v3  全程 shuffle（barrier 只剩 1 次）", [&]{ red_v3<<<grid_n, blk>>>(d_in, d_out, n); });
    run("v4  + float4 向量化访存", [&]{ red_v4<<<grid_n4, blk>>>((const float4*)d_in, d_out, n4); });

    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, red_v5, blk, 0));
    int wave = SMS * bps;
    printf("    （v5/v6 的 grid 固定为一个波 = %d SM × %d block/SM = %d）\n", SMS, bps, wave);
    run("v5  + grid-stride，grid 对齐波长", [&]{ red_v5<<<wave, blk>>>((const float4*)d_in, d_out, n4); });
    run("v6  + 每线程 4 个独立累加器", [&]{ red_v6<<<wave, blk>>>((const float4*)d_in, d_out, n4); });

    // 精度对照：v6 用 fp32 累加 vs CPU 的 fp64 参考
    {
        CK(cudaMemset(d_out, 0, sizeof(float)));
        red_v6<<<wave, blk>>>((const float4*)d_in, d_out, n4);
        CK(cudaDeviceSynchronize());
        float g6 = 0; CK(cudaMemcpy(&g6, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double cpu = 0; for (size_t i = 0; i < n; ++i) cpu += h[i];
        printf("\n    精度对照（%zu 个 1.0f 求和）：\n", n);
        printf("      CPU fp64 参考      %.1f\n", cpu);
        printf("      GPU v6 (fp32 分层)  %.1f   相对误差 %.2e\n",
               (double)g6, fabs(g6 - cpu) / cpu);
        printf("      ⇒ 分层归约（每线程局部和 → warp → block → 全局）本身就是一种\n");
        printf("        「成对求和」，把误差从 O(N) 降到 O(log N)。而 v0 的单点 atomicAdd\n");
        printf("        是纯串行累加，误差 O(N)：float 的 24 位尾数在累加到 2^24 后\n");
        printf("        再加 1.0 就完全丢失 —— 这就是它算不对的原因，与并发无关。\n");
    }

    printf("\n    参照：纯 memcpy 式读取（L1.1 的 bw_read_kernel）%.0f GB/s = 100%%\n", PEAK_GBS);
    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    return 0;
}
