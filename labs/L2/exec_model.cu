// L2.1 lab · CUDA 执行模型：warp、发散、占用率、并发。
//
// L1.1 讲了硬件有什么，本实验讲**软件怎么用它**，以及用错了代价多大。
//
// 五个实验：
//   A. warp 分支发散：if 两侧都执行，掩码屏蔽 —— 代价到底多大？
//   B. 占用率 vs 性能：更高的占用率一定更快吗？（memory-bound 与 compute-bound 分开测）
//   C. 寄存器压力与 spill：寄存器不够时会溢出到 local memory，代价多少？
//   D. 多流并发：独立 kernel 能同时跑吗？什么条件下不能？
//   E. grid 尺寸与尾效应（tail effect）：为什么 block 数最好是 SM 数的整数倍
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_120 -o exec_model exec_model.cu
//   # 想看寄存器用量加：-Xptxas -v

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); } } while(0)

static float time_ms(void (*launch)(int), int arg, int iters = 20) {
    launch(arg); CK(cudaDeviceSynchronize());              // 预热
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch(arg);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

// ---------------------------------------------------------------------------
// A. 分支发散
//
// warp 里 32 个线程共用一个程序计数器。遇到 if 且两侧都有线程要走时，
// 硬件**两条路径都执行一遍**，用掩码屏蔽不该生效的线程。
// 所以一个 N 路发散的 if，理论上要花 N 倍时间。
// ---------------------------------------------------------------------------
__global__ void divergence_kernel(float* out, int n, int ways, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float acc = tid * 0.001f;
    int branch = (threadIdx.x % 32) % ways;      // ways=1 时全 warp 走同一条
    for (int it = 0; it < iters; ++it) {
        // 每条分支做同样多的工作，唯一的区别是"要不要串行执行多条"
        switch (branch) {
            case 0: acc = fmaf(acc, 1.0001f, 0.1f); break;
            case 1: acc = fmaf(acc, 1.0002f, 0.2f); break;
            case 2: acc = fmaf(acc, 1.0003f, 0.3f); break;
            case 3: acc = fmaf(acc, 1.0004f, 0.4f); break;
            case 4: acc = fmaf(acc, 1.0005f, 0.5f); break;
            case 5: acc = fmaf(acc, 1.0006f, 0.6f); break;
            case 6: acc = fmaf(acc, 1.0007f, 0.7f); break;
            case 7: acc = fmaf(acc, 1.0008f, 0.8f); break;
            default: acc = fmaf(acc, 1.0009f, 0.9f); break;
        }
    }
    out[tid] = acc;
}

// ---------------------------------------------------------------------------
// B/C. 用共享内存人为压低占用率
//
// 每 SM 的共享内存是固定的（本机 100 KB 可 opt-in）。
// 一个 block 申请越多 smem，同时能驻留的 block 就越少 → 占用率越低。
// 这是最干净的"只改占用率、不改算法"的旋钮。
// ---------------------------------------------------------------------------
template <int SMEM_KB>
__global__ void occupancy_mem_kernel(const float4* __restrict__ in, size_t n4,
                                     float* out) {
    __shared__ char pad[SMEM_KB * 1024];
    pad[threadIdx.x] = (char)threadIdx.x;                 // 防止 smem 被优化掉
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = tid; i < n4; i += stride) {
        float4 v = in[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1234.5f && pad[0] == 7) *out = acc.x + acc.y;
}

template <int SMEM_KB>
__global__ void occupancy_compute_kernel(float* out, int iters) {
    __shared__ char pad[SMEM_KB * 1024];
    pad[threadIdx.x] = (char)threadIdx.x;
    float a = threadIdx.x * 0.001f, b = 1.0001f, c = 0.5f;
    float x0 = a, x1 = a + 1, x2 = a + 2, x3 = a + 3;     // 4 条独立链填流水
    for (int i = 0; i < iters; ++i) {
        x0 = fmaf(x0, b, c); x1 = fmaf(x1, b, c);
        x2 = fmaf(x2, b, c); x3 = fmaf(x3, b, c);
    }
    if (x0 + x1 + x2 + x3 == 1234.5f && pad[0] == 7) *out = x0;
}

// 动态共享内存版：静态 __shared__ 上限是 48 KB，
// 想申请更多必须用 extern __shared__ + cudaFuncSetAttribute 显式 opt-in（L1.1 讲过）。
// 只有这样才能把占用率压到 1 block/SM。
__global__ void occ_mem_dyn(const float4* __restrict__ in, size_t n4, float* out) {
    extern __shared__ char dpad[];
    dpad[threadIdx.x] = (char)threadIdx.x;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = tid; i < n4; i += stride) {
        float4 v = in[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1234.5f && dpad[0] == 7) *out = acc.x;
}

// ---------------------------------------------------------------------------
// E. 尾效应：grid 不是 SM 数整数倍时，最后一"波"只有部分 SM 在干活
// ---------------------------------------------------------------------------
__global__ void tail_kernel(float* out, int iters) {
    float x = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; ++i) x = fmaf(x, 1.0001f, 0.5f);
    if (x == 1234.5f) *out = x;
}

// ---------------------------------------------------------------------------
// F. 线程 / block / SM 映射与边界尺寸
//
// 三个问题：
//   F1 一个 block 的线程落在哪些 SM 上？32 个线程的 warp 边界在哪？
//   F2 元素数跨过 32/128/256 的整数倍时，结果还对不对（尾元素有没有漏）？
//   F3 元素数只多一个，为什么时间跳一个 block 的量级？
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned smid() {
    unsigned r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}

__global__ void map_kernel(int* smid_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) smid_out[tid] = (int)smid();
}

// 逐元素 kernel：块内线程数故意不整除，用来看尾 block
__global__ void scale_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// ---------------------------------------------------------------------------

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d  SM=%d  每SM最大线程=%d  smem/block(optin)=%zu KB\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.maxThreadsPerMultiProcessor, p.sharedMemPerBlockOptin / 1024);

    const int SMS = p.multiProcessorCount;
    float* d_out; CK(cudaMalloc(&d_out, sizeof(float) * (1 << 20)));

    // ---------------- A. 分支发散 ----------------
    printf("[A] warp 分支发散的代价（每条分支工作量相同，只是要不要串行）\n");
    printf("    %-10s %-12s %-10s %s\n", "发散路数", "耗时 ms", "相对 1 路", "理论");
    {
        int n = SMS * 8 * 256, iters = 20000;
        float base = 0;
        for (int ways : {1, 2, 4, 8}) {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            divergence_kernel<<<SMS * 8, 256>>>(d_out, n, ways, iters);
            CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i)
                divergence_kernel<<<SMS * 8, 256>>>(d_out, n, ways, iters);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 10;
            if (ways == 1) base = ms;
            printf("    %-10d %-12.3f %-10.2fx %dx\n", ways, ms, ms / base, ways);
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        printf("    ⇒ 实测倍数低于理论，因为编译器会把短分支转成**谓词执行**（predication）：\n");
        printf("      两条路都算，用谓词丢弃结果，省掉跳转但仍付出算力。\n\n");
    }

    // ---------------- B. 占用率 vs 性能 ----------------
    printf("[B] 占用率 vs 性能（用共享内存压低占用率，算法完全不变）\n");
    {
        const size_t bytes = 1024ull << 20;
        float4* d_in; CK(cudaMalloc(&d_in, bytes)); CK(cudaMemset(d_in, 1, bytes));
        size_t n4 = bytes / sizeof(float4);
        printf("    %-12s %-14s %-16s %-16s %s\n",
               "smem/block", "理论占用率", "memory-bound", "compute-bound", "");
        printf("    %-12s %-14s %-16s %-16s\n", "", "(blocks/SM)", "GB/s", "TFLOPS");

        auto run = [&](auto memk, auto compk, int smem_kb) {
            int blocks_per_sm = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, memk, 256, 0));
            int grid = SMS * std::max(blocks_per_sm, 1);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

            memk<<<grid, 256>>>(d_in, n4, d_out); CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 5; ++i) memk<<<grid, 256>>>(d_in, n4, d_out);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms_mem = 0; cudaEventElapsedTime(&ms_mem, a, b); ms_mem /= 5;

            const int iters = 100000;
            compk<<<grid, 256>>>(d_out, iters); CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 5; ++i) compk<<<grid, 256>>>(d_out, iters);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms_cmp = 0; cudaEventElapsedTime(&ms_cmp, a, b); ms_cmp /= 5;

            double gbs = bytes / (ms_mem * 1e-3) / 1e9;
            // 每次迭代 4 条 FMA = 8 FLOP
            double tflops = (double)grid * 256 * iters * 8 / (ms_cmp * 1e-3) / 1e12;
            printf("    %-12d %-14d %-16.1f %-16.1f\n", smem_kb, blocks_per_sm, gbs, tflops);
            cudaEventDestroy(a); cudaEventDestroy(b);
        };
        run(occupancy_mem_kernel<1>,  occupancy_compute_kernel<1>,  1);
        run(occupancy_mem_kernel<8>,  occupancy_compute_kernel<8>,  8);
        run(occupancy_mem_kernel<16>, occupancy_compute_kernel<16>, 16);
        run(occupancy_mem_kernel<32>, occupancy_compute_kernel<32>, 32);
        run(occupancy_mem_kernel<48>, occupancy_compute_kernel<48>, 48);

        // 用动态 smem 突破 48 KB，把占用率压到 1 block/SM
        CK(cudaFuncSetAttribute(occ_mem_dyn,
               cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        for (int kb : {64, 96}) {
            int bps = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, occ_mem_dyn, 256, kb * 1024));
            int grid = SMS * std::max(bps, 1);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            occ_mem_dyn<<<grid, 256, kb * 1024>>>(d_in, n4, d_out); CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 5; ++i) occ_mem_dyn<<<grid, 256, kb * 1024>>>(d_in, n4, d_out);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 5;
            printf("    %-12d %-14d %-16.1f %-16s  (动态 smem)\n",
                   kb, bps, bytes / (ms * 1e-3) / 1e9, "-");
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        printf("    ⇒ 对照 L1.1：2 blocks/SM 就能打满带宽，再高的占用率买不到带宽。\n");
        printf("      compute-bound 更不敏感——它靠的是**指令级并行(ILP)**而非线程级。\n\n");
        CK(cudaFree(d_in));
    }

    // ---------------- D. 多流并发 ----------------
    printf("[D] 多流并发：独立 kernel 能同时跑吗\n");
    {
        const int iters = 200000;
        // 故意用很小的 grid，让单个 kernel 占不满 GPU
        int small_grid = SMS / 4;
        cudaStream_t st[4];
        for (int i = 0; i < 4; ++i) cudaStreamCreate(&st[i]);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

        for (int nk : {1, 2, 4}) {
            // 串行：全在默认流
            tail_kernel<<<small_grid, 256>>>(d_out, iters); CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int k = 0; k < nk; ++k) tail_kernel<<<small_grid, 256>>>(d_out, iters);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms_seq = 0; cudaEventElapsedTime(&ms_seq, a, b);

            // 并发：各自一条流
            CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int k = 0; k < nk; ++k)
                tail_kernel<<<small_grid, 256, 0, st[k]>>>(d_out, iters);
            for (int k = 0; k < nk; ++k) CK(cudaStreamSynchronize(st[k]));
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms_par = 0; cudaEventElapsedTime(&ms_par, a, b);

            printf("    %d 个 kernel（每个 %d blocks，占 %d%% 的 SM）："
                   "同流 %6.3f ms，多流 %6.3f ms，重叠 %.2f×\n",
                   nk, small_grid, small_grid * 100 / SMS, ms_seq, ms_par, ms_seq / ms_par);
        }
        printf("    ⇒ 只有当单个 kernel **占不满 GPU** 时，多流才有意义。\n");
        printf("      推理里的小算子（RMSNorm、RoPE）正是这种情况。\n\n");
        for (int i = 0; i < 4; ++i) cudaStreamDestroy(st[i]);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }

    // ---------------- E. 尾效应 ----------------
    printf("[E] 尾效应：扫描 grid 大小，看波次边界上的台阶\n");
    {
        const int iters = 200000;
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, tail_kernel, 256, 0));
        int wave = SMS * bps;                       // 一"波"能同时驻留多少 block
        printf("    每 SM 可驻留 %d 个 block ⇒ 一波 = %d 个 block\n", bps, wave);
        printf("    %-9s %-9s %-12s %-16s %s\n",
               "blocks", "波数", "耗时 ms", "每 block 摊销 µs", "");
        double prev = 0;
        for (int g : {wave/2, wave-2, wave-1, wave, wave+1, wave+2,
                      2*wave-1, 2*wave, 2*wave+1, 3*wave, 3*wave+1}) {
            if (g <= 0) continue;
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            tail_kernel<<<g, 256>>>(d_out, iters); CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) tail_kernel<<<g, 256>>>(d_out, iters);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 10;
            const char* mark = (prev > 0 && ms > prev * 1.25) ? "   ← 跨过波次边界，跳了一台阶" : "";
            printf("    %-9d %-9.2f %-12.3f %-16.2f%s\n",
                   g, (double)g / wave, ms, ms * 1000 / g, mark);
            prev = ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        printf("    ⇒ 耗时是**阶梯**而不是斜线：只要多出 1 个 block，就要多跑一整波。\n");
        printf("      所以 grid 要按「SM 数 × 每 SM 可驻留 block 数」对齐；\n");
        printf("      persistent kernel（固定开一波、内部循环取任务）就是为了消灭这个台阶。\n\n");
    }

    // ---------------- F. 映射与边界尺寸 ----------------
    printf("[F] 线程/block/SM 映射与边界尺寸\n");
    {
        // F1：一个 block 的线程落在哪些 SM 上
        const int NB = 3, BT = 48;          // 48 = 1.5 个 warp
        int nmap = NB * BT;
        int* d_smid; CK(cudaMalloc(&d_smid, nmap * sizeof(int)));
        int* h_smid = (int*)malloc(nmap * sizeof(int));
        map_kernel<<<NB, BT>>>(d_smid, nmap);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_smid, d_smid, nmap * sizeof(int), cudaMemcpyDeviceToHost));
        printf("    %d 个 block × %d 线程（= 1.5 个 warp/block）\n", NB, BT);
        for (int b = 0; b < NB; ++b) {
            int sm = h_smid[b * BT], same = 1;
            for (int t = 0; t < BT; ++t)
                if (h_smid[b * BT + t] != sm) same = 0;
            printf("      block %d：%d 个线程全在 SM %d（同 SM: %s）"
                   "  尾 warp 活跃线程 = %d\n",
                   b, BT, sm, same ? "是" : "否", BT % 32 == 0 ? 32 : BT % 32);
        }
        printf("    ⇒ 一个 block 不会被拆到两个 SM；block 内按 32 线程切 warp，\n");
        printf("      %d 线程的 block = %d 个满 warp + 1 个 %d 线程的尾 warp。\n",
               BT, BT / 32, BT % 32);
        cudaFree(d_smid); free(h_smid);

        // F2：元素数跨过 warp / block 整数倍，结果必须逐元素对齐
        printf("\n    边界元素数正确性（block=128，逐元素与 CPU 参照比较）\n");
        const int B = 128;
        const int sizes[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                             255, 256, 257, 1023, 1024, 1025};
        float* h_in = (float*)malloc(1u << 21);
        float* h_out = (float*)malloc(1u << 21);
        float* h_ref = (float*)malloc(1u << 21);
        float *d_in, *d_out2;
        CK(cudaMalloc(&d_in, 1u << 21));
        CK(cudaMalloc(&d_out2, 1u << 21));
        printf("      %-8s %-8s %-10s %s\n", "n", "grid", "逐元素一致", "不一致个数");
        for (int n2 : sizes) {
            for (int i = 0; i < n2; ++i) {
                h_in[i] = i * 0.5f;
                h_ref[i] = h_in[i] * 2.0f + 1.0f;
            }
            int grid = (n2 + B - 1) / B;
            CK(cudaMemcpy(d_in, h_in, n2 * sizeof(float), cudaMemcpyHostToDevice));
            scale_kernel<<<grid, B>>>(d_in, d_out2, n2);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(h_out, d_out2, n2 * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < n2; ++i) if (h_out[i] != h_ref[i]) ++bad;
            printf("      %-8d %-8d %-10s %d\n", n2, grid, bad == 0 ? "是" : "否", bad);
        }
        printf("    ⇒ `if (i < n)` 覆盖尾 warp 与尾 block 的全部元素；去掉它就会越界写，\n");
        printf("      这类错误由 5.2 的失败小例与 compute-sanitizer 负责暴露。\n");

        // F3：元素数跨过一波的边界，耗时跳一台阶
        printf("\n    元素数跨过一波边界（block=256，grid=ceil(n/256)，每个 block 固定工作量）\n");
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, tail_kernel, 256, 0));
        int onewave_blocks = SMS * bps;
        int full = onewave_blocks * 256;
        printf("      一波 = %d 个 block = %d 个元素；下面只改 n，grid 跟着变\n",
               onewave_blocks, full);
        printf("      %-10s %-12s %-8s %-12s %s\n",
               "n", "grid", "波数", "耗时 ms", "相对一波");
        float base = 0, prev = 0;
        for (int delta : {-1, 0, 1, 256, 257}) {
            int n2 = full + delta;
            int grid = (n2 + 255) / 256;
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            tail_kernel<<<grid, 256>>>(d_out2, 200000);
            CK(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) tail_kernel<<<grid, 256>>>(d_out2, 200000);
            cudaEventRecord(b); CK(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 10;
            if (delta == 0) base = ms;
            const char* mark = (prev > 0 && ms > prev * 1.15) ? "   <- 多一波" : "";
            printf("      %-10d %-12d %-8.2f %-12.3f %.2fx%s\n",
                   n2, grid, (double)((grid + onewave_blocks - 1) / onewave_blocks), ms,
                   base > 0 ? ms / base : 1.0, mark);
            prev = ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        printf("    -> 元素数只多 1，grid 多 1 个 block 就跨进第二波；\n");
        printf("       第二波里只有那 1 个 block 在跑，其余 169 个 SM 空等（对应 [E] 的台阶）。\n\n");
        cudaFree(d_in); cudaFree(d_out2);
        free(h_in); free(h_out); free(h_ref);
    }

    CK(cudaFree(d_out));
    return 0;
}
