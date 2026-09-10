// L3.2 —— softmax 里的 exp 到底有多贵。
//
// FlashAttention 每一代的取舍，最后都绕回一个比值：
// **一次 exp 的时间里，tensor core 能做多少次乘加。**
// L1.2 已经测出 bf16 tensor core 的纯发射率（crater: 511 FLOP/clk/SM）。
// 这里用同样的方法测 exp 一侧：SFU（特殊功能单元）的发射率。
//
// 测的是**吞吐**不是延迟，所以每个线程维护 8 条互不依赖的链
// （L1.1 踩坑 #11：用依赖链测吞吐会测成延迟）。
//
// 编译：
//   nvcc -O3 -arch=sm_120 exp_throughput.cu -o exp_throughput
// 运行：
//   ./exp_throughput

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); \
    return 1; } } while (0)

#define CHAINS 8

// 每种「运算」一个 kernel，主体结构完全一样，只换中间那一行。
#define MAKE_KERNEL(NAME, OP)                                            \
__global__ void NAME(int iters, float* sink) {                           \
    float a[CHAINS];                                                     \
    for (int c = 0; c < CHAINS; ++c)                                     \
        a[c] = 0.001f * (threadIdx.x + c + 1);                           \
    for (int i = 0; i < iters; ++i) {                                    \
        _Pragma("unroll")                                                \
        for (int c = 0; c < CHAINS; ++c) a[c] = OP;                      \
    }                                                                    \
    float s = 0.f;                                                       \
    for (int c = 0; c < CHAINS; ++c) s += a[c];                          \
    if (s == 1234.5678f) sink[0] = s;   /* 防止被优化掉，但几乎不会执行 */ \
}

// __expf: 走 SFU 的快速版本（PTX 里是 ex2.approx.f32 + 一次乘）
MAKE_KERNEL(k_fast_exp,  __expf(a[c]) * 0.5f + 0.001f)
// expf: 精确版本，走软件展开
MAKE_KERNEL(k_exact_exp, expf(a[c]) * 0.5f + 0.001f)
// exp2f: 底为 2，SFU 原生指令，少一次乘
MAKE_KERNEL(k_exp2,      exp2f(a[c]) * 0.5f + 0.001f)
// 参照：纯 FMA，走普通 FP32 流水
MAKE_KERNEL(k_fma,       a[c] * 1.0001f + 0.001f)
// 参照：rcp（也走 SFU），用来确认 SFU 这条路的宽度
MAKE_KERNEL(k_rcp,       __frcp_rn(a[c]) * 0.5f + 0.001f)

template <typename K>
static double measure(K kernel, int sm_count, int iters, int blocks_per_sm,
                      int threads) {
    float* d_sink = nullptr;
    cudaMalloc(&d_sink, sizeof(float));
    dim3 grid(sm_count * blocks_per_sm), block(threads);
    kernel<<<grid, block>>>(iters / 10 + 1, d_sink);     // 预热
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    kernel<<<grid, block>>>(iters, d_sink);
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaFree(d_sink);
    cudaEventDestroy(a); cudaEventDestroy(b);

    // 总运算次数 = 线程数 × iters × CHAINS
    double ops = (double)grid.x * threads * iters * CHAINS;
    return ops / (ms * 1e-3);                            // ops/s
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int sm = p.multiProcessorCount;
    // CUDA 13 起 cudaDeviceProp 不再有 clockRate，改用属性查询。
    // 这是标称 boost 时钟，不是被测区间的真实频率（L1.2 踩坑 #12）。
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    printf("GPU: %s  sm_%d%d  SM 数 %d  标称时钟 %.2f GHz\n",
           p.name, p.major, p.minor, sm, clock_khz / 1e6);

    const int iters = 20000, threads = 256, bps = 2;
    struct { const char* name; double ops; } r[5];
    r[0] = {"__expf  (SFU 快速)", measure(k_fast_exp,  sm, iters, bps, threads)};
    r[1] = {"exp2f   (SFU 原生)", measure(k_exp2,      sm, iters, bps, threads)};
    r[2] = {"expf    (精确)",     measure(k_exact_exp, sm, iters, bps, threads)};
    r[3] = {"__frcp_rn (SFU)",    measure(k_rcp,       sm, iters, bps, threads)};
    r[4] = {"FMA     (普通流水)", measure(k_fma,       sm, iters, bps, threads)};

    double ghz = clock_khz / 1e6;
    printf("\n%-22s %14s %16s %14s\n", "运算", "G op/s", "每 SM 每周期", "相对 FMA");
    printf("%s\n", "--------------------------------------------------------------------");
    for (int i = 0; i < 5; ++i) {
        double per_sm_clk = r[i].ops / (sm * ghz * 1e9);
        printf("%-22s %14.1f %16.2f %13.3f×\n",
               r[i].name, r[i].ops / 1e9, per_sm_clk, r[i].ops / r[4].ops);
    }

    // ---- 用 FMA 反标定真实时钟 ----
    // FP32 FMA 的硬件发射率是确定的：每 SM 128 条 FP32 lane，即 128/clk/SM。
    // 用它反推被测区间的真实频率，比信标称 boost 时钟可靠（L1.2 踩坑 #12）。
    const double FMA_PER_CLK_SM = 128.0;
    double fma_measured = r[4].ops / (sm * ghz * 1e9);
    double real_ghz = ghz * fma_measured / FMA_PER_CLK_SM;
    double cal = FMA_PER_CLK_SM / fma_measured;
    printf("\n标定：FMA 实测 %.2f/clk/SM，硬件确定值 %.0f/clk/SM\n",
           fma_measured, FMA_PER_CLK_SM);
    printf("      => 被测区间真实频率约 %.2f GHz（标称 %.2f GHz）\n", real_ghz, ghz);

    printf("\n%-22s %20s\n", "运算", "标定后 每 SM 每周期");
    printf("%s\n", "--------------------------------------------");
    for (int i = 0; i < 5; ++i)
        printf("%-22s %20.2f\n", r[i].name,
               r[i].ops / (sm * ghz * 1e9) * cal);

    // ---- 与 tensor core 对照 ----
    const double TC_FLOP_PER_CLK_SM = 511.0;   // L1.2 在同一张卡上实测
    double exp_clk = r[0].ops / (sm * ghz * 1e9) * cal;
    printf("\n参照 L1.2 实测：bf16 tensor core %.0f FLOP/clk/SM（= NVIDIA 标称 512）\n",
           TC_FLOP_PER_CLK_SM);
    printf("SFU 的 exp 标定后约 %.0f/clk/SM。\n", exp_clk);
    printf("=> **一次 exp 的时间里，tensor core 能做约 %.0f 次浮点运算。**\n",
           TC_FLOP_PER_CLK_SM / exp_clk);

    printf("\n对 attention 的账（每个打分元素 S[i,j]）：\n");
    printf("  矩阵乘一侧：QK^T 贡献 2D FLOP，PV 贡献 2D FLOP，合计 4D\n");
    printf("  softmax 一侧：1 次 exp（还有 max/加/乘若干，这里只算 exp）\n");
    for (int D = 32; D <= 256; D *= 2) {
        double t_mm  = 4.0 * D / TC_FLOP_PER_CLK_SM;   // 周期
        double t_exp = 1.0 / exp_clk;
        printf("  D=%3d : 矩阵乘 %.4f clk, exp %.4f clk  -> exp 占 %.1f%%\n",
               D, t_mm, t_exp, 100.0 * t_exp / t_mm);
    }
    printf("\nD 越小，exp 占比越高。而且这个比值会随硬件代际**变差**：\n");
    printf("tensor core 每代大幅加宽，SFU 基本不变。\n");
    printf("（本条趋势只在 sm_120 上测过一个点，跨代对比需要别的卡，见正文「待验证」。）\n");
    return 0;
}
