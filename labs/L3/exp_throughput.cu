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

// ---------------------------------------------------------------------------
// 软件 exp2：把指数从 SFU 挪到 FMA 流水上（FA4 针对 Blackwell 的做法）。
// 拆成整数部分 n 与小数部分 f，f 上用多项式，n 用位运算拼回指数。
// 每个元素约 1 次 rint + 1 次减 + 6 次 FMA + 1 次整数移位 + 1 次乘。
// ---------------------------------------------------------------------------
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float soft_exp2(float x) {
    x = fminf(fmaxf(x, -126.0f), 126.0f);
    float n = rintf(x);
    float f = x - n;                       // f ∈ [-0.5, 0.5]
    // 2^f 的 6 阶多项式（Taylor 系数 ln2^k/k!）
    float p = 1.5403530393381610e-4f;
    p = fmaf(p, f, 1.3333558146428443e-3f);
    p = fmaf(p, f, 9.6181291076284770e-3f);
    p = fmaf(p, f, 5.5504108664821580e-2f);
    p = fmaf(p, f, 2.4022650695910070e-1f);
    p = fmaf(p, f, 6.9314718055994530e-1f);
    p = fmaf(p, f, 1.0f);
    int e = (int)n;
    float scale = __int_as_float((e + 127) << 23);
    return p * scale;
}

MAKE_KERNEL(k_soft_exp2, soft_exp2(a[c]) * 0.5f + 0.001f)
MAKE_KERNEL(k_soft_exp,  soft_exp2(a[c] * LOG2E) * 0.5f + 0.001f)

// 低阶版本：4 阶多项式（5 次 FMA），换更少的 FMA 占用。精度会差一档。
__device__ __forceinline__ float soft_exp2_d4(float x) {
    x = fminf(fmaxf(x, -126.0f), 126.0f);
    float n = rintf(x);
    float f = x - n;
    float p = 9.6181291076284770e-3f;
    p = fmaf(p, f, 5.5504108664821580e-2f);
    p = fmaf(p, f, 2.4022650695910070e-1f);
    p = fmaf(p, f, 6.9314718055994530e-1f);
    p = fmaf(p, f, 1.0f);
    int e = (int)n;
    return p * __int_as_float((e + 127) << 23);
}
MAKE_KERNEL(k_soft_exp2_d4, soft_exp2_d4(a[c]) * 0.5f + 0.001f)

// 两条流水并用：一半链走 SFU 的 exp2f，一半链走 FMA 多项式。
// SFU 与 FMA 是两条独立流水，理论上限是两者之和。
__global__ void k_mixed_exp(int iters, float* sink) {
    float a[CHAINS];
    for (int c = 0; c < CHAINS; ++c) a[c] = 0.001f * (threadIdx.x + c + 1);
    for (int i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int c = 0; c < CHAINS; ++c) {
            if (c % 2 == 0) a[c] = exp2f(a[c]) * 0.5f + 0.001f;
            else            a[c] = soft_exp2(a[c]) * 0.5f + 0.001f;
        }
    }
    float s = 0.f;
    for (int c = 0; c < CHAINS; ++c) s += a[c];
    if (s == 1234.5678f) sink[0] = s;
}

// 精度：在 [-30, 0] 上与 exp2f 比相对误差（attention 里减完最大值后的典型区间）
// D4=false 用 6 阶多项式，D4=true 用 4 阶；用模板参数而不是设备函数指针，
// 在较老的 nvcc 上也能编（设备端调用 __global__ 需要 rdc，这里不用）。
template <bool D4>
__global__ void k_soft_exp2_err(float lo, float hi, int n, float* out) {
    float worst = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float x = lo + (hi - lo) * (float)i / (float)(n - 1);
        float ref = exp2f(x);
        float got = D4 ? soft_exp2_d4(x) : soft_exp2(x);
        float rel = fabsf(got - ref) / ref;
        worst = fmaxf(worst, rel);
    }
    atomicMax((int*)out, __float_as_int(worst));
}

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
    struct { const char* name; double ops; } r[9];
    r[0] = {"__expf  (SFU 快速)", measure(k_fast_exp,  sm, iters, bps, threads)};
    r[1] = {"exp2f   (SFU 原生)", measure(k_exp2,      sm, iters, bps, threads)};
    r[2] = {"expf    (精确)",     measure(k_exact_exp, sm, iters, bps, threads)};
    r[3] = {"__frcp_rn (SFU)",    measure(k_rcp,       sm, iters, bps, threads)};
    r[4] = {"FMA     (普通流水)", measure(k_fma,       sm, iters, bps, threads)};
    r[5] = {"soft_exp2 d6 (FMA)", measure(k_soft_exp2, sm, iters, bps, threads)};
    r[6] = {"soft_expf d6 (FMA)", measure(k_soft_exp,  sm, iters, bps, threads)};
    r[7] = {"soft_exp2 d4 (FMA)", measure(k_soft_exp2_d4, sm, iters, bps, threads)};
    r[8] = {"SFU+FMA 并用",       measure(k_mixed_exp, sm, iters, bps, threads)};

    double ghz = clock_khz / 1e6;
    printf("\n%-24s %14s %16s %14s\n", "运算", "G op/s", "每 SM 每周期", "相对 FMA");
    printf("%s\n", "--------------------------------------------------------------------");
    for (int i = 0; i < 9; ++i) {
        double per_sm_clk = r[i].ops / (sm * ghz * 1e9);
        printf("%-24s %14.1f %16.2f %13.3f×\n",
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

    printf("\n%-24s %20s\n", "运算", "标定后 每 SM 每周期");
    printf("%s\n", "--------------------------------------------");
    for (int i = 0; i < 9; ++i)
        printf("%-24s %20.2f\n", r[i].name,
               r[i].ops / (sm * ghz * 1e9) * cal);

    // ---- 软件 exp2 的精度 ----
    {
        float* d_err = nullptr;
        cudaMalloc(&d_err, sizeof(float));
        float zero = 0.f;
        cudaMemcpy(d_err, &zero, sizeof(float), cudaMemcpyHostToDevice);
        k_soft_exp2_err<false><<<sm * 4, 256>>>(0.f, -30.f, 1 << 20, d_err);
        float worst = 0.f;
        cudaMemcpy(&worst, d_err, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_err, &zero, sizeof(float), cudaMemcpyHostToDevice);
        k_soft_exp2_err<true><<<sm * 4, 256>>>(0.f, -30.f, 1 << 20, d_err);
        float worst4 = 0.f;
        cudaMemcpy(&worst4, d_err, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_err);
        printf("\nsoft_exp2 在 [-30, 0] 上的最大相对误差：d6 = %.3e，d4 = %.3e"
               "（fp32 机器 epsilon %.1e）\n", worst, worst4, 1.1920929e-7f);
    }

    // ---- 与 tensor core 对照 ----
    const double TC_FLOP_PER_CLK_SM = 511.0;   // L1.2 在 sm_120 上实测；换卡须替换
    double exp_clk = r[0].ops / (sm * ghz * 1e9) * cal;
    printf("\n参照 L1.2 实测：bf16 tensor core %.0f FLOP/clk/SM（= NVIDIA 标称 512，"
           "此常数只在 sm_120 上实测，换卡须替换）\n", TC_FLOP_PER_CLK_SM);
    printf("SFU 的 exp 标定后约 %.0f/clk/SM。\n", exp_clk);
    printf("=> **一次 exp 的时间里，tensor core 能做约 %.0f 次浮点运算。**\n",
           TC_FLOP_PER_CLK_SM / exp_clk);
    printf("软件 exp2 标定后约 %.2f 次/clk/SM。\n",
           r[5].ops / (sm * ghz * 1e9) * cal);
    printf("  每个 soft_exp2 约 10 条 FMA 流水的指令，换算成 FMA 占用约 %.0f/clk/SM，"
           "与硬件 FMA 宽度 %.0f 对照即可看出它挤占的是哪条流水。\n",
           r[5].ops / (sm * ghz * 1e9) * cal * 10.0, FMA_PER_CLK_SM);

    printf("\n对 attention 的账（每个打分元素 S[i,j]）：\n");
    printf("  矩阵乘一侧：QK^T 贡献 2D FLOP，PV 贡献 2D FLOP，合计 4D\n");
    printf("  softmax 一侧：1 次 exp（还有 max/加/乘若干，这里只算 exp）\n");
    for (int D = 32; D <= 256; D *= 2) {
        double t_mm  = 4.0 * D / TC_FLOP_PER_CLK_SM;   // 周期
        double t_exp = 1.0 / exp_clk;
        double t_soft = 1.0 / (r[5].ops / (sm * ghz * 1e9) * cal);
        printf("  D=%3d : 矩阵乘 %.4f clk, SFU exp %.4f clk（%.1f%%）"
               ", FMA 软件 exp %.4f clk（%.1f%%）\n",
               D, t_mm, t_exp, 100.0 * t_exp / t_mm,
               t_soft, 100.0 * t_soft / t_mm);
    }
    printf("\nD 越小，exp 占比越高。SFU 宽度不随代际变化，而 tensor core 每代加宽；\n");
    printf("把 exp 挪到 FMA 流水上（soft_exp2）是另一条路，代价是占掉 FMA 的发射槽。\n");
    printf("跨代对照：同一份源码在 sm_89 / sm_120 / sm_121 上各跑一次，比 exp 与 FMA 的比值。\n");
    return 0;
}
