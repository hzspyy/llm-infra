// L1.2 lab · 直接写 tensor core 指令，量出它的纯发射吞吐。
//
// cuBLAS 给你的 232 TFLOPS 里混着访存、tile 调度、尾效应。
// 想知道 tensor core **本身**能跑多快，就得写一个完全不碰内存的 kernel：
// 数据全在寄存器里，反复发 mma 指令，测纯指令吞吐。
//
// 这个数字的用途：它是 L2.4 里 GEMM 优化阶梯的**真正天花板**。
// 你的 GEMM 达到 cuBLAS 的 90% 不算完，要看离这个纯发射上限还差多少。
//
// 三个实验：
//   A. mma.sync.m16n8k16 bf16 的纯发射吞吐（sm_80 血统，sm_89/sm_120 都支持）
//   B. 同上，fp8 e4m3（m16n8k32，Ada 起支持）
//   C. 打印本卡支持/不支持哪些 tensor core 指令代际
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_120a -o tensor_core tensor_core.cu
//   （注意是 sm_120**a**：架构专属指令需要 a 后缀，见正文）

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// ---------------------------------------------------------------------------
// A. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// 这条指令是**整个 warp 协作**完成一个 16×8×16 的矩阵乘加：
//     D[16×8] = A[16×16] × B[16×8] + C[16×8]
// 32 个线程各自持有 A、B、C 的一小片（fragment），
// 硬件把它们拼成完整矩阵送进 tensor core。
//
// 寄存器布局是**硬件规定死的**：
//   A: 4 个 uint32（每个装 2 个 bf16）  → 每线程 8 个 bf16 元素
//   B: 2 个 uint32                      → 每线程 4 个 bf16 元素
//   C/D: 4 个 float                     → 每线程 4 个 f32 累加值
// 记不住没关系，PTX ISA 文档里有图；重点是**它是寄存器级的，不碰共享内存**。
//
// 每条指令的 FLOP = 2 × 16 × 8 × 16 = 4096
// ---------------------------------------------------------------------------
__global__ void mma_bf16_kernel(int iters, float* sink) {
    uint32_t a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u;
    // ★ 必须用**多组互不依赖的累加器**。
    // 第一版把 8 条 mma 全累加到同一组 {c0..c3} 上，形成串行依赖链，
    // 测出 bf16 251 / fp8 503 TFLOPS —— 而 torch._scaled_mm 实测 fp8 有 560，
    // 真实 GEMM 反超"纯指令上限"，说明上限测错了。
    // tensor core 的 mma 有十几到几十周期的延迟，靠独立累加器才能填满流水线。
    float c[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = 0.f;

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int rep = 0; rep < 4; ++rep)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == 1234.5f) *sink = s;      // 防优化
}

// ---------------------------------------------------------------------------
// B. fp8 e4m3：mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// k 维从 16 翻到 32（元素只有 1 字节，同样的寄存器能装两倍），
// 所以每条指令的 FLOP 也翻倍 = 2 × 16 × 8 × 32 = 8192。
// 这就是 fp8 吞吐是 bf16 两倍的**指令级原因**——不是魔法，是每条指令干了两倍的活。
// ---------------------------------------------------------------------------
#if __CUDA_ARCH__ >= 890
__global__ void mma_fp8_kernel(int iters, float* sink) {
    uint32_t a0 = 0x38383838u, a1 = 0x38383838u, a2 = 0x38383838u, a3 = 0x38383838u;
    uint32_t b0 = 0x38383838u, b1 = 0x38383838u;
    float c[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = 0.f;

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int rep = 0; rep < 4; ++rep)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == 1234.5f) *sink = s;
}
#else
__global__ void mma_fp8_kernel(int, float*) {}
#endif


// ---------------------------------------------------------------------------
// B2. 累加精度的代价：f16 累加 vs f32 累加
//
// 消费级卡有一条历史悠久的限制：tensor core 以 **FP32 累加**运行时只有半速，
// 以 FP16 累加才是全速。数据中心卡没有这个阉割。
// 这条差异是本章的一个悬案的答案——见正文「一个超过物理上限的数字」。
// 代价：f16 累加在长 k 维上会累积舍入误差，训练几乎不能用，推理要看模型敏感度（L4.2）。
// ---------------------------------------------------------------------------
__global__ void mma_bf16_f16acc_kernel(int iters, float* sink) {
    uint32_t a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u;
    uint32_t c[4][2];                      // f16 累加：4 个 half2 = 2 个 uint32
    #pragma unroll
    for (int j = 0; j < 4; ++j) { c[j][0] = 0; c[j][1] = 0; }

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int rep = 0; rep < 4; ++rep)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                : "+r"(c[j][0]), "+r"(c[j][1])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) s ^= c[j][0] ^ c[j][1];
    if (s == 0xdeadbeefu) *sink = 1.f;
}

// ---------------------------------------------------------------------------

template <typename K>
static double measure(K kernel, int sm_count, int iters, double flop_per_mma,
                      int mmas_per_iter, int blocks_per_sm, int warps_per_block) {
    float* d_sink; cudaMalloc(&d_sink, sizeof(float));
    int grid = sm_count * blocks_per_sm, block = warps_per_block * 32;

    kernel<<<grid, block>>>(iters / 10 + 1, d_sink);          // 预热
    cudaDeviceSynchronize();

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    kernel<<<grid, block>>>(iters, d_sink);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(d_sink);

    // 每个 warp 每次迭代发 mmas_per_iter 条指令
    double n_warps = (double)grid * warps_per_block;
    double total_flop = n_warps * iters * mmas_per_iter * flop_per_mma;
    return total_flop / (ms * 1e-3) / 1e12;                   // TFLOPS
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int cc = p.major * 10 + p.minor;
    printf("=== %s  sm_%d  SM=%d\n", p.name, cc, p.multiProcessorCount);

    printf("\n[C] 本卡的 tensor core 指令代际支持\n");
    struct { const char* inst; const char* gen; int min_cc; int max_cc; } gens[] = {
        {"mma.sync.m8n8k4 (f16)",        "Volta  第一代",      70, 999},
        {"mma.sync.m16n8k16 (bf16)",     "Ampere 第三代",      80, 999},
        {"mma.sync.m16n8k32 (fp8)",      "Ada    第四代",      89, 999},
        {"wgmma.mma_async (warp-group)", "Hopper 第四代",      90, 90},
        {"tcgen05.mma (+ TMEM)",         "Blackwell 数据中心", 100, 103},
        {"mma.sync + FP4 (e2m1)",        "Blackwell 消费级",   120, 999},
    };
    for (auto& g : gens) {
        bool ok = cc >= g.min_cc && cc <= g.max_cc;
        printf("    %-32s %-22s %s\n", g.inst, g.gen, ok ? "支持" : "不支持");
    }
    printf("\n    注意 wgmma 与 tcgen05 的 max_cc：它们**不是**向后兼容的通用指令，\n");
    printf("    而是绑定特定架构。sm_120 虽然版本号最高，却用不了 sm_100 的 tcgen05。\n");

    // 扫并发度，取最高吞吐
    printf("\n[A] mma.sync.m16n8k16 bf16 纯发射吞吐（不碰内存）\n");
    printf("    %-14s %-14s %s\n", "warps/block", "blocks/SM", "TFLOPS");
    double best_bf16 = 0;
    for (int wpb : {4, 8}) for (int bps : {1, 2, 4}) {
        double t = measure(mma_bf16_kernel, p.multiProcessorCount, 20000, 4096.0, 16, bps, wpb);
        printf("    %-14d %-14d %.1f\n", wpb, bps, t);
        if (t > best_bf16) best_bf16 = t;
    }
    printf("    峰值 %.1f TFLOPS\n", best_bf16);

    {
        printf("\n[B2] 同一条 mma，改成 f16 累加（消费卡的半速限制在这里）\n");
        double best = 0;
        for (int wpb : {4, 8}) for (int bps : {1, 2, 4}) {
            double x = measure(mma_bf16_f16acc_kernel, p.multiProcessorCount,
                               20000, 4096.0, 16, bps, wpb);
            if (x > best) best = x;
        }
        printf("    f16 累加峰值 %.1f TFLOPS   （f32 累加是 %.1f，比值 %.2f×）\n",
               best, best_bf16, best / best_bf16);
    }

    if (cc >= 89) {
        printf("\n[B] mma.sync.m16n8k32 fp8-e4m3 纯发射吞吐\n");
        printf("    %-14s %-14s %s\n", "warps/block", "blocks/SM", "TFLOPS");
        double best_fp8 = 0;
        for (int wpb : {4, 8}) for (int bps : {1, 2, 4}) {
            double t = measure(mma_fp8_kernel, p.multiProcessorCount, 20000, 8192.0, 16, bps, wpb);
            printf("    %-14d %-14d %.1f\n", wpb, bps, t);
            if (t > best_fp8) best_fp8 = t;
        }
        printf("    峰值 %.1f TFLOPS   （是 bf16 的 %.2f×）\n", best_fp8, best_fp8 / best_bf16);
        printf("    指令级解释：k 维 16→32，每条指令的 FLOP 从 4096 翻到 8192。\n");
    }

    printf("\n");
    return 0;
}
