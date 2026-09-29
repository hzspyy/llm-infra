// L1.2 lab · tensor core 指令的发射阶梯：延迟、发射间隔、累加精度与 fragment 排列。
//
// tensor_core.cu 测的是「峰值 TFLOPS」，用墙钟除以 FLOP，频率靠 NVML 采样。
// 这个脚本换一种测法：**在 kernel 内部用 clock64() 数周期**，直接得到
// FLOP/clk/SM，不依赖任何频率采样，也就没有「采到空闲频率」这类误差。
//
// 四组实验：
//   [A] fragment 排列验证：按 PTX ISA 的 lane 映射把已知矩阵装进寄存器，
//       跑一条 mma，和 CPU FP64 参照逐元素对拍。排列错了这一步就会炸。
//   [B] 依赖链阶梯：独立累加器 1/2/4/8/16 条 × warp 数 4/8/16。
//       链数=1 量到的是指令延迟，链数足够多时量到的是发射间隔。
//   [C] 累加精度与数据格式：bf16/fp8 各自的 f32 累加与 f16 累加。
//   [D] 指令族编译探针：把候选 PTX 指令逐条送进 ptxas，记录支持与报错原文。
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -o tensor_core_ladder tensor_core_ladder.cu
// 运行：
//   ./tensor_core_ladder            # 全部四组
//   ./tensor_core_ladder --json out.json

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// ===========================================================================
// [A] fragment 排列验证
// ===========================================================================
// m16n8k16（bf16）的 lane 映射，取自 PTX ISA 的 mma 矩阵片段图：
//   groupID = lane >> 2      threadID_in_group = lane & 3
//   A(16x16): a0 -> (row=groupID,    col=tig*2+{0,1})
//             a1 -> (row=groupID+8,  col=tig*2+{0,1})
//             a2 -> (row=groupID,    col=tig*2+8+{0,1})
//             a3 -> (row=groupID+8,  col=tig*2+8+{0,1})
//   B(16x8) : b0 -> (k=tig*2+{0,1},    n=groupID)
//             b1 -> (k=tig*2+8+{0,1},  n=groupID)
//   D(16x8) : d0,d1 -> (row=groupID,   col=tig*2+{0,1})
//             d2,d3 -> (row=groupID+8, col=tig*2+{0,1})
__global__ void verify_bf16(const float* A, const float* B, float* D) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    auto pack = [](float lo, float hi) {
        __nv_bfloat16 l = __float2bfloat16(lo), h = __float2bfloat16(hi);
        uint32_t r;
        uint16_t lb = *reinterpret_cast<uint16_t*>(&l), hb = *reinterpret_cast<uint16_t*>(&h);
        r = (uint32_t)lb | ((uint32_t)hb << 16);
        return r;
    };
    uint32_t a0 = pack(A[g * 16 + t * 2], A[g * 16 + t * 2 + 1]);
    uint32_t a1 = pack(A[(g + 8) * 16 + t * 2], A[(g + 8) * 16 + t * 2 + 1]);
    uint32_t a2 = pack(A[g * 16 + t * 2 + 8], A[g * 16 + t * 2 + 9]);
    uint32_t a3 = pack(A[(g + 8) * 16 + t * 2 + 8], A[(g + 8) * 16 + t * 2 + 9]);
    uint32_t b0 = pack(B[(t * 2) * 8 + g], B[(t * 2 + 1) * 8 + g]);
    uint32_t b1 = pack(B[(t * 2 + 8) * 8 + g], B[(t * 2 + 9) * 8 + g]);
    float d[4] = {0, 0, 0, 0};
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    D[g * 8 + t * 2] = d[0];
    D[g * 8 + t * 2 + 1] = d[1];
    D[(g + 8) * 8 + t * 2] = d[2];
    D[(g + 8) * 8 + t * 2 + 1] = d[3];
}

// m16n8k32（fp8 e4m3）：每个寄存器装 4 个字节，A 每线程 16 个元素。
__global__ void verify_fp8(const float* A, const float* B, float* D) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    auto pack4 = [](float x0, float x1, float x2, float x3) {
        __nv_fp8_e4m3 v[4] = {__nv_fp8_e4m3(x0), __nv_fp8_e4m3(x1),
                              __nv_fp8_e4m3(x2), __nv_fp8_e4m3(x3)};
        uint32_t r = 0;
        for (int i = 0; i < 4; ++i)
            r |= (uint32_t)(*reinterpret_cast<uint8_t*>(&v[i])) << (8 * i);
        return r;
    };
    uint32_t a0 = pack4(A[g * 32 + t * 4], A[g * 32 + t * 4 + 1],
                        A[g * 32 + t * 4 + 2], A[g * 32 + t * 4 + 3]);
    uint32_t a1 = pack4(A[(g + 8) * 32 + t * 4], A[(g + 8) * 32 + t * 4 + 1],
                        A[(g + 8) * 32 + t * 4 + 2], A[(g + 8) * 32 + t * 4 + 3]);
    uint32_t a2 = pack4(A[g * 32 + t * 4 + 16], A[g * 32 + t * 4 + 17],
                        A[g * 32 + t * 4 + 18], A[g * 32 + t * 4 + 19]);
    uint32_t a3 = pack4(A[(g + 8) * 32 + t * 4 + 16], A[(g + 8) * 32 + t * 4 + 17],
                        A[(g + 8) * 32 + t * 4 + 18], A[(g + 8) * 32 + t * 4 + 19]);
    uint32_t b0 = pack4(B[(t * 4) * 8 + g], B[(t * 4 + 1) * 8 + g],
                        B[(t * 4 + 2) * 8 + g], B[(t * 4 + 3) * 8 + g]);
    uint32_t b1 = pack4(B[(t * 4 + 16) * 8 + g], B[(t * 4 + 17) * 8 + g],
                        B[(t * 4 + 18) * 8 + g], B[(t * 4 + 19) * 8 + g]);
    float d[4] = {0, 0, 0, 0};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    D[g * 8 + t * 2] = d[0];
    D[g * 8 + t * 2 + 1] = d[1];
    D[(g + 8) * 8 + t * 2] = d[2];
    D[(g + 8) * 8 + t * 2 + 1] = d[3];
}

// ===========================================================================
// [B][C] 发射阶梯：链数模板化，kernel 内部数周期
// ===========================================================================
#define MMA_BF16_F32(D0, D1, D2, D3)                                        \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"  \
                 : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                   \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// bf16 只有 f32 累加一种形式：写成 .f16.bf16.bf16.f16 时 ptxas 直接拒绝
// （"Unexpected instruction types specified for 'mma'"），见 [D] 的编译探针。

#define MMA_F16_F16(H0, H1)                                                 \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "       \
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"              \
                 : "+r"(H0), "+r"(H1)                                       \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

#define MMA_FP8_F32(D0, D1, D2, D3)                                         \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "     \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"  \
                 : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                   \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

#define MMA_FP8_F16(H0, H1)                                                 \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "     \
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"              \
                 : "+r"(H0), "+r"(H1)                                       \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// Blackwell 新增的 kind:: 前缀：同样的 m16n8k32，走的是另一条译码路径
#define MMA_KIND_F8_F32(D0, D1, D2, D3)                                     \
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32."      \
                 "e4m3.e4m3.f32 "                                           \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"  \
                 : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                   \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// block_scale：每 32 个元素共享一个 ue8m0 指数因子，scale 从额外的寄存器读
#define MMA_MXF8_BLOCK(D0, D1, D2, D3)                                      \
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."\
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "   \
                 "{%10}, {0,0}, {%11}, {0,0};\n"                            \
                 : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                   \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),    \
                   "r"(sa), "r"(sb))

// FP4：k 维再翻一倍到 64，每条指令 16384 FLOP
#define MMA_MXF4_BLOCK(D0, D1, D2, D3)                                      \
    asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X."   \
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "   \
                 "{%10}, {0,0}, {%11}, {0,0};\n"                            \
                 : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                   \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),    \
                   "r"(sa), "r"(sb))

enum Variant { V_BF16_F32, V_F16_F16, V_FP8_F32, V_FP8_F16,
               V_KIND_F8_F32, V_MXF8_BLOCK, V_MXF4_BLOCK };

template <int CHAINS, Variant V>
__global__ void ladder_kernel(int iters, float* sink, unsigned long long* cycles) {
    uint32_t a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;   // ue8m0 的 1.0（指数偏置 127）
    if (V != V_BF16_F32 && V != V_F16_F16) {
        a0 = a1 = a2 = a3 = 0x38383838u;      // e4m3 的 1.0
        b0 = b1 = 0x38383838u;
    }
    float f32acc[CHAINS][4];
    uint32_t f16acc[CHAINS][2];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
        f32acc[c][0] = f32acc[c][1] = f32acc[c][2] = f32acc[c][3] = 0.f;
        f16acc[c][0] = f16acc[c][1] = 0u;
    }
    __syncthreads();
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            if (V == V_BF16_F32) { MMA_BF16_F32(f32acc[c][0], f32acc[c][1], f32acc[c][2], f32acc[c][3]); }
            else if (V == V_F16_F16) { MMA_F16_F16(f16acc[c][0], f16acc[c][1]); }
            else if (V == V_FP8_F32) { MMA_FP8_F32(f32acc[c][0], f32acc[c][1], f32acc[c][2], f32acc[c][3]); }
            else if (V == V_FP8_F16) { MMA_FP8_F16(f16acc[c][0], f16acc[c][1]); }
            else if (V == V_KIND_F8_F32) { MMA_KIND_F8_F32(f32acc[c][0], f32acc[c][1], f32acc[c][2], f32acc[c][3]); }
            else if (V == V_MXF8_BLOCK) { MMA_MXF8_BLOCK(f32acc[c][0], f32acc[c][1], f32acc[c][2], f32acc[c][3]); }
            else { MMA_MXF4_BLOCK(f32acc[c][0], f32acc[c][1], f32acc[c][2], f32acc[c][3]); }
        }
    }
    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
    float s = 0;
    uint32_t u = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
        s += f32acc[c][0] + f32acc[c][1] + f32acc[c][2] + f32acc[c][3];
        u ^= f16acc[c][0] ^ f16acc[c][1];
    }
    // 无条件写回：让累加器成为真正的输出，编译器不能把 mma 循环消掉
    if (threadIdx.x == 0) sink[blockIdx.x] = s + (float)u;
}

struct Result {
    double tflops, cycles_per_mma, flop_per_clk_sm, ms;
    unsigned long long cycles;
};

template <int CHAINS, Variant V>
Result run_ladder(int sm_count, int iters, double flop_per_mma,
                  int blocks_per_sm, int warps_per_block) {
    int grid = sm_count * blocks_per_sm, block = warps_per_block * 32;
    float* d_sink;
    unsigned long long* d_cyc;
    cudaMalloc(&d_sink, sizeof(float) * grid);
    cudaMalloc(&d_cyc, sizeof(unsigned long long) * grid);
    ladder_kernel<CHAINS, V><<<grid, block>>>(iters / 10 + 1, d_sink, d_cyc);
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    ladder_kernel<CHAINS, V><<<grid, block>>>(iters, d_sink, d_cyc);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);

    std::vector<unsigned long long> cyc(grid);
    cudaMemcpy(cyc.data(), d_cyc, sizeof(unsigned long long) * grid, cudaMemcpyDeviceToHost);
    unsigned long long med = 0;
    for (auto c : cyc) med += c;
    med /= grid;                                   // 各 block 的平均周期数

    double n_warps = (double)grid * warps_per_block;
    double total_flop = n_warps * iters * CHAINS * flop_per_mma;
    Result r;
    r.ms = ms;
    r.cycles = med;
    r.tflops = total_flop / (ms * 1e-3) / 1e12;
    r.cycles_per_mma = (double)med / ((double)iters * CHAINS);   // 每 warp 每条 mma 的周期
    // 一个 SM 上同时有 blocks_per_sm 个 block、每 block warps_per_block 个 warp
    double flop_per_sm = (double)blocks_per_sm * warps_per_block * iters * CHAINS * flop_per_mma;
    r.flop_per_clk_sm = flop_per_sm / (double)med;
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(d_sink); cudaFree(d_cyc);
    return r;
}

static const char* vname(Variant v) {
    switch (v) {
        case V_BF16_F32: return "m16n8k16 bf16 -> f32";
        case V_F16_F16: return "m16n8k16 f16  -> f16";
        case V_FP8_F32: return "m16n8k32 e4m3 -> f32";
        case V_FP8_F16: return "m16n8k32 e4m3 -> f16";
        case V_KIND_F8_F32: return "kind::f8f6f4 m16n8k32 e4m3 -> f32";
        case V_MXF8_BLOCK: return "kind::mxf8f6f4.block_scale m16n8k32 e4m3 -> f32";
        default: return "kind::mxf4.block_scale m16n8k64 e2m1 -> f32";
    }
}

template <Variant V>
void sweep(int sm, double flop, int iters, FILE* js, const char* tag) {
    printf("\n    %s  （%.0f FLOP/指令）\n", vname(V), flop);
    printf("    %-7s %-7s %-9s %-11s %-13s %s\n",
           "链数", "warps", "TFLOPS", "周期/指令", "FLOP/clk/SM", "ms");
    struct Row { int chains, warps; Result r; };
    std::vector<Row> rows;
#define ONE(CH)                                                                  \
    for (int w : {4, 8, 16}) {                                                   \
        Result r = run_ladder<CH, V>(sm, iters, flop, 1, w);                      \
        rows.push_back({CH, w, r});                                              \
        printf("    %-7d %-7d %-9.1f %-11.2f %-13.1f %.2f\n",                     \
               CH, w, r.tflops, r.cycles_per_mma, r.flop_per_clk_sm, r.ms);       \
    }
    ONE(1) ONE(2) ONE(4) ONE(8) ONE(16)
#undef ONE
    // 单 warp、单链：整卡只有一个 warp 在发，量到的是纯依赖延迟
    Result lat = run_ladder<1, V>(sm, 4000, flop, 1, 1);
    double best = 0;
    for (auto& x : rows) if (x.r.flop_per_clk_sm > best) best = x.r.flop_per_clk_sm;
    printf("    单 warp 单链（纯延迟）%.2f 周期/指令；"
           "峰值 %.1f FLOP/clk/SM（%.1f 倍于延迟界）\n",
           lat.cycles_per_mma, best, best / (flop / lat.cycles_per_mma));
    if (js) {
        fprintf(js, "  \"%s\": {\"flop_per_mma\": %.0f, \"peak_flop_per_clk_sm\": %.2f,"
                    " \"latency_cycles_1warp_1chain\": %.3f, \"rows\": [",
                tag, flop, best, lat.cycles_per_mma);
        for (size_t i = 0; i < rows.size(); ++i)
            fprintf(js, "%s{\"chains\": %d, \"warps\": %d, \"tflops\": %.3f,"
                        " \"cycles_per_mma\": %.4f, \"flop_per_clk_sm\": %.3f, \"ms\": %.4f}",
                    i ? ", " : "", rows[i].chains, rows[i].warps, rows[i].r.tflops,
                    rows[i].r.cycles_per_mma, rows[i].r.flop_per_clk_sm, rows[i].r.ms);
        fprintf(js, "]},\n");
    }
}

int main(int argc, char** argv) {
    const char* json_path = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    FILE* js = json_path ? fopen(json_path, "w") : nullptr;

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int sm = p.multiProcessorCount;
    int clk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    printf("=== %s  sm_%d%d  SM=%d  标称时钟 %.2f GHz\n",
           p.name, p.major, p.minor, sm, clk_khz / 1e6);
    if (js) fprintf(js, "{\n  \"gpu\": \"%s\", \"sm_count\": %d, \"cc\": \"%d.%d\",\n",
                    p.name, sm, p.major, p.minor);

    // ---------------- [A] fragment 排列验证 ----------------
    printf("\n[A] fragment 排列验证：按 ISA 的 lane 映射装矩阵，和 CPU FP64 对拍\n");
    {
        float hA16[16 * 16], hB16[16 * 8], hD16[16 * 8];
        for (int i = 0; i < 16; ++i)
            for (int k = 0; k < 16; ++k) hA16[i * 16 + k] = (float)((i * 3 + k * 5) % 7 - 3);
        for (int k = 0; k < 16; ++k)
            for (int j = 0; j < 8; ++j) hB16[k * 8 + j] = (float)((k * 2 + j * 3) % 5 - 2);
        float *dA, *dB, *dD;
        CHECK(cudaMalloc(&dA, sizeof(hA16))); CHECK(cudaMalloc(&dB, sizeof(hB16)));
        CHECK(cudaMalloc(&dD, sizeof(hD16)));
        CHECK(cudaMemcpy(dA, hA16, sizeof(hA16), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB16, sizeof(hB16), cudaMemcpyHostToDevice));
        verify_bf16<<<1, 32>>>(dA, dB, dD);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hD16, dD, sizeof(hD16), cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double ref = 0;
                for (int k = 0; k < 16; ++k) ref += (double)hA16[i * 16 + k] * hB16[k * 8 + j];
                worst = fmax(worst, fabs(ref - hD16[i * 8 + j]));
            }
        printf("    m16n8k16 bf16：16x8 输出与 FP64 参照的最大绝对差 %.3e"
               "（整数输入，bf16 可精确表示）\n", worst);
        printf("    D[0][0]=%.1f D[0][1]=%.1f D[8][0]=%.1f D[15][7]=%.1f\n",
               hD16[0], hD16[1], hD16[8 * 8], hD16[15 * 8 + 7]);
        if (js) fprintf(js, "  \"verify_bf16_max_abs_err\": %.6e,\n", worst);
        cudaFree(dA); cudaFree(dB); cudaFree(dD);
    }
    {
        float hA[16 * 32], hB[32 * 8], hD[16 * 8];
        for (int i = 0; i < 16; ++i)
            for (int k = 0; k < 32; ++k) hA[i * 32 + k] = (float)((i + k) % 5 - 2);
        for (int k = 0; k < 32; ++k)
            for (int j = 0; j < 8; ++j) hB[k * 8 + j] = (float)((k * 3 + j) % 3 - 1);
        float *dA, *dB, *dD;
        CHECK(cudaMalloc(&dA, sizeof(hA))); CHECK(cudaMalloc(&dB, sizeof(hB)));
        CHECK(cudaMalloc(&dD, sizeof(hD)));
        CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        verify_fp8<<<1, 32>>>(dA, dB, dD);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double ref = 0;
                for (int k = 0; k < 32; ++k) ref += (double)hA[i * 32 + k] * hB[k * 8 + j];
                worst = fmax(worst, fabs(ref - hD[i * 8 + j]));
            }
        printf("    m16n8k32 e4m3：最大绝对差 %.3e（|值| ≤ 2，e4m3 可精确表示小整数）\n", worst);
        printf("    D[0][0]=%.1f D[0][1]=%.1f D[8][0]=%.1f D[15][7]=%.1f\n",
               hD[0], hD[1], hD[8 * 8], hD[15 * 8 + 7]);
        printf("    排列写错时这一步会出现成片的错值，而不是最后一位的舍入差。\n");
        if (js) fprintf(js, "  \"verify_fp8_max_abs_err\": %.6e,\n", worst);
        cudaFree(dA); cudaFree(dB); cudaFree(dD);
    }

    // ---------------- [B][C] 发射阶梯 ----------------
    const int iters = 20000;
    printf("\n[B] 依赖链阶梯：链数=1 是延迟，链数足够多是发射间隔\n");
    printf("    周期由 kernel 内的 clock64() 直接数，不经过 NVML 频率采样。\n");
    sweep<V_BF16_F32>(sm, 4096.0, iters, js, "bf16_f32acc");
    sweep<V_F16_F16>(sm, 4096.0, iters, js, "f16_f16acc");
    sweep<V_FP8_F32>(sm, 8192.0, iters, js, "fp8_f32acc");
    sweep<V_FP8_F16>(sm, 8192.0, iters, js, "fp8_f16acc");
    sweep<V_KIND_F8_F32>(sm, 8192.0, iters, js, "kind_f8f6f4_f32acc");
    sweep<V_MXF8_BLOCK>(sm, 8192.0, iters, js, "mxf8f6f4_block_scale_f32acc");
    sweep<V_MXF4_BLOCK>(sm, 16384.0, iters, js, "mxf4_block_scale_f32acc");

    if (js) { fprintf(js, "  \"iters\": %d\n}\n", iters); fclose(js); }
    printf("\n");
    return 0;
}
