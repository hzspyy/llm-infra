// L2.4-D lab · persistent scheduling、tile 遍历顺序与寄存器累加器预算（sm_120）
//
// 2.4-A/B 的阶梯默认"一个 block 一个 tile"（静态 grid，tile→block 的映射由 grid
// 的编号方式与硬件发块顺序共同决定）。数据中心 Blackwell（sm_100）的 CUTLASS
// kernel 普遍使用 persistent 调度：只发射常驻 block 数（SM × 每 SM 驻留数），
// 每个 block 在内部循环领取多个 tile。
//
// 本 lab 把两个常被混在一起的变量分开，全部共用同一个 gemm_body<...>：
//
//   1) tile 遍历顺序（rasterization）：连续 tile id 沿 M 走还是沿 N 走，
//      决定相邻 block 复用哪一块 panel，直接影响 L2 命中。
//   2) 调度方式：静态 grid（硬件动态发块）还是 persistent（软件 round-robin 领块）。
//
// 第三组只改每线程累加器规模（WM×WN）。sm_120 的累加器住在寄存器里，
// 规模直接换算成寄存器压力与驻留 block 数；sm_100 的 tcgen05 把累加器放在
// TMEM（128 行 × 512 列 × 32 bit = 256 KB/SM，见 include/cute/arch/
// tmem_capacity_sm100.hpp），走的是另一套预算。
//
//   bash labs/L2/run_persistent_gemm.sh <out_dir>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("[cuBLAS] %s @%d: %d\n", #x, __LINE__, (int)s); exit(1);} } while(0)

constexpr int M = 4096, N = 4096, K = 4096;

// ---------------------------------------------------------------------------
// 共享的 tile 计算体：C[cRow..][cCol..] += A[cRow..][:] * B[:][cCol..]
// 与 2.4 的 v6 相同（bf16 WMMA、smem padding、float4 向量化）。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int PAD>
__device__ __forceinline__ void gemm_body(
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
        float* __restrict__ C, int cRow, int cCol) {
    constexpr int LDA = BK + PAD, LDB = BN + PAD;
    __shared__ __nv_bfloat16 sA[BM * LDA], sB[BK * LDB];
    const int warp = threadIdx.x / 32;
    const int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    const int threads = blockDim.x;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) nvcuda::wmma::fill_fragment(acc[i][j], 0.0f);

    for (int t = 0; t < K; t += BK) {
        for (int idx = threadIdx.x * 8; idx < BM * BK; idx += threads * 8) {
            int r = idx / BK, c = idx % BK;
            *(float4*)&sA[r * LDA + c] = *(const float4*)&A[(cRow + r) * K + t + c];
        }
        for (int idx = threadIdx.x * 8; idx < BK * BN; idx += threads * 8) {
            int r = idx / BN, c = idx % BN;
            *(float4*)&sB[r * LDB + c] = *(const float4*)&B[(t + r) * N + cCol + c];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> fa;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> fb;
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                nvcuda::wmma::load_matrix_sync(fa, &sA[(wRow * WM + i * 16) * LDA + k], LDA);
                #pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
                    nvcuda::wmma::load_matrix_sync(fb, &sB[k * LDB + wCol * WN + j * 16], LDB);
                    nvcuda::wmma::mma_sync(acc[i][j], fa, fb, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j)
            nvcuda::wmma::store_matrix_sync(
                &C[(cRow + wRow * WM + i * 16) * N + cCol + wCol * WN + j * 16],
                acc[i][j], N, nvcuda::wmma::mem_row_major);
}

// 静态 grid：一个 block 一个 tile。TN_FAST=true 让连续 blockIdx.x 沿 N 方向走。
template <int BM, int BN, int BK, int WM, int WN, int PAD, bool TN_FAST>
__global__ void gemm_static(const __nv_bfloat16* __restrict__ A,
                           const __nv_bfloat16* __restrict__ B,
                           float* __restrict__ C) {
    const int tm = TN_FAST ? blockIdx.y : blockIdx.x;
    const int tn = TN_FAST ? blockIdx.x : blockIdx.y;
    gemm_body<BM, BN, BK, WM, WN, PAD>(A, B, C, tm * BM, tn * BN);
}

// persistent：grid 是常驻 block 数，block 内部按 round-robin 领 tile。
// TN_FAST=true 时 t%TILES_N 是 N 方向的 tile —— 与静态版同样的"沿 N 连续"。
template <int BM, int BN, int BK, int WM, int WN, int PAD, bool TN_FAST>
__global__ void gemm_persistent(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                float* __restrict__ C) {
    constexpr int TILES_M = M / BM, TILES_N = N / BN, TILES = TILES_M * TILES_N;
    for (int t = blockIdx.x; t < TILES; t += gridDim.x) {
        const int tm = TN_FAST ? t / TILES_N : t % TILES_M;
        const int tn = TN_FAST ? t % TILES_N : t / TILES_M;
        gemm_body<BM, BN, BK, WM, WN, PAD>(A, B, C, tm * BM, tn * BN);
        __syncthreads();   // smem 在 tile 之间复用，先确认全 block 读完
    }
}

static double gflop() { return 2.0 * M * N * K / 1e9; }

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int sms = p.multiProcessorCount;
    printf("=== %s  SM=%d  GEMM %dx%dx%d = %.1f GFLOP\n", p.name, sms, M, N, K, gflop());

    constexpr int BM = 128, BN = 128, BK = 32, WM = 64, WN = 32, PAD = 8;
    constexpr int TILES = (M / BM) * (N / BN);
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
    constexpr int WARMUP = 10, ROUNDS = 5, REPS = 20;

    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, gemm_persistent<BM, BN, BK, WM, WN, PAD, true>, THREADS, 0));
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, gemm_persistent<BM, BN, BK, WM, WN, PAD, true>));
    printf("    tile %dx%dx%d  WM×WN=%d×%d  %d 线程  寄存器 %d/线程  smem %zu B  "
           "驻留 %d block/SM\n",
           BM, BN, BK, WM, WN, THREADS, attr.numRegs, attr.sharedSizeBytes, occ);
    printf("    ⇒ 常驻槽位 %d，tile 总数 %d（%.2f 波）；预热 %d 次，%d 轮交错 × %d 次计时\n\n",
           occ * sms, TILES, (double)TILES / (occ * sms), WARMUP, ROUNDS, REPS);

    // ---- 输入与参照 ----
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = (rand() % 100 - 50) / 100.0f;
    for (auto& v : hB) v = (rand() % 100 - 50) / 100.0f;
    __nv_bfloat16 *bA, *bB; float *dC, *dRef;
    CK(cudaMalloc(&bA, (size_t)M * K * 2)); CK(cudaMalloc(&bB, (size_t)K * N * 2));
    CK(cudaMalloc(&dC, (size_t)M * N * 4)); CK(cudaMalloc(&dRef, (size_t)M * N * 4));
    {
        std::vector<__nv_bfloat16> tA((size_t)M * K), tB((size_t)K * N);
        for (size_t i = 0; i < tA.size(); ++i) tA[i] = __float2bfloat16(hA[i]);
        for (size_t i = 0; i < tB.size(); ++i) tB[i] = __float2bfloat16(hB[i]);
        CK(cudaMemcpy(bA, tA.data(), tA.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bB, tB.data(), tB.size() * 2, cudaMemcpyHostToDevice));
    }
    cublasHandle_t h; CB(cublasCreate(&h));
    const float alpha = 1.f, beta = 0.f;
    auto cublas_run = [&]{
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                        bB, CUDA_R_16BF, N, bA, CUDA_R_16BF, K, &beta,
                        dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    auto ref_run = [&]{
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                        bB, CUDA_R_16BF, N, bA, CUDA_R_16BF, K, &beta,
                        dRef, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    ref_run(); CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)M * N);
    CK(cudaMemcpy(ref.data(), dRef, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

    auto sample_err = [&]() {
        std::vector<float> got((size_t)M * N);
        CK(cudaMemcpy(got.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        double se = 0, sr = 0, mx = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double d = (double)got[i] - ref[i];
            se += d * d; sr += (double)ref[i] * ref[i];
            mx = std::max(mx, std::fabs(d));
        }
        return std::make_pair(std::sqrt(se / (sr + 1e-30)), mx);
    };

    // ---- 交错测量：预热后 5 轮，每轮内每个变体连测 20 次 ----
    struct Variant { const char* name; std::function<void()> launch; };
    std::vector<Variant> vs;
    std::vector<std::string> names;
    names.reserve(8);          // 先留够，后面保存的 c_str() 不会因扩容失效
    auto add_persist = [&](int P) {
        names.push_back("persist  grid=" + std::to_string(P) + " 沿 N 连续（" +
                        std::to_string(P / sms) + "×SM）");
        vs.push_back({names.back().c_str(),
            [&, P]{ gemm_persistent<BM,BN,BK,WM,WN,PAD,true><<<P,THREADS>>>(bA,bB,dC); }});
    };
    auto add_persist_mfast = [&](int P) {
        names.push_back("persist  grid=" + std::to_string(P) + " 沿 M 连续（" +
                        std::to_string(P / sms) + "×SM）");
        vs.push_back({names.back().c_str(),
            [&, P]{ gemm_persistent<BM,BN,BK,WM,WN,PAD,false><<<P,THREADS>>>(bA,bB,dC); }});
    };
    {
        // grid 必须在 lambda 生命周期内有效：按值捕获，不引用块内局部变量
        const dim3 g_tnfast(N / BN, M / BM), g_tmfast(M / BM, N / BN);
        vs.push_back({"static   grid=1024 沿 N 连续（TN_FAST）",
            [&, g_tnfast]{ gemm_static<BM,BN,BK,WM,WN,PAD,true><<<g_tnfast,THREADS>>>(bA,bB,dC); }});
        vs.push_back({"static   grid=1024 沿 M 连续（TM_FAST）",
            [&, g_tmfast]{ gemm_static<BM,BN,BK,WM,WN,PAD,false><<<g_tmfast,THREADS>>>(bA,bB,dC); }});
        add_persist(sms);
        add_persist(sms * 2);
        add_persist(sms * 4);
        add_persist(TILES);
        add_persist_mfast(sms * 2);
        add_persist_mfast(TILES);
        vs.push_back({"--  cuBLAS bf16 GemmEx（参照）", cublas_run});
    }

    // 先证明 dC 真的被这些 kernel 写过：填 NaN，跑一次，确认没有 NaN 残留。
    // 否则"误差恰好为 0"可能只是比较了两次都没被写过的缓冲区。
    {
        auto nan_run = [&](const char* who) {
            std::vector<float> nan_buf(1024, std::nanf(""));
            for (int i = 0; i < (int)((size_t)M * N / 1024); ++i)
                CK(cudaMemcpy(dC + (size_t)i * 1024, nan_buf.data(), 4096, cudaMemcpyHostToDevice));
            gemm_persistent<BM,BN,BK,WM,WN,PAD,true><<<sms * occ, THREADS>>>(bA, bB, dC);
            CK(cudaDeviceSynchronize());
            std::vector<float> probe((size_t)M * N);
            CK(cudaMemcpy(probe.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
            size_t nan_left = 0;
            for (float v : probe) if (std::isnan(v)) ++nan_left;
            printf("    写回检查（%s）：NaN 预填 %zu 个，跑完残留 %zu 个\n", who, (size_t)M * N, nan_left);
        };
        nan_run("persist grid=2×SM");
    }

    for (auto& v : vs) for (int i = 0; i < WARMUP; ++i) v.launch();
    CK(cudaDeviceSynchronize());

    std::vector<std::vector<double>> times(vs.size());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < ROUNDS; ++r) {
        for (size_t k = 0; k < vs.size(); ++k) {
            cudaEventRecord(e0);
            for (int i = 0; i < REPS; ++i) vs[k].launch();
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            times[k].push_back(ms / REPS);
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    const double PEAK_BF16 = 251.9;
    printf("    %-44s %8s %8s %8s %8s %s\n",
           "变体", "最短ms", "中位ms", "TFLOPS", "占峰值", "相对RMS/最大绝对差");
    for (size_t k = 0; k < vs.size(); ++k) {
        auto t = times[k];
        std::sort(t.begin(), t.end());
        double best = t.front(), med = t[t.size() / 2];
        vs[k].launch(); CK(cudaDeviceSynchronize());   // 用这次输出对参照
        auto err = sample_err();
        double tf = gflop() / 1e3 / (best * 1e-3);
        printf("    %-44s %8.3f %8.3f %8.1f %7.1f%%   %.2e / %.2e\n",
               vs[k].name, best, med, tf, tf / PEAK_BF16 * 100, err.first, err.second);
    }

    // ---- 累加器预算：只改每线程 WM×WN（线程数随 warp 数一起改，保证覆盖完整 tile）----
    printf("\n    只改每线程累加器规模（其余不变），观察 sm_120 的寄存器路径：\n");
    printf("      %-8s %8s %12s %12s %10s %12s\n",
           "WM×WN", "线程数", "寄存器/线程", "累加器/线程", "驻留block", "稳态TFLOPS");
    auto budget_row = [&](auto kern, int wm, int wn, int threads, const char* label) {
        cudaFuncAttributes a{}; CK(cudaFuncGetAttributes(&a, kern));
        int o = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kern, threads, 0));
        int P = sms * o;
        auto run = [&]{ kern<<<P, threads>>>(bA, bB, dC); };
        for (int i = 0; i < WARMUP; ++i) run();
        CK(cudaDeviceSynchronize());
        double best = 1e9;
        cudaEvent_t a0, a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
        for (int r = 0; r < ROUNDS; ++r) {
            cudaEventRecord(a0);
            for (int i = 0; i < REPS; ++i) run();
            cudaEventRecord(a1); CK(cudaEventSynchronize(a1));
            float ms = 0; cudaEventElapsedTime(&ms, a0, a1);
            best = std::min(best, (double)ms / REPS);
        }
        cudaEventDestroy(a0); cudaEventDestroy(a1);
        printf("      %-8s %8d %12d %12d %10d %12.1f\n", label, threads, a.numRegs,
               (wm / 16) * (wn / 16) * (16 * 16 / 32), o, gflop() / 1e3 / (best * 1e-3));
    };
    budget_row(gemm_persistent<BM,BN,BK,32,32,8,true>, 32, 32, (BM/32)*(BN/32)*32, "32×32");
    budget_row(gemm_persistent<BM,BN,BK,64,32,8,true>, 64, 32, (BM/64)*(BN/32)*32, "64×32");
    budget_row(gemm_persistent<BM,BN,BK,64,64,8,true>, 64, 64, (BM/64)*(BN/64)*32, "64×64");

    CB(cublasDestroy(h));
    CK(cudaFree(bA)); CK(cudaFree(bB)); CK(cudaFree(dC)); CK(cudaFree(dRef));
    return 0;
}
