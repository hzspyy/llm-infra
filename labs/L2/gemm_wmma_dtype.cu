// L2.5 lab · 受控 tile 对照用的手写 CUDA GEMM（同一份 kernel，dtype 与 tile 都是模板参数）。
//
// 2.4 的 gemm_ladder.cu 只实例化 bf16。2.5 的受控 tile 对照要在**同一 dtype、同一 tile**
// 下和 Triton / TileLang / CuTe-DSL 比，所以这里把元素类型也参数化：
//
//   T = __half / __nv_bfloat16，tile 由 <BM,BN,BK,WM,WN,PAD> 给定
//
// 计时协议与 2.4 的 persistent_gemm 一致：预热 10 次，5 轮交错 × 20 次，报最短与中位。
// 数值用 cuBLAS 同 dtype 参照，并做 NaN 预填的写回检查。
//
//   bash labs/L2/run_controlled_tile.sh <out_dir>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("[cuBLAS] %s @%d: %d\n", #x, __LINE__, (int)s); exit(1);} } while(0)

constexpr int M = 4096, N = 4096, K = 4096;
constexpr int WARMUP = 10, ROUNDS = 5, REPS = 20;
constexpr double PEAK = 251.9;   // L1.2 实测 bf16/fp16 tensor core 纯发射上限

// ---------------------------------------------------------------------------
// 与 2.4 v6 同构的 WMMA GEMM：smem 分块 + 寄存器分块 + padding 消 bank 冲突 + float4 向量化。
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN, int PAD>
__global__ void gemm_wmma(const T* __restrict__ A, const T* __restrict__ B,
                          float* __restrict__ C) {
    constexpr int LDA = BK + PAD, LDB = BN + PAD;
    __shared__ T sA[BM * LDA], sB[BK * LDB];
    const int warp = threadIdx.x / 32;
    const int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
    const int threads = blockDim.x;
    constexpr int VEC = 16 / sizeof(T);      // 16 字节 = 8 个 half / bf16

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) nvcuda::wmma::fill_fragment(acc[i][j], 0.0f);

    for (int t = 0; t < K; t += BK) {
        for (int idx = threadIdx.x * VEC; idx < BM * BK; idx += threads * VEC) {
            int r = idx / BK, c = idx % BK;
            *reinterpret_cast<float4*>(&sA[r * LDA + c]) =
                *reinterpret_cast<const float4*>(&A[(cRow + r) * K + t + c]);
        }
        for (int idx = threadIdx.x * VEC; idx < BK * BN; idx += threads * VEC) {
            int r = idx / BN, c = idx % BN;
            *reinterpret_cast<float4*>(&sB[r * LDB + c]) =
                *reinterpret_cast<const float4*>(&B[(t + r) * N + cCol + c]);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> fa;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major> fb;
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

static double gflop() { return 2.0 * M * N * K / 1e9; }

// ---------------------------------------------------------------------------
// 一个 (dtype, tile) 组合的完整测量
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN, int PAD>
static void run_case(const char* tag, const void* dA, const void* dB, float* dC,
                     const std::vector<float>& ref_host) {
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
    dim3 grid(N / BN, M / BM);
    auto launch = [&]{ gemm_wmma<T, BM, BN, BK, WM, WN, PAD><<<grid, THREADS>>>(
        (const T*)dA, (const T*)dB, dC); };

    cudaFuncAttributes at{};
    CK(cudaFuncGetAttributes(&at, gemm_wmma<T, BM, BN, BK, WM, WN, PAD>));
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, gemm_wmma<T, BM, BN, BK, WM, WN, PAD>, THREADS, 0));

    for (int i = 0; i < WARMUP; ++i) launch();
    CK(cudaDeviceSynchronize());

    double ts[ROUNDS], best = 1e9, med = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < ROUNDS; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < REPS; ++i) launch();
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        ts[r] = ms / REPS;
        best = std::min(best, ts[r]);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(ts, ts + ROUNDS);
    med = ts[ROUNDS / 2];

    // 写回检查：填 NaN 后跑一次，确认没有残留
    {
        std::vector<float> nan_v(1024, std::nanf(""));
        for (int i = 0; i < (int)((size_t)M * N / 1024); ++i)
            CK(cudaMemcpy(dC + (size_t)i * 1024, nan_v.data(), 4096, cudaMemcpyHostToDevice));
    }
    launch(); CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)M * N);
    CK(cudaMemcpy(got.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    size_t nan_left = 0, diff = 0;
    double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) ++nan_left;
        double d = (double)got[i] - ref_host[i];
        if (got[i] != ref_host[i]) ++diff;
        se += d * d; sr += (double)ref_host[i] * ref_host[i];
        mx = std::max(mx, std::fabs(d));
    }
    double rr = std::sqrt(se / (sr + 1e-30));
    double tf = gflop() / 1e3 / (best * 1e-3);

    printf("    %-38s %8.3f %8.3f %9.1f %8.1f%%  %5d %6d %8zu %8d  %.2e/%.2e  NaN残留 %zu 不同元素 %zu\n",
           tag, best, med, tf, tf / PEAK * 100, at.numRegs, THREADS,
           at.sharedSizeBytes, occ, rr, mx, nan_left, diff);
}

static double bench_cublas(cublasHandle_t h, cudaDataType_t dt, const void* A, const void* B,
                           float* dC) {
    const float alpha = 1.f, beta = 0.f;
    auto run = [&]{ CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                    B, dt, N, A, dt, K, &beta, dC, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); };
    for (int i = 0; i < WARMUP; ++i) run();
    CK(cudaDeviceSynchronize());
    double best = 1e9;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < ROUNDS; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < REPS; ++i) run();
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms / REPS);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  SM=%d  GEMM %dx%dx%d = %.1f GFLOP\n", p.name, p.multiProcessorCount,
           M, N, K, gflop());
    printf("    对照上限：bf16/fp16 tensor core 纯发射 251.9 TFLOPS（L1.2 实测）\n\n");

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = (rand() % 100 - 50) / 100.0f;
    for (auto& v : hB) v = (rand() % 100 - 50) / 100.0f;

    void *dA16, *dB16, *dAbf, *dBbf;
    float *dC, *dRef;
    CK(cudaMalloc(&dA16, (size_t)M * K * 2)); CK(cudaMalloc(&dB16, (size_t)K * N * 2));
    CK(cudaMalloc(&dAbf, (size_t)M * K * 2)); CK(cudaMalloc(&dBbf, (size_t)K * N * 2));
    CK(cudaMalloc(&dC, (size_t)M * N * 4)); CK(cudaMalloc(&dRef, (size_t)M * N * 4));
    {
        std::vector<__half> a16((size_t)M * K), b16((size_t)K * N);
        std::vector<__nv_bfloat16> abf((size_t)M * K), bbf((size_t)K * N);
        for (size_t i = 0; i < a16.size(); ++i) { a16[i] = __float2half(hA[i]); abf[i] = __float2bfloat16(hA[i]); }
        for (size_t i = 0; i < b16.size(); ++i) { b16[i] = __float2half(hB[i]); bbf[i] = __float2bfloat16(hB[i]); }
        CK(cudaMemcpy(dA16, a16.data(), a16.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB16, b16.data(), b16.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dAbf, abf.data(), abf.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBbf, bbf.data(), bbf.size() * 2, cudaMemcpyHostToDevice));
    }
    cublasHandle_t h; CB(cublasCreate(&h));
    std::vector<float> ref((size_t)M * N);

    printf("    %-38s %8s %8s %9s %9s  %5s %6s %8s %8s  %s\n",
           "变体（同一份源码，只改模板参数）", "最短ms", "中位ms", "TFLOPS", "占峰值",
           "寄存器", "线程", "smem B", "block/SM", "相对RMS/最大绝对差");

    // ---- fp16 ----
    bench_cublas(h, CUDA_R_16F, dA16, dB16, dRef); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), dRef, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    run_case<__half, 128, 128, 64, 64, 32, 8>("fp16 tile 128x128x64（受控 tile）", dA16, dB16, dC, ref);
    run_case<__half, 128, 128, 32, 64, 32, 8>("fp16 tile 128x128x32（2.4 的 v6 配置）", dA16, dB16, dC, ref);

    // ---- bf16 ----
    bench_cublas(h, CUDA_R_16BF, dAbf, dBbf, dRef); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), dRef, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    run_case<__nv_bfloat16, 128, 128, 64, 64, 32, 8>("bf16 tile 128x128x64（受控 tile）", dAbf, dBbf, dC, ref);

    {
        double b16 = bench_cublas(h, CUDA_R_16F, dA16, dB16, dC);
        double bbf = bench_cublas(h, CUDA_R_16BF, dAbf, dBbf, dC);
        printf("    %-38s %8.3f %8s %9.1f %8.1f%%\n", "--  cuBLAS fp16（参照）", b16, "-",
               gflop() / 1e3 / (b16 * 1e-3), gflop() / 1e3 / (b16 * 1e-3) / PEAK * 100);
        printf("    %-38s %8.3f %8s %9.1f %8.1f%%\n", "--  cuBLAS bf16（参照）", bbf, "-",
               gflop() / 1e3 / (bbf * 1e-3), gflop() / 1e3 / (bbf * 1e-3) / PEAK * 100);
    }

    CB(cublasDestroy(h));
    CK(cudaFree(dA16)); CK(cudaFree(dB16)); CK(cudaFree(dAbf)); CK(cudaFree(dBbf));
    CK(cudaFree(dC)); CK(cudaFree(dRef));
    return 0;
}
