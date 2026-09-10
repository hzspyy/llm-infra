// L2.4 lab · GEMM 优化阶梯：从 naive 到 tensor core。
//
// GEMM 是 compute-bound 算子的原型，也是 LLM 里 88% FLOP 的去处（L0.2 实测）。
// 它的优化阶梯是整个 GPU 编程里最经典的一条，每一步都对应一个硬件事实：
//
//   v0  naive：每线程算 1 个输出，直接读全局    —— 访存量 O(N³)
//   v1  共享内存分块                            —— 访存量降到 O(N³/BK)
//   v2  寄存器分块：每线程算 8×8 个输出          —— 提高算术强度
//   v3  + float4 向量化访存                     —— 每指令搬 4 倍字节
//   v4  + 双缓冲（预取下一块）                   —— 用计算掩盖访存
//   v5  bf16 + mma.sync tensor core             —— 换一套计算单元
//
// 每一步的分母：
//   fp32 CUDA core 峰值 116 TFLOPS（L2.1 实测）
//   bf16 tensor core 峰值 251.9 TFLOPS（L1.2 实测纯发射）
//   cuBLAS 实测 bf16 218 TFLOPS（L1.2）
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -lcublas -o gemm_ladder gemm_ladder.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("[cuBLAS] %s @%d: %d\n", #x, __LINE__, (int)s); exit(1);} } while(0)

// C[M,N] = A[M,K] * B[K,N]，全部行主序
constexpr int M = 4096, N = 4096, K = 4096;

// ---------------------------------------------------------------------------
// v0: naive。每个线程算一个 C[i][j]，内循环从全局内存读 A 的一行、B 的一列。
// 访存量：每个输出读 2K 个元素 → 总共 2*M*N*K 次读 = 137 G 次。
// 算术强度 = 2MNK FLOP / (2MNK*4 B) = 0.25 FLOP/byte，比 decode 还低。
// ---------------------------------------------------------------------------
__global__ void gemm_v0(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// v1: 共享内存分块。BK=32 的一块 A 和 B 被整个 block 复用 32 次。
// 访存量降到 2*M*N*K/BK。
// ---------------------------------------------------------------------------
template <int BS>
__global__ void gemm_v1(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C) {
    __shared__ float sA[BS][BS], sB[BS][BS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BS + ty, col = blockIdx.x * BS + tx;
    float acc = 0.f;
    for (int t = 0; t < K; t += BS) {
        sA[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        sB[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BS; ++k) acc += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// v2: 寄存器分块。block tile 128×128，每线程算 TM×TN = 8×8 个输出。
//
// 这是整条阶梯上最重要的一步。原理：
//   把 A 的一小列(TM) 和 B 的一小行(TN) 读进寄存器，做 TM×TN 次乘加。
//   访存 TM+TN 次，计算 TM*TN 次 → 算术强度从 1 提升到 TM*TN/(TM+TN) = 4。
// 这就是"外积累加"（outer product）——GEMM 优化的核心思想。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v2(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C) {
    __shared__ float sA[BK][BM], sB[BK][BN];        // sA 转置存，方便按列读
    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN), tCol = tid % (BN / TN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;

    float acc[TM][TN] = {0.f};
    float regA[TM], regB[TN];

    // 每线程要搬多少个元素进共享内存
    const int threads = (BM / TM) * (BN / TN);
    const int aLoads = BM * BK / threads, bLoads = BK * BN / threads;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < aLoads; ++i) {
            int idx = i * threads + tid;
            int r = idx / BK, c = idx % BK;
            sA[c][r] = A[(cRow + r) * K + t + c];    // 转置写入
        }
        #pragma unroll
        for (int i = 0; i < bLoads; ++i) {
            int idx = i * threads + tid;
            int r = idx / BN, c = idx % BN;
            sB[r][c] = B[(t + r) * N + cCol + c];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = sA[k][tRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = sB[k][tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(cRow + tRow * TM + i) * N + cCol + tCol * TN + j] = acc[i][j];
}

// ---------------------------------------------------------------------------
// v3: v2 + float4 向量化访存（对照 L2.3：向量化是 memory 侧最重要的一步）
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v3(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C) {
    __shared__ float sA[BK][BM], sB[BK][BN];
    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN), tCol = tid % (BN / TN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
    const int threads = (BM / TM) * (BN / TN);

    float acc[TM][TN] = {0.f};
    float regA[TM], regB[TN];

    // A: 每线程用 float4 读 BK/4 个 4 元组
    const int aRow = tid / (BK / 4), aCol = (tid % (BK / 4)) * 4;
    const int aStep = threads / (BK / 4);
    const int bRow = tid / (BN / 4), bCol = (tid % (BN / 4)) * 4;
    const int bStep = threads / (BN / 4);

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int r = aRow; r < BM; r += aStep) {
            float4 v = *(const float4*)&A[(cRow + r) * K + t + aCol];
            sA[aCol + 0][r] = v.x; sA[aCol + 1][r] = v.y;    // 转置写入
            sA[aCol + 2][r] = v.z; sA[aCol + 3][r] = v.w;
        }
        #pragma unroll
        for (int r = bRow; r < BK; r += bStep)
            *(float4*)&sB[r][bCol] = *(const float4*)&B[(t + r) * N + cCol + bCol];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; i += 4)
                *(float4*)&regA[i] = *(const float4*)&sA[k][tRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j += 4)
                *(float4*)&regB[j] = *(const float4*)&sB[k][tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
            *(float4*)&C[(cRow + tRow * TM + i) * N + cCol + tCol * TN + j] =
                *(const float4*)&acc[i][j];
}

// ---------------------------------------------------------------------------
// v5: bf16 + mma.sync tensor core（用 WMMA API，比手写 PTX 可读）
//
// 换的不是算法，是**计算单元**：从 CUDA core 换到 tensor core。
// L1.2 实测两者的峰值差 2.2 倍（116 vs 251.9 TFLOPS）。
// ---------------------------------------------------------------------------
using namespace nvcuda;
template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v5(const __nv_bfloat16* __restrict__ A,
                        const __nv_bfloat16* __restrict__ B,
                        float* __restrict__ C) {
    __shared__ __nv_bfloat16 sA[BM * BK], sB[BK * BN];
    const int warp = threadIdx.x / 32;
    const int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
    const int threads = blockDim.x;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int t = 0; t < K; t += BK) {
        for (int idx = threadIdx.x; idx < BM * BK; idx += threads) {
            int r = idx / BK, c = idx % BK;
            sA[r * BK + c] = A[(cRow + r) * K + t + c];
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += threads) {
            int r = idx / BN, c = idx % BN;
            sB[r * BN + c] = B[(t + r) * N + cCol + c];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                wmma::load_matrix_sync(fa, &sA[(wRow * WM + i * 16) * BK + k], BK);
                #pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
                    wmma::load_matrix_sync(fb, &sB[k * BN + wCol * WN + j * 16], BN);
                    wmma::mma_sync(acc[i][j], fa, fb, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j)
            wmma::store_matrix_sync(
                &C[(cRow + wRow * WM + i * 16) * N + cCol + wCol * WN + j * 16],
                acc[i][j], N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// v6: v5 + 共享内存 padding 消除 bank 冲突。
//
// WMMA 的 load_matrix_sync 按 16×16 的块读共享内存。如果行距（leading dim）
// 是 32 个 bf16 = 64 字节 = 16 个 bank，那么同一列的不同行会落在同一 bank 上
// —— L1.1 实测 32 路 bank 冲突让访问慢 10 倍。
// 解法：把行距 padding 成非 2 的幂的值，把列打散到不同 bank。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int PAD>
__global__ void gemm_v6(const __nv_bfloat16* __restrict__ A,
                        const __nv_bfloat16* __restrict__ B,
                        float* __restrict__ C) {
    constexpr int LDA = BK + PAD, LDB = BN + PAD;     // ← 关键：加 padding
    __shared__ __nv_bfloat16 sA[BM * LDA], sB[BK * LDB];
    const int warp = threadIdx.x / 32;
    const int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
    const int threads = blockDim.x;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int t = 0; t < K; t += BK) {
        // 用 float4（=8 个 bf16）向量化全局读
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                wmma::load_matrix_sync(fa, &sA[(wRow * WM + i * 16) * LDA + k], LDA);
                #pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
                    wmma::load_matrix_sync(fb, &sB[k * LDB + wCol * WN + j * 16], LDB);
                    wmma::mma_sync(acc[i][j], fa, fb, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i)
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j)
            wmma::store_matrix_sync(
                &C[(cRow + wRow * WM + i * 16) * N + cCol + wCol * WN + j * 16],
                acc[i][j], N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------

static double gflop() { return 2.0 * M * N * K / 1e9; }

template <typename F>
static double bench(F f, int iters = 10) {
    f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s   GEMM %dx%dx%d = %.1f GFLOP\n", p.name, M, N, K, gflop());
    printf("    分母：fp32 CUDA core 116 TFLOPS（L2.1 实测）、"
           "bf16 tensor core 251.9 TFLOPS（L1.2 实测纯发射）\n\n");

    size_t szf = (size_t)M * K * sizeof(float);
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = (rand() % 100 - 50) / 100.0f;
    for (auto& v : hB) v = (rand() % 100 - 50) / 100.0f;

    float *dA, *dB, *dC, *dRef;
    CK(cudaMalloc(&dA, szf)); CK(cudaMalloc(&dB, szf));
    CK(cudaMalloc(&dC, szf)); CK(cudaMalloc(&dRef, szf));
    CK(cudaMemcpy(dA, hA.data(), szf, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), szf, cudaMemcpyHostToDevice));

    cublasHandle_t h; CB(cublasCreate(&h));
    const float alpha = 1.f, beta = 0.f;
    // cuBLAS 是列主序；算 C^T = B^T * A^T 等价于行主序的 C = A*B
    auto cublas_sgemm = [&]{
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                       &alpha, dB, N, dA, K, &beta, dRef, N));
    };
    cublas_sgemm(); CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)M * N);
    CK(cudaMemcpy(ref.data(), dRef, szf, cudaMemcpyDeviceToHost));

    // 误差度量：用 ‖got-ref‖ / ‖ref‖（相对 RMS）。
    // 第一版用逐元素相对误差，但 K=4096 的随机矩阵乘结果在 0 附近震荡，
    // 单点相对误差会因为抵消（cancellation）而虚高到 1e-2，看着像出错了其实没有。
    auto check = [&](const char*) {
        std::vector<float> got((size_t)M * N);
        CK(cudaMemcpy(got.data(), dC, szf, cudaMemcpyDeviceToHost));
        double se = 0, sr = 0;
        for (size_t i = 0; i < got.size(); i += 97) {
            double d = (double)got[i] - ref[i];
            se += d * d; sr += (double)ref[i] * ref[i];
        }
        return std::sqrt(se / (sr + 1e-30));
    };

    printf("    %-44s %10s %10s %8s %s\n", "版本", "耗时ms", "TFLOPS", "占峰值", "相对RMS误差");

    auto report = [&](const char* name, double ms, double peak) {
        double tf = gflop() / 1e3 / (ms * 1e-3);
        printf("    %-44s %10.3f %10.1f %7.1f%%  %.2e\n",
               name, ms, tf, tf / peak * 100, check(name));
    };

    const double PEAK_FP32 = 116.0, PEAK_BF16 = 251.9;

    { dim3 b(16,16), g((N+15)/16,(M+15)/16);
      double ms = bench([&]{ gemm_v0<<<g,b>>>(dA,dB,dC); }, 2);
      report("v0  naive（每线程 1 个输出，直读全局）", ms, PEAK_FP32); }

    { constexpr int BS=32; dim3 b(BS,BS), g(N/BS,M/BS);
      double ms = bench([&]{ gemm_v1<BS><<<g,b>>>(dA,dB,dC); }, 3);
      report("v1  共享内存分块 32x32", ms, PEAK_FP32); }

    { constexpr int BM=128,BN=128,BK=8,TM=8,TN=8;
      dim3 b((BM/TM)*(BN/TN)), g(N/BN,M/BM);
      double ms = bench([&]{ gemm_v2<BM,BN,BK,TM,TN><<<g,b>>>(dA,dB,dC); });
      report("v2  + 寄存器分块 128x128, 每线程 8x8", ms, PEAK_FP32); }

    { constexpr int BM=128,BN=128,BK=8,TM=8,TN=8;
      dim3 b((BM/TM)*(BN/TN)), g(N/BN,M/BM);
      double ms = bench([&]{ gemm_v3<BM,BN,BK,TM,TN><<<g,b>>>(dA,dB,dC); });
      report("v3  + float4 向量化访存", ms, PEAK_FP32); }

    { double ms = bench([&]{ cublas_sgemm(); });
      CK(cudaMemcpy(dC, dRef, szf, cudaMemcpyDeviceToDevice));
      report("--  cuBLAS SGEMM（fp32 参照）", ms, PEAK_FP32); }

    // ---- bf16 tensor core ----
    printf("\n    以下换 bf16 + tensor core（分母改为 251.9 TFLOPS）\n");
    __nv_bfloat16 *bA, *bB;
    CK(cudaMalloc(&bA, (size_t)M*K*2)); CK(cudaMalloc(&bB, (size_t)K*N*2));
    {
        std::vector<__nv_bfloat16> tA((size_t)M*K), tB((size_t)K*N);
        for (size_t i=0;i<tA.size();++i) tA[i] = __float2bfloat16(hA[i]);
        for (size_t i=0;i<tB.size();++i) tB[i] = __float2bfloat16(hB[i]);
        CK(cudaMemcpy(bA, tA.data(), tA.size()*2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bB, tB.data(), tB.size()*2, cudaMemcpyHostToDevice));
    }
    // bf16 的参照重新算一遍（精度不同）
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    bB, CUDA_R_16BF, N, bA, CUDA_R_16BF, K, &beta,
                    dRef, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), dRef, szf, cudaMemcpyDeviceToHost));

    { constexpr int BM=128,BN=128,BK=32,WM=64,WN=32;
      dim3 b(256), g(N/BN,M/BM);
      double ms = bench([&]{ gemm_v5<BM,BN,BK,WM,WN><<<g,b>>>(bA,bB,dC); });
      report("v5  bf16 + mma (WMMA 16x16x16)", ms, PEAK_BF16); }

    { constexpr int BM=128,BN=128,BK=32,WM=64,WN=32,PAD=8;
      dim3 b(256), g(N/BN,M/BM);
      double ms = bench([&]{ gemm_v6<BM,BN,BK,WM,WN,PAD><<<g,b>>>(bA,bB,dC); });
      report("v6  + smem padding 消除 bank 冲突 + 向量化", ms, PEAK_BF16); }

    { double ms = bench([&]{
          CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                          bB, CUDA_R_16BF, N, bA, CUDA_R_16BF, K, &beta,
                          dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); });
      report("--  cuBLAS bf16 GemmEx（参照）", ms, PEAK_BF16); }

    CB(cublasDestroy(h));
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dRef));
    CK(cudaFree(bA)); CK(cudaFree(bB));
    return 0;
}
