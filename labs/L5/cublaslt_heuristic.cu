// L5.4 补测 · cuBLASLt 的候选与选择：为什么 batch 16 与 64 会挑不同的 GEMM。
//
// cuBLAS 的启发式是闭源的，但 `cublasLtMatmulAlgoGetHeuristic` 会把候选算法连同
// 它们的 tile / stages / splitK / workspace 一起返回，并且是按启发式排序的。
// 于是可以回答两件可核查的事：
//
//   1. 同一形状下，启发式返回的候选集里有哪些 splitK 变体，它们排第几；
//   2. 扫 M（batch），看 top-1 的 tile / splitK / workspace 在哪个 M 上换档。
//
// 形状取 Qwen3-1.7B 的 q_proj：N=2048（输出维）、K=2048（输入维），
// bf16 输入输出 + fp32 累加，列主序（对应 torch 的行主序 (M,K)x(K,N) 转置）。
//
// 编译（crater，nvcc 来自 serve venv 的 CUDA wheel）：
//   CUDIR=/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13
//   $CUDIR/bin/nvcc -O2 -arch=sm_120 -I$CUDIR/include -L$CUDIR/lib \
//       -lcublasLt -lcublas -lcudart -o /scratch/learn/work/out/5.4/cublaslt/cublaslt_heuristic \
//       labs/L5/cublaslt_heuristic.cu
//
// 运行： ./cublaslt_heuristic            # 扫 M=1..512

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CK(x) do { cublasStatus_t st__ = (x); if (st__ != CUBLAS_STATUS_SUCCESS) { \
  printf("cublas 调用失败 %s: %d\n", #x, (int)st__); exit(1); } } while (0)

static void report(cublasLtHandle_t lt, int M, int N, int K, int max_algo) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tn = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  CK(cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, M, K, M));
  CK(cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&C, CUDA_R_16BF, M, N, M));

  cublasLtMatmulPreference_t pref = nullptr;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = 256ull << 20;
  CK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      lt, op, A, B, C, C, pref, max_algo, res, &got);
  if (st != CUBLAS_STATUS_SUCCESS) {
    printf("M=%4d  启发式查询失败 %d\n", M, (int)st);
  } else {
    printf("M=%4d  候选 %d 个\n", M, got);
    for (int i = 0; i < got && i < 4; ++i) {
      const cublasLtMatmulAlgo_t& a = res[i].algo;
      uint64_t algo_id = 0; uint16_t tile = 0; uint32_t stages = 0;
      uint32_t splitk = 0, red = 0, swizzle = 0, cga = 0, inner = 0;
      uint32_t splitk_cap = 0, sw_cap = 0;
      size_t w = 0;
#define GET(cfg, dst) do { w = 0; cublasLtMatmulAlgoConfigGetAttribute( \
      &a, (cfg), &(dst), sizeof(dst), &w); } while (0)
      GET(CUBLASLT_ALGO_CONFIG_ID, algo_id);
      GET(CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
      GET(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner);
      GET(CUBLASLT_ALGO_CONFIG_STAGES_ID, stages);
      GET(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splitk);
      GET(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red);
      GET(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle);
      GET(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cga);
#undef GET
      w = 0;
      cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                        &splitk_cap, sizeof(splitk_cap), &w);
      w = 0;
      cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
                                        &sw_cap, sizeof(sw_cap), &w);
      printf("   #%d algo=%-5llu tile=%-3u stages=%-2u splitK=%-2u red=%-2u "
             "swizzle=%-2u inner=%-2u cluster=%-2u ws=%.1f MiB  cap[splitk/swizzle]=%u/%u\n",
             i, (unsigned long long)algo_id, (unsigned)tile, stages, splitk, red,
             swizzle, inner, cga, res[i].workspaceSize / 1048576.0, splitk_cap, sw_cap);
    }
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(A);
  cublasLtMatrixLayoutDestroy(B);
  cublasLtMatrixLayoutDestroy(C);
  cublasLtMatmulDescDestroy(op);
}

int main() {
  cublasLtHandle_t lt = nullptr;
  CK(cublasLtCreate(&lt));
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  printf("设备 %s  SM=%d  sm_%d%d\n", prop.name, prop.multiProcessorCount,
         prop.major, prop.minor);
  printf("形状取自 Qwen3-1.7B q_proj：N=2048（输出维）K=2048（输入维）"
         "bf16 输入输出 + fp32 累加\n\n");
  const int N = 2048, K = 2048;
  for (int M : {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 256, 512}) {
    report(lt, M, N, K, 16);
  }
  cublasLtDestroy(lt);
  return 0;
}
