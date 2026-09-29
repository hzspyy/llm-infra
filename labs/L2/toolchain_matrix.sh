#!/usr/bin/env bash
# L2.2-A · 编译制品矩阵：每个变体留下命令、字节数与哈希，并做源码行→指令映射。
#
# 用法：source <学习目录>/env.sh 后
#   bash labs/L2/toolchain_matrix.sh [输出目录]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="${1:-/scratch/learn/work/out/2.2/20260913-matrix}"
mkdir -p "$OUT"
cd "$OUT"

CUOBJ=$(command -v cuobjdump || ls "$LEARN_ROOT"/opt/cuobjdump/*/bin/cuobjdump 2>/dev/null | head -1)
NVDIS=$(command -v nvdisasm || echo "$CUDA_HOME/bin/nvdisasm")

cat > kernels.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + 1.0f;
}
__global__ void reduce_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    __shared__ float s[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; __syncthreads(); }
    if (t == 0) y[blockIdx.x] = s[0];
}
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
EOF

hash_of() { sha256sum "$1" 2>/dev/null | cut -c1-12; }
bytes_of() { stat -c%s "$1" 2>/dev/null || echo "-"; }

echo "############ 1. 制品矩阵：-O 级别 / fast-math / 目标架构"
printf "%-34s %-10s %-10s %-10s %-8s %-8s %-6s\n" "变体" "cpp1" "ptx" "sass" "regs" "smem" "spill"
printf "%s\n" "--------------------------------------------------------------------------------------------"
for V in "O0:-O0 -Xptxas -O0" "O2:-O2 -Xptxas -O2" "O3:-O3 -Xptxas -O3" "O3fast:-O3 -Xptxas -O3 --use_fast_math"; do
  NAME=${V%%:*}; FLAGS=${V#*:}
  rm -f v_*.ptx v_*.o v_*.cpp1
  nvcc -arch=sm_120 $FLAGS -E kernels.cu -o v_$NAME.cpp1 2>/dev/null
  nvcc -arch=sm_120 $FLAGS -ptx kernels.cu -o v_$NAME.ptx 2>/dev/null
  nvcc -arch=sm_120 $FLAGS -cubin kernels.cu -o v_$NAME.cubin -Xptxas -v 2> v_$NAME.log
  REGS=$(grep -oE "Used [0-9]+ registers" v_$NAME.log | head -1 | grep -oE "[0-9]+")
  SMEM=$(grep -oE "[0-9]+ bytes smem" v_$NAME.log | head -1 | grep -oE "[0-9]+")
  SPILL=$(grep -oE "[0-9]+ bytes spill stores" v_$NAME.log | head -1 | grep -oE "[0-9]+")
  "$CUOBJ" -sass v_$NAME.cubin > v_$NAME.sass 2>/dev/null
  printf "%-34s %-10s %-10s %-10s %-8s %-8s %-6s\n" \
    "$NAME ($FLAGS)" "$(bytes_of v_$NAME.cpp1)" "$(bytes_of v_$NAME.ptx)" \
    "$(bytes_of v_$NAME.sass)" "${REGS:-?}" "${SMEM:-?}" "${SPILL:-0}"
  printf "  %-32s sha256(cpp1/ptx/cubin)=%s/%s/%s\n" "" \
    "$(hash_of v_$NAME.cpp1)" "$(hash_of v_$NAME.ptx)" "$(hash_of v_$NAME.cubin)"
done

echo
echo "############ 2. 目标架构：sm_120 vs sm_90（PTX 版本与 JIT）"
printf "%-30s %-10s %-10s %s\n" "目标" "cubin" "ptx" "SASS 说明"
printf "%s\n" "-----------------------------------------------------------------"
for G in "-arch=sm_120" "-gencode=arch=compute_90,code=sm_90" "-gencode=arch=compute_120,code=compute_120"; do
  rm -f arch_*.o
  nvcc $G -O3 -c kernels.cu -o arch_$(echo "$G" | tr -cd 'a-z0-9').o 2>/dev/null
  F=$(ls arch_*.o 2>/dev/null | head -1); [ -z "$F" ] && continue
  N=$("$CUOBJ" -lelf "$F" 2>/dev/null | grep -c cubin)
  P=$("$CUOBJ" -lptx "$F" 2>/dev/null | grep -c ptx)
  echo "$G | cubin=$N ptx=$P | $(bytes_of "$F") 字节"
  rm -f arch_*.o
done
echo "  ⇒ 只有 PTX 的变体在首次加载时由驱动 JIT；这一点在 4 节用时间量出来。"

echo
echo "############ 3. 源码行 → SASS 指令数（-lineinfo + nvdisasm -gi）"
nvcc -arch=sm_120 -O3 -lineinfo -cubin kernels.cu -o line.cubin 2>/dev/null
"$CUOBJ" -xelf all line.cubin > /dev/null 2>&1
CUB=$(ls *.cubin 2>/dev/null | head -1)
if [ -n "$CUB" ] && [ -n "$NVDIS" ]; then
  "$NVDIS" -c -gi "$CUB" > line.sass 2>/dev/null
  echo "  按源码行的指令数（三个 kernel 合并统计，前 14 行）："
  printf "    %-8s %s\n" "源码行" "指令数"
  grep -oE '//## File "[^"]+", line [0-9]+' line.sass \
    | awk '{c[$NF]++} END {for (l in c) printf "    %-8s %s\n", l, c[l]}' | sort -k2 -n | head -14
  echo "  saxpy_kernel 只占 kernels.cu 的 2-5 行；出现更多行号是因为 -lineinfo" 
  echo "  映射的是预处理后的行。没有被映射到的源码行即被优化掉或与相邻行合并。"
fi

echo
echo "############ 4. fast-math 的数值代价（同一源文件，两个编译单元）"
cat > fm_impl.cu <<'EOF'
#include <cuda_runtime.h>
#ifndef FM_KERNEL
#define FM_KERNEL fm_kernel
#endif
__global__ void FM_KERNEL(const float* __restrict__ x, float* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i];
    float s = 0.f;
    for (int k = 1; k <= 8; ++k) s += 1.0f / (a + 0.001f * k) + sqrtf(a * k + 1.0f);
    o[i] = s;
}
extern "C" int RUN_FN(const float* x, float* o, int n) {
    FM_KERNEL<<<(n + 127) / 128, 128>>>(x, o, n);
    return (int)cudaDeviceSynchronize();
}
EOF
nvcc -arch=sm_120 -O3 -Xptxas -O3 -DFM_KERNEL=fm_kernel_precise -DRUN_FN=run_precise \
     -c fm_impl.cu -o fm_precise.o 2>/dev/null
nvcc -arch=sm_120 -O3 -Xptxas -O3 --use_fast_math -DFM_KERNEL=fm_kernel_fast -DRUN_FN=run_fast \
     -c fm_impl.cu -o fm_fast.o 2>/dev/null
cat > fm_main.cu <<'EOF'
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
extern "C" int run_precise(const float*, float*, int);
extern "C" int run_fast(const float*, float*, int);
int main() {
    const int n = 1 << 20;
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h[i] = 0.1f + i * 1e-6f;
    float *d, *o1, *o2;
    cudaMalloc(&d, n * 4); cudaMalloc(&o1, n * 4); cudaMalloc(&o2, n * 4);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    run_precise(d, o1, n); run_fast(d, o2, n);
    float* r1 = (float*)malloc(n * 4); float* r2 = (float*)malloc(n * 4);
    cudaMemcpy(r1, o1, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2, o2, n * 4, cudaMemcpyDeviceToHost);
    double max_abs = 0, max_rel = 0;
    for (int i = 0; i < n; ++i) {
        double dd = fabs((double)r1[i] - (double)r2[i]);
        double rel = dd / (fabs((double)r1[i]) + 1e-30);
        if (dd > max_abs) max_abs = dd;
        if (rel > max_rel) max_rel = rel;
    }
    printf("  precise[0]=%.7f  fast[0]=%.7f\n", r1[0], r2[0]);
    printf("  max |diff| = %.3e   max rel = %.3e\n", max_abs, max_rel);
    return 0;
}
EOF
if nvcc -arch=sm_120 -O3 fm_main.cu fm_precise.o fm_fast.o -o fm_test 2>/dev/null; then
  ./fm_test | sed 's/^/  /'
  echo "  SASS 指令数：precise=$("$CUOBJ" -sass fm_precise.o 2>/dev/null | grep -cE '^\s+/\*[0-9a-f]+\*/')" \
       "fast=$("$CUOBJ" -sass fm_fast.o 2>/dev/null | grep -cE '^\s+/\*[0-9a-f]+\*/')"
  echo "  （同一个 .cu，只差 --use_fast_math：结果有差异，指令数也不同）"
else
  echo "  fast-math 对照编译失败"
fi

echo
echo "############ 5. -Xptxas -v 原始日志（O3 变体）"
sed -n '1,12p' v_O3.log | sed 's/^/    /'

echo
echo "############ 6. unroll × 寄存器预算：时间 / 占用率 / spill 合并表"
nvcc -O3 -std=c++17 -arch=sm_120 -Xptxas -v -o sweep "$HERE/toolchain_sweep.cu" 2> sweep.log
if [ -x ./sweep ]; then
  ./sweep > sweep.txt 2>&1
  echo "  — memory-bound（1 GiB 流式读，UNROLL 个 float4/线程）—"
  sed -n '/^unroll/,/^$/p' sweep.txt | sed 's/^/    /'
  echo "  — compute-bound（独立 FMA 链，MINB 压缩寄存器预算）—"
  sed -n '/^\[compute\]/,$p' sweep.txt | grep -vE "^注：|local\(B\) > 0|^$" | sed 's/^/    /'
  echo
  echo "  ptxas 报的 spill（按符号，只列非零）："
  awk '/Function properties for/{sym=$NF} /spill stores/{ if ($1+0 > 0) print "    " sym " -> " $0 }' sweep.log | head -12
  echo "  （完整日志：$OUT/sweep.log）"
else
  echo "  sweep 编译失败，见 sweep.log"
fi

echo
echo "############ 7. -maxrregcount 硬上限：spill 与 TFLOPS 一起变"
printf "  %-16s %-8s %-10s %-10s %-10s %s\n" "maxrregcount" "regs" "spill(B)" "local(B)" "occ blk" "TFLOPS"
printf "  %s\n" "--------------------------------------------------------------------------------"
for CAP in 0 32 48 64 128 255; do
  if [ "$CAP" = "0" ]; then FLAG=""; NAME="(默认)"; else FLAG="-maxrregcount=$CAP"; NAME="$CAP"; fi
  nvcc -O3 -std=c++17 -arch=sm_120 $FLAG -Xptxas -v -o cap_bin "$HERE/sweep_capped.cu" 2> cap_$CAP.log
  if [ -x ./cap_bin ]; then
    OUT_LINE=$(./cap_bin)
    REGS=$(echo "$OUT_LINE" | grep -oE "regs=[0-9]+" | cut -d= -f2)
    LOC=$(echo "$OUT_LINE" | grep -oE "local=[0-9]+" | cut -d= -f2)
    OCC=$(echo "$OUT_LINE" | grep -oE "occ=[0-9]+" | cut -d= -f2)
    TF=$(echo "$OUT_LINE" | grep -oE "tflops=[0-9.]+" | cut -d= -f2)
    SP=$(grep -oE "[0-9]+ bytes spill stores" cap_$CAP.log | head -1 | grep -oE "^[0-9]+")
    printf "  %-16s %-8s %-10s %-10s %-10s %s\n" "$NAME" "${REGS:-?}" "${SP:-0}" "${LOC:-?}" "${OCC:-?}" "${TF:-?}"
  fi
done
echo "  ⇒ __launch_bounds__ 只是目标，ptxas 可以选择不满足；-maxrregcount 是硬上限，"
echo "     超了就只能 spill 到 local memory。"

echo
echo "所有产物在 $OUT"
