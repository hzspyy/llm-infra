#!/bin/bash
# L2.5 · 受控 tile 对照的三条推进：手写 CUDA、Triton/TileLang、CuTe-DSL。
#
# 受控条件是：4096³、输入 fp16、累加 fp32、输出 fp32、tile 128×128×64。
# CuTe-DSL 用的是 CUTLASS 官方 sm_120 例程（同一 tile），它的输出 dtype 固定为
# fp16，因此单独一栏标注，用于"同一 tile 下能不能跑、跑到多少"，不并入误差列。
#
#   bash labs/L2/run_controlled_tile.sh [out_dir]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$PWD/out/2.5/controlled-tile}"
mkdir -p "$OUT"
C="${CUDA_HOME:-/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13}"
PY=/scratch/learn/envs/serve/bin/python
CUTLASS_SRC="${CUTLASS_SRC:-/scratch/learn/opt/src/cutlass-147295a3}"

echo "== 1. 手写 CUDA（同一份源码，dtype 与 tile 是模板参数）==" | tee "$OUT/controlled_tile.txt"
"$C/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -I"$C/include" -L"$C/lib" -lcublas \
    -Xptxas -v -o "$OUT/gemm_wmma_dtype" "$HERE/gemm_wmma_dtype.cu" \
    2> "$OUT/cuda_ptxas.txt"
LD_LIBRARY_PATH="$C/lib" "$OUT/gemm_wmma_dtype" 2>&1 | tee -a "$OUT/controlled_tile.txt"
echo | tee -a "$OUT/controlled_tile.txt"

echo "== 2. Triton / TileLang（受控 tile、各自 tile、布局改动、融合）==" | tee -a "$OUT/controlled_tile.txt"
"$PY" "$HERE/dsl_controlled_tile.py" --out-dir "$OUT" 2>&1 | tee -a "$OUT/controlled_tile.txt"
echo | tee -a "$OUT/controlled_tile.txt"

echo "== 3. CuTe-DSL（CUTLASS 官方 sm_120 例程，tile 128×128×64）==" | tee -a "$OUT/controlled_tile.txt"
EX="$CUTLASS_SRC/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm"
export CUDA_TOOLKIT_PATH="$C"
{
  echo "CUTLASS commit: $(git -C "$CUTLASS_SRC" rev-parse HEAD)"
  # 冷/热编译的时间只测"编译 + 一次运行"：例程默认会做一次 CPU 上的参照检查
  # （4096³ 的 einsum 要 1–2 s），用它计时会把参照检查误算成编译。
  echo "-- 冷编译（空 cache，--skip_ref_check --iterations 1）--"
  rm -rf /scratch/learn/.cache/cute_dsl_cold
  t0=$(date +%s.%N)
  ( cd "$EX" && CUTE_DSL_CACHE_DIR=/scratch/learn/.cache/cute_dsl_cold \
      "$PY" dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk 128,128,64 \
      --skip_ref_check --iterations 1 --warmup_iterations 0 2>&1 ) \
    | grep -E "Execution time|Error|error" | head -3
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "冷编译 + 1 次运行: %.2f s\n", b-a}'
  echo "-- 缓存命中（复用 cache，同样只跑 1 次）--"
  t0=$(date +%s.%N)
  ( cd "$EX" && CUTE_DSL_CACHE_DIR=/scratch/learn/.cache/cute_dsl \
      "$PY" dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk 128,128,64 \
      --skip_ref_check --iterations 1 --warmup_iterations 0 2>&1 ) \
    | grep -E "Execution time|Error|error" | head -3
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "缓存命中 + 1 次运行: %.2f s\n", b-a}'
  echo
  echo "-- 受控 tile 的正确性与稳态（含参照检查）--"
  ( cd "$EX" && CUTE_DSL_CACHE_DIR=/scratch/learn/.cache/cute_dsl \
      "$PY" dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk 128,128,64 \
      --iterations 5 --warmup_iterations 2 2>&1 ) \
    | grep -E "Tile Shape|Execution time|PASS|Error|error" | head -6
  echo
  echo "-- 同 tile 下的布局改动：B 改成 N-major 存放（例程自带开关）--"
  ( cd "$EX" && CUTE_DSL_CACHE_DIR=/scratch/learn/.cache/cute_dsl \
      "$PY" dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk 128,128,64 \
      --a_major k --b_major n --iterations 5 --warmup_iterations 2 2>&1 ) \
    | grep -E "Tile Shape|Execution time|PASS|Error|error" | head -6
  echo
  echo "-- 例程拒绝的输入：tile 与 mnk 不整除（原文）--"
  ( cd "$EX" && CUTE_DSL_CACHE_DIR=/scratch/learn/.cache/cute_dsl \
      "$PY" dense_gemm.py --mnkl 4000,4096,4096,1 --tile_shape_mnk 128,128,64 \
      --iterations 1 2>&1 ) | tail -3
} 2>&1 | tee -a "$OUT/controlled_tile.txt"

echo | tee -a "$OUT/controlled_tile.txt"
echo "== 4. ptxas 资源账（手写 CUDA 的三个实例）==" | tee -a "$OUT/controlled_tile.txt"
grep -E "Function properties|Used |spill" "$OUT/cuda_ptxas.txt" | head -20 | tee -a "$OUT/controlled_tile.txt"
