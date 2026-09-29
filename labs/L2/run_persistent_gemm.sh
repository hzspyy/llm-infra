#!/bin/bash
# L2.4-D · 编译并运行 persistent scheduling 对照，同时保留 ptxas 的资源账。
#
#   bash labs/L2/run_persistent_gemm.sh [out_dir]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$PWD/out/2.4/persistent}"
mkdir -p "$OUT"

# CUDA 工具链来自 serve venv 的 nvidia wheel。`nvidia` 是命名空间包，
# `nvidia.__file__` 为 None，不能用 os.path.dirname 推导，直接给绝对路径。
C="${CUDA_HOME:-/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13}"
NVCC="$C/bin/nvcc"

echo "== 编译（-Xptxas -v 的资源账）==" | tee "$OUT/ptxas_resources.txt"
"$NVCC" -O3 -std=c++17 -arch=sm_120 -I"$C/include" -L"$C/lib" -lcublas \
    -Xptxas -v -o "$OUT/persistent_gemm" "$HERE/persistent_gemm.cu" \
    2>>"$OUT/ptxas_resources.txt"
echo "exit=$?" >> "$OUT/ptxas_resources.txt"

echo | tee -a "$OUT/ptxas_resources.txt"
echo "== 运行 ==" | tee "$OUT/persistent_gemm.txt"
LD_LIBRARY_PATH="$C/lib" "$OUT/persistent_gemm" 2>&1 | tee -a "$OUT/persistent_gemm.txt"
