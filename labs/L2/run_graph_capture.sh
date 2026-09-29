#!/usr/bin/env bash
# L2.1 · CUDA Graph 捕获/replay：编译、跑正常路径，再用 memcheck 观察 C5 的非法访问。
#
# 用法：source <学习目录>/env.sh 后
#   bash labs/L2/run_graph_capture.sh
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
CUDA=${CUDA_HOME:-$ROOT/envs/serve/lib/python3.12/site-packages/nvidia/cu13}
NVCC=${NVCC:-$CUDA/bin/nvcc}
# wheel 自带的 compute-sanitizer 缺少注入库，会报
#   "Target application terminated before first instrumented API call"
# 改用完整 redist（见 ENVIRONMENTS.md 的工具安装策略）：
SAN=${SAN:-$ROOT/opt/sanitizer-redist/cuda_sanitizer_api-linux-x86_64-13.0.85-archive/bin/compute-sanitizer}
SAN_LIB=${SAN_LIB:-$ROOT/opt/sanitizer-redist/cuda_sanitizer_api-linux-x86_64-13.0.85-archive/lib}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$ROOT/results/2.1-graph}
mkdir -p "$OUT"
BIN="$OUT/graph_capture"

echo "=== 编译 ==="
"$NVCC" -O2 -std=c++17 -arch=sm_120 -lineinfo -o "$BIN" "$HERE/graph_capture.cu" || exit 1

echo
echo "=== 正常路径 ==="
"$BIN" | tee "$OUT/normal.txt"

echo
echo "=== compute-sanitizer --tool memcheck ./graph_capture c5 ==="
LD_LIBRARY_PATH="$SAN_LIB:${LD_LIBRARY_PATH:-}" "$SAN" --tool memcheck --launch-timeout 120 "$BIN" c5 2>&1 \
    | tee "$OUT/sanitizer-memcheck-c5.txt" | tail -30

echo
echo "原始输出：$OUT/"
