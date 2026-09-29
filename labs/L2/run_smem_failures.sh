#!/usr/bin/env bash
# L2.1 · 失败小例：编译、正常跑一遍，再用 compute-sanitizer 跑对应工具。
#
# 用法：
#   bash labs/L2/run_smem_failures.sh              # 用默认路径
#   LEARN_ROOT=/scratch/learn bash labs/L2/run_smem_failures.sh
#
# 需先 source 学习目录的 env.sh（提供 nvcc 与 compute-sanitizer）。
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
OUT=${OUT:-$ROOT/results/2.1-smem}
mkdir -p "$OUT"
BIN="$OUT/smem_failures"

echo "=== 编译 $BIN ==="
"$NVCC" -O2 -std=c++17 -arch=sm_120 -lineinfo -o "$BIN" "$HERE/smem_failures.cu" || exit 1

echo
echo "=== 正常运行（20 次重复统计不一致率）==="
"$BIN" all | tee "$OUT/normal.txt"

run_san() {   # $1=tool $2=mode $3=输出文件名
    echo
    echo "=== compute-sanitizer --tool $1 ./smem_failures $2 ==="
    LD_LIBRARY_PATH="$SAN_LIB:${LD_LIBRARY_PATH:-}" "$SAN" --tool "$1" --launch-timeout 120 \
        "$BIN" "$2" 2>&1 | tee "$OUT/$3" | tail -20
}

run_san memcheck  oob        sanitizer-memcheck.txt
run_san racecheck race       sanitizer-racecheck-buggy.txt
run_san racecheck racefixed  sanitizer-racecheck-fixed.txt
run_san synccheck barrier    sanitizer-synccheck.txt

echo
echo "原始输出：$OUT/"
