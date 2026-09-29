#!/usr/bin/env bash
# L5.4 任务 B/C 的采集脚本（crater）。
#
#   execution_modes.txt   —— E 五种执行配置的重复分布、F 捕获桶、G CPU/GPU 归因
#   overlap_audit.txt     —— C1/C4/C5 重叠阶段开关、重叠判据、输出缓冲生命周期
#
# 用法：RUN_ID=20260913-bash labs/L5/run_execution_layer_revision.sh
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/execution-layer-$RUN_ID}
mkdir -p "$OUT"

echo "== [1/2] E/F/G：执行配置、捕获桶、CPU/GPU 归因 =="
VLLM_LOGGING_LEVEL=WARNING "$PY" "$HERE/execution_layer.py" E F G \
    > "$OUT/execution_modes.txt" 2>&1
echo "   -> $OUT/execution_modes.txt ($(wc -l < "$OUT/execution_modes.txt") 行)"
grep -E "^  (eager|COMPILE_ONLY|PIECEWISE|FULL|batch|config)|非单调" \
    "$OUT/execution_modes.txt" | head -30

echo "== [2/2] C：重叠阶段开关与输出缓冲生命周期 =="
VLLM_LOGGING_LEVEL=WARNING "$PY" "$HERE/overlap_audit.py" \
    > "$OUT/overlap_audit.txt" 2>&1
echo "   -> $OUT/overlap_audit.txt ($(wc -l < "$OUT/overlap_audit.txt") 行)"
grep -E "async|overlap|相同|前缀" "$OUT/overlap_audit.txt" | head -20

echo
echo "原始工件在 $OUT/"
