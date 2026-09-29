#!/usr/bin/env bash
# L5.4 任务 B 的「代价」采集：每个配置一个干净进程，量启动时间与 KV 容量。
#
# 为什么一个进程只建一个引擎：同进程连续建引擎时前一个的显存不一定归还
# （实测空闲显存 13.2 -> 12.9 -> 9.8 -> 6.4 -> 3.0 GiB），KV blocks 那一列
# 会变成「谁的预算大」而不是「图的代价」。分开进程后 gpu_memory_utilization
# 固定（L54_UTIL，默认 0.35），各行的 KV blocks 才可比。
#
# 用法：RUN_ID=20260913-bash labs/L5/run_execution_cost.sh
set -u
ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/execution-cost-$RUN_ID}
export L54_UTIL=${L54_UTIL:-0.35}
mkdir -p "$OUT"

SPECS=${SPECS:-"eager(NONE) COMPILE_ONLY PIECEWISE FULL_DECODE_ONLY FULL_AND_PIECEWISE bucket:1,8,64,512 bucket:1,8,16,32,64,128 bucket:1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512"}
for s in $SPECS; do
    VLLM_LOGGING_LEVEL=WARNING "$PY" "$HERE/execution_layer.py" --cost "$s" \
        > "$OUT/$(echo "$s" | tr -c 'a-zA-Z0-9' '_').log" 2>&1
done
grep -h "^COST " "$OUT"/*.log | sort -t= -k2 > "$OUT/cost.txt"
cat "$OUT/cost.txt"
echo
echo "引擎自报的图内存："
grep -h "CUDA graph pool memory\|Graph capturing finished" "$OUT"/*.log | sed 's/^/  /'
echo
echo "原始工件在 $OUT/"
