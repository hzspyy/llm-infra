#!/usr/bin/env bash
# 0.1-C server phase for SGLang: start a server, read its process tree, stop it.
# Usage: bash labs/L0/run_sglang_lifecycle.sh <output-dir> [port]
set -euo pipefail

OUT="${1:?output dir}"
PORT="${2:-8020}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PY="${PY:-/scratch/learn/envs/sgl/bin/python}"
TRACE_PY="${TRACE_PY:-/scratch/learn/envs/serve/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"
LOG="$OUT/server.log"

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-before.txt"

"$PY" -m sglang.launch_server \
  --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --context-length 2048 --mem-fraction-static 0.25 \
  --disable-radix-cache --disable-cuda-graph \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid.txt"
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fs --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 2
done
curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null

STATUS=0
"$TRACE_PY" "$HERE/request_lifecycle_trace.py" server \
  --output "$OUT/trace" --engine sglang --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --server-pid "$SERVER_PID" --server-log "$LOG" || STATUS=$?

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
sleep 8
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$OUT/gpu-after.txt"
echo "EXIT=$STATUS" > "$OUT/exit.txt"
exit "$STATUS"
