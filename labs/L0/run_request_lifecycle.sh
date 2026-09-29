#!/usr/bin/env bash
# 0.1-A server phase: start a real vllm serve, trace one request, stop it.
# Usage: bash labs/L0/run_request_lifecycle.sh <output-dir> [port]
set -euo pipefail

OUT="${1:?output dir}"
PORT="${2:-8010}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PY="${PY:-/scratch/learn/envs/serve/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"
LOG="$OUT/server.log"

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$OUT/gpu-before.txt"

"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 2048 --gpu-memory-utilization 0.25 \
  --enforce-eager --no-enable-prefix-caching --enable-log-requests \
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
"$PY" "$HERE/request_lifecycle_trace.py" server \
  --output "$OUT/trace" --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --server-pid "$SERVER_PID" --server-log "$LOG" || STATUS=$?

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
sleep 5
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$OUT/gpu-after.txt"
echo "EXIT=$STATUS" > "$OUT/exit.txt"
exit "$STATUS"
