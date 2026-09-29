#!/usr/bin/env bash
# 0.5-C: 起一个带 reasoning / tool-call parser 的 vLLM 服务，抓三次流式响应的原始字节，再停掉。
# 用法: bash labs/L0/run_stream_events.sh <output-dir> [port]
set -euo pipefail

OUT="${1:?output dir}"
PORT="${2:-8021}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PY="${PY:-/scratch/learn/envs/serve/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"
LOG="$OUT/server.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/gpu-before.txt"

"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 4096 --gpu-memory-utilization 0.35 \
  --enforce-eager --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
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
"$PY" "$HERE/stream_events_probe.py" --out-dir "$OUT" \
  --base-url "http://127.0.0.1:$PORT" --model "$MODEL" || STATUS=$?

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/gpu-after.txt"
echo "EXIT=$STATUS" > "$OUT/exit.txt"
exit "$STATUS"
