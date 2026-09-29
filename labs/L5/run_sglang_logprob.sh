#!/usr/bin/env bash
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1
PY=/scratch/learn/envs/sgl/bin/python
PORT=8195
OUT=${1:-/scratch/learn/work/out/5.10/sglang-logprob-20260921}
MODEL=/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/$(ls /scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots | head -1)
mkdir -p "$OUT"
pkill -9 -f "[p]ort $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m sglang.launch_server --model-path "$MODEL" --port $PORT \
  --host 127.0.0.1 --context-length 4096 --mem-fraction-static 0.45 \
  --page-size 16 --chunked-prefill-size 4096 --disable-radix-cache \
  > "$OUT/server.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 300); do
  C=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d '{"text":"ready","sampling_params":{"max_new_tokens":1,"temperature":0}}' 2>/dev/null || echo 000)
  [ "$C" = "200" ] && { echo "ready ${i}s"; READY=1; break; }
  sleep 1
done
[ "$READY" = 1 ] || { echo "未就绪"; tail -5 "$OUT/server.log" | cut -c1-160; kill -9 $PID 2>/dev/null; exit 1; }
cd /scratch/learn/work/labs/L5
"$PY" -u sglang_topk_logprob.py --base "http://127.0.0.1:$PORT" --out "$OUT" 2>&1 | tail -14
kill -9 "$PID" 2>/dev/null; sleep 5; pkill -9 -f "[p]ort $PORT" 2>/dev/null
echo SGLANG_LOGPROB_DONE
