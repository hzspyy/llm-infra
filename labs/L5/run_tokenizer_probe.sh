#!/usr/bin/env bash
# 5.11 补测：分词在前端占多少（同一串 token 的文本 vs 预分词两种送法）。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=/scratch/learn/envs/serve/bin/python
PORT=8165
OUT=${1:-/scratch/learn/work/out/5.11/tokenizer-20260921}
SNAP=/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/$(ls /scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots | head -1)
mkdir -p "$OUT"
pkill -9 -f "[p]ort $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --served-model-name Qwen/Qwen3-1.7B --port $PORT --host 127.0.0.1 \
  --max-model-len 8192 --gpu-memory-utilization 0.34 --dtype bfloat16 \
  --max-num-seqs 256 --no-enable-prefix-caching --no-enable-log-requests \
  > "$OUT/server.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 300); do
  C=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [ "$C" = "200" ] && { echo "ready ${i}s"; READY=1; break; }
  sleep 1
done
[ "$READY" = 1 ] || { echo "未就绪"; tail -5 "$OUT/server.log" | cut -c1-160; kill -9 $PID 2>/dev/null; exit 1; }
cd /scratch/learn/work/labs/L5
"$PY" -u tokenizer_bottleneck.py --base "http://127.0.0.1:$PORT" \
  --model "Qwen/Qwen3-1.7B" --out "$OUT" 2>&1 | tail -16
kill -9 "$PID" 2>/dev/null; sleep 5; pkill -9 -f "[p]ort $PORT" 2>/dev/null
echo TOKENIZER_DONE
