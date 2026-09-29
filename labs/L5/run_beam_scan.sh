#!/usr/bin/env bash
# L5.9 补测 · 起 SGLang 服务并跑 beam 扫描，跑完回收。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1
PY=/scratch/learn/envs/sgl/bin/python
PORT="${1:-8148}"
OUT="${2:-/scratch/learn/work/out/5.9/beam-scan-20260921}"
mkdir -p "$OUT"
pkill -f "port $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B --port "$PORT" --host 127.0.0.1 \
  --context-length 8192 --mem-fraction-static 0.45 --page-size 1 \
  --chunked-prefill-size 8192 --disable-radix-cache --enable-metrics \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 300); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text":"ready","sampling_params":{"max_new_tokens":1,"temperature":0}}' 2>/dev/null || echo 000)
  if [ "$CODE" = "200" ]; then echo "服务就绪 ${i}s"; READY=1; break; fi
  sleep 1
done
if [ "$READY" != "1" ]; then echo "未就绪"; kill "$SERVER_PID" 2>/dev/null; exit 1; fi
cd /scratch/learn/work/labs/L5
"$PY" -u beam_search_scan.py --base "http://127.0.0.1:$PORT" --out "$OUT" 2>&1 | tail -25
kill "$SERVER_PID" 2>/dev/null
sleep 8
pkill -f "port $PORT" 2>/dev/null
echo "BEAM_DONE"
