#!/usr/bin/env bash
# 5.6 补测：真实 SSE 里 tool-call 的 None 步分帧。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=/scratch/learn/envs/serve/bin/python
PORT=8160
OUT=${1:-/scratch/learn/work/out/5.6/tool-sse-20260921}
SNAP=/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/$(ls /scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots | head -1)
mkdir -p "$OUT"
pkill -9 -f "[p]ort $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --served-model-name Qwen/Qwen3-1.7B --port $PORT --host 127.0.0.1 \
  --max-model-len 4096 --gpu-memory-utilization 0.34 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$OUT/server.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 300); do
  C=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [ "$C" = "200" ] && { echo "ready ${i}s"; READY=1; break; }
  sleep 1
done
if [ "$READY" != 1 ]; then
  echo "服务未就绪；引擎错误见 server.log"
  tail -6 "$OUT/server.log" | cut -c1-160
  kill -9 "$PID" 2>/dev/null
  exit 1
fi
cd /scratch/learn/work/labs/L5
# 服务端的 --served-model-name 是 Qwen/Qwen3-1.7B；请求里的 model 必须与之一致
# （tokenizer 仍按仓库 id 从本地缓存解析）
"$PY" -u tool_call_sse_framing.py --base "http://127.0.0.1:$PORT" \
  --model "Qwen/Qwen3-1.7B" --out "$OUT" 2>&1 | tail -12
kill -9 "$PID" 2>/dev/null; sleep 5; pkill -9 -f "[p]ort $PORT" 2>/dev/null
echo TOOL_SSE_DONE
