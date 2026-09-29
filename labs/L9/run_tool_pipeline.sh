#!/usr/bin/env bash
# L9.2：起一个开了工具解析的 vLLM 服务，跑 tool_choice × 工具数矩阵，然后跑内容类型对照
# （含 DFlash 草稿，进程内 LLM）。
#
#   bash labs/L9/run_tool_pipeline.sh <run_root>
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_tool_pipeline.sh <run_root>}
PORT=${L92_PORT:-8012}
MODEL=${L92_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python

mkdir -p "$RUN_ROOT"
cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L92_GPU_UTIL:-0.45}" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"
$PY -u work/labs/L9/tool_pipeline.py choices \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/choices" > "$RUN_ROOT/choices.log" 2>&1
echo "$?" > "$RUN_ROOT/choices.exit"
cleanup
trap - EXIT
sleep 5

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-before-content.txt"
$PY -u work/labs/L9/tool_pipeline.py content --model "$MODEL" \
  --out "$RUN_ROOT/content" > "$RUN_ROOT/content.log" 2>&1
echo "$?" > "$RUN_ROOT/content.exit"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
