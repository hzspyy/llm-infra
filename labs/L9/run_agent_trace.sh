#!/usr/bin/env bash
# L9.1 任务 A：起 vLLM 服务并采集三类 agent 执行轨迹。
#
#   bash labs/L9/run_agent_trace.sh <run_root> [n_per_class] [concurrency]
#
# 例：bash labs/L9/run_agent_trace.sh /scratch/learn/work/out/9.1-trace 100 16
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_agent_trace.sh <run_root> [n_per_class] [concurrency]}
N=${2:-100}
CONC=${3:-16}
PORT=${L91_PORT:-8011}
MODEL=${L91_MODEL:-Qwen/Qwen3-4B}

mkdir -p "$RUN_ROOT"
cd /scratch/learn

/scratch/learn/envs/serve/bin/python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L91_GPU_UTIL:-0.45}" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "[server] ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[server] died; tail of log:"
    tail -40 "$RUN_ROOT/server.log"
    exit 1
  fi
  sleep 1
done

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/gpu-during.txt"

/scratch/learn/envs/serve/bin/python -u work/labs/L9/agent_trace_collect.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" \
  --out "$RUN_ROOT/trace" \
  --n-per-class "$N" \
  --concurrency "$CONC" \
  --max-turns 12 --max-tokens 512 \
  --thinking-classes compute \
  > "$RUN_ROOT/collect.log" 2>&1
RC=$?
echo "$RC" > "$RUN_ROOT/collect.exit"

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/gpu-after.txt"
exit "$RC"
