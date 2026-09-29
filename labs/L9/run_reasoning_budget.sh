#!/usr/bin/env bash
# L9.4：起 vLLM 服务，跑 budget × thinking 扫描、并发/长短混合、逐位置 decode 与采样聚合。
#
#   bash labs/L9/run_reasoning_budget.sh <run_root> [model]
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_reasoning_budget.sh <run_root> [model]}
MODEL=${2:-Qwen/Qwen3-4B}
PORT=${L94_PORT:-8013}
PY=/scratch/learn/envs/serve/bin/python

mkdir -p "$RUN_ROOT"
cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization "${L94_GPU_UTIL:-0.55}" \
  --reasoning-parser qwen3 \
  --enable-prefix-caching --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"
grep -o "GPU KV cache size: [0-9,]* tokens" "$RUN_ROOT/server.log" > "$RUN_ROOT/kv_capacity.txt" || true

run() {  # run <name> <args...>
  local name=$1; shift
  $PY -u work/labs/L9/reasoning_budget.py "$@" --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$MODEL" --out "$RUN_ROOT/$name" > "$RUN_ROOT/$name.log" 2>&1
  echo "$?" > "$RUN_ROOT/$name.exit"
}

run sweep-gsm8k   sweep --dataset gsm8k   --n 128 --budgets 128 512 2048 8192 --concurrency 16
run sweep-math500 sweep --dataset math500 --n 128 --budgets 128 512 2048 8192 --concurrency 16
run concurrency   concurrency --dataset gsm8k --n 64 --budget 512 --concurrency-levels 1 4 16
run samples       samples --dataset gsm8k --n 64 --budget 2048 --k 4 --concurrency 16

cleanup
trap - EXIT
sleep 5
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
