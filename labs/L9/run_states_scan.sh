#!/usr/bin/env bash
# L9.4 §5：逐位置 decode 延迟。纯 attention（Qwen3-4B）与混合架构（Qwen3.5-4B）各起一次服务，
# 扫两档：batch=1 短上下文（权重带宽受限）与 batch=16 + 6k 上下文（KV 读取与权重同量级）。
#
#   bash labs/L9/run_states_scan.sh <run_root>
set -u
source /scratch/learn/env.sh
RUN_ROOT=${1:?usage: run_states_scan.sh <run_root>}
PORT=${L94S_PORT:-8017}
PY=/scratch/learn/envs/serve/bin/python
mkdir -p "$RUN_ROOT"; cd /scratch/learn

for MODEL in Qwen/Qwen3-4B Qwen/Qwen3.5-4B; do
  TAG=$(echo "$MODEL" | tr '/' '_')
  $PY -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization "${L94S_GPU_UTIL:-0.55}" \
    --enable-prefix-caching --no-enable-log-requests \
    > "$RUN_ROOT/server-$TAG.log" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[$MODEL] ready after ${i}s"; ok=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" = "1" ]; then
    grep -o "GPU KV cache size: [0-9,]* tokens" "$RUN_ROOT/server-$TAG.log" > "$RUN_ROOT/kv-$TAG.txt" || true
    # batch=1，短上下文
    $PY -u work/labs/L9/reasoning_budget.py states --model "$MODEL" \
      --base-url "http://127.0.0.1:$PORT/v1" --out "$RUN_ROOT/states" \
      --budgets 512 --copies 1 --concurrency 1 > "$RUN_ROOT/states-b1-$TAG.log" 2>&1
    echo "$?" > "$RUN_ROOT/states-b1-$TAG.exit"
    # batch=16，长上下文（约 6k token）
    $PY -u work/labs/L9/reasoning_budget.py states --model "$MODEL" \
      --base-url "http://127.0.0.1:$PORT/v1" --out "$RUN_ROOT/states" \
      --budgets 512 --copies 24 --concurrency 16 > "$RUN_ROOT/states-b16-$TAG.log" 2>&1
    echo "$?" > "$RUN_ROOT/states-b16-$TAG.exit"
  else
    echo "server failed for $MODEL" > "$RUN_ROOT/states-$TAG.log"; echo 1 > "$RUN_ROOT/states-$TAG.exit"
  fi
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; sleep 5
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
