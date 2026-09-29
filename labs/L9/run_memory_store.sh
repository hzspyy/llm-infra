#!/usr/bin/env bash
# L9.6 任务 D：起 embedding 服务，跑长期记忆存储的生命周期检查（合成事实，5 写 5 查）。
#
#   bash labs/L9/run_memory_store.sh <run_root>
set -u
source /scratch/learn/env.sh
RUN_ROOT=${1:?usage: run_memory_store.sh <run_root>}
PORT=${L96M_PORT:-8019}
PY=/scratch/learn/envs/serve/bin/python
EMB_MODEL=${L96_EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B}
mkdir -p "$RUN_ROOT"; cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$EMB_MODEL" --served-model-name "$EMB_MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.15 --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
$PY -u work/labs/L9/memory_store_lifecycle.py --base-url "http://127.0.0.1:$PORT/v1" \
  --embed-model "$EMB_MODEL" --out "$RUN_ROOT/memory" > "$RUN_ROOT/memory.log" 2>&1
echo "$?" > "$RUN_ROOT/memory.exit"
kill $PID 2>/dev/null
