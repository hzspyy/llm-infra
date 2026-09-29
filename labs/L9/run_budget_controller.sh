#!/usr/bin/env bash
# L9.4 任务 B/C：预算控制器（三种实际支持的方式）+ 多候选执行与验证器。
#
#   bash labs/L9/run_budget_controller.sh <run_root> [n]
set -u
source /scratch/learn/env.sh
RUN_ROOT=${1:?usage: run_budget_controller.sh <run_root> [n]}
N=${2:-64}
PORT=${L94B_PORT:-8021}
PY=/scratch/learn/envs/serve/bin/python
MODEL=${L94B_MODEL:-Qwen/Qwen3-4B}
mkdir -p "$RUN_ROOT"; cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization "${L94B_GPU_UTIL:-0.5}" \
  --reasoning-parser qwen3 --enable-prefix-caching --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "[server] ready after ${i}s"; break; }
  kill -0 $PID 2>/dev/null || { tail -20 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"

$PY -u work/labs/L9/reasoning_budget_controller.py controller \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --out "$RUN_ROOT/controller" \
  --n "$N" --caps 256 1024 --margin 512 --concurrency 8 > "$RUN_ROOT/controller.log" 2>&1
echo "$?" > "$RUN_ROOT/controller.exit"

# 候选：串行 1/2/4 与并行 4，都用可执行算式核对
for spec in "1 0" "2 0" "4 0" "4 1" ; do
  set -- $spec
  K=$1; PAR=$2
  name="candidates-k${K}-par${PAR}"
  args=""
  [ "$PAR" = "1" ] && args="--parallel"
  $PY -u work/labs/L9/reasoning_budget_controller.py candidates \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --out "$RUN_ROOT/$name" \
    --n 32 --k "$K" $args --verify expr > "$RUN_ROOT/$name.log" 2>&1
  echo "$?" > "$RUN_ROOT/$name.exit"
done

kill $PID 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
