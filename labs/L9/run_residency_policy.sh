#!/usr/bin/env bash
# L9.3 任务 B：量重算与命中两条路径的单请求时间，然后用它驱动代价模型与 TTL 选择。
#
#   bash labs/L9/run_residency_policy.sh <run_root>
#
# 两个 vLLM 进程分先后起：一个关 prefix caching（量重算），一个开（量命中）。
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_residency_policy.sh <run_root>}
OFF_PORT=${L93_OFF_PORT:-8031}
ON_PORT=${L93_ON_PORT:-8032}
MODEL=${L93_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9
SPANS=${L93_SPANS:-/scratch/learn/work/out/9.1-spans-20260921/collect/spans.jsonl}

mkdir -p "$RUN_ROOT"
cd /scratch/learn

run_server() {   # $1=port  $2=extra args  $3=log
  $PY -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$1" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization "${L93_GPU_UTIL:-0.45}" \
    --enable-prompt-tokens-details --no-enable-log-requests $2 \
    > "$3" 2>&1 &
  echo $!
}

wait_ready() {   # $1=port  $2=pid
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null && { echo "[server $1] ready after ${i}s"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "[server $1] died"; return 1; }
    sleep 1
  done
  return 1
}

# --- 重算：关 prefix caching ---------------------------------------------------------
OFF_PID=$(run_server "$OFF_PORT" "--no-enable-prefix-caching" "$RUN_ROOT/server-cache-off.log")
if wait_ready "$OFF_PORT" "$OFF_PID"; then
  $PY -u "$LABS/session_residency_policy.py" probe \
    --base-url "http://127.0.0.1:$OFF_PORT/v1" --model "$MODEL" --mode cache_off \
    --lengths 512,4096,8192 --repeats 3 --out "$RUN_ROOT/probe" \
    > "$RUN_ROOT/probe-cache-off.log" 2>&1
  echo "$?" > "$RUN_ROOT/probe-cache-off.exit"
fi
kill "$OFF_PID" 2>/dev/null; wait "$OFF_PID" 2>/dev/null
sleep 5

# --- 命中：开 prefix caching ---------------------------------------------------------
ON_PID=$(run_server "$ON_PORT" "--enable-prefix-caching" "$RUN_ROOT/server-cache-on.log")
if wait_ready "$ON_PORT" "$ON_PID"; then
  $PY -u "$LABS/session_residency_policy.py" probe \
    --base-url "http://127.0.0.1:$ON_PORT/v1" --model "$MODEL" --mode cache_on \
    --lengths 512,4096,8192 --repeats 3 --out "$RUN_ROOT/probe" \
    > "$RUN_ROOT/probe-cache-on.log" 2>&1
  echo "$?" > "$RUN_ROOT/probe-cache-on.exit"
fi
kill "$ON_PID" 2>/dev/null; wait "$ON_PID" 2>/dev/null
sleep 3

# --- 离线部分：代价模型与 TTL --------------------------------------------------------
$PY -u "$LABS/session_residency_policy.py" derive \
  --out "$RUN_ROOT/derive" --probe-dir "$RUN_ROOT/probe" \
  --lengths 512,4096,8192 --gaps 0,1,10,60 \
  > "$RUN_ROOT/derive.log" 2>&1
echo "$?" > "$RUN_ROOT/derive.exit"

for L in 512 4096 8192; do
  $PY -u "$LABS/session_residency_policy.py" ttl \
    --spans "$SPANS" --out "$RUN_ROOT/ttl-$L" --length "$L" --recompute-ms 0 \
    > "$RUN_ROOT/ttl-$L.log" 2>&1
  echo "$?" > "$RUN_ROOT/ttl-$L.exit"
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
