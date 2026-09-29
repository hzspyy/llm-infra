#!/usr/bin/env bash
# L9.1 任务 B/C：在同一批真实请求上做合成负载对照与重放。
#
#   bash labs/L9/run_trace_replay.sh <run_root> <trace_dir> [max_num_seqs]
#
# run_root 放本次重放的输出，trace_dir 是 run_agent_trace.sh 采集到的 trace 目录。
# 环境变量：L91_PORT（默认 8011）、L91_GPU_UTIL（默认 0.45）。
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_trace_replay.sh <run_root> <trace_dir> [max_num_seqs]}
TRACE=${2:?missing trace dir}
MAX_SEQS=${3:-64}
PORT=${L91_PORT:-8011}
MODEL=${L91_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python

mkdir -p "$RUN_ROOT"
cd /scratch/learn

# 离线部分不需要 GPU
$PY -u work/labs/L9/trace_replay.py profile --run "$TRACE" --out "$RUN_ROOT/profile" \
  > "$RUN_ROOT/profile.log" 2>&1
$PY -u work/labs/L9/trace_replay.py synth --run "$TRACE" --out "$RUN_ROOT/synth" \
  > "$RUN_ROOT/synth.log" 2>&1
$PY -u work/labs/L9/trace_replay.py router --run "$TRACE" --out "$RUN_ROOT/router" --replicas 2 \
  > "$RUN_ROOT/router.log" 2>&1
$PY -u work/labs/L9/trace_replay.py audio --out "$RUN_ROOT/audio" --sessions 40 --chunks 40 \
  > "$RUN_ROOT/audio.log" 2>&1

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L91_GPU_UTIL:-0.45}" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "[server] ready after ${i}s (max_num_seqs=$MAX_SEQS)"
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"

# 冒烟：先发 2 个请求确认 tools 路径可用，失败就直接退出（避免整轮重放白跑）
$PY -u work/labs/L9/trace_replay.py replay --run "$TRACE" \
  --plan original --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --concurrency 2 --limit 2 --out "$RUN_ROOT/smoke" > "$RUN_ROOT/smoke.log" 2>&1
SMOKE_ERRORS=$($PY -c "import json;print(json.load(open('$RUN_ROOT/smoke/replay_summary.json'))['errors'])")
if [ "$SMOKE_ERRORS" != "0" ]; then
  echo "[smoke] replay errors=$SMOKE_ERRORS; aborting"
  head -c 400 "$RUN_ROOT/smoke/replay_requests.jsonl"
  exit 1
fi

for plan in original independent session; do
  $PY -u work/labs/L9/trace_replay.py replay --run "$TRACE" \
    --plan "$plan" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --concurrency 16 --out "$RUN_ROOT/replay-$plan-c16" \
    > "$RUN_ROOT/replay-$plan-c16.log" 2>&1
  echo "$?" > "$RUN_ROOT/replay-$plan-c16.exit"
done

# 同一份原样请求、低并发重放：用于重放一致性对照
$PY -u work/labs/L9/trace_replay.py replay --run "$TRACE" \
  --plan original --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --concurrency 8 --limit 300 --out "$RUN_ROOT/replay-original-c8" \
  > "$RUN_ROOT/replay-original-c8.log" 2>&1
echo "$?" > "$RUN_ROOT/replay-original-c8.exit"

# 开环泊松到达：同一请求清单，到达过程不再由客户端并发决定
$PY -u work/labs/L9/trace_replay.py replay --run "$TRACE" \
  --plan original --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --concurrency 64 --arrival-rate 12 --out "$RUN_ROOT/replay-original-poisson12" \
  > "$RUN_ROOT/replay-original-poisson12.log" 2>&1
echo "$?" > "$RUN_ROOT/replay-original-poisson12.exit"

$PY -u work/labs/L9/trace_replay.py consistency --run "$TRACE" \
  --replay "$RUN_ROOT/replay-original-c16" --out "$RUN_ROOT/consistency" \
  > "$RUN_ROOT/consistency.log" 2>&1
echo "$?" > "$RUN_ROOT/consistency.exit"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
