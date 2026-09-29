#!/usr/bin/env bash
# L9.1 任务 A/B：统一 span 采集、原始 SSE 审计、串行与 fork/join 任务图、依赖驱动重放。
#
#   bash labs/L9/run_agent_spans.sh <run_root> [n_per_class] [concurrency]
#
# 输出目录结构：
#   $RUN_ROOT/collect/          统一 span 轨迹（spans/nodes/sessions/requests/tasks/summary）
#   $RUN_ROOT/sse/              原始 SSE 字节 + 字段审计
#   $RUN_ROOT/graph/            串行与 fork/join 事件图、关键路径
#   $RUN_ROOT/timeout-probe/    受限 deadline 下 timeout 任务进入分母的证据
#   $RUN_ROOT/replay-dag-*/     依赖驱动重放（原工具时延 / 注入慢工具）
#   $RUN_ROOT/replay-<plan>-c16 请求级对照（multiset/session/independent）
#   $RUN_ROOT/compare/          结构、依赖与输入 token 对拍
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_agent_spans.sh <run_root> [n_per_class] [concurrency]}
N=${2:-100}
CONC=${3:-16}
PORT=${L91_PORT:-8011}
MODEL=${L91_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9

mkdir -p "$RUN_ROOT"
cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
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
$PY -c "import vllm,torch,transformers;print(vllm.__version__,torch.__version__,transformers.__version__)" \
  > "$RUN_ROOT/versions.txt" 2>&1

# --- A1: 原始 SSE 审计（不经过 SDK 直连引擎） -----------------------------------------
$PY -u "$LABS/agent_trace_spans.py" sse-audit \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/sse" --probe 3 \
  > "$RUN_ROOT/sse-audit.log" 2>&1
echo "$?" > "$RUN_ROOT/sse-audit.exit"

# --- A2: 统一 span 采集 ---------------------------------------------------------------
$PY -u "$LABS/agent_trace_spans.py" collect \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/collect" \
  --n-per-class "$N" --concurrency "$CONC" \
  --max-turns 12 --max-tokens 512 --thinking-classes compute \
  --arrival-rate "${L91_ARRIVAL_RATE:-4.5}" \
  > "$RUN_ROOT/collect.log" 2>&1
echo "$?" > "$RUN_ROOT/collect.exit"

# --- A3: timeout 任务进入分母的证据（受限 deadline 的小批量） -------------------------
$PY -u "$LABS/agent_trace_spans.py" collect \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/timeout-probe" \
  --classes codefix --n-per-class 8 --concurrency 4 \
  --max-turns 12 --max-tokens 512 --task-timeout 3.0 \
  > "$RUN_ROOT/timeout-probe.log" 2>&1
echo "$?" > "$RUN_ROOT/timeout-probe.exit"

# --- A4: 串行与 fork/join 事件图 ------------------------------------------------------
$PY -u "$LABS/agent_trace_spans.py" graph \
  --run "$RUN_ROOT/collect" --out "$RUN_ROOT/graph" \
  > "$RUN_ROOT/graph.log" 2>&1
echo "$?" > "$RUN_ROOT/graph.exit"

# --- B1: 依赖驱动重放（工具时延按真实轨迹重放） ---------------------------------------
$PY -u "$LABS/trace_replay.py" dagreplay \
  --run "$RUN_ROOT/collect" --out "$RUN_ROOT/replay-dag-c16" \
  --plan dag --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --concurrency "$CONC" --arrival-mode trace \
  > "$RUN_ROOT/replay-dag-c16.log" 2>&1
echo "$?" > "$RUN_ROOT/replay-dag-c16.exit"

# --- B2: 注入慢工具 1000 ms：验证拥塞下逻辑到达仍全部记录 ------------------------------
$PY -u "$LABS/trace_replay.py" dagreplay \
  --run "$RUN_ROOT/collect" --out "$RUN_ROOT/replay-dag-slowtool" \
  --plan dag --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --concurrency "$CONC" --arrival-mode trace --inject-tool-ms 1000 \
  > "$RUN_ROOT/replay-dag-slowtool.log" 2>&1
echo "$?" > "$RUN_ROOT/replay-dag-slowtool.exit"

# --- B3: 三种请求级对照 ---------------------------------------------------------------
for plan in multiset session independent; do
  $PY -u "$LABS/trace_replay.py" replay \
    --run "$RUN_ROOT/collect" --out "$RUN_ROOT/replay-$plan-c16" \
    --plan "$plan" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --concurrency "$CONC" \
    > "$RUN_ROOT/replay-$plan-c16.log" 2>&1
  echo "$?" > "$RUN_ROOT/replay-$plan-c16.exit"
done

# --- B4: 结构与工作量对拍 -------------------------------------------------------------
$PY -u "$LABS/trace_replay.py" dagcompare \
  --run "$RUN_ROOT/collect" --out "$RUN_ROOT/compare" \
  --dag "$RUN_ROOT/replay-dag-c16" \
  --mode multiset "$RUN_ROOT/replay-multiset-c16" \
  --mode session "$RUN_ROOT/replay-session-c16" \
  --mode independent "$RUN_ROOT/replay-independent-c16" \
  > "$RUN_ROOT/compare.log" 2>&1
echo "$?" > "$RUN_ROOT/compare.exit"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
