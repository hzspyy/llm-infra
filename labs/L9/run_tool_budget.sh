#!/usr/bin/env bash
# L9.2 任务 A 的剩余测量：工具预算（schema 未命中/命中前缀 + 完整集 vs 按需发现）在 vLLM 上，
# tool_choice 四模式的支持/拒绝路径在 vLLM 与 SGLang 两个引擎上各跑一遍。
#
#   bash labs/L9/run_tool_budget.sh <run_root>
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_tool_budget.sh <run_root>}
V_PORT=${L92B_VLLM_PORT:-8021}
S_PORT=${L92B_SGLANG_PORT:-8022}
MODEL=${L92B_MODEL:-Qwen/Qwen3-4B}
PY_V=/scratch/learn/envs/serve/bin/python
PY_S=/scratch/learn/envs/sgl/bin/python
LABS=/scratch/learn/work/labs/L9

mkdir -p "$RUN_ROOT"
cd /scratch/learn

# --- vLLM：工具预算 ----------------------------------------------------------------
$PY_V -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$V_PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L92B_GPU_UTIL:-0.45}" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-log-requests \
  > "$RUN_ROOT/vllm-server.log" 2>&1 &
VPID=$!
cleanup_v() { kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; }
trap cleanup_v EXIT

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$V_PORT/v1/models" > /dev/null && { echo "[vllm] ready after ${i}s"; break; }
  kill -0 "$VPID" 2>/dev/null || { tail -40 "$RUN_ROOT/vllm-server.log"; exit 1; }
  sleep 1
done

$PY_V -u "$LABS/tool_pipeline.py" budget \
  --base-url "http://127.0.0.1:$V_PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/budget-vllm" --counts 1,4,16,64 --repeats 2 \
  > "$RUN_ROOT/budget-vllm.log" 2>&1
echo "$?" > "$RUN_ROOT/budget-vllm.exit"

$PY_V -u "$LABS/tool_pipeline.py" choices \
  --base-url "http://127.0.0.1:$V_PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/choices-vllm" --repeats 2 \
  > "$RUN_ROOT/choices-vllm.log" 2>&1
echo "$?" > "$RUN_ROOT/choices-vllm.exit"

cleanup_v
trap - EXIT
sleep 5

# --- SGLang：支持/拒绝路径 ----------------------------------------------------------
$PY_S -u -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$S_PORT" \
  --dtype bfloat16 --context-length 8192 \
  --mem-fraction-static "${L92B_SGL_MEM:-0.35}" \
  --tool-call-parser qwen25 --reasoning-parser qwen3 \
  > "$RUN_ROOT/sglang-server.log" 2>&1 &
SPID=$!
cleanup_s() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
trap cleanup_s EXIT

for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$S_PORT/v1/models" > /dev/null && { echo "[sglang] ready after ${i}s"; break; }
  kill -0 "$SPID" 2>/dev/null || { tail -40 "$RUN_ROOT/sglang-server.log"; exit 1; }
  sleep 1
done

$PY_S -u "$LABS/tool_pipeline.py" choices \
  --base-url "http://127.0.0.1:$S_PORT/v1" --model "$MODEL" \
  --out "$RUN_ROOT/choices-sglang" --repeats 2 \
  > "$RUN_ROOT/choices-sglang.log" 2>&1
echo "$?" > "$RUN_ROOT/choices-sglang.exit"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
