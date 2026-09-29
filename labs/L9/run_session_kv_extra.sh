#!/usr/bin/env bash
# L9.3 补测：修正后的 trim_thinking 结构预测 + LoRA adapter 缓存身份。
# 两次起服务之间等待显存释放，避免 vLLM 的内存 profiling 断言
# （"Initial free memory ... current free memory ..."）。
#
#   bash labs/L9/run_session_kv_extra.sh <run_root>
set -u
source /scratch/learn/env.sh
export VLLM_SERVER_DEV_MODE=1
RUN_ROOT=${1:?usage: run_session_kv_extra.sh <run_root>}
PORT=${L93_PORT:-8014}
MODEL=${L93_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python
LORA=trl-lib/Qwen3-4B-LoRA
mkdir -p "$RUN_ROOT"; cd /scratch/learn

start_server() {
  local log=$1; shift
  $PY -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization "${L93_GPU_UTIL:-0.45}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-prompt-tokens-details --no-enable-log-requests "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -15 "$log"; return 1; }
    sleep 1
  done
  return 1
}
stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 15; }

start_server "$RUN_ROOT/server-structure2.log" --enable-prefix-caching || exit 1
$PY -u work/labs/L9/session_kv_bench.py structure --out "$RUN_ROOT/structure2" \
  --turns 2 4 8 > "$RUN_ROOT/structure2.log" 2>&1; echo "$?" > "$RUN_ROOT/structure2.exit"
stop_server

LORA_B="$RUN_ROOT/lora-copy"
mkdir -p "$LORA_B"
rm -rf "$LORA_B"; mkdir -p "$LORA_B"
  # 只复制适配器文件本体（snapshot 里是符号链接，解引用后写进独立目录）
  cp -rL "$HF_HOME/hub/models--trl-lib--Qwen3-4B-LoRA"/snapshots/*/. "$LORA_B/" 2>/dev/null || true
start_server "$RUN_ROOT/server-lora2.log" --enable-prefix-caching --enable-lora \
  --lora-modules "rev_a=$LORA" "rev_b=$LORA_B" || exit 1
$PY -u work/labs/L9/session_kv_bench.py migration --out "$RUN_ROOT/migration-lora" \
  --session-turns 2 --lora rev_a,rev_b > "$RUN_ROOT/migration-lora.log" 2>&1
echo "$?" > "$RUN_ROOT/migration-lora.exit"
stop_server
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
