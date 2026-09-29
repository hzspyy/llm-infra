#!/usr/bin/env bash
# L9.3：会话 KV 生命周期。三个阶段各自起一个引擎配置：
#   1) 前缀缓存开（KV 容量小，驱逐可达）——结构预测、驻留扫描、迁移对照
#   2) 前缀缓存关 —— 「全部重算」参照
#   3) 开 LoRA 两个 adapter —— 缓存身份是否包含 adapter
#
#   bash labs/L9/run_session_kv_bench.sh <run_root>
set -u
source /scratch/learn/env.sh
# /reset_prefix_cache 属于 dev 路由，需要显式打开
export VLLM_SERVER_DEV_MODE=1

RUN_ROOT=${1:?usage: run_session_kv_bench.sh <run_root>}
PORT=${L93_PORT:-8014}
MODEL=${L93_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python
LORA=${L93_LORA:-trl-lib/Qwen3-4B-LoRA}

mkdir -p "$RUN_ROOT"
cd /scratch/learn

start_server() {  # start_server <log> <extra args...>
  local log=$1; shift
  $PY -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization "${L93_GPU_UTIL:-0.45}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-prompt-tokens-details --no-enable-log-requests \
    "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s ($log)"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -20 "$log"; return 1; }
    sleep 1
  done
  return 1
}

stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 4; }

# ---- 阶段 1：前缀缓存开 ----
start_server "$RUN_ROOT/server-prefix.log" --enable-prefix-caching || exit 1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-prefix.txt"
grep -o "GPU KV cache size: [0-9,]* tokens" "$RUN_ROOT/server-prefix.log" > "$RUN_ROOT/kv_capacity.txt" || true

$PY -u work/labs/L9/session_kv_bench.py structure --out "$RUN_ROOT/structure" \
  --turns 2 4 8 > "$RUN_ROOT/structure.log" 2>&1; echo "$?" > "$RUN_ROOT/structure.exit"
$PY -u work/labs/L9/session_kv_bench.py residency --out "$RUN_ROOT/residency-cacheon" \
  --gaps 0 1 10 60 --pressure 0 24 --prefix-caching > "$RUN_ROOT/residency-cacheon.log" 2>&1
echo "$?" > "$RUN_ROOT/residency-cacheon.exit"
$PY -u work/labs/L9/session_kv_bench.py migration --out "$RUN_ROOT/migration" \
  --session-turns 4 > "$RUN_ROOT/migration.log" 2>&1; echo "$?" > "$RUN_ROOT/migration.exit"
stop_server

# ---- 阶段 2：前缀缓存关（全部重算参照） ----
start_server "$RUN_ROOT/server-noprefix.log" --no-enable-prefix-caching || exit 1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-noprefix.txt"
$PY -u work/labs/L9/session_kv_bench.py residency --out "$RUN_ROOT/residency-cacheoff" \
  --gaps 0 10 --pressure 0 --session-turns 4 > "$RUN_ROOT/residency-cacheoff.log" 2>&1
echo "$?" > "$RUN_ROOT/residency-cacheoff.exit"
stop_server

# ---- 阶段 3：LoRA adapter 身份 ----
if [ -d "$HF_HOME/hub/models--trl-lib--Qwen3-4B-LoRA" ]; then
  LORA_B="$RUN_ROOT/lora-copy"
  mkdir -p "$LORA_B"
  cp -r "$HF_HOME/hub/models--trl-lib--Qwen3-4B-LoRA"/snapshots/*/ "$LORA_B/" 2>/dev/null || true
  start_server "$RUN_ROOT/server-lora.log" --enable-prefix-caching --enable-lora \
    --lora-modules "rev_a=$LORA" "rev_b=$LORA_B" || exit 1
  $PY -u work/labs/L9/session_kv_bench.py migration --out "$RUN_ROOT/migration-lora" \
    --session-turns 2 --lora rev_a,rev_b > "$RUN_ROOT/migration-lora.log" 2>&1
  echo "$?" > "$RUN_ROOT/migration-lora.exit"
  stop_server
fi

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
