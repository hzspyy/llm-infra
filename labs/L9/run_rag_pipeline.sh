#!/usr/bin/env bash
# L9.6：起一个 embedding + 生成服务，跑检索参照、ANN 规模对照、端到端与片段 KV 探针。
# ANN 规模实验（10 万/100 万向量）在 CPU 上跑，放在生成服务之前，避免占显存。
#
#   bash labs/L9/run_rag_pipeline.sh <run_root>
set -u
source /scratch/learn/env.sh
# /reset_prefix_cache 属于 dev 路由，需要显式打开
export VLLM_SERVER_DEV_MODE=1

RUN_ROOT=${1:?usage: run_rag_pipeline.sh <run_root>}
PORT=${L96_PORT:-8016}
GEN_MODEL=${L96_GEN_MODEL:-Qwen/Qwen3-4B}
EMB_MODEL=${L96_EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B}
PY=/scratch/learn/envs/serve/bin/python

mkdir -p "$RUN_ROOT"
cd /scratch/learn

# ---- 先起 embedding 服务，算向量并做检索参照 ----
$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$EMB_MODEL" --served-model-name "$EMB_MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization "${L96_EMB_UTIL:-0.15}" --no-enable-log-requests \
  > "$RUN_ROOT/server-embed.log" 2>&1 &
EMB_PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[embed] ready after ${i}s"; break; }
  kill -0 "$EMB_PID" 2>/dev/null || { tail -20 "$RUN_ROOT/server-embed.log"; exit 1; }
  sleep 1
done
$PY -u work/labs/L9/rag_pipeline.py retrieve --base-url "http://127.0.0.1:$PORT/v1" \
  --embed-model "$EMB_MODEL" --out "$RUN_ROOT/retrieve" > "$RUN_ROOT/retrieve.log" 2>&1
echo "$?" > "$RUN_ROOT/retrieve.exit"
kill "$EMB_PID" 2>/dev/null; wait "$EMB_PID" 2>/dev/null; sleep 4

# ---- ANN 规模对照（CPU） ----
$PY -u work/labs/L9/rag_pipeline.py scale --vectors "$RUN_ROOT/retrieve" \
  --out "$RUN_ROOT/scale" > "$RUN_ROOT/scale.log" 2>&1
echo "$?" > "$RUN_ROOT/scale.exit"

# ---- 生成服务：端到端与片段 KV 探针 ----
$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$GEN_MODEL" --served-model-name "$GEN_MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization "${L96_GEN_UTIL:-0.55}" \
  --enable-prefix-caching --enable-prompt-tokens-details --no-enable-log-requests \
  > "$RUN_ROOT/server-gen.log" 2>&1 &
GEN_PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[gen] ready after ${i}s"; break; }
  kill -0 "$GEN_PID" 2>/dev/null || { tail -20 "$RUN_ROOT/server-gen.log"; exit 1; }
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-gen.txt"

$PY -u work/labs/L9/rag_pipeline.py e2e --base-url "http://127.0.0.1:$PORT/v1" \
  --vectors "$RUN_ROOT/retrieve" --out "$RUN_ROOT/e2e-norerank" --n 200 \
  > "$RUN_ROOT/e2e-norerank.log" 2>&1
echo "$?" > "$RUN_ROOT/e2e-norerank.exit"

$PY -u work/labs/L9/rag_pipeline.py e2e --base-url "http://127.0.0.1:$PORT/v1" \
  --vectors "$RUN_ROOT/retrieve" --out "$RUN_ROOT/e2e-rerank" --n 200 --rerank \
  > "$RUN_ROOT/e2e-rerank.log" 2>&1
echo "$?" > "$RUN_ROOT/e2e-rerank.exit"

$PY -u work/labs/L9/rag_kv_probe.py --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$RUN_ROOT/kv-probe" > "$RUN_ROOT/kv-probe.log" 2>&1
echo "$?" > "$RUN_ROOT/kv-probe.exit"

kill "$GEN_PID" 2>/dev/null; wait "$GEN_PID" 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
