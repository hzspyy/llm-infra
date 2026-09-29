#!/usr/bin/env bash
# L9.6 端到端阶段：HotpotQA 固定 200 题的向量离线准备 + 两种 arm 的端到端 + 片段 KV 探针。
# embedding 与生成是两次独立启动，e2e 用离线向量（一个 vLLM 实例只服务一个模型）。
#
#   bash labs/L9/run_rag_e2e.sh <run_root>
# 环境变量：L96_GEN_UTIL（默认 0.55）、L96_CONTEXT_CHARS（默认 24000）、L96_CONC（默认 8）
set -u
source /scratch/learn/env.sh
export VLLM_SERVER_DEV_MODE=1
RUN_ROOT=${1:?usage: run_rag_e2e.sh <run_root>}
PORT=${L96_PORT:-8018}
PY=/scratch/learn/envs/serve/bin/python
GEN_MODEL=${L96_GEN_MODEL:-Qwen/Qwen3-4B}
EMB_MODEL=${L96_EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B}
GEN_UTIL=${L96_GEN_UTIL:-0.55}
CTX_CHARS=${L96_CONTEXT_CHARS:-24000}
CONC=${L96_CONC:-8}
mkdir -p "$RUN_ROOT"; cd /scratch/learn

wait_ready() {  # wait_ready <pid> <log>：起不来就退出，绝不带着"服务没起来"继续跑
  for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s"; return 0; }
    kill -0 "$1" 2>/dev/null || { echo "[server] died; tail:"; tail -20 "$2"; return 1; }
    sleep 1
  done
  echo "[server] timeout"; return 1
}

# 1) embedding 服务：准备 HotpotQA 语料/问题向量
$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$EMB_MODEL" --served-model-name "$EMB_MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.15 --no-enable-log-requests \
  > "$RUN_ROOT/server-embed.log" 2>&1 &
EMB=$!
wait_ready "$EMB" "$RUN_ROOT/server-embed.log" || exit 1
$PY -u work/labs/L9/rag_pipeline.py prepare --base-url "http://127.0.0.1:$PORT/v1" \
  --embed-model "$EMB_MODEL" --out "$RUN_ROOT/hotpot-emb" --n 200 > "$RUN_ROOT/prepare.log" 2>&1
echo "$?" > "$RUN_ROOT/prepare.exit"
kill $EMB 2>/dev/null; wait $EMB 2>/dev/null; sleep 15

# 2) 生成服务：两种 arm + KV 探针
$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$GEN_MODEL" --served-model-name "$GEN_MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization "$GEN_UTIL" \
  --enable-prefix-caching --enable-prompt-tokens-details --no-enable-log-requests \
  > "$RUN_ROOT/server-gen.log" 2>&1 &
GEN=$!
wait_ready "$GEN" "$RUN_ROOT/server-gen.log" || exit 1
grep -o "GPU KV cache size: [0-9,]* tokens" "$RUN_ROOT/server-gen.log" > "$RUN_ROOT/kv_capacity.txt" || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-gen.txt"

$PY -u work/labs/L9/rag_pipeline.py e2e --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$RUN_ROOT/e2e-norerank" --embeddings "$RUN_ROOT/hotpot-emb" \
  --n 200 --concurrency "$CONC" --context-chars "$CTX_CHARS" > "$RUN_ROOT/e2e-norerank.log" 2>&1
echo "$?" > "$RUN_ROOT/e2e-norerank.exit"

$PY -u work/labs/L9/rag_pipeline.py e2e --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$RUN_ROOT/e2e-rerank" --embeddings "$RUN_ROOT/hotpot-emb" \
  --n 200 --rerank --concurrency "$CONC" --rerank-batch 16 \
  --context-chars "$CTX_CHARS" > "$RUN_ROOT/e2e-rerank.log" 2>&1
echo "$?" > "$RUN_ROOT/e2e-rerank.exit"

$PY -u work/labs/L9/rag_kv_probe.py --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$RUN_ROOT/kv-probe" > "$RUN_ROOT/kv-probe.log" 2>&1
echo "$?" > "$RUN_ROOT/kv-probe.exit"
kill $GEN 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
