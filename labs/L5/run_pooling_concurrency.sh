#!/usr/bin/env bash
# L5.12 补测 · 起 vLLM 池化服务 → 并发容量与残差扫描 → 停服务（crater）。
#
# 除扫描结果外，本脚本把启动日志里的显存分段（KV/图池/workspace/激活峰值）抽出来，
# 供"同一 checkpoint 的分段账"引用。
#
# 用法：RUN_ID=20260922 bash labs/L5/run_pooling_concurrency.sh
set -uo pipefail

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
MODEL=${POOL_MODEL:-Qwen/Qwen3-Embedding-0.6B}
PORT=${PORT:-8125}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/pooling-concurrency-$RUN_ID}
LEVELS=${LEVELS:-1,8,32,64,128,256}
REQCAP=${REQCAP:-512}
EXTRA=${EXTRA:-}
TEXT_REPEAT=${TEXT_REPEAT:-3}
mkdir -p "$OUT"

source "$ROOT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

pkill -f "port $PORT" 2>/dev/null
sleep 2

HF_HUB_OFFLINE=1 nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --runner pooling --convert embed \
    --port "$PORT" --host 127.0.0.1 --dtype bfloat16 \
    --gpu-memory-utilization 0.45 \
    > "$OUT/server.log" 2>&1 &
echo $! > "$OUT/server.pid"

for i in $(seq 1 300); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        "http://127.0.0.1:$PORT/pooling" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"input\":\"ready\"}" 2>/dev/null || echo 000)
    [ "$CODE" = "200" ] && { echo "服务就绪（${i}s）"; break; }
    sleep 1
done

# 启动日志里的分段账（KV / 图池 / workspace / 激活峰值）
grep -E "KV cache|graph memory|Graph memory|workspace|activation" "$OUT/server.log" \
    > "$OUT/memory_segments.txt" 2>/dev/null || true

"$PY" "$ROOT/work/labs/L5/pooling_concurrency_limit.py" --base "http://127.0.0.1:$PORT" \
    --out "$OUT" --model "$MODEL" --levels "$LEVELS" --requests-cap "$REQCAP" \
    --text-repeat "$TEXT_REPEAT" \
    $EXTRA 2>&1 | tail -30

kill "$(cat "$OUT/server.pid")" 2>/dev/null
sleep 5
pkill -f "port $PORT" 2>/dev/null
echo "POOLING_DONE"
