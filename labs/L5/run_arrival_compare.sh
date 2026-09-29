#!/usr/bin/env bash
# L5.3-C · 单引擎 ARRIVAL 扫描：起服务 → 闭环基线 + 0.3/0.6/0.9/1.1 倍 × 3 窗口 → 停服务。
#
# 两引擎必须固定同一组形状与预算，否则量到的是配置差：
#   prompt 2048 / 输出 128；vLLM 与 SGLang 都关掉前缀缓存、图执行保持默认。
#
# 用法： run_arrival_compare.sh <vllm|sglang> [out_dir] [port]
set -uo pipefail

ENGINE="${1:?用法: run_arrival_compare.sh <vllm|sglang> [out_dir] [port]}"
OUT="${2:-/scratch/learn/work/out/5.3/arrival-$ENGINE}"
PORT="${3:-8100}"
RATES="${RATES:-0.3,0.6,0.9,1.1}"
WINDOWS="${WINDOWS:-3}"
DURATION="${DURATION:-120}"
CONCURRENCY="${CONCURRENCY:-64}"
BASELINE_REQUESTS="${BASELINE_REQUESTS:-600}"
source /scratch/learn/env.sh
# 离线：引擎初始化时若去连 HuggingFace 会直接失败
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"

# 上一轮的服务可能还占着端口
pkill -f "port $PORT" 2>/dev/null
sleep 2

case "$ENGINE" in
  vllm)
    PY=/scratch/learn/envs/serve/bin/python
    nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-1.7B --port "$PORT" --host 127.0.0.1 \
      --max-model-len 16384 --gpu-memory-utilization 0.45 \
      --max-num-batched-tokens 8192 --dtype bfloat16 \
      --no-enable-prefix-caching > "$OUT/server.log" 2>&1 &
    ;;
  sglang)
    PY=/scratch/learn/envs/sgl/bin/python
    nohup "$PY" -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
      --port "$PORT" --host 127.0.0.1 --context-length 16384 \
      --mem-fraction-static 0.45 --page-size 16 --chunked-prefill-size 8192 \
      --enable-metrics --disable-radix-cache > "$OUT/server.log" 2>&1 &
    ;;
  *)
    echo "未知引擎 $ENGINE"; exit 2 ;;
esac
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"

# 就绪判据必须是"能真正完成一次推理"，不能只看 /health：
# SGLang 的 HTTP 端口先起来，模型还在加载，此时提交的请求会整批失败。
READY=0
for i in $(seq 1 300); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-1.7B","prompt":"ready","max_tokens":1,"temperature":0}' \
    2>/dev/null || echo 000)
  if [ "$CODE" = "200" ]; then
    echo "服务就绪（完成一次预热请求），用时 ${i}s"
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "服务在 300s 内未就绪，放弃本次扫描"
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi

"$PY" /scratch/learn/work/labs/L5/arrival_scan.py --engine "$ENGINE" \
  --base-url "http://127.0.0.1:$PORT" --out "$OUT" \
  --rates "$RATES" --windows "$WINDOWS" --duration "$DURATION" \
  --concurrency "$CONCURRENCY" --baseline-requests "$BASELINE_REQUESTS" 2>&1 | tail -40

kill "$SERVER_PID" 2>/dev/null
sleep 8
pkill -f "port $PORT" 2>/dev/null
sleep 3
echo "ARRIVAL_DONE $ENGINE"
