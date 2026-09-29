#!/usr/bin/env bash
# L9.2 任务 C：MCP 三条路径（direct / stdio / Streamable HTTP）× 并发 × 注入延迟的单轴扫描，
# 外加冷连接对照与下一轮 prefill 的引擎测量。
#
#   bash labs/L9/run_mcp_tool_path.sh <run_root>
#
# 输出：caps/（协商结果与 tools/list）、bench/（复用连接扫描）、cold/（每次重建连接）、
#       next-turn/（同一份工具结果回填给模型后的 TTFT）
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_mcp_tool_path.sh <run_root>}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9
PORT_JSON=${L92_JSON_PORT:-8071}
PORT_SSE=${L92_SSE_PORT:-8072}
PORT_ENGINE=${L92_ENGINE_PORT:-8011}
MODEL=${L92_MODEL:-Qwen/Qwen3-4B}
CALLS=${L92_CALLS:-64}

mkdir -p "$RUN_ROOT"
cd /scratch/learn

# --- 两个 HTTP 服务（JSON 回包 / SSE 回包） ------------------------------------------
$PY -u "$LABS/mcp_tool_path.py" serve-http --port "$PORT_JSON" --json \
  > "$RUN_ROOT/server-json.log" 2>&1 &
JSON_PID=$!
$PY -u "$LABS/mcp_tool_path.py" serve-http --port "$PORT_SSE" --sse \
  > "$RUN_ROOT/server-sse.log" 2>&1 &
SSE_PID=$!
cleanup_http() {
  kill "$JSON_PID" "$SSE_PID" 2>/dev/null
  wait "$JSON_PID" "$SSE_PID" 2>/dev/null
}
trap cleanup_http EXIT

for i in $(seq 1 60); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT_JSON/mcp" -X POST \
      -H 'content-type: application/json' \
      -H 'accept: application/json, text/event-stream' \
      -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' ; then
    echo "[http] ready after ${i}s"
    break
  fi
  sleep 1
done

# --- caps：协商版本、能力、tools/list ------------------------------------------------
$PY -u "$LABS/mcp_tool_path.py" caps --out "$RUN_ROOT/caps" > "$RUN_ROOT/caps.log" 2>&1
echo "$?" > "$RUN_ROOT/caps.exit"

# --- bench：复用连接下的单轴扫描 ------------------------------------------------------
$PY -u "$LABS/mcp_tool_path.py" bench --out "$RUN_ROOT/bench" \
  --stdio --http-json "http://127.0.0.1:$PORT_JSON/mcp" --http-sse "http://127.0.0.1:$PORT_SSE/mcp" \
  --concurrency 1,8,32 --delay-ms 0,10,1000 --calls "$CALLS" \
  > "$RUN_ROOT/bench.log" 2>&1
echo "$?" > "$RUN_ROOT/bench.exit"

# --- cold：每次调用重建连接并重新 initialize -----------------------------------------
$PY -u "$LABS/mcp_tool_path.py" bench --out "$RUN_ROOT/cold" \
  --stdio --http-json "http://127.0.0.1:$PORT_JSON/mcp" \
  --concurrency 1 --delay-ms 0 --calls 8 --no-reuse \
  > "$RUN_ROOT/cold.log" 2>&1
echo "$?" > "$RUN_ROOT/cold.exit"

# --- 下一轮 prefill：工具结果回填给模型后的首 token --------------------------------
/scratch/learn/envs/serve/bin/python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT_ENGINE" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L92_GPU_UTIL:-0.45}" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-log-requests \
  > "$RUN_ROOT/server-engine.log" 2>&1 &
ENGINE_PID=$!
cleanup_engine() {
  kill "$ENGINE_PID" 2>/dev/null
  wait "$ENGINE_PID" 2>/dev/null
}
trap 'cleanup_engine; cleanup_http' EXIT

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT_ENGINE/v1/models" > /dev/null; then
    echo "[engine] ready after ${i}s"
    break
  fi
  kill -0 "$ENGINE_PID" 2>/dev/null || { tail -30 "$RUN_ROOT/server-engine.log"; exit 1; }
  sleep 1
done

$PY -u "$LABS/mcp_tool_path.py" next-turn \
  --out "$RUN_ROOT/next-turn" --base-url "http://127.0.0.1:$PORT_ENGINE/v1" --model "$MODEL" \
  --bench "$RUN_ROOT/bench/mcp_path.json" --payload-bytes 96,4096,24576 \
  > "$RUN_ROOT/next-turn.log" 2>&1
echo "$?" > "$RUN_ROOT/next-turn.exit"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
