#!/usr/bin/env bash
# L9.5：起 vLLM 服务，跑正常路径、四个注入点的故障矩阵与取消恢复。
#
#   bash labs/L9/run_agent_runtime.sh <run_root>
set -u
source /scratch/learn/env.sh

RUN_ROOT=${1:?usage: run_agent_runtime.sh <run_root>}
PORT=${L95_PORT:-8015}
MODEL=${L95_MODEL:-Qwen/Qwen3-4B}
PY=/scratch/learn/envs/serve/bin/python

mkdir -p "$RUN_ROOT"
cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${L95_GPU_UTIL:-0.45}" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "[server] ready after ${i}s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -20 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"

# 正常路径
$PY -u work/labs/L9/agent_runtime.py run --out "$RUN_ROOT/happy" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" > "$RUN_ROOT/happy.log" 2>&1
echo "$?" > "$RUN_ROOT/happy.exit"

# 故障矩阵（每个格子自己起进程）
$PY -u work/labs/L9/agent_failure_matrix.py --root "$RUN_ROOT/matrix" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --python "$PY" \
  > "$RUN_ROOT/matrix.log" 2>&1
echo "$?" > "$RUN_ROOT/matrix.exit"

# 版本不一致时拒绝恢复：改掉工具 schema 后再 recover
$PY -u work/labs/L9/agent_runtime.py run --out "$RUN_ROOT/version-check" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --crash-at tool_running \
  > "$RUN_ROOT/version-check-crash.log" 2>&1
VERIFY_OUT="$RUN_ROOT/version-check" $PY - <<'PYEOF' > "$RUN_ROOT/version-check.log" 2>&1
import json, os, pathlib, sys
sys.path.insert(0, "work/labs/L9")
import agent_runtime as R
out = pathlib.Path(os.environ["VERIFY_OUT"])
# 伪造一次 schema 变更：把记录里的 tool_schema_hash 改成别的值，再走 recover
p = out / R.EVENTS
lines = p.read_text(encoding="utf-8").splitlines()
for i, l in enumerate(lines):
    rec = json.loads(l)
    if rec["kind"] == "session_start":
        rec["tool_schema_hash"] = "deadbeefdeadbeef"
        lines[i] = json.dumps(rec, ensure_ascii=False)
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("patched session_start.tool_schema_hash")
PYEOF
echo "$?" > "$RUN_ROOT/version-check-patch.exit"
$PY -u work/labs/L9/agent_runtime.py recover --out "$RUN_ROOT/version-check" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" >> "$RUN_ROOT/version-check.log" 2>&1
echo "$?" > "$RUN_ROOT/version-check-recover.exit"

# 取消
$PY -u work/labs/L9/agent_runtime.py run --out "$RUN_ROOT/cancel" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --cancel-at 2 \
  > "$RUN_ROOT/cancel.log" 2>&1
echo "$?" > "$RUN_ROOT/cancel.exit"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
