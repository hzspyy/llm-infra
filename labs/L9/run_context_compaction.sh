#!/usr/bin/env bash
# L9.3 任务 C：四条上下文策略（full / window / summary / external）在同一批多轮累加任务上对照。
#
#   bash labs/L9/run_context_compaction.sh <run_root> [sessions] [turns]
set -u
source /scratch/learn/env.sh
RUN_ROOT=${1:?usage: run_context_compaction.sh <run_root> [sessions] [turns]}
SESSIONS=${2:-12}
TURNS=${3:-8}
PORT=${L93C_PORT:-8020}
PY=/scratch/learn/envs/serve/bin/python
MODEL=${L93C_MODEL:-Qwen/Qwen3-4B}
mkdir -p "$RUN_ROOT"; cd /scratch/learn

$PY -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization "${L93C_GPU_UTIL:-0.45}" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-prefix-caching --enable-prompt-tokens-details --no-enable-log-requests \
  > "$RUN_ROOT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "[server] ready after ${i}s"; break; }
  kill -0 $PID 2>/dev/null || { tail -20 "$RUN_ROOT/server.log"; exit 1; }
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-during.txt"

for policy in full window summary external; do
  $PY -u work/labs/L9/context_compaction.py --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" --out "$RUN_ROOT/$policy" --policy "$policy" \
    --sessions "$SESSIONS" --turns "$TURNS" > "$RUN_ROOT/$policy.log" 2>&1
  echo "$?" > "$RUN_ROOT/$policy.exit"
done

# 汇总四条策略
$PY - "$RUN_ROOT" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
table = {}
for policy in ("full", "window", "summary", "external"):
    f = root / policy / f"compaction-{policy}.json"
    if not f.exists():
        table[policy] = None
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    table[policy] = {k: d[k] for k in ("turns_graded", "correct", "accuracy", "prompt_tokens_mean",
                                       "cached_ratio_mean", "ttft_ms_p50", "e2e_ms_mean",
                                       "recall_calls", "summary_calls")}
(root / "summary.json").write_text(json.dumps({"table": table}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
print(json.dumps(table, ensure_ascii=False, indent=1))
PYEOF
kill $PID 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-after.txt"
