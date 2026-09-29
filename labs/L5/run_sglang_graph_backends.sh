#!/usr/bin/env bash
# L5.10-J 补测 · SGLang decode 图后端三档扫描（full / breakable / disabled）。
#
# 5.10-J 只对照了「图开 / 图关」，没有区分 `breakable` 与 `full`。本脚本用同一模型、
# 同一批请求把 decode 后端扫到三档，并抓服务端日志里的 capture 行作为进程内证据。
#
# 用法：RUN_ID=20260922 bash labs/L5/run_sglang_graph_backends.sh
set -uo pipefail

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/sgl/bin/python}
MODEL=${SGL_MODEL:-$ROOT/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
PORT=${PORT:-8162}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
RUN=${RUN:-$ROOT/work/out/sglang-graph-backends-$RUN_ID}
mkdir -p "$RUN"

source "$ROOT/env.sh"
export HF_HUB_OFFLINE=1

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" > "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

run_one() {  # $1 标签，其余为 decode 后端参数
    local tag="$1"; shift
    pkill -f "port $PORT" 2>/dev/null; sleep 2
    HF_HUB_OFFLINE=1 "$PY" -m sglang.launch_server --model-path "$MODEL" \
        --port "$PORT" --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
        --attention-backend triton --mem-fraction-static 0.30 "$@" \
        > "$RUN/server-$tag.log" 2>&1 &
    local srv=$!
    local ok=0
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        sleep 1
    done
    if [ "$ok" != "1" ]; then
        echo "server($tag) not healthy" > "$RUN/timeout-$tag.txt"
        kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; return 1
    fi
    sleep 3
    grep -E "Capture target|cuda graph:|CUDA graph" "$RUN/server-$tag.log" | head -8 \
        > "$RUN/capture-$tag.txt"
    "$PY" -u "$ROOT/work/labs/L5/sglang_graph_audit.py" \
        --base "http://127.0.0.1:$PORT" --label "$tag" --out "$RUN" \
        > "$RUN/audit-$tag.log" 2>&1
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 3
}

run_one full --cuda-graph-backend-decode full
run_one breakable --cuda-graph-backend-decode breakable
run_one disabled --cuda-graph-backend-decode disabled

echo "== capture 行 =="
for tag in full breakable disabled; do
    echo "--- $tag"; head -3 "$RUN/capture-$tag.txt" 2>/dev/null
done
echo "== 结果 =="
for tag in full breakable disabled; do
    echo "--- $tag"; tail -4 "$RUN/audit-$tag.log" 2>/dev/null
done
echo "工件目录：$RUN"
