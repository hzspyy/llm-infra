#!/usr/bin/env bash
# 5.10：SGLang 的 CUDA Graph 开关对照（同一模型、同一批请求，只切图后端）。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/sglang-graph-${1:?pass run id}
PORT=8160
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

run_one() {  # $1 标签, $2... 额外参数
    local tag="$1"; shift
    HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
        --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
        --attention-backend triton --mem-fraction-static 0.30 "$@" \
        > "$RUN/server-$tag.log" 2>&1 &
    local srv=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server($tag) not healthy" > "$RUN/timeout-$tag.txt"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; return 1
    fi
    $PY -u /scratch/learn/work/labs/L5/sglang_graph_audit.py \
        --base "http://127.0.0.1:$PORT" --label "$tag" --out "$RUN" \
        > "$RUN/audit-$tag.log" 2>&1
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
}

run_one graph_on
run_one graph_off --cuda-graph-backend-decode disabled

grep -hE "Capture|cuda graph|CUDA graph" "$RUN"/server-graph_*.log | head -4 > "$RUN/capture-lines.txt"
echo "--- graph_on ---";  tail -5 "$RUN/audit-graph_on.log"
echo "--- graph_off ---"; tail -5 "$RUN/audit-graph_off.log"
