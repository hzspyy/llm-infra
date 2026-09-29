#!/usr/bin/env bash
# L5.11 SGLang 前端对照：动态批分词器 关 / 开 各起一次服务，跑同一批请求。
# 用法：bash run_sglang_frontend.sh <run-id>
set -euo pipefail
source /scratch/learn/env.sh
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
PORT=8145
PY=/scratch/learn/envs/sgl/bin/python
OUT=/scratch/learn/work/out
run_root="$OUT/sglang-frontend-${1:?pass unique run-id}"
mkdir -p "$run_root"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi

run_one() {  # $1 = 标签, $2... = 额外的分词器参数
    local tag="$1"; shift
    echo "=== $tag ==="
    HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
        --port $PORT --host 127.0.0.1 --dtype bfloat16 \
        --disable-radix-cache --attention-backend triton \
        --mem-fraction-static 0.45 "$@" > "$run_root/server-$tag.log" 2>&1 &
    local srv=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "server($tag) not healthy" > "$run_root/timeout-$tag.txt"
        kill $srv 2>/dev/null || true; wait $srv 2>/dev/null || true
        return 1
    fi
    $PY -u /scratch/learn/work/labs/L5/sglang_frontend_audit.py \
        --base "http://127.0.0.1:${PORT}" --out "$run_root/$tag" \
        > "$run_root/$tag.log" 2>&1 || true
    kill $srv 2>/dev/null || true
    wait $srv 2>/dev/null || true
    sleep 3
}

run_one dynamic_off
run_one dynamic_on --enable-dynamic-batch-tokenizer --dynamic-batch-tokenizer-batch-size 32

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-after.txt"
echo "--- dynamic_off ---"; sed -n '/前端成本/,$p' "$run_root/dynamic_off.log" 2>/dev/null | head -8
echo "--- dynamic_on ---";  sed -n '/前端成本/,$p' "$run_root/dynamic_on.log" 2>/dev/null | head -8
