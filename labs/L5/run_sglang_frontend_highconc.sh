#!/usr/bin/env bash
# 5.11：动态批分词器在**它针对的场景**（高并发 + 短 prompt）下有没有收益。
# 上一轮在 batch ≤ 32、prompt ≤ 4096 下测不到收益；这次把并发推到 128/256、prompt 压到 8/32。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/sglang-frontend-highconc-${1:?pass run id}
PORT=8162
LENGTHS="${LENGTHS:-8 32}"
BATCHES="${BATCHES:-64 128 256}"
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

run_one() {
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
    $PY -u /scratch/learn/work/labs/L5/sglang_frontend_audit.py \
        --base "http://127.0.0.1:$PORT" --out "$RUN" --lengths $LENGTHS \
        --batches $BATCHES --repeats 3 > "$RUN/audit-$tag.log" 2>&1
    # 每次运行写同名文件，改名保留两侧
    [ -f "$RUN/sglang_frontend.json" ] && mv "$RUN/sglang_frontend.json" "$RUN/sglang_frontend-$tag.json"
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
}

run_one dynamic_off
run_one dynamic_on --enable-dynamic-batch-tokenizer --dynamic-batch-tokenizer-batch-size 32
echo "--- off ---"; sed -n '/前端成本/,$p' "$RUN/audit-dynamic_off.log" | head -8
echo "--- on ---";  sed -n '/前端成本/,$p' "$RUN/audit-dynamic_on.log" | head -8
