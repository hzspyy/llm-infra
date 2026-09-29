#!/usr/bin/env bash
# 5.11：vLLM 的 --api-server-count 与 SGLang 的 --tokenizer-worker-num 对照。
set -uo pipefail
source /scratch/learn/env.sh
PYV=/scratch/learn/envs/serve/bin/python
PYS=/scratch/learn/envs/sgl/bin/python
MODEL=${MW_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/multiworker-${1:?pass run id}
VPORT=8180
SPORT=8181
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

wait_health() { for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

run_vllm() {  # $1 标签 $2 额外参数
    local tag="$1"; shift
    HF_HUB_OFFLINE=1 $PYV -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name m --port $VPORT --dtype bfloat16 \
        --max-model-len 4096 --gpu-memory-utilization 0.28 \
        --no-enable-prefix-caching --no-enable-log-requests "$@" \
        > "$RUN/vllm-$tag.log" 2>&1 &
    local srv=$!
    if wait_health $VPORT; then
        $PYV -u /scratch/learn/work/labs/L5/multiworker_audit.py \
            --base "http://127.0.0.1:$VPORT" --label "$tag" --out "$RUN" \
            > "$RUN/audit-vllm-$tag.log" 2>&1
    else
        echo "vllm($tag) not healthy" > "$RUN/timeout-$tag.txt"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
}

run_sglang() {  # $1 标签 $2... 额外参数
    local tag="$1"; shift
    HF_HUB_OFFLINE=1 $PYS -m sglang.launch_server --model-path "$MODEL" \
        --port $SPORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
        --attention-backend triton --mem-fraction-static 0.30 "$@" \
        > "$RUN/sglang-$tag.log" 2>&1 &
    local srv=$!
    if wait_health $SPORT; then
        # SGLang 的 /generate 收 text+sampling_params，客户端脚本已经是这个形状
        $PYS -u /scratch/learn/work/labs/L5/multiworker_audit.py \
            --base "http://127.0.0.1:$SPORT" --label "$tag" --out "$RUN" \
            > "$RUN/audit-sglang-$tag.log" 2>&1
    else
        echo "sglang($tag) not healthy" > "$RUN/timeout-$tag.txt"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
}

run_vllm vllm_api1 --api-server-count 1
run_vllm vllm_api4 --api-server-count 4
run_sglang sglang_tok1 --tokenizer-worker-num 1 --detokenizer-worker-num 1
run_sglang sglang_tok4 --tokenizer-worker-num 4 --detokenizer-worker-num 4

for f in "$RUN"/multiworker_*.json; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f")"
    /scratch/learn/envs/serve/bin/python -c "
import json,sys
d=json.load(open('$f')); print(json.dumps(d['summary'], ensure_ascii=False))"
done
