#!/usr/bin/env bash
# 5.11：vLLM 的 --api-server-count 1 vs 4（同一 EngineCore，多个前端进程）。
set -uo pipefail
source /scratch/learn/env.sh
PYV=/scratch/learn/envs/serve/bin/python
MODEL=${MW_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/vllm-apiserver-${1:?pass run id}
PORT=8182
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi
wait_health() { for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

for n in 1 4; do
    HF_HUB_OFFLINE=1 $PYV -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name m --port $PORT --dtype bfloat16 \
        --max-model-len 4096 --gpu-memory-utilization 0.28 \
        --no-enable-prefix-caching --no-enable-log-requests \
        --api-server-count $n > "$RUN/vllm-api$n.log" 2>&1 &
    srv=$!
    if wait_health; then
        $PYV -u /scratch/learn/work/labs/L5/multiworker_audit.py \
            --base "http://127.0.0.1:$PORT" --api openai --served-name m \
            --label "vllm_api$n" --out "$RUN" > "$RUN/audit-api$n.log" 2>&1
    else
        echo "api$n not healthy" > "$RUN/timeout-api$n.txt"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
done
for f in "$RUN"/multiworker_*.json; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f")"
    $PYV -c "
import json; d=json.load(open('$f')); print(json.dumps(d['summary'], ensure_ascii=False))"
done
