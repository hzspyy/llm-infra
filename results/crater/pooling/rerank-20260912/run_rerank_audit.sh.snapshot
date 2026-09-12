#!/usr/bin/env bash
# 5.12 rerank/classifier 真实链路：Qwen3-Reranker-0.6B HF 参考前向 + vLLM HTTP 合约核对。
# 用法：bash run_rerank_audit.sh <run-id>
set -euo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1

RERANK_MODEL="${RERANK_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473}"
PORT=8124
PY=/scratch/learn/envs/serve/bin/python

run_root="/scratch/learn/work/out/rerank-${1:?pass unique run-id}"
mkdir "$run_root"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"
    exit 1
fi

cp /scratch/learn/work/labs/L5/rerank_serving_audit.py "$run_root/rerank_serving_audit.py.snapshot"
cp "$0" "$run_root/run_rerank_audit.sh.snapshot"

# ---- 阶段 1：HF transformers 参考前向（直接用 CUDA，不起 server）------------
set +e
$PY /scratch/learn/work/labs/L5/rerank_serving_audit.py reference \
    --model "$RERANK_MODEL" \
    --out "$run_root" \
    > "$run_root/reference.log" 2>&1
ref_code=$?
printf '%s\n' "$ref_code" > "$run_root/exit-reference.txt"
set -e

if [ "$ref_code" -ne 0 ]; then
    echo "reference stage failed (exit $ref_code); see reference.log" > "$run_root/reference-failed.txt"
fi

# ---- 阶段 2：vLLM 服务端（pooling runner，自动检测 score 任务）---------------
# Qwen3-Reranker 是因果 LM 架构，vLLM 以 --runner pooling 加载并暴露 /score /rerank /classify
$PY -m vllm.entrypoints.openai.api_server \
    --model "$RERANK_MODEL" \
    --served-model-name reranker \
    --port $PORT \
    --runner pooling \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 4096 \
    --no-enable-log-requests \
    > "$run_root/server.log" 2>&1 &
SRV_PID=$!

wait_health() {
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

stop_srv() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null || true; }

set +e
if wait_health; then
    printf '0\n' > "$run_root/startup.txt"
    $PY /scratch/learn/work/labs/L5/rerank_serving_audit.py online \
        --model "$RERANK_MODEL" \
        --out "$run_root" \
        --base "http://127.0.0.1:${PORT}" \
        > "$run_root/online.log" 2>&1
    printf '%s\n' "$?" > "$run_root/exit-online.txt"
else
    printf '2\n' > "$run_root/startup.txt"
    echo 'server did not become healthy within 300 s' > "$run_root/server-timeout.txt"
    printf '2\n' > "$run_root/exit-online.txt"
fi
stop_srv
set -e

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
    > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$run_root/processes-after.txt"

exit 0
