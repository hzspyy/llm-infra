#!/usr/bin/env bash
# 5.12 客户端/服务端计时对齐：起一次 pooling 服务，同一批请求上采两侧时长。
# 用法：bash run_pooling_align.sh <run-id>
set -euo pipefail
source /scratch/learn/env.sh
MODEL="${POOL_MODEL:-BAAI/bge-small-en-v1.5}"
PORT=8125
PY=/scratch/learn/envs/serve/bin/python
run_root="/scratch/learn/work/out/pooling-align-${1:?pass unique run-id}"
mkdir "$run_root"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi

cp /scratch/learn/work/labs/L5/pooling_client_server_align.py "$run_root/lab.snapshot.py"
cp "$0" "$run_root/run.snapshot.sh"

HF_HUB_OFFLINE=1 $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name pooler --runner pooling --convert embed \
    --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.35 \
    --no-enable-log-requests > "$run_root/server.log" 2>&1 &
SRV=$!
stop() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true; }
trap stop EXIT

for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    echo 'server not healthy' > "$run_root/server-timeout.txt"; exit 1; }

set +e
$PY /scratch/learn/work/labs/L5/pooling_client_server_align.py \
    --base "http://127.0.0.1:${PORT}" --out "$run_root" --repeats 40 \
    > "$run_root/align.log" 2>&1
printf '%s\n' "$?" > "$run_root/exit.txt"
set -e

grep -h "Resolved pooling config" "$run_root/server.log" \
    > "$run_root/server-pooling-config.txt" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-after.txt"
tail -12 "$run_root/align.log"
