#!/usr/bin/env bash
# L5.9 SGLang beam search 探针：起一次服务，落盘原始响应 + 逐 beam 提取。
set -euo pipefail
source /scratch/learn/env.sh
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
PORT=8148
PY=/scratch/learn/envs/sgl/bin/python
run_root="/scratch/learn/work/out/sglang-beam-${1:?pass unique run-id}"
mkdir -p "$run_root"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi
HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" --port $PORT \
    --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.45 \
    > "$run_root/server.log" 2>&1 &
SRV=$!
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true; }
trap stop EXIT
for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    echo 'server not healthy' > "$run_root/server-timeout.txt"; exit 1; }
set +e
$PY -u /scratch/learn/work/labs/L5/sglang_beam_probe.py --base "http://127.0.0.1:${PORT}" \
    --out "$run_root" > "$run_root/probe.log" 2>&1
printf '%s\n' "$?" > "$run_root/exit.txt"
set -e
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-after.txt"
tail -8 "$run_root/probe.log"
