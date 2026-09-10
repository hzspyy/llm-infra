#!/usr/bin/env bash
# L5.8：nanoserve 的失败路径 + 真实 vLLM 的抢占与取消。每次运行写新目录。
set -u
source /scratch/learn/env.sh
PY=/scratch/learn/envs/serve/bin/python
OUT=/scratch/learn/work/out
stamp=$(date +%Y%m%d-%H%M)

cd /scratch/learn/work/labs/L5/nanoserve
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/failures-$stamp.gpu-before.txt"
$PY run_failures.py --out "$OUT/failures-$stamp" > "$OUT/failures-$stamp.log" 2>&1
echo "nanoserve=$?"
# HTTP 断连那条路复用 5.7 的 server.py 与 drive_server.py
$PY drive_server.py --out "$OUT/failures-http-$stamp" --port 8141 \
  > "$OUT/failures-http-$stamp.log" 2>&1
echo "http=$?"

cd /scratch/learn/work/labs/L5
L58_GPU_MEMORY_UTILIZATION=0.15 $PY vllm_failure_paths.py \
  --out "$OUT/vllm-failures-$stamp" > "$OUT/vllm-failures-$stamp.log" 2>&1
echo "vllm=$?"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/failures-$stamp.gpu-after.txt"
