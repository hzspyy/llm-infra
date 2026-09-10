#!/usr/bin/env bash
# L5.7 nanoserve：六次迭代 + HTTP 驱动。每次运行写新目录。
set -u
source /scratch/learn/env.sh
cd /scratch/learn/work/labs/L5/nanoserve
PY=/scratch/learn/envs/serve/bin/python
OUT=/scratch/learn/work/out
stamp=$(date +%Y%m%d-%H%M)

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
  > "$OUT/nanoserve-$stamp.gpu-before.txt"
$PY run_nanoserve.py --out "$OUT/nanoserve-$stamp" > "$OUT/nanoserve-$stamp.log" 2>&1
echo "stages=$?"
$PY drive_server.py --out "$OUT/nanoserve-http-$stamp" --port 8137 \
  > "$OUT/nanoserve-http-$stamp.log" 2>&1
echo "http=$?"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
  > "$OUT/nanoserve-$stamp.gpu-after.txt"
