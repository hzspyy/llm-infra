#!/usr/bin/env bash
set -euo pipefail

# 5.12 分阶段容量实测运行器
# crater 上运行，确保 GPU 空闲

TIMESTAMP=$(date +"%Y%m%d-%H%M")
OUT_DIR="/scratch/learn/work/out/pooling/capacity-stages-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] GPU status before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-before.txt"

echo "[$(date +%H:%M:%S)] Running capacity stages audit..."
source /scratch/learn/env.sh
HF_HUB_OFFLINE=1 /scratch/learn/envs/serve/bin/python \
  /scratch/learn/work/labs/L5/pooling_capacity_stages.py \
  --encoder BAAI/bge-small-en-v1.5 \
  --decoder Qwen/Qwen2.5-1.5B-Instruct \
  --seq-len 256 \
  --batch 8 \
  --out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/audit.log"

echo $? > "$OUT_DIR/exit.txt"

echo "[$(date +%H:%M:%S)] GPU status after:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-after.txt"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
