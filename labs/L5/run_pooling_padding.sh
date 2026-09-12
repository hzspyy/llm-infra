#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d-%H%M")
OUT_DIR="/scratch/learn/work/out/pooling/padding-controlled-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] GPU status before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-before.txt"

echo "[$(date +%H:%M:%S)] Running padding controlled experiment..."
source /scratch/learn/env.sh
HF_HUB_OFFLINE=1 /scratch/learn/envs/serve/bin/python \
  /scratch/learn/work/labs/L5/pooling_padding_controlled.py \
  --model BAAI/bge-small-en-v1.5 \
  --total-tokens 1024 \
  --num-sequences 16 \
  --out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/audit.log"

echo $? > "$OUT_DIR/exit.txt"

echo "[$(date +%H:%M:%S)] GPU status after:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-after.txt"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
