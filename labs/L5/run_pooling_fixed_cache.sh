#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d-%H%M")
OUT_DIR="/scratch/learn/work/out/pooling/fixed-cache-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] GPU status before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-before.txt"

echo "[$(date +%H:%M:%S)] Running fixed cache decode timing..."
source /scratch/learn/env.sh
HF_HUB_OFFLINE=1 /scratch/learn/envs/serve/bin/python \
  /scratch/learn/work/labs/L5/pooling_fixed_cache_decode.py \
  --encoder BAAI/bge-small-en-v1.5 \
  --decoder Qwen/Qwen2.5-1.5B-Instruct \
  --seq-len 256 \
  --context-len 256 \
  --batch 8 \
  --repeats 10 \
  --out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/audit.log"

echo $? > "$OUT_DIR/exit.txt"

echo "[$(date +%H:%M:%S)] GPU status after:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_DIR/gpu-after.txt"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
