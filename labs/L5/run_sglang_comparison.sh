#!/usr/bin/env bash
set -euo pipefail

# 运行 SGLang pooling 对照测试

cd /scratch/learn/work
source /scratch/learn/env.sh

TIMESTAMP=$(date +%Y%m%d-%H%M)
OUT_DIR="/scratch/learn/work/out/pooling/sglang-comparison-$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] Running SGLang pooling benchmark..."

/scratch/learn/envs/serve/bin/python3 labs/L5/pooling_sglang_comparison.py \
  --base-url http://localhost:8000 \
  --model BAAI/bge-small-en-v1.5 \
  --engine sglang \
  --repeats 30 \
  --out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/benchmark.log"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
