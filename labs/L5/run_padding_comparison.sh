#!/usr/bin/env bash
set -euo pipefail

# 运行 padding 对照实验

cd /scratch/learn/work
source /scratch/learn/env.sh

TIMESTAMP=$(date +%Y%m%d-%H%M)
OUT_DIR="/scratch/learn/work/out/pooling/padding-comparison-$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] Running padding comparison..."

# 确保 vLLM 服务在运行
if ! curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "Error: vLLM server not running"
    exit 1
fi

/scratch/learn/envs/serve/bin/python3 labs/L5/pooling_padding_comparison.py \
  --base-url http://localhost:8000 \
  --model BAAI/bge-small-en-v1.5 \
  --repeats 20 \
  --out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/benchmark.log"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
