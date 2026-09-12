#!/usr/bin/env bash
set -euo pipefail

# 运行客户端分段计时实验

cd /scratch/learn/work
source /scratch/learn/env.sh

TIMESTAMP=$(date +%Y%m%d-%H%M)
OUT_DIR="/scratch/learn/work/out/pooling/client-breakdown-$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] Running client breakdown..."

# 确保 vLLM 服务在运行
if ! curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "Error: vLLM server not running"
    exit 1
fi

/scratch/learn/envs/serve/bin/python3 labs/L5/pooling_client_breakdown.py \
  --host localhost \
  --port 8000 \
  --model BAAI/bge-small-en-v1.5 \
  --text "The quick brown fox jumps over the lazy dog." \
  --repeats 30 \
  --out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/benchmark.log"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
