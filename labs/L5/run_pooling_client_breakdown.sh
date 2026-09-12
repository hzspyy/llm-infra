#!/usr/bin/env bash
set -euo pipefail

# 5.12 客户端残差分段插桩
# 需要先启动 vLLM pooling 服务

TIMESTAMP=$(date +"%Y%m%d-%H%M")
OUT_DIR="/scratch/learn/work/out/pooling/client-breakdown-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# 检查服务是否运行
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Error: vLLM server not running on port 8000"
    echo "Please start with: vllm serve ... --port 8000"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Running client breakdown measurement..."
source /scratch/learn/env.sh
/scratch/learn/envs/serve/bin/python \
  /scratch/learn/work/labs/L5/pooling_client_breakdown.py \
  --base-url http://localhost:8000 \
  --text "The prefix cache mechanism enables multiple requests to reuse identical prompt prefixes efficiently." \
  --repeats 30 \
  --out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/audit.log"

echo $? > "$OUT_DIR/exit.txt"

echo "[$(date +%H:%M:%S)] Done. Output in $OUT_DIR"
