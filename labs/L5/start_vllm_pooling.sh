#!/usr/bin/env bash
set -euo pipefail

# 启动 vLLM pooling 服务

source /scratch/learn/env.sh

MODEL="BAAI/bge-small-en-v1.5"
PORT=8000
LOG_DIR="/scratch/learn/work/out/pooling"
mkdir -p "$LOG_DIR"

echo "[$(date +%H:%M:%S)] Starting vLLM pooling server..."
echo "Model: $MODEL"
echo "Port: $PORT"

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
nohup /scratch/learn/envs/serve/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --runner pooling \
  --convert embed \
  --port $PORT \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  > "$LOG_DIR/vllm-pooling-server.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务就绪
echo "Waiting for server to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "Server ready!"
    exit 0
  fi
  sleep 1
done

echo "Server failed to start in 30 seconds"
exit 1
