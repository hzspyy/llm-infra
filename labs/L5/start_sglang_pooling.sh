#!/usr/bin/env bash
set -euo pipefail

# 启动 SGLang pooling 服务用于对照测试

MODEL="BAAI/bge-small-en-v1.5"
PORT=8000

echo "[$(date +%H:%M:%S)] Starting SGLang pooling server..."
echo "Model: $MODEL"
echo "Port: $PORT"

source /scratch/learn/env.sh

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
nohup /scratch/learn/envs/sgl/bin/python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --is-embedding \
  --port $PORT \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --disable-radix-cache \
  --attention-backend triton \
  > /scratch/learn/work/out/pooling/sglang-pooling-server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > /scratch/learn/work/out/pooling/sglang-pooling-server.pid

# 等待服务就绪
echo "Waiting for server to be ready..."
for i in {1..60}; do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready!"
    exit 0
  fi
  sleep 1
done

echo "Timeout waiting for server"
exit 1
