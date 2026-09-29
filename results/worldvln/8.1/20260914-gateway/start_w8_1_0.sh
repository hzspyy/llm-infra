#!/bin/bash
set -e
source /root/learn/env.sh
export CUDA_VISIBLE_DEVICES=2
exec /root/learn/envs/serve/bin/vllm serve Qwen/Qwen3-1.7B --port 18000 \
  --gpu-memory-utilization 0.30 --max-model-len 8192 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:15557","topic":"kv"}'
