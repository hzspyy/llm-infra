#!/usr/bin/env bash
# labs/L8/run_8_1.sh - 8.1 真实双副本网关实验驱动 (在 worldvln 上执行).
#
#   用法: bash labs/L8/run_8_1.sh <run_dir> [bench|events|both]
#
# 起两个 Qwen3-1.7B 副本 (GPU2/GPU3), 打开 KV 事件发布, 等 /v1/models 就绪,
# 再跑 routing_bench 的策略对比或目录对照, 最后停掉两个副本。
set -euo pipefail

RUN_DIR=${1:?run_dir}
MODE=${2:-both}
source /root/learn/env.sh
mkdir -p "$RUN_DIR"

PY=/root/learn/envs/serve/bin/python
VLLM=/root/learn/envs/serve/bin/vllm

# 预检: 8.3/8.1 的 vLLM 都要现场编译 Triton 辅助 .so, 而本机系统 Python 没有
# 开发头文件, 靠 env.sh 里的 CPATH 指到 uv managed Python 的头文件目录;
# HF_HOME 没生效时会重新下载权重并写进根盘。两项都要在起服务前确认。
if [ -z "${CPATH:-}" ] || [ ! -e "${CPATH%%:*}/Python.h" ]; then
  echo "preflight failed: CPATH/Python.h 不可见, 拒绝启动 (会重新下载权重并编译失败)" >&2
  exit 2
fi
if [ -z "${HF_HOME:-}" ] || [ ! -d "$HF_HOME/hub/models--Qwen--Qwen3-1.7B" ]; then
  echo "preflight failed: HF_HOME 未指向本机模型缓存 ($HF_HOME)" >&2
  exit 2
fi

start_worker() {  # gpu port zmq_port name
  local gpu=$1 port=$2 zmq=$3 name=$4
  # 用 bash 并显式 source env.sh: /bin/sh 是 dash, 不认 source, 会静默跳过环境初始化。
  cat > "$RUN_DIR/start_$name.sh" <<EOF
#!/bin/bash
set -e
source /root/learn/env.sh
export CUDA_VISIBLE_DEVICES=$gpu
exec $VLLM serve Qwen/Qwen3-1.7B --port $port \\
  --gpu-memory-utilization 0.30 --max-model-len 8192 \\
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:$zmq","topic":"kv"}'
EOF
  chmod +x "$RUN_DIR/start_$name.sh"
  tmux kill-session -t "$name" 2>/dev/null || true
  tmux new-session -d -s "$name" "bash $RUN_DIR/start_$name.sh > $RUN_DIR/$name.log 2>&1"
}

wait_ready() {
  local port=$1
  for _ in $(seq 1 180); do
    if curl -s -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
      echo "worker on $port ready"; return 0
    fi
    sleep 5
  done
  echo "worker on $port NOT ready" >&2; return 1
}

cleanup() {
  tmux kill-session -t w8_1_0 2>/dev/null || true
  tmux kill-session -t w8_1_1 2>/dev/null || true
}
trap cleanup EXIT

start_worker 2 18000 15557 w8_1_0
start_worker 3 18001 15558 w8_1_1
wait_ready 18000
wait_ready 18001

COMMON="--workers w0=http://127.0.0.1:18000,w1=http://127.0.0.1:18001 \
--events w0=tcp://127.0.0.1:15557,w1=tcp://127.0.0.1:15558 \
--out-dir $RUN_DIR --rate 8 --n 150 --seed 0 --model Qwen/Qwen3-1.7B"

if [ "$MODE" = "bench" ] || [ "$MODE" = "both" ]; then
  echo "=== bench ==="
  # shellcheck disable=SC2086
  $PY /root/learn/work/labs/L8/routing_bench.py bench $COMMON > "$RUN_DIR/bench.log" 2>&1
  tail -5 "$RUN_DIR/bench.log"
fi

if [ "$MODE" = "events" ] || [ "$MODE" = "both" ]; then
  echo "=== events ==="
  # shellcheck disable=SC2086
  $PY /root/learn/work/labs/L8/routing_bench.py events $COMMON > "$RUN_DIR/events.log" 2>&1
  tail -5 "$RUN_DIR/events.log"
fi

echo "DONE $RUN_DIR"
