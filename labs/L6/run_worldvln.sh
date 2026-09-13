#!/bin/bash
# 在 worldvln 上运行 6.0 实验

set -e

cd /root/learn/work/labs/L6

# 加载环境
source /root/learn/env.sh

# 使用 serve 环境的 Python
PYTHON=/root/learn/envs/serve/bin/python3

# 2 卡实验
echo "=== Running 2-GPU NCCL experiments ==="
$PYTHON distributed_basics.py gpu 2

# 4 卡实验
echo "=== Running 4-GPU NCCL experiments ==="
$PYTHON distributed_basics.py gpu 4

echo "=== Experiments completed ==="
ls -lh /root/learn/work/out/6.0/
