#!/bin/bash
# 在 crater 上运行 7.1 训练循环系统实验

set -e

# 加载环境
source /scratch/learn/envs/serve/bin/activate

# 记录环境信息
echo "=== 环境信息 ==="
echo "主机: $(hostname)"
echo "时间: $(date)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU 状态（运行前）==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
fi

# 创建输出目录
mkdir -p results/crater/7.1

# 运行实验
echo ""
echo "=== 开始实验 ==="
python labs/L7/training_loop_systems.py 2>&1 | tee results/crater/7.1/training_loop_systems.txt

EXIT_CODE=${PIPESTATUS[0]}

# 记录完成状态
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU 状态（运行后）==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
fi

echo ""
echo "=== 实验完成 ==="
echo "退出码: $EXIT_CODE"
echo "时间: $(date)"
echo "输出目录: results/crater/7.1/"

exit $EXIT_CODE
