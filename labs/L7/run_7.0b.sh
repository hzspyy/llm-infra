#!/bin/bash
# 7.0b 训练步实验运行指南

set -e

echo "==================================================="
echo "7.0b 训练步实验 - SmolLM3"
echo "==================================================="

# 检查环境
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "警告: CUDA_VISIBLE_DEVICES 未设置"
fi

# 创建输出目录
OUTPUT_DIR="./work/out/7.0b"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Step 1: 核心机制验证（CPU，快速）"
echo "---------------------------------------------------"
python labs/L7/verify_training_mechanics.py | tee "$OUTPUT_DIR/verify_mechanics.txt"

echo ""
echo "Step 2: SmolLM 完整训练步（GPU）"
echo "---------------------------------------------------"
echo "  使用 SmolLM3-360M 验证（~5分钟）"
python labs/L7/training_step_smollm.py 2>&1 | tee "$OUTPUT_DIR/training_log.txt"

echo ""
echo "Step 3: Checkpoint 恢复测试"
echo "---------------------------------------------------"
python labs/L7/training_step_smollm.py test 2>&1 | tee "$OUTPUT_DIR/checkpoint_recovery.txt"

echo ""
echo "==================================================="
echo "实验完成"
echo "==================================================="
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "检查项："
echo "  1. verify_mechanics.txt - 4个核心测试全部通过"
echo "  2. training_log.txt - Loss 下降曲线"
echo "  3. checkpoints_7.0b/ - Checkpoint 文件"
echo "  4. checkpoint_recovery.txt - 恢复测试通过"
echo ""
echo "拉取结果："
echo "  rsync -av --ignore-existing crater:/scratch/learn/work/out/7.0b/ results/crater/7.0b/"
