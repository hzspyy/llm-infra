# L7 训练实验说明

## 实验概览

本目录包含 7.0b 章节的实验代码：

1. **verify_concepts.py** - 纯 Python 概念验证（本地运行）
2. **verify_training_mechanics.py** - PyTorch 核心机制验证（需要 crater）
3. **training_step_smollm.py** - SmolLM3 完整训练步（需要 crater + GPU）

## 本地运行（macOS）

```bash
cd labs/L7
python3 verify_concepts.py
```

验证通过：
- ✓ Labels 右移
- ✓ Loss mask
- ✓ 有效 token 归一化
- ✓ 梯度累积概念

## Crater 运行指南

### 1. 同步代码到 crater

```bash
# 在项目根目录
rsync -av labs/L7/ crater:/scratch/learn/work/labs/L7/
```

### 2. SSH 到 crater 并准备环境

```bash
ssh crater
cd /scratch/learn
source env.sh

# 检查显存
nvidia-smi

# 检查 Python 环境
which python
python --version

# 检查依赖
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 3. 运行实验

#### 实验 A: PyTorch 核心机制验证（~1 分钟，CPU 即可）

```bash
cd /scratch/learn/work/labs/L7
python verify_training_mechanics.py > /scratch/learn/work/out/7.0b/verify_mechanics.txt 2>&1
```

预期输出：
- Labels 移位验证
- Loss mask 验证
- 归一化验证
- 梯度累积验证
- Checkpoint 恢复验证

#### 实验 B: SmolLM3 完整训练步（~5 分钟，需要 GPU）

```bash
cd /scratch/learn/work/labs/L7

# 创建输出目录
mkdir -p /scratch/learn/work/out/7.0b

# 运行实验
python training_step_smollm.py > /scratch/learn/work/out/7.0b/training_step.txt 2>&1
```

预期输出：
- 样本内容与 tokenization
- Labels 移位与 loss mask
- 逐 token loss 统计
- 梯度范数与参数更新
- AdamW 优化器状态（m, v）
- Checkpoint 保存与恢复验证
- 梯度累积 vs 直接 batch 对比

### 4. 拉回结果

```bash
# 在本地项目根目录
mkdir -p results/crater/7.0b
rsync -av --ignore-existing crater:/scratch/learn/work/out/7.0b/ results/crater/7.0b/
```

## 预期结果

### verify_mechanics.txt

```
验证 1: Labels 移位
  ✓ Labels 移位验证通过

验证 2: Loss Mask
  有效 token 数: 8
  误差: < 1e-6
  ✓ Loss mask 验证通过

验证 3: 有效 Token 归一化
  场景 1 vs 场景 2 误差: < 1e-5
  ✓ 归一化验证通过

验证 4: 梯度累积等价性
  梯度最大差异: < 1e-6
  参数最大差异: < 1e-6
  ✓ 梯度累积验证通过

验证 5: Checkpoint 保存与恢复
  Loss 误差: < 1e-6
  参数最大差异: < 1e-6
  ✓ Checkpoint 恢复验证通过
```

### training_step.txt

```
实验 1: 完整训练步分析
  模型: SmolLM2-360M
  参数量: 360M
  
  样本内容（前 50 token）
  Loss 统计: 总 loss ~8-10（随机初始化）
  有效 token 数: ~200-250
  
  梯度统计:
    梯度范数均值: ~1e2
    最大梯度范数: ~1e3
  
  参数更新统计:
    最大参数变化: ~1e-4（lr=1e-4）
  
  AdamW 优化器状态:
    step: 1
    exp_avg (m): 形状与参数相同
    exp_avg_sq (v): 形状与参数相同
  
  Checkpoint 已保存: ~1.5 GB（360M FP32）
  
  恢复验证:
    原始 loss vs 恢复后 loss 差异: < 1e-5
    ✓ 恢复验证通过

实验 2: 梯度累积 vs 直接大 batch
  梯度累积总 loss vs 直接 batch loss 差异: < 0.01
  参数最大差异: < 1e-5
  ✓ 梯度累积与直接 batch 等价
```

## 故障排查

### PyTorch 未安装

```bash
# 检查环境
source /scratch/learn/env.sh

# 如果需要安装（通常已有）
pip install torch transformers
```

### 显存不足

修改 `training_step_smollm.py`：

```python
# 使用更小的模型
model_name = "HuggingFaceTB/SmolLM2-135M"  # 或 360M

# 或使用 FP16
torch_dtype=torch.float16
```

### 模型下载失败

```bash
# 设置 HuggingFace 镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com
```

## 下一步

完成实验后：
1. 检查 `results/crater/7.0b/` 的输出
2. 更新 STATUS.md 标记 7.0b 为"有输出"
3. 在正文中引用原始材料
4. 准备 7.1 训练循环系统实验
