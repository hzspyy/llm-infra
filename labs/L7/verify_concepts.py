#!/usr/bin/env python3
"""
训练机制概念验证（纯 Python，不依赖 PyTorch）

验证：
1. Labels 右移的概念
2. Loss mask 的作用
3. 有效 token 归一化的必要性

运行：python3 verify_concepts.py
"""

import math


def softmax(logits):
    """简单的 softmax"""
    exp_vals = [math.exp(x) for x in logits]
    total = sum(exp_vals)
    return [x / total for x in exp_vals]


def cross_entropy_loss(probs, target_idx):
    """单个 token 的交叉熵 loss"""
    return -math.log(probs[target_idx] + 1e-9)


def verify_labels_shift():
    print("\n" + "="*60)
    print("验证 1: Labels 右移")
    print("="*60)

    # 简单序列：token_ids = [1, 2, 3, 4, 5]
    # 目标：给定前 t 个 token，预测第 t+1 个

    input_ids = [1, 2, 3, 4, 5]

    # 错误：不移位（预测自己）
    labels_wrong = [1, 2, 3, 4, 5]

    # 正确：右移（预测下一个）
    labels_correct = [2, 3, 4, 5, None]  # 最后一个位置没有下一个 token

    print(f"input_ids:        {input_ids}")
    print(f"labels (错误):    {labels_wrong}  # 位置 0 预测 1（自己）")
    print(f"labels (正确):    {[x if x else 'N/A' for x in labels_correct]}  # 位置 0 预测 2（下一个）")

    print("\n推理时的对应关系：")
    print("  位置 0: 看到 token_1, 预测 token_2")
    print("  位置 1: 看到 token_1+2, 预测 token_3")
    print("  位置 2: 看到 token_1+2+3, 预测 token_4")
    print("  ...")

    print("\n✓ 右移保证训练目标与推理一致")


def verify_loss_mask():
    print("\n" + "="*60)
    print("验证 2: Loss Mask（padding 不参与计算）")
    print("="*60)

    # 两条样本，长度不同
    # 样本 1: [1, 2, 3] + padding [0, 0]
    # 样本 2: [4, 5] + padding [0, 0, 0]

    batch = [
        {"tokens": [1, 2, 3, 0, 0], "valid_length": 3},
        {"tokens": [4, 5, 0, 0, 0], "valid_length": 2},
    ]

    # 模拟 loss（这里用随机值）
    per_token_losses = [
        [0.8, 1.2, 0.9, 999.0, 999.0],  # 样本 1: 前 3 个有效
        [1.1, 0.7, 999.0, 999.0, 999.0] # 样本 2: 前 2 个有效
    ]

    # 错误：直接平均所有位置
    total_positions = sum(len(sample["tokens"]) for sample in batch)
    wrong_loss = sum(sum(losses) for losses in per_token_losses) / total_positions

    # 正确：只平均有效 token
    valid_losses = []
    for i, sample in enumerate(batch):
        valid_losses.extend(per_token_losses[i][:sample["valid_length"]])

    correct_loss = sum(valid_losses) / len(valid_losses)

    print(f"样本 1: tokens={batch[0]['tokens']}, 有效长度={batch[0]['valid_length']}")
    print(f"样本 2: tokens={batch[1]['tokens']}, 有效长度={batch[1]['valid_length']}")

    print(f"\nPer-token losses (999.0 = padding):")
    for i, losses in enumerate(per_token_losses):
        print(f"  样本 {i+1}: {losses}")

    print(f"\n错误方法（包含 padding）: {wrong_loss:.2f}")
    print(f"正确方法（只计有效）:     {correct_loss:.2f}")
    print(f"差异:                      {abs(wrong_loss - correct_loss):.2f}")

    print("\n✓ Loss mask 避免 padding 污染梯度")


def verify_normalization():
    print("\n" + "="*60)
    print("验证 3: 有效 Token 归一化")
    print("="*60)

    # 场景 1: batch_size=2, 每条 5 token（共 10）
    scenario_1_losses = [
        [1.0, 1.2, 0.8, 1.1, 0.9],  # 样本 1
        [1.3, 0.7, 1.0, 0.9, 1.1],  # 样本 2
    ]
    valid_1 = [loss for sample in scenario_1_losses for loss in sample]
    loss_1 = sum(valid_1) / len(valid_1)

    # 场景 2: batch_size=1, 共 10 token（拼接场景 1）
    scenario_2_losses = scenario_1_losses[0] + scenario_1_losses[1]
    loss_2 = sum(scenario_2_losses) / len(scenario_2_losses)

    print("场景 1: batch=2, 每条 5 token")
    print(f"  样本 1 losses: {scenario_1_losses[0]}")
    print(f"  样本 2 losses: {scenario_1_losses[1]}")
    print(f"  平均 loss: {loss_1:.4f}")

    print("\n场景 2: batch=1, 共 10 token（拼接）")
    print(f"  所有 losses: {scenario_2_losses}")
    print(f"  平均 loss: {loss_2:.4f}")

    print(f"\n两种配置的 loss 差异: {abs(loss_1 - loss_2):.9f}")

    print("\n✓ 按有效 token 归一化保证不同 batch 配置可比")


def verify_gradient_accumulation_concept():
    print("\n" + "="*60)
    print("验证 4: 梯度累积概念")
    print("="*60)

    # 简化模型：单个参数 w
    # 梯度：dL/dw

    # 4 个 micro_batch 的梯度
    micro_gradients = [0.5, 0.3, 0.7, 0.4]

    # 路径 1: 梯度累积
    # 每个 micro_batch 的 loss 除以 4，梯度也除以 4
    accumulated_grad = sum(g / 4 for g in micro_gradients)

    # 路径 2: 直接大 batch
    # 相当于 4 个样本的梯度平均
    direct_grad = sum(micro_gradients) / 4

    print("4 个 micro_batch 的梯度: ", micro_gradients)
    print(f"\n路径 1 (累积): 每个梯度 ÷4, 累加")
    print(f"  = ({micro_gradients[0]}/4) + ({micro_gradients[1]}/4) + ... = {accumulated_grad:.4f}")

    print(f"\n路径 2 (直接): 梯度之和 ÷4")
    print(f"  = ({micro_gradients[0]} + {micro_gradients[1]} + ...) / 4 = {direct_grad:.4f}")

    print(f"\n差异: {abs(accumulated_grad - direct_grad):.9f}")

    print("\n✓ 梯度累积在正确归一化下与直接大 batch 等价")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("训练机制概念验证（纯 Python）")
    print("="*60)

    verify_labels_shift()
    verify_loss_mask()
    verify_normalization()
    verify_gradient_accumulation_concept()

    print("\n" + "="*60)
    print("✓ 所有概念验证通过")
    print("="*60)
    print("\n下一步：在 crater 上运行完整的 PyTorch 实验")
    print("  - labs/L7/verify_training_mechanics.py")
    print("  - labs/L7/training_step_smollm.py")
