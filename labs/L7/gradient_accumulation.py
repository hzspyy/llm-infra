#!/usr/bin/env python3
"""
梯度累积演示

Usage:
    python3 gradient_accumulation.py > ../../results/local/7.0/accumulation.txt
"""
import torch

print("=" * 70)
print("梯度累积演示")
print("=" * 70)
print()

# 场景 1：参数被使用一次
print("场景 1: 参数被使用 1 次")
print("-" * 70)
W1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
x = torch.tensor([[1.0], [1.0]])
y1 = W1 @ x
loss1 = y1.sum()
loss1.backward()
print(f"W1 @ x = {y1.data.T}")
print(f"loss = y1.sum() = {loss1.item()}")
print(f"W1.grad: {W1.grad}")
print()

# 场景 2：参数被使用两次
print("场景 2: 参数被使用 2 次")
print("-" * 70)
W2 = torch.tensor([[1.0, 2.0]], requires_grad=True)
y2 = W2 @ x      # 第一次
y3 = W2 @ x      # 第二次（相同的计算）
loss2 = (y2 + y3).sum()
print(f"y2 = W2 @ x = {y2.data.T}")
print(f"y3 = W2 @ x = {y3.data.T}")
print(f"loss = (y2 + y3).sum() = {loss2.item()}")
loss2.backward()
print(f"W2.grad: {W2.grad}")
print(f"梯度是场景 1 的 2 倍（因为 W2 被使用了 2 次）")
print()

# 场景 3：手动多次 backward（不清空 grad）
print("场景 3: 多次 backward 不清空梯度")
print("-" * 70)
W3 = torch.tensor([[1.0, 2.0]], requires_grad=True)
for i in range(3):
    y = W3 @ x
    loss = y.sum()
    loss.backward()  # 不清空 grad
    print(f"第 {i+1} 次 backward 后 W3.grad: {W3.grad}")
print()

# 场景 4：清零后 backward
print("场景 4: 清零后 backward")
print("-" * 70)
W4 = torch.tensor([[1.0, 2.0]], requires_grad=True)
for i in range(3):
    if W4.grad is not None:
        W4.grad.zero_()  # 清零
    y = W4 @ x
    loss = y.sum()
    loss.backward()
    print(f"第 {i+1} 次 backward 后 W4.grad（已清零）: {W4.grad}")
print()

print("=" * 70)
print("总结")
print("=" * 70)
print("1. PyTorch 默认累积梯度（不自动清零）")
print("2. 训练循环需要手动 optimizer.zero_grad() 或 tensor.grad.zero_()")
print("3. 共享参数的梯度会自动累加")
