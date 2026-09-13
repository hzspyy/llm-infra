#!/usr/bin/env python3
"""
执行 backward 追踪

Usage:
    python3 trace_backward.py > ../../results/local/7.0/backward_trace.txt
"""
import torch

print("=" * 70)
print("执行 backward 追踪")
print("=" * 70)
print()

x = torch.randn(2, 3, requires_grad=True)
W = torch.randn(4, 3, requires_grad=True)
h = x @ W.t()
y = torch.relu(h)
loss = y.sum()

print("构建的计算图：")
print("  loss = sum(relu(x @ W.t()))")
print()

print("执行前的梯度状态：")
print(f"x.grad: {x.grad}")
print(f"W.grad: {W.grad}")
print()

# 执行反向传播
print("执行 loss.backward()...")
loss.backward()
print()

print("执行后的梯度状态：")
print(f"x.grad 形状: {x.grad.shape}")
print(f"x.grad:\n{x.grad}")
print()
print(f"W.grad 形状: {W.grad.shape}")
print(f"W.grad:\n{W.grad}")
print()

print("=" * 70)
print("backward() 做了什么")
print("=" * 70)
print("1. 创建 GraphTask")
print("2. 将 loss.grad_fn 入队（初始梯度为 1）")
print("3. 按拓扑序执行就绪队列中的节点")
print("4. 每个节点执行完后，更新后继的依赖计数")
print("5. 依赖计数变 0 的节点入队")
print("6. 直到所有 AccumulateGrad 执行完")
