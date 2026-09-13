#!/usr/bin/env python3
"""
追踪 grad_fn 的创建

Usage:
    python3 trace_grad_fn.py > ../../results/local/7.0/grad_fn_trace.txt
"""
import torch

print("=" * 70)
print("追踪 grad_fn 的创建")
print("=" * 70)
print()

x = torch.randn(2, 3, requires_grad=True)
print(f"x = torch.randn(2, 3, requires_grad=True)")
print(f"x.grad_fn: {x.grad_fn}")  # None（叶子节点）
print(f"x.requires_grad: {x.requires_grad}")
print(f"x.is_leaf: {x.is_leaf}")
print()

# 第一个算子：矩阵乘
W = torch.randn(4, 3, requires_grad=True)
h = x @ W.t()
print(f"W = torch.randn(4, 3, requires_grad=True)")
print(f"h = x @ W.t()")
print(f"h.grad_fn: {h.grad_fn}")
print(f"h.grad_fn.next_functions: {h.grad_fn.next_functions}")
print(f"h.is_leaf: {h.is_leaf}")
print()

# 第二个算子：ReLU
y = torch.relu(h)
print(f"y = torch.relu(h)")
print(f"y.grad_fn: {y.grad_fn}")
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")
print()

# 第三个算子：求和
loss = y.sum()
print(f"loss = y.sum()")
print(f"loss.grad_fn: {loss.grad_fn}")
print(f"loss.grad_fn.next_functions: {loss.grad_fn.next_functions}")
print()

print("=" * 70)
print("观察到什么")
print("=" * 70)
print("1. 叶子节点（x, W）没有 grad_fn")
print("2. 每个算子创建一个 grad_fn（MmBackward、ReluBackward、SumBackward）")
print("3. next_functions 指向前一个节点（反向图的边）")
print("4. AccumulateGrad 是终点（把梯度写入 tensor.grad）")
