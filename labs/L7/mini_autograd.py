#!/usr/bin/env python3
"""
最小的 autograd 引擎

Usage:
    python3 mini_autograd.py > ../../results/local/7.0/mini_autograd_output.txt
"""
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Tensor:
    """最小的 Tensor 类"""
    data: np.ndarray
    grad: Optional[np.ndarray] = None
    requires_grad: bool = False
    grad_fn: Optional['GradFn'] = None

    def backward(self):
        """执行反向传播"""
        if self.grad_fn is None:
            raise RuntimeError("No grad_fn (leaf tensor or no_grad context)")

        # 初始化梯度为 1（标量 loss）
        self.grad = np.ones_like(self.data)

        # 拓扑排序
        topo = []
        visited = set()

        def build_topo(fn):
            if fn is None or fn in visited:
                return
            visited.add(fn)
            for inp in fn.inputs:
                if inp.grad_fn is not None:
                    build_topo(inp.grad_fn)
            topo.append(fn)

        build_topo(self.grad_fn)

        # 反向执行
        for fn in reversed(topo):
            fn.backward()


class GradFn:
    """反向函数基类"""
    def __init__(self, *inputs):
        self.inputs = inputs
        self.output = None

    def backward(self):
        """计算输入的梯度"""
        raise NotImplementedError


class AddBackward(GradFn):
    """加法的反向"""
    def backward(self):
        # ∂loss/∂a = ∂loss/∂out
        # ∂loss/∂b = ∂loss/∂out
        a, b = self.inputs
        if a.requires_grad:
            if a.grad is None:
                a.grad = self.output.grad.copy()
            else:
                a.grad += self.output.grad
        if b.requires_grad:
            if b.grad is None:
                b.grad = self.output.grad.copy()
            else:
                b.grad += self.output.grad


class MulBackward(GradFn):
    """乘法的反向"""
    def backward(self):
        # ∂loss/∂a = ∂loss/∂out * b
        # ∂loss/∂b = ∂loss/∂out * a
        a, b = self.inputs
        if a.requires_grad:
            grad = self.output.grad * b.data
            if a.grad is None:
                a.grad = grad
            else:
                a.grad += grad
        if b.requires_grad:
            grad = self.output.grad * a.data
            if b.grad is None:
                b.grad = grad
            else:
                b.grad += grad


class MatmulBackward(GradFn):
    """矩阵乘的反向"""
    def backward(self):
        # ∂loss/∂a = ∂loss/∂out @ b.T
        # ∂loss/∂b = a.T @ ∂loss/∂out
        a, b = self.inputs
        if a.requires_grad:
            grad = self.output.grad @ b.data.T
            if a.grad is None:
                a.grad = grad
            else:
                a.grad += grad
        if b.requires_grad:
            grad = a.data.T @ self.output.grad
            if b.grad is None:
                b.grad = grad
            else:
                b.grad += grad


# 算子实现
def add(a: Tensor, b: Tensor) -> Tensor:
    """加法"""
    result = Tensor(
        data=a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )
    if result.requires_grad:
        result.grad_fn = AddBackward(a, b)
        result.grad_fn.output = result
    return result


def mul(a: Tensor, b: Tensor) -> Tensor:
    """乘法"""
    result = Tensor(
        data=a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )
    if result.requires_grad:
        result.grad_fn = MulBackward(a, b)
        result.grad_fn.output = result
    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """矩阵乘"""
    result = Tensor(
        data=a.data @ b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )
    if result.requires_grad:
        result.grad_fn = MatmulBackward(a, b)
        result.grad_fn.output = result
    return result


# 测试
if __name__ == "__main__":
    print("=" * 70)
    print("Mini Autograd 测试")
    print("=" * 70)
    print()

    # 场景 1: 简单加法
    print("场景 1: y = a + b")
    print("-" * 70)
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    y = add(a, b)
    y.backward()
    print(f"a.data: {a.data}")
    print(f"b.data: {b.data}")
    print(f"y.data: {y.data}")
    print(f"a.grad: {a.grad}  (应该是 [1, 1])")
    print(f"b.grad: {b.grad}  (应该是 [1, 1])")
    print()

    # 场景 2: 乘法
    print("场景 2: y = a * b")
    print("-" * 70)
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    y = mul(a, b)
    y.backward()
    print(f"a.data: {a.data}")
    print(f"b.data: {b.data}")
    print(f"y.data: {y.data}")
    print(f"a.grad: {a.grad}  (应该是 b.data = [4, 5])")
    print(f"b.grad: {b.grad}  (应该是 a.data = [2, 3])")
    print()

    # 场景 3: 矩阵乘
    print("场景 3: y = A @ B")
    print("-" * 70)
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    B = Tensor(np.array([[5.0], [6.0]]), requires_grad=True)
    y = matmul(A, B)
    y.backward()
    print(f"A.data:\n{A.data}")
    print(f"B.data:\n{B.data}")
    print(f"y.data:\n{y.data}")
    print(f"A.grad (应该是 [[5, 6], [5, 6]]):\n{A.grad}")
    print(f"B.grad (应该是 [[4], [6]]):\n{B.grad}")
    print()

    # 场景 4: 梯度累积（共享参数）
    print("场景 4: 梯度累积（共享参数）")
    print("-" * 70)
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c1 = Tensor(np.array([2.0, 2.0]))
    c2 = Tensor(np.array([3.0, 3.0]))
    y1 = mul(x, c1)
    y2 = mul(x, c2)
    y = add(y1, y2)
    y.backward()
    print(f"x.data: {x.data}")
    print(f"y1 = x * [2, 2] = {y1.data}")
    print(f"y2 = x * [3, 3] = {y2.data}")
    print(f"y = y1 + y2 = {y.data}")
    print(f"x.grad: {x.grad}  (应该是 [2+3, 2+3] = [5, 5])")
    print("梯度累积：x 被使用了 2 次，梯度相加")
    print()

    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("这个 mini autograd 演示了:")
    print("  1. 前向构建反向图（创建 grad_fn）")
    print("  2. 拓扑排序保证执行顺序")
    print("  3. 梯度累积（共享参数）")
    print("  4. 核心只需要约 150 行 Python")
