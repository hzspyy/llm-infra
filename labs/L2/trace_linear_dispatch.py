#!/usr/bin/env python3
"""
演示 dispatcher hook 看到的分解

Usage:
    python3 trace_linear_dispatch.py > ../../results/local/2.0b/dispatch_trace.txt
"""
import torch
import torch.nn.functional as F


class DispatchTracer(torch.utils._python_dispatch.TorchDispatchMode):
    """追踪所有 dispatcher 调用"""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"  dispatch: {func}")
        return func(*args, **(kwargs or {}))


def trace_linear():
    print("=" * 70)
    print("F.linear 的 dispatcher 分解")
    print("=" * 70)
    print()

    x = torch.randn(2, 3)
    w = torch.randn(4, 3)
    b = torch.randn(4)

    print("场景 1: F.linear(x, w) - 无 bias")
    with DispatchTracer():
        out1 = F.linear(x, w)
    print(f"输出形状: {out1.shape}")
    print()

    print("场景 2: F.linear(x, w, b) - 有 bias")
    with DispatchTracer():
        out2 = F.linear(x, w, b)
    print(f"输出形状: {out2.shape}")
    print()


def trace_other_ops():
    print("=" * 70)
    print("对照：其他算子的分解")
    print("=" * 70)
    print()

    x = torch.randn(2, 3)

    print("torch.relu(x):")
    with DispatchTracer():
        _ = torch.relu(x)
    print()

    print("x + 1:")
    with DispatchTracer():
        _ = x + 1
    print()


if __name__ == "__main__":
    trace_linear()
    trace_other_ops()

    print("=" * 70)
    print("观察到什么")
    print("=" * 70)
    print("1. F.linear 在 dispatcher 层被分解成 t + addmm")
    print("2. relu 直接调用 aten.relu，没有分解")
    print("3. x + 1 调用 aten.add，标量被提升为张量")
    print()
    print("这就是「观察层次」：不同算子在 dispatcher 层的粒度不同")
