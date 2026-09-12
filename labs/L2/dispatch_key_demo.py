#!/usr/bin/env python3
"""
演示 dispatch key 如何影响路径选择

Usage:
    python3 dispatch_key_demo.py > ../../results/local/2.0b/dispatch_keys.txt
"""
import torch
import torch.nn.functional as F


class VerboseDispatch(torch.utils._python_dispatch.TorchDispatchMode):
    """显示 dispatch key 的追踪器"""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        keys = [t.__name__ for t in types]
        func_name = str(func).split(".")[-1].replace(">", "")
        print(f"  {func_name:20s} keys: {keys}")
        return func(*args, **(kwargs or {}))


def scenario1_cpu_no_grad():
    """场景 1: CPU, no grad"""
    print("=" * 70)
    print("场景 1: CPU, requires_grad=False")
    print("=" * 70)

    x = torch.randn(2, 3, requires_grad=False)
    w = torch.randn(4, 3, requires_grad=False)

    print("调用: F.linear(x, w)")
    with VerboseDispatch():
        out = F.linear(x, w)

    print(f"\n输出 requires_grad: {out.requires_grad}")
    print()


def scenario2_cpu_with_grad():
    """场景 2: CPU, with grad"""
    print("=" * 70)
    print("场景 2: CPU, requires_grad=True")
    print("=" * 70)

    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)

    print("调用: F.linear(x, w)")
    with VerboseDispatch():
        out = F.linear(x, w)

    print(f"\n输出 requires_grad: {out.requires_grad}")
    print()


def scenario3_cuda():
    """场景 3: CUDA (如果可用)"""
    if not torch.cuda.is_available():
        print("=" * 70)
        print("场景 3: CUDA - 跳过（本机无 CUDA）")
        print("=" * 70)
        print("在 crater 上会看到:")
        print("  aten.t            keys: ['Tensor', 'AutogradCUDA']")
        print("  aten.mm           keys: ['Tensor', 'AutogradCUDA']")
        print()
        return

    print("=" * 70)
    print("场景 3: CUDA, requires_grad=True")
    print("=" * 70)

    x = torch.randn(2, 3, device='cuda', requires_grad=True)
    w = torch.randn(4, 3, device='cuda', requires_grad=True)

    print("调用: F.linear(x, w)")
    with VerboseDispatch():
        out = F.linear(x, w)

    print(f"\n输出 requires_grad: {out.requires_grad}")
    print()


if __name__ == "__main__":
    scenario1_cpu_no_grad()
    scenario2_cpu_with_grad()
    scenario3_cuda()

    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("requires_grad 改变了 dispatch key:")
    print("  False: 只有基础 key (CPU/CUDA)")
    print("  True:  额外有 AutogradCPU/AutogradCUDA")
    print()
    print("但分解后的算子（t + addmm）是相同的")
