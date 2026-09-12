#!/usr/bin/env python3
"""
演示如何用不同方法追踪算子分发

Usage:
    python3 trace_dispatch.py > ../../results/local/M1/dispatch_trace.txt
"""
import torch
import torch.nn.functional as F


def method_a_python_binding():
    """方法 A：最直接 —— 看 Python 绑定"""
    print("=== Method A: Python binding ===")
    print(f"F.linear 实际是: {F.linear}")
    print(f"类型: {type(F.linear)}")
    doc = F.linear.__doc__
    if doc:
        print(f"文档字符串前 200 字符:\n{doc[:200]}...")
    print()


def method_b_dispatch_hook():
    """方法 B：Hook dispatcher"""

    class DispatchTracer(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            print(f"  dispatch: {func}")
            return func(*args, **(kwargs or {}))

    print("=== Method B: Dispatch trace ===")
    x = torch.randn(2, 3)
    w = torch.randn(4, 3)
    print("调用: F.linear(x, w)")
    with DispatchTracer():
        out = F.linear(x, w)
    print(f"输出形状: {out.shape}")
    print()


def method_c_registration_lookup():
    """方法 C：看生成的注册表（需要源码）"""
    print("=== Method C: Registration lookup ===")
    print("在 PyTorch 源码中查找：")
    print("  1. aten/src/ATen/native/native_functions.yaml")
    print("  2. 搜索 'func: linear'")
    print("  3. 找到 dispatch 条目")
    print()
    print("预期看到：")
    print("  - func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")
    print("  - dispatch: 会列出 CPU/CUDA 等实现")
    print()


if __name__ == "__main__":
    method_a_python_binding()
    method_b_dispatch_hook()
    method_c_registration_lookup()

    print("=== Summary ===")
    print("观察到：")
    print("  - Python 层看到 F.linear")
    print("  - Dispatcher 层看到它分解成 t + addmm")
    print("  - 真实 kernel 在 addmm 里")
    print()
    print("这就是「观察层次」：你在不同层看到不同的名字和粒度。")
