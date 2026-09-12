#!/usr/bin/env python3
"""
演示 F.linear 在五层观察窗口中的样子

Usage:
    python3 trace_linear_layers.py > ../../results/local/2.0b/layers_trace.txt
"""
import torch
import torch.nn.functional as F


def layer1_python_entry():
    """Layer 1: Python 入口"""
    print("=" * 70)
    print("Layer 1: Python 入口")
    print("=" * 70)

    print(f"F.linear 是什么: {F.linear}")
    print(f"类型: {type(F.linear)}")
    print(f"模块: {F.linear.__module__}")
    print()

    doc = F.linear.__doc__
    if doc:
        print(f"文档前 200 字符:")
        print(doc[:200])
        print("...")
    print()


def layer2_python_binding():
    """Layer 2: Python binding（需要源码才能看到实际文件）"""
    print("=" * 70)
    print("Layer 2: Python binding")
    print("=" * 70)

    print("这一层在源码中的位置:")
    print("  torch/csrc/autograd/generated/python_functions.cpp")
    print("  由 tools/codegen/gen.py 从 native_functions.yaml 生成")
    print()
    print("功能: 参数解析、类型检查、转发到 C++ at::linear")
    print()


def layer3_dispatcher():
    """Layer 3: Dispatcher 层"""
    print("=" * 70)
    print("Layer 3: Dispatcher 层")
    print("=" * 70)

    class DispatchTracer(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            print(f"  {func}")
            return func(*args, **(kwargs or {}))

    x = torch.randn(2, 3)
    w = torch.randn(4, 3)

    print("调用: F.linear(x, w)")
    with DispatchTracer():
        out = F.linear(x, w)

    print(f"\n输出形状: {out.shape}")
    print("\n观察到: linear 被分解成 t (transpose) + addmm")
    print()


def layer4_native_impl():
    """Layer 4: Native 实现（需要源码）"""
    print("=" * 70)
    print("Layer 4: Native 实现")
    print("=" * 70)

    print("addmm 的 CPU 实现位置:")
    print("  文件: aten/src/ATen/native/LinearAlgebra.cpp")
    print("  函数: TORCH_IMPL_FUNC(addmm_out_cpu)")
    print("  功能: 调用 CPU BLAS 的 gemm")
    print()

    print("addmm 的 CUDA 实现位置:")
    print("  文件: aten/src/ATen/native/cuda/Blas.cpp")
    print("  函数: addmm_out_cuda_impl")
    print("  功能: 调用 cuBLAS 或 cutlass")
    print()


def layer5_kernel():
    """Layer 5: Kernel（需要 profiler）"""
    print("=" * 70)
    print("Layer 5: Kernel")
    print("=" * 70)

    print("观察工具:")
    print("  CPU: perf, gdb")
    print("  CUDA: nsys, ncu, cuobjdump")
    print()

    print("会看到:")
    print("  CPU: sgemm_ 或 dgemm_ (BLAS 函数)")
    print("  CUDA: kernel 名字如 volta_sgemm_128x128_nn")
    print()


if __name__ == "__main__":
    layer1_python_entry()
    layer2_python_binding()
    layer3_dispatcher()
    layer4_native_impl()
    layer5_kernel()

    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("每一层看到的名字和粒度都不同：")
    print("  L1: F.linear (用户 API)")
    print("  L2: THPVariable_linear (Python binding)")
    print("  L3: aten.t + aten.addmm (dispatcher 分解)")
    print("  L4: addmm_out_cpu/cuda (native 实现)")
    print("  L5: sgemm_ / volta_sgemm_* (实际 kernel)")
    print()
    print("关键：你在第 N 层看到的证据，只能证明第 N 层发生了什么。")
