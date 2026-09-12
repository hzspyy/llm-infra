#!/usr/bin/env python3
"""
最小的 dispatcher 实现，理解分发逻辑

Usage:
    python3 mini_dispatcher.py > ../../results/local/2.0b/mini_dispatcher_output.txt
"""
from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class DispatchKey:
    """简化的 dispatch key"""
    name: str
    priority: int  # 数字越大优先级越高


# 预定义的 keys
CPU = DispatchKey("CPU", 1)
CUDA = DispatchKey("CUDA", 1)
AutogradCPU = DispatchKey("AutogradCPU", 10)
AutogradCUDA = DispatchKey("AutogradCUDA", 10)


class MiniDispatcher:
    """最小的算子 dispatcher"""

    def __init__(self, op_name: str):
        self.op_name = op_name
        self.impls: Dict[str, Callable] = {}

    def register(self, key: DispatchKey, impl: Callable):
        """注册一个实现"""
        self.impls[key.name] = impl
        print(f"  [注册] {self.op_name}.{key.name} → {impl.__name__}")

    def call(self, *args, dispatch_keys: list, **kwargs):
        """根据 dispatch keys 选择实现"""
        # 按优先级排序
        sorted_keys = sorted(dispatch_keys, key=lambda k: k.priority, reverse=True)

        print(f"\n[调用] {self.op_name}")
        print(f"  dispatch keys: {[k.name for k in sorted_keys]}")

        # 选择第一个有实现的 key
        for key in sorted_keys:
            if key.name in self.impls:
                impl = self.impls[key.name]
                print(f"  → 选中 {key.name} 实现: {impl.__name__}")
                return impl(*args, **kwargs)

        raise RuntimeError(f"No implementation for {self.op_name}")


# 实现 addmm 算子
def addmm_cpu(mat1, mat2, bias):
    """CPU 实现（简化）"""
    print("    [执行] CPU GEMM (调用 BLAS)")
    return "result_cpu"


def addmm_cuda(mat1, mat2, bias):
    """CUDA 实现（简化）"""
    print("    [执行] CUDA GEMM (调用 cuBLAS)")
    return "result_cuda"


def addmm_autograd_cpu(mat1, mat2, bias):
    """Autograd 包装 (CPU)"""
    print("    [Autograd] 记录前向信息")

    # Redispatch 到下一层
    print("    [Redispatch] 到 CPU")
    result = addmm_dispatcher.call(mat1, mat2, bias, dispatch_keys=[CPU])

    print("    [Autograd] 注册反向函数")
    return result


def addmm_autograd_cuda(mat1, mat2, bias):
    """Autograd 包装 (CUDA)"""
    print("    [Autograd] 记录前向信息")

    # Redispatch 到下一层
    print("    [Redispatch] 到 CUDA")
    result = addmm_dispatcher.call(mat1, mat2, bias, dispatch_keys=[CUDA])

    print("    [Autograd] 注册反向函数")
    return result


# 创建 dispatcher 并注册
print("=" * 70)
print("初始化 addmm dispatcher")
print("=" * 70)

addmm_dispatcher = MiniDispatcher("addmm")
addmm_dispatcher.register(CPU, addmm_cpu)
addmm_dispatcher.register(CUDA, addmm_cuda)
addmm_dispatcher.register(AutogradCPU, addmm_autograd_cpu)
addmm_dispatcher.register(AutogradCUDA, addmm_autograd_cuda)


# 测试场景
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("场景 1: CPU, no grad")
    print("=" * 70)
    addmm_dispatcher.call("mat1", "mat2", "bias", dispatch_keys=[CPU])

    print("\n" + "=" * 70)
    print("场景 2: CPU, with grad")
    print("=" * 70)
    addmm_dispatcher.call("mat1", "mat2", "bias", dispatch_keys=[AutogradCPU, CPU])

    print("\n" + "=" * 70)
    print("场景 3: CUDA, with grad")
    print("=" * 70)
    addmm_dispatcher.call("mat1", "mat2", "bias", dispatch_keys=[AutogradCUDA, CUDA])

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("这个 mini dispatcher 演示了:")
    print("  1. 如何根据 dispatch key 选择实现")
    print("  2. Autograd 包装如何 redispatch 到下一层")
    print("  3. 为什么同一个算子有多个实现")
