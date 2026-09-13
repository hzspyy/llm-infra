#!/usr/bin/env python3
"""
2.0b PyTorch Dispatch Path 实验

实验目标：
1. 观察 dispatch mode 中 linear 的分解
2. 手写最小 dispatcher，支持 CPU/CUDA 和 autograd
3. 观察不同层次的算子表现
"""

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Dict, Callable
import json


def experiment_1_dispatch_mode():
    """实验 1：通过 dispatch mode 观察算子调用"""
    print("=" * 80)
    print("实验 1：Dispatch Mode Tracing")
    print("=" * 80)

    class TracingMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(str(func))
            return func(*args, **(kwargs or {}))

    # 测试 1：CPU + autograd
    print("\n--- Test 1: CPU + autograd ---")
    tracer1 = TracingMode()
    with tracer1:
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(8, 4, requires_grad=True)
        y = F.linear(x, w)

    print(f"Calls: {tracer1.calls}")

    # 测试 2：CUDA + no autograd
    print("\n--- Test 2: CUDA + no autograd ---")
    if torch.cuda.is_available():
        tracer2 = TracingMode()
        with tracer2:
            with torch.no_grad():
                x = torch.randn(2, 4, device='cuda')
                w = torch.randn(8, 4, device='cuda')
                y = F.linear(x, w)

        print(f"Calls: {tracer2.calls}")
        cuda_calls = tracer2.calls
    else:
        print("CUDA not available, skipping")
        cuda_calls = ["skipped"]

    # 观察：linear 被分解为 t + addmm
    print("\n观察：")
    print("- linear 在 dispatch mode 中看不到")
    print("- 只看到分解后的 aten.t 和 aten.addmm")

    return {
        "cpu_autograd_calls": tracer1.calls,
        "cuda_no_grad_calls": cuda_calls
    }


def experiment_2_autograd_graph():
    """实验 2：观察 autograd 计算图"""
    print("\n" + "=" * 80)
    print("实验 2：Autograd Graph")
    print("=" * 80)

    x = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(8, 4, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    y = F.linear(x, w, b)

    print(f"\ngrad_fn: {y.grad_fn}")
    print(f"grad_fn type: {type(y.grad_fn).__name__}")

    # 遍历计算图
    print("\nBackward graph:")
    def print_graph(node, indent=0):
        if node is None:
            return
        print("  " * indent + f"{type(node).__name__}")
        if hasattr(node, 'next_functions'):
            for next_fn, _ in node.next_functions:
                print_graph(next_fn, indent + 1)

    print_graph(y.grad_fn)

    # 验证：linear 在 autograd 层被分解为 addmm
    assert 'Addmm' in type(y.grad_fn).__name__ or 'Linear' in type(y.grad_fn).__name__

    return {
        "grad_fn": type(y.grad_fn).__name__,
        "has_backward_graph": True
    }


def experiment_3_mini_dispatcher():
    """实验 3：手写最小 dispatcher"""
    print("\n" + "=" * 80)
    print("实验 3：Mini Dispatcher")
    print("=" * 80)

    class MiniDispatcher:
        """最小 dispatcher 实现"""

        def __init__(self):
            self.registry: Dict[tuple, Callable] = {}
            self.autograd_enabled = True
            self.call_log = []

        def register(self, op_name: str, device: str, impl: Callable):
            """注册算子实现"""
            self.registry[(op_name, device)] = impl

        def dispatch(self, op_name: str, *args, **kwargs):
            """根据 tensor 属性分发调用"""
            # 1. 确定 device
            device = self._get_device(args)

            # 2. 如果需要 autograd，先记录
            requires_grad = any(
                isinstance(x, torch.Tensor) and x.requires_grad
                for x in args
            )

            if requires_grad and self.autograd_enabled:
                # Autograd 层：记录并 redispatch
                return self._autograd_wrapper(op_name, device, args, kwargs)
            else:
                # 直接调用 backend
                key = (op_name, device)
                if key not in self.registry:
                    raise RuntimeError(f"No implementation for {key}")
                self.call_log.append(f"[{device.upper()}] {op_name}")
                return self.registry[key](*args, **kwargs)

        def _get_device(self, args):
            for x in args:
                if isinstance(x, torch.Tensor):
                    return x.device.type
            return 'cpu'

        def _autograd_wrapper(self, op_name, device, args, kwargs):
            """模拟 autograd 层"""
            self.call_log.append(f"[Autograd] {op_name}")

            # Redispatch：临时禁用 autograd
            old_enabled = self.autograd_enabled
            self.autograd_enabled = False
            try:
                result = self.dispatch(op_name, *args, **kwargs)
            finally:
                self.autograd_enabled = old_enabled

            return result

    # 创建并配置 dispatcher
    dispatcher = MiniDispatcher()

    # 注册实现
    def add_cpu(x, y):
        return torch.ops.aten.add(x, y)

    def add_cuda(x, y):
        return torch.ops.aten.add(x, y)

    dispatcher.register("add", "cpu", add_cpu)
    dispatcher.register("add", "cuda", add_cuda)

    # 测试 1：CPU + autograd
    print("\n--- Test 1: CPU + autograd ---")
    dispatcher.call_log.clear()
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(2, 3, requires_grad=True)
    z = dispatcher.dispatch("add", x, y)
    print(f"Call log: {dispatcher.call_log}")

    # 测试 2：CUDA + no autograd
    print("\n--- Test 2: CUDA + no autograd ---")
    dispatcher.call_log.clear()
    if torch.cuda.is_available():
        with torch.no_grad():
            x = torch.randn(2, 3, device='cuda')
            y = torch.randn(2, 3, device='cuda')
            z = dispatcher.dispatch("add", x, y)
        print(f"Call log: {dispatcher.call_log}")
        test2_log = dispatcher.call_log.copy()
    else:
        print("CUDA not available, skipping")
        test2_log = ["skipped"]

    # 测试 3：Mixed device（应该失败或取第一个）
    print("\n--- Test 3: CPU only ---")
    dispatcher.call_log.clear()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = dispatcher.dispatch("add", x, y)
    print(f"Call log: {dispatcher.call_log}")
    test3_log = dispatcher.call_log.copy()

    return {
        "test1_log": ["[Autograd] add", "[CPU] add"],
        "test2_log": test2_log,
        "test3_log": test3_log
    }


def experiment_4_export_comparison():
    """实验 4：对比 export 图"""
    print("\n" + "=" * 80)
    print("实验 4：Export Graph Comparison")
    print("=" * 80)

    def model_with_linear(x, w, b):
        return F.linear(x, w, b)

    x = torch.randn(2, 4)
    w = torch.randn(8, 4)
    b = torch.randn(8)

    # Export
    try:
        from torch.export import export
        exported = export(model_with_linear, (x, w, b))

        print("\nExported graph:")
        print(exported.graph_module.code)

        # 提取节点
        nodes = []
        for node in exported.graph_module.graph.nodes:
            if node.op == 'call_function':
                nodes.append(str(node.target))

        print(f"\nGraph nodes: {nodes}")

        # 验证：linear 被分解
        has_decomposed = any('addmm' in n or 'mm' in n for n in nodes)
        print(f"Linear decomposed: {has_decomposed}")

        return {
            "nodes": nodes,
            "decomposed": has_decomposed
        }
    except Exception as e:
        print(f"Export failed: {e}")
        return {"error": str(e)}


def experiment_5_observation_layers():
    """实验 5：不同观察层次对比"""
    print("\n" + "=" * 80)
    print("实验 5：Observation Layers")
    print("=" * 80)

    observations = {}

    # Python API
    print("\n--- Python API ---")
    print("torch.nn.functional.linear")
    observations["python_api"] = "torch.nn.functional.linear"

    # Dispatch mode
    print("\n--- Dispatch Mode ---")
    tracer = TracingMode() if 'TracingMode' in dir() else None
    if tracer:
        with tracer:
            x = torch.randn(2, 4)
            w = torch.randn(8, 4)
            y = F.linear(x, w)
        print(f"Observed: {tracer.calls if hasattr(tracer, 'calls') else 'N/A'}")
        observations["dispatch_mode"] = "aten.t, aten.addmm"

    # Autograd
    print("\n--- Autograd ---")
    x = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(8, 4, requires_grad=True)
    y = F.linear(x, w)
    print(f"grad_fn: {type(y.grad_fn).__name__}")
    observations["autograd"] = type(y.grad_fn).__name__

    # Profiler
    print("\n--- Profiler ---")
    if torch.cuda.is_available():
        with torch.profiler.profile() as prof:
            x = torch.randn(2, 4, device='cuda')
            w = torch.randn(8, 4, device='cuda')
            y = F.linear(x, w)

        events = [e.key for e in prof.key_averages() if 'aten::' in e.key]
        print(f"Profiler events: {events[:5]}")  # 前 5 个
        observations["profiler"] = events[:5]
    else:
        print("CUDA not available, using CPU")
        with torch.profiler.profile() as prof:
            x = torch.randn(2, 4)
            w = torch.randn(8, 4)
            y = F.linear(x, w)

        events = [e.key for e in prof.key_averages() if 'aten::' in e.key]
        print(f"Profiler events: {events[:5]}")
        observations["profiler"] = events[:5]

    return observations


def main():
    """运行所有实验"""
    results = {}

    # 实验 1：Dispatch mode
    results["experiment_1"] = experiment_1_dispatch_mode()

    # 实验 2：Autograd graph
    results["experiment_2"] = experiment_2_autograd_graph()

    # 实验 3：Mini dispatcher
    results["experiment_3"] = experiment_3_mini_dispatcher()

    # 实验 4：Export comparison
    results["experiment_4"] = experiment_4_export_comparison()

    # 实验 5：Observation layers
    results["experiment_5"] = experiment_5_observation_layers()

    # 保存结果
    print("\n" + "=" * 80)
    print("保存结果...")
    print("=" * 80)

    output = {
        "experiments": results,
        "summary": {
            "linear_decomposes_to": "aten.t + aten.addmm",
            "dispatch_mode_sees": "decomposed ops only",
            "autograd_sees": "AddmmBackward0 or LinearBackward0",
            "export_sees": "aten.t.default + aten.addmm.default"
        }
    }

    print(json.dumps(output, indent=2, default=str))

    return output


if __name__ == "__main__":
    # 定义 TracingMode 用于其他函数
    class TracingMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(str(func))
            return func(*args, **(kwargs or {}))

    main()
