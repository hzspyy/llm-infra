#!/usr/bin/env python3
"""L2.0b-B 自己写一遍：位图 + 排除集 + redispatch 的最小 dispatcher。

优先级顺序照抄 `c10/core/DispatchKey.h` 的枚举顺序（越靠后优先级越高），
其中 Autocast 排在 Autograd 之后，源码注释给了理由：
`// Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
//  and inputs are saved for backward in the post-autocast type.`（:332-334）

最后一段用真实框架做同一件事：调用线程局部的排除集、autocast 下的
保存值 dtype，以及「不排除自己就重入」的真实报错。

    python mini_dispatcher.py
"""

import contextlib
import torch
import torch.nn.functional as F

# 摘自 c10/core/DispatchKey.h 的枚举顺序（只列本章用到的键）
KEY_ORDER = [
    "Undefined",
    "MkldnnCPU",
    "SparseCPU",
    "CPU", "CUDA",                       # 后端键（由 backend bit + Dense 组合而来）
    "NestedTensorCPU",
    "ZeroTensor",
    "ADInplaceOrView",                   # :289
    "AutogradOther", "AutogradCPU", "AutogradCUDA",   # :315
    "Tracer",                            # :329
    "AutocastCPU", "AutocastCUDA",       # :334 / :344
    "FuncTorchBatched",
    "PythonTLSSnapshot",
    "FuncTorchDynamicLayerFrontMode",
    "PreDispatch",
    "PythonDispatcher",
]
PRIORITY = {k: i for i, k in enumerate(KEY_ORDER)}

# 线程局部排除集：真实 dispatcher 用 `LocalDispatchKeySet` 存它
_tls_exclude = []


@contextlib.contextmanager
def exclude(*keys):
    _tls_exclude.extend(keys)
    try:
        yield
    finally:
        for k in keys:
            _tls_exclude.remove(k)


class MiniDispatcher:
    """按优先级选键、按排除集 redispatch。"""

    def __init__(self, name, keys, depth_limit=12):
        self.name = name
        self.keys = set(keys)
        self.impls = {}          # key -> callable(disp, keyset, args)
        self.kinds = {}          # key -> "kernel" / "fallback" / "math"
        self.calls = []          # 记录 (key, kind)
        self.depth = 0
        self.depth_limit = depth_limit

    def register(self, key, fn, kind="kernel"):
        self.impls[key] = fn
        self.kinds[key] = kind

    def _pick(self, keyset, exclude_set):
        cands = [k for k in keyset
                 if k not in exclude_set and k not in _tls_exclude and k in self.impls]
        if not cands:
            raise RuntimeError(f"{self.name}: 没有可用实现 keyset={sorted(keyset)}")
        return max(cands, key=lambda k: PRIORITY[k])

    def dispatch(self, keyset, args, exclude_set=frozenset()):
        key = self._pick(keyset, exclude_set)
        self.depth += 1
        try:
            if self.depth > self.depth_limit:
                raise RuntimeError(
                    f"{self.name}: 分发深度超过 {self.depth_limit}；"
                    f"{key} 实现 redispatch 时没有排除自己的键")
            self.calls.append((key, self.kinds[key]))
            return self.impls[key](self, key, keyset, args, exclude_set)
        finally:
            self.depth -= 1

    def redispatch(self, key, keyset, args, exclude_set):
        """已处理 key 的实现用这个入口往下走 —— 关键是把 key 放进排除集。"""
        return self.dispatch(keyset, args, exclude_set | {key})


# ------------------------------------------------------------------ 一个算子
def make_linear():
    d = MiniDispatcher("aten::linear", {"CPU", "CUDA", "AutogradCPU", "AutogradCUDA",
                                        "AutocastCPU", "ADInplaceOrView",
                                        "FuncTorchDynamicLayerFrontMode"})

    def cpu_kernel(disp, key, keyset, args, excl):
        x, w, b = args
        print("      [kernel] CPU: x @ w.T + b")
        return x @ w.t() + b

    def cuda_kernel(disp, key, keyset, args, excl):
        print("      [kernel] CUDA: cuBLAS gemm + epilogue")
        return args[0] @ args[1].t() + args[2]

    def autograd_wrapper(disp, key, keyset, args, excl):
        print(f"      [{key}] 建 LinearBackward0 并保存输入")
        out = disp.redispatch(key, keyset, args, excl)   # 排除自己
        print(f"      [{key}] 挂上 grad_fn")
        return out

    def autocast_wrapper(disp, key, keyset, args, excl):
        x, w, b = args
        print(f"      [{key}] 把 {x.dtype} 输入转成 bfloat16")
        out = disp.redispatch(key, keyset,
                              (x.to(torch.bfloat16), w.to(torch.bfloat16),
                               b.to(torch.bfloat16)), excl)
        print(f"      [{key}] 结果转回 float32")
        return out.to(torch.float32)

    def adinplace_wrapper(disp, key, keyset, args, excl):
        print(f"      [{key}] bump 版本计数 / 建 view meta")
        return disp.redispatch(key, keyset, args, excl)

    def front_mode_fallback(disp, key, keyset, args, excl):
        print(f"      [{key}] fallthrough")
        return disp.redispatch(key, keyset, args, excl)

    d.register("CPU", cpu_kernel)
    d.register("CUDA", cuda_kernel)
    d.register("AutogradCPU", autograd_wrapper)
    d.register("AutogradCUDA", autograd_wrapper)
    d.register("AutocastCPU", autocast_wrapper)
    d.register("ADInplaceOrView", adinplace_wrapper)
    d.register("FuncTorchDynamicLayerFrontMode", front_mode_fallback, kind="fallback")
    return d


SCENARIOS = [
    ("CPU, no_grad", {"CPU"}),
    ("CPU + Autograd", {"CPU", "AutogradCPU"}),
    ("CPU + Autograd + Autocast（真实顺序：Autocast 先）",
     {"CPU", "AutogradCPU", "AutocastCPU"}),
    ("CPU + Autograd + ADInplaceOrView（ADInplaceOrView 在 Autograd 之下）",
     {"CPU", "AutogradCPU", "ADInplaceOrView"}),
    ("CPU + CUDA 同时出现（后端键二选一）", {"CPU", "CUDA"}),
    ("CPU + 前端 fallthrough", {"CPU", "FuncTorchDynamicLayerFrontMode"}),
]


def section_mini():
    print("=" * 88)
    print("[mini] 位图 + 排除集 + redispatch")
    print("=" * 88)
    x = torch.randn(2, 4)
    w = torch.randn(5, 4)
    b = torch.randn(5)

    print(f"{'场景':<46} {'分发顺序'}")
    print("-" * 88)
    traces = []
    for name, keyset in SCENARIOS:
        d = make_linear()
        print()
        print(f"  {name}")
        out = d.dispatch(keyset, (x, w, b))
        seq = " -> ".join(k for k, _ in d.calls)
        traces.append((name, seq, tuple(out.shape)))
    print()
    print(f"{'场景':<46} {'分发顺序'}")
    print("-" * 88)
    for name, seq, shape in traces:
        print(f"{name:<46} {seq}   输出{shape}")

    print()
    print("读法：`AutocastCPU` 排在 `AutogradCPU` 之前，所以它先改输入、再让")
    print("autograd 看到转换后的张量 —— 与 DispatchKey.h:332 的注释一致。")
    print("`ADInplaceOrView` 排在 Autograd 之下，所以它最后才补版本计数。")


def section_recursion():
    print()
    print("=" * 88)
    print("[mini] 反例：redispatch 时不排除自己的键")
    print("=" * 88)
    d = make_linear()

    def bad_autograd(disp, key, keyset, args, excl):
        print(f"      [{key}] 直接重新分发，没有把自己放进排除集")
        return disp.dispatch(keyset, args, excl)      # 少了 | {key}

    d.register("AutogradCPU", bad_autograd)
    try:
        d.dispatch({"CPU", "AutogradCPU"}, (torch.randn(2, 4), torch.randn(5, 4),
                                            torch.randn(5)))
    except RuntimeError as exc:
        print(f"      RuntimeError: {exc}")

    print()
    print("真实框架的同一实验（用 torch.library 注册一个重入自己的 CPU kernel）：")
    torch.library.define("l2b_rec::rec", "(Tensor x) -> Tensor")
    lib = torch.library.Library("l2b_rec", "IMPL")

    def rec_impl(x):
        return torch.ops.l2b_rec.rec.default(x)

    lib.impl("rec", rec_impl, "CPU")
    try:
        torch.ops.l2b_rec.rec.default(torch.ones(2))
    except RecursionError as exc:
        print(f"      {type(exc).__name__}: {exc}")


def section_real():
    print()
    print("=" * 88)
    print("[对照] 真实框架里的同一套机制")
    print("=" * 88)

    print("线程局部排除集：")
    print(f"  普通上下文   {torch._C._dispatch_tls_local_exclude_set()}")
    with torch.no_grad():
        print(f"  no_grad      {torch._C._dispatch_tls_local_exclude_set()}")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        print(f"  autocast     {torch._C._dispatch_tls_local_exclude_set()}")
    print("  no_grad 不改变排除集（它翻的是 GradMode 布尔）；")
    print("  autocast 会把 AutocastCPU 从默认排除集里拿掉。")

    print()
    print("autocast 下面的保存值 dtype（用 saved_tensors_hooks 抓）：")
    saved = []
    x = torch.randn(4, 4, requires_grad=True)
    w = torch.randn(4, 4, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (saved.append((str(t.dtype), tuple(t.shape))), t)[1],
            lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = x @ w
    print(f"  y.dtype = {y.dtype}  grad_fn = {type(y.grad_fn).__name__}")
    print(f"  反向保存的张量（dtype, shape）：{saved}")
    y.sum().backward()
    print(f"  x.grad.dtype = {x.grad.dtype}")
    print("  保存下来的是 post-autocast 类型（bfloat16），与源码注释一致；")
    print("  叶子的梯度仍按叶子 dtype 累积。")

    print()
    print("autocast 下 F.linear 在 aten 层的算子序列：")
    from torch.utils._python_dispatch import TorchDispatchMode

    class Log(TorchDispatchMode):
        def __init__(self):
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(str(func).replace("torch.ops.", ""))
            return func(*args, **(kwargs or {}))

    x2 = torch.randn(2, 4)
    w2 = torch.randn(5, 4)
    log = Log()
    with log:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            F.linear(x2, w2)
    print(f"  {log.ops}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device=cpu")
    section_mini()
    section_recursion()
    section_real()
