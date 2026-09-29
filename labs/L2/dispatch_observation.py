#!/usr/bin/env python3
"""L2.0b-C 同一程序的三种观察层次：dispatch mode、export 图、profiler。

程序：`relu(linear(x, w, b))`。
第一遍用框架自带的算子（会被分解），第二遍换成一个不可分解的自定义算子，
对比三种观察层次各自看到什么名字、什么粒度。

自定义算子部分与 2.8 用的是同一套接口（`torch.library` + `register_fake`）。

    python dispatch_observation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

torch.manual_seed(0)


def title(s):
    print()
    print("=" * 96)
    print(s)
    print("=" * 96)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 84 - len(s)))


class OpLog(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func).replace("torch.ops.", ""))
        return func(*args, **(kwargs or {}))


class Baseline(nn.Module):
    def forward(self, x, w, b):
        return F.relu(F.linear(x, w, b))


def define_custom_op():
    """2.8 的最小前身：一个 schema、一个 CPU 实现、一个 fake 实现。"""
    torch.library.define(
        "l2b_obs::fused_linear_relu",
        "(Tensor x, Tensor w, Tensor b) -> Tensor")

    lib = torch.library.Library("l2b_obs", "IMPL")

    def impl(x, w, b):
        return torch.relu(F.linear(x, w, b))

    lib.impl("fused_linear_relu", impl, "CPU")

    @torch.library.register_fake("l2b_obs::fused_linear_relu")
    def _fake(x, w, b):
        return x.new_empty((*x.shape[:-1], w.shape[0]))

    class Custom(nn.Module):
        def forward(self, x, w, b):
            return torch.ops.l2b_obs.fused_linear_relu.default(x, w, b)

    return Custom(), lib


def run_dispatch_mode(mod, *args):
    log = OpLog()
    with log:
        mod(*args)
    return log.ops


def run_export(mod, x, w, b):
    ep = torch.export.export(mod, (x, w, b))
    nodes = [(n.name, n.op, getattr(n.target, "__name__", str(n.target)))
             for n in ep.graph_module.graph.nodes]
    return ep, nodes


def run_profiler(mod, x, w, b, iters=5):
    for _ in range(2):
        mod(x, w, b)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            mod(x, w, b)
    rows = []
    for ev in prof.key_averages():
        if ev.key.startswith("aten::") or ev.key.startswith("l2b_obs::"):
            rows.append((ev.key, ev.count,
                         ev.self_cpu_time_total / 1000.0,
                         ev.cpu_time_total / 1000.0))
    return rows, iters


def report(label, mod, x, w, b):
    sub(f"{label}：dispatch mode")
    print("  " + " ".join(run_dispatch_mode(mod, x, w, b)))

    sub(f"{label}：export 图")
    ep, nodes = run_export(mod, x, w, b)
    for name, op, target in nodes:
        print(f"  {name:<16} {op:<10} {target}")
    print("  graph code:")
    for line in ep.graph_module.code.strip().splitlines():
        print("    " + line)
    try:
        ep2 = ep.run_decompositions()
        targets = [n.target.__name__ for n in ep2.graph_module.graph.nodes
                   if n.op == "call_function"]
        print("  再过一遍分解 pass（ep.run_decompositions()）：")
        print(f"    {targets}")
    except Exception as exc:
        print(f"  分解 pass 报错：{type(exc).__name__}: {str(exc).splitlines()[0][:100]}")


def main():
    title("[C1] 框架自带算子：linear 在 aten 层已经不存在")

    x = torch.randn(2, 4)
    w = torch.randn(5, 4)
    b = torch.randn(5)
    baseline = Baseline()

    report("baseline", baseline, x, w, b)

    rows, iters = run_profiler(baseline, x, w, b)
    sub("baseline：profiler（CPU activity，5 次调用，单位 ms）")
    print(f"  {'op':<34} {'次数':>4} {'self':>8} {'total':>8}")
    for key, count, self_ms, total_ms in sorted(rows, key=lambda r: -r[3]):
        print(f"  {key:<34} {count:>4} {self_ms:>8.3f} {total_ms:>8.3f}")
    print("  注意 `aten::linear` 这一行：profiler 在 dispatcher 入口就打了点，")
    print("  所以它既记录复合算子本身，又记录它内部的 `t` / `addmm`。")
    print("  `total` 一列把子调用的时间也算了进去，逐行相加会重复计数；")
    print("  比较算子开销要看 `self`，或者只看叶子算子。")

    title("[C2] 自定义算子：同一程序，三层都只剩一个名字")

    custom, lib = define_custom_op()
    x2, w2, b2 = x.clone(), w.clone(), b.clone()
    report("custom", custom, x2, w2, b2)

    rows2, _ = run_profiler(custom, x2, w2, b2)
    sub("custom：profiler")
    print(f"  {'op':<34} {'次数':>4} {'self':>8} {'total':>8}")
    for key, count, self_ms, total_ms in sorted(rows2, key=lambda r: -r[3]):
        print(f"  {key:<34} {count:>4} {self_ms:>8.3f} {total_ms:>8.3f}")
    print("  自定义算子自己出现在表里（`l2b_obs::fused_linear_relu`），")
    print("  它的 Python 实现内部的 aten 调用同样会被记录 —— "
          "profiler 看执行，不看图的边界。")

    sub("数值对拍：自定义算子与分解实现必须一致")
    ref = baseline(x, w, b)
    got = custom(x, w, b)
    print(f"  baseline = {ref.flatten()[:4].tolist()}")
    print(f"  custom   = {got.flatten()[:4].tolist()}")
    print(f"  max |diff| = {(ref - got).abs().max().item():.3e}")

    title("[C3] 反例：没有 fake 实现的自定义算子进不了 export")

    torch.library.define("l2b_obs::no_fake", "(Tensor x) -> Tensor")
    lib2 = torch.library.Library("l2b_obs", "IMPL")

    def no_fake_impl(x):
        return x + 1

    lib2.impl("no_fake", no_fake_impl, "CPU")

    class NoFake(nn.Module):
        def forward(self, x):
            return torch.ops.l2b_obs.no_fake.default(x)

    print("dispatch mode 里它可以正常执行：")
    print("  " + " ".join(run_dispatch_mode(NoFake(), torch.ones(3))))

    print()
    print("export 却会失败，因为形状推导拿不到 fake 实现：")
    try:
        torch.export.export(NoFake(), (torch.ones(3),))
        print("  没有报错（预期会报错）")
    except Exception as exc:
        cur, depth = exc, 0
        while cur is not None and depth < 4:
            head = str(cur).splitlines()[0] if str(cur) else ""
            print(f"  [{type(cur).__name__}] {head[:110]}")
            cur = cur.__cause__ or cur.__context__
            depth += 1


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device=cpu")
    main()
