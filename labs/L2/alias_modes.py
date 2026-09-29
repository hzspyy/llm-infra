#!/usr/bin/env python3
"""L2.0-C 别名与原地修改：eager、functionalization、compile 三条路径。

同一个含别名与原地修改的程序，分别观察：
  1. dispatcher 层看到哪些算子（原地算子 vs 函数式算子 vs copy_）；
  2. 输入张量在前后的数值与版本计数；
  3. 原地修改打断保存值时，三条路径分别报错还是静默算完。

只打印事实，结论留给正文。

用法：
    python alias_modes.py          # 全跑
    python alias_modes.py C1 C3
"""

import sys
import warnings

import torch
import torch.func as tf
import torch.nn as nn
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


def fmt_ops(ops):
    return " ".join(("!" if "_." in o else "") + o for o in ops)


def program(x, w):
    """中间张量原地改、通过别名再改、最后改输入。"""
    y = x + w
    y.mul_(2.0)          # 原地改中间张量
    z = y.t()
    z.add_(1.0)          # 通过别名改同一块存储
    x.add_(10.0)         # 原地改输入
    return y.sum()


def fresh():
    return torch.ones(2, 3), torch.full((2, 3), 2.0)


# ------------------------------------------------------------------ C1
def section_C1():
    title("[C1] dispatcher 层：原地算子、函数式算子与 copy_")

    variants = [
        ("eager", lambda: program),
        ("functionalize(remove='mutations')",
         lambda: tf.functionalize(program, remove="mutations")),
        ("functionalize(remove='mutations_and_views')",
         lambda: tf.functionalize(program, remove="mutations_and_views")),
    ]
    for name, build in variants:
        x, w = fresh()
        fn = build()
        log = OpLog()
        with log:
            out = fn(x, w)
        print(f"{name}")
        print(f"  算子序列（! 标记原地算子）：{fmt_ops(log.ops)}")
        print(f"  输出 {float(out):.1f}   x 之后 {x.flatten()[:2].tolist()}   "
              f"x._version={x._version}")
        print()

    sub("reshape / transpose 在两种 remove 下的差别")
    def views(x):
        a = x.reshape(3, 2)
        b = a.t()
        return b.sum()

    for name, build in [
        ("eager", lambda: views),
        ("functionalize('mutations')", lambda: tf.functionalize(views, remove="mutations")),
        ("functionalize('mutations_and_views')",
         lambda: tf.functionalize(views, remove="mutations_and_views")),
    ]:
        x = torch.ones(2, 3)
        log = OpLog()
        with log:
            out = build()(x)
        print(f"  {name:<36} {fmt_ops(log.ops)}   -> {float(out):.1f}")


# ------------------------------------------------------------------ C2
def section_C2():
    title("[C2] 输入状态对拍：输出一样，输入和版本计数也未必一样")

    modes = [
        ("eager", lambda fn: fn),
        ("functionalize('mutations')",
         lambda fn: tf.functionalize(fn, remove="mutations")),
        ("compile(backend='eager')",
         lambda fn: torch.compile(fn, backend="eager", fullgraph=False)),
        ("compile(backend='aot_eager')",
         lambda fn: torch.compile(fn, backend="aot_eager", fullgraph=False)),
        ("compile(backend='inductor')",
         lambda fn: torch.compile(fn, backend="inductor", fullgraph=False)),
    ]

    x0, w0 = fresh()
    x0_before = x0.clone()
    ref = program(x0, w0)
    ref_x = x0.clone()
    print(f"eager 参照：输入前={x0_before.flatten()[:2].tolist()} "
          f"输出={float(ref):.1f}  x 之后={ref_x.flatten()[:2].tolist()} "
          f"x._version={x0._version}")
    print()
    print(f"{'模式':<26} {'输出':>6} {'输入被改':^8} {'x._version':>10} "
          f"{'与 eager 输出一致':^16} {'与 eager 输入一致':^16}")
    print("-" * 92)
    for name, wrap in modes:
        x, w = fresh()
        fn = wrap(program)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = fn(x, w)
                err = None
            except Exception as exc:
                out, err = None, exc
        if err is not None:
            print(f"{name:<26} {'ERR':>6} {type(err).__name__:<8} "
                  f"{str(err).splitlines()[0][:80]}")
            continue
        print(f"{name:<26} {float(out):>6.1f} "
              f"{'是' if not torch.equal(x, x0_before) else '否':^8} "
              f"{x._version:>10} "
              f"{'是' if torch.allclose(out, ref) else '否':^16} "
              f"{'是' if torch.equal(x, ref_x) else '否':^16}")

    sub("编译后的第二次调用：图复用与版本计数")
    x, w = fresh()
    fn = torch.compile(program, backend="inductor", fullgraph=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o1 = fn(x, w)
        v1 = x._version
        o2 = fn(x, w)
        v2 = x._version
    print(f"  第一次 out={float(o1):.1f} x._version={v1}")
    print(f"  第二次 out={float(o2):.1f} x._version={v2}  "
          f"（每次调用都真的改了输入：+{v2 - v1}）")


# ------------------------------------------------------------------ C3
def section_C3():
    title("[C3] 失效反例：原地修改打断保存值，三条路径的处置不同")

    def bad(x):
        y = x * 2
        z = y * y        # MulBackward 把 y 存下来
        y.add_(1.0)      # 改掉反向要用的保存值
        return z.sum()

    modes = [
        ("eager", lambda fn: fn),
        ("functionalize('mutations')",
         lambda fn: tf.functionalize(fn, remove="mutations")),
        ("compile(backend='eager')",
         lambda fn: torch.compile(fn, backend="eager", fullgraph=False)),
        ("compile(backend='aot_eager')",
         lambda fn: torch.compile(fn, backend="aot_eager", fullgraph=False)),
        ("compile(backend='inductor')",
         lambda fn: torch.compile(fn, backend="inductor", fullgraph=False)),
    ]
    print(f"{'模式':<26} {'前向':^6} {'反向':^6} {'输出':>7} {'grad[0]':>9} "
          f"{'报错首行'}")
    print("-" * 110)
    for name, wrap in modes:
        x = torch.ones(4, requires_grad=True)
        fn = wrap(bad)
        outv = gradv = None
        ferr = berr = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = fn(x)
                outv = float(out.detach())
            except Exception as exc:
                ferr = exc
            if ferr is None:
                try:
                    out.backward()
                    gradv = x.grad[0].item()
                except Exception as exc:
                    berr = exc
        err = ferr or berr
        print(f"{name:<26} {'ERR' if ferr else 'OK':^6} "
              f"{'—' if ferr else ('ERR' if berr else 'OK'):^6} "
              f"{'' if outv is None else f'{outv:>7.1f}'} "
              f"{'' if gradv is None else f'{gradv:>9.3f}'} "
              f"{'' if err is None else type(err).__name__ + ': ' + str(err).splitlines()[0][:60]}")

    sub("compile 的报错把 autograd 错误包在编译器异常里")
    x = torch.ones(4, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.compile(bad, backend="aot_eager", fullgraph=False)(x)
        except Exception as exc:
            lines = [ln for ln in str(exc).splitlines() if ln.strip()]
            for ln in lines[:6]:
                print("  " + ln.strip()[:110])
            print(f"  ...（共 {len(lines)} 行）")

    sub("版本计数就是判定依据")
    x = torch.ones(4, requires_grad=True)
    y = x * 2
    z = y * y
    print(f"  构造后 y._version={y._version}")
    y.add_(1.0)
    print(f"  原地改后 y._version={y._version}（MulBackward 记的是 0）")
    try:
        z.sum().backward()
    except RuntimeError as exc:
        print("  报错原文：" + str(exc).splitlines()[0])

    sub("functionalize 把原地修改换成 copy_，保存值因此不再失效")
    x = torch.ones(4, requires_grad=True)
    log = OpLog()
    with log:
        out = tf.functionalize(bad, remove="mutations")(x)
        out.backward()
    print(f"  算子序列：{fmt_ops(log.ops)}")
    print(f"  输出 {float(out.detach()):.1f}  grad[0]={x.grad[0].item():.3f}  无报错")
    print(f"  反向用的 y 仍是 2：dz/dx = 2 × 2y = {2 * 2 * 2.0:.1f}，"
          f"原地修改没有污染保存的 y")


# ------------------------------------------------------------------ C4
def section_C4():
    title("[C4] 观察层次之三：图里留下的算子与运行时的真实拷贝")

    class Clean(nn.Module):
        def forward(self, x):
            y = x + 1
            return y.sum()

    class Mutates(nn.Module):
        def forward(self, x):
            x.add_(1.0)
            return x.sum()

    for name, mod in [("clean", Clean()), ("mutates_input", Mutates())]:
        try:
            ep = torch.export.export(mod, (torch.ones(2, 3),))
            code = " ".join(ep.graph_module.code.split())
            print(f"  export {name:<14} 成功：{code[:150]}")
        except Exception as exc:
            print(f"  export {name:<14} 失败：{type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:120]}")

    sub("编译触发时 TorchDispatchMode 记录到的是追踪期算子序列")
    print("  （要看编译后的图本身用 2.6b/2.7 的 IR 与 AOT 工具，这里只看追踪期）")
    class Alias(nn.Module):
        def forward(self, x):
            y = x + 1
            z = y.t()
            z.add_(1.0)
            return y.sum()

    for backend in ["eager", "aot_eager", "inductor"]:
        x = torch.ones(2, 2)
        log = OpLog()
        mod = torch.compile(Alias(), backend=backend, fullgraph=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with log:
                out = mod(x)
        print(f"  compile({backend:<9}) {fmt_ops(log.ops)} -> {float(out):.1f}")

    sub("导出图中的 view 与拷贝")
    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(3, 2).sum() + x.t().reshape(-1).sum()

    ep = torch.export.export(Reshape(), (torch.ones(2, 3),))
    print("  " + " ".join(ep.graph_module.code.split())[:300])

    x = torch.ones(2, 3)
    v1 = x.reshape(3, 2)
    v2 = x.t().reshape(-1)
    print(f"  eager  x.reshape(3,2)      同 storage: "
          f"{v1.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()}")
    print(f"  eager  x.t().reshape(-1)   同 storage: "
          f"{v2.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()}"
          f"   <- 图里同样是 reshape，运行时却拷了")
    print("  图只记录算子名，view 还是拷贝由运行时的 stride 决定。")


SECTIONS = {"C1": section_C1, "C2": section_C2, "C3": section_C3, "C4": section_C4}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device=cpu  seed=0")
    for s in want:
        SECTIONS[s]()
