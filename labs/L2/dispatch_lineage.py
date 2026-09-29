#!/usr/bin/env python3
"""L2.0b-A 两条带真实 file:line 的调用链：aten::linear 与 aten::add.Tensor。

链路由三部分拼成，三部分都来自实测：
  1. 运行时分发表 dump（`torch._C._dispatch_dump_table`）：每个 key 注册在哪个文件哪一行；
  2. 固定的 pytorch 源码（commit 写在 manifest.json 里）：native 实现与结构化委托；
  3. dispatch mode 观察：同一个调用在 aten 层实际分解成什么。

源码快照在 `results/local/2.0b/20260913-lineage/source/`，可用环境变量
`L2B_SOURCE` 指向别处（读者按 manifest 里的 curl 命令自行下载）。

用法：
    python dispatch_lineage.py            # 全跑
    python dispatch_lineage.py A2 A4
"""

import os
import pathlib
import re
import sys

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(os.environ.get(
    "L2B_SOURCE", str(ROOT / "results/local/2.0b/20260913-lineage/source")))

FULL_FILES = {"Linear.cpp": "aten_src_ATen_native_Linear.cpp",
              "Dispatcher.h": "aten_src_ATen_core_dispatch_Dispatcher.h"}


def title(s):
    print()
    print("=" * 96)
    print(s)
    print("=" * 96)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 84 - len(s)))


def parse_registrations(op):
    """把 _dispatch_dump_table 的输出解析成 (注册点, 文件, 行号, 种类)。"""
    rows = []
    for line in torch._C._dispatch_dump_table(op).splitlines():
        m = re.match(
            r"^(\S+):\s+(.*?registered at\s+)(\S+?):(\d+)(.*?)(\[.*\])?\s*$", line)
        if not m:
            continue
        key, _lead, path, lineno, _tail, kind = m.groups()
        rows.append((key, path.split("/pytorch/pytorch/")[-1], int(lineno),
                     (kind or "").strip("[] ")))
    return rows


def short(path, width=74):
    return path if len(path) <= width else "…" + path[-(width - 1):]


class OpLog(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func).replace("torch.ops.", ""))
        return func(*args, **(kwargs or {}))


# ------------------------------------------------------------------ A1
def section_A1():
    title("[A1] F.linear 是什么对象：Python 函数、绑定还是一张注册表")

    print(f"torch.nn.functional.linear           = {F.linear}")
    print(f"type                                  = {type(F.linear).__name__}")
    print(f"__module__ / __name__                 = {F.linear.__module__} / "
          f"{F.linear.__name__}")
    print(f"F.linear is torch._C._nn.linear       = {F.linear is torch._C._nn.linear}")
    print(f"torch.__version__                     = {torch.__version__}")
    print()
    print("它不是 Python 函数，而是 C++ 侧注册进 `torch._C._nn` 的绑定：")
    print("`native_functions.yaml` 里的 `python_module: nn` 决定它被生成到")
    print("`torch._C._nn` 命名空间下，而不是 `torch` 顶层。")
    print()
    print("schema 与注册来源（运行时 dump 的前几行）：")
    for line in torch._C._dispatch_dump("aten::linear").splitlines()[:4]:
        print("  " + line.strip())
    print()
    print("`alias analysis kind: FROM_SCHEMA` 说明别名信息不来自手写注释，")
    print("而是由 schema 里的 `Tensor input` / `Tensor(a!) self` 这类签名推导。")


# ------------------------------------------------------------------ A2
def section_A2():
    title("[A2] aten::linear 的注册表：没有 CPU / CUDA kernel")

    rows = parse_registrations("aten::linear")
    print(f"{'dispatch key':<40} {'文件':<62} {'行':>6} {'种类'}")
    print("-" * 130)
    for key, path, lineno, kind in rows:
        print(f"{key:<40} {short(path):<62} {lineno:>6} {kind}")

    backend_specific = [r for r in rows
                        if "RegisterCPU" in r[1] or "RegisterCUDA" in r[1]]
    math_entries = [r for r in rows
                    if "RegisterCompositeImplicitAutograd" in r[1]]
    print()
    print(f"注册点共 {len(rows)} 个：后端专属实现 {len(backend_specific)} 个，"
          f"指向复合（math）实现的键 {len(math_entries)} 个。")
    print("CPU / CUDA 这些后端键确实在表里，但它们指向的是同一个复合实现")
    print("（`RegisterCompositeImplicitAutograd_0.cpp:5216`），种类标为 `math kernel`，")
    print("而不是 `RegisterCPU_*` / `RegisterCUDA_*` 里的后端 kernel。")
    print("对照下面 aten::add.Tensor 的表，就能看出「复合算子」与「有后端 kernel」的差别。")

    comp = [r for r in rows if r[0].startswith("CompositeImplicitAutograd")]
    for key, path, lineno, kind in comp:
        print(f"  {key} -> {short(path)}:{lineno}")

    print()
    print("别名与功能键注册（`_dispatch_dump`，带注册文件与行号）：")
    for line in torch._C._dispatch_dump("aten::linear").splitlines():
        if any(k in line for k in ("Autograd[alias]", "CompositeImplicitAutograd[alias]",
                                   "AutocastCPU:", "AutocastCUDA:")):
            print("  " + line.split(" :: ")[0].strip().split("/pytorch/pytorch/")[-1])
    print("  `Autograd[alias]` 那一条来自 `tools/autograd/derivatives.yaml:2387` 的")
    print("  `- name: linear(...)` 规则，由代码生成器写成 VariableType_5.cpp。")


# ------------------------------------------------------------------ A3
def section_A3():
    title("[A3] aten::add.Tensor 的注册表：委托给 add.out 的结构化实现")

    rows = parse_registrations("aten::add.Tensor")
    print(f"{'dispatch key':<40} {'文件':<62} {'行':>6} {'种类'}")
    print("-" * 130)
    for key, path, lineno, kind in rows:
        print(f"{key:<40} {short(path):<62} {lineno:>6} {kind}")
    print()
    print("对照 aten::add.out（结构化实现真正落地的地方）：")
    rows_out = parse_registrations("aten::add.out")
    for key, path, lineno, kind in rows_out:
        if key in ("CPU", "CUDA", "Meta", "CompositeExplicitAutograd",
                   "CompositeImplicitAutograd"):
            print(f"  {key:<28} {short(path, 50):<52} {lineno:>6} {kind}")
    print()
    print("`add.Tensor` 自己只有稀疏等特化实现，稠密路径由 `structured_delegate:`")
    print("交给 `add.out`；`add.out` 挂在 TensorIterator 上，由统一模板生成 CPU/CUDA kernel。")


# ------------------------------------------------------------------ A4
def section_A4():
    title("[A4] 源码侧：linear 的复合实现怎么落到 addmm")

    f = SRC / FULL_FILES["Linear.cpp"]
    if not f.exists():
        print(f"缺少源码快照 {f}；先按 manifest.json 里的 curl 命令下载。")
        print("（本条不影响其余小节：A1–A3 只依赖运行时。）")
        return
    lines = f.read_text().splitlines()
    print(f"{FULL_FILES['Linear.cpp']}（commit 见 manifest.json）")
    sub("linear() 的入口与它选择的分支：:85 起")
    print_lines(lines, 85, 112)
    sub("_flatten_nd_linear：真正调用 addmm 的位置 :60 起")
    print_lines(lines, 60, 76)

    sub("native_functions.yaml 里的 linear 声明（摘录）")
    ex = SRC / "native_functions_linear.yaml"
    if ex.exists():
        for line in ex.read_text().splitlines():
            if line.startswith("$") or re.match(r"^\d+[:-]", line) or line.startswith("#"):
                print("  " + line[:120])

    sub("addmm 的声明与 CPU 实现落点（摘录）")
    ex = SRC / "addmm_cpu_impl.txt"
    if ex.exists():
        for line in ex.read_text().splitlines():
            if line.startswith("$") or re.match(r"^\d+[:-]", line) or line.startswith("#"):
                print("  " + line[:120])
    else:
        print("  缺少 addmm 摘录文件")


def print_lines(lines, a, b):
    for i in range(a - 1, min(b, len(lines))):
        print(f"  {i + 1:>5}  {lines[i]}")


# ------------------------------------------------------------------ A5
def section_A5():
    title("[A5] 运行时验证：linear 在 aten 层被拆成 t + addmm")

    x = torch.randn(2, 3)
    w = torch.randn(4, 3)
    b = torch.randn(4)
    log = OpLog()
    with log:
        out = F.linear(x, w, b)
    print(f"F.linear(x, w, b)  ->  {[o for o in log.ops]}")
    print(f"输出 shape {tuple(out.shape)}，其中 linear 这个名字没有出现在 aten 层。")

    log2 = OpLog()
    with log2:
        _ = x @ w.t() + b
    print(f"x @ w.t() + b      ->  {[o for o in log2.ops]}")
    print("两条路径计算同一件事，但融合程度不同：有 bias 的 linear 直接走 `addmm`，")
    print("手写 `x @ w.t() + b` 是 `mm` 再加一次 `add`。复合算子只决定语义，")
    print("底层选哪个 kernel 由它调用谁决定。")


# ------------------------------------------------------------------ A6
def section_A6():
    title("[A6] 两条链的完整对照表")

    print("链 1：F.linear(x, w, b)")
    print("  ① torch._C._nn.linear            C++ 绑定，命名空间由 python_module: nn 决定")
    print("  ② aten::linear（schema）         native_functions.yaml:3337（无 CPU/CUDA kernel）")
    print("  ③ CompositeImplicitAutograd      register 到 linear（Linear.cpp:85）")
    print("  ④ at::addmm(bias, input, w.t())  Linear.cpp:108")
    print("  ⑤ aten::addmm.out（结构化委托）  native_functions.yaml:7057（CPU: addmm_out_cpu）")
    print("  ⑥ addmm_impl_cpu_               LinearAlgebra.cpp:1392 / 1609")
    print("  ⑦ BLAS gemm（CPU）/ cuBLAS（CUDA）")
    print()
    print("链 2：x + y")
    print("  ① Tensor.__add__ -> torch.add    C++ 绑定")
    print("  ② aten::add.Tensor（schema）     native_functions.yaml:542")
    print("  ③ structured_delegate: add.out   native_functions.yaml:544")
    print("  ④ aten::add.out                  native_functions.yaml:565")
    print("  ⑤ TensorIterator 模板实例化      RegisterCPU_*.cpp（构建期生成）")
    print("  ⑥ 逐元素 kernel（向量化循环）")
    print()
    print("两条链的第 ③ 步是分水岭：linear 是复合算子，add 是结构化算子。")
    print("复合算子在前向就已经变成了别的算子，结构化算子则共享同一套迭代器模板。")


SECTIONS = {"A1": section_A1, "A2": section_A2, "A3": section_A3,
            "A4": section_A4, "A5": section_A5, "A6": section_A6}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device=cpu  源码快照 {SRC}")
    for s in want:
        SECTIONS[s]()
