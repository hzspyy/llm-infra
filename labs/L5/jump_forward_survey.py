#!/usr/bin/env python3
"""L5.6 任务 C 前置 —— jump-forward 在固定 SGLang 版本里到底能不能被触发。

正文此前把 jump-forward 的端到端收益记为 `UNVERIFIED`，并计划「在 SGLang 上开关
`--disable-jump-forward`」。要在固定版本上给出结论，先得确认三件事：

  1. 后端类里有没有 `try_jump_forward` / `jump_forward_str_state` 的**定义**；
  2. 除定义之外有没有**调用点**（`obj.try_jump_forward(` 形式）；
  3. 服务端有没有对应的**开关**（`launch_server` 的参数名里含 jump）。

三条都在同一份已安装源码上静态核对，不做任何执行路径假设。
`--disable-jump-forward` 这类开关如果不存在，就不能用「打开/关闭」的方式测收益。

用装了 SGLang 的解释器运行（crater 上是 `envs/sgl`）：
    python labs/L5/jump_forward_survey.py --out "$OUT/jump-survey"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

SYMS = ["try_jump_forward", "jump_forward_str_state", "jump_forward_byte",
        "jump_forward_symbol"]


def package_root():
    import sglang
    return pathlib.Path(sglang.__file__).parent


def survey(root: pathlib.Path):
    defs, calls, files = [], [], 0
    for py in sorted(root.rglob("*.py")):
        files += 1
        text = py.read_text(errors="ignore")
        if not any(s in text for s in SYMS):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name in SYMS:
                defs.append(dict(file=str(py.relative_to(root)), line=node.lineno,
                                 symbol=node.name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in SYMS:
                calls.append(dict(file=str(py.relative_to(root)), line=node.lineno,
                                  symbol=node.func.attr))
    return dict(python_files_scanned=files, definitions=defs, call_sites=calls)


def server_args_with_jump():
    """launch_server 的参数名里有没有 jump 相关开关。"""
    from sglang.srt.server_args import ServerArgs
    fields = list(getattr(ServerArgs, "__dataclass_fields__", {}))
    return [f for f in fields if "jump" in f.lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import sglang
    root = package_root()
    s = survey(root)
    flags = server_args_with_jump()

    # 关键判据：真正决定「服务路径会不会触发」的是 constrained 包之外的调用点。
    outside = [d for d in s["call_sites"]
               if not d["file"].startswith("srt/constrained/")]
    report = dict(
        sglang_version=sglang.__version__,
        package_root=str(root),
        python_files_scanned=s["python_files_scanned"],
        definitions=s["definitions"],
        call_sites=s["call_sites"],
        call_sites_outside_constrained=outside,
        server_args_with_jump=flags,
        reachable=bool(outside),
    )
    (args.out / "jump_survey.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"sglang {report['sglang_version']}  {root}")
    print(f"扫描 {s['python_files_scanned']} 个 .py 文件")
    print(f"\n后端类里的定义（{len(s['definitions'])} 处）：")
    for d in s["definitions"]:
        print(f"  {d['file']}:{d['line']}  def {d['symbol']}")
    print(f"\n调用点（{len(s['call_sites'])} 处）：")
    for d in s["call_sites"]:
        print(f"  {d['file']}:{d['line']}  {d['symbol']}(...)")
    if not s["call_sites"]:
        print("  （没有调用点）")
    print(f"\nServerArgs 里含 jump 的字段：{flags or '（无）'}")
    print(f"\n`constrained/` 包之外的调用点：{len(outside)} 处")
    for d in outside:
        print(f"  {d['file']}:{d['line']}  {d['symbol']}(...)")
    print(f"\n结论：在 sglang {report['sglang_version']} 上，jump-forward 的"
          f"执行路径在服务侧{'有调用点' if report['reachable'] else '没有任何调用点'}"
          f"（constrained 包内的 {len(s['call_sites'])} 处都是后端自己的转调），"
          f"服务端开关{'存在' if flags else '不存在'}。")
    if not report["reachable"]:
        print("因此无法用「打开/关闭 jump-forward」的方式测端到端收益：")
        print("固定版本里这段能力只有定义与包内转调，没有服务路径上的调用者，")
        print("也没有对应参数。正文保留字符级上界，并把补测条件写成")
        print("「换到实际在请求路径上调用它的版本」。")


if __name__ == "__main__":
    sys.exit(main())
