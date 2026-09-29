#!/usr/bin/env python3
"""L2.0b 补测 · CUDA 分支的注册点（必须在带 CUDA 的 torch 上跑）。

本地是 CPU-only wheel（torch 2.14.0，`cuda_available=false`），
`torch._C._dispatch_dump_table` 里根本没有 CUDA 键，所以 2.0b 的正文在
"CUDA 分支注册点"这一项上保持 `UNVERIFIED`。本脚本在 crater（torch 2.13.0+cu130）
上用同一套解析复采，回答三件事：

  1. `aten::linear` 的 CUDA 键指向哪里——是 `RegisterCUDA_*.cpp` 里的后端 kernel，
     还是和 CPU 一样指向 `RegisterCompositeImplicitAutograd_*.cpp` 的复合实现；
  2. `aten::addmm` 与 `aten::add.Tensor` 的 CUDA 实现在哪个文件哪一行；
  3. 注册点与实际执行的 kernel 是否对得上（profiler 里看真实 kernel 名）。

    /scratch/learn/envs/serve/bin/python labs/L2/dispatch_cuda_branch.py \
        --out-json results/crater/2.0b/<run>/cuda_branch.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import torch

OPS = ["aten::linear", "aten::addmm", "aten::addmm.out", "aten::add.Tensor",
       "aten::add.out", "aten::mm", "aten::relu"]


def parse(op: str) -> list[dict]:
    rows = []
    for line in torch._C._dispatch_dump_table(op).splitlines():
        m = re.match(r"^(\S+):\s+(.*?registered at\s+)(\S+?):(\d+)(.*?)(\[.*\])?\s*$", line)
        if not m:
            continue
        key, _lead, path, lineno, _tail, kind = m.groups()
        rows.append({"key": key,
                     "file": path.split("/pytorch/pytorch/")[-1],
                     "line": int(lineno),
                     "kind": (kind or "").strip("[] ")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("本机没有 CUDA —— 这个脚本要在 crater 上跑")
        return 2
    print(f"device {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")

    result: dict = {"torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0),
                    "cc": list(torch.cuda.get_device_capability(0)),
                    "ops": {}}

    for op in OPS:
        rows = parse(op)
        result["ops"][op] = rows
        cuda = [r for r in rows if r["key"].startswith("CUDA")]
        backend = [r for r in cuda if "RegisterCUDA" in r["file"]]
        composite = [r for r in cuda if "Composite" in r["file"]]
        print(f"\n=== {op}：{len(rows)} 个注册点，其中 CUDA 键 {len(cuda)} 个 "
              f"（后端专属 {len(backend)}，复合实现 {len(composite)}）")
        for r in cuda:
            print(f"    {r['key']:34s} {r['file']}:{r['line']}  {r['kind']}")
        other = [r for r in rows if r["key"] in ("CPU", "Autograd", "AutogradCUDA",
                                                 "AutocastCUDA", "CompositeImplicitAutograd")]
        for r in other:
            print(f"    {r['key']:34s} {r['file']}:{r['line']}  {r['kind']}")

    # 实际执行的 kernel：注册点说的是"谁来接这个调用"，profiler 说的是"最后跑了什么"
    print("\n=== profiler：一次真实的 CUDA 前向跑了哪些 kernel ===")
    x = torch.randn(512, 2048, device="cuda")
    w = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, device="cuda")
    lin = torch.nn.Linear(2048, 2048).cuda()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = torch.nn.functional.linear(x, lin.weight, lin.bias)
        y2 = torch.relu(y + 1.0)
        z = x @ w + b
        torch.cuda.synchronize()
    kernels = {}
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA or ev.self_device_time_total > 0:
            kernels[ev.key] = {"calls": ev.count, "self_cuda_time_us": round(ev.self_device_time_total, 1)}
    for k, v in sorted(kernels.items(), key=lambda kv: -kv[1]["self_cuda_time_us"]):
        print(f"    {v['self_cuda_time_us']:10.1f} µs  ×{v['calls']:<3d} {k[:110]}")
    result["profiler_kernels"] = kernels

    if args.out_json:
        p = pathlib.Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
