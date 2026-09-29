#!/usr/bin/env python3
"""对 cubin/SASS 做指令普查：按助记符族统计，并打印资源用量。

用于 2.2（PTX → SASS 对应）与 2.4-D（sm_120 路径到底发射了哪一族 MMA）。

    python labs/L2/sass_census.py <cubin 或 .sass 文本> [...]

输出：每个文件的总指令数、助记符直方图（按首段归一，如
`OMMA.SF.16832.F32.E4M3.E4M3` 归到 `OMMA`）、以及 cuobjdump -res-usage 摘要。
"""

from __future__ import annotations

import collections
import pathlib
import re
import shutil
import subprocess
import sys

# SASS 行的形状： /*0080*/  @P0  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
SASS_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*)")

CUOBJDUMP_CANDIDATES = [
    "/scratch/learn/opt/cuobjdump/cuda_cuobjdump-linux-x86_64-13.0.85-archive/bin/cuobjdump",
    "cuobjdump",
]


def find_cuobjdump() -> str:
    for c in CUOBJDUMP_CANDIDATES:
        p = shutil.which(c) if "/" not in c else (c if pathlib.Path(c).exists() else None)
        if p:
            return p
    raise SystemExit("找不到 cuobjdump")


def nvdisasm_env() -> dict:
    """cuobjdump -sass 需要 PATH 里能找到 nvdisasm（wheel CUDA 只把 ptxas/nvcc 放进 bin）。"""
    import os
    env = dict(os.environ)
    extra = []
    for cand in ("nvdisasm",
                 "/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13/bin/nvdisasm"):
        d = os.path.dirname(cand) if "/" in cand else ""
        if "/" in cand and pathlib.Path(cand).exists():
            extra.append(d)
        elif shutil.which(cand):
            extra.append(os.path.dirname(shutil.which(cand)))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["NVDISASM_PATH"] = extra[0]
    return env


def sass_text(cuobjdump: str, path: pathlib.Path, env: dict) -> str:
    if path.suffix == ".sass" or path.name.endswith(".txt"):
        return path.read_text(errors="replace")
    proc = subprocess.run([cuobjdump, "-sass", str(path)],
                          capture_output=True, text=True, env=env)
    if "cuobjdump fatal" in proc.stderr + proc.stdout:
        raise SystemExit(f"cuobjdump 失败：{proc.stderr.strip()[:200]}")
    return proc.stdout


def census(text: str) -> tuple[int, collections.Counter]:
    hist: collections.Counter = collections.Counter()
    total = 0
    for line in text.splitlines():
        m = SASS_LINE.search(line)
        if not m:
            continue
        total += 1
        # 首段就是指令族：QMMA.SF.16832 → QMMA
        hist[m.group(1).split(".")[0]] += 1
    return total, hist


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cuobjdump = find_cuobjdump()
    env = nvdisasm_env()
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        text = sass_text(cuobjdump, path, env)
        total, hist = census(text)
        print(f"== {path.name}: {total} 条指令 ==")
        for mnem, n in hist.most_common(18):
            print(f"   {mnem:10s} {n:7d}  {n / max(total, 1) * 100:5.1f}%")
        if path.suffix == ".cubin":
            res = subprocess.run([cuobjdump, "-res-usage", str(path)],
                                 capture_output=True, text=True, env=env).stdout
            for line in res.splitlines():
                s = line.strip()
                if s.startswith(("REG:", "Function ", "SHARED:")) or "REG:" in s:
                    print("   " + s[:200])
        # 固定家族合计：低频但关键的指令（TMA、MMA）不会进 top-18，单独列
        families = ["UTMA", "HMMA", "QMMA", "OMMA", "IMMA", "LDSM", "STSM",
                    "LDL", "STL", "LDG", "STG", "TENSORMAP"]
        fam_totals = {f: sum(n for m, n in hist.items() if m.startswith(f)) for f in families}
        fam_totals = {f: n for f, n in fam_totals.items() if n}
        if fam_totals:
            print("   家族合计: " + "  ".join(f"{f} {n}" for f, n in fam_totals.items()))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
