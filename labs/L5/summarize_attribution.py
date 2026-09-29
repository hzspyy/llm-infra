#!/usr/bin/env python3
"""L5.4 任务 G —— 从 attribution 目录汇总「CPU 与 GPU 各占多少」。

读取 `run_execution_attribution.sh` 落下的两组窗口（gen=0 的 prefill 窗口与
gen=G 的 prefill+decode 窗口），对每个 (mode, bs) 相减得到**纯 decode** 的：

  wall_ms/步   无插桩墙钟差分
  cpu_ms/步    无插桩进程 CPU 时间差分（time.process_time）
  gpu_ms/步    nsys 里 kernel device 时间差分
  提交次数/步  图外的 kernel launch + GraphLaunch 次数

为什么不用 nsys 的 CPU API 时长：`--cuda-graph-trace=node` 会逐节点注入，
实测把 `cudaGraphLaunch` 从几十微秒放大到 6.1 ms/次（bs=64 的 15 次共 91.9 ms），
所以图模式的 API 时长不可用；提交**次数**和 kernel **device 时间**不受影响。

用法：
    python summarize_attribution.py <attribution 目录>
"""

from __future__ import annotations

import pathlib
import re
import sqlite3
import sys
from collections import Counter

FIELD = re.compile(r"(\w+)=(\S+)")


def parse_log(path: pathlib.Path) -> dict | None:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("MODE="):
            d = dict(FIELD.findall(line))
            return d
    return None


def sqlite_totals(path: pathlib.Path) -> dict:
    c = sqlite3.connect(str(path))
    def q(sql):
        try:
            return c.execute(sql).fetchone()[0] or 0
        except sqlite3.Error:
            return 0
    return dict(
        gpu_ns=q("select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL"),
        kernels=q("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL"),
        api=q("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME"),
    )


def main():
    root = pathlib.Path(sys.argv[1])
    nsys = root / "nsys"
    modes, batches = set(), set()
    for log in sorted(nsys.glob("*_gen*.log")):
        d = parse_log(log)
        if d:
            modes.add(d["MODE"])
            batches.add(int(d["bs"]))
    gen = 0
    for log in sorted(nsys.glob("*_gen*.log")):
        d = parse_log(log)
        if d:
            gen = max(gen, int(d["gen"]))

    print(f"decode 差分窗口：gen={gen}（B 窗口 = prefill + {gen} 步 decode，"
          f"A 窗口 = 仅 prefill）")
    print(f"\n  {'mode':<22}{'bs':>4}{'wall ms/步':>12}{'cpu ms/步':>11}"
          f"{'gpu ms/步':>11}{'提交数/步':>10}{'kernel/步':>10}"
          f"{'max(cpu,gpu)':>13}  瓶颈侧")
    rows = []
    for mode in sorted(modes):
        for b in sorted(batches):
            a_log = nsys / f"{mode}_bs{b}_gen0.log"
            b_log = nsys / f"{mode}_bs{b}_gen{gen}.log"
            a_db = a_log.with_suffix(".sqlite")
            b_db = b_log.with_suffix(".sqlite")
            if not (a_log.exists() and b_log.exists() and a_db.exists()
                    and b_db.exists()):
                continue
            da, db = parse_log(a_log), parse_log(b_log)
            if not da or not db:
                continue
            A, B = sqlite_totals(a_db), sqlite_totals(b_db)
            gpu_ms = (B["gpu_ns"] - A["gpu_ns"]) / gen / 1e6
            kernels = (B["kernels"] - A["kernels"]) / gen
            api = (B["api"] - A["api"]) / gen
            wall = float(db["WALL_MS"])
            cpu = float(db["CPU_MS"])
            side = "CPU" if cpu > gpu_ms else "GPU"
            rows.append((mode, b, wall, cpu, gpu_ms, api, kernels, side))
            print(f"  {mode:<22}{b:>4}{wall:>12.3f}{cpu:>11.3f}{gpu_ms:>11.3f}"
                  f"{api:>10.1f}{kernels:>10.1f}{max(cpu, gpu_ms):>13.3f}  {side}")

    print("\n  读法：wall 是唯一可以直接引用的性能数字；cpu 是进程 CPU 时间，")
    print("  gpu 是 kernel device 时间，两者都用于归因，不当作性能值。")
    print("  若 wall ≈ max(cpu, gpu)，说明两条流水基本重叠；")
    print("  若 wall ≈ cpu + gpu，说明它们被串行化了。")
    out = root / "attribution_summary.txt"
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"gen={gen} modes={sorted(modes)} batches={sorted(batches)}\n")
        for r in rows:
            fh.write("  ".join(str(x) for x in r) + "\n")
    print(f"\n  汇总写到 {out}")


if __name__ == "__main__":
    main()
