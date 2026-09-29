#!/usr/bin/env python3
"""L5.1 · nsys 侧对逐 step kernel 的独立核对（含时钟基准诊断）。

`step_kernel_align.py --no-profiler` 在每个引擎 step 上 push 一个 NVTX range，
nsys 用 `--cuda-graph-trace=node` 采完后，本脚本先尝试按区间把 kernel 归到 step
（NVTX → CUPTI_ACTIVITY_KIND_RUNTIME → kernel 的 `correlationId`），再给出不依赖
NVTX 的图结构统计。本机实测的两条边界写在输出里：

  1. **两套时间戳基准不同。** `NVTX_EVENTS` 与 `CUPTI_ACTIVITY_KIND_KERNEL`/`RUNTIME`
     在本机导出的 sqlite 里相差约 1.6 s，按区间比较会归到 0 个；`nsys stats` 的
     `nvtx_kern_sum` 与 `nvtx_gpu_proj_sum` 同样没有把 `step*` range 投影到 kernel；
     `--capture-range=nvtx --nvtx-capture=stepK` 在本机采到 0 个事件。因此**逐 step
     归属由 torch profiler 口径承担**（`step_kernel_align.py`），nsys 在这里提供
     图内 kernel 结构与数量的独立核对。

  2. 图内/图外由 `graphId` 区分：`graphId > 0` 是 CUDA Graph 重放执行的节点，
     `graphId IS NULL` 是单独 launch 的 kernel。图内节点数可与 profiler 报出的
     每步 kernel 数交叉核对。

用法：
    python summarize_step_nsys.py <report.sqlite> [label]
"""

from __future__ import annotations

import sqlite3
import sys


def name_of(c, key):
    try:
        row = c.execute("select value from StringIds where id = ?", (key,)).fetchone()
        return row[0] if row else f"<id {key}>"
    except sqlite3.Error:
        return f"<id {key}>"


def table_exists(c, name) -> bool:
    return bool(c.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone())


def span(c, table, where=""):
    try:
        return c.execute(f"select min(start), max(end), count(*) from {table} {where}"
                         ).fetchone()
    except sqlite3.Error:
        return None


def main() -> None:
    db = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else db
    c = sqlite3.connect(db)
    print(f"\n=== {label} ===")

    ranges = list(c.execute("""
        select text, start, end from NVTX_EVENTS
        where text is not null and text like 'step%' order by start"""))
    print(f"  NVTX step range：{len(ranges)} 个")

    if not table_exists(c, "CUPTI_ACTIVITY_KIND_KERNEL"):
        names = [r[0] for r in c.execute(
            "select name from sqlite_master where type='table' order by name")]
        print("  报告里没有 CUPTI_ACTIVITY_KIND_KERNEL：本次 nsys 未采到 CUDA kernel 活动")
        print(f"  现有表：{names}")
        return

    nv = span(c, "NVTX_EVENTS", "where text like 'step%'")
    kn = span(c, "CUPTI_ACTIVITY_KIND_KERNEL")
    rt = span(c, "CUPTI_ACTIVITY_KIND_RUNTIME") if table_exists(
        c, "CUPTI_ACTIVITY_KIND_RUNTIME") else None
    if nv and kn:
        print(f"  时钟诊断：NVTX {nv[0]}–{nv[1]} ns；kernel {kn[0]}–{kn[1]} ns"
              f"（末端相差 {(nv[1] - kn[1]) / 1e9:.3f} s）")
    if rt:
        print(f"            runtime {rt[0]}–{rt[1]} ns、{rt[2]:,} 次调用"
              f"（末端相差 {(nv[1] - rt[1]) / 1e9:.3f} s）")

    covered = 0
    for text, start, end in ranges:
        n_rt = c.execute("""
            select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME
            where start >= ? and end <= ?""", (start, end)).fetchone()[0] if rt else 0
        rows = list(c.execute("""
            select s.value, count(*), sum(k.end - k.start)
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId = r.correlationId
            join StringIds s on s.id = k.shortName
            where r.start >= ? and r.end <= ?
            group by s.value order by 3 desc""", (start, end))) if rt else []
        n = sum(r[1] for r in rows)
        covered += n
        if n:
            print(f"\n  {text}: {n} 个 kernel、runtime {n_rt} 次")
            for key, cnt, ns in rows[:6]:
                print(f"      {(ns or 0) / 1e6:>8.3f} ms ×{cnt:<5} {name_of(c, key)[:100]}")
    if covered == 0:
        print("  step range 内 runtime 调用 0 次 ⇒ 两套时钟基准下无法按区间归属，"
              "改用图结构口径")

    total = kn[2]
    ingraph = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL "
                        "where graphId > 0").fetchone()[0]
    gpu_ns = c.execute("select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL"
                       ).fetchone()[0] or 0
    print(f"\n  kernel 合计 {total:,}（图内 {ingraph:,}、图外 {total - ingraph:,}）、"
          f"device {gpu_ns / 1e6:.2f} ms")
    rows = list(c.execute("""
        select graphId, count(*) execs, count(distinct graphNodeId) nodes
        from CUPTI_ACTIVITY_KIND_KERNEL where graphId > 0
        group by graphId order by execs desc"""))
    if rows:
        nodes = sorted((r[2] for r in rows), reverse=True)
        print(f"  被重放的图 {len(rows)} 张；每张图节点数 最大 {nodes[0]}、"
              f"中位 {nodes[len(nodes) // 2]}、最小 {nodes[-1]}")
        for gid, execs, nd in rows[:5]:
            names = [name_of(c, k) for (k,) in c.execute("""
                select shortName from CUPTI_ACTIVITY_KIND_KERNEL
                where graphId = ? group by shortName order by count(*) desc limit 3""",
                (gid,))]
            print(f"      graphId={gid:<6} 节点={nd:<5} 重放={execs / max(nd, 1):>5.1f}×  "
                  f"代表: {', '.join(str(x)[:40] for x in names)}")
    if total - ingraph:
        print("  图外 kernel top：")
        for key, cnt in c.execute("""
                select shortName, count(*) from CUPTI_ACTIVITY_KIND_KERNEL
                where graphId is null group by shortName order by 2 desc limit 8"""):
            print(f"      {name_of(c, key)[:100]:<100} {cnt:>6,}")


if __name__ == "__main__":
    main()
