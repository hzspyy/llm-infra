#!/usr/bin/env python3
"""L5.4 任务 A（图内 kernel 普查）—— 从 nsys sqlite 里数图内/图外 kernel。

`--cuda-graph-trace=node` 让 nsys 把 CUDA Graph 展开成节点，于是
`CUPTI_ACTIVITY_KIND_KERNEL` 每行带 `graphId` / `graphNodeId`：

  - graphId > 0    : 这个 kernel 是某张图重放时执行的（图内）
  - graphId IS NULL: 这个 kernel 是单独 launch 的（图外）

这正好回答计划里的问题：「区分分段编译与 CUDA Graph」。
COMPILE_ONLY（只编译不捕获）与 NONE 都没有图；PIECEWISE 得到很多张小图，
切分算子留在图外；FULL_AND_PIECEWISE 每步一张大图，整步（含 attention）在图内。

注意 nsys 2026.3 里 `shortName`/`demangledName` 是 StringIds 的外键而非文本，
要 join `StringIds` 才是名字；某个 kernel 的 `graphId` 在图外时是 NULL 而不是 0。

用法：
    python summarize_graph_nsys.py <report.sqlite> [label]
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter


def q1(c, sql, *a):
    try:
        return c.execute(sql, a).fetchone()[0]
    except sqlite3.Error:
        return None


def name_of(c, key):
    try:
        row = c.execute("select value from StringIds where id = ?", (key,)).fetchone()
        return row[0] if row else f"<id {key}>"
    except sqlite3.Error:
        return f"<id {key}>"


def main():
    db = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else db
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else None
    c = sqlite3.connect(db)
    print(f"\n=== {label} ===")

    # ---- CPU 侧提交 API ----
    api = Counter()
    cpu_api_ns = {}
    for name, n, tot in c.execute("""
            select s.value, count(*), sum(r.end - r.start)
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on s.id = r.nameId
            group by s.value"""):
        api[name] = n
        cpu_api_ns[name] = tot or 0
    launch = sum(v for k, v in api.items() if "LaunchKernel" in k)
    graph = sum(v for k, v in api.items() if "GraphLaunch" in k)
    print(f"  CPU 侧提交 : LaunchKernel={launch:,}  GraphLaunch={graph:,}"
          f"  合计={launch + graph:,}")
    for k, v in sorted(api.items(), key=lambda kv: -kv[1])[:8]:
        if "aunch" in k or "Graph" in k:
            print(f"      {k:<44} {v:>8,}   CPU {cpu_api_ns[k] / 1e6:>8.3f} ms")
    submit_ns = sum(v for k, v in cpu_api_ns.items()
                    if "LaunchKernel" in k or "GraphLaunch" in k)

    # ---- GPU 侧 kernel：图内 vs 图外 ----
    tot = q1(c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL") or 0
    ingraph = q1(c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL "
                    "where graphId > 0") or 0
    outgraph = tot - ingraph
    gpu_ns = q1(c, "select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL") or 0
    print(f"  GPU kernel : 合计={tot:,}  图内={ingraph:,}  图外={outgraph:,}"
          f"   图内占比={ingraph / max(tot, 1):.1%}")
    print(f"  GPU 忙     : {gpu_ns / 1e6:.2f} ms（仅 profiler 窗口内）")
    if steps:
        print(f"  折算到每步 : 共 {steps} 次 forward；"
              f"CPU 提交 {submit_ns / steps / 1e6:.3f} ms/步  "
              f"GPU 忙 {gpu_ns / steps / 1e6:.3f} ms/步  "
              f"比值 {submit_ns / max(gpu_ns, 1):.3f}")

    # ---- 每张图：真实节点数（distinct graphNodeId）与重放次数 ----
    rows = list(c.execute("""
        select graphId, count(*) as execs, count(distinct graphNodeId) as nodes
        from CUPTI_ACTIVITY_KIND_KERNEL where graphId > 0
        group by graphId order by execs desc"""))
    print(f"  被重放的图 : {len(rows)} 张（按 graphId 去重）")
    if rows:
        nodes_sorted = sorted((r[2] for r in rows), reverse=True)
        print(f"  每张图的 kernel 节点数: 最大 {nodes_sorted[0]}  "
              f"中位 {nodes_sorted[len(nodes_sorted) // 2]}  最小 {nodes_sorted[-1]}")
        print("  最重的 5 张图（节点数=图的真实规模，重放次数=执行数/节点数）：")
        for gid, execs, nodes in rows[:5]:
            names = [name_of(c, k) for (k,) in c.execute("""
                select shortName from CUPTI_ACTIVITY_KIND_KERNEL
                where graphId = ? group by shortName
                order by count(*) desc limit 3""", (gid,))]
            print(f"      graphId={gid:<6} 节点={nodes:<5} 重放={execs / nodes:>5.1f}×  "
                  f"代表: {', '.join(str(n)[:34] for n in names)}")

    # ---- 图外 kernel 的名称（PIECEWISE 下应当是切分算子）----
    if outgraph:
        print(f"  图外 {outgraph:,} 个 kernel 的 top 名称：")
        for key, n in c.execute("""
                select shortName, count(*) from CUPTI_ACTIVITY_KIND_KERNEL
                where graphId is null group by shortName
                order by 2 desc limit 10"""):
            print(f"      {str(name_of(c, key))[:52]:<52} {n:>6,}")

    # ---- memcpy 同样分图内/图外 ----
    mc = q1(c, "select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY") or 0
    mc_in = q1(c, "select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY "
                  "where graphId > 0") or 0
    print(f"  memcpy     : 合计={mc:,}  图内={mc_in:,}  图外={mc - mc_in:,}")


if __name__ == "__main__":
    main()
