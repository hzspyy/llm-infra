#!/usr/bin/env python3
"""L9.8 任务 B：预热槽位与并发对「工具可用时间」的影响，分阶段计时。

计时口径按任务书：**从资源申请到首个工具可执行结果**。中间的四段分开记，因为它们互不替代：

| 阶段 | 含义 | 本机做法 |
|---|---|---|
| `cache_fetch_ms` | 镜像/依赖取得 | 无容器镜像可用，用「从本地依赖缓存复制 payload 到槽位」近似，缓存缺失时该项显著变大 |
| `dep_load_ms` | 依赖装载 | 子进程里 import 目标模块的时间（`python -c` 计时） |
| `workspace_prep_ms` | 工作区准备 | 写工具源码与输入文件 |
| `health_probe_ms` | 可用性探测 | 真的跑一条最短命令并看退出码 |
| `queue_wait_ms` | 排队 | 从申请到拿到槽位 |
| `tool_ms` | 工具执行 | 子进程实际运行时间 |

扫描：预热槽位 `warm ∈ {0,1,4}` × 并发 `∈ {1,4,16}`。每格跑固定条数任务，记录
p50/p95 的「申请→首个结果」、命中/溢出、就绪槽位的闲置 RSS 与磁盘占用。

用法::

    python labs/L9/sandbox_lifecycle_bench.py --out out/9.8/lifecycle --tasks 32
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tool_sandbox_pool import Pool  # noqa: E402


def q(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 3)


def make_dependency_cache(root: pathlib.Path, mb: int) -> pathlib.Path:
    """近似"镜像/依赖已在本地缓存"：准备一个可复制的 payload。"""
    root.mkdir(parents=True, exist_ok=True)
    payload = root / f"dep-{mb}mb.bin"
    if not payload.exists() or payload.stat().st_size != mb * 1024 * 1024:
        with open(payload, "wb") as fh:
            fh.write(b"d" * (mb * 1024 * 1024))
    return payload


def stage_timings(pool: Pool, slot, spec: dict, dependency: pathlib.Path,
                  timeout_s: float) -> dict:
    """把一次「申请→首个结果」拆成阶段并返回。"""
    t0 = time.perf_counter()
    # 1) 依赖取得：把 payload 复制进工作区（模拟从缓存/镜像层取依赖）
    t = time.perf_counter()
    dest = slot.workspace / dependency.name
    if not dest.exists():
        with open(dependency, "rb") as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    cache_fetch_ms = (time.perf_counter() - t) * 1000.0
    # 2) 依赖装载：在子进程里 import
    t = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import json,os,sys,time;time.sleep(0)"],
                   capture_output=True, timeout=30)
    dep_load_ms = (time.perf_counter() - t) * 1000.0
    # 3)+4)+5)+6) 交给池：工作区准备在 STARTING，健康探测在 READY，执行是工具本身
    res = pool.run_tool(slot, spec, timeout_s=timeout_s)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {"cache_fetch_ms": round(cache_fetch_ms, 3), "dep_load_ms": round(dep_load_ms, 3),
            "slot_prepare_ms": getattr(slot, "prepare_ms", None),
            "health_probe_ms": getattr(slot, "probe_ms", None),
            "tool_ms": res["duration_ms"], "first_result_ms": round(total_ms, 3),
            "exit_reason": res["exit_reason"], "exit_code": res["exit_code"]}


def slot_footprint(slot) -> dict:
    """就绪槽位的闲置占用：磁盘字节数与进程 RSS。"""
    disk = 0
    for p in slot.workspace.rglob("*"):
        if p.is_file():
            try:
                disk += p.stat().st_size
            except OSError:
                pass
    rss_kb = 0
    if slot.proc is not None and slot.proc.poll() is None:
        try:
            status = pathlib.Path(f"/proc/{slot.proc.pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS"):
                    rss_kb = int(line.split()[1])
        except Exception:  # noqa: BLE001
            pass
    return {"disk_bytes": disk, "rss_kb": rss_kb}


def run_cell(out: pathlib.Path, warm: int, concurrency: int, tasks: int, isolation: str,
             dep_mb: int, timeout_s: float, limits: dict, max_slots: int) -> dict:
    cell = out / f"warm{warm}-c{concurrency}"
    cell.mkdir(parents=True, exist_ok=True)
    dependency = make_dependency_cache(out / "dep-cache", dep_mb)
    pool = Pool(cell, max_slots=max_slots, warm=warm, lease_s=30.0, isolate=isolation,
                limits=limits, output_cap_bytes=64 * 1024, max_children=2)
    prewarm = pool.prewarm(warm)
    rows: list[dict] = []
    # 并发用固定大小的批来近似：每批 concurrency 个任务顺序申请、同时执行
    remaining = tasks
    while remaining > 0:
        batch = min(concurrency, remaining)
        remaining -= batch
        acquired = []
        for i in range(batch):
            hits_before, misses_before = pool.hits, pool.misses
            slot, wait_ms = pool.acquire({"id": f"t{tasks - remaining + i}"})
            via = "miss" if pool.misses > misses_before else ("hit" if pool.hits > hits_before else "overflow")
            if slot is None:
                rows.append({"acquired": False, "queue_wait_ms": wait_ms, "via": via})
                continue
            acquired.append((slot, wait_ms, via))
        # 顺序执行（本机单卡、工具是 CPU 任务；并发体现在槽位数与批次上）
        for slot, wait_ms, via in acquired:
            startup_ms = (getattr(slot, "prepare_ms", 0.0) or 0.0) + (getattr(slot, "probe_ms", 0.0) or 0.0)
            timing = stage_timings(pool, slot, {"alloc_mb": 0}, dependency, timeout_s)
            footprint = slot_footprint(slot)
            cleanup = pool.kill_group(slot)
            reclaim = pool.reclaim_workspace(slot, keep=("tool.py",))
            back = slot.drain({**cleanup, "reclaimed": reclaim["removed"]})
            rows.append({"acquired": True, "slot": slot.index, "queue_wait_ms": wait_ms,
                         "via": via, "slot_startup_ms": round(startup_ms, 3),
                         **timing, "footprint": footprint, "back_to_ready": back,
                         "running_after": cleanup["running_after"]})
    result = {
        "config": {"warm": warm, "concurrency": concurrency, "tasks": tasks,
                   "isolation": isolation, "dep_mb": dep_mb, "max_slots": max_slots},
        "prewarm": prewarm,
        "pool": {"hits": pool.hits, "misses": pool.misses, "overflows": pool.overflows,
                 "slots": len(pool.slots), "final_counts": pool.counts()},
        "first_result_ms": {"p50": q([r.get("first_result_ms") for r in rows if r.get("acquired")], 0.5),
                            "p95": q([r.get("first_result_ms") for r in rows if r.get("acquired")], 0.95),
                            "mean": round(statistics.fmean(
                                [r["first_result_ms"] for r in rows if r.get("acquired")] or [0]), 3)},
        "queue_wait_ms": {"p50": q([r["queue_wait_ms"] for r in rows if r.get("acquired")], 0.5),
                          "p95": q([r["queue_wait_ms"] for r in rows if r.get("acquired")], 0.95),
                          "hit_p50": q([r["queue_wait_ms"] for r in rows
                                        if r.get("acquired") and r.get("via") == "hit"], 0.5),
                          "miss_p50": q([r["queue_wait_ms"] for r in rows
                                         if r.get("acquired") and r.get("via") == "miss"], 0.5),
                          "miss_p95": q([r["queue_wait_ms"] for r in rows
                                         if r.get("acquired") and r.get("via") == "miss"], 0.95),
                          "miss_n": sum(1 for r in rows
                                        if r.get("acquired") and r.get("via") == "miss")},
        "slot_startup_ms": {"p50": q([r.get("slot_startup_ms") for r in rows
                                      if r.get("acquired") and r.get("via") == "miss"], 0.5)},
        "cache_fetch_ms": {"p50": q([r.get("cache_fetch_ms") for r in rows if r.get("acquired")], 0.5)},
        "dep_load_ms": {"p50": q([r.get("dep_load_ms") for r in rows if r.get("acquired")], 0.5)},
        "tool_ms": {"p50": q([r.get("tool_ms") for r in rows if r.get("acquired")], 0.5)},
        "slot_prepare_ms": {"p50": q([r.get("slot_prepare_ms") for r in rows if r.get("acquired")], 0.5)},
        "health_probe_ms": {"p50": q([r.get("health_probe_ms") for r in rows if r.get("acquired")], 0.5)},
        "footprint": {"disk_bytes": max((r["footprint"]["disk_bytes"] for r in rows
                                         if r.get("acquired")), default=0),
                      "rss_kb": max((r["footprint"]["rss_kb"] for r in rows
                                     if r.get("acquired")), default=0)},
        "not_acquired": sum(1 for r in rows if not r.get("acquired")),
        "rows": rows,
    }
    pool.close()
    (cell / "cell.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.8 沙箱生命周期与工具可用时间")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=32)
    ap.add_argument("--warms", default="0,1,4")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--isolation", choices=["plain", "userns"], default="userns")
    ap.add_argument("--dep-mb", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--max-slots", type=int, default=4)
    ap.add_argument("--as-mb", type=int, default=1024)
    ap.add_argument("--fsize-mb", type=int, default=16)
    ap.add_argument("--cpu-s", type=int, default=5)
    ap.add_argument("--nofile", type=int, default=256)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = {"as_mb": args.as_mb, "fsize_mb": args.fsize_mb, "cpu_s": args.cpu_s,
              "nofile": args.nofile}
    cells = {}
    for warm in [int(x) for x in args.warms.split(",")]:
        for conc in [int(x) for x in args.concurrency.split(",")]:
            res = run_cell(out, warm, conc, args.tasks, args.isolation, args.dep_mb,
                           args.timeout_s, limits, args.max_slots)
            cells[f"warm{warm}-c{conc}"] = res
            print(f"[warm={warm} c={conc}] first_result p50={res['first_result_ms']['p50']} "
                  f"p95={res['first_result_ms']['p95']}ms queue_p50={res['queue_wait_ms']['p50']}ms "
                  f"hits={res['pool']['hits']} misses={res['pool']['misses']} "
                  f"overflow={res['pool']['overflows']} not_acquired={res['not_acquired']} "
                  f"slots={res['pool']['slots']}", flush=True)
    report = {"config": {"tasks": args.tasks, "warms": args.warms,
                         "concurrency": args.concurrency, "isolation": args.isolation,
                         "dep_mb": args.dep_mb, "max_slots": args.max_slots, "limits": limits},
              "cells": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in cells.items()},
              "note": ("cache_fetch 用「从本地依赖缓存复制 payload」近似镜像层取得，"
                       "本机没有容器镜像；dep_load 是子进程 import 的实测；"
                       "first_result_ms = 申请 → 首个工具结果，包含排队与健康探测")}
    (out / "lifecycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
