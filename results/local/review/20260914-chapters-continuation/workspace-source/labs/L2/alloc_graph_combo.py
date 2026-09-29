#!/usr/bin/env python3
"""L2.0c 补测 · `expandable_segments` × CUDA Graph 的 2×2 组合。

2.0c 前面把两件事分开测过：变长轨迹下 native 与 `expandable_segments` 的差别、
以及 CUDA Graph 自己的内存池。两者都改分配路径，合在一起会怎样是本章的最后一个缺口。

四种配置各自起一个进程（分配器配置必须在 CUDA 初始化之前生效）：

    native      + 无图
    native      + 有图
    expandable  + 无图
    expandable  + 有图

每个进程里按两种顺序各跑一遍，看"谁先占用分配器"是否影响结论：

    trace-first  先跑 200 请求的变长轨迹，再捕获并 replay 固定形状的图
    graph-first  先建图池，再跑同一份变长轨迹

记录：峰值 active/reserved、段数与 `inactive_split`、图池的 reserved 增量、
捕获是否成功（失败保留原文）、以及图释放后池有没有归还。

    python labs/L2/alloc_graph_combo.py --out-dir <dir>          # 编排四个子进程
    python labs/L2/alloc_graph_combo.py --worker --conf expandable --graph 1 \
        --order trace-first --out <json>                          # 单个配置（内部用）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load_alloc_trace():
    spec = importlib.util.spec_from_file_location("alloc_trace", HERE / "alloc_trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snap():
    import torch
    segs = torch.cuda.memory_snapshot()
    states: dict = {}
    for s in segs:
        for b in s["blocks"]:
            states[b["state"]] = states.get(b["state"], 0) + b["size"]
    return {"segments": len(segs),
            "segment_total_mb": sum(s["total_size"] for s in segs) / 2**20,
            "inactive_split_mb": states.get("inactive_split", 0) / 2**20,
            "active_split_mb": states.get("active_split", 0) / 2**20}


def graph_phase(step_reps=200):
    """固定形状：不捕获 vs 捕获一次 replay。返回两套数与池增量。"""
    import torch

    def step(t):
        a = t @ t
        b = a * 2.0
        return b.sum()

    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(step_reps):
        step(x)
    torch.cuda.synchronize()
    eager = {"peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
             "end_reserved_mb": torch.cuda.memory_reserved() / 2**20}

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = {"reserved_mb": torch.cuda.memory_reserved() / 2**20}
    torch.cuda.reset_peak_memory_stats()
    result = {"capture_ok": False, "error": None}
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step(x)
        torch.cuda.current_stream().wait_stream(s)

        static_in = x.clone()
        static_out = torch.zeros((), device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out.copy_(step(static_in))
        pg = {"peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
              "end_reserved_mb": torch.cuda.memory_reserved() / 2**20}
        for _ in range(step_reps):
            g.replay()
        torch.cuda.synchronize()
        after_replay = {"peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                        "end_reserved_mb": torch.cuda.memory_reserved() / 2**20}
        # 图释放后池是否归还
        del g, static_in, static_out
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_del = {"reserved_mb": torch.cuda.memory_reserved() / 2**20,
                     "allocated_mb": torch.cuda.memory_allocated() / 2**20}
        result.update({"capture_ok": True, "graph_build": pg, "after_replay": after_replay,
                       "after_del": after_del,
                       "pool_delta_mb": after_replay["end_reserved_mb"] - before["reserved_mb"]})
    except Exception as exc:                                   # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"[:400]
        try:
            torch.cuda.synchronize()
        except Exception:                                      # noqa: BLE001
            pass
    result["eager"] = eager
    result["before_capture"] = before
    return result


def worker(conf: str, use_graph: bool, order: str, out_path: str) -> int:
    import torch

    at = load_alloc_trace()
    out: dict = {"conf": conf, "graph": use_graph, "order": order,
                 "torch": torch.__version__,
                 "device": torch.cuda.get_device_name(0),
                 "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"),
                 "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}
    tmp = pathlib.Path(out_path).with_suffix(".trace.json")

    if use_graph and order == "graph-first":
        out["graph_phase"] = graph_phase()
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    out["trace_phase"] = at.run_trace(
        f"{conf}{'+graph' if use_graph else ''}-{order}", str(tmp), order="interleaved")
    if use_graph and order == "trace-first":
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        out["graph_phase"] = graph_phase()
    out["final_snapshot"] = snap()
    out["stats"] = {k: v for k, v in torch.cuda.memory_stats().items()
                    if k in ("num_alloc_retries", "num_ooms", "segment.all.allocated",
                             "segment.all.freed", "reserved_bytes.all.peak",
                             "active_bytes.all.peak")}
    pathlib.Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    if tmp.exists():
        tmp.unlink()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--conf", default="default")
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--order", default="trace-first")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.worker:
        return worker(args.conf, bool(args.graph), args.order, args.out)

    out_dir = pathlib.Path(args.out_dir or "out/2.0c/graph-combo")
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [("default", 0), ("default", 1), ("expandable", 1), ("expandable", 0)]
    rows = []
    for conf, g in configs:
        for order in ("trace-first", "graph-first"):
            tag = f"{conf}{'_graph' if g else ''}_{order.replace('-', '')}"
            out = out_dir / f"{tag}.json"
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True" if conf == "expandable" else "expandable_segments:False")
            cmd = [sys.executable, str(HERE / "alloc_graph_combo.py"), "--worker",
                   "--conf", conf, "--graph", str(g), "--order", order, "--out", str(out)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            log = out_dir / f"{tag}.log"
            log.write_text(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                print(f"    {tag:34s} [失败 exit={proc.returncode}] "
                      f"{(proc.stderr.strip().splitlines() or ['?'])[-1][:90]}")
                continue
            rows.append(json.loads(out.read_text()))

    print(f"=== {rows[0]['device'] if rows else '?'} · torch "
          f"{rows[0]['torch'] if rows else '?'} ===")
    print(f"    {'配置':30s} {'峰值active':>9s} {'峰值reserved':>12s} {'段数':>5s} "
          f"{'inactive_split':>14s} {'图池增量':>9s} {'捕获':>6s}")
    for r in rows:
        t = r.get("trace_phase", {})
        g = r.get("graph_phase") or {}
        pool = g.get("pool_delta_mb")
        print(f"    {r['conf'] + ('+graph' if r['graph'] else '') + '/' + r['order']:30s} "
              f"{t.get('peak_active_mb', float('nan')):9.2f} "
              f"{t.get('peak_reserved_mb', float('nan')):12.2f} "
              f"{t.get('segments', -1):5d} "
              f"{t.get('peak_inactive_split_mb', float('nan')):14.3f} "
              f"{(f'{pool:.2f}' if isinstance(pool, (int, float)) else '-'):>9s} "
              f"{('OK' if g.get('capture_ok') else ('FAIL' if g else '-')):>6s}")
        if g and g.get("error"):
            print(f"        捕获原文：{g['error'][:150]}")

    (out_dir / "combo.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out_dir / 'combo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
