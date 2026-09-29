#!/usr/bin/env python3
"""L1.3 lab · 双缓冲输入流水：把 H2D 与计算重叠起来。

L1.3 测出 H2D 47 GB/s、主存聚合 48 GB/s。那么「读一块、算一块」能不能重叠？
答案是能，但重叠的收益被较慢的那一段封顶——本脚本就是把这件事量化出来。

做法（固定总工作量，只改 chunk 大小）：
    chunk 1 MiB / 4 MiB / 16 MiB / 64 MiB
    每种配置都传同样的总字节数、做同样的总 FLOPs

两种执行方式：
    串行：   copy(chunk) → gemm(chunk) → copy(next) → …
    双缓冲： copy 落在 s_copy 流、gemm 落在 s_comp 流；
             gemm 等 copy_done[p]；下一次 copy 等 comp_done[p]（复用同一个槽位）

三个要交付的证据：
    1. 每段的 per-chunk 时间线（同一时钟基准的 event 时刻），直接看出重叠区间
    2. 加速比与「较慢一段」的理论上限对照
    3. **反例**：故意不等 comp_done[p] 就覆写槽位（--show-race），
       输出与串行结果不一致——这就是「复用 buffer 前必须等正确事件」的证明

用法：
    python xfer_pipeline.py --gpu 0 --total-mib 1024 --out pipeline.json
    python xfer_pipeline.py --gpu 0 --total-mib 256 --show-race
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def build_inputs(chunk_bytes: int, k: int, device: str):
    """按 chunk 大小造一块可做 GEMM 的数据：rows x k 的 fp16。"""
    rows = max(8, chunk_bytes // (2 * k))
    rows -= rows % 8
    return rows


def run_serial(chunks_host: list[torch.Tensor], b: torch.Tensor, gpu: int,
               reps: int = 1) -> dict:
    """串行：copy(chunk) → gemm(chunk)（重复 reps 次），全在默认流上，两段不重叠。"""
    dev = f"cuda:{gpu}"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = None
    for host in chunks_host:
        d = host.to(dev, non_blocking=True)
        for _ in range(reps):
            c = (d @ b).float()      # fp32 累加，避免 fp16 溢出成 inf
            total = c if total is None else total + c
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"wall_ms": wall * 1e3, "checksum": float(total.sum().item())}


def run_pipeline(chunks_host: list[torch.Tensor], b: torch.Tensor, gpu: int,
                 race: bool = False, reps: int = 1) -> dict:
    """双缓冲：两个槽位，copy 与 gemm 各占一条流。"""
    s_copy = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    nbuf = 2
    dev = f"cuda:{gpu}"
    slots = [torch.empty_like(chunks_host[0], device=dev) for _ in range(nbuf)]
    copy_done = [torch.cuda.Event() for _ in range(nbuf)]
    comp_done = [torch.cuda.Event() for _ in range(nbuf)]
    for e in comp_done:
        e.record()                       # 初始状态：槽位可写

    timeline = []
    origin = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.cuda.stream(s_comp):
        origin.record()

    total = None
    wall0 = time.perf_counter()
    for i, host_chunk in enumerate(chunks_host):
        p = i % nbuf
        with torch.cuda.stream(s_copy):
            if not race:
                s_copy.wait_event(comp_done[p])        # 复用前必须等上一次计算读完
            c0 = torch.cuda.Event(enable_timing=True)
            c0.record(s_copy)
            slots[p].copy_(host_chunk, non_blocking=True)
            c1 = torch.cuda.Event(enable_timing=True)
            c1.record(s_copy)
            copy_done[p].record(s_copy)
        with torch.cuda.stream(s_comp):
            s_comp.wait_event(copy_done[p])
            g0 = torch.cuda.Event(enable_timing=True)
            g0.record(s_comp)
            c = slots[p] @ b
            # 用 fp32 累加：1 GiB 的 GEMM 结果量级超过 fp16 上限，
            # 直接在 fp16 里相加会溢出成 inf（第一版就是这么翻车的，
            # 检查和的「不一致」全都成了 inf vs inf，看不出对错）。
            c = c.float()
            for _ in range(reps - 1):
                c = c + (slots[p] @ b).float()
            total = c if total is None else total + c
            g1 = torch.cuda.Event(enable_timing=True)
            g1.record(s_comp)
            comp_done[p].record(s_comp)
        timeline.append({"chunk": i, "slot": p,
                         "copy": (c0, c1), "gemm": (g0, g1)})
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    rows = []
    for t in timeline:
        c0, c1 = t["copy"]
        g0, g1 = t["gemm"]
        rows.append({
            "chunk": t["chunk"], "slot": t["slot"],
            "copy_start_ms": round(origin.elapsed_time(c0), 4),
            "copy_end_ms": round(origin.elapsed_time(c1), 4),
            "gemm_start_ms": round(origin.elapsed_time(g0), 4),
            "gemm_end_ms": round(origin.elapsed_time(g1), 4),
        })
    copy_ms = sum(r["copy_end_ms"] - r["copy_start_ms"] for r in rows)
    gemm_ms = sum(r["gemm_end_ms"] - r["gemm_start_ms"] for r in rows)
    # 真正的流水重叠发生在「本块的 GEMM」与「下一块的 copy」之间——
    # 比较同一块的 copy 与 gemm 只会得到 0（第一版就是这么算错的）。
    overlap_next = 0.0
    for i in range(len(rows) - 1):
        lo = max(rows[i]["gemm_start_ms"], rows[i + 1]["copy_start_ms"])
        hi = min(rows[i]["gemm_end_ms"], rows[i + 1]["copy_end_ms"])
        overlap_next += max(0.0, hi - lo)
    overlap_same = 0.0
    for r in rows:
        lo = max(r["copy_start_ms"], r["gemm_start_ms"])
        hi = min(r["copy_end_ms"], r["gemm_end_ms"])
        overlap_same += max(0.0, hi - lo)
    span = max(r["gemm_end_ms"] for r in rows) - min(r["copy_start_ms"] for r in rows)
    return {"wall_ms": wall * 1e3, "checksum": float(total.sum().item()),
            "sum_copy_ms": round(copy_ms, 3), "sum_gemm_ms": round(gemm_ms, 3),
            "overlap_ms": round(overlap_next, 3),
            "overlap_next_ms": round(overlap_next, 3),
            "overlap_same_chunk_ms": round(overlap_same, 3),
            "timeline_span_ms": round(span, 3),
            "timeline": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--total-mib", type=int, default=1024)
    ap.add_argument("--chunks-mib", default="1,4,16,64")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=1,
                    help="每个 chunk 重复做几次 GEMM；>1 时计算成为较慢的一段，"
                         "这时才能暴露「复用槽位前不等计算完成」的竞态")
    ap.add_argument("--show-race", action="store_true",
                    help="额外跑一次「不复用前等待」的错误版本，验证它会算错")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    p = torch.cuda.get_device_properties(args.gpu)
    k = args.k
    total_bytes = args.total_mib * 1024 * 1024
    dev = f"cuda:{args.gpu}"
    b = torch.randn(k, k, dtype=torch.float16, device=dev)

    print(f"=== {p.name}   双缓冲 copy+GEMM 流水（总工作量固定 {args.total_mib} MiB）")
    print(f"    每个 chunk 做 (rows×{k})@({k}×{k}) 的 fp16 GEMM，"
          f"rows = chunk字节/(2k)，因此总 FLOPs 与 chunk 大小无关\n")
    print(f"    {'chunk':>8} {'块数':>6} {'串行ms':>9} {'流水ms':>9} {'加速':>7} "
          f"{'Σcopy':>8} {'Σgemm':>8} {'重叠ms':>8} {'一致':>5}")

    res = {"measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "gpu": p.name, "total_mib": args.total_mib, "k": k, "runs": []}
    for chunk_mib in [int(x) for x in args.chunks_mib.split(",")]:
        chunk_bytes = chunk_mib * 1024 * 1024
        nchunks = max(2, total_bytes // chunk_bytes)
        rows = build_inputs(chunk_bytes, k, dev)
        host = [torch.randn(rows, k, dtype=torch.float16, pin_memory=True)
                for _ in range(nchunks)]

        ser = run_serial(host, b, args.gpu, reps=args.reps)
        pipe = run_pipeline(host, b, args.gpu, reps=args.reps)
        tol = 1e-5 * max(1.0, abs(ser["checksum"]))
        same = abs(ser["checksum"] - pipe["checksum"]) <= tol
        speed = ser["wall_ms"] / pipe["wall_ms"]
        print(f"    {chunk_mib:>6}MiB {nchunks:>6} {ser['wall_ms']:>9.1f} "
              f"{pipe['wall_ms']:>9.1f} {speed:>6.2f}× {pipe['sum_copy_ms']:>8.1f} "
              f"{pipe['sum_gemm_ms']:>8.1f} {pipe['overlap_ms']:>8.1f} {str(same):>5}")

        entry = {"chunk_mib": chunk_mib, "nchunks": nchunks, "rows": rows,
                 "reps": args.reps,
                 "serial_ms": round(ser["wall_ms"], 2), "pipeline_ms": round(pipe["wall_ms"], 2),
                 "speedup": round(speed, 2), "checksum_equal": bool(same),
                 "sum_copy_ms": pipe["sum_copy_ms"], "sum_gemm_ms": pipe["sum_gemm_ms"],
                 "overlap_ms": pipe["overlap_ms"],
                 "stage_bound": {
                     "copy_bound_speedup": round(ser["wall_ms"] /
                                                 max(pipe["sum_copy_ms"], 1e-6), 2),
                     "gemm_bound_speedup": round(ser["wall_ms"] /
                                                 max(pipe["sum_gemm_ms"], 1e-6), 2)},
                 "timeline_head": pipe["timeline"][:6],
                 "timeline_tail": pipe["timeline"][-2:]}
        res["runs"].append(entry)

        if args.show_race:
            bad = run_pipeline(host, b, args.gpu, race=True, reps=args.reps)
            maxdiff = abs(ser["checksum"] - bad["checksum"])
            ok = maxdiff <= tol
            print(f"        [反例] 复用槽位前不等 comp_done：checksum {bad['checksum']:.3f} "
                  f"vs 串行 {ser['checksum']:.3f}  差 {maxdiff:.3f}  一致={ok}")
            entry["race"] = {"checksum": bad["checksum"], "diff": maxdiff,
                             "matches_serial": bool(ok),
                             "pipeline_ms": round(bad["wall_ms"], 2)}
        del host

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
