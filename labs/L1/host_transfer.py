#!/usr/bin/env python3
"""L1.3 lab · 主机与 GPU 之间那条路：PCIe、pinned 内存、NUMA。

显存带宽 1.6 TB/s，PCIe 只有 ~50 GB/s——差 32 倍。
凡是要过 PCIe 的东西（加载权重、KV 卸载到主机、PD 分离传 KV、
多卡之间没有 NVLink 时的通信）都被这条细管子卡着。

四个实验：
  A. pinned vs pageable：为什么 `pin_memory=True` 不是玄学
  B. 传输大小 vs 带宽：小块传输的固定开销有多大
  C. **NUMA 亲和性**：pinned 内存分配在哪个 NUMA 节点上，差多少
  D. 多流并发：DMA 引擎有几个，能不能重叠

C 需要在双路机器上跑（worldvln 有 2 个 NUMA 节点、5 张卡分属两侧）。

用法：
    python host_transfer.py --gpu 0 --out results/xfer_gpu0.json
    # NUMA 对照：
    numactl --cpunodebind=0 --membind=0 python host_transfer.py --gpu 0 ...
    numactl --cpunodebind=1 --membind=1 python host_transfer.py --gpu 0 ...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def time_cuda(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)


def numa_context() -> dict:
    """把当前进程实际的 NUMA / CPU 绑定情况记下来。

    这非常重要：如果你不记录"这次跑在哪个节点上"，
    NUMA 实验的结果就无法解释，也无法复现。
    """
    out = {}
    try:
        out["cpu_affinity"] = sorted(os.sched_getaffinity(0))[:8]
        out["n_cpus_allowed"] = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        pass
    for k in ("NUMA_NODE", "CUDA_VISIBLE_DEVICES"):
        if k in os.environ:
            out[k] = os.environ[k]
    try:
        # /proc/self/numa_maps 太长；用 numastat 看本进程的实际页分布
        r = subprocess.run(["numastat", "-p", str(os.getpid())],
                           capture_output=True, text=True, timeout=10)
        out["numastat"] = r.stdout.strip().splitlines()[-3:]
    except Exception:  # noqa: BLE001
        pass
    return out


def gpu_numa_node(idx: int) -> str:
    """从 sysfs 读这张卡挂在哪个 NUMA 节点上。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(idx)],
            capture_output=True, text=True, timeout=10)
        bus = r.stdout.strip().lower()
        if bus.startswith("00000000:"):
            bus = bus[len("00000000:"):]
        p = Path(f"/sys/bus/pci/devices/0000:{bus}/numa_node")
        if p.exists():
            return p.read_text().strip()
    except Exception:  # noqa: BLE001
        pass
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mb", type=int, default=256)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    p = torch.cuda.get_device_properties(args.gpu)
    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "gpu_index": args.gpu, "gpu": p.name,
        "gpu_numa_node": gpu_numa_node(args.gpu),
        "process": numa_context(),
    }
    print(f"=== GPU{args.gpu} {p.name}  (PCIe 挂在 NUMA node {res['gpu_numa_node']})")
    print(f"    本进程可用 CPU 数 {res['process'].get('n_cpus_allowed')}，"
          f"前几个核 {res['process'].get('cpu_affinity')}")

    n = args.mb * 1024 * 1024
    dev = torch.empty(n, dtype=torch.uint8, device=f"cuda:{args.gpu}")
    pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    paged = torch.empty(n, dtype=torch.uint8)

    def gbs(ms):
        return round(n / (ms * 1e-3) / 1e9, 1)

    # ---- A. pinned vs pageable ----
    print(f"\n[A] {args.mb} MB 单次传输")
    a = {
        "h2d_pinned": gbs(time_cuda(lambda: dev.copy_(pinned, non_blocking=True))),
        "d2h_pinned": gbs(time_cuda(lambda: pinned.copy_(dev, non_blocking=True))),
        "h2d_pageable": gbs(time_cuda(lambda: dev.copy_(paged))),
        "d2h_pageable": gbs(time_cuda(lambda: paged.copy_(dev))),
    }
    for k, v in a.items():
        print(f"    {k:16s} {v:8.1f} GB/s")
    print(f"    pinned 相对 pageable 的加速：H2D {a['h2d_pinned']/a['h2d_pageable']:.2f}×，"
          f"D2H {a['d2h_pinned']/a['d2h_pageable']:.2f}×")
    res["A_pinned_vs_pageable"] = a

    # ---- B. 传输大小 vs 带宽 ----
    print("\n[B] 传输大小 vs 有效带宽（pinned, H2D）")
    print(f"    {'大小':>10} {'GB/s':>9} {'耗时us':>10} {'固定开销占比':>12}")
    rows = []
    small_ms = None
    for kb in (4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144):
        sz = kb * 1024
        if sz > n:
            break
        d = dev[:sz]
        h = pinned[:sz]
        ms = time_cuda(lambda d=d, h=h: d.copy_(h, non_blocking=True), iters=50)
        if small_ms is None:
            small_ms = ms                       # 最小一次的耗时 ≈ 固定开销
        g = sz / (ms * 1e-3) / 1e9
        label = f"{kb} KB" if kb < 1024 else f"{kb//1024} MB"
        print(f"    {label:>10} {g:>9.1f} {ms*1000:>10.1f} {small_ms/ms:>11.0%}")
        rows.append({"kb": kb, "gbps": round(g, 2), "us": round(ms * 1000, 1)})
    res["B_size_sweep"] = rows
    res["B_fixed_overhead_us"] = round(small_ms * 1000, 1)
    print(f"    ⇒ 一次传输的固定开销约 {small_ms*1000:.1f} µs，"
          f"小于约 {int(small_ms*1e-3*a['h2d_pinned']*1e9/1024)} KB 的传输几乎全是开销")

    # ---- D. 多流并发 ----
    print("\n[D] 多流并发（H2D + D2H 能否重叠）")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    up = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    down = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev2 = torch.empty(n, dtype=torch.uint8, device=f"cuda:{args.gpu}")

    def seq():
        dev.copy_(up, non_blocking=True)
        down.copy_(dev2, non_blocking=True)

    def par():
        with torch.cuda.stream(s1):
            dev.copy_(up, non_blocking=True)
        with torch.cuda.stream(s2):
            down.copy_(dev2, non_blocking=True)
        s1.synchronize()
        s2.synchronize()

    ms_seq = time_cuda(seq, iters=10)
    ms_par = time_cuda(par, iters=10)
    print(f"    同一条流串行：{ms_seq:7.2f} ms   两条流并发：{ms_par:7.2f} ms   "
          f"重叠收益 {ms_seq/ms_par:.2f}×")
    print("    （H2D 与 D2H 走不同的 DMA 引擎，理论上可以完全重叠）")
    res["D_overlap"] = {"seq_ms": round(ms_seq, 2), "par_ms": round(ms_par, 2),
                        "speedup": round(ms_seq / ms_par, 2)}

    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
