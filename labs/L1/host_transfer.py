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
import time
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


def numa_maps_summary(tag: str = "") -> dict:
    """本进程的实际页归属，直接读 `/proc/self/numa_maps`。

    为什么必须有这一步：`numactl --membind=1` 只是设了内存策略，
    CPU 亲和性列表也只能证明 **CPU** 绑对了。真正要证明的是
    「这些字节的物理页落在哪个节点上」。numa_maps 的每一行形如

        7f2c... default anon=65536 dirty=65536 N0=65536 kernelpagesize_kB=4

    其中 `N0=`/`N1=` 就是该 VMA 在各节点的页数，这是内核给出的实际页归属。
    `numastat -p` 给的是同一类信息的汇总，但 crater 上没装 numactl，
    所以这里直接解析 /proc，任何 Linux 都能跑。

    **单位陷阱**：`N0=` 的计数单位是该行的 `kernelpagesize_kB`
    （普通页 4 KB，透明大页 2048 KB），同一个进程里两种都可能出现。
    把页数直接相加会把 2 MB 大页按 4 KB 计，得出「141 GB 匿名页」
    这种荒谬读数（第一版就是这么错的）。这里一律先折算成字节。
    """
    by_node_bytes: dict[str, int] = {}
    sizes_seen: set[int] = set()
    total_bytes = 0
    vmas = []
    try:
        with open(f"/proc/{os.getpid()}/numa_maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                addr, fields = parts[0], parts[2:]
                anon = 0
                kpage_kb = os.sysconf("SC_PAGE_SIZE") // 1024
                nodes: dict[str, int] = {}
                for fld in fields:
                    if fld.startswith("anon="):
                        anon = int(fld[5:]) if fld[5:].isdigit() else 0
                    elif fld.startswith("kernelpagesize_kB="):
                        kpage_kb = int(fld.split("=", 1)[1])
                    elif fld.startswith("N") and "=" in fld:
                        k, v = fld.split("=", 1)
                        if k[1:].isdigit() and v.isdigit():
                            nodes[k] = int(v)
                if not anon:
                    continue
                sizes_seen.add(kpage_kb)
                total_bytes += anon * kpage_kb * 1024
                for k, v in nodes.items():
                    by_node_bytes[k] = by_node_bytes.get(k, 0) + v * kpage_kb * 1024
                vmas.append({"addr": addr, "anon_pages": anon,
                             "kernel_page_kb": kpage_kb, "nodes": nodes})
    except OSError as e:  # noqa: BLE001
        return {"error": str(e)}
    vmas.sort(key=lambda x: -x["anon_pages"] * x["kernel_page_kb"])
    return {
        "tag": tag,
        "mib_by_node": {k: round(v / 1024 / 1024, 1)
                        for k, v in sorted(by_node_bytes.items())},
        "total_anon_mib": round(total_bytes / 1024 / 1024, 1),
        "kernel_page_kb_seen": sorted(sizes_seen),
        "top_vmas": vmas[:4],
        "note": "cudaHostAlloc 的 pinned 缓冲由驱动管理，不出现在 numa_maps 里；"
                "这里反映的是普通匿名页（含 torch 的 pageable 张量）",
    }


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
    out["numa_maps"] = numa_maps_summary()
    return out


def first_and_hot(fn, warmup: int = 5, iters: int = 20) -> dict:
    """首轮（含建页表/建映射）与热态分开记。

    `time_cuda` 先预热再看中位数，那是有意为之：它回答「稳态有多快」。
    但「第一次有多慢」是另一个问题——pinned 缓冲首次触碰要建页表、
    首次 H2D 要建映射，这些成本只出现一次，会被 warmup 抹掉。
    """
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.perf_counter_ns()
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    host_first_us = (time.perf_counter_ns() - t0) / 1000.0
    dev_first_ms = a.elapsed_time(b)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    hot = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        hot.append(e0.elapsed_time(e1))
    return {"first_device_ms": round(dev_first_ms, 4),
            "first_host_us": round(host_first_us, 2),
            "hot_device_ms": round(statistics.median(hot), 4),
            "hot_gbps": None}  # 由调用方按字节数补


def p2p_sweep(pairs: list[tuple[int, int]], sizes_kb: list[int]) -> list[dict]:
    """按尺寸扫 P2P 带宽。链路类型（NODE/SYS）由 nvidia-smi topo 单独记。"""
    rows = []
    link = {}
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=20).stdout
        for line in out.splitlines():
            p = line.split()
            if p and p[0].startswith("GPU") and len(p[0]) <= 5:
                src = int(p[0][3:])
                for j, tok in enumerate(p[1:]):
                    if tok in ("X", "NODE", "SYS", "PHB", "PXB", "PIX"):
                        link[(src, j)] = tok
                    elif tok.startswith("NV"):
                        link[(src, j)] = tok
                    else:
                        break
    except Exception:  # noqa: BLE001
        pass
    for src, dst in pairs:
        if src == dst:
            continue
        if not torch.cuda.can_device_access_peer(src, dst):
            rows.append({"src": src, "dst": dst, "peer": False})
            continue
        for kb in sizes_kb:
            n = kb * 1024
            a = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src}")
            b = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst}")
            r = first_and_hot(lambda a=a, b=b: b.copy_(a, non_blocking=True),
                              warmup=3, iters=15)
            r.update({"src": src, "dst": dst, "peer": True, "kb": kb,
                      "link": link.get((src, dst), "?"),
                      "hot_gbps": round(n / (r["hot_device_ms"] * 1e-3) / 1e9, 2),
                      "first_gbps": round(n / (r["first_device_ms"] * 1e-3) / 1e9, 2)})
            rows.append(r)
            del a, b
            torch.cuda.empty_cache()
    return rows


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
    ap.add_argument("--pairs", default="",
                    help="P2P 尺寸扫描的卡对，如 0:1,0:3（空则跳过）")
    ap.add_argument("--sizes-kb", default="4,16,64,256,1024,4096,16384,65536,262144")
    ap.add_argument("--skip-alloc-probe", action="store_true")
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
    nm = res["process"].get("numa_maps", {})
    if nm.get("mib_by_node"):
        dist = ", ".join(f"{k}={v} MiB" for k, v in nm["mib_by_node"].items())
        print(f"    本进程匿名页的实际归属：{dist}（共 {nm['total_anon_mib']} MiB）")

    n = args.mb * 1024 * 1024
    dev = torch.empty(n, dtype=torch.uint8, device=f"cuda:{args.gpu}")
    pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    paged = torch.empty(n, dtype=torch.uint8)

    # ---- A0. 绑定是否真的生效：看一块 pageable 内存的实际页归属 ----
    # 光有 CPU 亲和性列表不够——那只证明线程跑在哪个节点上。
    # 这里分配一块 pageable 张量并首次触碰，让内核按当前 mempolicy 落地，
    # 再从 numa_maps 读回每个节点的真实字节数。
    probe = torch.empty(n, dtype=torch.uint8)
    probe.fill_(3)
    placement = numa_maps_summary(f"pageable probe {args.mb} MiB after first touch")
    res["A0_page_placement"] = placement
    print("\n[A0] 绑定核对：pageable 缓冲首次触碰后的实际页归属")
    if placement.get("mib_by_node"):
        for k, v in placement["mib_by_node"].items():
            print(f"    {k}: {v} MiB")
        print(f"    （进程匿名页合计 {placement['total_anon_mib']} MiB，"
              f"页大小 {placement['kernel_page_kb_seen']} KB）")
        print("    注：cudaHostAlloc 的 pinned 缓冲由驱动管理，不出现在 numa_maps 里；")
        print("        这里用的是同策略下的 pageable 页，作为 mempolicy 生效的证据。")
    del probe

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

    # ---- A2. 首轮 vs 热态；分配与触碰放在计时内/外 ----
    if not args.skip_alloc_probe:
        print("\n[A2] 首轮 vs 热态；pinned 分配与触碰的代价（连续 5 次新建缓冲）")
        fresh = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        fh = first_and_hot(lambda: dev.copy_(fresh, non_blocking=True))
        fh["first_gbps"] = round(n / (fh["first_device_ms"] * 1e-3) / 1e9, 1)
        fh["hot_gbps"] = round(n / (fh["hot_device_ms"] * 1e-3) / 1e9, 1)
        print(f"    同一次传输：首次 H2D {fh['first_device_ms']*1e3:8.1f} µs ({fh['first_gbps']:5.1f} GB/s)"
              f"   热态 {fh['hot_device_ms']*1e3:8.1f} µs ({fh['hot_gbps']:5.1f} GB/s)"
              f"   比值 {fh['first_device_ms']/fh['hot_device_ms']:.2f}×")
        del fresh

        # pinned 分配：`torch.empty(pin_memory=True)` 只是向驱动的 pinned 池要一块，
        # 真正把物理页锁定发生在第一次触碰。连续新建 5 次可以看到池的预热曲线。
        alloc_rows = []
        torch.cuda.synchronize()
        for i in range(5):
            t0 = time.perf_counter_ns()
            tmp = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            alloc_ms = (time.perf_counter_ns() - t0) / 1e6
            t0 = time.perf_counter_ns()
            tmp.fill_(1)
            touch_ms = (time.perf_counter_ns() - t0) / 1e6
            t0 = time.perf_counter_ns()
            tmp.fill_(2)
            hot_ms = (time.perf_counter_ns() - t0) / 1e6
            alloc_rows.append({"round": i + 1, "alloc_ms": round(alloc_ms, 3),
                               "first_touch_ms": round(touch_ms, 3),
                               "hot_touch_ms": round(hot_ms, 3)})
            del tmp
        print(f"    {'轮次':>4} {'分配ms':>9} {'首次触碰ms':>11} {'热态触碰ms':>11}")
        for r in alloc_rows:
            print(f"    {r['round']:>4} {r['alloc_ms']:>9.3f} "
                  f"{r['first_touch_ms']:>11.3f} {r['hot_touch_ms']:>11.3f}")
        res["A2_first_vs_hot"] = {"h2d": fh, "pinned_rounds": alloc_rows}

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

    # ---- B2. P2P 尺寸扫描 ----
    if args.pairs:
        pairs = [tuple(int(x) for x in pr.split(":")) for pr in args.pairs.split(",")]
        sizes = [int(x) for x in args.sizes_kb.split(",")]
        print("\n[B2] P2P 带宽 vs 尺寸（单向）")
        print(f"    {'卡对':>8} {'链路':>5} {'大小':>9} {'热态GB/s':>10} {'首次GB/s':>10}")
        p2p_rows = p2p_sweep(pairs, sizes)
        for r in p2p_rows:
            if not r.get("peer"):
                print(f"    {r['src']}->{r['dst']:>2}  不可达")
                continue
            label = f"{r['kb']} KB" if r["kb"] < 1024 else f"{r['kb']//1024} MB"
            print(f"    {r['src']}->{r['dst']:>4} {r['link']:>5} {label:>9} "
                  f"{r['hot_gbps']:>10.1f} {r['first_gbps']:>10.1f}")
        res["B2_p2p_sweep"] = p2p_rows

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

    # 结束前再取一次页归属
    nm2 = numa_maps_summary("end of run")
    res["process_after"] = {"numa_maps": nm2}
    if nm2.get("mib_by_node"):
        dist = ", ".join(f"{k}={v} MiB" for k, v in nm2["mib_by_node"].items())
        print(f"\n    试验结束后匿名页归属：{dist}（共 {nm2['total_anon_mib']} MiB）")

    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
