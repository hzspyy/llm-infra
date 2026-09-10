#!/usr/bin/env python3
"""L1.3 lab · GPU 之间的点对点带宽矩阵。

worldvln 没有 NVLink，5 张卡分属两个 NUMA 节点：
    GPU0,1,2 -> NUMA0     GPU3,4 -> NUMA1
`nvidia-smi topo -m` 把链路标成 NODE（同节点内跨 PCIe 桥）与 SYS（跨 NUMA，走 UPI）。
这两种链路差多少？测出来。

这个矩阵是 L6（分布式）的地基：TP 的 all-reduce、EP 的 all-to-all、
PD 分离的 KV 传输，走的都是这些链路。
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def topo() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return {}
    link = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("GPU") and len(parts[0]) <= 5:
            src = int(parts[0][3:])
            for j, tok in enumerate(parts[1:]):
                if tok in ("X", "NODE", "SYS", "PHB", "PXB", "PIX") or tok.startswith("NV"):
                    link[(src, j)] = tok
                else:
                    break
    return {f"{a}->{b}": v for (a, b), v in link.items()}


def bench_pair(src: int, dst: int, mb: int = 256, iters: int = 20) -> float:
    n = mb * 1024 * 1024
    torch.cuda.set_device(src)
    a = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst}")
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    xs = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        b.copy_(a)
        e1.record()
        torch.cuda.synchronize()
        xs.append(e0.elapsed_time(e1))
    del a, b
    torch.cuda.empty_cache()
    return n / (statistics.median(xs) * 1e-3) / 1e9


def main() -> None:
    ngpu = torch.cuda.device_count()
    links = topo()
    print(f"=== {ngpu} 张 GPU\n")

    print("[P2P 可达性] cudaDeviceCanAccessPeer")
    print("      " + "".join(f"{j:>8}" for j in range(ngpu)))
    peer = {}
    for i in range(ngpu):
        row = []
        for j in range(ngpu):
            ok = i != j and torch.cuda.can_device_access_peer(i, j)
            peer[(i, j)] = ok
            row.append("  -  " if i == j else ("  是 " if ok else "  否 "))
        print(f"GPU{i} " + "".join(f"{c:>8}" for c in row))

    print("\n[单向带宽 GB/s]  括号里是 nvidia-smi 报的链路类型")
    print("      " + "".join(f"{j:>14}" for j in range(ngpu)))
    matrix = {}
    for i in range(ngpu):
        cells = []
        for j in range(ngpu):
            if i == j:
                cells.append("     -        ")
                continue
            g = bench_pair(i, j)
            matrix[f"{i}->{j}"] = round(g, 1)
            lk = links.get(f"{i}->{j}", "?")
            cells.append(f"{g:>7.1f}({lk:<4})")
        print(f"GPU{i} " + "".join(f"{c:>14}" for c in cells))

    # 汇总：按链路类型
    by_link: dict[str, list[float]] = {}
    for k, v in matrix.items():
        lk = links.get(k, "?")
        by_link.setdefault(lk, []).append(v)
    print("\n[按链路类型汇总]")
    for lk, vs in sorted(by_link.items()):
        print(f"    {lk:<6} n={len(vs):<3} 中位 {statistics.median(vs):6.1f} GB/s   "
              f"范围 {min(vs):.1f} – {max(vs):.1f}")

    out = {"n_gpu": ngpu, "topo": links, "bandwidth_gbps": matrix,
           "peer_access": {f"{i}->{j}": peer[(i, j)] for i in range(ngpu)
                           for j in range(ngpu) if i != j},
           "by_link": {k: {"median": statistics.median(v), "n": len(v)}
                       for k, v in by_link.items()}}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        print(f"\n写出 {sys.argv[1]}")


if __name__ == "__main__":
    main()
