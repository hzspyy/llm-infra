#!/usr/bin/env python3
"""L1.3 lab · CPU 侧的 NUMA 本地 vs 跨节点内存带宽（PCIe 实验的对照组）。

为什么需要这个对照：
`host_transfer.py` 测出 NUMA 亲和性对 H2D/D2H **完全没有影响**（25.0 vs 25.1 GB/s）。
这是负结果，但它有两种可能的解释：
  (a) 这台机器的跨 NUMA 链路根本不慢；
  (b) 跨 NUMA 确实慢，但 PCIe 更慢，所以瓶颈不在那里。

区分二者的办法：把 GPU 拿掉，只在 CPU 上测本地/远端内存带宽。
如果 CPU 侧有明显差异，就是 (b)——PCIe 把 NUMA 的影响掩盖了。

用法（两次运行，比较）：
    numactl --cpunodebind=0 --membind=0 python numa_bandwidth.py   # 本地
    numactl --cpunodebind=0 --membind=1 python numa_bandwidth.py   # 跨节点
"""
from __future__ import annotations

import os
import statistics
import time

import numpy as np


def bench(n_bytes: int = 2 << 30, repeats: int = 5) -> dict:
    n = n_bytes // 8
    a = np.ones(n, dtype=np.float64)
    b = np.empty_like(a)

    def once_copy():
        t0 = time.perf_counter()
        np.copyto(b, a)
        return time.perf_counter() - t0

    def once_read():
        t0 = time.perf_counter()
        s = a.sum()
        return time.perf_counter() - t0, s

    once_copy(); once_read()                       # 预热并让页真正落盘（first touch）
    copy_s = statistics.median(once_copy() for _ in range(repeats))
    read_s = statistics.median(once_read()[0] for _ in range(repeats))
    return {
        "copy_gbps": round(2 * n_bytes / copy_s / 1e9, 1),   # 读+写
        "read_gbps": round(n_bytes / read_s / 1e9, 1),
        "buffer_gb": round(n_bytes / 1e9, 2),
    }


def main() -> None:
    aff = sorted(os.sched_getaffinity(0))
    print(f"可用 CPU 数 {len(aff)}，前几个核 {aff[:6]}")
    try:
        print(open("/proc/self/numa_maps").read().count("N1=") and "", end="")
    except Exception:  # noqa: BLE001
        pass
    r = bench()
    print(f"  单线程 copy(读+写) {r['copy_gbps']:>7.1f} GB/s   "
          f"read {r['read_gbps']:>7.1f} GB/s   （{r['buffer_gb']} GB 缓冲）")


if __name__ == "__main__":
    main()
