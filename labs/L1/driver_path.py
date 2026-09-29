#!/usr/bin/env python3
"""L1.4 lab · 把「一次 kernel launch」拆开：用户态、驱动、内核。

L1.1 测出 kernel launch 要 2.68 µs（crater）/ 5.03 µs（worldvln），
而 decode 每步要发 444 次 —— 合计 1.2 ms，占墙钟的 27%（L0.1 实测）。
这 2.68 µs 到底花在哪？

五个实验：
  A. 初始化的一次性成本：cuInit / 创建 context / 第一个 kernel
  B. launch 开销的分解：空 kernel / 带参数 / 不同 grid 大小
  C. CUDA Graph 消除了什么（对照 A/B）
  D. 统一内存（UVM）的缺页代价 —— 移到 uvm_probe.cu
  E. 同步原语的代价：cudaDeviceSynchronize / event / stream query
  F. launch 到底进不进内核态（不需要 strace 也能回答）

用法：
    python driver_path.py --out results/driver_path.json
    # 看系统调用：
    strace -c -f python driver_path.py --syscall-demo
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


def med_us(fn, warmup=20, iters=200) -> float:
    """纯 host 侧计时：测的是「CPU 把请求交出去要多久」，不是 GPU 执行多久。"""
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        xs.append((time.perf_counter_ns() - t0) / 1000.0)
    return statistics.median(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--syscall-demo", action="store_true",
                    help="只跑一小段固定负载，供 strace 统计系统调用")
    args = ap.parse_args()

    # ---- A. 初始化的一次性成本 ----
    # 必须在 import torch 之后、任何 CUDA 调用之前打点
    t_import0 = time.perf_counter()
    import torch
    t_import = (time.perf_counter() - t_import0) * 1e3

    t0 = time.perf_counter()
    torch.cuda.init()                       # cuInit + 驱动握手
    t_init = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    x = torch.zeros(1, device="cuda")       # 首次分配 → 创建 context
    torch.cuda.synchronize()
    t_ctx = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    x.add_(1.0)                             # 第一个 kernel：可能触发模块加载/JIT
    torch.cuda.synchronize()
    t_first_kernel = (time.perf_counter() - t0) * 1e3

    p = torch.cuda.get_device_properties(0)
    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "gpu": p.name, "torch": torch.__version__, "cuda": torch.version.cuda,
        "A_startup_ms": {"import_torch": round(t_import, 1), "cuda_init": round(t_init, 1),
                         "create_context_first_alloc": round(t_ctx, 1),
                         "first_kernel": round(t_first_kernel, 2)},
    }
    print(f"=== {p.name}  torch {torch.__version__}  CUDA {torch.version.cuda}")
    print("\n[A] 一次性启动成本（每个进程都要付一遍）")
    for k, v in res["A_startup_ms"].items():
        print(f"    {k:28s} {v:9.2f} ms")
    print("    ⇒ 这就是为什么引擎要常驻进程，而不是每个请求起一个进程。")

    if args.syscall_demo:
        for _ in range(10000):
            x.add_(1.0)
        torch.cuda.synchronize()
        return

    # ---- B. launch 开销分解 ----
    print("\n[B] 单次 launch 的 host 侧耗时（不等 GPU，只看下发）")
    small = torch.zeros(1, device="cuda")
    big = torch.zeros(1 << 22, device="cuda")            # 4M 元素
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")

    cases = {
        "空操作(python 基线)": lambda: None,
        "1 元素 add_": lambda: small.add_(1.0),
        "4M 元素 add_": lambda: big.add_(1.0),
        "512x512 matmul": lambda: torch.mm(a, b),
        "torch.empty(1024)": lambda: torch.empty(1024, device="cuda"),
    }
    bl = med_us(cases["空操作(python 基线)"])
    rows = {}
    for name, fn in cases.items():
        us = med_us(fn)
        rows[name] = round(us, 3)
        extra = "" if name.startswith("空操作") else f"   (扣掉基线 {us - bl:6.3f})"
        print(f"    {name:22s} {us:8.3f} µs{extra}")
    print("    注意：4M 元素与 1 元素的**下发**耗时几乎一样——")
    print("          launch 开销与数据量无关，只与「发一次」有关。")
    res["B_launch_us"] = rows
    torch.cuda.synchronize()

    # ---- C. CUDA Graph ----
    print("\n[C] CUDA Graph：把 N 次下发压成 1 次")
    print(f"    {'N':>6} {'eager µs/kernel':>18} {'graph µs/kernel':>18} {'加速':>8}")
    graph_rows = []
    for n in (32, 128, 512):
        def burst(n=n):
            for _ in range(n):
                small.add_(1.0)
            torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(n):
                small.add_(1.0)

        def replay():
            g.replay()
            torch.cuda.synchronize()

        e = med_us(burst, warmup=5, iters=30) / n
        r = med_us(replay, warmup=5, iters=30) / n
        print(f"    {n:>6} {e:>18.3f} {r:>18.3f} {e/r:>7.2f}×")
        graph_rows.append({"n": n, "eager_us": round(e, 3), "graph_us": round(r, 3),
                           "speedup": round(e / r, 2)})
        del g
    res["C_cuda_graph"] = graph_rows

    # ---- D. 统一内存 ----
    # 这里曾经用 ctypes 直接调 cudaMallocManaged/cudaMemPrefetchAsync，
    # 结果 CUDA 13 改了 cudaMemPrefetchAsync 的签名（设备参数变成 cudaMemLocation 结构体），
    # ctypes 没有类型检查，调用静默失败、返回码被忽略，测出 9923 GB/s 的荒谬数字，
    # 而且**污染了 CUDA 上下文**，让后面的 kernel 全部报 invalid argument。
    # 教训：跨 ABI 调用一定要设 argtypes/restype 并检查返回码；
    #       更好的做法是直接写 CUDA C —— 见 labs/L1/uvm_probe.cu。
    print("\n[D] 统一内存的缺页代价 —— 见 labs/L1/uvm_probe.cu（CUDA C 版本）")

    # ---- E. 同步原语的代价 ----
    print("\n[E] 同步原语的 host 侧代价")
    ev = torch.cuda.Event(enable_timing=False)
    stream = torch.cuda.current_stream()
    syncs = {
        "cudaDeviceSynchronize (空闲)": lambda: torch.cuda.synchronize(),
        "event.record()": lambda: ev.record(),
        "stream.query() (非阻塞)": lambda: stream.query(),
        "tensor.item() (强制同步+拷回)": lambda: small.item(),
    }
    rows = {}
    for name, fn in syncs.items():
        us = med_us(fn, warmup=20, iters=200)
        rows[name] = round(us, 3)
        print(f"    {name:32s} {us:8.3f} µs")
    print("    `.item()` 是最常见的隐式同步来源——一行 Python 就能把流水打断。")
    res["E_sync_us"] = rows


    # ---- F. launch 到底进不进内核态 ----
    # CPU 时间按 clock ticks 计量，不能据此推断系统调用次数。
    print("\n[F] kernel launch 走内核态还是用户态？")
    HZ = os.sysconf("SC_CLK_TCK")

    def cpu_times():
        f = open(f"/proc/{os.getpid()}/stat").read().rsplit(") ", 1)[1].split()
        return int(f[11]) / HZ, int(f[12]) / HZ        # utime, stime（秒）

    N = 200000
    u0, s0 = cpu_times()
    t0 = time.perf_counter()
    for _ in range(N):
        small.add_(1.0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    u1, s1 = cpu_times()
    du, ds = u1 - u0, s1 - s0
    print(f"    {N} 次 launch：墙钟 {wall*1e3:8.1f} ms")
    print(f"    用户态 CPU {du*1e3:8.1f} ms ({du/wall:5.1%})   "
          f"内核态 CPU {ds*1e3:8.1f} ms ({ds/wall:5.1%})")
    print(f"    每次 launch 的内核态时间 ≈ {ds/N*1e9:.0f} ns")
    print(f"    CPU 时间的显示粒度为 {1000 / HZ:g} ms/tick（SC_CLK_TCK={HZ}）。")
    print("    该统计不是系统调用计数；stime 增量为零也不能证明没有系统调用。")
    res["F_kernel_vs_user"] = {"n": N, "wall_ms": round(wall*1e3, 1),
                               "utime_ms": round(du*1e3, 1), "stime_ms": round(ds*1e3, 1),
                               "stime_frac": round(ds/max(wall,1e-9), 4),
                               "clock_ticks_per_second": HZ,
                               "interpretation": "CPU time accounting, not syscall tracing"}

    # ---- G. 提交方式：逐次同步 / 批量提交 / CUDA Graph ----
    # 三者提交的是**同一组微小 kernel**，差别只在「什么时候等 GPU」。
    # 要分开的三笔成本：CPU 提交、GPU 执行、GPU 空闲（等 CPU 或等同步）。
    print("\n[G] 提交方式对照（同一组微小 kernel，N=2000）")

    def ctxt_switches() -> dict:
        out = {}
        for line in open(f"/proc/{os.getpid()}/status"):
            if line.startswith(("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")):
                k, _, v = line.partition(":")
                out[k.strip()] = int(v)
        return out

    def per_kernel_device_us(n: int = 200) -> float:
        """单次 kernel 的 GPU 侧耗时（event 测，不含 CPU 提交）。"""
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            small.add_(1.0)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1000 / n

    def run_mode(mode: str, n: int = 2000) -> dict:
        torch.cuda.synchronize()
        gaps = []
        c0 = ctxt_switches()
        submit_s = None
        capture_ms = None
        grabj = None
        if mode == "graph":
            # 捕获本身成本很高，但它不属于「执行」——必须挡在计时之外。
            t_cap = time.perf_counter()
            grabj = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(n):
                    small.add_(1.0)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(grabj):
                for _ in range(n):
                    small.add_(1.0)
            capture_ms = (time.perf_counter() - t_cap) * 1e3

        torch.cuda.synchronize()
        e_start, e_end = torch.cuda.Event(True), torch.cuda.Event(True)
        wall0 = time.perf_counter()
        e_start.record()
        if mode == "per-submit":
            for _ in range(n):
                t = time.perf_counter()
                small.add_(1.0)
                torch.cuda.synchronize()          # 每次提交后立刻等
                gaps.append((time.perf_counter() - t) * 1e6)
        elif mode == "batch":
            t_sub = time.perf_counter()
            prev = None
            for _ in range(n):
                now = time.perf_counter()
                if prev is not None:
                    gaps.append((now - prev) * 1e6)
                prev = now
                small.add_(1.0)
            submit_s = time.perf_counter() - t_sub
        elif mode == "graph":
            t_sub = time.perf_counter()
            grabj.replay()
            submit_s = time.perf_counter() - t_sub
        e_end.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall0
        device_span = e_start.elapsed_time(e_end)
        c1 = ctxt_switches()
        dev_us = per_kernel_device_us()
        busy_ms = dev_us * n / 1000
        if mode == "graph":
            del grabj
        # 「GPU 忙」的口径随提交方式而变，必须说清楚：
        #   batch / graph：提交后没有 CPU 插手，设备跨度就是执行时间；
        #   per-submit：设备跨度里混着等 CPU 往返的空隙，纯执行时间只能用
        #               单独测的单 kernel 设备时间 × N 估。
        # 用 eager 的单 kernel 时间去估 graph 会得到「空闲为负」的荒谬值
        # （graph 里每个 kernel 的设备时间本来就比 eager 低，见 [C] 段）。
        busy_ms = device_span if mode in ("batch", "graph") else dev_us * n / 1000
        return {
            "mode": mode, "n": n, "wall_ms": round(wall * 1e3, 2),
            "submit_ms": None if submit_s is None else round(submit_s * 1e3, 2),
            "capture_ms": None if capture_ms is None else round(capture_ms, 2),
            "device_busy_ms": round(busy_ms, 2),
            "device_span_ms": round(device_span, 2),
            "eager_busy_ms": round(dev_us * n / 1000, 2),
            "per_kernel_effective_us": round(busy_ms * 1000 / n, 3),
            "gpu_idle_ms": round(wall * 1e3 - busy_ms, 2),
            "gpu_idle_frac": round((wall * 1e3 - busy_ms) / (wall * 1e3), 3),
            "eager_per_kernel_us": round(dev_us, 3),
            "ctxt_switches": {k: c1[k] - c0.get(k, 0) for k in c1},
            "gap_us": {
                "n": len(gaps),
                "median": round(statistics.median(gaps), 2) if gaps else None,
                "p99": round(sorted(gaps)[int(len(gaps) * 0.99)], 2) if gaps else None,
                "max": round(max(gaps), 2) if gaps else None,
                "gt_100us": sum(1 for g in gaps if g > 100),
            },
        }

    print("    GPU忙：batch/graph 用设备跨度（提交后没有 CPU 插手）；"
          "per-submit 用单独测的单 kernel 设备时间 × N。")
    print(f"    {'方式':>12} {'墙钟ms':>9} {'提交ms':>9} {'GPU忙ms':>9} "
          f"{'设备跨度':>9} {'GPU空闲':>9} {'空闲占比':>8} {'每kernelµs':>10} "
          f"{'提交间隔中位µs':>14} {'>100µs':>7} {'换出':>5}")
    g_rows = []
    for mode in ("per-submit", "batch", "graph"):
        r = run_mode(mode)
        g_rows.append(r)
        nv = r["ctxt_switches"].get("nonvoluntary_ctxt_switches", 0)
        sub = "-" if r["submit_ms"] is None else f"{r['submit_ms']:.2f}"
        print(f"    {mode:>12} {r['wall_ms']:>9.2f} {sub:>9} "
              f"{r['device_busy_ms']:>9.2f} {r['device_span_ms']:>9.2f} "
              f"{r['gpu_idle_ms']:>9.2f} "
              f"{r['gpu_idle_frac']:>8.1%} {r['per_kernel_effective_us']:>10.3f} "
              f"{str(r['gap_us']['median']):>14} "
              f"{r['gap_us']['gt_100us']:>7} {nv:>5}")
    res["G_submit_modes"] = g_rows
    print("    读法：per-submit 的墙钟里大部分是同步往返；batch 把提交压到一次；")
    print("          graph 的 replay 提交只要一次调用，但执行仍受设备侧约束。")
    print("          捕获成本（graph）单独记在 capture_ms，不计入墙钟。")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
