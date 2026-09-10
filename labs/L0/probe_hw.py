#!/usr/bin/env python3
"""L0 lab · 测出这张卡真正的 roofline 顶点，而不是抄 datasheet。

厂商标称值（spec sheet）几乎永远达不到：显存带宽标称是引脚速率的理论上限，
tensor core TFLOPS 常常是 2:4 稀疏后的数字。做容量规划和判断「这个 kernel 慢不慢」时，
必须用**这台机器实际能达到的**上限做分母。

本脚本测四件事：
  1. 显存带宽 (device-to-device)  —— decode 阶段的分母
  2. 稠密 GEMM 吞吐 (bf16 / fp16 / fp8)  —— prefill 阶段的分母
  3. kernel launch 开销  —— 解释 CUDA Graph 为什么有用
  4. H2D / D2H PCIe 带宽  —— 权重加载与 KV 传输的分母

输出 JSON 到 stdout 或 --out 指定的文件。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import torch


# ---------------------------------------------------------------------------
# 计时基元：用 CUDA event 而不是 time.time()。
# CUDA kernel 是异步下发的，host 侧计时器测到的是「下发耗时」而非「执行耗时」。
# ---------------------------------------------------------------------------

def time_cuda(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """返回单次执行的中位耗时（毫秒）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# 1. 显存带宽
# ---------------------------------------------------------------------------

def probe_memory_bandwidth(mb: int = 512) -> dict:
    """用大块 copy 逼近显存带宽上限。

    copy_ 读一遍写一遍，所以搬运字节数 = 2 * size。
    选 512MB 是为了确保远超 L2（5090 约 96MB，L40S 约 96MB），
    否则测到的是 L2 带宽而不是显存带宽。
    """
    n = mb * 1024 * 1024 // 2  # fp16 元素数
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    src.normal_()

    ms = time_cuda(lambda: dst.copy_(src))
    moved_bytes = 2 * src.numel() * src.element_size()
    gbps = moved_bytes / (ms * 1e-3) / 1e9

    # 只读的场景（decode 读 KV cache 更接近这个）用 sum 近似
    ms_read = time_cuda(lambda: src.sum())
    read_gbps = src.numel() * src.element_size() / (ms_read * 1e-3) / 1e9

    return {
        "copy_gbps": round(gbps, 1),
        "readonly_gbps": round(read_gbps, 1),
        "buffer_mb": mb,
        "note": "copy_ 计入读+写两份流量；readonly 用 reduction 近似纯读",
    }


# ---------------------------------------------------------------------------
# 2. 稠密 GEMM 吞吐
# ---------------------------------------------------------------------------

def probe_gemm(sizes=(4096, 8192)) -> dict:
    """测 tensor core 实际能到的稠密 TFLOPS。

    GEMM 的 FLOP 数 = 2*M*N*K（一次乘 + 一次加）。
    用方阵 M=N=K，规模要够大才能摊薄启动与尾效应。
    """
    out: dict = {}
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
        best = 0.0
        detail = {}
        for s in sizes:
            a = torch.randn(s, s, dtype=dtype, device="cuda")
            b = torch.randn(s, s, dtype=dtype, device="cuda")
            ms = time_cuda(lambda: torch.mm(a, b), warmup=5, iters=20)
            tflops = 2 * s**3 / (ms * 1e-3) / 1e12
            detail[f"{s}"] = round(tflops, 1)
            best = max(best, tflops)
            del a, b
            torch.cuda.empty_cache()
        out[name] = {"peak_tflops": round(best, 1), "by_size": detail}

    # FP8：Ada(sm_89) 与 Blackwell(sm_120) 都有 FP8 tensor core，
    # 走 torch._scaled_mm（per-tensor scale）。
    try:
        s = 8192
        a = torch.randn(s, s, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(s, s, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
        scale = torch.tensor(1.0, device="cuda")
        ms = time_cuda(
            lambda: torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                     out_dtype=torch.bfloat16),
            warmup=5, iters=20)
        out["fp8_e4m3"] = {"peak_tflops": round(2 * s**3 / (ms * 1e-3) / 1e12, 1),
                           "by_size": {str(s): round(2 * s**3 / (ms * 1e-3) / 1e12, 1)}}
        del a, b
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out["fp8_e4m3"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out


# ---------------------------------------------------------------------------
# 3. kernel launch 开销
# ---------------------------------------------------------------------------

def probe_launch_overhead(n_kernels: int = 500) -> dict:
    """一个「什么都不做」的小 kernel，重复下发，测每次的摊销成本。

    这是 CUDA Graph 存在的理由：decode 阶段每步要下发几百个小 kernel，
    如果每个都要 CPU 走一遍 driver，CPU 就成了瓶颈。
    """
    x = torch.zeros(1, device="cuda")

    def burst():
        for _ in range(n_kernels):
            x.add_(1.0)

    ms_eager = time_cuda(burst, warmup=3, iters=10)

    # 同样的序列用 CUDA Graph 捕获后重放
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(n_kernels):
            x.add_(1.0)
    ms_graph = time_cuda(g.replay, warmup=3, iters=10)

    return {
        "n_kernels": n_kernels,
        "eager_us_per_kernel": round(ms_eager * 1000 / n_kernels, 2),
        "graph_us_per_kernel": round(ms_graph * 1000 / n_kernels, 2),
        "graph_speedup": round(ms_eager / ms_graph, 2),
    }


# ---------------------------------------------------------------------------
# 4. PCIe 带宽
# ---------------------------------------------------------------------------

def probe_pcie(mb: int = 256) -> dict:
    n = mb * 1024 * 1024
    host_pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    host_paged = torch.empty(n, dtype=torch.uint8)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")

    def gbps(ms):
        return round(n / (ms * 1e-3) / 1e9, 1)

    return {
        "h2d_pinned_gbps": gbps(time_cuda(lambda: dev.copy_(host_pinned, non_blocking=True), iters=20)),
        "d2h_pinned_gbps": gbps(time_cuda(lambda: host_pinned.copy_(dev, non_blocking=True), iters=20)),
        "h2d_pageable_gbps": gbps(time_cuda(lambda: dev.copy_(host_paged), iters=20)),
        "buffer_mb": mb,
        "note": "pageable 需要 driver 先拷到临时 pinned 缓冲，所以更慢——这是 pin_memory 的意义",
    }


# ---------------------------------------------------------------------------

def device_facts() -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        smi = ""
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "total_mem_gb": round(p.total_memory / 1e9, 1),
        "l2_cache_mb": round(getattr(p, "L2_cache_size", 0) / 1e6, 1),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="跳过大 GEMM，快速冒烟")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    result = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "device": device_facts(),
        "memory_bandwidth": probe_memory_bandwidth(128 if args.quick else 512),
        "gemm": probe_gemm((2048,) if args.quick else (4096, 8192)),
        "launch_overhead": probe_launch_overhead(100 if args.quick else 500),
        "pcie": probe_pcie(64 if args.quick else 256),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
