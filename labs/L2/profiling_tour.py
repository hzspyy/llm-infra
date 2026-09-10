#!/usr/bin/env python3
"""L2.6 lab · profiling 的四层尺子，各自能回答什么、各自骗你什么。

ncu 看单个 kernel 的内部（profile_ladder.sh 做了）。这个脚本管另外三层：

    [A] torch.profiler   —— 哪些 kernel、各占多少时间、CPU 与 GPU 怎么交错
    [B] chrome trace     —— 那个 .json 里到底是什么？自己解析一遍，算 GPU 空闲率
    [C] NVTX             —— 给时间轴贴标签，否则 444 个 kernel 你根本读不下去
    [D] 显存 profiling   —— _record_memory_history，看清每一块显存是谁分配的
    [E] profiler 的开销  —— 观测行为本身扰动被观测对象，先量出扰动有多大

用法：python profiling_tour.py --out-dir results/prof/
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def timeit(fn, iters: int = 30) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def workload(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """compute-bound 对照组：一层 MLP，时间几乎全在两次大 GEMM 上。"""
    h = x @ w1
    h = torch.nn.functional.silu(h)
    h = h @ w2
    return h + x


def tiny_workload(t: torch.Tensor) -> torch.Tensor:
    """launch-bound 对照组：120 个各自只跑几微秒的小 kernel。

    真实的 LLM decode 就长这样（L0.1 实测 vLLM 一步 444 个 kernel），
    所以 profiler 的开销、时间轴上的缝，都要在这种负载上看才有意义。
    """
    for _ in range(40):
        t = t + 1.0
        t = torch.nn.functional.relu(t)
        t = t * 0.999
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="prof")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")

    d, ff = 2048, 8192
    x = torch.randn(4096, d, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(d, ff, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(ff, d, device="cuda", dtype=torch.bfloat16)
    run = lambda: workload(x, w1, w2)                              # noqa: E731
    small = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    run_tiny = lambda: tiny_workload(small)                        # noqa: E731

    # -----------------------------------------------------------------
    # [E] 先量 profiler 自己的开销 —— 后面所有数字都要打这个折扣。
    #     必须在两种负载上分别量：开销是**按 kernel 计**的，
    #     所以 kernel 越小越密，观测扰动越大。
    # -----------------------------------------------------------------
    print("[E] profiler 的观测开销（两种负载对照）")
    from torch.profiler import ProfilerActivity, profile, record_function

    CFGS = [
        ("CUDA only", dict(activities=[ProfilerActivity.CUDA])),
        ("CPU+CUDA", dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])),
        ("CPU+CUDA+shapes", dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                 record_shapes=True)),
        ("再加 stack+memory", dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                   record_shapes=True, with_stack=True,
                                   profile_memory=True)),
    ]
    print(f"    {'':22s} {'GEMM 主导':>20s}   {'小 kernel 洪流':>20s}")
    base = {}
    for label, fn in (("compute", run), ("tiny", run_tiny)):
        base[label] = timeit(fn)
    print(f"    {'无 profiler':22s} {base['compute']:10.4f} ms         "
          f"{base['tiny']:10.4f} ms")
    for name, kw in CFGS:
        row = {}
        for label, fn in (("compute", run), ("tiny", run_tiny)):
            with profile(**kw):
                row[label] = timeit(fn, iters=30)
        print(f"    {name:22s} {row['compute']:10.4f} ms {row['compute']/base['compute']:5.2f}× "
              f"{row['tiny']:10.4f} ms {row['tiny']/base['tiny']:5.2f}×")
    print("    ↑ 同一个 profiler，对大 kernel 几乎免费，对小 kernel 洪流是实打实的税")

    # -----------------------------------------------------------------
    # [A] torch.profiler：kernel 成分表
    # -----------------------------------------------------------------
    print("\n[A] torch.profiler · kernel 成分")
    from torch.autograd import DeviceType

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(10):
            run()
        torch.cuda.synchronize()

    evts = [e for e in prof.key_averages()
            if e.device_type == DeviceType.CUDA and (e.device_time_total or 0) > 0]
    total_us = sum(e.device_time_total for e in evts)
    print(f"    {'kernel':52s} {'次数':>5s} {'总us':>9s} {'占比':>7s}")
    for e in sorted(evts, key=lambda e: -e.device_time_total)[:8]:
        print(f"    {e.key[:52]:52s} {e.count:>5d} {e.device_time_total:>9.1f} "
              f"{e.device_time_total/total_us*100:>6.1f}%")
    print(f"    合计 {len(evts)} 种 kernel, {total_us:.1f} us / 10 次迭代")

    (out / "50_key_averages.txt").write_text(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=40),
        encoding="utf-8")
    print("    写出 50_key_averages.txt")

    trace_path = out / "51_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"    写出 51_trace.json  {trace_path.stat().st_size/1e6:.2f} MB")

    # -----------------------------------------------------------------
    # [B] 自己解析 chrome trace：它里面到底是什么
    # -----------------------------------------------------------------
    print("\n[B] chrome trace 的内容（自己解析）")
    tr = json.loads(trace_path.read_text())
    evs = tr["traceEvents"]
    print(f"    顶层键        {list(tr.keys())}")
    print(f"    事件总数      {len(evs)}")

    by_cat: dict[str, int] = defaultdict(int)
    for e in evs:
        by_cat[e.get("cat", "<无 cat>")] += 1
    print("    按 cat 分类：")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {k:24s} {v}")

    # GPU 空闲率：把 kernel 事件按时间合并成区间，看时间轴上有多少缝
    def gpu_gap_analysis(events: list, tag: str) -> float:
        """返回 GPU 忙碌毫秒数。"""
        kern = sorted(((e["ts"], e["ts"] + e.get("dur", 0))
                       for e in events
                       if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                      key=lambda p: p[0])
        if not kern:
            print(f"    [{tag}] 没有 GPU 事件")
            return 0.0
        merged: list[list[float]] = []
        for s, t in kern:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], t)
            else:
                merged.append([s, t])
        span = merged[-1][1] - merged[0][0]
        busy = sum(t - s for s, t in merged)
        gaps = sorted((merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)),
                      reverse=True)
        print(f"    [{tag}] {len(kern)} 个 GPU 事件, 跨度 {span/1e3:.3f} ms, "
              f"忙 {busy/1e3:.3f} ms, **空闲 {(span-busy)/span*100:.1f}%**")
        if gaps:
            print(f"           最大的 5 个缝 {[f'{g:.1f}us' for g in gaps[:5]]}, "
                  f"缝总计 {sum(gaps):.1f}us")
        return busy / 1e3

    gpu_gap_analysis(evs, "GEMM 主导")

    # 同样的分析换到小 kernel 洪流上 —— 缝才会露出来
    ITERS = 3
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_t:
        for _ in range(ITERS):
            run_tiny()
        torch.cuda.synchronize()
    tiny_trace = out / "53_trace_tiny.json"
    prof_t.export_chrome_trace(str(tiny_trace))
    tiny_evs = json.loads(tiny_trace.read_text())["traceEvents"]
    gpu_busy_ms = gpu_gap_analysis(tiny_evs, "小 kernel 洪流") / ITERS
    print("    ↑ 缝 = GPU 干等 CPU。缝多说明是 launch-bound（对策见 L2.1 CUDA Graph）")

    # 自指的修正：上面那个空闲率是**在 profiler 底下**测的，
    # 而 [E] 已经量出 profiler 会让这种负载变慢。GPU 忙的时间不受影响，
    # 所以用干净的墙钟时间就能还原出真实空闲率。
    clean_ms = base["tiny"]
    print(f"    修正：GPU 忙 {gpu_busy_ms:.3f} ms/次（这个数不受 profiler 影响）")
    print(f"          干净墙钟 {clean_ms:.3f} ms/次  ->  真实空闲 "
          f"{(clean_ms-gpu_busy_ms)/clean_ms*100:.1f}%")
    print("          观测行为自己制造了一部分缝——量它之前先量自己")

    # 单条事件原样落盘，给"原始现场"用
    sample = [e for e in evs if e.get("cat") == "kernel"][:2]
    sample += [e for e in evs if e.get("cat") in ("cpu_op", "op")][:2]
    (out / "52_trace_sample_events.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    写出 52_trace_sample_events.json（4 条原样事件）")

    # -----------------------------------------------------------------
    # [C] NVTX：给时间轴贴标签
    # -----------------------------------------------------------------
    print("\n[C] NVTX 标注")
    for _ in range(5):                       # 先热身，否则首次调用的开销会算进去
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof2:
        with record_function("## MLP 前半 ##"):
            h = x @ w1
            h = torch.nn.functional.silu(h)
        with record_function("## MLP 后半 ##"):
            h = h @ w2 + x
        torch.cuda.synchronize()
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for e in prof2.key_averages():
        if e.key.startswith("## "):
            agg[e.key][0] += e.cpu_time_total
            agg[e.key][1] += e.device_time_total
    for k, (c, g) in agg.items():
        print(f"    {k:20s} CPU {c:8.1f} us  GPU {g:8.1f} us")
    print("    （nsys 下用 torch.cuda.nvtx.range_push/pop，时间轴上会显示成彩色区间）")

    # -----------------------------------------------------------------
    # [D] 显存 profiling
    # -----------------------------------------------------------------
    print("\n[D] 显存 profiling")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    keep = []
    for i in range(6):                    # 制造碎片：交替分配大小块，只留大块
        big = torch.empty(256 << 20 // 4, device="cuda", dtype=torch.float32)
        small = torch.empty(1 << 20, device="cuda", dtype=torch.float32)
        keep.append(small if i % 2 else big)
        del big, small
    y = workload(x, w1, w2)
    torch.cuda.synchronize()

    snap_path = out / "60_memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(str(snap_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"    写出 60_memory_snapshot.pickle  {snap_path.stat().st_size/1e6:.2f} MB")
    print("    （用 https://docs.pytorch.org/memory_viz 打开，纯前端，不上传）")

    st = torch.cuda.memory_stats()
    fields = [
        ("allocated_bytes.all.current", "当前已分配"),
        ("allocated_bytes.all.peak", "分配峰值"),
        ("reserved_bytes.all.current", "当前已保留(向驱动要的)"),
        ("reserved_bytes.all.peak", "保留峰值"),
        ("inactive_split_bytes.all.current", "碎片(保留但不可用)"),
        ("num_alloc_retries", "分配重试次数"),
        ("num_ooms", "OOM 次数"),
    ]
    for k, label in fields:
        v = st.get(k, 0)
        unit = f"{v/2**20:10.1f} MiB" if "bytes" in k else f"{v:10d}"
        print(f"    {label:24s} {unit}")
    frag = st.get("reserved_bytes.all.current", 0) - st.get("allocated_bytes.all.current", 0)
    print(f"    {'保留 - 已分配':24s} {frag/2**20:10.1f} MiB  ← 这部分显存你占着但用不上")

    (out / "61_memory_stats.json").write_text(
        json.dumps({k: v for k, v in st.items() if isinstance(v, int)},
                   indent=2), encoding="utf-8")
    print("    写出 61_memory_stats.json（allocator 的全部计数器）")

    del keep, y
    print(f"\n所有产物在 {out.absolute()}")


if __name__ == "__main__":
    main()
