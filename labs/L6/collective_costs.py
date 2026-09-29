#!/usr/bin/env python3
"""
6.1 集合通信的成本模型与实测扫描。

四个任务：
  A  成本模型：给出 ring/tree 的启动+传输公式，定义 M（每 rank 字节）、algbw、busbw，
     并用扫描数据拟合 alpha/beta，检查预测与实测的偏差。
  B  消息大小扫描：4 KiB→256 MiB 按 4 倍，auto 与强制算法/协议对照；从 NCCL 自己的
     tuning 日志里读出每个尺寸实际选中的算法、协议、通道数，而不是由带宽反推。
  C  通信与计算的重叠：同尺寸 GEMM 单独、通信单独、串行、带正确依赖的重叠。
  D  能力矩阵：NVLS / symmetric memory / 网络 transport 的触发条件与当前平台的实际状态。

用法：
    python collective_costs.py D --out <dir>
    python collective_costs.py B --out <dir> --world-size 2 --gpus 0,1 --tag near
    python collective_costs.py B --out <dir> --world-size 2 --gpus 0,3 --tag far \
        --nccl-algo Ring --nccl-proto Simple
    python collective_costs.py C --out <dir> --world-size 2
    python collective_costs.py A --sweep <sweep.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20,
         16 << 20, 64 << 20, 256 << 20]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "sendrecv"]


def human(n):
    if n < 1024:
        return f"{n}B"
    val = float(n)
    for unit in ("KiB", "MiB", "GiB"):
        val /= 1024
        if val < 1024 or unit == "GiB":
            return f"{val:.0f}{unit}" if abs(val - round(val)) < 1e-9 else f"{val:.1f}{unit}"
    return f"{val:.1f}GiB"


# ==========================================================================
# 成本模型（任务 A）
# ==========================================================================

def busbw_factor(op, nranks):
    """每 rank 需要搬运的字节相对 M 的倍数（ring 实现）。

    M 的定义统一为「每 rank 的输入字节数」，这是最不容易混的口径。
    """
    if op == "all_reduce":
        return 2 * (nranks - 1) / nranks      # reduce-scatter + all-gather 两轮
    if op in ("all_gather", "reduce_scatter"):
        return (nranks - 1) / nranks          # 每 rank 收/发 N-1 份 1/N 的块
    if op == "broadcast":
        return 1.0                            # root 发 M，其余各收 M
    if op == "sendrecv":
        return 1.0                            # 环上每 rank 发 M 收 M
    raise ValueError(op)


def model_rows(nranks):
    rows = []
    for op in OPS:
        f = busbw_factor(op, nranks)
        rows.append({
            "op": op,
            "M": "每 rank 输入字节",
            "ring_traffic_per_rank": f"M * {f:g}",
            "ring_steps": ("2(N-1)" if op == "all_reduce" else
                           ("(N-1)" if op in ("all_gather", "reduce_scatter") else
                            ("log2(N) 步（树）" if op == "broadcast" else "1 跳"))),
            "t_model": "alpha + M/beta",
            "algbw": "M / t",
            "busbw": f"M * {f:g} / t",
        })
    return rows


def run_A(sweep_path, out_dir):
    with open(sweep_path, encoding="utf-8") as f:
        sweep = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    report = {"source": sweep_path, "configs": []}
    for cfg in sweep["configs"]:
        n = cfg["world_size"]
        rows = model_rows(n)
        fits = []
        for op in OPS:
            pts = [p for p in cfg["points"]
                   if p["op"] == op and "t_s_max_rank" in p]
            if len(pts) < 3:
                continue

            def fit(subset):
                xs = [p["per_rank_bytes"] for p in subset]
                ys = [p["t_s_max_rank"] for p in subset]
                mx, my = statistics.fmean(xs), statistics.fmean(ys)
                denom = sum((x - mx) ** 2 for x in xs)
                slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
                         if denom else 0.0)
                alpha = my - slope * mx
                pred = [alpha + slope * x for x in xs]
                rel = [abs(p - y) / y for p, y in zip(pred, ys) if y > 0]
                return {
                    "points": len(subset),
                    "alpha_us": alpha * 1e6,
                    "beta_GBps": (1.0 / slope) / 1e9 if slope > 0 else None,
                    "max_rel_error": max(rel) if rel else None,
                    "median_rel_error": statistics.median(rel) if rel else None,
                }

            bus = [busbw_factor(op, n) * p["per_rank_bytes"] / p["t_s_max_rank"] / 1e9
                   for p in pts]
            large = [p for p in pts if p["per_rank_bytes"] >= (1 << 20)]
            fits.append({
                "op": op,
                "busbw_factor": busbw_factor(op, n),
                "fit_all": fit(pts),
                "fit_large_ge_1MiB": fit(large) if len(large) >= 3 else None,
                "busbw_median_GBps": statistics.median(bus),
                "busbw_max_GBps": max(bus),
                "busbw_at_256MiB_GBps": next(
                    (busbw_factor(op, n) * p["per_rank_bytes"] / p["t_s_max_rank"] / 1e9
                     for p in reversed(pts) if p["per_rank_bytes"] == (256 << 20)), None),
                "measured_points": [
                    {"bytes": p["per_rank_bytes"], "t_s": p["t_s_max_rank"],
                     "busbw_GBps": busbw_factor(op, n) * p["per_rank_bytes"]
                     / p["t_s_max_rank"] / 1e9} for p in pts],
            })
        report["configs"].append({
            "tag": cfg["tag"], "world_size": n, "gpus": cfg.get("gpus"),
            "nccl_algo": cfg.get("nccl_algo"), "nccl_proto": cfg.get("nccl_proto"),
            "model_rows": rows, "fits": fits,
        })
        print(f"== {cfg['tag']} (N={n})")
        for r in fits:
            fa, fl = r["fit_all"], r["fit_large_ge_1MiB"]
            print(f"   {r['op']:<15} 全区间 alpha={fa['alpha_us']:7.2f} us "
                  f"beta={fa['beta_GBps']:6.2f} GB/s 最大相对误差={fa['max_rel_error']:.3f}"
                  + (f" | ≥1MiB alpha={fl['alpha_us']:7.2f} us beta={fl['beta_GBps']:6.2f} GB/s "
                     f"最大相对误差={fl['max_rel_error']:.3f}" if fl else "")
                  + f" | 256MiB busbw={r['busbw_at_256MiB_GBps']:.2f} GB/s")
    with open(os.path.join(out_dir, "cost_model.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ==========================================================================
# 扫描（任务 B）
# ==========================================================================

def _bench_one(rank, nranks, op, nbytes, device=None, isolate=False):
    # device 显式给出时用给定设备（跨机档每台机器只有一个进程，本地编号恒为 cuda:0）
    # isolate=True 在每次计时前插一次 barrier + 同步，切断 NCCL proxy 跨轮次的流水，
    # 否则小消息测到的是入队/内核返回时间而不是真正送达时间。
    dev = torch.device(device) if device else torch.device(f"cuda:{rank}")
    elems = max(1, nbytes // 4)
    if op == "all_reduce":
        t = torch.ones(elems, dtype=torch.float32, device=dev)
        fn = lambda: dist.all_reduce(t)  # noqa: E731
        per_rank = nbytes
    elif op == "all_gather":
        t = torch.ones(elems, dtype=torch.float32, device=dev)
        out = torch.empty(elems * nranks, dtype=torch.float32, device=dev)
        fn = lambda: dist.all_gather_into_tensor(out, t)  # noqa: E731
        per_rank = nbytes
    elif op == "reduce_scatter":
        inp = torch.ones(elems * nranks, dtype=torch.float32, device=dev)
        out = torch.empty(elems, dtype=torch.float32, device=dev)
        fn = lambda: dist.reduce_scatter_tensor(out, inp)  # noqa: E731
        per_rank = nbytes
    elif op == "broadcast":
        t = torch.ones(elems, dtype=torch.float32, device=dev)
        fn = lambda: dist.broadcast(t, src=0)  # noqa: E731
        per_rank = nbytes
    elif op == "sendrecv":
        snd = torch.ones(elems, dtype=torch.float32, device=dev)
        rcv = torch.empty(elems, dtype=torch.float32, device=dev)
        def fn():
            if nranks == 1:
                rcv.copy_(snd)
                return
            ops = [dist.P2POp(dist.isend, snd, (rank + 1) % nranks),
                   dist.P2POp(dist.irecv, rcv, (rank - 1) % nranks)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()
        per_rank = nbytes
    else:
        raise ValueError(op)

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    iters = 20 if nbytes <= (4 << 20) else 8
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if isolate:
            dist.barrier()
            torch.cuda.synchronize()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]
    return {"rank": rank, "op": op, "per_rank_bytes": per_rank,
            "isolate": isolate,
            "t_s_median": statistics.median(times),
            "t_s": times}


def internal_sweep(rank, nranks, out_path, sizes, ops=None):
    dist.init_process_group("nccl", rank=rank, world_size=nranks,
                            timeout=timedelta(seconds=180))
    torch.cuda.set_device(rank)
    points = []
    for op in (ops or OPS):
        for nb in sizes:
            try:
                r = _bench_one(rank, nranks, op, nb)
                points.append(r)
                if rank == 0:
                    print(f"  {op:<15} {human(nb):>8} t={r['t_s_median'] * 1e3:9.3f} ms",
                          flush=True)
            except Exception as e:
                points.append({"rank": rank, "op": op, "per_rank_bytes": nb,
                               "error": f"{type(e).__name__}: {e}"})
                if rank == 0:
                    print(f"  {op:<15} {human(nb):>8} ERROR {e}", flush=True)
        dist.barrier()
    # 每个 rank 各自落盘，由驱动进程在 host 侧归并。
    # 不用 all_gather_object：它内部走 ncclAllGather(int8)，在强制 NCCL_ALGO=Tree 时
    # 会因为「Tree 不是 AllGather 的合法算法」直接失败，把测量本身带崩。
    per_rank_path = f"{out_path}.rank{rank}"
    with open(per_rank_path, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False)
    dist.barrier()  # 等所有 rank 都写完再让 rank 0 归并，否则会读到半截文件
    if rank == 0:
        merged = []
        for i, p0 in enumerate(points):
            op, nb = p0["op"], p0["per_rank_bytes"]
            meds = []
            err = None
            for r in range(nranks):
                with open(f"{out_path}.rank{r}", encoding="utf-8") as f:
                    data = json.load(f)
                if "error" in data[i]:
                    err = data[i]["error"]
                else:
                    meds.append(data[i]["t_s_median"])
            if err:
                merged.append({"op": op, "per_rank_bytes": nb, "error": err})
                continue
            merged.append({
                "op": op, "per_rank_bytes": nb,
                "t_s_per_rank": meds,
                "t_s_min_rank": min(meds),
                "t_s_max_rank": max(meds),
                "t_s_spread": (max(meds) - min(meds)) / statistics.median(meds),
                "algbw_GBps_max_rank": nb / max(meds) / 1e9,
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"points": merged,
                       "iters_note": "小消息 20 次、大消息 8 次 device-event 计时"}, f,
                      ensure_ascii=False, indent=2)
    dist.barrier()
    dist.destroy_process_group()


ALGO_RE = re.compile(
    r"NCCL INFO (\w+): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")


def parse_algo_log(path, op_map):
    """把 NCCL 自己打印的算法选择行解析成 {op: [(bytes, algo, proto, channels)]}。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ALGO_RE.search(line)
            if not m:
                continue
            name, nbytes, algo, proto, lo, hi = m.groups()
            op = op_map.get(name.lower())
            if not op:
                continue
            out.setdefault(op, {})[int(nbytes)] = {
                "algo": algo, "proto": proto,
                "channels": int(hi) - int(lo) + 1,
            }
    return out


def run_B(out_dir, world_size, gpus, tag, nccl_algo, nccl_proto, sizes, ops=None):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(29900 + abs(hash(tag)) % 90)
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "TUNING,INIT"
    env["NCCL_DEBUG_FILE"] = os.path.join(out_dir, "nccl_debug_%h_%p.log")
    if nccl_algo:
        env["NCCL_ALGO"] = nccl_algo
    if nccl_proto:
        env["NCCL_PROTO"] = nccl_proto
    raw_path = os.path.join(out_dir, "sweep_raw.json")
    log = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8")
    cmd = [sys.executable, os.path.abspath(__file__), "internal-sweep",
           "--world-size", str(world_size), "--out", raw_path,
           "--sizes", ",".join(str(s) for s in sizes)]
    if ops:
        cmd += ["--ops", ",".join(ops)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    log.close()
    if p.returncode != 0:
        raise RuntimeError(f"sweep 失败，见 {os.path.join(out_dir, 'run.log')}")

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    op_map = {"allreduce": "all_reduce", "allgather": "all_gather",
              "reducescatter": "reduce_scatter", "broadcast": "broadcast"}
    algo_sel = {}
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("nccl_debug_"):
            for op, d in parse_algo_log(os.path.join(out_dir, name), op_map).items():
                algo_sel.setdefault(op, {}).update(d)

    result = {
        "tag": tag, "world_size": world_size, "gpus": gpus,
        "nccl_algo_env": nccl_algo, "nccl_proto_env": nccl_proto,
        "ops": ops or OPS,
        "wall_s": wall,
        "points": [],
        "algo_selection": {op: {str(k): v for k, v in sorted(d.items())}
                           for op, d in algo_sel.items()},
    }
    for pt in raw["points"]:
        row = dict(pt)
        sel = algo_sel.get(pt["op"], {}).get(pt["per_rank_bytes"])
        row["selected"] = sel
        n = world_size
        if "t_s_max_rank" in pt:
            row["busbw_GBps"] = busbw_factor(pt["op"], n) * pt["per_rank_bytes"] / pt["t_s_max_rank"] / 1e9
        result["points"].append(row)
    with open(os.path.join(out_dir, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[B:{tag}] N={world_size} gpus={gpus} "
          f"algo_env={nccl_algo} proto_env={nccl_proto} wall={wall:.1f}s")
    for pt in result["points"]:
        if "error" in pt:
            print(f"   {pt['op']:<15} {human(pt['per_rank_bytes']):>8} ERROR {pt['error'][:60]}")
            continue
        sel = pt["selected"] or {}
        print(f"   {pt['op']:<15} {human(pt['per_rank_bytes']):>8} "
              f"t={pt['t_s_max_rank'] * 1e3:9.3f} ms  "
              f"busbw={pt['busbw_GBps']:6.2f} GB/s  "
              f"{sel.get('algo', '-'):<5}/{sel.get('proto', '-'):<6} "
              f"ch={sel.get('channels', '-')}")
    return result


# ==========================================================================
# 重叠（任务 C）
# ==========================================================================

def run_C(rank, nranks, out_dir, gemm_n, mb):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29950")
    dist.init_process_group("nccl", rank=rank, world_size=nranks,
                            timeout=timedelta(seconds=180))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    a = torch.randn(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(gemm_n, gemm_n, dtype=torch.bfloat16, device=dev)
    buf = torch.ones(mb * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    # 竞态用例需要一个明显短于通信的计算，才能让默认流先于 all_reduce 结束
    small_n = 2048
    as_ = torch.randn(small_n, small_n, dtype=torch.bfloat16, device=dev)
    bs_ = torch.randn(small_n, small_n, dtype=torch.bfloat16, device=dev)
    cs_ = torch.empty(small_n, small_n, dtype=torch.bfloat16, device=dev)
    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    flops = 2 * gemm_n ** 3

    def measure(fn, iters=10):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iters)]
        for s, e in ev:
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) / 1e3 for s, e in ev]
        return statistics.median(ts)

    def pre():
        """让两条工作流都排在上一轮计时结束之后，避免跨轮重叠污染区间。"""
        cur = torch.cuda.current_stream()
        s_comm.wait_stream(cur)
        s_comp.wait_stream(cur)

    def comm_only():
        pre()
        with torch.cuda.stream(s_comm):
            dist.all_reduce(buf)
        torch.cuda.current_stream().wait_stream(s_comm)

    def gemm_only():
        pre()
        with torch.cuda.stream(s_comp):
            torch.mm(a, b, out=c)
        torch.cuda.current_stream().wait_stream(s_comp)

    def serial():
        # 计算依赖通信结果：没有任何可重叠区间
        pre()
        with torch.cuda.stream(s_comm):
            dist.all_reduce(buf)
        s_comp.wait_stream(s_comm)
        with torch.cuda.stream(s_comp):
            torch.mm(a, b, out=c)
        torch.cuda.current_stream().wait_stream(s_comp)

    def overlap_independent():
        # 通信与计算互不依赖，两条流最后都汇回默认流
        pre()
        with torch.cuda.stream(s_comm):
            dist.all_reduce(buf)
        with torch.cuda.stream(s_comp):
            torch.mm(a, b, out=c)
        cur = torch.cuda.current_stream()
        cur.wait_stream(s_comm)
        cur.wait_stream(s_comp)
        return float(buf[0].item())

    def overlap_missing_wait():
        # 漏掉对通信流的等待：计时看起来更快，但读到的 buffer 是竞态结果
        pre()
        with torch.cuda.stream(s_comm):
            dist.all_reduce(buf)
        with torch.cuda.stream(s_comp):
            torch.mm(as_, bs_, out=cs_)
        torch.cuda.current_stream().wait_stream(s_comp)
        return float(buf[0].item())

    expected_buf = float(nranks)
    t_comm = measure(comm_only)
    t_gemm = measure(gemm_only)
    t_serial = measure(serial)
    t_over = measure(overlap_independent)

    def value_check(fn, n=20):
        """每次先把 buffer 重置为 1.0 再跑一次，期望值恒为 rank 数。"""
        out = []
        for _ in range(n):
            buf.fill_(1.0)
            torch.cuda.synchronize()
            out.append(fn())
            torch.cuda.synchronize()
        return out

    vals = value_check(overlap_independent)
    bad_vals = value_check(overlap_missing_wait)

    res = {
        "rank": rank, "world_size": nranks, "gemm_n": gemm_n, "comm_MiB": mb,
        "t_comm_s": t_comm, "t_gemm_s": t_gemm, "t_serial_s": t_serial,
        "t_overlap_independent_s": t_over,
        "gemm_TFLOPs_alone": flops / t_gemm / 1e12,
        "comm_GBps_alone": (mb * 1024 * 1024) / t_comm / 1e9,
        "gemm_TFLOPs_in_overlap": flops / t_over / 1e12,
        "comm_overlap_effective_GBps": (mb * 1024 * 1024) / t_over / 1e9,
        "overlap_gain_vs_serial": (t_serial - t_over) / t_serial,
        "theoretical_overlap_ceiling_s": min(t_comm, t_gemm),
        "overlap_ceiling_achieved": (t_serial - t_over) / min(t_comm, t_gemm),
        "expected_buf_value": expected_buf,
        "overlap_correct_values": vals,
        "missing_wait_values": bad_vals,
        "overlap_all_correct": all(abs(v - expected_buf) < 1e-6 for v in vals),
        "missing_wait_all_correct": all(abs(v - expected_buf) < 1e-6 for v in bad_vals),
    }
    gathered = [None] * nranks
    dist.all_gather_object(gathered, res)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "overlap.json"), "w", encoding="utf-8") as f:
            json.dump({"rank0": gathered[0], "ranks": gathered}, f,
                      ensure_ascii=False, indent=2)
        r = gathered[0]
        print(f"[C] GEMM {gemm_n}^3 bf16 = {flops / 1e9:.1f} GFLOP，通信 {mb} MiB")
        print(f"    通信单独 {r['t_comm_s'] * 1e3:.3f} ms ({r['comm_GBps_alone']:.2f} GB/s)  "
              f"计算单独 {r['t_gemm_s'] * 1e3:.3f} ms ({r['gemm_TFLOPs_alone']:.1f} TFLOP/s)")
        print(f"    串行 {r['t_serial_s'] * 1e3:.3f} ms  重叠 {r['t_overlap_independent_s'] * 1e3:.3f} ms  "
              f"收益 {(r['overlap_gain_vs_serial']) * 100:.1f}%  "
              f"理论上限 {r['theoretical_overlap_ceiling_s'] * 1e3:.3f} ms "
              f"（达成 {r['overlap_ceiling_achieved'] * 100:.0f}%）")
        print(f"    重叠中计算 {r['gemm_TFLOPs_in_overlap']:.1f} TFLOP/s、"
              f"通信有效 {(mb * 1024 * 1024) / r['t_overlap_independent_s'] / 1e9:.2f} GB/s")
        print(f"    值检查：正确重叠 {r['overlap_all_correct']}"
              f"（{sorted(set(round(v, 3) for v in r['overlap_correct_values']))}）  "
              f"漏等通信流 {r['missing_wait_all_correct']}"
              f"（{sorted(set(round(v, 3) for v in r['missing_wait_values']))[:5]}，期望 {r['expected_buf_value']}）")
    dist.barrier()
    dist.destroy_process_group()


# ==========================================================================
# 能力矩阵（任务 D）
# ==========================================================================

def run_legality(out_dir, world_size, gpus, size):
    """逐组合试跑，记录哪些 (algo, proto, op) 在当前平台上是合法的。

    强制 NCCL_ALGO/NCCL_PROTO 时，非法组合会在 NCCL 内部直接报
    「no algorithm/protocol available for function X」，而不是静默回退。
    """
    os.makedirs(out_dir, exist_ok=True)
    combos = []
    for algo in ("", "Ring", "Tree"):
        for op in OPS:
            if op == "sendrecv":
                continue
            combos.append({"algo": algo or "auto", "proto": "auto", "op": op})
    for proto in ("", "Simple", "LL", "LL128"):
        combos.append({"algo": "auto", "proto": proto or "auto", "op": "all_reduce"})
    rows = []
    for c in combos:
        tag = f"{c['op']}__{c['algo']}__{c['proto']}"
        d = os.path.join(out_dir, tag)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(30100 + len(rows))
        if c["algo"] != "auto":
            env["NCCL_ALGO"] = c["algo"]
        if c["proto"] != "auto":
            env["NCCL_PROTO"] = c["proto"]
        cmd = [sys.executable, os.path.abspath(__file__), "internal-sweep",
               "--world-size", str(world_size), "--out", os.path.join(d, "raw.json"),
               "--sizes", str(size), "--ops", c["op"]]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        err = ""
        if os.path.exists(os.path.join(d, "raw.json")):
            with open(os.path.join(d, "raw.json"), encoding="utf-8") as f:
                raw = json.load(f)
            pt = raw["points"][0]
            ok = "error" not in pt
            if not ok:
                err = pt["error"].split("\n")[-1][:180]
            t = pt.get("t_s_max_rank")
        else:
            ok, t = False, None
            err = p.stderr.strip().splitlines()[-1][:180] if p.stderr else "no output"
        rows.append({"op": c["op"], "algo": c["algo"], "proto": c["proto"],
                     "ok": ok, "t_s": t, "error": err})
        print(f"  {c['op']:<15} algo={c['algo']:<5} proto={c['proto']:<7} "
              f"{'OK' if ok else 'FAIL'} {'' if ok else err[:110]}")
    with open(os.path.join(out_dir, "legality.json"), "w", encoding="utf-8") as f:
        json.dump({"world_size": world_size, "size_bytes": size, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    return rows


def run_D(out_dir, world_size=2, gpus="0,1"):
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    env = dict(os.environ)
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,NET,GRAPH,TUNING"
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29999"
    log_path = os.path.join(out_dir, "nccl_capability.log")
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "internal-sweep", "--world-size", str(world_size),
                            "--out", os.path.join(out_dir, "probe.json"),
                            "--sizes", str(4 << 10), "--ops", "all_reduce"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    info["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["world_size"] = world_size
    info["gpus"] = gpus
    info["subprocess_returncode"] = p.returncode
    lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    keys = ["NET/IB", "NET/Socket", "NET/Plugin", "GIN/Plugin", "NVLS", "CollNet",
            "P2P", "MNNVL", "SHARP", "symmetric", "Symmetric", "Window",
            "Channel", "Trees", "TUNER", "Algo "]
    info["nccl_lines"] = [l for l in lines if any(k in l for k in keys)][:60]
    info["tuning_table"] = [l.split("NCCL INFO", 1)[-1].strip()
                            for l in lines if ("Algorithm" in l or "Protocol" in l
                                               or "AllReduce |" in l
                                               or "AllGather |" in l
                                               or "Broadcast |" in l
                                               or "ReduceScatter |" in l)][:24]
    info["algo_selections"] = [l.split("NCCL INFO", 1)[-1].strip()
                               for l in lines if "-> Algo" in l][:10]
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                              text=True, timeout=30).stdout
        info["topo_head"] = topo.splitlines()[:12]
    except Exception as e:
        info["topo_head"] = [f"nvidia-smi topo 失败: {e}"]
    import glob
    info["infiniband_devices"] = sorted(os.path.basename(p) for p in
                                        glob.glob("/sys/class/infiniband/*"))
    info["dev_infiniband_exists"] = os.path.exists("/dev/infiniband")
    info["env_nccl"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    with open(os.path.join(out_dir, "capability.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[D] NCCL {info['nccl_version']} torch {info['torch']} cuda {info['cuda']} "
          f"N={world_size} gpus={gpus}")
    for l in info["nccl_lines"]:
        print("   ", l.strip()[:150])
    print("    IB 设备:", info["infiniband_devices"],
          "/dev/infiniband:", info["dev_infiniband_exists"])
    print("    算法选择:", info["algo_selections"])
    return info


# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A", "B", "C", "D", "legality", "internal-sweep"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--tag", default="auto")
    ap.add_argument("--nccl-algo", default="")
    ap.add_argument("--nccl-proto", default="")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--ops", default="")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--size", type=int, default=1 << 20)
    ap.add_argument("--mb", type=int, default=64)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else SIZES
    ops = [x for x in a.ops.split(",") if x] if a.ops else None

    if a.task == "internal-sweep":
        import torch.multiprocessing as mp
        mp.spawn(internal_sweep, args=(a.world_size, a.out, sizes, ops),
                 nprocs=a.world_size, join=True)
    elif a.task == "B":
        run_B(a.out, a.world_size, a.gpus or "0,1", a.tag, a.nccl_algo,
              a.nccl_proto, sizes, ops)
    elif a.task == "C":
        import torch.multiprocessing as mp
        mp.spawn(run_C, args=(a.world_size, a.out, a.gemm_n, a.mb),
                 nprocs=a.world_size, join=True)
    elif a.task == "D":
        run_D(a.out, a.world_size, a.gpus or "0,1")
    elif a.task == "legality":
        run_legality(a.out, a.world_size, a.gpus or "0,1", a.size)
    elif a.task == "A":
        run_A(a.sweep, a.out)


if __name__ == "__main__":
    sys.exit(main())
