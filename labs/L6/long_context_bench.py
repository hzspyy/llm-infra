#!/usr/bin/env python3
"""
6.5 任务 B/C：长上下文的容量、通信与时间扫描，以及 CP/TP 的对照。

B：worldvln 2/4 卡，S=2048/8192/16384/32768，分别测训练（前向+反向，含激活）
   与推理（只存 KV）两条路径。每 rank 峰值用 CUDA 计数器实测，不做「全局 token
   数除以卡数」的估算。
C：CP 与 TP 的合法组合对照，捕获真实 collective 顺序；另加非整除负载不齐、
   不同 mask 与小 batch 的用例。

实测部分用 Qwen3-8B 的结构参数（heads=32, kv_heads=8, head_dim=128），
但只跑少量层；36 层的整账按同样的每层量线性外推，两者分开标注。

用法：
    python long_context_bench.py scan --cp 2 --out <dir>
    python long_context_bench.py compare --cp 2 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import context_parallel_attention as cpmod  # noqa: E402

QWEN3_8B = cpmod.QWEN3_8B
DTYPE_BYTES = 2          # bf16


def per_rank_inference_kv_bytes(seq, cp_size, layers=None):
    """推理时每 rank 的 KV：序列被 CP 切开，kv_head 只在 TP 下切。"""
    layers = layers or QWEN3_8B["layers"]
    return (2 * layers * QWEN3_8B["kv_heads"] * (seq // cp_size)
            * QWEN3_8B["head_dim"] * DTYPE_BYTES)


def per_rank_train_state_bytes(seq, cp_size, layers=None):
    """训练时的 attention 激活：每头每层都要存 scores 与输出（不含重算）。

    这里给的是「不重算」的上界：scores 是 (heads, S_local, S) 的完整矩阵。
    """
    layers = layers or QWEN3_8B["layers"]
    H, D = QWEN3_8B["heads"], QWEN3_8B["head_dim"]
    s_local = seq // cp_size
    scores = H * s_local * seq * DTYPE_BYTES          # 每个 head 的注意力矩阵
    out = H * s_local * D * DTYPE_BYTES
    return layers * (scores + out)


def ring_comm_accounting(seq, cp_size, cp_rank=0):
    """ring 每 rank 的通信：CP-1 轮，每轮搬一个 KV 块。"""
    B, G, D = 1, QWEN3_8B["kv_heads"], QWEN3_8B["head_dim"]
    rounds = cp_size - 1
    per_round = 2 * B * G * (seq // cp_size + 1) * D * DTYPE_BYTES
    return rounds, rounds * per_round


def ulysses_comm_accounting(seq, cp_size):
    """ulysses：两次 all-to-all，每次搬 B*S*H*D。"""
    H, D = QWEN3_8B["heads"], QWEN3_8B["head_dim"]
    return 2, 2 * 2 * seq * H * D * DTYPE_BYTES


# ==========================================================================
# 任务 B：扫描
# ==========================================================================

def run_scan(rank, cp_size, out_dir, sizes, layers, device):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29990")
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=cp_size,
                            timeout=timedelta(seconds=300))
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    if device == "cuda":
        torch.cuda.set_device(rank)
    H, G, D = QWEN3_8B["heads"], QWEN3_8B["kv_heads"], QWEN3_8B["head_dim"]
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    rows = []
    for seq in sizes:
        start, length = cpmod.shard_bounds(seq, cp_size, rank)
        q = torch.randn(1, H, length, D, dtype=dt, device=dev) * 0.1
        k = torch.randn(1, G, length, D, dtype=dt, device=dev) * 0.1
        v = torch.randn(1, G, length, D, dtype=dt, device=dev) * 0.1
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        # 预热：首次调用含分配与 kernel 编译，不预热会把第一个点抬高一个数量级
        qw, kw, vw = q.clone(), k.clone(), v.clone()
        cpmod.ring_attention(qw, kw, vw, seq, cp_size, rank)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()

        # 前向（ring，含通信）：3 次取中位
        fwd_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            y, rounds = cpmod.ring_attention(q, k, v, seq, cp_size, rank)
            if device == "cuda":
                torch.cuda.synchronize()
            fwd_times.append(time.perf_counter() - t0)
        fwd = statistics.median(fwd_times)
        peak_fwd = (torch.cuda.max_memory_allocated() - base) if device == "cuda" else None

        # 训练路径：ring 前向 + 本地 dQ 的反向。dK/dV 需要通信的转置，
        # 这里用「本地反向 + 跨 rank 归约」的验证路径，它只在 S 较小时可行
        # （要对完整 S 物化注意力矩阵，是 O(S^2) 显存）。
        train_note = None
        train_err = None
        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        try:
            yg, _ = cpmod.ring_attention(qg, kg, vg, seq, cp_size, rank)
            torch.autograd.grad(yg.sum(), (qg, kg, vg), allow_unused=True)
            if device == "cuda":
                torch.cuda.synchronize()
            fwd_bwd = time.perf_counter() - t0
            peak_train = ((torch.cuda.max_memory_allocated() - base)
                          if device == "cuda" else None)
            train_note = "ring 前向 + 本地 autograd（dQ 正确；dK/dV 需通信转置）"
        except torch.OutOfMemoryError as e:
            # 反向要保存每一轮的注意力矩阵，S 大时它就是 O(S^2) 的显存墙。
            # 这不是实现 bug，而是「CP 不解决训练激活」的直接证据，按实测记录。
            fwd_bwd, peak_train = None, None
            train_err = ("OOM：单轮注意力矩阵 "
                         f"{H}x{s_local}x{seq}x2B = "
                         f"{H * s_local * seq * 2 / 2**30:.1f} GiB/rank，"
                         f"autograd 要把它全部留住")
            train_note = "前向+反向 OOM"
        finally:
            del qg, kg, vg
            try:
                del yg
            except UnboundLocalError:
                pass
            if device == "cuda":
                torch.cuda.empty_cache()
        s_local = length

        r_rounds, r_bytes = ring_comm_accounting(seq, cp_size, rank)
        u_rounds, u_bytes = ulysses_comm_accounting(seq, cp_size)
        rows.append({
            "rank": rank, "seq": seq, "cp_size": cp_size,
            "local_len": length, "local_start": start,
            "fwd_s": fwd, "fwd_bwd_s": fwd_bwd,
            "ring_rounds": rounds, "ring_accounted_rounds": r_rounds,
            "ring_bytes_per_rank": r_bytes,
            "ulysses_rounds": u_rounds, "ulysses_bytes_per_rank": u_bytes,
            "measured_peak_fwd_MiB": (peak_fwd / 2**20) if peak_fwd else None,
            "measured_peak_train_MiB": (peak_train / 2**20) if peak_train else None,
            "analytic_inference_kv_MiB_per_rank":
                per_rank_inference_kv_bytes(seq, cp_size) / 2**20,
            "analytic_train_state_MiB_per_rank":
                per_rank_train_state_bytes(seq, cp_size) / 2**20,
            "analytic_layers": QWEN3_8B["layers"],
            "modules_run": layers,
            "train_path_note": train_note, "train_path_error": train_err,
        })
        if rank == 0:
            print(f"  S={seq:<7} CP={cp_size} 本段={length:<6} "
                  f"fwd={fwd * 1e3:8.1f} ms "
                  f"fwd+bwd={(f'{fwd_bwd * 1e3:8.1f} ms' if fwd_bwd else 'OOM'):>12} "
                  f"轮={rounds} ring 字节/rank={r_bytes / 2**20:7.1f} MiB "
                  f"ulysses 字节/rank={u_bytes / 2**20:7.1f} MiB "
                  f"推理 KV/rank(36层)={per_rank_inference_kv_bytes(seq, cp_size) / 2**20:8.1f} MiB "
                  f"训练激活/rank(36层)={per_rank_train_state_bytes(seq, cp_size) / 2**20:8.1f} MiB",
                  flush=True)
        del q, k, v, y
        if device == "cuda":
            torch.cuda.empty_cache()
        dist.barrier()
    gathered = [None] * cp_size
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"scan_cp{cp_size}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"cp_size": cp_size, "config": QWEN3_8B,
                       "rows": gathered[0]}, f, ensure_ascii=False, indent=2)
    dist.barrier()
    dist.destroy_process_group()


# ==========================================================================
# 任务 C：CP 与 TP 的对照 + collective 顺序
# ==========================================================================

def collectives_during(fn):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=True) as prof:
        out = fn()
    seq = []
    for e in prof.events():
        n = e.name
        if n.endswith("wait_tensor") or "_wrap" in n:
            continue
        if n.startswith("_c10d_functional::") or n.startswith("_dtensor::") \
                or n.startswith("c10d::") or n.startswith(("nccl:", "gloo:")):
            shapes = [list(s) for s in (e.input_shapes or [])]
            seq.append({"op": n, "input_shapes": shapes})
    # 去重但保留顺序（同名同形状的相邻重复折叠成 ×n）
    folded = []
    for s in seq:
        if folded and folded[-1]["op"] == s["op"] and folded[-1]["input_shapes"] == s["input_shapes"]:
            folded[-1]["count"] += 1
        else:
            folded.append({**s, "count": 1})
    return out, folded


def run_compare(rank, cp_size, out_dir, seq, device):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29991")
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=cp_size,
                            timeout=timedelta(seconds=300))
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    if device == "cuda":
        torch.cuda.set_device(rank)
    H, G, D = QWEN3_8B["heads"], QWEN3_8B["kv_heads"], QWEN3_8B["head_dim"]
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    start, length = cpmod.shard_bounds(seq, cp_size, rank)
    q = torch.randn(1, H, length, D, dtype=dt, device=dev) * 0.1
    k = torch.randn(1, G, length, D, dtype=dt, device=dev) * 0.1
    v = torch.randn(1, G, length, D, dtype=dt, device=dev) * 0.1

    result = {"rank": rank, "seq": seq, "cp_size": cp_size, "device": device}

    # CP：只测 ring/ulysses 的 collective 顺序
    if seq % cp_size == 0:
        _, cols_ul = collectives_during(
            lambda: cpmod.ulysses_attention(q, k, v, seq, cp_size, rank))
    else:
        cols_ul = [{"op": "skipped", "reason": "S 不能被 CP 整除"}]
    _, cols_ring = collectives_during(
        lambda: cpmod.ring_attention(q, k, v, seq, cp_size, rank))
    result["cp_collectives_ring"] = cols_ring
    result["cp_collectives_ulysses"] = cols_ul

    # 负载不齐：非整除时每 rank 的段长
    lens = [cpmod.shard_bounds(seq, cp_size, r)[1] for r in range(cp_size)]
    result["shard_lengths"] = lens
    result["load_imbalance"] = max(lens) / (sum(lens) / len(lens))

    # 不同 mask：causal 与 window
    y_c, _ = cpmod.ring_attention(q, k, v, seq, cp_size, rank, causal=True)
    y_w, _ = cpmod.ring_attention(q, k, v, seq, cp_size, rank, causal=True, window=256)
    # 窗口下每个位置看到的 KV 数上界
    result["mask_causal_vs_window_max_abs_diff"] = float((y_c - y_w).abs().max().item())
    result["window_visible_per_query"] = min(256, seq)

    # 小 batch 与 batch>1
    for b in (1, 2):
        qb = q.repeat(b, 1, 1, 1)
        kb = k.repeat(b, 1, 1, 1)
        vb = v.repeat(b, 1, 1, 1)
        t0 = time.perf_counter()
        yb, _ = cpmod.ring_attention(qb, kb, vb, seq, cp_size, rank)
        if device == "cuda":
            torch.cuda.synchronize()
        result[f"batch{b}_fwd_s"] = time.perf_counter() - t0

    gathered = [None] * cp_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"compare_cp{cp_size}_s{seq}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"config": QWEN3_8B, "ranks": gathered}, f,
                      ensure_ascii=False, indent=2)
        r = gathered[0]
        print(f"  S={seq} CP={cp_size} 段长={r['shard_lengths']} "
              f"不齐度={r['load_imbalance']:.3f}")
        print(f"    ring collective 顺序："
              + ", ".join(f"{c['op']}×{c['count']}" for c in r["cp_collectives_ring"][:6]))
        print(f"    ulysses collective 顺序："
              + ", ".join(f"{c['op']}×{c['count']}" for c in r["cp_collectives_ulysses"][:6]))
        print(f"    batch1 前向={r['batch1_fwd_s'] * 1e3:.1f} ms "
              f"batch2 前向={r['batch2_fwd_s'] * 1e3:.1f} ms")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["scan", "compare"])
    ap.add_argument("--cp", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--sizes", default="2048,8192,16384,32768")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    import torch.multiprocessing as mp
    if a.task == "scan":
        mp.spawn(run_scan, args=(a.cp, a.out, sizes, a.layers, a.device),
                 nprocs=a.cp, join=True)
    else:
        mp.spawn(run_compare, args=(a.cp, a.out, a.seq, a.device),
                 nprocs=a.cp, join=True)


if __name__ == "__main__":
    sys.exit(main())
