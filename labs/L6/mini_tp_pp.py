#!/usr/bin/env python3
"""
6.2 任务 A：手写张量并行与流水并行，并与单卡结果逐元素对拍。

覆盖：
  1. 列并行线性层（按输出维切权重，前向零通信）
  2. 行并行线性层（按输入维切权重，输出是部分和，需要 all_reduce）
  3. 由 1+2 组成的 MLP（Megatron 式：列并行 → 激活 → 行并行）
  4. 多头注意力的 head 切分：q/k/v 列并行，输出投影行并行
  5. 两阶段流水：按层切分，微批之间用 P2P 传递激活
  6. 整除条件表：TP 度必须整除 heads / kv_heads / intermediate / vocab

使用：
    python mini_tp_pp.py A --world-size 2 --device cpu --out <dir>
    python mini_tp_pp.py PP --world-size 2 --device cpu --out <dir> --micro 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# Qwen3-1.7B 的结构参数（与 checkpoint 的 config.json 一致）
QWEN3_1_7B = {"hidden": 2048, "inter": 6144, "heads": 16, "kv_heads": 8,
              "head_dim": 128, "vocab": 151936, "layers": 28}


def env_pins():
    info = {"host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    return info


def write_manifest(out_dir, task, world_size, extra=None):
    payload = {
        "task": task, "world_size": world_size,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env_pins(),
        "model_shape": QWEN3_1_7B,
        "input_spec": "FP64 随机权重与激活，seed 固定；整数索引与整除条件用同一组结构参数",
        "tolerance": "与单卡 FP64 参照逐元素比较，容差 1e-10",
        "source_pins": {
            "vllm_linear": "vllm/model_executor/layers/linear.py",
            "vllm_parallel_state": "vllm/distributed/parallel_state.py",
        },
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==========================================================================
# 权重与切分
# ==========================================================================

def make_weights(shape_cfg, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, I = shape_cfg["hidden"], shape_cfg["inter"]
    HD = shape_cfg["heads"] * shape_cfg["head_dim"]
    KVD = shape_cfg["kv_heads"] * shape_cfg["head_dim"]
    W = {
        "W1": torch.randn(I, H, generator=g, dtype=torch.float64) * 0.02,   # 门控/上投影
        "W2": torch.randn(H, I, generator=g, dtype=torch.float64) * 0.02,   # 下投影
        "Wq": torch.randn(HD, H, generator=g, dtype=torch.float64) * 0.02,
        "Wk": torch.randn(KVD, H, generator=g, dtype=torch.float64) * 0.02,
        "Wv": torch.randn(KVD, H, generator=g, dtype=torch.float64) * 0.02,
        "Wo": torch.randn(H, HD, generator=g, dtype=torch.float64) * 0.02,
    }
    return {k: v.to(device) for k, v in W.items()}


def shard_rows(W, tp_rank, tp_size, dim=0):
    """按 dim 均分权重，返回本 rank 的那一块。切不动时直接报错，不静默变换。"""
    n = W.shape[dim]
    if n % tp_size:
        raise ValueError(f"维度 {dim} 长度 {n} 不能被 TP={tp_size} 整除")
    per = n // tp_size
    return W.narrow(dim, tp_rank * per, per).contiguous()


def divisibility_table(cfg, tp_sizes):
    rows = []
    for tp in tp_sizes:
        row = {"tp": tp}
        for name, key in [("heads", "heads"), ("kv_heads", "kv_heads"),
                          ("intermediate", "inter"), ("vocab", "vocab"),
                          ("hidden", "hidden")]:
            n = cfg[key]
            row[name] = {"n": n, "ok": n % tp == 0, "per_rank": n // tp if n % tp == 0 else None}
        row["all_ok"] = all(row[k]["ok"] for k in
                            ("heads", "kv_heads", "intermediate", "vocab", "hidden"))
        rows.append(row)
    return rows


# ==========================================================================
# 并行算子的最小实现
# ==========================================================================

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / (2 ** 0.5)))


def col_parallel_linear(x, W_local):
    """按输出维切权重：每个 rank 独立算自己那段输出，前向不需要通信。"""
    return x @ W_local.t()


def row_parallel_linear(x_local, W_local, group=None):
    """按输入维切权重：输入必须按最后一维切好，各 rank 得到部分和，需要 all_reduce。"""
    partial = x_local @ W_local.t()
    dist.all_reduce(partial, group=group)
    return partial


def mlp_tp(x, W1_local, W2_local, group=None):
    h = gelu(col_parallel_linear(x, W1_local))
    return row_parallel_linear(h, W2_local, group=group)


def attention_tp(x, Wq_l, Wk_l, Wv_l, Wo_l, cfg, tp_size, group=None, causal=True):
    """head 维切分：q/k/v 列并行，输出投影行并行。"""
    S, H = x.shape
    hd = cfg["head_dim"]
    q = col_parallel_linear(x, Wq_l).view(S, cfg["heads"] // tp_size, hd)
    k = col_parallel_linear(x, Wk_l).view(S, cfg["kv_heads"] // tp_size, hd)
    v = col_parallel_linear(x, Wv_l).view(S, cfg["kv_heads"] // tp_size, hd)
    # GQA：每个 kv head 被 heads/kv_heads 个 q head 共享
    rep = (cfg["heads"] // tp_size) // (cfg["kv_heads"] // tp_size)
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    # 排成 (heads, S, hd) 之后再做 score：直接对 (S, heads, hd) 转置会把 head 维当 batch
    qp, kp, vp = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    att = (qp @ kp.transpose(1, 2)) / (hd ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1).unsqueeze(0)
        att = att.masked_fill(mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    o = (att @ vp).permute(1, 0, 2).reshape(S, -1)
    return row_parallel_linear(o, Wo_l, group=group)


def reference_mlp(x, W):
    return gelu(x @ W["W1"].t()) @ W["W2"].t()


def reference_attention(x, W, cfg, causal=True):
    S, H = x.shape
    hd = cfg["head_dim"]
    q = (x @ W["Wq"].t()).view(S, cfg["heads"], hd)
    k = (x @ W["Wk"].t()).view(S, cfg["kv_heads"], hd)
    v = (x @ W["Wv"].t()).view(S, cfg["kv_heads"], hd)
    rep = cfg["heads"] // cfg["kv_heads"]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    qp, kp, vp = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    att = (qp @ kp.transpose(1, 2)) / (hd ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1).unsqueeze(0)
        att = att.masked_fill(mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return (att @ vp).permute(1, 0, 2).reshape(S, -1) @ W["Wo"].t()


def run_A(rank, world_size, out_dir, device="cpu"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29960")
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    cfg = QWEN3_1_7B
    tp = world_size
    S = 64
    torch.manual_seed(0)
    W = make_weights(cfg, seed=0, device=dev)
    x = (torch.randn(S, cfg["hidden"], dtype=torch.float64, device=dev) * 0.1)
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    out = {"rank": rank, "tp": tp, "device": device, "backend": backend,
           "seq": S, "divisibility": divisibility_table(cfg, [1, 2, 4, 8, 16])}

    def split(tag):
        t0 = time.perf_counter()
        if device == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        val = fn()
        dist.barrier()
        if device == "cuda":
            torch.cuda.synchronize()
        return val, (time.perf_counter() - t0)

    # ---- MLP：列并行 + 行并行 ----
    W1_l = shard_rows(W["W1"], rank, tp, dim=0)     # 输出维切开
    W2_l = shard_rows(W["W2"], rank, tp, dim=1)     # 输入维切开
    def fn():
        return mlp_tp(x, W1_l, W2_l)
    y_mlp, t_mlp = split("mlp")
    ref_mlp = reference_mlp(x, W)
    out["mlp"] = {
        "local_hidden_shape": list((gelu(x @ W1_l.t())).shape),
        "y_shape": list(y_mlp.shape),
        "max_abs_diff": float((y_mlp - ref_mlp).abs().max().item()),
        "matches_reference": bool(torch.allclose(y_mlp, ref_mlp, atol=1e-10)),
        "wall_s": t_mlp,
    }

    # ---- attention：q/k/v 列并行，输出投影行并行 ----
    Wq_l = shard_rows(W["Wq"], rank, tp, dim=0)
    Wk_l = shard_rows(W["Wk"], rank, tp, dim=0)
    Wv_l = shard_rows(W["Wv"], rank, tp, dim=0)
    Wo_l = shard_rows(W["Wo"], rank, tp, dim=1)
    def fn():
        return attention_tp(x, Wq_l, Wk_l, Wv_l, Wo_l, cfg, tp)
    y_att, t_att = split("attn")
    ref_att = reference_attention(x, W, cfg)
    out["attention"] = {
        "local_q_shape": list((x @ Wq_l.t()).shape),
        "y_shape": list(y_att.shape),
        "max_abs_diff": float((y_att - ref_att).abs().max().item()),
        "matches_reference": bool(torch.allclose(y_att, ref_att, atol=1e-10)),
        "wall_s": t_att,
    }

    # ---- 通信量公式核对：行并行每次 all_reduce S×H 个 FP64 ----
    out["comm_accounting"] = {
        "mlp_all_reduce_bytes": S * cfg["hidden"] * 8,
        "attn_all_reduce_bytes": S * cfg["hidden"] * 8,
        "note": "列并行前向零通信；行并行的部分和必须 all_reduce",
    }

    gathered = [None] * world_size
    dist.all_gather_object(gathered, out)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "mini_tp.json"), "w", encoding="utf-8") as f:
            json.dump({"ranks": gathered}, f, ensure_ascii=False, indent=2)
        print(f"[A] TP={tp} device={device}")
        for name in ("mlp", "attention"):
            r0 = gathered[0][name]
            print(f"    {name:<10} 局部分片 shape={r0.get('local_q_shape') or r0['local_hidden_shape']} "
                  f"输出={r0['y_shape']} max|diff|={r0['max_abs_diff']:.3e} "
                  f"一致={r0['matches_reference']}")
        print("    整除条件：")
        for row in gathered[0]["divisibility"]:
            flags = " ".join(f"{k}={'Y' if row[k]['ok'] else 'N'}"
                             for k in ("heads", "kv_heads", "intermediate", "vocab", "hidden"))
            print(f"      TP={row['tp']:<2} {flags}  全部可整除={row['all_ok']}")
        write_manifest(out_dir, "A", world_size, extra={"tp": tp, "device": device})
    dist.barrier()
    dist.destroy_process_group()


# ==========================================================================
# 两阶段流水
# ==========================================================================

def run_PP(rank, world_size, out_dir, micro, device="cpu"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29961")
    backend = "nccl" if device == "cuda" else "gloo"
    if world_size != 2:
        raise SystemExit("两阶段流水用 world_size=2")
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    cfg = QWEN3_1_7B
    S, H, I = 64, 512, 2048
    g = torch.Generator(device="cpu").manual_seed(1)
    WA = torch.randn(I, H, generator=g, dtype=torch.float64).to(dev)
    WB = torch.randn(H, I, generator=g, dtype=torch.float64).to(dev)

    def stage_a(x):
        return gelu(x @ WA.t())

    def stage_b(h):
        return h @ WB.t()

    # 微批输入必须由两 rank 共用的 generator 产生，否则两端参照的不是同一批数据
    xs = [(torch.randn(S, H, generator=g, dtype=torch.float64) * 0.1).to(dev)
          for _ in range(micro)]
    ref = [stage_b(stage_a(x)) for x in xs]

    comp0, send0, comp1, wait1 = [], [], [], []
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for m in range(micro):
        if rank == 0:
            c0 = time.perf_counter()
            h = stage_a(xs[m])
            comp0.append(time.perf_counter() - c0)
            s0 = time.perf_counter()
            dist.send(h.contiguous(), dst=1)      # 阻塞式发送，等对端收完
            send0.append(time.perf_counter() - s0)
        else:
            recv = torch.empty(S, I, dtype=torch.float64, device=dev)
            r0 = time.perf_counter()
            dist.recv(recv, src=0)
            wait1.append(time.perf_counter() - r0)
            c1 = time.perf_counter()
            _ = stage_b(recv)
            comp1.append(time.perf_counter() - c1)
    total = time.perf_counter() - t0
    if device == "cuda":
        torch.cuda.synchronize()

    # 逐微批对拍：rank 1 把自己算出的 stage_b 与两端一致的参照比
    if rank == 0:
        results = []
        for m, x in enumerate(xs):
            dist.send(stage_a(x).contiguous(), dst=1)
            results.append(None)
    else:
        results = []
        for m in range(micro):
            recv = torch.empty(S, I, dtype=torch.float64, device=dev)
            dist.recv(recv, src=0)
            results.append(float((stage_b(recv) - ref[m]).abs().max().item()))

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {"rank": rank, "comp0": comp0, "send0": send0,
                                      "comp1": comp1, "wait1": wait1,
                                      "total": total,
                                      "diffs": results if rank == 1 else None})
    if rank == 0:
        stage1 = gathered[1]
        c0 = statistics.fsum(gathered[0]["comp0"])
        s0 = statistics.fsum(gathered[0]["send0"])
        c1 = statistics.fsum(stage1["comp1"])
        w1 = statistics.fsum(stage1["wait1"])
        payload = {
            "micro_batches": micro, "seq": S, "hidden": H, "inter": I,
            "total_s": total,
            "compute_stage0_s": c0, "send_stage0_s": s0,
            "compute_stage1_s": c1, "recv_wait_stage1_s": w1,
            "utilization_compute_only": (c0 + c1) / (2 * total),
            "stage0_idle_s": total - c0 - s0,
            "stage1_idle_s": total - c1,
            "stage1_recv_wait_s": w1,
            "slow_stage_compute_ratio": (c0 / c1) if c1 else None,
            "note": "stage0 计算+发送已占满观测窗口，是瓶颈；stage1 的空闲由等待 stage0 造成。"
                    "亚毫秒级的分项受计时分辨率限制，只用于看趋势",
            "max_abs_diff_vs_reference": max(stage1["diffs"]),
            "all_micro_batches_match": all(d < 1e-10 for d in stage1["diffs"]),
            "rank0_compute_per_micro": gathered[0]["comp0"],
            "rank0_send_per_micro": gathered[0]["send0"],
            "rank1_compute_per_micro": stage1["comp1"],
            "rank1_wait_per_micro": stage1["wait1"],
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "mini_pp.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[PP] 微批={micro} 总时间={total * 1e3:.2f} ms "
              f"stage0 计算 {c0 * 1e3:.2f} + 发送 {s0 * 1e3:.2f} ms / "
              f"stage1 计算 {c1 * 1e3:.2f} + 等收 {w1 * 1e3:.2f} ms  "
              f"计算利用率={payload['utilization_compute_only'] * 100:.1f}%")
        print(f"     stage0 空闲={payload['stage0_idle_s'] * 1e3:.2f} ms "
              f"stage1 空闲={payload['stage1_idle_s'] * 1e3:.2f} ms "
              f"（其中等收 {w1 * 1e3:.2f} ms）；"
              f"逐微批 max|diff|={payload['max_abs_diff_vs_reference']:.3e} "
              f"全部一致={payload['all_micro_batches_match']}")
        write_manifest(out_dir, "PP", world_size, extra={"micro_batches": micro})
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A", "PP"])
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--micro", type=int, default=4)
    a = ap.parse_args()
    import torch.multiprocessing as mp
    if a.task == "A":
        mp.spawn(run_A, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)
    else:
        mp.spawn(run_PP, args=(a.world_size, a.out, a.micro, a.device),
                 nprocs=a.world_size, join=True)


if __name__ == "__main__":
    sys.exit(main())
