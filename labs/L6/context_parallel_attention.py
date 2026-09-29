#!/usr/bin/env python3
"""
6.5 任务 A：手写上下文并行 attention，并与单进程参照对拍值和梯度。

两种换维方案都实现一遍：

  ring   —— 序列切分保持不变，KV 块沿环传递 CP-1 轮，每轮与本地 Q 做一次
            在线 softmax（m/l/O）归并。通信量随 CP 线性增长，但每轮只传 KV。
  ulysses—— 先 all-to-all 把「序列维」换成「head 维」，每个 rank 拿到完整序列
            但只负责 H/CP 个 head，本地做完整 attention，再 all-to-all 换回来。

对拍内容：值、以及 dQ/dK/dV 的梯度。用例覆盖非整除长度、GQA、窗口边界。

用法：
    python context_parallel_attention.py A --cp 2 --device cpu --out <dir>
    python context_parallel_attention.py A --cp 2 --device cuda --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# Qwen3-8B 的结构参数（取自官方 config.json）
QWEN3_8B = {"hidden": 4096, "heads": 32, "kv_heads": 8, "head_dim": 128,
            "layers": 36, "inter": 12288, "vocab": 151936,
            "max_position": 40960, "rope_theta": 1000000}


def env_pins():
    info = {"host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    return info


# ==========================================================================
# 参照实现（单进程、完整序列、FP64）
# ==========================================================================

def make_qkv(batch, seq, heads, kv_heads, head_dim, seed=0, dtype=torch.float64,
             device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, generator=g, dtype=dtype).to(device)
    k = torch.randn(batch, kv_heads, seq, head_dim, generator=g, dtype=dtype).to(device)
    v = torch.randn(batch, kv_heads, seq, head_dim, generator=g, dtype=dtype).to(device)
    return q, k, v


def causal_window_mask(seq_kv, window=None, device="cpu", q_offset=0, seq_q=None):
    """查询行 i 的全局位置是 q_offset + i，键列 j 的全局位置是 j。

    CP 下本地 Q 只是一段（q_offset>0），而 KV 可能是完整序列，
    所以掩码必须用**全局位置**算，不能默认 q 与 k 从同一个 0 开始。
    """
    seq_q = seq_q if seq_q is not None else seq_kv
    q_pos = torch.arange(q_offset, q_offset + seq_q, device=device)
    kv_pos = torch.arange(seq_kv, device=device)
    allow = kv_pos[None, :] <= q_pos[:, None]
    if window:
        allow = allow & (kv_pos[None, :] >= (q_pos[:, None] - window + 1))
    return allow


def repeat_kv(k, heads, kv_heads):
    return k.repeat_interleave(heads // kv_heads, dim=1)


def reference_attention(q, k, v, causal=True, window=None, q_offset=0):
    """单进程 attention；q/k/v 均为 (B, H, S, D) 或 (B, G, S, D)。

    q_offset 让本地 Q 段能对完整 KV 用全局位置做掩码（CP 的本地反向要用）。
    """
    B, H, Sq, D = q.shape
    Skv = k.shape[2]
    G = k.shape[1]
    kk, vv = repeat_kv(k, H, G), repeat_kv(v, H, G)
    scores = q @ kk.transpose(-1, -2) / (D ** 0.5)
    if causal or window:
        mask = causal_window_mask(Skv, window, device=q.device,
                                  q_offset=q_offset, seq_q=Sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return p @ vv


# ==========================================================================
# ring：KV 块沿环传递 + 在线 softmax 归并
# ==========================================================================

def shard_bounds(seq, cp_size, cp_rank):
    """把长度 seq 切成 cp_size 段，前 seq%cp 段各多 1（与 torch.chunk 一致）。"""
    base, rest = divmod(seq, cp_size)
    start = cp_rank * base + min(cp_rank, rest)
    length = base + (1 if cp_rank < rest else 0)
    return start, length


def _ring_pass(send, nxt, prv, recv_shape):
    """把 send 发给 nxt，从 prv 收一块形状为 recv_shape 的块。

    非整除长度下各 rank 的段长不同，接收缓冲必须按对方的段长分配，
    不能用 empty_like(send)。
    """
    out = torch.empty(recv_shape, dtype=send.dtype, device=send.device)
    ops = [dist.P2POp(dist.isend, send.contiguous(), nxt),
           dist.P2POp(dist.irecv, out, prv)]
    for w in dist.batch_isend_irecv(ops):
        w.wait()
    return out


def ring_attention(local_q, local_k, local_v, global_seq, cp_size, cp_rank,
                   causal=True, window=None, group=None):
    """local_q: (B, H, L, D)，local_k/v: (B, G, L, D)，L 为本 rank 的序列段长。

    每一轮：本地 Q 与手里的 KV 块做一次 attention，用在线 softmax 归并进
    运行量 (m, l, O)，然后把 KV 块传给下一个 rank。CP-1 轮之后所有 KV 都见过一遍。
    """
    B, H, L, D = local_q.shape
    G = local_k.shape[1]
    kk = repeat_kv(local_k, H, G)
    vv = repeat_kv(local_v, H, G)

    start_self, len_self = shard_bounds(global_seq, cp_size, cp_rank)
    q_pos = torch.arange(start_self, start_self + len_self, device=local_q.device)

    m = torch.full((B, H, L, 1), float("-inf"), device=local_q.device,
                   dtype=local_q.dtype)
    l = torch.zeros((B, H, L, 1), device=local_q.device, dtype=local_q.dtype)
    O = torch.zeros((B, H, L, D), device=local_q.device, dtype=local_q.dtype)

    cur_k, cur_v = kk, vv
    cur_rank = cp_rank
    rounds = 0
    for step in range(cp_size):
        blk_start, blk_len = shard_bounds(global_seq, cp_size, cur_rank)
        kv_pos = torch.arange(blk_start, blk_start + blk_len, device=local_q.device)
        s = (local_q @ cur_k.transpose(-1, -2)) / (D ** 0.5)
        allow = kv_pos[None, :] <= q_pos[:, None]
        if window:
            allow = allow & (kv_pos[None, :] >= (q_pos[:, None] - window + 1))
        s = s.masked_fill(~allow, float("-inf"))
        blk_max = s.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, blk_max)
        m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        alpha = torch.where(torch.isfinite(m - m_safe), torch.exp(m - m_safe),
                            torch.zeros_like(m))
        p = torch.where(torch.isfinite(s - m_safe), torch.exp(s - m_safe),
                        torch.zeros_like(s))
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        O = O * alpha + p @ cur_v
        m = m_safe
        if step < cp_size - 1:
            nxt = (cp_rank + 1) % cp_size
            prv = (cp_rank - 1) % cp_size
            src_rank = (cur_rank - 1) % cp_size
            _, recv_len = shard_bounds(global_seq, cp_size, src_rank)
            cur_k = _ring_pass(cur_k, nxt, prv, (B, H, recv_len, D))
            cur_v = _ring_pass(cur_v, nxt, prv, (B, H, recv_len, D))
            cur_rank = src_rank
            rounds += 1
    return O / l.clamp_min(1e-20), rounds


# ==========================================================================
# ulysses：all-to-all 把序列维换成 head 维
# ==========================================================================

def _seq_to_head(x, cp_size):
    """(B, h, L, D) -> (B, h/CP, S, D)：把 head 维切成 CP 段，序列维拼成完整长度。

    用 all_to_all 交换的是「本 rank 的序列段 × 各 head 段」：
    rank r 收到的第 j 块是 rank j 在它自己那段序列上、按 rank r 的 head 段切出来的片。
    """
    B, h, L, D = x.shape
    xs = x.view(B, cp_size, h // cp_size, L, D).permute(1, 0, 2, 3, 4)
    chunks = [xs[r].permute(0, 2, 1, 3).contiguous() for r in range(cp_size)]  # (B, L, h/CP, D)
    outs = [torch.empty_like(chunks[0]) for _ in range(cp_size)]
    dist.all_to_all(outs, chunks)
    y = torch.cat(outs, dim=1)                              # (B, S, h/CP, D)
    return y.permute(0, 2, 1, 3).contiguous()               # (B, h/CP, S, D)


def _head_to_seq(y, cp_size):
    """(B, h/CP, S, D) -> (B, h, L, D)：换回序列切分。"""
    B, hl, S, D = y.shape
    L = S // cp_size
    ys = y.permute(0, 2, 1, 3).reshape(B, cp_size, L, hl, D).permute(1, 0, 3, 2, 4)
    chunks = [ys[r].permute(0, 2, 1, 3).contiguous() for r in range(cp_size)]  # (B, L, hl, D)
    outs = [torch.empty_like(chunks[0]) for _ in range(cp_size)]
    dist.all_to_all(outs, chunks)
    out = torch.stack(outs, dim=2)                          # (B, L, CP, hl, D)
    return out.reshape(B, L, cp_size * hl, D).permute(0, 2, 1, 3).contiguous()


def ulysses_attention(local_q, local_k, local_v, global_seq, cp_size, cp_rank,
                      causal=True, window=None, group=None):
    """输入按序列切分；要求 S % CP == 0 且 head / kv_head 都能被 CP 整除。

    这两条不是实现偷懒：all_to_all 要对等分块，非整除长度必须补 padding，
    而 head 数不够分就只能退化成别的方案。做不到就明确报错。
    """
    if global_seq % cp_size:
        raise ValueError(f"S={global_seq} 不能被 CP={cp_size} 整除，"
                         f"Ulysses 需要等长分块（ring 不受此限）")
    H = local_q.shape[1]
    G = local_k.shape[1]
    if H % cp_size or G % cp_size:
        raise ValueError(f"heads={H}, kv_heads={G} 必须能被 CP={cp_size} 整除")
    qh = _seq_to_head(local_q, cp_size)
    kh = _seq_to_head(local_k, cp_size)
    vh = _seq_to_head(local_v, cp_size)
    out = reference_attention(qh, kh, vh, causal=causal, window=window)
    return _head_to_seq(out, cp_size), 2


# ==========================================================================
# CP 的正确梯度：本地反向 + 跨 rank 归约
# ==========================================================================

def cp_gradients(q, k, v, cp_size, cp_rank, window=None):
    """算出本 rank 的 dQ、以及本 rank 那个 KV 块的 dK/dV。

    关键点：dQ 是**局部**的——本 rank 的查询只由本 rank 的 Q 产生，autograd 直接对。
    但 dK/dV 不是：本 rank 的 K/V 块会被**所有** rank 的查询用到，所以它的梯度是
    若干份局部贡献之和。有两种做法：

      (a) 手写通信的转置（ring 的反向环、all-to-all 的逆变换）；
      (b) 把「本地反向得到的每块梯度」跨 rank 归约一次。

    这里用 (b)，它和 (a) 在数学上等价，但不依赖 autograd 能穿过 P2P：
    先把所有 rank 的 K/V 收齐（数据层面），重建为叶子张量，用本地 Q 跑完整
    attention，autograd 得到「本 rank 查询对每个 KV 块的贡献」，
    再 all_reduce 求和，就得到每个块的完整 dK/dV。

    直接把 autograd 穿过 dist 的 P2P / all_to_all 会**静默给错值**：
    这些算子没有注册 autograd kernel（PyTorch 只发一个 UserWarning）。
    """
    B, H, S, D = q.shape
    G = k.shape[1]
    start, length = shard_bounds(S, cp_size, cp_rank)
    # 收齐所有 rank 的 K/V（按序列拼回完整长度）
    k_parts = [torch.empty_like(k) for _ in range(cp_size)]
    v_parts = [torch.empty_like(v) for _ in range(cp_size)]
    dist.all_gather(k_parts, k.contiguous())
    dist.all_gather(v_parts, v.contiguous())
    k_all = torch.cat(k_parts, dim=2).detach().clone().requires_grad_(True)
    v_all = torch.cat(v_parts, dim=2).detach().clone().requires_grad_(True)
    q_local = q[:, :, start:start + length, :].detach().clone().requires_grad_(True)
    out = reference_attention(q_local, k_all, v_all, causal=True, window=window,
                              q_offset=start)
    dq, dk_all, dv_all = torch.autograd.grad(out.sum(), (q_local, k_all, v_all))
    # 每个 rank 只贡献了自己那批查询的梯度，求和才是完整的 dK/dV
    dist.all_reduce(dk_all)
    dist.all_reduce(dv_all)
    # 取回本 rank 那块
    off = 0
    for r in range(cp_size):
        _, ln = shard_bounds(S, cp_size, r)
        if r == cp_rank:
            dk_local = dk_all[:, :, off:off + ln, :]
            dv_local = dv_all[:, :, off:off + ln, :]
            break
        off += ln
    return dq, dk_local, dv_local


# ==========================================================================
# 任务 A
# ==========================================================================

def run_A(rank, cp_size, out_dir, device, case, seq, heads, kv_heads, head_dim,
          batch, window):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29980")
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=cp_size,
                            timeout=timedelta(seconds=180))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")

    q, k, v = make_qkv(batch, seq, heads, kv_heads, head_dim, seed=5, device=dev)
    # 参照：完整序列、全 head（再取本 rank 的序列段）
    ref_full = reference_attention(q, k, v, causal=True, window=window)
    start, length = shard_bounds(seq, cp_size, rank)
    ref_local = ref_full[:, :, start:start + length, :]

    # ---- ring ----
    lq = q[:, :, start:start + length, :].clone().requires_grad_(True)
    lk = k[:, :, start:start + length, :].clone().requires_grad_(True)
    lv = v[:, :, start:start + length, :].clone().requires_grad_(True)
    y_ring, rounds = ring_attention(lq, lk, lv, seq, cp_size, rank,
                                    causal=True, window=window)
    ring_diff = float((y_ring - ref_local).abs().max().item())
    gq_r, gk_r, gv_r = torch.autograd.grad(y_ring.sum(), (lq, lk, lv),
                                           allow_unused=True)

    # ---- ulysses ----
    uq = q[:, :, start:start + length, :].clone().requires_grad_(True)
    uk = k[:, :, start:start + length, :].clone().requires_grad_(True)
    uv = v[:, :, start:start + length, :].clone().requires_grad_(True)
    try:
        y_ul, ul_rounds = ulysses_attention(uq, uk, uv, seq, cp_size, rank,
                                            causal=True, window=window)
        ul_diff = float((y_ul - ref_local).abs().max().item())
        gq_u, gk_u, gv_u = torch.autograd.grad(y_ul.sum(), (uq, uk, uv),
                                               allow_unused=True)
        ul_err = None
    except Exception as e:                                    # noqa: BLE001
        y_ul, ul_rounds, ul_diff = None, None, None
        gq_u = gk_u = gv_u = None
        ul_err = f"{type(e).__name__}: {e}"

    # 梯度参照：对完整序列求梯度后取本 rank 段
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    ref_full2 = reference_attention(q2, k2, v2, causal=True, window=window)
    gq_ref, gk_ref, gv_ref = torch.autograd.grad(ref_full2.sum(), (q2, k2, v2))

    def gdiff(got, want):
        if got is None:
            return None
        return float((got - want[:, :, start:start + length, :]).abs().max().item())

    # ---- 正确的 CP 梯度（本地反向 + 跨 rank 归约）----
    dq_ok, dk_ok, dv_ok = cp_gradients(q, k, v, cp_size, rank, window=window)

    # 通信账
    rings_bytes = (cp_size - 1) * 2 * batch * kv_heads * length * head_dim * 8
    ulysses_bytes = 2 * 2 * batch * seq * heads * head_dim * 8   # 两次 all-to-all

    result = {
        "rank": rank, "case": case, "seq": seq, "heads": heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "batch": batch,
        "window": window, "cp_size": cp_size,
        "local_len": length, "local_start": start,
        "ring_value_max_abs_diff": ring_diff,
        "ring_grad_q_max_abs_diff": gdiff(gq_r, gq_ref),
        "ring_grad_k_max_abs_diff": gdiff(gk_r, gk_ref),
        "ring_grad_v_max_abs_diff": gdiff(gv_r, gv_ref),
        "ring_rounds": rounds,
        "ulysses_value_max_abs_diff": ul_diff,
        "ulysses_grad_q_max_abs_diff": gdiff(gq_u, gq_ref),
        "ulysses_grad_k_max_abs_diff": gdiff(gk_u, gk_ref),
        "ulysses_grad_v_max_abs_diff": gdiff(gv_u, gv_ref),
        "ulysses_rounds": ul_rounds,
        "ulysses_error": ul_err,
        "cp_grad_q_max_abs_diff": gdiff(dq_ok, gq_ref),
        "cp_grad_k_max_abs_diff": gdiff(dk_ok, gk_ref),
        "cp_grad_v_max_abs_diff": gdiff(dv_ok, gv_ref),
        "autograd_through_p2p_grad_k_error": gdiff(gk_r, gk_ref),
        "autograd_through_p2p_grad_v_error": gdiff(gv_r, gv_ref),
        "ring_bytes_per_rank": rings_bytes,
        "ulysses_bytes_per_rank": ulysses_bytes,
        "bytes_ratio_ulysses_over_ring": (ulysses_bytes / rings_bytes
                                          if rings_bytes else None),
    }
    gathered = [None] * cp_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"A_{case}.json"), "w", encoding="utf-8") as f:
            json.dump({"env": env_pins(), "config": QWEN3_8B, "ranks": gathered}, f,
                      ensure_ascii=False, indent=2)
        r = gathered[0]
        print(f"  正确 CP 梯度：dQ={r['cp_grad_q_max_abs_diff']:.2e} "
              f"dK={r['cp_grad_k_max_abs_diff']:.2e} dV={r['cp_grad_v_max_abs_diff']:.2e} "
              f"（直接穿 P2P 的 dK/dV 误差={r['autograd_through_p2p_grad_k_error']:.2e}/"
              f"{r['autograd_through_p2p_grad_v_error']:.2e}）")
        print(f"  {case:<22} S={seq:<4} CP={cp_size} 本 rank 段={r['local_len']} "
              f"ring 值差={r['ring_value_max_abs_diff']:.2e} "
              f"dQ={r['ring_grad_q_max_abs_diff']:.2e} "
              f"dK={r['ring_grad_k_max_abs_diff']:.2e} "
              f"dV={r['ring_grad_v_max_abs_diff']:.2e} 轮={r['ring_rounds']} "
              f"| ulysses 值差={r['ulysses_value_max_abs_diff']} "
              f"轮={r['ulysses_rounds']} {r['ulysses_error'] or ''}")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A"])
    ap.add_argument("--cp", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--case", default="auto")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    import torch.multiprocessing as mp
    mp.spawn(run_A, args=(a.cp, a.out, a.device, a.case, a.seq, a.heads,
                          a.kv_heads, a.head_dim, a.batch, a.window or None),
             nprocs=a.cp, join=True)


if __name__ == "__main__":
    sys.exit(main())
