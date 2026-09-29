#!/usr/bin/env python3
"""L5.7 任务 B —— 真实 paged decode attention 内核。

nanoserve 原来的读路径是「按 block table 展开成 gather_idx → 收集成连续张量 →
`scaled_dot_product_attention`」。那等于每步把整段 KV 复制一遍，paging 只体现在
寻址上，没有体现在 kernel 里。这个模块补的是真正按块寻址的 decode attention：
每个 (序列, query head) 一个 program，沿块表逐块读 KV，用在线 softmax 累加。

  q            [B, n_q, hd]
  k_cache      [num_slots, n_kv, hd]   （每层一份，槽位 = block_id * block_size + off）
  v_cache      同上
  block_tables [B, max_blocks] int32
  seq_lens     [B] int32
  → o          [B, n_q, hd]

GQA 通过 `n_q // n_kv` 分组复用 KV head；块内位置超过 `seq_len` 的按 -inf 屏蔽。

同一份分页数据还有第二条读路径：**prefill chunk**（Q>1）。它多了两件事——
每个 query 的绝对位置不同（因果掩码变成 `key_pos <= query_pos`），
以及一个 chunk 可能横跨多个块、且起点落在块内偏移上。`paged_chunk_attention`
把这些一起处理，于是引擎里不再有"只有 decode 走分页"的特例。

自测（不需要引擎）：
    python labs/L5/nanoserve/paged_attn.py            # 与 SDPA 对拍
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K, V, BT, SL, OUT,
    stride_qb, stride_qh,
    stride_ks, stride_kh,
    stride_btb,
    stride_ob, stride_oh,
    G,                      # n_q // n_kv（GQA 分组比）
    N_Q_HEADS,              # query head 数，grid = B * N_Q_HEADS
    MAX_BLOCKS,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // N_Q_HEADS
    hq = pid % N_Q_HEADS
    hkv = hq // G

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + hq * stride_qh + offs_d).to(tl.float32)
    # 与 SDPA 的默认口径一致：scale = 1/sqrt(head_dim)
    q = q * (1.0 / tl.sqrt(HEAD_DIM * 1.0))
    seq_len = tl.load(SL + b)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Triton 的 AST 不支持 `continue`，所以空槽位用掩码处理而不是跳过：
    # block_id < 0 时把基址钳到 0 并整体屏蔽，不参与 softmax。
    offs_n = tl.arange(0, BLOCK_N)
    bt_ptr = BT + b * stride_btb
    for blk in range(0, MAX_BLOCKS):
        block_id = tl.load(bt_ptr + blk)
        blk_ok = block_id >= 0
        base = tl.where(blk_ok, block_id, 0) * BLOCK_N
        pos = blk * BLOCK_N + offs_n
        mask = (pos < seq_len) & blk_ok
        k = tl.load(K + (base + offs_n)[:, None] * stride_ks
                    + hkv * stride_kh + offs_d[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        v = tl.load(V + (base + offs_n)[:, None] * stride_ks
                    + hkv * stride_kh + offs_d[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1)          # [BLOCK_N]
        s = tl.where(mask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    o = acc / l_i
    tl.store(OUT + b * stride_ob + hq * stride_oh + offs_d, o.to(OUT.dtype.element_ty))


def paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens,
                           block_size: int = 16):
    """按块表做 decode attention；q 每条序列一个 token。"""
    B, n_q, hd = q.shape
    n_kv = k_cache.shape[1]
    assert n_q % n_kv == 0, "n_q 必须能被 n_kv 整除（GQA）"
    g = n_q // n_kv
    assert hd & (hd - 1) == 0 and hd >= 16, "head_dim 需为 2 的幂"
    qc = q.contiguous()
    kc = k_cache.contiguous()
    vc = v_cache.contiguous()
    bt = block_tables.to(torch.int32).contiguous()
    sl = seq_lens.to(torch.int32).contiguous()
    out = torch.empty((B, n_q, hd), dtype=q.dtype, device=q.device)
    grid = (B * n_q,)
    _paged_decode_kernel[grid](
        qc, kc, vc, bt, sl, out,
        qc.stride(0), qc.stride(1),
        kc.stride(0), kc.stride(1),
        bt.stride(0),
        out.stride(0), out.stride(1),
        g, n_q, bt.shape[1],
        BLOCK_N=block_size, HEAD_DIM=hd,
        num_warps=4,
    )
    return out


def _reference(q, k_cache, v_cache, block_tables, seq_lens, block_size=16):
    """同一份分页数据的连续读参照：展开成 gather 后走 SDPA。"""
    import torch.nn.functional as F
    B, n_q, hd = q.shape
    n_kv = k_cache.shape[1]
    C = int(seq_lens.max().item())
    idx = torch.zeros((B, C), dtype=torch.long, device=q.device)
    for b in range(B):
        pos = 0
        for blk in block_tables[b].tolist():
            if blk < 0:
                break
            for off in range(block_size):
                if pos >= C:
                    break
                idx[b, pos] = blk * block_size + off
                pos += 1
    kk = k_cache[idx].transpose(1, 2)      # [B, n_kv, C, hd]
    vv = v_cache[idx].transpose(1, 2)
    mask = torch.arange(C, device=q.device)[None, :] < seq_lens[:, None]
    bias = torch.zeros((B, 1, 1, C), device=q.device, dtype=q.dtype)
    bias.masked_fill_(~mask[:, None, None, :], float("-inf"))
    # q: [B, n_q, 1, hd]；head 维在 -3，GQA 要求 n_kv 整除 n_q
    o = F.scaled_dot_product_attention(q.unsqueeze(2), kk, vv,
                                       attn_mask=bias, enable_gqa=True)
    return o.squeeze(2)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    B, n_q, n_kv, hd, bs = 3, 8, 2, 128, 16
    seq_lens = torch.tensor([17, 48, 5], dtype=torch.int32, device=dev)
    max_blocks = 4
    n_slots = 128          # 8 个块；块号必须 < n_slots/block_size
    q = torch.randn(B, n_q, hd, device=dev, dtype=torch.float16)
    k = torch.randn(n_slots, n_kv, hd, device=dev, dtype=torch.float16)
    v = torch.randn(n_slots, n_kv, hd, device=dev, dtype=torch.float16)
    bt = torch.full((B, max_blocks), -1, dtype=torch.int32, device=dev)
    # 故意让块表不连续，验证寻址确实按块走
    bt[0, :2] = torch.tensor([0, 3], dtype=torch.int32)
    bt[1, :3] = torch.tensor([1, 2, 5], dtype=torch.int32)
    bt[2, :1] = torch.tensor([7], dtype=torch.int32)

    got = paged_decode_attention(q, k, v, bt, seq_lens, bs)
    ref = _reference(q, k, v, bt, seq_lens, bs)
    diff = (got.float() - ref.float()).abs()
    print(f"shape {tuple(got.shape)}  max|Δ| {diff.max().item():.3e}  "
          f"mean|Δ| {diff.mean().item():.3e}")
    assert diff.max().item() < 2e-2, "与 SDPA 参照差距过大"
    print("块表非连续寻址与在线 softmax 结果与 gather+SDPA 参照一致")

    # ---- chunk 路径：起点落在块内、跨块、带 padding 的 query ----
    B, n_q, n_kv, hd, bs = 3, 8, 2, 128, 16
    max_blocks, n_slots = 6, 160
    k = torch.randn(n_slots, n_kv, hd, device=dev, dtype=torch.float16)
    v = torch.randn(n_slots, n_kv, hd, device=dev, dtype=torch.float16)
    # 三种序列：从块内偏移开始、从块边界开始、短序列
    starts = [5, 16, 0]
    lens = [11, 20, 3]
    Q = max(lens)
    bt = torch.full((B, max_blocks), -1, dtype=torch.int32, device=dev)
    bt[0, :3] = torch.tensor([0, 4, 2], dtype=torch.int32)
    bt[1, :4] = torch.tensor([1, 2, 5, 3], dtype=torch.int32)
    bt[2, :6] = torch.tensor([6, 7, 0, 1, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([s + n for s, n in zip(starts, lens)],
                            dtype=torch.int32, device=dev)
    positions = torch.full((B, Q), -1, dtype=torch.int32, device=dev)
    for b, (st, n) in enumerate(zip(starts, lens)):
        positions[b, :n] = torch.arange(st, st + n, dtype=torch.int32)
    qc = torch.randn(B, Q, n_q, hd, device=dev, dtype=torch.float16)
    got = paged_chunk_attention(qc, k, v, bt, seq_lens, positions, bs)
    ref = _reference_chunk(qc, k, v, bt, seq_lens, positions, bs)
    diff = (got.float() - ref.float()).abs()
    # padding 位置必须是 0（不是 NaN）
    padded = (positions < 0)
    pad_ok = bool((got[padded].abs().max().item() == 0.0)) if padded.any() else True
    print(f"chunk: shape {tuple(got.shape)}  max|Δ| {diff.max().item():.3e}  "
          f"mean|Δ| {diff.mean().item():.3e}  padding 全 0 {pad_ok}")
    assert diff.max().item() < 2e-2, "chunk 路径与参照差距过大"
    assert not torch.isnan(got).any(), "出现 NaN（空 softmax 行没有兜住）"
    print("chunk 路径（跨块、块内起点、因果掩码、padding）与 gather+SDPA 参照一致")


@triton.jit
def _paged_chunk_kernel(
    Q, K, V, BT, SL, POS, OUT,
    stride_qb, stride_qq, stride_qh,
    stride_ks, stride_kh,
    stride_btb,
    stride_pb, stride_pq,
    stride_ob, stride_oq, stride_oh,
    Q_LEN,                  # 本步的 query 长度（每个序列相同，padding 用 -1 位置标记）
    G,
    N_Q_HEADS,
    MAX_BLOCKS,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bq = tl.program_id(0)          # b * Q_LEN + qi
    hq = tl.program_id(1)
    b = bq // Q_LEN
    qi = bq % Q_LEN
    hkv = hq // G

    offs_d = tl.arange(0, HEAD_DIM)
    qpos = tl.load(POS + b * stride_pb + qi * stride_pq)
    q = tl.load(Q + b * stride_qb + qi * stride_qq + hq * stride_qh + offs_d).to(tl.float32)
    q = q * (1.0 / tl.sqrt(HEAD_DIM * 1.0))
    seq_len = tl.load(SL + b)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    bt_ptr = BT + b * stride_btb
    for blk in range(0, MAX_BLOCKS):
        block_id = tl.load(bt_ptr + blk)
        blk_ok = block_id >= 0
        base = tl.where(blk_ok, block_id, 0) * BLOCK_N
        pos = blk * BLOCK_N + offs_n
        # 因果 + 有效长度 + 该位置确实已写入（qpos>=0 表示这条 query 有效）
        mask = (pos <= qpos) & (pos < seq_len) & blk_ok & (qpos >= 0)
        k = tl.load(K + (base + offs_n)[:, None] * stride_ks
                    + hkv * stride_kh + offs_d[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        v = tl.load(V + (base + offs_n)[:, None] * stride_ks
                    + hkv * stride_kh + offs_d[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1)
        s = tl.where(mask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # padding query 没有参与任何 key：l_i 为 0，直接写 0 而不是做 0/0
    o = tl.where(l_i > 0, acc / tl.where(l_i > 0, l_i, 1.0), 0.0)
    tl.store(OUT + b * stride_ob + qi * stride_oq + hq * stride_oh + offs_d,
             o.to(OUT.dtype.element_ty))


def paged_chunk_attention(q, k_cache, v_cache, block_tables, seq_lens, positions,
                          block_size: int = 16):
    """按块表做 prefill chunk attention。

    q         [B, Q, n_q, hd]
    positions [B, Q]  每个 query 的绝对位置；padding 用 -1
    seq_lens  [B]     这一步之后的上下文长度（= start + chunk）
    → o       [B, Q, n_q, hd]
    """
    B, Q, n_q, hd = q.shape
    n_kv = k_cache.shape[1]
    assert n_q % n_kv == 0, "n_q 必须能被 n_kv 整除（GQA）"
    g = n_q // n_kv
    assert hd & (hd - 1) == 0 and hd >= 16, "head_dim 需为 2 的幂"
    qc = q.contiguous()
    kc = k_cache.contiguous()
    vc = v_cache.contiguous()
    bt = block_tables.to(torch.int32).contiguous()
    sl = seq_lens.to(torch.int32).contiguous()
    pos = positions.to(torch.int32).contiguous()
    out = torch.empty((B, Q, n_q, hd), dtype=q.dtype, device=q.device)
    grid = (B * Q, n_q)
    _paged_chunk_kernel[grid](
        qc, kc, vc, bt, sl, pos, out,
        qc.stride(0), qc.stride(1), qc.stride(2),
        kc.stride(0), kc.stride(1),
        bt.stride(0),
        pos.stride(0), pos.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        Q, g, n_q, bt.shape[1],
        BLOCK_N=block_size, HEAD_DIM=hd,
        num_warps=4,
    )
    return out


def _reference_chunk(q, k_cache, v_cache, block_tables, seq_lens, positions,
                     block_size=16):
    """连续读参照：把分页数据展开成 [B, C] 的 gather，再叠因果掩码走 SDPA。"""
    import torch.nn.functional as F
    B, Q, n_q, hd = q.shape
    C = int(seq_lens.max().item())
    idx = torch.zeros((B, C), dtype=torch.long, device=q.device)
    for b in range(B):
        slots = []
        for p in range(C):
            slots.append(int(block_tables[b, p // block_size]) * block_size
                         + p % block_size)
        idx[b] = torch.tensor(slots, device=q.device)
    kk = k_cache[idx].transpose(1, 2)                 # [B, n_kv, C, hd]
    vv = v_cache[idx].transpose(1, 2)
    qt = q.transpose(1, 2)                            # [B, n_q, Q, hd]
    ctx = torch.arange(C, device=q.device)[None, None, :]
    pos = positions[:, :, None]
    mask = (ctx <= pos) & (ctx < seq_lens[:, None, None])
    bias = torch.zeros((B, 1, Q, C), device=q.device, dtype=q.dtype)
    bias.masked_fill_(~mask[:, None], float("-inf"))
    o = F.scaled_dot_product_attention(qt, kk, vv, attn_mask=bias, enable_gqa=True)
    o = o.transpose(1, 2)
    o = torch.where((positions < 0)[:, :, None, None], torch.zeros_like(o), o)
    return o


if __name__ == "__main__":
    main()
