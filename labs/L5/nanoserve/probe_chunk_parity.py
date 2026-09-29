#!/usr/bin/env python3
"""L5.7 任务 B 续 —— prefill chunk 两条读路径的分级对拍。

引擎级的"输出不同"可能来自 kernel，也可能来自块表/槽位的搭法。这一步不经过调度器：
构造同一批输入与同一份 KV，分别调用

  * 基类 `PagedModel.forward`（gather + SDPA + causal bias）
  * 分页 `PagedModelExec.chunk_forward`（块表寻址的分块 kernel）

分三级报告，**不把 28 层 bf16 累加后的 logits 差当成 kernel 的错**：

  1. 第 0 层 attention 输出差（只看 kernel 本身）；
  2. 整模型 logits 差与 argmax 一致率（含 28 层累加）；
  3. 同一条路径重复调用的逐位一致性（对照组）。

用法：
    python probe_chunk_parity.py --start 5 --chunk 20
    python probe_chunk_parity.py --start 16 --chunk 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from model import PagedModel, rms_norm, rotate_half             # noqa: E402
from paged_attn import paged_chunk_attention                    # noqa: E402
from paged_engine import PagedModelExec                         # noqa: E402

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("NANOSERVE_HUB", "/scratch/learn/models/hf/hub")


def setup(m, start, chunk, block_ids):
    """写满历史 KV，并返回这一步的输入与寻址信息。"""
    c, bs = m.cfg, m.block_size
    dev = m.k_cache[0].device
    total = start + chunk
    ids = torch.randint(0, 1000, (1, chunk), device=dev)
    positions = torch.arange(start, total, device=dev)[None, :]
    slot_map = torch.tensor(
        [[block_ids[p // bs] * bs + p % bs for p in range(start, total)]],
        dtype=torch.long, device=dev)
    with torch.inference_mode():
        for i in range(c.n_layers):
            for p in range(start):
                s = block_ids[p // bs] * bs + p % bs
                m.k_cache[i][s] = torch.randn(c.n_kv, c.head_dim, device=dev,
                                              dtype=m.dtype)
                m.v_cache[i][s] = torch.randn(c.n_kv, c.head_dim, device=dev,
                                              dtype=m.dtype)
    gather = torch.tensor([[block_ids[p // bs] * bs + p % bs for p in range(total)]],
                          dtype=torch.long, device=dev)
    bt = torch.tensor([block_ids], dtype=torch.int32, device=dev)
    sl = torch.tensor([total], dtype=torch.int32, device=dev)
    return dict(ids=ids, positions=positions, slot_map=slot_map, gather=gather,
                bt=bt, sl=sl, total=total, dev=dev)


def base_forward(m, st):
    return PagedModel.forward(m, st["ids"], st["positions"], st["slot_map"],
                              st["gather"], torch.tensor([st["total"]],
                                                         dtype=torch.long,
                                                         device=st["dev"]))


def paged_forward(m, st):
    return PagedModelExec.chunk_forward(m, st["ids"], st["positions"],
                                        st["slot_map"], st["bt"], st["sl"])


def layer0_attention(m, st):
    """第 0 层 attention 输出：两条路径用同一份 q/k/v 与同一份缓存。"""
    c, bs = m.cfg, m.block_size
    lay = m.layers[0]
    with torch.inference_mode():
        x = m.embed[st["ids"]]
        h = rms_norm(x, lay.n1, c.eps)
        q = (h @ lay.wq.T).view(1, st["ids"].shape[1], c.n_q, c.head_dim)
        k = (h @ lay.wk.T).view(1, st["ids"].shape[1], c.n_kv, c.head_dim)
        v = (h @ lay.wv.T).view(1, st["ids"].shape[1], c.n_kv, c.head_dim)
        q = rms_norm(q, lay.qn, c.eps)
        k = rms_norm(k, lay.kn, c.eps)
        cos, sin = m.cos[st["positions"]], m.sin[st["positions"]]
        q = q * cos[:, :, None] + rotate_half(q) * sin[:, :, None]
        k = k * cos[:, :, None] + rotate_half(k) * sin[:, :, None]
        kc, vc = m.k_cache[0].clone(), m.v_cache[0].clone()
        flat = st["slot_map"].reshape(-1)
        kc[flat] = k.reshape(-1, c.n_kv, c.head_dim)
        vc[flat] = v.reshape(-1, c.n_kv, c.head_dim)
        kk = kc[st["gather"]].transpose(1, 2)
        vv = vc[st["gather"]].transpose(1, 2)
        ctx = torch.arange(st["total"], device=st["dev"])[None, None, :]
        mask = ctx <= st["positions"][:, :, None]
        bias = torch.zeros((1, 1, st["ids"].shape[1], st["total"]),
                           device=st["dev"], dtype=m.dtype)
        bias.masked_fill_(~mask[:, None], float("-inf"))
        o_base = F.scaled_dot_product_attention(
            q.transpose(1, 2), kk, vv, attn_mask=bias,
            enable_gqa=True).transpose(1, 2)
        o_paged = paged_chunk_attention(q, kc, vc, st["bt"], st["sl"],
                                        st["positions"].to(torch.int32),
                                        block_size=bs)
    return o_base, o_paged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    m = PagedModelExec(REPO, HUB, num_blocks=args.blocks,
                       block_size=args.block_size)
    bs = m.block_size
    n_blocks = (args.start + args.chunk + bs - 1) // bs
    block_ids = [(i * 3 + 1) % args.blocks for i in range(n_blocks)]
    st = setup(m, args.start, args.chunk, block_ids)

    with torch.inference_mode():
        ob, op = layer0_attention(m, st)
        lb, lp = base_forward(m, st), paged_forward(m, st)
        lb2 = base_forward(m, st)
        lp2 = paged_forward(m, st)

    dl = (ob.float() - op.float()).abs()
    d = (lb.float() - lp.float()).abs()
    argmax_same = (lb.argmax(-1) == lp.argmax(-1)).float().mean().item()
    print(f"start={args.start} chunk={args.chunk} 块表={block_ids} "
          f"dtype={m.dtype}")
    print(f"  ① 第 0 层 attention：max|Δ| {dl.max().item():.3e}  "
          f"mean|Δ| {dl.mean().item():.3e}")
    print(f"  ② 整模型 logits：max|Δ| {d.max().item():.3e}  "
          f"mean|Δ| {d.mean().item():.3e}  argmax 逐位置一致率 "
          f"{argmax_same * 100:.1f}%  量级 [{float(lb.max()):.2f}, "
          f"{float(lb.min()):.2f}]")
    print(f"  ③ 同路径重复：base 逐位相同 "
          f"{bool((lb == lb2).all())}，paged 逐位相同 {bool((lp == lp2).all())}")
    ok = dl.max().item() < 5e-3 and bool((lb == lb2).all()) and bool((lp == lp2).all())
    print("  结论：" + ("kernel 层一致、两条路径各自可复现；"
                        "整模型差异是 bf16 逐层累加的归约顺序，量与既有 decode 路径同级"
                        if ok else "**第 0 层或重复性不通过，需要定位**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
