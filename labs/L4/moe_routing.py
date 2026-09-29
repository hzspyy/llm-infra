#!/usr/bin/env python3
"""L4.4 修订（任务 A）—— MoE 路由与分组的参照实现。

[A1] top-k 路由、权重归一化、token pack、专家计算、weighted combine 的完整参照
[A2] 与"逐 token 逐专家"的 dense 参照逐元素对拍
[A3] 边界：空专家、单专家倾斜、重复专家、非法路由 id、尾 tile 与 padding

全部在小矩阵上做（E=8、H=64、I=128、top_k=2），输入和中间量都打印成表，
便于手算核对。TP/EP 的分片布局不在本任务里（见 6.3）。

用法：
    python labs/L4/moe_routing.py [--tokens 17] [--experts 8] [--topk 2]
"""

import argparse
import json
import os
import sys

import torch

SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def silu(x):
    return x * torch.sigmoid(x)


class MoE:
    """最小 MoE：router + 每专家一个 SwiGLU。"""

    def __init__(self, E, H, I, topk, seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.E, self.H, self.I, self.topk = E, H, I, topk
        self.Wg = torch.randn(E, H, generator=g, dtype=dtype) * 0.05
        self.W1 = torch.randn(E, I, H, generator=g, dtype=dtype) * 0.05
        self.W3 = torch.randn(E, I, H, generator=g, dtype=dtype) * 0.05
        self.W2 = torch.randn(E, H, I, generator=g, dtype=dtype) * 0.05

    # ---- 路由 ----
    def route(self, x, mode="renorm", forced=None):
        logits = x @ self.Wg.t()                       # [T, E]
        if forced is not None:
            ids = forced
            weights = torch.full((x.shape[0], ids.shape[1]), 1.0 / ids.shape[1],
                                 dtype=x.dtype)
            return ids, weights, logits
        top = torch.topk(logits, self.topk, dim=-1)
        ids, vals = top.indices, top.values
        if mode == "renorm":                            # 在 top-k 内重新归一化
            weights = torch.softmax(vals, dim=-1)
        elif mode == "full":                            # 全词表 softmax 后取值
            weights = torch.softmax(logits, dim=-1).gather(1, ids)
        else:
            weights = vals
        return ids, weights, logits

    def expert_forward(self, e, h):
        return (silu(h @ self.W1[e].t()) * (h @ self.W3[e].t())) @ self.W2[e].t()

    # ---- dense 参照：逐 token 逐专家，不分组 ----
    def dense(self, x, ids, weights):
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                out[t] += weights[t, j] * self.expert_forward(int(ids[t, j]), x[t:t + 1])[0]
        return out

    # ---- 分组执行：pack -> 逐专家 GEMM -> combine ----
    def grouped(self, x, ids, weights, block_m=8):
        T, K = ids.shape
        flat_e = ids.reshape(-1)
        order = torch.argsort(flat_e, stable=True)          # pack 顺序
        perm = order                                        # packed 位置 -> 原 flat 位置
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
        counts = torch.bincount(flat_e, minlength=self.E)
        offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)]).to(torch.int64)
        packed_x = x.repeat_interleave(K, dim=0)[perm]      # [T*K, H]
        packed_w = weights.reshape(-1)[perm]                # [T*K]
        # padding 到 block_m 的整数倍，pad 行用零填充且不参与 combine
        total = packed_x.shape[0]
        padded = (block_m - total % block_m) % block_m
        if padded:
            packed_x_p = torch.cat([packed_x, torch.zeros(padded, self.H,
                                                           dtype=x.dtype)], dim=0)
        else:
            packed_x_p = packed_x
        out_packed = torch.zeros_like(packed_x_p)
        for e in range(self.E):
            a, b = int(offsets[e]), int(offsets[e + 1])
            if a == b:
                continue                                    # 空专家：跳过，不启动 kernel
            out_packed[a:b] = self.expert_forward(e, packed_x_p[a:b])
        out_packed = out_packed[:total]
        contrib = out_packed * packed_w[:, None]
        out = torch.zeros_like(x)
        out.index_add_(0, perm // K, contrib)               # combine 回 token
        return out, {"perm": perm, "inv": inv, "counts": counts, "offsets": offsets,
                     "padded": padded, "packed_x": packed_x_p}


def section_A(args):
    torch.manual_seed(0)
    rep = {}
    title("[A] MoE 路由与分组的参照实现")
    m = MoE(args.experts, args.hidden, args.inter, args.topk)
    x = torch.randn(args.tokens, args.hidden, dtype=torch.float64)
    ids, w, logits = m.route(x, args.renorm)
    print(f"  x {tuple(x.shape)}  E={args.experts} top_k={args.topk} "
          f"H={args.hidden} I={args.inter}  归一化={args.renorm}")
    out_ref = m.dense(x, ids, w)
    out_grp, aux = m.grouped(x, ids, w, args.block_m)
    d = (out_grp - out_ref).abs().max().item()
    print(f"  分组 vs dense 参照：max|diff| {d:.3e}  "
          f"（相对 {(out_grp - out_ref).norm() / out_ref.norm():.3e}）")

    sub("A1 路由结果与分组元数据")
    K = args.topk
    print(f"  {'token':>5} " + " ".join(f"{'expert' :>6}" for _ in range(K))
          + "   " + " ".join(f"{'weight':>8}" for _ in range(K)))
    for t in range(min(args.tokens, 8)):
        print(f"  {t:>5} " + " ".join(f"{int(ids[t, j]):>6}" for j in range(K))
              + "   " + " ".join(f"{float(w[t, j]):>8.4f}" for j in range(K)))
    if args.tokens > 8:
        print(f"  ...（共 {args.tokens} 行）")
    print(f"  权重和（renorm 后应为 1）：{w.sum(-1)[:5].tolist()}")
    print(f"  专家计数 counts = {aux['counts'].tolist()}")
    print(f"  offsets（长度 E+1，前缀和）= {aux['offsets'].tolist()}")
    print(f"  perm 前 8 个 = {aux['perm'][:8].tolist()}；"
          f"inv[perm] == arange: {bool(torch.equal(aux['inv'][aux['perm']], torch.arange(aux['perm'].numel())))}")
    print(f"  尾 tile padding：{aux['padded']} 行（总 {aux['perm'].numel()} → "
          f"{aux['perm'].numel() + aux['padded']}）")

    sub("A2 分组执行的时间构成（CPU，仅看形状与调用次数）")
    calls = int((aux["counts"] > 0).sum())
    print(f"  非空专家 {calls}/{args.experts} 个；每专家 1 次 GEMM×3（w1/w3/w2）"
          f" → {calls * 3} 次小 GEMM")
    print(f"  packed 行数 {aux['perm'].numel()}（含 {aux['padded']} 行 padding），"
          f"每个专家处理 {aux['counts'].tolist()} 行")
    rep["basic"] = {"max_abs_diff_vs_dense": d,
                    "counts": aux["counts"].tolist(),
                    "offsets": aux["offsets"].tolist(),
                    "padded": aux["padded"], "nonempty_experts": calls}
    SUMMARY["A_basic"] = rep

    sub("A3 边界：空专家、单专家倾斜、重复与非法路由、尾 tile")
    edges = {}

    # 空专家：强制只路由到前 E-1 个专家，最后一个为空
    forced = torch.zeros(args.tokens, K, dtype=torch.long)
    forced[:, 0] = torch.arange(args.tokens) % (args.experts - 1)
    forced[:, 1] = (torch.arange(args.tokens) + 1) % (args.experts - 1)
    ids2, w2, _ = m.route(x, forced=forced)
    o_ref2 = m.dense(x, ids2, w2)
    o_grp2, aux2 = m.grouped(x, ids2, w2, args.block_m)
    d2 = (o_grp2 - o_ref2).abs().max().item()
    edges["empty_expert"] = {"diff": d2, "counts": aux2["counts"].tolist()}
    print(f"  空专家：counts {aux2['counts'].tolist()}（最后一个为 0），"
          f"与 dense 参照 max|diff| {d2:.3e}")

    # 单专家倾斜：所有 token 都路由到专家 0
    one = torch.zeros(args.tokens, K, dtype=torch.long)
    ids3, w3, _ = m.route(x, forced=one)
    o_ref3 = m.dense(x, ids3, w3)
    o_grp3, aux3 = m.grouped(x, ids3, w3, args.block_m)
    d3 = (o_grp3 - o_ref3).abs().max().item()
    edges["all_one_expert"] = {"diff": d3, "counts": aux3["counts"].tolist(),
                               "padded": aux3["padded"]}
    print(f"  单专家倾斜：counts {aux3['counts'].tolist()}，padding "
          f"{aux3['padded']} 行，与 dense 参照 max|diff| {d3:.3e}")

    # 重复专家：同一个 token 的 top-k 里出现两次同一个专家
    dup = torch.zeros(3, K, dtype=torch.long)
    dup[:, 0] = 0
    dup[:, 1] = 0
    xd = x[:3]
    try:
        o_dup = m.grouped(xd, dup, torch.full((3, K), 0.5, dtype=torch.float64), args.block_m)[0]
        o_dup_ref = m.dense(xd, dup, torch.full((3, K), 0.5, dtype=torch.float64))
        dup_diff = (o_dup - o_dup_ref).abs().max().item()
        note = ("与 dense 参照一致：重复路由被当成两次独立计算（权重各算一半）"
                if dup_diff < 1e-12 else "与 dense 参照不一致")
        edges["duplicate_expert"] = {"diff": dup_diff, "note": note}
        print(f"  重复专家：{note}（max|diff| {dup_diff:.3e}）")
    except Exception as e:
        edges["duplicate_expert"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  重复专家：{type(e).__name__}: {str(e)[:100]}")

    # 非法路由 id（越界）：counts/bincount 与索引会怎样
    bad = torch.zeros(2, K, dtype=torch.long)
    bad[0, 0] = args.experts + 5
    wb = torch.full((2, K), 0.5, dtype=torch.float64)
    grp_res = None
    try:
        grp_res = m.grouped(x[:2], bad, wb, args.block_m)[0]
        print(f"  非法 id（E+5）分组路径：不报错，该 token 的输出范数 "
              f"{grp_res[0].norm().item():.3e}（另一 token {grp_res[1].norm().item():.3e}）")
    except Exception as e:
        print(f"  非法 id（E+5）分组路径：{type(e).__name__}: {str(e)[:80]}")
    dense_err = None
    try:
        m.dense(x[:2], bad, wb)
    except Exception as e:
        dense_err = f"{type(e).__name__}: {str(e)[:80]}"
        print(f"  非法 id（E+5）dense 路径：{dense_err}")
    edges["illegal_id"] = {"grouped_raised": grp_res is None,
                           "grouped_first_token_norm":
                               None if grp_res is None else grp_res[0].norm().item(),
                           "dense_error": dense_err}
    print("  结论：分组路径把越界 id 当成一个不存在的专家，静默输出零；"
          "dense 路径才报越界。路由实现必须自己校验 id 范围。")

    # 尾 tile：tokens 不整除 block_m
    for T in (args.block_m - 1, args.block_m, args.block_m + 1):
        xt = torch.randn(T, args.hidden, dtype=torch.float64)
        idt, wt, _ = m.route(xt)
        ot, auxt = m.grouped(xt, idt, wt, args.block_m)
        dt = (ot - m.dense(xt, idt, wt)).abs().max().item()
        print(f"  尾 tile T={T:<3} padding {auxt['padded']} 行，"
              f"与 dense 参照 max|diff| {dt:.3e}")
    edges["tail_tiles"] = {"block_m": args.block_m}
    SUMMARY["A_edges"] = edges
    return SUMMARY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=17)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--inter", type=int, default=128)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--block-m", type=int, default=8)
    ap.add_argument("--renorm", choices=["renorm", "full", "none"], default="renorm")
    ap.add_argument("--outdir", default=os.path.expanduser("~/l44_routing"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    SUMMARY["config"] = vars(args)
    section_A(args)
    path = os.path.join(args.outdir, "moe_routing.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
