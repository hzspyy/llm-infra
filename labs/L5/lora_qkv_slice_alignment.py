#!/usr/bin/env python3
"""L5.10 补测 · `qkv_proj` 的切片对齐：v 到底落在哪几行。

SGLang 把适配器的 `[q_proj, k_proj, v_proj]` 归一成一条融合模块 `qkv_proj`
（`srt/lora/lora.py:264 normalize_qkv_proj`）：**按 [q, k, v] 的顺序在第 0 维拼接**，
没有 k_proj 的适配器用 `zeros_like(v)` 补零；融合模块的输出维由
`get_hidden_dim`（`srt/lora/utils.py:175`）给出：
`head_dim × (num_attention_heads + 2 × num_key_value_heads)`。

"切片对齐"要回答的是：v 的那几行落在**哪里**。本脚本在 CPU 上只读必要张量，
不加载 4B 权重其余部分，做三件事：

  1. 复现归一化：把适配器的 q/k/v LoRA 拼成 `qkv_proj` 的形状，核对与公式一致；
  2. **负例**：按"只拼 [q, v]"（少一次补零）拼，看 v 的行号偏移多少——
     这正是把 v 写到 4096 行而不是 5120 行的错误形态；
  3. **数值对拍**：用基座真实的 q/k/v 权重把 LoRA 合并进去，
     与"直接对 q_proj/v_proj 各做一次低秩更新"逐元素比较。

用法（crater，CPU 即可）：
    python lora_qkv_slice_alignment.py --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

BASE_GLOB = ("/scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots/*/"
             "model-*.safetensors")
ADAPTER = ("/scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots/"
           "036d6b7a5b589ea27bb9a855386e0ce1e281fa75/adapter_model.safetensors")
LAYER = 0


def load_base_layer(pattern, layer):
    """只读第 layer 层的 q/k/v 权重（safe_open 惰性读，单层约 31 MB）。"""
    out = {}
    keys = {f"model.layers.{layer}.self_attn.{m}_proj.weight": m
            for m in ("q", "k", "v")}
    for path in sorted(glob.glob(pattern)):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                if k in f.keys():
                    out[keys[k]] = f.get_tensor(k)
    return out


def load_adapter_layer(path, layer):
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            for m in ("q_proj", "v_proj"):
                if f".{layer}." in k and m in k:
                    out[k] = f.get_tensor(k)
    return out


def pick(adapter, layer, module, which):
    """取出某层某模块的 lora_A / lora_B。"""
    for k, v in adapter.items():
        if f".{layer}." in k and f"{module}.lora_{which}" in k:
            return v, k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-glob", default=BASE_GLOB)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--rank", type=float, default=8.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = load_base_layer(args.base_glob, args.layer)
    adapter = load_adapter_layer(args.adapter, args.layer)
    Aq, kq = pick(adapter, args.layer, "q_proj", "A")
    Bq, _ = pick(adapter, args.layer, "q_proj", "B")
    Av, kv = pick(adapter, args.layer, "v_proj", "A")
    Bv, _ = pick(adapter, args.adapter and args.layer, "v_proj", "B") \
        if False else pick(adapter, args.layer, "v_proj", "B")
    scale = args.alpha / args.rank

    out, rep = [], {}
    out.append(f"基座 {os.path.basename(os.path.dirname(args.base_glob))} 第 {args.layer} 层；"
               f"适配器 {os.path.basename(os.path.dirname(args.adapter))}")
    out.append(f"基座权重：q {tuple(base['q'].shape)}  k {tuple(base['k'].shape)}  "
               f"v {tuple(base['v'].shape)}")
    out.append(f"适配器：q A {tuple(Aq.shape)} B {tuple(Bq.shape)}；"
               f"v A {tuple(Av.shape)} B {tuple(Bv.shape)}；缩放 alpha/r = {scale}")
    head_dim = base["q"].shape[0] // 32
    out.append(f"按 get_hidden_dim 公式，qkv_proj 输出维 = head_dim × (n_q + 2×n_kv) = "
               f"{head_dim} × (32 + 2×8) = {head_dim * (32 + 16)}；"
               f"三块实测 {base['q'].shape[0]} + {base['k'].shape[0]} + "
               f"{base['v'].shape[0]} = "
               f"{base['q'].shape[0] + base['k'].shape[0] + base['v'].shape[0]}")
    rep["shapes"] = dict(q=list(base["q"].shape), k=list(base["k"].shape),
                         v=list(base["v"].shape),
                         qkv_rows=int(base["q"].shape[0] + base["k"].shape[0]
                                      + base["v"].shape[0]),
                         adapter=dict(Aq=list(Aq.shape), Bq=list(Bq.shape),
                                      Av=list(Av.shape), Bv=list(Bv.shape),
                                      scale=scale))

    # 1) 复现 SGLang 的 [q, k, v] 归一化（无 k_proj → 补零）
    zeros_B = torch.zeros_like(Bv)
    zeros_A = torch.zeros_like(Av)
    qkv_B = torch.cat([Bq, zeros_B, Bv], dim=0)
    qkv_A = torch.cat([Aq, zeros_A, Av], dim=0)
    out.append("")
    out.append(f"归一化后：B {tuple(qkv_B.shape)}（[q;k;v] 行序），"
               f"A {tuple(qkv_A.shape)}")
    q_rows = base["q"].shape[0]
    k_rows = base["k"].shape[0]
    v_rows = base["v"].shape[0]
    out.append(f"  v 的行区间 = [{q_rows + k_rows}, "
               f"{q_rows + k_rows + v_rows})；k 段全零："
               f"{bool((qkv_B[q_rows:q_rows + k_rows] == 0).all())}")

    # 2) 负例：只拼 [q; v]
    naive_B = torch.cat([Bq, Bv], dim=0)
    out.append(f"  负例（只拼 [q;v]）：B {tuple(naive_B.shape)}，"
               f"v 会落在 [{q_rows}, {q_rows + v_rows}) —— 与正确位置相差 "
               f"{k_rows} 行")

    # 3) 数值对拍：把 LoRA 合并进基座权重
    Wq_merged = base["q"].float() + scale * (Bq.float() @ Aq.float())
    Wv_merged = base["v"].float() + scale * (Bv.float() @ Av.float())
    delta = torch.zeros_like(base["q"], dtype=torch.float32)
    Wqkv = torch.cat([base["q"], base["k"], base["v"]], dim=0).float()
    # 融合权重上的低秩更新：B 的 q 段与 v 段分别作用
    Wqkv_merged = Wqkv.clone()
    Wqkv_merged[:q_rows] += scale * (Bq.float() @ Aq.float())
    Wqkv_merged[q_rows + k_rows:] += scale * (Bv.float() @ Av.float())
    dq = (Wqkv_merged[:q_rows] - Wq_merged).abs().max().item()
    dv = (Wqkv_merged[q_rows + k_rows:] - Wv_merged).abs().max().item()
    out.append("")
    out.append(f"数值对拍（融合权重 vs 分别合并）：q 段 max|Δ| {dq:.3e}，"
               f"v 段 max|Δ| {dv:.3e}")
    rep["numeric"] = dict(q_max_abs_diff=dq, v_max_abs_diff=dv,
                          q_rows=q_rows, k_rows=k_rows, v_rows=v_rows,
                          v_offset_correct=q_rows + k_rows,
                          v_offset_naive=q_rows)

    # 4) 在错误偏移上写入会怎样：与正确结果比较
    wrong = Wqkv.clone()
    wrong[:q_rows] += scale * (Bq.float() @ Aq.float())
    wrong[q_rows:q_rows + v_rows] += scale * (Bv.float() @ Av.float())
    d_wrong_q = (wrong[:q_rows] - Wq_merged).abs().max().item()
    d_wrong_v = (wrong[q_rows + k_rows:] - Wv_merged).abs().max().item()
    out.append(f"  错位写入（v 落在 [{q_rows}, {q_rows + v_rows})）："
               f"q 段差 {d_wrong_q:.3e}，v 段差 {d_wrong_v:.3e}")
    rep["misaligned"] = dict(q_max_abs_diff=d_wrong_q, v_max_abs_diff=d_wrong_v)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "qkv_slice_alignment.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "qkv_slice_alignment.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
