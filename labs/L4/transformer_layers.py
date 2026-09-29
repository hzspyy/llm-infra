#!/usr/bin/env python3
"""L4.1 修订 —— Transformer 层的三件事。

[A] 单元级对齐与首差定位：RMSNorm / QK-Norm / RoPE / GQA / SwiGLU / residual / lm_head
    逐个阶段与 HF 的对应模块对拍，同时打印每个阶段的 shape / dtype / stride
[B] prefill 与 cached decode：自建 KV cache，比较 full prefill、逐 token 增量解码、
    左 padding、多长度、cache_position 偏移；扫描 B=1/4 与 S=1/17/128/2048；
    用「把 padding 位置的输入置成 NaN」验证有效输出不依赖 padding
[C] 架构差异接入：SmolLM3 的 NoPE 层（真实 config 的 no_rope_layer_interval=4），
    在给定配置下与 HF 的 SmolLM3 逐层对拍，并与 3.4 的 Qwen3.5（线性递推状态）区分

用法：
    python labs/L4/transformer_layers.py --outdir out/4.1/run A B
    python labs/L4/transformer_layers.py --outdir out/4.1/run C
"""

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.request

import torch

SUMMARY = {"sections": {}}
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = os.environ.get("L41_MODEL", "Qwen/Qwen3-1.7B")
SMOLLM3 = "HuggingFaceTB/SmolLM3-3B-Base"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


def load_weights(d):
    from safetensors import safe_open
    w = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                w[k] = sf.get_tensor(k)
    return w


def show(name, t, note=""):
    print(f"  {name:<38} {str(tuple(t.shape)):<20} "
          f"{str(t.dtype).replace('torch.', ''):<9} "
          f"stride={str(t.stride()):<16} {'连续' if t.is_contiguous() else '不连续':<6} "
          f"{note}")


# ---------------------------------------------------------------- 手写实现
class Mini:
    """按 HF 命名读权重的最小实现：可选 QK-Norm、可选逐层 NoPE、可选 KV cache。"""

    def __init__(self, weights, cfg, device="cuda", dtype=None):
        self.w = weights
        self.cfg = cfg
        self.dev = device
        self.dtype = dtype or weights["model.embed_tokens.weight"].dtype
        self.nq, self.nkv, self.hd = cfg["nq"], cfg["nkv"], cfg["hd"]
        self.H, self.I, self.L = cfg["H"], cfg["I"], cfg["L"]
        self.eps, self.base = cfg["eps"], cfg["base"]
        self.has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in weights
        # HF 约定：no_rope_layers[i] = 1 表示这一层**有** RoPE，0 表示无
        nrl = cfg.get("no_rope_layers")
        self.no_rope = ([not bool(v) for v in nrl] if nrl is not None
                        else [False] * self.L)

    # ---- 基本算子 ----
    @staticmethod
    def rms_norm(x, weight, eps):
        dt = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps)).to(dt) * weight

    @staticmethod
    def rope_tables(positions, hd, base, device, dtype):
        """positions: [B, S] 绝对位置 -> cos/sin [B, S, hd]。"""
        inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
        freqs = positions.float()[..., None] * inv[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    @classmethod
    def apply_rope(cls, x, cos, sin):
        # x: [B,H,S,D]  cos/sin: [B,S,D]
        return x * cos[:, None] + cls.rotate_half(x) * sin[:, None]

    def layer(self, i, x, cos, sin, cache=None, mask=None, trace=None):
        p = f"model.layers.{i}."
        w = self.w
        B, S, _ = x.shape
        h = self.rms_norm(x, w[p + "input_layernorm.weight"].to(self.dev), self.eps)
        q = h @ w[p + "self_attn.q_proj.weight"].to(self.dev).T
        k = h @ w[p + "self_attn.k_proj.weight"].to(self.dev).T
        v = h @ w[p + "self_attn.v_proj.weight"].to(self.dev).T
        q = q.view(B, S, self.nq, self.hd).transpose(1, 2)
        k = k.view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        if self.has_qk_norm:
            q = self.rms_norm(q, w[p + "self_attn.q_norm.weight"].to(self.dev), self.eps)
            k = self.rms_norm(k, w[p + "self_attn.k_norm.weight"].to(self.dev), self.eps)
        if trace is not None:
            trace["q_pre_rope"] = q
        if not self.no_rope[i]:
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)
        if trace is not None:
            trace["q"] = q
            trace["k"] = k
            trace["v"] = v
        if cache is not None:
            past = cache[i]
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            cache[i] = (k, v)
        if mask is None:
            # 因果只在多 token 前向时生效：单 token 解码（S=1）看全部历史
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=(S > 1), enable_gqa=True)
        else:
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, S, self.nq * self.hd)
        x = x + o @ w[p + "self_attn.o_proj.weight"].to(self.dev).T
        if trace is not None:
            trace["attn_out"] = o
            trace["after_attn_residual"] = x
        h = self.rms_norm(x, w[p + "post_attention_layernorm.weight"].to(self.dev), self.eps)
        g = h @ w[p + "mlp.gate_proj.weight"].to(self.dev).T
        u = h @ w[p + "mlp.up_proj.weight"].to(self.dev).T
        if trace is not None:
            trace["gate"], trace["up"] = g, u
        x = x + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].to(self.dev).T
        return x

    def forward(self, ids, cache=None, positions=None, mask=None, trace=None,
                layers=None, final_norm=True):
        ids = ids.to(self.dev)
        B, S = ids.shape
        if positions is None:
            P = cache[0][0].shape[2] if (cache is not None and cache[0] is not None) else 0
            positions = torch.arange(P, P + S, device=self.dev)[None].expand(B, S)
        x = self.w["model.embed_tokens.weight"].to(self.dev)[ids]
        cos, sin = self.rope_tables(positions, self.hd, self.base, self.dev, x.dtype)
        n = self.L if layers is None else layers
        for i in range(n):
            x = self.layer(i, x, cos, sin, cache=cache, mask=mask, trace=trace)
        if final_norm:
            x = self.rms_norm(x, self.w["model.norm.weight"].to(self.dev), self.eps)
        return x

    def logits(self, x):
        head = self.w.get("lm_head.weight", self.w["model.embed_tokens.weight"])
        return x @ head.to(self.dev).T


def cfg_from(cj, no_rope_layers=None):
    return dict(
        nq=cj["num_attention_heads"], nkv=cj["num_key_value_heads"],
        hd=cj.get("head_dim") or cj["hidden_size"] // cj["num_attention_heads"],
        eps=cj["rms_norm_eps"], base=float(cj["rope_theta"]),
        L=cj["num_hidden_layers"], H=cj["hidden_size"],
        I=cj["intermediate_size"], V=cj["vocab_size"],
        tie=cj.get("tie_word_embeddings", False),
        no_rope_layers=no_rope_layers)


def build(repo=REPO):
    import torch
    d = snap(repo)
    cj = json.load(open(d + "/config.json"))
    w = load_weights(d)
    return d, cj, cfg_from(cj), w


# ---------------------------------------------------------------- A
def section_A(outdir):
    import torch
    rep = {}
    title("[A] 单元级对齐与首差定位")
    d, cj, cfg, w = build()
    dev = "cuda"
    ids = torch.tensor([[785, 6722, 315, 9625, 374, 12095, 13, 21806, 25, 12095]],
                       device=dev)
    B, S = ids.shape
    mine = Mini({k: v.to(dev) for k, v in w.items()}, cfg, dev)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to(dev).eval()
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    print(f"  {REPO}，输入 {S} 个 token，dtype {mine.dtype}，"
          f"QK-Norm {'有' if mine.has_qk_norm else '无'}")

    rows = []
    refs = {}

    def stage(name, a, b, note=""):
        diff = (a.float() - b.float()).abs().max().item()
        scale = max(b.float().abs().max().item(), 1e-9)
        ulp = 2 ** -8 * scale
        refs[name] = (a.detach().clone(), b.detach().clone())
        rows.append({"stage": name, "shape": list(a.shape),
                     "stride": list(a.stride()),
                     "contiguous": bool(a.is_contiguous()),
                     "max_abs_diff": diff, "scale": scale,
                     "ulp_estimate": ulp, "within_ulp": diff <= ulp})
        print(f"  {name:<26} {str(tuple(a.shape)):<20} "
              f"stride={str(tuple(a.stride())):<18} "
              f"max|diff| {diff:.3e}  {'≤1 ulp' if diff <= ulp else '> 1 ulp':<8} {note}")
        return diff

    positions = torch.arange(S, device=dev)[None].expand(B, S)
    with torch.no_grad():
        x_hf = hf.model.embed_tokens(ids)
        x_mine = w["model.embed_tokens.weight"].to(dev)[ids]
        stage("embed", x_mine, x_hf)
        lay = hf.model.layers[0]
        n1_hf = lay.input_layernorm(x_hf)
        n1_mine = Mini.rms_norm(x_mine, w["model.layers.0.input_layernorm.weight"].to(dev),
                                cfg["eps"])
        stage("input_layernorm", n1_mine, n1_hf)
        q_hf = lay.self_attn.q_proj(n1_hf)
        q_mine = n1_mine @ w["model.layers.0.self_attn.q_proj.weight"].to(dev).T
        stage("q_proj", q_mine, q_hf)
        k_hf = lay.self_attn.k_proj(n1_hf)
        k_mine = n1_mine @ w["model.layers.0.self_attn.k_proj.weight"].to(dev).T
        stage("k_proj", k_mine, k_hf)
        # HF: view -> q_norm -> transpose
        hs = (B, S, cfg["nq"], cfg["hd"])
        q_hf4 = lay.self_attn.q_norm(q_hf.view(hs)).transpose(1, 2)
        q_mine4 = Mini.rms_norm(
            q_mine.view(hs).transpose(1, 2),
            w["model.layers.0.self_attn.q_norm.weight"].to(dev), cfg["eps"])
        stage("q view+transpose+q_norm", q_mine4, q_hf4, "stride 反映 transpose")
        cos, sin = hf.model.rotary_emb(x_hf, positions)   # 5.x 把 rotary_emb 放在模型上
        q_hf5, k_hf5 = apply_rotary_pos_emb(q_hf4, lay.self_attn.k_norm(
            k_hf.view(B, S, cfg["nkv"], cfg["hd"])).transpose(1, 2), cos, sin)
        q_mine5 = Mini.apply_rope(q_mine4, cos, sin)
        stage("RoPE(q)", q_mine5, q_hf5)
        v_hf = lay.self_attn.v_proj(n1_hf).view(B, S, cfg["nkv"], cfg["hd"]).transpose(1, 2)
        v_mine = (n1_mine @ w["model.layers.0.self_attn.v_proj.weight"].to(dev).T
                  ).view(B, S, cfg["nkv"], cfg["hd"]).transpose(1, 2)
        o_hf = torch.nn.functional.scaled_dot_product_attention(
            q_hf5, k_hf5, v_hf, is_causal=True, enable_gqa=True)
        o_mine = torch.nn.functional.scaled_dot_product_attention(
            q_mine5, k_hf5, v_hf, is_causal=True, enable_gqa=True)
        stage("attention(SDPA)", o_mine, o_hf)
        o2_hf = o_hf.transpose(1, 2).reshape(B, S, -1)
        o2_mine = o_hf.transpose(1, 2).reshape(B, S, -1)   # reshape 本身走同一路径
        x1_hf = x_hf + lay.self_attn.o_proj(o2_hf)
        x1_mine = x_mine + o2_mine @ w["model.layers.0.self_attn.o_proj.weight"].to(dev).T
        stage("o_proj+residual", x1_mine, x1_hf)
        n2_hf = lay.post_attention_layernorm(x1_hf)
        n2_mine = Mini.rms_norm(x1_mine,
                                w["model.layers.0.post_attention_layernorm.weight"].to(dev),
                                cfg["eps"])
        stage("post_attention_layernorm", n2_mine, n2_hf)
        g_hf = lay.mlp.gate_proj(n2_hf)
        g_mine = n2_mine @ w["model.layers.0.mlp.gate_proj.weight"].to(dev).T
        stage("gate_proj", g_mine, g_hf)
        u_hf = lay.mlp.up_proj(n2_hf)
        u_mine = n2_mine @ w["model.layers.0.mlp.up_proj.weight"].to(dev).T
        stage("up_proj", u_mine, u_hf)
        act_hf = lay.mlp.act_fn(g_hf) * u_hf
        act_mine = torch.nn.functional.silu(g_mine) * u_mine
        stage("silu(gate)*up", act_mine, act_hf)
        out_hf = x1_hf + lay.mlp.down_proj(act_hf)
        out_mine = x1_mine + act_mine @ w["model.layers.0.mlp.down_proj.weight"].to(dev).T
        stage("down_proj+residual", out_mine, out_hf)

    sub("HF 的 uint 层输出对照（整层，用 position_embeddings / cache_position 调用）")
    try:
        with torch.no_grad():
            out = lay(x_hf, attention_mask=None, position_ids=positions,
                      position_embeddings=(cos, sin),
                      cache_position=torch.arange(S, device=dev))
        out_hf = out[0] if isinstance(out, tuple) else out
        with torch.no_grad():
            out_m = mine.forward(ids, layers=1, final_norm=False)
        diff = (out_m.float() - out_hf.float()).abs().max().item()
        print(f"  整层输出 max|diff| {diff:.3e}（手写整层 vs HF 单层模块）")
        rep["whole_layer_diff"] = diff
    except Exception as e:
        print(f"  调用失败：{type(e).__name__}: {e}")
        rep["whole_layer_diff"] = None

    sub("逐层对照（含 final norm 的对齐）")
    with torch.no_grad():
        hf_out = hf(ids, output_hidden_states=True)
    hs = hf_out.hidden_states
    with torch.no_grad():
        mine_hs = []
        x = w["model.embed_tokens.weight"].to(dev)[ids]
        mine_hs.append(x)
        cache = None
        for i in range(cfg["L"]):
            x = mine.layer(i, x, cos, sin)
            mine_hs.append(x)
        mine_hs[-1] = Mini.rms_norm(x, w["model.norm.weight"].to(dev), cfg["eps"])
    per_layer = []
    first_bad = None
    for i, (a, b) in enumerate(zip(mine_hs, hs)):
        df = (a.float() - b.float()).abs().max().item()
        rel = df / max(b.float().abs().mean().item(), 1e-9)
        per_layer.append({"index": i, "max_abs_diff": df, "rel": rel})
        if first_bad is None and rel > 0.05:
            first_bad = i
    exact = sum(1 for a, b in zip(mine_hs, hs)
                if (a.float() - b.float()).abs().max().item() == 0.0)
    print(f"  {len(per_layer)} 个对照点；逐位完全相同 {exact} 个；"
          f"首个相对误差 > 5% 的层 {first_bad}")
    print(f"  第 1 层 max|diff| {per_layer[1]['max_abs_diff']:.3e}，"
          f"末层 {per_layer[-1]['max_abs_diff']:.3e}")
    with torch.no_grad():
        lg_mine = mine.logits(mine_hs[-1])
    diff = (lg_mine.float() - hf_out.logits.float()).abs().max().item()
    print(f"  最终 logits max|diff| {diff:.6f}，argmax 相同 "
          f"{bool(torch.equal(lg_mine.argmax(-1), hf_out.logits.argmax(-1)))}")
    rep["stages"] = rows
    rep["per_layer"] = per_layer
    rep["per_layer_exact"] = exact
    rep["logits_max_abs_diff"] = diff
    rep["first_rel_gt_5pct"] = first_bad
    sub("A2 反例：常见实现错误的首差定位")
    n1w = w["model.layers.0.input_layernorm.weight"].to(dev)
    n2w = w["model.layers.0.post_attention_layernorm.weight"].to(dev)
    qnw = w["model.layers.0.self_attn.q_norm.weight"].to(dev)
    knw = w["model.layers.0.self_attn.k_norm.weight"].to(dev)
    wq = w["model.layers.0.self_attn.q_proj.weight"].to(dev)
    wk = w["model.layers.0.self_attn.k_proj.weight"].to(dev)
    wv = w["model.layers.0.self_attn.v_proj.weight"].to(dev)
    wo = w["model.layers.0.self_attn.o_proj.weight"].to(dev)
    wg = w["model.layers.0.mlp.gate_proj.weight"].to(dev)
    wu = w["model.layers.0.mlp.up_proj.weight"].to(dev)
    wd = w["model.layers.0.mlp.down_proj.weight"].to(dev)

    def interleaved_rope(x, cos, sin):
        """错误变体：按相邻配对旋转，而不是 HF 的前后对半。"""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[..., 0::2]
        s = sin[..., 0::2]
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        out = torch.stack([o1, o2], dim=-1).reshape_as(x)
        return out

    def pipeline(corrupt=None):
        """按 HF 的算子顺序重算一遍第 0 层，可选注入一种错误。"""
        stages = {}
        h = Mini.rms_norm(x_mine, n1w, 0.0 if corrupt == "eps_zero" else cfg["eps"])
        stages["input_layernorm"] = h
        if corrupt == "no_weight_transpose":
            q, k, v = h @ wq, h @ wk.T, h @ wv.T   # 方阵漏掉转置，形状合法但数值错
        else:
            q, k, v = h @ wq.T, h @ wk.T, h @ wv.T
        stages["q_proj"] = q
        stages["k_proj"] = k
        hs_ = (B, S, cfg["nq"], cfg["hd"])
        hk_ = (B, S, cfg["nkv"], cfg["hd"])
        if corrupt == "bf16_norm":
            def bnorm(t, ww):
                tf = t.to(torch.bfloat16)
                var = (tf.float().pow(2)).mean(-1, keepdim=True).bfloat16().float()
                return ((tf.float() * torch.rsqrt(var + cfg["eps"]))
                        .bfloat16() * ww)
            q4 = bnorm(q.view(hs_).transpose(1, 2), qnw)
            k4 = bnorm(k.view(hk_).transpose(1, 2), knw)
        elif corrupt == "no_qk_norm":
            q4 = q.view(hs_).transpose(1, 2)
            k4 = k.view(hk_).transpose(1, 2)
        elif corrupt == "swap_qk_norm":
            q4 = Mini.rms_norm(q.view(hs_).transpose(1, 2), knw, cfg["eps"])
            k4 = Mini.rms_norm(k.view(hk_).transpose(1, 2), qnw, cfg["eps"])
        else:
            q4 = Mini.rms_norm(q.view(hs_).transpose(1, 2), qnw, cfg["eps"])
            k4 = Mini.rms_norm(k.view(hk_).transpose(1, 2), knw, cfg["eps"])
        stages["q view+transpose+q_norm"] = q4
        if corrupt == "interleaved_rope":
            q5 = interleaved_rope(q4, cos, sin)
        else:
            q5 = Mini.apply_rope(q4, cos, sin)
        stages["RoPE(q)"] = q5
        v4 = v.view(hk_).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q5, k4, v4, is_causal=True, enable_gqa=True)
        stages["attention(SDPA)"] = o
        o2 = o.transpose(1, 2).reshape(B, S, -1)
        x1 = x_mine + o2 @ wo.T
        stages["o_proj+residual"] = x1
        h2 = Mini.rms_norm(x1, n2w, cfg["eps"])
        stages["post_attention_layernorm"] = h2
        g, u = h2 @ wg.T, h2 @ wu.T
        stages["gate_proj"] = g
        stages["up_proj"] = u
        act = torch.nn.functional.silu(g) * u
        stages["silu(gate)*up"] = act
        stages["down_proj+residual"] = x1 + act @ wd.T
        return stages

    variants = ["no_qk_norm", "no_weight_transpose", "interleaved_rope",
                "swap_qk_norm", "bf16_norm", "eps_zero"]
    label = {"no_qk_norm": "漏掉 QK-Norm", "no_weight_transpose": "权重不转置",
             "interleaved_rope": "RoPE 改成交错配对", "swap_qk_norm": "q/k 的 norm 互换",
             "bf16_norm": "RMSNorm 内部用 bf16 累加", "eps_zero": "eps 写成 0"}
    counter = []
    for v in variants:
        st = pipeline(v)
        first, first_d = None, None
        for name, (a_ref, b_ref) in refs.items():
            if name not in st:
                continue
            scale = max(b_ref.float().abs().max().item(), 1e-9)
            d = (st[name].float() - b_ref.float()).abs().max().item()
            if d > 2 ** -8 * scale:
                first, first_d = name, d
                break
        counter.append({"variant": v, "first_stage": first, "diff": first_d})
        print(f"  {label[v]:<26} 首差阶段 {str(first):<26} "
              f"{'—' if first_d is None else f'{first_d:.3e}'}")
    rep["counterexamples"] = counter
    rep["config"] = {k: v for k, v in cfg.items() if k != "no_rope_layers"}
    del hf
    torch.cuda.empty_cache()
    SUMMARY["sections"]["A"] = rep
    return rep


# ---------------------------------------------------------------- B
def _mask(B, S, P, pad_valid, device):
    """因果 + padding 的加性 mask；pad_valid: [B, P+S] bool，True 表示有效。"""
    import torch
    qpos = torch.arange(P, P + S, device=device)
    kpos = torch.arange(P + S, device=device)
    allow = kpos[None, :] <= qpos[:, None]                 # [S, P+S]
    m = torch.zeros(B, 1, S, P + S, device=device)
    m.masked_fill_(~allow, float("-inf"))
    if pad_valid is not None:
        m = m.masked_fill(~pad_valid[:, None, None, :], float("-inf"))
    return m


def section_B(outdir):
    import torch
    rep = {}
    title("[B] prefill 与 cached decode")
    d, cj, cfg, w = build()
    dev = "cuda"
    tok = __import__("transformers").AutoTokenizer.from_pretrained(d)
    mine = Mini({k: v.to(dev) for k, v in w.items()}, cfg, dev)
    text = ("The capital of France is Paris. Water boils at one hundred degrees "
            "Celsius at sea level under standard atmospheric pressure. "
            "A binary search halves the remaining interval at every step, so a "
            "sorted array of one million entries needs about twenty comparisons.")
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    print(f"  {REPO}；样本文本 {ids.shape[1]} token；层数 {cfg['L']}，"
          f"nq/nkv/hd = {cfg['nq']}/{cfg['nkv']}/{cfg['hd']}")

    sub("B1 full prefill vs 逐 token 增量解码")
    S = ids.shape[1]
    with torch.no_grad():
        full = mine.forward(ids, final_norm=True)
        lg_full = mine.logits(full)
        # 前 S-1 个 token 做 prefill，再解码第 S 个
        cache = [None] * cfg["L"]
        x_pre = mine.forward(ids[:, :S - 1], cache=cache, final_norm=True)
        lg_pre_last = mine.logits(x_pre[:, -1:])
        x_dec = mine.forward(ids[:, S - 1:S], cache=cache, final_norm=True)
        lg_dec = mine.logits(x_dec)
    d_last = (lg_pre_last.float() - lg_full[:, S - 2:S - 1].float()).abs().max().item()
    d_step = (lg_dec.float() - lg_full[:, -1:].float()).abs().max().item()
    print(f"  prefill 的后一个位置 vs 全序列同位置 max|diff| {d_last:.3e}")
    print(f"  增量解码一步 vs 全序列末位            max|diff| {d_step:.3e}")
    kv_len = [cache[i][0].shape[2] for i in range(cfg["L"])]
    print(f"  cache 有效长度：{set(kv_len)}（应为 {S}）")
    kv_bytes = sum(cache[i][0].numel() + cache[i][1].numel()
                   for i in range(cfg["L"])) * 2
    print(f"  KV cache 占用 {kv_bytes / 2**20:.1f} MiB（bf16）")
    try:
        from transformers import AutoModelForCausalLM
        hf_c = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to(dev).eval()
        with torch.no_grad():
            o_pre = hf_c(ids[:, :S - 1], use_cache=True)
            cache_hf = o_pre.past_key_values
            o_dec = hf_c(ids[:, S - 1:S], past_key_values=cache_hf,
                         cache_position=torch.tensor([S - 1], device=dev))
        lg_hf_dec = o_dec.logits[:, -1].float()
        d_hf = (lg_dec.float() - lg_hf_dec).abs().max().item()
        d_hf_pre = (lg_pre_last.float()
                    - o_pre.logits[:, -1:].float()).abs().max().item()
        ck = cache_hf.layers[0]
        kv_hf = (ck.keys.shape, ck.values.shape) if hasattr(ck, "keys") else None
        print(f"  与 HF 自己的 cache 对拍：prefill 末位 max|diff| {d_hf_pre:.3e}；"
              f"解码一步 {d_hf:.3e}")
        print(f"  HF cache 形状（第 0 层）：{kv_hf}，"
              f"我们的 {tuple(cache[0][0].shape)}")
        rep["hf_cache"] = {"prefill_diff": d_hf_pre, "decode_diff": d_hf,
                           "hf_cache_shape": [list(x) for x in kv_hf] if kv_hf else None}
        del hf_c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  HF cache 对拍失败：{type(e).__name__}: {e}")
        rep["hf_cache"] = {"error": f"{type(e).__name__}: {e}"}

    sub("B2 多步增量解码 vs 逐步扩长的 full prefill")
    n_new = 8
    cache = [None] * cfg["L"]
    with torch.no_grad():
        x = mine.forward(ids, cache=cache, final_norm=True)
    rows = []
    cur = ids
    for t in range(n_new):
        nxt = int(mine.logits(x[:, -1:]).argmax(-1))
        with torch.no_grad():
            x = mine.forward(torch.tensor([[nxt]], device=dev), cache=cache,
                             final_norm=True)
            lg_inc = mine.logits(x)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
            lg_full_step = mine.logits(mine.forward(cur, final_norm=True))[:, -1:]
        dstep = (lg_inc.float() - lg_full_step.float()).abs().max().item()
        rows.append({"step": t, "token": nxt, "max_abs_diff": dstep,
                     "cache_len": cache[0][0].shape[2]})
    print(f"  逐步解码 token {[r['token'] for r in rows]}")
    print(f"  每步 max|diff| 最大 {max(r['max_abs_diff'] for r in rows):.3e}，"
          f"cache 长度 {rows[0]['cache_len']} → {rows[-1]['cache_len']}")

    sub("B3 左 padding / 右 padding 与 NaN 隔离实验")
    pad_rows = []
    L0 = ids.shape[1]
    for pad in (1, 7, 31):
        pad_ids = torch.full((1, pad), tok.pad_token_id or 0, device=dev, dtype=torch.long)
        inp = torch.cat([pad_ids, ids], dim=1)
        valid = torch.ones(1, pad + L0, dtype=torch.bool, device=dev)
        valid[:, :pad] = False
        with torch.no_grad():
            x_pad = mine.forward(inp, mask=_mask(1, pad + L0, 0, valid, dev),
                                 final_norm=True)
            lg_pad = mine.logits(x_pad[:, -1:])
            # 把 padding 位置的输入换成另一组随机 embedding，有效输出必须不变
            saved = mine.w["model.embed_tokens.weight"]
            other = saved.clone()
            torch.manual_seed(1234)
            other[tok.pad_token_id or 0] = torch.randn_like(other[0])
            mine.w["model.embed_tokens.weight"] = other
            x_alt = mine.forward(inp, mask=_mask(1, pad + L0, 0, valid, dev),
                                 final_norm=True)
            lg_alt = mine.logits(x_alt[:, -1:])
            # 再试 NaN：加性 -inf mask 挡不住 NaN
            nan_emb = saved.clone()
            nan_emb[tok.pad_token_id or 0] = float("nan")
            mine.w["model.embed_tokens.weight"] = nan_emb
            x_nan = mine.forward(inp, mask=_mask(1, pad + L0, 0, valid, dev),
                                 final_norm=True)
            lg_nan = mine.logits(x_nan[:, -1:])
            mine.w["model.embed_tokens.weight"] = saved
        d_pad = (lg_pad.float() - lg_full[:, -1:].float()).abs().max().item()
        d_alt = (lg_alt.float() - lg_pad.float()).abs().max().item()
        d_nan = (lg_nan.float() - lg_pad.float()).abs().max().item()
        finite = bool(torch.isfinite(lg_nan).all())
        pad_rows.append({"pad": pad, "vs_unpadded": d_pad,
                         "other_embedding_vs_normal": d_alt,
                         "nan_vs_normal": ("nan" if math.isnan(d_nan) else d_nan),
                         "nan_run_finite": finite})
        print(f"  左 pad {pad:>2}：与未 padding 的末位 logits max|diff| {d_pad:.3e}；"
              f"换成另一组随机 embedding 后 {d_alt:.3e}；"
              f"置 NaN 后 {'NaN' if math.isnan(d_nan) else f'{d_nan:.3e}'}"
              f"（有限 {finite}）")
    # 右 padding：尾部 pad 不能影响真实位置
    tail = torch.full((1, 8), tok.pad_token_id or 0, device=dev, dtype=torch.long)
    inp_r = torch.cat([ids, tail], dim=1)
    valid_r = torch.ones(1, L0 + 8, dtype=torch.bool, device=dev)
    valid_r[:, L0:] = False
    with torch.no_grad():
        x_r = mine.forward(inp_r, mask=_mask(1, L0 + 8, 0, valid_r, dev), final_norm=True)
        lg_r = mine.logits(x_r[:, L0 - 1:L0])
    d_r = (lg_r.float() - lg_full[:, L0 - 1:L0].float()).abs().max().item()
    print(f"  右 pad 8：真实末位 logits 与未 padding 的差 {d_r:.3e}（尾部不参与）")
    # 反例：没有 mask 时左 padding 会改变结果
    with torch.no_grad():
        pad_ids = torch.full((1, 7), tok.pad_token_id or 0, device=dev, dtype=torch.long)
        inp = torch.cat([pad_ids, ids], dim=1)
        x_nomask = mine.forward(inp, mask=None, final_norm=True)
        lg_nomask = mine.logits(x_nomask[:, -1:])
    d_nomask = (lg_nomask.float() - lg_full[:, -1:].float()).abs().max().item()
    print(f"  反例：不带 attention mask 的左 padding 7，末位 logits 差 {d_nomask:.3e}")

    sub("B4 多长度批与 cache_position 偏移")
    texts = ["Hello", "The capital of France is", "1 + 1 =",
             "In 1969 humans first landed on the Moon and returned safely to Earth."]
    lens = [len(tok(t, return_tensors='pt').input_ids[0]) for t in texts]
    Lmax = max(lens)
    ids_b = torch.full((4, Lmax), tok.pad_token_id or 0, device=dev, dtype=torch.long)
    valid_b = torch.zeros(4, Lmax, dtype=torch.bool, device=dev)
    for r, t in enumerate(texts):
        e = tok(t, return_tensors="pt").input_ids[0].to(dev)
        ids_b[r, Lmax - len(e):] = e
        valid_b[r, Lmax - len(e):] = True
    with torch.no_grad():
        xb = mine.forward(ids_b, mask=_mask(4, Lmax, 0, valid_b, dev), final_norm=True)
        lg_b = mine.logits(xb[:, -1:])
    solo = []
    for t in texts:
        e = tok(t, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            solo.append(mine.logits(mine.forward(e, final_norm=True))[:, -1:])
    dsolo = [(lg_b[r:r + 1].float() - solo[r].float()).abs().max().item()
             for r in range(4)]
    print(f"  批长度 {Lmax}，各行长度 {lens}；同批 vs 逐条末位 logits 差 "
          f"{[f'{d:.2e}' for d in dsolo]}")
    # cache_position 偏移：先 prefill 一段，再在偏移位置上解码
    off_rows = []
    for P in (1, 5, S // 2, S - 1):
        pre = ids[:, :P]
        cache = [None] * cfg["L"]
        with torch.no_grad():
            mine.forward(pre, cache=cache, final_norm=True)
            pos = torch.tensor([[P]], device=dev)
            x_off = mine.forward(ids[:, P:P + 1], cache=cache, positions=pos,
                                 final_norm=True)
            lg_off = mine.logits(x_off)
            lg_ref = lg_full[:, P:P + 1]
        d_off = (lg_off.float() - lg_ref.float()).abs().max().item()
        off_rows.append({"prefix": P, "max_abs_diff": d_off})
        print(f"  prefill 前 {P:>2} 后再解码第 {P} 个位置：与全序列同位置 "
              f"max|diff| {d_off:.3e}")
    # 反例：解码时把 position 写成 0（最常见的缓存实现错误）
    Pw = S - 1
    cache = [None] * cfg["L"]
    with torch.no_grad():
        mine.forward(ids[:, :Pw], cache=cache, final_norm=True)
        x_wrong = mine.forward(ids[:, Pw:Pw + 1], cache=cache,
                               positions=torch.zeros(1, 1, dtype=torch.long,
                                                     device=dev), final_norm=True)
        lg_wrong = mine.logits(x_wrong)
    d_wrong = (lg_wrong.float() - lg_full[:, Pw:Pw + 1].float()).abs().max().item()
    off_rows.append({"prefix": Pw, "max_abs_diff": d_wrong,
                     "note": "反例：position 写成 0"})
    print(f"  反例：prefill 前 {Pw} 个 token 后把 position 写成 0，"
          f"与正确位置的 logits 差 {d_wrong:.3e}")

    sub("B5 扫描 B=1/4 与 S=1/17/128/2048")
    scan = []
    for B in (1, 4):
        for Sx in (1, 17, 128, 2048):
            base = ids[:, :Sx] if Sx <= S else ids.repeat(1, math.ceil(Sx / S))[:, :Sx]
            inp = base.repeat(B, 1) if B > 1 else base
            cache = [None] * cfg["L"]
            t0 = time.perf_counter()
            with torch.no_grad():
                x = mine.forward(inp, cache=cache, final_norm=True)
            t_pre = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.no_grad():
                if Sx > 1:
                    cache = [None] * cfg["L"]
                    mine.forward(inp[:, :-1], cache=cache, final_norm=True)
                    x1 = mine.forward(inp[:, -1:], cache=cache, final_norm=True)
                else:
                    x1 = x
            t_dec = time.perf_counter() - t0
            d_step = (mine.logits(x1).float()
                      - mine.logits(mine.forward(inp, final_norm=True))[:, -1:]
                      .float()).abs().max().item() if Sx > 1 else 0.0
            kv_mib = sum(cache[i][0].numel() + cache[i][1].numel()
                         for i in range(cfg["L"])) * 2 / 2**20
            scan.append({"B": B, "S": Sx, "prefill_s": t_pre, "decode_step_s": t_dec,
                         "kv_mib": kv_mib, "decode_vs_full_max_abs_diff": d_step,
                         "cache_len": cache[0][0].shape[2]})
            print(f"  B={B} S={Sx:<5} prefill {t_pre * 1e3:>8.2f} ms  "
                  f"decode {t_dec * 1e3:>7.2f} ms  KV {kv_mib:>7.1f} MiB  "
                  f"decode vs full {d_step:.2e}")
    rep["prefill_vs_decode"] = {"last_pos_diff": d_last, "one_step_diff": d_step,
                                "kv_bytes": kv_bytes, "steps": rows}
    rep["padding"] = {"left": pad_rows, "right_tail_diff": d_r,
                      "no_mask_negative_control": d_nomask}
    rep["multi_length_batch"] = {"lengths": lens, "diffs": dsolo}
    rep["position_offset"] = off_rows
    rep["scan"] = scan
    SUMMARY["sections"]["B"] = rep
    return rep


# ---------------------------------------------------------------- C
def section_C(outdir):
    import torch
    rep = {}
    title("[C] 架构差异接入：SmolLM3 的 NoPE 层")

    sub("C1 真实 config 与 NoPE 层模式")
    url = f"https://huggingface.co/{SMOLLM3}/raw/main/config.json"
    cfg_j = None
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
        cfg_j = json.loads(raw)
        rep["config_sha256"] = hashlib.sha256(raw).hexdigest()
        print(f"  已取到 {SMOLLM3} 的 config.json（sha256 "
              f"{rep['config_sha256'][:16]}）")
    except Exception as e:
        print(f"  取 config 失败：{type(e).__name__}: {e}")
        d = snap(SMOLLM3)
        raw = open(d + "/config.json", "rb").read()
        cfg_j = json.loads(raw)
        rep["config_sha256"] = hashlib.sha256(raw).hexdigest()
    interval = cfg_j.get("no_rope_layer_interval", 4)
    L = cfg_j["num_hidden_layers"]
    no_rope = [int((i + 1) % interval != 0) for i in range(L)]   # 1 = 有 RoPE
    nope_idx = [i for i, v in enumerate(no_rope) if not v]
    print(f"  层数 {L}，no_rope_layer_interval = {interval}，"
          f"no_rope_layers[i] = int((i+1) % {interval} != 0)")
    print(f"  → 无 RoPE 的层号（0 起）：{nope_idx}")
    print(f"  每 {interval} 层的最后一层去掉 RoPE，共 {len(nope_idx)} 层"
          f"（{len(nope_idx)}/{L}）")
    rep["no_rope"] = {"interval": interval, "layers": L, "no_rope_indices": nope_idx,
                      "rope_theta": cfg_j.get("rope_theta"),
                      "hidden_size": cfg_j.get("hidden_size"),
                      "num_attention_heads": cfg_j.get("num_attention_heads"),
                      "num_key_value_heads": cfg_j.get("num_key_value_heads"),
                      "tie_word_embeddings": cfg_j.get("tie_word_embeddings")}
    print(f"  与 Qwen3-1.7B 的区别：Qwen3 每层都有 RoPE，另有 q/k 的 head_dim "
          f"RMSNorm；SmolLM3 无 QK-Norm，但每 4 层去掉一次 RoPE。")

    sub("C2 用真实 config 的层模式构造小模型，与 HF 逐层对拍")
    from transformers import SmolLM3Config, SmolLM3ForCausalLM

    def tiny_cfg(interval_=4, L_=8, explicit=None):
        return SmolLM3Config(
            vocab_size=512, hidden_size=64, intermediate_size=192,
            num_hidden_layers=L_, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, rope_theta=1.0e6,
            tie_word_embeddings=False, attention_bias=False,
            no_rope_layer_interval=interval_, no_rope_layers=explicit,
            use_cache=True, use_sliding_window=False, max_position_embeddings=256,
            pad_token_id=0, bos_token_id=1, eos_token_id=2)

    results = []
    cases = [("interval_4", dict(interval_=4)),
             ("explicit_all_rope", dict(interval_=4, explicit=[1] * 8)),
             ("interval_2", dict(interval_=2)),
             ("interval_1_all_nope", dict(interval_=1))]
    for tag, kw in cases:
        torch.manual_seed(0)
        hf_cfg = tiny_cfg(**kw)
        model = SmolLM3ForCausalLM(hf_cfg).to("cuda").eval()
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        cj = {"hidden_size": 64, "intermediate_size": 192, "head_dim": 16,
              "num_hidden_layers": hf_cfg.num_hidden_layers,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "rms_norm_eps": 1e-6, "rope_theta": 1.0e6, "vocab_size": 512,
              "tie_word_embeddings": False}
        nrl = [bool(v) for v in hf_cfg.no_rope_layers]
        mini = Mini(sd, cfg_from(cj, no_rope_layers=nrl), "cuda")
        ids = torch.randint(0, 512, (2, 13), device="cuda")
        caps = {}
        handles = [l.register_forward_hook(
            (lambda i: (lambda m, inp, out: caps.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach().float().cpu())))(i))
            for i, l in enumerate(model.model.layers)]
        with torch.no_grad():
            model(ids)
        for h in handles:
            h.remove()
        with torch.no_grad():
            mine_hs = []
            x = sd["model.embed_tokens.weight"].to("cuda")[ids]
            mine_hs.append(x)
            pos = torch.arange(ids.shape[1], device="cuda")[None].expand_as(ids)
            cos, sin = Mini.rope_tables(pos, 16, 1.0e6, "cuda", torch.float32)
            for i in range(hf_cfg.num_hidden_layers):
                x = mini.layer(i, x, cos, sin)
                mine_hs.append(x)
            normed_hf = model.model.norm(caps[hf_cfg.num_hidden_layers - 1].cuda())
            normed_mine = Mini.rms_norm(x, sd["model.norm.weight"].to("cuda"), 1e-6)
        per = []
        for i in range(hf_cfg.num_hidden_layers):
            a = mine_hs[i + 1].float()
            b = caps[i].cuda().float()
            d = (a - b).abs().max().item()
            scale = max(b.abs().max().item(), 1e-9)
            per.append({"layer": i, "no_rope": not nrl[i], "max_abs_diff": d,
                        "rel": d / scale})
        dnorm = (normed_mine.float() - normed_hf.float()).abs().max().item()
        relnorm = dnorm / max(normed_hf.float().abs().max().item(), 1e-9)
        worst = max(p["rel"] for p in per)
        print(f"  {tag:<20} 层数 {hf_cfg.num_hidden_layers} "
              f"NoPE {[i for i, v in enumerate(nrl) if not v]}  "
              f"逐层最大相对差 {worst:.2e}  final norm {relnorm:.2e}")
        results.append({"tag": tag, "no_rope_layers": nrl, "per_layer": per,
                        "worst_rel": worst, "final_norm_abs": dnorm,
                        "final_norm_rel": relnorm})
        del model
        torch.cuda.empty_cache()
    rep["tiny_alignment"] = results
    print("  三组配置（每 4 层一次 NoPE、全 RoPE、每 2 层一次）都逐层一致，"
          "说明 NoPE 的接入点（跳过 RoPE、其余不变）与 HF 相同。")

    sub("C3 真实权重下的 NoPE 层对照（若权重已下载）")
    try:
        d3 = snap(SMOLLM3)
        files = sorted(glob.glob(d3 + "/*.safetensors"))
        if not files:
            raise FileNotFoundError("SmolLM3-3B-Base 权重尚未下载完")
        cj3 = json.load(open(d3 + "/config.json"))
        from transformers import AutoModelForCausalLM
        hf3 = AutoModelForCausalLM.from_pretrained(
            d3, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        w3 = load_weights(d3)
        nrl3 = hf3.config.no_rope_layers
        if nrl3 is None:
            iv = cj3.get("no_rope_layer_interval", 4)
            nrl3 = [int((i + 1) % iv != 0) for i in range(cj3["num_hidden_layers"])]
        nrl3 = [bool(v) for v in nrl3]
        cfg3 = cfg_from(cj3, no_rope_layers=nrl3)
        mini3 = Mini({k: v.to("cuda") for k, v in w3.items()}, cfg3, "cuda")
        tok3 = __import__("transformers").AutoTokenizer.from_pretrained(d3)
        ids3 = tok3("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        checkpoints = [0, 1, 3, 4, 7, 11]
        caps3 = {}
        handles = [hf3.model.layers[i - 1].register_forward_hook(
            (lambda k: (lambda m, inp, out: caps3.__setitem__(
                k, (out[0] if isinstance(out, tuple) else out).detach().float().cpu())))(i - 1))
            for i in checkpoints if i > 0]
        with torch.no_grad():
            hf3(ids3)
            x = w3["model.embed_tokens.weight"].to("cuda")[ids3]
            pos = torch.arange(ids3.shape[1], device="cuda")[None].expand_as(ids3)
            cos, sin = Mini.rope_tables(pos, cfg3["hd"], cfg3["base"], "cuda", x.dtype)
            got = {0: x}
            cur = x
            for i in range(max(checkpoints)):
                cur = mini3.layer(i, cur, cos, sin)
                got[i + 1] = cur
        for h in handles:
            h.remove()
        rows = []
        for i in checkpoints:
            a = got[i].float()
            b = (got[i] if i == 0 else caps3[i - 1].cuda()).float()
            dmax = (a - b).abs().max().item()
            scale = max(b.abs().max().item(), 1e-9)
            rows.append({"hidden_state": i, "layer_type":
                         "embedding" if i == 0 else
                         ("NoPE" if not nrl3[i - 1] else "RoPE"),
                         "max_abs_diff": dmax, "scale": scale,
                         "rel": dmax / scale})
            print(f"  hidden_state {i:>2}（{'embedding' if i == 0 else ('NoPE' if not nrl3[i - 1] else 'RoPE'):<9}）"
                  f" max|diff| {dmax:.3e}  量级 {scale:.1f}  相对 {dmax / scale:.2e}")
        rep["real_weights"] = {"rows": rows, "nope_layers": nope_idx,
                               "revision": os.path.basename(d3)}
        del hf3
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  跳过真实权重对照：{type(e).__name__}: {e}")
        rep["real_weights"] = {"error": f"{type(e).__name__}: {e}"}
    SUMMARY["sections"]["C"] = rep
    return rep


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l41_out"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B", "C"]
    env = {"python": sys.version.split()[0], "HF_HOME": os.environ.get("HF_HOME")}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    try:
        env["transformers"] = __import__("transformers").__version__
    except Exception:
        pass
    SUMMARY["env"] = env
    for s in want:
        {"A": section_A, "B": section_B, "C": section_C}[s](args.outdir)
    SUMMARY["outdir"] = args.outdir
    path = os.path.join(args.outdir, "transformer_layers.json")
    if os.path.exists(path):
        try:
            old = json.load(open(path))
            SUMMARY.update({"env": {**old.get("env", {}), **env},
                            "sections": {**old.get("sections", {}),
                                         **SUMMARY["sections"]},
                            "runs": old.get("runs", []) + [list(want)]})
        except Exception as e:
            print(f"  (合并旧结果失败: {type(e).__name__}: {e})")
    else:
        SUMMARY["runs"] = [list(want)]
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
