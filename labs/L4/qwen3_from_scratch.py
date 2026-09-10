#!/usr/bin/env python3
"""L4.1 —— 从 safetensors 直接读权重，手写一遍 Qwen3，逐层和 HF 对齐。

不用 transformers 的任何 Module，只用 torch 的基本算子。
每一步都打印张量的 shape / dtype / stride，然后和 HF 的对应中间结果比。

验收标准：**每一层的输出与 HF 的最大绝对误差在 bf16 舍入量级内。**
只比最终 logits 是不够的——中间某一层错了、后面又被 norm 拉回来的情况很常见。

  [A] 配置与权重清单
  [B] 一层的每个张量：shape / dtype / stride 全打印
  [C] 逐层对齐：手写 vs HF
  [D] 参数量与 FLOP 的解析模型 vs 实际

用法：
    python qwen3_from_scratch.py
"""

import glob
import json
import math
import os
import struct
import sys

import torch

MB = 1024 * 1024
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = os.environ.get("L41_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def snap(repo):
    d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots"
    return sorted(glob.glob(d + "/*"))[0]


def load_weights(d):
    from safetensors import safe_open
    w = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                w[k] = sf.get_tensor(k)
    return w


def show(name, t, note=""):
    print(f"  {name:<40} {str(tuple(t.shape)):<22} {str(t.dtype).replace('torch.',''):<9} "
          f"stride={str(t.stride()):<18} {'连续' if t.is_contiguous() else '不连续'} {note}")


# ---------------------------------------------------------------- 手写实现
def rms_norm(x, weight, eps):
    """RMSNorm：不减均值，只除以均方根。比 LayerNorm 少一次归约。"""
    dt = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x.to(dt) * weight)


def rope_tables(seq, head_dim, base, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float()
    freqs = torch.outer(pos, inv)                 # [S, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [S, head_dim]  HF 的排布
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, S, D]   cos/sin: [S, D]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class Layer:
    def __init__(self, w, i, cfg):
        p = f"model.layers.{i}."
        self.wq = w[p + "self_attn.q_proj.weight"]
        self.wk = w[p + "self_attn.k_proj.weight"]
        self.wv = w[p + "self_attn.v_proj.weight"]
        self.wo = w[p + "self_attn.o_proj.weight"]
        self.qn = w[p + "self_attn.q_norm.weight"]
        self.kn = w[p + "self_attn.k_norm.weight"]
        self.n1 = w[p + "input_layernorm.weight"]
        self.n2 = w[p + "post_attention_layernorm.weight"]
        self.gate = w[p + "mlp.gate_proj.weight"]
        self.up = w[p + "mlp.up_proj.weight"]
        self.down = w[p + "mlp.down_proj.weight"]
        self.cfg = cfg

    def forward(self, x, cos, sin, trace=False):
        c = self.cfg
        B, S, H = x.shape
        nq, nkv, hd = c["nq"], c["nkv"], c["hd"]
        eps = c["eps"]

        # ---- attention 分支 ----
        h = rms_norm(x, self.n1, eps)
        q = h @ self.wq.T
        k = h @ self.wk.T
        v = h @ self.wv.T
        q = q.view(B, S, nq, hd).transpose(1, 2)          # [B, nq, S, hd]
        k = k.view(B, S, nkv, hd).transpose(1, 2)
        v = v.view(B, S, nkv, hd).transpose(1, 2)
        # QK-Norm：在 head_dim 上做 RMSNorm（Qwen3 特有）
        q = rms_norm(q, self.qn, eps)
        k = rms_norm(k, self.kn, eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, S, nq * hd)
        x = x + o @ self.wo.T

        # ---- MLP 分支 ----
        h = rms_norm(x, self.n2, eps)
        g = h @ self.gate.T
        u = h @ self.up.T
        x = x + (torch.nn.functional.silu(g) * u) @ self.down.T
        if trace:
            return x, dict(after_norm1=h, q=q, k=k, v=v, attn_out=o, gate=g, up=u)
        return x


def build(repo=REPO):
    d = snap(repo)
    cfg_j = json.load(open(d + "/config.json"))
    w = load_weights(d)
    cfg = dict(
        nq=cfg_j["num_attention_heads"], nkv=cfg_j["num_key_value_heads"],
        hd=cfg_j.get("head_dim", cfg_j["hidden_size"] // cfg_j["num_attention_heads"]),
        eps=cfg_j["rms_norm_eps"], base=float(cfg_j["rope_theta"]),
        L=cfg_j["num_hidden_layers"], H=cfg_j["hidden_size"],
        I=cfg_j["intermediate_size"], V=cfg_j["vocab_size"],
        tie=cfg_j.get("tie_word_embeddings", False))
    return d, cfg_j, cfg, w


def run_mine(ids, w, cfg, upto=None):
    dev = ids.device
    emb = w["model.embed_tokens.weight"].to(dev)
    x = emb[ids]
    cos, sin = rope_tables(ids.shape[1], cfg["hd"], cfg["base"], dev, x.dtype)
    outs = [x]
    n = cfg["L"] if upto is None else upto
    for i in range(n):
        lay = Layer({k: v.to(dev) for k, v in w.items()
                     if k.startswith(f"model.layers.{i}.")}, i, cfg)
        x = lay.forward(x, cos, sin)
        outs.append(x)
    return x, outs


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 配置与权重清单")
    d, cfg_j, cfg, w = build()
    keys = ["hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "rms_norm_eps", "rope_theta", "vocab_size", "tie_word_embeddings",
            "hidden_act"]
    print(f"  {REPO}")
    for k in keys:
        if k in cfg_j:
            print(f"    {k:<26} {cfg_j[k]}")
    print(f"\n  GQA 分组数 g = nq/nkv = {cfg['nq']}/{cfg['nkv']} = "
          f"{cfg['nq'] // cfg['nkv']}")
    print(f"  注意 hidden_size({cfg['H']}) != nq×head_dim"
          f"({cfg['nq']}×{cfg['hd']}={cfg['nq'] * cfg['hd']})"
          if cfg["H"] != cfg["nq"] * cfg["hd"] else
          f"  hidden_size = nq×head_dim = {cfg['nq'] * cfg['hd']} ✓")
    print(f"  权重张量共 {len(w)} 个")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 一层里每个张量的 shape / dtype / stride")
    d, cfg_j, cfg, w = build()
    dev = "cuda"
    ids = torch.tensor([[785, 6722, 315, 9625, 374, 12095, 13]], device=dev)
    B, S = ids.shape
    emb = w["model.embed_tokens.weight"].to(dev)
    x = emb[ids]
    print(f"  输入 {S} 个 token")
    show("token ids", ids)
    show("embed_tokens 权重", emb)
    show("x = emb[ids]", x, "<- 这是一次 gather，不是矩阵乘")

    cos, sin = rope_tables(S, cfg["hd"], cfg["base"], dev, x.dtype)
    show("rope cos", cos)
    show("rope sin", sin)

    lay = Layer({k: v.to(dev) for k, v in w.items()
                 if k.startswith("model.layers.0.")}, 0, cfg)
    sub("attention 分支")
    h = rms_norm(x, lay.n1, cfg["eps"])
    show("input_layernorm 之后", h)
    q = h @ lay.wq.T; k = h @ lay.wk.T; v = h @ lay.wv.T
    show("q = h @ Wq^T", q)
    show("k = h @ Wk^T", k, "<- 只有 q 的一半宽（GQA）")
    show("v = h @ Wv^T", v)
    nq, nkv, hd = cfg["nq"], cfg["nkv"], cfg["hd"]
    q4 = q.view(B, S, nq, hd).transpose(1, 2)
    k4 = k.view(B, S, nkv, hd).transpose(1, 2)
    show("q.view(B,S,nq,hd).transpose(1,2)", q4, "<- 2.0 说过：transpose 之后不连续")
    show("k 同上", k4)
    q4 = rms_norm(q4, lay.qn, cfg["eps"])
    show("QK-Norm 之后的 q", q4, f"<- 权重形状 {tuple(lay.qn.shape)}，作用在 head_dim 上")
    q4 = apply_rope(q4, cos, sin)
    show("RoPE 之后的 q", q4)
    v4 = v.view(B, S, nkv, hd).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=True, enable_gqa=True)
    show("attention 输出", o)
    o2 = o.transpose(1, 2).reshape(B, S, nq * hd)
    show("reshape 回 [B,S,nq*hd]", o2, "<- 这里必须先 contiguous，reshape 会静默拷贝")
    show("o_proj 权重", lay.wo)

    sub("MLP 分支")
    x2 = x + o2 @ lay.wo.T
    h2 = rms_norm(x2, lay.n2, cfg["eps"])
    show("post_attention_layernorm 之后", h2)
    g = h2 @ lay.gate.T
    u = h2 @ lay.up.T
    show("gate_proj 输出", g)
    show("up_proj 输出", u)
    show("silu(gate)*up", torch.nn.functional.silu(g) * u)
    show("down_proj 权重", lay.down)


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 逐层对齐：手写实现 vs HF")
    from transformers import AutoModelForCausalLM
    d, cfg_j, cfg, w = build()
    dev = "cuda"
    ids = torch.tensor([[785, 6722, 315, 9625, 374, 12095, 13,
                         21806, 25, 12095]], device=dev)

    hf = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    with torch.no_grad():
        out = hf(ids, output_hidden_states=True)
    hs = out.hidden_states           # tuple: L+1 个 [B,S,H]，第 0 个是 embedding
    print(f"  HF 给出 {len(hs)} 个 hidden_states（embedding + {cfg['L']} 层）")

    with torch.no_grad():
        mine, outs = run_mine(ids, w, cfg)
    print(f"  手写实现给出 {len(outs)} 个")

    # 注意：HF 的 hidden_states 最后一个**已经过了 final norm**，
    # 而我的 outs 最后一个是第 L-1 层的原始输出。不处理这一点会看到
    # 最后一行相对误差 165473% —— 那是对比方式错了，不是实现错了。
    outs_cmp = list(outs[:-1]) + [
        rms_norm(outs[-1], w["model.norm.weight"].to(dev), cfg["eps"])]
    print("  （HF 的最后一个 hidden_state 已含 final norm，这里对齐了再比）")

    print(f"\n  {'层':>4} {'HF 的均方根':>13} {'max|diff|':>13} {'相对误差':>12} {'判定':>6}")
    bad = 0
    for i, (a, b) in enumerate(zip(outs_cmp, hs)):
        diff = (a.float() - b.float()).abs().max().item()
        scale = b.float().abs().mean().item()
        rel = diff / max(scale, 1e-9)
        ok = rel < 0.05
        bad += (not ok)
        if i < 3 or i > len(outs_cmp) - 4 or not ok:
            print(f"  {i:>4} {scale:>13.5f} {diff:>13.6f} {rel:>11.2%} "
                  f"{'✓' if ok else '✗':>6}")
    print(f"  ... 共 {len(outs_cmp)} 个对照点，{len(outs_cmp) - bad} 个通过")
    exact = sum(1 for a, b in zip(outs_cmp, hs)
                if (a.float() - b.float()).abs().max().item() == 0.0)
    print(f"  其中 **逐位完全相同** 的有 {exact} 个")

    sub("最终 logits")
    with torch.no_grad():
        fin = rms_norm(mine, w["model.norm.weight"].to(dev), cfg["eps"])
        head = w.get("lm_head.weight", w["model.embed_tokens.weight"]).to(dev)
        my_logits = fin @ head.T
    hf_logits = out.logits
    diff = (my_logits.float() - hf_logits.float()).abs().max().item()
    print(f"  手写 logits   {tuple(my_logits.shape)}")
    print(f"  HF   logits   {tuple(hf_logits.shape)}")
    print(f"  max|diff| = {diff:.6f}   (bf16 在这个量级的分辨率约 "
          f"{2 ** -8 * hf_logits.float().abs().max().item():.4f})")
    print(f"  argmax 相同: {torch.equal(my_logits.argmax(-1), hf_logits.argmax(-1))}")
    top = my_logits[0, -1].topk(5)
    top_hf = hf_logits[0, -1].topk(5)
    print(f"  手写 top5 id: {top.indices.tolist()}")
    print(f"  HF   top5 id: {top_hf.indices.tolist()}")
    del hf
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 参数量与 FLOP 的解析模型")
    d, cfg_j, cfg, w = build()
    H, I, L = cfg["H"], cfg["I"], cfg["L"]
    nq, nkv, hd, V = cfg["nq"], cfg["nkv"], cfg["hd"], cfg["V"]

    p_q = H * nq * hd
    p_k = H * nkv * hd
    p_v = H * nkv * hd
    p_o = nq * hd * H
    p_attn = p_q + p_k + p_v + p_o + 2 * hd            # +q_norm,k_norm
    p_mlp = 3 * H * I
    p_norm = 2 * H
    p_layer = p_attn + p_mlp + p_norm
    p_emb = V * H
    total = L * p_layer + p_emb + H + (0 if cfg["tie"] else V * H)

    print(f"  每层：")
    print(f"    q_proj  {H}×{nq * hd:<6} = {p_q:>12,}")
    print(f"    k_proj  {H}×{nkv * hd:<6} = {p_k:>12,}")
    print(f"    v_proj  {H}×{nkv * hd:<6} = {p_v:>12,}")
    print(f"    o_proj  {nq * hd}×{H:<6} = {p_o:>12,}")
    print(f"    q/k_norm  2×{hd:<10} = {2 * hd:>12,}")
    print(f"    attn 小计               {p_attn:>12,}  ({p_attn / p_layer:.1%})")
    print(f"    gate/up/down 3×{H}×{I} = {p_mlp:>12,}  ({p_mlp / p_layer:.1%})")
    print(f"    2 个 RMSNorm            {p_norm:>12,}")
    print(f"    每层合计                {p_layer:>12,}")
    print(f"\n  {L} 层           {L * p_layer:>12,}")
    print(f"  embedding        {p_emb:>12,}  ({p_emb / total:.1%})")
    print(f"  final norm       {H:>12,}")
    print(f"  tie={cfg['tie']}，lm_head {'复用 embedding' if cfg['tie'] else '单独一份'}")
    print(f"  ---------------------------------")
    print(f"  解析模型总计     {total:>12,}")

    real = sum(v.numel() for k, v in w.items()
               if not (cfg["tie"] and k == "lm_head.weight"))
    print(f"  文件里实际       {real:>12,}   差 {total - real:+,}")
    print(f"  （tie=True 时把文件里那份冗余的 lm_head 排除，见 4.0 §5）")

    sub("FLOP：一次前向，每个 token")
    f_attn_proj = 2 * (p_q + p_k + p_v + p_o)
    f_mlp = 2 * p_mlp
    print(f"    attention 投影  2×{p_q + p_k + p_v + p_o:,} = {f_attn_proj:>12,}")
    print(f"    MLP             2×{p_mlp:,} = {f_mlp:>12,}")
    print(f"    每层小计（不含 attention 本身）     {f_attn_proj + f_mlp:>12,}")
    print(f"    {L} 层                             {L * (f_attn_proj + f_mlp):>12,}")
    print(f"    lm_head 2×{V}×{H}                  {2 * V * H:>12,}")
    print(f"\n  经验规则：每 token 前向 FLOP ≈ 2 × 参数量 = {2 * total:,}")
    print(f"  上面逐项加起来 = {L * (f_attn_proj + f_mlp) + 2 * V * H:,}")
    print("  差在 embedding（gather 不是矩阵乘，0 FLOP）与 attention 本身")
    print(f"  （后者是 O(S²)，与 S 有关，见 0.2 与 3.1）。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  model {REPO}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
