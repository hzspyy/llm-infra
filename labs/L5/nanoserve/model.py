#!/usr/bin/env python3
"""nanoserve 的模型层：4.1 的手写 Qwen3 + 一个真正的 block table。

和 4.1 的区别只有一处：KV 不再是每条序列一段连续张量，而是写进一个全局块池，
读的时候按 block table 收集。写用 slot_mapping，收集用 gather —— 这两个名字
在 vLLM 里是同一个意思。

不是 paged attention kernel：这里先把 KV 收集成连续张量再交给 SDPA，
分页只体现在寻址上，不体现在 kernel 内部。代价的量级见 3.3。
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def snapshot_dir(repo: str, hub: str) -> str:
    d = f"{hub}/models--{repo.replace('/', '--')}/snapshots"
    matches = sorted(glob.glob(d + "/*"))
    if not matches:
        raise FileNotFoundError(f"{repo} 不在 {hub}")
    return matches[0]


def rms_norm(x, weight, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dt) * weight


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


@dataclass
class ModelConfig:
    n_layers: int
    n_q: int
    n_kv: int
    head_dim: int
    hidden: int
    vocab: int
    eps: float
    rope_base: float
    tie_embeddings: bool
    max_pos: int


class Layer:
    def __init__(self, w, i, cfg: ModelConfig, device, dtype):
        p = f"model.layers.{i}."
        def g(name):
            return w[p + name].to(device=device, dtype=dtype)
        self.wq, self.wk, self.wv, self.wo = (g("self_attn.q_proj.weight"),
                                              g("self_attn.k_proj.weight"),
                                              g("self_attn.v_proj.weight"),
                                              g("self_attn.o_proj.weight"))
        self.qn, self.kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.n1, self.n2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.gate, self.up, self.down = (g("mlp.gate_proj.weight"),
                                         g("mlp.up_proj.weight"),
                                         g("mlp.down_proj.weight"))
        self.cfg = cfg


class PagedModel:
    """一个 Qwen3 前向 + 一个全局 KV 块池。块的分配由 engine 负责。"""

    def __init__(self, repo, hub, num_blocks, block_size, device="cuda",
                 dtype=torch.bfloat16):
        d = snapshot_dir(repo, hub)
        raw = json.load(open(d + "/config.json"))
        self.cfg = ModelConfig(
            n_layers=raw["num_hidden_layers"], n_q=raw["num_attention_heads"],
            n_kv=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            hidden=raw["hidden_size"], vocab=raw["vocab_size"],
            eps=raw["rms_norm_eps"], rope_base=float(raw["rope_theta"]),
            tie_embeddings=raw.get("tie_word_embeddings", False),
            max_pos=raw["max_position_embeddings"])
        self.snapshot, self.raw_config = d, raw
        self.device, self.dtype = torch.device(device), dtype
        self.block_size, self.num_blocks = block_size, num_blocks

        from safetensors import safe_open
        w = {}
        for f in sorted(glob.glob(d + "/*.safetensors")):
            with safe_open(f, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    w[k] = sf.get_tensor(k)
        c = self.cfg
        self.embed = w["model.embed_tokens.weight"].to(self.device, dtype)
        self.norm_f = w["model.norm.weight"].to(self.device, dtype)
        self.lm_head = (self.embed if c.tie_embeddings
                        else w["lm_head.weight"].to(self.device, dtype))
        self.layers = [Layer(w, i, c, self.device, dtype) for i in range(c.n_layers)]
        del w

        # 块池：每层一份 K 和一份 V。形状里的 num_blocks*block_size 就是「槽位」。
        shape = (c.n_layers, num_blocks * block_size, c.n_kv, c.head_dim)
        self.k_cache = torch.zeros(shape, device=self.device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=self.device, dtype=dtype)

        inv = 1.0 / (c.rope_base ** (torch.arange(0, c.head_dim, 2,
                                                  device=self.device).float() / c.head_dim))
        pos = torch.arange(c.max_pos, device=self.device).float()
        emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
        self.cos, self.sin = emb.cos().to(dtype), emb.sin().to(dtype)

    def kv_bytes_per_token(self) -> int:
        c = self.cfg
        return 2 * c.n_layers * c.n_kv * c.head_dim * self.k_cache.element_size()

    def cache_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2

    @torch.inference_mode()
    def forward(self, input_ids, positions, slot_mapping, gather_idx, ctx_lens):
        """一次批量前向。

        input_ids   [B, Q]   本步要算的 token（prefill chunk 或 1 个 decode token）
        positions   [B, Q]   每个 token 的绝对位置，喂 RoPE
        slot_mapping[B, Q]   每个 token 的 KV 写到池子的哪个槽位；-1 表示 padding
        gather_idx  [B, C]   读 KV 时收集哪些槽位（C = 本批最长上下文）
        ctx_lens    [B]      收完这一步之后每条序列的上下文长度
        返回 logits [B, Q, V]。padding 位置的输出是垃圾，由调用方按 q_len 取。
        """
        c = self.cfg
        B, Q = input_ids.shape
        C = gather_idx.shape[1]
        x = self.embed[input_ids]                                   # [B, Q, H]
        cos = self.cos[positions]                                   # [B, Q, hd]
        sin = self.sin[positions]

        # 写入槽位用的下标：padding 位置统一指向 0 号槽，再用 valid 掩掉。
        write = slot_mapping.reshape(-1)
        valid_write = write >= 0
        write_safe = torch.where(valid_write, write, torch.zeros_like(write))

        # 读 mask：ctx 里位置 < ctx_len 的才是真数据；再叠一层因果。
        ctx_pos = torch.arange(C, device=self.device)[None, :]      # [1, C]
        keep = ctx_pos < ctx_lens[:, None]                          # [B, C]
        causal = ctx_pos[None, :, :] <= positions[:, :, None]       # [B, Q, C]
        mask = (keep[:, None, :] & causal)[:, None]                 # [B, 1, Q, C]
        bias = torch.zeros(mask.shape, device=self.device, dtype=self.dtype)
        bias.masked_fill_(~mask, float("-inf"))

        for i, lay in enumerate(self.layers):
            h = rms_norm(x, lay.n1, c.eps)
            q = (h @ lay.wq.T).view(B, Q, c.n_q, c.head_dim)
            k = (h @ lay.wk.T).view(B, Q, c.n_kv, c.head_dim)
            v = (h @ lay.wv.T).view(B, Q, c.n_kv, c.head_dim)
            q = rms_norm(q, lay.qn, c.eps)
            k = rms_norm(k, lay.kn, c.eps)
            q = q * cos[:, :, None] + rotate_half(q) * sin[:, :, None]
            k = k * cos[:, :, None] + rotate_half(k) * sin[:, :, None]

            # ---- 写：slot_mapping 决定这一步的 KV 落在池子哪里 ----
            kf = k.reshape(-1, c.n_kv, c.head_dim)
            vf = v.reshape(-1, c.n_kv, c.head_dim)
            keep_rows = valid_write.nonzero(as_tuple=True)[0]
            self.k_cache[i].index_copy_(0, write_safe[keep_rows], kf[keep_rows])
            self.v_cache[i].index_copy_(0, write_safe[keep_rows], vf[keep_rows])

            # ---- 读：block table 展开成的 gather_idx 决定收集哪些槽位 ----
            kk = self.k_cache[i][gather_idx].transpose(1, 2)         # [B, n_kv, C, hd]
            vv = self.v_cache[i][gather_idx].transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), kk, vv, attn_mask=bias, enable_gqa=True)
            x = x + o.transpose(1, 2).reshape(B, Q, c.n_q * c.head_dim) @ lay.wo.T

            h = rms_norm(x, lay.n2, c.eps)
            x = x + (F.silu(h @ lay.gate.T) * (h @ lay.up.T)) @ lay.down.T

        x = rms_norm(x, self.norm_f, c.eps)
        return x @ self.lm_head.T
