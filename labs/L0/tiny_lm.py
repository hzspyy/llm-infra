#!/usr/bin/env python3
"""L0.0 / L0.0b · 一个能完整读完的语言模型。

这个模型只有几万个参数，跑在 CPU 上，但结构和 Qwen3、Llama 是同一族：
token embedding、若干层「attention + MLP」、最后一个线性层输出词表上的分数。
后面每一层（L2 的 kernel、L4 的模型解剖、L7 的训练）都会回到这个模型。

刻意不用的东西，以及原因：
    F.scaled_dot_product_attention  —— 它把注意力的三步并成一个调用，看不见中间张量
    nn.TransformerEncoderLayer      —— 同上，而且它的默认设置与 decoder-only 不同
    nn.LayerNorm 之外的任何封装     —— 归一化本身在 L4.1 展开

被 walk_generate.py（L0.0）和 walk_train.py（L0.0b）共用。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 一小段中文，用作字符级语料。字符表就是词表。
CORPUS = "风吹过山谷，云落在水面。山不动，水在动，风来了又走。"


class CharVocab:
    """字符级词表：一个字符一个 token。真实模型用 BPE（见 L0.4）。"""

    def __init__(self, text: str) -> None:
        self.itos = sorted(set(text))          # id -> 字符
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class TinyAttention(nn.Module):
    """单层因果自注意力，多头。每一步的张量都留在局部变量里，便于打印。"""

    def __init__(self, d_model: int, n_head: int, block_size: int, scale: bool = True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = scale
        # 一个线性层同时产生 Q、K、V：输出维度是 3*d_model，之后再切三份。
        # 真实模型（含 vLLM 的 QKVParallelLinear）也这样做，因为一次 GEMM 比三次快。
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # 因果掩码：位置 i 只能看到 <= i。register_buffer 表示它是状态但不是参数。
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        B, T, C = x.shape                       # batch, 序列长度, d_model
        qkv = self.qkv(x)                       # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)           # 各 [B, T, C]
        # 拆头：把 C 拆成 (n_head, d_head)，再把 head 维换到前面，
        # 这样后面的矩阵乘就是每个头各算各的。
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)   # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1)           # [B, nh, T, T] 每对位置的相似度
        if self.scale:
            att = att / math.sqrt(self.d_head)  # 缩放：见 walk_generate.py 的消融 [A]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)            # 每一行是一个概率分布
        out = att @ v                           # [B, nh, T, hd] 按权重取 V 的加权和
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # 合头
        out = self.proj(out)

        if trace is not None:
            trace.update(qkv=qkv, q=q, k=k, v=v, att_logits=q @ k.transpose(-2, -1),
                         att=att, attn_out=out)
        return out


class TinyMLP(nn.Module):
    """两层前馈网络。中间维度取 4*d_model 是 Transformer 论文以来的惯例。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        h = self.fc1(x)
        a = F.gelu(h)
        y = self.fc2(a)
        if trace is not None:
            trace.update(mlp_hidden=h, mlp_act=a, mlp_out=y)
        return y


class TinyBlock(nn.Module):
    """一层：norm -> attention -> 残差；norm -> MLP -> 残差。

    先 norm 后子层（pre-norm）是现在的主流写法；原始 Transformer 是 post-norm。
    差别与训练稳定性有关，L4.1 展开。
    """

    def __init__(self, d_model: int, n_head: int, block_size: int, scale: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, n_head, block_size, scale)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TinyMLP(d_model)

    def forward(self, x: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), trace)
        x = x + self.mlp(self.ln2(x), trace)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_head: int = 4,
                 n_layer: int = 2, block_size: int = 16,
                 scale: bool = True, use_pos: bool = True):
        super().__init__()
        self.block_size = block_size
        self.use_pos = use_pos
        self.tok_emb = nn.Embedding(vocab_size, d_model)      # 查表：id -> 向量
        self.pos_emb = nn.Embedding(block_size, d_model)      # 位置也查表
        self.blocks = nn.ModuleList(
            [TinyBlock(d_model, n_head, block_size, scale) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        """idx: [B, T] 的 token id -> logits: [B, T, vocab]。"""
        B, T = idx.shape
        assert T <= self.block_size, f"序列长 {T} 超过 block_size {self.block_size}"
        tok = self.tok_emb(idx)                               # [B, T, C]
        if self.use_pos:
            pos = self.pos_emb(torch.arange(T, device=idx.device))   # [T, C]
            x = tok + pos                                     # 广播到 [B, T, C]
        else:
            x = tok                                           # 消融 [B]：不加位置
        if trace is not None:
            trace.update(tok_emb=tok, x_in=x)
        for blk in self.blocks:
            x = blk(x, trace)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # [B, T, vocab]
        if trace is not None:
            trace.update(x_final=x, logits=logits)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n_new: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """自回归生成：每次只用最后一个位置的 logits 采一个 token，再拼回去。"""
        for _ in range(n_new):
            ctx = idx[:, -self.block_size:]         # 超长就截断（真实系统见 L5.3）
            logits = self(ctx)[:, -1, :]            # 只要最后一个位置
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def param_table(model: nn.Module) -> tuple[list[tuple[str, tuple, int]], int]:
    """列出每个参数张量的名字、形状与元素个数。"""
    rows = [(n, tuple(p.shape), p.numel()) for n, p in model.named_parameters()]
    return rows, sum(r[2] for r in rows)


def show(name: str, t: torch.Tensor, k: int = 4) -> str:
    """统一的张量摘要：形状、dtype、是否连续、前 k 个数。"""
    flat = t.detach().reshape(-1)[:k]
    nums = ", ".join(f"{v:+.4f}" for v in flat.tolist())
    return (f"{name:<16s} {str(tuple(t.shape)):<18s} {str(t.dtype).replace('torch.',''):<8s} "
            f"contig={'是' if t.is_contiguous() else '否':<2s} 前{k}个: [{nums}]")
