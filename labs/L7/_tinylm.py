"""7.0b 共用的最小因果语言模型与二维 flow 小网络。

TinyLM 保留真实 causal LM 的全部监督接口：token embedding、因果 attention、
RMSNorm、绑定的 lm_head。默认 FP64，权重由固定 seed 生成，任何一次运行都可复算。
FlowNet 提供一个连续目标的对照，使"一次完整更新"不依赖语言模型 loss。
"""
from __future__ import annotations

import torch
import torch.nn as nn

DTYPE = torch.float64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=DTYPE):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class TinyLM(nn.Module):
    """vocab×dim 绑定词表的单层因果 LM：embed → attn → MLP → norm → lm_head。"""

    def __init__(self, vocab: int = 16, dim: int = 8, seed: int = 0, dtype=DTYPE):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab, self.dim = vocab, dim
        self.embed = nn.Embedding(vocab, dim, dtype=dtype)
        self.attn_norm = RMSNorm(dim, dtype=dtype)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False, dtype=dtype)
        self.proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.mlp_norm = RMSNorm(dim, dtype=dtype)
        self.up = nn.Linear(dim, 2 * dim, bias=False, dtype=dtype)
        self.down = nn.Linear(2 * dim, dim, bias=False, dtype=dtype)
        self.final_norm = RMSNorm(dim, dtype=dtype)
        self.dropout = nn.Dropout(0.0)
        for module in (self.qkv, self.proj, self.up, self.down):
            nn.init.normal_(module.weight, std=0.3)
        nn.init.normal_(self.embed.weight, std=0.3)

    def forward(self, input_ids, attention_mask=None):
        """attention_mask=None 表示所有位置都可见；padding 位置传 0。"""
        b, t = input_ids.shape
        h = self.embed(input_ids)

        q, k, v = self.qkv(self.attn_norm(h)).chunk(3, dim=-1)
        scores = q @ k.transpose(-1, -2) / (self.dim ** 0.5)
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        block = causal.expand(b, t, t).clone()
        if attention_mask is not None:
            block = block & attention_mask[:, None, :].bool()
        scores = scores.masked_fill(~block, float("-inf"))
        # 一行全部被屏蔽时 softmax 会产生 NaN，用全零权重代替。
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        h = h + self.proj(self.dropout(weights @ v))

        h = h + self.down(torch.nn.functional.silu(self.up(self.mlp_norm(h))))
        return self.final_norm(h) @ self.embed.weight.T  # 绑定词表


class FlowNet(nn.Module):
    """二维 flow matching 的速度场：输入 (x_t, t)，输出两个坐标的速度。"""

    def __init__(self, hidden: int = 8, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(3, hidden, dtype=DTYPE)
        self.fc2 = nn.Linear(hidden, 2, dtype=DTYPE)

    def forward(self, x_t, t):
        z = torch.cat([x_t, t], dim=-1)
        return self.fc2(torch.tanh(self.fc1(z)))


def parameter_table(model: nn.Module) -> list[tuple[str, tuple[int, ...], int, bool]]:
    return [(name, tuple(p.shape), p.numel(), p.requires_grad)
            for name, p in model.named_parameters()]
