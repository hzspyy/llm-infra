#!/usr/bin/env python3
"""L5.7 任务 B —— 把 nanoserve 的 decode 读路径换成真实分页 kernel（不改原文件）。

约定：`engine.py` / `model.py` 保持不动，这样 5.7 与 5.8 的既有工件仍可复现。
这里用子类补一条并行执行器：

  `PagedModelExec.decode_forward(...)`  逐层与 `PagedModel.forward` 相同，
  只把「gather 成连续张量 → SDPA」换成 `paged_decode_attention`（按块表寻址）。
  只在 Q=1（纯 decode）时启用；prefill 仍走原路径。

  `PagedEngine._run(...)`  给 decode 组传块表与序列长度，其余交给基类。

两条路径共用同一份 `k_cache`/`v_cache`，所以可以逐 logits 对拍。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import Engine                                  # noqa: E402
from model import PagedModel, rms_norm, rotate_half        # noqa: E402
from paged_attn import paged_chunk_attention, paged_decode_attention  # noqa: E402


class PagedModelExec(PagedModel):
    """在同一个权重与缓存上，为 decode 步提供分页读路径。"""

    def decode_forward(self, input_ids, positions, slot_mapping,
                       block_tables, seq_lens):
        """Q=1 的 decode 前向：KV 按块表读，不写回（KV 由调用方或本函数写入）。"""
        c = self.cfg
        B, Q = input_ids.shape
        assert Q == 1, "分页路径只处理 decode（每个序列一个 token）"
        x = self.embed[input_ids]
        cos = self.cos[positions]
        sin = self.sin[positions]

        write = slot_mapping.reshape(-1)
        valid_write = write >= 0
        write_safe = torch.where(valid_write, write, torch.zeros_like(write))

        for i, lay in enumerate(self.layers):
            h = rms_norm(x, lay.n1, c.eps)
            q = (h @ lay.wq.T).view(B, Q, c.n_q, c.head_dim)
            k = (h @ lay.wk.T).view(B, Q, c.n_kv, c.head_dim)
            v = (h @ lay.wv.T).view(B, Q, c.n_kv, c.head_dim)
            q = rms_norm(q, lay.qn, c.eps)
            k = rms_norm(k, lay.kn, c.eps)
            q = q * cos[:, :, None] + rotate_half(q) * sin[:, :, None]
            k = k * cos[:, :, None] + rotate_half(k) * sin[:, :, None]

            kf = k.reshape(-1, c.n_kv, c.head_dim)
            vf = v.reshape(-1, c.n_kv, c.head_dim)
            keep = valid_write.nonzero(as_tuple=True)[0]
            self.k_cache[i].index_copy_(0, write_safe[keep], kf[keep])
            self.v_cache[i].index_copy_(0, write_safe[keep], vf[keep])

            # ---- 分页读：按块表寻址，不做 gather ----
            o = paged_decode_attention(
                q.reshape(B, c.n_q, c.head_dim),
                self.k_cache[i], self.v_cache[i],
                block_tables, seq_lens, block_size=self.block_size)
            x = x + o.reshape(B, 1, c.n_q * c.head_dim) @ lay.wo.T

            h = rms_norm(x, lay.n2, c.eps)
            x = x + (F.silu(h @ lay.gate.T) * (h @ lay.up.T)) @ lay.down.T

        x = rms_norm(x, self.norm_f, c.eps)
        return x[:, -1, :] @ self.lm_head.T




    def chunk_forward(self, input_ids, positions, slot_mapping,
                      block_tables, seq_lens):
        """prefill chunk 前向：Q>=1，读路径同样按块表走。

        与 `decode_forward` 的唯一区别是 attention 换成分块版本：每个 query
        有自己的绝对位置，因果掩码在 kernel 里按 `key_pos <= query_pos` 判。
        `positions < 0` 表示 padding，kernel 会写 0、调用方忽略这些位置。
        """
        c = self.cfg
        B, Q = input_ids.shape
        x = self.embed[input_ids]
        cos = self.cos[positions.clamp(min=0)]
        sin = self.sin[positions.clamp(min=0)]

        write = slot_mapping.reshape(-1)
        valid_write = write >= 0
        write_safe = torch.where(valid_write, write, torch.zeros_like(write))

        for i, lay in enumerate(self.layers):
            h = rms_norm(x, lay.n1, c.eps)
            q = (h @ lay.wq.T).view(B, Q, c.n_q, c.head_dim)
            k = (h @ lay.wk.T).view(B, Q, c.n_kv, c.head_dim)
            v = (h @ lay.wv.T).view(B, Q, c.n_kv, c.head_dim)
            q = rms_norm(q, lay.qn, c.eps)
            k = rms_norm(k, lay.kn, c.eps)
            q = q * cos[:, :, None] + rotate_half(q) * sin[:, :, None]
            k = k * cos[:, :, None] + rotate_half(k) * sin[:, :, None]

            kf = k.reshape(-1, c.n_kv, c.head_dim)
            vf = v.reshape(-1, c.n_kv, c.head_dim)
            keep = valid_write.nonzero(as_tuple=True)[0]
            self.k_cache[i].index_copy_(0, write_safe[keep], kf[keep])
            self.v_cache[i].index_copy_(0, write_safe[keep], vf[keep])

            o = paged_chunk_attention(q, self.k_cache[i], self.v_cache[i],
                                      block_tables, seq_lens, positions,
                                      block_size=self.block_size)
            x = x + o.reshape(B, Q, c.n_q * c.head_dim) @ lay.wo.T

            h = rms_norm(x, lay.n2, c.eps)
            x = x + (F.silu(h @ lay.gate.T) * (h @ lay.up.T)) @ lay.down.T

        x = rms_norm(x, self.norm_f, c.eps)
        return x @ self.lm_head.T


class PagedEngine(Engine):
    """**整条前向**都走分页读路径：prefill chunk 与 decode 用同一张块表。

    与基类的唯一差别是 attention 的读路径：基类把 KV 按块表 gather 成连续张量
    再交给 SDPA，这里直接按块表在 kernel 里寻址。调度、采样、块池与取消逻辑
    完全复用基类，所以两条路径的输出应当逐 token 相同。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.paged_steps = 0
        self.paged_tokens = 0

    def _run(self, groups):
        dev = self.model.device
        B = len(groups)
        Q = max(n for _, _, n in groups)
        max_blocks = max(len(r.block_table) for r, _, _ in groups)
        bt = torch.full((B, max_blocks), -1, dtype=torch.int32, device=dev)
        seq_lens = torch.zeros(B, dtype=torch.int32, device=dev)
        input_ids = torch.zeros((B, Q), dtype=torch.long)
        # padding 位置用 -1：chunk kernel 见到 qpos<0 会写 0，不参与 softmax
        positions = torch.full((B, Q), -1, dtype=torch.long)
        slot_map = torch.full((B, Q), -1, dtype=torch.long)
        for b, (r, start, n) in enumerate(groups):
            bt[b, :len(r.block_table)] = torch.tensor(
                r.block_table, dtype=torch.int32, device=dev)
            seq_lens[b] = start + n
            input_ids[b, :n] = torch.tensor(r.all_ids[start:start + n])
            positions[b, :n] = torch.arange(start, start + n)
            slot_map[b, :n] = torch.tensor(self._slots(r, start, n))
        self.paged_steps += 1
        self.paged_tokens += sum(n for _, _, n in groups)
        logits = self.model.chunk_forward(
            input_ids.to(dev), positions.to(dev), slot_map.to(dev),
            bt, seq_lens)
        last = torch.tensor([n - 1 for _, _, n in groups], device=dev)
        return logits[torch.arange(B, device=dev), last]
