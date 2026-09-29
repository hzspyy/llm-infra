#!/usr/bin/env python3
"""草稿模型训练目标的小张量参照（7.7-E）。

按 SpecForge 固定 commit 3d64e7a6 的实现语义，用 FP64 CPU 复现三件事：

1. `dflash_family_model.create_dflash_block_mask` 的块 mask 语义（context 严格小于
   anchor、draft 槽只在同一块内、滑动窗口按块内偏移截断、无效块整行屏蔽）；
2. EAGLE3 的 `LogSoftmaxLoss`：按教师分布加权的 log-softmax，分母是 **B×T 行数**，
   以及 acceptance rate 与三种 LK 变体（alpha/tv/lambda）；
3. DFlash 的 per-position CE：按 `exp(-(pos-1)_+/gamma)` 衰减、分母是 **权重之和**，
   与上一条的分母口径不同；外加 `compact_teacher` 的分块 logsumexp/argmax 等价性。

Usage:
    python labs/L7/draft_objectives.py --outdir "$RUN_DIR/draft-objectives"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# [A] DFlash 块 mask
# ---------------------------------------------------------------------------
def dflash_block_mask(anchor_positions: torch.Tensor, block_keep_mask: torch.Tensor,
                      context_len: int, block_size: int, sliding_window: int | None = None):
    """返回稠密布尔 mask，形状 (B, N*block_size, context_len + N*block_size)。"""
    batch, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = context_len + num_blocks * block_size
    mask = torch.zeros((batch, q_len, kv_len), dtype=torch.bool)
    for b in range(batch):
        for block_id in range(num_blocks):
            anchor = int(anchor_positions[b, block_id])
            keep = bool(block_keep_mask[b, block_id])
            for offset in range(block_size):
                q_idx = block_id * block_size + offset
                if not keep or block_id >= num_blocks:
                    continue
                for kv_idx in range(kv_len):
                    if kv_idx < context_len:
                        visible = kv_idx < anchor
                        if sliding_window is not None:
                            visible = visible and kv_idx >= anchor + offset - (sliding_window - 1)
                    else:
                        kv_block = (kv_idx - context_len) // block_size
                        kv_offset = (kv_idx - context_len) % block_size
                        visible = kv_block == block_id
                        if sliding_window is not None:
                            visible = visible and kv_offset <= offset
                    mask[b, q_idx, kv_idx] = visible
    return mask


def check_block_mask() -> dict:
    context_len, block_size, num_blocks = 4, 2, 2
    anchors = torch.tensor([[2, 4]])
    keep_all = torch.tensor([[True, True]])
    dense = dflash_block_mask(anchors, keep_all, context_len, block_size)

    # 手算参照：Q 行顺序为 (block0, off0), (block0, off1), (block1, off0), (block1, off1)
    # KV 顺序为 context 0..3，然后 draft0..3
    expected = torch.tensor([
        [1, 1, 0, 0, 1, 1, 0, 0],   # block0/off0: context<2, 同块 draft 全部（无窗口）
        [1, 1, 0, 0, 1, 1, 0, 0],   # block0/off1
        [1, 1, 1, 1, 0, 0, 1, 1],   # block1/off0: context<4，同块 draft0..1
        [1, 1, 1, 1, 0, 0, 1, 1],   # block1/off1
    ], dtype=torch.bool)
    hand_ok = torch.equal(dense[0], expected)

    sliding = dflash_block_mask(anchors, keep_all, context_len, block_size, sliding_window=2)
    # 窗口 2：block0/off0 只看到 context[0..0]（anchor+0-(2-1)=1 → KV>=1 且 <2）；
    # block0/off1 看到 context[0..1] 与 draft offset<=1
    sliding_rows = sliding[0][:2].tolist()

    keep_second = torch.tensor([[True, False]])
    dropped = dflash_block_mask(anchors, keep_second, context_len, block_size)

    checks = {
        "hand_computed_case_matches": hand_ok,
        "invalid_block_row_is_empty": bool(dropped[0, 2:].sum() == 0),
        "no_context_at_or_after_anchor": bool(
            all(not dense[0, b * block_size + o, int(anchors[0, b]):context_len].any()
                for b in range(num_blocks) for o in range(block_size))
        ),
        "no_cross_block_draft_attention": bool(
            all(not dense[0, b * block_size + o, context_len + other * block_size:
                            context_len + (other + 1) * block_size].any()
                for b in range(num_blocks) for o in range(block_size)
                for other in range(num_blocks) if other != b)
        ),
        "sliding_window_limits_context_and_draft": bool(
            sliding[0, 0, 0].item() is False and sliding[0, 0, 4].item() is True
            and sliding[0, 1, 5].item() is True
        ),
    }
    return {"checks": checks, "hand_case_rows": expected[0].tolist(),
            "dense_rows": dense[0].tolist(), "sliding_first_two_rows": sliding_rows}


# ---------------------------------------------------------------------------
# [B] EAGLE3 LogSoftmaxLoss
# ---------------------------------------------------------------------------
def eagle3_logsoftmax_loss(block_logits: torch.Tensor, teacher_probs: torch.Tensor,
                           position_mask: torch.Tensor) -> torch.Tensor:
    """按 SpecForge `LogSoftmaxLoss`：逐行按教师分布加权，最后对 B×T 行取 mean。"""
    logsumexp = torch.logsumexp(block_logits, dim=-1, keepdim=True)
    per_token = (teacher_probs * (logsumexp - block_logits)).sum(dim=-1)
    masked = per_token * position_mask
    return masked.mean()


def eagle3_reference_autograd(block_logits, teacher_probs, position_mask):
    """用 PyTorch 自身算子写同一函数，用于梯度对拍。"""
    log_prob = F.log_softmax(block_logits, dim=-1)
    per_token = -(teacher_probs * log_prob).sum(dim=-1)
    return (per_token * position_mask).mean()


def check_eagle3_objective() -> dict:
    b, t, v = 2, 3, 4
    logits = torch.randn(b, t, v, dtype=DTYPE, generator=torch.Generator().manual_seed(7)).requires_grad_()
    teacher_logits = torch.randn(b, t, v, dtype=DTYPE, generator=torch.Generator().manual_seed(11))
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=DTYPE)

    value = eagle3_logsoftmax_loss(logits, teacher_probs, mask)
    value.backward()
    grad_mine = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_()
    ref = eagle3_reference_autograd(logits2, teacher_probs, mask)
    ref.backward()

    # 分母口径：屏蔽位置仍然进入 B×T 分母
    dense_mask = torch.ones_like(mask)
    value_dense = eagle3_logsoftmax_loss(logits.detach(), teacher_probs, dense_mask)
    value_sparse = eagle3_logsoftmax_loss(logits.detach(), teacher_probs, mask)
    unchanged_masked_positions = value_sparse / value_dense  # 3/6 位置有效时 loss 约为一半

    # forward KL 与教师熵的关系：L_row = KL(q||p) + H(q)
    log_prob = F.log_softmax(logits.detach(), dim=-1)
    kl = (teacher_probs * (torch.log(teacher_probs.clamp_min(1e-300)) - log_prob)).sum(dim=-1)
    entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-300))).sum(dim=-1)
    per_row = (torch.logsumexp(logits.detach(), dim=-1, keepdim=True) - logits.detach())
    loss_row = (teacher_probs * per_row).sum(dim=-1)
    kl_gap = float((loss_row - (kl + entropy)).abs().max())

    # 平移不变性：给所有 logits 加常数不改变 loss 或梯度
    shifted = logits.detach() + 5.0
    shifted_value = eagle3_logsoftmax_loss(shifted, teacher_probs, mask)

    return {
        "value": value.detach().item(),
        "grad_max_abs_diff_vs_autograd": float((grad_mine - logits2.grad).abs().max()),
        "half_masked_over_dense_loss_ratio": float(unchanged_masked_positions),
        "row_equals_KL_plus_teacher_entropy_max_gap": kl_gap,
        "shift_invariance_gap": abs(float(shifted_value) - float(value_sparse)),
        "numerator_weight_of_valid_rows": float(mask.sum() / mask.numel()),
    }


# ---------------------------------------------------------------------------
# [C] acceptance rate 与 LK 变体
# ---------------------------------------------------------------------------
def acceptance_rate(draft_logits, target_probs_on_draft, position_mask, eps=1e-8):
    draft_p = F.softmax(draft_logits, dim=-1)
    per_token = torch.minimum(target_probs_on_draft, draft_p).sum(dim=-1)
    m = position_mask
    acc = (per_token * m).sum() / m.sum().clamp_min(eps)
    log_per_token = torch.where(per_token > 0, torch.log(per_token.clamp_min(eps)), torch.zeros_like(per_token))
    log_acc = (log_per_token * m).sum() / m.sum().clamp_min(eps)
    return acc, log_acc


def check_lk_variants() -> dict:
    v = 5
    draft_logits = torch.randn(1, 3, v, dtype=DTYPE, generator=torch.Generator().manual_seed(3)).requires_grad_()
    teacher = F.softmax(torch.randn(1, 3, v, dtype=DTYPE, generator=torch.Generator().manual_seed(5)), dim=-1)
    mask = torch.tensor([[0.0, 1.0, 1.0]], dtype=DTYPE)

    acc, log_acc = acceptance_rate(draft_logits, teacher, mask)
    kl = eagle3_logsoftmax_loss(draft_logits, teacher, mask)
    kl_weight = 1.0 * torch.exp(-1.0 * acc.detach())
    losses = {
        "alpha": -log_acc,
        "tv": 1 - acc,
        "lambda": kl_weight * kl + (1 - kl_weight) * (1 - acc),
        "none": kl,
    }
    grads = {}
    for name, loss in losses.items():
        g = torch.autograd.grad(loss, draft_logits, retain_graph=True)[0]
        grads[name] = float(g.norm())
    return {"acceptance_rate": float(acc), "log_acceptance_rate": float(log_acc),
            "kl_weight_lambda": float(kl_weight), "grad_norms": grads}


# ---------------------------------------------------------------------------
# [D] DFlash per-position CE
# ---------------------------------------------------------------------------
def dflash_ce(lm_logits, target_ids, weight_mask, block_size, gamma=None):
    ce = F.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]),
                         target_ids.reshape(-1), reduction="none").reshape_as(target_ids)
    weights = weight_mask
    if gamma is not None and gamma > 0:
        positions = torch.arange(block_size, dtype=weights.dtype).view(1, 1, -1)
        decay = torch.exp(-(positions - 1).clamp(min=0) / gamma)
        weights = weights * decay
    return (ce * weights).sum() / weights.sum(), ce, weights


def check_dflash_objective() -> dict:
    b, n, block, v = 1, 2, 4, 6
    logits = torch.randn(b, n * block, v, dtype=DTYPE,
                         generator=torch.Generator().manual_seed(13)).requires_grad_()
    target = torch.randint(0, v, (b, n, block), generator=torch.Generator().manual_seed(17))
    weight_mask = torch.ones((b, n, block), dtype=DTYPE)
    gamma = 2.0

    loss_decay, ce, weights = dflash_ce(logits, target, weight_mask, block, gamma)
    loss_plain, _, weights_plain = dflash_ce(logits, target, weight_mask, block, None)
    grad = torch.autograd.grad(loss_decay, logits, retain_graph=True)[0]

    # 用同一批数字比较两种分母口径
    eagle_style = eagle3_logsoftmax_loss(
        logits.detach().reshape(b * n, block, v),
        F.one_hot(target.reshape(-1), v).to(DTYPE).reshape(b * n, block, v),
        torch.ones((b * n, block), dtype=DTYPE),
    )
    ratio = float(eagle_style) / float(loss_decay.detach())

    # 同上数字、只改分母：一半位置被屏蔽时两种口径的差别
    half_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=DTYPE)
    ce_full = F.cross_entropy(logits.detach().reshape(-1, v), target.reshape(-1),
                              reduction="none").reshape_as(target)
    weighted_sum = float((ce_full * half_mask).sum())
    dflash_den = float((ce_full * half_mask).sum() / half_mask.sum())
    eagle_den = float((ce_full * half_mask).sum() / half_mask.numel())
    denominator_ratio = dflash_den / eagle_den

    return {
        "denominator_ratio_when_half_masked_dflash_over_eagle3": denominator_ratio,
        "weighted_ce_sum": weighted_sum,
        "dflash_denominator_numerator": float(half_mask.sum()),
        "eagle3_denominator_numerator": float(half_mask.numel()),
        "loss_with_decay": float(loss_decay),
        "loss_without_decay": float(loss_plain),
        "grad_norm": float(grad.norm()),
        "mean_position_weight": float(weights.mean()),
        "first_block_weights": weights[0, 0].tolist(),
        "eagle3_denominator_style_same_numbers": float(eagle_style),
        "denominator_ratio_eagle_over_dflash": ratio,
    }


# ---------------------------------------------------------------------------
# [E] compact teacher 的分块 logsumexp/argmax
# ---------------------------------------------------------------------------
def tiled_logsumexp_argmax(hidden, weight, chunk_size):
    running_max = None
    running_sum = None
    argmax_index = None
    for start in range(0, weight.shape[0], chunk_size):
        chunk = F.linear(hidden, weight[start:start + chunk_size]).float()
        chunk_max = chunk.max(dim=-1, keepdim=True).values
        if running_max is None:
            running_max = chunk_max
            running_sum = torch.exp(chunk - chunk_max).sum(dim=-1, keepdim=True)
            argmax_index = chunk.argmax(dim=-1) + start
        else:
            new_max = torch.maximum(running_max, chunk_max)
            running_sum = (running_sum * torch.exp(running_max - new_max)
                           + torch.exp(chunk - new_max).sum(dim=-1, keepdim=True))
            chunk_argmax = chunk.argmax(dim=-1) + start
            better = chunk_max.squeeze(-1) > running_max.squeeze(-1)
            argmax_index = torch.where(better, chunk_argmax, argmax_index)
            running_max = new_max
    return torch.log(running_sum.squeeze(-1)) + running_max.squeeze(-1), argmax_index


def check_compact_teacher() -> dict:
    seq, v, hidden = 4, 64, 8
    h = torch.randn(seq, hidden, dtype=DTYPE, generator=torch.Generator().manual_seed(23))
    w = torch.randn(v, hidden, dtype=DTYPE, generator=torch.Generator().manual_seed(29))
    full_logits = F.linear(h, w).float()
    full_lse = torch.logsumexp(full_logits, dim=-1)
    full_argmax = full_logits.argmax(dim=-1)

    tiled_lse, tiled_argmax = tiled_logsumexp_argmax(h, w, chunk_size=16)
    draft_ids = torch.tensor([1, 5, 9, 13])
    t2d = torch.zeros(v, dtype=torch.bool)
    t2d[draft_ids] = True
    draft_logits = full_logits[:, t2d]
    renorm = F.softmax(draft_logits, dim=-1)
    full_vocab_probs = F.softmax(full_logits, dim=-1)[:, draft_ids]
    renorm_from_full = full_vocab_probs / full_vocab_probs.sum(dim=-1, keepdim=True)

    return {
        "lse_max_abs_diff": float((tiled_lse - full_lse).abs().max()),
        "argmax_equal": bool(torch.equal(tiled_argmax, full_argmax)),
        "renormalization_max_abs_diff": float((renorm - renorm_from_full).abs().max()),
        "full_logits_bytes_fp32": int(full_logits.numel() * 4),
        "streaming_peak_bytes_estimate": int(seq * 16 * 4 * 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="草稿模型训练目标的小张量参照")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    result = {
        "draft_block_mask": check_block_mask(),
        "eagle3_logsoftmax": check_eagle3_objective(),
        "acceptance_and_lk": check_lk_variants(),
        "dflash_ce": check_dflash_objective(),
        "compact_teacher": check_compact_teacher(),
        "source": "SpecForge 固定 commit 3d64e7a61f5fcc7f7d78ba6164c881f831943947 的 "
                  "specforge/algorithms/common/dflash_family_model.py、specforge/core/loss.py、"
                  "specforge/core/lk_loss.py、specforge/core/compact_teacher.py",
        "claim_scope": "FP64 CPU 语义与梯度对拍；不含真实草稿模型训练、接受长度或部署收益",
    }
    (args.outdir / "draft_objectives.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.outdir / "cases.json").write_text(json.dumps({
        "dtype": "float64", "seeds": [0, 3, 5, 7, 11, 13, 17, 23, 29],
        "checks": list(result["draft_block_mask"]["checks"]) +
                  ["eagle3_grad_vs_autograd", "denominator_ratio", "compact_teacher_equivalence"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())