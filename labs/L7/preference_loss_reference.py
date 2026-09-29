#!/usr/bin/env python3
"""偏好优化的目标与梯度：DPO、APO-zero 与它们的边界。

四段：
  A 序列 logprob  在固定的小词表上手算 chosen/rejected 的序列对数概率
  B DPO           从 Bradley–Terry 推到 DPO 的 loss 与解析梯度，与 autograd 对拍
  C APO-zero      SmolLM3 用的那个变体与 DPO 的差别
  D 反例          长度归一化、配对错位、reference 跟着更新、模板不一致

全部 FP64，无外部依赖。

Usage:
    python labs/L7/preference_loss_reference.py > "$RUN_DIR/preference.txt"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VOCAB, SEQ = 8, 5
BETA = 0.05                      # SmolLM3 的 APO 配方用的 beta
torch.set_default_dtype(torch.float64)


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def seq_logprob(logits: torch.Tensor, tokens: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """Σ_t log p(y_t | y_<t)，只统计 mask 为真的位置（即回答部分）。"""
    logp = F.log_softmax(logits, dim=-1)
    picked = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)


def make_case(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    policy = torch.randn(2, SEQ, VOCAB, generator=gen, requires_grad=True)
    ref = torch.randn(2, SEQ, VOCAB, generator=gen)
    tokens = torch.randint(0, VOCAB, (2, SEQ), generator=gen)
    mask = torch.zeros(2, SEQ)
    mask[0, 2:] = 1.0            # chosen 的回答从第 3 个位置开始
    mask[1, 2:4] = 1.0           # rejected 的回答更短
    return policy, ref, tokens, mask


# ------------------------------------------------------------------ A
def section_a():
    head("A 序列 logprob：只统计回答部分")
    policy, ref, tokens, mask = make_case()
    lp = seq_logprob(policy, tokens, mask)
    lp_ref = seq_logprob(ref, tokens, mask)
    print(f"  chosen 回答长度 {int(mask[0].sum())}，rejected 回答长度 {int(mask[1].sum())}")
    print(f"  policy  logπ(chosen)={lp[0]:.8f}  logπ(rejected)={lp[1]:.8f}")
    print(f"  ref     logπ(chosen)={lp_ref[0]:.8f}  logπ(rejected)={lp_ref[1]:.8f}")
    print(f"  log-ratio：chosen {lp[0] - lp_ref[0]:+.8f}，"
          f"rejected {lp[1] - lp_ref[1]:+.8f}")
    print("  两条序列的 prompt 部分必须完全相同且都不计入 logprob；")
    print("  否则比较的是两个不同问题上的概率，差值没有意义。")
    return policy, ref, tokens, mask


# ------------------------------------------------------------------ B
def dpo_loss(policy, ref, tokens, mask, beta=BETA):
    lp = seq_logprob(policy, tokens, mask)
    lp_ref = seq_logprob(ref, tokens, mask)
    delta = (lp[0] - lp_ref[0]) - (lp[1] - lp_ref[1])
    return -F.logsigmoid(beta * delta), delta


def section_b(policy, ref, tokens, mask):
    head("B DPO：Bradley–Terry 给出的目标与它的梯度")
    print("  Bradley–Terry：P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))")
    print("  DPO 把隐式奖励取成 r(y) = β·log[π(y)/π_ref(y)]，于是")
    print("    L = −log σ(β·Δ)，Δ = [logπ(y_w)−logπ_ref(y_w)] − [logπ(y_l)−logπ_ref(y_l)]")
    loss, delta = dpo_loss(policy, ref, tokens, mask)
    loss.backward()
    grad_auto = policy.grad.clone()
    policy.grad = None

    # 解析梯度：dL/dΔ = −β·σ(−βΔ)
    coef = -BETA * torch.sigmoid(-BETA * delta)
    policy2 = policy.detach().clone().requires_grad_(True)
    lp = seq_logprob(policy2, tokens, mask)
    (lp[0] - lp[1]).backward()
    grad_manual = coef.detach() * policy2.grad
    print(f"\n  Δ={delta:.8f}  loss={loss:.8f}  σ(βΔ)={torch.sigmoid(BETA * delta):.8f}")
    print(f"  解析系数 dL/dΔ = −β·σ(−βΔ) = {coef:.8f}")
    print(f"  与 autograd 的梯度最大差 {float((grad_auto - grad_manual).abs().max()):.3e}")
    print("\n  梯度的方向：把 chosen 的 logprob 推高、rejected 的推低，")
    print(f"  权重 σ(−βΔ)={float(torch.sigmoid(-BETA * delta)):.6f} 在模型已经分得很开时趋近 0，")
    print("  所以 DPO 会自动忽略已经学会的样本——这也是它对标注噪声敏感的原因。")

    print("\n  β 的作用：")
    for beta in (0.01, 0.05, 0.5):
        p = policy.detach().clone().requires_grad_(True)
        l, d = dpo_loss(p, ref, tokens, mask, beta)
        l.backward()
        print(f"    β={beta:<5} loss={float(l):.6f}  梯度范数={float(p.grad.norm()):.6f}")
    print("  β 越大越贴近 reference，越小越放任 policy 偏离——它不是学习率。")


# ------------------------------------------------------------------ C
def section_c(policy, ref, tokens, mask):
    head("C APO-zero：SmolLM3 的 DPO 配方用的是它")
    lp = seq_logprob(policy.detach(), tokens, mask)
    lp_ref = seq_logprob(ref, tokens, mask)
    d_w = lp[0] - lp_ref[0]
    d_l = lp[1] - lp_ref[1]
    dpo = -F.logsigmoid(BETA * (d_w - d_l))
    apo_zero = -(F.logsigmoid(BETA * d_w) + F.logsigmoid(-BETA * d_l))
    print("  DPO 只约束两者之差；APO-zero 把两项拆开：")
    print("    L_APO0 = −log σ(β·Δ_w) − log σ(−β·Δ_l)")
    print(f"    Δ_w={d_w:+.6f}  Δ_l={d_l:+.6f}")
    print(f"    DPO loss={float(dpo):.6f}   APO-zero loss={float(apo_zero):.6f}")
    print("  含义不同：DPO 允许两者一起下降只要差值拉大；")
    print("  APO-zero 明确要求 chosen 的概率相对 reference 上升、rejected 下降。")
    print("  SmolLM3 的 apo.yaml 写的是 loss_type: apo_zero、beta: 0.05；")
    print("  换 loss_type 等于换优化目标，不是换一个超参。")


# ------------------------------------------------------------------ D
def section_d():
    head("D 四个反例")
    policy, ref, tokens, mask = make_case()

    loss_plain, delta_plain = dpo_loss(policy, ref, tokens, mask)
    len_w, len_l = mask[0].sum(), mask[1].sum()
    lp = seq_logprob(policy, tokens, mask)
    lp_ref = seq_logprob(ref, tokens, mask)
    delta_norm = ((lp[0] - lp_ref[0]) / len_w) - ((lp[1] - lp_ref[1]) / len_l)
    print(f"  1) 长度归一化：chosen {int(len_w)} token、rejected {int(len_l)} token")
    print(f"     不归一化 Δ={float(delta_plain):+.6f}；按长度归一化 Δ={float(delta_norm):+.6f}")
    print("     两者符号可以相反。序列 logprob 随长度单调下降，")
    print("     长度不等的偏好对里，不归一化的 DPO 天然偏向短回答。")

    swapped_mask = mask.flip(0)
    loss_swap, delta_swap = dpo_loss(policy, ref, tokens, swapped_mask)
    print(f"\n  2) 配对错位（把 chosen 的 mask 用到 rejected 上）：")
    print(f"     Δ 从 {float(delta_plain):+.6f} 变成 {float(delta_swap):+.6f}，"
          f"loss 从 {float(loss_plain):.6f} 变成 {float(loss_swap):.6f}")
    print("     它不会报错，只是把优化方向反过来。")

    ref_follow = policy.detach().clone()
    lp = seq_logprob(policy.detach(), tokens, mask)
    lp_ref2 = seq_logprob(ref_follow, tokens, mask)
    delta_follow = (lp[0] - lp_ref2[0]) - (lp[1] - lp_ref2[1])
    print(f"\n  3) reference 跟着 policy 一起更新：Δ={float(delta_follow):.1e}，"
          f"loss={float(-F.logsigmoid(BETA * delta_follow)):.6f}=−log σ(0)")
    print("     两个 log-ratio 恒为零，梯度也恒为零：reference 必须冻结。")

    shifted = tokens.clone()
    shifted[1, 2] = (shifted[1, 2] + 1) % VOCAB      # 模板差一个 token
    _, delta_tpl = dpo_loss(policy, ref, shifted, mask)
    print(f"\n  4) 两侧模板不一致（rejected 多一个特殊 token）：")
    print(f"     Δ 从 {float(delta_plain):+.6f} 变成 {float(delta_tpl):+.6f}")
    print("     偏好数据必须用与训练完全相同的 chat template 渲染，")
    print("     否则差值里混进了模板差异，而不是回答质量差异。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU | FP64 | beta={BETA}")
    policy, ref, tokens, mask = section_a()
    section_b(policy, ref, tokens, mask)
    section_c(policy, ref, tokens, mask)
    section_d()
