#!/usr/bin/env python3
"""
labs/L7/rl_objective_reference.py
PPO 与 GRPO 目标函数推导、解析梯度对拍与失效模式 (PPO & GRPO Objective Parity)

严谨推导并数值验证大模型强化学习两大核心目标：
1. PPO (Proximal Policy Optimization) 裁剪代理目标与梯度
2. GRPO (Group Relative Policy Optimization) 组内基线目标：
   - 显式剔除独立的 Critic 价值网络
   - 组内奖励归一化：A_i = (R_i - mean(R)) / (std(R) + eps)
   - Token 级裁剪目标与 KL 惩罚项
3. 解析梯度 (Analytical Gradient) 与 PyTorch Autograd 双向对拍 (FP64, 误差 < 1e-15)
4. 注入并剖析三类典型工程失效模式：
   - 模式 1 (Zero-Variance Group): 组内回答得分全部相同 (std=0) 触发数值除零异常
   - 模式 2 (Prompt Masking Leak): 错误将 Advantage 应用到 Prompt Token，破坏输入语义分布
   - 模式 3 (Stale Ratio Divergence): 异步策略滞后未裁剪，比率大于裁剪上界时仍可能产生非零代理梯度
"""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_grpo_objective_analytical(
    logpi_new: torch.Tensor,  # shape [G]
    logpi_old: torch.Tensor,  # shape [G]
    rewards: torch.Tensor,    # shape [G]
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    logpi_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算标量 GRPO 损失以及关于 logpi_new 的解析梯度 (FP64)
    """
    G = rewards.shape[0]
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r

    ratios = torch.exp(logpi_new - logpi_old)
    ratios_clamped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)

    surr1 = ratios * advantages
    surr2 = ratios_clamped * advantages

    # PPO/GRPO 最小化代理目标: max E[min(surr1, surr2)] => min E[-min(surr1, surr2)]
    # 取 element-wise 最小值
    is_surr1_smaller = (surr1 < surr2)
    selected_surr = torch.where(is_surr1_smaller, surr1, surr2)

    # KL 惩罚项: D_KL(pi || ref) approx (exp(logref - lognew) - (logref - lognew) - 1) 或简单的反向 KL
    if logpi_ref is not None:
        kl_pen = torch.exp(logpi_ref - logpi_new) - (logpi_ref - logpi_new) - 1.0
    else:
        kl_pen = torch.zeros_like(logpi_new)

    loss_per_item = -selected_surr + kl_beta * kl_pen
    total_loss = loss_per_item.mean()

    # 解析梯度推导 d Loss / d logpi_new:
    # d ratio / d logpi_new = ratio
    # d surr1 / d logpi_new = ratio * adv
    # d surr2 / d logpi_new = 0 (当处于 clamp 边界之外时)
    # 因此若 surr1 < surr2 或处于 clamp 内部未饱和，导数为 ratio * adv；若饱和且被 clip，导数为 0
    in_unclipped_region = (ratios >= 1.0 - clip_eps) & (ratios <= 1.0 + clip_eps)
    active_grad_mask = is_surr1_smaller | in_unclipped_region

    grad_ratio = torch.where(active_grad_mask, ratios * advantages, torch.zeros_like(ratios))
    
    # KL 导数: d/d lognew [exp(ref - new) - (ref - new) - 1] = -exp(ref - new) + 1
    if logpi_ref is not None:
        grad_kl = kl_beta * (1.0 - torch.exp(logpi_ref - logpi_new))
    else:
        grad_kl = torch.zeros_like(logpi_new)

    analytical_grad = (-grad_ratio + grad_kl) / G

    return total_loss, analytical_grad, advantages


def verify_grpo_gradient_parity() -> Dict[str, Any]:
    """验证解析梯度与 Autograd 的绝对误差对拍 (FP64)"""
    # 4 条合成轨迹 (Group Size = 4)
    logpi_old = torch.tensor([-2.1, -1.8, -3.2, -0.9], dtype=torch.float64, requires_grad=False)
    logpi_new = torch.tensor([-2.0, -1.9, -3.1, -1.0], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=False)
    logpi_ref = torch.tensor([-2.15, -1.85, -3.15, -0.95], dtype=torch.float64, requires_grad=False)

    loss, ana_grad, adv = compute_grpo_objective_analytical(
        logpi_new=logpi_new,
        logpi_old=logpi_old,
        rewards=rewards,
        clip_eps=0.2,
        kl_beta=0.04,
        logpi_ref=logpi_ref,
    )

    loss.backward()
    auto_grad = logpi_new.grad

    max_diff = (ana_grad - auto_grad).abs().max().item()
    torch.testing.assert_close(ana_grad, auto_grad, rtol=0, atol=1e-15)

    return {
        "loss_value": loss.item(),
        "advantages": [round(a, 4) for a in adv.tolist()],
        "analytical_grad": [round(g, 8) for g in ana_grad.tolist()],
        "autograd_grad": [round(g, 8) for g in auto_grad.tolist()],
        "max_absolute_diff": max_diff,
        "is_exact_parity": (max_diff < 1e-15),
    }


def verify_rl_failure_modes():
    cases = []
    for advantage in (-1., 1.):
        for value in (.1, 1., 10.):
            ratio = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            grad = torch.autograd.grad(loss, ratio)[0]
            cases.append({'advantage':advantage,'ratio':value,'loss':loss.item(),
                          'gradient_wrt_ratio':grad.item()})
    assert cases[2]['gradient_wrt_ratio'] == 1 and cases[2]['loss'] == 10
    rewards = torch.ones(4,dtype=torch.float64)
    stable = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-8)
    assert torch.equal(stable,torch.zeros_like(stable))
    logp = torch.zeros(3,dtype=torch.float64,requires_grad=True)
    mask = torch.tensor([0.,1.,1.],dtype=torch.float64)
    good = -(logp.exp()*mask).sum()/mask.sum()
    bad = -logp.exp().mean()
    good_grad = torch.autograd.grad(good,logp,retain_graph=True)[0]
    bad_grad = torch.autograd.grad(bad,logp)[0]
    return {'ppo_cases':cases,'clip_is_not_ratio_constraint':True,
            'zero_variance_advantages':stable.tolist(),
            'prompt_mask':{'valid_mask':mask.tolist(),'correct_gradient':good_grad.tolist(),
                           'incorrect_gradient':bad_grad.tolist()},
            'scope':'surrogate gradients; no policy quality or collapse measured'}


def gae(rewards, values, terminated, gamma=.99, lam=.95):
    advantages=torch.zeros_like(rewards)
    carry=torch.zeros_like(rewards[...,0])
    for t in range(rewards.shape[-1]-1,-1,-1):
        continuation=1-terminated[...,t]
        delta=rewards[...,t]+gamma*values[...,t+1]*continuation-values[...,t]
        carry=delta+gamma*lam*continuation*carry
        advantages[...,t]=carry
    return advantages


def trajectory_cases():
    rows=[]
    for seed in (0,1):
        torch.manual_seed(seed)
        rewards=torch.randn(4,3,dtype=torch.float64)
        values=torch.randn(4,4,dtype=torch.float64)
        ended=torch.zeros_like(rewards)
        ended[:2,-1]=1 # true terminations; other trajectories bootstrap a truncated tail
        advantages=gae(rewards,values,ended)
        reference=torch.zeros_like(advantages)
        for i in range(4):
            for t in range(3):
                factor=1.
                for k in range(t,3):
                    delta=rewards[i,k]+.99*values[i,k+1]*(1-ended[i,k])-values[i,k]
                    reference[i,t]+=factor*delta
                    factor*=.99*.95*(1-ended[i,k])
        torch.testing.assert_close(advantages,reference,atol=1e-12,rtol=0)
        rows.append({'seed':seed,'rewards':rewards.tolist(),'values':values.tolist(),
                     'terminated':ended.tolist(),'gae':advantages.tolist(),
                     'max_abs_error':float((advantages-reference).abs().max())})
    return rows


def main():
    from _evidence import new_output,write_result
    out=new_output('PPO/GRPO surrogate gradients, masks and two groups of trajectories')
    result={'grpo_gradient_parity_fp64':verify_grpo_gradient_parity(),
            'rl_failure_modes':verify_rl_failure_modes(),'trajectory_groups':trajectory_cases()}
    write_result(out,'rl_objective_report.json',result,
                 {'device':'CPU','dtype':'float64','trajectory_groups':2,'trajectories_per_group':4,
                  'transitions_per_trajectory':3,'gamma':.99,'lambda':.95,'optimizer_updates':0,
                  'quality_measured':False},[__file__])


if __name__=='__main__':
    main()
