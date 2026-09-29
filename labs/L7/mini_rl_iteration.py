#!/usr/bin/env python3
"""
labs/L7/mini_rl_iteration.py
端到端强化学习迭代与权重同步循环模拟 (End-to-End RL Iteration & Weight Synchronization)

模拟现代大语言模型强化学习运行时 (veRL / slime / OpenRLHF) 的多角色协作环路：
1. 角色职责与状态流向：
   - Rollout Worker: 依据行为策略 pi_old 采样生成轨迹 (Action Sequences)
   - Reward Evaluator: 计算可验证奖励 (Rule/Format Verification)
   - Advantage Estimator: 计算组内相对优势 (GRPO: A_i = (R_i - mean(R)) / (std(R) + eps))
   - Learner Worker: 基于 PPO / GRPO 目标执行反向求导并更新策略权重 pi_new
   - Weight Synchronizer: 将新权重广播同步回 Rollout Worker (参数发布)
2. 单进程 bandit 式三轮循环，权重通过 load_state_dict 同步。
   不包含真实文本多步轨迹、并行 worker 或异步 lag 的验证。
"""

import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class TinyPolicyNet(nn.Module):
    """极简策略网络 (用于动作分布输出)"""

    def __init__(self, vocab_size: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        logits = self.fc(h)
        return logits


@dataclass
class TrajectorySample:
    prompt_id: str
    tokens: List[int]
    old_logprobs: List[float]
    reward: float
    advantage: float = 0.0
    policy_version: int = 0


class MiniRLRuntime:
    """模拟轻量级 RL 运行时"""

    def __init__(self, vocab_size: int = 8, hidden_dim: int = 16, clip_eps: float = 0.2):
        self.vocab_size = vocab_size
        self.clip_eps = clip_eps
        self.policy_version = 0

        # Learner 持有的可训练策略
        self.learner_policy = TinyPolicyNet(vocab_size, hidden_dim)
        self.optimizer = optim.AdamW(self.learner_policy.parameters(), lr=0.01)

        # Rollout Worker 持有的行为策略副本
        self.rollout_policy = TinyPolicyNet(vocab_size, hidden_dim)
        self.sync_weights()  # 初始同步

    def sync_weights(self):
        """权重发布与同步 (Learner -> Rollout Actor)"""
        self.rollout_policy.load_state_dict(self.learner_policy.state_dict())
        self.policy_version += 1

    def generate_rollouts(self, prompt_token: int, group_size: int = 4) -> List[TrajectorySample]:
        """Rollout Worker 生成一个组 (Group) 的轨迹"""
        trajectories = []
        x = torch.tensor([prompt_token], dtype=torch.long)
        with torch.no_grad():
            logits = self.rollout_policy(x)[0]
            probs = torch.softmax(logits, dim=-1)

        # 采样生成 group_size 条回答
        for i in range(group_size):
            # 随机采样一个动作 token
            action = torch.multinomial(probs, num_samples=1).item()
            logp = math.log(probs[action].item() + 1e-12)

            # 模拟规则奖励：假设动作为偶数时正确 (Reward=1.0)，奇数时错误 (Reward=0.0)
            reward = 1.0 if (action % 2 == 0) else 0.0

            trajectories.append(
                TrajectorySample(
                    prompt_id="prompt_001",
                    tokens=[prompt_token, action],
                    old_logprobs=[logp],
                    reward=reward,
                    policy_version=self.policy_version,
                )
            )
        return trajectories

    def compute_grpo_advantages(self, trajectories: List[TrajectorySample]):
        """GRPO 组内相对基线归一化"""
        rewards = [t.reward for t in trajectories]
        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(variance) + 1e-8

        for t in trajectories:
            t.advantage = (t.reward - mean_r) / std_r

    def train_step(self, trajectories: List[TrajectorySample]) -> Dict[str, Any]:
        """Learner 执行一步策略更新"""
        self.optimizer.zero_grad()
        total_loss = 0.0
        ratios = []

        for t in trajectories:
            x = torch.tensor([t.tokens[0]], dtype=torch.long)
            action = t.tokens[1]
            old_logp = t.old_logprobs[0]

            logits = self.learner_policy(x)[0]
            new_logp = torch.log_softmax(logits, dim=-1)[action]

            # 重要性采样比率 r_t = exp(new_logp - old_logp)
            ratio = torch.exp(new_logp - old_logp)
            ratios.append(ratio.item())

            # PPO / GRPO 裁切目标
            surr1 = ratio * t.advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * t.advantage
            loss = -torch.min(surr1, surr2)
            total_loss += loss

        total_loss = total_loss / len(trajectories)
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_ratio": sum(ratios) / len(ratios),
            "max_ratio": max(ratios),
            "min_ratio": min(ratios),
        }


def main():
    output_dir = Path("/Users/nyxri/Documents/llm-infra/results/local/7.6")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 开始端到端 RL 运行时与策略同步模拟 ===")
    runtime = MiniRLRuntime(vocab_size=8, hidden_dim=16, clip_eps=0.2)

    iteration_records = []

    # 运行 3 轮完整 RL 迭代 (Rollout -> Reward -> Advantage -> Update -> Weight Sync)
    for epoch in range(1, 4):
        print(f"\n[RL 迭代轮次 {epoch}]")
        # 1. Rollout
        trajectories = runtime.generate_rollouts(prompt_token=1, group_size=4)
        # 2. Advantage
        runtime.compute_grpo_advantages(trajectories)
        # 3. Learner Update
        step_metrics = runtime.train_step(trajectories)
        # 4. Weight Sync
        runtime.sync_weights()

        rewards = [t.reward for t in trajectories]
        advs = [round(t.advantage, 3) for t in trajectories]
        actions = [t.tokens[1] for t in trajectories]

        print(f"  -> 生成动作: {actions} | 奖励: {rewards} | 组内优势: {advs}")
        print(f"  -> 策略损失: {step_metrics['loss']:.6f} | 平均采样比率: {step_metrics['mean_ratio']:.4f}")
        print(f"  -> 权重同步完成，策略版本升至: v{runtime.policy_version}")

        iteration_records.append({
            "epoch": epoch,
            "actions": actions,
            "rewards": rewards,
            "advantages": advs,
            "step_metrics": step_metrics,
            "policy_version": runtime.policy_version,
        })

    report_path = output_dir / "mini_rl_iteration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"iterations": iteration_records}, f, indent=2, ensure_ascii=False)
    print(f"\n[工件写入] RL 迭代测试报告已持久化至: {report_path}")


if __name__ == "__main__":
    main()
