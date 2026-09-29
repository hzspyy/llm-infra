#!/usr/bin/env python3
"""activation checkpoint：保存值、重算次数与 RNG。

同一份 FP64 小网络分别用普通前向和 checkpoint 执行，统计：
前向调用次数、saved_tensors pack/unpack 次数、梯度差、dropout 下的 RNG 行为。

Usage:
    python labs/L7/checkpoint_autograd.py > "$RUN_DIR/checkpoint.txt"
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

torch.manual_seed(0)

FORWARD_CALLS = {"n": 0}


class Block(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, dtype=torch.float64)
        self.fc2 = nn.Linear(dim, dim, dtype=torch.float64)
        self.dropout = dropout

    def forward(self, x):
        FORWARD_CALLS["n"] += 1
        h = torch.tanh(self.fc1(x))
        if self.dropout:
            h = nn.functional.dropout(h, p=self.dropout, training=True)
        return self.fc2(h)


def make(dim: int = 8, dropout: float = 0.0, seed: int = 0):
    torch.manual_seed(seed)
    return Block(dim, dropout)


def run_plain(dropout: float = 0.0):
    model = make(dropout=dropout)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(123)
    packed, unpacked = [], []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (packed.append(tuple(t.shape)), t)[1],
        lambda t: (unpacked.append(tuple(t.shape)), t)[1],
    ):
        FORWARD_CALLS["n"] = 0
        y = model(x)
    after_forward = FORWARD_CALLS["n"]
    loss = y.sum()
    loss.backward()
    return {"graph": "普通前向", "forward_calls_after_forward": after_forward,
            "forward_calls_total": FORWARD_CALLS["n"],
            "pack_events": len(packed), "unpack_events": len(unpacked),
            "x_grad": x.grad.clone(), "loss": loss.detach().item()}


def run_checkpoint(dropout: float = 0.0, use_reentrant: bool = False):
    model = make(dropout=dropout)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(123)
    packed, unpacked = [], []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (packed.append(tuple(t.shape)), t)[1],
        lambda t: (unpacked.append(tuple(t.shape)), t)[1],
    ):
        FORWARD_CALLS["n"] = 0
        y = checkpoint(model, x, use_reentrant=use_reentrant)
    after_forward = FORWARD_CALLS["n"]
    loss = y.sum()
    loss.backward()
    return {"graph": f"checkpoint(use_reentrant={use_reentrant})",
            "forward_calls_after_forward": after_forward,
            "forward_calls_total": FORWARD_CALLS["n"],
            "pack_events": len(packed), "unpack_events": len(unpacked),
            "x_grad": x.grad.clone(), "loss": loss.detach().item()}


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    print("activation checkpoint：保存值、重算与 RNG（FP64 CPU，torch", torch.__version__, "）")

    section("[A] 普通前向 vs checkpoint：保存值与重算次数")
    plain = run_plain()
    ckpt = run_checkpoint(use_reentrant=False)
    for row in (plain, ckpt):
        print(f"  {row['graph']:<34} 前向后调用={row['forward_calls_after_forward']} "
              f"总调用={row['forward_calls_total']} pack={row['pack_events']} unpack={row['unpack_events']}")
    diff = (plain["x_grad"] - ckpt["x_grad"]).abs().max().item()
    print(f"  两种路径 loss 差={abs(plain['loss'] - ckpt['loss']):.3e}，x.grad 最大绝对差={diff:.3e}")

    section("[B] dropout 下的 RNG：checkpoint 重算是否得到同一 mask")
    p2 = run_plain(dropout=0.5)
    c2 = run_checkpoint(dropout=0.5, use_reentrant=False)
    diff2 = (p2["x_grad"] - c2["x_grad"]).abs().max().item()
    print(f"  普通前向 x.grad 前两个元素={p2['x_grad'][0, :2].tolist()}")
    print(f"  checkpoint x.grad 前两个元素={c2['x_grad'][0, :2].tolist()}")
    print(f"  dropout=0.5 时 x.grad 最大绝对差={diff2:.3e}（非重入 checkpoint 恢复 RNG，重算得到同一 mask）")

    section("[C] early stop：反向未到达 checkpoint 时不重算")
    model = make()
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    FORWARD_CALLS["n"] = 0
    y = checkpoint(model, x, use_reentrant=False)
    after_forward = FORWARD_CALLS["n"]
    loss = (y.detach() * 2.0).sum() + (x * 3.0).sum()
    loss.backward()
    print(f"  detach 截断后：前向后调用={after_forward}，backward 后总调用={FORWARD_CALLS['n']}"
          f"（未增加说明没有任何重算）")
    print(f"  x.grad={x.grad[0, :2].tolist()}（解析 3.0）")

    section("[D] 非重入 checkpoint 的 Rule 5 早停开关（实测为负结果）")
    class Step(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            STEPS["n"] += 1
            ctx.save_for_backward(t)
            return t * 2.0

        @staticmethod
        def backward(ctx, g):
            (t,) = ctx.saved_tensors
            return 2.0 * g

    STEPS = {"n": 0}

    def fn(t):
        a = Step.apply(t)
        b = Step.apply(a)
        return a, b

    for early in (True, False):
        set_checkpoint_early_stop(early)
        t = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        STEPS["n"] = 0
        a, b = checkpoint(fn, t, use_reentrant=False, early_stop=early)
        forward_steps = STEPS["n"]
        STEPS["n"] = 0
        g, = torch.autograd.grad(a.sum(), t)
        print(f"  early_stop={early}: 前向 step={forward_steps} 重算 step={STEPS['n']} grad={g.tolist()}")
    set_checkpoint_early_stop(True)
    print("  两种设置的重算 step 数相同，本构造未触发 Rule 5 的提前退出；")
    print("  该构造只说明非重入路径能正确重算，不构成 early-stop 收益的证据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
