#!/usr/bin/env python3
"""冻结、no_grad、detach、teacher 分支与 LoRA 的梯度路径对照。

同一 encoder→projector→loss 结构，逐个改变控制面，记录：
参数 requires_grad、backward 后的 .grad、各区域 saved tensors 数量、optimizer 参数组。

Usage:
    python labs/L7/frozen_branches.py > "$RUN_DIR/frozen-branches.txt"
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

torch.manual_seed(0)
DTYPE = torch.float64


def count_pack(fn):
    """执行 fn，返回 (结果, 该区域内 saved tensors 数量)。"""
    box = {"n": 0}

    def pack(t):
        box["n"] += 1
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, box["n"]


class Net(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8, out: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, dtype=DTYPE)
        self.encoder = nn.Sequential(nn.Linear(dim, dim, dtype=DTYPE), nn.ReLU())
        self.projector = nn.Linear(dim, out, dtype=DTYPE)

    def forward(self, ids):
        e, n_embed = count_pack(lambda: self.embed(ids))
        h, n_enc = count_pack(lambda: self.encoder(e))
        p, n_proj = count_pack(lambda: self.projector(h))
        self.last_pack = {"embed": n_embed, "encoder": n_enc, "projector": n_proj}
        return p


def grad_table(model: nn.Module) -> list[tuple[str, bool, str]]:
    rows = []
    for name, param in model.named_parameters():
        g = param.grad
        rows.append((name, param.requires_grad,
                     "None" if g is None else f"shape={tuple(g.shape)} norm={g.norm().item():.4f}"))
    return rows


def show(title: str, model: Net, loss: torch.Tensor) -> None:
    print(f"  {title}: loss={loss.item():.6f}，saved tensors {model.last_pack}")
    for name, req, g in grad_table(model):
        print(f"    {name:<20} requires_grad={str(req):<5} grad={g}")


def make(seed: int = 0):
    torch.manual_seed(seed)
    return Net()


def main() -> int:
    ids = torch.tensor([[1, 2, 3, 4]])
    target = torch.randn(1, 4, dtype=DTYPE)

    print("=" * 72)
    print("[V1] 全部可训练")
    print("=" * 72)
    m1 = make()
    out1 = m1(ids)
    loss1 = ((out1 - target) ** 2).mean()
    loss1.backward()
    show("全参", m1, loss1)
    opt1 = torch.optim.SGD([
        {"params": m1.embed.parameters(), "lr": 1e-2, "name": "embed"},
        {"params": m1.encoder.parameters(), "lr": 1e-3, "name": "encoder"},
        {"params": m1.projector.parameters(), "lr": 1e-2, "name": "projector"},
    ])
    print("  optimizer 参数组:", [(g["name"], sum(p.numel() for p in g["params"])) for g in opt1.param_groups])

    print()
    print("=" * 72)
    print("[V2] 冻结 encoder 参数（requires_grad=False），embedding 仍可训练")
    print("=" * 72)
    m2 = make(seed=1)
    for p in m2.encoder.parameters():
        p.requires_grad_(False)
    out2 = m2(ids)
    loss2 = ((out2 - target) ** 2).mean()
    loss2.backward()
    show("冻结 encoder 权重", m2, loss2)
    print("  结论：encoder 权重无梯度，但 embedding 仍拿到输入梯度，encoder 区域仍有 saved tensors。")

    print()
    print("=" * 72)
    print("[V3] encoder 前向放在 no_grad 内")
    print("=" * 72)
    m3 = make(seed=1)
    e3, n_embed3 = count_pack(lambda: m3.embed(ids))
    with torch.no_grad():
        h3, n_enc3 = count_pack(lambda: m3.encoder(e3))
    p3, n_proj3 = count_pack(lambda: m3.projector(h3))
    m3.last_pack = {"embed": n_embed3, "encoder": n_enc3, "projector": n_proj3}
    loss3 = ((p3 - target) ** 2).mean()
    loss3.backward()
    show("no_grad encoder", m3, loss3)
    print("  结论：encoder 输出不需要梯度，projector 的输入梯度与 embedding 梯度都不存在。")

    print()
    print("=" * 72)
    print("[V4] projector 输出 detach 后进入冻结辅助头，主损失仍经过 projector")
    print("=" * 72)
    m4 = make(seed=2)
    head4 = nn.Linear(4, 4, dtype=DTYPE)
    for p in head4.parameters():
        p.requires_grad_(False)
    out4 = m4(ids)
    aux4 = head4(out4.detach())
    loss4_main = ((out4 - target) ** 2).mean()
    loss4_aux = ((aux4 - target) ** 2).mean()
    loss4 = loss4_main + loss4_aux
    loss4.backward()
    show("主损失 + detach 辅助损失", m4, loss4)
    m4b = make(seed=2)
    out4b = m4b(ids)
    loss4b = ((out4b - target) ** 2).mean()
    loss4b.backward()
    same = all(
        torch.allclose(p.grad, q.grad)
        for p, q in zip(m4.parameters(), m4b.parameters())
    )
    print(f"  去掉辅助损失后梯度完全相同={same}（detach 的辅助项不产生任何梯度贡献）")

    print()
    print("=" * 72)
    print("[V5] 冻结 teacher 监督分支（teacher 前向在 no_grad 内）")
    print("=" * 72)
    m5 = make(seed=3)
    teacher = make(seed=4)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student_out = m5(ids)
    teacher_out, teacher_pack = count_pack(
        lambda: torch.no_grad()(lambda: teacher(ids))()
    )
    loss5 = ((student_out - teacher_out.detach()) ** 2).mean()
    loss5.backward()
    print(f"  teacher 区域 saved tensors={teacher_pack}，teacher 输出 requires_grad={teacher_out.requires_grad}")
    show("student 学 teacher", m5, loss5)
    teacher_grads = [p.grad is None for p in teacher.parameters()]
    print(f"  teacher 参数全部无梯度: {all(teacher_grads)}")

    print()
    print("=" * 72)
    print("[V6] LoRA：base 冻结，A/B 可训练，B 零初始化")
    print("=" * 72)
    torch.manual_seed(5)
    base = nn.Linear(8, 8, dtype=DTYPE)
    for p in base.parameters():
        p.requires_grad_(False)
    r = 2
    A = nn.Parameter(torch.randn(r, 8, dtype=DTYPE) * 0.01)
    B = nn.Parameter(torch.zeros(8, r, dtype=DTYPE))
    x6 = torch.randn(3, 8, dtype=DTYPE)
    y_base = base(x6)
    y_lora = y_base + (x6 @ A.t() @ B.t())
    print(f"  初始化时增量最大绝对差={(y_lora - y_base).abs().max().item():.3e}（B=0）")
    opt6 = torch.optim.SGD([{"params": [A], "lr": 1e-2}, {"params": [B], "lr": 1e-2}])

    def lora_forward():
        return base(x6) + (x6 @ A.t() @ B.t())

    base_before = base.weight.detach().clone()
    opt6.zero_grad()
    loss6 = (lora_forward() ** 2).mean()
    loss6.backward()
    print(f"  base.weight.grad is None: {base.weight.grad is None}，"
          f"step1 A.grad 范数={A.grad.norm().item():.3e}，B.grad 范数={B.grad.norm().item():.3e}")
    A_before = A.detach().clone()
    opt6.step()
    print(f"  第 1 步后 base 权重是否改变={not torch.equal(base_before, base.weight)}，"
          f"A 是否改变={not torch.equal(A_before, A)}（B=0 时 A 的梯度为 0）")
    opt6.zero_grad()
    loss6b = (lora_forward() ** 2).mean()
    loss6b.backward()
    print(f"  step2 A.grad 范数={A.grad.norm().item():.3e}，B.grad 范数={B.grad.norm().item():.3e}"
          "（B 非零后 A 开始收到梯度）")
    A_before2 = A.detach().clone()
    opt6.step()
    print(f"  第 2 步后 A 是否改变={not torch.equal(A_before2, A)}")
    trainable = sum(p.numel() for p in [A, B])
    frozen = sum(p.numel() for p in base.parameters())
    print(f"  可训练参数={trainable}，冻结参数={frozen}，比例={trainable / (trainable + frozen):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
