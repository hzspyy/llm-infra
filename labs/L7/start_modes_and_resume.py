#!/usr/bin/env python3
"""一次更新的起点与接续：五种启动方式的状态账，以及严格恢复的逐项反例。

前半部分在同一个 TinyLM 上对比从零训练、加载权重、冻结、LoRA、teacher/student
五种启动，记录可训练参数、optimizer 参数组与第一次 step 后创建的状态。
后半部分做"更新—保存—恢复—再更新"的短序列，并逐项移除 optimizer、scheduler、
RNG、数据游标，记录第一个被破坏的不变量。

checkpoint 走 torch.save/torch.load 的真实序列化，但只落在内存缓冲区，不写文件。

Usage:
    python labs/L7/start_modes_and_resume.py > "$RUN_DIR/start-resume.txt"
"""
from __future__ import annotations

import copy
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from _tinylm import DTYPE, TinyLM

IGNORE = -100
LR, WD = 5e-3, 0.1

# 固定数据流：每条带样本 ID，游标决定下一步取哪一批。
STREAM = [("b0", [[5, 9, 3, 7, 2], [4, 11, 6, 13, 1]]),
          ("b1", [[8, 1, 12, 10, 6], [2, 14, 5, 3, 9]]),
          ("b2", [[7, 6, 11, 4, 15], [10, 2, 8, 13, 5]]),
          ("b3", [[3, 12, 9, 1, 14], [6, 7, 15, 11, 2]])]


def batch_at(cursor):
    name, rows = STREAM[cursor % len(STREAM)]
    ids = torch.tensor(rows)
    return name, ids, torch.ones_like(ids)


def loss_of(model, ids, attn):
    logits = model(ids, attention_mask=attn)
    labels = ids.clone()
    labels[attn == 0] = IGNORE
    targets = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    return total / int((targets != IGNORE).sum())


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# --------------------------------------------------------------------------
# 第一部分：五种启动方式
# --------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """W 冻结，只训练 A、B；B 初始化为 0 使初始增量为 0。"""

    def __init__(self, base: nn.Linear, rank: int = 2):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=DTYPE) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=DTYPE))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


def build_mode(mode: str, pretrained: dict):
    """返回 (model, teacher 或 None, 说明)。"""
    model = TinyLM()
    if mode == "from_scratch":
        return model, None, "重新采样全部参数"
    model.load_state_dict(pretrained)
    if mode == "load_pretrained":
        return model, None, "加载已有权重，全参可训练"
    if mode == "freeze_backbone":
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(("down", "final_norm")))
        return model, None, "只训练 down 与 final_norm，其余冻结"
    if mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        torch.manual_seed(3)
        model.qkv = LoRALinear(model.qkv, rank=2)
        return model, None, "base 全冻结，只训练 qkv 的 A/B"
    if mode == "teacher_student":
        teacher = TinyLM(seed=21)               # 另一份权重，teacher 不等于 student 起点
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return model, teacher, "学生从已有权重起步，teacher 冻结且只前向"
    raise ValueError(mode)


def start_modes():
    torch.manual_seed(11)
    pretrained = copy.deepcopy(TinyLM(seed=5).state_dict())
    name, ids, attn = batch_at(0)
    print(f"共同输入 batch={name}，shape={tuple(ids.shape)}\n")
    header = ("启动方式         | 总参数 | 可训练 | 参数组 | step 后有状态的张量 |"
              " 数值改变的张量 | 最大 |Δ|")
    print(header)
    print("-" * len(header))
    for mode in ("from_scratch", "load_pretrained", "freeze_backbone", "lora",
                 "teacher_student"):
        model, teacher, note = build_mode(mode, pretrained)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([p for _, p in trainable], lr=LR, weight_decay=WD,
                                foreach=False)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        if teacher is None:
            loss = loss_of(model, ids, attn)
        else:
            with torch.no_grad():
                t_logits = teacher(ids, attention_mask=attn)
            s_logits = model(ids, attention_mask=attn)
            loss = F.kl_div(F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1),
                            reduction="batchmean", log_target=True)
        loss.backward()
        opt.step()
        changed = [(n, float((p.detach() - before[n]).abs().max()))
                   for n, p in model.named_parameters()
                   if not torch.equal(p.detach(), before[n])]
        with_state = sum(1 for _, p in trainable if opt.state.get(p))
        total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for _, p in trainable)
        biggest = max((d for _, d in changed), default=0.0)
        print(f"{mode:16s} | {total:6d} | {n_train:6d} | {len(trainable):6d} |"
              f" {with_state:19d} | {len(changed):14d} | {biggest:.3e}")
        print(f"                 └ {note}；loss={loss.item():.8f}")
        if mode == "lora":
            names = [n for n, _ in changed]
            print(f"                   本次改变的张量: {names}"
                  "（B=0 时 A 的梯度为 0，第一次只更新 B）")
        if mode == "teacher_student":
            print(f"                   teacher 参数 requires_grad 全为 False，"
                  f"不进入 optimizer；学生 loss 是 KL 而不是 token CE")
    print("\n五种启动共用同一套系统接口（batch → loss → backward → optimizer.step），"
          "区别只在参数来源、requires_grad 划分和监督定义；\n"
          "optimizer 状态在第一次 step 时按参数组现场创建，冻结的参数既不占状态也不产生梯度。")


# --------------------------------------------------------------------------
# 第二部分：严格恢复与逐项反例
# --------------------------------------------------------------------------

def make_trainer(dropout=0.1):
    model = TinyLM(seed=5)
    model.dropout.p = dropout
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.2,
                                              total_iters=6)
    return model, opt, sched


def train_step(model, opt, sched, cursor):
    name, ids, attn = batch_at(cursor)
    opt.zero_grad(set_to_none=True)
    loss = loss_of(model, ids, attn)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    lr = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
    return {"sample": name, "loss": loss.item(), "grad_norm": norm, "lr": lr,
            "param": torch.cat([p.detach().flatten().clone() for p in model.parameters()]),
            "exp_avg": torch.cat([opt.state[p]["exp_avg"].flatten().clone()
                                  for p in model.parameters()])}


def save_blob(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def load_blob(blob):
    return torch.load(io.BytesIO(blob), weights_only=False)


def resume_trace():
    model, opt, sched = make_trainer()
    torch.manual_seed(2026)
    cursor = 0
    first = train_step(model, opt, sched, cursor)
    cursor += 1
    checkpoint = save_blob({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(),
                            "rng": torch.get_rng_state(), "cursor": cursor})
    print(f"第 1 次更新: sample={first['sample']} loss={first['loss']:.8f} "
          f"lr={first['lr']:.6f} grad_norm={first['grad_norm']:.6f}")
    print(f"checkpoint 序列化后 {len(checkpoint)} 字节，包含 "
          f"{sorted(load_blob(checkpoint).keys())}")

    reference = [train_step(model, opt, sched, cursor + i) for i in range(2)]
    for i, r in enumerate(reference, 2):
        print(f"不中断的第 {i} 次更新: sample={r['sample']} loss={r['loss']:.8f} "
              f"lr={r['lr']:.6f}")
    return checkpoint, reference


def replay(checkpoint, drop: str | None):
    state = load_blob(checkpoint)
    model, opt, sched = make_trainer()
    model.load_state_dict(state["model"])
    opt_state = copy.deepcopy(state["optimizer"])
    if drop == "optimizer_state":
        opt_state["state"] = {}                 # 保留 param_groups（LR 在这里），清掉 m/v/step
    opt.load_state_dict(opt_state)
    if drop != "scheduler":
        sched.load_state_dict(state["scheduler"])
    if drop != "rng":
        torch.set_rng_state(state["rng"])
    else:
        torch.manual_seed(12345)
    cursor = 0 if drop == "cursor" else state["cursor"]
    return [train_step(model, opt, sched, cursor + i) for i in range(2)]


def first_break(reference, actual):
    """按 样本 → loss → 梯度范数 → LR → 参数 → m 的顺序找第一个不一致的量。"""
    for i, (ref, act) in enumerate(zip(reference, actual), 2):
        if ref["sample"] != act["sample"]:
            return f"第 {i} 步的样本 ID（{ref['sample']} → {act['sample']}）", None
        for key in ("loss", "grad_norm", "lr"):
            if ref[key] != act[key]:
                return f"第 {i} 步的 {key}", abs(ref[key] - act[key])
        for key in ("param", "exp_avg"):
            d = float((ref[key] - act[key]).abs().max())
            if d != 0.0:
                return f"第 {i} 步的 {key}", d
    return "无（逐元素一致）", 0.0


def main():
    section("A. 五种启动方式的第一次 step")
    start_modes()

    section("B. 更新 — 保存 — 恢复 — 再更新")
    checkpoint, reference = resume_trace()

    print("\n完整恢复后重跑同样两步：")
    full = replay(checkpoint, drop=None)
    for i, (ref, act) in enumerate(zip(reference, full), 2):
        print(f"  第 {i} 步 sample={act['sample']} loss={act['loss']:.8f} "
              f"参数最大差 {float((ref['param'] - act['param']).abs().max()):.1e} "
              f"m 最大差 {float((ref['exp_avg'] - act['exp_avg']).abs().max()):.1e}")

    section("C. 逐项移除一种状态，记录第一个被破坏的不变量")
    print("移除项      | 第一个不一致的量              | 差值")
    print("-" * 68)
    for drop, label in (("optimizer_state", "m/v/step"), ("scheduler", "scheduler"),
                        ("rng", "RNG"), ("cursor", "数据游标")):
        where, delta = first_break(reference, replay(checkpoint, drop))
        shown = "—" if delta is None else f"{delta:.3e}"
        print(f"{label:11s} | {where:28s} | {shown}")
    print("\n顺序是固定的：游标错了在样本 ID 上就能看见；RNG 错了要到 loss 才暴露；"
          "scheduler 错了 loss、梯度和当前 LR 都对，下一步的 LR 才偏；\n"
          "m/v/step 错了前四项全部正确，第一次参数写回才出现差异——"
          "只比一次 forward loss 的恢复检查正好漏掉这一类。")
    print("学习率本身存在 optimizer.state_dict()['param_groups'] 里，"
          "scheduler 的 state_dict 只有 last_epoch 一类计数；两者缺一都不能还原 LR 曲线。")

    section("D. 只加载权重：另一次训练，而不是同一条轨迹的接续")
    state = load_blob(checkpoint)
    model, opt, sched = make_trainer()
    model.load_state_dict(state["model"])
    torch.manual_seed(999)
    warm = [train_step(model, opt, sched, i) for i in range(2)]
    saved_m = max(float(v["exp_avg"].abs().max()) for v in state["optimizer"]["state"].values())
    print(f"  参数起点与 checkpoint 一致，其余三项全部重置：")
    print(f"    样本 = {warm[0]['sample']}（从数据流开头重新取，不是 {reference[0]['sample']}）")
    print(f"    LR = {warm[0]['lr']:.6f}（回到调度起点，不是 {reference[0]['lr']:.6f}）")
    print(f"    m 起点 = 0，而 checkpoint 里保存的 m 最大值是 {saved_m:.6f}")
    print(f"    第一步 loss = {warm[0]['loss']:.8f}，不中断轨迹的同一步是 "
          f"{reference[0]['loss']:.8f}")
    print("  这条路径是合法的 fine-tuning 初始化，但它不满足严格恢复的任何一项不变量；"
          "把它当成 resume 会改变优化轨迹而不报任何错。")


if __name__ == "__main__":
    main()
