#!/usr/bin/env python3
"""梯度累积的分母：三条路径的梯度与参数对照，以及顺序控制的三个反例。

四个有效数不同的 microbatch 上比较
  P1 直接大 batch        Σ_all / N_total
  P2 累积 + 全局分母      每份 S_m / N_total
  P3 累积 + 每份自己的均值 (S_m / N_m) / M   —— HF 与 Nanotron 的默认口径
并给出 dropout/RNG、clip 位置、scheduler 推进三处次序反例。
最后用归档的 SmolLM2-360M 运行验算 P3 与 P1 的关系。

Usage:
    python labs/L7/accumulation_denominator.py > "$RUN_DIR/accumulation.txt"
    python labs/L7/accumulation_denominator.py --tokenizer <tokenizer.json>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from _tinylm import TinyLM

IGNORE = -100
LR, WD = 1e-2, 0.1
ARCHIVE = Path("results/crater/7.0b/training_complete.txt")

# 四条固定样本，长度 7/5/4/6，分别产生 6/4/3/5 个 target。
MICRO = [[5, 9, 3, 7, 2, 11, 6],
         [4, 11, 6, 13, 1],
         [8, 1, 12, 10],
         [2, 14, 5, 3, 9, 7]]
EQUAL = [[5, 9, 3, 7], [4, 11, 6, 13], [8, 1, 12, 10], [2, 14, 5, 3]]


def pad(rows):
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    attn = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    return ids, attn


def loss_sum_and_count(model, ids, attn):
    logits = model(ids, attention_mask=attn)
    labels = ids.clone()
    labels[attn == 0] = IGNORE
    targets = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    return total, int((targets != IGNORE).sum())


def grads_of(model):
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def max_diff(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def fresh(train_mode=False, dropout=0.0):
    model = TinyLM()
    model.dropout.p = dropout
    model.train(train_mode)
    return model


def run_direct(rows, dropout=0.0, seed=None):
    model = fresh(train_mode=dropout > 0, dropout=dropout)
    if seed is not None:
        torch.manual_seed(seed)
    ids, attn = pad(rows)
    total, n = loss_sum_and_count(model, ids, attn)
    (total / n).backward()
    return model, grads_of(model), n, (total / n).item()


def run_accumulated(rows, global_denominator: bool, dropout=0.0, seed=None,
                    clip_each=None, clip_end=None):
    model = fresh(train_mode=dropout > 0, dropout=dropout)
    counts = []
    for r in rows:
        ids, attn = pad([r])
        counts.append(loss_sum_and_count(model, ids, attn)[1])
    n_total, m = sum(counts), len(rows)
    per_micro, norms = [], []
    for i, r in enumerate(rows):
        if seed is not None:
            torch.manual_seed(seed + i)
        ids, attn = pad([r])
        total, n = loss_sum_and_count(model, ids, attn)
        scaled = total / n_total if global_denominator else (total / n) / m
        scaled.backward()
        per_micro.append({"n": n, "sum": total.item(), "mean": (total / n).item(),
                          "scaled": scaled.item()})
        if clip_each is not None:
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_each)))
    if clip_end is not None:
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_end)))
    return model, grads_of(model), per_micro, n_total, norms


def one_step(model):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    opt.step()
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def compare(rows, label):
    _, g1, n_total, loss1 = run_direct(rows)
    m2, g2, rec2, _, _ = run_accumulated(rows, global_denominator=True)
    m3, g3, rec3, _, _ = run_accumulated(rows, global_denominator=False)
    print(f"\n[{label}] 每份有效数 {[r['n'] for r in rec2]}，N_total={n_total}，M={len(rows)}")
    print("  microbatch |  N  |     Σloss    |   Σloss/N   |  P2 系数贡献 |  P3 系数贡献")
    for i, (a, b) in enumerate(zip(rec2, rec3)):
        print(f"      {i}      | {a['n']:3d} | {a['sum']:12.8f} | {a['mean']:11.8f} |"
              f" {a['scaled']:12.8f} | {b['scaled']:12.8f}")
    print(f"  P1 直接大 batch loss = {loss1:.8f}")
    print(f"  P2 累积和            = {sum(r['scaled'] for r in rec2):.8f}")
    print(f"  P3 累积和            = {sum(r['scaled'] for r in rec3):.8f}")
    print(f"  梯度最大差 P1↔P2 = {max_diff(g1, g2):.3e}")
    print(f"  梯度最大差 P1↔P3 = {max_diff(g1, g3):.3e}")
    _, gd, _, _ = run_direct(rows)
    p1 = one_step(fresh_with(gd))
    p2, p3 = one_step(m2), one_step(m3)
    print(f"  一次 AdamW 更新后参数最大差 P1↔P2 = "
          f"{max(float((p1[k] - p2[k]).abs().max()) for k in p1):.3e}")
    print(f"  一次 AdamW 更新后参数最大差 P1↔P3 = "
          f"{max(float((p1[k] - p3[k]).abs().max()) for k in p1):.3e}")
    return rec2


def fresh_with(grads):
    model = fresh()
    for name, p in model.named_parameters():
        p.grad = grads[name].clone()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE,
                        help="归档的 SmolLM2-360M 运行日志")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="SmolLM2 的 tokenizer.json，用于重新推导每份有效数")
    args = parser.parse_args()

    section("1. 三条路径的梯度与参数（有效数不等）")
    rec = compare(MICRO, "长度 7/5/4/6")
    n_total, m = sum(r["n"] for r in rec), len(rec)
    print("\n  每个 target token 在两种口径下的权重：")
    print("  microbatch |  N  |  P1/P2 权重 1/N_total |  P3 权重 1/(M·N)")
    for i, r in enumerate(rec):
        print(f"      {i}      | {r['n']:3d} |      {1 / n_total:.8f}      |"
              f"    {1 / (m * r['n']):.8f}")
    print("  P3 给每个 microbatch 相同的总权重 1/M，短序列里的每个 token 因此被放大。")

    section("2. 有效数相等时 P3 与 P1 重合")
    compare(EQUAL, "长度全为 4")

    section("3. 反例：dropout 打开后，分母正确也不等价")
    _, g_direct, _, _ = run_direct(MICRO, dropout=0.2, seed=7)
    _, g_accum, _, _, _ = run_accumulated(MICRO, global_denominator=True, dropout=0.2, seed=7)
    print(f"  P1 与 P2 的梯度最大差 = {max_diff(g_direct, g_accum):.3e}")
    print("  两条路径抽到的 dropout mask 不同：一次大 batch 抽一张 [B,T,D] 掩码，"
          "四次 microbatch 抽四张 [1,T,D]。分母只负责权重，RNG 消费顺序是另一条控制面。")
    _, g_eval, _, _ = run_direct(MICRO, dropout=0.0)
    _, g_eval_acc, _, _, _ = run_accumulated(MICRO, global_denominator=True, dropout=0.0)
    print(f"  关闭 dropout 后同一对照 = {max_diff(g_eval, g_eval_acc):.3e}")

    section("4. 反例：clip 放在每个 microbatch 之后")
    _, g_end, _, _, norms_end = run_accumulated(MICRO, global_denominator=True, clip_end=1.0)
    _, g_each, _, _, norms_each = run_accumulated(MICRO, global_denominator=True, clip_each=1.0)
    print(f"  窗口末尾一次 clip：total_norm={norms_end[0]:.8f}")
    print(f"  每份都 clip：逐份 total_norm={[round(x, 8) for x in norms_each]}")
    print(f"  两种写法的最终梯度最大差 = {max_diff(g_end, g_each):.3e}")
    print("  逐份裁剪改变的是各 microbatch 的相对权重，窗口末尾的全局范数不再是被裁剪的对象。")

    section("5. 反例：scheduler 跟着 microbatch 走")
    for name, per_micro in (("按 optimizer step 推进", False), ("按 microbatch 推进", True)):
        model = fresh()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1,
                                                  total_iters=10)
        for _ in range(3):                                 # 3 个 optimizer step
            for i, r in enumerate(MICRO):
                ids, attn = pad([r])
                total, n = loss_sum_and_count(model, ids, attn)
                (total / n / len(MICRO)).backward()
                if per_micro:
                    sched.step()
            opt.step()
            opt.zero_grad()
            if not per_micro:
                sched.step()
        print(f"  {name}: 3 次更新后 LR = {opt.param_groups[0]['lr']:.8f}，"
              f"scheduler.last_epoch = {sched.last_epoch}")
    print("  microbatch 数一变，学习率曲线就跟着变；scheduler 的时钟只能是 optimizer step。")

    section("6. 归档的 SmolLM2-360M 运行：P3 与 P1 的关系验算")
    if not args.archive.exists():
        print(f"  未找到 {args.archive}，跳过。")
        return
    text = args.archive.read_text()
    means = [float(x) for x in re.findall(r"Micro-batch \d: loss = ([0-9.]+)", text)]
    direct = float(re.search(r"Direct batch loss: ([0-9.]+)", text).group(1))
    counts = [8, 8, 9, 9]
    source = "按 SmolLM2 tokenizer 的 token 数减 1 写入"
    if args.tokenizer is not None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(args.tokenizer))
        sentences = [
            "The first example sentence for gradient accumulation test.",
            "The second example with different content and length.",
            "Third sentence is about machine learning and deep learning.",
            "Fourth and final sentence concludes the micro batch set.",
        ]
        counts = [len(tok.encode(s).ids) - 1 for s in sentences]
        source = f"由 {args.tokenizer} 现场推导"
    weighted = sum(n * mu for n, mu in zip(counts, means)) / sum(counts)
    print(f"  逐 microbatch 均值（日志原文）: {means}")
    print(f"  每份有效 target 数: {counts}（{source}），合计 {sum(counts)}")
    print(f"  Σ 均值            = {sum(means):.6f}   （日志记为 loss_accumulated）")
    print(f"  Σ 均值 / 4        = {sum(means) / 4:.6f}")
    print(f"  按 token 加权平均  = {weighted:.6f}")
    print(f"  日志的 direct loss = {direct:.6f}，差 {abs(weighted - direct):.2e}")
    print("  该运行的累积路径就是 P3：四个 microbatch 均值等权相加。它与直接大 batch 的"
          "差别不是浮点噪声，而是 8/8/9/9 这组不等有效数带来的权重差，参数差 2e-4 由此而来。")


if __name__ == "__main__":
    main()
