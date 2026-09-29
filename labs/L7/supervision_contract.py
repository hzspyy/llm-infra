#!/usr/bin/env python3
"""从样本 ID 到 loss 分母：shift、attention mask、loss mask 与有效元素计数。

在 TinyLM（FP64）上逐位置打印监督关系，并给出四组反例：
重复 shift、prompt mask 改变分母、左 padding 缺 attention mask、有效数为 0。
末尾用二维 flow matching 的连续目标做对照，说明分母单位不同、数值不可并列。

Usage:
    python labs/L7/supervision_contract.py > "$RUN_DIR/supervision.txt"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from _tinylm import DTYPE, FlowNet, TinyLM

PAD, IGNORE = 0, -100

# 三条固定样本：ID 可追溯，长度不同，第三条是 prompt+answer 的 SFT 结构。
SAMPLES = [
    {"id": "s0", "tokens": [5, 9, 3, 7, 2], "prompt_len": 0},
    {"id": "s1", "tokens": [4, 11, 6], "prompt_len": 0},
    {"id": "s2", "tokens": [8, 1, 12, 10, 6, 14], "prompt_len": 3},
]


def right_pad(samples, width):
    """右 padding：input_ids、attention_mask、与输入同位置的 labels。"""
    ids, attn, labels = [], [], []
    for s in samples:
        n = len(s["tokens"])
        ids.append(s["tokens"] + [PAD] * (width - n))
        attn.append([1] * n + [0] * (width - n))
        row = list(s["tokens"]) + [IGNORE] * (width - n)
        for i in range(s["prompt_len"]):       # prompt 位置不作为预测目标
            row[i] = IGNORE
        labels.append(row)
    return (torch.tensor(ids), torch.tensor(attn), torch.tensor(labels))


def shift_targets(labels):
    """transformers 的对齐方式：右补一个 ignore 再左移，logits 不切片。

    见 transformers v5.17.0 src/transformers/loss/loss_utils.py:61-64。
    """
    padded = F.pad(labels, (0, 1), value=IGNORE)
    return padded[..., 1:]


def token_loss(logits, labels):
    """返回 (逐位置 loss, 有效掩码, 有效数, sum, sum/N)。"""
    targets = shift_targets(labels)
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="none")
    per_pos = flat.view(targets.shape)
    valid = targets != IGNORE
    n = int(valid.sum())
    total = per_pos[valid].sum()
    return per_pos, valid, n, total, (total / n if n else None)


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    torch.manual_seed(0)
    model = TinyLM()
    model.eval()                                   # dropout 关闭，结果只由输入决定
    width = max(len(s["tokens"]) for s in SAMPLES)
    ids, attn, labels = right_pad(SAMPLES, width)
    logits = model(ids, attention_mask=attn)

    section("1. batch 逐位置监督表（右 padding，width=%d）" % width)
    per_pos, valid, n, total, mean = token_loss(logits, labels)
    targets = shift_targets(labels)
    print("样本 | 位置 | input_id | attn | label | shift 目标 | 逐位置 loss")
    for b, s in enumerate(SAMPLES):
        for t in range(width):
            tgt = int(targets[b, t])
            mark = f"{per_pos[b, t].item():.6f}" if valid[b, t] else "—（不计入）"
            print(f"  {s['id']} |  {t}   |    {int(ids[b, t]):3d}   |  {int(attn[b, t])}   |"
                  f" {int(labels[b, t]):5d} | {tgt:5d}      | {mark}")
    print(f"\n有效 target 数 N = {n}（= Σ(len_i − 1) − prompt 屏蔽数）")
    print(f"Σloss = {total.item():.8f}    Σloss/N = {mean.item():.8f}")
    builtin = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              targets.reshape(-1), ignore_index=IGNORE, reduction="mean")
    print(f"F.cross_entropy(reduction='mean') = {builtin.item():.8f}"
          f"    与手算差 {abs(builtin - mean).item():.3e}")

    section("2. 反例：labels 已预移位，模型内部再移一次")
    pre_shifted = torch.full_like(labels, IGNORE)
    pre_shifted[:, :-1] = labels[:, 1:]            # 调用方先做了一次 shift
    _, _, n2, _, mean2 = token_loss(logits, pre_shifted)
    tgt2 = shift_targets(pre_shifted)
    print(f"单次 shift 的目标（样本 s0）: {shift_targets(labels)[0].tolist()}")
    print(f"两次 shift 的目标（样本 s0）: {tgt2[0].tolist()}")
    print(f"N: {n} → {n2}；loss: {mean.item():.8f} → {mean2.item():.8f}")
    print("位置 0 的目标从第 2 个 token 变成第 3 个，整条序列错一位，loss 仍然是有限数。")

    section("3. prompt mask：attention 可见范围与监督范围是两个控制面")
    no_prompt_mask = [dict(s, prompt_len=0) for s in SAMPLES]
    _, _, n_full, _, mean_full = token_loss(logits, right_pad(no_prompt_mask, width)[2])
    print(f"不屏蔽 prompt: N={n_full}，loss={mean_full.item():.8f}")
    print(f"屏蔽 s2 的前 3 个 prompt 位置: N={n}，loss={mean.item():.8f}")
    boundary = SAMPLES[2]["prompt_len"] - 1
    print(f"s2 的最后一个 prompt 位置 {boundary} 仍在监督范围内："
          f"它的 shift 目标是 {int(targets[2, boundary])}，即第一个 answer token；"
          f"prompt 屏蔽只到位置 {boundary - 1}。")

    section("4. 反例：左 padding 不传 attention_mask")
    left_ids = torch.tensor([[PAD, PAD, PAD] + SAMPLES[1]["tokens"]])
    left_attn = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with_mask = model(left_ids, attention_mask=left_attn)
    without = model(left_ids, attention_mask=None)
    real = slice(3, 6)
    delta = (with_mask[0, real] - without[0, real]).abs().max()
    right_only = model(ids[1:2, :3], attention_mask=attn[1:2, :3])
    print(f"左 padding，有效位置 logits 最大差（传/不传 mask）: {delta.item():.6e}")
    print(f"与无 padding 的同一序列比较: {(with_mask[0, real] - right_only[0]).abs().max().item():.3e}")
    right_pad_ids = torch.tensor([SAMPLES[1]['tokens'] + [PAD] * 3])
    r_with = model(right_pad_ids, attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]]))
    r_without = model(right_pad_ids, attention_mask=None)
    print(f"右 padding 同一比较: {(r_with[0, :3] - r_without[0, :3]).abs().max().item():.3e}"
          "（因果 mask 已经挡住右侧 pad，所以缺 attention_mask 不改变有效位置）")
    print("loss mask 在两种情形下完全相同：它决定哪些预测计入目标，不决定谁能被看见。")

    section("5. 有效数为 0 的整步协议")
    empty = torch.full_like(labels[:1], IGNORE)
    _, _, n0, sum0, _ = token_loss(logits[:1], empty)
    bad = F.cross_entropy(logits[:1].reshape(-1, logits.shape[-1]),
                          shift_targets(empty).reshape(-1), ignore_index=IGNORE, reduction="mean")
    print(f"N={n0}，Σloss={sum0.item():.1f}；F.cross_entropy(reduction='mean') = {bad.item()}")
    print("直接相除得到 nan，一次 backward 就会污染全部梯度；N==0 必须按整步协议跳过。")

    section("6. 连续目标对照：二维 flow matching 的分母单位")
    torch.manual_seed(1)
    flow = FlowNet()
    x0 = torch.randn(4, 3, 2, dtype=DTYPE)         # [样本, 帧, 坐标]
    x1 = torch.randn(4, 3, 2, dtype=DTYPE)
    t = torch.rand(4, 3, 1, dtype=DTYPE)
    frame_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=DTYPE)
    x_t = (1 - t) * x0 + t * x1
    target_u = x1 - x0
    sq = (flow(x_t, t) - target_u).pow(2) * frame_mask[..., None]
    n_frames = int(frame_mask.sum())
    n_coords = n_frames * 2
    print(f"有效帧 {n_frames}，有效坐标 {n_coords}")
    print(f"按坐标平均 MSE = {(sq.sum() / n_coords).item():.8f}")
    print(f"按帧平均   MSE = {(sq.sum() / n_frames).item():.8f}（正好是前者的 2 倍）")
    print(f"同一 batch 的 token CE = {mean.item():.8f} nats/token")
    print("CE 的单位是每个 target token 的负对数似然，MSE 的单位是每个坐标的平方误差；"
          "两者只共用「Σ 目标项 / 有效元素数」这套系统接口，数值本身不可排名。")
    print("改变分母约定（帧 vs 坐标）等价于把学习率乘上坐标数，梯度方向不变、步长变。")


if __name__ == "__main__":
    main()
