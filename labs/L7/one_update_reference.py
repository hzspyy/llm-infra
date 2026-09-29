#!/usr/bin/env python3
"""一次有效更新的数值参照：梯度 → global clip → AdamW，与 PyTorch 逐参数对拍。

TinyLM（FP64）上执行两次更新。手写 AdamW 同时给出教科书写法和 PyTorch 的
lerp/addcdiv 写法，用来区分"公式等价"和"浮点顺序一致"。
最后给 L2 正则 Adam 与 decoupled AdamW 的对照，说明 weight_decay 的语义差别。

Usage:
    python labs/L7/one_update_reference.py > "$RUN_DIR/one-update.txt"
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from _tinylm import TinyLM

LR, BETA1, BETA2, EPS, WD = 1e-2, 0.9, 0.95, 1e-8, 0.1
MAX_NORM = 1.0
IGNORE = -100

BATCH = torch.tensor([[5, 9, 3, 7, 2, 1],
                      [4, 11, 6, 13, 0, 0]])
ATTN = torch.tensor([[1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0]])


def labels_from(batch, attn):
    labels = batch.clone()
    labels[attn == 0] = IGNORE
    return labels


def loss_sum_and_count(model, batch, attn):
    logits = model(batch, attention_mask=attn)
    targets = F.pad(labels_from(batch, attn), (0, 1), value=IGNORE)[..., 1:]
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="sum")
    return flat, int((targets != IGNORE).sum())


def manual_clip(params, max_norm):
    """复现 torch.nn.utils.clip_grad_norm_ 的二范数与缩放系数。"""
    norms = torch.stack([p.grad.norm(2) for p in params])
    total = norms.norm(2)
    coef = max_norm / (total + 1e-6)
    coef = torch.clamp(coef, max=1.0)
    return total, coef


class ManualAdamW:
    """decoupled weight decay 的 AdamW。textbook=True 用教科书写法，否则复现 torch 顺序。"""

    def __init__(self, params, textbook: bool):
        self.params = list(params)
        self.textbook = textbook
        self.state = [{"step": 0,
                       "m": torch.zeros_like(p),
                       "v": torch.zeros_like(p)} for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, st in zip(self.params, self.state):
            g = p.grad
            st["step"] += 1
            k = st["step"]
            p.mul_(1 - LR * WD)                      # decoupled：只作用在参数上
            if self.textbook:
                st["m"] = BETA1 * st["m"] + (1 - BETA1) * g
                st["v"] = BETA2 * st["v"] + (1 - BETA2) * g * g
                m_hat = st["m"] / (1 - BETA1 ** k)
                v_hat = st["v"] / (1 - BETA2 ** k)
                p.sub_(LR * m_hat / (v_hat.sqrt() + EPS))
            else:
                st["m"].lerp_(g, 1 - BETA1)
                st["v"].mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
                bc1 = 1 - BETA1 ** k
                bc2_sqrt = math.sqrt(1 - BETA2 ** k)
                denom = (st["v"].sqrt() / bc2_sqrt).add_(EPS)
                p.addcdiv_(st["m"], denom, value=-LR / bc1)


def max_diff(a, b):
    return float((a - b).abs().max())


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def run_reference(model, opt, manual: bool, steps: int):
    """执行 steps 次完整更新，返回每次更新的记录。"""
    records = []
    for _ in range(steps):
        for p in model.parameters():
            p.grad = None
        loss_sum, n = loss_sum_and_count(model, BATCH, ATTN)
        (loss_sum / n).backward()
        params = [p for p in model.parameters() if p.grad is not None]
        total, coef = manual_clip(params, MAX_NORM)
        if manual:
            for p in params:
                p.grad.mul_(coef)
        else:
            total_t = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
            total = total_t
        opt.step()
        records.append({"loss": (loss_sum / n).item(), "n": n,
                        "grad_norm": float(total), "coef": float(coef)})
    return records


def main():
    section("1. batch 与有效数")
    model = TinyLM()
    model.eval()
    loss_sum, n = loss_sum_and_count(model, BATCH, ATTN)
    print(f"input_ids shape={tuple(BATCH.shape)}，有效 target N={n}")
    print(f"Σloss={loss_sum.item():.8f}，loss=Σ/N={(loss_sum / n).item():.8f}")

    section("2. 逐参数梯度与 global clip")
    (loss_sum / n).backward()
    params = [p for p in model.parameters()]
    names = [name for name, _ in model.named_parameters()]
    total, coef = manual_clip(params, MAX_NORM)
    print("参数 | shape | numel | grad 二范数")
    for name, p in zip(names, params):
        print(f"  {name:16s} | {str(tuple(p.shape)):12s} | {p.numel():5d} | {p.grad.norm(2).item():.8f}")
    # Parameter 的 deepcopy 不带 .grad，这里显式复制一份梯度再交给 PyTorch。
    mirror = [torch.zeros_like(p) for p in params]
    for m_t, p in zip(mirror, params):
        m_t.grad = p.grad.clone()
    torch_total = torch.nn.utils.clip_grad_norm_(mirror, MAX_NORM)
    print(f"\n手算 total_norm = {total.item():.10f}，clip_grad_norm_ 返回 {torch_total.item():.10f}"
          f"，差 {abs(total - torch_total).item():.3e}")
    print(f"缩放系数 coef = min(1, {MAX_NORM}/(total+1e-6)) = {coef.item():.10f}"
          f"（{'发生裁剪' if coef.item() < 1 else '未裁剪'}）")

    section("3. 手写 AdamW 与 torch.optim.AdamW 逐参数对拍（两次更新）")
    torch_model = TinyLM()
    torch_opt = torch.optim.AdamW(torch_model.parameters(), lr=LR, betas=(BETA1, BETA2),
                                  eps=EPS, weight_decay=WD, foreach=False, fused=False)
    torch_rec = run_reference(torch_model, torch_opt, manual=False, steps=2)

    variants = {}
    for textbook in (True, False):
        m = TinyLM()
        opt = ManualAdamW(m.parameters(), textbook=textbook)
        rec = run_reference(m, opt, manual=True, steps=2)
        variants["textbook" if textbook else "torch 写法"] = (m, opt, rec)

    for i, rec in enumerate(torch_rec, 1):
        print(f"第 {i} 次更新: loss={rec['loss']:.8f}  N={rec['n']}  "
              f"grad_norm={rec['grad_norm']:.8f}  coef={rec['coef']:.6f}")

    torch_state = {name: torch_opt.state[p] for name, p in torch_model.named_parameters()}
    for label, (m, opt, rec) in variants.items():
        p_err = m_err = v_err = 0.0
        for idx, (name, p) in enumerate(m.named_parameters()):
            ref_p = dict(torch_model.named_parameters())[name]
            p_err = max(p_err, max_diff(p.detach(), ref_p.detach()))
            m_err = max(m_err, max_diff(opt.state[idx]["m"], torch_state[name]["exp_avg"]))
            v_err = max(v_err, max_diff(opt.state[idx]["v"], torch_state[name]["exp_avg_sq"]))
        steps_ok = all(opt.state[i]["step"] == int(torch_state[nm]["step"])
                       for i, nm in enumerate(dict(m.named_parameters())))
        print(f"\n[{label}] 与 torch.optim.AdamW 的最大绝对差")
        print(f"  参数 {p_err:.3e}   exp_avg {m_err:.3e}   exp_avg_sq {v_err:.3e}   "
              f"step 计数一致={steps_ok}")

    print("\n教科书写法与 torch 写法在数学上相同；差值全部来自 lerp_/addcdiv_ 的运算顺序，"
          "在 FP64 上是 1e-17 量级。BF16 训练里同一顺序差别会放大，见 7.9。")

    section("4. optimizer 状态清单")
    total_numel = sum(p.numel() for p in torch_model.parameters())
    sample = torch_state["qkv.weight"]
    print(f"参数总数 {total_numel}；AdamW 为每个参数保存 exp_avg、exp_avg_sq 与 step")
    print(f"  参数 dtype={next(torch_model.parameters()).dtype}，"
          f"exp_avg dtype={sample['exp_avg'].dtype}，step={sample['step']}")
    per_param_bytes = (sample["exp_avg"].element_size() + sample["exp_avg_sq"].element_size())
    print(f"  本例每个参数的 optimizer 状态 {per_param_bytes} 字节（FP64 参照），"
          f"全模型 {per_param_bytes * total_numel} 字节")
    print("  FP32 训练的同一项是 8 字节/参数；master weight、FP32 梯度缓冲与通信 dtype "
          "另算，完整字节账见 7.9。")

    section("5. 反例：L2 正则 Adam 不等于 decoupled AdamW")
    a = TinyLM()
    b = TinyLM()
    opt_a = torch.optim.AdamW(a.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS,
                              weight_decay=WD, foreach=False)
    opt_b = torch.optim.Adam(b.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS,
                             weight_decay=WD, foreach=False)          # 把 wd 加进梯度
    for model_x, opt_x in ((a, opt_a), (b, opt_b)):
        for p in model_x.parameters():
            p.grad = None
        loss_sum_x, n_x = loss_sum_and_count(model_x, BATCH, ATTN)
        (loss_sum_x / n_x).backward()
        torch.nn.utils.clip_grad_norm_(model_x.parameters(), MAX_NORM)
        opt_x.step()
    err = max(max_diff(pa.detach(), pb.detach()) for pa, pb in zip(a.parameters(), b.parameters()))
    print(f"同一 loss、同一 wd={WD}，一次更新后参数最大差 {err:.3e}")
    print("AdamW 把衰减直接乘在参数上，Adam(weight_decay=) 把 wd*param 加进梯度后再过 m/v，"
          "两者的有效衰减强度随 v 变化，不是同一个目标。")


if __name__ == "__main__":
    main()
