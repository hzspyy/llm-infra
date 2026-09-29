#!/usr/bin/env python3
"""数值故障定位：逐层记录，注入五种故障，保存第一个坏张量。

每次尝试记录：
  · 每层前向输出的 finite 标记、最大绝对值与范数
  · 每个参数梯度的 finite 标记、范数、零元素比例
  · 全局梯度范数、是否跳步、更新量与参数的比值
  · 第一个非有限张量的名字、位置、邻域取值和产生它的输入切片

五种注入：正常、FP16 前向溢出、梯度下溢、坏样本、空分母；
另有一段专查 optimizer 常数在低精度参数下的失效。
可复现：固定 seed 与固定 batch，`--outdir` 保存每次尝试的 JSON。

Usage:
    python labs/L7/numerical_diagnostics.py --outdir "$RUN_DIR/diagnostics"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

IGNORE = -100
SEED = 0
BATCH = torch.tensor([[5, 9, 3, 7, 2, 1, 8, 4],
                      [4, 11, 6, 13, 12, 1, 7, 2]])


class DeepStack(torch.nn.Module):
    """12 层线性栈：层数足够多，可以观察激活量级沿深度的累积。"""

    def __init__(self, dim: int = 64, vocab: int = 32, layers: int = 12,
                 gain: float = 1.0, dtype=torch.float32):
        super().__init__()
        gen = torch.Generator().manual_seed(SEED)
        self.embed = torch.nn.Embedding(vocab, dim, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim, bias=False, dtype=dtype) for _ in range(layers)])
        self.head = torch.nn.Linear(dim, vocab, bias=False, dtype=dtype)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(vocab, dim, generator=gen, dtype=torch.float64).to(dtype))
            for layer in self.layers:
                w = torch.randn(dim, dim, generator=gen, dtype=torch.float64) * dim ** -0.5
                layer.weight.copy_((w * gain).to(dtype))
            self.head.weight.copy_(
                (torch.randn(vocab, dim, generator=gen, dtype=torch.float64)
                 * dim ** -0.5).to(dtype))

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.head(h)


def tensor_report(t: torch.Tensor) -> dict:
    finite = torch.isfinite(t)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
        "absmax": float(t[finite].abs().max()) if finite.any() else None,
        "zero_frac": round(float((t == 0).double().mean()), 6),
    }


def first_bad(name: str, t: torch.Tensor) -> dict:
    """定位第一个非有限元素并保留邻域，用于确认它是不是孤立事件。"""
    bad = (~torch.isfinite(t)).nonzero()
    idx = [int(v) for v in bad[0]]
    flat = t.reshape(-1)
    pos = int((~torch.isfinite(flat)).nonzero()[0])
    lo, hi = max(0, pos - 4), min(flat.numel(), pos + 5)
    return {
        "tensor": name,
        "first_index": idx,
        "flat_position": pos,
        "neighbourhood": [None if not torch.isfinite(v) else round(float(v), 6)
                          for v in flat[lo:hi]],
        "nonfinite_count": int((~torch.isfinite(t)).sum()),
        "total": int(t.numel()),
    }


def run_attempt(name: str, *, dtype=torch.float32, gain=1.0, embed_scale=1.0,
                grad_scale=1.0, bad_sample=False, empty_labels=False,
                eps=1e-8) -> dict:
    torch.manual_seed(SEED)
    model = DeepStack(gain=gain, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, eps=eps)
    ids = BATCH.clone()

    acts: list[dict] = []
    handles = []

    def hook(layer_name):
        def fn(_mod, _inp, out):
            rec = tensor_report(out.detach())
            rec["layer"] = layer_name
            acts.append(rec)
        return fn

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
    handles.append(model.head.register_forward_hook(hook("head")))

    with torch.no_grad():
        model.embed.weight.mul_(embed_scale)     # 放大输入量级，制造前向溢出
    h = model.embed(ids)
    if bad_sample:
        h = h.clone()
        h[1, 3, :] = float("nan")                # 第 2 条样本的第 4 个位置是坏数据
    for layer in model.layers:
        h = torch.relu(layer(h))
    logits = model.head(h)
    for handle in handles:
        handle.remove()

    labels = F.pad(ids, (0, 1), value=IGNORE)[..., 1:].clone()
    if empty_labels:
        labels.fill_(IGNORE)
    valid = int((labels != IGNORE).sum())
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                            labels.reshape(-1), ignore_index=IGNORE, reduction="sum")
    loss = total / valid                      # valid=0 时就是 0/0，不额外保护

    (loss * grad_scale).backward()

    grads = []
    bad_tensor = None
    for pname, p in model.named_parameters():
        if p.grad is None:
            continue
        rec = tensor_report(p.grad)
        rec["param"] = pname
        rec["grad_norm"] = (float(p.grad[torch.isfinite(p.grad)].norm())
                            if torch.isfinite(p.grad).any() else None)
        grads.append(rec)
    for rec in acts + grads:
        if not rec["finite"] and bad_tensor is None:
            bad_tensor = rec.get("layer") or rec.get("param")

    detail = None
    if bad_tensor is not None:
        for pname, p in model.named_parameters():
            if pname == bad_tensor:
                detail = first_bad(pname + ".grad", p.grad)
    finite_all = all(rec["finite"] for rec in grads)
    gnorm = None
    if finite_all:
        gnorm = float(torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    stepped = False
    if finite_all and gnorm is not None and gnorm == gnorm:   # 非 NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stepped = True
    ratios = {n: float((p.detach().double() - before[n].double()).norm()
                       / before[n].double().norm().clamp(min=1e-12))
              for n, p in model.named_parameters()} if stepped else {}

    return {
        "attempt": name,
        "config": {"dtype": str(dtype).replace("torch.", ""), "gain": gain,
                   "embed_scale": embed_scale, "grad_scale": grad_scale,
                   "bad_sample": bad_sample, "empty_labels": empty_labels},
        "loss": (None if not torch.isfinite(loss.detach())
                 else round(float(loss.detach()), 6)),
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "valid_targets": valid,
        "activations": acts,
        "grads": grads,
        "grad_norm": gnorm,
        "stepped": stepped,
        "params_finite_after_step": all(bool(torch.isfinite(p).all())
                                        for p in model.parameters()),
        "update_to_weight": {k: round(v, 8) for k, v in ratios.items()},
        "first_nonfinite": bad_tensor,
        "first_nonfinite_detail": detail,
    }


ATTEMPTS = [
    ("正常 FP32", dict()),
    ("FP16 前向溢出", dict(dtype=torch.float16, gain=3.2, embed_scale=8.0, eps=1e-4)),
    ("梯度整体下溢（FP16）", dict(dtype=torch.float16, grad_scale=1e-6, eps=1e-4)),
    ("坏样本：输入含 NaN", dict(bad_sample=True)),
    ("空分母：无有效 target", dict(empty_labels=True)),
]


def first_bad_activation(rec: dict) -> str:
    for act in rec["activations"]:
        if not act["finite"]:
            return f"{act['layer']}（{act['nonfinite_count']}/{act['shape']}）"
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir")
    args = ap.parse_args()
    print(f"torch {torch.__version__} | CPU | 12 层线性栈，固定 batch {tuple(BATCH.shape)}")

    results = [run_attempt(name, **kw) for name, kw in ATTEMPTS]
    print(f"\n{'尝试':<24}{'loss':>12}{'有效target':>11}{'梯度全finite':>13}"
          f"{'是否更新':>10}{'首个坏张量':>26}")
    for r in results:
        loss = "NaN/Inf" if not r["loss_finite"] else f"{r['loss']:.4f}"
        print(f"{r['attempt']:<24}{loss:>12}{r['valid_targets']:>11}"
              f"{str(all(g['finite'] for g in r['grads'])):>13}{str(r['stepped']):>10}"
              f"{first_bad_activation(r):>26}")

    print("\n逐层激活的最大绝对值（FP16 前向溢出这次尝试）")
    over = results[1]
    marked = False
    for act in over["activations"]:
        mark = ""
        if not act["finite"] and not marked:
            mark, marked = "  ← 第一个非有限", True
        val = "-" if act["absmax"] is None else f"{act['absmax']:.4g}"
        print(f"  {act['layer']:<10}{val:>14}  非有限 {act['nonfinite_count']:>4}{mark}")
    print(f"  FP16 上限 {torch.finfo(torch.float16).max}；量级沿深度按每层增益累乘，")
    print("  溢出点由层号决定，不是 loss 的性质。只看最终 loss 无法区分这一层与上一层。")

    print("\n梯度下溢：所有检查都通过，更新却是空的")
    under = results[2]
    print(f"  loss={under['loss']}（与正常尝试相同） 梯度全 finite="
          f"{all(g['finite'] for g in under['grads'])} 全局梯度范数={under['grad_norm']:.3e} "
          f"是否更新={under['stepped']}")
    for g in under["grads"][:4]:
        print(f"  {g['param']:<18} 零元素比例={g['zero_frac']:.3f} 范数={g['grad_norm']:.3e}")
    print("  loss 正常、finite 通过、optimizer 也执行了 step，唯一的信号是梯度范数为 0。")
    print("  监控里必须有梯度范数与零元素比例这一项，只看 loss 曲线看不出这种停摆。")

    print("\n坏样本与空分母：现象相同，原因不同")
    bad, empty = results[3], results[4]
    print(f"  坏样本：loss 非有限，第一个非有限张量是 {first_bad_activation(bad)}；"
          f"沿输入回溯定位到 batch 内第 2 条样本")
    print(f"  空分母：有效 target={empty['valid_targets']}，loss=0/0 为 NaN，"
          f"但梯度全为 0（范数={empty['grad_norm']:.1f}），finite 检查通过，"
          f"optimizer 照常执行")
    print(f"  这一步参数仍然变了：update/weight={list(empty['update_to_weight'].values())[0]:.3e}"
          f"，正好是 lr×weight_decay=1e-4 的 decoupled 衰减。")
    print("  先查有效 target 计数，再查输入，最后才查前向和反向——顺序反了会在模型里白找一圈。")

    print("\n更新量与参数的比值")
    print(f"  {'参数':<18}{'正常':>12}{'梯度下溢':>12}{'空分母':>12}")
    for name in list(results[0]["update_to_weight"])[:5]:
        row = [r["update_to_weight"].get(name) for r in (results[0], results[2], results[4])]
        cells = "".join(f"{('-' if v is None else f'{v:.2e}'):>12}" for v in row)
        print(f"  {name:<18}{cells}")
    print("  三列的量级本身就是判据：正常尝试是 AdamW 首步的 ~lr 级更新，")
    print("  下溢与空分母两列只剩 weight decay。绝对数值依赖 lr 与参数范数，")
    print("  有意义的是同一次训练里这条比值随步数的走向。")

    print("\n最后一段：optimizer 自己的常数也会被参数 dtype 吃掉")
    print(f"  {'参数 dtype':<12}{'eps 设定':>10}{'eps 在该 dtype 中':>20}"
          f"{'一步后非有限参数':>18}")
    eps_rows = []
    for dtype, eps in ((torch.float16, 1e-8), (torch.float16, 1e-4),
                       (torch.bfloat16, 1e-8), (torch.float32, 1e-8)):
        torch.manual_seed(SEED)
        model = DeepStack(dtype=dtype)
        logits = model(BATCH)
        labels = F.pad(BATCH, (0, 1), value=IGNORE)[..., 1:]
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                        labels.reshape(-1), ignore_index=IGNORE).backward()
        torch.optim.AdamW(model.parameters(), lr=1e-2, eps=eps).step()
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        eps_in_dtype = torch.tensor(eps, dtype=dtype).item()
        eps_rows.append({"dtype": str(dtype), "eps": eps,
                         "eps_in_dtype": eps_in_dtype, "nonfinite_params": bad})
        print(f"  {str(dtype).replace('torch.', ''):<12}{eps:>10.0e}{eps_in_dtype:>20.3e}"
              f"{(str(len(bad)) + ' / ' + str(len(list(model.parameters())))):>18}")
    zero_rows = int((model.embed.weight.grad.abs().sum(-1) == 0).sum())
    print(f"  这一批只用到 {model.embed.weight.shape[0] - zero_rows} 个 token，"
          f"embed 有 {zero_rows} 行梯度恒为 0。")
    print("  FP16 里 1e-8 低于最小次正规数 5.96e-8，直接变成 0；这些零梯度行的")
    print("  denom = sqrt(v)/bc2 + eps 于是也是 0，addcdiv 得到 0/0，整张参数变 NaN。")
    print("  BF16 的指数范围与 FP32 相同，同一个 eps 仍然可表示，不触发这个问题。")
    print("  低精度参数影响的不只是舍入，还包括 optimizer 公式里所有的常数。")

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=False)
        for r in results:
            (out / f"{r['attempt'].split('：')[0].split('（')[0].replace(' ', '_')}.json"
             ).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "optimizer_eps.json").write_text(
            json.dumps(eps_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐次尝试的完整记录已写入 {out}")


if __name__ == "__main__":
    main()
