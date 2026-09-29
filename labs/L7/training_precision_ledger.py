#!/usr/bin/env python3
"""混合精度的状态账：一次真实更新之后，逐张量打印 dtype 与字节。

五种配置各跑一次完整更新（前向 → 反向 → optimizer.step），记录
  · 参数 / 梯度 / m / v / step 的实际 dtype 与字节
  · autograd 为反向保存了哪些张量、各是什么 dtype（saved_tensors_hooks 抓取）
  · 前向输出与 loss 的 dtype
再按全参 / LoRA / QLoRA / QAT / KD 五种训练方法列常驻状态，
最后打印 FSDP2 MixedPrecisionPolicy 的三个 dtype 字段。
按每参数字节的外推不等于显存实测。

Usage:
    python labs/L7/training_precision_ledger.py > "$RUN_DIR/ledger.txt"
"""
from __future__ import annotations

import dataclasses
from collections import Counter

import torch
import torch.nn.functional as F

from _tinylm import TinyLM

VOCAB, DIM, SEED = 64, 32, 0
BATCH = torch.tensor([[5, 9, 3, 7, 2, 1, 8, 4],
                      [4, 11, 6, 13, 12, 1, 0, 0]])
ATTN = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 0, 0]])
IGNORE = -100
PARAM_3B = 3_086_000_000     # SmolLM3-3B 量级，仅用于按每参数字节的外推


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build(dtype):
    torch.manual_seed(SEED)
    return TinyLM(vocab=VOCAB, dim=DIM, seed=SEED, dtype=dtype)


def loss_of(model, autocast_dtype=None):
    ctx = (torch.autocast("cpu", dtype=autocast_dtype)
           if autocast_dtype is not None else torch.autocast("cpu", enabled=False))
    with ctx:
        logits = model(BATCH, attention_mask=ATTN)
        targets = F.pad(torch.where(ATTN.bool(), BATCH, torch.full_like(BATCH, IGNORE)),
                        (0, 1), value=IGNORE)[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               targets.reshape(-1), ignore_index=IGNORE)
    return logits, loss


class SavedTensorRecorder:
    """记录 autograd 为反向保存的张量，按 dtype 汇总字节。"""

    def __init__(self):
        self.by_dtype: Counter = Counter()
        self.bytes_by_dtype: Counter = Counter()
        self._seen: set[int] = set()

    def __enter__(self):
        def pack(t):
            key = t.data_ptr()
            if key not in self._seen:
                self._seen.add(key)
                self.by_dtype[str(t.dtype)] += 1
                self.bytes_by_dtype[str(t.dtype)] += t.numel() * t.element_size()
            return t

        self._hook = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hook.__enter__()
        return self

    def __exit__(self, *exc):
        return self._hook.__exit__(*exc)


def run_config(name: str, param_dtype, autocast_dtype=None, use_scaler=False,
               master_fp32=False):
    model = build(param_dtype)
    params = list(model.parameters())
    if master_fp32:
        master = [p.detach().float().clone().requires_grad_(False) for p in params]
        opt = torch.optim.AdamW(master, lr=1e-2)
    else:
        master = None
        opt = torch.optim.AdamW(params, lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=use_scaler) if use_scaler else None

    rec = SavedTensorRecorder()
    with rec:
        logits, loss = loss_of(model, autocast_dtype)
        (scaler.scale(loss) if scaler else loss).backward()
    if scaler:
        scaler.unscale_(opt)
    if master_fp32:
        for m_p, p in zip(master, params):
            m_p.grad = p.grad.float()
    if scaler:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    if master_fp32:
        with torch.no_grad():
            for m_p, p in zip(master, params):
                p.copy_(m_p)

    tracked = master if master_fp32 else params
    state = opt.state[tracked[0]]
    numel = sum(p.numel() for p in params)
    bytes_param = sum(p.numel() * p.element_size() for p in params)
    bytes_grad = sum(p.grad.numel() * p.grad.element_size() for p in params)
    bytes_master = (sum(p.numel() * p.element_size() for p in master)
                    if master_fp32 else 0)
    bytes_opt = sum(opt.state[p]["exp_avg"].numel() * opt.state[p]["exp_avg"].element_size()
                    + opt.state[p]["exp_avg_sq"].numel() * opt.state[p]["exp_avg_sq"].element_size()
                    for p in tracked)
    total = bytes_param + bytes_grad + bytes_master + bytes_opt
    return {
        "name": name,
        "param": str(params[0].dtype),
        "grad": str(params[0].grad.dtype),
        "master": str(master[0].dtype) if master_fp32 else "-",
        "m": str(state["exp_avg"].dtype),
        "v": str(state["exp_avg_sq"].dtype),
        "step": f'{state["step"].dtype}@{state["step"].device}',
        "logits": str(logits.dtype),
        "loss": str(loss.dtype),
        "numel": numel,
        "bytes_per_param": total / numel,
        "saved": dict(rec.bytes_by_dtype),
        "saved_count": dict(rec.by_dtype),
        "model": model,
        "opt": opt,
    }


CONFIGS = [
    ("FP32 参数与计算", dict(param_dtype=torch.float32)),
    ("FP32 参数 + autocast BF16", dict(param_dtype=torch.float32,
                                       autocast_dtype=torch.bfloat16)),
    ("FP32 参数 + autocast FP16 + scaler", dict(param_dtype=torch.float32,
                                                autocast_dtype=torch.float16,
                                                use_scaler=True)),
    ("BF16 参数 + 原生 AdamW", dict(param_dtype=torch.bfloat16)),
    ("BF16 参数 + FP32 master", dict(param_dtype=torch.bfloat16, master_fp32=True)),
]


def main() -> None:
    print(f"torch {torch.__version__} | CPU | TinyLM vocab={VOCAB} dim={DIM} "
          f"| 输入 shape={tuple(BATCH.shape)}")

    results = [run_config(name, **kw) for name, kw in CONFIGS]

    head("A 一次更新之后，各配置的实际 dtype")
    print(f"{'配置':<34}{'参数':>9}{'梯度':>9}{'master':>9}{'m':>9}{'v':>9}{'logits':>10}")
    for r in results:
        short = lambda s: s.replace("torch.float", "fp").replace("torch.bfloat", "bf")
        print(f"{r['name']:<34}{short(r['param']):>9}{short(r['grad']):>9}"
              f"{short(r['master']):>9}{short(r['m']):>9}{short(r['v']):>9}"
              f"{short(r['logits']):>10}")
    print(f"\nstep 计数器：{results[0]['step']}；loss 一律 {results[0]['loss']}（本 lab 显式 .float() 算 CE）")
    print("autocast 不转换持久参数：前两行的参数、梯度和 m/v 全是 FP32，只有算子输出变了。")

    head("B 常驻状态的字节账（参数+梯度+master+m+v，不含激活与 workspace）")
    print(f"{'配置':<34}{'字节/参数':>10}{'TinyLM 合计':>14}{'3.09B 外推':>14}")
    for r in results:
        print(f"{r['name']:<34}{r['bytes_per_param']:>10.1f}"
              f"{r['bytes_per_param'] * r['numel'] / 1024:>12.1f}KiB"
              f"{r['bytes_per_param'] * PARAM_3B / 2 ** 30:>12.2f}GiB")
    print("BF16 参数省的是常驻状态；FP32 master 把省下的又加了回来，换回 FP32 精度的写回。")
    print("3.09B 列是按每参数字节的算术外推，不是加载 3B 模型的显存实测。")

    head("C autograd 为反向保存了什么")
    for r in results[:4]:
        total = sum(r["saved"].values())
        detail = "  ".join(f"{k.replace('torch.', '')}×{r['saved_count'][k]}={v / 1024:.1f}KiB"
                           for k, v in sorted(r["saved"].items()))
        print(f"{r['name']:<34} 合计 {total / 1024:>7.1f} KiB | {detail}")
    print("autocast 行的保存值出现 BF16/FP16：省下来的是激活内存，")
    print("而参数、梯度与 optimizer 状态仍按上表的 dtype 常驻。")

    head("D 逐张量明细：FP32 参数 + autocast BF16")
    r = results[1]
    model, opt = r["model"], r["opt"]
    print(f"{'张量':<20}{'shape':>14}{'参数dtype':>12}{'梯度dtype':>12}{'m dtype':>12}{'字节':>10}")
    for n, p in model.named_parameters():
        st = opt.state[p]
        b = (p.numel() * p.element_size() + p.grad.numel() * p.grad.element_size()
             + st["exp_avg"].numel() * st["exp_avg"].element_size() * 2)
        print(f"{n:<20}{str(tuple(p.shape)):>14}{str(p.dtype).replace('torch.', ''):>12}"
              f"{str(p.grad.dtype).replace('torch.', ''):>12}"
              f"{str(st['exp_avg'].dtype).replace('torch.', ''):>12}{b:>10}")

    head("E 五种训练方法的状态账：谁被更新，谁只是常驻")
    dim = 1024
    base = torch.nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16)
    n_base = base.weight.numel()
    rank = 16
    n_adapter = 2 * rank * dim

    def report_method(name, trainable_numel, trainable_bytes_per, base_bytes_per,
                      extra=0, note=""):
        opt_state = trainable_numel * 8          # FP32 的 m 与 v
        grad = trainable_numel * trainable_bytes_per
        total = n_base * base_bytes_per + trainable_numel * trainable_bytes_per + grad \
            + opt_state + extra
        print(f"{name:<26}{trainable_numel:>12,}{total / 2 ** 20:>14.2f}"
              f"{total / n_base:>14.2f}  {note}")

    print(f"基座 Linear({dim},{dim}) 共 {n_base:,} 参数；LoRA rank={rank} 引入 "
          f"{n_adapter:,} 个可训练参数")
    print(f"{'方法':<26}{'可训练参数':>12}{'合计 MiB':>14}{'字节/基座参数':>14}")
    report_method("全参 BF16 + FP32 master", n_base, 2, 2, extra=n_base * 4,
                  note="master 4B + m/v 各 4B")
    report_method("LoRA（BF16 冻结基座）", n_adapter, 4, 2,
                  note="基座无梯度、无 optimizer 状态")
    # QLoRA 的基座按 bitsandbytes NF4 的公开定义推算：4 bit 数据 + 每 64 个元素
    # 一个 FP8 absmax + double quant 的每 256 块一个 FP32，共 4 + 8/64 + 32/256 bit。
    qlora_base_bytes = (4 + 8 / 64 + 32 / 256) / 8
    report_method("QLoRA（NF4 冻结基座）", n_adapter, 4, qlora_base_bytes,
                  note="基座 4.25 bit/参数，为公开格式定义的推算")
    report_method("QAT（BF16 + fake quant）", n_base, 2, 2,
                  extra=n_base * 4 + n_base // 64 * 4,
                  note="加 master 与每 64 元素一个 scale")
    report_method("KD 学生（teacher 常驻）", n_base, 2, 2,
                  extra=n_base * 4 + n_base * 2,
                  note="teacher 只前向：加一份 BF16 权重，无梯度无状态")
    print("LoRA 把梯度与 optimizer 状态限制在 adapter 上，省下的是这两项，不是基座本身；")
    print("QLoRA 再把常驻基座压到 4.25 bit，但反量化后的计算精度仍由 compute dtype 决定。")
    print("teacher 常驻的是权重与前向激活，它不产生梯度和 optimizer 状态；")
    print("若改为离线缓存 teacher logits，这笔开销从显存换成存储与带宽，见 7.7。")

    head("F FSDP2 的 MixedPrecisionPolicy 是三个独立的 dtype")
    from torch.distributed.fsdp import MixedPrecisionPolicy
    for f in dataclasses.fields(MixedPrecisionPolicy):
        print(f"  {f.name:<24} 默认={f.default}")
    print("  param_dtype 只作用于展开后的参数与计算，optimizer 看到的分片保持原 dtype；")
    print("  reduce_dtype 决定梯度归约用什么精度，与 param_dtype 可以不同；")
    print("  分片副本、短暂的完整参数、梯度归约缓冲要分开记账，实测见 7.2。")


if __name__ == "__main__":
    main()
