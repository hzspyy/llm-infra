#!/usr/bin/env python3
"""优化器状态与调度的 FP64 参照：手写更新与 PyTorch 逐步对拍。

五段内容：
  A 状态与公式   SGD / momentum / nesterov / Adam(L2) / AdamW 三步逐项对拍
  B eps 的位置   非 capturable 与 capturable 两条路径的 denom 写法差异
  C 参数组       decay / no-decay 划分的实际作用范围与错划的后果
  D 调度         warmup+cosine 与 WSD，按 step 与按 token 的两种时钟
  E EMA          decay 与 dtype 决定 EMA 能不能真的跟上参数

Usage:
    python labs/L7/optimizer_state_reference.py > "$RUN_DIR/optimizer.txt"
"""
from __future__ import annotations

import math

import torch

from _tinylm import TinyLM

torch.manual_seed(0)
FP64 = torch.float64


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- A 状态与公式
GRADS = [0.8, -0.3, 0.05]          # 固定的三步梯度序列，避免依赖某个模型的前向
P0, LR, WD = 1.0, 0.1, 0.1
B1, B2, EPS = 0.9, 0.999, 1e-8


def manual_sgd(momentum: float, nesterov: bool):
    p, buf = P0, None
    rows = []
    for t, g in enumerate(GRADS, 1):
        d = g
        if momentum:
            buf = d if buf is None else momentum * buf + d
            d = d + momentum * buf if nesterov else buf
        p -= LR * d
        rows.append((t, buf, d, p))
    return rows


def manual_adam(decoupled: bool):
    """decoupled=False 复现 Adam 的 L2（把 wd*p 加进梯度），True 复现 AdamW。"""
    p, m, v = P0, 0.0, 0.0
    rows = []
    for t, g in enumerate(GRADS, 1):
        if decoupled:
            p *= 1 - LR * WD
        else:
            g = g + WD * p
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
        denom = math.sqrt(v) / math.sqrt(bc2) + EPS
        step = (LR / bc1) * m / denom
        p -= step
        rows.append((t, m, v, bc1, bc2, step, p))
    return rows


def torch_run(make_opt):
    p = torch.tensor([P0], dtype=FP64, requires_grad=True)
    opt = make_opt([p])
    out = []
    for g in GRADS:
        p.grad = torch.tensor([g], dtype=FP64)
        opt.step()
        state = opt.state[p]
        out.append((p.item(), {k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
                               for k, v in state.items()}))
    return out


def section_a() -> None:
    head("A 一维 FP64：五种更新规则的状态与参数轨迹")
    print(f"p0={P0} lr={LR} weight_decay={WD} betas=({B1},{B2}) eps={EPS}")
    print(f"梯度序列 {GRADS}（固定输入，与任何模型无关）\n")

    for name, make_opt, manual in [
        ("SGD", lambda ps: torch.optim.SGD(ps, lr=LR), manual_sgd(0.0, False)),
        ("SGD+momentum0.9", lambda ps: torch.optim.SGD(ps, lr=LR, momentum=0.9),
         manual_sgd(0.9, False)),
        ("SGD+nesterov0.9", lambda ps: torch.optim.SGD(ps, lr=LR, momentum=0.9, nesterov=True),
         manual_sgd(0.9, True)),
    ]:
        ref = torch_run(make_opt)
        print(f"[{name}] 手写与 torch 的参数轨迹")
        for (t, buf, d, p_manual), (p_torch, state) in zip(manual, ref):
            buf_txt = "-" if buf is None else f"{buf:+.12f}"
            print(f"  step {t}: buf={buf_txt} 实际更新方向={d:+.12f} "
                  f"p_manual={p_manual:.15f} p_torch={p_torch:.15f} "
                  f"diff={abs(p_manual - p_torch):.3e}")
        keys = sorted(ref[-1][1])
        print(f"  torch 状态字段：{keys or '无（普通 SGD 不存状态）'}")

    for name, decoupled, make_opt in [
        ("Adam(L2 weight_decay)", False,
         lambda ps: torch.optim.Adam(ps, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)),
        ("AdamW(decoupled)", True,
         lambda ps: torch.optim.AdamW(ps, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)),
    ]:
        manual, ref = manual_adam(decoupled), torch_run(make_opt)
        print(f"\n[{name}] m/v/bias correction 与参数")
        for (t, m, v, bc1, bc2, step, p_manual), (p_torch, state) in zip(manual, ref):
            print(f"  step {t}: m={m:+.12f} v={v:.12e} bc1={bc1:.6f} bc2={bc2:.9f} "
                  f"更新量={step:+.12f}")
            print(f"           p_manual={p_manual:.15f} p_torch={p_torch:.15f} "
                  f"diff={abs(p_manual - p_torch):.3e} | "
                  f"torch exp_avg={state['exp_avg']:+.12f} exp_avg_sq={state['exp_avg_sq']:.12e} "
                  f"step={state['step']}")
    print("\nv 是梯度平方的指数移动平均，不减均值；两种 weight decay 的差别在第一步就出现：")
    l2, w = manual_adam(False)[-1][-1], manual_adam(True)[-1][-1]
    print(f"  三步后 Adam(L2)={l2:.15f}  AdamW={w:.15f}  差={abs(l2 - w):.6e}")

    p = torch.tensor([P0], dtype=FP64, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=LR)
    p.grad = torch.tensor([GRADS[0]], dtype=FP64)
    opt.step()
    st = opt.state[p]
    print(f"\ntorch 的 step 计数是张量：dtype={st['step'].dtype} device={st['step'].device} "
          f"（非 fused/capturable 时刻意留在 CPU）")
    print(f"m/v 的 dtype 跟随参数：{st['exp_avg'].dtype}，"
          "由 zeros_like(p) 决定，不是数学上要求 FP32")


# ---------------------------------------------------------- B 同一公式的多条实现
def run_adamw(dtype, foreach: bool, steps: int = 20, seed: int = 0):
    """同一份初值和梯度序列，走不同的 AdamW 实现路径。"""
    gen = torch.Generator().manual_seed(seed)
    init = torch.randn(4096, generator=gen, dtype=torch.float64) * 0.1
    grads = [torch.randn(4096, generator=gen, dtype=torch.float64) * 0.05
             for _ in range(steps)]
    p = init.to(dtype).clone().requires_grad_(True)
    opt = torch.optim.AdamW([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD,
                            foreach=foreach)
    for g in grads:
        p.grad = g.to(dtype)
        opt.step()
    return p.detach()


def section_b() -> None:
    head("B 同一个 AdamW 公式，不同实现路径的浮点结果")
    ref = run_adamw(torch.float64, foreach=False)
    runs = {}
    for label, dtype, foreach in [
        ("FP64 foreach=True ", torch.float64, True),
        ("FP32 foreach=False", torch.float32, False),
        ("FP32 foreach=True ", torch.float32, True),
    ]:
        got = run_adamw(dtype, foreach).to(torch.float64)
        runs[label.strip()] = got
        diff = (got - ref).abs()
        print(f"  {label} 与 FP64 单张量路径：最大绝对差={diff.max():.3e} "
              f"最大相对差={(diff / ref.abs().clamp(min=1e-12)).max():.3e}")
    pair = (runs["FP32 foreach=False"] - runs["FP32 foreach=True"]).abs().max()
    print(f"  两条 FP32 路径之间：最大差={pair:.3e}")
    print("20 步之后，4096 个参数里 FP32 与 FP64 的最大相对差已到 1e-4——dtype 才是主导项。")
    print("foreach 只改内存访问与 kernel 数量，本例中两条 FP32 路径逐位一致；")
    print("换设备、换 fused 实现或改归约顺序时这一点需要重新验证，不能假设。")

    m, v, t = 0.08, 6.4e-4, 1
    bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
    step_size = LR / bc1
    print("\neps 在两条源码路径里的位置不同（capturable 把它折进 step_size）：")
    for eps in (1e-8, 1e-3, 1e-1):
        plain = step_size * m / (math.sqrt(v) / math.sqrt(bc2) + eps)
        denom_cap = math.sqrt(v) / (math.sqrt(bc2) * -step_size) + eps / -step_size
        cap = -m / denom_cap
        print(f"  eps={eps:<8} 非capturable={plain:.15f} capturable={cap:.15f} "
              f"相对差={abs(plain - cap) / abs(plain):.3e}")
    print("两种写法在实数域等价，FP64 下的差停在舍入量级；真正会改变结果的是 eps 本身的大小，")
    print(f"eps 从 1e-8 增到 1e-1，这一步的更新量少了 "
          f"{(1 - (step_size * m / (math.sqrt(v) / math.sqrt(bc2) + 1e-1)) / (step_size * m / (math.sqrt(v) / math.sqrt(bc2) + 1e-8))) * 100:.1f}%。")


# ---------------------------------------------------------------- C 参数组
def no_decay(name: str) -> bool:
    return name.endswith("bias") or "norm" in name or name == "embed.weight"


def section_c() -> None:
    head("C 参数组：weight_decay 实际作用在哪些张量上")
    model = TinyLM(vocab=16, dim=8, seed=0)
    decay = [(n, p) for n, p in model.named_parameters() if not no_decay(n)]
    plain = [(n, p) for n, p in model.named_parameters() if no_decay(n)]
    print(f"decay 组（{sum(p.numel() for _, p in decay)} 参数）：{[n for n, _ in decay]}")
    print(f"no-decay 组（{sum(p.numel() for _, p in plain)} 参数）：{[n for n, _ in plain]}")

    opt = torch.optim.AdamW([
        {"params": [p for _, p in decay], "weight_decay": 0.1},
        {"params": [p for _, p in plain], "weight_decay": 0.0, "lr": LR * 0.5},
    ], lr=LR)
    for g_i, g in enumerate(opt.param_groups):
        print(f"  group{g_i}: lr={g['lr']} weight_decay={g['weight_decay']} "
              f"张量数={len(g['params'])}")

    # 只有 decay 的效果：给零梯度，观察 decoupled decay 单独造成的参数变化
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    print("\n梯度全零时，只有 decoupled weight decay 会改参数：")
    for n, p in model.named_parameters():
        delta = (p - before[n]).abs().max().item()
        print(f"  {n:<18} 最大变化={delta:.6e} {'(no-decay 组)' if no_decay(n) else ''}")

    print("\n反例：整模型一组、统一 weight_decay=0.1，零梯度下 RMSNorm 的 weight 被直接拉低")
    model2 = TinyLM(vocab=16, dim=8, seed=0)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=0.1)
    for p in model2.parameters():
        p.grad = torch.zeros_like(p)
    w_before = model2.final_norm.weight.detach().clone()
    for _ in range(10):
        opt2.step()
    w_after = model2.final_norm.weight.detach()
    print(f"  final_norm.weight: 起点={w_before[0].item():.6f} "
          f"10 步零梯度后={w_after[0].item():.6f} "
          f"（每步乘 1-lr*wd={1 - LR * 0.1:.3f}）")
    print("  RMSNorm 的 scale 与 bias 没有\"越小越好\"的先验，衰减它们等于在改模型结构的先验。")


# ---------------------------------------------------------------- D 学习率调度
def lr_warmup_cosine(step: int, total: int, warmup: int, base: float, final_frac: float):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def lr_wsd(step: int, total: int, warmup: int, base: float, decay_frac: float):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return base * (step + 1) / warmup
    if step < decay_start:
        return base
    return base * (1 - (step - decay_start) / max(1, total - decay_start))


def section_d() -> None:
    head("D 学习率调度：两种形状，两种时钟")
    total, warmup, base = 100, 10, 3e-4
    print(f"total_steps={total} warmup={warmup} base_lr={base}")
    print(f"{'step':>5} {'warmup+cosine':>16} {'WSD(尾部20%衰减)':>20}")
    for s in (0, 4, 9, 10, 25, 50, 79, 80, 90, 99):
        print(f"{s:>5} {lr_warmup_cosine(s, total, warmup, base, 0.1):>16.8f} "
              f"{lr_wsd(s, total, warmup, base, 0.2):>20.8f}")
    print("cosine 需要在开始前知道 total；WSD 的稳定段可以延长后再决定何时进入衰减。")

    print("\n同一份配方按 token 计时与按 step 计时：")
    gbs_tokens = 2_048_000
    budget = 8e12
    steps = int(budget / gbs_tokens)
    print(f"  token 预算 {budget:.1e}，global batch {gbs_tokens} token → {steps} 次更新")
    print(f"  若 micro_batch 或 DP 变化而 global batch 不变，step 数不变；"
          f"  global batch 减半则 step 数翻倍，同一条 cosine 曲线被拉长一倍")

    print("\n反例：scheduler 跟着 microbatch 走（accumulation=4）")
    acc = 4
    wrong = [lr_warmup_cosine(min(s * acc, total - 1), total, warmup, base, 0.1)
             for s in range(6)]
    right = [lr_warmup_cosine(s, total, warmup, base, 0.1) for s in range(6)]
    print(f"  前 6 次更新的 LR，正确={['%.2e' % x for x in right]}")
    print(f"                   错误={['%.2e' % x for x in wrong]}")
    print("  错误时钟让 warmup 提前 4 倍结束，整条曲线在 1/4 的训练量内跑完。")


# ---------------------------------------------------------------- E EMA
def section_e() -> None:
    head("E EMA：decay 与缓冲区 dtype 共同决定它能不能跟上参数")
    print("参数固定在 FP32 且每步确定地下降 1e-3，只改 EMA 缓冲区的 dtype：")
    p = torch.tensor([1.0], dtype=torch.float32)
    ema32 = p.clone()
    ema_bf = p.to(torch.bfloat16).clone()
    marks = {1, 100, 500, 1000, 1500, 1950, 2000}
    for step in range(1, 2001):
        p = p - 1e-3
        ema32 = ema32 * 0.999 + p * 0.001
        ema_bf = (ema_bf.float() * 0.999 + p * 0.001).to(torch.bfloat16)
        if step in marks:
            print(f"  step {step:>4}: p={p.item():+.6f} ema_fp32={ema32.item():+.6f} "
                  f"ema_bf16={ema_bf.float().item():+.6f}")
    print("FP32 的 EMA 滞后但一直在动；BF16 缓冲区在开头长期停在 1.0——")
    print("每步增量 0.001×(p−ema) 小于 1.0 附近 BF16 的半 ULP(0.00195)，被整体舍掉，")
    print("直到 p 与 ema 的差拉开到约 2 才开始移动。EMA 的 dtype 要与 decay、参数量级一起判断。")

    print("\ndecay warmup（ema_pytorch 风格）让早期的 EMA 不至于一直贴着初值：")
    ema32 = torch.tensor([1.0], dtype=torch.float32)
    p32 = torch.tensor([1.0], dtype=torch.float32)
    for step in range(1, 6):
        p32 = p32 - 1e-3
        d = min(0.999, (1 + step) / (10 + step))   # ema_pytorch 风格的 decay warmup
        ema32 = ema32 * d + p32 * (1 - d)
        print(f"  step {step}: decay={d:.4f} p={p32.item():.6f} ema={ema32.item():.6f}")
    print("EMA 在 optimizer.step 之后更新；它是另一份权重，导出、评测和恢复都要单独记账。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | 默认 dtype {torch.get_default_dtype()} | CPU")
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
