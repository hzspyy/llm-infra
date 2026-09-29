#!/usr/bin/env python3
"""损失缩放：官方 GradScaler 的状态机、次序反例与跳步时钟。

五段内容：
  A 为什么需要缩放   FP16 下溢/上溢边界与一个 2^k 缩放能救回多少梯度
  B 官方状态机       六次尝试的 scale、growth tracker、found_inf 与参数是否变化
  C 次序反例         clip 早于 unscale、窗口内 update、跳步仍推进 scheduler
  D step 的两条路径  普通 optimizer 与 fused optimizer 的 skip 契约
  E BF16            不用 scaler 不等于不会溢出

Usage:
    python labs/L7/loss_scaling_reference.py > "$RUN_DIR/scaler.txt"
"""
from __future__ import annotations

import torch


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------- A 为什么需要缩放
def section_a() -> None:
    head("A FP16 的两端：下溢与上溢，缩放能搬多远")
    info = torch.finfo(torch.float16)
    print(f"FP16 max={info.max} 最小正规数={info.smallest_normal:.3e} "
          f"最小正次正规数={torch.tensor(6e-8).half().item():.3e}")
    grads = [1e-3, 1e-5, 1e-7, 6e-8, 1e-9, 1e-12]
    print(f"\n{'真实梯度':>10}{'直接转FP16':>14}{'×2^16 后':>14}{'再 unscale':>16}{'结论':>10}")
    for g in grads:
        raw = torch.tensor(g, dtype=torch.float32).half()
        scaled = torch.tensor(g * 65536.0, dtype=torch.float32).half()
        back = scaled.float() / 65536.0
        verdict = "救回" if raw.item() == 0 and back.item() != 0 else (
            "本来就在" if raw.item() != 0 else "仍为零")
        print(f"{g:>10.0e}{raw.item():>14.3e}{scaled.item():>14.3e}{back.item():>16.3e}{verdict:>10}")
    print("\n上溢一侧：scale 过大时缩放后的梯度越过 65504 变 Inf。")
    for s in (2 ** 10, 2 ** 16, 2 ** 20):
        v = torch.tensor(2.0 * s, dtype=torch.float32).half()
        print(f"  梯度 2.0 × scale {s:>8} = {v.item()}")
    print("scale 取 2 的幂：乘除只改指数，不引入额外舍入。")


# ---------------------------------------------------------- B 官方状态机
def make_case():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    x = torch.randn(16, 8)
    return model, opt, x


def section_b() -> None:
    head("B 官方 GradScaler 的六次尝试：正常、Inf、正常、NaN、正常、正常")
    model, opt, x = make_case()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16, growth_factor=2.0,
                                  backoff_factor=0.5, growth_interval=2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    inject = {2: float("inf"), 4: float("nan")}
    updates = 0
    print(f"{'尝试':>4}{'进入时scale':>12}{'注入':>8}{'found_inf':>11}{'参数是否变':>12}"
          f"{'step返回':>10}{'update后scale':>14}{'有效更新数':>11}")
    for attempt in range(1, 7):
        before = model.weight.detach().clone()
        scale_in = scaler.get_scale()
        opt.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        scaler.scale(loss).backward()
        if attempt in inject:
            model.weight.grad[0, 0] = inject[attempt]
        scaler.unscale_(opt)
        finite = all(torch.isfinite(p.grad).all().item()
                     for p in model.parameters() if p.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                           error_if_nonfinite=True)
        ret = scaler.step(opt)
        scaler.update()
        if finite:
            sched.step()
            updates += 1
        moved = not torch.equal(before, model.weight.detach())
        tag = {2: "Inf", 4: "NaN"}.get(attempt, "-")
        print(f"{attempt:>4}{scale_in:>12.0f}{tag:>8}{str(not finite):>11}{str(moved):>12}"
              f"{str(ret):>10}{scaler.get_scale():>14.0f}{updates:>11}")
    print(f"\ngrowth_interval=2：连续 2 次成功后 scale 翻倍，任何一次非有限立即减半并清零计数。")
    print(f"结束时 scale={scaler.get_scale():.0f} growth_tracker={scaler._get_growth_tracker()} "
          f"scheduler.last_epoch={sched.last_epoch}（只在有效更新时前进）")
    print("step() 在成功与跳步时都返回 None，它不是成功标志。")


# ---------------------------------------------------------- C 次序反例
def section_c() -> None:
    head("C 三个次序反例")
    model, opt, x = make_case()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)
    loss = model(x).pow(2).mean()
    scaler.scale(loss).backward()
    scaled_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    scaler.unscale_(opt)
    real_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters()]).item()
    print("反例 1：在 unscale 之前 clip")
    print(f"  缩放梯度的范数={scaled_norm:.3f}，按 max_norm=1.0 裁剪后再 unscale，")
    print(f"  真实梯度范数只剩 {real_norm:.6e}——阈值实际被缩小了 {scaled_norm:.0f} 倍。")
    print("  正确次序是 unscale_ → 检查 finite → clip → step。")

    print("\n反例 2：梯度累积窗口内改 scale")
    def accumulate(change_scale: bool):
        model, opt, x = make_case()
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)
        opt.zero_grad(set_to_none=True)
        xs = [x, x.flip(0)]
        used = []
        for micro, xb in enumerate(xs):
            used.append(scaler.get_scale())
            scaler.scale(model(xb).pow(2).mean() / len(xs)).backward()
            if change_scale and micro == 0:      # 错误：窗口没结束就把 scale 翻倍
                scaler.update(scaler.get_scale() * 2)
        scaler.unscale_(opt)                     # 按窗口结束时的 scale 统一还原
        return model.weight.grad.detach().clone(), used

    good, used_good = accumulate(False)
    bad, used_bad = accumulate(True)
    rel = ((bad - good).abs().max() / good.abs().max()).item()
    print(f"  正确：两份 microbatch 都按 scale {used_good[0]:.0f} 缩放")
    print(f"  错误：第一份按 {used_bad[0]:.0f}，第二份按 {used_bad[1]:.0f}，unscale 只能用一个数")
    print(f"  还原后的梯度最大相对偏差 = {rel:.3f}（第一份 microbatch 的贡献被砍掉一半）")
    print("  scale 必须在一个累积窗口内保持不变，update() 只能在窗口结束后调用。")

    print("\n反例 3：跳步时 scheduler 照常前进")
    steps_total, skips = 100, 7
    print(f"  {steps_total} 次尝试里跳过 {skips} 次，按尝试计时的 scheduler 会比按有效更新计时的")
    print(f"  多走 {skips} 步；warmup 与 cosine 的位置都随之偏移，两份日志的 LR 曲线不可比。")
    print("  有效更新数要单独计数，它也是 checkpoint 里 global_step 的语义。")


# ---------------------------------------------------------- D 两条 step 路径
def section_d() -> None:
    head("D scaler.step 的两条路径")
    plain = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=0.1)
    print(f"  普通 AdamW: _step_supports_amp_scaling="
          f"{getattr(plain, '_step_supports_amp_scaling', False)}")
    try:
        fused = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=0.1, fused=True)
        flag = getattr(fused, "_step_supports_amp_scaling", False)
        print(f"  fused AdamW（CPU）: _step_supports_amp_scaling={flag}")
    except Exception as exc:
        print(f"  fused AdamW 在本机不可用：{type(exc).__name__}: {exc}")
    print("\n普通路径：scaler 在 host 上把 found_inf 求和成 Python 数，决定是否调用 optimizer.step。")
    print("fused 路径：scaler 把 grad_scale 与 found_inf 两个张量挂到 optimizer 上，")
    print("           由融合 kernel 自己 unscale 并在非有限时放弃写回，全程不回主机。")
    print("两条路径的跳步语义相同，但只有前者能用 Python 布尔值直接观察，")
    print("自定义训练循环判断是否发生有效更新时要按所用路径选择观测点。")


# ---------------------------------------------------------- E BF16
def section_e() -> None:
    head("E BF16 不用 scaler，不代表不会溢出")
    x = torch.full((4, 4), 1e20, dtype=torch.bfloat16)
    y = x @ x
    print(f"  1e20 的 4×4 矩阵自乘（BF16）：{y[0, 0].item()}，"
          f"BF16 max={torch.finfo(torch.bfloat16).max:.3e}")
    logits = torch.tensor([[0.0, 1e38]], dtype=torch.bfloat16)
    print(f"  softmax 前先减最大值可以避免 exp 溢出：{torch.softmax(logits.float(), -1).tolist()}")
    p = torch.tensor([float("nan")], dtype=torch.bfloat16)
    print(f"  BF16 同样有 Inf/NaN 编码：isnan={torch.isnan(p).item()}")
    print("\nBF16 的指数范围与 FP32 相同，所以通常不需要损失缩放；")
    print("但 attention 分数、重建损失、对抗项、累加器仍可能越界，")
    print("finite 检查与首个坏张量定位不能因为换成 BF16 就省掉。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU")
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
