#!/usr/bin/env python3
"""FP8 / MXFP8 / NVFP4 的格式、缩放与三类 GEMM 的 CPU 参照。

七段内容：
  A 格式          E4M3 / E5M2 / E2M1 / E8M0 的范围、间距与饱和行为
  B 缩放粒度      per-tensor / rowwise / MX 32 元素块 / NVFP4 16 元素两级
  C 转置          块缩放在什么条件下不随转置保持，必须量化两次
  D 三类 GEMM     前向、dX、dW 各自的量化轴与累加精度
  E 随机舍入      重复量化累加下 round-to-nearest 的系统偏差
  F Hadamard      16×16 变换摊平块内尖峰之后，误差在各量级之间怎么重新分配
  G 字节账        数据位宽之外，scale 本身占多少

量化按 torchao 的实际公式实现（amax→scale、power-of-2 floor、E8M0 指数、
NVFP4 两级 scale），GEMM 用 FP32 累加模拟，不代表真实 tensor core 的分块累加。

Usage:
    python labs/L7/low_precision_formats.py > "$RUN_DIR/formats.txt"
"""
from __future__ import annotations

import math

import torch

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
F4_MAX, F8E4M3_MAX_POW2, F4_MAX_POW2 = 6.0, 8, 2
EPS = 1e-12


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """信号与量化噪声比，越大越好；20 dB 约等于 10% 的相对误差。"""
    err = (ref - got).pow(2).sum()
    return float("inf") if err == 0 else 10 * math.log10((ref.pow(2).sum() / err).item())


# ---------------------------------------------------------------- 量化原语
def amax_to_scale(amax: torch.Tensor, fp8_dtype, pow2: bool = False) -> torch.Tensor:
    """torchao float8_utils.amax_to_scale：scale = finfo.max / clamp(amax, EPS)。"""
    scale = torch.finfo(fp8_dtype).max / amax.double().clamp(min=EPS)
    if pow2:
        scale = torch.exp2(torch.floor(torch.log2(scale)))
    return scale


def quantize_fp8(x: torch.Tensor, fp8_dtype, dim=None, pow2: bool = False):
    """按 per-tensor（dim=None）或沿 dim 的 rowwise 缩放量化，返回反量化结果。"""
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = amax_to_scale(amax, fp8_dtype, pow2)
    q = (x.double() * scale).to(torch.float32).to(fp8_dtype)
    return q.float().double() / scale


def quantize_mx(x: torch.Tensor, elem_dtype, block: int = 32):
    """MX：每 block 个连续元素共享一个 E8M0（2 的整数幂）scale。

    torchao mx_tensor.to_mx 的 FLOOR 模式：
        X = 2^(floor(log2(amax)) - target_max_pow2)
    target_max_pow2 是元素格式里最大的 2 的幂（E4M3 是 8 即 256，E2M1 是 2 即 4）。
    """
    shape = x.shape
    blocks = x.reshape(-1, block).double()
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exp = torch.floor(torch.log2(amax.clamp(min=1e-300)))
    target = F8E4M3_MAX_POW2 if elem_dtype == "e4m3" else F4_MAX_POW2
    scale_exp = torch.clamp(exp - target, min=-127, max=128)
    scale = torch.exp2(scale_exp)                       # E8M0：只存指数
    scaled = blocks / scale
    if elem_dtype == "e4m3":
        deq = scaled.to(torch.float32).to(E4M3).float().double()
    else:
        deq = round_e2m1(scaled)
    return (deq * scale).reshape(shape), scale.numel()


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """E2M1 只有 8 个非负可表示值，按最近邻舍入（ties 取偶数下标）。"""
    sign = torch.sign(x)
    mag = x.abs().clamp(max=F4_MAX)
    levels = E2M1_LEVELS.to(x.device)
    idx = torch.bucketize(mag, levels)
    lo = levels[(idx - 1).clamp(min=0)]
    hi = levels[idx.clamp(max=len(levels) - 1)]
    take_hi = (hi - mag) < (mag - lo)
    tie = (hi - mag) == (mag - lo)
    take_hi = take_hi | (tie & ((idx.clamp(max=len(levels) - 1)) % 2 == 0))
    return sign * torch.where(take_hi, hi, lo)


def quantize_nvfp4(x: torch.Tensor, block: int = 16, two_level: bool = True):
    """NVFP4：每 16 个元素一个 E4M3 块 scale，可选再套一个 per-tensor FP32 scale。"""
    shape = x.shape
    blocks = x.reshape(-1, block).double()
    block_scale = blocks.abs().amax(dim=-1, keepdim=True) / F4_MAX
    if two_level:
        per_tensor = block_scale.amax().clamp(min=EPS)
        s_fp8 = (block_scale / per_tensor).clamp(min=2 ** -9, max=448.0)
        s_fp8 = s_fp8.to(torch.float32).to(E4M3).float().double()
        total = s_fp8 * per_tensor
    else:
        total = block_scale.clamp(min=2 ** -9, max=448.0)
        total = total.to(torch.float32).to(E4M3).float().double()
    scaled = blocks / total.clamp(min=1e-300)
    deq = round_e2m1(scaled) * total
    return deq.reshape(shape), block_scale.numel()


def make_tensor(rows: int = 128, cols: int = 256, outlier: float = 40.0, seed: int = 0):
    """典型的激活分布：正态主体加少量离群点。"""
    gen = torch.Generator().manual_seed(seed)
    t = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    t[0, 0] = outlier
    t[7, 19] = -outlier * 0.8
    return t


# ---------------------------------------------------------------- A 格式
def section_a() -> None:
    head("A 四种低位宽格式")
    print(f"{'格式':<14}{'位宽':>6}{'max':>12}{'最小正规':>12}{'最小次正规':>12}"
          f"{'1 附近间距':>12}{'非零取值数':>12}")
    for name, dt in (("FP8 E4M3", E4M3), ("FP8 E5M2", E5M2)):
        info = torch.finfo(dt)
        vals = torch.arange(0, 256, dtype=torch.uint8).view(dt).float()
        finite = vals[torch.isfinite(vals)]
        print(f"{name:<14}{8:>6}{info.max:>12.1f}{info.smallest_normal:>12.3e}"
              f"{finite[finite > 0].min():>12.3e}{info.eps:>12.4f}"
              f"{len(torch.unique(finite)):>12}")
    print(f"{'FP4 E2M1':<14}{4:>6}{F4_MAX:>12.1f}{1.0:>12.3e}{0.5:>12.3e}"
          f"{0.5:>12.4f}{2 * len(E2M1_LEVELS) - 1:>12}")
    print(f"{'E8M0(scale)':<14}{8:>6}{'2^127':>12}{'2^-127':>12}{'-':>12}{'×2':>12}{255:>12}")
    print(f"\nE2M1 的全部非负取值：{E2M1_LEVELS.tolist()}")

    print("\n饱和与归零（torch 2.14 的 eager 转换）：")
    probe = torch.tensor([500.0, 448.0, 447.9, 0.002, 0.001, 2e-4, 1e-5], dtype=torch.float32)
    print(f"  输入          {[f'{v:.4g}' for v in probe.tolist()]}")
    print(f"  → E4M3        {[f'{v:.4g}' for v in probe.to(E4M3).float().tolist()]}")
    print("  500 被截到 448 而不是变成 Inf：E4M3 的 fn 变体没有 Inf 编码。")
    print("  2e-4 低于 E4M3 的最小次正规数 1.95e-3，直接归零。")
    print("  torchao 的 to_fp8_saturated 注明：PyTorch 2.11 及更早的 eager 转换不饱和，")
    print("  需要显式 clamp；2.12 起 eager 也饱和，这条差异按版本判断。")


# ---------------------------------------------------------------- B 缩放粒度
def section_b() -> None:
    head("B 同一个张量，五种缩放粒度")
    x = make_tensor()
    print(f"张量 {tuple(x.shape)}，主体 ~N(0,1)，两个离群点 ±40/±32")
    print(f"{'方案':<34}{'scale 个数':>12}{'SQNR(dB)':>12}{'最大相对误差':>14}")
    rows = [
        ("per-tensor E4M3", quantize_fp8(x, E4M3), 1),
        ("per-tensor E5M2", quantize_fp8(x, E5M2), 1),
        ("rowwise E4M3（沿最后一维）", quantize_fp8(x, E4M3, dim=-1, pow2=True), x.shape[0]),
    ]
    mx, n_mx = quantize_mx(x, "e4m3", 32)
    rows.append(("MXFP8 block=32 E8M0 scale", mx, n_mx))
    nv1, n_nv = quantize_nvfp4(x, 16, two_level=False)
    rows.append(("NVFP4 block=16 单级 scale", nv1, n_nv))
    nv2, _ = quantize_nvfp4(x, 16, two_level=True)
    rows.append(("NVFP4 block=16 两级 scale", nv2, n_nv))
    for name, got, n_scale in rows:
        rel = ((got - x).abs() / x.abs().clamp(min=1e-6)).max().item()
        print(f"{name:<34}{n_scale:>12}{sqnr_db(x, got):>12.2f}{rel:>14.2e}")
    print("\n离群点决定 per-tensor 的 scale，其余元素被压进极少的几个量级——")
    print("块缩放把一个离群点的影响限制在它所在的块内。")
    print("这张表上 NVFP4 的单级与两级 scale 完全相同：所有块 scale 都落在 E4M3 的范围内。")

    print("\n两级 scale 在块之间量级差距很大时才起作用：")
    wide = make_tensor(64, 64, seed=5)
    wide = wide.reshape(-1, 16)
    factors = torch.logspace(-4, 4, wide.shape[0], base=10, dtype=torch.float64)
    wide = (wide * factors.unsqueeze(-1)).reshape(64, 64)
    one, _ = quantize_nvfp4(wide, 16, two_level=False)
    two, _ = quantize_nvfp4(wide, 16, two_level=True)
    print(f"  块 amax 跨 8 个数量级时：单级 scale SQNR={sqnr_db(wide, one):.2f} dB，"
          f"两级 scale SQNR={sqnr_db(wide, two):.2f} dB")
    print("  单级时块 scale 要直接存进 E4M3（最大 448、最小次正规 1.95e-3），超范围的块整体失真；")
    print("  两级把块 scale 先除以一个 per-tensor FP32 再存，等于把 E4M3 的窗口搬到合适的位置。")

    print("\nMX 的 scale 只取 2 的幂（FLOOR 模式），代价是每块最多浪费近 2 倍动态范围：")
    blk = x.reshape(-1, 32)[0]
    amax = blk.abs().amax()
    exp = math.floor(math.log2(amax))
    print(f"  第 0 块 amax={amax:.6f} → floor(log2)={exp} → scale=2^{exp - F8E4M3_MAX_POW2}")
    print(f"  缩放后该块最大值={amax / 2 ** (exp - F8E4M3_MAX_POW2):.2f}，E4M3 上限 448，"
          f"利用率 {amax / 2 ** (exp - F8E4M3_MAX_POW2) / 448 * 100:.1f}%")
    print("  换成 CEIL/RCEIL 模式可以提高利用率，代价是块内最大值可能需要饱和处理。")


# ---------------------------------------------------------------- C 转置
def section_c() -> None:
    head("C 转置：什么时候必须重新量化")
    x = make_tensor(64, 64)
    print("scale 取 2 的幂时（torchao rowwise 的默认）：")
    q_row = quantize_fp8(x, E4M3, dim=-1, pow2=True)       # 沿最后一维
    q_col = quantize_fp8(x, E4M3, dim=0, pow2=True)        # 沿第 0 维
    diff = q_row != q_col
    print(f"  沿最后一维 SQNR={sqnr_db(x, q_row):.2f} dB，沿第 0 维 SQNR={sqnr_db(x, q_col):.2f} dB")
    print(f"  两种分组下结果不同的元素占 {diff.double().mean() * 100:.2f}%，"
          f"最大差={((q_row - q_col).abs().max()).item():.4f}")
    if diff.any():
        vals = x[diff].abs()
        print(f"  这些元素的绝对值区间 [{vals.min():.2e}, {vals.max():.2e}]，"
              f"整张量绝对值中位数 {x.abs().median():.3f}")
    print("  乘 2 的幂只改指数，E4M3 的舍入网格是相对的，多数元素不受分组影响；")
    print("  只有掉出窗口两端（饱和，或落进次正规与零）的元素会因为分组不同而改变。")

    print("\n换成非 2 的幂 scale（tensorwise recipe 的默认）：")
    q_row2 = quantize_fp8(x, E4M3, dim=-1, pow2=False)
    q_col2 = quantize_fp8(x, E4M3, dim=0, pow2=False)
    print(f"  结果不同的元素占 {((q_row2 != q_col2).double().mean() * 100):.1f}%，"
          f"最大差={((q_row2 - q_col2).abs().max()).item():.4f}")
    print("  scale 不是 2 的幂时，缩放本身引入舍入，分组一变整张网格都变。")

    print("\nMX 的 32 元素块同样沿最后一维切分：")
    mx_row, _ = quantize_mx(x, "e4m3", 32)
    mx_colT, _ = quantize_mx(x.t().contiguous(), "e4m3", 32)
    print(f"  先按行量化再转置 vs 直接对转置量化：最大差={((mx_row.t() - mx_colT).abs().max()).item():.4f}，"
          f"不同元素占 {((mx_row.t() != mx_colT).double().mean() * 100):.2f}%")
    double_q, _ = quantize_mx(mx_row.t().contiguous(), "e4m3", 32)
    print(f"  从高精度原张量量化 SQNR={sqnr_db(x.t(), mx_colT):.2f} dB；"
          f"对已量化结果再量化 SQNR={sqnr_db(x.t(), double_q):.2f} dB")
    print("\n块沿归约维连续是 tensor core 的要求，前向与反向的归约维不同，")
    print("所以同一个张量需要两个方向的量化结果。Transformer Engine 的做法是")
    print("从同一份高精度输入直接产出正反两份，避免对量化结果再量化的二次损失。")


# ---------------------------------------------------------------- D 三类 GEMM
def section_d() -> None:
    head("D 三类 GEMM：量化轴、累加精度与 fast_accum")
    m, k, n = 128, 512, 256
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(m, k, generator=gen, dtype=torch.float64) * 0.5
    w = torch.randn(n, k, generator=gen, dtype=torch.float64) * 0.1
    go = torch.randn(m, n, generator=gen, dtype=torch.float64) * 0.05

    print("torchao rowwise recipe 的量化轴（Float8Linear 源码）：")
    print(f"  {'GEMM':<28}{'左操作数':<26}{'右操作数':<26}")
    print(f"  {'output = X·Wᵀ':<28}{'input 沿 -1':<26}{'weight 沿 0':<26}")
    print(f"  {'grad_input = dY·W':<28}{'grad_output 沿 -1':<26}{'weight 沿 -1':<26}")
    print(f"  {'grad_weight = dYᵀ·X':<28}{'grad_output 沿 0':<26}{'input 沿 0':<26}")
    print("  同一个 weight 在前向沿 0、在 grad_input 沿 -1，两次量化结果不同。")

    ref_out, ref_gi, ref_gw = x @ w.t(), go @ w, go.t() @ x
    def q(t, dim):
        return quantize_fp8(t, E4M3, dim=dim, pow2=True)
    cases = [
        ("output = X·Wᵀ", q(x, -1) @ q(w, 0).t(), ref_out),
        ("grad_input = dY·W", q(go, -1) @ q(w, -1), ref_gi),
        ("grad_weight = dYᵀ·X", q(go, 0).t() @ q(x, 0), ref_gw),
    ]
    print(f"\n  {'GEMM':<24}{'SQNR(dB)':>12}{'相对 Frobenius 误差':>22}")
    for name, got, ref in cases:
        print(f"  {name:<24}{sqnr_db(ref, got):>12.2f}"
              f"{((got - ref).norm() / ref.norm()).item():>22.3e}")
    gw_hp = (go.bfloat16().double().t() @ x.bfloat16().double())
    print(f"  {'grad_weight 走 BF16':<24}{sqnr_db(ref_gw, gw_hp):>12.2f}"
          f"{((gw_hp - ref_gw).norm() / ref_gw.norm()).item():>22.3e}   ← rowwise_with_gw_hp")
    print("  这条配方把三个 GEMM 里最影响权重更新的一个留在高精度，其余两个仍是 FP8。")

    print("\n累加精度：把 K 维分成 32 元素一组依次累加，只改累加器 dtype")
    print(f"  {'输入量级':<12}{'累加 dtype':<12}{'SQNR(dB)':>10}{'|输出|最大':>14}{'非有限元素':>12}")
    for mag in (1.0, 150.0):
        xs, ws = x * mag, w * mag
        ref_mag = xs @ ws.t()
        xq, wq = q(xs, -1).float(), q(ws, 0).float()
        for acc_dtype in (torch.float32, torch.float16):
            acc = torch.zeros(m, n, dtype=acc_dtype)
            for s in range(0, k, 32):                # 模拟 tensor core 的分块累加
                acc = (acc + (xq[:, s:s + 32] @ wq[:, s:s + 32].t()).to(acc_dtype)).to(acc_dtype)
            finite = torch.isfinite(acc)
            sq = sqnr_db(ref_mag[finite], acc.double()[finite]) if finite.any() else float("nan")
            print(f"  ×{mag:<11.0f}{str(acc_dtype).replace('torch.', ''):<12}{sq:>10.2f}"
                  f"{acc.double()[finite].abs().max():>14.1f}{int((~finite).sum()):>12}")
    print("  常规量级下量化噪声盖过累加噪声，两种累加器看不出差别；")
    print("  输入放大 150 倍后部分和越过 65504，FP16 累加器直接产生 Inf。")
    print("  低精度累加的风险在动态范围，不在有效位数。")
    print("  torchao 默认只给前向 GEMM 开 use_fast_accum，反向两个 GEMM 保持关闭：")
    print("  梯度的累加长度和数值范围都更不友好，省下的时间换不回精度。")


# ---------------------------------------------------------------- E 随机舍入
def stochastic_round_e2m1(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    levels = E2M1_LEVELS
    sign, mag = torch.sign(x), x.abs().clamp(max=F4_MAX)
    idx = torch.bucketize(mag, levels)
    lo = levels[(idx - 1).clamp(min=0)]
    hi = levels[idx.clamp(max=len(levels) - 1)]
    span = (hi - lo).clamp(min=1e-12)
    p_hi = (mag - lo) / span
    draw = torch.rand(mag.shape, generator=gen, dtype=torch.float64)
    return sign * torch.where(draw < p_hi, hi, lo)


def section_e() -> None:
    head("E 随机舍入：消除重复量化的系统偏差")
    gen = torch.Generator().manual_seed(2)
    val = 1.2                                   # 落在 1.0 与 1.5 之间，最近邻永远选 1.0
    x = torch.full((4096,), val, dtype=torch.float64)
    rn = round_e2m1(x)
    sr = stochastic_round_e2m1(x, gen)
    print(f"  待量化值 {val}，E2M1 的相邻可表示值 1.0 / 1.5")
    print(f"  最近邻舍入：全部得到 {rn.unique().tolist()}，均值={rn.mean():.6f}，"
          f"偏差={rn.mean() - val:+.6f}")
    print(f"  随机舍入：取值 {sorted(sr.unique().tolist())}，均值={sr.mean():.6f}，"
          f"偏差={sr.mean() - val:+.6f}（期望为 0）")

    print("\n把同一个梯度量化后累加 200 次（模拟多步累积或多 rank 归约）：")
    steps = 200
    acc_rn = round_e2m1(torch.full((1,), val, dtype=torch.float64)).item() * steps
    acc_sr = sum(stochastic_round_e2m1(torch.full((1,), val, dtype=torch.float64), gen).item()
                 for _ in range(steps))
    print(f"  真值={val * steps:.1f}  最近邻={acc_rn:.1f}（相对偏差 {acc_rn / (val * steps) - 1:+.2%}）"
          f"  随机舍入={acc_sr:.1f}（{acc_sr / (val * steps) - 1:+.2%}）")
    print("  偏差不随步数平均掉，它是确定性的方向误差；随机舍入把它换成方差。")
    print("  TE 的 NVFP4 配方只对梯度使用随机舍入，权重与激活仍用最近邻。")


# ---------------------------------------------------------------- F Hadamard
def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def section_f() -> None:
    head("F Hadamard 变换：换一个坐标系再量化")
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(256, 16, generator=gen, dtype=torch.float64)
    w = torch.randn(64, 16, generator=gen, dtype=torch.float64) * 0.5
    # 稀疏离群：少量元素被放大，这正是 RHT 针对的分布
    mask = torch.rand(x.shape, generator=gen, dtype=torch.float64) < 0.01
    x = torch.where(mask, x * 30, x)
    h = hadamard(16)
    ref = x @ w.t()

    qx, _ = quantize_nvfp4(x, 16)
    qw, _ = quantize_nvfp4(w, 16)
    plain = qx @ qw.t()

    # H 正交：(XH)(WH)ᵀ = X H Hᵀ Wᵀ = XWᵀ，量化放在变换之后
    qxh, _ = quantize_nvfp4(x @ h, 16)
    qwh, _ = quantize_nvfp4(w @ h, 16)
    rotated = qxh @ qwh.t()

    rec = qxh @ h.t()                       # 把旋转域的量化结果转回原坐标便于逐元素比较
    ratio = lambda t: (t.reshape(-1, 16).abs().amax(1)
                       / t.reshape(-1, 16).abs().median(1).values).mean()
    print(f"  块内 最大/中位数 之比：变换前 {ratio(x):.1f} → 变换后 {ratio(x @ h):.1f}")
    print(f"  被量化成 0 的非零元素：直接 {int(((qx == 0) & (x != 0)).sum())}/{x.numel()}，"
          f"经 Hadamard {int(((rec.abs() < 1e-12) & (x != 0)).sum())}/{x.numel()}")
    print(f"  张量 SQNR：直接 {sqnr_db(x, qx):.2f} dB，经 Hadamard {sqnr_db(x, rec):.2f} dB")
    print(f"  GEMM 输出 SQNR：直接 {sqnr_db(ref, plain):.2f} dB，"
          f"先变换再量化 {sqnr_db(ref, rotated):.2f} dB")

    mag = x.abs()
    err_plain, err_rot = (qx - x).abs(), (rec - x).abs()
    print(f"\n  {'元素分位':<12}{'平均相对误差(直接)':>20}{'平均相对误差(Hadamard)':>24}")
    for lo, hi, name in ((0.0, 0.5, "最小 50%"), (0.5, 0.9, "50–90%"),
                         (0.9, 0.99, "90–99%"), (0.99, 1.0, "最大 1%")):
        a, b = torch.quantile(mag, torch.tensor([lo, hi], dtype=torch.float64))
        sel = (mag >= a) & (mag <= b) & (mag > 0)
        print(f"  {name:<12}{(err_plain[sel] / mag[sel]).mean():>20.3f}"
              f"{(err_rot[sel] / mag[sel]).mean():>24.3f}")
    print("\n  变换把块内尖峰摊平，块 scale 不再被一个元素绑架，被压成零的元素少了约 4/5；")
    print("  代价是原本很小的元素接收到了从大元素摊过来的噪声，它们的相对误差反而变大，")
    print("  按能量计的 SQNR 因此略降。换句话说，选什么判据决定这个变换是赚是赔。")
    print("  正交性保证 (XH)(WH)ᵀ=XWᵀ，所以它是一次可以抵消的坐标变换，代价是两次小矩阵乘。")
    print("  TE 只在 weight 梯度这条路径上使用 16×16 随机 Hadamard；它的收益依据是")
    print("  NVFP4 预训练配方的端到端质量，本段合成张量的读数不能代替那个结论。")


# ---------------------------------------------------------------- G 字节账
def section_g() -> None:
    head("G scale 本身的开销")
    print(f"{'格式':<28}{'数据位/元素':>14}{'scale 位/元素':>16}{'合计':>10}{'相对 BF16':>12}")
    rows = [
        ("BF16", 16, 0.0),
        ("FP8 per-tensor", 8, 32 / (128 * 256)),
        ("FP8 rowwise(K=256)", 8, 32 / 256),
        ("MXFP8 block=32", 8, 8 / 32),
        ("NVFP4 block=16 两级", 4, 8 / 16),
    ]
    for name, data, scale in rows:
        total = data + scale
        print(f"{name:<28}{data:>14}{scale:>16.3f}{total:>10.3f}{total / 16:>12.3f}")
    print("\nNVFP4 的块 scale 让每元素的实际位宽从 4 升到 4.5，另加一个 per-tensor FP32；")
    print("换算显存和带宽时按实际位宽算，不要按标称的 4 bit。")
    print("训练还要记另一份账：quantize kernel 的读写、转置副本、以及")
    print("delayed scaling 的 amax 历史（每个张量一段定长 buffer）都是额外状态。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU | 量化与 GEMM 均为 CPU 参照实现")
    for fn in (section_a, section_b, section_c, section_d, section_e, section_f, section_g):
        fn()
