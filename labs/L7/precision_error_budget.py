#!/usr/bin/env python3
"""误差预算：autocast 选了什么、小更新去了哪里、四个位置各错多少。

六段内容：
  A autocast 分类   实测一批算子在 autocast 下的输出 dtype
  B 舍入与下溢      间距、次正规数与"更新被吸收"的相对阈值
  C 长归约          顺序累加、成对累加与高精度累加的误差
  D 归一化与 CE     softmax / RMSNorm / cross entropy 的误差来源
  E 四处误差        前向、dX、dW、参数更新分别相对 FP64 参照的误差
  F TF32            以尾数截断模拟的 TF32 matmul 误差（GPU 实测见 fp8_gemm_probe.py）

Usage:
    python labs/L7/precision_error_budget.py > "$RUN_DIR/error-budget.txt"
    python labs/L7/precision_error_budget.py --device cuda --sections A
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def short(dtype) -> str:
    return str(dtype).replace("torch.", "")


# ------------------------------------------------------------ A autocast 分类
def section_a(device: str) -> None:
    head(f"A autocast 在 {device} 上把哪些算子降到低精度（实测输出 dtype）")
    dev = torch.device(device)
    x = torch.randn(64, 64, device=dev)
    w = torch.randn(64, 64, device=dev)
    b = torch.randn(64, device=dev)
    idx = torch.randint(0, 64, (64,), device=dev)
    cases = [
        ("linear", lambda: F.linear(x, w, b)),
        ("matmul", lambda: x @ w),
        ("bmm", lambda: torch.bmm(x.unsqueeze(0), w.unsqueeze(0))),
        ("conv1d", lambda: F.conv1d(x.unsqueeze(0), w.unsqueeze(-1))),
        ("scaled_dot_product_attention",
         lambda: F.scaled_dot_product_attention(x.unsqueeze(0).unsqueeze(0),
                                                x.unsqueeze(0).unsqueeze(0),
                                                x.unsqueeze(0).unsqueeze(0))),
        ("softmax", lambda: torch.softmax(x, dim=-1)),
        ("log_softmax", lambda: torch.log_softmax(x, dim=-1)),
        ("layer_norm", lambda: F.layer_norm(x, (64,))),
        ("cross_entropy", lambda: F.cross_entropy(x, idx)),
        ("sum", lambda: x.sum()),
        ("mean", lambda: x.mean()),
        ("exp", lambda: torch.exp(x)),
        ("pow2", lambda: x.pow(2)),
        ("rsqrt", lambda: torch.rsqrt(x.abs() + 1)),
        ("silu", lambda: F.silu(x)),
        ("add(fp32,fp32)", lambda: x + w),
    ]
    for amp_dtype in (torch.bfloat16, torch.float16):
        if device == "cpu" and amp_dtype is torch.float16:
            continue
        print(f"\n autocast dtype = {short(amp_dtype)}")
        for name, fn in cases:
            try:
                with torch.autocast(device, dtype=amp_dtype):
                    out = fn()
                got = short(out.dtype)
            except Exception as exc:                       # 记录真实报错而不是跳过
                got = f"ERROR {type(exc).__name__}"
            print(f"   {name:<32} → {got}")
    print("\n低精度组是 GEMM 类算子；softmax / 归一化 / CE / 归约留在 FP32。")
    print("这张表按设备分别成立，CPU 与 CUDA 的策略表是两份实现，不能互相推断。")


# ------------------------------------------------------------ B 舍入与下溢
def section_b() -> None:
    head("B 间距、次正规数与更新被吸收的阈值")
    print(f"{'dtype':<10}{'1 以下间距':>14}{'1 以上间距':>14}{'最小正规数':>14}"
          f"{'最小正次正规数':>16}{'max':>12}")
    for dt in (torch.float32, torch.float16, torch.bfloat16,
               torch.float8_e4m3fn, torch.float8_e5m2):
        info = torch.finfo(dt)
        up = info.eps                      # 1 与下一个可表示数的间距
        down = info.eps / 2                # 1 是指数区间边界，下侧间距是上侧的一半
        # 从 FP32 侧逐次减半，找到该 dtype 仍能表示的最小正值
        tiny = 1.0
        while torch.tensor(tiny / 2, dtype=torch.float32).to(dt).float().item() > 0:
            tiny /= 2
        print(f"{short(dt):<10}{down:>14.3e}{up:>14.3e}{info.smallest_normal:>14.3e}"
              f"{tiny:>16.3e}{info.max:>12.3e}")

    print("\n参数 p=1.0 时，一次更新 delta 要多大才不被舍掉：")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        p0 = torch.tensor(1.0, dtype=dt)
        for delta in (1e-2, 3.9e-3, 1e-3, 1e-5, 1e-8):
            p1 = (p0.float() - delta).to(dt)
            moved = "变" if p1.item() != p0.item() else "不变"
            print(f"  {short(dt):<10} delta={delta:<9.1e} → p={p1.float().item():.10f} {moved}")
    print("BF16 在 1.0 附近的半 ULP 是 2^-9≈1.95e-3：小于它的更新整步消失，")
    print("这与梯度是否下溢无关，是参数写回时的舍入吸收。相对更新量 |Δ|/|p| 才是判据。")

    print("\n同一个数在不同 dtype 里是否为零：")
    for val in (1e-5, 1e-7, 6e-8, 1e-40):
        row = [f"{short(dt)}={torch.tensor(val, dtype=torch.float32).to(dt).float().item():.3e}"
               for dt in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)]
        print(f"  {val:<8.0e} " + "  ".join(row))
    print("FP16 的 1e-5、1e-7 落在次正规区仍非零；FP8 E4M3 在 1e-5 就归零。")
    print("是否 flush-to-zero 由具体算子和硬件路径决定，不能从存储格式直接推断。")


# ------------------------------------------------------------ C 长归约
def section_c() -> None:
    head("C 长归约：累加 dtype 与累加顺序")
    n = 1_000_000
    torch.manual_seed(0)
    base = (torch.rand(n, dtype=torch.float64) * 2 - 1) * 1e-2
    ref = base.sum().item()
    print(f"N={n}，元素量级 1e-2，FP64 参照 = {ref:.12f}")

    def seq_sum(t):
        acc = torch.zeros((), dtype=t.dtype)
        for chunk in t.split(65536):          # 分块的顺序累加，避免 Python 逐元素
            for v in chunk.split(4096):
                acc = acc + v.sum(dtype=t.dtype)
        return acc.item()

    for dt in (torch.float32, torch.bfloat16):
        t = base.to(dt)
        pairwise = t.sum().item()             # torch 的成对/分块归约
        seq = seq_sum(t)
        hi = t.sum(dtype=torch.float32).item()
        print(f"  {short(dt):<10} torch.sum={pairwise:+.9f} 相对误差={abs(pairwise - ref) / abs(ref):.3e}")
        print(f"  {'':<10} 分块顺序累加={seq:+.9f} 相对误差={abs(seq - ref) / abs(ref):.3e}")
        print(f"  {'':<10} sum(dtype=fp32)={hi:+.9f} 相对误差={abs(hi - ref) / abs(ref):.3e}")
    print("同一份数据、同一个 dtype，换累加顺序就换结果；BF16 直接累加已不可用。")
    print("这解释了为什么 loss、归一化统计量和梯度归约都要指定更高的累加精度。")


# ------------------------------------------------------------ D 归一化与 CE
def section_d() -> None:
    head("D softmax / RMSNorm / cross entropy 的误差来源")
    torch.manual_seed(0)
    logits64 = torch.randn(4, 4096, dtype=torch.float64) * 6
    logits64[0, 0] = 40.0                      # 一个大 logit，考察平移与溢出
    target = torch.tensor([0, 7, 11, 3])

    print("softmax：")
    ref = torch.softmax(logits64, dim=-1)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        got = torch.softmax(logits64.to(dt), dim=-1).double()
        print(f"  {short(dt):<10} 最大绝对误差={abs(got - ref).max():.3e} "
              f"概率和偏差={abs(got.sum(-1) - 1).max():.3e}")
    no_shift = torch.exp(logits64.to(torch.float16))
    print(f"  不做最大值平移时，FP16 的 exp(40) = {no_shift[0, 0].item()} "
          f"（上溢，softmax 内部必须先减最大值）")

    print("\nRMSNorm 的统计量：")
    x64 = torch.randn(4096, dtype=torch.float64) * 3
    ref_rms = torch.sqrt(x64.pow(2).mean())
    for dt in (torch.float32, torch.bfloat16):
        x = x64.to(dt)
        low = torch.sqrt(x.pow(2).mean()).double()
        hi = torch.sqrt(x.double().pow(2).mean())
        print(f"  {short(dt):<10} 全程低精度={low:.9f} 误差={abs(low - ref_rms) / ref_rms:.3e}"
              f" | 升精度求均值={hi:.9f} 误差={abs(hi - ref_rms) / ref_rms:.3e}")

    print("\ncross entropy：")
    ref_loss = F.cross_entropy(logits64, target)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        low = F.cross_entropy(logits64.to(dt), target).double()
        up = F.cross_entropy(logits64.to(dt).float(), target).double()
        print(f"  logits {short(dt):<10} 直接算={low:.9f} 误差={abs(low - ref_loss):.3e}"
              f" | 先转 FP32={up:.9f} 误差={abs(up - ref_loss):.3e}")
    print("logits 一旦落到 BF16，先升精度也救不回已经丢掉的位；")
    print("autocast 把 CE 留在 FP32 指的是计算精度，不能替代 logits 本身的精度。")


# ------------------------------------------------------------ E 四处误差
class MLP(torch.nn.Module):
    """权重由同一份 FP64 初值转换而来，保证各 dtype 从同一个点出发。"""

    def __init__(self, dim: int, dtype, init: dict[str, torch.Tensor]):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 4 * dim, bias=False, dtype=dtype)
        self.fc2 = torch.nn.Linear(4 * dim, dim, bias=False, dtype=dtype)
        self.fc3 = torch.nn.Linear(dim, dim, bias=False, dtype=dtype)
        with torch.no_grad():
            for name, mod in (("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)):
                mod.weight.copy_(init[name].to(dtype))

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return self.fc3(F.layer_norm(x + h, (x.shape[-1],)))


def section_e() -> None:
    head("E 前向、dX、dW、参数更新四处分别错多少")
    dim, bs = 256, 64
    gen = torch.Generator().manual_seed(0)
    x64 = torch.randn(bs, dim, generator=gen, dtype=torch.float64)
    init = {
        "fc1": torch.randn(4 * dim, dim, generator=gen, dtype=torch.float64) * dim ** -0.5,
        "fc2": torch.randn(dim, 4 * dim, generator=gen, dtype=torch.float64) * (4 * dim) ** -0.5,
        "fc3": torch.randn(dim, dim, generator=gen, dtype=torch.float64) * dim ** -0.5,
    }

    def run(dtype, autocast_dtype=None):
        model = MLP(dim, dtype, init)
        x = x64.detach().to(dtype).requires_grad_(True)
        ctx = (torch.autocast("cpu", dtype=autocast_dtype) if autocast_dtype
               else torch.autocast("cpu", enabled=False))
        with ctx:
            out = model(x)
            loss = out.float().pow(2).mean()
        loss.backward()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        w0 = model.fc1.weight.detach().double().clone()
        opt.step()
        return {
            "out": out.detach().double(),
            "dx": x.grad.detach().double(),
            "dw": model.fc1.weight.grad.detach().double(),
            "update": (model.fc1.weight.detach().double() - w0),
        }

    ref = run(torch.float64)
    print("误差口径：max|got-ref| / max|ref|，更新一列同时给 AdamW 首步的符号翻转比例")
    print(f"{'配置':<28}{'前向':>12}{'dX':>12}{'dW':>12}{'参数更新':>12}{'符号翻转':>10}")
    for label, kw in [
        ("FP32", dict(dtype=torch.float32)),
        ("FP32 + autocast BF16", dict(dtype=torch.float32, autocast_dtype=torch.bfloat16)),
        ("BF16 参数与计算", dict(dtype=torch.bfloat16)),
    ]:
        got = run(**kw)
        rel = {k: (abs(got[k] - ref[k]).max() / abs(ref[k]).max()).item() for k in ref}
        flip = ((got["update"].sign() != ref["update"].sign()).double().mean()).item()
        print(f"{label:<28}{rel['out']:>12.2e}{rel['dx']:>12.2e}{rel['dw']:>12.2e}"
              f"{rel['update']:>12.2e}{flip * 100:>9.2f}%")
    print("前向、dX、dW 三处误差逐级放大，量级与输入 dtype 一致。")
    print("参数更新一列反而最大：AdamW 首步的更新量约等于 ±lr，与梯度大小无关，")
    print("接近零的分量只要符号被扰动，整个更新就换方向。判断数值是否可用要看 dW，")
    print("不是看参数变化；反过来，更新看着正常也不代表梯度是准的。")


# ------------------------------------------------------------ F TF32
def to_tf32_like(t: torch.Tensor) -> torch.Tensor:
    """把 FP32 的尾数截到 10 位，模拟 TF32 的输入表示（累加仍是 FP32）。"""
    bits = t.float().view(torch.int32)
    rounded = (bits + (1 << 12)) & ~((1 << 13) - 1)
    return rounded.view(torch.float32)


def section_f() -> None:
    head("F TF32 的表示：10 位尾数输入 + FP32 累加（CPU 模拟）")
    torch.manual_seed(0)
    a64 = torch.randn(512, 512, dtype=torch.float64)
    b64 = torch.randn(512, 512, dtype=torch.float64)
    ref = a64 @ b64
    fp32 = (a64.float() @ b64.float()).double()
    tf32 = (to_tf32_like(a64.float()) @ to_tf32_like(b64.float())).double()
    bf16 = (a64.bfloat16() @ b64.bfloat16()).double()
    for name, got in [("FP32", fp32), ("TF32 模拟", tf32), ("BF16 输入", bf16)]:
        print(f"  {name:<12} 最大绝对误差={abs(got - ref).max():.3e} "
              f"相对 Frobenius 误差={(got - ref).norm() / ref.norm():.3e}")
    print("TF32 与 BF16 的尾数分别是 10 位和 7 位，指数范围都与 FP32 相同。")
    print("TF32 是 matmul 的计算模式：张量仍以 FP32 存储和读写，它不是一种权重格式。")
    print("本段是 CPU 上的表示模拟，真实 TF32 kernel 的累加与分块另见 GPU 实测。")


SECTIONS = {"A": section_a, "B": section_b, "C": section_c,
            "D": section_d, "E": section_e, "F": section_f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sections", default="ABCDEF")
    args = ap.parse_args()
    print(f"torch {torch.__version__} | device={args.device}")
    for key in args.sections:
        fn = SECTIONS[key]
        fn(args.device) if key == "A" else fn()


if __name__ == "__main__":
    main()
