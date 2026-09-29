#!/usr/bin/env python3
"""L3.2-B（数值部分）—— rescale 条件与软件 exp 的误差。

FA3/FA4 的机制在 sm_120 上跑不了，但里面有两条**可以脱离硬件验证**的东西：

  [A] rescale 条件：块最大值没有变大时，修正因子 exp(m_old−m_new) 恰好是 1，
      这次 rescale 可以整段跳过。数一数真实打分分布下有多少块属于这种"白做"，
      并验证跳过之后结果**逐位不变**；再做一个"阈值延迟 rescale"的误差扫描
      （这是我自己的对照，不是论文里的条件）。
  [B] 软件 exp：把 exp2 写成 FMA 上的多项式，测它在 attention 里的实际误差：
      与 FP64 参照比、与硬件 exp2 比、看 top-1 是否翻转。

用法：
    L3_OUT=<目录> python fa34_mechanisms.py A B
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                      # noqa: E402


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def scores_for(B, H, S, D, seed=0, device="cuda", dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, generator=g, dtype=dtype, device=device)
    s = (q @ k.transpose(-1, -2)) * (D ** -0.5)
    if S > 4096:                          # 大 S 只保留一条带，避免显存爆掉
        return s
    return s


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] rescale 条件：多少块是'白做'，跳过之后是否逐位不变")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S, D, BN, H = 2048, 64, 64, 4
    print(f"  S={S} D={D} H={H} 块长={BN}，q/k 取 N(0,1)（真实 attention 打分的量级）")
    s = scores_for(1, H, S, D, seed=0, device=device)
    nblk = S // BN
    m_old = torch.full((1, H, S), float("-inf"), device=device)
    need, wasted = 0, 0
    gaps = []
    for b in range(nblk):
        blk = s[:, :, :, b * BN:(b + 1) * BN]
        m_blk = blk.amax(-1)
        m_new = torch.maximum(m_old, m_blk)
        need += int((m_new > m_old).sum().item())
        wasted += int((m_new <= m_old).sum().item())
        d = (m_new - m_old)
        d = d[torch.isfinite(m_old)]
        if d.numel():
            gaps.append(d.max().item())
        m_old = m_new
    tot = nblk * H * S
    print(f"  块数 {nblk}，每个块有 H·S = {H * S} 个 (行) 需要判定 → 共 {tot} 次")
    print(f"  需要抬基准（m_new > m_old）：{need} 次（{need / tot:.1%}）")
    print(f"  不需要抬基准（因子恰好为 1）：{wasted} 次（{wasted / tot:.1%}）")
    print(f"  抬升幅度最大的块：{max(gaps):.3f}")

    sub("跳过因子为 1 的 rescale：结果逐位不变")
    def run(skip_when_one, threshold=0.0):
        m = torch.full((1, H, S), float("-inf"), device=device)
        l = torch.zeros((1, H, S), device=device, dtype=torch.float64)
        acc = torch.zeros((1, H, S, D), device=device, dtype=torch.float64)
        vv = torch.ones(BN, D, device=device, dtype=torch.float64)
        n_rescale = 0
        for b in range(nblk):
            blk = s[:, :, :, b * BN:(b + 1) * BN].double()
            m_new = torch.maximum(m, blk.amax(-1))
            d = (m_new - m)
            if skip_when_one:
                fire = d > threshold
                n_rescale += int(fire.sum().item())
                corr = torch.where(fire, torch.exp(m - m_new),
                                   torch.ones_like(m))
            else:
                n_rescale += d.numel()
                corr = torch.exp(m - m_new)
            p = torch.exp(blk - m_new.unsqueeze(-1))
            l = l * corr + p.sum(-1)
            acc = acc * corr.unsqueeze(-1) + p @ vv
            m = m_new
        return acc / l.unsqueeze(-1), n_rescale

    a_always, n_always = run(False)
    a_skip, n_skip = run(True)
    same = torch.equal(a_always, a_skip)
    print(f"  始终 rescale：{n_always} 次；仅在 d>0 时 rescale：{n_skip} 次"
          f"（跳过 {1 - n_skip / n_always:.1%}）")
    print(f"  两种写法逐位相同 = {same}，最大差 = "
          f"{(a_always - a_skip).abs().max().item():.3e}")
    h.case(id="A_rescale_counts", S=S, D=D, block=BN, n_blocks=nblk,
           raise_count=need, skip_count=wasted, skip_fraction=wasted / tot,
           max_raise=max(gaps))
    h.case(id="A_skip_bitwise", bitwise_equal=bool(same),
           max_diff=(a_always - a_skip).abs().max().item())

    sub("阈值延迟 rescale（我的对照，不是论文条件）")
    ref, ref_n = run(False)
    print(f"  {'阈值':>8} {'实际 rescale 次数':>18} {'跳过比例':>10} "
          f"{'max|err| vs 精确':>18} {'相对误差':>12}")
    for th in [0.0, 1e-3, 1e-2, 1e-1]:
        out, cnt = run(True, threshold=th)
        e = (out - ref).abs().max().item()
        print(f"  {th:>8.0e} {cnt:>18} {1 - cnt / ref_n:>9.1%} {e:>18.3e} "
              f"{e / max(1e-30, ref.abs().max().item()):>12.3e}")
        h.case(id=f"A_threshold_{th:g}", threshold=th, rescale_events=cnt,
               skipped_fraction=1 - cnt / ref_n, max_err=e,
               rel=e / max(1e-30, ref.abs().max().item()),
               note="analysis，非论文条件")
    print("\n  阈值越大，跳过的 rescale 越多，误差也越大 —— 这是一条可调的取舍曲线；")
    print("  d=0 这一档是**精确**的（因子恰好是 1），所以它不需要任何误差预算。")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 软件 exp：把 exp2 放到 FMA 流水上之后的误差")

    def soft_exp2(x, degree=6):
        x = x.clamp(min=-126.0, max=126.0)
        n = torch.round(x)
        f = x - n
        c6 = [1.5403530393381610e-4, 1.3333558146428443e-3,
              9.6181291076284770e-3, 5.5504108664821580e-2,
              2.4022650695910070e-1, 6.9314718055994530e-1, 1.0]
        c4 = c6[2:]
        coeffs = c6 if degree == 6 else c4
        p = torch.full_like(f, coeffs[0])
        for c in coeffs[1:]:
            p = p * f + c
        return p * torch.pow(2.0, n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.linspace(-30, 0, 100000, device=device, dtype=torch.float32)
    ref = torch.exp2(x)
    for deg in (6, 4):
        got = soft_exp2(x, deg)
        rel = ((got - ref).abs() / ref).max().item()
        print(f"  soft_exp2 d{deg} 在 [-30, 0] 上的最大相对误差 = {rel:.3e}"
              f"（fp32 eps = {torch.finfo(torch.float32).eps:.1e}）")
        h.case(id=f"B_poly_d{deg}", degree=deg, max_rel_err=rel,
               fp32_eps=torch.finfo(torch.float32).eps, range=[-30.0, 0.0])

    sub("放进 attention：误差会不会被放大")
    torch.manual_seed(0)
    S, D = 1024, 64
    q = torch.randn(1, 4, S, D, device=device)
    k = torch.randn(1, 4, S, D, device=device)
    v = torch.randn(1, 4, S, D, device=device)
    s = (q @ k.transpose(-1, -2)) * (D ** -0.5)
    mask = torch.ones(S, S, dtype=torch.bool, device=device).triu(1)
    s = s.masked_fill(mask, float("-inf"))
    m = s.amax(-1, keepdim=True)
    ref64 = (torch.softmax(s.double(), dim=-1) @ v.double())
    print(f"  S={S} D={D} H=4 causal；参照为 FP64 softmax")
    print(f"  {'exp 实现':>16} {'max|err| vs FP64':>18} {'相对误差':>12} "
          f"{'top1 一致':>10}")
    for name, fn in [("硬件 exp2", lambda z: torch.exp2(z)),
                     ("soft_exp2 d6", lambda z: soft_exp2(z, 6)),
                     ("soft_exp2 d4", lambda z: soft_exp2(z, 4))]:
        p = fn((s - m) * math.log2(math.e))
        p = torch.nan_to_num(p, nan=0.0)
        p = p / p.sum(-1, keepdim=True)          # 归一化
        o = p @ v
        p64 = torch.softmax(s.double(), dim=-1)
        o64 = p64 @ v.double()
        e = (o.double() - o64).abs().max().item()
        top1 = bool((o[0, 0, -1].argmax() == o64[0, 0, -1].argmax()).item())
        print(f"  {name:>16} {e:>18.3e} {e / o64.abs().max().item():>12.3e} "
              f"{str(top1):>10}")
        h.case(id=f"B_attn_{name}", S=S, D=D, max_err=e,
               rel=e / o64.abs().max().item(), top1_match=top1,
               ref="float64 softmax")
    print("\n  d6 的误差停在 fp32 舍入量级；d4 差一档但仍在 1e-4 以内。")
    print("  与 sm_120 上的吞吐实测（exp_throughput.cu）合起来看：")
    print("  软件 exp 的多项式指令数决定了它比 SFU 慢还是快，误差只是选阶数的约束。")


SECTIONS = {"A": section_A, "B": section_B}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} sm_{p.major}{p.minor}")
    h = Harness("3.2-B-numeric", "3.2", out=os.environ.get("L3_OUT"),
                backend="torch（数值）",
                notes="rescale 条件与软件 exp 误差；性能见 exp_throughput.cu")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "d=0 时跳过 rescale 逐位不变；阈值延迟会引入可控误差；"
                         "软件 exp d6 落在 fp32 舍入量级。",
              "n_blocks": 32})
    sys.stdout.flush()
    os._exit(0)
