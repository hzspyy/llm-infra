#!/usr/bin/env python3
"""L3.1 —— online softmax 与分块 attention，从零写一遍。

纯 torch（CPU 就能跑），不用任何 attention 库。目的是把
「为什么可以不materialize 那个 S×S 矩阵」这件事推到能自己写出来。

四步：
  [A] softmax 为什么要减最大值 —— 不减会溢出
  [B] 三遍扫描 -> 两遍 -> 一遍（online），每一步的代数与实测等价
  [C] 把 V 的加权和也放进同一遍 -> 分块 attention
  [D] 和 PyTorch 的 SDPA 对拍

用法：
    python online_softmax.py
    python online_softmax.py B C
"""

import sys

import torch

torch.manual_seed(0)


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


# ---------------------------------------------------------------- A
def section_A():
    title("[A] softmax 为什么必须减去最大值")

    print("  softmax(x)_i = exp(x_i) / Σ_j exp(x_j)")
    print("  数学上，给每个 x 减去同一个常数 c，结果不变：")
    print("    exp(x_i - c) / Σ exp(x_j - c) = [exp(x_i)/exp(c)] / [Σ exp(x_j)/exp(c)]")
    print("  分子分母的 exp(c) 约掉了。所以减常数是**恒等变换**，不是近似。")

    sub("但在浮点里不减就会炸")
    for val in [10.0, 88.0, 100.0, 1000.0]:
        x = torch.tensor([val, val - 1, val - 2], dtype=torch.float32)
        naive_num = torch.exp(x)
        naive = naive_num / naive_num.sum()
        safe = torch.softmax(x, dim=0)
        print(f"  x 最大 {val:>7.1f}   exp(x) = {naive_num.tolist()}")
        print(f"  {'':17}   朴素 softmax = {naive.tolist()}")
        print(f"  {'':17}   减最大值后   = {[round(v, 6) for v in safe.tolist()]}")
    print("\n  float32 能表示的最大值约 3.4e38，exp(88.7) 就到顶了。")
    print("  一旦 exp 溢出成 inf，inf/inf = nan，整行结果全毁。")

    sub("反方向：全是很小的负数")
    x = torch.tensor([-800.0, -801.0, -802.0])
    num = torch.exp(x)
    print(f"  exp(x) = {num.tolist()}  -> 全部下溢成 0")
    print(f"  朴素 softmax = {(num / num.sum()).tolist()}   (0/0)")
    print(f"  减最大值后   = {[round(v, 6) for v in torch.softmax(x, 0).tolist()]}")
    print("\n  减最大值之后，最大那一项恰好是 exp(0)=1，其余都在 (0,1]，")
    print("  既不会上溢也不会全部下溢。这就是所谓 safe softmax。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 三遍 -> 两遍 -> 一遍")

    x = torch.randn(8) * 3
    print(f"  x = {[round(v, 3) for v in x.tolist()]}")

    sub("写法一：三遍扫描（教科书写法）")
    m = x.max()                       # 第 1 遍：求最大值
    p = torch.exp(x - m)              # 第 2 遍：求指数
    l = p.sum()                       # 第 3 遍：求和
    y3 = p / l
    print(f"  第1遍 m = max(x)      = {m.item():.6f}")
    print(f"  第2遍 p = exp(x - m)  = {[round(v, 4) for v in p.tolist()]}")
    print(f"  第3遍 l = Σp          = {l.item():.6f}")
    print(f"  y = p / l             = {[round(v, 4) for v in y3.tolist()]}")
    print("  代价：x 要从内存读 2 次（求 m 一次，求 p 一次）。")

    sub("写法二：一遍就把 m 和 l 都算出来（online）")
    print("  关键恒等式：已经处理完前 k 个元素，手上有")
    print("      m_old = max(x_1..x_k)")
    print("      l_old = Σ_{i<=k} exp(x_i - m_old)")
    print("  来了第 k+1 个元素 x_new：")
    print("      m_new = max(m_old, x_new)")
    print("      l_new = l_old * exp(m_old - m_new) + exp(x_new - m_new)")
    print()
    print("  为什么那个 exp(m_old - m_new) 是对的：l_old 里每一项的基准是 m_old，")
    print("  现在基准要换成 m_new，每一项都要乘 exp(m_old)/exp(m_new)")
    print("  = exp(m_old - m_new)。整个和是线性的，所以提到外面乘一次就行。")
    print("  m_new >= m_old，所以这个因子 <= 1，永远不会放大 —— 数值上是安全的。")

    m_run = torch.tensor(float("-inf"))
    l_run = torch.tensor(0.0)
    print(f"\n  {'步':>3} {'x_new':>9} {'m_old':>10} {'m_new':>10} "
          f"{'修正因子':>10} {'l_new':>12}")
    for i, xv in enumerate(x):
        m_old, l_old = m_run.clone(), l_run.clone()
        m_run = torch.maximum(m_old, xv)
        corr = torch.exp(m_old - m_run) if torch.isfinite(m_old) else torch.tensor(0.0)
        l_run = l_old * corr + torch.exp(xv - m_run)
        print(f"  {i:>3} {xv.item():>9.4f} {m_old.item():>10.4f} "
              f"{m_run.item():>10.4f} {corr.item():>10.6f} {l_run.item():>12.6f}")

    print(f"\n  一遍之后 m={m_run.item():.6f}  l={l_run.item():.6f}")
    print(f"  三遍写法   m={m.item():.6f}  l={l.item():.6f}")
    print(f"  m 相同: {torch.allclose(m_run, m)}   l 相同: {torch.allclose(l_run, l)}")

    y1 = torch.exp(x - m_run) / l_run
    print(f"  最终结果与三遍写法的 max|err| = {(y1 - y3).abs().max().item():.3e}")
    print(f"  与 torch.softmax 的 max|err|   = "
          f"{(y1 - torch.softmax(x, 0)).abs().max().item():.3e}")

    sub("按块做也一样（这才是 kernel 里的写法）")
    print("  逐元素只是块大小为 1 的特例。块大小 B 时：")
    print("      m_blk = max(块内)          l_blk = Σ exp(块内 - m_blk)")
    print("      m_new = max(m_old, m_blk)")
    print("      l_new = l_old*exp(m_old-m_new) + l_blk*exp(m_blk-m_new)")
    for B in [2, 4, 8]:
        m_run = torch.tensor(float("-inf")); l_run = torch.tensor(0.0)
        for s in range(0, x.numel(), B):
            blk = x[s:s + B]
            m_blk = blk.max()
            l_blk = torch.exp(blk - m_blk).sum()
            m_new = torch.maximum(m_run, m_blk)
            corr = torch.exp(m_run - m_new) if torch.isfinite(m_run) else torch.tensor(0.0)
            l_run = l_run * corr + l_blk * torch.exp(m_blk - m_new)
            m_run = m_new
        print(f"  块大小 {B}:  m={m_run.item():.6f}  l={l_run.item():.6f}   "
              f"与三遍写法一致: {torch.allclose(l_run, l, atol=1e-5)}")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 把 V 的加权和也放进同一遍：分块 attention")

    S, D = 12, 4
    q = torch.randn(D)
    K = torch.randn(S, D)
    V = torch.randn(S, D)
    scale = D ** -0.5

    sub("参照：先算完整的 S 维打分，再 softmax，再乘 V")
    s_full = (K @ q) * scale
    p_full = torch.softmax(s_full, 0)
    o_full = p_full @ V
    print(f"  打分 s（长度 {S}）= {[round(v, 3) for v in s_full.tolist()]}")
    print(f"  o = {[round(v, 5) for v in o_full.tolist()]}")
    print(f"  这条路要把长度 {S} 的 s 完整存下来。真实场景里它是 S×S 的矩阵。")

    sub("分块：每块只看 K/V 的一段，从不存完整的 s")
    print("  额外的量：输出累加器 O。基准换了之后 O 也要跟着修正。")
    print("      O_new = O_old * exp(m_old - m_new) + exp(s_blk - m_new) @ V_blk")
    print("  最后统一除以 l。**除法留到最后一次做**，中间只累加未归一化的和。")

    B = 4
    m_run = torch.tensor(float("-inf"))
    l_run = torch.tensor(0.0)
    o_run = torch.zeros(D)
    print(f"\n  {'块':>3} {'m_old':>9} {'m_new':>9} {'修正':>9} {'l_run':>10}  o_run（未归一化）")
    for bi, s0 in enumerate(range(0, S, B)):
        Kb, Vb = K[s0:s0 + B], V[s0:s0 + B]
        s_blk = (Kb @ q) * scale
        m_blk = s_blk.max()
        m_old = m_run.clone()
        m_new = torch.maximum(m_old, m_blk)
        corr = torch.exp(m_old - m_new) if torch.isfinite(m_old) else torch.tensor(0.0)
        p_blk = torch.exp(s_blk - m_new)
        l_run = l_run * corr + p_blk.sum()
        o_run = o_run * corr + p_blk @ Vb
        m_run = m_new
        print(f"  {bi:>3} {m_old.item():>9.4f} {m_new.item():>9.4f} "
              f"{corr.item():>9.6f} {l_run.item():>10.5f}  "
              f"{[round(v, 4) for v in o_run.tolist()]}")

    o_online = o_run / l_run
    print(f"\n  归一化后 o = {[round(v, 5) for v in o_online.tolist()]}")
    print(f"  参照     o = {[round(v, 5) for v in o_full.tolist()]}")
    print(f"  max|err| = {(o_online - o_full).abs().max().item():.3e}")
    print(f"\n  峰值额外存储：一块的打分 {B} 个数 + m,l 各 1 个 + O 的 {D} 个。")
    print(f"  与 S={S} 无关。S 变成 100 万也是这么多。")


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 写成真正的多头版本，与 PyTorch SDPA 对拍")

    def flash_like(Q, K, V, block=64, causal=True):
        """分块 attention。Q/K/V: [B, H, S, D]。只用基本张量运算。"""
        B, H, S, D = Q.shape
        scale = D ** -0.5
        O = torch.zeros_like(Q)
        # 外层遍历 Q 的块（FA2 的顺序），内层遍历 K/V 的块
        for q0 in range(0, S, block):
            q1 = min(q0 + block, S)
            Qb = Q[:, :, q0:q1]                          # [B,H,bq,D]
            m = torch.full((B, H, q1 - q0), float("-inf"), device=Q.device,
                           dtype=Q.dtype)
            l = torch.zeros((B, H, q1 - q0), device=Q.device, dtype=Q.dtype)
            acc = torch.zeros_like(Qb)
            for k0 in range(0, S, block):
                k1 = min(k0 + block, S)
                if causal and k0 > q1 - 1:               # 整块都在未来，跳过
                    continue
                Kb, Vb = K[:, :, k0:k1], V[:, :, k0:k1]
                s = (Qb @ Kb.transpose(-1, -2)) * scale  # [B,H,bq,bk]
                if causal:
                    qi = torch.arange(q0, q1, device=Q.device).view(-1, 1)
                    ki = torch.arange(k0, k1, device=Q.device).view(1, -1)
                    s = s.masked_fill(ki > qi, float("-inf"))
                m_blk = s.amax(dim=-1)                   # [B,H,bq]
                m_new = torch.maximum(m, m_blk)
                corr = torch.exp(m - m_new)
                corr = torch.nan_to_num(corr, nan=0.0)   # 首块 m=-inf
                p = torch.exp(s - m_new.unsqueeze(-1))
                p = torch.nan_to_num(p, nan=0.0)         # 整行被 mask 掉时
                l = l * corr + p.sum(dim=-1)
                acc = acc * corr.unsqueeze(-1) + p @ Vb
                m = m_new
            O[:, :, q0:q1] = acc / l.clamp(min=1e-20).unsqueeze(-1)
        return O

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (B, H, S, D) in [(1, 2, 128, 32), (2, 4, 256, 64)]:
        Q = torch.randn(B, H, S, D, device=dev, dtype=torch.float32)
        K = torch.randn(B, H, S, D, device=dev, dtype=torch.float32)
        V = torch.randn(B, H, S, D, device=dev, dtype=torch.float32)
        for causal in (False, True):
            mine = flash_like(Q, K, V, block=64, causal=causal)
            ref = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=causal)
            err = (mine - ref).abs().max().item()
            print(f"  B={B} H={H} S={S} D={D} causal={causal!s:<5} "
                  f"max|err| = {err:.3e}   {'一致' if err < 1e-4 else '不一致'}")

    sub("块大小不影响结果（只影响速度与显存）")
    Q = torch.randn(1, 2, 256, 32, device=dev)
    K = torch.randn(1, 2, 256, 32, device=dev)
    V = torch.randn(1, 2, 256, 32, device=dev)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    for blk in [16, 32, 64, 128, 256]:
        err = (flash_like(Q, K, V, block=blk, causal=True) - ref).abs().max().item()
        print(f"  block={blk:>4}   max|err| vs SDPA = {err:.3e}")
    print("\n  分块是**精确**的重排，不是近似。块大小只改变访存与并行度。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    for s in want:
        SECTIONS[s]()
