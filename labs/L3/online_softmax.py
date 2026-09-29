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


# ---------------------------------------------------------------- E
def _keep_matrix(Sq, Sk=None, causal=False, keep=None, device="cpu"):
    """可见性矩阵：True 表示该 (query, key) 位置可见。形状 [Sq, Sk]。"""
    Sk = Sq if Sk is None else Sk
    if keep is None:
        keep = torch.ones(Sq, Sk, dtype=torch.bool, device=device)
    else:
        keep = keep.to(device=device, dtype=torch.bool)
    if causal:
        keep = keep & torch.ones(Sq, Sk, dtype=torch.bool, device=device).tril()
    return keep


def attention_ref_fp64(Q, K, V, keep=None, causal=True):
    """FP64 参照：完整打分矩阵 + safe softmax + PV。

    全 mask 行的约定：该行输出全 0，不计入任何归一化。
    返回 (O, n_allmask)。
    """
    Q, K, V = Q.double(), K.double(), V.double()
    Sq, D = Q.shape[-2], Q.shape[-1]
    km = _keep_matrix(Sq, K.shape[-2], causal, keep, Q.device)
    s = (Q @ K.transpose(-1, -2)) * (D ** -0.5)
    s = s.masked_fill(~km, float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    dead = ~torch.isfinite(m)                       # 整行都被 mask
    p = torch.exp(s - m.masked_fill(dead, 0.0))
    p = torch.where(km, p, torch.zeros_like(p))
    l = p.sum(dim=-1, keepdim=True)
    O = (p / l.clamp(min=1e-300)) @ V
    O = torch.where(l > 0, O, torch.zeros_like(O))
    return O, int(dead.sum().item())


def flash_states(Q, K, V, block, keep=None, causal=True, dtype=torch.float32):
    """分块 attention，返回 (O, states)。

    states[i] 是处理第 i 个 K/V 块之后的运行状态 (m, l, acc)，
    shape 为 [B, H, Sq]。这是"逐块状态可检查"的入口。
    Q/K/V 先 cast 到 dtype；dtype 决定了这个实现的舍入量级。
    """
    Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)
    B, H, S, D = Q.shape
    km = _keep_matrix(S, K.shape[-2], causal, keep, Q.device)
    O = torch.zeros_like(Q)
    states = []
    for q0 in range(0, S, block):
        q1 = min(q0 + block, S)
        Qb = Q[:, :, q0:q1]
        m = torch.full((B, H, q1 - q0), float("-inf"), dtype=dtype, device=Q.device)
        l = torch.zeros((B, H, q1 - q0), dtype=dtype, device=Q.device)
        acc = torch.zeros_like(Qb)
        for k0 in range(0, S, block):
            k1 = min(k0 + block, S)
            if causal and k0 > q1 - 1:
                break                                    # 整块在未来
            Kb, Vb = K[:, :, k0:k1], V[:, :, k0:k1]
            kb = km[q0:q1, k0:k1]
            s = (Qb @ Kb.transpose(-1, -2)) * (D ** -0.5)
            s = s.masked_fill(~kb, float("-inf"))
            m_blk = s.amax(dim=-1)
            m_new = torch.maximum(m, m_blk)
            corr = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
            p = torch.nan_to_num(torch.exp(s - m_new.unsqueeze(-1)), nan=0.0)
            l = l * corr + p.sum(dim=-1)
            acc = acc * corr.unsqueeze(-1) + p @ Vb
            m = m_new
            states.append({"q": (q0, q1), "k": (k0, k1),
                           "m": m.clone(), "l": l.clone(), "acc": acc.clone()})
        O[:, :, q0:q1] = torch.where(
            (l > 0).unsqueeze(-1), acc / l.clamp(min=torch.finfo(dtype).tiny).unsqueeze(-1),
            torch.zeros_like(acc))
    return O, states


def subset_state(Q, K, V, keep, kblock=None, dtype=torch.float64):
    """在给定的 K/V 子集上算一个部分状态 (m, l, acc)，key 数可以与 Q 不同。

    keep 的 shape 是 [Sq, Sk]，直接给出可见性（不再叠加 causal）。
    这是 E.3 用来构造"任意切分的部分状态"的入口。
    """
    Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)
    B, H, Sq, D = Q.shape
    Sk = K.shape[-2]
    keep = keep.to(device=Q.device, dtype=torch.bool)
    m = torch.full((B, H, Sq), float("-inf"), dtype=dtype, device=Q.device)
    l = torch.zeros((B, H, Sq), dtype=dtype, device=Q.device)
    acc = torch.zeros_like(Q)
    step = kblock or Sk
    for k0 in range(0, Sk, step):
        k1 = min(k0 + step, Sk)
        kb = keep[:, k0:k1]
        s = (Q @ K[:, :, k0:k1].transpose(-1, -2)) * (D ** -0.5)
        s = s.masked_fill(~kb, float("-inf"))
        m_new = torch.maximum(m, s.amax(dim=-1))
        corr = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
        p = torch.nan_to_num(torch.exp(s - m_new.unsqueeze(-1)), nan=0.0)
        l = l * corr + p.sum(dim=-1)
        acc = acc * corr.unsqueeze(-1) + p @ V[:, :, k0:k1]
        m = m_new
    return {"m": m, "l": l, "acc": acc}


def merge_states(states):
    """把若干个部分状态 (m, l, acc) 合并成一个。这就是跨块/跨 split 的归并公式。"""
    ms = torch.stack([s["m"] for s in states])
    m = ms.amax(dim=0)
    l = torch.zeros_like(m)
    acc = torch.zeros_like(states[0]["acc"])
    for s in states:
        w = torch.nan_to_num(torch.exp(s["m"] - m), nan=0.0)
        l = l + s["l"] * w
        acc = acc + s["acc"] * w.unsqueeze(-1)
    return {"m": m, "l": l, "acc": acc}


def states_to_out(st, dtype):
    l = st["l"]
    return torch.where((l > 0).unsqueeze(-1),
                       st["acc"] / l.clamp(min=torch.finfo(dtype).tiny).unsqueeze(-1),
                       torch.zeros_like(st["acc"]))


def section_E(harness=None):
    title("[E] FP64 参照、块长 1/3/16、状态合并，以及极值和全 mask 行")

    sub("E.1 逐块状态：m / l / acc 每处理完一个 K/V 块都在变")
    torch.manual_seed(3)
    S, D, BLK = 12, 4, 3
    Q = torch.randn(1, 1, S, D)
    K = torch.randn(1, 1, S, D)
    V = torch.randn(1, 1, S, D)
    _, st = flash_states(Q, K, V, BLK, causal=True, dtype=torch.float64)
    q0 = ((S - 1) // BLK) * BLK
    q1 = min(q0 + BLK, S)
    print(f"  S={S} D={D} block={BLK} causal  "
          f"（打印最后一个 query 块 [{q0},{q1}) 的状态，它看得到全部 K/V 块）")
    print(f"  {'k块':>7} {'m_new[q0]':>12} {'l[q0]':>10}  acc[q0, 0:4]")
    for s in st:
        if s["q"] != (q0, q1):
            continue
        a = s["acc"][0, 0, 0]
        print(f"  {str(s['k']):>7} {s['m'][0, 0, 0].item():>12.4f} "
              f"{s['l'][0, 0, 0].item():>10.4f}  "
              f"{[round(v, 4) for v in a.tolist()]}")
    print("  m 单调不减；l 与 acc 在基准抬高时被同一个修正因子缩小。")

    sub("E.2 块长 1/3/16 与不同分块：结果与 FP64 参照的差")
    ref, n_dead = attention_ref_fp64(Q, K, V, causal=True)
    print(f"  FP64 参照（完整 S×S）全 mask 行数 = {n_dead}")
    print(f"  {'块长':>5} {'dtype':>9} {'max|err| vs FP64':>18} {'相对误差':>12} "
          f"{'top1 一致':>10}")
    cases = []
    for blk in [1, 3, S]:
        for dt in [torch.float64, torch.float32, torch.bfloat16]:
            O, _ = flash_states(Q, K, V, blk, causal=True, dtype=dt)
            Od = O.double()
            err = (Od - ref).abs().max().item()
            denom = ref.abs().max().item()
            top1 = int(Od[0, 0, -1].argmax()) == int(ref[0, 0, -1].argmax())
            print(f"  {blk:>5} {str(dt).split('.')[-1]:>9} {err:>18.3e} "
                  f"{err / denom:>12.3e} {str(top1):>10}")
            cases.append({"block": blk, "dtype": str(dt).split(".")[-1],
                          "max_err_vs_fp64": err, "rel_err": err / denom,
                          "top1_match": top1})
    print("  同一种数值类型下换块长只改变舍入顺序；跨 dtype 的差是 dtype 的精度。")

    sub("E.3 状态合并：把任意切分的部分状态并起来，应当等于一次跑完")
    print("  合并公式（m/l/acc 三个统计量）：")
    print("      m = max(m_i)")
    print("      l = Σ l_i · exp(m_i − m)")
    print("      acc = Σ acc_i · exp(m_i − m)")
    parts = {
        "逐元素(1)": [1] * S,
        "均匀(3)": [3] * (S // 3),
        "单块(≥S)": [S],
        "不等长(5,1,6)": [5, 1, 6],
    }
    print(f"  {'切分':>16} {'max|err| vs 一次跑完':>22} {'max|err| vs FP64':>18}")
    merge_cases = []
    for name, sizes in parts.items():
        sts = []
        pos = 0
        for sz in sizes:
            idx = torch.arange(pos, min(pos + sz, S))
            pos += sz
            Kp, Vp = K[:, :, idx], V[:, :, idx]
            # 每个 K 子集只在这一段上可见；按全局列索引取它的可见性
            sub_keep = _keep_matrix(S, S, True, None)[:, idx]
            st = subset_state(Q, Kp, Vp, sub_keep, kblock=min(sz, 4))
            sts.append(st)
        merged = states_to_out(merge_states(sts), torch.float64)
        one, _ = flash_states(Q, K, V, S, causal=True, dtype=torch.float64)
        e1 = (merged - one).abs().max().item()
        e2 = (merged - ref).abs().max().item()
        print(f"  {name:>16} {e1:>22.3e} {e2:>18.3e}")
        merge_cases.append({"partition": name, "sizes": sizes,
                            "max_err_vs_onepass": e1, "max_err_vs_fp64": e2})

    sub("E.4 极值输入：不减最大值的写法在哪里崩")
    torch.manual_seed(4)
    S2, D2 = 32, 8
    Qx = torch.randn(1, 1, S2, D2) * 300.0
    Kx = torch.randn(1, 1, S2, D2)
    Vx = torch.randn(1, 1, S2, D2)
    refx, _ = attention_ref_fp64(Qx, Kx, Vx, causal=True)
    s_naive = (Qx @ Kx.transpose(-1, -2)) * (D2 ** -0.5)
    # 真正的"不减最大值"：直接 exp 再归一化（torch.softmax 内部会减最大值）
    num = torch.exp(s_naive)
    p_naive = num / num.sum(dim=-1, keepdim=True)
    o_naive = (p_naive @ Vx).double()
    print(f"  打分范围 [{s_naive.min().item():.1f}, {s_naive.max().item():.1f}]，"
          f"fp32 exp 上限约 88.7")
    print(f"  {'实现':>22} {'max|err|':>12} {'NaN 数':>8}")
    rows = [("不减最大值的朴素写法", o_naive, int(torch.isnan(o_naive).sum().item()))]
    for blk, dt in [(32, torch.float32), (3, torch.float32), (16, torch.bfloat16)]:
        O, _ = flash_states(Qx, Kx, Vx, blk, causal=True, dtype=dt)
        rows.append((f"online 块长{blk} {str(dt).split('.')[-1]}", O.double(),
                     int(torch.isnan(O).sum().item())))
    for name, O, nn in rows:
        err = (O - refx).abs().max().item() if nn == 0 else float("nan")
        print(f"  {name:>22} {err:>12.3e} {nn:>8}")
    print("  极端打分下朴素写法溢出成 NaN；online 写法逐块抬基准，全程有限。")

    sub("E.5 全 mask 行：输出约定")
    keep = _keep_matrix(8, causal=True)
    keep[3, :] = False                                   # 第 3 行整行屏蔽
    keep[6, :] = False
    Qm, Km, Vm = (torch.randn(1, 1, 8, 4) for _ in range(3))
    refm, n_dead = attention_ref_fp64(Qm, Km, Vm, keep=keep, causal=False)
    Om, _ = flash_states(Qm, Km, Vm, 4, keep=keep, causal=False, dtype=torch.float64)
    print(f"  屏蔽第 3、6 两行全部 key：FP64 参照报告 {n_dead} 个全 mask 行")
    print(f"  参照第 3 行输出 = {[round(v, 6) for v in refm[0, 0, 3].tolist()]}")
    print(f"  分块第 3 行输出 = {[round(v, 6) for v in Om[0, 0, 3].tolist()]}")
    print(f"  两实现对拍 max|err| = {(Om - refm).abs().max().item():.3e}，"
          f"NaN 数 = {int(torch.isnan(Om).sum().item())}")
    print("  约定：一行里所有 key 都不可见时，该行输出全 0，不参与归一化。")
    print("  未定这个约定时，l=0 会走到 0/0；真实 kernel 里通常把该行单独置零。")

    sub("E.6 容差：dtype 与规模决定误差量级")
    print(f"  {'S':>6} {'块长':>5} {'fp32 max|err|':>15} {'bf16 max|err|':>15} "
          f"{'fp32 相对':>12}")
    tol_cases = []
    torch.manual_seed(5)
    for S3 in [64, 256, 512]:
        Q3 = torch.randn(1, 2, S3, 64)
        K3 = torch.randn(1, 2, S3, 64)
        V3 = torch.randn(1, 2, S3, 64)
        ref3, _ = attention_ref_fp64(Q3, K3, V3, causal=True)
        scale = ref3.abs().max().item()
        for blk in [16, 64]:
            e32 = (flash_states(Q3, K3, V3, blk, causal=True,
                                dtype=torch.float32)[0].double() - ref3).abs().max().item()
            e16 = (flash_states(Q3, K3, V3, blk, causal=True,
                                dtype=torch.bfloat16)[0].double() - ref3).abs().max().item()
            print(f"  {S3:>6} {blk:>5} {e32:>15.3e} {e16:>15.3e} {e32 / scale:>12.3e}")
            tol_cases.append({"S": S3, "block": blk, "fp32_max_err": e32,
                              "bf16_max_err": e16, "fp32_rel": e32 / scale})
    print(f"  fp32 的机器 epsilon = {torch.finfo(torch.float32).eps:.3e}，"
          f"bf16 = {torch.finfo(torch.bfloat16).eps:.3e}")
    print("  容差按 dtype 与累加长度取：fp32 用 ~1e-6·|O|max，bf16 要放宽到 ~1e-2·|O|max。")

    if harness is not None:
        for c in cases:
            harness.case(id=f"E2_block{c['block']}_{c['dtype']}", **c)
        for c in merge_cases:
            harness.case(id=f"E3_{c['partition']}", **c)
        for c in tol_cases:
            harness.case(id=f"E6_S{c['S']}_blk{c['block']}", **c)
        harness.case(id="E4_extreme_naive", max_err=None,
                     nan=int(torch.isnan(o_naive).sum().item()),
                     score_min=s_naive.min().item(), score_max=s_naive.max().item())
        harness.case(id="E5_allmask_rows", n_dead=n_dead,
                     ref_row=[round(v, 6) for v in refm[0, 0, 3].tolist()],
                     blocked_row=[round(v, 6) for v in Om[0, 0, 3].tolist()],
                     max_err=(Om - refm).abs().max().item(),
                     nan=int(torch.isnan(Om).sum().item()))
        harness.finish({
            "verdict": "分块与合并公式在 FP64/Fp32 下与完整参照一致；"
                       "块长只改舍入顺序；全 mask 行按输出全 0 处理；"
                       "极值下不减最大值的写法溢出为 NaN。",
            "fp32_eps": torch.finfo(torch.float32).eps,
            "bf16_eps": torch.finfo(torch.bfloat16).eps,
        })


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D,
            "E": section_E}

if __name__ == "__main__":
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import Harness

    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    h = Harness("3.1-A", "3.1", out=os.environ.get("L3_OUT"))
    for s in want:
        if s == "E":
            section_E(harness=h)          # 内部收尾并落盘
        else:
            SECTIONS[s]()
    if "E" not in want:
        h.finish({"verdict": "仅运行了 A–D 节，未跑数值参照"})
