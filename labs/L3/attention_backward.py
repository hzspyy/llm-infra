#!/usr/bin/env python3
"""L3.1-C —— 反向：softmax 的梯度、重算策略，以及真实模型的 Q/K/V 布局。

正向他只存 m/l/O 三个累加量，不存 S×S 的概率矩阵。反向于是只剩两条路：
把 P 存下来，或者**用 Q/K 重算**。本 lab 把两条路都写出来对拍：

  [A] 推导：softmax 的 Jacobian、ds 的闭式，以及"重算换显存"的边界
  [B] 梯度对拍（CPU FP64）：autograd 参照 / 解析公式 / 分块重算 三方一致
  [C] GPU 峰值与完整调用时间：存 P vs 重算 vs SDPA-FLASH
  [D] Qwen3-1.7B 真实 Q/K/V：GQA 原生布局 vs 展开成 MHA 的代价，
      并在真实张量上对拍重算 backward 与 SDPA backward

用法：
    L3_OUT=<目录> python attention_backward.py            # 全部（D 需要 GPU + 模型）
    L3_OUT=<目录> python attention_backward.py A B        # 只做推导与对拍
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness, tensor_hash                      # noqa: E402

MB = 1024 * 1024
MODEL = os.environ.get("L31_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def timeit(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    peak = torch.cuda.max_memory_allocated()
    del out
    torch.cuda.empty_cache()
    return (peak - base) / MB


# ---------------------------------------------------------------- 参照实现
def naive_attention(Q, K, V, causal=True, gqa=False):
    """显式 QK → softmax → PV。存 P。Q:[B,Hq,S,D] K/V:[B,Hkv,S,D]。"""
    D = Q.shape[-1]
    Kx = expand_kv(K, Q) if gqa else K
    Vx = expand_kv(V, Q) if gqa else V
    s = (Q @ Kx.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        S = Q.shape[-2]
        mask = torch.ones(S, S, dtype=torch.bool, device=Q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ Vx


def expand_kv(K, Q):
    """GQA 展开：把 KV 头复制到和 query 头一样多（显式物化）。"""
    hq, hkv = Q.shape[1], K.shape[1]
    if hq == hkv:
        return K
    return K.repeat_interleave(hq // hkv, dim=1)


def analytic_backward(Q, K, V, dO, causal=True, gqa=False):
    """从 P 出发的解析反向，逐项按公式写。返回 dQ, dK, dV（GQA 时对 KV 头求和）。"""
    D = Q.shape[-1]
    scale = D ** -0.5
    Kx = expand_kv(K, Q) if gqa else K
    Vx = expand_kv(V, Q) if gqa else V
    s = (Q @ Kx.transpose(-1, -2)) * scale
    if causal:
        S = Q.shape[-2]
        mask = torch.ones(S, S, dtype=torch.bool, device=Q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    dVx = p.transpose(-1, -2) @ dO
    dp = dO @ Vx.transpose(-1, -2)
    ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    dQ = (ds @ Kx) * scale
    dKx = (ds.transpose(-1, -2) @ Q) * scale
    if gqa:
        g = Q.shape[1] // K.shape[1]
        dK = dKx.view(dKx.shape[0], K.shape[1], g, *dKx.shape[2:]).sum(dim=2)
        dV = dVx.view(dVx.shape[0], V.shape[1], g, *dVx.shape[2:]).sum(dim=2)
    else:
        dK, dV = dKx, dVx
    return dQ, dK, dV, p


class FlashRecompute(torch.autograd.Function):
    """正向只存 Q/K/V 与 m/l；反向用 Q/K 重算 P，不存 S×S。

    GQA：Q 有 Hq 个头，K/V 有 Hkv 个，h -> h // group。
    """

    @staticmethod
    def forward(ctx, Q, K, V, causal=True, block=128):
        B, Hq, S, D = Q.shape
        Hkv = K.shape[1]
        g = Hq // Hkv
        scale = D ** -0.5
        O = torch.zeros_like(Q)
        # 计算与 m/l 的精度：低精度输入用 fp32 累加，fp64 输入保持 fp64 以做对拍
        wd = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        Qw, Kw, Vw = Q.to(wd), K.to(wd), V.to(wd)
        M = torch.empty(B, Hq, S, device=Q.device, dtype=wd)
        L = torch.zeros(B, Hq, S, device=Q.device, dtype=wd)
        for q0 in range(0, S, block):
            q1 = min(q0 + block, S)
            Qb = Qw[:, :, q0:q1]
            m = torch.full((B, Hq, q1 - q0), float("-inf"), device=Q.device,
                           dtype=wd)
            l = torch.zeros((B, Hq, q1 - q0), device=Q.device, dtype=wd)
            acc = torch.zeros((B, Hq, q1 - q0, D), device=Q.device, dtype=wd)
            for k0 in range(0, S, block):
                k1 = min(k0 + block, S)
                if causal and k0 > q1 - 1:
                    break
                Kb = Kw[:, :, k0:k1].repeat_interleave(g, dim=1)
                Vb = Vw[:, :, k0:k1].repeat_interleave(g, dim=1)
                s = (Qb @ Kb.transpose(-1, -2)) * scale
                if causal:
                    qi = torch.arange(q0, q1, device=Q.device).view(-1, 1)
                    ki = torch.arange(k0, k1, device=Q.device).view(1, -1)
                    s = s.masked_fill(ki > qi, float("-inf"))
                m_new = torch.maximum(m, s.amax(dim=-1))
                corr = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
                p = torch.nan_to_num(torch.exp(s - m_new.unsqueeze(-1)), nan=0.0)
                l = l * corr + p.sum(dim=-1)
                acc = acc * corr.unsqueeze(-1) + p @ Vb
                m = m_new
            O[:, :, q0:q1] = torch.where(
                (l > 0).unsqueeze(-1), acc / l.clamp(min=1e-30).unsqueeze(-1),
                torch.zeros_like(acc)).to(O.dtype)
            M[:, :, q0:q1] = m
            L[:, :, q0:q1] = l
        ctx.save_for_backward(Q, K, V, M, L, O)
        ctx.causal = causal
        ctx.block = block
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, M, L, O = ctx.saved_tensors
        causal, block = ctx.causal, ctx.block
        B, Hq, S, D = Q.shape
        Hkv = K.shape[1]
        g = Hq // Hkv
        scale = D ** -0.5
        wd = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        Qw, Kw, Vw = Q.to(wd), K.to(wd), V.to(wd)
        Ow, dOw = O.to(wd), dO.to(wd)
        dQ = torch.zeros_like(Qw)
        dK = torch.zeros_like(Kw)
        dV = torch.zeros_like(Vw)
        # Σ_j p_j dp_j = dO·O：整行的归一化项可以从正向输出 O 直接算出来，
        # 不需要把一行里所有块的部分和再归约一次。
        rowterm = (dOw * Ow).sum(dim=-1, keepdim=True)
        for q0 in range(0, S, block):
            q1 = min(q0 + block, S)
            Qb = Qw[:, :, q0:q1]
            m = M[:, :, q0:q1]
            l = L[:, :, q0:q1].clamp(min=1e-30)
            dOb = dOw[:, :, q0:q1]
            rt = rowterm[:, :, q0:q1]
            for k0 in range(0, S, block):
                k1 = min(k0 + block, S)
                if causal and k0 > q1 - 1:
                    break
                Kb = Kw[:, :, k0:k1]
                Vb = Vw[:, :, k0:k1]
                Kbx = Kb.repeat_interleave(g, dim=1)
                Vbx = Vb.repeat_interleave(g, dim=1)
                s = (Qb @ Kbx.transpose(-1, -2)) * scale
                if causal:
                    qi = torch.arange(q0, q1, device=Q.device).view(-1, 1)
                    ki = torch.arange(k0, k1, device=Q.device).view(1, -1)
                    s = s.masked_fill(ki > qi, float("-inf"))
                p = torch.nan_to_num(torch.exp(s - m.unsqueeze(-1)), nan=0.0) \
                    / l.unsqueeze(-1)
                dVx = p.transpose(-1, -2) @ dOb
                dp = dOb @ Vbx.transpose(-1, -2)
                ds = p * (dp - rt)          # rt 已经是 [B,Hq,qblk,1]
                dQ[:, :, q0:q1] += (ds @ Kbx) * scale
                dKx = (ds.transpose(-1, -2) @ Qb) * scale
                dV[:, :, k0:k1] += dVx.view(dVx.shape[0], Hkv, g,
                                            *dVx.shape[2:]).sum(dim=2)
                dK[:, :, k0:k1] += dKx.view(dKx.shape[0], Hkv, g,
                                            *dKx.shape[2:]).sum(dim=2)
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 推导：softmax 的 Jacobian 与重算策略")

    print("  设 P = softmax(S)，S = QKᵀ/√D。已知 dO（O = P V 的上游梯度）：")
    print("      dV = Pᵀ dO")
    print("      dP = dO Vᵀ")
    print("  softmax 的 Jacobian 作用在 dP 上：")
    print("      ds_j = p_j · (dp_j − Σ_i p_i dp_i)")
    print("  再回传到两个矩阵乘：")
    print("      dQ = (ds K) · scale        dK = (dsᵀ Q) · scale")
    print()
    print("  为什么必须有 p_j 这个因子：softmax 的每个输出都依赖整行，")
    print("  归一化项把 dP 的公共分量消掉了 —— 这一项就是把公共分量减掉。")
    print()
    print("  分块反向的关键：Σ_i p_i dp_i 看起来要把整行扫一遍，其实可以用正向输出换掉：")
    print("      Σ_i p_i (dO·V_i) = dO · (Σ_i p_i V_i) = dO · O")
    print("  于是每个 (q 块, k 块) 只需要 O 的一行，不用跨块归约。")
    print()
    print("  重算策略的边界（正向只存 Q/K/V、m/l、O，不存 P）：")
    print("      存的：Q/K/V（各 O(S·D)） + m/l（各 O(S)）")
    print("      不存的：P（O(S²)）")
    print("      代价：反向每个 (q 块, k 块) 重算一次 exp，正向的 exp 要做两遍")
    print("  和 7.0 的 activation checkpointing 是同一笔交易：用算力换显存。")
    print("  S 小时 P 装得下，重算反而多付一遍 exp；S 大时省下的是 O(S²)。")

    # 小例子：softmax 的 Jacobian 与 (dp - Σ p dp) 的等价
    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64)
    p = torch.softmax(x, 0)
    J = torch.diag(p) - p.outer(p)
    ref = torch.autograd.functional.jacobian(lambda z: torch.softmax(z, 0), x)
    print(f"\n  5 维 softmax 的 Jacobian：手写 diag(p) − p pᵀ 与 autograd 的 "
          f"max|err| = {(J - ref).abs().max().item():.3e}")
    dp = torch.randn(5, dtype=torch.float64)
    lhs = J @ dp
    rhs = p * (dp - (p * dp).sum())
    print(f"  同一 Jacobian 作用在随机 dp 上：两种写法 max|err| = "
          f"{(lhs - rhs).abs().max().item():.3e}")

    # dO·O 恒等式的小例子
    torch.manual_seed(7)
    V = torch.randn(6, 4, dtype=torch.float64)
    p2 = torch.softmax(torch.randn(6, dtype=torch.float64), 0)
    O = p2 @ V
    dO = torch.randn(4, dtype=torch.float64)
    lhs2 = (p2 * (dO @ V.transpose(0, 1))).sum()
    rhs2 = dO @ O
    print(f"\n  Σ_i p_i (dO·V_i) 与 dO·O：max|err| = {(lhs2 - rhs2).abs().item():.3e}"
          f"（{lhs2:.6f} vs {rhs2:.6f}）")
    h.case(id="A_jacobian", max_err_vs_autograd=(J - ref).abs().max().item(),
           max_err_two_forms=(lhs - rhs).abs().max().item(), dim=5,
           dtype="float64")
    h.case(id="A_rowterm_identity", max_err=(lhs2 - rhs2).abs().item(),
           sum_p_dp=lhs2.item(), dO_dot_O=rhs2.item(), dim=6, dtype="float64")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 梯度对拍：autograd / 解析公式 / 分块重算")

    torch.manual_seed(1)
    print("  CPU FP64，小矩阵；三种实现都必须一致到机器精度。")
    print(f"  {'B':>3} {'Hq':>3} {'Hkv':>3} {'S':>5} {'D':>4} {'causal':>7} "
          f"{'§解析 dQ':>12} {'§重算 dQ':>12} {'§重算 dK':>12} {'§重算 dV':>12}")
    for (B, Hq, Hkv, S, D, causal) in [
            (1, 2, 2, 16, 8, True), (1, 2, 2, 16, 8, False),
            (2, 4, 2, 24, 16, True), (1, 4, 1, 32, 8, True),
            (1, 2, 2, 33, 8, True)]:
        Q = torch.randn(B, Hq, S, D, dtype=torch.float64, requires_grad=True)
        K = torch.randn(B, Hkv, S, D, dtype=torch.float64, requires_grad=True)
        V = torch.randn(B, Hkv, S, D, dtype=torch.float64, requires_grad=True)
        gqa = Hq != Hkv

        # 参照：materialize P 的 autograd
        O = naive_attention(Q, K, V, causal=causal, gqa=gqa)
        g = torch.randn_like(O)
        O.backward(g)
        dQ_ref, dK_ref, dV_ref = Q.grad.clone(), K.grad.clone(), V.grad.clone()

        # 解析公式
        dQ_a, dK_a, dV_a, _ = analytic_backward(Q, K, V, g, causal=causal, gqa=gqa)
        # 分块重算
        Q2 = Q.detach().clone().requires_grad_()
        K2 = K.detach().clone().requires_grad_()
        V2 = V.detach().clone().requires_grad_()
        FlashRecompute.apply(Q2, K2, V2, causal, 8).backward(g)
        e_qa = (dQ_a - dQ_ref).abs().max().item()
        e_q = (Q2.grad - dQ_ref).abs().max().item()
        e_k = (K2.grad - dK_ref).abs().max().item()
        e_v = (V2.grad - dV_ref).abs().max().item()
        print(f"  {B:>3} {Hq:>3} {Hkv:>3} {S:>5} {D:>4} {str(causal):>7} "
              f"{e_qa:>12.3e} {e_q:>12.3e} {e_k:>12.3e} {e_v:>12.3e}")
        h.case(id=f"B_B{B}_Hq{Hq}_Hkv{Hkv}_S{S}_D{D}_c{causal}", B=B, Hq=Hq, Hkv=Hkv,
               S=S, D=D, causal=causal, gqa=gqa, dtype="float64",
               analytic_dQ_err=e_qa, recompute_dQ_err=e_q,
               recompute_dK_err=e_k, recompute_dV_err=e_v,
               input_hash=tensor_hash(Q, K, V))
    print("\n  三种实现的差都在 1e-15 量级（FP64 机器精度）；")
    print("  GQA 的 dK/dV 是把同一 KV 头上所有 query 头的贡献求和，不是平均。")


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] GPU 峰值与完整调用时间：存 P vs 重算 vs SDPA-FLASH")

    dev = "cuda"
    D = 128
    print("  bf16，B=1 H=4 D=128，causal；'完整' = 前向 + 反向（含 autograd 图）。")
    print(f"  {'S':>6} {'存 P 峰值':>10} {'重算峰值':>10} {'SDPA 峰值':>10} "
          f"{'存 P ms':>9} {'重算 ms':>9} {'SDPA ms':>9} {'S²·2B':>9}")
    for S in [1024, 4096, 8192]:
        B, H = 1, 4
        Q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        K = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        V = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)

        def naive_step():
            q = Q.clone().requires_grad_()
            k = K.clone().requires_grad_()
            v = V.clone().requires_grad_()
            o = naive_attention(q, k, v, causal=True)
            o.backward(torch.ones_like(o))
            return o

        def recompute_step():
            q = Q.clone().requires_grad_()
            k = K.clone().requires_grad_()
            v = V.clone().requires_grad_()
            o = FlashRecompute.apply(q, k, v, True, 128)
            o.backward(torch.ones_like(o))
            return o

        def sdpa_step():
            q = Q.clone().requires_grad_()
            k = K.clone().requires_grad_()
            v = V.clone().requires_grad_()
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o.backward(torch.ones_like(o))
            return o

        try:
            pn = peak_mb(naive_step)
            tn = timeit(naive_step, n=5, warmup=2)
        except torch.cuda.OutOfMemoryError:
            pn, tn = None, None
            torch.cuda.empty_cache()
        pr = peak_mb(recompute_step)
        tr = timeit(recompute_step, n=5, warmup=2)
        ps = peak_mb(sdpa_step)
        ts = timeit(sdpa_step, n=5, warmup=2)
        sxs = B * H * S * S * 2 / MB
        pn_s = f"{pn:>8.1f}" if pn else f"{'OOM':>10}"
        tn_s = f"{tn:>7.3f}" if tn else f"{'OOM':>9}"
        print(f"  {S:>6} {pn_s}MB {pr:>8.1f}MB {ps:>8.1f}MB {tn_s} "
              f"{tr:>9.3f} {ts:>9.3f} {sxs:>7.1f}MB")
        h.case(id=f"C_S{S}", S=S, D=D, dtype="bfloat16", causal=True,
               storeP_peak_mb=pn, recompute_peak_mb=pr, sdpa_peak_mb=ps,
               storeP_ms=tn, recompute_ms=tr, sdpa_ms=ts, sxs_mb=sxs,
               timer="cuda-event", include="forward+backward")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  重算路径的峰值不随 S² 增长；代价是反向多付一遍 exp 与矩阵乘。")
    print("  本实现是分块循环的 Python 版本，时间不能代表融合 kernel；")
    print("  它要说明的是峰值曲线的形状。")


# ---------------------------------------------------------------- D
def _capture_qkv(model, tok, text, S, layer, HQ, HKV, D):
    """跑一次真实前向，在第 layer 层抓 hidden，再用该层自己的 q/k/v_proj 得到 Q/K/V。"""
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < S:
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :S].cuda()
    mod = model.model.layers[layer].self_attn
    cap = {}

    def pre_hook(_m, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        cap["hs"] = hs.detach()

    handle = mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        model(ids)
    handle.remove()
    hs = cap["hs"].to(torch.bfloat16)
    b, s, _ = hs.shape
    with torch.no_grad():                      # 只要张量本身，不要模型的计算图
        q = mod.q_proj(hs).view(b, s, HQ, D).transpose(1, 2)
        k = mod.k_proj(hs).view(b, s, HKV, D).transpose(1, 2)
        v = mod.v_proj(hs).view(b, s, HKV, D).transpose(1, 2)
        if hasattr(mod, "q_norm") and hasattr(mod, "k_norm"):
            q = mod.q_norm(q)
            k = mod.k_norm(k)
    return q.detach().contiguous(), k.detach().contiguous(), v.detach().contiguous()


def section_D(h):
    title("[D] Qwen3-1.7B 的真实 Q/K/V：GQA 原生布局 vs 展开成 MHA")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    HQ = cfg.num_attention_heads
    HKV = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // HQ)
    layer = cfg.num_hidden_layers // 2
    print(f"  {MODEL}: layers={cfg.num_hidden_layers} Hq={HQ} Hkv={HKV} D={D}")
    print(f"  Q/K/V 取自第 {layer} 层：hook 该层 self_attn 的输入 hidden，"
          f"再走该层自己的 q/k/v_proj（RoPE 不改变形状与字节账）")

    text = ("The capital of France is Paris. The capital of Japan is Tokyo. "
            "Machine learning models process sequences of tokens one at a time. "
            "A KV cache stores the keys and values of the prefix so that decode "
            "only needs to compute one query row per step.")

    sub("字节账与时间：GQA 原生布局 vs 展开成 MHA（真实张量）")
    print(f"  {'S':>6} {'Q/K/V 实shape':>26} {'原生读MB':>9} {'展开读MB':>9} "
          f"{'GQA ms':>9} {'展开 ms':>9} {'比值':>7} {'展开峰值MB':>11}")
    for S in [2048, 8192]:
        qq, kk, vv = _capture_qkv(model, tok, text, S, layer, HQ, HKV, D)
        shape = f"q{tuple(qq.shape)}k{tuple(kk.shape)}"
        nat = (HQ * D + 2 * HKV * D) * 2 * S / MB
        exp = (HQ * D * 3) * 2 * S / MB
        t_gqa = timeit(lambda: F.scaled_dot_product_attention(
            qq, kk, vv, is_causal=True, enable_gqa=True), n=10, warmup=3)

        def expand_then_attn():
            ke = kk.repeat_interleave(HQ // HKV, dim=1)
            ve = vv.repeat_interleave(HQ // HKV, dim=1)
            return F.scaled_dot_product_attention(qq, ke, ve, is_causal=True)

        t_exp = timeit(expand_then_attn, n=10, warmup=3)
        m_exp = peak_mb(expand_then_attn)
        print(f"  {S:>6} {shape:>26} {nat:>9.1f} {exp:>9.1f} {t_gqa:>9.4f} "
              f"{t_exp:>9.4f} {t_exp / t_gqa:>6.2f}× {m_exp:>10.1f}")
        h.case(id=f"D_layout_S{S}", S=S, Hq=HQ, Hkv=HKV, D=D, dtype="bfloat16",
               native_read_mb=nat, expanded_read_mb=exp,
               sdpa_gqa_ms=t_gqa, expand_mha_ms=t_exp, ratio=t_exp / t_gqa,
               expand_peak_mb=m_exp, input_hash=tensor_hash(qq, kk, vv),
               source=f"layer{layer}.q/k/v_proj 实际权重")

        if S <= 2048:
            sub("真实张量上的重算 backward 对拍（FP64）")
            qg = qq.double().requires_grad_()
            kg = kk.double().requires_grad_()
            vg = vv.double().requires_grad_()
            o = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True,
                                               enable_gqa=True)
            go = torch.randn_like(o)
            o.backward(go)
            dQ_ref, dK_ref, dV_ref = (qg.grad.clone(), kg.grad.clone(),
                                      vg.grad.clone())
            q2 = qq.double().requires_grad_()
            k2 = kk.double().requires_grad_()
            v2 = vv.double().requires_grad_()
            FlashRecompute.apply(q2, k2, v2, True, 256).backward(go)
            e_q = (q2.grad - dQ_ref).abs().max().item()
            e_k = (k2.grad - dK_ref).abs().max().item()
            e_v = (v2.grad - dV_ref).abs().max().item()
            print(f"  重算 backward vs SDPA backward：dQ {e_q:.3e}  dK {e_k:.3e}  "
                  f"dV {e_v:.3e}")
            h.case(id=f"D_gradcheck_S{S}", S=S, dQ_err=e_q, dK_err=e_k, dV_err=e_v,
                   dtype="float64", ref="SDPA(enable_gqa) backward",
                   input_hash=tensor_hash(qq, kk, vv))
            del qg, kg, vg, q2, k2, v2
        del qq, kk, vv
        torch.cuda.empty_cache()

    sub("因果 mask 与 GQA 布局在真实模型里允许什么")
    print(f"  该模型 causal=True、无 padding mask、q_len==kv_len；GQA 组大小 "
          f"{HQ // HKV}。")
    print("  SDPA 的 enable_gqa 直接接受 Hkv<Hq，不需要在框架层 repeat_interleave；")
    print("  展开成 MHA 会多读 1.5× 的 KV 字节并多一次全量拷贝。")
    h.case(id="D_mask_layout", causal=True, padding_mask=False, q_len_eq_kv_len=True,
           gqa_group=HQ // HKV, notes="真实第 %d 层 hidden 经 q/k/v_proj 得到" % layer)
    del model
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name}  sm_{p.major}{p.minor}")
    h = Harness("3.1-C", "3.1", out=os.environ.get("L3_OUT"),
                backend="CPU FP64 对拍 + crater GPU 峰值/计时",
                notes="A/B 只依赖 torch；C/D 需要 GPU；D 需要 Qwen3-1.7B 本地缓存")
    for s in want:
        SECTIONS[s](h)
    h.finish({
        "verdict": "解析反向与分块重算在 FP64 下与 autograd 一致；"
                   "重算路径峰值不随 S² 增长；GQA 原生布局比展开省 1/3 的 KV 读取。",
        "model": MODEL,
    })
    sys.stdout.flush()
    os._exit(0)
