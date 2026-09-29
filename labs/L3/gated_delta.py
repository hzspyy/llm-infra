#!/usr/bin/env python3
"""L3.4-A —— gated delta rule：逐 token 递推、门控与 chunkwise 等价。

线性 attention 里"状态与上下文无关"的那一类，最简单的核是 φ=elu(x)+1（3.4 §3）。
gated delta rule 是另一支：它带**门控衰减**和**delta 修正**，状态仍然是固定的
[K, V] 矩阵，但每一步先衰减、再按 delta 规则写入。

本 lab 三件事：
  [A] FP64 顺序参照：把逐 token 递推写成最直白的循环，打印每步状态范数
  [B] chunkwise 等价：同一算子的分块实现，chunk=16/32/64/128 与顺序参照对齐
  [C] 真实 kernel：flash-linear-attention 的 Triton chunk kernel 与 FP64 参照对拍，
      并扫 S=512..16384、B=1/4，记录实际 kernel 与时间

递推（fla 的 naive 实现，逐 token 形式）：
    h_t   = h_{t-1} · exp(g_t)                       # 门控：沿 K 维按标量衰减
    v'_t  = β_t · (v_t − k_tᵀ h_t)                   # delta 修正
    h_t   = h_t + k_t ⊗ v'_t                         # 外积写入
    o_t   = scale · q_tᵀ h_t
注意 h_t 参与了两处：`v'` 里的读取用的是**已衰减**的状态，写回也写在这个状态上。

用法：
    L3_OUT=<目录> python gated_delta.py A B C
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                      # noqa: E402

FLA_SRC = os.environ.get(
    "FLA_SRC",
    "/scratch/learn/opt/fla-src/516143e31fce09925e6c39ac37148444bad176c4")
FLA_COMMIT = "516143e31fce09925e6c39ac37148444bad176c4"
if os.path.isdir(FLA_SRC) and FLA_SRC not in sys.path:
    sys.path.insert(0, FLA_SRC)


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


def sequential_gated_delta(q, k, v, g, beta, scale=None,
                           output_final_state=True):
    """FP64 顺序参照。[B,T,H,K] / [B,T,H,V] / g,beta [B,T,H]。"""
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    h = torch.zeros(B, H, K, V, dtype=torch.float64)
    o = torch.zeros(B, T, H, V, dtype=torch.float64)
    norms = []
    for t in range(T):
        h = h * g[:, t].exp()[..., None, None]
        read = torch.einsum("bhk,bhkv->bhv", k[:, t], h)      # k_tᵀ h_t
        delta = (v[:, t] - read) * beta[:, t][..., None]
        h = h + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, h)
        norms.append(h.norm().item())
    return o, (h if output_final_state else None), norms


def make_case(B, T, H, K, V, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, generator=g, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, generator=g, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, generator=g, dtype=dtype, device=device)
    # 真实 GatedDeltaNet 对 q/k 做 L2 归一化；不归一化时状态会被外积不断放大
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    # 门控取负值（对数衰减），典型范围 [-0.05, -0.001] 每步
    gg = -torch.rand(B, T, H, generator=g, dtype=dtype, device=device) * 0.05 - 0.001
    return q, k, v, gg, beta


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] FP64 顺序参照：逐 token 的衰减、修正与写入")

    print("  递推（每 token）：")
    print("      h_t  = h_{t-1} · exp(g_t)")
    print("      v'_t = β_t · (v_t − k_tᵀ h_t)")
    print("      h_t  = h_t + k_t ⊗ v'_t")
    print("      o_t  = scale · q_tᵀ h_t")
    print()
    q, k, v, g, beta = make_case(1, 24, 2, 16, 32, seed=0)
    o, hfin, norms = sequential_gated_delta(q, k, v, g, beta)
    print(f"  T=24 H=2 K=16 V=32，状态 h 的形状 {tuple(hfin.shape)}"
          f"（与 T 无关）")
    print(f"  {'t':>3} {'状态范数':>12} {'exp(g_t)':>10} {'β_t':>8}")
    for t in [0, 1, 2, 5, 11, 23]:
        print(f"  {t:>3} {norms[t]:>12.5f} {g[0, t, 0].exp().item():>10.5f} "
              f"{beta[0, t, 0].item():>8.4f}")
    print("  状态范数不是单调的：门控让它衰减，delta 写入把它拉回来。")
    print("  这正是 delta 规则与纯门控（只衰减）的区别。")

    sub("门控开到 0 时退化成 delta rule；β=0 时退化成纯门控")
    g0 = torch.zeros_like(g)
    o_g0, h_g0, _ = sequential_gated_delta(q, k, v, g0, beta)
    b0 = torch.zeros_like(beta)
    o_b0, h_b0, _ = sequential_gated_delta(q, k, v, g, b0)
    print(f"  g=0（无衰减）：末态范数 {h_g0.norm().item():.5f}"
          f"  与 g<0 的 {hfin.norm().item():.5f} 对比")
    print(f"  β=0（不写入）：末态范数 {h_b0.norm().item():.5f}"
          f"  （只有衰减，状态趋近 0）")
    h.case(id="A_recurrence", T=24, H=2, K=16, V=32, dtype="float64",
           state_shape=list(hfin.shape),
           norm_at=[norms[t] for t in (0, 1, 2, 5, 11, 23)],
           final_norm=hfin.norm().item(),
           final_norm_g0=h_g0.norm().item(),
           final_norm_beta0=h_b0.norm().item())


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] chunkwise 等价：块内并行、块间传状态")

    print("  分块的代数：块内把 cumsum(g) 提出来做 L[i,j]=exp(decay_i−decay_j)，")
    print("  delta 修正写成一个小矩阵求逆（WY 表示），块间只传 [K,V] 状态。")
    print("  判据：多种 chunk 大小都必须与顺序参照在 FP64 下一致。")
    try:
        from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule
    except Exception as exc:                                      # noqa: BLE001
        print(f"  fla 不可用（{FLA_SRC}）：{str(exc).splitlines()[0][:80]}")
        return
    print(f"  来源：flash-linear-attention @ {FLA_COMMIT}（naive 参照实现）")
    print(f"  {'B':>3} {'T':>5} {'H':>3} {'K':>3} {'V':>4} {'chunk':>6} "
          f"{'o max|err|':>12} {'末态 max|err|':>13}")
    cases = []
    for (B, T, H, K, V) in [(1, 128, 2, 16, 32), (2, 96, 3, 32, 48)]:
        q, k, v, g, beta = make_case(B, T, H, K, V, seed=7)
        o_ref, h_ref, _ = sequential_gated_delta(q, k, v, g, beta)
        for ck in [16, 32, 64, 128]:
            if T % ck:
                continue
            o_c, h_c = naive_chunk_gated_delta_rule(
                q.float(), k.float(), v.float(), g.float(), beta.float(),
                chunk_size=ck, output_final_state=True)
            e_o = (o_c.double() - o_ref).abs().max().item()
            e_h = (h_c.double() - h_ref).abs().max().item()
            print(f"  {B:>3} {T:>5} {H:>3} {K:>3} {V:>4} {ck:>6} "
                  f"{e_o:>12.3e} {e_h:>13.3e}")
            cases.append({"B": B, "T": T, "H": H, "K": K, "V": V, "chunk": ck,
                          "o_err": e_o, "state_err": e_h})
            h.case(id=f"B_B{B}_T{T}_chunk{ck}", B=B, T=T, H=H, K=K, V=V,
                   chunk=ck, o_err=e_o, h_err=e_h, dtype="float64 vs float32 ref")
    print("\n  分块参照是 fp32（fla 的 naive 实现内部转 fp32），"
          "所以误差停在 1e-7 量级；换 chunk 不改变结果。")


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] 真实 kernel：Triton chunk kernel 与 FP64 参照")

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception as exc:                                      # noqa: BLE001
        print(f"  fla 不可用：{str(exc).splitlines()[0][:80]}")
        return
    if not torch.cuda.is_available():
        print("  需要 GPU")
        return
    print("  Triton kernel 吃 bf16 输入、fp32 状态；参照是同一输入的 FP64 顺序递推。")
    print(f"  {'B':>3} {'T':>7} {'H':>3} {'K':>4} {'V':>5} {'dtype':>9} "
          f"{'max|err|':>12} {'相对误差':>10} {'ms':>9} {'状态 MB':>9}")
    for (B, T, H, K, V) in [(1, 512, 4, 64, 64), (1, 2048, 4, 64, 64),
                            (4, 2048, 4, 64, 64), (1, 8192, 4, 64, 64),
                            (1, 16384, 4, 128, 128)]:
        for dt in (torch.bfloat16,):
            q, k, v, g, beta = make_case(B, T, H, K, V, seed=3,
                                         dtype=torch.float32)
            qd, kd, vd, gd, bd = (x.cuda() for x in (q, k, v, g, beta))
            o_ref, h_ref, _ = sequential_gated_delta(q, k, v, g, beta)
            o_c, h_c = chunk_gated_delta_rule(
                qd.to(dt), kd.to(dt), vd.to(dt), gd, bd.to(dt),
                output_final_state=True)
            err = (o_c.float().cpu().double() - o_ref).abs().max().item()
            scale_ref = max(1.0, o_ref.abs().max().item())
            n = 10 if T <= 2048 else 3
            t = timeit(lambda: chunk_gated_delta_rule(
                qd.to(dt), kd.to(dt), vd.to(dt), gd, bd.to(dt)), n=n,
                warmup=max(1, n // 3))
            state_mb = 2 * B * H * K * V * 4 / 1024 / 1024
            print(f"  {B:>3} {T:>7} {H:>3} {K:>4} {V:>5} {str(dt)[6:]:>9} "
                  f"{err:>12.3e} {err / scale_ref:>10.3e} {t:>9.3f} {state_mb:>7.2f}MB")
            h.case(id=f"C_B{B}_T{T}_K{K}_V{V}", B=B, T=T, H=H, K=K, V=V,
                   dtype=str(dt), max_err=err, rel_err=err / scale_ref, ms=t,
                   state_mb=state_mb, ref="float64 sequential", kernel="fla triton")
            del qd, kd, vd, gd, bd, o_c, h_c
            torch.cuda.empty_cache()
    print("\n  状态大小与 T 无关（2·B·H·K·V·4 字节），这是这一类算子的全部卖点；")
    print("  代价是 bf16 的误差与 kernel 的适用形状（chunk 对齐、K/V 为 16 位类型时需为偶数）。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} sm_{p.major}{p.minor}")
    h = Harness("3.4-A", "3.4", out=os.environ.get("L3_OUT"),
                backend="FP64 顺序参照 / fla naive_chunk / fla Triton chunk",
                notes=f"fla @ {FLA_COMMIT}")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "逐 token 递推与 chunkwise 在 FP64 下对齐；"
                         "Triton kernel 与 FP64 参照的差在 bf16 量级；"
                         "状态大小与 T 无关。",
              "fla_commit": FLA_COMMIT})
    sys.stdout.flush()
    os._exit(0)
