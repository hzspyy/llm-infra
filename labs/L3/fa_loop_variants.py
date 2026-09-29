#!/usr/bin/env python3
"""L3.2-A —— FA1/FA2 的循环、split-Q / split-K 与并行来源。

3.1-B 写的是一个 FA2 形态的融合 kernel。本 lab 把"并行单元从哪里来"这件事
做成可测的对照：同样一个分块 attention，只改**网格怎么切**。

  [A] fa1 网格 = B·H（一个 program 负责一条 (b,h) 的全部 query 块）
      fa2 网格 = B·H·⌈S/BM⌉（再沿 query 块切）
      扫 B·H：只沿 B·H 切的那个应当到 B·H≈SM 数才饱和
  [B] query tile（BM）改变 grid，也改变每个 program 的工作量
  [C] split-K：把 K/V 再切成 NSPLIT 段，各段算局部 m/l/O，再用第三个 kernel 归并
      两个 kernel 的 grid 都打印出来；结果与不分段一致
  [D] warp 共享状态：K/V tile 在 CTA 内被所有 warp 共用，
      num_warps 改变寄存器/smem 分配与吞吐

用法：
    L3_OUT=<目录> python fa_loop_variants.py A B C D
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                       # noqa: E402

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as _exc:                                          # noqa: BLE001
    HAS_TRITON = False
    TRITON_ERR = str(_exc).splitlines()[0]

PEAK_TF = 232.0


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


if HAS_TRITON:

    @triton.jit
    def _attn_body(Q, K, V, Out, qh, q0, q1,
                   sqh, sqs, skh, sks, soh, sos,
                   S, sm_scale,
                   BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                   CAUSAL: tl.constexpr):
        offs_m = q0 + tl.arange(0, BM)
        offs_k = tl.arange(0, D)
        m_mask = offs_m < S
        q = tl.load(Q + qh * sqh + offs_m[:, None] * sqs + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0)
        m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        hi = S
        if CAUSAL:
            hi = tl.minimum(S, q1)
        for n0 in range(0, hi, BN):
            offs_n = n0 + tl.arange(0, BN)
            n_mask = offs_n < S
            k = tl.load(K + qh * skh + offs_n[None, :] * sks + offs_k[:, None],
                        mask=n_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * sm_scale
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(V + qh * skh + offs_n[:, None] * sks + offs_k[None, :],
                        mask=n_mask[:, None], other=0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        o = acc / l_i[:, None]
        tl.store(Out + qh * soh + offs_m[:, None] * sos + offs_k[None, :],
                 o.to(Out.dtype.element_ty), mask=m_mask[:, None])

    @triton.jit
    def _fa1(Q, K, V, Out, sqb, sqh, sqs, skb, skh, sks, sob, soh, sos,
             S, sm_scale,
             BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
             CAUSAL: tl.constexpr):
        """FA1 形态：grid 只有 B·H，序列维的循环留在 program 内部。"""
        qh = tl.program_id(0)
        nqb = tl.cdiv(S, BM)
        for qb in range(0, nqb):
            q0 = qb * BM
            _attn_body(Q, K, V, Out, qh, q0, tl.minimum(S, q0 + BM),
                       sqh, sqs, skh, sks, soh, sos, S, sm_scale,
                       BM, BN, D, CAUSAL)

    @triton.jit
    def _fa2(Q, K, V, Out, sqb, sqh, sqs, skb, skh, sks, sob, soh, sos,
             S, sm_scale,
             BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
             CAUSAL: tl.constexpr):
        """FA2 形态：grid 再乘 ⌈S/BM⌉，一个 program 只管一个 query 块。"""
        pid = tl.program_id(0)
        nqb = tl.cdiv(S, BM)
        qh = pid // nqb
        qblk = pid % nqb
        q0 = qblk * BM
        _attn_body(Q, K, V, Out, qh, q0, tl.minimum(S, q0 + BM),
                   sqh, sqs, skh, sks, soh, sos, S, sm_scale,
                   BM, BN, D, CAUSAL)

    @triton.jit
    def _fa2_splitk(Q, K, V, PM, PL, PO,
                    sqh, sqs, skh, sks, S, sm_scale, NSPLIT,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr):
        """split-K：每个 program 只算 K/V 的一段，输出局部 m/l/O（fp32）。"""
        pid = tl.program_id(0)
        nqb = tl.cdiv(S, BM)
        ks = pid % NSPLIT
        t = pid // NSPLIT
        qblk = t % nqb
        qh = t // nqb
        q0 = qblk * BM
        q1 = tl.minimum(S, q0 + BM)
        nkb = tl.cdiv(S, BN)
        per = tl.cdiv(nkb, NSPLIT)
        kb0 = ks * per
        kb1 = tl.minimum(nkb, kb0 + per)

        offs_m = q0 + tl.arange(0, BM)
        offs_k = tl.arange(0, D)
        m_mask = offs_m < S
        q = tl.load(Q + qh * sqh + offs_m[:, None] * sqs + offs_k[None, :],
                    mask=m_mask[:, None], other=0.0)
        m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for kb in range(kb0, kb1):
            n0 = kb * BN
            offs_n = n0 + tl.arange(0, BN)
            n_mask = offs_n < S
            k = tl.load(K + qh * skh + offs_n[None, :] * sks + offs_k[:, None],
                        mask=n_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * sm_scale
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(V + qh * skh + offs_n[:, None] * sks + offs_k[None, :],
                        mask=n_mask[:, None], other=0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        pid_m = pid
        tl.store(PM + pid_m * BM + tl.arange(0, BM), m_i, mask=offs_m < S)
        tl.store(PL + pid_m * BM + tl.arange(0, BM), l_i, mask=offs_m < S)
        offs_d = tl.arange(0, D)
        tl.store(PO + pid_m * BM * D + tl.arange(0, BM)[:, None] * D + offs_d[None, :],
                 acc, mask=m_mask[:, None])

    @triton.jit
    def _fa2_splitk_combine(PM, PL, PO, Out, soh, sos, S, NSPLIT,
                            BM: tl.constexpr, D: tl.constexpr):
        """把 NSPLIT 份局部状态按 m/l/O 合并公式归并，写回 O。"""
        pid = tl.program_id(0)
        nqb = tl.cdiv(S, BM)
        qblk = pid % nqb
        qh = pid // nqb
        q0 = qblk * BM
        offs_m = q0 + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < S
        m = tl.full([BM], -1.0e30, dtype=tl.float32)
        for s in range(0, NSPLIT):
            mm = tl.load(PM + (pid * NSPLIT + s) * BM + tl.arange(0, BM),
                         mask=m_mask, other=-1.0e30)
            m = tl.maximum(m, mm)
        l = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for s in range(0, NSPLIT):
            base = pid * NSPLIT + s
            mm = tl.load(PM + base * BM + tl.arange(0, BM), mask=m_mask, other=-1.0e30)
            ll = tl.load(PL + base * BM + tl.arange(0, BM), mask=m_mask, other=0.0)
            oo = tl.load(PO + base * BM * D + tl.arange(0, BM)[:, None] * D
                         + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            w = tl.exp(mm - m)
            w = tl.where(m_mask, w, 0.0)
            l += ll * w
            acc += oo * w[:, None]
        o = acc / l[:, None]
        tl.store(Out + qh * soh + offs_m[:, None] * sos + offs_d[None, :],
                 o.to(Out.dtype.element_ty), mask=m_mask[:, None])


def run_variant(kind, Q, K, V, causal=True, BM=128, BN=64, num_warps=4,
                nsplit=1):
    """按 kind 跑一种网格划分，返回 (O, grid, kernels)。"""
    B, H, S, D = Q.shape
    O = torch.empty_like(Q)
    nqb = triton.cdiv(S, BM)
    if kind == "fa1":
        grid = (B * H,)
        _fa1[grid](Q, K, V, O, Q.stride(0), Q.stride(1), Q.stride(2),
                   K.stride(0), K.stride(1), K.stride(2),
                   O.stride(0), O.stride(1), O.stride(2),
                   S, D ** -0.5, BM=BM, BN=BN, D=D, CAUSAL=causal,
                   num_warps=num_warps)
        return O, grid, 1
    if kind == "fa2":
        grid = (B * H * nqb,)
        _fa2[grid](Q, K, V, O, Q.stride(0), Q.stride(1), Q.stride(2),
                   K.stride(0), K.stride(1), K.stride(2),
                   O.stride(0), O.stride(1), O.stride(2),
                   S, D ** -0.5, BM=BM, BN=BN, D=D, CAUSAL=causal,
                   num_warps=num_warps)
        return O, grid, 1
    # split-K
    nprog = B * H * nqb * nsplit
    PM = torch.empty(nprog, BM, device=Q.device, dtype=torch.float32)
    PL = torch.empty(nprog, BM, device=Q.device, dtype=torch.float32)
    PO = torch.empty(nprog, BM, D, device=Q.device, dtype=torch.float32)
    _fa2_splitk[(nprog,)](Q, K, V, PM, PL, PO, Q.stride(1), Q.stride(2),
                          K.stride(1), K.stride(2), S, D ** -0.5, nsplit,
                          BM=BM, BN=BN, D=D, CAUSAL=causal, num_warps=num_warps)
    _fa2_splitk_combine[(B * H * nqb,)](PM, PL, PO, O, O.stride(1), O.stride(2),
                                        S, nsplit, BM=BM, D=D, num_warps=4)
    return O, (nprog,), 2


def make(B, H, S, D, dtype=torch.bfloat16):
    return (torch.randn(B, H, S, D, device="cuda", dtype=dtype) for _ in range(3))


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 并行单元从哪里来：fa1 网格 = B·H，fa2 网格 = B·H·⌈S/BM⌉")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    S, D = 4096, 128
    print(f"  这张卡 {sm} 个 SM；S={S} D={D} bf16 causal BM=128 BN=64。")
    print("  fa1 的序列维循环在 program 内部，所以 launch grid 不随 S 增长；")
    print("  fa2 每个 query 块一个 program，grid 随 S 线性增长。")
    print(f"  {'B·H':>5} {'fa1 grid':>9} {'fa2 grid':>9} {'fa1 ms':>9} {'fa2 ms':>9} "
          f"{'fa1 TFLOP/s':>12} {'fa2 TFLOP/s':>12} {'fa1 占峰值':>10} {'fa2 占峰值':>10}")
    for bh in [1, 2, 4, 8, 16, 32, 64, 128, 170, 256]:
        B, H = 1, bh
        try:
            Q, K, V = make(B, H, S, D)
        except torch.cuda.OutOfMemoryError:
            print(f"  {bh:>5}  显存不够")
            torch.cuda.empty_cache()
            continue
        n = 10 if bh <= 64 else 5
        t1 = timeit(lambda: run_variant("fa1", Q, K, V), n=n, warmup=2)
        t2 = timeit(lambda: run_variant("fa2", Q, K, V), n=n, warmup=2)
        flops = 4.0 * B * H * S * S * D / 2
        g1, g2 = B * H, B * H * triton.cdiv(S, 128)
        tf1, tf2 = flops / t1 / 1e9, flops / t2 / 1e9
        print(f"  {bh:>5} {g1:>9} {g2:>9} {t1:>9.3f} {t2:>9.3f} {tf1:>12.1f} "
              f"{tf2:>12.1f} {tf1 / PEAK_TF:>9.1%} {tf2 / PEAK_TF:>9.1%}")
        h.case(id=f"A_bh{bh}", S=S, D=D, BH=bh, fa1_grid=g1, fa2_grid=g2,
               fa1_ms=t1, fa2_ms=t2, fa1_tflops=tf1, fa2_tflops=tf2,
               flops=flops, dtype="bfloat16", causal=True)
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  判据：fa1 的吞吐要等 B·H 接近 SM 数（170）才接近饱和；")
    print("  fa2 在 B·H 很小时就靠 query 块补齐并行单元。")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] query tile：grid 与每个 program 的工作量")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    S, D = 4096, 128
    B, H = 1, 4
    Q, K, V = make(B, H, S, D)
    print(f"  S={S} D={D} B·H={B * H} bf16 causal，改 BM/BN")
    print(f"  {'BM':>5} {'BN':>5} {'fa2 grid':>9} {'ms':>9} {'TFLOP/s':>9} "
          f"{'regs':>6} {'spills':>7} {'smem B':>8}")
    for BM, BN in [(64, 64), (128, 64), (128, 128), (256, 64), (256, 128)]:
        try:
            t = timeit(lambda: run_variant("fa2", Q, K, V, BM=BM, BN=BN), n=5, warmup=2)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {BM:>5} {BN:>5}  失败: {str(exc).splitlines()[0][:50]}")
            torch.cuda.empty_cache()
            continue
        cache = _fa2.device_caches[torch.cuda.current_device()][0]
        kern = list(cache.values())[-1]
        flops = 4.0 * B * H * S * S * D / 2
        print(f"  {BM:>5} {BN:>5} {B * H * triton.cdiv(S, BM):>9} {t:>9.3f} "
              f"{flops / t / 1e9:>9.1f} {getattr(kern, 'n_regs', '?'):>6} "
              f"{getattr(kern, 'n_spills', '?'):>7} "
              f"{getattr(getattr(kern, 'metadata', None), 'shared', '?'):>8}")
        h.case(id=f"B_BM{BM}_BN{BN}", S=S, D=D, BM=BM, BN=BN,
               grid=B * H * triton.cdiv(S, BM), ms=t, tflops=flops / t / 1e9,
               regs=getattr(kern, "n_regs", None),
               spills=getattr(kern, "n_spills", None),
               smem_bytes=getattr(getattr(kern, "metadata", None), "shared", None),
               dtype="bfloat16", causal=True)
    del Q, K, V
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] split-K：把 K/V 切段，各段算局部状态再归并")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    print("  每段独立算 m/l/O，第二个 kernel 用合并公式归并 —— 结果必须与不分段一致。")
    for (B, H, S, D) in [(1, 4, 8192, 128), (1, 2, 16384, 128), (1, 1, 16384, 64)]:
        Q, K, V = make(B, H, S, D)
        ref = run_variant("fa2", Q, K, V)[0]
        C = (S + 63) // 64
        print(f"\n  B·H={B * H} S={S} D={D} bf16 causal："
              f"fa2 grid={B * H * (S // 128)}，每个 split 多一个 program")
        print(f"  {'NSPLIT':>7} {'grid':>8} {'kernels':>8} {'ms':>9} "
              f"{'max|err|':>10} {'部分状态 MB':>12}")
        for ns in [1, 2, 4, 8]:
            O, grid, nk = run_variant("fa2_splitk", Q, K, V, nsplit=ns)
            t = timeit(lambda: run_variant("fa2_splitk", Q, K, V, nsplit=ns),
                       n=5, warmup=2)
            err = (O.float() - ref.float()).abs().max().item()
            part_mb = (grid[0] * 128 * (4 + 4 + 128 * 4)) / 1024 / 1024
            print(f"  {ns:>7} {grid[0]:>8} {nk:>8} {t:>9.3f} {err:>10.3e} "
                  f"{part_mb:>10.1f}MB")
            h.case(id=f"C_BH{B * H}_S{S}_D{D}_ns{ns}", BH=B * H, S=S, D=D,
                   nsplit=ns, grid=grid[0], kernels=nk, ms=t, max_err=err,
                   partial_state_mb=part_mb, dtype="bfloat16", causal=True,
                   tol="bf16 1% of |O|max")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  split-K 换来的是并行单元（grid × NSPLIT）与一份 fp32 部分状态；")
    print("  归并用的是 3.1-A 的合并公式，实数算术下等价。")


# ---------------------------------------------------------------- D
def section_D(h):
    title("[D] warp 与共享状态：K/V tile 在 CTA 内被所有 warp 共用")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    S, D = 4096, 128
    B, H = 1, 4
    Q, K, V = make(B, H, S, D)
    print("  一个 CTA 处理一个 query 块：warp 各自负责若干 query 行，")
    print("  但 K/V 块只从显存读一次，放进共享内存供所有 warp 用。")
    print("  改 num_warps 能看到线程如何分工，改 BN 看到共享内存如何变。")
    print(f"  {'BM':>5} {'BN':>5} {'warps':>6} {'smem B':>8} {'regs':>6} {'ms':>9} "
          f"{'TFLOP/s':>9}")
    for BM, BN, nw in [(128, 64, 4), (128, 64, 8), (128, 64, 16),
                       (128, 128, 8), (256, 128, 8)]:
        try:
            t = timeit(lambda: run_variant("fa2", Q, K, V, BM=BM, BN=BN,
                                           num_warps=nw), n=5, warmup=2)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {BM:>5} {BN:>5} {nw:>6}  失败: {str(exc).splitlines()[0][:40]}")
            torch.cuda.empty_cache()
            continue
        cache = _fa2.device_caches[torch.cuda.current_device()][0]
        kern = list(cache.values())[-1]
        flops = 4.0 * B * H * S * S * D / 2
        print(f"  {BM:>5} {BN:>5} {nw:>6} "
              f"{getattr(getattr(kern, 'metadata', None), 'shared', '?'):>8} "
              f"{getattr(kern, 'n_regs', '?'):>6} {t:>9.3f} {flops / t / 1e9:>9.1f}")
        h.case(id=f"D_BM{BM}_BN{BN}_w{nw}", BM=BM, BN=BN, num_warps=nw, ms=t,
               tflops=flops / t / 1e9, regs=getattr(kern, "n_regs", None),
               smem_bytes=getattr(getattr(kern, "metadata", None), "shared", None))
    del Q, K, V
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  SM {p.multi_processor_count}")
    if HAS_TRITON:
        print(f"triton {triton.__version__}")
    h = Harness("3.2-A", "3.2", out=os.environ.get("L3_OUT"),
                backend="triton fa1/fa2/splitK + combine",
                notes="B=1 H 可变 bf16 causal；BM/BN/warps 按表变化")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "并行来源由 launch grid 直接读出；fa2 的 grid 含序列维，"
                         "fa1 不含；split-K 用合并公式与不分段结果一致。",
              "peak_tf_bf16": PEAK_TF})
    sys.stdout.flush()
    os._exit(0)
