#!/usr/bin/env python3
"""L3.1-B —— 真正分块的 GPU attention kernel，以及和显式写法 / SDPA 的对照。

3.1-A 的 `flash_like` 是 Python 里的中间张量模拟，只用来说明语义。
本 lab 写一个**真正融合的分块 kernel**（Triton，一次 launch，S×S 从不落显存），
和三条参照路径在同一批形状上对照：

  [A] 正确性：分块 kernel vs SDPA-MATH（fp32 参照），causal × 非 causal
  [B] 实际 kernel 名：profiler 抓四条路径各自发射了什么
  [C] 峰值显存与临时张量：显式 QK→softmax→PV 的 O(S²) vs 分块
  [D] 时间、达成 TFLOP/s、以及按流量模型算出的有效带宽
  [E] 资源矩阵：Triton 编译出的寄存器 / 共享内存 / spills，block 大小的影响

形状：S=128/1024/4096/16384，D=64/128。显式路径在 S×S 装不下时记 OOM，
不改成别的形状——那一格本身就是结论的一部分。

用法：
    L3_OUT=<目录> python tiled_attention_gpu.py
    L3_OUT=<目录> python tiled_attention_gpu.py A C
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness, tensor_hash                      # noqa: E402

MB = 1024 * 1024
PEAK_TF = 232.0          # L1.2 实测 bf16 tensor core 上限，TFLOP/s
PEAK_BW = 1608.6         # L1.1 实测只读带宽，GB/s
L2_MIB = 96.0


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


# ---------------------------------------------------------------- kernel
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as _exc:                                        # noqa: BLE001
    HAS_TRITON = False
    TRITON_ERR = str(_exc).splitlines()[0]


if HAS_TRITON:

    @triton.jit
    def _attn_fwd(Q, K, V, Out,
                  sqb, sqh, sqs, skb, skh, sks, sob, soh, sos,
                  S, sm_scale,
                  BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                  CAUSAL: tl.constexpr):
        """每个 program 负责一个 query tile 的全部 K/V 块（FA2 的循环顺序）。

        m/l/acc 三个累加量全程留在寄存器里；只有 Q 块、K/V 块和最终 O 过显存。
        """
        pid = tl.program_id(0)
        nqb = tl.cdiv(S, BM)
        qh = pid // nqb
        qblk = pid % nqb
        offs_m = qblk * BM + tl.arange(0, BM)
        offs_k = tl.arange(0, D)
        q_ptrs = Q + qh * sqh + offs_m[:, None] * sqs + offs_k[None, :]
        m_mask = offs_m < S
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)

        hi = S
        if CAUSAL:
            hi = tl.minimum(S, (qblk + 1) * BM)
        for n0 in range(0, hi, BN):
            offs_n = n0 + tl.arange(0, BN)
            n_mask = offs_n < S
            k_ptrs = K + qh * skh + offs_n[None, :] * sks + offs_k[:, None]
            v_ptrs = V + qh * skh + offs_n[:, None] * sks + offs_k[None, :]
            k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * sm_scale
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        o = acc / l_i[:, None]
        o_ptrs = Out + qh * soh + offs_m[:, None] * sos + offs_k[None, :]
        tl.store(o_ptrs, o.to(Out.dtype.element_ty), mask=m_mask[:, None])


def tiled_attention(Q, K, V, causal=True, BM=128, BN=64, num_warps=4, num_stages=2):
    """Triton 分块 attention 的前向。Q/K/V: [B,H,S,D]，连续。"""
    B, H, S, D = Q.shape
    O = torch.empty_like(Q)
    grid = (B * H * triton.cdiv(S, BM),)
    _attn_fwd[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        S, D ** -0.5,
        BM=BM, BN=BN, D=D, CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return O


def explicit_attention(Q, K, V, causal=True):
    """显式 QK → softmax → PV：把 S×S 打分矩阵完整写出来。"""
    D = Q.shape[-1]
    s = (Q @ K.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        S = Q.shape[-2]
        mask = torch.ones(S, S, dtype=torch.bool, device=Q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ V


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


def reps_for(S):
    return 30 if S <= 1024 else (10 if S <= 4096 else 3)


def kernel_names(fn):
    """跑一次 fn，用 profiler 抓实际发射的 CUDA kernel 名。"""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.key:
            names.append(evt.key)
    out, seen = [], set()
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


SHAPES = [(128, 64), (128, 128), (1024, 64), (1024, 128),
          (4096, 64), (4096, 128), (16384, 64), (16384, 128)]
H_FIXED = 4


def make(shape, dtype=torch.bfloat16):
    S, D = shape
    B, H = 1, H_FIXED
    Q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    K = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    return Q, K, V


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 正确性：分块 kernel vs SDPA-MATH（fp32 参照）")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    print("  bf16 输入、fp32 累加的 Triton kernel，对照 fp32 的显式参照。")
    print("  容差按输出量级取 1%：bf16 的机器 epsilon 是 7.8e-3，"
          "末位舍入不该被判成实现错误。")
    print(f"  {'S':>7} {'D':>5} {'causal':>7} {'max|err|':>12} {'|O|max':>10} "
          f"{'err/|O|max':>12} {'判定':>8}")
    for S in [128, 1024, 4096]:
        for D in [64, 128]:
            Q, K, V = make((S, D))
            for causal in (True, False):
                R = explicit_attention(Q.float(), K.float(), V.float(),
                                       causal=causal).to(torch.bfloat16)
                out = tiled_attention(Q, K, V, causal=causal)
                err = (out.float() - R.float()).abs().max().item()
                scale = max(1.0, R.float().abs().max().item())
                rel = err / scale
                ok = rel < 1e-2
                print(f"  {S:>7} {D:>5} {str(causal):>7} {err:>12.4e} "
                      f"{scale:>10.3f} {rel:>12.4e} {'一致' if ok else '偏大':>8}")
                h.case(id=f"A_S{S}_D{D}_causal{causal}", S=S, D=D, dtype="bfloat16",
                       causal=causal, max_err=err, out_scale=scale, rel_err=rel,
                       ok=ok, input_hash=tensor_hash(Q, K, V), tol=1e-2,
                       ref="fp32 explicit")
            del Q, K, V
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 实际发射了什么 kernel")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    S, D = 4096, 128
    Q, K, V = make((S, D))
    paths = [("Triton 分块", lambda: tiled_attention(Q, K, V, causal=True)),
             ("显式 QK-softmax-PV", lambda: explicit_attention(Q, K, V, causal=True))]
    for name, be in [("SDPA-FLASH", SDPBackend.FLASH_ATTENTION),
                     ("SDPA-MATH", SDPBackend.MATH),
                     ("SDPA-CUDNN", SDPBackend.CUDNN_ATTENTION),
                     ("SDPA-EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)]:
        def mk(be=be):
            with sdpa_kernel(be):
                return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        paths.append((name, mk))
    print(f"  S={S} D={D} bf16 causal，每次只跑一个调用")
    for name, fn in paths:
        try:
            kn = kernel_names(fn)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {name:<20} 失败: {str(exc).splitlines()[0][:60]}")
            torch.cuda.empty_cache()
            continue
        print(f"  {name:<20} {len(kn)} 个 kernel")
        for k in kn[:4]:
            print(f"      {k[:100]}")
        h.case(id=f"B_{name}", S=S, D=D, kernel_count=len(kn),
               kernels=[k[:160] for k in kn[:6]])
    del Q, K, V
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] 峰值显存与临时张量：显式写法的 O(S²) 对分块")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    print("  bf16，B=1 H=4；“显式”一列包含打分矩阵与 softmax 概率两份 S×S。")
    print(f"  {'S':>7} {'D':>5} {'Q/K/V':>9} {'S×S 一份':>10} {'显式峰值':>10} "
          f"{'Triton 峰值':>12} {'SDPA 峰值':>10}")
    for S, D in SHAPES:
        Q, K, V = make((S, D))
        qkv = 3 * 1 * H_FIXED * S * D * 2 / MB
        ss = 1 * H_FIXED * S * S * 2 / MB
        try:
            pe = peak_mb(lambda: explicit_attention(Q, K, V, causal=True))
            pe_s = f"{pe:>8.1f}MB"
        except torch.cuda.OutOfMemoryError:
            pe, pe_s = None, f"{'OOM':>10}"
            torch.cuda.empty_cache()
        pt = peak_mb(lambda: tiled_attention(Q, K, V, causal=True))
        ps = peak_mb(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        print(f"  {S:>7} {D:>5} {qkv:>7.1f}MB {ss:>8.1f}MB {pe_s:>10} "
              f"{pt:>10.1f}MB {ps:>8.1f}MB")
        h.case(id=f"C_S{S}_D{D}", S=S, D=D, qkv_mb=qkv, sxs_mb=ss,
               explicit_peak_mb=pe, triton_peak_mb=pt, sdpa_peak_mb=ps)
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  显式路径的峰值随 S² 增长；分块路径只有 Q/K/V/O 加上一块打分与 m/l/acc，"
          "与 S 无关。")


# ---------------------------------------------------------------- D
def section_D(h):
    title("[D] 时间、达成 TFLOP/s 与按流量模型算的有效带宽")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    print("  causal bf16，B=1 H=4。FLOP 按实际计算区域计（causal 取一半）。")
    print(f"  {'S':>7} {'D':>5} {'显式 ms':>10} {'Triton ms':>10} {'SDPA ms':>10} "
          f"{'Triton TFLOP/s':>15} {'占峰值':>8} {'有效带宽 GB/s':>14}")
    for S, D in SHAPES:
        Q, K, V = make((S, D))
        flops = 4.0 * 1 * H_FIXED * S * S * D / 2
        n = reps_for(S)
        try:
            te = timeit(lambda: explicit_attention(Q, K, V, causal=True), n=n, warmup=1)
            te_s = f"{te:>8.3f}"
        except torch.cuda.OutOfMemoryError:
            te, te_s = None, f"{'OOM':>10}"
            torch.cuda.empty_cache()
        tt = timeit(lambda: tiled_attention(Q, K, V, causal=True), n=n, warmup=max(1, n // 3))
        ts = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True),
                    n=n, warmup=max(1, n // 3))
        tflops = flops / tt / 1e9
        # 流量模型：至少读 Q/K/V 各一次、写 O 一次；分块时 KV 被 nqb 个 Q 块重复读，
        # 因果下平均约 nqb/2 次。这里给的是模型值，不是计数器读数。
        nqb = (S + 127) // 128
        kv_bytes = 2 * 1 * H_FIXED * S * D * 2
        qo_bytes = 2 * 1 * H_FIXED * S * D * 2
        model_bytes = qo_bytes + kv_bytes * max(1.0, nqb / 2)
        bw = model_bytes / tt / 1e6
        print(f"  {S:>7} {D:>5} {te_s} {tt:>10.3f} {ts:>10.3f} {tflops:>15.1f} "
              f"{tflops / PEAK_TF:>7.1%} {bw:>14.1f}")
        h.case(id=f"D_S{S}_D{D}", S=S, D=D, dtype="bfloat16", causal=True,
               repeats=n, warmup=max(1, n // 3), explicit_ms=te, triton_ms=tt,
               sdpa_ms=ts, flops=flops, triton_tflops=tflops,
               pct_peak=tflops / PEAK_TF, model_bytes=model_bytes, model_bw_gbs=bw,
               timer="cuda-event", sync="after-loop")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  有效带宽高于 DRAM 上限时说明 KV 被 L2 命中了（工作集小于 96 MiB 的档）；")
    print("  只读带宽参照 1608.6 GB/s。")


# ---------------------------------------------------------------- E
def section_E(h):
    title("[E] Triton kernel 的资源矩阵：block 大小与寄存器/共享内存")

    if not HAS_TRITON:
        print(f"  Triton 不可用：{TRITON_ERR}")
        return
    S, D = 4096, 128
    Q, K, V = make((S, D))
    print(f"  S={S} D={D} bf16 causal；grid = B·H·⌈S/BM⌉ = "
          f"{1 * H_FIXED * ((S + 127) // 128)}（BM=128）")
    print(f"  {'BM':>5} {'BN':>5} {'warps':>6} {'regs':>6} {'spills':>7} "
          f"{'smem B':>9} {'ms':>9} {'TFLOP/s':>9}")
    for BM, BN, nw in [(64, 64, 4), (128, 64, 4), (128, 64, 8),
                       (128, 128, 4), (128, 128, 8), (256, 64, 8)]:
        try:
            t = timeit(lambda: tiled_attention(Q, K, V, causal=True, BM=BM, BN=BN,
                                               num_warps=nw), n=5, warmup=2)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {BM:>5} {BN:>5} {nw:>6}  失败: {str(exc).splitlines()[0][:50]}")
            torch.cuda.empty_cache()
            continue
        dev = torch.cuda.current_device()
        cache = _attn_fwd.device_caches[dev][0]                 # 本进程编译出的全部变体
        kern = list(cache.values())[-1]                         # 刚跑的那个
        regs = getattr(kern, "n_regs", None)
        spills = getattr(kern, "n_spills", None)
        smem = getattr(getattr(kern, "metadata", None), "shared", None)
        tflops = 4.0 * H_FIXED * S * S * D / 2 / t / 1e9
        print(f"  {BM:>5} {BN:>5} {nw:>6} {str(regs):>6} {str(spills):>7} "
              f"{str(smem):>9} {t:>9.3f} {tflops:>9.1f}")
        h.case(id=f"E_BM{BM}_BN{BN}_w{nw}", BM=BM, BN=BN, num_warps=nw,
               regs=regs, spills=spills, smem_bytes=smem, ms=t, tflops=tflops)
    del Q, K, V
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  sm_{p.major}{p.minor}  SM "
          f"{p.multi_processor_count}")
    if HAS_TRITON:
        print(f"triton {triton.__version__}")
    h = Harness("3.1-B", "3.1", out=os.environ.get("L3_OUT"),
                backend="triton(fused tiled) / torch explicit / SDPA",
                notes="B=1 H=4 bf16；块长 BM=128 BN=64 默认；计时用 cuda event")
    for s in want:
        SECTIONS[s](h)
    h.finish({
        "verdict": "融合分块 kernel 一次 launch、S×S 不落显存；峰值显存与 S 无关；"
                   "显式路径在 S×S 装不下时报 OOM。",
        "peak_tf_bf16": PEAK_TF, "read_bw_gbs": PEAK_BW, "l2_mib": L2_MIB,
    })
    sys.stdout.flush()
    os._exit(0)
