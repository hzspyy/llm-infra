#!/usr/bin/env python3
"""L3.2-C / 3.2-D —— 后端对照矩阵，以及 attention 在真实模型里占多少。

  [A] 同一批形状上的后端对照：SDPA 四后端、FlashInfer、vLLM 自带 FA2，
      以及可选的 SageAttention —— 每个都记录**实际发射的 kernel 名**
  [B] 正确性：每个后端与 fp32 参照对拍（同形状、同 mask）
  [C] 资源矩阵：时间、峰值显存、是否可用（不可用记录原因）
  [D] 真实模型：Qwen3-1.7B 前向的 CUDA kernel 时间按类别汇总，
      看 attention 占总体多少；并核对这个模型允许哪些 mask/layout

用法：
    L3_OUT=<目录> python fa_backends.py A B C D
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness, tensor_hash                       # noqa: E402

MB = 1024 * 1024
MODEL = os.environ.get("L32_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def timeit(fn, n=20, warmup=5):
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


def kernel_names(fn, limit=3):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out, seen = [], set()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.key:
            if e.key not in seen:
                seen.add(e.key)
                out.append(e.key)
    return out[:limit]


def make(BH, S, D, Hkv=None, dtype=torch.bfloat16):
    H = BH
    q = torch.randn(1, H, S, D, device="cuda", dtype=dtype)
    if Hkv:
        k = torch.randn(1, Hkv, S, D, device="cuda", dtype=dtype)
        v = torch.randn(1, Hkv, S, D, device="cuda", dtype=dtype)
    else:
        k = torch.randn(1, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(1, H, S, D, device="cuda", dtype=dtype)
    return q, k, v


def backend_runners(q, k, v, causal=True):
    """返回 {名字: (调用函数, 是否需要 GQA 展开)}。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    runs = {}
    for name, be in [("SDPA-FLASH", SDPBackend.FLASH_ATTENTION),
                     ("SDPA-CUDNN", SDPBackend.CUDNN_ATTENTION),
                     ("SDPA-EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                     ("SDPA-MATH", SDPBackend.MATH)]:
        def mk(be=be):
            def f():
                with sdpa_kernel(be):
                    return F.scaled_dot_product_attention(q, k, v,
                                                          is_causal=causal,
                                                          enable_gqa=True)
            return f
        runs[name] = mk()

    try:
        import flashinfer
        B, H, S, D = q.shape
        Hkv = k.shape[1]
        qf = q[0].transpose(0, 1).contiguous()
        kf = k[0].transpose(0, 1).contiguous()
        vf = v[0].transpose(0, 1).contiguous()

        def fi():
            if H != Hkv:
                kk = kf.repeat_interleave(H // Hkv, dim=1)
                vv = vf.repeat_interleave(H // Hkv, dim=1)
            else:
                kk, vv = kf, vf
            return flashinfer.single_prefill_with_kv_cache(qf, kk, vv,
                                                           causal=causal)
        runs["FlashInfer"] = fi
    except Exception as exc:                                       # noqa: BLE001
        runs["FlashInfer"] = ("ERR", str(exc).splitlines()[0][:80])

    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        B, H, S, D = q.shape
        Hkv = k.shape[1]
        qf = q[0].transpose(0, 1).contiguous()
        kf = k[0].transpose(0, 1).contiguous()
        vf = v[0].transpose(0, 1).contiguous()
        cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")

        def vfa():
            # 这份 vLLM 自带实现的参数顺序与上游不同：max_seqlen 在前
            return flash_attn_varlen_func(q=qf, k=kf, v=vf, cu_seqlens_q=cu,
                                          cu_seqlens_k=cu, max_seqlen_q=S,
                                          max_seqlen_k=S, causal=causal)
        runs["vLLM-FA2"] = vfa
    except Exception as exc:                                       # noqa: BLE001
        runs["vLLM-FA2"] = ("ERR", str(exc).splitlines()[0][:80])

    try:
        from sageattention import sageattn
        runs["SageAttention"] = lambda: sageattn(q, k, v, is_causal=causal,
                                                 tensor_layout="HND")
    except Exception as exc:                                       # noqa: BLE001
        runs["SageAttention"] = ("ERR", str(exc).splitlines()[0][:100])
    return runs


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 后端对照：同形状、同 mask，各自发射了什么 kernel")

    shapes = [(8, 4096, 128, None), (16, 4096, 128, 8), (2, 8192, 64, None)]
    for BH, S, D, Hkv in shapes:
        q, k, v = make(BH, S, D, Hkv)
        print(f"\n  B·H={BH} S={S} D={D} Hkv={Hkv or BH} bf16 causal")
        runs = backend_runners(q, k, v)
        for name, fn in runs.items():
            if isinstance(fn, tuple):
                print(f"  {name:<16} 不可用：{fn[1]}")
                h.case(id=f"A_{name}_S{S}_D{D}", backend=name, status="unavailable",
                       reason=fn[1], S=S, D=D, BH=BH, Hkv=Hkv or BH)
                continue
            try:
                t = timeit(fn, n=10, warmup=3)
                kn = kernel_names(fn)
                p = peak_mb(fn)
                flops = 4.0 * BH * S * S * D / 2
                print(f"  {name:<16} {t:>8.4f} ms  {flops / t / 1e9:>7.1f} TFLOP/s  "
                      f"峰值 {p:>7.1f}MB  kernel: {kn[0][:60] if kn else '?'}")
                h.case(id=f"A_{name}_S{S}_D{D}", backend=name, status="ok", S=S,
                       D=D, BH=BH, Hkv=Hkv or BH, ms=t, tflops=flops / t / 1e9,
                       peak_mb=p, kernels=kn)
            except Exception as exc:                               # noqa: BLE001
                msg = str(exc).splitlines()[0][:100]
                print(f"  {name:<16} 失败：{msg}")
                h.case(id=f"A_{name}_S{S}_D{D}", backend=name, status="failed",
                       error=msg, S=S, D=D, BH=BH, Hkv=Hkv or BH)
                torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 正确性：每个后端与 fp32 参照对拍")

    BH, S, D, Hkv = 8, 1024, 128, 4
    q, k, v = make(BH, S, D, Hkv)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         is_causal=True, enable_gqa=True)
    runs = backend_runners(q, k, v)
    print(f"  B·H={BH} S={S} D={D} Hkv={Hkv} bf16；参照是 fp32 的 SDPA-MATH")
    print(f"  {'后端':<16} {'max|err|':>12} {'|O|max':>10} {'相对':>10} {'判定':>8}")
    for name, fn in runs.items():
        if isinstance(fn, tuple):
            print(f"  {name:<16} 不可用")
            continue
        try:
            o = fn()
            if isinstance(o, tuple):
                o = o[0]
            if o.shape != ref.shape:      # FlashInfer 返回 [S,H,D]
                o = o.permute(1, 0, 2).unsqueeze(0)
            o = o.float()
            e = (o - ref).abs().max().item()
            sc = max(1.0, ref.abs().max().item())
            print(f"  {name:<16} {e:>12.4e} {sc:>10.3f} {e / sc:>10.2e} "
                  f"{'一致' if e / sc < 1e-2 else '偏大':>8}")
            h.case(id=f"B_{name}", backend=name, max_err=e, out_scale=sc,
                   rel=e / sc, ok=e / sc < 1e-2, ref="float32 SDPA-MATH",
                   input_hash=tensor_hash(q, k, v))
        except Exception as exc:                                   # noqa: BLE001
            msg = str(exc).splitlines()[0][:90]
            print(f"  {name:<16} 失败：{msg}")
            h.case(id=f"B_{name}", backend=name, status="failed", error=msg)
            torch.cuda.empty_cache()
    del q, k, v
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] 资源矩阵：可用性、时间、峰值")

    print("  把 A/B 两节的结论合并成一张矩阵；不可用的一律记原因，不写成'不支持'。")
    print(f"  {'后端':<16} {'可用':>6} {'S=4096 BH=8 相对':>18} {'峰值 MB':>10}")
    BH, S, D = 8, 4096, 128
    q, k, v = make(BH, S, D)
    runs = backend_runners(q, k, v)
    base = None
    for name, fn in runs.items():
        if isinstance(fn, tuple):
            print(f"  {name:<16} {'否':>6} {'—':>18} {'—':>10}   {fn[1][:40]}")
            h.case(id=f"C_{name}", available=False, reason=fn[1])
            continue
        try:
            t = timeit(fn, n=10, warmup=3)
            base = base or t
            p = peak_mb(fn)
            print(f"  {name:<16} {'是':>6} {t / base:>18.2f}× {p:>10.1f}")
            h.case(id=f"C_{name}", available=True, ms=t, peak_mb=p,
                   ratio_vs_first=t / base)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {name:<16} {'否':>6} {'—':>18} {'—':>10}   "
                  f"{str(exc).splitlines()[0][:40]}")
            h.case(id=f"C_{name}", available=False,
                   reason=str(exc).splitlines()[0][:100])
            torch.cuda.empty_cache()
    del q, k, v
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D(h):
    title("[D] 真实模型：attention 占总体多少，允许哪些 mask/layout")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config
    S = 2048
    text = ("A KV cache stores the keys and values of the prefix so that decode "
            "only needs one query row per step. " * 400)
    ids = tok(text, return_tensors="pt").input_ids[:, :S].cuda()
    n = ids.shape[1]
    print(f"  {MODEL}: {cfg.num_hidden_layers} 层 Hq={cfg.num_attention_heads} "
          f"Hkv={cfg.num_key_value_heads} D={cfg.head_dim}；输入 {n} token")
    print(f"  attn_implementation = "
          f"{getattr(model.config, '_attn_implementation', '?')}")

    sub("CUDA kernel 时间按类别汇总")
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(3):
            model(ids)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(ids)
            torch.cuda.synchronize()
    agg = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.key:
            agg[e.key] = agg.get(e.key, 0.0) + e.self_device_time_total
    total = sum(agg.values())
    attn = sum(v for k, v in agg.items()
               if any(s in k.lower() for s in ("flash", "fmha", "attention",
                                               "cudnn")))
    gemm = sum(v for k, v in agg.items()
               if any(s in k.lower() for s in ("gemm", "cutlass", "nvjet",
                                               "mma", "wgrad")))
    print(f"  CUDA 总时间 {total / 1000:.3f} ms（{len(agg)} 个不同 kernel）")
    print(f"  attention 类 {attn / 1000:.3f} ms 占 {attn / total:.1%}")
    print(f"  GEMM 类      {gemm / 1000:.3f} ms 占 {gemm / total:.1%}")
    print(f"  其它         {(total - attn - gemm) / 1000:.3f} ms "
          f"占 {(total - attn - gemm) / total:.1%}")
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
    for k, t in top:
        print(f"    {t / 1000:>8.3f} ms  {k[:70]}")
    h.case(id="D_kernel_share", S=n, layers=cfg.num_hidden_layers,
           cuda_total_ms=total / 1000, attn_ms=attn / 1000,
           attn_share=attn / total, gemm_share=gemm / total,
           distinct_kernels=len(agg),
           top_kernels=[{"name": k, "ms": t / 1000} for k, t in top])
    print("\n  组件提升与整模型提升是两回事：attention 占比就是那个换算系数；")
    print("  它不是常数，随 S、batch 和 dtype 变。")

    sub("这个模型允许哪些 mask/layout")
    checks = []
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    q, k, v = make(Hq, 512, cfg.head_dim, Hkv)
    try:
        t = timeit(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True), n=10, warmup=3)
        checks.append(("GQA 原生（enable_gqa）", "可用", f"{t:.4f} ms"))
    except Exception as exc:                                       # noqa: BLE001
        checks.append(("GQA 原生（enable_gqa）", "不可用",
                       str(exc).splitlines()[0][:50]))
    # 显式 4D mask：看看会不会把 flash 后端挤掉
    S2 = 512
    mask = torch.zeros(1, 1, S2, S2, device="cuda", dtype=torch.bfloat16)
    causal = torch.ones(S2, S2, device="cuda", dtype=torch.bool).tril()
    mask = mask.masked_fill(~causal, torch.finfo(torch.bfloat16).min)
    k4 = k.repeat_interleave(Hq // Hkv, dim=1)
    v4 = v.repeat_interleave(Hq // Hkv, dim=1)
    try:
        kn = kernel_names(lambda: F.scaled_dot_product_attention(
            q, k4, v4, attn_mask=mask), limit=2)
        checks.append(("显式 4D mask", "可用", kn[0][:60] if kn else "?"))
    except Exception as exc:                                       # noqa: BLE001
        checks.append(("显式 4D mask", "不可用", str(exc).splitlines()[0][:50]))
    for name, status, detail in checks:
        print(f"  {name:<24} {status:<6} {detail}")
        h.case(id=f"D_layout_{name}", status=status, detail=detail,
               causal=True, gqa_group=Hq // Hkv, padding_mask=False)
    h.case(id="D_mask_layout", causal=True, gqa_group=Hq // Hkv,
           q_len_eq_kv_len=True, notes="真实 config 读取")
    del model
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  SM {p.multi_processor_count}")
    h = Harness("3.2-CD", "3.2", out=os.environ.get("L3_OUT"),
                backend="SDPA 四后端 / FlashInfer / vLLM FA2 / SageAttention",
                notes=f"model={MODEL}")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "同形状下各后端的 kernel 名与时间矩阵；"
                         "真实模型里 attention 占 CUDA 时间的比例已实测。",
              "model": MODEL})
    sys.stdout.flush()
    os._exit(0)
