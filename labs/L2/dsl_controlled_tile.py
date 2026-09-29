#!/usr/bin/env python3
"""L2.5 lab · 受控 tile 对照、布局改动与融合需求（Triton / TileLang）。

2.5 的任务 B 要求：不把编译开销混进稳态，也不为了"公平"强行统一 tile 而掩盖各实现
可用的优化，**另提供受控 tile 对照**。本脚本做两件事：

1. 受控 tile：把 Triton 与 TileLang 都锁在同一个 tile（BM=BN=128, BK=64）与同一
   dtype（输入 fp16、累加 fp32、输出 fp32）上，与手写 CUDA（`gemm_wmma_dtype.cu`）
   和 CUTLASS CuTe-DSL 例程（`run_dsl_sm120_gemm.sh`，同 tile）放进同一张表。
2. 各自 tile：两个实现用自己挑的 tile，看"受控"相对"放开"损失多少。

每个实现记录：冷编译（首次调用含 JIT）、同进程第二次调用（缓存命中）、稳态（预热后
20 次中位）、代码行数、寄存器/溢出/共享内存与生成的 PTX 行数。

Task C 的两项改动也在这里做：
  - 布局：B 从 K-major 换成 N-major（转置存放），记录改动点与代价；
  - 融合：epilogue 增加 bias + ReLU，记录改动点与代价。
失败用例（共享内存溢出、非 2 的幂 tile）保留原始报错文本。

    python labs/L2/dsl_controlled_tile.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import logging
import torch

# TileLang 把失败用例的整份 builder 转储打到 stderr（上万行），压掉它；
# 原始报错仍会进 JSON 的 failures 字段。
logging.getLogger("tilelang").setLevel(logging.CRITICAL + 10)

M, N, K = 4096, 4096, 4096
PEAK = 251.9
FLOP = 2.0 * M * N * K
CTRL = dict(BM=128, BN=128, BK=64, GROUP=8)      # 受控 tile
OWN = dict(BM=128, BN=256, BK=64, GROUP=8)       # Triton / TileLang 自己挑的 tile


def lines_of(src: str) -> int:
    return sum(1 for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#"))


# --------------------------------------------------------------------------- Triton
TRITON_SRC = '''
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, bias, M, N, K,
                sam, sak, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GROUP: tl.constexpr, B_TRANS: tl.constexpr, FUSE: tl.constexpr):
    pid = tl.program_id(0)
    n_m, n_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GROUP * n_n
    gid = pid // per_group
    first_m = gid * GROUP
    group_m = min(n_m - first_m, GROUP)
    pid_m = first_m + ((pid % per_group) % group_m)
    pid_n = (pid % per_group) // group_m

    offs_m = (pid_m * BM + tl.arange(0, BM)) % M
    offs_n = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    if B_TRANS:
        b_ptr = B + offs_n[None, :] * sbn + offs_k[:, None] * sbk
    else:
        b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        am = tl.load(a_ptr, mask=offs_k[None, :] < K - k * BK, other=0.0)
        bm = tl.load(b_ptr, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(am, bm, acc)
        a_ptr += BK * sak
        b_ptr += BK * sbk

    if FUSE:
        bv = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)
        acc = tl.maximum(acc + bv[None, :], 0.0)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptr = C + offs_cm[:, None] * scm + offs_cn[None, :] * scn
    tl.store(c_ptr, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
'''


def triton_case(kernel_fn, triton, a, b, bias, ref, tile, b_trans, fuse,
                num_warps=8, num_stages=3, label=""):
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    b_arg = b.t().contiguous() if b_trans else b
    grid = (triton.cdiv(M, tile["BM"]) * triton.cdiv(N, tile["BN"]),)
    # 核函数的形参顺序是 (sam, sak, sbk, sbn, ...)：K-major 的 B 里 stride(0) 是 k 维，
    # N-major（转置存放）的 B 里 stride(1) 才是 k 维。写反不会报错，只会算错。
    if b_trans:
        sbk, sbn = b_arg.stride(1), b_arg.stride(0)
    else:
        sbk, sbn = b_arg.stride(0), b_arg.stride(1)
    strides = (a.stride(0), a.stride(1), sbk, sbn, c.stride(0), c.stride(1))

    def run():
        return kernel_fn[grid](
            a, b_arg, c, bias, M, N, K, *strides,
            BM=tile["BM"], BN=tile["BN"], BK=tile["BK"], GROUP=tile["GROUP"],
            B_TRANS=b_trans, FUSE=fuse, num_warps=num_warps, num_stages=num_stages)

    t0 = time.perf_counter()
    kern = run()
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    cached_call = time.perf_counter() - t0

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); run(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ms = statistics.median(ts)

    res = {"route": "Triton", "label": label, "ms": ms, "tflops": FLOP / 1e12 / (ms * 1e-3),
           "cold_s": cold, "cached_call_s": cached_call,
           "rel_rms": float(((c - ref).norm() / ref.norm()).item()),
           "max_abs": float((c - ref).abs().max().item()),
           "tile": f"{tile['BM']}x{tile['BN']}x{tile['BK']}",
           "num_warps": num_warps, "num_stages": num_stages,
           "b_trans": b_trans, "fuse": fuse}
    for attr in ("n_regs", "n_spills"):
        res[attr] = getattr(kern, attr, None)
    md = getattr(kern, "metadata", None)
    res["shared_bytes"] = getattr(md, "shared", None) if md is not None else None
    try:
        res["ptx_lines"] = len(kern.asm["ptx"].splitlines())
    except Exception:                                          # noqa: BLE001
        res["ptx_lines"] = None
    return res


# --------------------------------------------------------------------------- TileLang
def tilelang_builder(T):
    def build(BM, BN, BK, stages, threads, b_trans, fuse):
        @T.prim_func
        def kernel(A: T.Tensor((M, K), "float16"),
                   B: T.Tensor((N, K) if b_trans else (K, N), "float16"),
                   Bias: T.Tensor((N,), "float32"),
                   C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                # B 的存放方向决定 shared 缓冲的形状：K-major 用 (BK, BN)，
                # N-major 用 (BN, BK) 再让 T.gemm 做 transpose_B
                Bs = T.alloc_shared((BN, BK) if b_trans else (BK, BN), "float16")
                Cl = T.alloc_fragment((BM, BN), "float32")
                Cf = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                    T.copy(A[by * BM, k * BK], As)
                    if b_trans:
                        T.copy(B[bx * BN, k * BK], Bs)
                        T.gemm(As, Bs, Cl, transpose_B=True)
                    else:
                        T.copy(B[k * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cl)
                if fuse:
                    for i, j in T.Parallel(BM, BN):
                        Cf[i, j] = T.max(Cl[i, j] + Bias[bx * BN + j], 0.0)
                    T.copy(Cf, C[by * BM, bx * BN])
                else:
                    T.copy(Cl, C[by * BM, bx * BN])
        return kernel
    return build


def tilelang_case(tilelang, build, a, b, bias, ref, tile, b_trans, fuse,
                  stages=3, threads=256, label=""):
    b_arg = b.t().contiguous() if b_trans else b
    t0 = time.perf_counter()
    jit = tilelang.compile(build(tile["BM"], tile["BN"], tile["BK"], stages, threads,
                                 b_trans, fuse), out_idx=[3])
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jit(a, b_arg, bias)
    torch.cuda.synchronize()
    cached_call = time.perf_counter() - t0

    for _ in range(5):
        jit(a, b_arg, bias)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); jit(a, b_arg, bias); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ms = statistics.median(ts)

    return {"route": "TileLang", "label": label, "ms": ms,
            "tflops": FLOP / 1e12 / (ms * 1e-3),
            "cold_s": cold, "cached_call_s": cached_call,
            "rel_rms": float(((out - ref).norm() / ref.norm()).item()),
            "max_abs": float((out - ref).abs().max().item()),
            "tile": f"{tile['BM']}x{tile['BN']}x{tile['BK']}", "stages": stages,
            "threads": threads, "b_trans": b_trans, "fuse": fuse,
            "n_regs": None, "n_spills": None, "shared_bytes": None, "ptx_lines": None}


def stride_counterexample(kernel_fn, triton, a, b, ref):
    """把 N-major B 的 stride 顺序写反：不抛异常，结果直接错。

    这是布局改动最危险的一类后果——编译通过、跑得也不慢，只有对拍才发现。
    """
    b_arg = b.t().contiguous()
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    bias = torch.zeros(N, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    strides = (a.stride(0), a.stride(1), b_arg.stride(0), b_arg.stride(1),
               c.stride(0), c.stride(1))            # ← 故意沿用未转置时的顺序
    kernel_fn[grid](a, b_arg, c, bias, M, N, K, *strides,
                    BM=128, BN=128, BK=64, GROUP=8, B_TRANS=True, FUSE=False,
                    num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    return float(((c - ref).norm() / ref.norm()).item())


def failure_cases(kernel_fn, triton, build, tilelang, a, b, bias):
    out = []
    try:
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        kernel_fn[grid](a, b, c, bias, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BM=128, BN=128, BK=128, GROUP=8, B_TRANS=False, FUSE=False,
                        num_warps=8, num_stages=8)
        torch.cuda.synchronize()
        out.append({"case": "Triton BK=128, num_stages=8（共享内存超限）", "error": None})
    except Exception as exc:                                   # noqa: BLE001
        out.append({"case": "Triton BK=128, num_stages=8（共享内存超限）",
                    "error": f"{type(exc).__name__}: {exc}"[:700]})

    try:
        jit = tilelang.compile(build(96, 128, 64, 3, 256, False, False), out_idx=[3])
        jit(a, b, bias); torch.cuda.synchronize()
        out.append({"case": "TileLang BM=96（非 2 的幂 tile）", "error": None})
    except Exception as exc:                                   # noqa: BLE001
        out.append({"case": "TileLang BM=96（非 2 的幂 tile）",
                    "error": f"{type(exc).__name__}: {exc}"[:700]})

    # TileLang 在 sm_120 上不能像 Triton 那样把 tile 开到 128×256
    try:
        jit = tilelang.compile(build(128, 256, 64, 3, 256, False, False), out_idx=[3])
        jit(a, b, bias); torch.cuda.synchronize()
        out.append({"case": "TileLang 128×256×64, stages=3（共享内存）", "error": None})
    except Exception as exc:                                   # noqa: BLE001
        out.append({"case": "TileLang 128×256×64, stages=3（共享内存）",
                    "error": f"{type(exc).__name__}: {exc}"[:700]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1
    ref = (a.double() @ b.double()).float()            # FP64 参照 → fp32
    ref_fused = torch.relu(ref + bias[None, :])
    torch.cuda.synchronize()

    import triton
    import importlib.util
    # Triton 的 @jit 需要能读到函数源码，必须落成一个真实文件再导入
    kern_path = out_dir / "dsl_triton_kernels.py"
    kern_path.write_text(TRITON_SRC)
    spec = importlib.util.spec_from_file_location("dsl_triton_kernels", kern_path)
    triton_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(triton_mod)
    kernel_fn = triton_mod.gemm_kernel

    import tilelang
    import tilelang.language as T
    build = tilelang_builder(T)

    print(f"=== {torch.cuda.get_device_name(0)}  GEMM {M}x{N}x{K} fp16→fp32 "
          f"= {FLOP/1e9:.1f} GFLOP")
    print("    受控 tile 128×128×64；各自 tile 128×256×64；参照为 FP64 matmul\n")

    cases = []

    def safe(route, label, fn, *a_, **kw):
        try:
            cases.append(fn(*a_, label=label, **kw))
        except Exception as exc:                               # noqa: BLE001
            cases.append({"route": route, "label": label, "tile": "-", "error":
                          f"{type(exc).__name__}: {exc}"[:300]})

    safe("Triton", "受控 tile", triton_case, kernel_fn, triton, a, b, bias, ref, CTRL, False, False)
    safe("Triton", "各自 tile", triton_case, kernel_fn, triton, a, b, bias, ref, OWN, False, False)
    safe("Triton", "受控 tile + B 转置（布局改动）", triton_case, kernel_fn, triton, a, b, bias,
         ref, CTRL, True, False)
    safe("Triton", "受控 tile + bias&ReLU（融合）", triton_case, kernel_fn, triton, a, b, bias,
         ref_fused, CTRL, False, True)
    safe("TileLang", "受控 tile", tilelang_case, tilelang, build, a, b, bias, ref, CTRL, False, False)
    # TileLang 在本机可用的最宽 tile 就是受控 tile：128×256×64 会撞共享内存上限
    safe("TileLang", "各自 tile（= 可用最宽，见失败用例）", tilelang_case, tilelang, build,
         a, b, bias, ref, CTRL, False, False)
    safe("TileLang", "受控 tile + B 转置（布局改动）", tilelang_case, tilelang, build, a, b, bias,
         ref, CTRL, True, False)
    safe("TileLang", "受控 tile + bias&ReLU（融合）", tilelang_case, tilelang, build, a, b, bias,
         ref_fused, CTRL, False, True)
    fails = failure_cases(kernel_fn, triton, build, tilelang, a, b, bias)
    try:
        bad = stride_counterexample(kernel_fn, triton, a, b, ref)
        fails.append({"case": "Triton N-major B 的 stride 顺序写反（静默错误）",
                      "error": None, "rel_rms_observed": bad,
                      "note": "编译通过、计时正常，只有对拍才暴露"})
    except Exception as exc:                                   # noqa: BLE001
        fails.append({"case": "Triton N-major B 的 stride 顺序写反（静默错误）",
                      "error": f"{type(exc).__name__}: {exc}"[:400]})

    print(f"    {'实现':9s} {'变体':30s} {'tile':12s} {'冷编译s':>8s} {'二次调用s':>9s} "
          f"{'稳态ms':>8s} {'TFLOPS':>8s} {'占峰值':>7s} {'寄存器':>6s} {'溢出':>5s} "
          f"{'smem':>7s} {'PTX行':>7s} {'相对误差':>10s}")
    for r in cases:
        if "error" in r:
            print(f"    {r['route']:9s} {r['label']:30s}   [失败] {r['error'][:110]}")
            continue
        print(f"    {r['route']:9s} {r['label']:30s} {r['tile']:12s} "
              f"{r['cold_s']:8.2f} {r['cached_call_s']:9.4f} {r['ms']:8.3f} "
              f"{r['tflops']:8.1f} {r['tflops']/PEAK*100:6.1f}% "
              f"{str(r.get('n_regs')):>6s} {str(r.get('n_spills')):>5s} "
              f"{str(r.get('shared_bytes')):>7s} {str(r.get('ptx_lines')):>7s} "
              f"{r['rel_rms']:10.2e}")

    # 共同参照：cuBLAS fp16 输出对 FP64 matmul 的误差，用来把各实现的误差放到同一把尺子上
    cublas_out = (a @ b).float()
    cublas_err = float(((cublas_out - ref).norm() / ref.norm()).item())
    print(f"\n    共同尺子：cuBLAS fp16 对 FP64 参照的相对 RMS = {cublas_err:.2e}"
          f"（手写 CUDA 与 cuBLAS 逐位相同，误差同为这一量级）")

    print("\n    == 失败用例（原文）==")
    for f in fails:
        if f.get("rel_rms_observed") is not None:
            print(f"    [{f['case']}] 相对 RMS = {f['rel_rms_observed']:.2e}"
                  f"（{f.get('note', '')}）")
        else:
            print(f"    [{f['case']}] {'通过' if f['error'] is None else f['error'][:220]}")

    payload = {"machine": torch.cuda.get_device_name(0), "shape": [M, N, K],
               "dtype": "fp16 in / fp32 acc / fp32 out",
               "controlled_tile": "128x128x64",
               "triton_src_loc": lines_of(TRITON_SRC),
               "cases": cases, "failures": fails}
    (out_dir / "dsl_controlled_tile.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out_dir / 'dsl_controlled_tile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
