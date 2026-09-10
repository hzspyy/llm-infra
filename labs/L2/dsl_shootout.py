#!/usr/bin/env python3
"""L2.5 lab · 同一个 bf16 GEMM，五种写法的横评。

L2.4 用手写 CUDA 把 GEMM 从 7.6 推到 144.9 TFLOPS（cuBLAS 是 218.6）。
那一路写了 400 多行 C++、调了 tile 尺寸、算了 bank 冲突的 padding。

现代 kernel DSL 声称能用几十行 Python 拿到接近的性能。本实验直接验证：
同样的 4096³ bf16 GEMM，比较**性能、代码量、编译时间**。

对照基线（均为本站实测）：
    手写 CUDA v6      144.9 TFLOPS   （L2.4，约 60 行 kernel 代码）
    cuBLAS            218.6 TFLOPS   （L2.4）
    tensor core 上限   251.9 TFLOPS   （L1.2 纯发射）

用法：python dsl_shootout.py [M N K]
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

try:                                  # Helion 要求 hl 是全局名字，不能是闭包变量
    import helion
    import helion.language as hl

    @helion.kernel(static_shapes=True)
    def helion_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        k2, n = y.size()
        out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype),
                          device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(out.dtype)
        return out
except Exception:                                             # noqa: BLE001
    helion_gemm = None

M, N, K = (int(x) for x in (sys.argv[1:4] or [4096, 4096, 4096]))
PEAK = 251.9              # L1.2 实测 bf16 tensor core 纯发射上限
FLOP = 2.0 * M * N * K

results: list[dict] = []


def bench(fn, name: str, loc: int, compile_s: float, ref=None) -> None:
    """统一的计时与验证。编译时间单独计，不混进耗时。"""
    try:
        out = fn()
        torch.cuda.synchronize()
    except Exception as exc:                                  # noqa: BLE001
        results.append({"name": name, "err": f"{type(exc).__name__}: {exc}"[:90]})
        return

    err = float("nan")
    if ref is not None:
        d = (out.float() - ref.float())
        err = (d.norm() / ref.float().norm()).item()

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ms = statistics.median(ts)
    results.append({"name": name, "ms": ms, "tflops": FLOP / 1e12 / (ms * 1e-3),
                    "loc": loc, "compile_s": compile_s, "err": err})


def main() -> None:
    print(f"=== {torch.cuda.get_device_name(0)}   GEMM {M}x{N}x{K} bf16 "
          f"= {FLOP/1e9:.1f} GFLOP")
    print(f"    对照：手写 CUDA v6 144.9 / cuBLAS 218.6 / tensor core 上限 {PEAK} TFLOPS\n")

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ref = torch.matmul(a, b)
    torch.cuda.synchronize()

    # ---------------- 0. cuBLAS 基线 ----------------
    bench(lambda: torch.matmul(a, b), "torch.matmul (cuBLAS)", 1, 0.0, ref)

    # ---------------- 1. Triton ----------------
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _triton_gemm(A, B, C, M, N, K,
                         sam, sak, sbk, sbn, scm, scn,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         GROUP: tl.constexpr):
            # L2 友好的 block 重排：让同时执行的 block 复用相同的 A/B 行列
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
            b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

            acc = tl.zeros((BM, BN), dtype=tl.float32)      # 累加器必须 fp32（L4.2）
            for k in range(0, tl.cdiv(K, BK)):
                am = tl.load(a_ptr, mask=offs_k[None, :] < K - k * BK, other=0.0)
                bm = tl.load(b_ptr, mask=offs_k[:, None] < K - k * BK, other=0.0)
                acc = tl.dot(am, bm, acc)                   # ← 一行就是 tensor core
                a_ptr += BK * sak
                b_ptr += BK * sbk

            offs_cm = pid_m * BM + tl.arange(0, BM)
            offs_cn = pid_n * BN + tl.arange(0, BN)
            c_ptr = C + offs_cm[:, None] * scm + offs_cn[None, :] * scn
            tl.store(c_ptr, acc.to(tl.bfloat16),
                     mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        BM, BN, BK, GROUP = 128, 256, 64, 8
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

        def run_triton():
            _triton_gemm[grid](a, b, c, M, N, K,
                               a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1),
                               BM=BM, BN=BN, BK=BK, GROUP=GROUP,
                               num_stages=3, num_warps=8)
            return c

        t0 = time.perf_counter()
        run_triton(); torch.cuda.synchronize()
        bench(run_triton, "Triton", 34, time.perf_counter() - t0, ref)
    except Exception as exc:                                  # noqa: BLE001
        results.append({"name": "Triton", "err": f"{type(exc).__name__}: {exc}"[:90]})

    # ---------------- 2. TileLang ----------------
    try:
        import tilelang
        import tilelang.language as T

        def tl_gemm(M, N, K, BM=128, BN=128, BK=64, stages=3, threads=256):
            @T.prim_func
            def kernel(A: T.Tensor((M, K), "bfloat16"),
                       B: T.Tensor((K, N), "bfloat16"),
                       C: T.Tensor((M, N), "bfloat16")):
                with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                    As = T.alloc_shared((BM, BK), "bfloat16")
                    Bs = T.alloc_shared((BK, BN), "bfloat16")
                    Cl = T.alloc_fragment((BM, BN), "float")
                    T.clear(Cl)
                    for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                        T.copy(A[by * BM, k * BK], As)      # 自动生成 cp.async
                        T.copy(B[k * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cl)                  # 自动选 mma 指令与布局
                    T.copy(Cl, C[by * BM, bx * BN])
            return kernel

        t0 = time.perf_counter()
        jit = tilelang.compile(tl_gemm(M, N, K), out_idx=[2])
        compile_s = time.perf_counter() - t0
        bench(lambda: jit(a, b), "TileLang", 18, compile_s, ref)
    except Exception as exc:                                  # noqa: BLE001
        results.append({"name": "TileLang", "err": f"{type(exc).__name__}: {exc}"[:90]})

    # ---------------- 3. Helion ----------------
    if helion_gemm is not None:
        try:
            t0 = time.perf_counter()
            helion_gemm(a, b)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            bench(lambda: helion_gemm(a, b), "Helion", 12, compile_s, ref)
        except Exception as exc:                              # noqa: BLE001
            results.append({"name": "Helion", "err": f"{type(exc).__name__}: {exc}"[:90]})
    else:
        results.append({"name": "Helion", "err": "导入失败"})

    # ---------------- 4. torch.compile ----------------
    try:
        f = torch.compile(lambda x, y: x @ y, mode="max-autotune-no-cudagraphs")
        t0 = time.perf_counter()
        f(a, b); torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        bench(lambda: f(a, b), "torch.compile (Inductor)", 1, compile_s, ref)
    except Exception as exc:                                  # noqa: BLE001
        results.append({"name": "torch.compile", "err": f"{type(exc).__name__}: {exc}"[:90]})

    # ---------------- 汇总 ----------------
    print(f"    {'实现':28s} {'耗时ms':>9s} {'TFLOPS':>9s} {'占上限':>8s} "
          f"{'kernel行数':>10s} {'编译s':>8s} {'相对误差':>10s}")
    for r in results:
        if "err" in r and "ms" not in r:
            print(f"    {r['name']:28s}   [失败] {r['err']}")
            continue
        print(f"    {r['name']:28s} {r['ms']:9.3f} {r['tflops']:9.1f} "
              f"{r['tflops']/PEAK*100:7.1f}% {r['loc']:>10d} {r['compile_s']:8.1f} "
              f"{r['err']:10.2e}")
    print(f"\n    参照：手写 CUDA v6 = 144.9 TFLOPS（57.5%），约 60 行 C++ kernel 代码")
    print(f"          手写 CUDA v0 =   7.6 TFLOPS（3.0%），约 8 行")


if __name__ == "__main__":
    main()
