#!/usr/bin/env python3
"""L2.5 补 · 三类算子 × 可用 DSL：冷编译与稳态分开记，并给 CuTe-DSL 一个明确结论。

算子（复用 2.3/2.4 的形状）：
  R1 RMSNorm        [rows, 14336] fp16
  R2 分段 gather+reduce（变长段求和）

实现：CUDA（load_inline）、Triton、TileLang（R1）。
每个都记：冷编译（首次调用含 JIT）、稳态、代码行数、PTX 指令行、与 torch 参照的误差。
CuTe-DSL 用最小 kernel 实测，成功就记时间，失败就记原文（不填性能）。

    python dsl_ops_extra.py --out-json <path>
"""

import argparse
import json
import os
import pathlib
import sys
import time

import torch

COLS = 14336
EPS = 1e-6
ROWS = 512


def lines_of(src):
    return sum(1 for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#"))


# ------------------------------------------------------------------ CUDA
def cuda_rmsnorm():
    from torch.utils.cpp_extension import load_inline
    build = os.path.join(os.environ.get("LEARN_ROOT", "/scratch/learn"), ".cache", "torchext", "l25")
    os.makedirs(build, exist_ok=True)
    src = r'''
#include <cuda_fp16.h>
#include <torch/extension.h>
__global__ void rms_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                           __half* __restrict__ out, int cols, float eps) {
    __shared__ float s[256];
    int r = blockIdx.x;
    const __half* xr = x + (size_t)r * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(xr[i]);
        ss += v * v;
    }
    s[threadIdx.x] = ss; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    float rstd = rsqrtf(s[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out[(size_t)r * cols + i] = __float2half(__half2float(xr[i]) * rstd * __half2float(w[i]));
}
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
    int rows = x.size(0), cols = x.size(1);
    auto out = torch::empty_like(x);
    rms_kernel<<<rows, 256>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), cols, (float)eps);
    return out;
}
'''
    mod = load_inline(name="l25_rms", cpp_sources="torch::Tensor rms_norm(torch::Tensor, torch::Tensor, double);",
                      cuda_sources=src, functions=["rms_norm"], build_directory=build, verbose=False)
    return mod.rms_norm, lines_of(src)


def cuda_seg_reduce():
    from torch.utils.cpp_extension import load_inline
    build = os.path.join(os.environ.get("LEARN_ROOT", "/scratch/learn"), ".cache", "torchext", "l25b")
    os.makedirs(build, exist_ok=True)
    src = r'''
#include <torch/extension.h>
__global__ void seg_kernel(const float* __restrict__ in, const int* __restrict__ off,
                           float* __restrict__ out, int nseg, int maxlen) {
    int seg = blockIdx.x;
    if (seg >= nseg) return;
    int b = off[seg], e = off[seg + 1];
    float acc = 0.f;
    for (int i = b + threadIdx.x; i < e; i += blockDim.x) acc += in[i];
    __shared__ float s[256];
    s[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[seg] = s[0];
}
torch::Tensor seg_reduce(torch::Tensor x, torch::Tensor off, int64_t nseg) {
    auto out = torch::empty({nseg}, x.options());
    seg_kernel<<<(int)nseg, 256>>>(x.data_ptr<float>(), off.data_ptr<int>(),
                                   out.data_ptr<float>(), (int)nseg, 0);
    return out;
}
'''
    mod = load_inline(name="l25_seg", cpp_sources="torch::Tensor seg_reduce(torch::Tensor, torch::Tensor, int64_t);",
                      cuda_sources=src, functions=["seg_reduce"], build_directory=build, verbose=False)
    return mod.seg_reduce, lines_of(src)


# ------------------------------------------------------------------ Triton
TRITON_SRC = '''
import torch, triton, triton.language as tl

@triton.jit
def rms_kernel(x_ptr, w_ptr, o_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * cols
    offs = tl.arange(0, BLOCK)
    ss = 0.0
    for start in range(0, cols, BLOCK):
        o = start + offs
        m = o < cols
        v = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        ss += tl.sum(v * v, axis=0)
    rstd = 1.0 / tl.sqrt(ss / cols + eps)
    for start in range(0, cols, BLOCK):
        o = start + offs
        m = o < cols
        v = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + o, mask=m, other=0.0).to(tl.float32)
        tl.store(o_ptr + base + o, (v * rstd * w).to(tl.float16), mask=m)

@triton.jit
def seg_kernel(in_ptr, off_ptr, o_ptr, BLOCK: tl.constexpr):
    seg = tl.program_id(0)
    b = tl.load(off_ptr + seg)
    e = tl.load(off_ptr + seg + 1)
    acc = 0.0
    for start in range(0, 8192, BLOCK):
        i = b + start + tl.arange(0, BLOCK)
        m = i < e
        v = tl.load(in_ptr + i, mask=m, other=0.0)
        acc += tl.sum(v, axis=0)
    tl.store(o_ptr + seg, acc)
'''


def triton_ops(out_dir):
    src = pathlib.Path(out_dir) / "triton_l25.py"
    src.write_text(TRITON_SRC)
    sys.path.insert(0, str(out_dir))
    import importlib
    mod = importlib.import_module("triton_l25")
    return mod, lines_of(TRITON_SRC)


# ------------------------------------------------------------------ TileLang
def tilelang_rmsnorm():
    import tilelang
    import tilelang.language as T
    src = '''
import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[2])
def rms(rows, cols, blk):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "float16"),
             W: T.Tensor((cols,), "float16"),
             O: T.Tensor((rows, cols), "float16")):
        with T.Kernel(rows, threads=256) as bx:
            x_sh = T.alloc_shared((blk,), "float16")
            ss = T.alloc_fragment((1,), "float32")
            T.clear(ss)
            for ko in T.serial(T.ceildiv(cols, blk)):
                for i in T.Parallel(blk):
                    idx = ko * blk + i
                    x_sh[i] = T.if_then_else(idx < cols, X[bx, idx], T.float16(0))
                for i in T.Parallel(blk):
                    ss[0] += T.cast(x_sh[i], "float32") * T.cast(x_sh[i], "float32")
            for ko in T.serial(T.ceildiv(cols, blk)):
                for i in T.Parallel(blk):
                    idx = ko * blk + i
                    if idx < cols:
                        O[bx, idx] = T.cast(T.cast(X[bx, idx], "float32") *
                                            T.rsqrt(ss[0] / cols + 1e-6) * T.cast(W[idx], "float32"),
                                            "float16")
    return main
'''
    return src, lines_of(src)


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def ptx_lines(mod, name):
    try:
        dev = torch.cuda.current_device()
        cache = mod.__dict__[name].device_caches[dev][0]
        kern = list(cache.values())[0]
        ptx = kern.asm.get("ptx", "")
        return sum(1 for ln in ptx.splitlines()
                   if ln.strip() and not ln.strip().startswith("//") and not ln.strip().startswith("."))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float16)
    w = torch.ones(COLS, device="cuda", dtype=torch.float16)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)).half() * w
    result = {"rmsnorm": {}, "seg_reduce": {}, "cute_dsl": {}}
    print(f"=== R1 RMSNorm [{ROWS}, {COLS}] fp16 ===")

    cuda_fn, cuda_loc = cuda_rmsnorm()
    t0 = time.perf_counter(); out = cuda_fn(x, w, EPS); torch.cuda.synchronize()
    cold = (time.perf_counter() - t0) * 1000
    steady = timeit(lambda: cuda_fn(x, w, EPS))
    err = (out.float() - ref.float()).abs().max().item()
    result["rmsnorm"]["cuda"] = {"cold_ms": cold, "steady_ms": steady, "loc": cuda_loc, "max_err": err}
    print(f"  CUDA     冷 {cold:8.1f} ms  稳态 {steady:7.4f} ms  LOC {cuda_loc:3d}  max|err| {err:.3e}")

    mod, tri_loc = triton_ops(out_dir)
    o = torch.empty_like(x)
    t0 = time.perf_counter()
    mod.rms_kernel[(ROWS,)](x, w, o, COLS, EPS, BLOCK=1024, num_warps=8); torch.cuda.synchronize()
    cold = (time.perf_counter() - t0) * 1000
    steady = timeit(lambda: mod.rms_kernel[(ROWS,)](x, w, o, COLS, EPS, BLOCK=1024, num_warps=8))
    err = (o.float() - ref.float()).abs().max().item()
    result["rmsnorm"]["triton"] = {"cold_ms": cold, "steady_ms": steady, "loc": tri_loc,
                                   "max_err": err, "ptx_lines": ptx_lines(mod, "rms_kernel")}
    print(f"  Triton   冷 {cold:8.1f} ms  稳态 {steady:7.4f} ms  LOC {tri_loc:3d}  "
          f"PTX 行 {result['rmsnorm']['triton']['ptx_lines']:4d}  max|err| {err:.3e}")

    try:
        src, tl_loc = tilelang_rmsnorm()
        tl_path = out_dir / "tl_l25.py"
        tl_path.write_text(src)
        import importlib
        sys.path.insert(0, str(out_dir))
        tl_mod = importlib.import_module("tl_l25")
        ns = {"rms": tl_mod.rms}
        t0 = time.perf_counter()
        tl_out = ns["rms"](ROWS, COLS, 1024)(x, w)
        torch.cuda.synchronize()
        cold = (time.perf_counter() - t0) * 1000
        steady = timeit(lambda: ns["rms"](ROWS, COLS, 1024)(x, w))
        err = (tl_out.float() - ref.float()).abs().max().item()
        result["rmsnorm"]["tilelang"] = {"cold_ms": cold, "steady_ms": steady, "loc": tl_loc, "max_err": err}
        print(f"  TileLang 冷 {cold:8.1f} ms  稳态 {steady:7.4f} ms  LOC {tl_loc:3d}  max_err {err:.3e}")
    except Exception as exc:
        result["rmsnorm"]["tilelang"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(f"  TileLang 失败：{type(exc).__name__}: {str(exc)[:160]}")

    print(f"\n=== R2 分段 gather+reduce（512 段，变长 1..4096）===")
    lens = (torch.arange(512) % 4096 + 1).to(torch.int32)
    total = int(lens.sum())
    data = torch.randn(int(total), device="cuda")
    off = torch.zeros(513, dtype=torch.int32, device="cuda")
    off[1:] = torch.cumsum(lens, 0)
    lens_dev = lens.to("cuda")
    ref_seg = torch.segment_reduce(data, "sum", lengths=lens_dev) if hasattr(torch, "segment_reduce") else \
        torch.stack([data[int(off[i]):int(off[i + 1])].sum() for i in range(512)])
    seg_fn, seg_loc = cuda_seg_reduce()
    t0 = time.perf_counter(); o1 = seg_fn(data, off, 512); torch.cuda.synchronize()
    cold = (time.perf_counter() - t0) * 1000
    steady = timeit(lambda: seg_fn(data, off, 512))
    err = (o1 - ref_seg).abs().max().item()
    result["seg_reduce"]["cuda"] = {"cold_ms": cold, "steady_ms": steady, "loc": seg_loc, "max_err": err}
    print(f"  CUDA     冷 {cold:8.1f} ms  稳态 {steady:7.4f} ms  LOC {seg_loc:3d}  max|err| {err:.3e}")

    o2 = torch.empty(512, device="cuda")
    t0 = time.perf_counter()
    mod.seg_kernel[(512,)](data, off, o2, BLOCK=1024); torch.cuda.synchronize()
    cold = (time.perf_counter() - t0) * 1000
    steady = timeit(lambda: mod.seg_kernel[(512,)](data, off, o2, BLOCK=1024))
    err = (o2 - ref_seg).abs().max().item()
    result["seg_reduce"]["triton"] = {"cold_ms": cold, "steady_ms": steady, "max_err": err}
    print(f"  Triton   冷 {cold:8.1f} ms  稳态 {steady:7.4f} ms  max|err| {err:.3e}")

    print("\n=== CuTe-DSL：最小 kernel 的结论 ===")
    # 正确写法（2026-09 复核）：host 函数不接收 stream，launch 走当前流；
    # 运行前必须把 CUDA_TOOLKIT_PATH 指向 CUDA 安装根，否则 libNVVM 报后端失败。
    cute_src = '''
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N = 4096

@cute.kernel
def add_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk, _, _ = cute.arch.block_dim()
    i = bidx * blk + tidx
    if i < N:
        gC[i] = gA[i] + gB[i]

@cute.jit
def add_host(a, b, c):
    blk = 128
    add_kernel(a, b, c).launch(grid=((N + blk - 1) // blk, 1, 1), block=(blk, 1, 1))

if __name__ == "__main__":
    a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda")
    c = torch.zeros(N, device="cuda")
    add_host(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()
    print("ok", (c - (a + b)).abs().max().item())
'''
    src_path = out_dir / "cute_min.py"
    src_path.write_text(cute_src)
    import subprocess
    r = subprocess.run([sys.executable, str(src_path)], capture_output=True, text=True, cwd=out_dir)
    result["cute_dsl"] = {"returncode": r.returncode,
                          "stdout": r.stdout.strip()[-400:],
                          "stderr": r.stderr.strip()[-400:]}
    if r.returncode == 0:
        print(f"  成功：{r.stdout.strip()[-200:]}")
        print("  注：RMSNorm 与分段归约没有 CuTe-DSL 版本；受控 tile 的 CuTe-DSL GEMM 见"
              " labs/L2/run_controlled_tile.sh")
    else:
        print(f"  失败（原文）：{r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else '?'}")
        print("  ⇒ 记录原文；本脚本不给 CuTe-DSL 性能数字。")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
