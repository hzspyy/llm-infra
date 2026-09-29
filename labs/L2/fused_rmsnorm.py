#!/usr/bin/env python3
"""L2.3-C · add+RMSNorm 融合：分离实现 / CUDA 融合 / Triton / 引擎 op。

形状按 Qwen3-1.7B 的 hidden=14336，rows ∈ {1, 8, 128, 512}，fp16。
三件事一起报：时间与字节、与 FP64 参照的误差、以及别名语义。

    python fused_rmsnorm.py --out-json <path>
"""

import argparse
import json
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

COLS = 14336
EPS = 1e-6
ROWS_LIST = [1, 8, 128, 512, 4096]


def cuda_fused():
    """用 load_inline 现场编译一个融合 kernel（vLLM 同款语义：就地更新 residual）。"""
    from torch.utils.cpp_extension import load_inline
    root = os.environ.get("LEARN_ROOT", "/scratch/learn")
    build = os.path.join(root, ".cache", "torchext", "l23")
    os.makedirs(build, exist_ok=True)
    cuda_src = r'''
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void fused_kernel(const __half* __restrict__ x, __half* __restrict__ residual,
                             const __half* __restrict__ w, __half* __restrict__ out,
                             int rows, int cols, float eps) {
    __shared__ float s[256];
    int r = blockIdx.x;
    if (r >= rows) return;
    const __half* xr = x + (size_t)r * cols;
    __half* rr = residual + (size_t)r * cols;
    __half* orr = out + (size_t)r * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(xr[i]) + __half2float(rr[i]);
        rr[i] = __float2half(v);
        ss += v * v;
    }
    s[threadIdx.x] = ss;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    float rstd = rsqrtf(s[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        orr[i] = __float2half(__half2float(rr[i]) * rstd * __half2float(w[i]));
}

torch::Tensor fused_add_rmsnorm(torch::Tensor x, torch::Tensor residual,
                                torch::Tensor w, double eps) {
    int rows = x.size(0), cols = x.size(1);
    auto out = torch::empty_like(x);
    fused_kernel<<<rows, 256>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), rows, cols, (float)eps);
    return out;
}
'''
    mod = load_inline(name="l23_fused", cpp_sources="torch::Tensor fused_add_rmsnorm(torch::Tensor, torch::Tensor, torch::Tensor, double);",
                      cuda_sources=cuda_src, functions=["fused_add_rmsnorm"],
                      build_directory=build, verbose=False)
    return mod.fused_add_rmsnorm


def triton_fused():
    import triton
    import triton.language as tl

    @triton.jit
    def kern(x_ptr, r_ptr, w_ptr, out_ptr, cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        ss = 0.0
        for start in range(0, cols, BLOCK):
            o = start + offs
            m = o < cols
            x = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + base + o, mask=m, other=0.0).to(tl.float32)
            v = x + r
            tl.store(r_ptr + base + o, v.to(tl.float16), mask=m)
            ss += tl.sum(v * v, axis=0)
        rstd = 1.0 / tl.sqrt(ss / cols + eps)
        for start in range(0, cols, BLOCK):
            o = start + offs
            m = o < cols
            v = tl.load(r_ptr + base + o, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + o, mask=m, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + o, (v * rstd * w).to(tl.float16), mask=m)

    def run(x, residual, w, eps, out):
        kern[(x.shape[0],)](x, residual, w, out, x.shape[1], eps, BLOCK=1024, num_warps=8)

    return run


def reference_fp64(x, residual, w, eps):
    r = (x.double() + residual.double())          # residual 的更新值
    rstd = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
    return (r * rstd * w.double()).to(torch.float16), r.to(torch.float16)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    result = {"shape": {"cols": COLS, "rows": ROWS_LIST}, "dtype": "float16",
              "eps": EPS, "rows": {}}
    cuda_run = cuda_fused()
    triton_run = triton_fused()

    engine_probe = None
    try:
        import vllm._custom_ops as ops
        names = [n for n in dir(ops) if "rms" in n.lower() or "norm" in n.lower()]
        engine_probe = {"module": "vllm._custom_ops", "candidates": names}
    except Exception as exc:
        engine_probe = {"error": f"{type(exc).__name__}: {exc}"}
    result["engine_probe"] = engine_probe
    print(f"引擎 op 探测：{engine_probe}")

    for rows in ROWS_LIST:
        x = torch.randn(rows, COLS, device="cuda", dtype=torch.float16)
        r0 = torch.randn(rows, COLS, device="cuda", dtype=torch.float16)
        w = torch.ones(COLS, device="cuda", dtype=torch.float16)
        ref_out, ref_res = reference_fp64(x, r0, w, EPS)

        # 分离实现：先算 x+r（多一份中间张量），再做 RMSNorm
        def separate():
            res = x + r0
            r0.copy_(res)
            return F.rms_norm(res, (COLS,), w, EPS)
        sep_fn = lambda: separate()

        def cuda_fn():
            r = r0.clone()
            return cuda_run(x, r, w, EPS)
        def tri_fn():
            r = r0.clone()
            out = torch.empty_like(x)
            triton_run(x, r, w, EPS, out)
            return out
        def sep_fn2():
            r = r0.clone()
            res = x + r
            r.copy_(res)
            return F.rms_norm(res, (COLS,), w, EPS)

        engine_fn = None
        try:
            import vllm._custom_ops as _ops
            if "fused_add_rms_norm" in dir(_ops):
                def engine_fn():
                    inp = x.clone()
                    r = r0.clone()
                    _ops.fused_add_rms_norm(inp, r, w, EPS)   # 就地：inp 变归一化输出，r 变残差和
                    return inp
        except Exception:
            engine_fn = None

        outs = {}
        variants = [("separate", sep_fn2), ("cuda_fused", cuda_fn), ("triton_fused", tri_fn)]
        if engine_fn is not None:
            variants.append(("engine_vllm", engine_fn))
        for name, fn in variants:
            o = fn()
            err = ((o.float() - ref_out.float()).abs().max().item())
            outs[name] = {"max_abs_err": err,
                          "ms": timeit(fn),
                          "out_is_residual_alias": o.data_ptr() == r0.data_ptr()}

        # 别名语义：融合 kernel 就地更新 residual
        r = r0.clone()
        out = cuda_run(x, r, w, EPS)
        alias_ok = (r.data_ptr() != out.data_ptr()) and torch.allclose(
            r.float(), ref_res.float(), atol=2e-2, rtol=2e-3)
        inplace_ok = not torch.equal(r, r0)

        # 逐次访存账（单位：rows*COLS*2 字节）
        #   融合：读 x、读 r、写 r、再读 r、写 out            = 5
        #   分离：读 x、读 r、写中间、读中间(copy_)、再读中间(rms_norm)、写 r、写 out = 7
        bytes_fused = rows * COLS * 2 * 5 + COLS * 2
        bytes_sep = rows * COLS * 2 * 7 + COLS * 2
        rec = {"times_ms": {k: v["ms"] for k, v in outs.items()},
               "max_abs_err": {k: v["max_abs_err"] for k, v in outs.items()},
               "gbs": {k: (bytes_fused if k != "separate" else bytes_sep)
                            / (v["ms"] * 1e-3) / 1e9 for k, v in outs.items()},
               "bytes": {"fused": bytes_fused, "separate": bytes_sep},
               "residual_inplace_updated": inplace_ok,
               "out_not_alias_of_residual": alias_ok}
        result["rows"][str(rows)] = rec
        print(f"\nrows={rows}  工作集 {rows*COLS*2/1048576:.1f} MB")
        for name in [n for n, _ in variants]:
            print(f"  {name:<14} {outs[name]['ms']:8.4f} ms   "
                  f"{rec['gbs'][name]:8.1f} GB/s   max|err|={outs[name]['max_abs_err']:.3e}")
        print(f"  residual 就地更新: {inplace_ok}   out 与 residual 不同存储: {alias_ok}")
        print(f"  字节账：融合 5 次访存 {bytes_fused/1048576:.1f} MB vs 分离 7 次 {bytes_sep/1048576:.1f} MB "
              f"（分离多一份中间张量的写+两次读）")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
