#!/usr/bin/env python3
"""L4.3 修订（任务 C）—— 真实 shape 上的量化线性层对照。

对照路线（同一份权重、同一输入，比数值与时间）：
  1. BF16 权重 + BF16 激活（基线）
  2. INT4 权重显式反量化 → BF16 GEMM（非融合参照）
  3. INT4 权重 Marlin 融合 kernel（W4A16，vLLM apply_gptq_marlin_linear）
  4. FP8 权重 + FP8 激活（W8A8，cutlass_scaled_mm）
  5. NVFP4 权重 + NVFP4 激活（W4A4，cutlass_scaled_fp4_mm，若硬件支持）
消融：
  - 形状：Qwen3-1.7B 的 q_proj [2048,2048] 与 down_proj [2048,6144]，M=1/8/32/128/512/2048
  - 驻留 vs 大于 L2：在 8 份权重上轮转，使工作集超过 L2
  - 在线旋转：Hadamard 旋转后再量化（against identity），比较输出误差与额外成本

用法：
    python labs/L4/quantized_linear_bench.py --outdir out/4.3/run
"""

import argparse
import json
import math
import os
import sys
import time

import torch

SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def rel_err(got, ref):
    return ((got.double() - ref.double()).norm() / ref.double().norm()).item()


# ---------------- INT4（对称，GPTQ 存储：q+8 ∈ [1,15]） ----------------
def int4_quant(W, group_size=128):
    """W [N,K] -> (codes [N,K] in 1..15, scales [ng,N] fp16) 供 marlin 使用。"""
    N, K = W.shape
    g = W.reshape(N, K // group_size, group_size)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
    q = torch.round(g / scale).clamp(-7, 7)
    return (q.reshape(N, K) + 8).to(torch.int32), scale.squeeze(-1).t().contiguous().half()


def int4_dequant(codes, scales, group_size=128):
    N, K = codes.shape
    q = (codes - 8).reshape(N, K // group_size, group_size).float()
    s = scales.t().reshape(N, K // group_size, 1).float()
    return (q * s).reshape(N, K)


def gptq_pack_int4(codes):
    """codes [K,N] in 0..15 -> qweight [K/8, N] int32（沿输入维打包）。"""
    qi = codes.t().contiguous()
    qq = qi.reshape(qi.shape[0] // 8, 8, qi.shape[1]).permute(0, 2, 1)
    sh = torch.arange(0, 32, 4, device=qi.device)
    return (qq << sh).sum(-1).to(torch.int32)


def build_marlin(W, group_size=128):
    """按 vLLM marlin.py 的步骤把 INT4 权重建到执行布局。"""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx, marlin_pad_qweight, marlin_pad_scales,
        marlin_padded_nk, marlin_permute_scales)
    from vllm.scalar_type import scalar_types
    N, K = W.shape
    codes, scales = int4_quant(W, group_size)
    qweight = gptq_pack_int4(codes.cuda())                     # [K/8, N]
    padded_n, padded_k = marlin_padded_nk(N, K, group_size)
    dev = W.device
    perm = marlin_make_empty_g_idx(dev)
    q_pad = marlin_pad_qweight(qweight, N, K, padded_n, padded_k)
    q_marlin = ops.gptq_marlin_repack(q_pad, perm=perm, size_k=padded_k,
                                      size_n=padded_n, num_bits=4)
    s = marlin_permute_scales(
        marlin_pad_scales(scales.contiguous(), N, K, padded_n, padded_k, group_size),
        size_k=padded_k, size_n=padded_n, group_size=group_size)
    return {"qweight": q_marlin, "scales": s,
            "wtype": scalar_types.uint4b8,
            "zp": marlin_make_empty_g_idx(dev), "gidx": marlin_make_empty_g_idx(dev),
            "padded_nk": (padded_n, padded_k), "codes": codes, "scales_raw": scales}


def marlin_linear(x, mq, K, N):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear, marlin_make_workspace_new)
    ws = marlin_make_workspace_new(x.device)
    return apply_gptq_marlin_linear(
        input=x, weight=mq["qweight"], weight_scale=mq["scales"],
        weight_zp=mq["zp"], g_idx=mq["gidx"], g_idx_sort_indices=mq["gidx"],
        workspace=ws, wtype=mq["wtype"], input_size_per_partition=K,
        output_size_per_partition=N, is_k_full=True, bias=None,
        input_dtype=torch.bfloat16)


# ---------------- FP8 / NVFP4 ----------------
def fp8_prepare(W):
    """把权重预先量化成 fp8（离线成本，不计入 kernel 计时）。"""
    sw = W.abs().amax().float() / 448.0
    return {"wq": (W / sw).to(torch.float8_e4m3fn).t(), "sw": sw.reshape(1)}


def fp8_kernel(xq, sx, wp):
    """只算 kernel：激活已量化。"""
    from vllm import _custom_ops as ops
    return ops.cutlass_scaled_mm(xq, wp["wq"], scale_a=sx.reshape(1),
                                 scale_b=wp["sw"], out_dtype=torch.bfloat16)


def fp8_online(x, wp):
    """kernel + 在线激活量化（每次调用都要算 amax 与转换）。"""
    sx = x.abs().amax().float() / 448.0
    xq = (x / sx).to(torch.float8_e4m3fn)
    return fp8_kernel(xq, sx, wp)


def nvfp4_prepare(W):
    from vllm import _custom_ops as ops
    g = torch.tensor([1.0], device=W.device)
    wq, ws = ops.scaled_fp4_quant(W, g)
    return {"wq": wq, "ws": ws, "g": g}


def nvfp4_kernel(xq, xs, wp):
    from vllm import _custom_ops as ops
    alpha = torch.tensor(1.0, device=xq.device)
    return ops.cutlass_scaled_fp4_mm(xq, wp["wq"], xs, wp["ws"], alpha, torch.bfloat16)


def nvfp4_online(x, wp):
    from vllm import _custom_ops as ops
    xq, xs = ops.scaled_fp4_quant(x, wp["g"])
    return nvfp4_kernel(xq, xs, wp)


def nvfp4_quant(x, wp):
    from vllm import _custom_ops as ops
    return ops.scaled_fp4_quant(x, wp["g"])


def hadamard(n, device):
    """简单的 Hadamard 旋转（Sylvester 构造，n 必须是 2 的幂）。"""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/l43_bench"))
    ap.add_argument("--ms", default="1,8,32,128,512,2048")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--l2-copies", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda"
    title("[C] 真实 shape 的量化线性层：数值与时间")
    print(f"  torch {torch.__version__}  {torch.cuda.get_device_name(0)}  "
          f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB")
    shapes = {"q_proj": (2048, 2048), "down_proj": (2048, 6144)}
    ms = [int(v) for v in args.ms.split(",")]
    rows = []

    for name, (N, K) in shapes.items():
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        print(f"\n  === {name}  N={N} K={K}  权重 {N * K * 2 / 2**20:.1f} MiB（bf16）")
        mq = None
        try:
            mq = build_marlin(W, args.group)
            print(f"  marlin 布局：qweight {tuple(mq['qweight'].shape)} "
                  f"scales {tuple(mq['scales'].shape)} padded {mq['padded_nk']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  marlin 构建失败：{type(e).__name__}: {str(e)[:160]}")
        fp8w = None
        nv4w = None
        try:
            fp8w = fp8_prepare(W)
        except Exception as e:
            print(f"  FP8 权重预量化失败：{type(e).__name__}: {str(e)[:100]}")
        try:
            nv4w = nvfp4_prepare(W)
        except Exception as e:
            print(f"  NVFP4 权重预量化失败：{type(e).__name__}: {str(e)[:100]}")
        for M in ms:
            x = (torch.randn(M, K, device=dev) * 1.0).to(torch.bfloat16)
            ref = (x.double() @ W.double().t())
            res = {}
            t = bench(lambda: torch.matmul(x, W.t()))
            res["bf16"] = {"ms": t * 1e3, "err": rel_err(torch.matmul(x, W.t()), ref)}
            xd = int4_dequant(mq["codes"], mq["scales_raw"], args.group).to(torch.bfloat16)
            t = bench(lambda: torch.matmul(x, xd.t()))
            res["int4_dequant_bf16"] = {"ms": t * 1e3,
                                        "err": rel_err(torch.matmul(x, xd.t()), ref)}
            if mq is not None:
                try:
                    y = marlin_linear(x, mq, K, N)
                    t = bench(lambda: marlin_linear(x, mq, K, N))
                    res["int4_marlin_w4a16"] = {"ms": t * 1e3, "err": rel_err(y, ref)}
                except Exception as e:
                    res["int4_marlin_w4a16"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            if fp8w is not None:
                try:
                    xq = (x / (x.abs().amax().float() / 448.0)).to(torch.float8_e4m3fn)
                    sx = x.abs().amax().float() / 448.0
                    y = fp8_kernel(xq, sx, fp8w)
                    t_k = bench(lambda: fp8_kernel(xq, sx, fp8w))
                    t_o = bench(lambda: fp8_online(x, fp8w))
                    res["fp8_w8a8_kernel"] = {"ms": t_k * 1e3, "err": rel_err(y, ref)}
                    res["fp8_w8a8_online"] = {"ms": t_o * 1e3, "err": rel_err(fp8_online(x, fp8w), ref)}
                except Exception as e:
                    res["fp8_w8a8_kernel"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            if nv4w is not None:
                try:
                    xq4, xs4 = nvfp4_quant(x, nv4w)
                    y = nvfp4_kernel(xq4, xs4, nv4w)
                    t_k = bench(lambda: nvfp4_kernel(xq4, xs4, nv4w))
                    t_o = bench(lambda: nvfp4_online(x, nv4w))
                    res["nvfp4_w4a4_kernel"] = {"ms": t_k * 1e3, "err": rel_err(y, ref)}
                    res["nvfp4_w4a4_online"] = {"ms": t_o * 1e3, "err": rel_err(nvfp4_online(x, nv4w), ref)}
                except Exception as e:
                    res["nvfp4_w4a4_kernel"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            rows.append({"shape": name, "N": N, "K": K, "M": M, **res})
            print(f"  M={M:<5} " + "  ".join(
                f"{k} {v.get('ms', float('nan')):6.3f}ms/{v.get('err', float('nan')):.1e}"
                if "error" not in v else f"{k} 失败"
                for k, v in res.items()))
            del x, ref
            torch.cuda.empty_cache()

    sub("C2 驻留 vs 大于 L2：权重轮转")
    N, K = shapes["down_proj"]
    copies = []
    for i in range(args.l2_copies):
        Wi = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        copies.append(Wi)
    total_mib = sum(w.numel() * 2 for w in copies) / 2**20
    print(f"  {len(copies)} 份 down_proj 权重，合计 {total_mib:.0f} MiB "
          f"(L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB)")
    M = 8
    x = (torch.randn(M, K, device=dev)).to(torch.bfloat16)

    def rotate_bf16():
        for w in copies:
            torch.matmul(x, w.t())

    t_rot = bench(rotate_bf16, warmup=2, iters=10)
    one = bench(lambda: torch.matmul(x, copies[0].t()))
    print(f"  单份（驻留）{one * 1e3:.3f} ms/次；{len(copies)} 份轮转 "
          f"{t_rot * 1e3:.3f} ms/轮（平均 {t_rot / len(copies) * 1e3:.3f} ms/份，"
          f"{t_rot / one:.2f}× 单份）")
    SUMMARY["l2"] = {"copies": len(copies), "total_mib": total_mib,
                     "single_ms": one * 1e3, "rotate_ms": t_rot * 1e3,
                     "ratio_per_copy": t_rot / len(copies) / one}

    sub("C3 在线旋转消融：identity vs Hadamard（FP8 W8A8 与 NVFP4）")
    N3, K3 = 2048, 2048
    W3 = (torch.randn(N3, K3, device=dev) * 0.02)
    W3[::64, ::64] *= 8.0            # 权重里的离群点（MR-GPTQ 针对的对象）
    W3 = W3.to(torch.bfloat16)
    x3 = torch.randn(64, K3, device=dev).to(torch.bfloat16)
    x3[:, ::16] *= 20.0              # 激活里的离群通道（QuaRot 针对的对象）
    ref3 = x3.double() @ W3.double().t()
    H = hadamard(K3, dev)
    # 先验证旋转的数学等价性（FP64）
    eq = ((x3.double() @ H.double()) @ (W3.double() @ H.double()).t()
          - x3.double() @ W3.double().t()).abs().max().item()
    print(f"  旋转等价性检查（FP64）：max|(xH)(WH)ᵀ − xWᵀ| = {eq:.3e}")
    rot = []
    t0 = time.perf_counter()
    Wr = (W3.float() @ H).to(torch.bfloat16)
    xr = (x3.float() @ H).to(torch.bfloat16)
    rot_s = time.perf_counter() - t0
    fp8w_id, fp8w_rot = fp8_prepare(W3), fp8_prepare(Wr)
    nv4w_id, nv4w_rot = nvfp4_prepare(W3), nvfp4_prepare(Wr)
    for tag, (xx, ww, w8, w4) in (("identity", (x3, W3, fp8w_id, nv4w_id)),
                                  ("hadamard", (xr, Wr, fp8w_rot, nv4w_rot))):
        e8 = rel_err(fp8_online(xx, w8), ref3)
        y4 = rel_err(nvfp4_online(xx, w4), ref3)
        rot.append({"variant": tag, "fp8_err": e8, "nvfp4_err": y4,
                    "x_absmax": xx.abs().max().item(),
                    "w_absmax": ww.abs().max().item()})
        print(f"  {tag:<10} FP8 总 {e8:.4e}  NVFP4 总 {y4:.4e}  "
              f"|x|max {xx.abs().max().item():.2f}  |w|max {ww.abs().max().item():.3f}")
    print(f"  旋转本身耗时 {rot_s * 1e3:.3f} ms（一次性，可离线折进权重）")
    SUMMARY["rotation"] = {"rows": rot, "rotate_ms": rot_s * 1e3}
    SUMMARY["rows"] = rows
    SUMMARY["config"] = {"ms": ms, "group_size": args.group,
                         "shapes": {k: list(v) for k, v in shapes.items()},
                         "gpu": torch.cuda.get_device_name(0),
                         "l2_mib": torch.cuda.get_device_properties(0).L2_cache_size / 2**20}
    path = os.path.join(args.outdir, "quantized_linear_bench.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
