#!/usr/bin/env python3
"""GPU 上的低精度 GEMM 实测：TF32、FP8 三种 scaling、MXFP8 与 NVFP4 的支持边界。

四段内容：
  A TF32        FP32 存储下三种 matmul 精度模式的实际误差
  B FP8         tensorwise / rowwise 的数值与约束，越界时保留真实报错
  C 块缩放      BlockWise1x32(MXFP8) 与 BlockWise1x16(NVFP4) 在本机的支持情况
  D 计时        BF16 与 FP8 的大 GEMM 对照，含量化开销

判据是相对 FP64 参照的 SQNR 与真实报错原文；计时用 CUDA event，10 次预热、
5 轮交错、每轮 20 次。

Usage:
    python labs/L7/fp8_gemm_probe.py --outdir "$RUN_DIR/fp8"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

E4M3 = torch.float8_e4m3fn


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    err = (ref - got).pow(2).sum()
    return float("inf") if err == 0 else 10 * math.log10((ref.pow(2).sum() / err).item())


def make(m: int, k: int, n: int, seed: int = 0, b_col_major: bool = True):
    """b_col_major=True 时 b.stride(0)==1，这是 rowwise/blockwise kernel 的布局要求。"""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float32)
    if b_col_major:
        b = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.float32).t()
    else:
        b = torch.randn(k, n, generator=gen, device="cuda", dtype=torch.float32)
    return a, b


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """把 (H, W) 的块 scale 重排成 cuBLAS 要求的 32×4×4 swizzle 布局。

    与 torchao.prototype.mx_formats.utils.to_blocked 一致，依据是 cuBLAS 文档的
    "block scaling factors layout"。
    """
    rows, cols = scales.shape
    n_row_blocks, n_col_blocks = -(-rows // 128), -(-cols // 4)
    padded_rows, padded_cols = n_row_blocks * 128, n_col_blocks * 4
    padded = scales
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=scales.device,
                             dtype=scales.dtype)
        padded[:rows, :cols] = scales
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def to_fp8_tensorwise(t):
    amax = t.abs().amax().float()
    scale = (torch.finfo(E4M3).max / amax.clamp(min=1e-12)).float()
    return (t.float() * scale).to(E4M3), (1.0 / scale).reshape(1, 1)


def to_fp8_rowwise(t, dim: int):
    amax = t.abs().amax(dim=dim, keepdim=True).float()
    scale = (torch.finfo(E4M3).max / amax.clamp(min=1e-12)).float()
    scale = torch.exp2(torch.floor(torch.log2(scale)))
    return (t.float() * scale).to(E4M3), (1.0 / scale)


def timed(fn, iters: int = 20, rounds: int = 5, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best.append(start.elapsed_time(end) / iters)
    best.sort()
    return best[len(best) // 2]


def section_a(report: dict) -> None:
    head("A TF32：FP32 存储，三种 matmul 精度模式")
    a, b = make(1024, 1024, 1024)
    ref = a.double() @ b.double()
    rows = []
    for mode in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(mode)
        got = a @ b
        rows.append({"mode": mode, "out_dtype": str(got.dtype),
                     "sqnr_db": round(sqnr_db(ref, got), 2),
                     "max_abs_err": float((got.double() - ref).abs().max())})
        print(f"  precision={mode:<8} 输出 dtype={got.dtype} "
              f"SQNR={rows[-1]['sqnr_db']:>7.2f} dB  最大绝对误差={rows[-1]['max_abs_err']:.3e}")
    torch.set_float32_matmul_precision("highest")
    bf = (a.bfloat16() @ b.bfloat16())
    print(f"  BF16 输入对照         输出 dtype={bf.dtype} SQNR={sqnr_db(ref, bf):>7.2f} dB")
    print("  三种模式的张量都以 FP32 存储；改变的是乘法输入被截到多少位尾数。")
    report["tf32"] = rows


def section_b(report: dict) -> None:
    head("B FP8 scaled_mm：tensorwise 与 rowwise")
    m, k, n = 512, 1024, 256
    a, b = make(m, k, n)
    ref = a.double() @ b.double()
    rows = []

    qa_t, sa_t = to_fp8_tensorwise(a)
    qb_t, sb_t = to_fp8_tensorwise(b)
    for fast in (False, True):
        try:
            out = scaled_mm(qa_t, qb_t, sa_t, ScalingType.TensorWise,
                            sb_t, ScalingType.TensorWise,
                            output_dtype=torch.bfloat16, use_fast_accum=fast)
            rec = {"recipe": "TensorWise", "fast_accum": fast,
                   "sqnr_db": round(sqnr_db(ref, out), 2), "error": None}
        except Exception as exc:
            rec = {"recipe": "TensorWise", "fast_accum": fast,
                   "sqnr_db": None, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(f"  TensorWise fast_accum={str(fast):<5} → "
              f"{rec['sqnr_db'] if rec['error'] is None else rec['error']}")

    qa_r, sa_r = to_fp8_rowwise(a, dim=-1)          # 沿 K
    qb_r, sb_r = to_fp8_rowwise(b, dim=0)           # 沿 K
    try:
        out = scaled_mm(qa_r, qb_r, sa_r, ScalingType.RowWise,
                        sb_r, ScalingType.RowWise, output_dtype=torch.bfloat16)
        rec = {"recipe": "RowWise", "sqnr_db": round(sqnr_db(ref, out), 2), "error": None}
        print(f"  RowWise                     → SQNR={rec['sqnr_db']:.2f} dB")
    except Exception as exc:
        rec = {"recipe": "RowWise", "sqnr_db": None,
               "error": f"{type(exc).__name__}: {exc}"}
        print(f"  RowWise                     → {rec['error']}")
    rows.append(rec)

    print("\n  故意违反约束，保留真实报错：")
    probes = []
    bad_k = 1000                                    # 不是 16 的倍数
    a2, b2 = make(m, bad_k, n, seed=1)
    qa2, sa2 = to_fp8_tensorwise(a2)
    qb2, sb2 = to_fp8_tensorwise(b2)
    probes.append(("K=1000 不是 16 的倍数",
                   lambda: scaled_mm(qa2, qb2, sa2, ScalingType.TensorWise,
                                     sb2, ScalingType.TensorWise,
                                     output_dtype=torch.bfloat16)))
    a3, b3 = make(m, k, n, seed=1, b_col_major=False)     # b.stride(0)=n
    qa3, sa3 = to_fp8_rowwise(a3, -1)
    qb3, sb3 = to_fp8_rowwise(b3, 0)
    probes.append(("rowwise + mat_b 行主序",
                   lambda: scaled_mm(qa3, qb3, sa3, ScalingType.RowWise,
                                     sb3, ScalingType.RowWise,
                                     output_dtype=torch.bfloat16)))
    probes.append(("tensorwise + mat_b 行主序",
                   lambda: scaled_mm(qa3, qb3, sa3.amax().reshape(1, 1),
                                     ScalingType.TensorWise,
                                     sb3.amax().reshape(1, 1), ScalingType.TensorWise,
                                     output_dtype=torch.bfloat16)))
    probes.append(("output_dtype=float64",
                   lambda: scaled_mm(qa_t, qb_t, sa_t, ScalingType.TensorWise,
                                     sb_t, ScalingType.TensorWise,
                                     output_dtype=torch.float64)))
    for name, fn in probes:
        try:
            fn()
            msg = "通过"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        print(f"    {name:<24} {msg}")
        report.setdefault("constraint_probes", []).append({"case": name, "result": msg})
    report["fp8"] = rows


def section_c(report: dict) -> None:
    head("C 块缩放：BlockWise1x32（MXFP8）与 BlockWise1x16（NVFP4）")
    m, k, n = 512, 1024, 256
    a, b = make(m, k, n, seed=2)
    ref = a.double() @ b.double()
    rows = []

    def to_mx(t, block=32):
        """沿最后一维每 32 个元素一个 E8M0 scale，返回 E4M3 数据与未 swizzle 的 scale。"""
        rows_, cols_ = t.shape
        blocks = t.float().reshape(-1, block)
        amax = blocks.abs().amax(-1, keepdim=True)
        exp = torch.floor(torch.log2(amax.clamp(min=1e-30))) - 8
        q = (blocks / torch.exp2(exp)).to(E4M3).reshape(rows_, cols_)
        e8m0 = (exp.reshape(rows_, cols_ // block) + 127).clamp(0, 254).to(torch.uint8)
        return q, e8m0.view(torch.float8_e8m0fnu)

    qa, sa = to_mx(a)
    bt = b.t().contiguous()                          # (n, k)：沿 K 连续，块也沿 K
    qbt, sbt = to_mx(bt)
    for swizzle in (SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4):
        scale_a = sa if swizzle is SwizzleType.NO_SWIZZLE else to_blocked(sa)
        scale_b = sbt if swizzle is SwizzleType.NO_SWIZZLE else to_blocked(sbt)
        try:
            out = scaled_mm(qa, qbt.t(), scale_a, ScalingType.BlockWise1x32,
                            scale_b, ScalingType.BlockWise1x32,
                            swizzle_a=swizzle, swizzle_b=swizzle,
                            output_dtype=torch.bfloat16)
            rec = {"recipe": "BlockWise1x32", "swizzle": str(swizzle).split(".")[-1],
                   "sqnr_db": round(sqnr_db(ref, out), 2), "error": None}
            print(f"  MXFP8 {rec['swizzle']:<18} → SQNR={rec['sqnr_db']:.2f} dB")
        except Exception as exc:
            rec = {"recipe": "BlockWise1x32", "swizzle": str(swizzle).split(".")[-1],
                   "sqnr_db": None,
                   "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}
            rows.append(rec)
            print(f"  MXFP8 {rec['swizzle']:<18} → {rec['error']}")
            continue
        rows.append(rec)

    # NVFP4：E2M1 打包成每字节两个值，块 scale 为 E4M3，外加 per-tensor FP32
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")

    def to_nvfp4(t, block=16):
        rows_, cols_ = t.shape
        blocks = t.float().reshape(-1, block)
        block_scale = blocks.abs().amax(-1, keepdim=True) / 6.0
        per_tensor = block_scale.amax().clamp(min=1e-12)
        s_fp8 = (block_scale / per_tensor).clamp(min=2 ** -9, max=448.0).to(E4M3)
        total = s_fp8.float() * per_tensor
        scaled = blocks / total.clamp(min=1e-30)
        idx = (scaled.abs().unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
        code = idx | (scaled < 0).to(torch.uint8) * 8
        code = code.reshape(rows_, cols_)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
        return (packed.view(torch.float4_e2m1fn_x2),
                s_fp8.reshape(rows_, cols_ // block), per_tensor)

    fa, sa4, pa = to_nvfp4(a)
    fbt, sb4, pb = to_nvfp4(bt)
    try:
        out = scaled_mm(fa, fbt.t(), to_blocked(sa4), ScalingType.BlockWise1x16,
                        to_blocked(sb4), ScalingType.BlockWise1x16,
                        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                        output_dtype=torch.bfloat16)
        out = out.float() * (pa * pb)                # 两级 scale 的外层在 kernel 之外还原
        rec = {"recipe": "BlockWise1x16", "sqnr_db": round(sqnr_db(ref, out), 2),
               "error": None}
        print(f"  NVFP4 SWIZZLE_32_4_4  → SQNR={rec['sqnr_db']:.2f} dB")
    except Exception as exc:
        rec = {"recipe": "BlockWise1x16", "sqnr_db": None,
               "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}
        print(f"  NVFP4 SWIZZLE_32_4_4  → {rec['error']}")
    rows.append(rec)
    print("  块 scale 必须按 cuBLAS 的 32×4×4 布局重排；直接传扁平 scale 会被拒绝或算错。")
    report["blockwise"] = rows


def section_d(report: dict) -> None:
    head("D 计时：同一个 GEMM 的四条路径，以及量化本身的代价")
    rows = []
    for size in (1024, 4096):
        m = k = n = size
        a, b = make(m, k, n, seed=3)
        ab, bb = a.bfloat16(), b.bfloat16()
        qa_t, sa_t = to_fp8_tensorwise(a)
        qb_t, sb_t = to_fp8_tensorwise(b)
        qa_r, sa_r = to_fp8_rowwise(a, -1)
        qb_r, sb_r = to_fp8_rowwise(b, 0)
        flops = 2 * m * k * n
        entries = [
            ("BF16 mm", lambda: ab @ bb, True),
            ("FP8 tensorwise", lambda: scaled_mm(qa_t, qb_t, sa_t, ScalingType.TensorWise,
                                                 sb_t, ScalingType.TensorWise,
                                                 output_dtype=torch.bfloat16), True),
            ("FP8 rowwise", lambda: scaled_mm(qa_r, qb_r, sa_r, ScalingType.RowWise,
                                              sb_r, ScalingType.RowWise,
                                              output_dtype=torch.bfloat16), True),
            ("量化 A 一次(rowwise)", lambda: to_fp8_rowwise(a, -1), False),
        ]
        print(f"\n  M=N=K={size}")
        for name, fn, is_gemm in entries:
            try:
                ms = timed(fn)
                tflops = flops / (ms * 1e-3) / 1e12 if is_gemm else None
                rows.append({"size": size, "case": name, "ms": round(ms, 4),
                             "tflops": None if tflops is None else round(tflops, 1)})
                print(f"    {name:<22}{ms:>10.4f} ms"
                      + (f"{tflops:>12.1f} TFLOP/s" if tflops else ""))
            except Exception as exc:
                rows.append({"size": size, "case": name,
                             "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"})
                print(f"    {name:<22} {rows[-1]['error']}")
    print("\n  量化一行是未融合的 eager 实现：amax、log2、floor、乘法、cast 各一次 kernel，")
    print("  小尺寸下几乎全是启动开销，所以 1024 这一档它比整个 FP8 GEMM 还贵。")
    print("  上游把 torch.compile 列为 float8 训练的前提，正是要把这几次 cast 融进相邻算子。")
    print("  训练每步对 input/weight/grad_output 都要量化，而且前向与反向的归约维不同，")
    print("  同一个张量常常要量化两次；GEMM 越小这笔固定开销占比越高，")
    print("  上游按 (K,N) 大小自动跳过小 linear 就是这个原因。")
    report["timing"] = rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir")
    args = ap.parse_args()
    cap = torch.cuda.get_device_capability()
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}")
    report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "capability": f"sm_{cap[0]}{cap[1]}"}
    for fn in (section_a, section_b, section_c, section_d):
        fn(report)
    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "fp8_gemm_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化结果写入 {out}/fp8_gemm_probe.json")


if __name__ == "__main__":
    main()
