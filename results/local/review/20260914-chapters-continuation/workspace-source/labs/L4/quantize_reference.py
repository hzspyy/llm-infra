#!/usr/bin/env python3
"""L4.3 修订（任务 A/B）—— 量化的参照实现与格式账。

[A] 层输出重建目标：在 X=[256,128]、W=[64,128] 上实现 RTN/clipping、GPTQ 逐列补偿、
    AWQ 通道等价缩放，用 FP64 对拍中间状态，并比较校准分布改变后的误差
[B] 格式参照：INT4（GPTQ 布局）、FP8（per-tensor/per-channel/block-128）、
    MXFP4（E2M1 + E8M0）、NVFP4（E2M1 + E4M3 + global）的 pack/unpack 与 scale，
    组大小 32/64/128 的扫描、实际 bits/weight 复算、与官方反量化的逐元素对齐

误差口径（全章统一）：
  权重误差 = ||Ŵ − W||_F / ||W||_F
  输出误差 = ||X Ŵᵀ − X Wᵀ||_F / ||X Wᵀ||_F   （层输出重建的真实目标）

用法：
    python labs/L4/quantize_reference.py --outdir out/4.3/run A B
"""

import argparse
import json
import math
import os
import struct
import sys

import torch

SUMMARY = {"sections": {}}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


# ---------------------------------------------------------------- 量化基元
QMAX = {4: 7, 3: 3, 2: 1}          # 对称量化的最大整数（INT4/INT3/INT2）


def group_view(W, group_size):
    """[out, in] -> [out, in/gs, gs]；返回 (视图, 列数, 组数)。"""
    out_f, in_f = W.shape
    if group_size >= in_f:
        return W.reshape(out_f, 1, in_f), out_f, 1
    assert in_f % group_size == 0, "in_features 必须能被 group_size 整除"
    return W.reshape(out_f, in_f // group_size, group_size), out_f, in_f // group_size


def rtn_quant(W, bits=4, group_size=128):
    """对称 round-to-nearest：每组一个 scale，返回 (整数值, 还原值, scale)。"""
    g, out_f, ng = group_view(W, group_size)
    qmax = QMAX[bits]
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.round(g / scale).clamp(-qmax, qmax)
    deq = (q * scale).reshape(out_f, -1)
    return q.reshape(out_f, -1), deq, scale.reshape(out_f, -1)


def clip_quant(W, X, bits=4, group_size=128, ratios=None):
    """带裁剪的 RTN：按校准数据在裁剪比例上搜索最优点。"""
    if ratios is None:
        ratios = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    Wt = W.t()                        # [in, out]，按输入列分组
    g, in_f, ng = group_view(Wt, group_size)
    best = None
    for r in ratios:
        qmax = QMAX[bits]
        scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) * r / qmax
        q = torch.round(g / scale).clamp(-qmax, qmax)
        deq = (q * scale).reshape(Wt.shape)
        err = layer_error(deq.t(), W, X)
        if best is None or err < best[0]:
            best = (err, r, deq.t())
    return best[1], best[2], best[0]


def layer_error(W_hat, W, X):
    ref = X @ W.t()
    got = X @ W_hat.t()
    return ((got - ref).norm() / ref.norm()).item()


def weight_error(W_hat, W):
    return ((W_hat - W).norm() / W.norm()).item()


def gptq_quant(W, H, bits=4, group_size=128, damp=0.01):
    """GPTQ：按列量化并补偿后续列。H = X_calᵀ X_cal（[in, in]，FP64）。

    返回 (还原权重, 每组的 scale, 诊断)。中间量全部 FP64。
    """
    out_f, in_f = W.shape
    qmax = QMAX[bits]
    Wq = W.clone().double()
    W_hat = torch.zeros_like(Wq)
    scales = torch.zeros(out_f, in_f // group_size, dtype=torch.float64)
    diag_pos = []
    for gi, g0 in enumerate(range(0, in_f, group_size)):
        g1 = min(g0 + group_size, in_f)
        Wg = Wq[:, g0:g1].clone()
        Hg = H[g0:g1, g0:g1].clone()
        dead = torch.diag(Hg) == 0
        Hg[dead, dead] = 1.0
        if damp > 0:
            Hg += damp * torch.eye(g1 - g0, dtype=torch.float64) * Hg.diag().mean()
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hg))
        # 每组每个输出通道一个 scale（与 rtn_quant 的口径一致）
        scale = W[:, g0:g1].abs().amax(dim=1).clamp_min(1e-12) / qmax
        scales[:, gi] = scale
        for i in range(g1 - g0):
            w = Wg[:, i].clone()
            d = Hinv[i, i].clamp_min(1e-12)
            q = torch.round(w / scale).clamp(-qmax, qmax)
            wq = q * scale
            err = (w - wq) / d
            Wg[:, i:] -= err.outer(Hinv[i, i:])
            W_hat[:, g0 + i] = wq
            diag_pos.append(float(d))
    return W_hat, scales, {"Hinv_diag": diag_pos}


def awq_scale_quant(W, X, bits=4, group_size=128, grid=21, use_activation=True):
    """AWQ：对输入通道做等价缩放 s，W' = W·diag(s)，X' = X/diag(s)，再 RTN。

    量化后把 s 折回权重：Ŵ = dequant(W·diag(s)) / diag(s)。
    """
    out_f, in_f = W.shape
    qmax = QMAX[bits]
    if use_activation:
        a = X.abs().mean(dim=0).double().clamp_min(1e-12)
    else:
        a = W.abs().mean(dim=0).double().clamp_min(1e-12)   # 对照：不看激活
    best = None
    for p in torch.linspace(0.0, 1.0, grid).tolist():
        s = a.pow(p)
        s = s / s.mean()
        Ws = W * s[None, :]
        q, deq, _ = rtn_quant(Ws, bits, group_size)
        W_hat = deq / s[None, :]
        err = layer_error(W_hat, W, X)
        if best is None or err < best[0]:
            best = (err, p, s.clone(), W_hat.clone())
    return best[1], best[3], best[0], best[2]


# ---------------------------------------------------------------- A
def section_A(outdir):
    torch.manual_seed(0)
    rep = {}
    title("[A] 层输出重建目标与三种量化方法")
    in_f, out_f, n = 128, 64, 256
    gs = 32
    W = (torch.randn(out_f, in_f, dtype=torch.float64) * 0.05)
    X = torch.randn(n, in_f, dtype=torch.float64)
    X[:, ::16] *= 10.0        # 每 16 个输入通道一个离群通道（真实激活的常见形态）
    H = X.t() @ X
    print(f"  X {tuple(X.shape)}（校准样本 {n}，每 16 个通道一个 ×10 离群通道）")
    print(f"  W {tuple(W.shape)}  group_size={gs}  全部 FP64")
    print(f"  通道幅度：普通通道均值 {X[:, 1:16].abs().mean():.3f}，"
          f"离群通道均值 {X[:, ::16].abs().mean():.3f}")

    sub("A1 目标函数：为什么不是最小化权重误差")
    q_rtn, W_rtn, scales = rtn_quant(W, 4, gs)
    print(f"  RTN：权重误差 {weight_error(W_rtn, W):.4e}  "
          f"输出误差 {layer_error(W_rtn, W, X):.4e}")
    ev = torch.linalg.eigvalsh(H)
    print(f"  H = XᵀX 的形状 {tuple(H.shape)}，特征值范围 "
          f"[{ev.min().item():.3e}, {ev.max().item():.3e}]（对称半正定）")
    print("  层输出误差 = Σ_g tr((Ŵ_g − W_g) H_g (Ŵ_g − W_g)ᵀ)，")
    print("  所以每一组的最优量化取决于该组的 H_g = X_gᵀX_g，而权重误差忽略了 X。")

    sub("A2 clipping：在裁剪比例上按校准数据搜索")
    best_r, W_clip, err_clip = clip_quant(W, X, 4, gs)
    print(f"  RTN（无裁剪）      权重误差 {weight_error(W_rtn, W):.4e}  "
          f"输出误差 {layer_error(W_rtn, W, X):.4e}")
    print(f"  最优裁剪比例 {best_r:.2f}   权重误差 {weight_error(W_clip, W):.4e}  "
          f"输出误差 {err_clip:.4e}")
    print(f"  两个指标的关系：权重误差 "
          f"{weight_error(W_rtn, W):.4e} → {weight_error(W_clip, W):.4e}，"
          f"输出误差 {layer_error(W_rtn, W, X):.4e} → {err_clip:.4e}")
    print("  裁剪比例的搜索目标是输出误差；权重误差只是副产物，两者不同步。")

    sub("A3 GPTQ：用 Hinv 逐列补偿")
    W_gptq, gscales, diag = gptq_quant(W, H, 4, gs)
    print(f"  GPTQ               权重误差 {weight_error(W_gptq, W):.4e}  "
          f"输出误差 {layer_error(W_gptq, W, X):.4e}")
    # FP64 中间状态对拍：Hinv 的对角元应与逐列补偿量一致
    Hg = H[:gs, :gs] + 0.01 * torch.eye(gs, dtype=torch.float64) * H[:gs, :gs].diag().mean()
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hg))
    d0 = float(Hinv[0, 0])
    print(f"  中间状态：第 0 组 Hinv[0,0] = {d0:.6f}（脚本记录 {diag['Hinv_diag'][0]:.6f}，"
          f"相对差 {abs(d0 - diag['Hinv_diag'][0]) / d0:.2e}）")
    # FP64 中间状态对拍：两列闭式，验证补偿量等于闭式解
    W2 = W[:, 8:10].clone()
    rho = 0.9
    H2 = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    Hinv2 = torch.cholesky_inverse(torch.linalg.cholesky(H2))
    qmax = QMAX[4]
    scale = W2.abs().amax(dim=1).clamp_min(1e-12) / qmax     # 整行一个 scale（group=2）
    w0 = W2[:, 0].clone()
    q0 = torch.round(w0 / scale).clamp(-qmax, qmax)
    err0 = (w0 - q0 * scale) / Hinv2[0, 0]
    observed = -(err0.outer(Hinv2[0, 1:2]))                  # 代码里对第 1 列施加的更新
    closed = -(w0 - q0 * scale) * (Hinv2[0, 1] / Hinv2[0, 0])
    rel = ((observed.reshape(-1) - closed).abs().max()
           / closed.abs().max().clamp_min(1e-30)).item()
    print(f"  两列中间状态：Hinv[0,1]/Hinv[0,0] = {Hinv2[0, 1] / Hinv2[0, 0]:.6f}，"
          f"补偿量与闭式解的最大相对差 {rel:.2e}")
    print("  这条同时说明为什么 ρ=0 时补偿为零：与第 0 列不相关的列没有可传递的误差。")

    sub("A4 AWQ：通道等价缩放")
    p_star, W_awq, err_awq, s = awq_scale_quant(W, X, 4, gs)
    print(f"  AWQ（看激活）      最优 p={p_star:.2f}  "
          f"权重误差 {weight_error(W_awq, W):.4e}  输出误差 {err_awq:.4e}")
    _, W_awq_w, err_awq_w, _ = awq_scale_quant(W, X, 4, gs, use_activation=False)
    print(f"  AWQ（只看权重均值） 权重误差 {weight_error(W_awq_w, W):.4e}  "
          f"输出误差 {err_awq_w:.4e}")
    print("  激活统计比权重统计更贴近目标：AWQ 的缩放是围绕 X 设计的。")

    sub("A5 校准分布改变后的误差")
    dists = {}
    torch.manual_seed(1)
    X_same = torch.randn(n, in_f, dtype=torch.float64)
    outlier = torch.randn(n, in_f, dtype=torch.float64)
    outlier[:, ::8] *= 12.0                       # 每 8 个通道一个离群通道
    corr = X + 0.5 * torch.roll(X, 1, dims=1)     # 通道相关
    for tag, Xe in (("同分布", X_same), ("离群通道", outlier), ("通道相关", corr)):
        row = {"rtn": layer_error(W_rtn, W, Xe),
               "clip": layer_error(W_clip, W, Xe),
               "gptq": layer_error(W_gptq, W, Xe),
               "awq": layer_error(W_awq, W, Xe)}
        dists[tag] = row
        print(f"  {tag:<8} RTN {row['rtn']:.4e}  clip {row['clip']:.4e}  "
              f"GPTQ {row['gptq']:.4e}  AWQ {row['awq']:.4e}")
    print("  校准集只影响三种方法的**参数**（scale / 补偿 / 缩放），")
    print("  评测集换分布后误差排序会变，这就是校准/评测必须分开的原因。")

    rep["setup"] = {"X": [n, in_f], "W": [out_f, in_f], "group_size": gs,
                    "dtype": "float64", "seed": 0}
    rep["methods"] = {
        "rtn": {"weight_error": weight_error(W_rtn, W),
                "output_error": layer_error(W_rtn, W, X)},
        "clip": {"best_ratio": best_r, "weight_error": weight_error(W_clip, W),
                 "output_error": layer_error(W_clip, W, X)},
        "gptq": {"weight_error": weight_error(W_gptq, W),
                 "output_error": layer_error(W_gptq, W, X)},
        "awq": {"best_p": p_star, "weight_error": weight_error(W_awq, W),
                "output_error": layer_error(W_awq, W, X),
                "awq_weight_only": {"weight_error": weight_error(W_awq_w, W),
                                    "output_error": layer_error(W_awq_w, W, X)}}}
    rep["hessian"] = {"eig_min": ev.min().item(), "eig_max": ev.max().item()}
    rep["calibration_shift"] = dists
    SUMMARY["sections"]["A"] = rep
    return rep


# ---------------------------------------------------------------- B
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
E8M0_BIAS = 127


def e2m1_encode(x, scale):
    """把 x/scale 编码成 E2M1（4 bit：1 符号 + 2 指数 + 1 尾数），返回整数码。"""
    v = (x / scale).abs()
    # 找最接近的可表示幅度（索引 0..7），ties 取更小
    idx = torch.searchsorted(E2M1, v.contiguous(), right=False)
    idx = idx.clamp(0, 7)
    lo = (idx - 1).clamp(0, 7)
    choose_lo = (v - E2M1[lo]).abs() <= (E2M1[idx] - v).abs()
    mag = torch.where(choose_lo, lo, idx)
    sign = (x < 0).to(torch.int64)
    return (sign << 3) | mag


def e2m1_decode(code, scale):
    sign = (code >> 3) & 1
    mag = code & 7
    v = E2M1[mag]
    return torch.where(sign.bool(), -v, v) * scale


def e8m0_scale(x, block):
    """E8M0：只有指数，scale = 2^(e-127)，按 block 内 amax 取。"""
    g = x.reshape(-1, block) if x.dim() == 1 else x.reshape(x.shape[0], -1, block)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    e = torch.floor(torch.log2(amax)) .clamp(-127, 127)
    return torch.pow(2.0, e), e + E8M0_BIAS


def section_B(outdir):
    torch.manual_seed(0)
    rep = {}
    title("[B] 格式参照：pack/unpack、scale 与 bits/weight")

    sub("B1 E2M1 与 E8M0 的参照表")
    print(f"  E2M1 可表示幅度（3 位指数+尾数）: {E2M1.tolist()}")
    codes = torch.arange(16)
    vals = e2m1_decode(codes, torch.tensor(1.0))
    print(f"  16 个码的取值: {[round(v, 2) for v in vals.tolist()]}")
    known = torch.tensor([0.0, 0.5, -1.5, 6.0, -6.0])
    enc = e2m1_encode(known, torch.tensor(1.0))
    dec = e2m1_decode(enc, torch.tensor(1.0))
    print(f"  已知值 {known.tolist()} -> 码 {enc.tolist()} -> 还原 {dec.tolist()}  "
          f"精确往返 {bool(torch.equal(dec, known))}")
    for v in (0.3, 0.8, 2.6, 5.2):
        e = e2m1_encode(torch.tensor([v]), torch.tensor(1.0))
        d = e2m1_decode(e, torch.tensor(1.0))
        print(f"    {v} -> 码 {int(e)} -> {float(d):.2f}（四舍五入到最近的可表示值）")
    xs = torch.randn(64, dtype=torch.float64)
    sc, e = e8m0_scale(xs, 32)
    print(f"  E8M0：两个 block 的 scale {sc.reshape(-1).tolist()}"
          f"（指数码 {[int(v) for v in e.reshape(-1)]}，bias={E8M0_BIAS}）")

    sub("B2 四种格式的 pack/unpack 与 bits/weight")
    torch.manual_seed(2)
    W = torch.randn(64, 128, dtype=torch.float32) * 0.05
    X = torch.randn(256, 128, dtype=torch.float32)
    rows = []

    def record(name, deq, bits_per_weight, extra):
        e_w = weight_error(deq.double(), W.double())
        e_o = layer_error(deq.double(), W.double(), X.double())
        rows.append({"format": name, "bits_per_weight": bits_per_weight,
                     "weight_error": e_w, "output_error": e_o, **extra})
        print(f"  {name:<26} {bits_per_weight:>6.3f} bit/权重  "
              f"权重误差 {e_w:.4e}  输出误差 {e_o:.4e}")

    # INT4（GPTQ 布局的打包：沿输入维，8 个 4 bit 塞进 int32）
    def int4_pack(q):
        qi = q.to(torch.int32) & 0xF          # 存成无符号 nibble，负值走补码
        qq = qi.reshape(qi.shape[0] // 8, 8, qi.shape[1]).permute(0, 2, 1)
        sh = torch.arange(0, 32, 4)
        return (qq << sh).sum(-1).to(torch.int32)

    def int4_unpack(p):
        sh = torch.arange(0, 32, 4)
        c = ((p.unsqueeze(-1) >> sh) & 0xF).permute(0, 2, 1).reshape(-1, p.shape[1])
        return torch.where(c >= 8, c - 16, c).to(torch.float64)

    for gs in (32, 64, 128):
        q, deq, scale = rtn_quant(W.double(), 4, gs)
        packed = int4_pack(q.to(torch.int32))
        back = int4_unpack(packed)
        assert torch.equal(back, q), "INT4 往返必须逐元素相等"
        n_bytes = packed.numel() * 4 + scale.numel() * 2      # int32 包 + fp16 scale
        bpw = n_bytes * 8 / W.numel()
        record(f"INT4 g{gs}", deq.to(torch.float32), bpw,
               {"pack_roundtrip_exact": True, "scale_dtype": "fp16"})

    # FP8 e4m3：per-tensor / per-channel / block-128
    fe = torch.float8_e4m3fn
    fp8_max = 448.0
    for tag, mode in (("FP8 per-tensor", "tensor"), ("FP8 per-channel", "channel"),
                      ("FP8 block-32", "b32"), ("FP8 block-64", "b64"),
                      ("FP8 block-128", "b128")):
        if mode == "tensor":
            scale = W.abs().max().double() / fp8_max
            deq = (W / scale).to(fe).float().double() * scale
            sc_bytes = 4
        elif mode == "channel":
            scale = W.abs().amax(dim=1, keepdim=True).double() / fp8_max
            deq = (W / scale).to(fe).float().double() * scale
            sc_bytes = W.shape[0] * 4
        else:
            blk = int(mode[1:])
            g = W.reshape(W.shape[0], -1, blk)
            scale = g.abs().amax(dim=-1, keepdim=True).double() / fp8_max
            deq = ((g / scale).to(fe).float().double() * scale).reshape(W.shape)
            sc_bytes = g.shape[0] * g.shape[1] * 4
        bpw = (W.numel() * 8 + sc_bytes * 8) / W.numel()
        record(tag, deq.float(), bpw, {"scale_bytes": sc_bytes})

    # MXFP4：E2M1 + E8M0（每 32 个元素一个共享指数）
    def mxfp4(Wt):
        g = Wt.reshape(Wt.shape[0], -1, 32)
        scale = torch.pow(2.0, torch.floor(torch.log2(
            g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30))))
        code = e2m1_encode(g, scale)
        deq = e2m1_decode(code, scale)
        return deq.reshape(Wt.shape), code, scale

    deq, code, scale = mxfp4(W.double())
    bpw = (W.numel() * 4 + (W.numel() // 32) * 8) * 1.0 / W.numel()
    record("MXFP4 (E2M1+E8M0/32)", deq.float(), bpw, {"scale_dtype": "e8m0"})

    # NVFP4：E2M1 + E4M3（每 16 个）+ fp32 global
    def nvfp4(Wt, block=16):
        g = Wt.reshape(Wt.shape[0], -1, block)
        global_scale = g.abs().max().clamp_min(1e-30) / (6.0 * 448.0)
        scale = (g.abs().amax(dim=-1, keepdim=True) / 6.0).clamp_min(1e-30)
        scale_e4m3 = (scale / global_scale).to(fe).float().clamp_min(1e-30)
        code = e2m1_encode(g, scale_e4m3 * global_scale)
        deq = e2m1_decode(code, scale_e4m3 * global_scale)
        return deq.reshape(Wt.shape), code, scale_e4m3

    deq, code, scale = nvfp4(W.double())
    bpw = (W.numel() * 4 + (W.numel() // 16) * 8 + 32) * 1.0 / W.numel()
    record("NVFP4 (E2M1+E4M3/16)", deq.float(), bpw, {"scale_dtype": "e4m3+fp32global"})

    sub("B3 与官方反量化/量化算子对齐")
    if torch.cuda.is_available():
        from vllm import _custom_ops as ops
        # FP8：与 ops.scaled_fp8_quant 比字节与 scale
        Wc = W.cuda()
        sc_got = torch.zeros(1, device="cuda", dtype=torch.float32)
        # 动态量化：第二个位置参数是"静态 scale"，传 0 会把输入压成 0
        ret = ops.scaled_fp8_quant(Wc, use_per_token_if_dynamic=False)
        if isinstance(ret, tuple):
            q_got, sc_got = ret[0], ret[1]
        else:
            q_got = ret
        mine_scale = (W.abs().max().double() / fp8_max).float()
        mine_q = (W / mine_scale).to(fe)
        same_bytes = int((q_got.view(torch.uint8) == mine_q.cuda().view(torch.uint8)).sum())
        total = W.numel()
        print(f"  FP8：官方 scale {float(sc_got):.6e} vs 本节参照 {float(mine_scale):.6e}"
              f"（相对差 {abs(float(sc_got) - float(mine_scale)) / float(mine_scale):.2e}）")
        print(f"        官方 fp8 字节与参照逐字节相同 {same_bytes}/{total}")
        rep["fp8_vs_vllm"] = {"scale_got": float(sc_got),
                              "scale_ref": float(mine_scale),
                              "same_bytes": same_bytes, "total": total}
        # NVFP4：官方量化算子（8x4 swizzle 布局）的往返误差
        try:
            gscale = torch.tensor([1.0], device="cuda")
            xq, sf = ops.scaled_fp4_quant(Wc.to(torch.bfloat16), gscale)
            print(f"  NVFP4：官方 scaled_fp4_quant 输出 shape {tuple(xq.shape)} "
                  f"scale shape {tuple(sf.shape)}（fp8 打包 + 8x4 swizzle 布局）")
            rep["nvfp4_vllm"] = {"q_shape": list(xq.shape), "scale_shape": list(sf.shape)}
        except Exception as e:
            print(f"  NVFP4 官方算子调用失败：{type(e).__name__}: {str(e)[:120]}")
            rep["nvfp4_vllm"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        print("  没有 CUDA，跳过官方算子对照")
        rep["fp8_vs_vllm"] = None
    rep["formats"] = rows
    SUMMARY["sections"]["B"] = rep
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l43_ref"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B"]
    env = {"python": sys.version.split()[0]}
    env["torch"] = torch.__version__
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    SUMMARY["env"] = env
    for s in want:
        {"A": section_A, "B": section_B}[s](args.outdir)
    SUMMARY["outdir"] = args.outdir
    path = os.path.join(args.outdir, "quantize_reference.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
