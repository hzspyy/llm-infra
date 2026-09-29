#!/usr/bin/env python
"""10.1-A 预测参数化与转换的最小 FP64 参照实现。

覆盖两条时间方向相反的族：
  * 方差保持（VP，DDPM/score）: x_t = a_t x_0 + s_t eps，a_t^2 + s_t^2 = 1，t=0 是数据、t=T 是噪声；
  * 直线插值（FM/Rectified Flow）: x_t = (1-t) x_noise + t x_data，t=0 是噪声、t=1 是数据。

对每种参数化（epsilon / x0 / v / score / flow velocity）给出：
  1. 由 (x0, eps) 生成训练目标；2. 由模型输出反解 (x0_hat, eps_hat)；3. 参数化之间的等价性检查。

不依赖 diffusers；若运行环境存在 diffusers，则追加一节与官方 scheduler 的对拍。

用法：
  python labs/L10/prediction_parameterization.py --out results/<机器>/10.1/<run_id>/parameterization.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

DTYPE = torch.float64
PTYPES = ("eps", "x0", "v", "score")

# 文献中 v 的定义差异（正文据此区分，不在代码里混用）：
#   * Salimans & Ho 2022 (arXiv:2202.00512, eq.30-31)：alpha=cos(phi), sigma=sin(phi),
#     z_phi = cos(phi) x + sin(phi) eps,  v_phi = dz_phi/dphi = cos(phi) eps - sin(phi) x。
#   * Diffusers DDPMScheduler.get_velocity 与上式同号：v = sqrt(alphabar) eps - sqrt(1-alphabar) x0。
#   * Flow matching / rectified flow 的 "velocity" 是另一族对象，方向与时间约定都不同：
#     u = x_data - x_noise（本文件 flow 段），不能与上面的 v 互换。


# --------------------------------------------------------------------------------------
# 方差保持（VP）连续/离散 schedule
# --------------------------------------------------------------------------------------
def vp_discrete(name: str = "linear", num_train_timesteps: int = 1000) -> dict:
    """返回离散 VP schedule：alphas_cumprod 即 a_t^2，索引 t=0..T-1，t=0 最接近数据。"""
    if name == "linear":
        betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=DTYPE)
    elif name == "scaled_linear":
        betas = torch.linspace(0.0001**0.5, 0.02**0.5, num_train_timesteps, dtype=DTYPE) ** 2
    elif name == "cosine":
        steps = num_train_timesteps + 1
        x = torch.linspace(0, num_train_timesteps, steps, dtype=DTYPE)
        f = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = torch.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0.0001, 0.9999)
    else:
        raise ValueError(name)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return {"betas": betas, "alphas_cumprod": alphas_cumprod}


def vp_coeffs(alphas_cumprod: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """离散索引版本：a = sqrt(alphabar_t), s = sqrt(1-alphabar_t)。t 为整数索引张量。"""
    ab = alphas_cumprod[t]
    return torch.sqrt(ab), torch.sqrt(1.0 - ab)


def vp_coeffs_angle(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """连续角度版本：a = cos(phi), s = sin(phi)。phi=0 数据，phi=pi/2 纯噪声。"""
    return torch.cos(phi), torch.sin(phi)


# --------------------------------------------------------------------------------------
# 参数化互转：给定 x_t 与模型输出，反解 (x0_hat, eps_hat)
# --------------------------------------------------------------------------------------
def split_model_output(x_t: torch.Tensor, out: torch.Tensor, a: torch.Tensor,
                       s: torch.Tensor, ptype: str) -> tuple[torch.Tensor, torch.Tensor]:
    """把任意参数化的模型输出翻译成 (x0_hat, eps_hat)。

    v 的定义采用旋转形式：[x_t; v] = [[a, s], [-s, a]] [x0; eps]，
    因此 [x0; eps] = [[a, -s], [s, a]] [x_t; v]。
    """
    if ptype == "eps":
        eps_hat = out
        x0_hat = (x_t - s * out) / a
    elif ptype == "x0":
        x0_hat = out
        eps_hat = (x_t - a * out) / s
    elif ptype == "v":
        x0_hat = a * x_t - s * out
        eps_hat = s * x_t + a * out
    elif ptype == "score":
        # score = -eps / s  =>  eps = -s * score
        eps_hat = -s * out
        x0_hat = (x_t + s * s * out) / a
    else:
        raise ValueError(ptype)
    return x0_hat, eps_hat


def target_from_x0eps(x0: torch.Tensor, eps: torch.Tensor, a: torch.Tensor,
                      s: torch.Tensor, ptype: str) -> torch.Tensor:
    """由真实 (x0, eps) 生成该参数化下的训练目标。"""
    if ptype == "eps":
        return eps
    if ptype == "x0":
        return x0
    if ptype == "v":
        return a * eps - s * x0
    if ptype == "score":
        return -eps / s
    raise ValueError(ptype)


def ddim_step(x_t: torch.Tensor, eps_hat: torch.Tensor, a_t: torch.Tensor, s_t: torch.Tensor,
              a_prev: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
    """确定性 DDIM 一步：先由 eps 重建 x0_hat，再按目标 (a_prev, s_prev) 重新加噪。"""
    x0_hat = (x_t - s_t * eps_hat) / a_t
    return a_prev * x0_hat + s_prev * eps_hat


# --------------------------------------------------------------------------------------
# 检查 1：参数化往返
# --------------------------------------------------------------------------------------
def check_roundtrip(alphas_cumprod: torch.Tensor, seed: int, batch: int = 4096) -> dict:
    g = torch.Generator().manual_seed(seed)
    T = alphas_cumprod.numel()
    t = torch.randint(1, T, (batch,), generator=g)
    x0 = torch.randn(batch, 8, generator=g, dtype=DTYPE)
    eps = torch.randn(batch, 8, generator=g, dtype=DTYPE)
    a, s = vp_coeffs(alphas_cumprod, t)
    a, s = a[:, None], s[:, None]
    x_t = a * x0 + s * eps

    res = {}
    for ptype in PTYPES:
        out = target_from_x0eps(x0, eps, a, s, ptype)
        x0_hat, eps_hat = split_model_output(x_t, out, a, s, ptype)
        res[ptype] = {
            "max_abs_err_x0": float((x0_hat - x0).abs().max()),
            "max_abs_err_eps": float((eps_hat - eps).abs().max()),
            "max_abs_err_xt": float((a * x0_hat + s * eps_hat - x_t).abs().max()),
        }
    return res


# --------------------------------------------------------------------------------------
# 检查 2：采样步对参数化不变
# --------------------------------------------------------------------------------------
def check_sampler_equivariance(alphas_cumprod: torch.Tensor, seed: int, batch: int = 512) -> dict:
    g = torch.Generator().manual_seed(seed)
    T = alphas_cumprod.numel()
    t = torch.randint(1, T, (batch,), generator=g)
    x0 = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    eps = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    a, s = vp_coeffs(alphas_cumprod, t)
    a, s = a[:, None], s[:, None]
    x_t = a * x0 + s * eps
    a_prev, s_prev = vp_coeffs(alphas_cumprod, (t - 1).clamp(min=0))
    a_prev, s_prev = a_prev[:, None], s_prev[:, None]

    ref = None
    out = {}
    for ptype in PTYPES:
        pred = target_from_x0eps(x0, eps, a, s, ptype)
        _, eps_hat = split_model_output(x_t, pred, a, s, ptype)
        step = ddim_step(x_t, eps_hat, a, s, a_prev, s_prev)
        if ref is None:
            ref = step
        out[ptype] = float((step - ref).abs().max())
    return out


# --------------------------------------------------------------------------------------
# 检查 3：端点与病态
# --------------------------------------------------------------------------------------
def check_endpoints(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(4, generator=g, dtype=DTYPE)
    eps = torch.randn(4, generator=g, dtype=DTYPE)
    rows = []
    phis = [0.0, 1e-8, math.pi / 4, math.pi / 2 - 1e-8, math.pi / 2]
    for phi in phis:
        a, s = vp_coeffs_angle(torch.tensor(phi, dtype=DTYPE))
        x_t = a * x0 + s * eps
        row = {"phi": phi, "a": float(a), "s": float(s), "log_snr": float(2 * math.log(a / s)) if 0 < float(s) and float(a) > 0 else None}
        for ptype in PTYPES:
            out = target_from_x0eps(x0, eps, a, s, ptype)
            x0_hat, eps_hat = split_model_output(x_t, out, a, s, ptype)
            err = float((x0_hat - x0).abs().max())
            finite = bool(torch.isfinite(x0_hat).all() and torch.isfinite(eps_hat).all())
            row[ptype] = {"max_abs_err_x0": err if finite else None, "finite": finite}
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# 检查 4：零噪声
# --------------------------------------------------------------------------------------
def check_zero_noise(alphas_cumprod: torch.Tensor, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(4, generator=g, dtype=DTYPE)
    t = torch.tensor([500, 900], dtype=torch.long)
    a, s = vp_coeffs(alphas_cumprod, t)
    a, s = a[:, None], s[:, None]
    x0r = x0[None, :].repeat(2, 1)
    x_t = a * x0r                      # eps = 0
    res = {}
    for ptype in PTYPES:
        out_finite = {}
        out = target_from_x0eps(x0r, torch.zeros_like(x0r), a, s, ptype)
        x0_hat, eps_hat = split_model_output(x_t, out, a, s, ptype)
        out_finite["max_abs_err_x0"] = float((x0_hat - x0r).abs().max())
        out_finite["max_abs_err_eps"] = float(eps_hat.abs().max())
        out_finite["target_abs_max"] = float(out.abs().max())
        res[ptype] = out_finite
    return res


# --------------------------------------------------------------------------------------
# 检查 5：v 的符号反转 / 检查 6：prediction_type 错配
# --------------------------------------------------------------------------------------
def check_sign_and_mismatch(alphas_cumprod: torch.Tensor, seed: int = 0, batch: int = 1024) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(100, 900, (batch,), generator=g)
    x0 = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    eps = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    a, s = vp_coeffs(alphas_cumprod, t)
    a, s = a[:, None], s[:, None]
    x_t = a * x0 + s * eps

    v_true = a * eps - s * x0
    v_flipped = -v_true                                   # 等价于 s*x0 - a*eps
    # 用文档公式解释一个被翻转符号的 v
    x0_from_flip, eps_from_flip = split_model_output(x_t, v_flipped, a, s, "v")
    rel_flip = float(((x0_from_flip - x0).norm(dim=1) / x0.norm(dim=1)).mean())
    resid_flip = float((a * x0_from_flip + s * eps_from_flip - x_t).abs().max())

    # 一个 epsilon 预测模型被当成 x0 预测模型解释
    x0_from_mismatch, eps_from_mismatch = split_model_output(x_t, eps, a, s, "x0")
    rel_mismatch = float(((x0_from_mismatch - x0).norm(dim=1) / x0.norm(dim=1)).mean())
    resid_mismatch = float((a * x0_from_mismatch + s * eps_from_mismatch - x_t).abs().max())
    # 被误解释为 eps 预测模型的 x0 输出
    x0_as_eps, _ = split_model_output(x_t, x0, a, s, "eps")
    rel_reverse = float(((x0_as_eps - x0).norm(dim=1) / x0.norm(dim=1)).mean())

    # 错配是否破坏 DDIM 一步：同一 x_t 用真实/错配解释各走一步
    a_prev, s_prev = vp_coeffs(alphas_cumprod, t - 10)
    a_prev, s_prev = a_prev[:, None], s_prev[:, None]
    step_true = ddim_step(x_t, eps, a, s, a_prev, s_prev)
    step_wrong = ddim_step(x_t, eps_from_mismatch, a, s, a_prev, s_prev)
    return {
        "v_sign_flip": {
            "mean_rel_err_x0": rel_flip,
            "max_abs_consistency_residual": resid_flip,
            "target_abs_mean": float(v_true.abs().mean()),
            "flip_abs_diff": float((v_flipped - v_true).abs().max()),
        },
        "eps_read_as_x0": {"mean_rel_err_x0": rel_mismatch, "max_abs_consistency_residual": resid_mismatch},
        "x0_read_as_eps": {"mean_rel_err_x0": rel_reverse},
        "wrong_ptype_ddim_step": {
            "max_abs_step_diff": float((step_true - step_wrong).abs().max()),
            "mean_rel_step_diff": float(((step_true - step_wrong).norm(dim=1) / step_true.norm(dim=1)).mean()),
        },
        "note": (
            "一致性残差 a*x0_hat + s*eps_hat - x_t 对任何模型输出都恒为 0（转换本身是旋转的逆），"
            "因此它不能用来发现符号或 prediction_type 错误；只有与真实 (x0, eps) 比较或端到端输出才能发现。"
        ),
    }


# --------------------------------------------------------------------------------------
# 检查 7：EDM (alpha/sigma) 桥接
# --------------------------------------------------------------------------------------
def check_edm_bridge(alphas_cumprod: torch.Tensor, seed: int = 0, batch: int = 512) -> dict:
    """EDM 形式 x = x0 + sigma * n，sigma = s/a，x = x_t/a，n 与 eps 同分布。"""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(1, alphas_cumprod.numel(), (batch,), generator=g)
    x0 = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    eps = torch.randn(batch, 4, generator=g, dtype=DTYPE)
    a, s = vp_coeffs(alphas_cumprod, t)
    a, s = a[:, None], s[:, None]
    x_t = a * x0 + s * eps
    sigma = s / a
    x_edm = x_t / a
    # EDM 的 score 是对缩放变量 x=x_t/a 求的：score_x = -n/sigma = -eps/sigma = a * score_{x_t}
    score_edm = -eps / sigma
    score_ddpm = -eps / s
    return {
        "max_abs_diff_to_a_times_ddpm_score": float((score_edm - a * score_ddpm).abs().max()),
        "max_abs_diff_if_conventions_mixed": float((score_edm - score_ddpm).abs().max()),
        "max_abs_x0_from_edm": float((x_edm - sigma * eps - x0).abs().max()),
        "sigma_range": [float(sigma.min()), float(sigma.max())],
    }


# --------------------------------------------------------------------------------------
# 检查 8：flow（FM / rectified flow）参数化
# --------------------------------------------------------------------------------------
def check_flow(seed: int = 0, batch: int = 4096) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(batch, generator=g, dtype=DTYPE)
    x_data = torch.randn(batch, 8, generator=g, dtype=DTYPE)
    x_noise = torch.randn(batch, 8, generator=g, dtype=DTYPE)
    tc = t[:, None]
    x_t = (1 - tc) * x_noise + tc * x_data
    u = x_data - x_noise

    # 反解：x_data = x_t + (1-t) u, x_noise = x_t - t u
    x_data_hat = x_t + (1 - tc) * u
    x_noise_hat = x_t - tc * u
    err_data = float((x_data_hat - x_data).abs().max())
    err_noise = float((x_noise_hat - x_noise).abs().max())
    # 一致性：x_t = (1-t) x_noise_hat + t x_data_hat
    resid = float(((1 - tc) * x_noise_hat + tc * x_data_hat - x_t).abs().max())

    # 方向翻转的 u 会得到什么
    x_data_flip = x_t - (1 - tc) * u
    rel_flip = float(((x_data_flip - x_data).norm(dim=1) / x_data.norm(dim=1)).mean())

    # 端点：t=0 时 x_t 必须等于噪声样本，t=1 时必须等于数据样本
    endpoint = {}
    for tt in (0.0, 1e-12, 0.5, 1 - 1e-12, 1.0):
        x_t0 = (1 - tt) * x_noise[0, 0] + tt * x_data[0, 0]
        endpoint[str(tt)] = {
            "abs_diff_to_noise": float(abs(x_t0 - x_noise[0, 0])),
            "abs_diff_to_data": float(abs(x_t0 - x_data[0, 0])),
        }
    return {
        "roundtrip": {"max_abs_err_data": err_data, "max_abs_err_noise": err_noise, "max_abs_residual": resid},
        "direction_flip_mean_rel_err_data": rel_flip,
        "u_abs_mean": float(u.abs().mean()),
        "endpoints": endpoint,
        "note": "FM/RF 的 u 与 VP 的 v 是不同对象；t=0 是噪声、t=1 是数据，方向与 DDPM 相反。",
    }


# --------------------------------------------------------------------------------------
# 检查 9：与 diffusers 官方 scheduler 对拍（可选）
# --------------------------------------------------------------------------------------
def check_diffusers(seed: int = 0) -> dict:
    try:
        import diffusers
        from diffusers import DDPMScheduler, DDIMScheduler
    except Exception as exc:  # pragma: no cover - 本地无 diffusers 时跳过
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    T = 1000
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="linear", prediction_type="epsilon")
    abc = sched.alphas_cumprod.to(DTYPE)
    g = torch.Generator().manual_seed(seed)
    n = 256
    x0 = torch.randn(n, 4, generator=g, dtype=DTYPE)
    eps = torch.randn(n, 4, generator=g, dtype=DTYPE)
    t = torch.randint(1, T, (n,), generator=g)

    acc = {}
    # get_velocity 对拍
    v_ref = sched.get_velocity(x0, eps, t).to(DTYPE)
    a, s = vp_coeffs(abc, t)
    v_ours = a[:, None] * eps - s[:, None] * x0
    acc["velocity_max_abs_diff"] = float((v_ref - v_ours).abs().max())

    # add_noise 对拍
    x_t_ref = sched.add_noise(x0, eps, t).to(DTYPE)
    x_t_ours = a[:, None] * x0 + s[:, None] * eps
    acc["add_noise_max_abs_diff"] = float((x_t_ref - x_t_ours).abs().max())

    # step 对拍：三种 prediction_type 各解释一次同一模型输出
    torch.manual_seed(seed)
    x_t = x_t_ref
    outs = {}
    for ptype in ("epsilon", "sample", "v_prediction"):
        sched_t = DDPMScheduler(num_train_timesteps=T, beta_schedule="linear", prediction_type=ptype)
        sched_t.alphas_cumprod = sched.alphas_cumprod
        model_out = target_from_x0eps(x0, eps, a[:, None], s[:, None],
                                      {"epsilon": "eps", "sample": "x0", "v_prediction": "v"}[ptype])
        torch.manual_seed(seed)
        prev = sched_t.step(model_out, t[0].item(), x_t[:1]).prev_sample.to(DTYPE)
        _ = prev
        outs[ptype] = float(prev.abs().mean())
    acc["step_branches_run"] = outs

    # 错配 prediction_type 的真实影响：用 epsilon 目标喂给 v_prediction scheduler
    sched_v = DDPMScheduler(num_train_timesteps=T, beta_schedule="linear", prediction_type="v_prediction")
    sched_v.alphas_cumprod = sched.alphas_cumprod
    x_t1 = x_t[:1]
    t0 = t[0].item()
    torch.manual_seed(seed)
    good = sched.step(eps[:1], t0, x_t1).prev_sample.to(DTYPE)
    torch.manual_seed(seed)
    bad = sched_v.step(eps[:1], t0, x_t1).prev_sample.to(DTYPE)
    acc["mismatch_prev_sample_max_abs_diff"] = float((good - bad).abs().max())
    acc["mismatch_prev_sample_rel"] = float(((good - bad).norm() / good.norm()))

    # DDIM 对拍（确定性）
    ddim = DDIMScheduler(num_train_timesteps=T, beta_schedule="linear", prediction_type="epsilon",
                         clip_sample=False, set_alpha_to_one=False)
    ddim.alphas_cumprod = sched.alphas_cumprod
    ddim.set_timesteps(50)
    ts = int(ddim.timesteps[10].item())
    a_t, s_t = vp_coeffs(abc, torch.tensor([ts]))
    prev_t = int(ddim.timesteps[11].item())
    a_p, s_p = vp_coeffs(abc, torch.tensor([prev_t]))
    ours = ddim_step(x_t1, eps[:1], a_t[0, None], s_t[0, None], a_p[0, None], s_p[0, None])
    ref = ddim.step(eps[:1], ts, x_t1, eta=0.0, use_clipped_model_output=False).prev_sample.to(DTYPE)
    acc["ddim_step_max_abs_diff"] = float((ref - ours).abs().max())

    return {"available": True, "diffusers_version": diffusers.__version__, "checks": acc}


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    report: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float64",
        "device": "cpu",
        "schedule": {},
        "roundtrip": {},
        "sampler_equivariance": {},
        "endpoints": {},
        "zero_noise": {},
        "sign_and_mismatch": {},
        "edm_bridge": {},
        "flow": {},
    }
    for name in ("linear", "scaled_linear", "cosine"):
        abc = vp_discrete(name)["alphas_cumprod"]
        report["schedule"][name] = {
            "T": int(abc.numel()),
            "alphabar_first": float(abc[0]),
            "alphabar_last": float(abc[-1]),
        }

    abc = vp_discrete("linear")["alphas_cumprod"]
    for seed in args.seeds:
        report["roundtrip"][str(seed)] = check_roundtrip(abc, seed)
        report["sampler_equivariance"][str(seed)] = check_sampler_equivariance(abc, seed)
        report["edm_bridge"][str(seed)] = check_edm_bridge(abc, seed)
        report["flow"][str(seed)] = check_flow(seed)
    report["endpoints"] = check_endpoints()
    report["zero_noise"] = check_zero_noise(abc)
    report["sign_and_mismatch"] = check_sign_and_mismatch(abc)
    report["diffusers"] = check_diffusers()

    text = json.dumps(report, indent=2, ensure_ascii=False, default=float)
    print(text)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
