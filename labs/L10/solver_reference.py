#!/usr/bin/env python
"""10.2-A 采样器参照实现：Euler / Heun / DPM-Solver++(2M) 与多步历史状态。

三类检查：
  1. 解析 ODE（有闭式解）上量各方法的观测阶；
  2. 10.1 的解析二维场（VP Gaussian / FM GMM）上对手写求解器与 RK4 参照；
  3. 有 diffusers 时与官方 scheduler 逐步对拍，并打印 sigma、模型输入缩放与历史状态。

只依赖 torch；diffusers 缺失时自动跳过第 3 节。用法：
  python labs/L10/solver_reference.py --out results/local/10.2/<run_id>/solver_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

DTYPE = torch.float64


def _col(v):
    if not torch.is_tensor(v):
        return torch.tensor(v, dtype=DTYPE)
    return v.unsqueeze(-1) if v.dim() == 1 else v


# --------------------------------------------------------------------------------------
# 解析场（与 10.1-B 同一族，便于两章对齐）
# --------------------------------------------------------------------------------------
class Gaussian2D:
    def __init__(self, mu, var):
        self.mu = mu.to(DTYPE)
        self.var = var.to(DTYPE)

    def score_vp(self, x, a, s):
        a, s = _col(a), _col(s)
        D = a * a * self.var + s * s
        return (a * self.mu - x) / D


class GMM2D:
    def __init__(self, pi, mu, var):
        self.pi = pi.to(DTYPE)
        self.mu = mu.to(DTYPE)
        self.var = var.to(DTYPE)

    def _resp_flow(self, x, t):
        tc = _col(t)
        A = 1 - tc
        logp = []
        for k in range(self.mu.shape[0]):
            D = A ** 2 + tc * tc * self.var[k]
            d = x - tc * self.mu[k]
            logp.append(torch.log(self.pi[k]) - 0.5 * (torch.log(D).sum(dim=-1) + (d * d / D).sum(dim=-1)))
        return torch.softmax(torch.stack(logp, dim=0), dim=0)

    def flow_velocity(self, x, t):
        tc = _col(t)
        A = 1 - tc
        w = self._resp_flow(x, t)
        out = torch.zeros_like(x)
        for k in range(self.mu.shape[0]):
            D = A ** 2 + tc * tc * self.var[k]
            m = tc * self.mu[k]
            e_noise = A / D * (x - m)
            e_data = self.mu[k] + tc * self.var[k] / D * (x - m)
            out = out + w[k][:, None] * (e_data - e_noise)
        return out


# --------------------------------------------------------------------------------------
# 通用积分器
# --------------------------------------------------------------------------------------
def integrate(field, x0, t0, t1, steps, method):
    """返回 (末端样本, 轨迹, NFE)。field(x, t) 返回速度。"""
    x = x0
    h = (t1 - t0) / steps
    t = t0
    traj = [x.clone()]
    nfe = 0
    for i in range(steps):
        if method == "euler":
            x = x + h * field(x, t); t += h; nfe += 1
        elif method == "heun":
            k1 = field(x, t); nfe += 1
            xp = x + h * k1
            k2 = field(xp, t + h); nfe += 1
            x = x + 0.5 * h * (k1 + k2); t += h
        elif method == "midpoint":
            k1 = field(x, t); nfe += 1
            xm = x + 0.5 * h * k1
            k2 = field(xm, t + 0.5 * h); nfe += 1
            x = x + h * k2; t += h
        else:
            raise ValueError(method)
        traj.append(x.clone())
    return x, traj, nfe


def rk4_reference(field, x0, t0, t1, steps=4096):
    x = x0
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = field(x, t)
        k2 = field(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = field(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = field(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return x


# --------------------------------------------------------------------------------------
# 检查 1：解析 ODE dx/dt = -lam x，闭式解
# --------------------------------------------------------------------------------------
def check_linear_ode(lam: float = 0.7, x0_val: float = 1.0, steps_list=(8, 32, 128)) -> dict:
    def field(x, t):
        return -lam * x

    x0 = torch.full((64, 1), x0_val, dtype=DTYPE)
    exact = x0 * math.exp(-lam * 1.0)
    rows = {}
    prev = {}
    for steps in steps_list:
        row = {}
        for method in ("euler", "heun", "midpoint"):
            x, _, nfe = integrate(field, x0, 0.0, 1.0, steps, method)
            row[method] = {"rel_err": float((x - exact).abs().max() / exact.abs().max()), "nfe": nfe}
        rows[str(steps)] = row
    lo, hi = str(steps_list[0]), str(steps_list[-1])
    ratio = steps_list[-1] / steps_list[0]
    order = {m: float(math.log(rows[lo][m]["rel_err"] / rows[hi][m]["rel_err"]) / math.log(ratio))
             for m in ("euler", "heun", "midpoint")}
    return {"rows": rows, "observed_order": order,
            "note": "闭式解 exp(-lam t)；Euler 一阶、Heun/midpoint 二阶是条件，不是恒等式。"}


# --------------------------------------------------------------------------------------
# 检查 2：VP 解析场上的手写 Euler/Heun/DPM++(2M)
# --------------------------------------------------------------------------------------
def vp_sigmas(num_train=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_train, dtype=DTYPE)
    ab = torch.cumprod(1 - betas, dim=0)
    sigmas = torch.sqrt((1 - ab) / ab)
    return torch.cat([sigmas, torch.zeros(1, dtype=DTYPE)])


def sigma_to_alpha_sigma(sigma):
    """扩散工程的 VP 约定：alpha = 1/sqrt(1+sigma^2)，sigma_t = sigma*alpha。"""
    alpha = 1 / torch.sqrt(1 + sigma ** 2)
    return alpha, sigma * alpha


def eps_model_vp(model, x, sigma):
    """给定 sigma 与解析 score，返回 epsilon 预测。"""
    a, s = sigma_to_alpha_sigma(sigma)
    a, s = _col(a), _col(s)
    score = model.score_vp(x, a, s)
    return -s * score


def dpmpp2m_vp(model, x0, sigmas_grid, final_sigmas_type="zero", lower_order_final_cfg=True):
    """手写 DPM-Solver++(2M)：一阶起步、二阶用历史、末步降阶。返回 (x, nfe, history)。"""
    sigmas = sigmas_grid
    n = len(sigmas) - 1
    x = x0
    nfe = 0
    history = []
    m_prev = None
    lower_order_nums = 0
    for i in range(n):
        sigma_s0 = sigmas[i]
        sigma_t = sigmas[i + 1]
        m0 = eps_model_vp(model, x, sigma_s0)          # 模型输出 = epsilon
        nfe += 1
        alpha_s0, sig_s0 = sigma_to_alpha_sigma(sigma_s0)
        alpha_t, sig_t = sigma_to_alpha_sigma(sigma_t)
        lam_s0 = torch.log(alpha_s0) - torch.log(sig_s0)
        lam_t = torch.log(alpha_t) - torch.log(sig_t)
        h = lam_t - lam_s0
        # convert_model_output: epsilon -> x0
        d0 = (x - sig_s0 * m0) / alpha_s0
        # 与 diffusers 的判据一致：末步在 euler_at_final / (lower_order_final 且步数<15) /
        # final_sigmas_type == "zero" 任一成立时降为一阶（默认 final_sigmas_type="zero"）。
        lower_order_final = (i == n - 1) and ((lower_order_final_cfg and n < 15)
                                              or final_sigmas_type == "zero")
        if lower_order_nums < 1 or lower_order_final:
            x = (sig_t / sig_s0) * x - alpha_t * (torch.exp(-h) - 1.0) * d0
            used = "first"
        else:
            sigma_s1 = sigmas[i - 1]
            alpha_s1, sig_s1 = sigma_to_alpha_sigma(sigma_s1)
            lam_s1 = torch.log(alpha_s1) - torch.log(sig_s1)
            h0 = lam_s0 - lam_s1
            r0 = h0 / h
            d1 = (1.0 / r0) * (d0 - m_prev)
            x = ((sig_t / sig_s0) * x
                 - alpha_t * (torch.exp(-h) - 1.0) * d0
                 - 0.5 * alpha_t * (torch.exp(-h) - 1.0) * d1)
            used = "second"
        m_prev = d0
        lower_order_nums = min(lower_order_nums + 1, 2)
        h_val = float(h)
        history.append({"step": i, "sigma": float(sigma_s0), "order": used,
                        "lower_order_nums": lower_order_nums,
                        "h": h_val if math.isfinite(h_val) else "inf(last step to sigma=0)"})
    return x, nfe, history


def vp_field_from_score(model):
    """连续 VP 概率流 ODE 的速度场（角度时间）：dx/dphi = a*eps - s*x0 = v。"""
    def field(x, phi):
        a = torch.full((x.shape[0],), math.cos(phi), dtype=DTYPE)
        s = torch.full((x.shape[0],), math.sin(phi), dtype=DTYPE)
        score = model.score_vp(x, a, s)
        a, s = _col(a), _col(s)
        eps = -s * score
        x0 = (x + s * s * score) / a
        return a * eps - s * x0
    return field


def check_vp_orders(model, steps_list=(8, 32, 128), seed=0, n=1024) -> dict:
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    field = vp_field_from_score(model)
    phi0 = math.pi / 2 - 1e-4
    ref = rk4_reference(field, x0, phi0, 0.0, steps=4096)
    rows = {}
    for steps in steps_list:
        row = {}
        for method in ("euler", "heun", "midpoint"):
            x, traj, nfe = integrate(field, x0, phi0, 0.0, steps, method)
            row[method] = {"rel_err": float((x - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()), "nfe": nfe}
        rows[str(steps)] = row
    lo, hi = str(steps_list[0]), str(steps_list[-1])
    ratio = steps_list[-1] / steps_list[0]
    order = {m: float(math.log(rows[lo][m]["rel_err"] / rows[hi][m]["rel_err"]) / math.log(ratio))
             for m in ("euler", "heun", "midpoint")}
    return {"rows": rows, "observed_order": order,
            "note": "参照为 RK4(4096)；VP 场的 phi 端点为 pi/2-1e-4，避开 a=0。"}


# --------------------------------------------------------------------------------------
# 检查 3：与 diffusers 官方 scheduler 对拍
# --------------------------------------------------------------------------------------
def _eps_from_model_input(model, x_in, sigma, scheduler=None, t=None):
    """先按 scheduler.scale_model_input 得到网络输入，再算 epsilon 预测。

    Euler/Heun 的状态是 EDM 缩放的 x，输入会乘 1/sqrt(1+sigma^2)；DPM-Solver++ 的状态就是
    VP 的 x_t，输入不缩放。统一走 scale_model_input 可以避免把两套变量混用。
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), dtype=DTYPE)
    alpha = 1 / torch.sqrt(1 + sigma ** 2)
    s = sigma * alpha
    return -s * model.score_vp(x_in, alpha, s)


def check_diffusers_vp(model, seed=0, n=64, steps_list=(10, 20, 40)) -> dict:
    try:
        from diffusers import (EulerDiscreteScheduler, HeunDiscreteScheduler,
                               DPMSolverMultistepScheduler)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    T = 1000
    out = {"available": True, "cases": {}}

    def run_library(sched, steps, kind):
        sched.set_timesteps(steps)
        init = float(sched.init_noise_sigma)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 2, generator=g, dtype=DTYPE) * init
        nfe = 0
        sigmas_seen = []
        try:
            for t in sched.timesteps:
                idx = 0 if sched.step_index is None else sched.step_index
                # 模型输入用的 sigma 是 scale_model_input 在时间戳 t 上查到的 sigmas[step_index]；
                # Heun 的第二次调用用同一时间戳，因此两次的输入 sigma 相同。
                j = min(idx, len(sched.sigmas) - 1)
                sigma = float(sched.sigmas[j])
                sigmas_seen.append(sigma)
                x_in = sched.scale_model_input(x, t)
                eps = _eps_from_model_input(model, x_in, sched.sigmas[j])
                nfe += 1
                x = sched.step(eps, t, x, return_dict=False)[0]
            ok = True
        except Exception as exc:                              # 记录失败而不是静默
            ok = False
            sigmas_seen.append(f"error: {type(exc).__name__}: {exc}")
        return x, nfe, sigmas_seen, init, ok

    for steps in steps_list:
        cases = {}
        for name in ("euler", "heun", "dpmpp_2m"):
            if name == "euler":
                sched = EulerDiscreteScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02,
                                               beta_schedule="linear", prediction_type="epsilon",
                                               timestep_spacing="linspace", interpolation_type="linear")
                kind = "euler"
            elif name == "heun":
                sched = HeunDiscreteScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02,
                                              beta_schedule="linear", prediction_type="epsilon",
                                              timestep_spacing="linspace")
                kind = "heun"
            else:
                sched = DPMSolverMultistepScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02,
                                                    beta_schedule="linear", prediction_type="epsilon",
                                                    algorithm_type="dpmsolver++", solver_order=2,
                                                    solver_type="midpoint", lower_order_final=True,
                                                    timestep_spacing="linspace")
                kind = "dpmpp"
            x_lib, nfe_lib, sig_seen, init, ok = run_library(sched, steps, kind)
            g = torch.Generator().manual_seed(seed)
            x_start = torch.randn(n, 2, generator=g, dtype=DTYPE) * init

            if name in ("euler", "heun"):
                # Heun 库内把 sigmas 复制成 2N 项，真实的时间网格是 [sigma_0, sigma_1, sigma_3, ...]
                all_sig = [float(v) for v in sched.sigmas]
                grid = all_sig if name == "euler" else [all_sig[0]] + all_sig[1::2]
                x_hand = x_start.clone()
                nfe_hand = 0
                if name == "euler":
                    for i in range(steps):
                        s0, s1 = grid[i], grid[i + 1]
                        eps = _eps_from_model_input(model, x_hand * (1 / math.sqrt(1 + s0 ** 2)), s0); nfe_hand += 1
                        x0_hat = x_hand - s0 * eps
                        deriv = (x_hand - x0_hat) / s0
                        x_hand = x_hand + (s1 - s0) * deriv
                else:
                    for i in range(steps):
                        s0, s1 = grid[i], grid[i + 1]
                        eps = _eps_from_model_input(model, x_hand * (1 / math.sqrt(1 + s0 ** 2)), s0); nfe_hand += 1
                        d1 = (x_hand - (x_hand - s0 * eps)) / s0
                        x_pred = x_hand + (s1 - s0) * d1
                        if i < steps - 1:
                            eps2 = _eps_from_model_input(model, x_pred * (1 / math.sqrt(1 + s1 ** 2)), s1); nfe_hand += 1
                            d2 = (x_pred - (x_pred - s1 * eps2)) / s1
                            x_hand = x_hand + 0.5 * (s1 - s0) * (d1 + d2)
                        else:
                            x_hand = x_pred
            else:
                # DPM++ 的 sigma 序列：每次调用前的 sigmas[step_index]，末步目标是 0
                seq = [v for v in sig_seen if isinstance(v, float)]
                grid_t = torch.tensor(seq + [0.0], dtype=DTYPE)
                x_hand, nfe_hand, _hist = dpmpp2m_vp(model, x_start, grid_t,
                                                     final_sigmas_type=sched.config.final_sigmas_type,
                                                     lower_order_final_cfg=sched.config.lower_order_final)
            cases[name] = {
                "steps": steps,
                "nfe_library": nfe_lib,
                "nfe_hand": nfe_hand,
                "library_ok": ok,
                "max_abs_diff": float((x_lib - x_hand).abs().max()) if ok else None,
                "mean_abs_library": float(x_lib.abs().mean()) if ok else None,
                "mean_abs_hand": float(x_hand.abs().mean()),
                "init_noise_sigma": init,
                "n_sigmas": len(sched.sigmas),
                "n_timesteps": len(sched.timesteps),
                "sigma_first": float(sched.sigmas[0]),
                "sigma_input_first": sig_seen[0] if sig_seen else None,
                "sigma_input_last": sig_seen[-2] if len(sig_seen) > 1 else None,
            }
        out["cases"][str(steps)] = cases
    out["note"] = ("Euler/Heun 在 EDM 缩放的 x 上积分、模型输入再乘 1/sqrt(1+sigma^2)；"
                   "DPM-Solver++ 直接以 VP 的 x_t 为变量。两套变量不能互相代入。"
                   "Heun 的 timesteps 在库内被复制成 2N-1 个，每次调用一次模型求值。")
    return out


# --------------------------------------------------------------------------------------
# 检查 4：历史状态与起步/末步降阶
# --------------------------------------------------------------------------------------
def check_history(model, steps=10, seed=0, n=32) -> dict:
    sigmas = vp_sigmas()
    grid = torch.linspace(0, 999, steps + 1).round().long()
    sig_grid = sigmas[grid]
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    _, nfe, hist = dpmpp2m_vp(model, x0, sig_grid)
    return {"nfe": nfe, "steps": steps,
            "history": hist,
            "first_step_order": hist[0]["order"],
            "last_step_order": hist[-1]["order"],
            "note": "起步用一阶、二阶从第二步开始；步数 <15 时末步降为一阶（lower_order_final）。"}


# --------------------------------------------------------------------------------------
# 检查 5：flow matching 的 sigma / 输入缩放
# --------------------------------------------------------------------------------------
def check_flow_match(model_gmm, seed=0, n=256, steps_list=(8, 32, 128)) -> dict:
    out = {}
    # 手写 Euler/Heun（时间变量 t∈[0,1]，与 10.1 的约定一致）
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    vf = lambda x, t: model_gmm.flow_velocity(x, t)
    ref = rk4_reference(vf, x0, 0.0, 1.0, steps=4096)
    rows = {}
    for steps in steps_list:
        row = {}
        for method in ("euler", "heun"):
            x, _, nfe = integrate(vf, x0, 0.0, 1.0, steps, method)
            row[method] = {"rel_err": float((x - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()), "nfe": nfe}
        rows[str(steps)] = row
    out["hand_rows"] = rows
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        sched.set_timesteps(steps_list[-1])
        heun = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        n = steps_list[-1]
        shift = float(sched.config.shift)
        T = int(sched.config.num_train_timesteps)
        lib = sched.sigmas[:-1].to(DTYPE)
        # 用逆映射 u = s/(shift-(shift-1)s) 从库内 sigmas 还原基网格，记录端点与步长
        u = lib / (shift - (shift - 1) * lib)
        # flow match 的输入缩放：库内默认 x / sqrt(sigma^2+1)（该版本没有 scale_model_input 方法，
        # 由 pipeline 自己乘 init_noise_sigma 后直接前向）
        sig = float(sched.sigmas[5])
        out["diffusers"] = {
            "available": True,
            "shift": shift,
            "sigma_first": float(sched.sigmas[0]),
            "sigma_last": float(sched.sigmas[-1]),
            "base_grid_first": float(u[0]),
            "base_grid_last": float(u[-1]),
            "base_grid_step": float((u[0] - u[-1]) / (len(u) - 1)),
            "shift_roundtrip_max_abs_diff": float((shift * u / (1 + (shift - 1) * u) - lib).abs().max()),
            "input_scaling_formula": "x / sqrt(sigma^2 + 1)",
            "input_scaling_at_sigma5": float(1.0 / math.sqrt(sig ** 2 + 1)),
            "has_scale_model_input": hasattr(sched, "scale_model_input"),
            "heun_shift_default": float(heun.config.shift),
        }
    except Exception as exc:  # pragma: no cover
        out["diffusers"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gauss = Gaussian2D(mu=torch.tensor([1.0, -0.5]), var=torch.tensor([0.25, 1.0]))
    gmm = GMM2D(pi=torch.tensor([0.4, 0.35, 0.25]),
                mu=torch.tensor([[1.5, 0.0], [-1.0, 1.0], [-0.5, -1.5]]),
                var=torch.tensor([[0.09, 0.09], [0.16, 0.09], [0.04, 0.25]]))

    rep = {
        "python": platform.python_version(), "torch": torch.__version__, "dtype": "float64",
        "linear_ode": check_linear_ode(),
        "vp_orders": check_vp_orders(gauss, seed=args.seed),
        "history": check_history(gauss),
        "diffusers_vp": check_diffusers_vp(gauss, seed=args.seed),
        "flow_match": check_flow_match(gmm, seed=args.seed),
    }
    text = json.dumps(rep, indent=2, ensure_ascii=False, default=float)
    print(text)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
