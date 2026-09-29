#!/usr/bin/env python
"""10.1-B 解析 Gaussian / 二维 GMM 上的 score、velocity、采样与求解误差。

数据分布已知，因此 VP（DDPM/score）与 FM（rectified flow）的边际 score 与边际 velocity
都有解析式。用它们可以：
  1. 检查 DDIM/DDPM 与 Euler/Heun 的误差如何随步数变化（对照 RK4 数值参照）；
  2. 把「条件插值的直线路径」与「边际向量场的实际轨迹」分开；
  3. 记录 NFE、逐步轨迹与分布指标。

解析场只作参照，不代表任何已训练模型。用法：
  python labs/L10/toy_diffusion_flow.py --out results/local/10.1/<run_id>/toy_fields.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

DTYPE = torch.float64
# 连续 VP 场在 phi=pi/2 处 a=cos(phi)=0，x0 反解除以 a，故积分从略小于 pi/2 处开始
PHI_START = math.pi / 2 - 1e-4


# --------------------------------------------------------------------------------------
# 解析分布
# --------------------------------------------------------------------------------------
def _col(v: torch.Tensor) -> torch.Tensor:
    """把 [N] 形状的系数变成 [N,1]，便于与 [2] 维方差广播。"""
    return v.unsqueeze(-1) if v.dim() == 1 else v


class Gaussian2D:
    """单个二维高斯：p(x) = N(mu, diag(var))。"""

    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        self.mu = mu.to(DTYPE)
        self.var = var.to(DTYPE)

    def score_vp(self, x: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """VP 扰动后 p_t = N(a*mu, a^2*var + s^2)，score = (a*mu - x) / D。"""
        a, s = _col(a), _col(s)
        D = a * a * self.var + s * s
        return (a * self.mu - x) / D

    def flow_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = (1-t) x_noise + t x_data 的边际速度 E[x_data - x_noise | x_t]。

        x_noise ~ N(0, I)，x_data ~ N(mu, var)，两者独立：
          D = (1-t)^2 + t^2 var,  m = t mu
          E[x_noise|x_t] = (1-t)/D (x - m)
          E[x_data|x_t]  = mu + t var/D (x - m)
        """
        tc = _col(t)
        A = 1 - tc
        D = A ** 2 + tc ** 2 * self.var
        m = tc * self.mu
        e_noise = A / D * (x - m)
        e_data = self.mu + tc * self.var / D * (x - m)
        return e_data - e_noise

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return self.mu + torch.sqrt(self.var) * torch.randn(n, 2, generator=g, dtype=DTYPE)


class GMM2D:
    """二维高斯混合：p(x) = sum_k pi_k N(mu_k, diag(var_k))。"""

    def __init__(self, pi: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
        self.pi = pi.to(DTYPE)
        self.mu = mu.to(DTYPE)          # [K, 2]
        self.var = var.to(DTYPE)        # [K, 2]

    def _resp_vp(self, x: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a, s = _col(a), _col(s)
        logp = []
        for k in range(self.mu.shape[0]):
            D = a * a * self.var[k] + s * s
            d = x - a * self.mu[k]
            logp.append(torch.log(self.pi[k]) - 0.5 * (torch.log(D).sum(dim=-1) + (d * d / D).sum(dim=-1)))
        return torch.softmax(torch.stack(logp, dim=0), dim=0)      # [K, N]

    def score_vp(self, x: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a, s = _col(a), _col(s)
        w = self._resp_vp(x, a, s)
        out = torch.zeros_like(x)
        for k in range(self.mu.shape[0]):
            D = a * a * self.var[k] + s * s
            out = out + w[k][:, None] * (a * self.mu[k] - x) / D
        return out

    def _resp_flow(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tc = _col(t)
        A = 1 - tc
        logp = []
        for k in range(self.mu.shape[0]):
            D = A ** 2 + tc * tc * self.var[k]
            d = x - tc * self.mu[k]
            logp.append(torch.log(self.pi[k]) - 0.5 * (torch.log(D).sum(dim=-1) + (d * d / D).sum(dim=-1)))
        return torch.softmax(torch.stack(logp, dim=0), dim=0)

    def flow_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tc = _col(t)
        A = 1 - tc
        w = self._resp_flow(x, t)
        out = torch.zeros_like(x)
        for k in range(self.mu.shape[0]):
            D = A ** 2 + tc ** 2 * self.var[k]
            m = tc * self.mu[k]
            e_noise = A / D * (x - m)
            e_data = self.mu[k] + tc * self.var[k] / D * (x - m)
            out = out + w[k][:, None] * (e_data - e_noise)
        return out

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        k = torch.multinomial(self.pi, n, replacement=True, generator=g)
        u = torch.randn(n, 2, generator=g, dtype=DTYPE)
        return self.mu[k] + torch.sqrt(self.var[k]) * u


# --------------------------------------------------------------------------------------
# VP 采样器
# --------------------------------------------------------------------------------------
def ddim_sample(model, x_start, idx_grid, alphas_cumprod, score_fn):
    """时间索引从 idx_grid[0]（噪声端）走到 idx_grid[-1]=0（数据端）的确定性 DDIM。"""
    x = x_start
    traj = [x.clone()]
    for i in range(len(idx_grid) - 1):
        t, tp = idx_grid[i], idx_grid[i + 1]
        ab, abp = alphas_cumprod[t], alphas_cumprod[tp]
        a, s = torch.sqrt(ab), torch.sqrt(1 - ab)
        ap, sp = torch.sqrt(abp), torch.sqrt(1 - abp)
        score = score_fn(x, a, s)
        eps = -s * score
        x0 = (x + s * s * score) / a
        x = ap * x0 + sp * eps
        traj.append(x.clone())
    return x, traj, len(idx_grid) - 1


def ddpm_ancestral_sample(model, x_start, idx_grid, betas, alphas_cumprod, score_fn, g):
    """DDPM 祖先采样：均值用论文式 (7)，方差用 beta_tilde。

    跳过中间索引时，式中的 alpha_t/beta_t 是这一步跨越的 alpha_bar 比值，
    不是单步表里的 betas[t]；混用会在粗网格上把方差放大到 1。
    """
    x = x_start
    traj = [x.clone()]
    for i in range(len(idx_grid) - 1):
        t, tp = idx_grid[i], idx_grid[i + 1]
        ab, abp = alphas_cumprod[t], alphas_cumprod[tp]
        alpha_jump = ab / abp
        beta_jump = 1 - alpha_jump
        a, s = torch.sqrt(ab), torch.sqrt(1 - ab)
        score = score_fn(x, a, s)
        x0 = (x + s * s * score) / a
        mean = (torch.sqrt(abp) * beta_jump / (1 - ab)) * x0 + (torch.sqrt(alpha_jump) * (1 - abp) / (1 - ab)) * x
        beta_tilde = (1 - abp) / (1 - ab) * beta_jump
        x = mean + torch.sqrt(beta_tilde) * torch.randn_like(x, generator=g)
        traj.append(x.clone())
    return x, traj, len(idx_grid) - 1


# --------------------------------------------------------------------------------------
# 连续时间 ODE 求解
# --------------------------------------------------------------------------------------
def _time_vec(x: torch.Tensor, val: float) -> torch.Tensor:
    return torch.full((x.shape[0],), val, dtype=DTYPE)


def rk4_reference(field, x0, t0, t1, steps=4096):
    """RK4 高精度数值参照；只作参照，不当作真值。"""
    x = x0
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = field(x, _time_vec(x, t))
        k2 = field(x + 0.5 * h * k1, _time_vec(x, t + 0.5 * h))
        k3 = field(x + 0.5 * h * k2, _time_vec(x, t + 0.5 * h))
        k4 = field(x + h * k3, _time_vec(x, t + h))
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return x


def euler_flow(field, x0, t0, t1, steps):
    x = x0
    h = (t1 - t0) / steps
    t = t0
    traj = [x.clone()]
    for _ in range(steps):
        x = x + h * field(x, _time_vec(x, t))
        t = t + h
        traj.append(x.clone())
    return x, traj


def heun_flow(field, x0, t0, t1, steps):
    x = x0
    h = (t1 - t0) / steps
    t = t0
    traj = [x.clone()]
    nfe = 0
    for _ in range(steps):
        k1 = field(x, _time_vec(x, t))
        x_pred = x + h * k1
        k2 = field(x_pred, _time_vec(x, t + h))
        x = x + 0.5 * h * (k1 + k2)
        t = t + h
        nfe += 2
        traj.append(x.clone())
    return x, traj, nfe


def vp_field_from_score(score_fn):
    """VP 概率流 ODE 在角度时间下的向量场：dx/dphi = a*eps - s*x0 = v。"""
    def field(x, phi):
        a = torch.cos(phi)
        s = torch.sin(phi)
        score = score_fn(x, a, s)
        a, s = _col(a), _col(s)
        eps = -s * score
        x0 = (x + s * s * score) / a
        return a * eps - s * x0
    return field


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------
def mean_cov(x):
    m = x.mean(dim=0)
    xc = x - m
    return m, xc.t() @ xc / (x.shape[0] - 1)


def w2_gaussian(x, mu, var):
    """样本高斯拟合与目标高斯（对角 var）之间的 2-Wasserstein 距离。"""
    m, cov = mean_cov(x)
    tr = (cov.diagonal() + var).sum() - 2 * torch.sqrt(cov.diagonal() * var).sum()
    return float(torch.sqrt(torch.clamp((m - mu).pow(2).sum() + tr, min=0.0)))


def energy_distance(x, y):
    """无偏 energy distance：2E|X-Y| - E|X-X'| - E|Y-Y'|，对角线排除。"""
    def offdiag_mean(a):
        d = torch.cdist(a, a)
        n = a.shape[0]
        return d.sum() / (n * (n - 1))

    xy = torch.cdist(x, y).mean()
    return float(2 * xy - offdiag_mean(x) - offdiag_mean(y))


def curvature(traj_stacked):
    """轨迹相对首末端点连线的最大垂距，按线段长度归一化。"""
    p0, p1 = traj_stacked[0], traj_stacked[-1]
    d = p1 - p0
    L = d.norm(dim=1).clamp(min=1e-12)
    dev = []
    for p in traj_stacked[1:-1]:
        rel = p - p0
        proj = (rel * d).sum(dim=1) / (L ** 2)
        perp = rel - proj[:, None] * d
        dev.append(perp.norm(dim=1) / L)
    return float(torch.stack(dev, dim=1).max())


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 32, 128])
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    rep: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float64",
        "seed": args.seed,
        "n_samples": args.samples,
        "phi_start": PHI_START,
        "note": "解析场参照；误差相对 RK4(4096 步) 数值参照或闭式高斯 W2，不涉及任何训练模型。",
    }

    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, dtype=DTYPE)
    abc = torch.cumprod(1.0 - betas, dim=0)

    gauss = Gaussian2D(mu=torch.tensor([1.0, -0.5]), var=torch.tensor([0.25, 1.0]))
    gmm = GMM2D(
        pi=torch.tensor([0.4, 0.35, 0.25]),
        mu=torch.tensor([[1.5, 0.0], [-1.0, 1.0], [-0.5, -1.5]]),
        var=torch.tensor([[0.09, 0.09], [0.16, 0.09], [0.04, 0.25]]),
    )
    models = {"gaussian": gauss, "gmm": gmm}

    # ---------------- VP 离散：DDIM 与 DDPM 祖先采样的分布误差 ----------------
    vp = {}
    for steps in args.steps:
        idx = torch.linspace(T - 1, 0, steps + 1).round().long()
        x_start = torch.randn(args.samples, 2, generator=g, dtype=DTYPE)
        for name, model in models.items():
            sf = (lambda x, a, s, m=model: m.score_vp(x, a, s))
            x_out, _, nfe = ddim_sample(model, x_start, idx, abc, sf)
            g2 = torch.Generator().manual_seed(args.seed + 1)
            x_out2, _, nfe2 = ddpm_ancestral_sample(model, x_start.clone(), idx, betas, abc, sf, g2)
            target = model.sample(args.samples, g)
            if name == "gaussian":
                m1 = w2_gaussian(x_out, model.mu, model.var)
                m2 = w2_gaussian(x_out2, model.mu, model.var)
            else:
                m1 = energy_distance(x_out, target)
                m2 = energy_distance(x_out2, target)
            vp[f"{name}_{steps}"] = {
                "ddim": {"nfe": nfe, "dist_metric": m1},
                "ddpm_ancestral": {"nfe": nfe2, "dist_metric": m2},
            }
    rep["vp_discrete"] = vp
    rep["energy_noise_floor"] = energy_distance(gmm.sample(1024, g), gmm.sample(1024, g))
    rep["w2_noise_floor"] = w2_gaussian(gauss.sample(args.samples, g), gauss.mu, gauss.var)

    # ---------------- VP 端点敏感度：反解 x0 的 1/a 放大 ----------------
    sens = []
    x0_t = gauss.sample(512, g)
    eps_t = torch.randn(512, 2, generator=g, dtype=DTYPE)
    for a_val in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        a = torch.full((512,), a_val, dtype=DTYPE)
        s = torch.sqrt(1 - a * a)
        x_t = a[:, None] * x0_t + s[:, None] * eps_t
        x0_hat = (x_t - s[:, None] * eps_t) / a[:, None]
        sens.append({
            "a": a_val,
            "rel_err_x0": float((x0_hat - x0_t).norm(dim=1).mean() / x0_t.norm(dim=1).mean()),
            "predicted_scale": 1e-16 / a_val,
        })
    rep["vp_endpoint_sensitivity"] = sens

    # ---------------- VP 连续 ODE 的求解阶数 ----------------
    conv = {}
    for name, model in models.items():
        sf = (lambda x, a, s, m=model: m.score_vp(x, a, s))
        field = vp_field_from_score(sf)
        x0 = torch.randn(1024, 2, generator=g, dtype=DTYPE)
        ref = rk4_reference(field, x0, PHI_START, 0.0, steps=4096)
        rows = {}
        for steps in args.steps:
            xe, traje = euler_flow(field, x0, PHI_START, 0.0, steps)
            xh, _, nfe_h = heun_flow(field, x0, PHI_START, 0.0, steps)
            rows[str(steps)] = {
                "euler_rel_err": float((xe - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()),
                "euler_nfe": steps,
                "heun_rel_err": float((xh - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()),
                "heun_nfe": nfe_h,
                "euler_traj_curvature": curvature(torch.stack(traje)),
            }
        s0, s1 = str(args.steps[0]), str(args.steps[-1])
        ratio = args.steps[-1] / args.steps[0]
        rows["observed_order"] = {
            "euler": float(math.log(rows[s0]["euler_rel_err"] / rows[s1]["euler_rel_err"]) / math.log(ratio)),
            "heun": float(math.log(rows[s0]["heun_rel_err"] / rows[s1]["heun_rel_err"]) / math.log(ratio)),
        }
        conv[name] = rows
    rep["vp_ode_convergence"] = conv

    # ---------------- FM：边际向量场与条件直线 ----------------
    fm = {}
    for name, model in models.items():
        vf = (lambda x, t, m=model: m.flow_velocity(x, t))
        x0 = torch.randn(1024, 2, generator=g, dtype=DTYPE)
        ref = rk4_reference(vf, x0, 0.0, 1.0, steps=4096)
        rows = {}
        for steps in args.steps:
            xe, traje = euler_flow(vf, x0, 0.0, 1.0, steps)
            xh, _, nfe_h = heun_flow(vf, x0, 0.0, 1.0, steps)
            rows[str(steps)] = {
                "euler_rel_err": float((xe - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()),
                "heun_rel_err": float((xh - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()),
                "heun_nfe": nfe_h,
            }
        rows["marginal_trajectory_curvature"] = curvature(torch.stack(traje))
        x_data = model.sample(x0.shape[0], g)
        line = torch.stack([(1 - tt) * x0 + tt * x_data
                            for tt in torch.linspace(0, 1, args.steps[-1] + 1, dtype=DTYPE)])
        rows["conditional_line_curvature"] = curvature(line)
        rows["marginal_sample_energy"] = energy_distance(ref, model.sample(ref.shape[0], g))
        fm[name] = rows
    rep["fm"] = fm

    # ---------------- 固定初始点的逐步轨迹（可重放） ----------------
    x_fixed = torch.tensor([[2.0, -1.5]], dtype=DTYPE)
    _, tr = euler_flow(lambda x, t: gmm.flow_velocity(x, t), x_fixed, 0.0, 1.0, 8)
    rep["fixed_point_flow_trajectory_8step"] = [[float(v) for v in p[0]] for p in tr]
    sf = lambda x, a, s: gmm.score_vp(x, a, s)
    idx = torch.linspace(T - 1, 0, 9).round().long()
    _, tr2, _ = ddim_sample(gmm, x_fixed, idx, abc, sf)
    rep["fixed_point_ddim_trajectory_8step"] = [[float(v) for v in p[0]] for p in tr2]

    text = json.dumps(rep, indent=2, ensure_ascii=False, default=float)
    print(text)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
