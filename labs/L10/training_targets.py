#!/usr/bin/env python
"""10.1-C 训练 pair、目标与权重的 FP64 小参照。

同一批二维数据上构造 denoising（VP）与 flow（直线插值）训练对，检查四件容易混淆的事：
  1. 不同 target（eps / x0 / v / score）在同一模型输出下的逐项 loss 之间存在精确换算；
  2. time/noise 采样分布如何改变各时间段的实际权重；
  3. loss weighting 与 parameterization 可以互相抵消，也可以人为制造差别；
  4. 条件 dropout 后的目标是边缘期望（多个分量混合），不是某个分量的目标；
  5. 「训练一步」与「采样一步」是不同对象。

小 MLP 最多做一次更新，只用于检查梯度方向与等价性，不用单步 loss 排名目标。

用法：
  python labs/L10/training_targets.py --out results/local/10.1/<run_id>/training_targets.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

DTYPE = torch.float64


# --------------------------------------------------------------------------------------
def vp_discrete(T: int = 1000, name: str = "linear") -> dict:
    if name == "linear":
        betas = torch.linspace(1e-4, 0.02, T, dtype=DTYPE)
    elif name == "cosine":
        steps = T + 1
        x = torch.linspace(0, T, steps, dtype=DTYPE)
        f = torch.cos(((x / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = f / f[0]
        betas = torch.clip(1 - ab[1:] / ab[:-1], 0.0001, 0.9999)
    else:
        raise ValueError(name)
    abc = torch.cumprod(1.0 - betas, dim=0)
    return {"betas": betas, "alphas_cumprod": abc}


def coeffs(abc, t):
    ab = abc[t]
    return torch.sqrt(ab), torch.sqrt(1.0 - ab)


def col(v):
    return v.unsqueeze(-1) if v.dim() == 1 else v


# --------------------------------------------------------------------------------------
# 数据与阈值：两个分量的二维 GMM，条件 = 分量编号
# --------------------------------------------------------------------------------------
def gmm_sample(n, comp, g):
    mu = torch.tensor([[1.6, 0.2], [-1.2, -0.9]], dtype=DTYPE)
    var = torch.tensor([[0.09, 0.16], [0.25, 0.04]], dtype=DTYPE)
    return mu[comp] + torch.sqrt(var[comp]) * torch.randn(n, 2, generator=g, dtype=DTYPE)


def class_priors():
    return torch.tensor([0.65, 0.35], dtype=DTYPE)


# --------------------------------------------------------------------------------------
# 精确换算：同一模型输出在四种参数化下的逐样本 loss 关系
#   设模型给出 (x0_hat, eps_hat)，两者都复现同一个 x_t：a*Δx0 + s*Δeps = 0
#   于是 L_v = L_eps / a^2, L_x0 = (s^2/a^2) L_eps, L_score = L_eps / s^2
# --------------------------------------------------------------------------------------
def check_loss_relations(abc, seed: int = 0, n: int = 4096) -> dict:
    g = torch.Generator().manual_seed(seed)
    T = abc.numel()
    t = torch.randint(50, T - 50, (n,), generator=g)
    a, s = coeffs(abc, t)
    a, s = col(a), col(s)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    eps = torch.randn(n, 2, generator=g, dtype=DTYPE)
    x_t = a * x0 + s * eps

    # 任意模型输出：这里用真实目标加扰动，保证 (x0_hat, eps_hat) 一致
    d_eps = 0.1 * torch.randn(n, 2, generator=g, dtype=DTYPE)
    eps_hat = eps - d_eps
    x0_hat = (x_t - s * eps_hat) / a

    def mse(u, v):
        return ((u - v) ** 2).mean(dim=1)

    L_eps = mse(eps, eps_hat)
    L_x0 = mse(x0, x0_hat)
    L_v = mse(a * eps - s * x0, a * eps_hat - s * x0_hat)
    L_score = mse(-eps / s, -eps_hat / s)

    return {
        "max_abs_Lv_minus_Leps_over_a2": float((L_v - L_eps / a[:, 0] ** 2).abs().max()),
        "max_abs_Lx0_minus_s2_over_a2_Leps": float((L_x0 - s[:, 0] ** 2 / a[:, 0] ** 2 * L_eps).abs().max()),
        "max_abs_Lscore_minus_Leps_over_s2": float((L_score - L_eps / s[:, 0] ** 2).abs().max()),
        "mean_Leps_a05": float(L_eps[a[:, 0] < 0.5].mean()),
        "mean_Lv_a05": float(L_v[a[:, 0] < 0.5].mean()),
        "mean_Lv_a05_over_Leps_a05": float((L_v[a[:, 0] < 0.5] / L_eps[a[:, 0] < 0.5]).mean()),
        "note": "同一模型输出的四种 loss 由 a、s 严格换算；比较不同 target 的 loss 值本身没有意义。",
    }


# --------------------------------------------------------------------------------------
# time / noise 采样
# --------------------------------------------------------------------------------------
def sample_time(n, scheme, abc, g):
    """返回 0..T-1 的整数索引。"""
    T = abc.numel()
    u = torch.rand(n, generator=g, dtype=DTYPE)
    if scheme == "uniform":
        t = u
    elif scheme == "logit_normal":
        # SD3 式 logit-normal（sigma=1）
        t = torch.sigmoid(u * 4 - 2)
    elif scheme == "uniform_log_snr":
        # 在 log-SNR 上均匀：log-SNR = log(ab/(1-ab))，随 t 单调递减
        logsnr = torch.log(abc / (1 - abc))
        lo, hi = logsnr[-1], logsnr[0]          # lo < 0 < hi
        target = lo + u * (hi - lo)
        # -logsnr 随 t 单调递增，直接在其中查找 -target；再换回 [0,1] 分数
        idx = torch.searchsorted(-logsnr, -target).clamp(0, T - 1)
        t = idx.to(DTYPE) / (T - 1)
    else:
        raise ValueError(scheme)
    return (t * (T - 1)).round().long().clamp(1, T - 1)


def snr_bands(abc, t):
    ab = abc[t]
    snr = ab / (1 - ab)
    return {
        "low_snr_lt1": float((snr < 1).double().mean()),
        "mid_snr_1_10": float(((snr >= 1) & (snr < 10)).double().mean()),
        "high_snr_ge10": float((snr >= 10).double().mean()),
        "mean_log_snr": float(torch.log(snr).mean()),
    }


# --------------------------------------------------------------------------------------
# 小 MLP：最多一次更新
# --------------------------------------------------------------------------------------
class TinyMLP(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2)
        )

    def forward(self, x_t, t_norm, cond):
        return self.net(torch.cat([x_t, t_norm, cond], dim=1))


def one_update_grad(model, x_t, t_norm, cond, target, weight):
    """一次反向：返回梯度（不更新参数），weight 为逐样本权重 [n]。"""
    model.zero_grad(set_to_none=True)
    pred = model(x_t, t_norm, cond)
    per_sample = ((pred - target) ** 2).mean(dim=1)
    loss = (per_sample * weight).mean()
    loss.backward()
    flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    return loss.detach(), flat.detach()


def check_gradient_equivalence(abc, seed: int = 0, n: int = 2048) -> dict:
    """等价性的正确形式：网络输出 m 被当作 v 时，eps-loss 必须经过换算。

    eps_hat = s*x_t + a*m  ==>  ||eps - eps_hat||^2 = a^2 ||v - m||^2。
    因此「v-loss 乘 a^2」与「换算后的 eps-loss」对 m 的梯度逐元素相同；
    直接把同一个 m 当作 eps 目标来回归是另一件事（错配参数化）。
    """
    g = torch.Generator().manual_seed(seed)
    T = abc.numel()
    t = torch.randint(50, T - 50, (n,), generator=g)
    a, s = coeffs(abc, t)
    a, s = col(a), col(s)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    eps = torch.randn(n, 2, generator=g, dtype=DTYPE)
    x_t = a * x0 + s * eps
    v_true = a * eps - s * x0
    t_norm = (t.to(DTYPE) / T)[:, None]
    cond = torch.zeros(n, 1, dtype=DTYPE)

    torch.manual_seed(seed)
    model = TinyMLP().to(DTYPE)

    def grad_of(fn):
        model.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        return float(loss.detach()), flat.detach()

    w = a[:, 0] ** 2
    l_weighted, g_weighted = grad_of(lambda: (((model(x_t, t_norm, cond) - v_true) ** 2).mean(dim=1) * w).mean())
    l_conv, g_conv = grad_of(lambda: (((s * x_t + a * model(x_t, t_norm, cond) - eps) ** 2).mean(dim=1)).mean())
    l_eps, g_eps = grad_of(lambda: ((model(x_t, t_norm, cond) - eps) ** 2).mean())
    l_raw_v, g_raw_v = grad_of(lambda: ((model(x_t, t_norm, cond) - v_true) ** 2).mean())

    def cos(u, v):
        return float((u * v).sum() / (u.norm() * v.norm()))

    return {
        "loss_v_weight_a2": l_weighted,
        "loss_eps_from_conversion": l_conv,
        "max_abs_grad_diff_v_weighted_vs_eps_converted": float((g_weighted - g_conv).abs().max()),
        "relative_grad_gap": float((g_weighted - g_conv).norm() / g_weighted.norm()),
        "loss_eps_raw": l_eps,
        "loss_v_raw": l_raw_v,
        "cos_eps_raw_vs_v_raw": cos(g_eps, g_raw_v),
        "grad_norm_eps_raw": float(g_eps.norm()),
        "grad_norm_v_raw": float(g_raw_v.norm()),
        "note": "v-loss 乘 a^2 与换算后的 eps-loss 等价；把同一输出直接当作另一种参数化回归不是等价变换。",
    }


# --------------------------------------------------------------------------------------
# min-SNR 权重与有效维度归一化
# --------------------------------------------------------------------------------------
def check_weighting(abc, seed: int = 0, n: int = 8192, gamma: float = 5.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    T = abc.numel()
    out = {}
    for scheme in ("uniform", "logit_normal", "uniform_log_snr"):
        t = sample_time(n, scheme, abc, g)
        a, s = coeffs(abc, t)
        snr = (a / s) ** 2
        w_min_snr = torch.clamp(snr, max=gamma) / snr          # min-SNR-gamma
        out[scheme] = {
            **snr_bands(abc, t),
            "mean_min_snr_weight": float(w_min_snr.mean()),
            "share_weight_above_half": float((w_min_snr > 0.5).double().mean()),
            "mean_target_rms_eps": float((torch.sqrt(1 - abc[t])).mean()),
        }
    # 有效维度归一化：同一 t 上 eps 目标的每维方差都是 1，而 x0 目标的方差随 a 变化
    t = sample_time(4096, "uniform", abc, g)
    a, s = coeffs(abc, t)
    out["target_scales"] = {
        "eps_target_std": [1.0, 1.0],
        "x0_target_std_scaled_by_a": [float(a.mean()), float(a.mean())],
        "score_target_std_ratio_over_eps": float((1 / s).mean()),
        "note": "把不同 t 的目标按各自标准差归一化，等于给 loss 乘 1/std^2；这会和 SNR 权重叠加，必须显式记录。",
    }
    return out


# --------------------------------------------------------------------------------------
# 条件 dropout：目标是边缘期望
# --------------------------------------------------------------------------------------
def check_condition_dropout(abc, seed: int = 0, n: int = 20000) -> dict:
    """条件 dropout 的目标变化：分开看「条件解释的目标方差」和「模型误差增益」。

    这里换用两个对称、可分性可控的分量，便于按责任度熵分档观察。
    """
    g = torch.Generator().manual_seed(seed)
    pi = torch.tensor([0.5, 0.5], dtype=DTYPE)
    comp = torch.multinomial(pi, n, replacement=True, generator=g)
    x0 = gmm_sample(n, comp, g)
    mu = torch.tensor([[2.2, 0.0], [-2.2, 0.0]], dtype=DTYPE)
    var = torch.tensor([[0.04, 0.04], [0.04, 0.04]], dtype=DTYPE)
    rows = {}
    for t_val in (100, 200, 400, 800):
        t = torch.full((n,), t_val, dtype=torch.long)
        eps = torch.randn(n, 2, generator=g, dtype=DTYPE)
        a, s = coeffs(abc, t)
        a, s = col(a), col(s)
        x_t = a * x0 + s * eps

        def comp_post(k):
            D_k = a * a * var[k] + s * s
            d_k = x_t - a * mu[k]
            logp_k = torch.log(pi[k]) - 0.5 * (torch.log(D_k).sum(dim=-1) + (d_k * d_k / D_k).sum(dim=-1))
            mean_k = a * mu[k] + s * s / D_k * d_k
            return logp_k, mean_k

        logps, means = [], []
        for k in range(2):
            lp, mn = comp_post(k)
            logps.append(lp)
            means.append(mn)
        w = torch.softmax(torch.stack(logps, dim=0), dim=0)
        x0_marginal = sum(w[k][:, None] * means[k] for k in range(2))
        x0_cond = means[0] * (comp == 0).unsqueeze(1) + means[1] * (comp == 1).unsqueeze(1)
        # 全方差分解：条件能解释的目标方差 = total - within
        total = float(x0_cond.var(dim=0).mean())
        within = float(sum(((x0_cond[comp == k] - x0_cond[comp == k].mean(0)) ** 2).mean() * (comp == k).double().mean()
                           for k in range(2)))
        err_cond = float(((x0 - x0_cond) ** 2).mean())
        err_marg = float(((x0 - x0_marginal) ** 2).mean())
        ent = -(w * torch.log(w + 1e-30)).sum(0) / math.log(2)
        ambiguous = ent > 0.9
        gap = ((x0_marginal - x0_cond) ** 2).mean(dim=1)
        rows[str(t_val)] = {
            "a": float(a[0, 0]),
            "target_total_var": total,
            "target_within_component_var": within,
            "target_between_component_var": total - within,
            "mse_with_condition": err_cond,
            "mse_dropped_condition": err_marg,
            "mse_gap_dropped_minus_kept": err_marg - err_cond,
            "mean_responsibility_entropy_over_log2": float(ent.mean()),
            "argmax_matches_true_component": float((w.argmax(0) == comp).double().mean()),
            "ambiguous_share": float(ambiguous.double().mean()),
            "mse_gap_on_ambiguous": float(gap[ambiguous].mean()) if bool(ambiguous.any()) else 0.0,
        }
    return {
        "rows": rows,
        "note": "条件丢弃后目标是边缘期望 E[x0|x_t]；t 越小（SNR 越高）分量越可分，边缘目标与真实分量目标的差别越大。",
    }


# --------------------------------------------------------------------------------------
# 训练一步 vs 采样一步
# --------------------------------------------------------------------------------------
def check_train_vs_sample(abc, seed: int = 0, n: int = 512) -> dict:
    """训练一步只算 loss 与梯度；采样一步要用更新后的参数重新前向并按后验更新 x。"""
    g = torch.Generator().manual_seed(seed)
    T = abc.numel()
    t = torch.full((n,), 500, dtype=torch.long)
    a, s = coeffs(abc, t)
    a, s = col(a), col(s)
    x0 = torch.randn(n, 2, generator=g, dtype=DTYPE)
    eps = torch.randn(n, 2, generator=g, dtype=DTYPE)
    x_t = a * x0 + s * eps
    torch.manual_seed(seed)
    model = TinyMLP().to(DTYPE)
    t_norm = (t.to(DTYPE) / T)[:, None]
    cond = torch.zeros(n, 1, dtype=DTYPE)

    loss, grad = one_update_grad(model, x_t, t_norm, cond, eps, torch.ones(n, dtype=DTYPE))
    # 更新一步
    lr = 1e-3
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            k = p.numel()
            p -= lr * grad[off:off + k].reshape(p.shape)
            off += k
    # 采样一步（DDIM，使用更新后的模型预测）
    with torch.no_grad():
        eps_hat = model(x_t, t_norm, cond)
    a_prev, s_prev = coeffs(abc, t - 10)
    a_prev, s_prev = col(a_prev), col(s_prev)
    x0_hat = (x_t - s * eps_hat) / a
    x_prev = a_prev * x0_hat + s_prev * eps_hat
    return {
        "train_step_loss": float(loss),
        "grad_norm": float(grad.norm()),
        "sample_step_x_change_rms": float((x_prev - x_t).pow(2).mean().sqrt()),
        "sample_step_nfe": 1,
        "note": "训练一步更新参数、不移动 x；采样一步移动 x、不改参数。两者的时间索引、目标与状态都不同。",
    }


# --------------------------------------------------------------------------------------
def check_flow_targets(seed: int = 0, n: int = 2048) -> dict:
    """FM 路径：u = x_data - x_noise；同一 u 与 VP 的 v 不是同一对象。"""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g, dtype=DTYPE)
    x_data = torch.randn(n, 2, generator=g, dtype=DTYPE)
    x_noise = torch.randn(n, 2, generator=g, dtype=DTYPE)
    u = x_data - x_noise
    tc = col(t)
    x_t = (1 - tc) * x_noise + tc * x_data
    return {
        "target_mean_norm": float(u.norm(dim=1).mean()),
        "x_data_reconstruction_max_err": float((x_t + (1 - tc) * u - x_data).abs().max()),
        "x_noise_reconstruction_max_err": float((x_t - tc * u - x_noise).abs().max()),
        "note": "flow 的 u 与 DDPM 的 v（a*eps - s*x0）维度同、含义不同，不能互相代入。",
    }


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    abc = vp_discrete()["alphas_cumprod"]
    rep = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float64",
        "seed": args.seed,
        "loss_relations": check_loss_relations(abc, args.seed),
        "gradient_equivalence": check_gradient_equivalence(abc, args.seed),
        "weighting": check_weighting(abc, args.seed),
        "condition_dropout": check_condition_dropout(abc, args.seed),
        "train_vs_sample": check_train_vs_sample(abc, args.seed),
        "flow_targets": check_flow_targets(args.seed),
    }
    text = json.dumps(rep, indent=2, ensure_ascii=False, default=float)
    print(text)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
