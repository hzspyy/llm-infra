#!/usr/bin/env python3
"""扩散/flow 蒸馏的目标与状态契约（7.7-G/H）：解析场验证 + 四份公开实现定位。

`objectives`（CPU，无随机拟合）用一维高斯场把两类目标写成闭式，因此可以逐项检查：

  * LCM：参数化 `f(x,σ)=c_skip(σ)x+c_out(σ)g_θ(x,σ)`、教师 DDIM 一步得到的相邻点、
    一致性 MSE，以及 σ=0 处"边界条件把数值固定为 x、梯度恒为 0"。
  * DMD2：生成器 `x=μ_θ+s_θε`、真假 score 之差乘 ∂x/∂θ 的估计量，与
    `KL(p_fake‖p_real)` 的解析梯度逐项对拍；再跑交替更新（critic 拟合假 score、
    generator 走 KL 梯度）看分布差距是否单调下降。

`assumptions` 显式写出场、参数化与估计量，避免把"解析场上的等价"读成"真实蒸馏可复现"。

`sources` 只读仓库内快照，定位 LCM/DMD2/ZipVoice/ModelOpt 的阶段、损失、优化器与可训练模块。

Usage:
    python labs/L7/distill_generative_contracts.py --section objectives --outdir "$RUN_DIR/generative"
    python labs/L7/distill_generative_contracts.py --section sources \
      --source-root results/local/7.7/<run>/source --outdir "$RUN_DIR/generative-src"
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

SIGMA_DATA = 0.5
SIGMA_MIN = 0.02
SIGMA_MAX = 5.0


# ---------------------------------------------------------------------------
# 解析场：一维两分量高斯混合 + VE 扩散
# ---------------------------------------------------------------------------
def mixture_params():
    return {"weights": [0.5, 0.5], "means": [-1.5, 1.5], "std": 0.5}


def posterior(x, sigma):
    """返回每个分量在给定噪声水平下的后验权重。"""
    field = mixture_params()
    total = x.new_zeros(x.shape)
    weights = []
    for w, mu in zip(field["weights"], field["means"]):
        var = field["std"] ** 2 + sigma ** 2               # sigma 可能是逐元素张量
        density = (w * torch.exp(-0.5 * (x - mu) ** 2 / var)
                   / torch.sqrt(2 * math.pi * var))
        weights.append(density)
        total = total + density
    return [item / total for item in weights]


def denoiser(x, sigma):
    """解析去噪器 D(x,σ)=E[x0|x_σ=x]，VE 形式下 D=x+σ²·score。"""
    field = mixture_params()
    var = field["std"] ** 2 + sigma ** 2
    out = torch.zeros_like(x)
    for weight, mu in zip(posterior(x, sigma), field["means"]):
        out = out + weight * (field["std"] ** 2 * mu + sigma ** 2 * x) / var
    return out


def score(x, sigma):
    return (denoiser(x, sigma) - x) / sigma ** 2


def ddim_step(x, sigma_start, sigma_end):
    """VE 概率流 ODE 的欧拉（DDIM 同形）一步：dx/dσ = -σ·score。"""
    return x - sigma_start * (sigma_end - sigma_start) * score(x, sigma_start)


def scalings_for_boundary_conditions(timestep, sigma_data=SIGMA_DATA, timestep_scaling=10.0):
    """与 diffusers `scalings_for_boundary_conditions` 同一公式。"""
    scaled = timestep * timestep_scaling
    c_skip = sigma_data ** 2 / (scaled ** 2 + sigma_data ** 2)
    c_out = scaled / (scaled ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


class Student(torch.nn.Module):
    """g_θ(x,σ)：线性基上的小模型，够用来检查梯度流与边界条件。"""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))
        self.c = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, sigma):
        return self.a + self.b * x + self.c * sigma


def lcm_objective(seed=0, batch=512):
    torch.manual_seed(seed)
    student = Student()
    x0 = torch.randn(batch)
    field = mixture_params()
    component = (torch.rand(batch) > 0.5).long()
    mu = torch.tensor([field["means"][i] for i in component.tolist()])
    x0 = mu + field["std"] * x0

    sigma_start = torch.full((batch,), 2.0)
    sigma_end = torch.full((batch,), 1.0)
    noise = torch.randn(batch)
    x_start = x0 + sigma_start * noise

    c_skip_start, c_out_start = scalings_for_boundary_conditions(sigma_start)
    c_skip_end, c_out_end = scalings_for_boundary_conditions(sigma_end)
    model_pred = c_skip_start * x_start + c_out_start * student(x_start, sigma_start)

    with torch.no_grad():
        x_end = ddim_step(x_start, sigma_start, sigma_end)          # 教师轨迹上的相邻点
        target = c_skip_end * x_end + c_out_end * denoiser(x_end, sigma_end)
    loss = torch.nn.functional.mse_loss(model_pred, target)
    grad = torch.autograd.grad(loss, list(student.parameters()), retain_graph=True)
    grad_norm = math.sqrt(sum(float(g ** 2) for g in grad))

    # 数值梯度对拍（中心差分，只查第一个参数）
    eps = 1e-4
    numeric = []
    for param in student.parameters():
        original = param.detach().clone()
        with torch.no_grad():
            param.copy_(original + eps)
        plus = float(torch.nn.functional.mse_loss(
            c_skip_start * x_start + c_out_start * student(x_start, sigma_start),
            target).detach())
        with torch.no_grad():
            param.copy_(original - eps)
        minus = float(torch.nn.functional.mse_loss(
            c_skip_start * x_start + c_out_start * student(x_start, sigma_start),
            target).detach())
        with torch.no_grad():
            param.copy_(original)
        numeric.append((plus - minus) / (2 * eps))

    # 边界条件：σ̃=0 时 f(x,0)=x，且对 θ 的导数恒为 0
    c_skip_zero, c_out_zero = scalings_for_boundary_conditions(torch.zeros(1))
    x_probe = torch.tensor([0.7])
    model_zero = c_skip_zero * x_probe + c_out_zero * student(x_probe, torch.zeros(1))
    boundary_grad = torch.autograd.grad(model_zero.sum(), list(student.parameters()),
                                        retain_graph=False)

    # 一次梯度下降后的 loss
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = float(torch.nn.functional.mse_loss(
            c_skip_start * x_start + c_out_start * student(x_start, sigma_start), target))
    return {
        "loss_initial": float(loss), "loss_after_one_step": after,
        "grad_norm": grad_norm,
        "grad_autograd": [float(g) for g in grad],
        "grad_numeric_central_difference": numeric,
        "grad_max_abs_diff": max(abs(float(g) - n) for g, n in zip(grad, numeric)),
        "c_skip_at_sigma_end": float(c_skip_end[0]), "c_out_at_sigma_end": float(c_out_end[0]),
        "boundary_condition": {
            "c_skip_at_timestep_0": float(c_skip_zero[0]),
            "c_out_at_timestep_0": float(c_out_zero[0]),
            "model_output_minus_x_at_timestep_0": float((model_zero - x_probe).abs().max()),
            "grad_wrt_theta_max_abs": max(float(g.abs().max()) for g in boundary_grad),
        },
    }


def dmd2_objective(mu_data=0.0, std_data=1.0, alternations=6, samples=4096, seed=1):
    """真假 score 之差乘 ∂x/∂θ 的估计量 vs KL 的解析梯度，并跑交替更新。"""
    torch.manual_seed(seed)
    mu = torch.nn.Parameter(torch.tensor(2.0))
    log_s = torch.nn.Parameter(torch.tensor(math.log(0.6)))

    def fake_score(x):
        var = torch.exp(log_s) ** 2
        return -(x - mu) / var

    def real_score(x):
        return -(x - mu_data) / std_data ** 2

    def kl_exact():
        s = torch.exp(log_s)
        return (torch.log(torch.tensor(std_data) / s)
                + (s ** 2 + (mu - mu_data) ** 2) / (2 * std_data ** 2) - 0.5)

    # 估计量：E[(s_fake - s_real)·∂x/∂θ]，∂x/∂μ=1、∂x/∂log_s=ε
    x = mu + torch.exp(log_s) * torch.randn(samples)
    diff = (fake_score(x) - real_score(x)).detach()
    # x=μ+s·ε ⇒ ∂x/∂μ=1、∂x/∂log s=s·ε=x-μ
    estimate = {"mu": float((diff * 1.0).mean()),
                "log_s": float((diff * (x - mu).detach()).mean())}
    exact = torch.autograd.grad(kl_exact(), [mu, log_s])
    exact_values = {"mu": float(exact[0]), "log_s": float(exact[1])}

    history, critic = [], {"a": 0.0, "b": 0.0}
    for _ in range(alternations):
        # critic 步：用假样本上的去噪 MSE 拟合假 score（闭式最小二乘）
        x = (mu + torch.exp(log_s) * torch.randn(samples)).detach()
        target = (-(x - mu.detach()) / torch.exp(log_s).detach() ** 2)
        design = torch.stack([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
        critic = {"a": float(solution[0]), "b": float(solution[1])}
        critic_loss = float(((design @ solution - target) ** 2).mean())
        # generator 步：用 critic 与真 score 之差作为优势，沿 KL 梯度下降
        with torch.no_grad():
            x = mu + torch.exp(log_s) * torch.randn(samples)
            advantage = critic["a"] * x + critic["b"] - real_score(x)
        grad_mu = float((advantage * 1.0).mean())
        grad_log_s = float((advantage * (x - mu).detach()).mean())
        with torch.no_grad():
            mu -= 0.05 * grad_mu
            log_s -= 0.05 * grad_log_s
        history.append({"kl": float(kl_exact()), "critic_loss": critic_loss,
                        "grad_mu": grad_mu, "grad_log_s": grad_log_s,
                        "mu": float(mu), "std": float(torch.exp(log_s))})
    return {
        "gradient_estimator": estimate, "gradient_exact": exact_values,
        "gradient_max_abs_diff": max(abs(estimate[k] - exact_values[k]) for k in estimate),
        "kl_initial": history[0]["kl"], "kl_final": history[-1]["kl"],
        "history": history,
        "state": {"generator_params": ["mu", "log_s"], "critic_params": ["a", "b"],
                  "generator_optimizer": "SGD(0.05) 于 mu/log_s",
                  "critic_optimizer": "解析最小二乘（每步重解）",
                  "ema": "本节不含 EMA：DMD2 的 EMA 只用于评估权重"},
    }


# ---------------------------------------------------------------------------
# 源码定位
# ---------------------------------------------------------------------------
PROBES = {
    "lcm": {
        "file": "lcm/train_lcm_distill_sdxl_wds.py",
        "items": {
            "边界条件公式": r"def scalings_for_boundary_conditions",
            "EMA 更新": r"def update_ema",
            "引导尺度嵌入": r"def guidance_scale_embedding",
            "教师 CFG 组合": r"pred_x0 = cond_pred_x0 \+ w \* \(cond_pred_x0 - uncond_pred_x0\)",
            "教师 DDIM 一步": r"x_prev = solver\.ddim_step",
            "在线学生输出": r"model_pred = c_skip_start \* noisy_model_input",
            "一致性目标": r"target = c_skip \* x_prev \+ c_out \* pred_x_0",
            "损失": r"loss = F\.mse_loss\(model_pred\.float\(\), target\.float\(\)",
            "EMA 调用": r"update_ema\(target_unet\.parameters\(\), unet\.parameters\(\)",
        },
    },
    "dmd2": {
        "file": "dmd2/train_sd.py",
        "items": {
            "generator 优化器": r"self\.optimizer_generator = torch\.optim\.AdamW",
            "guidance 优化器": r"self\.optimizer_guidance = torch\.optim\.AdamW",
            "交替比例": r"COMPUTE_GENERATOR_GRADIENT = self\.step % self\.dfake_gen_update_ratio",
            "generator 前向": r"generator_turn=True",
            "distribution matching 权重": r"generator_loss \+= generator_loss_dict\[\"loss_dm\"\]",
            "分类损失权重": r"generator_loss \+= generator_loss_dict\[\"gen_cls_loss\"\]",
            "generator 更新": r"self\.optimizer_generator\.step\(\)",
            "guidance 更新": r"self\.optimizer_guidance\.step\(\)",
        },
    },
}


def locate(text: str, pattern: str):
    for index, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return index, line.strip()
    return None, None


def sources_section(source_root: Path) -> dict:
    result = {"section": "generative-distill-sources", "frameworks": {}, "missing": []}
    for name, spec in PROBES.items():
        text = (source_root / spec["file"]).read_text(encoding="utf-8", errors="replace")
        rows = {}
        for label, pattern in spec["items"].items():
            line_no, code = locate(text, pattern)
            rows[label] = {"line": line_no, "code": code}
            if line_no is None:
                result["missing"].append(f"{name}:{label}")
        result["frameworks"][name] = {"file": spec["file"], "items": rows}

    run_sh = (source_root / "zipvoice" / "run_emilia.sh").read_text(encoding="utf-8")
    stop_stage = re.search(r"stop_stage=(\d+)", run_sh)
    commands = []
    for block in re.split(r"\n(?=\s*python3 -m )", run_sh):
        match = re.match(r"\s*python3 -m ([\w.]+)", block)
        if not match:
            continue
        body = []
        for line in block.splitlines():
            body.append(line)
            if not line.rstrip().endswith("\\"):
                break
        text = " ".join(body)
        commands.append({"module": match.group(1),
                         "args": dict(re.findall(r"--([\w-]+)\s+([^\s\\]+)", text))})
    result["zipvoice"] = {
        "stop_stage": int(stop_stage.group(1)) if stop_stage else None,
        "stage_labels": re.findall(r'echo "(Stage[^"]+)"', run_sh),
        "commands": commands,
        "distill_invocations": [c for c in commands if "distill" in c["module"]],
        "averaging_command": next((c for c in commands if "averaged_model" in c["module"]), None),
        "onnx_commands": [c for c in commands if "onnx" in c["module"]],
        "configs": sorted(p.name for p in (source_root / "zipvoice" / "conf").glob("*.json")),
    }

    arguments = (source_root / "modelopt" / "ARGUMENTS.md").read_text(encoding="utf-8",
                                                                     errors="replace")
    keys = sorted(set(re.findall(r"^\|\s*`?([a-z0-9_./-]+)`?\s*\|", arguments, re.M)))
    quantize = (source_root / "modelopt" / "quantize.py").read_text(encoding="utf-8",
                                                                   errors="replace")
    result["modelopt"] = {
        "argument_table_keys": keys[:40], "argument_key_count": len(keys),
        "qat_entry": locate(quantize, r"def main|quantize\(")[0],
        "has_distillation_arg": bool(re.search(r"distill|teacher", arguments, re.I)),
        "distillation_mentions": len(re.findall(r"distill|teacher", arguments, re.I)),
    }
    result["claim_scope"] = ("源码定位与脚本阶段解析：行号来自仓库内快照；"
                             "ZipVoice 阶段表与 ModelOpt 参数表是文本解析，不代表任何一次实际运行")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", required=True, choices=["objectives", "sources"])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    if args.section == "objectives":
        torch.manual_seed(0)
        result = {
            "section": "generative-distill-objectives",
            "assumptions": {
                "field": "一维两分量高斯混合（均值 ±1.5、标准差 0.5）+ VE 扩散 x_σ=x0+σε",
                "denoiser": "解析后验均值 D=E[x0|x_σ]；score=(D-x)/σ²",
                "ddim_step": "VE 概率流 ODE 的欧拉一步 dx/dσ=-σ·score",
                "lcm_parameterization": "c_skip=σ_d²/(σ̃²+σ_d²)、c_out=σ̃/√(σ̃²+σ_d²)，σ_d=0.5、σ̃=t·10",
                "lcm_target": "教学上把 target_unet 的理想情形取为同一个解析教师；EMA 只另做收敛检查",
                "dmd2_generator": "x=μ_θ+s_θε（高斯族），真假 score 解析；估计量=E[(s_fake-s_real)·∂x/∂θ]",
                "why_analytic": "闭式场才能把'目标与梯度流'与'拟合是否收敛'分开检查",
            },
            "lcm": lcm_objective(),
            "dmd2": dmd2_objective(),
        }
    else:
        if args.source_root is None:
            raise SystemExit("sources 需要 --source-root")
        result = sources_section(args.source_root)
    (args.outdir / "generative_distill.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
