#!/usr/bin/env python3
"""跨阶段失败定位练习（7.10-I）：四个"看起来在训练、其实监督接错了"的注入用例。

每个用例都给出三样东西：**症状**（loss 曲线看起来正常）、**可观测差别**（哪个字段/统计量
不同）、**定位判据**（先查什么）。用例都是小实现，不加载真实模型；它们要证明的是"检测点
存在且可计算"，不是复现某次真实故障。

  mask    视觉仍在编码，但 loss mask 把 prompt 也监督了
  codec   音频 codec 的帧率/码本与标签不一致
  flow    flow matching 的 prediction_type 用错
  action  动作归一化统计量用错来源

Usage:
    python labs/L7/multimodal_failure_localization.py --outdir "$RUN_DIR/mm-failure"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def case_mask(seed=0):
    """视觉特征照常进入序列，只有监督位置写错。"""
    torch.manual_seed(seed)
    n_prompt, n_image, n_answer = 4, 3, 5
    vision = torch.randn(1, n_image, 8, requires_grad=True)
    text = torch.randn(1, n_prompt + n_answer, 8)
    tokens = torch.cat([vision, text], dim=1)                 # 图像 token 在序列前部
    tokens = tokens + tokens.mean(dim=1, keepdim=True)        # 位置间混合：否则逐位置 head 看不到图像
    head = torch.nn.Linear(8, 7, bias=False)
    labels = torch.randint(0, 7, (1, n_prompt + n_answer))
    total = n_image + n_prompt + n_answer
    answer_positions = list(range(n_image + n_prompt, total))
    prompt_positions = list(range(n_image, n_image + n_prompt))

    def loss_with(supervise_prompt: bool):
        logits = head(tokens)[0]
        positions = answer_positions + (prompt_positions if supervise_prompt else [])
        target = labels[0][[position - n_image for position in positions]]
        return torch.nn.functional.cross_entropy(logits[positions], target)

    correct = loss_with(False)
    wrong = loss_with(True)
    grad = torch.autograd.grad(correct, vision, retain_graph=True)[0]
    grad_wrong = torch.autograd.grad(wrong, vision, retain_graph=True)[0]
    return {
        "symptom": "两种 mask 下 loss 都在下降，图像 token 的梯度都非零",
        "observable": {
            "loss_answer_only": float(correct.detach()), "loss_with_prompt": float(wrong.detach()),
            "supervised_positions_correct": n_answer,
            "supervised_positions_wrong": n_answer + n_prompt,
            "vision_grad_norm_correct": float(grad.norm()),
            "vision_grad_norm_wrong": float(grad_wrong.norm()),
        },
        "detection": "梯度非零只证明图像进了计算图，不能证明监督位置对；"
                     "打印每个位置的 label 是否被监督（本用例 5 对 9 个位置）",
    }


def case_codec():
    """帧率与码本大小不一致：一个报错、一个静默错位。"""
    configs = {"trained": {"frame_rate": 12.5, "codebook_size": 1024, "n_q": 16},
               "loaded": {"frame_rate": 25.0, "codebook_size": 2048, "n_q": 8}}
    duration_s = 2.0
    frames = {name: int(round(duration_s * cfg["frame_rate"])) for name, cfg in configs.items()}
    table = torch.nn.Embedding(configs["trained"]["codebook_size"], 4)
    ids = torch.randint(0, configs["loaded"]["codebook_size"], (frames["loaded"],))
    out_of_range = bool((ids >= configs["trained"]["codebook_size"]).any())
    strict = None
    try:
        table(torch.tensor([configs["trained"]["codebook_size"]]))
        strict = "no_error"
    except IndexError:
        strict = "IndexError"
    clamped = table.weight[torch.clamp(torch.tensor([configs["trained"]["codebook_size"]]), 0,
                                       configs["trained"]["codebook_size"] - 1)]
    return {
        "symptom": "训练照常进行，loss 下降但合成音频/时长系统性偏移",
        "observable": {
            "frames_at_trained_rate": frames["trained"], "frames_at_loaded_rate": frames["loaded"],
            "frame_ratio": frames["loaded"] / frames["trained"],
            "ids_out_of_trained_range": out_of_range,
            "lookup_behaviour_on_out_of_range": strict,
            "clamped_lookup_returns_row": int(torch.argmax(clamped).item()),
        },
        "detection": "先断言 max(id) < codebook_size 且 frames == round(duration*frame_rate)，"
                     "再比对两边的码本层数与帧率；只靠 loss 无法发现 2 倍时长错位",
    }


def case_flow():
    """flow matching 的 prediction_type 用错：同一模型、同一标签，目标含义改变。"""
    torch.manual_seed(1)
    batch = 256
    x0 = torch.randn(batch)
    x1 = torch.randn(batch) * 0.5 + 1.0
    t = torch.rand(batch)
    x_t = (1 - t) * x0 + t * x1
    velocity = x1 - x0
    # 三个目标：velocity（正确）、sample（x1）、noise（x1-x0 的另一种常见写法）
    model = torch.nn.Linear(2, 1)                               # 输入 (x_t, t)

    def fit(target, steps=400, lr=0.05):
        torch.manual_seed(2)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        inputs = torch.stack([x_t, t], dim=1)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs).squeeze(-1), target)
            loss.backward()
            opt.step()
        return float(loss)

    losses = {"velocity": fit(velocity), "sample": fit(x1),
              "sign_flipped_velocity": fit(-velocity)}
    # 采样：用训练出的“速度”走 1 步欧拉，与正确目标比较均值
    inputs = torch.stack([x_t, t], dim=1)
    with torch.no_grad():
        pred = model(inputs).squeeze(-1)
    euler = x_t + (1 - t) * pred
    return {
        "symptom": "换个 prediction_type 后 loss 依然能下降，采样结果只是略糊",
        "observable": {
            "loss_floor_velocity_target": losses["velocity"],
            "loss_floor_sample_target": losses["sample"],
            "loss_floor_sign_flipped": losses["sign_flipped_velocity"],
            "target_std_velocity": float(velocity.std()), "target_std_x1": float(x1.std()),
            "sampled_mean_after_one_euler": float(euler.mean()),
            "data_mean_x1": float(x1.mean()),
        },
        "detection": "先核对 prediction_type 与 target 的构造式（velocity=x1-x0、"
                     "sample=x1、noise=x1-x0 的符号约定），再用'完美模型应达到 0 loss'的"
                     "解析检查：同一标签换个名字不会报错，但采样均值会偏",
    }


def case_action(seed=3):
    """动作归一化统计量来自错误的来源（未归一化 / 用了别的数据集）。

    用固定的梯度预算训练同一个小 MLP：归一化正确时特征尺度一致、优化步长有效；
    不做归一化时各维量纲差 1–30 倍，同样步数下明显拟合不足。
    """
    torch.manual_seed(seed)
    scales = torch.tensor([1.0, 0.2, 0.05, 3.0, 0.5, 1.0, 0.1, 2.0])
    train = torch.randn(512, 8) * scales + 0.3
    other = train * 2.0 + 5.0                                   # 另一台机器/另一批数据
    stats_train = {"mean": train.mean(0), "std": train.std(0)}
    stats_other = {"mean": other.mean(0), "std": other.std(0)}
    weight = torch.randn(8)
    target = torch.sin(train @ weight) + 0.1 * train[:, 0] * train[:, 3]

    def fit(stats):
        torch.manual_seed(seed + 1)
        features = train if stats is None else (train - stats["mean"]) / stats["std"]
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(),
                                    torch.nn.Linear(16, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(300):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(features).squeeze(-1), target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = model(features).squeeze(-1)
        return {"mse": float(((prediction - target) ** 2).mean()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max())}

    correct = fit(stats_train)
    none = fit(None)
    wrong = fit(stats_other)
    return {
        "symptom": "三种归一化下 loss 都在下降，部署时动作幅度与精度异常",
        "observable": {
            "mse_correct_stats": correct["mse"], "mse_no_normalization": none["mse"],
            "mse_wrong_source_stats": wrong["mse"],
            "mse_ratio_none_over_correct": none["mse"] / max(correct["mse"], 1e-12),
            "prediction_range_correct": [correct["prediction_min"], correct["prediction_max"]],
            "prediction_range_none": [none["prediction_min"], none["prediction_max"]],
            "stored_mean_drift": float((stats_other["mean"] - stats_train["mean"]).abs().max()),
            "stored_std_drift": float((stats_other["std"] - stats_train["std"]).abs().max()),
        },
        "detection": "把 normalizer 的 mean/std 与训练集实际统计、以及动作维度量纲逐项比对；"
                     "再比较固定步数下的拟合误差与预测 min/max——只跑 loss 曲线会以为在做同一件事",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    result = {
        "section": "multimodal-failure-localization",
        "cases": {"mask": case_mask(), "codec": case_codec(), "flow": case_flow(),
                  "action": case_action()},
        "quality_metrics_by_modality": {
            "vision": ["留出问答精确匹配", "打乱/移除图像对照", "文字能力回归（预训练 CE）"],
            "speech": ["WER/CER", "说话人相似度", "首包时延与 RTF", "码本层级对拍"],
            "diffusion": ["留出目标误差", "固定噪声的样本与多样性", "采样 NFE 与质量"],
            "action": ["离线动作误差（每维）", "归一化统计对齐", "闭环成功率"],
        },
        "claim_scope": "四个注入用例都是小实现，用于展示检测点；不代表任何真实训练故障的复现",
    }
    (args.outdir / "failure_localization.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
