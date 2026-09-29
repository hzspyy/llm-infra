#!/usr/bin/env python3
"""多基座训练的阶段矩阵、异构 batch 与缓存身份（7.10-A/D/E/F/G/H）。

`matrix`：只读仓库内快照，把六个基座实现（Cosmos 生成/Reasoner、MiniMind-V、Qwen3-Omni、
Qwen3-TTS、openpi DROID）的"可训练模块、优化器、目标字段、阶段预算"定位到 `file:line`。
矩阵的价值在于回答"这个 loss 到底 reach 了哪些参数"，而不是罗列模型名。

`hetero`：两件可算的事——按分辨率/长度分桶相对整批 padding 的浪费，以及缓存键的身份字段。
缓存部分用构造用例证明"参数已冻结"不是可复用的充分条件：换增强、换 crop、换时间采样、
换 processor 或换随机 posterior 都必须让键失效。

Usage:
    python labs/L7/multimodal_stage_matrix.py --section matrix \
      --source-root results/local/7.10/<run>/source --outdir "$RUN_DIR/mm-matrix"
    python labs/L7/multimodal_stage_matrix.py --section hetero --outdir "$RUN_DIR/mm-hetero"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# (基座, 文件, 事实标签, 正则)
PROBES = {
    "cosmos_generator": [
        ("vision_sft_edge.toml", "任务类型", r'task\s*=\s*"(\w+)"'),
        ("vision_sft_edge.toml", "优化器覆盖的参数组", r"keys_to_select"),
        ("vision_sft_edge.toml", "学习率", r"lr\s*=\s*([\d.e-]+)"),
        ("vision_sft_edge.toml", "激活重算", r'mode\s*=\s*"full"'),
        ("vision_sft_edge.toml", "打包预算", r"max_num_tokens_after_packing\s*=\s*(\d+)"),
        ("vision_sft_edge.toml", "EMA", r"\[model\.ema\]"),
    ],
    "cosmos_reasoner": [
        ("videophy2_sft_edge.toml", "任务类型", r'task\s*=\s*"(\w+)"'),
        ("videophy2_sft_edge.toml", "优化器段未列参数组限制", r"\[optimizer\]"),
        ("videophy2_sft_edge.toml", "学习率", r"lr\s*=\s*([\d.e-]+)"),
        ("videophy2_sft_edge.toml", "冻结主干", r"model_name\s*=\s*\"([^\"]+)\""),
        ("videophy2_sft_edge.toml", "每批样本数", r"max_samples_per_batch\s*=\s*(\d+)"),
    ],
    "minimind_v": [
        ("vlm/train_sft_vlm.py", "只有语言模型冻结开关", r"--freeze_llm"),
        ("vlm/train_sft_vlm.py", "冻结语言模型开关", r"freeze_llm"),
        ("vlm/model_vlm.py", "视觉塔冻结", r"param\.requires_grad = False"),
        ("vlm/model_vlm.py", "视觉 token 数", r"image_token_len"),
    ],
    "qwen3_omni": [
        ("omni/transformers.sh", "训练器入口", r"swift sft"),
        ("omni/transformers.sh", "冻结/可训练模块", r"freeze_|trainable"),
        ("omni/zero3.sh", "并行方式", r"deepspeed"),
    ],
    "qwen3_tts": [
        ("tts/sft_12hz.py", "优化器", r"optimizer = AdamW"),
        ("tts/sft_12hz.py", "学习率", r"--lr\", type=float, default=([\d.e-]+)"),
        ("tts/sft_12hz.py", "第一码本标签", r"codec_0_labels"),
        ("tts/sft_12hz.py", "码本掩码", r"codec_mask"),
        ("tts/sft_12hz.py", "说话人嵌入", r"model\.speaker_encoder"),
    ],
    "openpi_droid": [
        ("openpi/optimizer.py", "优化器", r"tx = optax\.adamw"),
        ("openpi/optimizer.py", "峰值学习率", r"peak_lr: float = ([\d.e-]+)"),
        ("openpi/config.py", "冻结过滤器", r"freeze_filter"),
        ("openpi/config.py", "动作视野", r"action_horizon=(\d+)"),
        ("openpi/config.py", "EMA 衰减", r"ema_decay: float \| None = ([\d.]+)"),
        ("openpi/config.py", "批大小/步数", r"num_train_steps: int = ([\d_]+)"),
    ],
}

CACHE_KEY_FIELDS = ("model_revision", "processor_hash", "augmentation_seed", "crop",
                    "time_sampling", "stochastic_posterior", "trainable_state")


def locate(text: str, pattern: str):
    for index, line in enumerate(text.splitlines(), 1):
        match = re.search(pattern, line)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            return index, line.strip(), value
    return None, None, None


def matrix_section(source_root: Path) -> dict:
    result = {"section": "multimodal-stage-matrix", "bases": {}, "missing": []}
    for base, probes in PROBES.items():
        rows = {}
        for relative, label, pattern in probes:
            path = source_root / relative
            if not path.exists():
                result["missing"].append(f"{base}:{relative}")
                rows[label] = None
                continue
            line_no, code, value = locate(path.read_text(encoding="utf-8", errors="replace"),
                                          pattern)
            rows[label] = {"file": relative, "line": line_no, "value": value, "code": code}
            if line_no is None:
                result["missing"].append(f"{base}:{label}")
        result["bases"][base] = rows
    result["claim_scope"] = ("源码定位：行号来自仓库内快照；矩阵描述的是配置与代码声明的"
                            "可训练范围，不是某次运行的实际梯度到达范围")
    return result


def hetero_section(seed: int = 7) -> dict:
    import random
    rng = random.Random(seed)
    # 异构 batch：图像分辨率、视频帧数、音频时长、动作视野混在同一条流里
    samples = []
    for index in range(64):
        samples.append({"id": index,
                        "pixels": rng.choice([224 * 224, 448 * 448, 672 * 672]),
                        "frames": rng.choice([1, 8, 16, 32]),
                        "audio_frames": rng.choice([50, 100, 200]),
                        "action_horizon": rng.choice([10, 25, 50])})
    def token_cost(sample):
        return (sample["pixels"] // (32 * 32) * sample["frames"] + sample["audio_frames"] // 2
                + sample["action_horizon"])
    naive = max(token_cost(s) for s in samples) * len(samples)
    packed = sum(token_cost(s) for s in samples)
    buckets = {}
    for sample in samples:
        key = (sample["pixels"], sample["frames"])
        buckets.setdefault(key, []).append(token_cost(sample))
    bucketed = sum(max(costs) * len(costs) for costs in buckets.values())
    branch_usage = {"vision_only": sum(1 for s in samples if s["pixels"] > 0),
                    "video_only": sum(1 for s in samples if s["frames"] > 1),
                    "audio_only": sum(1 for s in samples if s["audio_frames"] < 100),
                    "action_only": sum(1 for s in samples if s["action_horizon"] > 10)}

    # 缓存身份：任一字段变化都要换键
    baseline = {"model_revision": "rev-a", "processor_hash": "p1", "augmentation_seed": 0,
                "crop": "center", "time_sampling": "uniform", "stochastic_posterior": False,
                "trainable_state": "frozen"}
    def key(payload):
        text = json.dumps(payload, sort_keys=True)
        return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()

    base_key = key(baseline)
    mutations = {}
    for field in CACHE_KEY_FIELDS:
        changed = dict(baseline)
        changed[field] = ("free" if field == "trainable_state" else
                          ("train" if field == "crop" else
                           (not baseline[field] if isinstance(baseline[field], bool)
                            else str(baseline[field]) + "-x")))
        mutations[field] = {"changed_to": changed[field], "same_key": key(changed) == base_key}
    # 冻结但换了增强：键必须变，说明"参数没变"不等于"可以复用"
    frozen_but_aug = dict(baseline)
    frozen_but_aug["augmentation_seed"] = 1
    return {
        "section": "multimodal-hetero-cache", "samples": len(samples),
        "padding_waste": {
            "naive_max_padding_tokens": naive, "content_tokens": packed,
            "per_key_padding_tokens": bucketed,
            "naive_overhead_ratio": naive / packed - 1.0,
            "bucketed_overhead_ratio": bucketed / packed - 1.0,
            "buckets": len(buckets),
            "reduction_vs_naive": 1.0 - (bucketed - packed) / max(naive - packed, 1),
        },
        "unused_branch_tracking": branch_usage,
        "cache_identity": {
            "baseline_key": base_key,
            "mutations_change_key": {k: not v["same_key"] for k, v in mutations.items()},
            "frozen_but_new_augmentation_changes_key":
                key(frozen_but_aug) != base_key,
            "note": "冻结参数不是缓存可复用的充分条件：增强/crop/时间采样/processor/posterior "
                    "任一变化都要换键；trainable_state 也进键，因为同一权重在冻结与解冻下的输出语义不同",
        },
        "claim_scope": "分桶浪费是按样本形状的解析计算，不含真实 dataloader；"
                       "缓存键检查是构造用例，不是缓存命中率测量",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", required=True, choices=["matrix", "hetero"])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    if args.section == "matrix":
        if args.source_root is None:
            raise SystemExit("matrix 需要 --source-root")
        result = matrix_section(args.source_root)
        name = "stage_matrix.json"
    else:
        result = hetero_section()
        name = "hetero_cache.json"
    (args.outdir / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
