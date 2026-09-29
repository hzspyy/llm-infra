#!/usr/bin/env python3
"""样本身份与拆分单元：从固定版本上游源码里抽出真实字段，建统一 schema 并检查泄漏。

第 1 节直接解析快照源码，取出各模态真正存在的字段名与过滤常量（不是转述）。
第 2 节把它们归进 model / audit / partition / resume 四组，形成七类样本的统一 schema。
第 3 节用分组 split 与随机 split 对比泄漏量。
第 4 节按上游 DROID 的过滤规则重放，量化"改一条数据规则"的下游影响。

Usage:
    python labs/L7/sample_schema_survey.py > "$RUN_DIR/schema.txt"
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

import numpy as np

SRC = Path("results/local/7.8/20260915-data-engineering/source")


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# --------------------------------------------------------------------------
# 1. 从快照源码里抽字段
# --------------------------------------------------------------------------

def dataclass_fields(path: Path, name: str):
    """用 AST 取一个 dataclass 的字段名与类型，避免手抄。"""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [(item.target.id, ast.unparse(item.annotation))
                    for item in node.body if isinstance(item, ast.AnnAssign)]
    return []


def module_constants(path: Path, names):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in names:
                out[key] = ast.literal_eval(node.value)
    return out


def survey_sources():
    facts = {}

    doc = SRC / "datatrove/src/datatrove/data.py"
    facts["datatrove_document"] = dataclass_fields(doc, "Document")
    facts["datatrove_media"] = dataclass_fields(doc, "Media")
    print(f"datatrove Document（{doc.name}）字段: "
          f"{[n for n, _ in facts['datatrove_document']]}")
    print(f"  Media 字段: {[n for n, _ in facts['datatrove_media']]}")
    print("  正文、id、media、metadata 四项分别承担监督内容、身份、媒体引用和一切审计字段；"
          "\n  过滤器把判定结果写回 metadata，所以「为什么被删」跟着样本走。")

    minhash = SRC / "datatrove/src/datatrove/pipeline/dedup/minhash.py"
    text = minhash.read_text()
    cfg = re.search(r"class MinhashConfig.*?(?=\n@|\nclass )", text, re.S).group(0)
    defaults = dict(re.findall(r"^\s{4}(n_grams|num_buckets|hashes_per_bucket|seed): \w+ = (\d+)",
                               cfg, re.M))
    stages = re.findall(r"^class (Minhash\w+)\(PipelineStep\)", text, re.M)
    b, r = int(defaults["num_buckets"]), int(defaults["hashes_per_bucket"])
    facts["minhash"] = {"defaults": defaults, "stages": stages, "b": b, "r": r}
    print(f"\ndatatrove MinHash 默认配置: {defaults}，总签名数 {b * r}")
    print(f"  PipelineStep 子类: {stages}")
    print("  去重按四步走：算签名 → 按桶找候选对 → 并查集聚类 → 过滤；"
          "MinhashBuildIndex 是另建索引的可选步骤。")
    print(f"  banding LSH 的隐含阈值 (1/b)^(1/r) = (1/{b})^(1/{r}) = {(1 / b) ** (1 / r):.4f}")
    for j in (0.6, 0.72, 0.8, 0.9):
        p = 1 - (1 - j ** r) ** b
        print(f"    真实 Jaccard={j:.2f} 时成为候选对的概率 1-(1-J^{r})^{b} = {p:.4f}")
    print("  这是一条 S 形曲线而不是硬阈值：0.6 的相似度也有几率被判为重复，0.8 也有几率漏掉。")

    asr = SRC / "Qwen3-ASR/finetuning/qwen3_asr_sft.py"
    asr_text = asr.read_text()
    sr = re.search(r"def load_audio\(path: str, sr: int = (\d+)\)", asr_text).group(1)
    mask_lines = [i + 1 for i, line in enumerate(asr_text.splitlines())
                  if "-100" in line]
    facts["asr"] = {"sample_rate": int(sr), "mask_lines": mask_lines}
    print(f"\nQwen3-ASR SFT: load_audio 统一重采样到 {sr} Hz 单声道；"
          f"labels 用 -100 屏蔽的位置在第 {mask_lines} 行")
    print("  屏蔽的两处分别是 prompt 前缀与 padding——和纯文本 SFT 是同一套 loss mask 契约，"
          "\n  只是输入侧多了一步重采样。")

    droid = SRC / "openpi/examples/droid/compute_droid_nonidle_ranges.py"
    consts = module_constants(droid, {"min_idle_len", "min_non_idle_len",
                                      "filter_last_n_in_ranges"})
    droid_text = droid.read_text()
    idle_expr = re.search(r"np\.all\(np\.abs\(joint_velocities\[1:\] - joint_velocities\[:-1\]\) < ([\deE.\-]+), axis=1\)",
                          droid_text)
    key_expr = re.search(r'key = f"(.+?)"', droid_text)
    facts["droid"] = {**consts, "idle_threshold": float(idle_expr.group(1)),
                      "episode_key": key_expr.group(1)}
    print(f"\nopenpi DROID idle 过滤: {consts}，idle 判据 |Δjoint_velocity| < "
          f"{idle_expr.group(1)}（全关节同时成立）")
    print(f"  episode 身份 = {key_expr.group(1)}，这也是唯一合法的拆分单元。")

    cosmos = SRC / "cosmos-framework/docs/dataset_jsonl.md"
    block = re.search(r"```json\n(\{.*?\n\})\n```", cosmos.read_text(), re.S).group(1)
    sample = json.loads(block)
    window = sample["t2w_windows"][0]
    facts["cosmos"] = {"top": list(sample), "window": list(window),
                       "caption_json": list(window["caption_json"])}
    print(f"\nCosmos3 视频 SFT JSONL 顶层字段: {list(sample)}")
    print(f"  每个 t2w_window: {list(window)}")
    print(f"  caption_json 内部: {list(window['caption_json'])}")
    print("  start_frame/end_frame/temporal_interval 决定这一段监督覆盖哪些帧；"
          "\n  caption 由 VLM 生成（文档里指定 Qwen3-VL-8B-Instruct-FP8 经 vLLM 产出），"
          "\n  所以视频样本同时是一条「教师生成记录」，teacher 的身份必须一起存。")
    return facts


# --------------------------------------------------------------------------
# 2. 统一 schema
# --------------------------------------------------------------------------

SCHEMA = [
    ("文本文档", "input_ids / labels / 文档边界",
     "来源、许可、质量分、过滤规则版本", "document id",
     "shard、文件内偏移、全局样本号", "datatrove Document.metadata"),
    ("图文样本", "pixel_values / grid / input_ids / loss_mask",
     "caption 来源、处理器版本、min_pixels/max_pixels", "图像或相册 id",
     "annotation 文件 + 行号", "Qwen3-VL data_processor"),
    ("音频 utterance", "audio(16 kHz 单声道) / text_ids / labels",
     "转写来源、语言前缀、信噪比", "speaker、session",
     "JSONL 行号", "Qwen3-ASR collator"),
    ("视频 clip", "帧序列 / start_frame..end_frame / prompt",
     "caption 的生成模型与提示词版本、分辨率、fps", "video uuid、clip 序号",
     "JSONL 行号 + window 序号", "Cosmos3 dataset_jsonl"),
    ("机器人轨迹", "观测帧 / proprio / action / action_mask",
     "机器人配置、成功标志、normalizer 统计版本", "episode key",
     "episode 内 step 区间", "openpi DROID"),
    ("偏好对", "prompt / chosen / rejected 及各自 loss_mask",
     "标注来源、一致性、模板版本", "prompt 归一化后的身份",
     "JSONL 行号", "TRL/handbook 的 DPO 列"),
    ("教师生成", "student 输入 / teacher logits 或 hidden",
     "teacher revision、温度、topk、提取层", "原始 prompt 身份",
     "特征块 id + 块内偏移", "SpecForge / Cosmos caption"),
]


def print_schema():
    print("类别         | 送进前向的字段 | 审计字段 | 拆分单元 | 恢复字段 | 字段依据")
    for row in SCHEMA:
        print(f"{row[0]:12s} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]}")
    print("\n四组字段的用途完全不同：model 决定梯度，audit 决定能不能复现和能不能用，"
          "\npartition 决定 train/val 怎么切，resume 决定中断后从哪继续。"
          "\n把它们压成一个扁平的 input_ids 之后，后三组就再也找不回来了。")


# --------------------------------------------------------------------------
# 3. 拆分单元与泄漏
# --------------------------------------------------------------------------

def leakage_demo(n_speakers=40, per_speaker=25, seed=0):
    rng = random.Random(seed)
    samples = [{"id": f"{s:02d}-{k:02d}", "speaker": s} for s in range(n_speakers)
               for k in range(per_speaker)]
    rng.shuffle(samples)
    cut = int(len(samples) * 0.9)

    random_train, random_val = samples[:cut], samples[cut:]
    train_spk = {x["speaker"] for x in random_train}
    leaked = [x for x in random_val if x["speaker"] in train_spk]

    speakers = list(range(n_speakers))
    rng.shuffle(speakers)
    val_spk = set(speakers[:4])
    grouped_val = [x for x in samples if x["speaker"] in val_spk]
    grouped_train = [x for x in samples if x["speaker"] not in val_spk]
    grouped_leak = [x for x in grouped_val
                    if x["speaker"] in {y["speaker"] for y in grouped_train}]

    print(f"{n_speakers} 个说话人 × {per_speaker} 条 = {len(samples)} 条")
    print(f"  随机 9:1 切分: val {len(random_val)} 条，其中 {len(leaked)} 条"
          f"（{len(leaked) / len(random_val):.0%}）的说话人在 train 里出现过")
    print(f"  按说话人分组切分: val {len(grouped_val)} 条（{len(val_spk)} 个说话人），"
          f"泄漏 {len(grouped_leak)} 条")
    print("  文本 hash 去重查不出这类泄漏：两条录音的字面内容可以完全不同，"
          "\n  但同一个说话人的音色已经被模型见过。视频的同一场景、机器人同一 episode 同理。")


# --------------------------------------------------------------------------
# 4. 改一条规则的下游影响
# --------------------------------------------------------------------------

def droid_keep_ranges(is_idle, min_idle_len, min_non_idle_len, filter_last_n):
    """按 openpi 的规则求可训练区间：容忍不超过 min_idle_len 的连续 idle。"""
    ranges, start, idle_run = [], 0, 0
    for i, idle in enumerate(is_idle):
        if idle:
            idle_run += 1
            if idle_run > min_idle_len:
                end = i - idle_run + 1
                if end - start >= min_non_idle_len:
                    ranges.append((start, max(start, end - filter_last_n)))
                start, idle_run = i + 1, 0
        else:
            idle_run = 0
    if len(is_idle) - start >= min_non_idle_len:
        ranges.append((start, max(start, len(is_idle) - filter_last_n)))
    return [(a, b) for a, b in ranges if b > a]


def rule_change_demo(facts, seed=3):
    rng = np.random.default_rng(seed)
    length = 600
    moving = rng.random(length) > 0.35
    for a, b in ((80, 120), (300, 316), (500, 512)):
        moving[a:b] = False                     # 三段停顿，长度 40 / 16 / 12
    is_idle = ~moving
    base = facts["droid"]
    print(f"合成一条 {length} 帧的 episode，其中 idle 帧 {int(is_idle.sum())} 帧")
    print("min_idle_len | min_non_idle_len | filter_last_n | 区间数 | 保留帧 | 保留比例")
    for mi in (base["min_idle_len"], 3, 15):
        for last_n in (base["filter_last_n_in_ranges"], 0):
            ranges = droid_keep_ranges(is_idle, mi, base["min_non_idle_len"], last_n)
            kept = sum(b - a for a, b in ranges)
            print(f"{mi:12d} | {base['min_non_idle_len']:16d} | {last_n:13d} |"
                  f" {len(ranges):6d} | {kept:6d} | {kept / length:8.1%}")
    print(f"\n上游默认是 min_idle_len={base['min_idle_len']}、"
          f"min_non_idle_len={base['min_non_idle_len']}、"
          f"filter_last_n_in_ranges={base['filter_last_n_in_ranges']}。")
    print("把 min_idle_len 从 7 调到 3，同一条 episode 被切成更多更短的区间，保留帧数下降；"
          "\n这不是超参数微调，而是换了一个训练集。")

    print("\n改一条数据规则时必须同时检查的下游：")
    rows = [
        ("过滤/去重阈值", "训练集组成", "评测污染结论、数据混合比例", "已缓存的 tokenized shard 全部作废"),
        ("packing 最大长度", "每步有效 token 数", "loss 分母与学习率的有效标度", "batch 边界变了，游标不可直接复用"),
        ("图像 min/max_pixels", "视觉 token 数", "上下文长度与显存峰值", "预计算的视觉特征缓存作废"),
        ("音频重采样率", "编码器输入长度", "每秒的监督量", "预抽的特征与 codec token 作废"),
        ("action 归一化统计", "回归目标的量纲", "动作误差指标不可跨版本比较", "导出的 normalizer 必须同版本更新"),
        ("teacher revision", "监督信号本身", "学生质量结论只对该 teacher 成立", "特征 store 整体失效"),
    ]
    print("规则                | 直接改变      | 指标口径的影响 | 缓存与 checkpoint")
    for a, b, c, d in rows:
        print(f"{a:19s} | {b:13s} | {c} | {d}")


def main():
    global SRC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SRC)
    args = parser.parse_args()
    SRC = args.source

    section("1. 上游源码里真实存在的字段与常量")
    facts = survey_sources()

    section("2. 七类样本的统一 schema")
    print_schema()

    section("3. 拆分单元：随机切分与分组切分的泄漏量")
    leakage_demo()

    section("4. 改一条数据规则的下游影响")
    rule_change_demo(facts)


if __name__ == "__main__":
    main()
