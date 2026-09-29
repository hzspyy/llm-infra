#!/usr/bin/env python3
"""配置字段 → 数学与状态：SmolLM3 的 Nanotron 配方与发布 config 的逐项映射。

只读解析固定版本的 YAML 与 config.json，算出参数量、NoPE/GQA 布局、全局 batch、
token 预算、学习率分段和每参数的训练状态字节，并与实际做过一次更新的
SmolLM2-360M 对照。不加载权重、不执行任何配置里的代码。

Usage:
    python labs/L7/recipe_state_ledger.py > "$RUN_DIR/recipe-ledger.txt"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

RECIPES = Path("results/local/7.3/20260914-review-fixes/source/smollm/"
               "text/pretraining/smollm3")
CONFIGS = Path("results/local/7.0b/20260915-one-step/source/hf")
STAGES = ["stage1_8T.yaml", "stage2_8T_9T.yaml", "stage3_9T_11T.yaml",
          "long_context_4k_to_32k.yaml", "long_context_32k_to_64.yaml"]


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def param_breakdown(cfg: dict) -> dict:
    """按 config 字段算参数量；head_dim = hidden/num_attention_heads。"""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    head_dim = h // heads
    kv_dim = kv_heads * head_dim
    per_layer = {
        "q_proj": h * h, "k_proj": h * kv_dim, "v_proj": h * kv_dim, "o_proj": h * h,
        "gate_proj": h * inter, "up_proj": h * inter, "down_proj": inter * h,
        "input_layernorm": h, "post_attention_layernorm": h,
    }
    embed = cfg["vocab_size"] * h
    out = {"embed_tokens": embed, **{k: v * layers for k, v in per_layer.items()},
           "norm": h}
    if not cfg.get("tie_word_embeddings", False):
        out["lm_head"] = embed
    out["_per_layer"] = sum(per_layer.values())
    out["_head_dim"] = head_dim
    out["_kv_dim"] = kv_dim
    return out


def show_model(name: str, cfg: dict):
    parts = param_breakdown(cfg)
    total = sum(v for k, v in parts.items() if not k.startswith("_"))
    print(f"\n[{name}] hidden={cfg['hidden_size']} inter={cfg['intermediate_size']} "
          f"layers={cfg['num_hidden_layers']} heads={cfg['num_attention_heads']} "
          f"kv_heads={cfg['num_key_value_heads']} vocab={cfg['vocab_size']} "
          f"tie={cfg.get('tie_word_embeddings')}")
    print(f"  head_dim={parts['_head_dim']}，KV 投影输出维 {parts['_kv_dim']}，"
          f"GQA 每 {cfg['num_attention_heads'] // cfg['num_key_value_heads']} 个 q head 共用一组 KV")
    print("  模块                        参数量        占比")
    for key, value in parts.items():
        if key.startswith("_"):
            continue
        print(f"    {key:24s} {value:12,d}   {value / total:6.2%}")
    print(f"    {'合计':24s} {total:12,d}")
    if cfg.get("tie_word_embeddings"):
        print("    lm_head 与 embed_tokens 绑定：只存一份权重，梯度来自两处使用")
    interval = cfg.get("no_rope_layer_interval") or cfg.get("no_rope_layer")
    if interval:
        nope = [i for i in range(1, cfg["num_hidden_layers"] + 1) if i % interval == 0]
        print(f"  NoPE：interval={interval}，跳过 RoPE 的层（1 起数）{nope}，"
              f"共 {len(nope)}/{cfg['num_hidden_layers']} 层")
    return total


def state_bytes(total_params: int, recipe: dict) -> None:
    dtype = recipe["model"]["dtype"]
    fp32_grad = recipe["optimizer"]["accumulate_grad_in_fp32"]
    zero = recipe["optimizer"]["zero_stage"]
    tp = recipe["parallelism"]["tp"]
    dp = recipe["parallelism"]["dp"]
    width = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    rows = [(f"模型参数（{dtype}）", width, "前向/反向的计算权重")]
    if fp32_grad:
        rows += [("FP32 master 参数", 4, "gradient_accumulator.py:113 从半精度拷贝而来"),
                 ("FP32 梯度缓冲", 4, "gradient_accumulator.py:222 累加，:226 随即释放半精度 .grad")]
    else:
        rows += [(f"梯度（{dtype}）", width, "直接用参数精度累加")]
    rows += [("AdamW exp_avg", 4, "建立在 FP32 参数上" if fp32_grad else "建立在模型参数上"),
             ("AdamW exp_avg_sq", 4, "同上")]
    per = sum(r[1] for r in rows)
    print(f"  zero_stage={zero}（optimizer 状态在 {dp} 个 DP rank 上完整复制），tp={tp}")
    print("  项目                    字节/参数   说明")
    for label, b, note in rows:
        print(f"    {label:20s} {b:6d}     {note}")
    print(f"    {'合计':20s} {per:6d}")
    gib = total_params * per / 1024 ** 3
    print(f"  全模型常驻训练状态 {total_params:,} × {per} B = {gib:.2f} GiB，"
          f"按 tp={tp} 平分后每 rank {gib / tp:.2f} GiB")
    print("  这份账不含激活、通信缓冲与 workspace；完整预算见 7.2/7.9/7.11。")


def schedule_lr(sched: dict, step: int) -> float:
    base = sched["learning_rate"]
    warmup = sched["lr_warmup_steps"]
    start = sched["lr_decay_starting_step"]
    span = sched["lr_decay_steps"]
    if step <= warmup:
        return base * step / warmup
    if step < start:
        return base
    frac = min(1.0, (step - start) / span)
    if sched["lr_decay_style"] == "linear":
        return base + (sched["min_decay_lr"] - base) * frac
    import math
    return sched["min_decay_lr"] + (base - sched["min_decay_lr"]) * 0.5 * (1 + math.cos(math.pi * frac))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=RECIPES)
    parser.add_argument("--configs", type=Path, default=CONFIGS)
    args = parser.parse_args()

    recipes = {name: yaml.safe_load((args.recipes / name).read_text()) for name in STAGES}
    smollm3 = json.loads((args.configs / "SmolLM3-3B-Base-config.json").read_text())
    smollm2 = json.loads((args.configs / "SmolLM2-360M-config.json").read_text())

    section("1. 参数量、GQA 与 NoPE")
    total3 = show_model("SmolLM3-3B-Base（发布 config）", smollm3)
    total2 = show_model("SmolLM2-360M（crater 上实测一次更新的模型）", smollm2)
    print(f"\n两者结构不同：层数 {smollm2['num_hidden_layers']}→{smollm3['num_hidden_layers']}、"
          f"词表 {smollm2['vocab_size']}→{smollm3['vocab_size']}、"
          f"SmolLM3 有 NoPE 层而 SmolLM2 没有；参数量相差 {total3 / total2:.1f} 倍。"
          "\n本章的实测读数只属于 SmolLM2-360M，SmolLM3 的部分全部来自公开配方。")

    section("2. 全局 batch 与 token 预算")
    print("阶段                       | mbs | accum |  dp | tp | cp |  seq  | 序列/步 |"
          " token/步   | train_steps")
    for name, r in recipes.items():
        t, p = r["tokens"], r["parallelism"]
        seqs = t["micro_batch_size"] * t["batch_accumulation_per_replica"] * p["dp"]
        toks = seqs * t["sequence_length"]
        print(f"{name:26s} | {t['micro_batch_size']:3d} | {t['batch_accumulation_per_replica']:5d} |"
              f" {p['dp']:3d} | {p['tp']:2d} | {p['context_parallel_size']:2d} |"
              f" {t['sequence_length']:5d} | {seqs:7d} | {toks:9,d} | {t['train_steps']:,}")
    base = recipes["stage1_8T.yaml"]["tokens"]
    unit = (base["micro_batch_size"] * base["batch_accumulation_per_replica"]
            * recipes["stage1_8T.yaml"]["parallelism"]["dp"] * base["sequence_length"])
    print(f"\n五份配方的 token/步全部等于 {unit:,}：序列长度从 4096 涨到 65536 的同时，"
          "mbs×accum×dp 同比例减小。\n"
          "全局 batch 的不变量是 token 数而不是序列条数——分母按 token 计时这一点才成立。")
    print(f"预训练三阶段共享同一条时间轴：train_steps={base['train_steps']:,} × {unit:,} "
          f"token/步 = {base['train_steps'] * unit / 1e12:.1f}T token。")
    for name in STAGES[:3]:
        starts = re.findall(r"start_training_step:\s*(\d+)",
                            (args.recipes / name).read_text())
        first = max(int(s) for s in starts)
        print(f"  {name:26s} 最后一个 data_stage 起点 step {first:,} "
              f"≈ {first * unit / 1e12:.1f}T token；"
              f"resume 自 {recipes[name]['checkpoints']['resume_checkpoint_path'].split('/')[-1]}")
    print("阶段切换的是数据混合与续接的 checkpoint，不是重新计数的新一轮训练。")

    section("3. 学习率分段")
    for name in ("stage1_8T.yaml", "long_context_4k_to_32k.yaml", "long_context_32k_to_64.yaml"):
        s = recipes[name]["optimizer"]["learning_rate_scheduler"]
        probes = [1, s["lr_warmup_steps"], s["lr_decay_starting_step"],
                  s["lr_decay_starting_step"] + s["lr_decay_steps"] // 2,
                  s["lr_decay_starting_step"] + s["lr_decay_steps"]]
        print(f"\n{name}: peak={s['learning_rate']} warmup={s['lr_warmup_steps']}({s['lr_warmup_style']}) "
              f"decay={s['lr_decay_style']} 自 step {s['lr_decay_starting_step']:,} 起 "
              f"{s['lr_decay_steps']:,} 步到 {s['min_decay_lr']}")
        print("  step:      " + "  ".join(f"{p:>9,}" for p in probes))
        print("  lr:        " + "  ".join(f"{schedule_lr(s, p):>9.3e}" for p in probes))
    print("\nLR 的自变量是 optimizer step，不是消耗的 token 数；改 mbs×accum×dp 会同时移动"
          "「每步多少 token」和「同一 step 的 LR」两件事。")

    section("4. 参数组：weight_decay 的排除规则")
    opt = recipes["stage1_8T.yaml"]["optimizer"]
    patterns = opt["weight_decay_exclude_named_params"]
    hf_names = [k for k in param_breakdown(smollm3) if not k.startswith("_")]
    hit = [n for n in hf_names for p in patterns if re.match(p, n)]
    print(f"  weight_decay={opt['weight_decay']}，排除规则 {patterns}")
    print(f"  规则按 Nanotron 的参数名书写；用它匹配发布 config 的模块名 {hf_names} 命中 {len(hit)} 个。")
    print("  helpers.py:208-230 逐个参数建组，:210-219 对绑定参数额外用 tied name 再匹配一次，"
          "所以 tie_word_embeddings=true 时被排除的是同一块权重的两个名字。")
    print("  norm 与 bias 是否衰减由这份规则决定，本配方只排除了 token embedding。")

    section("5. 训练状态字节（按 stage1 的实际开关）")
    state_bytes(total3, recipes["stage1_8T.yaml"])

    section("6. 训练期配置与发布 config 的差异")
    train_model = recipes["stage1_8T.yaml"]["model"]["model_config"]
    last = recipes["long_context_32k_to_64.yaml"]["model"]["model_config"]
    rows = [("rope_theta", train_model["rope_theta"], last["rope_theta"], smollm3["rope_theta"]),
            ("max_position_embeddings", train_model["max_position_embeddings"],
             last["max_position_embeddings"], smollm3["max_position_embeddings"]),
            ("no_rope_layer(_interval)", train_model["no_rope_layer"], last["no_rope_layer"],
             smollm3["no_rope_layer_interval"]),
            ("vocab_size", train_model["vocab_size"], last["vocab_size"], smollm3["vocab_size"]),
            ("tie_word_embeddings", train_model["tie_word_embeddings"],
             last["tie_word_embeddings"], smollm3["tie_word_embeddings"])]
    print("字段                     | stage1 训练期 | 最后一个长上下文阶段 | 发布 config")
    for key, a, b, c in rows:
        print(f"{key:24s} | {str(a):13s} | {str(b):20s} | {c}")
    print("\nrope_theta 与 max_position_embeddings 由最后一个训练阶段决定，发布 config 记录的是"
          "该阶段的值；\n用发布 config 反推训练全过程会把 50000 这一段完全丢掉。")
    lc = recipes["long_context_32k_to_64.yaml"]["checkpoints"]
    print(f"\n32k→64k 阶段的 load_optimizer={lc['load_optimizer']}、"
          f"load_lr_scheduler={lc['load_lr_scheduler']}：它从上一阶段的权重出发，"
          "但新建 optimizer 与调度器。\n这是 warm start 而不是严格 resume，"
          "对应 start_modes_and_resume.py 的 D 段。")
    print("训练期还声明了 _use_doc_masking、_use_qkv_packed、z_loss_enabled 等只在训练存在的开关，"
          "发布 config 里没有对应字段。")

    section("7. 只有静态检查能回答的与不能回答的")
    print("  可以：结构与参数量、分母与 batch 的数量关系、LR 曲线、状态字节、阶段续接关系。")
    print("  不能：真实吞吐、收敛质量、数据实际混合比例（dataset_folder 指向内部 S3）、"
          "以及发布权重是否带有可继续原训练的 optimizer/RNG。")
    print("  配方完整性检查与四框架接口对照在 7.3；本节只做配置到状态的映射。")


if __name__ == "__main__":
    main()
