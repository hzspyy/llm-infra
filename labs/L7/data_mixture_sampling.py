#!/usr/bin/env python3
"""混合权重的单位：同一个"50%"在文档/样本/token/时长四种口径下的实际占比。

用固定 RNG 生成真实采样序列（不是期望公式），比较声明权重与实测 token 占比；
再检查有放回/重复 epoch、阶段性课程混合与 DistributedSampler 的尾部补齐。
最后解析两份真实配方里的权重字段，指出它们各自的单位。

Usage:
    python labs/L7/data_mixture_sampling.py > "$RUN_DIR/mixture.txt"
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import yaml
from torch.utils.data.distributed import DistributedSampler

RECIPE = Path("results/local/7.3/20260914-review-fixes/source/smollm/"
              "text/pretraining/smollm3/stage1_8T.yaml")
VL_REGISTRY = Path("results/local/7.8/20260915-data-engineering/source/"
                   "Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py")

# 三个小数据源：文档数相同，平均长度差一个数量级。
SOURCES = {
    "web":  {"docs": 400, "mean_tokens": 120, "mean_seconds": 0.0},
    "book": {"docs": 400, "mean_tokens": 1800, "mean_seconds": 0.0},
    "talk": {"docs": 400, "mean_tokens": 300, "mean_seconds": 12.0},
}
DRAWS = 4000


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_corpus(seed: int = 0):
    """给每篇文档一个固定长度，长度分布用对数正态近似真实语料的长尾。"""
    rng = random.Random(seed)
    corpus = {}
    for name, spec in SOURCES.items():
        lengths = [max(8, int(rng.lognormvariate(0, 0.6) * spec["mean_tokens"]))
                   for _ in range(spec["docs"])]
        if spec["mean_seconds"]:
            # 语速在 1.5–8 token/秒 之间变化，秒数与 token 数不成固定比例。
            seconds = [length / rng.uniform(1.5, 8.0) for length in lengths]
        else:
            seconds = [0.0] * len(lengths)
        corpus[name] = {"lengths": lengths, "seconds": seconds}
    return corpus


def sample_sequence(corpus, probs, draws, seed):
    """按 probs 有放回抽源、再在源内均匀抽一篇，返回实际抽样序列。"""
    rng = random.Random(seed)
    names = list(probs)
    weights = [probs[n] for n in names]
    picks = []
    for _ in range(draws):
        name = rng.choices(names, weights=weights, k=1)[0]
        idx = rng.randrange(len(corpus[name]["lengths"]))
        picks.append((name, idx))
    return picks


def tally(corpus, picks):
    docs, tokens, seconds = Counter(), Counter(), Counter()
    for name, idx in picks:
        docs[name] += 1
        tokens[name] += corpus[name]["lengths"][idx]
        seconds[name] += corpus[name]["seconds"][idx]
    return docs, tokens, seconds


def share(counter):
    total = sum(counter.values())
    return {k: (v / total if total else 0.0) for k, v in counter.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--vl-registry", type=Path, default=VL_REGISTRY)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()

    corpus = build_corpus()
    mean_len = {n: sum(c["lengths"]) / len(c["lengths"]) for n, c in corpus.items()}

    section("1. 三个数据源的真实长度分布")
    print("源    | 文档数 | 平均 token | 中位 token | 最长 token | 总 token")
    for name, c in corpus.items():
        ls = sorted(c["lengths"])
        print(f"{name:5s} | {len(ls):6d} | {sum(ls) / len(ls):10.1f} | {ls[len(ls) // 2]:10d} |"
              f" {ls[-1]:10d} | {sum(ls):8d}")

    section("2. 同一组权重，四种口径下的实测占比")
    declared = {"web": 0.5, "book": 0.25, "talk": 0.25}
    picks = sample_sequence(corpus, declared, args.draws, seed=1)
    docs, tokens, seconds = tally(corpus, picks)
    print(f"声明权重（按抽样概率）: {declared}，实际抽样 {args.draws} 次")
    print("源    | 声明 | 实测文档占比 | 实测 token 占比 | 期望 token 占比(p·L 公式)")
    exp_token_total = sum(declared[n] * mean_len[n] for n in declared)
    for name in declared:
        print(f"{name:5s} | {declared[name]:.2f} | {share(docs)[name]:12.4f} |"
              f" {share(tokens)[name]:15.4f} | {declared[name] * mean_len[name] / exp_token_total:16.4f}")
    print("\n声明的 50% 落在「被抽中的文档数」上。book 的平均长度是 web 的 "
          f"{mean_len['book'] / mean_len['web']:.1f} 倍，于是 25% 的文档权重换来 "
          f"{share(tokens)['book']:.1%} 的 token 权重。")

    section("3. 要让 token 占比等于 50%，权重要反过来按长度折算")
    target = {"web": 0.5, "book": 0.25, "talk": 0.25}
    adjusted = {n: target[n] / mean_len[n] for n in target}
    norm = sum(adjusted.values())
    adjusted = {n: v / norm for n, v in adjusted.items()}
    picks2 = sample_sequence(corpus, adjusted, args.draws, seed=1)
    _, tokens2, _ = tally(corpus, picks2)
    print(f"折算后的抽样概率: { {k: round(v, 5) for k, v in adjusted.items()} }")
    print("源    | 目标 token 占比 | 实测 token 占比 | 实测文档占比")
    for name in target:
        print(f"{name:5s} | {target[name]:15.4f} | {share(tokens2)[name]:15.4f} |"
              f" {share(tally(corpus, picks2)[0])[name]:12.4f}")
    print("同一份目标，两套权重；写配方时必须说清 50% 是文档、样本、token 还是秒。")

    section("4. 时长口径：同一个音频源内部，秒和 token 也不成比例")
    talk = corpus["talk"]
    rates = [length / sec for length, sec in zip(talk["lengths"], talk["seconds"])]
    order = sorted(range(len(rates)), key=lambda i: rates[i])
    slow, fast = order[:100], order[-100:]
    for label, group in (("最慢 100 条", slow), ("最快 100 条", fast)):
        tok = sum(talk["lengths"][i] for i in group)
        sec = sum(talk["seconds"][i] for i in group)
        print(f"  {label}: {tok:6d} token / {sec:8.1f} 秒 = {tok / sec:.2f} token/秒，"
              f"占该源 token 的 {tok / sum(talk['lengths']):.1%}、占秒数的 {sec / sum(talk['seconds']):.1%}")
    print(f"  talk 源整体 {sum(talk['lengths'])} token / {sum(talk['seconds']):.0f} 秒")
    print("  按转写 token 配比和按音频小时配比会挑出不同的子集：慢速语音贡献的秒数远超它的 token 数。")
    print("  ASR 的监督量是转写 token，编码器成本按采样点/mel 帧算，两者必须分别记账。")

    section("5. 有放回抽样与重复 epoch")
    counts = Counter(idx for name, idx in picks if name == "web")
    print(f"web 源 400 篇，在 {docs['web']} 次抽取中被覆盖 {len(counts)} 篇，"
          f"未被抽到 {400 - len(counts)} 篇，最多的一篇被抽 {max(counts.values())} 次")
    rng = random.Random(7)
    epoch_order = []
    for epoch in range(2):
        order = list(range(10))
        rng.shuffle(order)
        epoch_order.append(order)
    print(f"无放回、重复 2 个 epoch（10 篇小集合）: {epoch_order[0]} / {epoch_order[1]}")
    print("有放回采样在一个 epoch 里既漏掉一部分文档也重复另一部分；"
          "只有无放回+重排才保证每轮覆盖一次。混合权重与覆盖率是两件事。")

    section("6. 阶段性课程混合：累计占比的轨迹")
    stages = [({"web": 0.8, "book": 0.1, "talk": 0.1}, 2000),
              ({"web": 0.3, "book": 0.5, "talk": 0.2}, 2000)]
    cum, seed = [], 11
    all_picks = []
    for weights, n in stages:
        all_picks += sample_sequence(corpus, weights, n, seed)
        seed += 1
        _, t, _ = tally(corpus, all_picks)
        cum.append((weights, len(all_picks), share(t)))
    for weights, n, s in cum:
        print(f"  截至第 {n} 次抽样（当前阶段权重 {weights}）: "
              f"累计 token 占比 { {k: round(v, 4) for k, v in s.items()} }")
    print("阶段切换后累计占比不会立刻跟上当前阶段的权重；"
          "报告数据配比时必须说明是瞬时权重还是累计消耗。")

    section("7. DistributedSampler 的尾部")
    for drop_last in (False, True):
        seen = {}
        for rank in range(3):
            sampler = DistributedSampler(range(7), num_replicas=3, rank=rank,
                                         shuffle=True, seed=0, drop_last=drop_last)
            sampler.set_epoch(0)
            seen[rank] = list(sampler)
        flat = [i for v in seen.values() for i in v]
        dup = [i for i, c in Counter(flat).items() if c > 1]
        missing = sorted(set(range(7)) - set(flat))
        print(f"  drop_last={drop_last}: 每 rank {[seen[r] for r in range(3)]}，"
              f"总条数 {len(flat)}，重复 {dup}，丢失 {missing}")
    print("7 个样本分给 3 个 rank：drop_last=False 靠重复补齐，drop_last=True 直接丢。"
          "两种都不是「每个样本恰好一次」，恢复时的样本清单要按实际取到的 ID 记录。")

    section("8. 真实配方里的权重字段与它们的单位")
    if args.recipe.exists():
        cfg = yaml.safe_load(args.recipe.read_text())
        data = cfg["data_stages"][0]["data"]["dataset"]
        w = data["dataset_weights"]
        folders = data["dataset_folder"]
        tokens_cfg = cfg["tokens"]
        gbs = (tokens_cfg["micro_batch_size"] * tokens_cfg["batch_accumulation_per_replica"]
               * cfg["parallelism"]["dp"])
        seq = tokens_cfg["sequence_length"]
        total = sum(w)
        print(f"  SmolLM3 stage1: {len(w)} 个数据源，权重和 {total:.6f}")
        order = sorted(range(len(w)), key=lambda i: -w[i])[:6]
        print("  前 6 个源与按 train_steps 折算的 token 预算：")
        for i in order:
            share_i = w[i] / total
            toks = tokens_cfg["train_steps"] * gbs * share_i * seq
            print(f"    {folders[i].split('/')[-2]:22s} w={w[i]:<7.4f} 归一化 {share_i:.4f}"
                  f"  → {toks / 1e12:6.2f}T token")
        print(f"  单位来自加载器：tokenized_bytes.py:576-579 先把权重归一化，:589-590 用"
              f"\n  int((train_steps - it) * global_batch_size * w_i) 算样本数、再乘 sequence_length "
              f"({seq}) 得 token 数。")
        print("  这批数据已经预分词成等长序列，样本数与 token 数成固定比例，权重才等于 token 占比；"
              "\n  同一份权重作用在变长文档上就变成文档占比。")
    else:
        print(f"  未找到 {args.recipe}，跳过。")

    if args.vl_registry.exists():
        text = args.vl_registry.read_text()
        names = ["cambrian_737k%50", "mp_doc", "clevr_mc%25"]
        parsed = {n: (float(m.group(1)) / 100.0 if (m := re.search(r"%(\d+)$", n)) else 1.0)
                  for n in names}
        print(f"\n  Qwen3-VL 的数据集名后缀：{parsed}")
        print(f"  解析位置 qwenvl/data/__init__.py:38-44；消费位置 data_processor.py:277-280 是"
              "\n  random.sample(annotations, int(len(annotations) * rate))——对 annotation 条目取样。")
        print("  一条 annotation 可能带 1 张图也可能带一段视频，因此 50% 的条目"
              "既不是 50% 的图像数，也不是 50% 的视觉 token 数。")
        assert "parse_sampling_rate" in text


if __name__ == "__main__":
    main()
