#!/usr/bin/env python3
"""0.5-A: 四分类玩具分布上，把采样的每一步算成看得见的向量。

词表只有 4 个 token（A/B/C/D），logits 含负值，全部用 FP64 计算，
每一步都打印完整向量：temperature、top-k、top-p、min-p、三种惩罚。
在这个尺度上手算可以核对每个数字，顺序交换与并列的反例也能直接看出来。

三件事是本脚本要证明的：

  1. 过滤顺序不可交换。温度在前和在后，候选集与最终概率都不同。
  2. 并列（ties）会让「保留全部」和「截断到 k 个」给出不同的分布。
  3. greedy 与随机采样的「正确」不是一回事：前者要求 id 精确相等，
     后者只能要求频率与理论分布在统计容差内一致。

顺序与公式对照 vLLM 0.29.0 的 `v1/sample/sampler.py`，行号见正文。

用法：
    python labs/L0/sampling_math.py --out-dir results/local/0.5/<run>
纯 CPU，无需模型。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

SEP = "-" * 78
NAMES = ["A", "B", "C", "D"]
LOGITS = [2.0, 0.5, -1.0, -3.0]      # 含负值，便于看清 repetition penalty 的不对称
PROMPT_TOKENS = [1]                  # B 出现在 prompt 里
OUTPUT_TOKENS = [0, 0, 2]            # A 出现两次，C 一次
SEEDS = (0, 1, 2)
N_DRAWS = 200_000


def vec(x: torch.Tensor, width: int = 12, prec: int = 6) -> str:
    out = []
    for v in x.tolist():
        out.append("-inf".rjust(width) if v == -math.inf else f"{v:{width}.{prec}f}")
    return " ".join(out)


def head(title: str) -> None:
    print(f"\n{title}\n{SEP}")


def softmax(z: torch.Tensor) -> torch.Tensor:
    return torch.softmax(z, dim=-1)


# ------------------------------------------------------------------ 过滤算子

def top_k_filter(z: torch.Tensor, k: int, keep_ties: bool = True) -> torch.Tensor:
    """keep_ties=True: 保留所有等于第 k 大值的候选；False: 严格留 k 个（按索引）。"""
    out = z.clone()
    if k >= z.numel():
        return out
    kth = torch.topk(z, k).values[-1]
    if keep_ties:
        out[z < kth] = -math.inf
        return out
    idx = torch.topk(z, k).indices
    mask = torch.ones_like(z, dtype=torch.bool)
    mask[idx] = False
    out[mask] = -math.inf
    return out


def top_p_filter(z: torch.Tensor, p: float) -> torch.Tensor:
    """保留「累计概率首次达到 p」为止的候选，含跨过阈值的那一个。"""
    probs = softmax(z)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    remove = cum - sorted_probs > p          # 不含自己的累计已经超过 p
    out = z.clone()
    out[sorted_idx[remove]] = -math.inf
    return out


def min_p_filter(z: torch.Tensor, min_p: float) -> torch.Tensor:
    """阈值 = 最大概率 × min_p，低于阈值的全部丢掉（vLLM builtin.py 的写法）。"""
    probs = softmax(z)
    thresh = probs.max() * min_p
    out = z.clone()
    out[probs < thresh] = -math.inf
    return out


def apply_penalties(z: torch.Tensor, prompt_tokens, output_tokens,
                    repetition: float, frequency: float, presence: float) -> torch.Tensor:
    """顺序与 vLLM 一致：repetition -> frequency -> presence。"""
    out = z.clone()
    n = z.numel()
    prompt_mask = torch.zeros(n, dtype=torch.bool)
    output_mask = torch.zeros(n, dtype=torch.bool)
    counts = torch.zeros(n, dtype=torch.float64)
    for t in prompt_tokens:
        prompt_mask[t] = True
    for t in output_tokens:
        output_mask[t] = True
        counts[t] += 1
    pen = torch.where(prompt_mask | output_mask,
                      torch.full_like(out, repetition), torch.ones_like(out))
    out = out * torch.where(out > 0, 1.0 / pen, pen)     # 正的除、负的乘
    out = out - frequency * counts
    out = out - presence * output_mask.to(out.dtype)
    return out


# ---------------------------------------------------------------- [0] 基础

def section_base(z: torch.Tensor) -> dict:
    head("[0] 四分类玩具分布（FP64，含负 logits）")
    p = softmax(z)
    print(f"    token        {'  '.join(n.rjust(11) for n in NAMES)}")
    print(f"    logits    {vec(z)}")
    print(f"    exp(z)    {vec(torch.exp(z))}")
    print(f"    prob      {vec(p)}")
    print(f"    logprob   {vec(torch.log(p))}")
    print(f"\n    手算核对：exp(2)=7.389056，sum={float(torch.exp(z).sum()):.6f}，"
          f"p(A)=7.389056/{float(torch.exp(z).sum()):.6f}={float(p[0]):.6f}")
    print(f"    概率之和 {float(p.sum()):.12f}，4 个数求和没有可见误差；"
          f"词表 15 万时这一项会偏离 1e-5 量级。")
    return dict(logits=z.tolist(), prob=p.tolist(), logprob=torch.log(p).tolist())


# ------------------------------------------------------------ [1] temperature

def section_temperature(z: torch.Tensor) -> dict:
    head("[1] temperature：逐步向量")
    rows = {}
    print(f"    {'T':>6s}  {'缩放后 logits':^50s}   {'概率':^50s}   熵")
    for T in (0.2, 0.5, 1.0, 1.5, 3.0):
        zt = z / T
        p = softmax(zt)
        ent = float(-(p * torch.log(p)).sum())
        rows[str(T)] = dict(logits=zt.tolist(), prob=p.tolist(), entropy=ent,
                            effective=math.exp(ent))
        print(f"    {T:>6.1f}  {vec(zt, 11, 4)}  {vec(p, 11, 6)}  {ent:6.4f}")
    print(f"\n    T→0 不是除以 0：引擎把 T<1e-5 的请求直接走 argmax 分支"
          f"（vLLM sampler.py:235 把温度替换成 1.0，再用 argmax 的结果覆盖）。")
    print(f"    熵的指数 exp(H) 是「有效候选数」：T=0.2 时 {math.exp(rows['0.2']['entropy']):.2f}，"
          f"T=3.0 时 {math.exp(rows['3.0']['entropy']):.2f}（词表只有 4）。")
    return rows


# ------------------------------------------------- [2] top-k / top-p / min-p

def section_filters(z: torch.Tensor) -> dict:
    head("[2] 三种截断：每一步的 mask 与重新归一化后的概率")
    p = softmax(z)
    print(f"    原始概率  {vec(p)}   累计 {vec(torch.cumsum(torch.sort(p, descending=True).values, 0))}")
    out = {}
    cases = [("top_k=1", lambda: top_k_filter(z, 1)),
             ("top_k=2", lambda: top_k_filter(z, 2)),
             ("top_p=0.78", lambda: top_p_filter(z, 0.78)),
             ("top_p=0.80", lambda: top_p_filter(z, 0.80)),
             ("top_p=0.96", lambda: top_p_filter(z, 0.96)),
             ("min_p=0.04", lambda: min_p_filter(z, 0.04)),
             ("min_p=0.05", lambda: min_p_filter(z, 0.05)),
             ("min_p=0.10", lambda: min_p_filter(z, 0.10)),
             ("min_p=0.30", lambda: min_p_filter(z, 0.30))]
    print(f"\n    {'策略':<12s}{'候选数':>6s}  {'过滤后 logits':^50s}   归一化概率")
    for name, fn in cases:
        zf = fn()
        pf = softmax(zf)
        keep = int((zf > -math.inf).sum())
        out[name] = dict(logits=zf.tolist(), prob=pf.tolist(), kept=keep,
                         kept_names=[NAMES[i] for i in range(len(NAMES)) if zf[i] > -math.inf])
        print(f"    {name:<12s}{keep:>6d}  {vec(zf, 11, 4)}  {vec(pf, 11, 6)}")
    print(f"\n    top_p=0.78 只留 1 个，因为 A 一个就占 {float(p[0]):.4f}；")
    print(f"    top_p=0.80 留 2 个：跨过阈值的那一个要保留，否则 p 小于最大概率时会一个都不剩。")
    print(f"    min_p 的阈值随分布走：max_prob × min_p。min_p=0.04 时阈值 "
          f"{float(p.max()) * 0.04:.6f}，C（{float(p[2]):.6f}）留下；"
          f"min_p=0.05 时阈值 {float(p.max()) * 0.05:.6f}，C 恰好低于它，被丢掉。")
    return out


# --------------------------------------------------------------- [3] 惩罚

def section_penalties(z: torch.Tensor) -> dict:
    head("[3] 三种惩罚：repetition 对正负 logits 的方向相反")
    print(f"    prompt token {PROMPT_TOKENS} = {[NAMES[i] for i in PROMPT_TOKENS]}，"
          f"已生成 {OUTPUT_TOKENS} = {[NAMES[i] for i in OUTPUT_TOKENS]}（A 两次、C 一次）")
    out = {}
    print(f"\n    {'设置':<34s}{'logits':^50s}   概率")
    base = dict(repetition=1.0, frequency=0.0, presence=0.0)
    for label, kw in [
        ("无惩罚", base),
        ("repetition=1.2", dict(base, repetition=1.2)),
        ("frequency=0.5", dict(base, frequency=0.5)),
        ("presence=0.3", dict(base, presence=0.3)),
        ("三者同时", dict(repetition=1.2, frequency=0.5, presence=0.3)),
    ]:
        zp = apply_penalties(z, PROMPT_TOKENS, OUTPUT_TOKENS, **kw)
        out[label] = dict(logits=zp.tolist(), prob=softmax(zp).tolist(), **kw)
        print(f"    {label:<34s}{vec(zp, 11, 4)}  {vec(softmax(zp), 11, 6)}")
    rep = out["repetition=1.2"]["logits"]
    print(f"\n    repetition=1.2 的作用方向按 logit 的符号分两支："
          f"A 是 {LOGITS[0]:+.1f} → {rep[0]:+.4f}（除以 1.2），"
          f"C 是 {LOGITS[2]:+.1f} → {rep[2]:+.4f}（乘以 1.2）。")
    print("    两支都让概率下降，但幅度不同；把公式写成「一律乘 1/penalty」会让负 logit 反而被奖励。")
    print("    D 从未出现，三种惩罚都不动它——惩罚只改已出现 token 的分数，不改其余部分。")
    print("    repetition 看 prompt ∪ 已生成，frequency 按出现次数线性扣，presence 只看出现与否。")
    print("    B 只在 prompt 里：repetition 罚到它，frequency/presence 不罚"
          f"（frequency=0.5 那行 B 仍是 {out['frequency=0.5']['logits'][1]:+.4f}）。")
    return out


# ---------------------------------------------------------- [4] 顺序不可交换

def section_order(z: torch.Tensor) -> dict:
    head("[4] 顺序交换的反例")
    out = {}

    # 温度 vs top-p
    a = top_p_filter(z / 0.5, 0.8)
    b = top_p_filter(z, 0.8) / 0.5
    out["temp_then_topp"] = dict(logits=a.tolist(), prob=softmax(a).tolist(),
                                 kept=int((a > -math.inf).sum()))
    out["topp_then_temp"] = dict(logits=b.tolist(), prob=softmax(b).tolist(),
                                 kept=int((b > -math.inf).sum()))
    print(f"    T=0.5 与 top_p=0.8")
    print(f"      先温度再截断   候选 {out['temp_then_topp']['kept']}   {vec(softmax(a), 11, 6)}")
    print(f"      先截断再温度   候选 {out['topp_then_temp']['kept']}   {vec(softmax(b), 11, 6)}")
    print("      先温度：分布被拉尖，A 一个就超过 0.8，候选集只剩 1 个；")
    print("      先截断：按原分布截断留 2 个，再降温只是把这 2 个的比例拉开。")

    # 温度 vs min-p
    c = min_p_filter(z / 0.5, 0.1)
    d = min_p_filter(z, 0.1) / 0.5
    out["temp_then_minp"] = dict(prob=softmax(c).tolist(), kept=int((c > -math.inf).sum()))
    out["minp_then_temp"] = dict(prob=softmax(d).tolist(), kept=int((d > -math.inf).sum()))
    print(f"\n    T=0.5 与 min_p=0.1")
    print(f"      先温度再 min_p  候选 {out['temp_then_minp']['kept']}   {vec(softmax(c), 11, 6)}")
    print(f"      先 min_p 再温度  候选 {out['minp_then_temp']['kept']}   {vec(softmax(d), 11, 6)}")
    print("      min_p 的阈值是「最大概率的比例」，而温度会改变最大概率，所以它对顺序敏感。")

    # 惩罚 vs 温度：repetition 可交换，frequency/presence 不可
    e = apply_penalties(z / 0.5, PROMPT_TOKENS, OUTPUT_TOKENS, 1.2, 0.0, 0.0)
    f = apply_penalties(z, PROMPT_TOKENS, OUTPUT_TOKENS, 1.2, 0.0, 0.0) / 0.5
    rep_same = float((e - f).abs().max())
    g = apply_penalties(z / 0.5, PROMPT_TOKENS, OUTPUT_TOKENS, 1.0, 0.5, 0.0)
    h = apply_penalties(z, PROMPT_TOKENS, OUTPUT_TOKENS, 1.0, 0.5, 0.0) / 0.5
    out["temp_then_rep"] = dict(logits=e.tolist(), prob=softmax(e).tolist())
    out["rep_then_temp"] = dict(logits=f.tolist(), prob=softmax(f).tolist())
    out["repetition_commutes_max_abs_diff"] = rep_same
    out["temp_then_freq"] = dict(logits=g.tolist(), prob=softmax(g).tolist())
    out["freq_then_temp"] = dict(logits=h.tolist(), prob=softmax(h).tolist())
    out["frequency_commutes_max_abs_diff"] = float((g - h).abs().max())
    print(f"\n    T=0.5 与 repetition=1.2（乘性）")
    print(f"      先温度再惩罚   {vec(e, 11, 4)}  {vec(softmax(e), 11, 6)}")
    print(f"      先惩罚再温度   {vec(f, 11, 4)}  {vec(softmax(f), 11, 6)}")
    print(f"      两者最大差 {rep_same:.3e}：repetition 是按符号分支的乘法，"
          f"而除以正的 T 不改变符号，所以这一对确实可交换。")
    print(f"\n    T=0.5 与 frequency=0.5（加性）")
    print(f"      先温度再惩罚   {vec(g, 11, 4)}  {vec(softmax(g), 11, 6)}")
    print(f"      先惩罚再温度   {vec(h, 11, 4)}  {vec(softmax(h), 11, 6)}")
    print(f"      两者最大差 {float((g - h).abs().max()):.4f}：减一个常数与除以 T 不可交换，"
          f"惩罚力度被温度整体放大了 1/T 倍。")
    print("      vLLM 的固定顺序是：惩罚（sampler.py:405）→ 温度（:277）→ min_p（:283）→ top-k/top-p（:287）；")
    print("      写自己的采样器时要对齐这个顺序，否则同样的参数含义不同。")
    return out


# --------------------------------------------------------------- [5] 并列

def section_ties() -> dict:
    head("[5] 并列：两种截断策略给出不同的分布")
    z = torch.tensor([2.0, 2.0, 0.5, -1.0], dtype=torch.float64)
    print(f"    logits {vec(z)}   A 与 B 完全相同")
    keep = top_k_filter(z, 1, keep_ties=True)
    strict = top_k_filter(z, 1, keep_ties=False)
    print(f"    top_k=1 保留并列   候选 {int((keep > -math.inf).sum())}   {vec(softmax(keep), 11, 6)}")
    print(f"    top_k=1 严格 k 个  候选 {int((strict > -math.inf).sum())}   {vec(softmax(strict), 11, 6)}")
    argmax = [int(z.argmax()) for _ in range(5)]
    print(f"\n    torch.argmax 在并列时的返回：{argmax}（取下标最小的那个，确定但任意）")
    print("    真实 logits 出现精确并列并不罕见：bf16 在 19 附近的间隔是 0.125，"
          "两个不同实数会落到同一个可表示值上（0.5 章的真实前向里排名 4 和 5 就是这种情况）。")
    return dict(logits=z.tolist(), keep_ties=softmax(keep).tolist(),
                strict_k=softmax(strict).tolist(), argmax_repeat=argmax)


# ----------------------------------------------- [6] 两种「正确」的定义

def section_correctness(z: torch.Tensor) -> dict:
    head("[6] greedy 与随机采样：两种正确性判据")
    zf = top_k_filter(z, 3)
    p = softmax(zf)
    print(f"    检验分布（top_k=3 后归一化）{vec(p)}")

    greedy = [int(zf.argmax()) for _ in SEEDS]
    print(f"\n    greedy：judgement 是 id 精确相等。三次调用都得到 {greedy}，"
          f"与 FP64 参照 argmax {int(torch.tensor(LOGITS).argmax())} 一致。")

    rows = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        draws = torch.multinomial(p, N_DRAWS, replacement=True, generator=g)
        freq = torch.bincount(draws, minlength=len(NAMES)).double() / N_DRAWS
        dev = float((freq - p).abs().max())
        expected = p * N_DRAWS
        mask = expected > 0
        chi2 = float((((freq * N_DRAWS - expected)[mask]) ** 2 / expected[mask]).sum())
        rows.append(dict(seed=seed, freq=freq.tolist(), max_abs_dev=dev, chi2=chi2))
    dof = int((p > 0).sum()) - 1
    crit = {1: 3.841, 2: 5.991, 3: 7.815}[dof]
    for r in rows:
        r["below_critical"] = r["chi2"] < crit
        print(f"    seed={r['seed']}  频率 {vec(torch.tensor(r['freq']), 11, 6)}  "
              f"max|Δ|={r['max_abs_dev']:.5f}  χ²={r['chi2']:.3f}  "
              f"{'通过' if r['below_critical'] else '超过临界值'}")
    n_fail = sum(1 for r in rows if not r["below_critical"])
    print(f"\n    随机采样：judgement 是分布检验，不是逐次相等。自由度 {dof}，"
          f"α=0.05 的临界值 {crit}。")
    print(f"    {len(rows)} 个 seed 里有 {n_fail} 个超过临界值。这本身就是检验的含义："
          f"α=0.05 表示正确的实现也有 5% 的单次运行会被判为不通过，")
    print(f"    跑 3 个 seed 至少出现一次的概率是 1-0.95³≈14%。"
          f"判断实现是否正确要看多次重复的整体表现，不能凭一次 χ² 下结论。")
    print(f"    统计涨落的量级是 1/√N = {1 / math.sqrt(N_DRAWS):.5f}，"
          f"实测 max|Δ| 与之同量级——要求「同 seed 同 token」在跨引擎场景下没有意义。")
    return dict(greedy=greedy, draws=N_DRAWS, dof=dof, critical=crit, per_seed=rows,
                reference_prob=p.tolist())


# -------------------------------------------- [7] 与 vLLM 的实现逐项对拍

def section_vllm_parity(z: torch.Tensor) -> dict:
    head("[7] 与 vLLM 0.29.0 的公式对拍（同一份 FP64 输入，两种写法）")
    n = len(NAMES)
    logits = z.clone().unsqueeze(0)
    prompt_mask = torch.zeros(1, n, dtype=torch.bool)
    output_mask = torch.zeros(1, n, dtype=torch.bool)
    counts = torch.zeros(1, n, dtype=torch.float64)
    for t in PROMPT_TOKENS:
        prompt_mask[0, t] = True
    for t in OUTPUT_TOKENS:
        output_mask[0, t] = True
        counts[0, t] += 1
    # vllm/_custom_ops.py:377 apply_repetition_penalties_torch 的原式
    penalties = torch.where(prompt_mask | output_mask,
                            torch.full((1, n), 1.2, dtype=torch.float64),
                            torch.ones(1, n, dtype=torch.float64))
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    ref = logits * scaling
    # vllm/model_executor/layers/utils.py:87-88
    ref = ref - 0.5 * counts
    ref = ref - 0.3 * output_mask.double()
    mine = apply_penalties(z, PROMPT_TOKENS, OUTPUT_TOKENS, 1.2, 0.5, 0.3)
    diff = float((ref[0] - mine).abs().max())
    print(f"    vLLM 公式 {vec(ref[0], 11, 6)}")
    print(f"    本脚本   {vec(mine, 11, 6)}")
    print(f"    最大绝对差 {diff:.3e}（FP64，判据是精确相等）")

    # min_p：vllm/v1/sample/logits_processor/builtin.py:102-116
    probs = torch.softmax(z, dim=-1)
    thresh = probs.amax() * 0.1
    ref_minp = z.clone()
    ref_minp[probs < thresh] = -math.inf
    mine_minp = min_p_filter(z, 0.1)
    same = bool(torch.equal(torch.isinf(ref_minp), torch.isinf(mine_minp)))
    print(f"    min_p 候选集与 vLLM 写法一致：{same}")
    return dict(penalty_max_abs_diff=diff, minp_same_mask=same,
                vllm_penalty=ref[0].tolist(), mine_penalty=mine.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_printoptions(precision=6, sci_mode=False)
    z = torch.tensor(LOGITS, dtype=torch.float64)

    print(f"词表 {NAMES}，logits {LOGITS}，dtype float64，torch {torch.__version__}")
    payload = dict(
        base=section_base(z),
        temperature=section_temperature(z),
        filters=section_filters(z),
        penalties=section_penalties(z),
        order=section_order(z),
        ties=section_ties(),
        correctness=section_correctness(z),
        vllm_parity=section_vllm_parity(z),
    )
    (out / "sampling_math.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.5-A", vocab=NAMES, logits=LOGITS, dtype="float64",
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        seeds=list(SEEDS), draws=N_DRAWS, torch=torch.__version__,
        python=platform.python_version(), host=platform.node(),
        reference="vLLM 0.29.0 sampler.py / _custom_ops.py / builtin.py 的公式",
        outputs=["sampling_math.json", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
