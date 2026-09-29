#!/usr/bin/env python3
"""L5.9 —— 独立验证"保留数差一"的边界规则（不依赖内核调试输出）。

上一轮从合并分支的代码读到判决式

    num_keep = num_duplicate_logit - uint32((p_pivots_sum - p) / min_larger_prob)

并据此提出：内核的保留规则是"pivot 加上修正不到一个候选的边界候选"，
与排序参照的"累计刚好达到 p 的最小前缀"是两套定义，因此会在边界上差一个。
这一轮用**同一批随机行**把这条推断做成可证伪的统计：

  对每一行同时算出
    参照保留数 r_ref = 最小前缀（在 top-50 上重新归一化后累计 ≥ 0.9）
    内核保留数 r_kernel = apply_top_k_top_p(k=50, p=0.9) 的保留个数
    越过 p 时的超出量与边界候选质量的比 ratio = (cum[r_ref]-0.9)/prob[r_ref]

  然后看**不一致是否只出现在 ratio < 1 的行**。
  若成立，规则假说得到独立支持；若不一致在 ratio 远大于 1 时也出现，
  则判决式不是唯一机制，需要继续找。

用法：python topk_topp_keep_rule_probe.py --out DIR [--rounds 20] [--batch 256]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

VOCAB = 151936
TOPK, TOPP = 50, 0.9


def reference_keep(row):
    """排序参照：top-50 上重新归一化，最小前缀使累计 ≥ p。返回 (保留数, ratio)。"""
    vals = torch.topk(row, TOPK).values.double()
    probs = torch.softmax(vals, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    c = int(torch.searchsorted(cum, torch.tensor(TOPP, dtype=torch.float64,
                                                device=cum.device)).item())
    c = min(c, TOPK - 1)
    overshoot = float(cum[c] - TOPP)
    mass = float(probs[c])
    return c + 1, (overshoot / mass if mass > 0 else float("inf"))


def kernel_keep(rows):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    k = torch.full((rows.shape[0],), TOPK, dtype=torch.int32, device=rows.device)
    p = torch.full((rows.shape[0],), TOPP, dtype=torch.float32, device=rows.device)
    with torch.inference_mode():
        out = apply_top_k_top_p(rows.clone(), k, p)
    finite = torch.isfinite(out) & (out > -1e30)
    return finite.sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    agree, disagree = [], []
    for r in range(args.rounds):
        rows = torch.randn(args.batch, VOCAB, device="cuda", dtype=torch.float32,
                           generator=g)
        rk = kernel_keep(rows).cpu()
        for b in range(args.batch):
            ref, ratio = reference_keep(rows[b])
            got = int(rk[b])
            rec = dict(round=r, row=b, reference=ref, kernel=got, ratio=ratio)
            (agree if got == ref else disagree).append(rec)
        print(f"  round {r}: 累计 {len(agree)} 行一致 / {len(disagree)} 行不一致",
              flush=True)

    def stats(rows_):
        if not rows_:
            return dict(n=0)
        rs = sorted(x["ratio"] for x in rows_)
        return dict(n=len(rows_), ratio_min=rs[0], ratio_p50=rs[len(rs) // 2],
                    ratio_max=rs[-1], ratio_lt_1=sum(1 for x in rs if x < 1.0))

    # 只保留 ratio < 1 的行做成"同分布对照"：看这个区间内一致率是多少
    band = [x for x in disagree if x["ratio"] < 1.0]
    out = dict(rounds=args.rounds, batch=args.batch, vocab=VOCAB,
               total_rows=len(agree) + len(disagree),
               agree=len(agree), disagree=len(disagree),
               disagree_stats=stats(disagree), agree_stats=stats(agree),
               disagreements=disagree[:20],
               disagree_with_ratio_lt_1=len(band),
               max_ratio_among_disagreements=max((x["ratio"] for x in disagree),
                                                 default=None),
               min_ratio_among_agreements=min((x["ratio"] for x in agree),
                                              default=None))
    (args.out / "keep_rule_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n共 {out['total_rows']} 行：一致 {len(agree)}，不一致 {len(disagree)}")
    print(f"不一致行的 ratio 分布：{out['disagree_stats']}")
    print(f"一致行的 ratio 分布：{out['agree_stats']}")
    print(f"不一致行的最大 ratio = {out['max_ratio_among_disagreements']}")
    print(f"一致行的最小 ratio = {out['min_ratio_among_agreements']}")
    print("\n判读：若不一致行全部落在 ratio < 1 而一致行在 ratio ≫ 1 时也出现，")
    print("      则'边界修正不到一个候选'的规则得到独立支持；")
    print("      若不一致行里出现 ratio 明显大于 1 的，说明还有别的机制。")


if __name__ == "__main__":
    main()
