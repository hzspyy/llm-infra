#!/usr/bin/env python3
"""L5.9 任务 A 续 —— 用保存下来的整行 logits 定位 5.850e-03 那一例。

上一轮把三条不一致分成了两类：两条的边界间隔（1.4e-6、1.7e-6）落在 pivot 搜索
18 次二分的分辨率带内，一条间隔 5.850e-03 却更大——那一例的解释仍未落地。
这一轮把那一条的**整行 logits** 从工件里读回来，直接问三个问题：

  1. 排序参照在 float32 与 float64 下的判决是否相同（判决本身稳不稳）；
  2. Triton 路径实际用的 pivot 落在哪个区间（最后被保留的 logit 与最先被丢弃的 logit）；
  3. 该区间与"精确边界"的差，换算到概率空间后，与二分分辨率 R/2^18 相比谁大。

第 3 问是判据：若差落在分辨率带内，这一例也是精度问题；若明显超出，
说明还有别的机制（例如 top-p 前置的 outlier 预筛把候选漏掉了，或并列处理）。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

VOCAB = 151936
TOPK, TOPP = 50, 0.9
BLOCK = 128          # 内核里第一个 BLOCK 用于估计 max_sample 的那一段


def load_row(path: pathlib.Path):
    return torch.from_numpy(
        np.frombuffer(path.read_bytes(), dtype="<f4").copy()).cuda()


def triton_kept(row):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    k = torch.full((1,), TOPK, dtype=torch.int32).cuda()
    p = torch.full((1,), TOPP, dtype=torch.float32).cuda()
    with torch.inference_mode():
        # 该接口要求 [batch, vocab] 二维输入
        out = apply_top_k_top_p(row.unsqueeze(0).clone(), k, p)
    return torch.isfinite(out[0]) & (out[0] > -1e30)


def sort_cut(vals, p=TOPP, dtype=torch.float32):
    v = vals.to(dtype).double() if dtype == torch.float64 else vals.to(dtype)
    probs = torch.softmax(v, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    return int(torch.searchsorted(cum, torch.tensor(p, dtype=v.dtype, device=v.device)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    row = load_row(args.row)
    assert row.numel() == VOCAB, row.numel()

    top = torch.topk(row, TOPK)
    vals, ids = top.values, top.indices

    c32 = sort_cut(vals, dtype=torch.float32)
    c64 = sort_cut(vals, dtype=torch.float64)
    kept = triton_kept(row)
    n_kept = int(kept.sum())

    # 精确边界 logit：参照保留到第 c32 名，所以边界落在 vals[c32] 与 vals[c32+1] 之间
    boundary_hi = float(vals[c32])
    boundary_lo = float(vals[c32 + 1]) if c32 + 1 < TOPK else float("-inf")

    # Triton 实际用的 pivot 区间：最后被保留的 logit 与最先被丢弃的 logit
    # （只看前 TOPK 名以内的候选，尾部大量低值不参与边界）
    head = row[ids]
    kept_head = kept[ids]
    kept_vals = head[kept_head]
    drop_vals = head[~kept_head]
    min_kept = float(kept_vals.min()) if kept_vals.numel() else float("nan")
    max_drop = float(drop_vals.max()) if drop_vals.numel() else float("nan")

    # 概率空间换算：d(logit) -> d(prob)，用边界处单个候选的概率份额
    probs = torch.softmax(vals.float(), dim=-1)
    share = float(probs[c32])
    # 二分分辨率：区间 R 减半 18 次
    res = {}
    for R in (1.0, 0.0158, 1e-3):
        dp = R / (2 ** 18)
        res[f"R={R}"] = dict(dp=dp, dlogit=dp / share)

    # 边界上是不是并列值：把与 boundary_lo 逐位相同的候选数出来
    tie_val = boundary_lo
    tie_mask = (row == tie_val)
    n_tie_total = int(tie_mask.sum())
    head_tie = int((head == tie_val).sum())
    # Triton 多留的那个是不是并列值之一
    extra_kept = kept_head & (head == tie_val)
    # 复算 top-p 前置的 outlier 预筛：内核用第一个 BLOCK 估 avg/std，
    # 再用 NORMAL_CDF_TO_SIGMA_TABLE 取 sigma，得到 outlier_prob。
    # 若精确边界上的候选低于这个阈值，它就不在搜索用的候选集里。
    from vllm.v1.sample.ops.topk_topp_triton import _NORMAL_CDF_TO_SIGMA_TABLE as NORMAL_CDF_TO_SIGMA_TABLE
    blk0 = row[:BLOCK].float()
    finite = blk0[torch.isfinite(blk0)]
    avg = float(finite.mean())
    std = float(finite.std(unbiased=False))
    max_sample = avg + std * 10.0
    S = float(torch.exp(row.float() - max_sample).sum())
    idx = max(0, min(int(TOPP * 200), 199))
    sigma = float(NORMAL_CDF_TO_SIGMA_TABLE[idx])
    sigma = sigma + abs(sigma) * -0.25
    outlier_pivot = avg + std * sigma
    outlier_prob = float(torch.exp(torch.tensor(outlier_pivot - max_sample)) / S)
    n_top50_in_outliers = int((probs > outlier_prob).sum())
    boundary_in_outliers = bool(probs[c32 + 1] > outlier_prob)
    outlier = dict(
        block=BLOCK, avg_logit=avg, std_logit=std, max_sample=max_sample,
        sigma=sigma, outlier_pivot=outlier_pivot, outlier_prob=outlier_prob,
        sum_exp_logits=S,
        top50_candidates_above_outlier_prob=n_top50_in_outliers,
        boundary_candidate_above_outlier_prob=boundary_in_outliers,
        boundary_probability=float(probs[c32 + 1]),
    )
    out = dict(
        row_file=str(args.row),
        outlier_prefilter=outlier,
        boundary_lo_bit_pattern=format(int(torch.tensor(tie_val).view(torch.int32).item()) & 0xFFFFFFFF, "08x"),
        ties_at_boundary_full_vocab=n_tie_total,
        ties_at_boundary_in_top50=head_tie,
        triton_extra_kept_is_tie=int(extra_kept.sum()),
        top50_logits=[round(float(x), 8) for x in vals.tolist()],
        top50_ids=[int(x) for x in ids.tolist()],
        cut_rank_float32=c32, cut_rank_float64=c64,
        cut_stable=(c32 == c64),
        triton_kept_count=n_kept,
        triton_kept_count_matches_reference=(n_kept == c32 + 1),
        boundary_logit_hi=boundary_hi, boundary_logit_lo=boundary_lo,
        boundary_gap=boundary_hi - boundary_lo if boundary_lo != float("-inf") else None,
        triton_pivot_interval=[max_drop, min_kept],
        pivot_vs_boundary=dict(
            upper=min_kept - boundary_hi,     # >0：pivot 高于精确边界
            lower=max_drop - boundary_lo),    # >0：被丢的最低者仍高于边界下沿
        boundary_probability_share=share,
        resolution=res,
    )
    (args.out / "pivot_replay.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"行文件 {args.row.name}  词汇 {row.numel()}")
    print(f"排序参照判决：float32 保留 {c32 + 1} 个，float64 保留 {c64 + 1} 个"
          f"（判决稳定 {c32 == c64}）")
    print(f"Triton 实际保留 {n_kept} 个，与参照一致 {n_kept == c32 + 1}")
    print(f"精确边界：{boundary_hi:.8f} ↓ {boundary_lo:.8f}"
          f"（间隔 {(boundary_hi - boundary_lo):.3e}）")
    print(f"Triton 的 pivot 区间：[{max_drop:.8f}, {min_kept:.8f}]")
    print(f"  相对精确边界：上沿 {min_kept - boundary_hi:+.3e}，"
          f"下沿 {max_drop - boundary_lo:+.3e}")
    print(f"边界值 {tie_val:.8f}（位型 {out['boundary_lo_bit_pattern']}）："
          f"全词表并列 {n_tie_total} 个，前 50 名内并列 {head_tie} 个；"
          f"Triton 多留的并列候选 {int(extra_kept.sum())} 个")
    print(f"边界处单候选概率份额 {share:.6f}")
    for k, v in res.items():
        print(f"  分辨率 {k:<8} 概率 {v['dp']:.3e} → logit {v['dlogit']:.3e}"
              f"  {'≥ 偏差（分辨率可解释）' if v['dlogit'] >= abs(min_kept - boundary_hi) else '< 偏差（分辨率解释不了）'}")
    print(f"outlier 预筛：avg {outlier['avg_logit']:.4f} std {outlier['std_logit']:.4f} "
          f"sigma {outlier['sigma']:.4f} → outlier_prob {outlier['outlier_prob']:.3e}")
    print(f"  前 50 名里超过该阈值的有 {outlier['top50_candidates_above_outlier_prob']} 个；"
          f"边界候选（p={outlier['boundary_probability']:.3e}）"
          f"{'在' if outlier['boundary_candidate_above_outlier_prob'] else '不在'}候选集内")
    print("\n判读：若 Triton 的 pivot 区间与精确边界相差远大于 R/2^18，")
    print("      说明这一例不是搜索精度问题，需要继续查 outlier 预筛与并列处理。")


if __name__ == "__main__":
    main()
