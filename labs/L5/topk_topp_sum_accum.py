#!/usr/bin/env python3
"""L5.9 任务 A 收尾 —— 检验"跨 tile 串行累加 sum_exp_logits"能否解释第三例。

上一轮排除了并列值、pivot 分辨率与 outlier 预筛，把候选机制收窄到
bisection 的跨 tile 归并。读内核源码后可以看到更具体的一处：

    sum_exp_logits = 0.0
    for i in range(0, NUM_TILES):          # 151936 / 128 = 1187 个 tile
        probs_blk = tl.exp(logits_blk - max_sample)
        sum_exp_logits += tl.sum(probs_blk)   # float32 串行累加

`sum_exp_logits` 是按 **tile 顺序串行相加的 float32**，而参照实现（torch）用自己
的分块/向量化求和。第 1187 次串行相加的舍入误差量级是 √1187·eps ≈ 4e-6（相对），
而判决只取决于 `p_pivots_sum >= p` 这条边——**只要累计概率与 0.9 的余量小于
这个误差，判决就会翻**。

这个脚本把那点余量算出来，并用三种求和方式各判一次，看谁能复现 Triton 的 45。

用法：
  python topk_topp_sum_accum.py --row mismatch-rows/B256-t4-r85.bin --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

VOCAB = 151936
TOPK, TOPP = 50, 0.9
BLOCK_SIZE = 128          # NUM_TILES = ceil(VOCAB/BLOCK_SIZE)，与内核一致


def keep_count(probs_sorted: np.ndarray, p: float = TOPP) -> int:
    """保留多少个：最小的 k 使前 k 个概率之和 ≥ p。"""
    cum = np.cumsum(probs_sorted)
    idx = int(np.searchsorted(cum, p))
    return idx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    row = np.frombuffer(args.row.read_bytes(), dtype="<f4").copy()
    assert row.size == VOCAB, row.size

    # 内核的 max_sample：取第一个 BLOCK，排除 -inf，算 avg/std（float32）
    blk0 = row[:BLOCK_SIZE].astype(np.float32)
    finite = blk0[np.isfinite(blk0)]
    avg = np.float32(finite.sum(dtype=np.float32) / np.float32(finite.size))
    sq = np.float32((finite.astype(np.float32) ** 2).sum(dtype=np.float32)
                    / np.float32(finite.size))
    var = np.float32(max(sq - avg * avg, 0.0))
    std = np.float32(np.sqrt(var))
    max_sample = np.float32(avg + std * np.float32(10.0))

    e = np.exp(row.astype(np.float32) - max_sample).astype(np.float32)

    # 三种求和
    S_serial = np.float32(0.0)
    for i in range(0, VOCAB, BLOCK_SIZE):
        S_serial = np.float32(S_serial + e[i:i + BLOCK_SIZE].sum(dtype=np.float32))
    S_torch_like = e.sum(dtype=np.float32)                  # 向量化分块
    S_exact = float(e.astype(np.float64).sum())              # float64 参照

    order = np.argsort(-row)[:TOPK]
    top_vals = row[order]
    res = {}
    for name, S in (("serial_tile_fp32", S_serial),
                    ("vectorized_fp32", S_torch_like),
                    ("float64", S_exact)):
        probs = (e / np.float32(S)).astype(np.float32)
        top_p = probs[order].astype(np.float64)
        k = keep_count(top_p)
        margin = float(np.cumsum(top_p)[k - 1] - TOPP)
        # 名次 k-1 处累计与 0.9 的余量：小于它才会翻
        res[name] = dict(sum_exp_logits=float(S), kept=k, margin=margin,
                         margin_over_kept=margin / TOPP)

    # 串行累加误差的相对量级
    eps = float(np.finfo(np.float32).eps)
    n_tiles = (VOCAB + BLOCK_SIZE - 1) // BLOCK_SIZE
    rel_err_bound = eps * n_tiles            # 最坏（全部同向）
    rel_err_rms = eps * np.sqrt(n_tiles)     # 随机游走
    out = dict(
        row_file=str(args.row), vocab=VOCAB, block_size=BLOCK_SIZE,
        num_tiles=n_tiles, max_sample=float(max_sample),
        avg_logit=float(avg), std_logit=float(std),
        sum_exp_logits=dict(serial_tile_fp32=float(S_serial),
                            vectorized_fp32=float(S_torch_like),
                            float64=float(S_exact)),
        rel_diff_serial_vs_exact=float((S_serial - S_exact) / S_exact),
        rel_diff_vector_vs_exact=float((S_torch_like - S_exact) / S_exact),
        decisions=res,
        error_scale=dict(eps=eps, worst_case_rel=rel_err_bound,
                         rms_rel=rel_err_rms),
        reference_kept=44,
    )
    (args.out / "sum_accum.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tile 数 {n_tiles}，max_sample {float(max_sample):.4f}")
    print(f"sum_exp_logits：串行 tile {float(S_serial):.6f} / 向量化 "
          f"{float(S_torch_like):.6f} / float64 {S_exact:.6f}")
    print(f"  串行相对 float64 偏差 {(S_serial - S_exact) / S_exact:+.3e}，"
          f"向量化 {(S_torch_like - S_exact) / S_exact:+.3e}")
    print(f"  float32 舍入量级：最坏 {rel_err_bound:.2e}，随机游走 {rel_err_rms:.2e}")
    for name, r in res.items():
        print(f"  {name:<18} 保留 {r['kept']:>3} 个  累计余量 "
              f"{r['margin']:+.3e}（相对 {r['margin_over_kept']:+.3e}）")
    print(f"  参照保留 44，Triton 实际保留 45")
    flip = [n for n, r in res.items() if r["kept"] == 45]
    print(f"\n判读：复现出 45 的求和方式：{flip or '无'}")
    print("      若某个 float32 求和方式给出 45 而 float64 给出 44，且其相对偏差与"
          "余量同量级，则第三例的机制就是 sum_exp_logits 的串行累加。")


if __name__ == "__main__":
    main()
