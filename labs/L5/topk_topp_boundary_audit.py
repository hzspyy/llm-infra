#!/usr/bin/env python3
"""L5.9 任务 A —— 定位 Triton 与排序参照在 top-k/top-p 边界上的候选差异。

已保存的两条差异（`results/crater/sampling/20260911-1045/gpu-r2/`）：
batch 64 的 token 40077 排序参照保留、dispatch 丢弃；batch 256 的 token 80704 反过来。
本章此前只记录了差异，没有定位。本脚本做三件事：

  1. **重建输入**：用记录里的 top-55 logits/ids 还原一整行 logits，
     低尾填成远低于第 55 名（贡献可忽略），并验证重建行的前 50 累计概率
     与记录里的 `top50_cumulative` 一致；
  2. **复现差异**：在 GPU 上跑真实的 `apply_top_k_top_p`（Triton pivot 路径）与
     `apply_top_k_top_p_pytorch`（排序参照），看争议 token 的去留是否与记录一致；
  3. **量化边界分辨率**：构造一组合成行，让 top-p 的截断正好落在两个相邻候选之间，
     扫描它们的间隔 g，找出两条路径开始不一致的阈值；再与 pivot 搜索自身的
     分辨率（`(max_range − min_range) / 2^18`）对照。

用装了 vLLM 与 CUDA 的 serve venv 运行：
    python labs/L5/topk_topp_boundary_audit.py --out "$OUT/boundary"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

VOCAB = 151936
TOPK, TOPP = 50, 0.9
DEFAULT_MISMATCH_DIR = pathlib.Path(
    "results/crater/sampling/20260911-1045/gpu-r2")


def load_records(base: pathlib.Path):
    out = []
    for name, tag in [("mismatch-B64.json", "B64"), ("mismatch-B256.json", "B256")]:
        p = base / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())[0]
        out.append(dict(tag=tag, **d))
    return out


def reconstruct(rec, tail_drop=40.0):
    """把 top-55 记录还原成一整行 logits；低尾远低于第 55 名，贡献可忽略。"""
    import torch
    row = torch.full((1, VOCAB), float(rec["top55_logits"][-1]) - tail_drop,
                     dtype=torch.float32)
    ids = torch.tensor(rec["top55_ids"], dtype=torch.long)
    vals = torch.tensor(rec["top55_logits"], dtype=torch.float32)
    row[0, ids] = vals
    return row


def boundary_facts(logits_row, topk=TOPK, topp=TOPP):
    """按排序参照给出边界秩、累计概率与相邻间隔。"""
    import torch
    vals = logits_row[0].float()
    top = torch.topk(vals, topk).values
    probs = torch.softmax(top, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    cut = int(torch.searchsorted(cum, torch.tensor(topp, device=cum.device)).item())
    gaps = (top[:-1] - top[1:]).tolist()
    return dict(cut_rank_zero_based=cut,
                cum= cum.tolist(),
                top=top.tolist(),
                gaps=gaps)


def run_paths(logits_row):
    """返回 (排序参照保留集, dispatch 保留集)。"""
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p, apply_top_k_top_p_pytorch)
    k = torch.full((1,), TOPK, dtype=torch.int32)
    p = torch.full((1,), TOPP, dtype=torch.float32)
    row = logits_row.cuda()
    with torch.inference_mode():
        a = apply_top_k_top_p(row.clone(), k.cuda(), p.cuda())
        b = apply_top_k_top_p_pytorch(row.clone(), k.cuda(), p.cuda())
    keep_a = torch.isfinite(a[0]) & (a[0] > -1e30)
    keep_b = torch.isfinite(b[0]) & (b[0] > -1e30)
    return keep_b, keep_a


def synth_row(gap, boundary_rank=43, k=TOPK, topp=TOPP, tail_drop=8.0):
    """造一行 logits，使 top-p 的截断**正好落在** swept 的那一对候选之间。

    先解一个几何递减的头部（k 个值），让累计概率在第 boundary_rank 名刚好越过 topp
    （即 cum[b-1] < topp <= cum[b]）；再把第 b+1 名放到第 b 名下方 gap 处。
    此时正确边界位于这两名之间，宽度就是 gap —— 分辨率不够就会翻。
    """
    import torch

    def head_for(d):
        return -d * torch.arange(k, dtype=torch.float32)

    lo, hi = 1e-4, 1.0
    for _ in range(80):
        d = (lo + hi) / 2
        cum = torch.softmax(head_for(d), -1).cumsum(-1)
        if cum[boundary_rank] < topp:
            hi = d
        else:
            lo = d
    head = head_for((lo + hi) / 2)
    head = head.clone()
    head[boundary_rank + 1] = head[boundary_rank] - gap
    for j in range(boundary_rank + 2, k):
        head[j] = head[boundary_rank] - tail_drop
    row = torch.full((1, VOCAB), float(head[-1]) - 40.0, dtype=torch.float32)
    row[0, torch.arange(k, dtype=torch.long)] = head
    return row


def random_repro(batches, trials, seed, out):
    """复现原实验：randn logits、k=50、p=0.9，统计不一致并把边界事实记全。

    关键补充：原工件只留了 top-55，无法重建判决；这里对每一行都记下
    排序参照的边界秩与边界间隔，并推出 dispatch 路径实际用的 pivot 区间
    （最后被保留的 logit 与最先被丢弃的 logit 之间）。
    """
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p, apply_top_k_top_p_pytorch)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, summary = [], []
    for B in batches:
        n_dis = 0
        gaps_agree, gaps_dis = [], []
        for trial in range(trials):
            x = torch.randn((B, VOCAB), device="cuda", dtype=torch.float32,
                            generator=gen)
            k = torch.full((B,), TOPK, device="cuda", dtype=torch.int32)
            p = torch.full((B,), TOPP, device="cuda")
            with torch.inference_mode():
                a = apply_top_k_top_p(x.clone(), k, p)
                b = apply_top_k_top_p_pytorch(x.clone(), k, p)
            ka = torch.isfinite(a)
            kb = torch.isfinite(b)
            neq = (ka != kb).any(dim=1)
            for row in torch.nonzero(neq).flatten().tolist():
                # 原工件只留 top-55 导致无法重建判决；这里把整行原始 logits 落盘
                (out / "mismatch-rows").mkdir(exist_ok=True)
                (out / "mismatch-rows" / f"B{B}-t{trial}-r{row}.bin").write_bytes(
                    x[row].detach().cpu().numpy().astype("<f4").tobytes())
                top = torch.topk(x[row], TOPK).values
                pr = torch.softmax(top, -1).cumsum(-1)
                c = int(torch.searchsorted(pr, torch.tensor(TOPP, device=pr.device)).item())
                gap = float(top[c] - top[c + 1]) if c + 1 < TOPK else float("nan")
                kept = x[row][ka[row]]
                drop_head = x[row][~ka[row]]
                min_kept = float(kept.min())
                # 只把靠前的被丢候选算进 pivot 区间（尾部有大量低值）
                max_drop = float(torch.topk(drop_head, min(TOPK, drop_head.numel())).values.max())
                rows.append(dict(batch=B, trial=trial, row=row,
                                 ref_cut_rank=c, boundary_gap=gap,
                                 min_kept_logit=min_kept,
                                 max_dropped_logit=max_drop,
                                 pivot_interval=(max_drop, min_kept),
                                 kept_sort=int(kb[row].sum()),
                                 kept_dispatch=int(ka[row].sum()),
                                 sort_kept=bool(kb[row, row])))
                gaps_dis.append(gap)
                n_dis += 1
            # 同批里抽若干"一致"的行做对照
            for row in range(0, min(B, 8)):
                if bool(neq[row]):
                    continue
                top = torch.topk(x[row], TOPK).values
                pr = torch.softmax(top, -1).cumsum(-1)
                c = int(torch.searchsorted(pr, torch.tensor(TOPP, device=pr.device)).item())
                gaps_agree.append(float(top[c] - top[c + 1]) if c + 1 < TOPK
                                  else float("nan"))
        summary.append(dict(batch=B, trials=trials, disagreements=n_dis,
                            gap_disagree_max=max(gaps_dis) if gaps_dis else None,
                            gap_disagree_min=min(gaps_dis) if gaps_dis else None,
                            gap_agree_min=min(g for g in gaps_agree
                                              if g == g) if gaps_agree else None))
        print(f"  B={B:>3}  {trials} 轮 × {B} 行：不一致 {n_dis} 次"
              f"  不一致行的边界间隔 "
              f"{min(gaps_dis) if gaps_dis else float('nan'):.3e}"
              f"–{max(gaps_dis) if gaps_dis else float('nan'):.3e}"
              f"  一致行的最小间隔 "
              f"{min(g for g in gaps_agree if g == g) if gaps_agree else float('nan'):.3e}")
    (out / "random_repro.json").write_text(
        json.dumps(dict(seed=seed, batches=batches, trials=trials,
                        summary=summary, rows=rows), ensure_ascii=False,
                   indent=2), encoding="utf-8")
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--mismatch-dir", type=pathlib.Path,
                    default=DEFAULT_MISMATCH_DIR)
    ap.add_argument("--random-repro", action="store_true",
                    help="复现原实验：randn logits 扫 batch，统计不一致并记录边界事实")
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--gaps", type=float, nargs="+",
                    default=[1e-2, 1e-3, 1e-4, 1e-5, 4e-6, 2e-6, 1e-6,
                             5e-7, 2e-7, 1e-7, 1e-8])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import torch
    print(f"torch {torch.__version__}  vocab {VOCAB}  top_k {TOPK}  top_p {TOPP}")
    if args.random_repro:
        print("\n复现原实验（randn logits，k=50，p=0.9）：")
        summary, rows = random_repro(args.batches, args.trials, args.seed, args.out)
        if rows:
            ds = [r["boundary_gap"] for r in rows if r["boundary_gap"] == r["boundary_gap"]]
            print(f"\n  不一致共 {len(rows)} 行；边界间隔最小 {min(ds):.3e} 最大 {max(ds):.3e}")
            print("  其中 pivot 区间（最后保留 − 最先丢弃）宽度示例：")
            for r in rows[:5]:
                print(f"    B={r['batch']} row={r['row']} 边界秩 {r['ref_cut_rank']} "
                      f"边界间隔 {r['boundary_gap']:.3e} 保留数 "
                      f"sort={r['kept_sort']} dispatch={r['kept_dispatch']}")
        else:
            print("  本轮未出现不一致（换 seed 或增加 trials 再试）")
        sys.stdout.flush()
        os._exit(0)
    report = dict(torch=torch.__version__, vocab=VOCAB, topk=TOPK, topp=TOPP,
                  records=[], sweep=[])

    # ---------- 1/2：重建并复现记录里的差异 ----------
    for rec in load_records(args.mismatch_dir):
        row = reconstruct(rec)
        facts = boundary_facts(row)
        keep_sort, keep_disp = run_paths(row)
        tid = rec["token_id"]
        rank = rec["rank_in_top55_zero_based"][0]
        got_sort = bool(keep_sort[tid].item())
        got_disp = bool(keep_disp[tid].item())
        n_sort, n_disp = int(keep_sort.sum().item()), int(keep_disp.sum().item())
        # 重建保真度：前 50 累计概率与记录对比
        max_dev = max(abs(a - b) for a, b in
                      zip(facts["cum"][:50], rec["top50_cumulative"]))
        entry = dict(tag=rec["tag"], token_id=tid, rank=rank,
                     recorded=dict(sort_kept=rec["sort_kept"],
                                   dispatch_kept=rec["dispatch_kept"]),
                     reproduced=dict(sort_kept=got_sort, dispatch_kept=got_disp),
                     reproduced_ok=(got_sort == rec["sort_kept"]
                                    and got_disp == rec["dispatch_kept"]),
                     kept_counts=dict(sort=n_sort, dispatch=n_disp),
                     cumulative_reconstruction_max_dev=max_dev,
                     gap_above=facts["top"][rank - 1] - facts["top"][rank],
                     gap_below=facts["top"][rank] - facts["top"][rank + 1],
                     cum_before=facts["cum"][rank - 1],
                     cum_at=facts["cum"][rank])
        report["records"].append(entry)
        print(f"\n[{rec['tag']}] token {tid}（第 {rank} 名）"
              f"  记录 sort_kept={rec['sort_kept']} dispatch_kept={rec['dispatch_kept']}")
        print(f"  重建保真度：前 50 累计概率与记录最大偏差 {max_dev:.3e}")
        print(f"  复现结果：sort_kept={got_sort} dispatch_kept={got_disp}"
              f"  一致={entry['reproduced_ok']}  保留数 {n_sort} vs {n_disp}")
        print(f"  相邻 logit 间隔：上 {entry['gap_above']:.3e}  下 {entry['gap_below']:.3e}")
        print(f"  累计概率：到上一名为 {entry['cum_before']:.10f}，"
              f"到本名为 {entry['cum_at']:.10f}（top_p={TOPP}）")

    # ---------- 2b：用 (k,p) 组合隔离差异出在哪一级 ----------
    print("\n隔离：把 top-k 或 top-p 单独关掉，看差异是否还在")
    for rec in load_records(args.mismatch_dir):
        row = reconstruct(rec)
        tid = rec["token_id"]
        import torch
        from vllm.v1.sample.ops.topk_topp_sampler import (
            apply_top_k_top_p, apply_top_k_top_p_pytorch)
        combos = [("k=50,p=0.9", 50, 0.9), ("k=50,p=1.0", 50, 1.0),
                  (f"k={VOCAB},p=0.9", VOCAB, 0.9)]
        line = []
        for label, kk, pp in combos:
            k = torch.full((1,), kk, dtype=torch.int32).cuda()
            pp_t = torch.full((1,), pp, dtype=torch.float32).cuda()
            r = row.cuda()
            with torch.inference_mode():
                a = apply_top_k_top_p(r.clone(), k, pp_t)
                b = apply_top_k_top_p_pytorch(r.clone(), k, pp_t)
            ka = torch.isfinite(a[0]) & (a[0] > -1e30)
            kb = torch.isfinite(b[0]) & (b[0] > -1e30)
            same = int(ka.sum().item()) == int(kb.sum().item()) and bool(
                torch.equal(ka, kb))
            line.append(f"{label}: 保留 {int(kb.sum())}/{int(ka.sum())} 一致={same}")
        print(f"  [{rec['tag']}] " + " | ".join(line))

    # ---------- 3：合成扫描，找不一致的间隔阈值 ----------
    print("\n合成扫描：让截断落在两个间隔 g 的候选之间，看两条路径何时开始不一致")
    print(f"  {'gap':>9} {'排序参照保留数':>14} {'dispatch 保留数':>16}  一致")
    first_bad = None
    for g in args.gaps:
        row = synth_row(g)
        ks, kd = run_paths(row)
        ns, nd = int(ks.sum().item()), int(kd.sum().item())
        ok = ns == nd
        if not ok and first_bad is None:
            first_bad = g
        report["sweep"].append(dict(gap=g, kept_sort=ns, kept_dispatch=nd, agree=ok))
        print(f"  {g:>9.0e} {ns:>14} {nd:>16}  {ok}")
    print(f"\n  首次出现不一致的间隔：{first_bad if first_bad is not None else '未出现'}")

    # ---------- 分辨率算术 ----------
    # 概率空间：bisection 区间宽度 (max_range - min_range) 每次减半，最多 18 次
    print("\n分辨率算术（pivot 搜索最多 18 次二分，收敛判据 1e-9 达不到）")
    res = dict(iters_cap=18, tol=1e-9, per_record=[],
               note="区间每次减半，18 次后宽度 = R/2^18；把宽度换算到 logit 空间"
                    "要与该处的边界概率相除（d(logit) = d(prob)/prob）")
    for rec, entry in zip(load_records(args.mismatch_dir), report["records"]):
        p_edge = entry["cum_at"] - entry["cum_before"]   # 边界处单个候选的概率份额
        for R, label in [(1.0, "R=1（pivot 全域）"), (0.0158, "R=0.0158"),
                         (1e-3, "R=1e-3")]:
            dp = R / (2 ** 18)
            rowres = dict(tag=rec["tag"], range=label, dp=dp,
                          dlogit=dp / max(p_edge, 1e-12),
                          observed_gap_below=entry["gap_below"])
            res["per_record"].append(rowres)
        print(f"  [{rec['tag']}] 边界处单候选概率 {p_edge:.6f}，"
              f"与下一名的 logit 间隔 {entry['gap_below']:.3e}")
        for R, label in [(1.0, "R=1"), (0.0158, "R=0.0158"), (1e-3, "R=1e-3")]:
            dp = R / 2 ** 18
            print(f"      {label:<12} 概率分辨率 {dp:.3e} → logit 分辨率 "
                  f"{dp / max(p_edge, 1e-12):.3e}"
                  f"  {'≥ 观测间隔（会翻）' if dp / max(p_edge, 1e-12) >= entry['gap_below'] else '< 观测间隔'}")
    report["resolution"] = res

    (args.out / "boundary_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
