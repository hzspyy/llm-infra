#!/usr/bin/env python3
"""
6.3 任务 D：静态专家布局与 EPLB 式重平衡的对照。

问的是同一个问题：把热专家多复制几份、搬到别的 rank 上，换来的负载均衡能不能盖住
迁移本身的代价。所以每次都要同时给出三个数——均衡指标、迁移字节、以及迁移期间的
同步开销——只报其中一个都不足以支持结论。

路由轨迹取自 4.4 在 OLMoE 上采集的真实分布形状（每层都有自己的热专家，
逐层最热/最冷可达数百倍），这里用「每层独立的热专家集合」重建这个形状。

用法：
    python moe_placement_bench.py D --tokens 4096 --experts 64 --topk 8 --ranks 4 --out <dir>
    python moe_placement_bench.py D --rebalance-periods 1,4,16 --skew 0.5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time

import torch

OLMOE = {"num_experts": 64, "top_k": 8, "layers": 16, "hidden": 2048,
         "expert_bytes_bf16": 3 * 2048 * 1024 * 2}


def env_pins():
    return {"host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__}


def make_layer_routing(tokens, experts, top_k, skew, layer, seed=0):
    """造一层路由：每层有自己的一小撮热专家（与 4.4 观察到的跨层差异一致）。"""
    g = torch.Generator().manual_seed(seed * 1000 + layer)
    ids = torch.randint(0, experts, (tokens, top_k), generator=g)
    hot = max(1, int(experts * 0.05))
    n_hot = int(tokens * skew)
    if n_hot:
        hot_pool = torch.randint(0, experts, (hot,), generator=g)
        pick = hot_pool[torch.randint(0, hot, (n_hot, top_k), generator=g)]
        ids[:n_hot] = pick
    return ids


def rank_loads(ids, placement, ranks, experts, top_k):
    """给定专家→rank 的放置，算出每个 rank 收到的 (token,k) 数。"""
    owner = torch.tensor([placement[e] for e in range(experts)])
    own = owner[ids]                       # (T,K)
    return [int((own == r).sum()) for r in range(ranks)]


def rank_loads_replicated(ids, placement, replicas, ranks, experts):
    """把每个专家的 token 均分给它的所有持有者（主副本 + 复制副本）。

    这是「理想分流」下的负载：真实 router 不会把 token 精确均分，所以这条曲线给出的是
    EPLB 收益的乐观上界，实际收益要打折。
    """
    counts = torch.bincount(ids.reshape(-1), minlength=experts).tolist()
    loads = [0] * ranks
    for e in range(experts):
        holders = [placement[e]] + list(replicas.get(e, []))
        per, rem = divmod(counts[e], len(holders))
        for i, r in enumerate(holders):
            loads[r] += per + (1 if i < rem else 0)
    return loads


def balanced_metrics(loads):
    mean = statistics.fmean(loads) if loads else 0.0
    mn, mx = min(loads), max(loads)
    return {"loads": loads, "mean": mean,
            "max_over_mean": (mx / mean) if mean else None,
            "max_over_min": (mx / mn) if mn else None,
            "imbalance_cv": (statistics.pstdev(loads) / mean) if mean else None}


def eplb_placement(ids, ranks, experts, replicas_per_hot, hot_fraction=0.1):
    """EPLB 式重平衡：把最热的若干专家复制到负载最轻的 rank 上。

    返回 (placement, replica_count, moved_experts)。placement[e] 是主副本所在 rank；
    replicas 记录额外副本落在哪些 rank。
    """
    counts = torch.bincount(ids.reshape(-1), minlength=experts).tolist()
    order = sorted(range(experts), key=lambda e: -counts[e])
    n_hot = max(1, int(experts * hot_fraction))
    placement = {e: (e * ranks // experts) for e in range(experts)}   # 静态连续切分
    replicas = {e: [] for e in range(experts)}
    loads = rank_loads(ids, placement, ranks, experts, ids.shape[1])
    moved = []
    for e in order[:n_hot]:
        for _ in range(replicas_per_hot):
            # 找一个当前最轻、且还没有该专家副本的 rank
            cand = sorted(range(ranks), key=lambda r: loads[r])
            for r in cand:
                if r != placement[e] and r not in replicas[e]:
                    replicas[e].append(r)
                    # 复制过去的副本承担一半负载（简化的分流模型）
                    share = counts[e] // (len(replicas[e]) + 1)
                    loads[r] += share
                    loads[placement[e]] -= share
                    moved.append({"expert": e, "to_rank": r,
                                  "bytes": OLMOE["expert_bytes_bf16"]})
                    break
    return placement, replicas, moved


def run_D(args):
    os.makedirs(args.out, exist_ok=True)
    report = {"config": vars(args), "model": "OLMOE-1B-7B-0924", "layers": [], "env": env_pins()}
    for layer in range(args.layers):
        ids = make_layer_routing(args.tokens, args.experts, args.topk,
                                 args.skew, layer, seed=args.seed)
        static = {e: (e * args.ranks // args.experts) for e in range(args.experts)}
        base = rank_loads(ids, static, args.ranks, args.experts, args.topk)
        bm = balanced_metrics(base)
        row = {"layer": layer,
               "static": bm,
               "tokens": args.tokens, "top_k": args.topk}
        for rep in args.replicas:
            pl, replicas, moved = eplb_placement(ids, args.ranks, args.experts, rep)
            after = rank_loads_replicated(ids, pl, replicas, args.ranks, args.experts)
            am = balanced_metrics(after)
            bytes_moved = sum(m["bytes"] for m in moved)
            row[f"eplb_r{rep}"] = {
                "metrics": am,
                "moved_experts": len(moved),
                "migration_bytes": bytes_moved,
                "migration_MiB": bytes_moved / 2**20,
                "gain_max_over_mean": ((bm["max_over_mean"] - am["max_over_mean"])
                                       / bm["max_over_mean"]) if bm["max_over_mean"] else None,
                "hot_experts": sorted(range(args.experts),
                                      key=lambda e: -int((ids == e).sum()))[:5],
            }
        report["layers"].append(row)

    # 汇总：静态 vs 各复制度
    summary = {}
    for key in ["static"] + [f"eplb_r{r}" for r in args.replicas]:
        vals = []
        for row in report["layers"]:
            if key == "static":
                vals.append(row["static"]["max_over_mean"])
            else:
                vals.append(row[key]["metrics"]["max_over_mean"])
        summary[key] = {
            "max_over_mean_median": statistics.median(vals),
            "max_over_mean_max": max(vals),
            "max_over_mean_min": min(vals),
        }
    report["summary"] = summary

    # 重平衡周期：一个 epoch 走完 args.layers 层；每 period 层重新求解一次放置。
    # 周期越短 → 求解次数越多 → 迁移字节越多，这是迁移代价的直接来源。
    import math
    period_rows = []
    rep0 = args.replicas[0] if args.replicas else 1
    for period in args.rebalance_periods:
        solves = math.ceil(args.layers / period)
        per_solve_bytes = 0
        for i in range(solves):
            layer = min(i * period, args.layers - 1)
            ids = make_layer_routing(args.tokens, args.experts, args.topk,
                                     args.skew, layer, seed=args.seed)
            _, _, moved = eplb_placement(ids, args.ranks, args.experts, rep0)
            per_solve_bytes += sum(m["bytes"] for m in moved)
        period_rows.append({"period_layers": period, "solves_per_epoch": solves,
                            "bytes_per_solve": per_solve_bytes,
                            "migration_MiB_per_epoch": per_solve_bytes / 2**20})
    report["rebalance_periods"] = period_rows

    with open(os.path.join(args.out, "placement.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[D] tokens={args.tokens} experts={args.experts} topk={args.topk} "
          f"ranks={args.ranks} skew={args.skew}")
    for key, v in summary.items():
        print(f"    {key:<10} max/mean 中位={v['max_over_mean_median']:.2f} "
              f"最大={v['max_over_mean_max']:.2f} 最小={v['max_over_mean_min']:.2f}")
    for row in report["layers"][:3]:
        line = f"    层{row['layer']}: 静态 max/mean={row['static']['max_over_mean']:.2f} 负载={row['static']['loads']}"
        for rep in args.replicas:
            e = row[f"eplb_r{rep}"]
            line += (f" | r{rep} max/mean={e['metrics']['max_over_mean']:.2f} "
                     f"迁移={e['migration_MiB']:.1f} MiB 收益={e['gain_max_over_mean'] * 100:.0f}%")
        print(line)
    for pr in period_rows:
        print(f"    重平衡周期={pr['period_layers']} 层 → 每 epoch 求解 {pr['solves_per_epoch']} 次，"
              f"迁移 {pr['migration_MiB_per_epoch']:.1f} MiB/epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["D"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--skew", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--replicas", default="1,2")
    ap.add_argument("--rebalance-periods", default="1,4,16")
    a = ap.parse_args()
    a.replicas = [int(x) for x in a.replicas.split(",") if x]
    a.rebalance_periods = [int(x) for x in a.rebalance_periods.split(",") if x]
    run_D(a)


if __name__ == "__main__":
    sys.exit(main())
