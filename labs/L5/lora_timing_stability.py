#!/usr/bin/env python3
"""L5.10 —— 用已有的交错重复数据判定"小幅非单调"是真效应还是噪声。

正文第 2 节写过：「1/4/7/8 的小幅非单调变化不作机制解释：这条趋势未解释，
不要引用。需要更多交错重复、功耗/频率记录和各档 kernel trace 才能判断是否稳定。」
但 `timings.json` 里其实已经有 **5 轮交错重复**（rep 是外层循环），
所以"是否稳定"这个问题可以用已有数据回答，不必等新实验：

  * 同一档内的重复离散度（min–max、CV）是多少；
  * 相邻档的差是不是**换一轮就变号**（配对比较，5 对）；
  * rep 维度上有没有系统性漂移（若有，说明存在时间相关因素，如频率/温度）。

判据（写死在脚本里，避免事后挑选）：
  一对配置的差若满足「5 轮中同号 ≥ 4 次」且「中位差 > 两档各自离散度的较大者」，
  判为稳定效应；否则判为落在噪声内、应读作"持平"。

用法：python labs/L5/lora_timing_stability.py --timings <timings.json> [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st


def load(path):
    rows = json.loads(pathlib.Path(path).read_text())
    by = {}
    for r in rows:
        by.setdefault(r["adapters"], {})[r["rep"]] = r["wall_ms"]
    return rows, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rows, by = load(args.timings)
    configs = sorted(by)
    report = {"n_configs": len(configs), "reps": len(next(iter(by.values()))),
              "per_config": {}, "per_rep_mean": {}, "pairs": []}

    for c in configs:
        xs = [by[c][r] for r in sorted(by[c])]
        report["per_config"][c] = dict(
            n=len(xs), median=round(st.median(xs), 2), min=round(min(xs), 2),
            max=round(max(xs), 2),
            spread_pct=round((max(xs) - min(xs)) / st.median(xs) * 100, 2),
            cv_pct=round(st.pstdev(xs) / st.mean(xs) * 100, 2))

    reps = sorted(next(iter(by.values())))
    for r in reps:
        vals = [by[c][r] for c in configs]
        report["per_rep_mean"][r] = round(st.mean(vals), 2)

    for a, b in zip(configs, configs[1:]):
        deltas = [by[b][r] - by[a][r] for r in reps]
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        med = st.median(deltas)
        spread = max(report["per_config"][a]["max"] - report["per_config"][a]["min"],
                     report["per_config"][b]["max"] - report["per_config"][b]["min"])
        stable = (max(pos, neg) >= 4) and (abs(med) > spread)
        report["pairs"].append(dict(
            pair=f"{a}->{b}", deltas_ms=[round(d, 2) for d in deltas],
            positive=pos, negative=neg, median_delta_ms=round(med, 2),
            within_config_spread_ms=round(spread, 2),
            verdict="稳定效应" if stable else "落在噪声内（读作持平）",
            pct=round(med / st.median([by[a][r] for r in reps]) * 100, 2)))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "lora_timing_stability.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'adapters':>9}{'median':>10}{'min':>9}{'max':>9}{'spread%':>9}{'CV%':>7}")
    for c in configs:
        p = report["per_config"][c]
        print(f"{c:>9}{p['median']:>10.1f}{p['min']:>9.1f}{p['max']:>9.1f}"
              f"{p['spread_pct']:>9.2f}{p['cv_pct']:>7.2f}")
    print("\n每轮各档均值（看有没有时间漂移）：")
    print("  " + "  ".join(f"rep{r}={v:.1f}" for r, v in report["per_rep_mean"].items()))
    print("\n相邻档配对比较（5 轮交错，delta>0 表示后一档更慢）：")
    for p in report["pairs"]:
        print(f"  {p['pair']:>9}  同号 {max(p['positive'], p['negative'])}/5  "
              f"中位差 {p['median_delta_ms']:>8.2f} ms（{p['pct']:>6.2f}%）  "
              f"档内离散 {p['within_config_spread_ms']:>7.2f} ms  → {p['verdict']}")


if __name__ == "__main__":
    main()
