#!/usr/bin/env python3
"""L5.3 任务 C 的预测部分 · step 时间模型、mini 预测与预测误差注入。

调度器要用「这一步大概要跑多久」做决策（选分块大小、决定准入）。这个时间只能预测，
所以要知道预测错了会怎样。本文件只用已经采到的数字做三件事：

  1. 从预算扫描的原始逐 step 记录里拟合 step 时间模型。先说明为什么一个线性模型
     不够：纯 decode 步（每步 8 个 token）约 3.3 ms，含长 prefill 的步可以到 158 ms，
     两者不落在同一条直线上。所以按两段拟合：纯 decode 步取中位数；
     含 prefill 的步拟合 t = c + b·n。
  2. 把 mini 调度器的计数轨迹过一遍这个模型，预测长请求 TTFT 与 decode 最大间隔，与实测并排。
  3. 给 b 注入 ±25%/±50% 误差，看一个「把单步控制在 15 ms」的分块控制器在真实斜率下落到哪里。

用法：
    python schedule_prediction.py --results results/crater/5.3/budget-20260921 \
        --out results/local/5.3/schedule-prediction
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_scheduler import Scheduler, trace_long_insert    # noqa: E402

TARGET_STEP_MS = 15.0


def fit_linear(rows):
    xs = [x for x, _ in rows]
    ys = [y for _, y in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = (sum((x - mx) * (y - my) for x, y in rows)
         / sum((x - mx) ** 2 for x in xs))
    return my - b * mx, b


def load_steps(results_dir):
    """逐 step 记录：(调度 token 数, 墙钟 ms, 这一步是否含 prefill)。"""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "run_b*_r*.json"))):
        with open(path) as f:
            run = json.load(f)["run"]
        for st in run["steps"]:
            if st["tokens"] <= 0:
                continue
            has_prefill = any(r["phase"].startswith("prefill")
                              for r in st.get("reqs", {}).values())
            out.append(dict(budget=run["budget"], repeat=run["repeat"],
                            tokens=int(st["tokens"]), wall_ms=float(st["wall_ms"]),
                            has_prefill=has_prefill))
    return out


def fit_two_regime(steps):
    dec = [s["wall_ms"] for s in steps if not s["has_prefill"]]
    pre = [(float(s["tokens"]), s["wall_ms"]) for s in steps if s["has_prefill"]]
    c, b = fit_linear(pre)
    return dict(decode_ms=statistics.median(dec), decode_n=len(dec),
                c=c, b=b, prefill_n=len(pre))


def simulate(model, budget):
    """mini 计数轨迹 → 时间：含 prefill 的步用 c+b·n，纯 decode 步用中位数。"""
    sched = Scheduler("decode_priority", budget, num_blocks=4096,
                      trace=trace_long_insert())
    sched.run()
    times = {}
    for s in sched.steps:
        has_prefill = any(p.startswith("prefill") for p in s["phases"].values())
        times[s["step"]] = (model["c"] + model["b"] * s["tokens"]
                            if has_prefill else model["decode_ms"])
    long_ttft, started = None, False
    for s in sched.steps:
        if "LONG" in s["scheduled"]:
            started = True
        if started:
            long_ttft = (long_ttft or 0.0) + times[s["step"]]
            if "LONG" in s["produced"]:
                break
    emit = {}
    for s in sched.steps:
        for rid in s["produced"]:
            emit.setdefault(rid, []).append(s["step"])
    gaps = [sum(times[t] for t in range(ss[i] + 1, ss[i + 1] + 1))
            for rid, ss in emit.items() if rid != "LONG"
            for i in range(len(ss) - 1)]
    return dict(long_ttft_ms=long_ttft,
                decode_max_gap_ms=max(gaps) if gaps else None,
                n_steps=len(sched.steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/crater/5.3/budget-20260921")
    ap.add_argument("--out", default="results/local/5.3/schedule-prediction")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.results, "chunked_budget.json")) as f:
        blob = json.load(f)
    steps = load_steps(args.results)
    model = fit_two_regime(steps)

    worst = []
    for budget, runs in blob["reps"].items():
        for r in runs:
            worst.append((float(max(r["requests"]["LONG"]["prefill_chunk_sizes"])),
                          r["slowest_step_ms"]))
    cw, bw = fit_linear(worst)

    out = ["step 时间模型（Qwen3-1.7B，8 条 decode + 一条 8192 prompt，"
           f"{len(steps)} 个 step 样本）", ""]
    out.append(f"  纯 decode 步（n={model['decode_n']}）：中位 {model['decode_ms']:.2f} ms")
    out.append(f"  含 prefill 的步（n={model['prefill_n']}）：t = {model['c']:.2f} + "
               f"{model['b']:.5f} × tokens（ms）")
    out.append(f"  只用最慢步的对照拟合：t = {cw:.2f} + {bw:.5f} × tokens —— "
               f"截距被抬高到 {cw / model['decode_ms']:.1f} 倍，因为样本里没有 decode 步。")
    out.append("")

    buckets = collections.defaultdict(list)
    for s in steps:
        if s["has_prefill"]:
            buckets["prefill"].append(s["wall_ms"] - (model["c"] + model["b"] * s["tokens"]))
        else:
            buckets["decode"].append(s["wall_ms"] - model["decode_ms"])
    out.append("  残差（ms）：纯 decode 步 中位 "
               f"{statistics.median(buckets['decode']):+.2f}，极差 "
               f"[{min(buckets['decode']):+.2f}, {max(buckets['decode']):+.2f}]；"
               f"含 prefill 步 中位 {statistics.median(buckets['prefill']):+.2f}，极差 "
               f"[{min(buckets['prefill']):+.2f}, {max(buckets['prefill']):+.2f}]")
    out.append("")

    out.append("  用这套模型预测 mini 计数轨迹的结果（含 prefill 步用拟合式，"
               "纯 decode 步用中位数）：")
    out.append(f"  {'预算':>6}{'预测 LONG TTFT ms':>18}{'实测 ms':>9}{'偏差':>9}"
               f"{'预测 decode 最大间隔 ms':>24}{'实测 ms':>9}{'偏差':>9}")
    pred_rows = {}
    for budget in sorted(blob["reps"], key=int):
        b = int(budget)
        sim = simulate(model, b)
        runs = blob["reps"][budget]
        m_ttft = statistics.median(x["requests"]["LONG"]["eng_ttft_ms"] for x in runs)
        m_gap = statistics.median(
            max(q["tpot_max_ms"] for k, q in x["requests"].items() if k != "LONG")
            for x in runs)
        out.append(f"  {b:>6}{sim['long_ttft_ms']:>18.1f}{m_ttft:>9.1f}"
                   f"{sim['long_ttft_ms'] / m_ttft - 1:>+8.0%}"
                   f"{sim['decode_max_gap_ms']:>24.1f}{m_gap:>9.1f}"
                   f"{sim['decode_max_gap_ms'] / m_gap - 1:>+8.0%}")
        pred_rows[b] = dict(pred_ttft=sim["long_ttft_ms"], meas_ttft=m_ttft,
                            pred_gap=sim["decode_max_gap_ms"], meas_gap=m_gap)
    out.append("")

    # 实测曲线作为"真值"：5 个预算档的 (分块, 最慢 step) 折线插值
    truth_pts = sorted((r["chunk"], r["step_ms"]) for r in
                       (dict(chunk=statistics.median([max(x["requests"]["LONG"]["prefill_chunk_sizes"])
                                                      for x in runs]),
                             step_ms=statistics.median([x["slowest_step_ms"] for x in runs]))
                        for runs in blob["reps"].values()))

    def truth_ms(chunk: float) -> float:
        if chunk <= truth_pts[0][0]:
            return truth_pts[0][1]
        for (x0, y0), (x1, y1) in zip(truth_pts, truth_pts[1:]):
            if chunk <= x1:
                return y0 + (chunk - x0) / (x1 - x0) * (y1 - y0)
        x0, y0 = truth_pts[-2]
        x1, y1 = truth_pts[-1]
        return y1 + (chunk - x1) / (x1 - x0) * (y1 - y0)

    out.append(f"预测误差注入：控制器用估计的斜率选分块，目标是预测单步 ≈ {TARGET_STEP_MS} ms。")
    out.append("  控制器模型是 5 点最小二乘 t = "
               f"{cw:.2f} + {bw:.5f} × n；真实单步由实测折线插值给出。")
    out.append(f"  {'斜率误差':>8}{'控制器用的 b':>14}{'选出的分块':>11}"
               f"{'真实单步 ms':>12}{'相对目标':>10}{'长请求分块数':>13}")
    inj = {}
    for tag, factor in (("-50%", 0.5), ("-25%", 0.75), ("0", 1.0),
                        ("+25%", 1.25), ("+50%", 1.5)):
        b_ctl = bw * factor
        chunk = max(1.0, (TARGET_STEP_MS - cw) / b_ctl)
        actual = truth_ms(chunk)
        out.append(f"  {tag:>8}{b_ctl:>14.5f}{chunk:>11.0f}{actual:>12.1f}"
                   f"{actual / TARGET_STEP_MS:>9.2f}×{8192 / chunk:>13.1f}")
        inj[tag] = dict(b_ctl=b_ctl, chunk=chunk, actual_ms=actual)
    out.append("")
    out.append(f"  斜率低估 50%：控制器以为 686 token 只要 15 ms，实测 {inj['-50%']['actual_ms']:.1f} ms"
               f"（{inj['-50%']['actual_ms'] / TARGET_STEP_MS:.2f} 倍），decode 的停顿随之拉长；")
    out.append(f"  高估 50%：单步只有 {inj['+50%']['actual_ms']:.1f} ms，看起来更安全，"
               f"但分块被切到 {inj['+50%']['chunk']:.0f} token，长请求要多等 "
               f"{8192 / inj['+50%']['chunk']:.0f} 个 step。")
    out.append("  预测误差的代价是双向的：低估伤尾延迟，高估伤长请求的 TTFT。")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "schedule_prediction.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "schedule_prediction.json"), "w") as f:
        json.dump(dict(model=model, worst_only=dict(c=cw, b=bw),
                       prediction=pred_rows, injection=inj), f, indent=1)
    print(f"\n写入 {args.out}/schedule_prediction.txt")


if __name__ == "__main__":
    main()
