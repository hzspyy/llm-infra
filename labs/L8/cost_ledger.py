#!/usr/bin/env python3
"""labs/L8/cost_ledger.py - 8.5-C: 把实测能量换算成"每千有效输出 token 成本".

输入是 8.5-B 的实测产物 (`energy_summary.json` + 逐请求 `records_*.jsonl`), 输出是:

  * `price_assumptions.json`  —— **假设**, 与读数分开存放, 每个价格带币种/单位/来源/日期;
  * `cost_table.json`         —— 同 SLO 下每个配置的每千有效输出 token 成本;
  * 敏感度表                  —— GPU 小时价 × 电价的网格。

有效输出 token 的定义与 8.3 一致: 在冻结 SLO 下达标的请求, 其输出 token 之和。分母
是**计划到达**的请求, 所以超时/拒绝/不达标的那部分输出不计入分子。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.request_metrics import TEACHING_SLOS, attained, read_jsonl  # noqa: E402

# 价格是假设, 不是读数。来源与日期一并记下, 未取得可机器读取的报价时如实标注。
DEFAULT_PRICES = {
    "gpu_hour": {
        "currency": "USD", "unit": "per GPU-hour", "value": 0.80,
        "source": "results/worldvln/8.5/20260914-refs/manifest.txt 记录的公开报价页；"
                  "页面为交互式渲染, 未取到可机器读取的价目, 故数值为**假设**",
        "retrieved_at": "2026-09-14", "kind": "assumption",
    },
    "electricity": {
        "currency": "USD", "unit": "per kWh", "value": 0.12,
        "source": "results/worldvln/8.5/20260914-refs/manifest.txt 记录的公开电价页；"
                  "同样未取到可机器读取的数值, 故为**假设**",
        "retrieved_at": "2026-09-14", "kind": "assumption",
    },
    "scope": {
        "measured": "单张 GPU 的 NVML power.draw 积分 (J)",
        "not_measured": "主机 CPU/内存/网卡/NFS/机房 PUE 的能耗; 因此本表不能当整机或机房成本",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-hour-usd", type=float, default=None)
    ap.add_argument("--kwh-usd", type=float, default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run
    summary = json.loads((run / "energy_summary.json").read_text(encoding="utf-8"))
    prices = json.loads(json.dumps(DEFAULT_PRICES))
    if args.gpu_hour_usd is not None:
        prices["gpu_hour"]["value"] = args.gpu_hour_usd
        prices["gpu_hour"]["kind"] = "override"
    if args.kwh_usd is not None:
        prices["electricity"]["value"] = args.kwh_usd
        prices["electricity"]["kind"] = "override"
    (out / "price_assumptions.json").write_text(
        json.dumps(prices, ensure_ascii=False, indent=2), encoding="utf-8")

    gpu_hour = prices["gpu_hour"]["value"]
    kwh = prices["electricity"]["value"]

    rows: List[Dict[str, Any]] = []
    for case in summary["ledger"]:
        name = case["case"]
        recs_path = run / f"records_{name}.jsonl"
        if not recs_path.exists():
            continue
        recs = read_jsonl(str(recs_path))
        dur = case["window_s"] or 0.0
        attained_tokens: Dict[str, int] = {}
        attained_reqs: Dict[str, int] = {}
        for slo_name, slo in TEACHING_SLOS.items():
            ok = [r for r in recs if attained(r, slo)]
            attained_reqs[slo_name] = len(ok)
            attained_tokens[slo_name] = sum(r.num_tokens for r in ok)
        inc_j = case["incremental_energy_j"] or 0.0
        # GPU 时间成本摊到本配置的窗口上; 能量成本用同一段能量
        gpu_s = dur
        cost_gpu = gpu_s / 3600.0 * gpu_hour
        cost_energy = inc_j / 3.6e6 * kwh
        row = {
            "case": name, "kind": case["kind"],
            "concurrency": case.get("concurrency"), "rate_qps": case.get("rate_qps"),
            "window_s": dur, "incremental_energy_j": inc_j,
            "tokens_total": case["tokens"],
            "gpu_seconds": gpu_s,
            "cost_gpu_usd": cost_gpu, "cost_energy_usd": cost_energy,
            "cost_total_usd": cost_gpu + cost_energy,
            "attained_requests": attained_reqs,
            "attained_tokens": attained_tokens,
            "ttft_p50": case.get("ttft_p50"), "ttft_p99": case.get("ttft_p99"),
            "tpot_p50": case.get("tpot_p50"),
        }
        for slo_name, tok in attained_tokens.items():
            row[f"usd_per_1k_attained_tokens[{slo_name}]"] = (
                (cost_gpu + cost_energy) / tok * 1000 if tok else None)
            row[f"j_per_1k_attained_tokens[{slo_name}]"] = (
                inc_j / tok * 1000 if tok else None)
        rows.append(row)

    # 敏感度: GPU 小时价 x 电价, 取 c32 在 SLO-1s/50ms 下的每千有效 token 成本
    ref = next((r for r in rows if r["case"] == "closed_c32"), rows[-1] if rows else None)
    sens: List[Dict[str, Any]] = []
    if ref:
        tok = ref["attained_tokens"].get("SLO-1s/50ms") or 0
        for g in (0.4, 0.8, 1.6):
            for e in (0.06, 0.12, 0.24):
                c = (ref["gpu_seconds"] / 3600.0 * g) + (ref["incremental_energy_j"] / 3.6e6 * e)
                sens.append({"gpu_hour_usd": g, "kwh_usd": e,
                             "usd_per_1k_attained_tokens": c / tok * 1000 if tok else None,
                             "cost_gpu_usd": c, "attained_tokens": tok})

    (out / "cost_table.json").write_text(
        json.dumps({"prices": prices, "rows": rows, "sensitivity_closed_c32": sens},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    hdr = f"{'case':12s} {'incJ':>9s} {'tok':>7s} {'att_tok(1s/50ms)':>17s} {'J/1k_att':>9s} {'$/1k_att':>9s}"
    print(hdr)
    for r in rows:
        print(f"{r['case']:12s} {r['incremental_energy_j']:9.1f} {r['tokens_total']:7d} "
              f"{r['attained_tokens'].get('SLO-1s/50ms', 0):17d} "
              f"{(r.get('j_per_1k_attained_tokens[SLO-1s/50ms]') or 0):9.1f} "
              f"{(r.get('usd_per_1k_attained_tokens[SLO-1s/50ms]') or 0):9.5f}")
    print(f"\n-> {out/'cost_table.json'}")


if __name__ == "__main__":
    main()