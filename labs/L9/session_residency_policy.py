#!/usr/bin/env python3
"""L9.3 任务 B：会话驻留、换出与重算的成本模型，以及从实测工具时延分布选 TTL。

三件事分开：

* ``probe``  —— 在引擎上量「重算前缀」与「命中前缀」两条路径的单请求时间（长度 512/4096/8192，
  缓存关与缓存开各起一个服务），得到成本模型要用的真实输入。
* ``derive`` —— 用实测输入做推导：保留的字节×等待时间、换出/取回字节与时间、重算时间，以及
  「默认容量驱逐 / 重算 / 有界驻留」三种策略在工具间隔 0/1/10/60 s 与上下文 512/4096/8192 上的代价。
* ``ttl``    —— 从 9.1 采集到的真实工具时延分布选 TTL，给出各候选 TTL 的期望代价，并检查
  工具时延被 ±50% 扰动后容量上界是否仍然成立。

口径说明（避免把三个不同的量混起来）：

* **字节×等待时间**（byte·s）是保留一段 KV 的「占用积分」，与具体硬件无关；
* **换出/取回**是 2×B 的搬运，时间由实测带宽决定；
* **重算**是重新 prefill 这段前缀，时间由实测得到，与当前排队状况有关（本 lab 只报单请求时间）。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KV_BYTES_PER_TOKEN = 147_456      # Qwen3-4B：2 × 36 层 × 8 KV 头 × 128 × 2（9.4 已核对）
SWAP_BW_GBPS = 36.0               # 5.8 实测的换出聚合带宽
DEFAULT_BUDGET_GIB = 12.0         # 本 lab 假定的 KV 预算（≈0.45×32 GiB 减权重后的可用池）


# --------------------------------------------------------------------------------------
# probe：引擎侧的重算与命中对照
# --------------------------------------------------------------------------------------

def _filler_prompt(target_tokens: int) -> str:
    """用重复词元构造接近目标 token 数的提示（按 ~1 token/词 粗估，实测以 usage 为准）。"""
    unit = "alpha beta gamma delta epsilon zeta eta theta "
    return ("请阅读以下文本并只回复 OK。\n" + unit * max(1, target_tokens // 8))[: target_tokens * 5]


async def cmd_probe_async(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    rows = []
    for target in [int(x) for x in args.lengths.split(",")]:
        prompt = _filler_prompt(target)
        for rep in range(args.repeats):
            t0 = time.perf_counter()
            ttft = None
            prompt_tokens = cached = None
            stream = await client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=4, temperature=0.0, stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    d = getattr(chunk.usage, "prompt_tokens_details", None)
                    if d is not None:
                        cached = getattr(d, "cached_tokens", None)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if (getattr(delta, "content", None) or getattr(delta, "reasoning", None)
                        or getattr(delta, "tool_calls", None)):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
            rows.append({
                "mode": args.mode, "target_tokens": target, "repeat": rep,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "uncached_tokens": (None if prompt_tokens is None or cached is None
                                    else prompt_tokens - cached),
                "ttft_ms": round(ttft, 3) if ttft is not None else None,
                "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            })
            print(f"[probe] mode={args.mode} target={target} rep={rep} "
                  f"prompt={prompt_tokens} cached={cached} ttft={rows[-1]['ttft_ms']}", flush=True)
    await client.close()
    report = {"config": {"base_url": args.base_url, "model": args.model, "mode": args.mode,
                         "lengths": args.lengths, "repeats": args.repeats},
              "rows": rows,
              "note": ("mode=cache_off 的服务用于量重算；mode=cache_on 的服务用于量命中。"
                       "两者是不同进程，不能直接比较绝对值以外的东西")}
    (out / f"probe_{args.mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    return 0


def cmd_probe(args) -> int:
    import asyncio
    return asyncio.run(cmd_probe_async(args))


# --------------------------------------------------------------------------------------
# derive：三种策略的代价
# --------------------------------------------------------------------------------------

def _p50(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[len(vals) // 2]


def load_tool_wait_spans(spans_path: pathlib.Path) -> list[tuple[float, float]]:
    """从 9.1 的 spans 里取每个节点的工具等待区间 (start_s, wait_s)。"""
    starts: dict[tuple[str, str], float] = {}
    out: list[tuple[float, float]] = []
    for line in open(spans_path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["session_id"], r.get("node_id") or "")
        if r["span"] == "tool_start":
            starts[key] = min(starts.get(key, float("inf")), r["t_ms"])
        elif r["span"] == "tool_end" and key in starts:
            st = starts.pop(key)
            out.append((st / 1000.0, (r["t_ms"] - st) / 1000.0))
    return out


def max_concurrent(intervals: list[tuple[float, float]], scale: float = 1.0) -> int:
    """工具等待区间的最大重叠数（等待时长按 scale 缩放）。"""
    events = []
    for start, wait in intervals:
        w = wait * scale
        events.append((start, 1))
        events.append((start + w, -1))
    events.sort(key=lambda e: (e[0], -e[1]))   # 同一时刻先加后减（闭区间口径）
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def load_tool_waits(spans_path: pathlib.Path) -> list[float]:
    """从 9.1 的 spans 里取每个节点的工具占用（秒）。"""
    starts: dict[tuple[str, str], list[float]] = {}
    ends: dict[tuple[str, str], list[float]] = {}
    for line in open(spans_path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["session_id"], r.get("node_id") or "")
        if r["span"] == "tool_start":
            starts.setdefault(key, []).append(r["t_ms"])
        elif r["span"] == "tool_end":
            ends.setdefault(key, []).append(r["t_ms"])
    waits = []
    for key, ss in starts.items():
        ee = ends.get(key, [])
        if ee:
            waits.append((max(ee) - min(ss)) / 1000.0)
    return waits


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def cmd_derive(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kv_bpt = args.kv_bytes_per_token
    budget_bytes = args.budget_gib * (1 << 30)
    bw = args.swap_bw_gbps * 1e9

    # 重算时间：优先用 probe 实测（cache_off 的 TTFT 减命中态的固定开销），否则用给出的速率
    prefill_ms: dict[int, float] = {}
    probe_dir = pathlib.Path(args.probe_dir) if args.probe_dir else None
    if probe_dir and (probe_dir / "probe_cache_off.json").exists():
        rows = json.load(open(probe_dir / "probe_cache_off.json", encoding="utf-8"))["rows"]
        by_len: dict[int, list[float]] = {}
        for r in rows:
            by_len.setdefault(r["target_tokens"], []).append(r["uncached_tokens"] or 0)
        for r in rows:
            by_len.setdefault(r["target_tokens"], [])
        per_len: dict[int, list[float]] = {}
        for r in rows:
            per_len.setdefault(r["target_tokens"], []).append(r["ttft_ms"])
        # 用每个目标长度下「prompt token 最多」的那一轮作为该长度的重算时间
        for target, rs in per_len.items():
            richest = max((r for r in rows if r["target_tokens"] == target),
                          key=lambda r: r["prompt_tokens"] or 0)
            prefill_ms[target] = richest["ttft_ms"]
    if not prefill_ms:
        # 没有实测时按声明的速率外推，并在报告里标出来
        rate_tok_s = args.prefill_tokens_per_s
        for L in [int(x) for x in args.lengths.split(",")]:
            prefill_ms[L] = L / rate_tok_s * 1000.0

    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        B = kv_bpt * L
        recompute_ms = prefill_ms.get(L) or (L / args.prefill_tokens_per_s * 1000.0)
        swap_ms = 2 * B / bw * 1000.0
        for gap in [float(x) for x in args.gaps.split(",")]:
            retained_byte_s = B * gap
            # 保留占用的「会话槽·秒」：把占用积分折算成预算被占满的等效时间
            slot_seconds = retained_byte_s / budget_bytes
            rows.append({
                "length": L, "gap_s": gap,
                "kv_bytes": B, "kv_MiB": round(B / (1 << 20), 3),
                "retained_byte_s": round(retained_byte_s, 3),
                "budget_equivalent_session_seconds": round(slot_seconds, 6),
                "recompute_ms": round(recompute_ms, 3),
                "swap_ms": round(swap_ms, 4),
                "cheapest": min(("retain", "swap", "recompute"),
                                key=lambda k: {"retain": slot_seconds * budget_bytes / bw,
                                               "swap": swap_ms / 1000.0,
                                               "recompute": recompute_ms / 1000.0}[k]),
                "swap_vs_recompute": round(swap_ms / max(1e-9, recompute_ms), 4),
            })
    report = {
        "inputs": {"kv_bytes_per_token": kv_bpt, "budget_gib": args.budget_gib,
                   "swap_bw_gbps": args.swap_bw_gbps, "prefill_ms": prefill_ms,
                   "prefill_source": ("probe" if probe_dir and (probe_dir / "probe_cache_off.json").exists()
                                      else f"assumed {args.prefill_tokens_per_s} tok/s")},
        "rows": rows,
        "note": ("retain 的代价用「占用积分 ÷ 预算 ÷ 带宽」折算成等效时间，只能跨配置比较量级；"
                 "swap 是 2×B 的搬运；recompute 是单请求实测/假定值，未含排队"),
    }
    (out / "residency_derive.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps({"inputs": report["inputs"]}, ensure_ascii=False, indent=1))
    print(f"{'L':>6}{'gap_s':>7}{'kv_MiB':>10}{'retain(byte·s)':>16}{'slots·s':>10}{'recompute_ms':>14}{'swap_ms':>10}{'swap/recompute':>16}{'cheapest':>10}")
    for r in rows:
        print(f"{r['length']:>6}{r['gap_s']:>7}{r['kv_MiB']:>10}{r['retained_byte_s']:>16}{r['budget_equivalent_session_seconds']:>10}{r['recompute_ms']:>14}{r['swap_ms']:>10}{r['swap_vs_recompute']:>16}{r['cheapest']:>10}")
    return 0


def cmd_ttl(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spans_path = pathlib.Path(args.spans)
    waits = load_tool_waits(spans_path)
    intervals = load_tool_wait_spans(spans_path)
    if not waits:
        raise SystemExit("no tool waits found in spans")
    L = args.length
    B = args.kv_bytes_per_token * L
    budget_bytes = args.budget_gib * (1 << 30)
    recompute_ms = args.recompute_ms
    # 候选 TTL：0（总是丢）、实测分布的分位数、以及「永不丢」
    quants = [0.5, 0.75, 0.9, 0.95, 0.99]
    candidates = [0.0] + [_quantile(waits, q) for q in quants] + [None]

    def cost(ttl: float | None, scale: float = 1.0) -> dict:
        retained = 0.0
        dropped = 0
        for w in waits:
            ws = w * scale
            if ttl is None or ws <= ttl:
                retained += B * ws
            else:
                dropped += 1
        recompute_s = dropped * recompute_ms / 1000.0
        return {
            "ttl_s": ttl, "retained_byte_s": round(retained, 3),
            "retained_sessions": len(waits) - dropped, "dropped_sessions": dropped,
            "recompute_total_s": round(recompute_s, 4),
            "peak_concurrent_retained": None,
            "occupancy_seconds": round(retained / budget_bytes, 6),
        "capacity_ok": retained / budget_bytes <= args.max_occupancy_seconds,
        }

    base = [("ttl=" + ("inf" if t is None else f"{t:.3f}"), cost(t)) for t in candidates]
    # 扰动：工具时延整体 ×0.5 / ×1.5，检查容量上界
    perturbed = []
    for scale in (0.5, 1.5):
        for t in candidates:
            perturbed.append({"scale": scale, **cost(t, scale)})
    fit = int(budget_bytes // B) if B > 0 else 0
    conc_now = max_concurrent(intervals, 1.0)
    conc_15 = max_concurrent(intervals, 1.5)
    report = {
        "spans": str(args.spans),
        "capacity": {
            "kv_bytes_per_session": B,
            "sessions_fit_in_budget": fit,
            "max_concurrent_tool_waits_observed": conc_now,
            "max_concurrent_tool_waits_scaled_1.5": conc_15,
            "fits_now": conc_now <= fit,
            "fits_after_perturbation": conc_15 <= fit,
            "note": ("驻留的容量约束是**同时在等工具的会话数**，不是累计占用积分："
                     "并发等待数 × 单会话 KV 必须落在预算内；工具变慢会让等待区间拉长、重叠数上升"),
        },
        "tool_wait_stats": {
            "n": len(waits),
            "p50_s": round(_quantile(waits, 0.5), 4),
            "p90_s": round(_quantile(waits, 0.9), 4),
            "p95_s": round(_quantile(waits, 0.95), 4),
            "p99_s": round(_quantile(waits, 0.99), 4),
            "max_s": round(max(waits), 4),
            "mean_s": round(statistics.fmean(waits), 4),
        },
        "config": {"length": L, "kv_bytes": B, "budget_gib": args.budget_gib,
                   "recompute_ms": recompute_ms,
                   "max_occupancy_seconds": args.max_occupancy_seconds},
        "candidates": [{"label": lb, **c} for lb, c in base],
        "perturbed": perturbed,
    }
    # 选 TTL：在容量上界内 retained_byte_s 最大者（即尽量少丢，但不超过预算）
    feasible = [c for lb, c in base] if conc_now <= fit else []
    report["chosen"] = max(feasible, key=lambda c: c["retained_byte_s"]) if feasible else None
    worst = max((p["retained_byte_s"] for p in perturbed), default=0.0)
    report["capacity_upper_bound"] = {
        "nominal_max_retained_byte_s": max(c["retained_byte_s"] for lb, c in base),
        "perturbed_max_retained_byte_s": worst,
        "conservative_reservation_factor": round(
            worst / max(1e-9, max(c["retained_byte_s"] for lb, c in base)), 3),
        "note": "工具时延 ±50% 扰动后占用积分同步变化，容量预留要按最坏情况放大",
    }
    (out / "residency_ttl.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "perturbed"}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.3 会话驻留成本模型与 TTL 选择")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--mode", required=True, choices=["cache_off", "cache_on"])
    p.add_argument("--lengths", default="512,4096,8192")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("derive")
    p.add_argument("--out", required=True)
    p.add_argument("--probe-dir", default=None)
    p.add_argument("--lengths", default="512,4096,8192")
    p.add_argument("--gaps", default="0,1,10,60")
    p.add_argument("--kv-bytes-per-token", type=int, default=KV_BYTES_PER_TOKEN)
    p.add_argument("--budget-gib", type=float, default=DEFAULT_BUDGET_GIB)
    p.add_argument("--swap-bw-gbps", type=float, default=SWAP_BW_GBPS)
    p.add_argument("--prefill-tokens-per-s", type=float, default=6000.0)
    p.set_defaults(func=cmd_derive)

    p = sub.add_parser("ttl")
    p.add_argument("--spans", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--kv-bytes-per-token", type=int, default=KV_BYTES_PER_TOKEN)
    p.add_argument("--budget-gib", type=float, default=DEFAULT_BUDGET_GIB)
    p.add_argument("--recompute-ms", type=float, default=0.0)
    p.add_argument("--max-occupancy-seconds", type=float, default=3600.0)
    p.set_defaults(func=cmd_ttl)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
