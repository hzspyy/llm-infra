#!/usr/bin/env python3
"""labs/L8/routing_bench.py - 8.1-B/C: 真实双副本网关上的路由策略与 KV 目录实验.

两个模式:
    bench   固定到达清单跑三种策略, 记录每请求选择/队列/预测命中/真实命中与 TTFT。
            前缀长度与共享比例按单轴扫描 (不做笛卡尔积), 与 8.1-B 的验收口径一致。
    events  只跑 prefix_aware, 对比"精确事件目录"与"近似请求目录", 并注入事件丢失,
            用引擎自报的真实 cached_tokens 检查预测是否变成假阳性命中。

真实命中来自两处: 逐请求的 `usage.prompt_tokens_details.cached_tokens`, 以及
worker `/metrics` 的 `prefix_cache_hits_total` 前后差值 (总量对账)。

用法:
    python labs/L8/routing_bench.py bench \
        --workers w0=http://127.0.0.1:18000,w1=http://127.0.0.1:18001 \
        --events w0=tcp://127.0.0.1:15557,w1=tcp://127.0.0.1:15558 \
        --out-dir results/worldvln/8.1/<run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.gateway import Gateway, build_app  # noqa: E402
from labs.L8.load_generator import generate_arrivals, run_open_loop  # noqa: E402
from labs.L8.request_metrics import evaluate, write_jsonl  # noqa: E402

METRIC_RE = {
    "prefix_cache_hits_total": re.compile(
        rb"^vllm:prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+]+)$", re.M),
    "prefix_cache_queries_total": re.compile(
        rb"^vllm:prefix_cache_queries_total\{[^}]*\}\s+([0-9.eE+]+)$", re.M),
}


async def worker_metrics(url: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url.rstrip('/')}/metrics",
                             timeout=aiohttp.ClientTimeout(total=10)) as r:
                raw = await r.read()
        for name, rx in METRIC_RE.items():
            vals = [float(m) for m in rx.findall(raw)]
            out[name] = sum(vals) if vals else None
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def build_workload(n: int, shared_ratio: float, prefix_len: int, num_clusters: int = 16,
                   suffix_len: int = 32, output_len: int = 32, seed: int = 0,
                   hot_share: float = 0.6, vocab_size: int = 151936) -> List[Dict[str, Any]]:
    """固定请求清单: 一部分请求共享若干"业务前缀", 其余各自带唯一前缀。

    token id 必须落在模型词表内: 超出词表的 id 会让 vLLM 在**流中途**抛
    `Token id ... is out of vocabulary`, 响应只送出第一块就断开, 客户端会把它
    记成截断而不是错误。共享前缀与唯一前缀用互不相交的两个区间, 避免"唯一"前缀
    恰好等于某个共享前缀。
    """
    rng = random.Random(seed)
    lo, hi = 1000, min(vocab_size - 1000, 150000)
    mid = (lo + hi) // 2
    clusters = [[rng.randrange(lo, mid) for _ in range(prefix_len)]
                for _ in range(num_clusters)]
    reqs = []
    for i in range(n):
        if rng.random() < shared_ratio:
            c = 0 if rng.random() < hot_share else rng.randrange(1, num_clusters)
            prefix = list(clusters[c])
            tag = f"cluster-{c}"
        else:
            prefix = [rng.randrange(mid, hi) for _ in range(prefix_len)]
            tag = "unique"
        reqs.append({"id": f"req-{i:05d}",
                     "tokens": prefix + [rng.randrange(1000, 150000) for _ in range(suffix_len)],
                     "tag": tag, "output_len": output_len})
    return reqs


async def start_gateway(gw: Gateway, port: int) -> web.AppRunner:
    runner = web.AppRunner(build_app(gw), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


def _fmt(x: Any) -> str:
    return "None" if x is None else f"{x:.3f}"


def _count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


async def run_case(case: Dict[str, Any], workers: List[Tuple[str, str]], events: Dict[str, str],
                   port: int, out_dir: Path, rate: float, seed: int) -> Dict[str, Any]:
    policy = case["policy"]
    reqs = build_workload(case["n"], case["shared_ratio"], case["prefix_len"], seed=seed)
    duration = case["n"] / rate
    arrivals = generate_arrivals("poisson", rate, duration, seed=seed)
    ids = [r["tokens"] for r in reqs]
    if len(arrivals) != len(reqs):
        arrivals = [i * (duration / len(reqs)) for i in range(len(reqs))]

    gw = Gateway(workers, policy, block_size=case.get("block_size", 16),
                 queue_penalty=case.get("queue_penalty", 0.0),
                 event_endpoints=events,
                 drop_rate=case.get("drop_rate", 0.0),
                 delay_s=case.get("delay", 0.0),
                 duplicate=case.get("duplicate", False),
                 prediction_source=case.get("prediction_source", "events"))
    runner = await start_gateway(gw, port)
    await asyncio.sleep(1.0)

    before = {w: await worker_metrics(u) for w, u in workers}
    t0 = time.monotonic()
    records = await run_open_loop(
        f"http://127.0.0.1:{port}", arrivals, prompt_len=case["prefix_len"],
        output_len=reqs[0]["output_len"], backend="openai", model=case.get("model", ""),
        timeout_s=case.get("timeout", 120.0), prompt_ids=ids,
    )
    window = time.monotonic() - t0
    after = {w: await worker_metrics(u) for w, u in workers}
    await runner.cleanup()

    summary = evaluate(records, f"{policy}-{case['tag']}", window, reserved_events=arrivals)
    metrics_delta = {}
    for w, u in workers:
        b, a = before.get(w, {}), after.get(w, {})
        metrics_delta[w] = {
            k: (a.get(k) - b.get(k))
            if (a.get(k) is not None and b.get(k) is not None) else None
            for k in ("prefix_cache_hits_total", "prefix_cache_queries_total")
        }

    gw_by_id = {rec["request_id"]: rec for rec in gw.request_log}
    per_req = []
    fp = 0
    have_actual = 0
    for r in records:
        g = gw_by_id.get(r.request_id, {})
        pred = int(r.metadata.get("predicted_hit_tokens", 0) or 0)
        actual = g.get("cached_tokens")
        if actual is None:
            actual_v = None
        else:
            actual_v = actual
            have_actual += 1
            if pred > 0 and actual == 0:
                fp += 1
        per_req.append({
            "request_id": r.request_id, "worker": g.get("worker"),
            "predicted_hit_tokens": pred, "actual_cached_tokens": actual_v,
            "true_ttft_s": r.true_ttft_s, "observed_ttft_s": r.observed_ttft_s,
            "status": r.status,
        })

    result = {
        "case": case, "policy": policy, "seed": seed, "rate_qps": rate,
        "summary": summary.to_dict(),
        "metrics_delta": metrics_delta,
        "gateway_counters": gw.counters,
        "directory_stats": gw.directory.stats,
        "subscribers": {w: s.stats() for w, s in gw.subscribers.items()},
        "false_positive_predictions": fp,
        "requests_with_actual": have_actual,
        "worker_assignment": _count_by(per_req, "worker"),
    }
    tag = f"{case['tag']}-{policy}-s{seed}"
    write_jsonl(str(out_dir / f"records_{tag}.jsonl"), records)
    with (out_dir / f"per_request_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for x in per_req:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] planned={summary.planned_requests} counts={summary.counts} "
          f"trueTTFT p50={_fmt(summary.latency['true_ttft']['p50'])} "
          f"p99={_fmt(summary.latency['true_ttft']['p99'])} "
          f"hits={ {w: metrics_delta[w]['prefix_cache_hits_total'] for w, _ in workers} } "
          f"fp={fp}/{have_actual}", flush=True)
    return result


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "events"])
    ap.add_argument("--workers", required=True)
    ap.add_argument("--events", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--queue-penalty", type=float, default=0.0)
    ap.add_argument("--prefix-order", default="4096,128")
    ap.add_argument("--ratios", default="0.0,0.25,0.75")
    args = ap.parse_args()

    workers = [tuple(x.split("=", 1)) for x in args.workers.split(",") if x]
    events = dict(x.split("=", 1) for x in args.events.split(",") if x)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    if args.mode == "bench":
        for ratio in [float(x) for x in args.ratios.split(",")]:
            cases.append({"tag": f"ratio{int(ratio*100)}-len2048", "shared_ratio": ratio,
                          "prefix_len": 2048, "n": args.n})
        for L in [int(x) for x in args.prefix_order.split(",") if int(x) != 2048]:
            cases.append({"tag": f"ratio75-len{L}", "shared_ratio": 0.75,
                          "prefix_len": L, "n": args.n})
        cases = [{**c, "policy": p, "queue_penalty": args.queue_penalty}
                 for c in cases for p in ("round_robin", "shortest_queue", "prefix_aware")]
    else:
        for src in ("events", "approx"):
            for drop in (0.0, 0.2, 0.5):
                cases.append({"tag": f"src{src}-drop{int(drop*100)}", "shared_ratio": 0.75,
                              "prefix_len": 4096, "n": args.n, "policy": "prefix_aware",
                              "prediction_source": src, "drop_rate": drop,
                              "queue_penalty": args.queue_penalty})

    results = []
    for i, case in enumerate(cases):
        # 每个 case 用独立端口: 复用端口会在上一轮 runner 未完全释放时直接
        # OSError: address already in use, 让整轮实验静默中断。
        results.append(await run_case(case, workers, events, args.port + i, out,
                                      args.rate, args.seed))

    (out / f"{args.mode}_summary.json").write_text(
        json.dumps({"cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
