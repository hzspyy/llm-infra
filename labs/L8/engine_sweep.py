#!/usr/bin/env python3
"""labs/L8/engine_sweep.py - 8.3-B/C: 真实引擎上的到达率阶梯扫描与 SLO goodput.

基线口径 (在扫描任何档位之前冻结, 并写入 baseline.json):
  * 模型 Qwen3-1.7B, 输入固定 2048 token, 输出固定 128 token (`ignore_eos`)。
  * 关闭 prefix caching, 每次请求用互不相同的 token 序列, 使 prefill 成本可比。
  * 参考容量 R_ref = 闭环并发 32 跑满 `--baseline-window` 秒的**实际完成速率**
    (请求/秒)。这是一个"能做到多少"的实测值, 不是理论值。
  * 阶梯 = 0.3/0.6/0.9/1.1 × R_ref, 每档 3 个泊松种子, 每个窗口不少于 120 秒。

额外两组负载: 突发到达 (每 2s 投入 20 个请求) 与长短混合 (512/4096 token 交替),
同样 3 个种子。所有请求——包括拒绝、超时、截断和缺事件——都进 goodput 分母。

用法 (服务端已就绪):
    python labs/L8/engine_sweep.py --base-url http://127.0.0.1:8000 \
        --out-dir results/worldvln/8.3/<run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.load_generator import (  # noqa: E402
    generate_arrivals,
    run_closed_loop,
    run_open_loop,
)
from labs.L8.request_metrics import (  # noqa: E402
    TEACHING_SLOS,
    evaluate,
    write_jsonl,
)

SHORT_LEN = 512
LONG_LEN = 4096
PROMPT_POOL = 192


def build_prompts(tok, lengths: List[int], seed: int = 0) -> Dict[int, List[List[int]]]:
    """为每个长度构造一批互不相同的 token 序列 (固定 seed 可复现)。"""
    rng = random.Random(seed)
    vocab = int(getattr(tok, "vocab_size", 151936))
    lo, hi = 1000, min(vocab - 1000, 150000)
    pool: Dict[int, List[List[int]]] = {}
    for L in lengths:
        pool[L] = [[rng.randrange(lo, hi) for _ in range(L)] for _ in range(PROMPT_POOL)]
    return pool


async def wait_ready(base_url: str, timeout_s: float = 900.0) -> Dict[str, Any]:
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=10.0) as c:
        while time.monotonic() - t0 < timeout_s:
            try:
                r = await c.get(f"{base_url}/v1/models")
                if r.status_code == 200:
                    return {"ready_s": time.monotonic() - t0, "models": r.json()}
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout_s}s")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--baseline-window", type=float, default=60.0)
    ap.add_argument("--baseline-concurrency", type=int, default=32)
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = build_prompts(tok, [args.prompt_len, SHORT_LEN, LONG_LEN], seed=20260914)
    base_pool = pool[args.prompt_len]

    meta: Dict[str, Any] = {"run_id": out.name, "args": vars(args), "seeds": seeds,
                            "prompt_pool": PROMPT_POOL, "slos": {k: v.to_dict() for k, v in TEACHING_SLOS.items()}}
    meta["ready"] = await wait_ready(args.base_url)
    print(f"server ready in {meta['ready']['ready_s']:.1f}s", flush=True)

    # ---------------- 基线: 闭环饱和吞吐 ---------------------------------
    t0 = time.monotonic()
    rng = random.Random(99)
    n_total = 100000  # 由 wall-clock 截断, 见下
    baseline_records = await run_closed_loop(
        args.base_url, concurrency=args.baseline_concurrency, total_requests=n_total,
        prompt_len=args.prompt_len, output_len=args.output_len,
        model=args.model, timeout_s=args.timeout,
        prompt_ids=[base_pool[i % PROMPT_POOL] for i in range(n_total)],
        stop_after_s=args.baseline_window, t0_monotonic=t0,
    )
    baseline_win = time.monotonic() - t0
    base_summary = evaluate(baseline_records, "baseline", baseline_win)
    r_ref = base_summary.throughput["success_requests"] / baseline_win
    meta["baseline"] = {
        "concurrency": args.baseline_concurrency,
        "window_s": baseline_win,
        "completed": base_summary.throughput["success_requests"],
        "r_ref_qps": r_ref,
        "summary": base_summary.to_dict(),
    }
    write_jsonl(str(out / "baseline_records.jsonl"), baseline_records)
    (out / "baseline.json").write_text(json.dumps(meta["baseline"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline: {base_summary.throughput['success_requests']} req in {baseline_win:.1f}s "
          f"-> R_ref={r_ref:.3f} QPS", flush=True)

    # ---------------- 阶梯扫描 -------------------------------------------
    configs: List[Dict[str, Any]] = []
    for label, mult in (("0.3x", 0.3), ("0.6x", 0.6), ("0.9x", 0.9), ("1.1x", 1.1)):
        configs.append({"tag": label, "kind": "poisson", "rate": mult * r_ref})
    configs.append({"tag": "burst", "kind": "burst", "rate": 0.6 * r_ref})
    configs.append({"tag": "mixed", "kind": "mixed", "rate": 0.6 * r_ref})

    all_summaries: List[Dict[str, Any]] = []
    for cfg in configs:
        for seed in seeds:
            tag = f"{cfg['tag']}-s{seed}"
            if cfg["kind"] == "burst":
                arrivals = generate_arrivals("burst", cfg["rate"], args.window, seed=seed,
                                             burst_size=20, burst_gap_s=2.0)
            else:
                arrivals = generate_arrivals("poisson", cfg["rate"], args.window, seed=seed)

            if cfg["kind"] == "mixed":
                rng2 = random.Random(seed)
                ids = []
                plens = []
                for i in range(len(arrivals)):
                    if i % 2 == 0:
                        L = SHORT_LEN
                    else:
                        L = LONG_LEN
                    plens.append(L)
                    ids.append(pool[L][rng2.randrange(PROMPT_POOL)])
                records = await run_open_loop(
                    args.base_url, arrivals, prompt_len=args.prompt_len,
                    output_len=args.output_len, model=args.model, timeout_s=args.timeout,
                    prompt_ids=ids, prompt_lens=plens,
                )
            else:
                rng2 = random.Random(seed * 7 + 1)
                ids = [base_pool[rng2.randrange(PROMPT_POOL)] for _ in arrivals]
                records = await run_open_loop(
                    args.base_url, arrivals, prompt_len=args.prompt_len,
                    output_len=args.output_len, model=args.model, timeout_s=args.timeout,
                    prompt_ids=ids,
                )

            summary = evaluate(records, tag, args.window, reserved_events=arrivals)
            entry = {
                "config": cfg,
                "seed": seed,
                "rate_qps": cfg["rate"],
                "summary": summary.to_dict(),
            }
            all_summaries.append(entry)
            write_jsonl(str(out / f"records_{tag}.jsonl"), records)
            (out / f"summary_{tag}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            gp = summary.goodput["SLO-1s/50ms"]
            print(f"[{tag}] planned={summary.planned_requests} "
                  f"counts={summary.counts} "
                  f"trueTTFT p50={summary.latency['true_ttft']['p50']} "
                  f"p99={summary.latency['true_ttft']['p99']} "
                  f"goodput(1s/50ms)={gp['ratio']}", flush=True)

    (out / "sweep_summary.json").write_text(
        json.dumps({"meta": meta, "runs": all_summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nall done -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
