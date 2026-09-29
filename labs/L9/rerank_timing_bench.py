#!/usr/bin/env python3
"""L9.6 任务 B 的缺口：把**重排自身耗时**单独计时，并用同卡生成量化竞争。

9.6 原来的端到端表把重排与生成放在同一张卡上，重排组的生成耗时（2.4–2.9 s）里混进了
"重排正在跑"的那段时间，所以**重排自身的耗时没有单独的数**。本脚本补上这一段：

* ``isolated``：只有 reranker 在卡上，按 top-k ∈ {5,20,100} × batch ∈ {1,8,32} 逐档测
  每对 (query, passage) 的耗时、每问总耗时与 pairs/s，并记录显存峰值；
* ``contended``：同一台机器上再起一个 vLLM 引擎并持续生成，重测同样的档位，
  用两档之比给出"同卡竞争"把重排抬高了多少。

判分器与打分方式与 9.6 正文一致：把 `rag_pipeline.rerank_pairs` 原样 import 进来，
不另写一份实现。所有数字都落盘，含每档的逐次测量值。

用法::

    python labs/L9/rerank_timing_bench.py run --out out/9.6/rerank \\
        --rerank-model Qwen/Qwen3-Reranker-0.6B --topks 5,20,100 --batches 1,8,32
    # contended 档需先起好引擎
    python labs/L9/rerank_timing_bench.py run --out out/9.6/rerank --base-url http://127.0.0.1:8061/v1 \\
        --phase contended --concurrency 4 --gen-max-tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rag_pipeline as R  # noqa: E402


class Args(dict):
    """`rag_pipeline.rerank_pairs` 通过属性访问参数，这里做个薄适配。"""

    def __getattr__(self, item):
        return self[item]


def make_pairs(n_pairs: int, seed: int = 0) -> list[tuple[str, str]]:
    import random

    rng = random.Random(seed)
    words = ["budget", "analysis", "report", "archive", "temperature", "pipeline", "kernel",
             "latency", "cache", "prefix", "token", "scheduler", "sandbox", "session"]
    pairs = []
    for i in range(n_pairs):
        q = " ".join(rng.sample(words, 4))
        p = " ".join(rng.choices(words, k=40)) + f" [doc-{i}]"
        pairs.append((q, p))
    return pairs


def gpu_mem_mb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return (float("nan"), float("nan"))
    return (round(torch.cuda.memory_allocated() / 2**20, 1),
            round(torch.cuda.max_memory_allocated() / 2**20, 1))


def engine_running(base_url: str | None) -> float | None:
    """读引擎当时的并发请求数，用来证明"重排测量期间生成确实在跑"。"""
    if not base_url:
        return None
    import httpx2

    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    try:
        with httpx2.Client(timeout=3.0) as cli:
            text = cli.get(root.rstrip("/") + "/metrics").text
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                return float(line.rsplit(" ", 1)[1])
    except Exception:  # noqa: BLE001
        return None
    return None


def measure(args, pairs: list[tuple[str, str]], batch: int, repeats: int) -> dict:
    import torch

    a = Args(rerank_model=args.rerank_model, rerank_batch=batch)
    per_repeat = []
    running_seen = []
    for _ in range(repeats):
        running_seen.append(engine_running(args.base_url))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        scores = R.rerank_pairs(a, pairs)
        dt = (time.perf_counter() - t0) * 1000.0
        per_repeat.append(round(dt, 3))
        assert len(scores) == len(pairs)
    best = min(per_repeat)
    alloc, peak = gpu_mem_mb()
    return {"pairs": len(pairs), "batch": batch, "repeats": repeats,
            "engine_running_seen": running_seen,
            "engine_running_max": (max([r for r in running_seen if r is not None])
                                   if any(r is not None for r in running_seen) else None),
            "total_ms_best": round(best, 3), "per_repeat_ms": per_repeat,
            "ms_per_pair": round(best / len(pairs), 4),
            "pairs_per_s": round(len(pairs) / (best / 1000.0), 2),
            "gpu_alloc_mb": alloc, "gpu_peak_mb": peak}


async def generate_load(base_url: str, model: str, concurrency: int, max_tokens: int,
                        stop: asyncio.Event) -> dict:
    """后台生成负载：让同卡引擎保持忙，用于 contended 档。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=300.0)
    done = {"requests": 0, "tokens": 0, "errors": 0}

    async def one() -> None:
        while not stop.is_set():
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "写一段关于推理服务调度的文字。"}],
                    max_tokens=max_tokens, temperature=0.2,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                done["requests"] += 1
                done["tokens"] += (resp.usage.completion_tokens if resp.usage else 0)
            except Exception:  # noqa: BLE001
                done["errors"] += 1
                await asyncio.sleep(0.2)

    await asyncio.gather(*[one() for _ in range(concurrency)])
    await client.close()
    return done


async def cmd_run_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topks = [int(x) for x in args.topks.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    rows = []
    load_stats = None
    stop = asyncio.Event()
    load_task = None
    if args.phase == "contended":
        if not args.base_url:
            raise SystemExit("contended 档需要 --base-url 指向同卡引擎")
        load_task = asyncio.create_task(generate_load(args.base_url, args.gen_model,
                                                      args.concurrency, args.gen_max_tokens, stop))
        await asyncio.sleep(args.warmup_s)   # 让引擎先跑起来

    # 先预热 reranker（首次加载含权重搬运，不计入测量）
    warm = make_pairs(8, seed=99)
    R.rerank_pairs(Args(rerank_model=args.rerank_model, rerank_batch=8), warm)

    for topk in topks:
        pairs = make_pairs(topk * args.repeats)
        for batch in batches:
            row = measure(args, pairs, batch, args.repeats)
            row["topk"] = topk
            row["phase"] = args.phase
            rows.append(row)
            print(f"[{args.phase}] topk={topk:<4} batch={batch:<3} "
                  f"ms/pair={row['ms_per_pair']:<8} pairs/s={row['pairs_per_s']:<9} "
                  f"peak_mb={row['gpu_peak_mb']}", flush=True)

    if load_task is not None:
        stop.set()
        load_stats = await load_task
    report = {"config": {k: v for k, v in vars(args).items()
                         if isinstance(v, (str, int, float, bool, type(None)))},
              "phase": args.phase, "rows": rows, "generation_load": load_stats,
              "note": ("重排打分方式与 9.6 正文一致（复用 rag_pipeline.rerank_pairs，Qwen3-Reranker-0.6B 的 "
                       "yes/no logits）；isolated 档引擎未加载，contended 档引擎持续生成。"
                       "每档取 repeats 次里的最小值，避免把首次分配与抖动算进去")}
    (out / f"rerank_timing_{args.phase}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


async def cmd_load_async(args) -> int:
    """独立进程的生成负载：必须在**另一个进程**里跑。

    同一进程里用 asyncio 起负载是无效的——重排前向是同步阻塞调用，会把事件循环占住，
    负载请求根本发不出去（实测此时引擎的 `num_requests_running` 全程为 0）。
    """
    stop = asyncio.Event()

    async def stopper() -> None:
        await asyncio.sleep(args.seconds)
        stop.set()

    t0 = time.perf_counter()
    done = await asyncio.gather(generate_load(args.base_url, args.gen_model, args.concurrency,
                                             args.gen_max_tokens, stop),
                               stopper())
    stats = done[0]
    wall = time.perf_counter() - t0
    stats["wall_s"] = round(wall, 3)
    stats["requests_per_s"] = round(stats["requests"] / wall, 3)
    stats["tokens_per_s"] = round(stats["tokens"] / wall, 1)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.6 重排自身耗时与同卡竞争")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", choices=["isolated", "contended"], default="isolated")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--topks", default="5,20,100")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--base-url", default=None)
    p.add_argument("--gen-model", default="Qwen/Qwen3-4B")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--gen-max-tokens", type=int, default=256)
    p.add_argument("--warmup-s", type=float, default=3.0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_run_async(a)))

    p = sub.add_parser("load", help="独立进程的生成负载（用于真竞争档）")
    p.add_argument("--base-url", required=True)
    p.add_argument("--gen-model", default="Qwen/Qwen3-4B")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--gen-max-tokens", type=int, default=512)
    p.add_argument("--seconds", type=float, default=120.0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_load_async(a)))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
