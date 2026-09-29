#!/usr/bin/env python3
"""L9.4 × 9.7：把预算策略接到任务级调度上，并用任务完成时间与每成功任务成本衡量。

9.4 已经量过「预算 → 质量」的曲线，9.7 已经量过「任务级优先级 → SLO 内任务吞吐」。本脚本
把两者接起来：同一批长短混合任务、同一个固定到达过程，唯一变量是**预算策略**（以及由预算
类别折算出的引擎优先级），看三件事怎么变：

* 任务完成时间（JCT p50/p95）；
* SLO 内成功任务数/秒（主指标）；
* 每成功任务成本（总生成 token / 成功任务数）。

三个策略：

1. ``uniform_long``        —— 所有任务都给长预算（2048），无优先级；
2. ``budget_aware``        —— 短任务 256、长任务 2048，无优先级；
3. ``budget_aware_priority`` —— 同上，但短任务 priority=0、长任务 priority=100，
   由引擎的 priority 调度器（`--scheduling-policy priority`）决定入队次序。

题目取自与 9.4 相同的 GSM8K 固定子集，评分用同一个 `score()`；短预算档会把思考吃满、
最终答案被截断，这类样本必须留在分母里（`answered` 与 `correct` 分开记）。

用法::

    /scratch/learn/envs/serve/bin/python labs/L9/budget_task_scheduling.py run \\
        --base-url http://127.0.0.1:8061/v1 --model Qwen/Qwen3-4B \\
        --out out/9.4/budget-scheduling/report.json --n 64 --rate 3.0
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

from reasoning_budget import load_questions, score  # noqa: E402

SHORT_BUDGET = 256
LONG_BUDGET = 2048


def build_tasks(n: int, dataset: str, seed: int, rate: float, slo_ms: float) -> list[dict]:
    """长短交替的固定任务清单；到达时间由索引与到达率给出（开环，不依赖完成）。"""
    questions = load_questions(dataset, n, seed)
    tasks: list[dict] = []
    for i, q in enumerate(questions):
        short = (i % 2 == 0)
        tasks.append({
            "task_id": q["qid"], "index": i, "prompt": q["prompt"], "gold": q.get("gold"),
            "budget_class": "short" if short else "long",
            "budget": SHORT_BUDGET if short else LONG_BUDGET,
            "priority": 0 if short else 100,
            "arrive_at": round(i / rate, 4),
            "slo_s": slo_ms / 1000.0,
        })
    return tasks


async def one_task(client, base_url: str, model: str, task: dict, policy: str,
                   results: list[dict], wall_origin: float) -> None:
    import httpx

    use_priority = policy == "budget_aware_priority"
    budget = task["budget"] if policy != "uniform_long" else LONG_BUDGET
    body = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "temperature": 0.0,
        "max_tokens": budget,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if use_priority:
        body["priority"] = task["priority"]
    wait_start = time.perf_counter()
    row: dict = {"task_id": task["task_id"], "index": task["index"],
                 "budget_class": task["budget_class"],
                 "budget": budget, "priority": body.get("priority"),
                 "arrive_at": task["arrive_at"], "wait_start": round(wait_start, 4)}
    try:
        t0 = time.perf_counter()
        r = await client.post(base_url.rstrip("/") + "/chat/completions", json=body,
                              timeout=600.0)
        dt = time.perf_counter() - t0
        d = r.json()
        msg = d["choices"][0]["message"]
        text = (msg.get("content") or "")
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
        usage = d.get("usage") or {}
        sc = score("gsm8k", text, task["gold"]) if task.get("gold") is not None else {}
        row.update({
            "status": r.status_code, "e2e_s": round(dt, 4),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "finish_reason": d["choices"][0].get("finish_reason"),
            "reasoning_chars": len(reasoning), "answer_chars": len(text),
            "answered": bool(text.strip()), "correct": bool(sc.get("correct")),
            "pred": sc.get("extracted"), "gold": task.get("gold"),
        })
    except Exception as exc:  # noqa: BLE001
        row.update({"status": None, "error": f"{type(exc).__name__}: {exc}",
                    "answered": False, "correct": False})
    finished = time.perf_counter()
    # 到达时间与完成时间必须用同一个时钟原点（都在 `run_policy` 的 t_start 上），
    # 否则 perf_counter 的任意原点会把 JCT 算成几十万秒。
    row["finished_rel_s"] = round(finished - wall_origin, 4)
    row["jct_s"] = round(finished - (wall_origin + task["arrive_at"]), 4)
    row["slo_met"] = bool(row.get("correct") and row["jct_s"] <= task["slo_s"])
    results.append(row)


async def run_policy(base_url: str, model: str, tasks: list[dict], policy: str,
                     concurrency: int) -> dict:
    import httpx

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        results: list[dict] = []

        async def guarded(task: dict) -> None:
            async with sem:
                await one_task(client, base_url, model, task, policy, results, t_start)

        t_start = time.perf_counter()

        handles: list[asyncio.Task] = []

        async def release_all() -> None:
            for task in tasks:
                now = time.perf_counter() - t_start
                if task["arrive_at"] > now:
                    await asyncio.sleep(task["arrive_at"] - now)
                handles.append(asyncio.create_task(guarded(task)))

        await release_all()
        while len(results) < len(tasks):
            await asyncio.sleep(0.2)
        await asyncio.gather(*handles)
        wall = time.perf_counter() - t_start

    results.sort(key=lambda r: r["index"] if "index" in r else 0)
    jcts = [r["jct_s"] for r in results if r.get("jct_s") is not None]
    tokens = [r.get("completion_tokens") or 0 for r in results]
    correct = [r for r in results if r.get("correct")]
    slo = [r for r in results if r.get("slo_met")]
    return {
        "policy": policy, "tasks": len(results), "wall_s": round(wall, 3),
        "correct": len(correct), "answered": sum(1 for r in results if r.get("answered")),
        "slo_met": len(slo), "slo_met_per_s": round(len(slo) / wall, 3),
        "success_rate": round(len(correct) / max(1, len(results)), 4),
        "jct_p50_s": round(statistics.median(jcts), 3) if jcts else None,
        "jct_p95_s": round(sorted(jcts)[min(len(jcts) - 1, int(round(0.95 * (len(jcts) - 1))))], 3)
        if jcts else None,
        "max_jct_s": round(max(jcts), 3) if jcts else None,
        "tokens_total": sum(tokens),
        "tokens_per_success": round(sum(tokens) / len(correct), 1) if correct else None,
        "tokens_per_success_by_class": {
            cls: (round(sum(r.get("completion_tokens") or 0
                            for r in results if r["budget_class"] == cls and r.get("correct"))
                    / max(1, sum(1 for r in results if r["budget_class"] == cls
                                 and r.get("correct"))), 1))
            for cls in ("short", "long")},
        "errors": sum(1 for r in results if r.get("error")),
        "per_task": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.4×9.7 预算策略与任务级调度集成")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--dataset", default="gsm8k")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rate", type=float, default=3.0)
    r.add_argument("--slo-ms", type=float, default=30000.0)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--policies",
                   default="uniform_long,budget_aware,budget_aware_priority")
    args = ap.parse_args()

    tasks = build_tasks(args.n, args.dataset, args.seed, args.rate, args.slo_ms)
    report = {
        "config": {"base_url": args.base_url, "model": args.model, "n": args.n,
                   "dataset": args.dataset, "seed": args.seed, "rate": args.rate,
                   "slo_ms": args.slo_ms, "concurrency": args.concurrency,
                   "short_budget": SHORT_BUDGET, "long_budget": LONG_BUDGET,
                   "policies": args.policies.split(",")},
        "task_plan": [{"task_id": t["task_id"], "budget_class": t["budget_class"],
                       "budget": t["budget"], "arrive_at": t["arrive_at"]} for t in tasks],
        "policies": {},
    }
    for policy in args.policies.split(","):
        print(f"[policy] {policy}", flush=True)
        out = asyncio.run(run_policy(args.base_url, args.model, tasks, policy,
                                     args.concurrency))
        report["policies"][policy] = out
        print(json.dumps({k: v for k, v in out.items() if k != "per_task"},
                         ensure_ascii=False), flush=True)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
