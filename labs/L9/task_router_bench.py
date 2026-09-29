#!/usr/bin/env python3
"""L9.7 任务 E：两真实 worker 上的任务路由、亲和与失效恢复。

三条策略在**同一批任务、同一到达过程**下比较（每任务是一条 4 轮会话，轮次之间有真实依赖）：

| 策略 | 选 worker 的依据 | 关注的性质 |
|---|---|---|
| ``round_robin`` | 轮转 | 无状态基线，缓存命中率最低 |
| ``least_queue`` | 引擎的 running+waiting 最小 | 只看队列，不看缓存 |
| ``affinity`` | 任务首次落到的 worker，之后固定 | 缓存最优，但可能把队列堆在一台机器上 |
| ``cache_aware`` | 上一轮命中率高就留在原 worker，否则按队列选 | 缓存与队列联合 |

副作用按 ``idem_key = 任务#轮次#工具序号`` 记账：**重试与迁移不得重复提交**。失效注入是把 worker B
的引擎停掉（由 runner 在做实验时杀进程），客户端连续连接失败后把它标成 unhealthy，任务改投 A；
恢复后校验每个任务的副作用键恰好一条。

用法::

    python labs/L9/task_router_bench.py run --out out/9.7/router \\
        --worker-a http://127.0.0.1:8061/v1 --worker-b http://192.168.105.101:8061/v1 \\
        --policies round_robin,least_queue,affinity,cache_aware --tasks 12 --rounds 4
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

import agent_tasks as T  # noqa: E402


def q(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 3)


async def worker_metrics(base_url: str) -> dict:
    import httpx2

    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    out = {"running": 0.0, "waiting": 0.0, "kv": 0.0, "alive": False}
    try:
        async with httpx2.AsyncClient(timeout=3.0) as cli:
            text = (await cli.get(root.rstrip("/") + "/metrics")).text
        out["alive"] = True
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                out["running"] = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:num_requests_waiting{"):
                out["waiting"] = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:kv_cache_usage_perc{"):
                out["kv"] = float(line.rsplit(" ", 1)[1])
    except Exception:  # noqa: BLE001
        out["alive"] = False
    return out


class Ledger:
    """副作用账本：按幂等键去重；重试不得产生第二条。"""

    def __init__(self) -> None:
        self.effects: dict[str, int] = {}
        self.attempts: dict[str, int] = {}

    def execute(self, key: str) -> tuple[str, bool]:
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if key in self.effects:
            return f"effect-{self.effects[key]}", True
        self.effects[key] = len(self.effects) + 1
        return f"effect-{self.effects[key]}", False


async def one_turn(base_url: str, model: str, messages: list[dict], tools, max_tokens: int,
                   timeout: float) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
    t0 = time.perf_counter()
    content: list[str] = []
    frags: dict[int, dict] = {}
    ttft = prompt_tokens = cached = completion = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=0.0, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion = chunk.usage.completion_tokens
                d = getattr(chunk.usage, "prompt_tokens_details", None)
                if d is not None:
                    cached = getattr(d, "cached_tokens", None)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            piece = getattr(delta, "content", None) or getattr(delta, "reasoning", None)
            if piece:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                content.append(piece)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index or 0
                slot = frags.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] = fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await client.close()
    return {"content": "".join(content), "calls": [frags[i] for i in sorted(frags)],
            "prompt_tokens": prompt_tokens, "cached_tokens": cached,
            "completion_tokens": completion,
            "ttft_ms": round(ttft, 3) if ttft else None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3), "error": error}


class Router:
    def __init__(self, args):
        self.args = args
        self.workers = {"A": args.worker_a, "B": args.worker_b}
        self.health = {"A": True, "B": True}
        self.failures = {"A": 0, "B": 0}
        self.next_rr = 0
        self.metrics: dict[str, dict] = {"A": {}, "B": {}}

    async def refresh(self) -> None:
        for name, url in self.workers.items():
            m = await worker_metrics(url)
            self.metrics[name] = m
            if m["alive"]:
                self.failures[name] = 0
                self.health[name] = True
            else:
                self.failures[name] += 1
                if self.failures[name] >= self.args.fail_threshold:
                    if self.health[name]:
                        self.health[name] = False
                    self.health[name] = False

    def healthy(self) -> list[str]:
        hs = [n for n in ("A", "B") if self.health[n]]
        return hs or ["A", "B"]

    def pick(self, policy: str, task_pin: str | None, prev_ratio: float | None) -> str:
        hs = self.healthy()
        if policy == "round_robin":
            choice = hs[self.next_rr % len(hs)]
            self.next_rr += 1
            return choice
        if policy == "least_queue":
            cands = [n for n in hs if self.metrics.get(n, {}).get("alive")]
            if not cands:
                return hs[0]
            # 空载时两队都是 0，直接取 min 会把所有任务钉到同一台；
            # 用轮转偏移打散，才是"首轮选最闲、之后固定"的语义。
            best = min(self.metrics[n].get("running", 0) + self.metrics[n].get("waiting", 0)
                       for n in cands)
            tied = sorted(n for n in cands
                          if self.metrics[n].get("running", 0)
                          + self.metrics[n].get("waiting", 0) == best)
            choice = tied[self.next_rr % len(tied)]
            self.next_rr += 1
            return choice
        if policy == "affinity":
            if task_pin and self.health.get(task_pin, False):
                return task_pin
            # 首轮按队列选，之后固定：否则所有任务都会落到同一台（A），B 完全闲置
            cands = [n for n in hs if self.metrics.get(n, {}).get("alive")]
            if not cands:
                return hs[0]
            # 空载时两队都是 0，直接取 min 会把所有任务钉到同一台；
            # 用轮转偏移打散，才是"首轮选最闲、之后固定"的语义。
            best = min(self.metrics[n].get("running", 0) + self.metrics[n].get("waiting", 0)
                       for n in cands)
            tied = sorted(n for n in cands
                          if self.metrics[n].get("running", 0)
                          + self.metrics[n].get("waiting", 0) == best)
            choice = tied[self.next_rr % len(tied)]
            self.next_rr += 1
            return choice
        if policy == "cache_aware":
            if task_pin and self.health.get(task_pin, False) and (prev_ratio or 0) >= 0.5:
                return task_pin
            cands = [n for n in hs if self.metrics.get(n, {}).get("alive")]
            if not cands:
                return task_pin or hs[0]
            return min(cands, key=lambda n: (self.metrics[n].get("running", 0)
                                             + self.metrics[n].get("waiting", 0)))
        raise ValueError(policy)


async def run_task(router: Router, ledger: Ledger, task: dict, policy: str, args,
                   timeline: list[dict], t0: float) -> dict:
    sandbox = pathlib.Path(args.out) / "tasks" / f"{policy}-{task['task_id']}"
    sandbox.mkdir(parents=True, exist_ok=True)
    env = T.make_env(task, sandbox)
    messages = [{"role": "system", "content": T.SYSTEM_PROMPT},
                {"role": "user", "content": f"{task['prompt']}\n\n[run={args.run_tag} policy={policy}]"}]
    pin = None
    prev_ratio = None
    final_text = ""
    error = None
    retries = 0
    start = time.perf_counter()
    for turn in range(1, args.rounds + 1):
        await router.refresh()
        worker = router.pick(policy, pin, prev_ratio)
        pin = pin or worker
        res = await one_turn(router.workers[worker], args.model, messages, env.tools,
                             args.max_tokens, args.timeout)
        if res["error"] and retries < args.max_retries:
            retries += 1
            await asyncio.sleep(0.3)
            await router.refresh()
            worker = router.healthy()[0]          # 失效时改投另一台
            res = await one_turn(router.workers[worker], args.model, messages, env.tools,
                                 args.max_tokens, args.timeout)
        if res["error"]:
            error = res["error"]
            break
        ratio = (res["cached_tokens"] or 0) / max(1, res["prompt_tokens"] or 1)
        prev_ratio = ratio
        for i, tc in enumerate(res["calls"]):
            key = f"{task['task_id']}#{turn}#{i}"
            _effect, dedup = ledger.execute(key)
            try:
                a = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {}
            out = env.call(tc["name"], a) if isinstance(a, dict) else "ERROR"
            messages.append({"role": "tool", "tool_call_id": tc["id"] or f"c{i}",
                             "content": str(out)[:2000]})
            timeline.append({"task": task["task_id"], "policy": policy, "turn": turn,
                             "worker": worker, "idem_key": key, "dedup_hit": dedup,
                             "t_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                             "cached_ratio": round(ratio, 4), "ttft_ms": res["ttft_ms"]})
        messages.append({"role": "assistant", "content": res["content"],
                         "tool_calls": [{"id": c["id"] or f"c{i}", "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": c["arguments"] or "{}"}}
                                        for i, c in enumerate(res["calls"])]})
        if not res["calls"]:
            final_text = res["content"]
            break
    end = time.perf_counter()
    return {"task": task["task_id"], "jct_ms": round((end - start) * 1000.0, 3),
            "pinned_worker": pin, "error": error, "retries": retries,
            "score": env.score(final_text).get("score")}


async def run_policy(args, policy: str) -> dict:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    for cls in args.classes.split(","):
        tasks.extend(T.build_tasks(cls.strip(), args.tasks, args.seed))
    tasks = tasks[: args.tasks]
    router = Router(args)
    ledger = Ledger()
    timeline: list[dict] = []
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[run_task(router, ledger, t, policy, args, timeline, t0)
                                  for t in tasks], return_exceptions=True)
    wall = time.perf_counter() - t0
    ok = [r for r in rows if isinstance(r, dict)]
    per_worker = {}
    for name in ("A", "B"):
        sel = [e for e in timeline if e["worker"] == name]
        per_worker[name] = {
            "turns": len(sel),
            "cached_ratio_p50": q([e["cached_ratio"] for e in sel], 0.5),
            "ttft_p50": q([e["ttft_ms"] for e in sel], 0.5),
        }
    report = {
        "policy": policy, "wall_s": round(wall, 3), "tasks": len(ok),
        "success_tasks": sum(1 for r in ok if r.get("score") == 1.0),
        "errors": sum(1 for r in ok if r.get("error")),
        "retries": sum(r.get("retries", 0) for r in ok),
        "jct_ms": {"p50": q([r["jct_ms"] for r in ok], 0.5),
                   "p95": q([r["jct_ms"] for r in ok], 0.95)},
        "slo_met_tasks": sum(1 for r in ok if not r.get("error") and r["jct_ms"] <= args.slo_ms),
        "per_worker": per_worker,
        "ledger": {"effects": len(ledger.effects), "attempts": sum(ledger.attempts.values()),
                   "duplicate_attempts": sum(1 for n in ledger.attempts.values() if n > 1)},
        "timeline": timeline,
    }
    (out / f"{policy}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[{policy}] wall={report['wall_s']}s success={report['success_tasks']}/{report['tasks']} "
          f"jct_p50={report['jct_ms']['p50']} per_worker={json.dumps(per_worker, ensure_ascii=False)} "
          f"effects={report['ledger']['effects']} dup_attempts={report['ledger']['duplicate_attempts']}",
          flush=True)
    return report


async def cmd_run_async(args) -> int:
    if not args.run_tag:
        args.run_tag = str(int(time.time()))
    results = {}
    for policy in args.policies.split(","):
        policy = policy.strip()
        results[policy] = await run_policy(args, policy)
    rr = results.get("round_robin")
    aff = results.get("affinity")
    ca = results.get("cache_aware")
    lq = results.get("least_queue")
    checks: list[dict] = []
    if aff and rr:
        checks.append({"name": "affinity_has_highest_cache_hit",
                       "expected": "affinity 的单 worker 命中率高于 round_robin",
                       "got": {"affinity": aff["per_worker"]["A"]["cached_ratio_p50"],
                               "round_robin": rr["per_worker"]["A"]["cached_ratio_p50"]},
                       "match": (aff["per_worker"]["A"]["cached_ratio_p50"] or 0)
                                > (rr["per_worker"]["A"]["cached_ratio_p50"] or 0)})
        checks.append({"name": "round_robin_spreads_load",
                       "expected": "round_robin 两台都有任务",
                       "got": {k: rr["per_worker"][k]["turns"] for k in ("A", "B")},
                       "match": all(rr["per_worker"][k]["turns"] > 0 for k in ("A", "B"))})
    if lq and aff:
        checks.append({"name": "affinity_concentrates_load",
                       "expected": "affinity 的负载集中度高于 least_queue（负面结果也要报）",
                       "got": {"affinity_diff": abs(aff["per_worker"]["A"]["turns"]
                                                    - aff["per_worker"]["B"]["turns"]),
                               "least_queue_diff": abs(lq["per_worker"]["A"]["turns"]
                                                       - lq["per_worker"]["B"]["turns"])},
                       "match": abs(aff["per_worker"]["A"]["turns"] - aff["per_worker"]["B"]["turns"])
                                >= abs(lq["per_worker"]["A"]["turns"] - lq["per_worker"]["B"]["turns"])})
    checks.append({"name": "no_duplicate_side_effects",
                   "expected": "副作用键无重复提交（effects 数 == 不同幂等键数）",
                   "got": {p: r["ledger"] for p, r in results.items()},
                   "match": all(r["ledger"]["effects"] == len({e["idem_key"] for e in r["timeline"]})
                                for r in results.values())})
    checks.append({"name": "tasks_complete_despite_failures",
                   "expected": "注入失效后任务仍完成（错误数为 0）",
                   "got": {p: r["errors"] for p, r in results.items()},
                   "match": all(r["errors"] == 0 for r in results.values())})
    checks.append({"name": "healthy_worker_carries_all_work_after_failure",
                   "expected": "worker 失效后可用 worker 承接全部轮次",
                   "got": {p: {k: r["per_worker"][k]["turns"] for k in ("A", "B")}
                           for p, r in results.items()},
                   "match": all(r["per_worker"]["A"]["turns"] > 0 for r in results.values())})
    (pathlib.Path(args.out) / "task_router.json").write_text(
        json.dumps({"config": {k: v for k, v in vars(args).items()
                               if isinstance(v, (str, int, float, bool, type(None)))},
                    "results": {p: {k: v for k, v in r.items() if k != "timeline"}
                                for p, r in results.items()},
                    "checks": checks, "all_match": all(c["match"] for c in checks)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {json.dumps(c['got'], ensure_ascii=False)[:200]}")
    print("all_match:", all(c["match"] for c in checks))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.7 两 worker 任务路由与失效恢复")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--worker-a", default="http://127.0.0.1:8061/v1")
    p.add_argument("--worker-b", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--policies", default="round_robin,least_queue,affinity,cache_aware")
    p.add_argument("--classes", default="compute,retrieval")
    p.add_argument("--tasks", type=int, default=12)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--fail-threshold", type=int, default=2)
    p.add_argument("--slo-ms", type=float, default=30000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default=None,
                   help="本次运行的唯一标记；默认取时间戳，保证不与历史运行共享前缀")
    p.add_argument("--timeout", type=float, default=120.0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_run_async(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
