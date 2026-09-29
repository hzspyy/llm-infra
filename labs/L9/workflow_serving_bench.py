#!/usr/bin/env python3
"""L9.7 任务 B：把任务级优先级接进真实引擎（vLLM 0.29.0 的 `priority` 字段）。

对照的四条策略只在**优先级怎么算**上不同，任务清单、到达过程、模型与采样参数完全一致：

| 策略 | 优先级来源 | 对应实现 |
|---|---|---|
| ``engine_fcfs`` | 全部为 0 | 基线：请求级 FCFS（引擎默认次序） |
| ``program_fcfs`` | 程序到达序号 | 程序级 FCFS |
| ``plas`` | 程序已完成调用的服务量之和 | Autellix PLAS |
| ``atlas`` | 程序已观测到的最长路径 | Autellix ATLAS |

vLLM 的等待队列在 ``--scheduling-policy priority`` 下是小顶堆（``priority`` 小者先出，并列取更早到达），
并且抢占受害者取「优先级最低、并列时最晚到达」的那个。因此这里**测的是引擎自身的排队与抢占**，
不是网关侧的排序：网关只做一件事——把程序级状态折算成一个整数 priority 注入请求。

每个任务是一条串行链（轮次之间是真实依赖：下一轮要等上一轮的输出），长短混合。记录：

* 任务级：JCT、是否在 SLO 内、每任务请求数、每任务 token；
* 请求级：TTFT、prompt/completion/cached token（cached 下降是抢占重算的代理量）；
* 策略自身开销：算优先级花掉的 CPU 时间（总时长与每次决策均值）；
* 引擎日志里的抢占计数。

用法::

    python labs/L9/workflow_serving_bench.py run --out out/9.7/serving \
        --base-url http://127.0.0.1:8051/v1 --policies engine_fcfs,program_fcfs,plas,atlas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import statistics
import time

POLICIES = ("engine_fcfs", "program_fcfs", "plas", "atlas")
PRIORITY_BUCKET_MS = 50.0      # 把毫秒级服务量折算成整数 priority 的桶宽


def q(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 3)


class Program:
    """一个任务（程序）的观测状态：PLAS 与 ATLAS 的标量都只从已完成调用累出来。"""

    __slots__ = ("pid", "arrival_index", "turns", "cls", "arrive_ms", "end_ms",
                 "service_ms", "critical_ms", "completed", "requests", "prompt_tokens",
                 "completion_tokens", "cached_tokens", "ttfts", "slo_met", "error")

    def __init__(self, pid: str, arrival_index: int, turns: int, cls: str):
        self.pid = pid
        self.arrival_index = arrival_index
        self.turns = turns
        self.cls = cls
        self.arrive_ms = None
        self.end_ms = None
        self.service_ms = 0.0        # PLAS：已完成调用耗时之和
        self.critical_ms = 0.0       # ATLAS：最长已观测路径
        self.completed = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.ttfts: list[float] = []
        self.slo_met = None
        self.error = None

    def priority(self, policy: str) -> int:
        if policy == "engine_fcfs":
            return 0
        if policy == "program_fcfs":
            return self.arrival_index
        if policy == "plas":
            return int(self.service_ms / PRIORITY_BUCKET_MS)
        if policy == "atlas":
            return int(self.critical_ms / PRIORITY_BUCKET_MS)
        raise ValueError(policy)

    def on_node_done(self, duration_ms: float) -> None:
        """一次调用完成：更新两个标量并累计。"""
        self.completed += 1
        self.service_ms += duration_ms
        self.critical_ms = max(self.critical_ms, self.service_ms)


def build_programs(n: int) -> list[Program]:
    """长短混合：前 1/3 是 6 轮长任务，其余是 2 轮短任务，交错排列。"""
    progs = []
    for i in range(n):
        long_task = (i % 3 == 0)
        turns = 6 if long_task else 2
        progs.append(Program(f"p{i:02d}", i, turns, "long" if long_task else "short"))
    return progs


def messages_for(prog: Program, turn: int, filler: str) -> list[dict]:
    msgs = [{"role": "system", "content": filler}]
    for t in range(1, turn):
        msgs.append({"role": "user", "content": f"{prog.pid} 第 {t} 问"})
        msgs.append({"role": "assistant", "content": f"（{prog.pid} 第 {t} 答）"})
    msgs.append({"role": "user", "content": f"{prog.pid} 第 {turn} 问：只回一个词。"})
    return msgs


async def one_turn(client, model: str, messages: list[dict], priority: int, max_tokens: int,
                   timeout: float) -> dict:
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = completion = cached = None
    error = None
    text = 0
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
            stream=True, stream_options={"include_usage": True},
            # priority 走 extra_body：openai SDK 的 create() 不认识该关键字（会 TypeError），
            # 而 vLLM 的 OpenAI 协议把它作为请求体字段读取。
            extra_body={"priority": priority,
                        "chat_template_kwargs": {"enable_thinking": False}},
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
                text += len(piece)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
            "cached_tokens": cached, "ttft_ms": round(ttft, 3) if ttft else None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "chars": text, "error": error}


async def run_program(client, model: str, prog: Program, policy: str, filler: str,
                      max_tokens: int, timeout: float, sem: asyncio.Semaphore,
                      t0: float, stats: dict) -> None:
    async with sem:
        prog.arrive_ms = (time.perf_counter() - t0) * 1000.0
        for turn in range(1, prog.turns + 1):
            # --- 策略自身开销：只算"把程序状态折算成 priority"这一段 ---
            t_pri = time.perf_counter()
            priority = prog.priority(policy)
            stats["priority_decisions"] += 1
            stats["priority_cpu_ms"] += (time.perf_counter() - t_pri) * 1000.0

            res = await one_turn(client, model, messages_for(prog, turn, filler), priority,
                                 max_tokens, timeout)
            prog.requests += 1
            for key, attr in (("prompt_tokens", "prompt_tokens"),
                              ("completion_tokens", "completion_tokens"),
                              ("cached_tokens", "cached_tokens")):
                setattr(prog, attr, getattr(prog, attr) + (res[key] or 0))
            if res["ttft_ms"]:
                prog.ttfts.append(res["ttft_ms"])
            if res["error"]:
                prog.error = res["error"]
                break
            prog.on_node_done(res["e2e_ms"])
        prog.end_ms = (time.perf_counter() - t0) * 1000.0
        jct = prog.end_ms - prog.arrive_ms
        prog.slo_met = (prog.error is None) and jct <= stats["slo_ms"]


async def metrics_probe(base_url: str, stop: asyncio.Event, out: dict) -> None:
    """后台采样引擎指标：累计抢占次数与 KV 使用率峰值。

    抢占在 `scheduler.py:669-673` 里没有日志行（只在优先级模式下换受害者），所以判据用
    Prometheus 计数器 `vllm:num_preemptions_total` 与 `vllm:kv_cache_usage_perc`，而不是 grep 日志。
    """
    import httpx2

    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    url = root.rstrip("/") + "/metrics"
    out.setdefault("samples", 0)
    out.setdefault("kv_peak", 0.0)
    out.setdefault("preempt_start", None)
    out.setdefault("preempt_end", None)
    async with httpx2.AsyncClient(timeout=5.0) as cli:
        while not stop.is_set():
            try:
                text = (await cli.get(url)).text
                for line in text.splitlines():
                    if line.startswith("vllm:num_preemptions_total"):
                        val = float(line.rsplit(" ", 1)[1])
                        if out["preempt_start"] is None:
                            out["preempt_start"] = val
                        out["preempt_end"] = val
                    elif line.startswith("vllm:kv_cache_usage_perc"):
                        out["kv_peak"] = max(out["kv_peak"], float(line.rsplit(" ", 1)[1]))
                out["samples"] += 1
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.1)


def parse_preemptions(log_path: pathlib.Path | None) -> dict:
    if not log_path or not log_path.exists():
        return {"log": None, "count": None}
    text = log_path.read_text(errors="replace")
    hits = re.findall(r"[Pp]reempt\w*", text)
    return {"log": str(log_path), "count": len(hits),
            "samples": list(dict.fromkeys(hits))[:5]}


async def run_policy(args, policy: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    progs = build_programs(args.tasks)
    filler = ("系统提示：你是一个只回答事实的助手。 " * max(1, args.prefix_repeat))
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"priority_decisions": 0, "priority_cpu_ms": 0.0, "slo_ms": args.slo_ms}
    probe_stop = asyncio.Event()
    probe = {}
    probe_task = asyncio.create_task(metrics_probe(args.base_url, probe_stop, probe))
    t0 = time.perf_counter()
    await asyncio.gather(*[run_program(client, args.model, p, policy, filler,
                                       args.max_tokens, args.timeout, sem, t0, stats)
                           for p in progs])
    wall = time.perf_counter() - t0
    probe_stop.set()
    await asyncio.gather(probe_task, return_exceptions=True)
    await client.close()

    jcts = [p.end_ms - p.arrive_ms for p in progs if p.end_ms is not None]
    ok = [p for p in progs if p.error is None]
    sla = [p for p in progs if p.slo_met]
    per_class = {}
    for cls in ("short", "long"):
        sel = [p for p in progs if p.cls == cls]
        per_class[cls] = {
            "tasks": len(sel),
            "jct_p50": q([p.end_ms - p.arrive_ms for p in sel if p.end_ms is not None], 0.5),
            "jct_p95": q([p.end_ms - p.arrive_ms for p in sel if p.end_ms is not None], 0.95),
            "slo_met": sum(1 for p in sel if p.slo_met),
            "requests": sum(p.requests for p in sel),
            "cached_tokens": sum(p.cached_tokens for p in sel),
        }
    preempt_delta = None
    if probe.get("preempt_start") is not None and probe.get("preempt_end") is not None:
        preempt_delta = probe["preempt_end"] - probe["preempt_start"]
    return {
        "policy": policy,
        "engine_metrics": {"samples": probe.get("samples"), "kv_usage_peak": probe.get("kv_peak"),
                           "preemptions_delta": preempt_delta,
                           "preemptions_total_end": probe.get("preempt_end")},
        "config": {"tasks": args.tasks, "concurrency": args.concurrency,
                   "max_tokens": args.max_tokens, "slo_ms": args.slo_ms,
                   "priority_bucket_ms": PRIORITY_BUCKET_MS},
        "wall_s": round(wall, 3),
        "tasks": len(progs),
        "tasks_ok": len(ok),
        "tasks_error": len(progs) - len(ok),
        "slo_met": len(sla),
        "slo_met_per_s": round(len(sla) / wall, 4) if wall > 0 else None,
        "task_jct_ms": {"p50": q(jcts, 0.5), "p95": q(jcts, 0.95), "max": q(jcts, 1.0),
                        "mean": round(statistics.fmean(jcts), 3) if jcts else None},
        "requests_total": sum(p.requests for p in progs),
        "requests_per_task": round(sum(p.requests for p in progs) / max(1, len(progs)), 3),
        "ttft_ms": {"p50": q([t for p in progs for t in p.ttfts], 0.5),
                    "p95": q([t for p in progs for t in p.ttfts], 0.95)},
        "tokens": {"prompt": sum(p.prompt_tokens for p in progs),
                   "completion": sum(p.completion_tokens for p in progs),
                   "cached": sum(p.cached_tokens for p in progs)},
        "cache_hit_ratio": round(
            sum(p.cached_tokens for p in progs) / max(1, sum(p.prompt_tokens for p in progs)), 4),
        "per_class": per_class,
        "policy_cpu": {"decisions": stats["priority_decisions"],
                       "total_ms": round(stats["priority_cpu_ms"], 4),
                       "per_decision_us": round(stats["priority_cpu_ms"] * 1000.0
                                                / max(1, stats["priority_decisions"]), 3)},
        "programs": [{"pid": p.pid, "cls": p.cls, "turns": p.turns, "requests": p.requests,
                      "jct_ms": round(p.end_ms - p.arrive_ms, 3) if p.end_ms is not None else None,
                      "slo_met": p.slo_met, "service_ms": round(p.service_ms, 3),
                      "critical_ms": round(p.critical_ms, 3),
                      "cached_tokens": p.cached_tokens, "error": p.error} for p in progs],
    }


async def cmd_run_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for policy in args.policies.split(","):
        policy = policy.strip()
        res = await run_policy(args, policy)
        res["preemptions"] = parse_preemptions(pathlib.Path(args.engine_log)
                                               if args.engine_log else None)
        results[policy] = res
        print(f"[{policy}] wall={res['wall_s']}s jct_p50={res['task_jct_ms']['p50']} "
              f"p95={res['task_jct_ms']['p95']} slo_met={res['slo_met']}/{res['tasks']} "
              f"slo/s={res['slo_met_per_s']} cache={res['cache_hit_ratio']} "
              f"prio_cpu={res['policy_cpu']['per_decision_us']}us "
              f"preempt={res['preemptions']['count']}", flush=True)
    base = results.get("engine_fcfs")
    comparisons = {}
    for policy, res in results.items():
        if policy == "engine_fcfs":
            continue
        comparisons[policy] = {
            "jct_p50_delta_pct": _delta(res["task_jct_ms"]["p50"],
                                        base["task_jct_ms"]["p50"] if base else None),
            "jct_p95_delta_pct": _delta(res["task_jct_ms"]["p95"],
                                        base["task_jct_ms"]["p95"] if base else None),
            "slo_met_per_s_delta_pct": _delta(res["slo_met_per_s"],
                                              base["slo_met_per_s"] if base else None),
            "cache_hit_ratio_delta": (round(res["cache_hit_ratio"]
                                            - (base["cache_hit_ratio"] if base else 0), 4)),
        }
    cfg = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
    report = {"config": cfg, "policies": results, "vs_engine_fcfs": comparisons,
              "note": ("四条策略使用同一任务清单、同一到达过程与同一模型；唯一变量是注入的 priority。"
                       "网关只做程序级状态折算，排队与抢占都在引擎内完成")}
    (out / "workflow_serving.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(json.dumps({"vs_engine_fcfs": comparisons}, ensure_ascii=False, indent=1))
    return 0


def _delta(new, old):
    if new is None or old in (None, 0):
        return None
    return round((new - old) / old * 100.0, 3)


def cmd_run(args) -> int:
    return asyncio.run(cmd_run_async(args))


# --------------------------------------------------------------------------------------
# 抢占专项：优先级模式下受害者的选择规则
# --------------------------------------------------------------------------------------

async def one_big(client, model: str, prompt: str, priority: int, max_tokens: int,
                  timeout: float) -> dict:
    t0 = time.perf_counter()
    ttft = None
    cached = prompt_tokens = completion = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, stream=True,
            stream_options={"include_usage": True},
            extra_body={"priority": priority,
                        "chat_template_kwargs": {"enable_thinking": False}},
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
            if (getattr(delta, "content", None) or getattr(delta, "reasoning", None)):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"priority": priority, "ttft_ms": round(ttft, 3) if ttft else None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion,
            "cached_tokens": cached, "error": error}


async def cmd_preempt_async(args) -> int:
    """填满 KV 池以触发真实抢占，观察优先级模式下的受害者选择。

    受害者规则见 `vllm/v1/core/sched/scheduler.py:669-673`：优先级模式下取
    `max(running, key=(priority, arrival_time))`，即 priority 数值最大（优先级最低）、
    并列时最晚到达的那个。判据用引擎的 `vllm:num_preemptions_total` 计数器，
    不以日志文本为依据（该分支没有日志行）。
    """
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    # 每请求一段互不相同的长前缀：只有这样才能真正占满 KV 池（共享前缀会被缓存摊薄）
    prompt_tokens = args.prompt_tokens
    prompts = [
        "以下是编号 %d 的独立资料：" % i
        + " ".join(f"t{i}_{j}" for j in range(max(1, prompt_tokens // 3)))
        for i in range(args.n)
    ]
    priorities = []
    for i in range(args.n):
        if args.priority_mode == "zero":
            priorities.append(0)
        else:
            priorities.append(0 if i % 2 == 0 else args.low_priority)
    probe_stop = asyncio.Event()
    probe: dict = {}
    probe_task = asyncio.create_task(metrics_probe(args.base_url, probe_stop, probe))
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[one_big(client, args.model, prompts[i], priorities[i],
                                          args.max_tokens, args.timeout)
                                  for i in range(args.n)])
    wall = time.perf_counter() - t0
    probe_stop.set()
    await asyncio.gather(probe_task, return_exceptions=True)
    await client.close()
    for i, r in enumerate(rows):
        r["index"] = i
        r["group"] = "high" if priorities[i] == 0 else "low"
    groups = {}
    for g in ("high", "low"):
        sel = [r for r in rows if r["group"] == g]
        groups[g] = {
            "n": len(sel),
            "ttft_p50": q([r["ttft_ms"] for r in sel], 0.5),
            "ttft_p95": q([r["ttft_ms"] for r in sel], 0.95),
            "e2e_p50": q([r["e2e_ms"] for r in sel], 0.5),
            "e2e_p95": q([r["e2e_ms"] for r in sel], 0.95),
            "cached_ratio_p50": q([(r["cached_tokens"] or 0) / max(1, r["prompt_tokens"] or 1)
                                   for r in sel], 0.5),
            "errors": sum(1 for r in sel if r["error"]),
        }
    report = {
        "config": {"n": args.n, "prompt_tokens_target": prompt_tokens, "max_tokens": args.max_tokens,
                   "priority_mode": args.priority_mode, "low_priority": args.low_priority},
        "engine_metrics": {"samples": probe.get("samples"), "kv_usage_peak": probe.get("kv_peak"),
                           "preemptions_delta": (None if probe.get("preempt_start") is None
                                                 else probe["preempt_end"] - probe["preempt_start"]),
                           "preemptions_total_end": probe.get("preempt_end")},
        "wall_s": round(wall, 3),
        "groups": groups,
        "rows": rows,
        "rule_under_test": ("priority 模式下受害者 = max(running, key=(priority, arrival_time))；"
                            "低优先组应承担更多重算与更差尾延迟；FCFS 档受害者由到达次序决定"),
    }
    (out / f"preempt_{args.priority_mode}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


def cmd_preempt(args) -> int:
    return asyncio.run(cmd_preempt_async(args))


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.7 任务级优先级接入真实引擎")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8051/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--policies", default="engine_fcfs,program_fcfs,plas,atlas")
    p.add_argument("--tasks", type=int, default=18)
    p.add_argument("--concurrency", type=int, default=18)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prefix-repeat", type=int, default=24)
    p.add_argument("--slo-ms", type=float, default=20000.0)
    p.add_argument("--engine-log", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("preempt")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8053/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--prompt-tokens", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--priority-mode", choices=["spread", "zero"], default="spread")
    p.add_argument("--low-priority", type=int, default=100)
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(func=cmd_preempt)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
