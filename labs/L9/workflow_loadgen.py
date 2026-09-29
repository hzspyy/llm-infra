#!/usr/bin/env python3
"""L9.7 任务 D：按 ARRIVAL 协议在固定任务到达率下运行完整任务，比较任务级 SLO 与成本。

与 `workflow_serving_bench.py`（9.7-B，固定任务清单看排队次序）的区别：这里跑的是**闭环任务**——
每个任务从 9.1 的教学任务清单里取，用真实工具执行、真实评分，任务的成功/失败/超时都进入分母；
到达过程由独立到达器按绝对时钟打点（不依赖客户端并发槽），策略只影响注入给引擎的 `priority`。

两个子命令：

* ``calibrate``：在若干候选到达率下各跑一小段，报告完成率与在飞任务数，用来确定"任务基线到达率"
  （ARRIVAL 协议里的 1.0×）。判据是**队列是否稳定**（窗口结束时在飞任务数不持续增长）。
* ``arrival``：在 0.3/0.6/0.9/1.1 倍基线到达率下按窗口运行，每档若干窗口、策略交错。

指标（每窗口一条记录）：

* 主表：SLO 内成功任务数/秒、任务 JCT p50/p95、最长等待、成功率、超时/拒绝数、每成功任务 token 成本；
* 解释项：请求 TTFT p50、请求吞吐、命中 token 占比。

样本门槛：每窗口都记录样本数；**少于 10000 个样本时不宣称 p99 稳定**（正文据此只报 p50/p95）。

用法::

    python labs/L9/workflow_loadgen.py calibrate --out out/9.7/cal --rates 2,4,6
    python labs/L9/workflow_loadgen.py arrival --out out/9.7/arrival --baseline-rate 4.0 \
        --scales 0.3,0.6,0.9,1.1 --windows 3 --window-s 120 --policies engine_fcfs,plas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

PRIORITY_BUCKET_MS = 50.0


def q(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 3)


class Program:
    """任务（程序）状态：PLAS/ATLAS 标量只从已完成调用累出来。"""

    __slots__ = ("tid", "cls", "turns_done", "service_ms", "critical_ms", "arrive_ms",
                 "admit_ms", "end_ms", "requests", "prompt_tokens", "completion_tokens",
                 "cached_tokens", "ttfts", "error", "score", "arrival_index", "termination")

    def __init__(self, tid: str, cls: str, arrival_index: int):
        self.tid = tid
        self.cls = cls
        self.arrival_index = arrival_index
        self.turns_done = 0
        self.service_ms = 0.0
        self.critical_ms = 0.0
        self.arrive_ms = None
        self.admit_ms = None
        self.end_ms = None
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.ttfts: list[float] = []
        self.error = None
        self.score = None
        self.termination = None

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

    def on_call_done(self, duration_ms: float) -> None:
        self.turns_done += 1
        self.service_ms += duration_ms
        self.critical_ms = max(self.critical_ms, self.service_ms)


async def one_call(client, model: str, messages: list[dict], tools, priority: int,
                   max_tokens: int, timeout: float, thinking: bool) -> dict:
    t0 = time.perf_counter()
    ttft = None
    content_parts: list[str] = []
    tool_frags: dict[int, dict] = {}
    prompt_tokens = completion = cached = None
    finish = None
    error = None
    engine_id = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=0.0, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"priority": priority,
                        "chat_template_kwargs": {"enable_thinking": thinking}},
        )
        async for chunk in stream:
            if chunk.id:
                engine_id = chunk.id
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion = chunk.usage.completion_tokens
                d = getattr(chunk.usage, "prompt_tokens_details", None)
                if d is not None:
                    cached = getattr(d, "cached_tokens", None)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            piece_c = getattr(delta, "content", None)
            piece_r = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            tcs = getattr(delta, "tool_calls", None)
            if (piece_c or piece_r or tcs) and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            if piece_c:
                content_parts.append(piece_c)
            for tc in tcs or []:
                idx = tc.index or 0
                slot = tool_frags.setdefault(idx, {"id": None, "name": None, "arguments": ""})
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
    return {"content": "".join(content_parts), "calls": [tool_frags[i] for i in sorted(tool_frags)],
            "prompt_tokens": prompt_tokens, "completion_tokens": completion,
            "cached_tokens": cached, "finish_reason": finish,
            "ttft_ms": round(ttft, 3) if ttft is not None else None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "engine_request_id": engine_id, "error": error}


async def run_task(client, task: dict, prog: Program, policy: str, *, model: str, max_turns: int,
                   max_tokens: int, timeout: float, thinking: bool, workdir: pathlib.Path,
                   stats: dict) -> None:
    """一个任务的完整闭环：多轮模型调用 + 真实工具 + 评分。"""
    sandbox = workdir / "tasks" / prog.tid
    sandbox.mkdir(parents=True, exist_ok=True)
    env = T.make_env(task, sandbox)
    prompt = task["prompt"]
    if task["task_class"] == "codefix":
        failure = T.run_repo_tests(sandbox)
        prompt += "\n\n当前测试失败的输出如下：\n```\n" + failure["tail"] + "\n```"
    messages = [{"role": "system", "content": T.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    final_text = ""
    for turn in range(1, max_turns + 1):
        # 上下文预算守卫：多轮会话 + 工具结果会把 prompt 撑过 max_model_len，
        # 一旦超限引擎返回 400，任务会以"模型报错"计入分母，掩盖真正的终止原因。
        approx_chars = sum(len(str(m.get("content") or "")) for m in messages)
        if approx_chars > stats["context_char_budget"]:
            prog.error = None
            prog.termination = "context_budget"
            stats["context_budget_stops"] += 1
            break
        t_pri = time.perf_counter()
        priority = prog.priority(policy)
        stats["priority_decisions"] += 1
        stats["priority_cpu_ms"] += (time.perf_counter() - t_pri) * 1000.0
        res = await one_call(client, model, messages, env.tools, priority, max_tokens,
                             timeout, thinking)
        prog.requests += 1
        prog.prompt_tokens += res["prompt_tokens"] or 0
        prog.completion_tokens += res["completion_tokens"] or 0
        prog.cached_tokens += res["cached_tokens"] or 0
        if res["ttft_ms"]:
            prog.ttfts.append(res["ttft_ms"])
        stats["requests"] += 1
        stats["prompt_tokens"] += res["prompt_tokens"] or 0
        stats["completion_tokens"] += res["completion_tokens"] or 0
        stats["cached_tokens"] += res["cached_tokens"] or 0
        if res["ttft_ms"]:
            stats["ttfts"].append(res["ttft_ms"])
        if res["error"]:
            prog.error = res["error"]
            break
        prog.on_call_done(res["e2e_ms"])
        calls = res["calls"]
        if not calls:
            final_text = res["content"]
            break
        assistant_msg = {"role": "assistant", "content": res["content"] or "",
                         "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": c["arguments"] or "{}"}}
                                        for i, c in enumerate(calls)]}
        messages.append(assistant_msg)
        for i, c in enumerate(calls):
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                out = "ERROR: bad JSON arguments"
            else:
                out = env.call(c["name"], args) if isinstance(args, dict) else "ERROR: not an object"
            messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                             "content": str(out)[:4000]})
        if turn == max_turns:
            final_text = res["content"]
    scored = env.score(final_text)
    prog.score = scored.get("score")
    if prog.termination is None:
        prog.termination = "final" if not prog.error else "error"
    prog.end_ms = (time.perf_counter() - stats["t0"]) * 1000.0


async def run_window(args, policy: str, rate: float, window_s: float, tasks: list[dict],
                     out: pathlib.Path, window_idx: int) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"priority_decisions": 0, "priority_cpu_ms": 0.0, "requests": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "cached_tokens": 0, "ttfts": [], "t0": None,
             "logical_arrivals": 0, "admitted": 0, "rejected": 0,
             "context_char_budget": args.context_char_budget, "context_budget_stops": 0}
    progs: list[Program] = []
    rng = random.Random(args.seed + int(rate * 1000) + window_idx)
    loop = asyncio.get_running_loop()
    stats["t0"] = time.perf_counter()
    t0 = stats["t0"]

    async def launch(task: dict, idx: int) -> None:
        prog = Program(f"{task['task_id']}-w{window_idx}-{idx}", task["task_class"], idx)
        prog.arrive_ms = (time.perf_counter() - t0) * 1000.0
        stats["logical_arrivals"] += 1
        progs.append(prog)
        if len(progs) > args.max_tasks_per_window:
            stats["rejected"] += 1
            prog.error = "window_task_budget_exceeded"
            return
        async with sem:
            prog.admit_ms = (time.perf_counter() - t0) * 1000.0
            stats["admitted"] += 1
            await run_task(client, task, prog, policy, model=args.model, max_turns=args.max_turns,
                           max_tokens=args.max_tokens, timeout=args.timeout,
                           thinking=task["task_class"] == "compute", workdir=out, stats=stats)

    # 到达器：独立于并发槽的绝对时钟
    pending = []
    idx = 0
    deadline = t0 + window_s
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        gap = rng.expovariate(rate) if rate > 0 else 0.0
        nxt = now + gap
        if nxt >= deadline:
            await asyncio.sleep(max(0.0, deadline - now))
            break
        await asyncio.sleep(gap)
        pending.append(asyncio.create_task(launch(tasks[idx % len(tasks)], idx)))
        idx += 1
    if pending:
        await asyncio.gather(*pending)
    wall = time.perf_counter() - t0
    await client.close()

    engine_alive = True
    try:
        import httpx2
        async with httpx2.AsyncClient(timeout=5.0) as cli:
            root = args.base_url[:-3] if args.base_url.endswith("/v1") else args.base_url
            engine_alive = (await cli.get(root.rstrip("/") + "/v1/models")).status_code == 200
    except Exception:  # noqa: BLE001
        engine_alive = False
    finished = [p for p in progs if p.end_ms is not None]
    # SLO 阈值在比较策略前预先声明为一组（曲线），不在跑完之后挑一个好看的门限。
    thresholds = [float(x) for x in args.slo_ms_list.split(",")]
    slo_by_threshold = {}
    for th in thresholds:
        met = [p for p in finished if p.score == 1.0
               and (p.end_ms - p.arrive_ms) <= th and not p.error]
        slo_by_threshold[str(int(th))] = {
            "tasks": len(met),
            "per_s": round(len(met) / wall, 4) if wall else None,
        }
    slo = [p for p in finished if p.score == 1.0
           and (p.end_ms - p.arrive_ms) <= thresholds[0] and not p.error]
    jcts = [p.end_ms - p.arrive_ms for p in finished]
    waits = [p.admit_ms - p.arrive_ms for p in progs if p.admit_ms is not None]
    return {
        "policy": policy, "rate": rate, "window": window_idx, "window_s": round(window_s, 2),
        "wall_s": round(wall, 3),
        "tasks_arrived": stats["logical_arrivals"], "tasks_admitted": stats["admitted"],
        "tasks_finished": len(finished), "tasks_in_flight": len(progs) - len(finished),
        "tasks_rejected": stats["rejected"],
        "success_tasks": sum(1 for p in finished if p.score == 1.0),
        "slo_met_tasks": len(slo),
        "slo_met_per_s": round(len(slo) / wall, 4) if wall else None,
        "slo_thresholds_ms": thresholds,
        "slo_by_threshold": slo_by_threshold,
        "jct_ms": {"p50": q(jcts, 0.5), "p95": q(jcts, 0.95), "max": q(jcts, 1.0)},
        "max_queue_wait_ms": q(waits, 1.0),
        "success_rate": round(len([p for p in finished if p.score == 1.0])
                              / max(1, len(finished)), 4),
        "errors": sum(1 for p in progs if p.error),
        "context_budget_stops": stats["context_budget_stops"],
        "terminations": {t: sum(1 for p in progs if p.termination == t)
                         for t in {p.termination for p in progs}},
        "requests": stats["requests"],
        "requests_per_s": round(stats["requests"] / wall, 3) if wall else None,
        "ttft_ms": {"p50": q(stats["ttfts"], 0.5), "p95": q(stats["ttfts"], 0.95)},
        "tokens": {"prompt": stats["prompt_tokens"], "completion": stats["completion_tokens"],
                   "cached": stats["cached_tokens"]},
        "tokens_per_success": round((stats["prompt_tokens"] + stats["completion_tokens"])
                                    / max(1, len([p for p in finished if p.score == 1.0])), 1),
        "cache_hit_ratio": round(stats["cached_tokens"] / max(1, stats["prompt_tokens"]), 4),
        "policy_cpu_us_per_decision": round(stats["priority_cpu_ms"] * 1000.0
                                            / max(1, stats["priority_decisions"]), 3),
        "engine_alive_after_window": engine_alive,
        "samples_note": ("本窗口请求数不足 10000，p95 只作同口径比较，不宣称 p99 稳定"
                         if stats["requests"] < 10000 else "请求数达标"),
    }


def load_tasks(args) -> list[dict]:
    if args.tasks_json:
        d = json.loads(pathlib.Path(args.tasks_json).read_text(encoding="utf-8"))
        return d["tasks"]
    tasks = []
    for cls in args.classes.split(","):
        tasks.extend(T.build_tasks(cls.strip(), args.n_per_class, args.seed))
    return tasks


async def cmd_calibrate_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args)
    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        res = await run_window(args, "engine_fcfs", rate, args.window_s, tasks, out, 0)
        rows.append(res)
        print(f"[calibrate] rate={rate} finished={res['tasks_finished']} "
              f"in_flight={res['tasks_in_flight']} requests/s={res['requests_per_s']} "
              f"slo/s={res['slo_met_per_s']}", flush=True)
    report = {"config": {"rates": args.rates, "window_s": args.window_s,
                         "classes": args.classes, "n_per_class": args.n_per_class},
              "rows": rows,
              "note": "基线到达率取「在飞任务数不持续增长」的最大档；1.0× 由 arrival 子命令按 0.3/0.6/0.9/1.1 展开"}
    (out / "calibrate.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return 0


async def cmd_arrival_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args)
    scales = [float(x) for x in args.scales.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    rows = []
    # 策略交错：每档到达率下按窗口轮转策略，避免某个策略总是先跑（缓存/编译状态不同）
    for scale in scales:
        rate = args.baseline_rate * scale
        for w in range(args.windows):
            for k, policy in enumerate(policies):
                res = await run_window(args, policy, rate, args.window_s, tasks, out,
                                       w * len(policies) + k)
                res["scale"] = scale
                rows.append(res)
                print(f"[arrival] scale={scale} rate={rate:.2f} policy={policy} w={w} "
                      f"finished={res['tasks_finished']} slo/s={res['slo_met_per_s']} "
                      f"jct_p95={res['jct_ms']['p95']} success={res['success_rate']} "
                      f"in_flight={res['tasks_in_flight']}", flush=True)
    summary = {}
    for scale in scales:
        for policy in policies:
            sel = [r for r in rows if r["scale"] == scale and r["policy"] == policy]
            if not sel:
                continue
            summary[f"{scale}|{policy}"] = {
                "windows": len(sel),
                "slo_met_per_s_mean": round(statistics.fmean(
                    [r["slo_met_per_s"] or 0 for r in sel]), 4),
                "slo_by_threshold_mean": {
                    th: round(statistics.fmean(
                        [r["slo_by_threshold"][th]["per_s"] or 0 for r in sel]), 4)
                    for th in sel[0]["slo_by_threshold"]},
                "jct_p50_mean": round(statistics.fmean(
                    [r["jct_ms"]["p50"] or 0 for r in sel]), 3),
                "jct_p95_mean": round(statistics.fmean(
                    [r["jct_ms"]["p95"] or 0 for r in sel]), 3),
                "max_queue_wait_ms_max": max(r["max_queue_wait_ms"] or 0 for r in sel),
                "success_rate_mean": round(statistics.fmean([r["success_rate"] for r in sel]), 4),
                "tokens_per_success_mean": round(statistics.fmean(
                    [r["tokens_per_success"] for r in sel]), 1),
                "requests": sum(r["requests"] for r in sel),
                "requests_per_s_mean": round(statistics.fmean(
                    [r["requests_per_s"] or 0 for r in sel]), 3),
                "cache_hit_ratio_mean": round(statistics.fmean(
                    [r["cache_hit_ratio"] for r in sel]), 4),
            }
    report = {"config": {"baseline_rate": args.baseline_rate, "scales": args.scales,
                         "windows": args.windows, "window_s": args.window_s,
                         "policies": policies, "slo_ms_list": args.slo_ms_list,
                         "classes": args.classes, "n_per_class": args.n_per_class,
                         "max_turns": args.max_turns, "max_tokens": args.max_tokens},
              "windows": rows, "summary": summary,
              "note": ("主表是 SLO 内成功任务数/秒与任务 JCT；请求 TTFT/吞吐与缓存命中用于解释原因。"
                       "每窗口请求数都远小于 10000，因此只报 p50/p95")}
    (out / "arrival.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.7 ARRIVAL 任务级负载与 SLO")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("calibrate", cmd_calibrate_async), ("arrival", cmd_arrival_async)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
        p.add_argument("--model", default="Qwen/Qwen3-4B")
        p.add_argument("--tasks-json", default=None)
        p.add_argument("--classes", default="compute,retrieval")
        p.add_argument("--n-per-class", type=int, default=40)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--concurrency", type=int, default=16)
        p.add_argument("--max-turns", type=int, default=4)
        p.add_argument("--context-char-budget", type=int, default=20000,
                       help="对话字符预算（约 5k token），超过就停止该任务并记入分母")
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--slo-ms-list", default="10000,20000,30000,60000",
                       help="预先声明的 SLO 阈值曲线（毫秒），第一个作为主表口径")
        p.add_argument("--max-tasks-per-window", type=int, default=400)
        p.add_argument("--timeout", type=float, default=300.0)
        if name == "calibrate":
            p.add_argument("--rates", default="2,4,6")
            p.add_argument("--window-s", type=float, default=40.0)
        else:
            p.add_argument("--baseline-rate", type=float, required=True)
            p.add_argument("--scales", default="0.3,0.6,0.9,1.1")
            p.add_argument("--windows", type=int, default=3)
            p.add_argument("--window-s", type=float, default=120.0)
            p.add_argument("--policies", default="engine_fcfs,plas")
        p.set_defaults(func=lambda a, f=fn: asyncio.run(f(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
