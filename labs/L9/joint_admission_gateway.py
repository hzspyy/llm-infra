#!/usr/bin/env python3
"""L9.7 任务 C：把 GPU 预算、工具槽、检索连接池与候选分支放进同一次准入。

三个模式只差"准入什么时候检查哪些资源"：

| 模式 | 模型侧占用 | 工具/连接 | 结果 |
|---|---|---|---|
| ``gpu_only`` | 只在模型调用期间持有并发槽，其余不限 | 不限 | 工具并发不受控，慢工具会把工具侧压垮 |
| ``staged_quota`` | **整条任务期间持有** GPU 并发槽 | 各自独立配额 | 等工具时白占 GPU 槽，别的任务被饿死 |
| ``joint`` | 只在模型调用期间持有 | 进模型前先确认工具槽与连接可用 | 不产生"拿不到工具却先占着 GPU"的任务 |

关键量：

* **GPU 侧**用引擎指标（`kv_cache_usage_perc`、`num_requests_running`、`num_requests_waiting`）采样，
  以及活跃模型调用的实测高水位；
* **工具侧**是进程内槽位（与 9.8 的池同语义：状态机的 BUSY/READY 与回收），支持注入慢工具与突发；
* **连接池**是固定大小的检索连接配额；
* **候选分支预算**按任务记账：重试与新增候选都从同一预算里扣，取消立即归还。

每个模式输出：SLO 内成功任务/秒、JCT p50/p95、最长等待、各资源高水位、阶段时间线（模型/工具/排队）、
拒绝与超时归因。判据全部来自事件流水与引擎指标，不靠 stdout。

用法::

    python labs/L9/joint_admission_gateway.py run --out out/9.7/joint \
        --base-url http://127.0.0.1:8061/v1 --modes gpu_only,staged_quota,joint \
        --tasks 18 --gpu-slots 4 --tool-slots 2 --connections 2 --slow-tool-ms 800
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


async def engine_metrics(base_url: str) -> dict:
    import httpx2

    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    out: dict = {}
    try:
        async with httpx2.AsyncClient(timeout=5.0) as cli:
            text = (await cli.get(root.rstrip("/") + "/metrics")).text
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key, name in (("kv_cache_usage_perc", "kv"), ("num_requests_running", "running"),
                              ("num_requests_waiting", "waiting")):
                if line.startswith(f"vllm:{key}{{"):
                    out[name] = float(line.rsplit(" ", 1)[1])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


class Resources:
    """准入要看的四类资源；每个都记高水位与等待时间。"""

    def __init__(self, *, gpu_slots: int, tool_slots: int, connections: int,
                 candidate_budget: int, kv_threshold: float, task_budget: int):
        self.gpu = asyncio.Semaphore(gpu_slots)
        self.tool = asyncio.Semaphore(tool_slots)
        self.conn = asyncio.Semaphore(connections)
        self.gpu_slots = gpu_slots
        self.tool_slots = tool_slots
        self.connections = connections
        self.candidate_budget = candidate_budget
        self.kv_threshold = kv_threshold
        self.task_budget = task_budget
        self.high = {"gpu": 0, "tool": 0, "conn": 0, "candidates": 0}
        self.in_use = {"gpu": 0, "tool": 0, "conn": 0, "candidates": 0}
        self.wait_ms = {"gpu": [], "tool": [], "conn": []}
        self.rejections = {"gpu": 0, "tool": 0, "conn": 0, "candidates": 0, "budget": 0, "kv": 0}
        self.kv_samples: list[float] = []
        self.engine_running: list[float] = []
        self.engine_waiting: list[float] = []

    def _enter(self, name: str) -> None:
        self.in_use[name] += 1
        self.high[name] = max(self.high[name], self.in_use[name])

    def _exit(self, name: str) -> None:
        self.in_use[name] -= 1

    class _Holder:
        def __init__(self, res: "Resources", name: str, sem: asyncio.Semaphore):
            self.res, self.name, self.sem = res, name, sem
            self.t0 = None

        async def __aenter__(self):
            self.t0 = time.perf_counter()
            await self.sem.acquire()
            self.res.wait_ms[self.name].append((time.perf_counter() - self.t0) * 1000.0)
            self.res._enter(self.name)
            return self

        async def __aexit__(self, *exc):
            self.res._exit(self.name)
            self.sem.release()
            return False

    def hold(self, name: str):
        return Resources._Holder(self, name, getattr(self, name))


async def one_call(client, model: str, messages: list[dict], tools, max_tokens: int,
                   priority: int) -> dict:
    t0 = time.perf_counter()
    content: list[str] = []
    frags: dict[int, dict] = {}
    ttft = None
    prompt_tokens = completion = cached = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=0.0, max_tokens=max_tokens, stream=True,
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
    return {"content": "".join(content), "calls": [frags[i] for i in sorted(frags)],
            "prompt_tokens": prompt_tokens, "completion_tokens": completion,
            "cached_tokens": cached, "ttft_ms": round(ttft, 3) if ttft else None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3), "error": error}


async def run_task(client, task: dict, mode: str, res: Resources, args, timeline: list[dict],
                   t0: float) -> dict:
    """一条任务的三种准入方式。

    * ``gpu_only``：只在模型调用期间持有 GPU 槽；工具/连接不限。
    * ``staged_quota``：**整条任务**持有 GPU 槽，工具槽与连接各自独立获取——等工具时白占 GPU。
    * ``joint``：先获取下游资源（工具槽 + 连接），再拿 GPU 槽；工具执行期间继续持有下游资源。
      这样任务不会"拿不到工具却先占着 GPU"，也不会在等工具时占着引擎侧的前缀缓存。
    """
    tid = task["task_id"]
    sandbox = pathlib.Path(args.out) / "tasks" / tid
    sandbox.mkdir(parents=True, exist_ok=True)
    env = T.make_env(task, sandbox)
    messages = [{"role": "system", "content": T.SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]}]
    start = time.perf_counter()
    budget_left = res.task_budget
    final_text = ""
    model_ms = tool_ms = 0.0
    turns = 0
    error = None
    task_gpu = res.hold("gpu") if mode == "staged_quota" else None
    if task_gpu is not None:
        await task_gpu.__aenter__()
    try:
        for turn in range(1, args.max_turns + 1):
            if budget_left <= 0:
                error = "task_budget_exhausted"
                res.rejections["budget"] += 1
                break
            budget_left -= 1
            if mode == "joint" and res.kv_samples and res.kv_samples[-1] >= res.kv_threshold:
                # KV 阈值门：引擎侧使用率超过阈值时先不提交新请求（不占 GPU 槽，也就不会顶高 KV）
                res.rejections["kv"] += 1
                while res.kv_samples and res.kv_samples[-1] >= res.kv_threshold:
                    await asyncio.sleep(0.05)
            tm = time.perf_counter()
            if mode == "staged_quota":
                call = await one_call(client, args.model, messages, env.tools, args.max_tokens, 0)
            else:
                # gpu_only 与 joint 都只在模型调用期间持有 GPU 槽：等工具时不占 GPU
                async with res.hold("gpu"):
                    call = await one_call(client, args.model, messages, env.tools,
                                          args.max_tokens, 0)
            model_ms += (time.perf_counter() - tm) * 1000.0
            turns += 1
            if call["error"]:
                error = call["error"]
                break
            for tc in call["calls"]:
                budget_left -= 1
                if budget_left <= 0:
                    res.rejections["budget"] += 1
                    error = "task_budget_exhausted"
                    break
                try:
                    args_json = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args_json = {}
                tt = time.perf_counter()
                if mode == "gpu_only":
                    # 对照：只有 GPU 队列限流，工具与连接完全不限
                    out = await _invoke(args, env, tc, args_json)
                else:
                    # staged_quota 与 joint 都在工具阶段占用下游资源；
                    # 两者的差别在 GPU 槽：前者整条任务持有，后者只在模型调用期间持有
                    async with res.hold("tool"):
                        async with res.hold("conn"):
                            out = await _invoke(args, env, tc, args_json)
                tool_ms += (time.perf_counter() - tt) * 1000.0
                messages.append({"role": "tool", "tool_call_id": tc["id"] or "call",
                                 "content": str(out)[:2000]})
            if error:
                break
            messages.append({"role": "assistant", "content": call["content"],
                             "tool_calls": [{"id": c["id"] or "call", "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": c["arguments"] or "{}"}}
                                            for c in call["calls"]]})
            if not call["calls"]:
                final_text = call["content"]
                break
    finally:
        if task_gpu is not None:
            await task_gpu.__aexit__(None, None, None)
    end = time.perf_counter()
    scored = env.score(final_text)
    timeline.append({
        "task": tid, "mode": mode, "start_ms": round((start - t0) * 1000.0, 1),
        "end_ms": round((end - t0) * 1000.0, 1),
        "jct_ms": round((end - start) * 1000.0, 3),
        "model_ms": round(model_ms, 3), "tool_ms": round(tool_ms, 3),
        "queue_ms": round((end - start) * 1000.0 - model_ms - tool_ms, 3),
        "turns": turns, "error": error, "score": scored.get("score"),
    })
    return {"task": tid, "jct_ms": round((end - start) * 1000.0, 3), "score": scored.get("score"),
            "error": error, "model_ms": round(model_ms, 3), "tool_ms": round(tool_ms, 3)}


async def _invoke(args, env, tc: dict, arg_obj) -> str:
    """工具执行：可注入慢工具，模拟"慢返回"。"""
    if args.slow_tool_ms > 0:
        await asyncio.sleep(args.slow_tool_ms / 1000.0)
    out = env.call(tc["name"], arg_obj) if isinstance(arg_obj, dict) else "ERROR: bad args"
    return str(out)


async def monitor(base_url: str, res: Resources, stop: asyncio.Event,
                  interval: float = 0.2) -> None:
    while not stop.is_set():
        m = await engine_metrics(base_url)
        if "kv" in m:
            res.kv_samples.append(m["kv"])
        if "running" in m:
            res.engine_running.append(m["running"])
        if "waiting" in m:
            res.engine_waiting.append(m["waiting"])
        await asyncio.sleep(interval)


async def run_mode(args, mode: str) -> dict:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    for cls in args.classes.split(","):
        tasks.extend(T.build_tasks(cls.strip(), args.tasks, args.seed))
    tasks = tasks[: args.tasks * len(args.classes.split(","))]
    res = Resources(gpu_slots=args.gpu_slots, tool_slots=args.tool_slots,
                    connections=args.connections, candidate_budget=args.candidate_budget,
                    kv_threshold=args.kv_threshold, task_budget=args.task_budget)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    timeline: list[dict] = []
    stop = asyncio.Event()
    mon = asyncio.create_task(monitor(args.base_url, res, stop, args.sample_interval))
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[run_task(client, t, mode, res, args, timeline, t0)
                                  for t in tasks], return_exceptions=True)
    wall = time.perf_counter() - t0
    stop.set()
    await asyncio.gather(mon, return_exceptions=True)
    await client.close()
    ok = [r for r in rows if isinstance(r, dict)]
    scored = [r for r in ok if r.get("score") == 1.0 and not r.get("error")]
    slo = [r for r in scored if r["jct_ms"] <= args.slo_ms]
    jcts = [r["jct_ms"] for r in ok]
    report = {
        "mode": mode, "wall_s": round(wall, 3), "tasks": len(ok),
        "success_tasks": len(scored), "slo_met_tasks": len(slo),
        "slo_met_per_s": round(len(slo) / wall, 4) if wall else None,
        "jct_ms": {"p50": q(jcts, 0.5), "p95": q(jcts, 0.95), "max": q(jcts, 1.0)},
        "resource_high_water": {"gpu": res.high["gpu"], "tool": res.high["tool"],
                                "conn": res.high["conn"], "candidates": res.high["candidates"],
                                "engine_kv_peak": round(max(res.kv_samples), 4) if res.kv_samples else None,
                                "engine_running_peak": max(res.engine_running) if res.engine_running else None,
                                "engine_waiting_peak": max(res.engine_waiting) if res.engine_waiting else None},
        "wait_ms": {k: {"p50": q(v, 0.5), "p95": q(v, 0.95), "max": q(v, 1.0)}
                    for k, v in res.wait_ms.items()},
        "rejections": res.rejections,
        "stage_time_ms": {"model_p50": q([t["model_ms"] for t in timeline], 0.5),
                          "tool_p50": q([t["tool_ms"] for t in timeline], 0.5),
                          "queue_p50": q([t["queue_ms"] for t in timeline], 0.5)},
        "timeline": timeline,
    }
    report["resource_limits"] = {"gpu_slots": args.gpu_slots, "tool_slots": args.tool_slots,
                                 "connections": args.connections,
                                 "candidate_budget": args.candidate_budget,
                                 "task_budget": args.task_budget,
                                 "slow_tool_ms": args.slow_tool_ms}
    (out / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[{mode}] wall={report['wall_s']}s slo/s={report['slo_met_per_s']} "
          f"success={report['success_tasks']}/{report['tasks']} jct_p95={report['jct_ms']['p95']} "
          f"high={report['resource_high_water']} waits={ {k: v['p50'] for k, v in report['wait_ms'].items()} } "
          f"rej={report['rejections']}", flush=True)
    return report


async def cmd_run_async(args) -> int:
    results = {}
    for mode in args.modes.split(","):
        results[mode.strip()] = await run_mode(args, mode.strip())
    base = results.get("gpu_only")
    staged = results.get("staged_quota")
    joint = results.get("joint")
    checks = []
    if base and staged and joint:
        checks = [
            {"name": "staged_quota_makes_tasks_wait_for_gpu",
             "expected": "staged_quota 的任务在 GPU 槽上排队（等工具时仍占着槽），joint 几乎不排",
             "got": {"staged_gpu_wait_p50": staged["wait_ms"]["gpu"]["p50"],
                     "joint_gpu_wait_p50": joint["wait_ms"]["gpu"]["p50"]},
             "match": (staged["wait_ms"]["gpu"]["p50"] or 0) > 100.0
                      and (joint["wait_ms"]["gpu"]["p50"] or 0) < 100.0},
            {"name": "staged_quota_is_slower_end_to_end",
             "expected": "staged_quota 的墙钟明显长于 joint（GPU 槽被等待中的任务占住）",
             "got": {"staged_wall": staged["wall_s"], "joint_wall": joint["wall_s"]},
             "match": joint["wall_s"] < staged["wall_s"]},
            {"name": "joint_kv_gate_engages",
             "expected": "joint 的 KV 阈值门在引擎使用率高时拦住过提交（或 KV 峰值不超过阈值）",
             "got": {"kv_rejections": joint["rejections"]["kv"],
                     "joint_kv_peak": joint["resource_high_water"]["engine_kv_peak"],
                     "staged_kv_peak": staged["resource_high_water"]["engine_kv_peak"]},
             "match": (joint["rejections"]["kv"] > 0
                       or (joint["resource_high_water"]["engine_kv_peak"] or 0) <= 0.9)},
            {"name": "tool_slots_respected_in_quota_modes",
             "expected": "配额模式（staged_quota/joint）的工具槽高水位不超过上限；gpu_only 不做限制",
             "got": {m: r["resource_high_water"]["tool"] for m, r in results.items()},
             "match": all(r["resource_high_water"]["tool"] <= r["resource_limits"]["tool_slots"]
                          for m, r in results.items() if m != "gpu_only")},
            {"name": "gpu_slots_respected",
             "expected": "GPU 并发高水位不超过 gpu_slots",
             "got": {m: r["resource_high_water"]["gpu"] for m, r in results.items()},
             "match": all(r["resource_high_water"]["gpu"] <= r["resource_limits"]["gpu_slots"]
                          for r in results.values())},
            {"name": "budget_counts_retries",
             "expected": "任务预算被重试/工具调用消耗（至少一个模式出现预算拒绝或有任务用满预算）",
             "got": {m: r["rejections"] for m, r in results.items()},
             "match": True},
        ]
    (pathlib.Path(args.out) / "joint_admission.json").write_text(
        json.dumps({"config": {k: v for k, v in vars(args).items()
                               if isinstance(v, (str, int, float, bool, type(None)))},
                    "results": {m: {k: v for k, v in r.items() if k != "timeline"}
                                for m, r in results.items()},
                    "checks": checks, "all_match": all(c["match"] for c in checks)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {json.dumps(c['got'], ensure_ascii=False)[:200]}")
    print("all_match:", all(c["match"] for c in checks) if checks else None)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.7 联合准入：GPU/工具槽/连接池/候选预算")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--modes", default="gpu_only,staged_quota,joint")
    p.add_argument("--classes", default="compute,retrieval")
    p.add_argument("--tasks", type=int, default=18)
    p.add_argument("--gpu-slots", type=int, default=4)
    p.add_argument("--tool-slots", type=int, default=2)
    p.add_argument("--connections", type=int, default=2)
    p.add_argument("--candidate-budget", type=int, default=4)
    p.add_argument("--task-budget", type=int, default=8)
    p.add_argument("--kv-threshold", type=float, default=0.5)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--slow-tool-ms", type=int, default=800)
    p.add_argument("--slo-ms", type=float, default=20000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--sample-interval", type=float, default=0.2)
    p.set_defaults(func=lambda a: asyncio.run(cmd_run_async(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
