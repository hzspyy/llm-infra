#!/usr/bin/env python3
"""L9.1 任务 A：统一因果事件（span）采集、原始 SSE 审计与任务图分析。

本文件把 9.1 需要的所有标识与时间点收敛到一套事件模型里，供 9.2–9.8 共用：

* 标识链：``session_id`` → ``task_id`` → ``node_id``（``<task>#tN``）→ ``parent_node_id``
  → ``attempt`` → ``tool_call_id``（``<node>/toolK``，引擎侧再带 ``engine_call_id``）
  → ``engine_request_id``（SSE 里真正回传的 ``id``）。
* 时间点：``task_arrive``（逻辑到达，由独立到达器在绝对时刻打点）→ ``task_admit``
  （拿到并发槽）→ ``node_ready`` → ``model_start`` → 首 reasoning/final/tool delta 三选一
  及全部三者 → ``model_end`` → ``tool_start``/``tool_end`` → ``task_end``。
* 首输出口径：首 reasoning delta、首 final delta、首 tool delta 分别记时；三者都不存在时
  记 ``missing_delta``。``ttft_ms`` 只是"首个任意 delta"的别名，**不是 prefill**，
  ``client_e2e_ms - ttft_ms`` 也只是客户端观测到的流式尾段，**不是 GPU decode**。
* 到达器与客户端并发槽解耦：到达时刻先在独立循环里按绝对时钟打点并落盘，任务随后才去
  竞争 ``asyncio.Semaphore``。因此注入慢工具或队列拥塞后，全部逻辑到达仍然被记录。
* 工具执行分两批：只读工具（``calculate``/``search_corpus``/``list_files``/``read_file``）
  并行执行（fork/join），有副作用的工具（``write_file``/``edit_file``/``run_tests``）按原
  顺序串行执行。回填进上下文的消息顺序始终按模型给出的调用顺序，保证后续轮次的输入可复现。

子命令：

* ``collect``   —— 跑真实任务并写 ``spans.jsonl`` / ``nodes.jsonl`` / ``sessions.jsonl`` /
  ``requests.jsonl``；可用 ``--capture-sse N`` 额外抓取前 N 个请求的原始 SSE 字节。
* ``graph``     —— 从 spans 里挑一条纯串行任务与一条含 fork/join 的任务，输出事件图、
  关键路径与"并行区间不能简单求和"的对照。
* ``sse-audit`` —— 用 httpx 直连引擎抓原始 SSE（不经过 SDK），核对字段与版本字段，
  并与采集器实际消费的字段对照。

用法（crater，serve venv）::

    python labs/L9/agent_trace_spans.py collect \
        --base-url http://127.0.0.1:8011/v1 --out "$OUT" \
        --n-per-class 100 --concurrency 16 --arrival-rate 0
    python labs/L9/agent_trace_spans.py graph --run "$OUT" --out "$OUT/graph"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

# 只读工具可以并行；有副作用的工具必须按模型给出的顺序串行，否则同一工作区会互相踩。
READ_ONLY_TOOLS = {"calculate", "search_corpus", "list_files", "read_file"}

# 采集器真正消费的 SSE 字段；sse-audit 用它和原始字节对照，避免"字段其实不存在"。
CONSUMED_CHUNK_FIELDS = ("id", "choices", "usage")
CONSUMED_DELTA_FIELDS = ("content", "reasoning_content", "reasoning", "tool_calls")
CONSUMED_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_tokens_details.cached_tokens",
    "completion_tokens_details.reasoning_tokens",
)


class Clock:
    """全进程唯一的单调时钟；所有 span 的 ``t_ms`` 都来自它。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


CLOCK = Clock()

# vLLM 0.29.0 的 SSE 把思考文本放在 ``delta.reasoning``；部分 OpenAI 兼容端用
# ``delta.reasoning_content``。采集器两个都读，并把命中的字段名记进 span，避免"读了一个
# 不存在的字段"这种静默失效（早期 9.1 轨迹的 reasoning_chars 因此恒为 0，见 sse-audit）。
REASONING_FIELDS = ("reasoning_content", "reasoning")


def delta_reasoning(delta) -> tuple[str | None, str | None]:
    """返回 (思考文本, 命中字段名)；两个字段都不存在时返回 (None, None)。"""
    for name in REASONING_FIELDS:
        v = getattr(delta, name, None)
        if v:
            return v, name
    extra = getattr(delta, "model_extra", None) or {}
    for name in REASONING_FIELDS:
        v = extra.get(name)
        if v:
            return v, name
    return None, None


class SpanSink:
    """把 span 落盘；asyncio 单线程模型下无需加锁，逐行 flush 便于崩溃后保留前缀。"""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.fh = open(path, "w", encoding="utf-8")
        self.count = 0

    def emit(self, span: str, *, session_id: str, task_id: str, task_class: str,
             node_id: str | None = None, parent_node_id: str | None = None,
             attempt: int | None = None, t_ms: float | None = None, **kw) -> dict:
        row = {
            "span": span,
            "t_ms": round(CLOCK.ms() if t_ms is None else t_ms, 3),
            "session_id": session_id,
            "task_id": task_id,
            "task_class": task_class,
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "attempt": attempt,
        }
        row.update(kw)
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.count += 1
        return row

    def close(self) -> None:
        self.fh.close()


# --------------------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------------------

async def run_one_task(
    client,
    task: dict,
    out_dir: pathlib.Path,
    *,
    model: str,
    concurrency_tag: int,
    max_turns: int,
    max_tokens: int,
    thinking: bool,
    retries: int,
    temperature: float,
    sem: asyncio.Semaphore,
    sink: SpanSink,
    requests_fh,
    task_timeout: float,
    arrival_ms: float,
) -> dict:
    """跑一个会话：多轮模型调用 + 并行只读工具 / 串行副作用工具 + 最终评分。"""
    session_id = f"{task['task_id']}-c{concurrency_tag}-{uuid.uuid4().hex[:8]}"
    sandbox = out_dir / "sandboxes" / session_id
    sandbox.mkdir(parents=True, exist_ok=True)
    env = T.make_env(task, sandbox)
    base = {"session_id": session_id, "task_id": task["task_id"], "task_class": task["task_class"]}
    sink.emit("task_arrive", t_ms=arrival_ms, **base)

    async def _run() -> dict:
        prompt = task["prompt"]
        if task["task_class"] == "codefix":
            failure = T.run_repo_tests(sandbox)
            prompt += (
                "\n\n当前测试失败的输出如下（供定位用）：\n```\n" + failure["tail"] + "\n```"
            )
        messages = [
            {"role": "system", "content": T.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        turns: list[dict] = []
        tool_total_ms = 0.0
        tool_parallel_gain_ms = 0.0
        total_tool_calls = 0
        retry_total = 0
        error: str | None = None
        final_text = ""
        prev_prompt_text = ""
        session_t0 = CLOCK.ms()
        sink.emit("task_start", **base, t_ms=session_t0)

        for turn in range(1, max_turns + 1):
            node_id = f"{task['task_id']}#t{turn}"
            parent = f"{task['task_id']}#t{turn - 1}" if turn > 1 else None
            nbase = dict(base, node_id=node_id, parent_node_id=parent)
            t_ready = CLOCK.ms()
            sink.emit("node_ready", t_ms=t_ready, **nbase)
            requests_fh.write(
                json.dumps({**base, "turn": turn, "thinking": thinking,
                            "t_global_ms": round(t_ready, 3), "messages": messages},
                           ensure_ascii=False) + "\n"
            )
            requests_fh.flush()

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_frags: dict[int, dict] = {}
            ttft_ms = first_reasoning_ms = first_final_ms = first_tool_ms = None
            prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = None
            finish_reason = None
            engine_request_id = None
            engine_call_id = None
            reasoning_field = None
            chunk_count = delta_chunks = 0
            attempt = 0
            t0 = CLOCK.ms()
            while True:
                attempt += 1
                content_parts.clear()
                reasoning_parts.clear()
                tool_frags.clear()
                ttft_ms = first_reasoning_ms = first_final_ms = first_tool_ms = None
                chunk_count = delta_chunks = 0
                engine_request_id = None
                reasoning_field = None
                sink.emit("model_start", attempt=attempt, t_ms=CLOCK.ms(), **nbase)
                try:
                    stream = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=env.tools,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                    )
                    async for chunk in stream:
                        chunk_count += 1
                        if chunk.id:
                            engine_request_id = chunk.id
                        if chunk.usage is not None:
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens
                            details = getattr(chunk.usage, "prompt_tokens_details", None)
                            if details is not None:
                                cached_tokens = getattr(details, "cached_tokens", None)
                            cdetails = getattr(chunk.usage, "completion_tokens_details", None)
                            if cdetails is not None:
                                reasoning_tokens = getattr(cdetails, "reasoning_tokens", None)
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        delta = choice.delta
                        if delta is None:
                            continue
                        piece_c = getattr(delta, "content", None)
                        piece_r, rfield = delta_reasoning(delta)
                        if rfield:
                            reasoning_field = rfield
                        tcs = getattr(delta, "tool_calls", None)
                        if piece_c or piece_r or tcs:
                            delta_chunks += 1
                        now = CLOCK.ms() - t0
                        if ttft_ms is None and (piece_c or piece_r or tcs):
                            ttft_ms = now
                        if piece_c:
                            if first_final_ms is None:
                                first_final_ms = now
                            content_parts.append(piece_c)
                        if piece_r:
                            if first_reasoning_ms is None:
                                first_reasoning_ms = now
                            reasoning_parts.append(piece_r)
                        if tcs:
                            if first_tool_ms is None:
                                first_tool_ms = now
                            for tc in tcs:
                                idx = tc.index or 0
                                slot = tool_frags.setdefault(
                                    idx, {"id": None, "name": None, "arguments": ""}
                                )
                                if tc.id:
                                    slot["id"] = tc.id
                                    engine_call_id = engine_call_id or tc.id
                                fn = getattr(tc, "function", None)
                                if fn is not None:
                                    if fn.name:
                                        slot["name"] = fn.name
                                    if fn.arguments:
                                        slot["arguments"] += fn.arguments
                    break
                except Exception as exc:  # noqa: BLE001
                    sink.emit("model_error", attempt=attempt, t_ms=CLOCK.ms(),
                              error=f"{type(exc).__name__}: {exc}", **nbase)
                    if attempt > retries:
                        error = f"{type(exc).__name__}: {exc}"
                        break
                    await asyncio.sleep(0.5 * attempt)
            e2e_ms = CLOCK.ms() - t0
            retry_total += attempt - 1
            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            for name, at in (("model_first_reasoning", first_reasoning_ms),
                             ("model_first_final", first_final_ms),
                             ("model_first_tool", first_tool_ms)):
                if at is not None:
                    sink.emit(name, t_ms=t0 + at, **nbase)
            sink.emit(
                "model_end", attempt=attempt, t_ms=CLOCK.ms(), **nbase,
                engine_request_id=engine_request_id,
                engine_call_id=engine_call_id,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                chunk_count=chunk_count,
                delta_chunks=delta_chunks,
                reasoning_field=reasoning_field,
                reasoning_chars=len(reasoning),
                content_chars=len(content),
                ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
                first_reasoning_ms=round(first_reasoning_ms, 3) if first_reasoning_ms is not None else None,
                first_final_ms=round(first_final_ms, 3) if first_final_ms is not None else None,
                first_tool_ms=round(first_tool_ms, 3) if first_tool_ms is not None else None,
                missing_delta=int(ttft_ms is None),
                client_e2e_ms=round(e2e_ms, 3),
                stream_tail_ms=round(e2e_ms - (ttft_ms or 0.0), 3),
                error=error,
            )

            calls = [tool_frags[i] for i in sorted(tool_frags)]
            node = {
                **nbase,
                "turn": turn,
                "engine_request_id": engine_request_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cached_tokens": cached_tokens,
                "reasoning_field": reasoning_field,
                "reasoning_chars": len(reasoning),
                "content_chars": len(content),
                "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                "first_reasoning_ms": round(first_reasoning_ms, 3) if first_reasoning_ms is not None else None,
                "first_final_ms": round(first_final_ms, 3) if first_final_ms is not None else None,
                "first_tool_ms": round(first_tool_ms, 3) if first_tool_ms is not None else None,
                "missing_delta": int(ttft_ms is None),
                "client_e2e_ms": round(e2e_ms, 3),
                "stream_tail_ms": round(e2e_ms - (ttft_ms or 0.0), 3),
                "finish_reason": finish_reason,
                "tool_names": [c["name"] for c in calls],
                "retries": attempt - 1,
                "error": error,
            }

            if error is not None or not calls:
                final_text = content
            if error is not None or not calls or turn == max_turns:
                if not calls and content:
                    final_text = content
                node["tool_wait_ms"] = 0.0
                turns.append(node)
                break

            # 工具执行：只读并行（fork），副作用串行；回填顺序始终按模型调用顺序（join）。
            assistant_msg = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": c["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for i, c in enumerate(calls)
                ],
            }
            messages.append(assistant_msg)
            call_ids = [c["id"] or f"call_{i}" for i, c in enumerate(calls)]
            t_tool = CLOCK.ms()
            results: list[str | None] = [None] * len(calls)

            def _invoke(i: int) -> str:
                c = calls[i]
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    return f"ERROR: bad JSON arguments: {exc}"
                if not isinstance(args, dict):
                    return "ERROR: arguments must be a JSON object"
                return env.call(c["name"], args)

            durations: list[float] = [0.0] * len(calls)

            def _wrap(i: int) -> str:
                tid = f"{node_id}/tool{i}"
                t_start = CLOCK.ms()
                sink.emit("tool_start", t_ms=t_start, **nbase,
                          tool_call_id=tid, engine_call_id=call_ids[i],
                          tool_name=calls[i]["name"],
                          parallel=int(calls[i]["name"] in READ_ONLY_TOOLS))
                try:
                    out = _invoke(i)
                except Exception as exc:  # noqa: BLE001
                    out = f"ERROR: {type(exc).__name__}: {exc}"
                t_end = CLOCK.ms()
                durations[i] = t_end - t_start
                sink.emit("tool_end", t_ms=t_end, **nbase,
                          tool_call_id=tid, engine_call_id=call_ids[i],
                          tool_name=calls[i]["name"],
                          parallel=int(calls[i]["name"] in READ_ONLY_TOOLS),
                          duration_ms=round(t_end - t_start, 3),
                          result_chars=len(str(out)), ok=int(not str(out).startswith("ERROR")))
                return out

            parallel = [i for i, c in enumerate(calls) if c["name"] in READ_ONLY_TOOLS]
            serial = [i for i, c in enumerate(calls) if c["name"] not in READ_ONLY_TOOLS]
            if parallel:
                outs = await asyncio.gather(*[asyncio.to_thread(_wrap, i) for i in parallel])
                for i, out in zip(parallel, outs):
                    results[i] = out
            for i in serial:
                results[i] = _wrap(i)
            tool_ms = CLOCK.ms() - t_tool
            tool_total_ms += tool_ms
            total_tool_calls += len(calls)
            # join：回填顺序始终按模型给出的调用顺序，与并行执行无关。
            for i in range(len(calls)):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_ids[i],
                        "content": str(results[i])[:4000],
                    }
                )
            node["tool_wait_ms"] = round(tool_ms, 3)
            node["tool_names"] = [c["name"] for c in calls]
            node["parallel_tool_calls"] = len(parallel)
            node["serial_tool_calls"] = len(serial)
            node["tool_durations_ms"] = [round(x, 3) for x in durations]
            # fork/join 并行收益：只读工具时长之和 减去 它们真实占用的墙钟跨度（下界用 max 时长）
            if len(parallel) > 1:
                par_dur = [durations[i] for i in parallel]
                tool_parallel_gain_ms += max(0.0, sum(par_dur) - max(par_dur))
            turns.append(node)

        session_end = CLOCK.ms()
        scored = env.score(final_text)
        sink.emit("task_end", t_ms=session_end, **base,
                  termination=("error" if error else ("missing_delta" if any(t["missing_delta"] for t in turns)
                                                      else ("final" if turns and not turns[-1]["tool_names"] else "max_turns"))),
                  turns=len(turns), score=scored.get("score"), error=error)
        return {
            **base,
            "concurrency": concurrency_tag,
            "thinking": thinking,
            "turns": len(turns),
            "tool_calls": total_tool_calls,
            "tool_wait_ms": round(tool_total_ms, 3),
            "tool_parallel_gain_ms": round(tool_parallel_gain_ms, 3),
            "retries": retry_total,
            "prompt_tokens": sum(t["prompt_tokens"] or 0 for t in turns),
            "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns),
            "reasoning_tokens": sum(t["reasoning_tokens"] or 0 for t in turns),
            "cached_tokens": sum(t["cached_tokens"] or 0 for t in turns),
            "missing_delta_turns": sum(t["missing_delta"] for t in turns),
            "node_ids": [t["node_id"] for t in turns],
            "parent_edges": [[t["parent_node_id"], t["node_id"]] for t in turns if t["parent_node_id"]],
            "tool_edges": [
                [t["node_id"], f"{t['node_id']}/tool{i}"]
                for t in turns for i in range(len(t["tool_names"]))
            ],
            "t_arrive_ms": round(arrival_ms, 3),
            "t_start_ms": round(session_t0, 3),
            "t_end_ms": round(session_end, 3),
            "task_wall_ms": round(session_end - session_t0, 3),
            "sum_node_ms": round(sum(t["client_e2e_ms"] for t in turns), 3),
            "error": error,
            "final_text": final_text[-400:],
            **scored,
        }

    async def _admit_and_run() -> dict:
        # 到达已在到达器里打点；这里才开始竞争并发槽，admission delay = 排队等待。
        async with sem:
            sink.emit("task_admit", **base)
            return await _run()

    try:
        if task_timeout > 0:
            return await asyncio.wait_for(_admit_and_run(), timeout=task_timeout)
        return await _admit_and_run()
    except (asyncio.TimeoutError, TimeoutError):
        sink.emit("task_end", **base, termination="timeout", turns=0, score=None,
                  error=f"task timeout after {task_timeout}s")
        return {
            **base, "concurrency": concurrency_tag, "thinking": thinking, "turns": 0,
            "tool_calls": 0, "tool_wait_ms": 0.0, "tool_parallel_gain_ms": 0.0, "retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "cached_tokens": 0, "missing_delta_turns": 0, "node_ids": [], "parent_edges": [],
            "tool_edges": [], "t_arrive_ms": round(arrival_ms, 3),
            "t_start_ms": round(arrival_ms, 3), "t_end_ms": round(CLOCK.ms(), 3),
            "task_wall_ms": round(CLOCK.ms() - arrival_ms, 3), "sum_node_ms": 0.0,
            "error": f"task timeout after {task_timeout}s", "termination": "timeout",
            "final_text": "", "score": 0.0,
        }


async def amain_collect(args) -> int:
    from openai import AsyncOpenAI

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict] = []
    for cls in args.classes.split(","):
        cls = cls.strip()
        tasks = T.build_tasks(cls, args.n_per_class, args.seed)
        plan.extend(tasks)
        print(f"[plan] {cls}: {len(tasks)} tasks", flush=True)

    serializable = [{k: v for k, v in t.items() if not k.startswith("_")} for t in plan]
    (out_dir / "tasks.json").write_text(
        json.dumps({"seed": args.seed, "n_per_class": args.n_per_class, "classes": args.classes,
                    "max_turns": args.max_turns, "max_tokens": args.max_tokens,
                    "thinking_classes": args.thinking_classes, "tasks": serializable},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    thinking_set = {c.strip() for c in args.thinking_classes.split(",") if c.strip()}
    sink = SpanSink(out_dir / "spans.jsonl")
    sessions: list[dict] = []
    loop = asyncio.get_running_loop()

    with open(out_dir / "requests.jsonl", "w", encoding="utf-8") as req_fh:
        coros: list[asyncio.Task] = []
        if args.arrival_rate > 0:
            rng = random.Random(args.seed)
            gaps = [rng.expovariate(args.arrival_rate) for _ in plan]
        else:
            gaps = [0.0] * len(plan)
        t_next = loop.time()
        for task, gap in zip(plan, gaps):
            t_next += gap
            wait = t_next - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            arrival_ms = CLOCK.ms()
            coros.append(asyncio.create_task(
                run_one_task(
                    client, task, out_dir, model=args.model, concurrency_tag=args.concurrency,
                    max_turns=args.max_turns, max_tokens=args.max_tokens,
                    thinking=task["task_class"] in thinking_set, retries=args.retries,
                    temperature=args.temperature, sem=sem, sink=sink, requests_fh=req_fh,
                    task_timeout=args.task_timeout, arrival_ms=arrival_ms,
                )
            ))
        done = 0
        for coro in asyncio.as_completed(coros):
            sessions.append(await coro)
            done += 1
            if done % 10 == 0 or done == len(coros):
                ok = sum(1 for s in sessions if s.get("score") == 1.0)
                print(f"[progress] {done}/{len(coros)} score={ok}", flush=True)
    await client.close()
    sink.close()

    with open(out_dir / "sessions.jsonl", "w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_dir / "nodes.jsonl", "w", encoding="utf-8") as fh:
        seen = set()
        for line in open(out_dir / "spans.jsonl", encoding="utf-8"):
            row = json.loads(line)
            if row["span"] != "model_end":
                continue
            key = row["node_id"]
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_class: dict[str, list[dict]] = {}
    for s in sessions:
        by_class.setdefault(s["task_class"], []).append(s)
    term = {}
    for line in open(out_dir / "spans.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["span"] == "task_end":
            term[row.get("termination")] = term.get(row.get("termination"), 0) + 1
    summary = {
        "config": {
            "base_url": args.base_url, "model": args.model, "concurrency": args.concurrency,
            "n_per_class": args.n_per_class, "max_turns": args.max_turns,
            "max_tokens": args.max_tokens, "seed": args.seed,
            "thinking_classes": sorted(thinking_set), "temperature": args.temperature,
            "arrival_rate_rps": args.arrival_rate, "task_timeout_s": args.task_timeout,
        },
        "spans": sink.count,
        "tasks": len(sessions),
        "termination_reasons": term,
        "denominator_note": "所有任务（含 timeout / missing_delta / error / max_turns）都进入分母",
        "per_class": {
            cls: {
                "tasks": len(rows),
                "scored": sum(1 for r in rows if r.get("score") == 1.0),
                "failed": sum(1 for r in rows if r.get("score") != 1.0),
                "error_tasks": sum(1 for r in rows if r.get("error")),
                "missing_delta_turns": sum(r["missing_delta_turns"] for r in rows),
                "turns": sum(r["turns"] for r in rows),
                "tool_calls": sum(r["tool_calls"] for r in rows),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                "completion_tokens": sum(r["completion_tokens"] for r in rows),
                "reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
                "cached_tokens": sum(r["cached_tokens"] for r in rows),
            }
            for cls, rows in sorted(by_class.items())
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


# --------------------------------------------------------------------------------------
# graph：串行任务与 fork/join 任务的事件图、关键路径
# --------------------------------------------------------------------------------------

def load_spans(run: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in open(run / "spans.jsonl", encoding="utf-8")]


def build_session_graph(spans: list[dict], session_id: str) -> dict:
    """把一条会话的 span 折成节点图：每个 model 轮次是一个节点，工具是它的子节点。"""
    rows = sorted([s for s in spans if s["session_id"] == session_id], key=lambda s: s["t_ms"])
    nodes: dict[str, dict] = {}
    order: list[str] = []
    for s in rows:
        if s["span"] == "node_ready":
            nid = s["node_id"]
            nodes[nid] = {
                "node_id": nid, "parent": s["parent_node_id"], "t_ready": s["t_ms"],
                "t_start": None, "t_first_output": None, "t_end": None,
                "first_reasoning": None, "first_final": None, "first_tool": None,
                "tools": [], "prompt_tokens": None, "completion_tokens": None,
                "cached_tokens": None, "missing_delta": None,
            }
            order.append(nid)
        elif s["span"] == "model_start" and s["node_id"] in nodes:
            nodes[s["node_id"]]["t_start"] = s["t_ms"]
        elif s["span"].startswith("model_first_") and s["node_id"] in nodes:
            key = s["span"].replace("model_first_", "first_")
            nodes[s["node_id"]][key] = s["t_ms"]
            if nodes[s["node_id"]]["t_first_output"] is None:
                nodes[s["node_id"]]["t_first_output"] = s["t_ms"]
        elif s["span"] == "model_end" and s["node_id"] in nodes:
            n = nodes[s["node_id"]]
            n["t_end"] = s["t_ms"]
            n["prompt_tokens"] = s["prompt_tokens"]
            n["completion_tokens"] = s["completion_tokens"]
            n["cached_tokens"] = s["cached_tokens"]
            n["missing_delta"] = s["missing_delta"]
            n["finish_reason"] = s["finish_reason"]
        elif s["span"] == "tool_start" and s["node_id"] in nodes:
            nodes[s["node_id"]]["tools"].append(
                {"tool_call_id": s["tool_call_id"], "name": s["tool_name"],
                 "parallel": s["parallel"], "t_start": s["t_ms"], "t_end": None, "duration_ms": None}
            )
        elif s["span"] == "tool_end" and s["node_id"] in nodes:
            for t in nodes[s["node_id"]]["tools"]:
                if t["tool_call_id"] == s["tool_call_id"]:
                    t["t_end"] = s["t_ms"]
                    t["duration_ms"] = s["duration_ms"]
    ends = [s["t_ms"] for s in rows if s["span"] == "task_end"]
    arrives = [s["t_ms"] for s in rows if s["span"] == "task_arrive"]
    admits = [s["t_ms"] for s in rows if s["span"] == "task_admit"]

    # 关键路径：串行链沿节点累加，fork 处取最长子路径（不是各分支求和）。
    memo: dict[str, float] = {}

    def longest(nid: str) -> float:
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        own = max(0.0, (n["t_end"] or n["t_start"] or n["t_ready"]) - n["t_start"]) if n["t_start"] else 0.0
        children = [c for c in order if nodes[c]["parent"] == nid]
        best = max([longest(c) for c in children], default=0.0)
        memo[nid] = own + best
        return memo[nid]

    roots = [nid for nid in order if not nodes[nid]["parent"]]
    critical_ms = max([longest(r) for r in roots], default=0.0)
    sum_node_ms = sum(max(0.0, (n["t_end"] or 0) - (n["t_start"] or 0)) for n in nodes.values())
    sum_tool_ms = sum(t["duration_ms"] or 0.0 for n in nodes.values() for t in n["tools"])
    fork_nodes = []
    for nid in order:
        par = [t for t in nodes[nid]["tools"] if t["parallel"]]
        if len(par) > 1:
            span = max(t["t_end"] for t in par) - min(t["t_start"] for t in par)
            fork_nodes.append({
                "node_id": nid,
                "tools": [t["name"] for t in par],
                "tool_durations_ms": [t["duration_ms"] for t in par],
                "fork_wall_ms": round(span, 3),
                "sum_tool_ms": round(sum(t["duration_ms"] for t in par), 3),
                "join_saving_ms": round(sum(t["duration_ms"] for t in par) - span, 3),
            })
    return {
        "session_id": session_id,
        "task_id": rows[0]["task_id"] if rows else None,
        "task_class": rows[0]["task_class"] if rows else None,
        "t_arrive_ms": arrives[0] if arrives else None,
        "admission_delay_ms": round(admits[0] - arrives[0], 3) if arrives and admits else None,
        "nodes": [nodes[nid] for nid in order],
        "edges": [[nodes[nid]["parent"], nid] for nid in order if nodes[nid]["parent"]],
        "tool_edges": [[n["node_id"], t["tool_call_id"]] for n in nodes.values() for t in n["tools"]],
        "serial_chain": all(len(n["tools"]) <= 1 for n in nodes.values()) and not fork_nodes,
        "fork_join_nodes": fork_nodes,
        "critical_path_ms": round(critical_ms, 3),
        "sum_node_ms": round(sum_node_ms, 3),
        "sum_tool_ms": round(sum_tool_ms, 3),
        "task_wall_ms": round((ends[0] - admits[0]), 3) if ends and admits else None,
        "note": "关键路径在 fork 处取最长分支；sum_node_ms 把所有并行节点直接相加，只有串行链两者才相等",
    }


def cmd_graph(args) -> int:
    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spans = load_spans(run)
    by_session: dict[str, list[dict]] = {}
    for s in spans:
        by_session.setdefault(s["session_id"], []).append(s)
    # 挑"最长纯串行链"和"并行工具最多、join 收益最大"的会话作为示例，而不是遇到即停：
    # 单看一条 2 节点的小 fork 说明不了并行区间与关键路径的关系。
    serial_pick = fork_pick = None
    serial_score = fork_score = -1.0
    fork_count = 0
    for sid in by_session:
        g = build_session_graph(spans, sid)
        if not g["nodes"]:
            continue
        if g["serial_chain"]:
            score = len(g["nodes"])
            if g["critical_path_ms"]:
                score += min(1.0, g["critical_path_ms"] / 10000.0)
            if score > serial_score:
                serial_pick, serial_score = g, score
        else:
            fork_count += 1
            par = [len(f["tools"]) for f in g["fork_join_nodes"]]
            score = (max(par) if par else 0) * 1000 + sum(f["join_saving_ms"] for f in g["fork_join_nodes"])
            if score > fork_score:
                fork_pick, fork_score = g, score
    result = {
        "run": str(run),
        "sessions": len(by_session),
        "sessions_with_parallel_readonly_tools": fork_count,
        "serial_example": serial_pick or {"note": "本次轨迹里没有纯串行会话"},
        "fork_join_example": fork_pick or {"note": "本次轨迹里没有并行只读工具的会话"},
        "first_output_definition": [
            "model_first_reasoning", "model_first_final", "model_first_tool",
        ],
        "ttft_definition": "首个任意 delta（三者中最早出现的那个）；不是 prefill",
        "stream_tail_definition": "client_e2e_ms - ttft_ms，客户端观测的流式尾段；不是 GPU decode",
    }
    (out / "graph.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) or "nodes" not in v else
                          {kk: vv for kk, vv in v.items() if kk not in ("nodes",)})
                      for k, v in result.items()}, ensure_ascii=False, indent=1)[:5000])
    return 0


# --------------------------------------------------------------------------------------
# sse-audit：不经过 SDK 的原始 SSE 抓取与字段核对
# --------------------------------------------------------------------------------------

async def _probe_one(base_url: str, model: str, messages: list[dict], tools, out_path: pathlib.Path,
                     thinking: bool) -> dict:
    import httpx

    body = {
        "model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
        "temperature": 0.0, "max_tokens": 256, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    raw = bytearray()
    async with httpx.AsyncClient(timeout=300.0) as cli:
        async with cli.stream("POST", base_url.rstrip("/") + "/chat/completions", json=body) as resp:
            status = resp.status_code
            headers = {k: v for k, v in resp.headers.items()}
            async for piece in resp.aiter_bytes():
                raw.extend(piece)
    out_path.write_bytes(bytes(raw))
    return {"status": status, "bytes": len(raw), "headers": headers}


def _parse_raw_sse(data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace")
    events, chunks, done = [], 0, 0
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev = None
        payload = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                payload.append(line[5:].lstrip())
        body = "\n".join(payload).strip()
        if body == "[DONE]":
            done += 1
            events.append({"event": ev, "kind": "done"})
            continue
        if not body:
            events.append({"event": ev, "kind": "other"})
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            events.append({"event": ev, "kind": "unparsed", "sample": body[:120]})
            continue
        chunks += 1
        events.append({"event": ev, "kind": "chunk", "keys": sorted(obj.keys()), "obj": obj})
    chunk_keys: set[str] = set()
    delta_keys: set[str] = set()
    usage_keys: set[str] = set()
    finish_reasons: list[str] = []
    models: set[str] = set()
    fingerprints: set[str] = set()
    ids: set[str] = set()
    for e in events:
        if e["kind"] != "chunk":
            continue
        obj = e["obj"]
        chunk_keys.update(obj.keys())
        models.add(str(obj.get("model")))
        if "system_fingerprint" in obj:
            fingerprints.add(str(obj.get("system_fingerprint")))
        if obj.get("id"):
            ids.add(str(obj["id"]))
        if isinstance(obj.get("usage"), dict):
            usage_keys.update(obj["usage"].keys())
            for k, v in obj["usage"].items():
                if isinstance(v, dict):
                    usage_keys.update(f"{k}.{kk}" for kk in v.keys())
        for ch in obj.get("choices") or []:
            if ch.get("finish_reason"):
                finish_reasons.append(ch["finish_reason"])
            d = ch.get("delta")
            if isinstance(d, dict):
                delta_keys.update(d.keys())
    consumed_ok = {
        "chunk.id": "id" in chunk_keys,
        "chunk.choices": "choices" in chunk_keys,
        "chunk.usage": "usage" in chunk_keys,
        "delta.content": "content" in delta_keys,
        "delta.reasoning_content": "reasoning_content" in delta_keys,
        "delta.reasoning": "reasoning" in delta_keys,
        "delta.tool_calls": "tool_calls" in delta_keys,
        "usage.prompt_tokens": "prompt_tokens" in usage_keys,
        "usage.completion_tokens": "completion_tokens" in usage_keys,
        "usage.prompt_tokens_details.cached_tokens": "prompt_tokens_details.cached_tokens" in usage_keys,
        "usage.completion_tokens_details.reasoning_tokens": "completion_tokens_details.reasoning_tokens" in usage_keys,
    }
    # "两个字段读到一个即可"：reasoning / reasoning_content 是同一语义的两种拼法。
    if not (consumed_ok["delta.reasoning"] or consumed_ok["delta.reasoning_content"]):
        consumed_ok["delta.reasoning*"] = False
    else:
        consumed_ok["delta.reasoning*"] = True
    return {
        "chunks": chunks, "done_markers": done, "events": len(events),
        "chunk_keys": sorted(chunk_keys), "delta_keys": sorted(delta_keys),
        "usage_keys": sorted(usage_keys), "finish_reasons": sorted(set(finish_reasons)),
        "models": sorted(models), "system_fingerprints": sorted(fingerprints),
        "request_ids": sorted(ids)[:5], "request_id_count": len(ids),
        "consumed_field_present": consumed_ok,
        "missing_consumed_fields": sorted(k for k, v in consumed_ok.items() if not v),
    }


async def amain_sse_audit(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "sse_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.probe > 0:
        # 探针必须拿到真实任务对象（检索任务带 BM25 索引），因此按 seed 重建而不用 tasks.json。
        tasks: list[dict] = []
        for cls in args.classes.split(","):
            tasks.extend(T.build_tasks(cls.strip(), args.n_per_class, args.seed))
        for i, task in enumerate(tasks[: args.probe]):
            sandbox = out / "probe-sandboxes" / f"probe{i:02d}"
            sandbox.mkdir(parents=True, exist_ok=True)
            env = T.make_env(task, sandbox)
            messages = [
                {"role": "system", "content": T.SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ]
            info = await _probe_one(args.base_url, args.model, messages, env.tools,
                                    raw_dir / f"req{i:02d}.txt", task["task_class"] == "compute")
            print(f"[probe {i}] {info['status']} {info['bytes']} bytes", flush=True)
    report = {"probes": []}
    for p in sorted(raw_dir.glob("*.txt")):
        parsed = _parse_raw_sse(p.read_bytes())
        parsed["file"] = p.name
        report["probes"].append(parsed)
    report["engine"] = {"base_url": args.base_url, "model": args.model}
    report["consumed_fields_checked"] = list(CONSUMED_CHUNK_FIELDS) + list(CONSUMED_DELTA_FIELDS) + list(CONSUMED_USAGE_FIELDS)
    required = ("chunk.id", "chunk.choices", "delta.content", "delta.reasoning*",
                "usage.prompt_tokens", "usage.completion_tokens")
    report["required_fields"] = list(required)
    report["missing_required_fields"] = sorted({k for p in report["probes"]
                                                for k in p["missing_consumed_fields"] if k in required})
    report["all_required_present"] = not report["missing_required_fields"]
    report["reasoning_spelling_observed"] = sorted({k for p in report["probes"]
                                                    for k in ("reasoning", "reasoning_content")
                                                    if p["consumed_field_present"].get("delta." + k)})
    report["all_consumed_present"] = all(not p["missing_consumed_fields"] for p in report["probes"])
    (out / "sse_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, ensure_ascii=False, indent=1))
    for p in report["probes"]:
        print(json.dumps({k: v for k, v in p.items() if k != "obj"}, ensure_ascii=False)[:800])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1 统一 span 采集 / 任务图 / 原始 SSE 审计")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default="compute,retrieval,codefix")
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--task-timeout", type=float, default=0.0, help=">0 时给每个任务设总 deadline")
    p.add_argument("--arrival-rate", type=float, default=0.0, help=">0 时按泊松绝对时刻到达")
    p.add_argument("--thinking-classes", default="compute")
    p.set_defaults(func=lambda a: asyncio.run(amain_collect(a)))

    p = sub.add_parser("graph")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("sse-audit")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default="compute,retrieval,codefix")
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", type=int, default=0, help=">0 时直连引擎抓 N 个请求的原始 SSE")
    p.set_defaults(func=lambda a: asyncio.run(amain_sse_audit(a)))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
