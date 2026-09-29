#!/usr/bin/env python3
"""L9.1 任务 A：真实 agent 执行轨迹采集。

对 vLLM 的 OpenAI 兼容端点跑三类任务（计算 / 多轮检索 / 代码修复），把每一次模型调用
记成一条事件，把每一个会话记成一条结果。字段设计成可以直接支撑 9.1-B 的画像与 9.1-C
的重放：

* ``session_id`` / ``turn`` / ``tool_call_ids``：会话结构；
* ``prompt_tokens`` / ``completion_tokens`` / ``reasoning_tokens`` / ``cached_tokens``：
  长度与真实前缀命中（来自引擎 usage，不是估算）；
* ``ttft_ms`` / ``decode_ms`` / ``e2e_ms``：阶段时间；
* ``tool_wait_ms`` / ``tool_names`` / ``retries``：工具等待与重试；
* 会话末尾的 ``score``：任务得分（失败会话照样入库，不删）。

用法（crater，serve venv；先按 run_agent_trace.sh 起服务）::

    python labs/L9/agent_trace_collect.py \
        --base-url http://127.0.0.1:8011/v1 --out "$OUT"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402


def _now() -> float:
    return time.perf_counter()


# 采集开始时刻（perf_counter），给每条事件一个全局单调时间戳，便于按到达顺序重排。
_T0 = time.perf_counter()


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
    events_fh,
    requests_fh,
) -> dict:
    """跑一个会话：多轮模型调用 + 本地工具执行 + 最终评分。"""
    async with sem:
        session_id = f"{task['task_id']}-c{concurrency_tag}-{uuid.uuid4().hex[:8]}"
        sandbox = out_dir / "sandboxes" / session_id
        sandbox.mkdir(parents=True, exist_ok=True)
        env = T.make_env(task, sandbox)
        prompt = task["prompt"]
        if task["task_class"] == "codefix":
            failure = T.run_repo_tests(sandbox)
            prompt += (
                "\n\n当前测试失败的输出如下（供定位用）：\n```\n"
                + failure["tail"]
                + "\n```"
            )
        messages = [
            {"role": "system", "content": T.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        turns: list[dict] = []
        tool_total_ms = 0.0
        total_tool_calls = 0
        retry_total = 0
        error: str | None = None
        final_text = ""
        prev_prompt_text = ""
        session_t0 = _now()

        for turn in range(1, max_turns + 1):
            t_global_ms = (_now() - _T0) * 1000.0
            # 逐轮保存完整消息，供 trace_replay 原样重放（“原始请求可重放”）
            requests_fh.write(
                json.dumps(
                    {
                        "session_id": session_id,
                        "task_id": task["task_id"],
                        "task_class": task["task_class"],
                        "turn": turn,
                        "t_global_ms": round(t_global_ms, 3),
                        "thinking": thinking,
                        "messages": messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            requests_fh.flush()
            prompt_text = "\n".join(
                f"<{m['role']}>{m.get('content') or ''}" for m in messages
            )
            common = 0
            for a, b in zip(prev_prompt_text, prompt_text):
                if a != b:
                    break
                common += 1
            prev_prompt_text = prompt_text

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_frags: dict[int, dict] = {}
            ttft_ms = None
            # 三种"首输出"分开记：首 reasoning delta / 首 final delta / 首 tool delta。
            # 把三者合成一个 TTFT 会让"思考型"负载的 prefill 与 decode 边界消失。
            first_reasoning_ms = None
            first_final_ms = None
            first_tool_ms = None
            prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = None
            finish_reason = None
            request_id = None
            attempt = 0
            t0 = _now()
            while True:
                attempt += 1
                content_parts.clear()
                reasoning_parts.clear()
                tool_frags.clear()
                ttft_ms = None
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
                        if chunk.id:
                            request_id = chunk.id
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
                        piece_r = getattr(delta, "reasoning_content", None)
                        if ttft_ms is None and (piece_c or piece_r or getattr(delta, "tool_calls", None)):
                            ttft_ms = (_now() - t0) * 1000.0
                        now_ms = (_now() - t0) * 1000.0
                        if piece_c:
                            if first_final_ms is None:
                                first_final_ms = now_ms
                            content_parts.append(piece_c)
                        if piece_r:
                            if first_reasoning_ms is None:
                                first_reasoning_ms = now_ms
                            reasoning_parts.append(piece_r)
                        if getattr(delta, "tool_calls", None) and first_tool_ms is None:
                            first_tool_ms = now_ms
                        for tc in getattr(delta, "tool_calls", None) or []:
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
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt > retries:
                        error = f"{type(exc).__name__}: {exc}"
                        break
                    await asyncio.sleep(0.5 * attempt)
            e2e_ms = (_now() - t0) * 1000.0
            retry_total += attempt - 1

            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            calls = [tool_frags[i] for i in sorted(tool_frags)]
            event = {
                "session_id": session_id,
                "task_id": task["task_id"],
                "task_class": task["task_class"],
                "turn": turn,
                "t_global_ms": round(t_global_ms, 3),
                "request_id": request_id,
                "thinking": thinking,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_chars": len(reasoning),
                "cached_tokens": cached_tokens,
                "prompt_chars": len(prompt_text),
                "common_prefix_chars": common,
                "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                "first_reasoning_ms": round(first_reasoning_ms, 3) if first_reasoning_ms is not None else None,
                "first_final_ms": round(first_final_ms, 3) if first_final_ms is not None else None,
                "first_tool_ms": round(first_tool_ms, 3) if first_tool_ms is not None else None,
                "node": f"{task['task_id']}#t{turn}",
                "parent": (f"{task['task_id']}#t{turn - 1}" if turn > 1 else None),
                "attempt_index": attempt - 1,
                "e2e_ms": round(e2e_ms, 3),
                "decode_ms": round(e2e_ms - (ttft_ms or 0.0), 3),
                "finish_reason": finish_reason,
                "tool_names": [c["name"] for c in calls],
                "tool_arg_chars": [len(c["arguments"] or "") for c in calls],
                "tool_args_preview": [(c["arguments"] or "")[:200] for c in calls],
                "retries": attempt - 1,
                "error": error,
            }

            if error is not None or not calls:
                final_text = content
            if error is not None or not calls or turn == max_turns:
                if not calls and content:
                    final_text = content
                events_fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                events_fh.flush()
                turns.append(event)
                break

            # 有工具调用：执行工具，把 assistant 消息与工具结果追加进上下文
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
            names = []
            t_tool = _now()
            for c in calls:
                names.append(c["name"])
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    result = f"ERROR: bad JSON arguments: {exc}"
                    args = {}
                else:
                    if not isinstance(args, dict):
                        result = "ERROR: arguments must be a JSON object"
                    else:
                        result = env.call(c["name"], args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"] or "call_0",
                        "content": str(result)[:4000],
                    }
                )
            tool_ms = (_now() - t_tool) * 1000.0
            tool_total_ms += tool_ms
            total_tool_calls += len(calls)
            event["tool_wait_ms"] = round(tool_ms, 3)
            event["tool_names"] = names
            events_fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            events_fh.flush()
            turns.append(event)

        session_wall_ms = (_now() - session_t0) * 1000.0
        session_start_ms = (session_t0 - _T0) * 1000.0
        scored = env.score(final_text)
        return {
            "session_id": session_id,
            "task_id": task["task_id"],
            "task_class": task["task_class"],
            "concurrency": concurrency_tag,
            "thinking": thinking,
            "turns": len(turns),
            "tool_calls": total_tool_calls,
            "tool_wait_ms": round(tool_total_ms, 3),
            "retries": retry_total,
            "prompt_tokens": sum(t["prompt_tokens"] or 0 for t in turns),
            "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns),
            "reasoning_tokens": sum(t["reasoning_tokens"] or 0 for t in turns),
            "cached_tokens": sum(t["cached_tokens"] or 0 for t in turns),
            "node_ids": [f"{task['task_id']}#t{i}" for i in range(1, len(turns) + 1)],
            "parent_edges": [[f"{task['task_id']}#t{i - 1}", f"{task['task_id']}#t{i}"]
                             for i in range(2, len(turns) + 1)],
            "t_start_ms": round(session_start_ms, 3),
            "t_end_ms": round(session_start_ms + session_wall_ms, 3),
            "session_wall_ms": round(session_wall_ms, 3),
            "sum_e2e_ms": round(sum(t["e2e_ms"] for t in turns), 3),
            "error": error,
            "final_text": final_text[-400:],
            **scored,
        }


async def amain(args) -> int:
    from openai import AsyncOpenAI

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict] = []
    for cls in args.classes.split(","):
        cls = cls.strip()
        tasks = T.build_tasks(cls, args.n_per_class, args.seed)
        plan.extend(tasks)
        print(f"[plan] {cls}: {len(tasks)} tasks", flush=True)

    serializable = [
        {k: v for k, v in t.items() if not k.startswith("_")} for t in plan
    ]
    (out_dir / "tasks.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_per_class": args.n_per_class,
                "classes": args.classes,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "thinking_classes": args.thinking_classes,
                "tasks": serializable,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    thinking_set = {c.strip() for c in args.thinking_classes.split(",") if c.strip()}
    t0 = _now()
    sessions: list[dict] = []
    with open(out_dir / "events.jsonl", "w", encoding="utf-8") as fh, open(
        out_dir / "requests.jsonl", "w", encoding="utf-8"
    ) as req_fh:
        coros = [
            run_one_task(
                client,
                task,
                out_dir,
                model=args.model,
                concurrency_tag=args.concurrency,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                thinking=task["task_class"] in thinking_set,
                retries=args.retries,
                temperature=args.temperature,
                sem=sem,
                events_fh=fh,
                requests_fh=req_fh,
            )
            for task in plan
        ]
        done = 0
        for coro in asyncio.as_completed(coros):
            res = await coro
            sessions.append(res)
            done += 1
            if done % 10 == 0 or done == len(coros):
                ok = sum(1 for s in sessions if s.get("score") == 1.0)
                print(f"[progress] {done}/{len(coros)}  score={ok}", flush=True)
    wall = _now() - t0

    await client.close()
    with open(out_dir / "sessions.jsonl", "w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_class: dict[str, list[dict]] = {}
    for s in sessions:
        by_class.setdefault(s["task_class"], []).append(s)
    summary = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "n_per_class": args.n_per_class,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "thinking_classes": sorted(thinking_set),
            "temperature": args.temperature,
        },
        "wall_s": round(wall, 3),
        "sessions": len(sessions),
        "per_class": {
            cls: {
                "sessions": len(rows),
                "scored": sum(1 for r in rows if r.get("score") == 1.0),
                "failed_sessions": sum(1 for r in rows if r.get("score") != 1.0),
                "error_sessions": sum(1 for r in rows if r.get("error")),
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
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1 agent 轨迹采集")
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="compute,retrieval,codefix")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--thinking-classes", default="compute")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
