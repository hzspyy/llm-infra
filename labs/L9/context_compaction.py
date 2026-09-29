#!/usr/bin/env python3
"""L9.3 任务 C：上下文压缩作为"会改变模型输入"的近似策略，单独评分。

四条上下文策略，跑同一批多轮任务：

* ``full``：完整历史（不压缩的参照）；
* ``window``：只保留最近 k 轮（丢掉更早的轮次）；
* ``summary``：丢掉更早的轮次，但先用目标模型把"已处理的数字与当前累计值"压成一句话放进 system；
* ``external``：历史里的旧工具结果不再进 prompt，改为提供一个 ``recall_history`` 工具按需取回。

任务本身要求"知道全部历史才能算对"：每轮给一个整数，模型要把它们累加。这样
**压缩省下的 token 与丢掉的正确率可以同时被测量**，而不会退化成"只看 prompt 变短了"。

逐轮记录：prompt/cached token、TTFT、端到端、解析出的 TOTAL、是否正确、工具调用、压缩动作。
统计口径：正确率按**每轮**独立统计（不把多轮合成一个成功/失败），失败与截断不删样本。

用法::

    python labs/L9/context_compaction.py --base-url http://127.0.0.1:8020/v1 \
        --out DIR --sessions 12 --turns 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import re
import statistics
import time

SYSTEM = ("你是一个累加助手。每一轮我会给你一个整数，你要用 calculate 工具把它加到累计值上，"
          "并在最后一行写 `TOTAL: <累计值>`。")

CALC_TOOL = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算一个只含整数与 + - * 的算术表达式的精确整数值",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"], "additionalProperties": False},
    },
}]

RECALL_TOOL = [{
    "type": "function",
    "function": {
        "name": "recall_history",
        "description": "取回此前所有轮次给出的整数（按顺序）",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}]


def safe_eval(expr: str) -> int:
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.USub: operator.neg, ast.UAdd: operator.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


class ToolEnv:
    """被压缩的"外部状态"：数字清单与累计值都放在这里，不由 prompt 承载。"""

    def __init__(self, numbers: list[int]):
        self.numbers = numbers
        self.calls = {"calculate": 0, "recall_history": 0}

    @property
    def tools(self):
        return CALC_TOOL + RECALL_TOOL

    def call(self, name: str, args: dict):
        self.calls[name] = self.calls.get(name, 0) + 1
        if name == "calculate":
            try:
                return str(safe_eval(str(args.get("expression", ""))))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"
        if name == "recall_history":
            return json.dumps(self.numbers)
        return f"ERROR: unknown tool {name}"


async def stream_call(client, model, messages, tools, *, thinking=False, max_tokens=256):
    """一次流式调用，返回内容、工具调用、usage 与计时。"""
    content: list[str] = []
    calls: dict[int, dict] = {}
    ttft = None
    usage = None
    finish = None
    t0 = time.perf_counter()
    extra: dict = {"temperature": 0.0, "max_tokens": max_tokens, "stream": True,
                   "stream_options": {"include_usage": True},
                   "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
    if tools:
        # 空数组会被服务端拒绝（"tools must not be an empty array"），摘要调用因此不带工具
        extra.update(tools=tools, tool_choice="auto")
    stream = await client.chat.completions.create(model=model, messages=messages, **extra)
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            finish = ch.finish_reason
        d = ch.delta
        if d is None:
            continue
        if d.content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            content.append(d.content)
        for tc in d.tool_calls or []:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            slot = calls.setdefault(tc.index or 0, {"id": None, "name": None, "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if fn.name:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments
    e2e = (time.perf_counter() - t0) * 1000.0
    u = usage
    details = getattr(u, "prompt_tokens_details", None) if u else None
    return {
        "content": "".join(content),
        "calls": [calls[i] for i in sorted(calls)],
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
        "ttft_ms": round(ttft, 3) if ttft else None,
        "e2e_ms": round(e2e, 3),
        "finish_reason": finish,
    }


def parse_total(text: str) -> int | None:
    m = re.findall(r"TOTAL:\s*(-?\d+)", text or "")
    return int(m[-1]) if m else None


async def summarize_conversation(client, model, dropped: list[dict], thinking=False) -> str:
    """用目标模型把**被丢弃的真实对话**压成一句话（摘要成本计入统计）。

    摘要里保留什么由模型决定：如果它没记住累计值，后续轮次就会算错——这正是压缩策略的风险。
    """
    text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in dropped)
    prompt = ("把下面这段对话压缩成一句话，尽量保留已经给出的整数与当前累计值；"
              "只输出这一句话。\n\n" + text)
    out = await stream_call(client, model, [{"role": "user", "content": prompt}], [],
                            thinking=thinking, max_tokens=160)
    return (out["content"] or "").strip()


async def run_session(client, args, sid: int, numbers: list[int]) -> list[dict]:
    env = ToolEnv(numbers)
    running = 0
    rows: list[dict] = []
    base_history = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "开始累加，每轮我给一个整数。"}]
    for turn, x in enumerate(numbers, start=1):
        running += x
        if args.policy == "full":
            messages = base_history + [{"role": "user", "content": f"第 {turn} 个数是 {x}，请更新累计值。"}]
        elif args.policy == "window":
            keep = base_history[:2] + [m for m in base_history[2:] if m["role"] != "system"][-2 * args.window * 2:]
            messages = keep + [{"role": "user", "content": f"第 {turn} 个数是 {x}，请更新累计值。"}]
        elif args.policy == "summary":
            if turn > 1:
                # 摘要由模型对**被丢弃的真实对话**生成；是否保留住累计值由模型自己决定
                dropped = base_history[2:]
                summary = await summarize_conversation(client, args.model, dropped)
                rows.append({"session": sid, "turn": turn, "kind": "summary_cost"})
            else:
                summary = "尚无历史。"
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "system", "content": f"（历史摘要）{summary}"},
                        {"role": "user", "content": f"第 {turn} 个数是 {x}，请更新累计值。"}]
        elif args.policy == "external":
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "system", "content": "（历史不保留在上下文里；需要时用 recall_history 取回全部整数）"},
                        {"role": "user", "content": f"第 {turn} 个数是 {x}，请更新累计值。"}]
        else:
            raise ValueError(args.policy)

        model_text = ""
        turn_max_calls = 4
        for _hop in range(turn_max_calls):
            out = await stream_call(client, args.model, messages, env.tools, thinking=args.thinking)
            rows.append({
                "session": sid, "turn": turn, "policy": args.policy, "hop": _hop,
                "prompt_tokens": out["prompt_tokens"], "cached_tokens": out["cached_tokens"],
                "completion_tokens": out["completion_tokens"],
                "ttft_ms": out["ttft_ms"], "e2e_ms": out["e2e_ms"],
                "finish_reason": out["finish_reason"],
                "tool_names": [c["name"] for c in out["calls"]],
                "content_chars": len(out["content"]),
            })
            if not out["calls"]:
                model_text = out["content"]
                total = parse_total(out["content"])
                rows[-1].update({"parsed_total": total, "expected_total": running,
                                 "correct": total == running, "final": True})
                messages.append({"role": "assistant", "content": out["content"]})
                break
            messages.append({"role": "assistant", "content": out["content"] or "",
                             "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                            for i, c in enumerate(out["calls"])]})
            for i, c in enumerate(out["calls"]):
                try:
                    cargs = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    cargs = {}
                result = env.call(c["name"], cargs)
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                                 "content": str(result)})
        else:
            rows.append({"session": sid, "turn": turn, "policy": args.policy,
                         "parsed_total": None, "expected_total": running, "correct": False,
                         "final": True, "note": "max hops reached"})
        # 把本轮的问答追进历史：assistant 一侧只记**模型自己的输出**，不注入真值，
        # 否则"仅保留最近 k 轮"也能从上一轮的真值 TOTAL 直接读出答案，压缩就测不出代价
        base_history.append({"role": "user", "content": f"第 {turn} 个数是 {x}，请更新累计值。"})
        base_history.append({"role": "assistant", "content": model_text})
    rows.append({"session": sid, "policy": args.policy, "kind": "session_done",
                 "tool_calls": dict(env.calls), "expected_final": running})
    return rows


async def amain(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    rows: list[dict] = []
    for sid in range(args.sessions):
        numbers = [rng.randint(100, 9999) for _ in range(args.turns)]
        rows.extend(await run_session(client, args, sid, numbers))
        if (sid + 1) % 4 == 0:
            print(f"[progress] {sid + 1}/{args.sessions}", flush=True)
    await client.close()

    graded = [r for r in rows if r.get("final")]
    summary = {
        "config": {"policy": args.policy, "sessions": args.sessions, "turns": args.turns,
                   "window": args.window, "model": args.model, "thinking": args.thinking},
        "turns_graded": len(graded),
        "correct": sum(1 for r in graded if r.get("correct")),
        "accuracy": round(sum(1 for r in graded if r.get("correct")) / max(1, len(graded)), 4),
        "prompt_tokens_mean": round(statistics.fmean(
            [r["prompt_tokens"] for r in graded if r.get("prompt_tokens")]), 1) if graded else None,
        "cached_ratio_mean": round(statistics.fmean(
            [(r.get("cached_tokens") or 0) / max(1, r["prompt_tokens"]) for r in graded
             if r.get("prompt_tokens")]), 4) if graded else None,
        "ttft_ms_p50": round(statistics.median([r["ttft_ms"] for r in graded if r.get("ttft_ms")]), 2)
        if any(r.get("ttft_ms") for r in graded) else None,
        "e2e_ms_mean": round(statistics.fmean([r["e2e_ms"] for r in graded if r.get("e2e_ms")]), 2)
        if graded else None,
        "recall_calls": sum(1 for r in rows if "recall_history" in (r.get("tool_names") or [])),
        "summary_calls": sum(1 for r in rows if r.get("kind") == "summary_cost"),
        "rows": rows,
    }
    (out / f"compaction-{args.policy}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.3 上下文压缩策略对照")
    ap.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True, choices=["full", "window", "summary", "external"])
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())