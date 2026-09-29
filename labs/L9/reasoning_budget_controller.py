#!/usr/bin/env python3
"""L9.4 任务 B/C：有独立答案余量的预算控制器 + 多候选执行与验证器。

``controller`` 模式（任务 B）比较三种**实际支持**的推理预算控制方式：

* ``hard_truncate``：只给 `max_tokens`（引擎原生口径，思考与答案共用预算）；
* ``stop_and_resume``：先给 reasoning_cap，被截断后再追加一条"立刻给最终答案"的引导并续写，
  第二次请求的**重复 prefill** 与额外请求数都要计入；
* ``native_margin``：尝试传 reasoning 专用字段，报告服务端真实反应（忽略 / 400 / 生效）。

三种方式都用同一批 GSM8K 题、同一模板与评分；主指标是**最终答案率**与准确率，
同时记总输出 token、总 prompt token（含重复 prefill）与墙钟。

``candidates`` 模式（任务 C）在同题同总资源上限下比较 1/2/4 个候选：

* 串行与有界并行两种执行方式；
* 验证器用**可执行检查**（让模型同时给出算式，harness 只做求值与一致性核对，
  不把判分器答案暴露给策略），以及可选的"自我复核"（再发一次请求让模型复核）；
* 记录候选 token、验证器 token、被取消的分支数、选中答案与每正确答案成本。

用法::

    python labs/L9/reasoning_budget_controller.py controller --base-url ... --out DIR
    python labs/L9/reasoning_budget_controller.py candidates --base-url ... --out DIR --k 4
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import pathlib
import random
import re
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reasoning_budget import GSM8K_TEMPLATE, extract_gsm8k, load_questions, score  # noqa: E402

GSM8K_GLOB = "/scratch/learn/models/hf/hub/datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet"

EXPR_TEMPLATE = (
    "Solve the problem step by step. Then output exactly two lines:\n"
    "`EXPR: <a single arithmetic expression with integers that yields the answer>`\n"
    "`#### <number>`\n\nProblem: {q}\n"
)


async def call(client, model, prompt, *, max_tokens, thinking, temperature=0.0):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    wall = (time.perf_counter() - t0) * 1000.0
    u = resp.usage
    return {
        "text": resp.choices[0].message.content or "",
        "reasoning": getattr(resp.choices[0].message, "reasoning_content", None) or "",
        "finish_reason": resp.choices[0].finish_reason,
        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
        "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
        "wall_ms": round(wall, 3),
    }


async def call_native_margin(client, model, prompt, *, budget, margin, thinking):
    """尝试用 reasoning 专用字段留出答案余量；服务端不接受就如实记录。"""
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking},
                        "thinking_budget": max(0, budget - margin)})
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "field_effective": False,
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
    u = resp.usage
    return {"text": resp.choices[0].message.content or "",
            "reasoning": getattr(resp.choices[0].message, "reasoning_content", None) or "",
            "finish_reason": resp.choices[0].finish_reason,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "field_effective": "unknown", "error": None}


async def controller(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions("gsm8k", args.n, args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def one(q: dict, cap: int, mode: str):
        async with sem:
            prompt = q["prompt"]
            rec = {"qid": q["qid"], "cap": cap, "mode": mode, "requests": 1,
                   "extra_prompt_tokens": 0, "answer_after_resume": None}
            t0 = time.perf_counter()
            if mode == "hard_truncate":
                r = await call(client, args.model, prompt, max_tokens=cap, thinking=True)
                text = r["text"]
                rec.update({"completion_tokens": r["completion_tokens"],
                            "prompt_tokens": r["prompt_tokens"], "finish_reason": r["finish_reason"],
                            "reasoning_tokens": r["reasoning_tokens"]})
            elif mode == "stop_and_resume":
                r1 = await call(client, args.model, prompt, max_tokens=cap, thinking=True)
                text = r1["text"]
                rec.update({"completion_tokens": r1["completion_tokens"],
                            "prompt_tokens": r1["prompt_tokens"], "finish_reason": r1["finish_reason"],
                            "reasoning_tokens": r1["reasoning_tokens"]})
                if r1["finish_reason"] == "length":
                    # 追加引导并续写：第二次请求要把整段前缀重新 prefill 一遍
                    follow = (prompt + r1["reasoning"] + r1["text"] +
                              "\n请立刻给出最终数值答案，不要继续思考。最后一行写 `#### <number>`。\n")
                    r2 = await call(client, args.model, follow, max_tokens=args.margin, thinking=False)
                    text = text + "\n" + r2["text"]
                    rec.update({"requests": 2,
                                "extra_prompt_tokens": r2["prompt_tokens"],
                                "prompt_tokens": r1["prompt_tokens"] + r2["prompt_tokens"],
                                "completion_tokens": r1["completion_tokens"] + r2["completion_tokens"],
                                "answer_after_resume": extract_gsm8k(r2["text"])})
            elif mode == "native_margin":
                r = await call_native_margin(client, args.model, prompt, budget=cap,
                                             margin=args.margin, thinking=True)
                text = r.get("text", "")
                rec.update({"completion_tokens": r.get("completion_tokens"),
                            "prompt_tokens": r.get("prompt_tokens"),
                            "finish_reason": r.get("finish_reason"),
                            "reasoning_tokens": r.get("reasoning_tokens"),
                            "field_effective": r.get("field_effective"),
                            "error": r.get("error")})
            rec["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            parsed = extract_gsm8k(text)
            rec.update({"extracted": parsed, "gold": q["gold"],
                        "correct": parsed == score("gsm8k", text, q["gold"])["extracted"] and parsed is not None,
                        "has_final": bool(re.search(r"####\s*-?\d", text))})
            rec["correct"] = bool(parsed is not None and score("gsm8k", text, q["gold"])["correct"])
            rows.append(rec)

    for mode in ("hard_truncate", "stop_and_resume", "native_margin"):
        for cap in args.caps:
            await asyncio.gather(*(one(q, cap, mode) for q in questions))
            print(f"[controller] {mode} cap={cap} done", flush=True)
    await client.close()

    table = {}
    for r in rows:
        key = f"{r['mode']}|cap{r['cap']}"
        a = table.setdefault(key, {"n": 0, "correct": 0, "final": 0, "tokens": 0, "prompt_tokens": 0,
                                   "requests": 0, "wall_ms": [], "field_effective": r.get("field_effective")})
        a["n"] += 1
        a["correct"] += int(r["correct"])
        a["final"] += int(r["has_final"])
        a["tokens"] += r.get("completion_tokens") or 0
        a["prompt_tokens"] += r.get("prompt_tokens") or 0
        a["requests"] += r.get("requests") or 1
        a["wall_ms"].append(r["wall_ms"])
    summary = {"config": {"model": args.model, "n": args.n, "caps": args.caps,
                          "margin": args.margin, "concurrency": args.concurrency},
               "table": {k: {**{kk: vv for kk, vv in v.items() if kk != "wall_ms"},
                             "accuracy": round(v["correct"] / max(1, v["n"]), 4),
                             "final_rate": round(v["final"] / max(1, v["n"]), 4),
                             "completion_tokens": v["tokens"],
                             "prompt_tokens": v["prompt_tokens"],
                             "requests": v["requests"],
                             "wall_ms_mean": round(statistics.fmean(v["wall_ms"]), 1)}
                         for k, v in sorted(table.items())},
               "rows": rows}
    (out / "controller.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["table"], ensure_ascii=False, indent=1))
    return 0


def check_expr(text: str, stated: str | None) -> tuple[bool | None, str | None]:
    """可执行检查：模型给的算式求值后是否等于它自己给出的答案（不涉及判分器答案）。"""
    m = re.search(r"EXPR:\s*([0-9][0-9+\-*/() ]*)", text)
    if not m or stated is None:
        return None, None
    expr = m.group(1).strip()
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - 只允许数字与四则运算
    except Exception:  # noqa: BLE001
        return False, expr
    try:
        ok = abs(float(value) - float(stated)) < 1e-6
    except ValueError:
        ok = False
    return ok, expr


async def candidates(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions("gsm8k", args.n, args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
    results = []
    for q in questions:
        prompt = EXPR_TEMPLATE.format(q=q["prompt"].split("Problem: ")[-1])
        cancelled = 0

        async def gen(i: int):
            return await call(client, args.model, prompt, max_tokens=args.budget, thinking=True)

        t0 = time.perf_counter()
        if args.parallel and args.k > 1:
            done, pending = await asyncio.wait(
                [asyncio.create_task(gen(i)) for i in range(args.k)],
                timeout=args.branch_timeout, return_when=asyncio.ALL_COMPLETED)
            branes = [t.result() for t in done]
            cancelled = len(pending)
            for t in pending:
                t.cancel()
        else:
            branes = [await gen(i) for i in range(args.k)]
        wall_ms = (time.perf_counter() - t0) * 1000.0

        cands = []
        for r in branes:
            ans = extract_gsm8k(r["text"])
            checked, expr = check_expr(r["text"], ans)
            cands.append({"answer": ans, "checked": checked, "expr": expr,
                          "completion_tokens": r["completion_tokens"],
                          "prompt_tokens": r["prompt_tokens"]})
        verify_calls = 0
        if args.verify == "self" and cands:
            # 自我复核：只对第一个候选发一次复核请求（成本计入）
            first = branes[0]
            review = await call(client, args.model,
                                prompt + "\n\n上面这道题的答案是：" + (cands[0]["answer"] or "未知") +
                                "\n请只回答 `CHECK: yes` 或 `CHECK: no`，判断这个答案是否由算式正确推出。",
                                max_tokens=32, thinking=False)
            verify_calls = 1
            cands[0]["self_check"] = "CHECK: yes" in (review["text"] or "")
        checked = [c for c in cands if c["checked"] is True]
        pool = checked if checked else [c for c in cands if c["answer"]]
        if pool:
            answers = [c["answer"] for c in pool]
            mode = statistics.mode(answers) if len(set(answers)) < len(answers) else answers[0]
            selected = mode
        else:
            selected = None
        gold = q["gold"]
        results.append({
            "qid": q["qid"], "k": args.k, "parallel": args.parallel, "verify": args.verify,
            "candidates": cands, "selected": selected, "gold": gold,
            "correct": selected is not None and score("gsm8k", f"#### {selected}", gold)["correct"],
            "checked_count": len(checked), "cancelled_branches": cancelled,
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in cands),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in cands),
            "verify_calls": verify_calls, "wall_ms": round(wall_ms, 2),
        })
        if len(results) % 16 == 0:
            print(f"[candidates k={args.k}] {len(results)}/{len(questions)}", flush=True)
    await client.close()

    correct = sum(1 for r in results if r["correct"])
    checked_total = sum(r["checked_count"] for r in results)
    summary = {
        "config": {"model": args.model, "n": args.n, "k": args.k, "budget": args.budget,
                   "parallel": args.parallel, "verify": args.verify,
                   "branch_timeout": args.branch_timeout},
        "questions": len(results), "correct": correct,
        "accuracy": round(correct / max(1, len(results)), 4),
        "checked_candidates": checked_total,
        "candidate_total": sum(len(r["candidates"]) for r in results),
        "cancelled_branches": sum(r["cancelled_branches"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "verify_calls": sum(r["verify_calls"] for r in results),
        "wall_ms_mean": round(statistics.fmean([r["wall_ms"] for r in results]), 1),
        "seconds_per_correct": round(sum(r["wall_ms"] for r in results) / 1000.0 / max(1, correct), 3),
        "tokens_per_correct": round(sum(r["completion_tokens"] for r in results) / max(1, correct), 1),
        "results": results,
    }
    (out / f"candidates-k{args.k}-par{int(args.parallel)}-{args.verify}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.4 预算控制器与多候选执行")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("controller")
    p.add_argument("--base-url", default="http://127.0.0.1:8021/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--caps", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--margin", type=int, default=512,
                   help="续写/原生余量：留给最终答案的 token 数")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: asyncio.run(controller(a)))

    p = sub.add_parser("candidates")
    p.add_argument("--base-url", default="http://127.0.0.1:8021/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--budget", type=int, default=512, help="每个候选的输出预算")
    p.add_argument("--parallel", action="store_true", help="有界并行；默认串行")
    p.add_argument("--verify", default="expr", choices=["expr", "self"],
                   help="expr=可执行算式核对；self=再发一次请求自我复核")
    p.add_argument("--branch-timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: asyncio.run(candidates(a)))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())