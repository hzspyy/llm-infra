#!/usr/bin/env python3
"""L9.1 任务 C：BFCL `multi_turn_base` 固定子集，用**官方评分器与官方模拟环境**对拍。

与 `agent_tasks.py` 的教学任务分开：这里跑的是公开基准题，模型侧接本机 vLLM 的 OpenAI 接口，
工具侧用 BFCL 自带的模拟实现（`execute_multi_turn_func_call`）维护状态，最后交给官方
`multi_turn_checker` 判分。三件事都来自 `bfcl-eval==2025.12.17` 的同一份安装，不做二次实现：

* 题目与工具文档：`bfcl_eval/data/BFCL_v4_multi_turn_base.json` 与 `data/multi_turn_func_doc/<class>.json`；
* 环境执行：`bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call`；
* 判分：`bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker.multi_turn_checker`。

注意 BFCL 的数据随 wheel 一起分发（`pip install bfcl-eval==2025.12.17` 即得，不需要访问 Hub），
本机 pip 默认镜像里没有该包，需显式用 `--index-url https://pypi.org/simple`。

固定子集的取法：按 `id` 的数字后缀排序取前 N 个（默认 50），并在结果里记录题目 ID 列表，
保证可复核。

用法::

    python labs/L9/bfcl_multi_turn.py run --out out/9.1/bfcl \\
        --base-url http://127.0.0.1:8061/v1 --model Qwen/Qwen3-4B --n 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import time


def bfcl_root() -> pathlib.Path:
    import bfcl_eval

    return pathlib.Path(bfcl_eval.__file__).resolve().parent


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tools(involved_classes: list[str], root: pathlib.Path) -> list[dict]:
    """按 BFCL 自己的映射取工具文档：类名到文件名不是简单的大小写变换
    （`TwitterAPI` 的文件叫 `posting_api.json`），必须用 `MULTI_TURN_FUNC_DOC_FILE_MAPPING`。"""
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

    tools: list[dict] = []
    for cls in involved_classes:
        fname = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
        if fname is None:
            raise KeyError(f"BFCL 映射里没有 {cls}")
        path = root / "data" / "multi_turn_func_doc" / fname
        if not path.exists():
            raise FileNotFoundError(f"缺少工具文档：{path}")
        for row in load_jsonl(path):
            tools.append({"type": "function",
                          "function": {"name": row["name"], "description": row["description"],
                                       "parameters": row["parameters"]}})
    return tools


async def engine_call(client, model: str, messages: list[dict], tools: list[dict],
                      max_tokens: int, thinking: bool) -> dict:
    t0 = time.perf_counter()
    calls: list[dict] = []
    content = ""
    usage = None
    error = None
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=tools or None,
            tool_choice="auto" if tools else None, temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        usage = resp.usage
        msg = resp.choices[0].message
        content = msg.content or ""
        for tc in (msg.tool_calls or []):
            calls.append({"id": tc.id, "name": tc.function.name,
                          "arguments": tc.function.arguments or "{}"})
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"content": content, "calls": calls, "usage": usage, "error": error,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3)}


async def run_entry(client, model: str, entry: dict, gt: list[list[str]], tools: list[dict],
                    args, root: pathlib.Path) -> dict:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.model_handler.utils import convert_to_function_call

    entry_id = entry["id"]
    category = entry_id.rsplit("_", 1)[0]
    messages: list[dict] = []
    per_turn_steps: list[list[list[str]]] = []
    turn_stats: list[dict] = []
    error = None
    prompt_tokens = completion_tokens = 0
    t_start = time.perf_counter()

    for turn_idx, turn_msgs in enumerate(entry["question"]):
        for m in turn_msgs:
            messages.append({"role": m["role"], "content": m["content"]})
        steps: list[list[str]] = []
        turn_calls = 0
        for _step in range(args.max_steps_per_turn):
            res = await engine_call(client, model, messages, tools, args.max_tokens,
                                    args.thinking)
            if res["usage"] is not None:
                prompt_tokens += getattr(res["usage"], "prompt_tokens", 0) or 0
                completion_tokens += getattr(res["usage"], "completion_tokens", 0) or 0
            if res["error"]:
                error = res["error"]
                break
            if not res["calls"]:
                messages.append({"role": "assistant", "content": res["content"]})
                break
            call_strings = convert_to_function_call([{c["name"]: c["arguments"]}
                                                     for c in res["calls"]])
            steps.append(call_strings)
            turn_calls += len(call_strings)
            messages.append({"role": "assistant", "content": res["content"],
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": c["arguments"]}}
                                            for c in res["calls"]]})
            # 用官方模拟环境执行，拿到工具返回值再继续该轮
            try:
                results, _instances = execute_multi_turn_func_call(
                    func_call_list=call_strings,
                    initial_config=entry.get("initial_config", {}),
                    involved_classes=entry["involved_classes"],
                    model_name=model.replace("/", "_").replace("-", "_").replace(".", "_"),
                    test_entry_id=entry_id,
                    long_context=("long_context" in category or "composite" in category),
                    is_evaL_run=False,
                )
            except Exception as exc:  # noqa: BLE001
                results = [f"ERROR: {type(exc).__name__}: {exc}"]
                error = f"tool_exec: {type(exc).__name__}: {exc}"
            for call, out in zip(res["calls"], results):
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": str(out)[:4000]})
            if error:
                break
        per_turn_steps.append(steps)
        turn_stats.append({"turn": turn_idx, "steps": len(steps), "calls": turn_calls,
                           "gt_calls": len(gt[turn_idx]) if turn_idx < len(gt) else None})
        if error:
            break

    verdict = None
    if error is None and len(per_turn_steps) == len(gt):
        try:
            verdict = multi_turn_checker(per_turn_steps, gt, entry, category, model)
            # 官方判定里带模拟环境的实例对象（如 Directory），先转成可序列化结构再落盘
            verdict = json.loads(json.dumps(verdict, default=str, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            verdict = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"id": entry_id, "turns": len(entry["question"]), "turns_run": len(per_turn_steps),
            "involved_classes": entry["involved_classes"],
            "model_calls": per_turn_steps, "gt_calls": gt, "turn_stats": turn_stats,
            "error": error, "verdict": verdict,
            "valid": bool(verdict and verdict.get("valid")),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "wall_s": round(time.perf_counter() - t_start, 3)}


async def cmd_run_async(args) -> int:
    from openai import AsyncOpenAI

    root = bfcl_root()
    questions = load_jsonl(root / "data" / "BFCL_v4_multi_turn_base.json")
    gts = {r["id"]: r["ground_truth"]
           for r in load_jsonl(root / "data" / "possible_answer" / "BFCL_v4_multi_turn_base.json")}
    questions.sort(key=lambda e: int(e["id"].rsplit("_", 1)[-1]))
    selected = questions[: args.n]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def one(entry: dict) -> None:
        async with sem:
            tools = load_tools(entry["involved_classes"], root)
            r = await run_entry(client, args.model, entry, gts[entry["id"]], tools, args, root)
            results.append(r)
            print(f"[{r['id']}] valid={r['valid']} turns={r['turns_run']}/{r['turns']} "
                  f"calls={sum(t['calls'] for t in r['turn_stats'])} err={r['error']}",
                  flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*[one(e) for e in selected])
    wall = time.perf_counter() - t0
    await client.close()
    results.sort(key=lambda r: int(r["id"].rsplit("_", 1)[-1]))
    valid = [r for r in results if r["valid"]]
    fanout = [t["calls"] for r in results for t in r["turn_stats"]]
    turns = [r["turns"] for r in results]
    summary = {
        "config": {"base_url": args.base_url, "model": args.model, "n": args.n,
                   "thinking": args.thinking, "max_tokens": args.max_tokens,
                   "max_steps_per_turn": args.max_steps_per_turn,
                   "concurrency": args.concurrency,
                   "bfcl_version": _bfcl_version()},
        "task_ids": [r["id"] for r in results],
        "accuracy": round(len(valid) / max(1, len(results)), 4),
        "valid": len(valid), "total": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "distribution": {
            "turns": {"min": min(turns) if turns else None,
                      "max": max(turns) if turns else None,
                      "mean": round(statistics.fmean(turns), 3) if turns else None},
            "calls_per_turn_p50": _q(fanout, 0.5), "calls_per_turn_max": max(fanout) if fanout else None,
            "calls_total": sum(fanout),
        },
        "tokens": {"prompt": sum(r["prompt_tokens"] for r in results),
                   "completion": sum(r["completion_tokens"] for r in results)},
        "wall_s": round(wall, 3),
        "per_entry": results,
    }
    (out / "bfcl_multi_turn.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_entry", "task_ids")}, ensure_ascii=False, indent=1))
    return 0


def _bfcl_version() -> str:
    import importlib.metadata as md

    return md.version("bfcl-eval")


def _q(values: list[float], p: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1-C：BFCL multi_turn_base 固定子集")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-steps-per-turn", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_run_async(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
