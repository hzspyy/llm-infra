#!/usr/bin/env python3
"""L9.5：故障矩阵——在四个位置注入崩溃/取消，比较几种恢复策略。

四个注入点（与运行时状态机对齐）：

| 注入点 | 崩溃时的状态 | 已提交的东西 |
|---|---|---|
| ``model_wait`` | 模型请求在途 | 本轮无任何提交 |
| ``tool_running`` | 工具已开始、结果未写账本 | 上几轮已提交 |
| ``tool_done_unacked`` | 工具结果已写账本、轮次未确认 | 账本里有 tool_result |
| ``next_prefill`` | 上一轮已确认、下一轮尚未发模型 | 上一轮完整提交 |

三种恢复策略：

* ``restart``：直接从头重跑整个会话（最容易造成重复副作用）；
* ``resume``：从账本恢复，只补做未提交的步骤（本文件的默认，幂等键去重）；
* ``abort``：不再继续，检查资源与账本状态。

同时测一次 ``--cancel-at 2``：取消后第 2 轮不得推进，且恢复时不得复用被取消轮次的工具结果。

用法::

    python labs/L9/agent_failure_matrix.py --root DIR --base-url http://127.0.0.1:8015/v1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = HERE / "agent_runtime.py"


def read_events(out: pathlib.Path) -> list[dict]:
    path = out / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def side_effect_keys(out: pathlib.Path) -> list[str]:
    path = out / "side_effects.jsonl"
    if not path.exists():
        return []
    return [json.loads(l)["idem_key"] for l in path.open(encoding="utf-8") if l.strip()]


def run(cmd: list[str], out: pathlib.Path | None = None) -> dict:
    """跑一次 runtime 子进程。

    运行时会把自己的结论写进 ``runtime_state.json`` / ``recover.json``（多行 JSON），
    比解析 stdout 的最后一行可靠。
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for name in ("recover.json", "runtime_state.json"):
        if out is not None and (out / name).exists():
            try:
                return json.loads((out / name).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    text = proc.stdout.strip()
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pass
    tail = ((proc.stdout + proc.stderr).strip().splitlines() or [""])[-1]
    return {"status": "unparsed", "stderr": tail[:200]}


def cell(python: str, root: pathlib.Path, point: str, strategy: str, base_url: str, model: str) -> dict:
    out = root / f"{point}-{strategy}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    common = ["--out", str(out), "--base-url", base_url, "--model", model]
    first = run([python, str(RUNTIME), "run", *common, "--crash-at", point], out)
    keys_after_crash = side_effect_keys(out)
    events_after_crash = [e["kind"] for e in read_events(out)]

    if strategy == "abort":
        return {
            "point": point, "strategy": strategy, "first": first, "final_status": "aborted",
            "side_effects_after_crash": keys_after_crash,
            "duplicate_side_effects": 0, "completed": False,
            "turn_done_after": sum(1 for k in events_after_crash if k == "turn_done"),
        }

    if strategy == "restart":
        # 进程重启后盲目重放：运行时自己的事件账本丢了，但**外部副作用日志还在**
        # （副作用发生在外部系统上，重启不会把它抹掉）——这正是重复副作用的来源。
        p = out / "events.jsonl"
        if p.exists():
            p.unlink()
        second = run([python, str(RUNTIME), "run", *common], out)
        final_status = second.get("status")
    else:  # resume
        second = run([python, str(RUNTIME), "recover", *common], out)
        final_status = second.get("status")

    keys_final = side_effect_keys(out)
    dup = len(keys_final) - len(set(keys_final))
    turns = sum(1 for e in read_events(out) if e["kind"] == "turn_done")
    return {
        "point": point,
        "strategy": strategy,
        "first": first,
        "second": second,
        "final_status": final_status,
        "side_effects_after_crash": keys_after_crash,
        "side_effects_final": keys_final,
        "duplicate_side_effects": dup,
        "turn_done_after": turns,
        "completed": final_status == "done",
    }


def cancel_case(python: str, root: pathlib.Path, base_url: str, model: str) -> dict:
    out = root / "cancel-turn2"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    common = ["--out", str(out), "--base-url", base_url, "--model", model]
    first = run([python, str(RUNTIME), "run", *common, "--cancel-at", "2"], out)
    events = read_events(out)
    turns_done_before = {e["turn"] for e in events if e["kind"] == "turn_done"}
    keys_before = side_effect_keys(out)
    # 取消后继续：应只补第 2 轮及之后，且不复用第 2 轮被取消时可能留下的结果
    second = run([python, str(RUNTIME), "recover", *common], out)
    events_after = read_events(out)
    turns_done_after = {e["turn"] for e in events_after if e["kind"] == "turn_done"}
    keys_after = side_effect_keys(out)
    return {
        "cancelled_at_turn": 2,
        "first": first,
        "turns_done_before_recover": sorted(turns_done_before),
        "side_effects_before_recover": keys_before,
        "second": second,
        "turns_done_after_recover": sorted(turns_done_after),
        "side_effects_after_recover": keys_after,
        "duplicates": len(keys_after) - len(set(keys_after)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 故障矩阵")
    ap.add_argument("--root", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for point in ("model_wait", "tool_running", "tool_done_unacked", "next_prefill"):
        for strategy in ("restart", "resume", "abort"):
            row = cell(args.python, root, point, strategy, args.base_url, args.model)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("point", "strategy", "final_status",
                                                  "duplicate_side_effects", "turn_done_after",
                                                  "completed")}, ensure_ascii=False), flush=True)
    cancel = cancel_case(args.python, root, args.base_url, args.model)
    print(json.dumps(cancel, ensure_ascii=False), flush=True)

    summary = {
        "config": {"base_url": args.base_url, "model": args.model, "python": args.python},
        "matrix": rows,
        "cancel": cancel,
    }
    (root / "failure_matrix.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps({"cells": len(rows),
                      "resume_completed": sum(1 for r in rows if r["strategy"] == "resume" and r["completed"]),
                      "restart_duplicates": sum(r["duplicate_side_effects"] for r in rows if r["strategy"] == "restart"),
                      "resume_duplicates": sum(r["duplicate_side_effects"] for r in rows if r["strategy"] == "resume"),
                      "abort_side_effects": [len(r["side_effects_after_crash"]) for r in rows if r["strategy"] == "abort"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
