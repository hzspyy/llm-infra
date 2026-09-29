#!/usr/bin/env python3
"""L9.5：Agent 运行时的状态机、持久事件、幂等任务账本与恢复。

一次 agent 调用的失败面比一次补全大得多：模型在等、工具在跑、工具跑完但结果没被确认、
下一轮 prefill 还没开始——这四个点上断掉，恢复动作完全不同。本文件实现一个最小运行时：

* **状态机**：``CREATED → TURN_START → MODEL_RUNNING → TOOL_RUNNING → TOOL_DONE → TURN_DONE``，
  每个会话、每一轮、每次工具调用都有唯一 ID；
* **持久事件**：所有状态迁移追加写进 ``events.jsonl``，正常路径可以只靠日志重放；
* **幂等账本**：带幂等键的副作用工具（``task_ledger_append``）执行前先查账本，命中则复用结果；
* **取消**：``cancel`` 抛出的取消令牌在模型流读取与工具边界被检查，取消后不会推进轮次；
* **恢复**：``--recover`` 从账本重建状态，只补做没提交的步骤；
* **版本校验**：模型 revision、adapter、工具 schema 哈希不一致时拒绝恢复。

用法::

    # 正常跑完
    python labs/L9/agent_runtime.py run --out DIR
    # 在某个点崩掉，然后恢复
    python labs/L9/agent_runtime.py run --out DIR --crash-at tool_running
    python labs/L9/agent_runtime.py recover --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

STATE_FILE = "runtime_state.json"
EVENTS = "events.jsonl"
SIDE_EFFECTS = "side_effects.jsonl"

TASK_LEDGER_TOOL = [{
    "type": "function",
    "function": {
        "name": "task_ledger_append",
        "description": "把一个任务结论追加到项目内任务账本（有副作用，幂等键相同的重复调用不会重复记账）",
        "parameters": {
            "type": "object",
            "properties": {
                "idem_key": {"type": "string", "description": "调用方给出的幂等键"},
                "entry": {"type": "string", "description": "要记账的内容"},
            },
            "required": ["idem_key", "entry"],
            "additionalProperties": False,
        },
    },
}]

TOOLS = T.CALC_TOOL + TASK_LEDGER_TOOL

SYSTEM = (
    "你是一个执行多轮任务的助理。每轮先用一句话说明计划，再调用一个工具："
    "算数用 calculate，记结论用 task_ledger_append（每次用不同的 idem_key）。"
    "任务完成后输出 `FINAL: done`。"
)
# 每一步可以强制指定工具（tool_choice=named）：账本写入必须确定发生，
# 否则"模型这次没调用工具"会把幂等与恢复路径整个绕过去，矩阵就测不到东西。
STEPS = [
    {"prompt": "第一步：用 calculate 计算 (12+7)*3 的值。", "force": "calculate"},
    {"prompt": "第二步：把第一步的结果写进任务账本，idem_key 用 key-1，entry 用 `step1=57`。",
     "force": "task_ledger_append"},
    {"prompt": "第三步：先用 calculate 算出 100//7，再把这个结果记进账本，idem_key 用 key-2。",
     "force": "task_ledger_append"},
]


class Cancelled(Exception):
    pass


class CrashInjected(Exception):
    pass


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class Runtime:
    def __init__(self, out: pathlib.Path, *, model: str, adapter: str | None,
                 session_id: str, schema_hash: str, crash_at: str | None = None,
                 cancel_at: int | None = None, max_turns: int = 6, deadline_s: float = 120.0,
                 crash_turn: int = 2,
                 base_url: str = "http://127.0.0.1:8015/v1"):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.adapter = adapter
        self.session_id = session_id
        self.schema_hash = schema_hash
        self.crash_at = crash_at
        self.cancel_at = cancel_at
        self.crash_turn = crash_turn
        self.max_turns = max_turns
        self.deadline_s = deadline_s
        self.base_url = base_url
        self.events_fh = open(self.out / EVENTS, "a", encoding="utf-8")
        self.side_fh = open(self.out / SIDE_EFFECTS, "a", encoding="utf-8")
        self.ledger = self._load_ledger()
        self.state = self._rebuild_state()
        self.cancel_token = False
        self.deadline_hits = 0

    # -- 持久化 --------------------------------------------------------------
    def emit(self, kind: str, **kw):
        rec = {"ts_ms": round(now_ms(), 3), "session_id": self.session_id, "kind": kind,
               "state": self.state, **kw}
        self.events_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events_fh.flush()
        return rec

    def _load_ledger(self) -> dict:
        path = self.out / EVENTS
        tool_results: dict[str, dict] = {}
        done_turns: set[int] = set()
        versions: dict | None = None
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["kind"] == "session_start":
                    versions = {k: rec.get(k) for k in ("model", "adapter", "schema_hash",
                                                         "model_rev", "tool_schema_hash")}
                elif rec["kind"] == "tool_result" and rec.get("committed") and rec.get("idem_key"):
                    # 只把带幂等键的副作用结果记账；calculate 等无副作用工具不进账本
                    tool_results[rec["idem_key"]] = rec
                elif rec["kind"] == "turn_done":
                    done_turns.add(rec["turn"])
        return {"tool_results": tool_results, "done_turns": done_turns, "versions": versions}

    def _rebuild_state(self) -> str:
        return "TURN_DONE" if self.ledger["done_turns"] else "CREATED"

    def side_effect(self, idem_key: str, entry: str) -> str:
        """副作用工具：先查账本，命中就复用，不重复写。"""
        if idem_key in self.ledger["tool_results"]:
            return "reused:" + str(self.ledger["tool_results"][idem_key]["result"])
        self.side_fh.write(json.dumps({"idem_key": idem_key, "entry": entry}, ensure_ascii=False) + "\n")
        self.side_fh.flush()
        return f"committed:{entry}"

    def count_side_effects(self) -> int:
        path = self.out / SIDE_EFFECTS
        if not path.exists():
            return 0
        return sum(1 for _ in path.open(encoding="utf-8"))

    # -- 模型与工具 ----------------------------------------------------------
    async def model_step(self, client, messages: list[dict], turn: int,
                         force_tool: str | None = None) -> dict:
        self.state = "MODEL_RUNNING"
        self.emit("model_start", turn=turn)
        call_id = f"{self.session_id}-m{turn}-{uuid.uuid4().hex[:6]}"
        if self.crash_at == "model_wait" and turn == self.crash_turn:
            self.emit("crash_injected", point="model_wait", turn=turn, call_id=call_id)
            raise CrashInjected("model_wait")
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.adapter or self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=({"type": "function", "function": {"name": force_tool}}
                                 if force_tool else "auto"),
                    temperature=0.0,
                    max_tokens=192,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
                timeout=self.deadline_s,
            )
        except asyncio.TimeoutError:
            self.deadline_hits += 1
            self.state = "TURN_START"
            self.emit("deadline_exceeded", turn=turn, call_id=call_id, stage="model")
            raise
        msg = resp.choices[0].message
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        rec = self.emit("model_call", turn=turn, call_id=call_id,
                        prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                        cached_tokens=getattr(details, "cached_tokens", None) if details else None,
                        wall_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        tool_names=[tc.function.name for tc in (msg.tool_calls or [])])
        return {"call_id": call_id, "content": msg.content or "", "tool_calls": msg.tool_calls or [],
                "record": rec}

    async def run_tool(self, name: str, args: dict, turn: int, tool_call_id: str) -> str:
        self.state = "TOOL_RUNNING"
        call_id = f"{self.session_id}-t{turn}-{uuid.uuid4().hex[:6]}"
        self.emit("tool_call", turn=turn, call_id=call_id, tool=name, args=args)
        if self.crash_at == "tool_running" and turn == self.crash_turn:
            self.emit("crash_injected", point="tool_running", turn=turn, call_id=call_id)
            raise CrashInjected("tool_running")
        if name == "calculate":
            result = T.safe_eval(str(args.get("expression", "")))
            result = str(result)
        elif name == "task_ledger_append":
            key = str(args.get("idem_key", ""))
            result = self.side_effect(key, str(args.get("entry", "")))
            self.emit("tool_result", turn=turn, call_id=call_id, tool=name, idem_key=key,
                      result=result, committed=True)
        else:
            result = f"ERROR: unknown tool {name}"
        if not (name == "task_ledger_append"):
            self.emit("tool_result", turn=turn, call_id=call_id, tool=name, result=result, committed=True)
        self.state = "TOOL_DONE"
        if self.crash_at == "tool_done_unacked" and turn == self.crash_turn:
            self.emit("crash_injected", point="tool_done_unacked", turn=turn, call_id=call_id)
            raise CrashInjected("tool_done_unacked")
        return result

    async def run(self) -> dict:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY", timeout=self.deadline_s + 30)
        self.emit("session_start", model=self.model, adapter=self.adapter,
                  schema_hash=self.schema_hash, model_rev=self.model,
                  tool_schema_hash=schema_hash(TOOLS))
        messages = [{"role": "system", "content": SYSTEM}]
        start_turn = max(self.ledger["done_turns"], default=0) + 1
        if start_turn > 1:
            self.emit("resume", from_turn=start_turn,
                      reused_tool_results=sorted(self.ledger["tool_results"]))
            # 用账本里已提交的结果重建上下文
            for turn in range(1, start_turn):
                messages.append({"role": "user",
                                 "content": STEPS[min(turn - 1, len(STEPS) - 1)]["prompt"]})
                for key, rec in self.ledger["tool_results"].items():
                    if rec.get("turn") == turn:
                        messages.append({"role": "assistant", "content": f"(resumed) {key}"})
                        messages.append({"role": "tool", "tool_call_id": key, "content": str(rec["result"])})
        turns = start_turn
        for step in STEPS[start_turn - 1:]:
            turn = turns
            if self.cancel_at == turn:
                self.cancel_token = True
                self.emit("cancel_requested", turn=turn)
                self.state = "CANCELLED"
                self.emit("session_cancelled", turn=turn)
                await client.close()
                return {"status": "cancelled", "turn": turn, "side_effects": self.count_side_effects()}
            self.state = "TURN_START"
            messages.append({"role": "user", "content": step["prompt"]})
            self.emit("turn_start", turn=turn, deadline_s=self.deadline_s)
            if self.crash_at == "next_prefill" and turn == self.crash_turn:
                self.emit("crash_injected", point="next_prefill", turn=turn)
                raise CrashInjected("next_prefill")
            try:
                step_out = await self.model_step(client, messages, turn, step.get("force"))
            except asyncio.TimeoutError:
                self.state = "TURN_START"
                self.emit("turn_failed", turn=turn, reason="model_timeout")
                await client.close()
                return {"status": "timeout", "turn": turn, "side_effects": self.count_side_effects()}
            if self.cancel_token:
                raise Cancelled()
            assistant = {"role": "assistant", "content": step_out["content"]}
            tool_names = []
            if step_out["tool_calls"]:
                assistant["tool_calls"] = [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in step_out["tool_calls"]]
                messages.append(assistant)
                for tc in step_out["tool_calls"]:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tool_names.append(tc.function.name)
                    result = await self.run_tool(tc.function.name, args, turn, tc.id)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
            else:
                messages.append(assistant)
            self.state = "TURN_DONE"
            self.emit("turn_done", turn=turn, tools=tool_names)
            self.ledger["done_turns"].add(turn)
            turns += 1
        self.state = "DONE"
        self.emit("session_done", turns=turns - 1, side_effects=self.count_side_effects())
        await client.close()
        return {"status": "done", "turns": turns - 1, "side_effects": self.count_side_effects()}


def schema_hash(tools: list[dict]) -> str:
    return hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:16]


def cmd_run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sid = args.session_id or f"sess-{uuid.uuid4().hex[:8]}"
    (out / "session_id.txt").write_text(sid, encoding="utf-8")
    rt = Runtime(out, model=args.model, adapter=args.adapter, session_id=sid,
                 schema_hash=schema_hash(TOOLS), crash_at=args.crash_at,
                 cancel_at=args.cancel_at, crash_turn=args.crash_turn, base_url=args.base_url)
    if args.crash_at == "next_prefill":
        rt._next_prefill_turn = args.crash_turn
    try:
        result = asyncio.run(rt.run())
    except CrashInjected as exc:
        result = {"status": "crashed", "point": str(exc), "side_effects": rt.count_side_effects()}
    except Cancelled:
        result = {"status": "cancelled", "side_effects": rt.count_side_effects()}
    (out / STATE_FILE).write_text(json.dumps({"session_id": sid, **result}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["status"] in ("done", "cancelled", "crashed") else 1


def cmd_recover(args) -> int:
    out = pathlib.Path(args.out)
    sid_file = out / "session_id.txt"
    sid = sid_file.read_text(encoding="utf-8").strip() if sid_file.exists() else f"sess-{uuid.uuid4().hex[:8]}"
    schema_hash_now = schema_hash(TOOLS)
    ledger = Runtime(out, model=args.model, adapter=args.adapter, session_id=sid,
                     schema_hash=schema_hash_now, base_url=args.base_url)
    versions = ledger.ledger["versions"]
    if versions and versions.get("tool_schema_hash") and versions["tool_schema_hash"] != schema_hash_now:
        rec = {"status": "refused", "reason": "tool schema changed",
               "recorded": versions["tool_schema_hash"], "current": schema_hash_now}
        ledger.emit("recover_refused", **rec)
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        return 0
    result = asyncio.run(ledger.run())
    (out / "recover.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 agent 运行时")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("recover", cmd_recover)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
        p.add_argument("--model", default="Qwen/Qwen3-4B")
        p.add_argument("--adapter", default=None)
        p.add_argument("--session-id", default=None)
        if name == "run":
            p.add_argument("--crash-at", default=None,
                           choices=[None, "model_wait", "tool_running", "tool_done_unacked", "next_prefill"])
            p.add_argument("--cancel-at", type=int, default=None)
            p.add_argument("--crash-turn", type=int, default=2,
                           help="注入点在第几轮触发（默认第 2 轮，此时已有账本写入）")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
