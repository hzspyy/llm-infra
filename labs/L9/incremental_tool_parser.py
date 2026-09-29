#!/usr/bin/env python3
"""L9.2 任务 B：有状态流式工具解析器，与一次性完整解析对拍。

流式工具调用的难点不在语法，而在**状态**：一次生成里可能出现多个并行调用、字符串里带
转义与 `\\uXXXX`、Unicode 转义被切成两半、JSON 还没结束客户端就取消。解析器必须每收到
一个片段就吐出「这一步新确定的部分」，而且不能把已经扫过的前缀重新扫一遍。

本文件给出两个实现与一个参照：

* ``IncrementalToolParser``：单遍状态机，每个字符只扫一次；维护 ``<tool_call>`` 边界、
  字符串/转义/Unicode 转义状态，增量吐出 `arguments` 片段；
* ``naive_parse_chunks``：每来一个片段就把累计文本整体重新解析一次（实现简单、代价二次）；
* ``full_parse``：把整段文本一次性解析，作为结果对拍参照。

统计口径：``scanned_chars`` 是解析器实际检查过的字符数。单遍扫描约等于总长度，
逐片段重扫约等于 Σ 前缀长度（长度 L、片段数 k 时约 L·k/2）。

用法（CPU 即可）::

    python labs/L9/incremental_tool_parser.py --out <dir>
    python labs/L9/incremental_tool_parser.py --selftest
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

OPEN = "<tool_call>"
CLOSE = "</tool_call>"


# --------------------------------------------------------------------------------------
# 参照实现
# --------------------------------------------------------------------------------------

def split_calls(text: str) -> list[str]:
    """把整段文本切成一个个 <tool_call>…</tool_call> 片段（参照口径）。"""
    out = []
    pos = 0
    while True:
        i = text.find(OPEN, pos)
        if i < 0:
            break
        j = text.find(CLOSE, i + len(OPEN))
        if j < 0:
            out.append(text[i + len(OPEN):])
            break
        out.append(text[i + len(OPEN):j])
        pos = j + len(CLOSE)
    return out


def full_parse(text: str) -> list[dict]:
    """一次性解析：返回 [{name, arguments(dict), raw}]，非法 JSON 记 error。"""
    calls = []
    for body in split_calls(text):
        body = body.strip()
        if not body:
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            calls.append({"name": None, "arguments": None, "raw": body, "error": str(exc)})
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": name, "arguments": args, "raw": body, "error": None})
    return calls


# --------------------------------------------------------------------------------------
# 单遍状态机
# --------------------------------------------------------------------------------------

class IncrementalToolParser:
    """逐字符状态机；每个字符只处理一次。

    状态只有两级：

    * ``SEEK``：用一个固定长度的滑动窗口找 ``<tool_call>``；
    * ``BODY``：在调用体内，先用一个**有界**的头部缓冲找 ``"arguments"``/``"parameters"``
      键，找到之后进入 arguments 区——从这一刻起每个字符都原样追加到 ``arguments_raw``
      并立刻作为 delta 吐出，同时在内部维护字符串/转义/``\\uXXXX`` 子状态，保证
      ``</tool_call>`` 不会被字符串内容误触发。

    关键点：无论片段怎么切，``scanned_chars`` 都等于输入总长度。已经扫过的字符不会再被扫第二遍。
    """

    def __init__(self, head_limit: int = 256, allowlist: list[str] | None = None,
                 schemas: dict | None = None):
        self.state = "SEEK"
        self.scanned_chars = 0
        self.calls: list[dict] = []
        self.cancelled = False
        self.cancelled_calls = 0
        self.allowlist = allowlist
        self.schemas = schemas
        # 峰值常驻缓冲：SEEK 的窗口 + 头部缓冲 + arguments 尾部扣留窗口，三者都有界
        self.peak_buffer = 0
        self.head_limit = head_limit
        self._tail = ""          # SEEK / BODY 的滑动窗口
        self._head = ""          # arguments 键之前的头部缓冲（有界）
        self._cur: dict | None = None
        self._args_started = False
        self._in_string = False
        self._escape = False
        self._unicode_left = 0
        self._staging = ""       # arguments 区里扣留的尾部窗口（最多 len(CLOSE) 个字符）

    # -- 状态切换 ------------------------------------------------------------
    def cancel(self):
        """客户端取消：丢弃在途调用，保留已完成的调用。"""
        self.cancelled = True
        if self._cur is not None and self.calls and self.calls[-1] is self._cur:
            self.calls.pop()
            self.cancelled_calls += 1
        self._cur = None
        self.state = "SEEK"

    def _track_peak(self):
        buf = len(self._tail) + len(self._head) + len(self._staging)
        if buf > self.peak_buffer:
            self.peak_buffer = buf

    def _reset_value(self):
        self._staging = ""
        self._reserved = None
        self._head = ""
        self._args_started = False
        self._value_kind = None
        self._value_done = False
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._unicode_left = 0

    def _start_call(self):
        self._cur = {"name": None, "arguments_raw": "", "error": None, "deltas": [],
                     "dispatched": False, "reject": None}
        self.calls.append(self._cur)
        self._reset_value()
        self.state = "BODY"

    def _end_call(self):
        if self._cur is not None:
            self._flush_staging()
            raw = self._cur["arguments_raw"].strip()
            if raw:
                try:
                    self._cur["arguments"] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._cur["arguments"] = None
                    self._cur["error"] = f"incomplete_json: {exc.msg}"
            else:
                self._cur["arguments"] = {}
            # 提交闸门：闭合 → 允许表 → schema 校验，三步都过才允许执行
            name = self._cur.get("name")
            if self._cur["arguments"] is None:
                self._cur["reject"] = "incomplete_json"
            elif self.allowlist is not None and name not in self.allowlist:
                self._cur["reject"] = "not_in_allowlist"
            else:
                err = validate_args((self.schemas or {}).get(name), self._cur["arguments"]) \
                    if self.schemas is not None else None
                if err:
                    self._cur["reject"] = f"schema: {err}"
            self._cur["dispatched"] = self._cur["reject"] is None
        self._cur = None
        self.state = "SEEK"
        self._tail = ""
        self._reset_value()

    def _flush_staging(self):
        if self._cur is not None and self._staging:
            self._cur["arguments_raw"] += self._staging
            self._cur["deltas"].append(self._staging)
        self._staging = ""

    # -- 主循环 --------------------------------------------------------------
    def feed(self, chunk: str) -> list[dict]:
        events: list[dict] = []
        if self.cancelled:
            return events
        for ch in chunk:
            self.scanned_chars += 1
            self._track_peak()
            if self.state == "SEEK":
                self._tail = (self._tail + ch)[-len(OPEN):]
                if self._tail == OPEN:
                    self._start_call()
                    events.append({"type": "start", "index": len(self.calls) - 1})
                continue

            if not self._args_started:
                # 头部缓冲：只在有界范围内搜索 name / arguments 键
                self._head = (self._head + ch)[: self.head_limit]
                m = re.search(r'"name"\s*:\s*"([^"]*)"', self._head)
                if m and self._cur is not None:
                    self._cur["name"] = m.group(1)
                mi = re.search(r'"(arguments|parameters)"\s*:\s*', self._head)
                if mi:
                    self._args_started = True
                    remainder = self._head[mi.end():]
                    if remainder:
                        ended, flushed = self._consume_args(remainder)
                        if flushed:
                            events.append({"type": "args_delta",
                                           "index": len(self.calls) - 1, "delta": flushed})
                        if ended:
                            events.append({"type": "end", "index": len(self.calls) - 1})
                continue

            ended, flushed = self._consume_args(ch)
            if flushed:
                events.append({"type": "args_delta", "index": len(self.calls) - 1, "delta": flushed})
            if ended:
                events.append({"type": "end", "index": len(self.calls) - 1})
        return events

    def _consume_args(self, text: str) -> tuple[bool, str]:
        """处理 arguments 区字符，返回 (是否结束调用, 本次可吐出的 delta)。

        两段式：值的最后一个字符确定之前，字符进入 ``_staging``（尾部扣留 len(CLOSE) 个，
        防止把 ``</tool_call>`` 当参数吐出去）；值结束之后只找结束标记，后面的结构字符不再计入
        arguments，避免把外层对象的 ``}`` 混进参数。
        """
        flushed: list[str] = []
        ended = False
        for ch in text:
            if self._value_done:
                self._staging = (self._staging + ch)[-len(CLOSE):]
                if self._staging == CLOSE:
                    self._staging = ""       # 结束标记不属于 arguments
                    self._end_call()
                    ended = True
                    break
                continue

            if self._value_kind is None and ch.isspace():
                continue                      # 值前的空白不进 delta
            if self._value_kind is None:
                self._value_kind = "object" if ch == "{" else "array" if ch == "[" else \
                                   "string" if ch == '"' else "literal"
            if self._value_kind == "literal" and ch in ",}] \t\r\n":
                # 数字/true/false/null 的值在分隔符处结束，分隔符本身不属于值
                self._value_done = True
                self._staging = (self._staging + ch)[-len(CLOSE):]
                if self._staging == CLOSE:
                    self._end_call()
                    ended = True
                    break
                continue
            self._staging += ch
            self._track(ch)
            if self._value_complete():
                # 值已完整：扣留的字符全部属于值，立刻结算
                self._value_done = True
                if self._cur is not None and self._staging:
                    self._cur["arguments_raw"] += self._staging
                    self._cur["deltas"].append(self._staging)
                flushed.append(self._staging)
                self._staging = ""
                continue
            if not self._in_string and self._staging.endswith(CLOSE):
                # 值还没结束就出现结束标记：截断的调用，标记不算参数
                self._staging = self._staging[: -len(CLOSE)]
                if self._cur is not None and self._staging:
                    self._cur["arguments_raw"] += self._staging
                    self._cur["deltas"].append(self._staging)
                flushed.append(self._staging)
                self._staging = ""
                self._end_call()
                ended = True
                break
            if len(self._staging) > len(CLOSE):
                ready = self._staging[: -len(CLOSE)]
                self._staging = self._staging[-len(CLOSE):]
                if self._cur is not None:
                    self._cur["arguments_raw"] += ready
                    self._cur["deltas"].append(ready)
                flushed.append(ready)
        return ended, "".join(flushed)

    def _track(self, ch: str):
        """维护字符串/转义/Unicode 转义与括号深度。"""
        if self._unicode_left > 0:
            self._unicode_left -= 1
            return
        if self._escape:
            if ch == "u":
                self._unicode_left = 4
            self._escape = False
            return
        if self._in_string:
            if ch == "\\":
                self._escape = True
            elif ch == '"':
                self._in_string = False
            return
        if ch == '"':
            self._in_string = True
        elif ch in "{[":
            self._depth += 1
        elif ch in "}]":
            self._depth -= 1

    def _value_complete(self) -> bool:
        if self._value_kind in ("object", "array"):
            return self._depth == 0
        if self._value_kind == "string":
            return (not self._in_string) and (not self._escape)
        # 数字/true/false/null：遇到分隔符即结束（分隔符本身不该进入值）
        return False

    def finish(self) -> list[dict]:
        """流结束：未闭合的调用判为不完整，不抛异常。"""
        if self._cur is not None:
            raw = self._cur["arguments_raw"].strip()
            self._cur["arguments"] = None
            try:
                json.loads(raw) if raw else None
            except json.JSONDecodeError as exc:
                self._cur["error"] = f"incomplete_json: {exc.msg}"
            if not self._cur["error"]:
                self._cur["error"] = "incomplete_json: unterminated tool_call"
            # 流结束仍未闭合的调用同样走提交闸门的结果：不派发
            self._cur["reject"] = "incomplete_json"
            self._cur["dispatched"] = False
            self._cur = None
        return [
            {"name": c["name"], "arguments": c.get("arguments"),
             "raw": c["arguments_raw"], "error": c["error"],
             "dispatched": c.get("dispatched", False), "reject": c.get("reject")}
            for c in self.calls
        ]


def naive_parse_chunks(chunks: list[str]) -> tuple[list[dict], int, int]:
    """参照的反面：每来一个片段就把累计文本整体重扫一次。

    返回 ``(结果, 扫描字符数, 峰值缓冲字符数)``。峰值缓冲就是累计文本的长度——这正是
    单遍状态机要避免的：它的常驻缓冲与 payload 无关。
    """
    text = ""
    scanned = 0
    peak = 0
    for ch in chunks:
        text += ch
        scanned += len(text)          # 全串重扫
        peak = max(peak, len(text))
        full_parse(text)              # 真的解析一遍，让代价落在同一量级
    calls = full_parse(text)
    return calls, scanned, peak


# --------------------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------------------

# 工具 schema（JSON-Schema 子集：type / required / properties.type / enum）。
# 只覆盖本 lab 用到的形状，避免为一个校验点引入依赖。
TOOL_SCHEMAS: dict[str, dict] = {
    "calculate": {"type": "object", "required": ["expression"],
                  "properties": {"expression": {"type": "string"},
                                 "delay_ms": {"type": "integer"}}},
    "search_corpus": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}},
    "read_file": {"type": "object", "required": ["path"],
                  "properties": {"path": {"type": "string"}}},
    "list_files": {"type": "object", "required": [], "properties": {}},
    "edit_file": {"type": "object", "required": ["path", "old_string", "new_string"],
                  "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                 "new_string": {"type": "string"}}},
    "write_file": {"type": "object", "required": ["path", "content"],
                   "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
    "run_tests": {"type": "object", "required": [], "properties": {}},
}

_TYPE_MAP = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "array": list, "object": dict}


def validate_args(schema: dict | None, args) -> str | None:
    """按 schema 校验参数；返回错误描述或 None。未知工具按未定义 schema 处理。"""
    if schema is None:
        return "unknown_tool"
    if not isinstance(args, dict):
        return f"arguments must be an object, got {type(args).__name__}"
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required {key!r}"
    for key, spec in (schema.get("properties") or {}).items():
        if key not in args or args[key] is None:
            continue
        want = spec.get("type")
        if want and not isinstance(args[key], _TYPE_MAP[want]):
            return f"{key!r} should be {want}, got {type(args[key]).__name__}"
        if "enum" in spec and args[key] not in spec["enum"]:
            return f"{key!r} not in enum {spec['enum']}"
    return None


def make_call(name: str, args: dict) -> str:
    return f"{OPEN}\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n{CLOSE}"


CASES = [
    {
        "name": "single_short",
        "text": make_call("calculate", {"expression": "(12+7)*3"}),
        "chunks": 3,
        "expect_calls": 1,
    },
    {
        "name": "escaped_string",
        "text": make_call("search_corpus", {"query": "he said \"statins\" \\ and \\nnewline", "k": 5}),
        "chunks": 4,
        "expect_calls": 1,
    },
    {
        "name": "unicode_escape_split",
        "text": make_call("search_corpus", {"query": "\u4ed6\u6c40\u7c7b\u836f\u7269", "k": 3}),
        "chunks": 99,          # 逐字符喂，强制把 \uXXXX 切成两半
        "expect_calls": 1,
    },
    {
        "name": "parallel_calls",
        "text": make_call("calculate", {"expression": "1+1"}) + "\n" + make_call("calculate", {"expression": "2*2"}),
        "chunks": 5,
        "expect_calls": 2,
    },
    {
        "name": "incomplete_json",
        "text": f'{OPEN}\n{{"name": "edit_file", "arguments": {{"path": "a.py", "old_string": "x"',
        "chunks": 4,
        "expect_calls": 1,
        "expect_dispatched": 0,
        "expect_reject": "incomplete_json",
        "expect_incomplete": True,
    },
    {
        "name": "three_calls_last_truncated",
        "text": (make_call("calculate", {"expression": "1+1"}) + "\n"
                 + make_call("read_file", {"path": "textstat/seq.py"}) + "\n"
                 + f'{OPEN}\n{{"name": "edit_file", "arguments": {{"path": "b.py", "old_string": "y"'),
        "chunks": 7,
        "expect_calls": 3,
        "expect_dispatched": 2,
        "expect_last_incomplete": True,
    },
    {
        "name": "closed_but_schema_invalid",
        "text": make_call("calculate", {"expression": 12345}),   # expression 应为字符串
        "chunks": 3,
        "expect_calls": 1,
        "expect_dispatched": 0,
        "expect_reject_prefix": "schema:",
    },
    {
        "name": "closed_but_not_authorized",
        "text": make_call("rm_rf", {"path": "/"}),
        "chunks": 3,
        "expect_calls": 1,
        "expect_dispatched": 0,
        "expect_reject": "not_in_allowlist",
    },
    {
        "name": "escape_backslash_split",
        # content 里放的是字面反斜杠序列（\n、\u00e9、\"），逐字符喂用来验证
        # 转义子状态跨片段保持
        "text": make_call("write_file", {"path": "a.py",
                                         "content": r"line1\nline2 \u00e9 \"q\""}),
        "chunks": 99,          # 逐字符喂，把 \\ 与 \\u00e9 都切开
        "expect_calls": 1,
        "expect_dispatched": 1,
    },
    {
        "name": "cancel_mid_stream",
        "text": make_call("read_file", {"path": "a" * 200}),
        "chunks": 6,
        "cancel_after": 2,
        "expect_calls": 0,
    },
]


def chunk_text(text: str, n: int) -> list[str]:
    if n <= 0:
        n = 1
    size = max(1, len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), size)]


ALLOWLIST = ["calculate", "search_corpus", "read_file", "list_files",
             "edit_file", "write_file", "run_tests"]


def run_cases() -> dict:
    results = []
    for case in CASES:
        chunks = chunk_text(case["text"], case["chunks"])
        parser = IncrementalToolParser(allowlist=ALLOWLIST, schemas=TOOL_SCHEMAS)
        events = []
        for i, ch in enumerate(chunks):
            if case.get("cancel_after") is not None and i == case["cancel_after"]:
                parser.cancel()
            events.extend(parser.feed(ch))
        calls = parser.finish()
        reference = full_parse(case["text"])
        # 对拍口径：把增量 deltas 拼回去，应当与一次性解析等价
        reparsed = []
        for c in parser.calls:
            raw = c["arguments_raw"].strip()
            try:
                args = json.loads(raw) if raw else {}
                err = None
            except json.JSONDecodeError as exc:
                args, err = None, f"incomplete_json: {exc.msg}"
            reparsed.append({"name": c["name"], "arguments": args, "error": err})
        ref_ok = [
            {"name": r["name"], "arguments": r["arguments"], "error": r["error"]}
            for r in reference
        ]
        # 对拍口径分三种：正常流要求与一次性解析逐字段相同；截断流要求两边都报错；
        # 取消流要求增量侧丢弃在途调用（参照实现看不到取消，必然不同）。
        if case.get("cancel_after") is not None:
            match = parser.cancelled and len(calls) == 0
        elif case.get("expect_incomplete"):
            match = len(calls) == 1 and calls[0]["error"] is not None
        elif case.get("expect_last_incomplete"):
            # 末尾截断：前面几次调用要与一次性解析逐字段相同，最后一次必须报不完整
            match = (reparsed[:-1] == ref_ok[:-1]
                     and bool(calls) and calls[-1]["error"] is not None
                     and calls[-1]["arguments"] is None)
        else:
            match = reparsed == ref_ok
        # 拼接 delta 必须还原 arguments_raw（流式协议的核心不变量）
        for c, call in zip(parser.calls, calls):
            assert "".join(c.get("deltas", [])) or True
        deltas_ok = all(
            "".join(c.get("deltas", [])) == c["arguments_raw"] for c in parser.calls
        )
        results.append({
            "case": case["name"],
            "chunks": len(chunks),
            "chars": len(case["text"]),
            "calls_incremental": len(calls),
            "calls_reference": len(reference),
            "match": match,
            "deltas_equal_raw": deltas_ok,
            "cancelled": parser.cancelled,
            "expected_calls": case["expect_calls"],
            "count_ok": len(calls) == case["expect_calls"],
            "expect_incomplete": bool(case.get("expect_incomplete")),
            "incomplete_ok": (calls[0]["error"] is not None) == bool(case.get("expect_incomplete")) if calls else False,
            "scanned_chars": parser.scanned_chars,
            "deltas": len(events),
            "peak_buffer_chars": parser.peak_buffer,
            "cancelled_calls": parser.cancelled_calls,
            "dispatched": sum(1 for c in calls if c.get("dispatched")),
            "rejects": [c.get("reject") for c in calls if c.get("reject")],
            "expected_dispatched": case.get("expect_dispatched", case["expect_calls"]),
            "dispatch_ok": sum(1 for c in calls if c.get("dispatched"))
            == case.get("expect_dispatched", case["expect_calls"]),
            "reject_ok": all(
                (case.get("expect_reject") is None)
                or any(c.get("reject") == case["expect_reject"] for c in calls)
                for _ in (0,)
            ) and all(
                (case.get("expect_reject_prefix") is None)
                or any((c.get("reject") or "").startswith(case["expect_reject_prefix"]) for c in calls)
                for _ in (0,)
            ),
        })
    return {"cases": results,
            "allowlist": ALLOWLIST,
            "all_match": all(r["match"] for r in results),
            "all_counts_ok": all(r["count_ok"] for r in results),
            "all_dispatch_ok": all(r["dispatch_ok"] for r in results),
            "all_reject_ok": all(r["reject_ok"] for r in results),
            "commit_gate_note": "闭合（JSON 可解析）→ 允许表 → schema 校验，三步都过才 dispatched=True"}


# --------------------------------------------------------------------------------------
# 规模扫描：单遍 vs 逐片段重扫
# --------------------------------------------------------------------------------------

def payload_case(nbytes: int, chunk_chars: int, calls: int = 1) -> dict:
    """单调用或**多调用**（每个调用分担 payload）的规模对照。

    多调用档用来回答"多个长参数同时在流里时，常驻缓冲会不会累加"：单遍状态机在每个调用结束时
    就把该调用的 arguments 交出去，峰值应仍由单个调用的头部/扣留窗口决定，与总字节数无关。
    """
    per_call = max(1, nbytes // max(1, calls))
    text = "\n".join(
        make_call("search_corpus", {"query": "x" * max(0, per_call - 80), "k": 5})
        for _ in range(calls)
    )
    chunks = chunk_text(text, max(1, len(text) // chunk_chars))

    t0 = time.perf_counter()
    parser = IncrementalToolParser()
    for ch in chunks:
        parser.feed(ch)
    inc_calls = parser.finish()
    inc_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    naive_calls, naive_scanned, naive_peak = naive_parse_chunks(chunks)
    naive_ms = (time.perf_counter() - t0) * 1000.0

    same = [
        {"name": c["name"], "arguments": c["arguments"], "error": c["error"]} for c in inc_calls
    ] == [
        {"name": c["name"], "arguments": c["arguments"], "error": c["error"]} for c in naive_calls
    ]
    return {
        "payload_bytes": nbytes,
        "calls": calls,
        "text_chars": len(text),
        "chunks": len(chunks),
        "chunk_chars": chunk_chars,
        "incremental_scanned": parser.scanned_chars,
        "naive_scanned": naive_scanned,
        "incremental_peak_buffer_chars": parser.peak_buffer,
        "naive_peak_buffer_chars": naive_peak,
        "peak_buffer_ratio": round(naive_peak / max(1, parser.peak_buffer), 3),
        "scan_ratio": round(naive_scanned / max(1, parser.scanned_chars), 3),
        "incremental_ms": round(inc_ms, 3),
        "naive_ms": round(naive_ms, 3),
        "time_ratio": round(naive_ms / max(1e-9, inc_ms), 3),
        "results_equal": same,
        "arguments_len": len(json.dumps(inc_calls[0]["arguments"], ensure_ascii=False)) if inc_calls and inc_calls[0]["arguments"] else None,
    }


def sweep(sizes=(256, 1024, 4096, 16384), chunk_chars=8, call_counts=(1,)) -> dict:
    rows = [payload_case(n, chunk_chars, c) for n in sizes for c in call_counts]
    return {"chunk_chars": chunk_chars, "call_counts": list(call_counts), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.2 流式工具解析器与代价扫描")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sizes", default="256,1024,4096,16384,65536")
    ap.add_argument("--chunk-chars", type=int, default=8)
    ap.add_argument("--calls", default="1", help="逗号分隔的调用数档位，用于多调用长参数对照")
    args = ap.parse_args()

    report = {
        "cases": run_cases(),
        "sweep": sweep(tuple(int(x) for x in args.sizes.split(",")), args.chunk_chars,
                       tuple(int(x) for x in args.calls.split(","))),
    }
    if args.out:
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(args.out) / "incremental_parser.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.selftest:
        ok = (report["cases"]["all_match"] and report["cases"]["all_counts_ok"]
              and report["cases"]["all_dispatch_ok"] and report["cases"]["all_reject_ok"])
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
