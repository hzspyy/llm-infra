#!/usr/bin/env python3
"""L9.2 任务 C：同一工具的直接调用 / stdio MCP / Streamable HTTP MCP 三条路径对拍。

三条路径共用**同一个**工具实现与同一套参数/返回值：

* ``direct``——在客户端进程内直接调用工具函数，作为没有协议层的基线；
* ``stdio``——MCP 服务器是子进程，帧走 stdin/stdout（JSON-RPC，逐行）；
* ``http``——MCP 服务器是独立 HTTP 服务，帧走 Streamable HTTP（``json`` 或 ``sse`` 两种回包模式）。

每个采样点记录四段（客户端口径）：

1. ``client_queue_ms``：从调用提交到真正发出（受客户端并发上限约束的部分）；
2. ``server_ms``：服务端工具处理器内部自报的执行时间（由返回值携带 ``server_ms``）；
3. ``transport_ms``：``total_ms − client_queue_ms − server_ms``，含帧编解码、进程管道/HTTP 往返与调度；
4. ``result_bytes``：结果序列化后的字节数（用于核对背压与传输成本）。

工具故意做成「带显式延迟」的形式：``calculate(expression, delay_ms)``。延迟作为参数注入，
所以延迟档位不需要换 schema，服务端也无需重启。

子命令：

* ``serve-stdio`` —— 在 stdio 上提供 MCP 服务（客户端会以子进程方式启动同一个文件）。
* ``serve-http``  —— 在 HTTP 上提供 MCP 服务；``--json``/``--sse`` 选回包模式，``--idle-timeout`` 控会话过期。
* ``caps``         —— 打印 initialize 协商结果（协议版本、能力、serverInfo）与 tools/list 的 schema。
* ``bench``        —— 三/四条路径 × 并发 × 延迟的单轴扫描，写 ``mcp_path.json``。

用法（先起 HTTP 服务，再跑 bench）::

    python labs/L9/mcp_tool_path.py serve-http --port 8071 --json &
    python labs/L9/mcp_tool_path.py serve-http --port 8072 --sse &
    python labs/L9/mcp_tool_path.py caps --out out/9.2/mcp/caps
    python labs/L9/mcp_tool_path.py bench --out out/9.2/mcp/bench \
        --http-json http://127.0.0.1:8071/mcp --http-sse http://127.0.0.1:8072/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

PROTOCOL_PIN = "2025-11-25"   # 9.2 任务书钉住的 MCP 规范版本
DEFAULT_TIMEOUT = 30.0


# --------------------------------------------------------------------------------------
# 工具实现（三条路径共用）
# --------------------------------------------------------------------------------------

def calculate(expression: str, delay_ms: int = 0) -> dict:
    """AST 白名单整数求值；``delay_ms`` 用于注入服务端执行延迟。"""
    t0 = time.perf_counter()
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    try:
        value = T.safe_eval(str(expression))
        error = None
    except Exception as exc:  # noqa: BLE001 - 工具错误原样回给调用方
        value = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "value": value,
        "error": error,
        "server_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        "pid": os.getpid(),
    }


def build_server(name: str = "l9-tool"):
    """构造 MCP 服务端；stdio 与 HTTP 两种传输共用同一个实例。"""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name=name, version="0.1.0")

    @server.tool(name="calculate", description="求值一个只含整数与 + - * // % ** 的表达式")
    def calculate_tool(expression: str, delay_ms: int = 0) -> str:
        return json.dumps(calculate(expression, delay_ms), ensure_ascii=False)

    return server


# --------------------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------------------

def cmd_serve_stdio(args) -> int:
    server = build_server()
    asyncio.run(server.run_stdio_async())
    return 0


def cmd_serve_http(args) -> int:
    import uvicorn

    server = build_server()
    app = server.streamable_http_app(
        json_response=bool(args.json), session_idle_timeout=float(args.idle_timeout)
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------------------
# 客户端封装
# --------------------------------------------------------------------------------------

class StdioPath:
    """stdio 路径；``reuse=False`` 时每次调用重新拉起子进程并重新 initialize。"""

    kind = "stdio"

    def __init__(self, module: pathlib.Path, python: str, reuse: bool, timeout: float):
        self.module = module
        self.python = python
        self.reuse = reuse
        self.timeout = timeout
        self._stack = None
        self._session = None
        self.init_ms = None

    async def start(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        t0 = time.perf_counter()
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.python, args=[str(self.module), "serve-stdio"], cwd=str(self.module.parent.parent.parent)
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=self.timeout)
        )
        await self._session.initialize()
        self.init_ms = (time.perf_counter() - t0) * 1000.0

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def call(self, expression: str, delay_ms: int) -> dict:
        if not self.reuse:
            await self.start()
        result = await self._session.call_tool(
            "calculate", {"expression": expression, "delay_ms": delay_ms}
        )
        if not self.reuse:
            await self.stop()
        return _payload(result)


class HttpPath:
    """Streamable HTTP 路径；``reuse=False`` 时每次调用重新建连接与 initialize。"""

    def __init__(self, url: str, kind: str, reuse: bool, timeout: float):
        self.url = url
        self.kind = kind
        self.reuse = reuse
        self.timeout = timeout
        self._stack = None
        self._session = None
        self.init_ms = None

    async def start(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        t0 = time.perf_counter()
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(streamable_http_client(self.url))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=self.timeout)
        )
        await self._session.initialize()
        self.init_ms = (time.perf_counter() - t0) * 1000.0

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def call(self, expression: str, delay_ms: int) -> dict:
        if not self.reuse:
            await self.start()
        result = await self._session.call_tool(
            "calculate", {"expression": expression, "delay_ms": delay_ms}
        )
        if not self.reuse:
            await self.stop()
        return _payload(result)


def _payload(result) -> dict:
    """从 CallToolResult 取出工具返回的 JSON 文本。"""
    if getattr(result, "isError", False):
        return {"value": None, "error": "tool reported isError", "server_ms": 0.0}
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            return json.loads(text)
    return {"value": None, "error": "empty content", "server_ms": 0.0}


async def _direct_call(expression: str, delay_ms: int) -> dict:
    return calculate(expression, delay_ms)


# --------------------------------------------------------------------------------------
# caps：协商结果与 tools/list
# --------------------------------------------------------------------------------------

async def cmd_caps_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=[str(pathlib.Path(__file__).resolve()), "serve-stdio"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=DEFAULT_TIMEOUT) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            # 字段命名随 SDK 大版本变化（2.x 用 server_info/protocol_version），
            # 直接整对象 dump，避免按名字取值失败漏掉协商结果。
            init_dump = json.loads(init.model_dump_json())
            report = {
                "protocol_pin": PROTOCOL_PIN,
                "negotiated_protocol_version": init_dump.get("protocolVersion")
                or init_dump.get("protocol_version"),
                "initialize_result": init_dump,
                "tools": [json.loads(t.model_dump_json()) for t in tools.tools],
                "next_cursor": getattr(tools, "nextCursor", None) or getattr(tools, "next_cursor", None),
            }
    try:
        import importlib.metadata as md

        report["sdk_version"] = md.version("mcp")
        report["sdk_version_note"] = (
            "python-sdk 2.x 的握手版本上限与 9.2 钉住的 2025-11-25 不一定相同；"
            "negotiated_protocol_version 是实际生效值，与 pin 不同则按实际值解释协议行为"
        )
    except Exception:  # noqa: BLE001
        pass
    (out / "caps.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tools"}, ensure_ascii=False, indent=1))
    print("tools:", json.dumps(report["tools"], ensure_ascii=False)[:600])
    return 0


def cmd_caps(args) -> int:
    return asyncio.run(cmd_caps_async(args))


# --------------------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------------------

def _quantiles(values: list[float], qs=(0.0, 0.5, 0.95, 1.0)) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    out = {}
    for q in qs:
        out[f"p{int(q * 100)}"] = round(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))], 4)
    out["mean"] = round(statistics.fmean(vals), 4)
    out["n"] = len(vals)
    return out


async def _run_cell(path, concurrency: int, delay_ms: int, calls: int, value: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    records: list[dict] = []

    async def one(i: int) -> None:
        async with sem:
            t_submit = time.perf_counter()
            t_send = time.perf_counter()
            error = None
            payload = None
            encoded = 0
            try:
                if isinstance(path, str):
                    payload = await _direct_call(f"{value}+{i}", delay_ms)
                else:
                    payload = await path.call(f"{value}+{i}", delay_ms)
                encoded = len(json.dumps(payload, ensure_ascii=False).encode())
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            t_done = time.perf_counter()
            total_ms = (t_done - t_submit) * 1000.0
            server_ms = float((payload or {}).get("server_ms") or 0.0)
            queue_ms = (t_send - t_submit) * 1000.0
            records.append({
                "total_ms": round(total_ms, 4),
                "client_queue_ms": round(queue_ms, 4),
                "server_ms": round(server_ms, 4),
                "transport_ms": round(max(0.0, total_ms - queue_ms - server_ms), 4),
                "result_bytes": encoded,
                "value_ok": int(bool(payload) and payload.get("value") == value + i),
                "error": error,
            })

    t0 = time.perf_counter()
    await asyncio.gather(*[one(i) for i in range(calls)])
    wall = time.perf_counter() - t0
    ok = [r for r in records if not r["error"]]
    return {
        "concurrency": concurrency,
        "delay_ms": delay_ms,
        "calls": calls,
        "wall_s": round(wall, 4),
        "errors": sum(1 for r in records if r["error"]),
        "value_mismatches": sum(1 for r in ok if not r["value_ok"]),
        "total_ms": _quantiles([r["total_ms"] for r in ok]),
        "client_queue_ms": _quantiles([r["client_queue_ms"] for r in ok]),
        "server_ms": _quantiles([r["server_ms"] for r in ok]),
        "transport_ms": _quantiles([r["transport_ms"] for r in ok]),
        "result_bytes": _quantiles([float(r["result_bytes"]) for r in ok]),
        "setup_ms": None if isinstance(path, str) else path.init_ms,
        "sample_errors": [r["error"] for r in records if r["error"]][:3],
    }


async def cmd_bench_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    module = pathlib.Path(__file__).resolve()
    report: dict = {
        "protocol_pin": PROTOCOL_PIN,
        "calls_per_cell": args.calls,
        "concurrencies": [int(x) for x in args.concurrency.split(",")],
        "delays_ms": [int(x) for x in args.delay_ms.split(",")],
        "reuse": not args.no_reuse,
        "cells": {},
        "phases": {
            "client_queue_ms": "提交到发出（客户端并发上限内）",
            "server_ms": "服务端处理器自报执行时间（随结果返回）",
            "transport_ms": "total − queue − server：帧编解码 + 管道/HTTP 往返 + 调度",
            "result_bytes": "结果 JSON 的字节数",
        },
    }
    paths: list[tuple[str, object]] = [("direct", "direct")]
    if args.stdio:
        paths.append(("stdio", StdioPath(module, args.python or sys.executable, not args.no_reuse, args.timeout)))
    if args.http_json:
        paths.append(("http-json", HttpPath(args.http_json, "http-json", not args.no_reuse, args.timeout)))
    if args.http_sse:
        paths.append(("http-sse", HttpPath(args.http_sse, "http-sse", not args.no_reuse, args.timeout)))

    for name, path in paths:
        live = not isinstance(path, str)
        if live and not args.no_reuse:
            await path.start()
            report.setdefault("setup_ms", {})[name] = round(path.init_ms or 0.0, 4)
        for delay in report["delays_ms"]:
            for conc in report["concurrencies"]:
                cell = await _run_cell(path, conc, delay, args.calls, args.value)
                report["cells"][f"{name}|c{conc}|d{delay}"] = cell
                print(
                    f"[{name}] c={conc} delay={delay}ms wall={cell['wall_s']}s "
                    f"total p50={cell['total_ms'].get('p50')} p95={cell['total_ms'].get('p95')} "
                    f"server p50={cell['server_ms'].get('p50')} transport p50={cell['transport_ms'].get('p50')} "
                    f"err={cell['errors']} mism={cell['value_mismatches']}",
                    flush=True,
                )
        if live and not args.no_reuse:
            await path.stop()

    (out / "mcp_path.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_bench(args) -> int:
    return asyncio.run(cmd_bench_async(args))


# --------------------------------------------------------------------------------------
# next-turn：工具结果回填后的下一轮首 token（引擎侧，含命中与排队）
# --------------------------------------------------------------------------------------

def _tool_result_messages(result_json: str, filler: int) -> list[dict]:
    """构造「assistant 发起 calculate 调用 + 工具返回结果」的下一轮上下文。

    ``filler`` 用重复的空白段把工具结果撑到指定字节数，用来测结果体积对下一轮 prefill 的影响。
    """
    body = result_json if filler <= 0 else result_json + (" " * filler)
    return [
        {"role": "system", "content": "你是一个只做整数运算的助手。"},
        {"role": "user", "content": "请计算 1234+5678，并只回一个整数。"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "calculate", "arguments": json.dumps({"expression": "1234+5678"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": body},
    ]


async def cmd_next_turn_async(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 从 bench 结果里取各路径的实测结果体积，作为下一轮上下文的规模依据
    sizes: dict[str, int] = {}
    bench_path = pathlib.Path(args.bench)
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        for key, cell in bench["cells"].items():
            path = key.split("|")[0]
            if cell.get("delay_ms") == 0 and cell.get("concurrency") == 1:
                size = int((cell.get("result_bytes") or {}).get("p50") or 0)
                sizes[path] = size
    if not sizes:
        sizes = {"direct": 96}

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    rows = []
    for path, size in sorted(sizes.items()):
        for payload_bytes in args.payload_bytes.split(","):
            target = int(payload_bytes)
            for round_idx in range(args.rounds):
                messages = _tool_result_messages('{"value":6912,"error":null}', max(0, target - 32))
                t0 = time.perf_counter()
                ttft = None
                prompt_tokens = cached = None
                stream = await client.chat.completions.create(
                    model=args.model, messages=messages, max_tokens=args.max_tokens,
                    temperature=0.0, stream=True, stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    if chunk.usage is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                        d = getattr(chunk.usage, "prompt_tokens_details", None)
                        if d is not None:
                            cached = getattr(d, "cached_tokens", None)
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if (getattr(delta, "content", None) or getattr(delta, "reasoning", None)
                            or getattr(delta, "tool_calls", None)):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000.0
                e2e = (time.perf_counter() - t0) * 1000.0
                rows.append({
                    "path": path, "bench_result_bytes_p50": size,
                    "constructed_payload_bytes": target, "round": round_idx,
                    "ttft_ms": round(ttft, 3) if ttft is not None else None,
                    "e2e_ms": round(e2e, 3),
                    "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                })
                print(f"[next-turn] {path} payload={target}B round={round_idx} "
                      f"ttft={rows[-1]['ttft_ms']} prompt={prompt_tokens} cached={cached}", flush=True)
    await client.close()

    summary: dict = {"base_url": args.base_url, "model": args.model, "rows": rows, "by_cell": {}}
    keys = {(r["path"], r["constructed_payload_bytes"]) for r in rows}
    for path, payload in sorted(keys):
        sel = [r for r in rows if r["path"] == path and r["constructed_payload_bytes"] == payload]
        # 第一轮是冷前缀，后续轮命中同一前缀；两者分开报
        summary["by_cell"][f"{path}|{payload}B"] = {
            "first_round_ttft_ms": sel[0]["ttft_ms"],
            "later_rounds_ttft_ms": _quantiles([r["ttft_ms"] for r in sel[1:]]),
            "prompt_tokens": sel[-1]["prompt_tokens"],
            "cached_tokens_first": sel[0]["cached_tokens"],
            "cached_tokens_later": sel[-1]["cached_tokens"],
        }
    (out / "next_turn.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["by_cell"], ensure_ascii=False, indent=1))
    return 0


def cmd_next_turn(args) -> int:
    return asyncio.run(cmd_next_turn_async(args))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="L9.2 工具调用三条路径对拍")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve-stdio")
    p.set_defaults(func=cmd_serve_stdio)

    p = sub.add_parser("serve-http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8071)
    p.add_argument("--json", action="store_true", help="用 JSON 回包（默认 SSE 流）")
    p.add_argument("--sse", action="store_true", help="显式选择 SSE 回包")
    p.add_argument("--idle-timeout", type=float, default=1800.0)
    p.set_defaults(func=cmd_serve_http)

    p = sub.add_parser("caps")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_caps)

    p = sub.add_parser("bench")
    p.add_argument("--out", required=True)
    p.add_argument("--stdio", action="store_true", help="加入 stdio 路径")
    p.add_argument("--http-json", default=None)
    p.add_argument("--http-sse", default=None)
    p.add_argument("--concurrency", default="1,8,32")
    p.add_argument("--delay-ms", default="0,10,1000")
    p.add_argument("--calls", type=int, default=64)
    p.add_argument("--value", type=int, default=1000)
    p.add_argument("--no-reuse", action="store_true", help="每次调用重建连接并重新 initialize")
    p.add_argument("--python", default=None)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("next-turn")
    p.add_argument("--out", required=True)
    p.add_argument("--bench", required=True, help="bench 写出的 mcp_path.json，用于取各路径的结果体积")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--payload-bytes", default="96,4096,65536")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_next_turn)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
