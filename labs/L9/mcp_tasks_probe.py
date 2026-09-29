#!/usr/bin/env python3
"""L9.2 任务 D：MCP Tasks 扩展的能力协商、轮询、结果取得与取消回滚边界。

9.2 的失败矩阵已经覆盖断连、重复回包、晚到结果、schema 变更与过期会话；本轮补的是任务
书里剩下的那一格：**Tasks 扩展**。它需要先协商 capability，再测轮询（`tasks/get`）、结果
取得（`tasks/result`）与 `tasks/cancel`；同时把「普通取消通知不保证副作用回滚」变成可数
的账本记录，而不是一句规范引文。

脚本自带一个最小 MCP 服务（stdio）与一个客户端。服务端每次工具调用都往
``--state-dir/effects.jsonl`` 追加副作用记录，因此「取消后副作用有没有回滚」可以直接数。

四个用例：

* ``task_augmented_call``：带 ``task`` 元数据的 ``tools/call`` → 立刻返回 ``CreateTaskResult``，
  轮询 ``tasks/get`` 到终态，再 ``tasks/result`` 取结果；记录协商到的 ``capabilities.tasks``。
* ``task_cancel``：任务运行中发 ``tasks/cancel``。服务端的 worker 与请求作用域解耦，所以
  「已提交的副作用」不会因为任务被取消而消失——这正是账本要证明的点。
* ``abandon_last``／``abandon_first``：客户端放弃一个在飞请求（asyncio 取消 → SDK 发
  ``notifications/cancelled``），对照两种工具实现：副作用在等待**之后**提交（可协作取消，
  无副作用）与在等待**之前**提交（已发生的副作用不因取消而回滚）。
* ``sdk_boundary``：记录本机 SDK 对 Tasks 的实际支持范围（类型存在、方法表刻意不含
  ``tasks/*``、``tasks/result`` 载荷类型只有 ``meta``、取消模式默认 ``interrupt``），
  逐条带文件与行号。

用法::

    /Volumes/data/venvs/mcp/bin/python labs/L9/mcp_tasks_probe.py run \\
        --out results/local/9.2/20260922-mcp-tasks --state-dir results/local/9.2/20260922-mcp-tasks/state
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import uuid

PROTOCOL_PIN = "2025-11-25"
TOOLS = ("echo", "slow_write")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _append_effect(state_dir: pathlib.Path, record: dict) -> None:
    record = {"at": _now(), **record}
    fd = os.open(state_dir / "effects.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(record, ensure_ascii=False) + "\n").encode())
    finally:
        os.close(fd)


# --------------------------------------------------------------------------------------
# 服务端：lowlevel Server + Tasks 扩展处理器
# --------------------------------------------------------------------------------------
def build_server(state_dir: pathlib.Path):
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server

    state_file = state_dir / "tasks.json"

    def load_state() -> dict:
        if state_file.exists():
            return json.loads(state_file.read_text(encoding="utf-8"))
        return {}

    def save_state(state: dict) -> None:
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(state_file)

    async def list_tools(ctx, params):
        schema = {"type": "object", "properties": {
            "key": {"type": "string"}, "seconds": {"type": "number"},
            "commit": {"type": "string", "enum": ["first", "last"]}},
            "required": ["key", "seconds"]}
        return types.ListToolsResult(tools=[
            types.Tool(name="echo", description="立即返回输入",
                       input_schema={"type": "object", "properties": {"text": {"type": "string"}}}),
            types.Tool(name="slow_write", description="等待 seconds 后写一条副作用账本记录",
                       input_schema=schema,
                       execution=types.ToolExecution(task_support="optional")),
            types.Tool(name="start_task",
                       description="显式创建一个后台任务并立刻返回 task_id（用于在 SDK "
                                   "拒绝把 CreateTaskResult 作为 tools/call 结果时仍能测 "
                                   "tasks/get、tasks/cancel 与 tasks/result）",
                       input_schema=schema,
                       execution=types.ToolExecution(task_support="optional")),
        ])

    async def run_work(task_id: str, key: str, seconds: float, commit: str) -> None:
        """任务 worker：与请求作用域解耦，运行在自己的 asyncio task 里。"""
        if commit == "first":
            _append_effect(state_dir, {"key": key, "task_id": task_id, "commit": "first",
                                       "phase": "before_wait"})
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            raise
        _append_effect(state_dir, {"key": key, "task_id": task_id, "commit": commit,
                                   "phase": "after_wait"})
        st = load_state()
        if task_id in st:
            st[task_id].update(status="completed", last_updated_at=_now(),
                               status_message=f"wrote {key}")
            save_state(st)

    async def call_tool(ctx, params: types.CallToolRequestParams):
        name = params.name
        args = params.arguments or {}
        if name == "echo":
            return types.CallToolResult(content=[types.TextContent(type="text",
                                                                  text=str(args.get("text", "")))])
        key = str(args.get("key", "k"))
        seconds = float(args.get("seconds", 0.5))
        commit = str(args.get("commit", "last"))
        task_meta = getattr(params, "task", None)
        if name == "start_task" or task_meta is not None:
            ttl = task_meta.ttl if task_meta is not None else 60000
            task_id = f"task-{uuid.uuid4().hex[:12]}"
            st = load_state()
            st[task_id] = {"task_id": task_id, "status": "working", "status_message": None,
                           "created_at": _now(), "last_updated_at": _now(),
                           "ttl": ttl, "poll_interval": 100, "key": key}
            save_state(st)
            asyncio.create_task(run_work(task_id, key, seconds, commit))
            if task_meta is not None:
                # 规范里任务增强调用应返回 CreateTaskResult；本机 SDK 服务端按
                # tools/call 的结果面校验它（SERVER_RESULTS 只有 CallToolResult），
                # 因此这条分支会被判成 invalid result——这是要记录的边界。
                return types.CreateTaskResult(task=types.Task(
                    task_id=task_id, status="working", created_at=st[task_id]["created_at"],
                    last_updated_at=st[task_id]["last_updated_at"], ttl=ttl,
                    poll_interval=100))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=task_id)],
                structured_content={"task_id": task_id})
        if name != "slow_write":
            raise ValueError(f"unknown tool {name}")
        # 普通调用：`commit=last` 先等待后提交（取消能拦住副作用），`first` 先提交后
        # 等待（已发生的副作用不因取消而回滚）——两种实现都由账本逐条数出来。
        if commit == "last":
            await asyncio.sleep(seconds)
            _append_effect(state_dir, {"key": key, "task_id": None, "commit": commit,
                                       "phase": "after_wait"})
        else:
            _append_effect(state_dir, {"key": key, "task_id": None, "commit": commit,
                                       "phase": "before_wait"})
            await asyncio.sleep(seconds)
        return types.CallToolResult(content=[types.TextContent(
            type="text", text=f"inline {key} after {seconds}s")])

    async def tasks_get(ctx, params: types.GetTaskRequestParams):
        st = load_state()
        row = st.get(params.task_id)
        if row is None:
            raise ValueError(f"unknown task {params.task_id}")
        return types.GetTaskResult(**{k: row[k] for k in
                                      ("task_id", "status", "status_message", "created_at",
                                       "last_updated_at", "ttl", "poll_interval")})

    async def tasks_cancel(ctx, params: types.CancelTaskRequestParams):
        st = load_state()
        row = st.get(params.task_id)
        if row is None:
            raise ValueError(f"unknown task {params.task_id}")
        if row["status"] == "working":
            row.update(status="cancelled", last_updated_at=_now(),
                       status_message="cancel requested; 已提交的副作用不回滚")
            save_state(st)
        return types.CancelTaskResult(**{k: row[k] for k in
                                         ("task_id", "status", "status_message", "created_at",
                                          "last_updated_at", "ttl", "poll_interval")})

    async def tasks_result(ctx, params: types.GetTaskPayloadRequestParams):
        st = load_state()
        row = st.get(params.task_id)
        if row is None:
            raise ValueError(f"unknown task {params.task_id}")
        return types.GetTaskPayloadResult(meta={"task": row})

    async def on_cancelled(ctx, params: types.CancelledNotificationParams) -> None:
        """直接记录服务端收到的 `notifications/cancelled`，避免只靠副作用反推。"""
        _append_effect(state_dir, {"kind": "notifications/cancelled",
                                   "request_id": str(getattr(params, "request_id", None)),
                                   "reason": getattr(params, "reason", None)})

    server = Server("l9-mcp-tasks", version="1.0.0", on_list_tools=list_tools)
    server.add_request_handler("tools/call", types.CallToolRequestParams, call_tool)
    server.add_request_handler("tasks/get", types.GetTaskRequestParams, tasks_get)
    server.add_request_handler("tasks/cancel", types.CancelTaskRequestParams, tasks_cancel)
    server.add_request_handler("tasks/result", types.GetTaskPayloadRequestParams, tasks_result)
    server.add_notification_handler("notifications/cancelled",
                                    types.CancelledNotificationParams, on_cancelled)
    _ = anyio  # 保持导入以便与 SDK 相同的运行环境
    return server


def tasks_capabilities():
    """把一个完整的 Tasks 能力块显式声明出去（SDK 不会从处理器自动推导 tasks）。"""
    import mcp.types as types

    return types.ServerCapabilities(
        tools=types.ToolsCapability(),
        tasks=types.ServerTasksCapability(
            list=types.TasksListCapability(),
            cancel=types.TasksCancelCapability(),
            requests=types.ServerTasksRequestsCapability(
                tools=types.TasksToolsCapability(call=types.TasksCallCapability())),
        ),
    )


async def serve(state_dir: pathlib.Path) -> None:
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    server = build_server(state_dir)
    opts = InitializationOptions(server_name="l9-mcp-tasks", server_version="1.0.0",
                                 capabilities=tasks_capabilities())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, opts)


# --------------------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------------------
def _sdk_evidence() -> list[dict]:
    """把本机 SDK 里与 Tasks 支持范围相关的源码位置逐条记下来。"""
    import mcp
    import mcp_types

    out: list[dict] = []
    targets = [
        (pathlib.Path(mcp_types.__file__).parent / "methods.py",
         ["tasks/* deliberately absent", "tasks/status deliberately absent"]),
        (pathlib.Path(mcp.__file__).parent / "server" / "extension.py",
         ["e.g. `tasks/get`"]),
        (pathlib.Path(mcp.__file__).parent / "shared" / "jsonrpc_dispatcher.py",
         ["PeerCancelMode", "cancels the handler's scope"]),
    ]
    for path, needles in targets:
        if not path.exists():
            out.append({"file": str(path), "missing": True})
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for needle in needles:
            for i, line in enumerate(lines, 1):
                if needle in line:
                    out.append({"file": str(path), "line": i, "text": line.strip(),
                                "needle": needle})
                    break
    import mcp.types as types

    out.append({"type": "GetTaskPayloadResult.model_fields",
                "value": list(types.GetTaskPayloadResult.model_fields)})
    out.append({"type": "CallToolRequestParams.has_task",
                "value": "task" in types.CallToolRequestParams.model_fields})
    return out


async def client_run(out_path: pathlib.Path, state_dir: pathlib.Path) -> dict:
    import mcp.types as types
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    script = str(pathlib.Path(__file__).resolve())
    params = StdioServerParameters(command=sys.executable,
                                   args=[script, "serve", "--state-dir", str(state_dir)])
    result: dict = {"sdk_evidence": _sdk_evidence(), "cases": {}}

    def ledger() -> list[dict]:
        f = state_dir / "effects.jsonl"
        if not f.exists():
            return []
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

    def effect_keys(key: str) -> list[dict]:
        return [r for r in ledger() if r.get("key") == key]

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            result["initialize"] = {
                "protocol_version": init.protocol_version,
                "capabilities": init.capabilities.model_dump(exclude_none=True, by_alias=True),
                "server_info": init.server_info.model_dump(exclude_none=True, by_alias=True),
            }
            result["capabilities_tasks_present"] = init.capabilities.tasks is not None

            tools = await session.list_tools()
            result["tools"] = [t.model_dump(exclude_none=True, by_alias=True) for t in tools.tools]

            echo = await session.call_tool("echo", {"text": "hi"})
            result["cases"]["plain_call"] = {"is_error": echo.is_error,
                                             "text": echo.content[0].text if echo.content else None}

            # ---- 用例 1：任务增强 tools/call（规范路径）在本机 SDK 上的结果 ---------
            from pydantic import TypeAdapter

            union = TypeAdapter(types.CallToolResult | types.CreateTaskResult)
            req = types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="slow_write", arguments={"key": "t1", "seconds": 0.8, "commit": "last"},
                    task=types.TaskMetadata(ttl=60000)))
            case1: dict = {}
            try:
                raw = await session.send_request(req, result_type=union)
                case1["result_type"] = type(raw).__name__
                case1["result"] = raw.model_dump(exclude_none=True, by_alias=True)
            except Exception as exc:  # noqa: BLE001
                case1["error"] = f"{type(exc).__name__}: {exc}"
            case1["effects"] = effect_keys("t1")
            result["cases"]["task_augmented_call"] = case1

            # ---- 用例 2：显式建任务 → 轮询 tasks/get → tasks/result ----------------
            started = await session.call_tool(
                "start_task", {"key": "t2", "seconds": 0.8, "commit": "last"})
            tid = (started.structured_content or {}).get("task_id") or (
                started.content[0].text if started.content else None)
            case2: dict = {"task_id": tid, "start_call_structured": started.structured_content}
            polls = []
            for _ in range(60):
                g = await session.send_request(
                    types.GetTaskRequest(params=types.GetTaskRequestParams(task_id=tid)),
                    result_type=types.GetTaskResult)
                polls.append({"status": g.status, "updated": g.last_updated_at})
                if g.status in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.15)
            case2["polls"] = polls
            try:
                payload = await session.send_request(
                    types.GetTaskPayloadRequest(
                        params=types.GetTaskPayloadRequestParams(task_id=tid)),
                    result_type=types.GetTaskPayloadResult)
                case2["task_result_visible_fields"] = payload.model_dump(exclude_none=True,
                                                                         by_alias=True)
            except Exception as exc:  # noqa: BLE001
                case2["task_result_error"] = f"{type(exc).__name__}: {exc}"
            case2["effects"] = effect_keys("t2")
            result["cases"]["task_poll_and_result"] = case2

            # ---- 用例 3：任务运行中 tasks/cancel，副作用是否回滚 --------------------
            started3 = await session.call_tool(
                "start_task", {"key": "t3", "seconds": 2.0, "commit": "first"})
            tid3 = (started3.structured_content or {}).get("task_id")
            case3: dict = {"task_id": tid3}
            await asyncio.sleep(0.4)
            cancelled = await session.send_request(
                types.CancelTaskRequest(params=types.CancelTaskRequestParams(task_id=tid3)),
                result_type=types.CancelTaskResult)
            case3["cancel_result"] = cancelled.model_dump(exclude_none=True, by_alias=True)
            await asyncio.sleep(2.2)          # 等 worker 自己跑完
            g3 = await session.send_request(
                types.GetTaskRequest(params=types.GetTaskRequestParams(task_id=tid3)),
                result_type=types.GetTaskResult)
            case3["status_after_worker_finishes"] = g3.model_dump(exclude_none=True, by_alias=True)
            case3["effects"] = effect_keys("t3")
            case3["effects_rolled_back_by_cancel"] = not effect_keys("t3")
            result["cases"]["task_cancel"] = case3

            # ---- 用例 3：放弃在飞请求（SDK 发 notifications/cancelled）→ 副作用 ----
            for mode in ("last", "first"):
                key = f"abandon-{mode}"
                call = asyncio.create_task(
                    session.call_tool("slow_write",
                                      {"key": key, "seconds": 1.5, "commit": mode}))
                await asyncio.sleep(0.4)
                call.cancel()
                outcome = None
                try:
                    await call
                except asyncio.CancelledError:
                    outcome = "CancelledError"
                except Exception as exc:  # noqa: BLE001
                    outcome = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1.6)
                result["cases"][f"abandon_{mode}"] = {
                    "client_outcome": outcome,
                    "effects": effect_keys(key),
                    "side_effect_survived": bool(effect_keys(key)),
                    "server_saw_cancelled_notification": [
                        r for r in ledger() if r.get("kind") == "notifications/cancelled"],
                }
    result["ledger_total"] = len(ledger())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.2-D MCP Tasks 扩展核对")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--state-dir", required=True)

    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--state-dir", required=True)
    args = ap.parse_args()

    state_dir = pathlib.Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "serve":
        asyncio.run(serve(state_dir))
        return 0
    res = asyncio.run(client_run(pathlib.Path(args.out), state_dir))
    print(json.dumps({k: res[k] for k in ("initialize", "capabilities_tasks_present",
                                          "tools", "cases")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
