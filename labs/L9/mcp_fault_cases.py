#!/usr/bin/env python3
"""L9.2 任务 D：MCP 客户端的失败路径与授权边界。

覆盖 9.2-D 要求的注入：断连、重复回包/重复执行、晚到结果、schema 变更、过期会话、
HTTP 授权上下文与工具 allowlist；Tasks 扩展先协商 capability 再决定是否可测。

服务端每次工具调用都会往 ``--state-dir`` 里的账本追加一条记录，所以「协议层看不到的
副作用」可以被逐条数出来。工具带一个业务幂等键 ``idem_key``：服务端在同一条提交边界里
写账本与去重缓存，重复键返回首次结果、不再产生第二条副作用。

子命令：

* ``serve-http`` —— 起一个可注入故障的 MCP 服务：``--schema v1|v2``（v2 的 ``mode`` 为必填，
  用于制造「客户端 schema 过期」）、``--no-idem``（关闭幂等去重，用于对照重复执行）、
  ``--require-token``（缺 Authorization 直接 401）。
* ``run``        —— 顺序执行全部用例并写 ``mcp_faults.json``。

用法::

    python labs/L9/mcp_fault_cases.py run --out out/9.2/mcp-faults
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Context 必须在模块级可解析：工具函数带 `from __future__ import annotations`，SDK 用
# get_type_hints 还原注解来识别 Context 参数，注解里的名字必须在模块全局可见。
try:
    from mcp.server.mcpserver import Context
except Exception:  # noqa: BLE001 - 缺 SDK 时给出明确报错，而不是导入期崩溃
    Context = None  # type: ignore[assignment]

PROTOCOL_PIN = "2025-11-25"


# --------------------------------------------------------------------------------------
# 服务端：带账本与幂等去重的工具
# --------------------------------------------------------------------------------------

def _ledger_append(state_dir: pathlib.Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    fd = os.open(state_dir / "effects.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def _idem_lookup(state_dir: pathlib.Path, key: str) -> dict | None:
    path = state_dir / "idem.json"
    if not path.exists():
        return None
    import fcntl
    with open(path, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            data = {}
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return data.get(key)


def _idem_store(state_dir: pathlib.Path, key: str, value: dict) -> None:
    import fcntl
    path = state_dir / "idem.json"
    path.touch(exist_ok=True)
    with open(path, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            data = {}
        data[key] = value
        fh.seek(0)
        fh.truncate()
        json.dump(data, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def build_server(state_dir: pathlib.Path, schema: str, use_idem: bool,
                 require_token: str | None, effect_delay_ms: int = 0):
    """构造故障可注入的 MCP 服务端。"""
    from mcp.server.mcpserver import MCPServer

    kwargs: dict = {"name": f"l9-fault-{schema}", "version": "0.1.0"}
    if require_token:
        from mcp.server.auth.provider import AccessToken, TokenVerifier
        from mcp.server.auth.settings import AuthSettings

        class _Verifier(TokenVerifier):
            async def verify_token(self, token: str) -> AccessToken | None:
                if token != require_token:
                    return None
                return AccessToken(token=token, client_id="fault-client", scopes=["tools"],
                                   subject="tenant-a")

        kwargs["token_verifier"] = _Verifier()
        kwargs["auth"] = AuthSettings(issuer_url="https://fault.local",
                                      resource_server_url="http://127.0.0.1/mcp",
                                      required_scopes=["tools"])
    server = MCPServer(**kwargs)

    if schema == "v1":
        @server.tool(name="calculate", description="求值整数表达式（v1 schema）")
        def calculate_v1(expression: str, delay_ms: int = 0, idem_key: str | None = None,
                         ctx: Context | None = None) -> str:
            return _run_tool(state_dir, expression, delay_ms, idem_key, use_idem,
                             effect_delay_ms, ctx, schema="v1")

        @server.tool(name="echo", description="回显一个字符串")
        def echo_v1(text: str) -> str:
            return text
    else:
        @server.tool(name="calculate",
                     description="求值整数表达式（v2 schema：mode 必填）")
        def calculate_v2(expression: str, mode: str, delay_ms: int = 0,
                         idem_key: str | None = None, ctx: Context | None = None) -> str:
            if mode not in ("plain",):
                raise ValueError(f"unsupported mode {mode!r}")
            return _run_tool(state_dir, expression, delay_ms, idem_key, use_idem,
                             effect_delay_ms, ctx, schema="v2")

        @server.tool(name="echo", description="回显一个字符串")
        def echo_v2(text: str) -> str:
            return text

    return server


def _run_tool(state_dir: pathlib.Path, expression: str, delay_ms: int, idem_key: str | None,
              use_idem: bool, effect_delay_ms: int, ctx, schema: str) -> str:
    import agent_tasks as T

    t0 = time.perf_counter()
    auth_header = None
    if ctx is not None:
        try:
            headers = ctx.headers or {}
            auth_header = headers.get("authorization")
        except Exception:  # noqa: BLE001
            auth_header = None

    if use_idem and idem_key:
        cached = _idem_lookup(state_dir, idem_key)
        if cached is not None:
            return json.dumps({**cached, "dedup_hit": True,
                               "server_ms": round((time.perf_counter() - t0) * 1000.0, 4)},
                              ensure_ascii=False)

    if effect_delay_ms > 0:
        time.sleep(effect_delay_ms / 1000.0)
    try:
        value = T.safe_eval(str(expression))
        error = None
    except Exception as exc:  # noqa: BLE001
        value = None
        error = f"{type(exc).__name__}: {exc}"

    effect = {
        "effect_id": uuid.uuid4().hex[:12],
        "idem_key": idem_key,
        "expression": expression,
        "value": value,
        "error": error,
        "schema": schema,
        "auth_present": bool(auth_header),
        "auth_masked": (auth_header[:14] + "...") if auth_header else None,
        "ts": round(time.time(), 4),
    }
    _ledger_append(state_dir, effect)
    if use_idem and idem_key:
        _idem_store(state_dir, idem_key, {"value": value, "error": error,
                                          "effect_id": effect["effect_id"]})
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    return json.dumps({"value": value, "error": error, "effect_id": effect["effect_id"],
                       "dedup_hit": False,
                       "server_ms": round((time.perf_counter() - t0) * 1000.0, 4)},
                      ensure_ascii=False)


def cmd_serve_http(args) -> int:
    import uvicorn

    state_dir = pathlib.Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    server = build_server(state_dir, args.schema, not args.no_idem,
                          args.require_token, args.effect_delay_ms)
    app = server.streamable_http_app(json_response=not args.sse,
                                     session_idle_timeout=float(args.idle_timeout))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------------------

async def _session(url: str, token: str | None = None, timeout: float = 30.0):
    """建立一个 Streamable HTTP 会话；返回 (exit_stack, session)。"""
    from contextlib import AsyncExitStack

    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    stack = AsyncExitStack()
    client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"} if token else None)
    await stack.enter_async_context(client)
    read, write = await stack.enter_async_context(
        streamable_http_client(url, http_client=client)
    )
    session = await stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=timeout))
    init = await session.initialize()
    return stack, session, init


def _schema_of(tool) -> dict:
    """取工具的输入 schema；SDK 2.x 用 input_schema，旧版用 inputSchema。"""
    for attr in ("input_schema", "inputSchema"):
        val = getattr(tool, attr, None)
        if val is not None:
            return val if isinstance(val, dict) else json.loads(json.dumps(val))
    return {}


async def _one_shot(url: str, expression: str, extra: dict | None = None,
                    token: str | None = None, timeout: float = 30.0,
                    state_dir: pathlib.Path | None = None, allowlist: list[str] | None = None):
    """一次调用：返回 (ok, payload_or_error, server_effect_count_delta)。"""
    args = {"expression": expression}
    if extra:
        args.update(extra)
    if allowlist is not None and args.get("tool", "calculate") not in allowlist:
        return False, "refused by client allowlist", 0
    before = _count_effects(state_dir) if state_dir else None
    stack = None
    try:
        stack, session, _ = await _session(url, token=token, timeout=timeout)
        if allowlist is not None and "calculate" not in allowlist:
            return False, "refused by client allowlist", 0
        result = await session.call_tool("calculate", args)
        payload = json.loads(result.content[0].text) if result.content else {}
        ok = not getattr(result, "isError", False) and payload.get("error") is None
        return ok, payload, (_count_effects(state_dir) - before) if state_dir else 0
    except BaseException as exc:  # noqa: BLE001 - BaseExceptionGroup 里可能含 CancelledError
        return False, f"{type(exc).__name__}: {exc}"[:300], (_count_effects(state_dir) - before) if state_dir else 0
    finally:
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException:  # noqa: BLE001
                pass


def _count_effects(state_dir: pathlib.Path | None) -> int:
    if state_dir is None:
        return 0
    path = state_dir / "effects.jsonl"
    if not path.exists():
        return 0
    return sum(1 for _ in open(path, encoding="utf-8"))


_SPAWNED: list[subprocess.Popen] = []


def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _stop_all() -> None:
    for p in _SPAWNED:
        p.terminate()
    for p in _SPAWNED:
        try:
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            p.kill()
    _SPAWNED.clear()


def _start_server(args, schema: str, port: int, state_sub: str, extra: list[str]) -> subprocess.Popen:
    state_dir = pathlib.Path(args.state_dir) / state_sub
    state_dir.mkdir(parents=True, exist_ok=True)
    # 先确认端口空闲：残留的旧服务会让本次请求打到错误配置上，结果看起来"成功"却无效
    if _port_in_use(port):
        raise RuntimeError(f"port {port} 已被占用（疑似残留服务），先清理再跑")
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "serve-http",
           "--port", str(port), "--schema", schema, "--state-dir", str(state_dir)] + extra
    log = open(pathlib.Path(args.out) / f"server-{state_sub}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    _SPAWNED.append(proc)
    try:
        _wait_port(port)
    except RuntimeError:
        _stop_all()
        raise
    return proc


def _wait_port(port: int, tries: int = 60) -> None:
    import socket
    for _ in range(tries):
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise RuntimeError(f"port {port} not listening")


async def _case_disconnect(args, url: str, state_dir: pathlib.Path) -> dict:
    """调用中途杀掉服务进程：客户端必须报错；账本记录副作用是否已经发生。"""
    proc = _start_server(args, "v1", args.port_disconnect, "disconnect", ["--effect-delay-ms", "1500"])
    before = _count_effects(state_dir)
    async def call():
        return await _one_shot(url_disconnect(args), "7*8", {"idem_key": "disc-1"},
                               state_dir=state_dir, timeout=20)
    task = asyncio.create_task(call())
    await asyncio.sleep(0.6)
    proc.kill()
    proc.wait(timeout=10)
    ok, payload, _ = await task
    await asyncio.sleep(0.2)
    after = _count_effects(state_dir)
    return {
        "case": "disconnect_mid_call",
        "client_ok": ok,
        "client_result": payload if isinstance(payload, str) else json.dumps(payload)[:200],
        "effects_before": before,
        "effects_after": after,
        "effect_happened_despite_error": after > before,
        "note": "客户端报错不等于副作用没发生：副作用已提交但回包丢失时，重试必须靠业务幂等键",
    }


async def _case_disconnect_after_effect(args, port: int, state_dir: pathlib.Path) -> dict:
    """副作用已提交、回包未送达：客户端报错但账本已有记录，重试必须靠幂等键拿回原结果。

    与 ``disconnect_mid_call`` 的区别在注入时机：这里服务端把副作用写在前面、把长延迟放在
    后面，进程在延迟期间被杀，所以覆盖的正是「执行成功但响应丢失」这个窗口。
    """
    proc = _start_server(args, "v1", port, "disconnect-after-effect", ["--effect-delay-ms", "0"])
    before = _count_effects(state_dir)
    key = "disc-after-1"

    async def call():
        return await _one_shot(url_port(port), "21*2", {"delay_ms": 3000, "idem_key": key},
                               state_dir=state_dir, timeout=20)

    task = asyncio.create_task(call())
    await asyncio.sleep(0.9)
    proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass
    if proc in _SPAWNED:
        _SPAWNED.remove(proc)
    ok, payload, _ = await task
    await asyncio.sleep(0.3)
    after = _count_effects(state_dir)
    # 重启同端口，用同一业务键重试：应命中去重缓存，不产生第二条副作用
    _start_server(args, "v1", port, "disconnect-after-effect", ["--effect-delay-ms", "0"])
    retry_ok, retry_payload, _ = await _one_shot(url_port(port), "21*2", {"idem_key": key},
                                                 state_dir=state_dir, timeout=20)
    rp = retry_payload if isinstance(retry_payload, dict) else {}
    return {
        "case": "disconnect_after_effect_committed",
        "client_ok": ok,
        "client_result": payload if isinstance(payload, str) else json.dumps(payload)[:200],
        "effects_before": before,
        "effects_after": after,
        "effect_happened_despite_error": after > before,
        "retry_ok": retry_ok,
        "retry_dedup_hit": bool(rp.get("dedup_hit")),
        "effects_total": _count_effects(state_dir),
        "note": ("协议层只保证 attempt 级失败可见；副作用已经提交时，唯一能避免重复执行的是"
                 "服务端在同一提交边界里的业务幂等键"),
    }


async def _case_retry_no_key(args, url: str, state_dir: pathlib.Path) -> dict:
    """同一业务动作重试两次、不带幂等键：账本必须出现两条副作用。"""
    r1 = await _one_shot(url, "11*11", {"idem_key": None}, state_dir=state_dir, timeout=20)
    r2 = await _one_shot(url, "11*11", {"idem_key": None}, state_dir=state_dir, timeout=20)
    return {
        "case": "retry_without_idem_key",
        "first_ok": r1[0], "second_ok": r2[0],
        "effects": _count_effects(state_dir),
        "duplicate_effects": _count_effects(state_dir),
        "note": "协议层没有 exactly-once；没有业务幂等键时重试就是第二次执行",
    }


async def _case_retry_with_key(args, url: str, state_dir: pathlib.Path) -> dict:
    """同一业务键重试两次：第二次应命中去重缓存，账本只有一条副作用。"""
    key = "idem-alpha"
    r1 = await _one_shot(url, "12*12", {"idem_key": key}, state_dir=state_dir, timeout=20)
    r2 = await _one_shot(url, "12*12", {"idem_key": key}, state_dir=state_dir, timeout=20)
    p2 = r2[1] if isinstance(r2[1], dict) else {}
    return {
        "case": "retry_with_idem_key",
        "first_ok": r1[0], "second_ok": r2[0],
        "second_dedup_hit": bool(p2.get("dedup_hit")),
        "effects": _count_effects(state_dir),
        "note": "去重发生在服务端的同一条提交边界里；客户端先查后写不能替代它",
    }


async def _case_late_result(args, url: str, state_dir: pathlib.Path) -> dict:
    """客户端超时但服务端继续执行：晚到结果被丢弃，副作用只发生一次。"""
    key = "late-1"
    before = _count_effects(state_dir)
    ok, payload, _ = await _one_shot(url, "13*13", {"delay_ms": 3000, "idem_key": key},
                                     state_dir=state_dir, timeout=0.8)
    await asyncio.sleep(3.5)                      # 等晚到的执行真正完成
    mid = _count_effects(state_dir)
    r = await _one_shot(url, "13*13", {"delay_ms": 3000, "idem_key": key},
                        state_dir=state_dir, timeout=20)
    p = r[1] if isinstance(r[1], dict) else {}
    return {
        "case": "late_result_after_client_timeout",
        "first_ok": ok,
        "first_error": payload if isinstance(payload, str) else None,
        "effects_after_late_completion": mid - before,
        "retry_dedup_hit": bool(p.get("dedup_hit")),
        "effects_total": _count_effects(state_dir),
        "note": "客户端断开不停止执行；晚到回包被丢弃后，重试只能靠服务端去重拿回原结果",
    }


async def _case_schema_change(args, state_url_v2: str) -> dict:
    """客户端持有 v1 schema 去调 v2 服务：必填字段缺失被拒；重新 tools/list 后成功。"""
    stack = None
    out: dict = {"case": "schema_change"}
    try:
        stack, session, _ = await _session(state_url_v2, timeout=20)
        stale = await session.list_tools()
        out["v2_schema_required"] = _schema_of(stale.tools[0]).get("required")
        try:
            r = await session.call_tool("calculate", {"expression": "5*5"})
            out["stale_call_isError"] = bool(getattr(r, "isError", False))
            out["stale_call_text"] = (r.content[0].text if r.content else "")[:200]
        except Exception as exc:  # noqa: BLE001
            out["stale_call_isError"] = True
            out["stale_call_text"] = f"{type(exc).__name__}: {exc}"[:200]
        fresh = await session.list_tools()
        out["relisted_required"] = _schema_of(fresh.tools[0]).get("required")
        r2 = await session.call_tool("calculate", {"expression": "5*5", "mode": "plain"})
        out["retry_ok"] = not getattr(r2, "isError", False)
        out["retry_text"] = (r2.content[0].text if r2.content else "")[:200]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stack is not None:
            await stack.aclose()
    out["note"] = "schema 变更会让缓存的调用方在参数校验处失败；发现方式是重新 tools/list，而不是重试同一请求"
    return out


async def _case_expired_session(args, port: int, state_sub: str) -> dict:
    """会话失效后的客户端行为。

    先验证配置的空闲超时是否真的会让会话过期（本 SDK 下没有），再用「服务重启导致会话消失」
    这一必然发生的传输状态变化测客户端能否重建会话并继续。
    """
    out: dict = {"case": "expired_session", "idle_timeout_s": args.idle_timeout}
    url = url_port(port)
    proc = _start_server(args, "v1", port, state_sub, ["--idle-timeout", str(args.idle_timeout)])
    stack = None
    try:
        stack, session, _ = await _session(url, timeout=20)
        r1 = await session.call_tool("calculate", {"expression": "3*3"})
        out["before_ok"] = not getattr(r1, "isError", False)
        wait_s = max(args.idle_timeout * 3, 6.0)
        await asyncio.sleep(wait_s)
        try:
            r2 = await session.call_tool("calculate", {"expression": "3*4"})
            out["idle_survived_s"] = wait_s
            out["idle_timeout_effective"] = False
            out["idle_survived_text"] = (r2.content[0].text if r2.content else "")[:120]
        except Exception as exc:  # noqa: BLE001
            out["idle_survived_s"] = wait_s
            out["idle_timeout_effective"] = True
            out["idle_expiry_error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException:  # noqa: BLE001
                pass
    # 服务重启：会话在服务端已不存在，旧会话必须失败、新会话必须能继续工作
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()
    _SPAWNED.remove(proc) if proc in _SPAWNED else None
    _start_server(args, "v1", port, state_sub + "-restart", ["--idle-timeout", str(args.idle_timeout)])
    stack2 = None
    try:
        stack2, session2, _ = await _session(url, timeout=20)
        # 用服务端已不认识的会话 ID 直接发请求：模拟客户端持有旧会话继续调用
        old_session_id = getattr(session2, "session_id", None)
        old_session_id = old_session_id or getattr(session2, "_session_id", None)
        out["session_id_available_to_client"] = old_session_id is not None
        r3 = await session2.call_tool("calculate", {"expression": "3*5"})
        out["after_restart_ok"] = not getattr(r3, "isError", False)
    except Exception as exc:  # noqa: BLE001
        out["after_restart_ok"] = False
        out["after_restart_text"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if stack2 is not None:
            try:
                await stack2.aclose()
            except BaseException:  # noqa: BLE001
                pass
    out["restart_recovery_ok"] = bool(out.get("after_restart_ok"))
    out["note"] = (
        "配置的 session_idle_timeout 在本次观测里没有让会话过期（记录为负结果）；"
        "服务重启后会话确定消失，客户端必须重新 initialize 才能继续，这一步是重建会话的验收点"
    )
    return out


async def _case_auth(args, url: str, state_dir: pathlib.Path) -> dict:
    """HTTP 授权边界：缺 token 被拒（原始 HTTP 探测），带 token 通过且工具能看到 Authorization 头。"""
    import httpx2

    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_PIN, "capabilities": {},
                       "clientInfo": {"name": "probe", "version": "0"}}}
    headers = {"content-type": "application/json",
               "accept": "application/json, text/event-stream"}
    async with httpx2.AsyncClient(timeout=20.0) as cli:
        r_no = await cli.post(url, json=body, headers=headers)
        r_yes = await cli.post(url, json=body,
                               headers={**headers, "authorization": f"Bearer {args.token}"})
    with_token = await _one_shot(url, "2*3", {"idem_key": "auth-yes"}, token=args.token,
                                 state_dir=state_dir, timeout=20)
    ledger = []
    path = state_dir / "effects.jsonl"
    if path.exists():
        ledger = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {
        "case": "http_auth_context",
        "no_token_http_status": r_no.status_code,
        "with_token_http_status": r_yes.status_code,
        "with_token_call_ok": with_token[0],
        "tool_saw_auth_header": bool(ledger and ledger[-1].get("auth_present")),
        "tool_saw_masked_header": ledger[-1].get("auth_masked") if ledger else None,
        "effects": len(ledger),
        "note": "传输层鉴权与工具授权是两层：工具注解或合法 JSON 都不授予执行权限",
    }


async def _case_allowlist(args, url: str, state_dir: pathlib.Path) -> dict:
    """客户端 allowlist 在发出前拒绝未授权工具：账本不应新增。"""
    before = _count_effects(state_dir)
    ok, payload, _ = await _one_shot(url, "9*9", {"idem_key": "allow-1"}, state_dir=state_dir,
                                    timeout=20, allowlist=["search_corpus"])
    after = _count_effects(state_dir)
    return {
        "case": "client_allowlist",
        "ok": ok,
        "text": payload if isinstance(payload, str) else json.dumps(payload)[:200],
        "effects_before": before,
        "effects_after": after,
        "note": "允许表在客户端先拦一道，服务端仍要独立校验；两者都不能只靠提示词",
    }


async def _case_tasks_extension(args, url: str) -> dict:
    """Tasks 扩展：先协商 capability，再决定能否测轮询/结果取得/tasks-cancel。"""
    out: dict = {"case": "tasks_extension", "protocol_pin": PROTOCOL_PIN}
    stack = None
    try:
        stack, session, init = await _session(url, timeout=20)
        caps = init.capabilities
        tasks_cap = getattr(caps, "tasks", None)
        out["server_capabilities_tasks"] = None if tasks_cap is None else json.loads(
            tasks_cap.model_dump_json())
        out["tasks_capability_negotiated"] = tasks_cap is not None
        if tasks_cap is None:
            out["polling_testable"] = False
            out["result_retrieval_testable"] = False
            out["cancel_testable"] = False
            out["note"] = ("服务端在 initialize 里没有声明 tasks 能力，轮询/结果取得/tasks/cancel "
                           "无法在本机协商成功；这三项保持 UNVERIFIED")
        else:
            out["polling_testable"] = True
            try:
                from mcp import types
                resp = await session.send_request(
                    types.ClientRequest(types.CallToolRequest(
                        method="tools/call",
                        params=types.CallToolRequestParams(name="calculate",
                                                           arguments={"expression": "1+1"},
                                                           task=types.TaskMetadata(ttl=5000)))),
                    types.CallToolResult,
                )
                out["task_creation_response"] = json.loads(resp.model_dump_json())[:1200]
            except Exception as exc:  # noqa: BLE001
                out["task_creation_response"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stack is not None:
            await stack.aclose()
    return out


def url_port(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


def url_disconnect(args) -> str:
    return url_port(args.port_disconnect)


CASES = (
    "retry_without_idem_key",
    "retry_with_idem_key",
    "late_result_after_client_timeout",
    "schema_change",
    "expired_session",
    "http_auth_context",
    "client_allowlist",
    "tasks_extension",
    "disconnect_mid_call",
    "disconnect_after_effect_committed",
)


async def run_case(name: str, args) -> dict:
    """执行单个用例。每个用例在独立进程里跑：MCP 会话的取消作用域一旦被超时打断，
    会污染同一条事件循环上的后续会话，隔离进程比事后清理更可靠。"""
    state_root = pathlib.Path(args.state_dir)
    if name == "retry_without_idem_key":
        return await _case_retry_no_key(args, url_port(args.port_noidem), state_root / "noidem")
    if name == "retry_with_idem_key":
        return await _case_retry_with_key(args, url_port(args.port_main), state_root / "main")
    if name == "late_result_after_client_timeout":
        return await _case_late_result(args, url_port(args.port_main), state_root / "main")
    if name == "schema_change":
        return await _case_schema_change(args, url_port(args.port_v2))
    if name == "expired_session":
        return await _case_expired_session(args, args.port_short, "short")
    if name == "http_auth_context":
        return await _case_auth(args, url_port(args.port_auth), state_root / "auth")
    if name == "client_allowlist":
        return await _case_allowlist(args, url_port(args.port_main), state_root / "main")
    if name == "tasks_extension":
        return await _case_tasks_extension(args, url_port(args.port_main))
    if name == "disconnect_mid_call":
        return await _case_disconnect(args, url_port(args.port_disconnect), state_root / "disconnect")
    if name == "disconnect_after_effect_committed":
        return await _case_disconnect_after_effect(args, args.port_disconnect_after,
                                                   state_root / "disconnect-after-effect")
    raise SystemExit(f"unknown case {name}")


def cmd_case(args) -> int:
    try:
        result = asyncio.run(run_case(args.name, args))
    except BaseException as exc:  # noqa: BLE001
        result = {"case": args.name, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        _stop_all()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"case-{args.name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[case] {args.name}: {json.dumps(result, ensure_ascii=False)[:320]}", flush=True)
    return 0


def cmd_run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_root = pathlib.Path(args.state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    [
        _start_server(args, "v1", args.port_main, "main", []),
        _start_server(args, "v2", args.port_v2, "v2", []),
        _start_server(args, "v1", args.port_noidem, "noidem", ["--no-idem"]),
        _start_server(args, "v1", args.port_auth, "auth", ["--require-token", args.token]),
    ]
    cases: list[dict] = []
    try:
        for name in CASES:
            cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "case",
                   "--name", name, "--out", str(out), "--state-dir", str(state_root),
                   "--port-main", str(args.port_main), "--port-v2", str(args.port_v2),
                   "--port-noidem", str(args.port_noidem), "--port-auth", str(args.port_auth),
                   "--port-short", str(args.port_short),
                   "--port-disconnect", str(args.port_disconnect),
                   "--port-disconnect-after", str(args.port_disconnect_after),
                   "--token", args.token, "--idle-timeout", str(args.idle_timeout)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.case_timeout)
            path = out / f"case-{name}.json"
            if path.exists():
                cases.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                cases.append({"case": name,
                              "error": f"case process rc={proc.returncode}: "
                                       f"{(proc.stdout + proc.stderr)[-400:]}"})
    finally:
        _stop_all()

    report = {
        "protocol_pin": PROTOCOL_PIN,
        "cases": cases,
        "effects_total": {d.name: _count_effects(d) for d in sorted(state_root.iterdir()) if d.is_dir()},
        "unverified": [
            "Tasks 扩展的轮询/结果取得/tasks-cancel（服务端未声明 tasks 能力）",
            "跨进程取消对已验证副作用回滚（普通取消通知不保证回滚）",
        ],
    }
    (out / "mcp_faults.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("effects_total:", report["effects_total"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.2 工具调用失败路径与授权边界")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve-http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--schema", choices=["v1", "v2"], default="v1")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--no-idem", action="store_true", help="关闭幂等去重（对照重复执行）")
    p.add_argument("--require-token", default=None)
    p.add_argument("--effect-delay-ms", type=int, default=0)
    p.add_argument("--idle-timeout", type=float, default=1800.0)
    p.add_argument("--sse", action="store_true")
    p.set_defaults(func=cmd_serve_http)

    p = sub.add_parser("case")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--port-main", type=int, default=8081)
    p.add_argument("--port-v2", type=int, default=8082)
    p.add_argument("--port-noidem", type=int, default=8083)
    p.add_argument("--port-auth", type=int, default=8084)
    p.add_argument("--port-short", type=int, default=8085)
    p.add_argument("--port-disconnect", type=int, default=8086)
    p.add_argument("--port-disconnect-after", type=int, default=8087)
    p.add_argument("--token", default="l9-secret-token")
    p.add_argument("--idle-timeout", type=float, default=2.0)
    p.set_defaults(func=cmd_case)

    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--case-timeout", type=float, default=120.0)
    p.add_argument("--state-dir", default=None)
    p.add_argument("--port-main", type=int, default=8081)
    p.add_argument("--port-v2", type=int, default=8082)
    p.add_argument("--port-noidem", type=int, default=8083)
    p.add_argument("--port-auth", type=int, default=8084)
    p.add_argument("--port-short", type=int, default=8085)
    p.add_argument("--port-disconnect", type=int, default=8086)
    p.add_argument("--port-disconnect-after", type=int, default=8087)
    p.add_argument("--token", default="l9-secret-token")
    p.add_argument("--idle-timeout", type=float, default=2.0)
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if getattr(args, "state_dir", None) is None:
        args.state_dir = str(pathlib.Path(args.out) / "state")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
