#!/usr/bin/env python3
"""L5.7 第六次迭代的客户端：起 server.py，打三种请求，把服务端事件收回来。

    python drive_server.py --out NEW_DIRECTORY [--port 8123]

三种请求：
  A 正常流式：逐条 SSE 收到底，记录 TTFT 与 token 间隔
  B 并发三条：看服务端把它们批到同一步里
  C 读到一半就关掉 socket：断连 -> abort -> 块回收，在服务端日志里对账
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = []


def emit(kind, **fields):
    row = {"kind": kind, **fields}
    LOG.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


async def sse(port, prompt, max_tokens, req_id, stop_after=None):
    """发一条请求，逐个事件读回来。stop_after 不为 None 时读够就关 socket。"""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "id": req_id}).encode()
    writer.write(b"POST /generate HTTP/1.1\r\nHost: localhost\r\n"
                 b"Content-Type: application/json\r\n"
                 + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    await writer.drain()
    t0 = time.perf_counter()
    await reader.readuntil(b"\r\n\r\n")            # 跳过响应头
    stamps, pieces, done = [], [], False
    while True:
        try:
            line = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            break
        text = line.decode().strip()
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        stamps.append(time.perf_counter() - t0)
        if payload == "[DONE]":
            done = True
            break
        pieces.append(json.loads(payload)["text"])
        if stop_after is not None and len(pieces) >= stop_after:
            writer.close()                          # 客户端主动跑路
            return {"id": req_id, "aborted_by_client": True,
                    "received_tokens": len(pieces), "text": "".join(pieces)}
    writer.close()
    gaps = [round((b - a) * 1000, 2) for a, b in zip(stamps, stamps[1:])]
    return {"id": req_id, "done": done, "received_tokens": len(pieces),
            "ttft_ms": round(stamps[0] * 1000, 2) if stamps else None,
            "inter_token_ms_first10": gaps[:10],
            "text": "".join(pieces)}


async def wait_health(port, timeout=180):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            n = int(dict(l.split(": ", 1) for l in head.decode().split("\r\n")
                         if ": " in l)["Content-Length"])
            body = json.loads(await reader.readexactly(n))
            writer.close()
            return body
        except Exception:
            await asyncio.sleep(0.5)
    raise TimeoutError("server 没起来")


async def main(port, out):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log_path = out / "server.log"
    with log_path.open("w") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "server.py"), "--port", str(port),
             "--blocks", "512", "--max-num-seqs", "8"],
            stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        try:
            health = await wait_health(port)
            emit("server_up", port=port, health=health)

            emit("A_stream", **await sse(port, "Count from one to ten.", 24, "A1"))

            t0 = time.perf_counter()
            rs = await asyncio.gather(*[
                sse(port, f"Write one sentence about the number {i}.", 24, f"B{i}")
                for i in range(3)])
            emit("B_concurrent", wall_s=round(time.perf_counter() - t0, 4), results=rs)

            emit("C_disconnect", **await sse(
                port, "Tell me a very long story about a robot.", 64, "C1",
                stop_after=5))
            await asyncio.sleep(1.0)
            emit("after_disconnect_health", health=await wait_health(port))
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    server_events = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                server_events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    emit("server_events", events=server_events)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8123)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    asyncio.run(main(args.port, args.out))
    (args.out / "client.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in LOG) + "\n")
