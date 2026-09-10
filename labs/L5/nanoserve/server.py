#!/usr/bin/env python3
"""nanoserve 的第六次迭代：HTTP + SSE 流式，外加一条从断连到释放块的通路。

用裸 asyncio 写 HTTP，不是为了少一个依赖，而是因为**断连必须看得见**：
框架会把「客户端走了」这件事包装成一个异常或回调，这里它就是
`writer.drain()` 抛出的 ConnectionResetError，和 reader 读到 EOF。
5.8 会沿着这条线继续往上接超时、背压和 OOM。

    python server.py --port 8000 --blocks 512
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from engine import Engine, Request, State


@dataclass
class Stream:
    req: Request
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    sent: int = 0


class Server:
    def __init__(self, engine: Engine, tokenizer, idle_sleep=0.001):
        self.engine = engine
        self.tok = tokenizer
        self.streams: dict[str, Stream] = {}
        self.idle_sleep = idle_sleep
        self.counter = 0
        self.events: list[dict] = []
        self._loop_task: asyncio.Task | None = None

    def log(self, kind, **fields):
        row = {"t": round(time.perf_counter(), 6), "kind": kind, **fields}
        self.events.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # ---------------------------------------------------------------- 引擎循环
    async def engine_loop(self):
        """一个协程推进引擎。step() 是同步阻塞的，所以它独占事件循环。

        这正是 5.8 要处理的问题之一：一次 step 有多长，HTTP 侧就有多久
        没人理。真实引擎把它放到单独的进程里（vLLM 的 EngineCore）。
        """
        while True:
            if not self.engine.waiting and not self.engine.running:
                await asyncio.sleep(self.idle_sleep)
                continue
            trace = self.engine.step()
            for rid in trace.prefill + trace.decode:
                st = self.streams.get(rid)
                if st is None:
                    continue
                while st.sent < len(st.req.output_ids):
                    st.queue.put_nowait(st.req.output_ids[st.sent])
                    st.sent += 1
            for rid in trace.finished:
                st = self.streams.get(rid)
                if st is not None:
                    while st.sent < len(st.req.output_ids):
                        st.queue.put_nowait(st.req.output_ids[st.sent])
                        st.sent += 1
                    st.queue.put_nowait(None)
            await asyncio.sleep(0)          # 让出，HTTP 协程才有机会跑

    # ---------------------------------------------------------------- HTTP
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            writer.close()
            return
        lines = head.decode("latin1").split("\r\n")
        method, path, _ = lines[0].split(" ")
        headers = dict(l.split(": ", 1) for l in lines[1:] if ": " in l)
        body = b""
        n = int(headers.get("Content-Length", 0))
        if n:
            body = await reader.readexactly(n)

        if path == "/health":
            await self._json(writer, {"status": "ok",
                                      "running": len(self.engine.running),
                                      "waiting": len(self.engine.waiting),
                                      "free_blocks": self.engine.pool.num_free})
            writer.close()
            return
        if method != "POST" or path != "/generate":
            await self._json(writer, {"error": "not found"}, status="404 Not Found")
            writer.close()
            return

        payload = json.loads(body or b"{}")
        self.counter += 1
        rid = payload.get("id") or f"h{self.counter}"
        ids = self.tok.encode(payload["prompt"], add_special_tokens=False)
        req = Request(rid, ids, max_tokens=int(payload.get("max_tokens", 32)))
        st = Stream(req)
        self.streams[rid] = st
        self.engine.add(req)
        self.log("accept", req=rid, prompt_tokens=len(ids),
                 max_tokens=req.max_tokens, peer=str(peer))

        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n")
        # 客户端半关或整个断开时，这个任务先醒过来。
        watcher = asyncio.create_task(reader.read(1))
        try:
            while True:
                get = asyncio.create_task(st.queue.get())
                done, _ = await asyncio.wait({get, watcher},
                                             return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    get.cancel()
                    raise ConnectionResetError("client closed")
                token = get.result()
                if token is None:
                    writer.write(b"data: [DONE]\r\n\r\n")
                    await writer.drain()
                    break
                chunk = json.dumps({"token": token, "text": self.tok.decode([token])},
                                   ensure_ascii=False)
                writer.write(f"data: {chunk}\r\n\r\n".encode())
                await writer.drain()        # 对端不收时，反压在这里出现
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            # 断连 -> abort -> 释放块。这条链就是 5.8 的主题。
            freed_before = self.engine.pool.num_free
            aborted = self.engine.abort(rid, "client-disconnect")
            self.log("disconnect", req=rid, aborted=aborted,
                     blocks_freed=self.engine.pool.num_free - freed_before,
                     sent_tokens=st.sent)
        finally:
            watcher.cancel()
            self.streams.pop(rid, None)
            self.log("close", req=rid, state=req.state.value,
                     finish_reason=req.finish_reason,
                     free_blocks=self.engine.pool.num_free)
            try:
                writer.close()
            except Exception:
                pass

    async def _json(self, writer, obj, status="200 OK"):
        body = json.dumps(obj, ensure_ascii=False).encode()
        writer.write(f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
                     .encode() + body)
        await writer.drain()

    async def serve(self, host, port):
        self._loop_task = asyncio.create_task(self.engine_loop())
        srv = await asyncio.start_server(self.handle, host, port)
        self.log("listen", host=host, port=port,
                 blocks=len(self.engine.pool.blocks),
                 block_size=self.engine.block_size)
        async with srv:
            await srv.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--blocks", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-batched-tokens", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=8)
    args = p.parse_args()
    from transformers import AutoTokenizer, GenerationConfig
    from model import PagedModel
    repo = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
    hub = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    gen = GenerationConfig.from_pretrained(repo, local_files_only=True)
    eos = gen.eos_token_id
    eos = (eos,) if isinstance(eos, int) else tuple(eos)
    model = PagedModel(repo, hub, num_blocks=args.blocks, block_size=args.block_size)
    engine = Engine(model, block_size=args.block_size,
                    max_batched_tokens=args.max_batched_tokens,
                    max_num_seqs=args.max_num_seqs, eos_ids=eos)
    asyncio.run(Server(engine, tok).serve(args.host, args.port))


if __name__ == "__main__":
    main()
