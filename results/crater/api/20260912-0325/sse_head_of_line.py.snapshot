#!/usr/bin/env python3
"""把 tokenizer 放在事件循环里，会阻塞住所有在跑的请求——用 nanoserve 实测这条链。

对照两种写法，其余代码完全相同：
  inloop  —— 在 HTTP 协程里直接 `tok.encode(...)`。这正是 5.7 那个 server.py 的写法，
             也是 5.11 要展示的失败模式。
  thread  —— `await asyncio.to_thread(tok.encode, ...)`。vLLM 在
             v1/engine/async_llm.py 里对 raw prompt 走的是「不许阻塞事件循环」这条线。

负载：先起一条长输出的流式请求 A，等它稳定吐若干 token；再发一条请求 B。
B 的 prompt 长度由 --b-reps 控制：用超长 prompt 是实验组，用极短 prompt 是对照组。
tokenization 只发生在 B 的接收路径上，所以 A 的 token 间隔里多出来的那一段，
只能来自 B 的 tokenization（对照组扣掉「B 到达本身」的开销）。

    python sse_head_of_line.py --mode inloop --b-reps 700 --out DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nanoserve"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from engine import Engine, Request  # noqa: E402

UNIT = "The paged KV cache stores key and value blocks in a fixed-size pool. "


def trace(msg):
    """进度写到 stderr，stdout 只留最终 JSON。"""
    print(f"[hol] {msg}", file=sys.stderr, flush=True)


class Server:
    """nanoserve 的最小 HTTP+SSE 外壳，唯一变量是 tokenizer 在哪里跑。"""

    def __init__(self, engine, tokenizer, mode, idle_sleep=0.001):
        self.engine = engine
        self.tok = tokenizer
        self.mode = mode
        self.idle_sleep = idle_sleep
        self.reqs: dict[str, Request] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.sent: dict[str, int] = {}
        self.counter = 0
        self.log: list[dict] = []
        self.tok_ms: list[dict] = []
        self.engine_error: str | None = None

    def record(self, kind, **fields):
        self.log.append({"t": round(time.perf_counter(), 6), "kind": kind, **fields})

    async def engine_loop(self):
        try:
            while True:
                if not self.engine.waiting and not self.engine.running:
                    await asyncio.sleep(self.idle_sleep)
                    continue
                self.engine.step()
                for rid, req in list(self.reqs.items()):
                    q = self.queues.get(rid)
                    if q is None:
                        continue
                    while self.sent[rid] < len(req.output_ids):
                        q.put_nowait(req.output_ids[self.sent[rid]])
                        self.sent[rid] += 1
                    if req.state.value in ("finished", "aborted") and not q.hol_done:
                        q.hol_done = True
                        q.put_nowait(None)
                await asyncio.sleep(0)
        except Exception:
            import traceback
            self.engine_error = traceback.format_exc()
            print("ENGINE_LOOP_ERROR\n" + self.engine_error, flush=True)
            for q in list(self.queues.values()):
                if not q.hol_done:
                    q.hol_done = True
                    q.put_nowait(None)

    async def tokenize(self, text):
        t = time.perf_counter()
        if self.mode == "inloop":
            ids = self.tok.encode(text, add_special_tokens=False)
        else:
            ids = await asyncio.to_thread(self.tok.encode, text, add_special_tokens=False)
        self.tok_ms.append({"mode": self.mode, "ms": round((time.perf_counter() - t) * 1000, 3),
                            "tokens": len(ids), "chars": len(text)})
        return ids

    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            writer.close()
            return
        lines = head.decode("latin1").split("\r\n")
        _, path, _ = lines[0].split(" ")
        headers = dict(l.split(": ", 1) for l in lines[1:] if ": " in l)
        n = int(headers.get("Content-Length", 0))
        payload = json.loads((await reader.readexactly(n)) if n else b"{}")

        if path == "/tokenize":
            # 只走 tokenizer，不进引擎。用来把「事件循环被 tokenization 占住」
            # 与「引擎多了一条请求」这两件事分开。
            t0 = time.perf_counter()
            ids = await self.tokenize(payload["prompt"])
            body = json.dumps({"tokens": len(ids),
                               "server_ms": round((time.perf_counter() - t0) * 1000, 3)}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Connection: close\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
            writer.close()
            return

        t_accept = time.perf_counter()
        trace(f"handle: tokenizing {len(payload['prompt'])} chars")
        ids = await self.tokenize(payload["prompt"])          # ← 唯一的变量
        t_tokenized = time.perf_counter()
        trace(f"handle: tokenized {len(ids)} tokens in {(t_tokenized - t_accept) * 1000:.1f} ms")

        self.counter += 1
        rid = payload.get("id") or f"r{self.counter}"
        req = Request(rid, ids, max_tokens=int(payload.get("max_tokens", 32)))
        q: asyncio.Queue = asyncio.Queue()
        q.hol_done = False
        self.reqs[rid], self.queues[rid], self.sent[rid] = req, q, 0
        self.engine.add(req)
        self.record("accept", req=rid, prompt_tokens=len(ids),
                    tokenize_ms=round((t_tokenized - t_accept) * 1000, 3))

        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n")
        writer.write(f"data: {json.dumps({'accepted': rid})}\r\n\r\n".encode())
        await writer.drain()
        try:
            while True:
                tok = await q.get()
                if tok is None:
                    writer.write(b"data: [DONE]\r\n\r\n")
                    await writer.drain()
                    break
                writer.write(f"data: {json.dumps({'token': tok})}\r\n\r\n".encode())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self.engine.abort(rid, "client-disconnect")
        finally:
            self.reqs.pop(rid, None)
            self.queues.pop(rid, None)
            self.record("close", req=rid)
            try:
                writer.close()
            except Exception:
                pass

    async def serve(self, host, port):
        self._loop = asyncio.create_task(self.engine_loop())
        self.srv = await asyncio.start_server(self.handle, host, port)


async def drive(host, port, payload, marks):
    """客户端：把「accepted」和每个 token 的到达时刻追加进 marks（就地写，方便外部观察）。"""
    body = json.dumps(payload).encode()
    head = (f"POST /generate HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode()
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(head + body)
    await writer.drain()
    buf = ""
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            buf += data.decode()
            while "\r\n\r\n" in buf:
                part, _, buf = buf.partition("\r\n\r\n")
                for line in part.split("\r\n"):
                    if not line.startswith("data: "):
                        continue
                    now = time.perf_counter()
                    body_json = line[6:]
                    if body_json.startswith('{"accepted"'):
                        marks.append(("accepted", now))
                    elif body_json.startswith("[DONE]"):
                        marks.append(("done", now))
                    else:
                        marks.append(("token", now))
    finally:
        writer.close()
    return marks


def gaps(marks):
    toks = [m for m in marks if m[0] == "token"]
    return [(toks[i][1] - toks[i - 1][1]) * 1000 for i in range(1, len(toks))], toks


async def post_json(port, path, payload):
    body = json.dumps(payload).encode()
    head = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    t0 = time.perf_counter()
    writer.write(head + body)
    await writer.drain()
    raw = await reader.read(-1)
    writer.close()
    sep = raw.find(b"\r\n\r\n")
    return {"client_ms": round((time.perf_counter() - t0) * 1000, 3),
            "body": json.loads(raw[sep + 4:]) if sep >= 0 else None}


async def scenario(args, engine, tokenizer):
    server = Server(engine, tokenizer, args.mode)
    await server.serve("127.0.0.1", args.port)
    trace(f"listening on {args.port}, mode={args.mode}, probe={args.probe}, b_reps={args.b_reps}")
    a_marks: list = []
    a = asyncio.create_task(drive("127.0.0.1", args.port,
                                  {"id": "A", "prompt": UNIT * 4, "max_tokens": args.a_tokens},
                                  a_marks))
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.005)
        if a.done() or len([m for m in a_marks if m[0] == "token"]) >= args.warm_tokens:
            break
    trace(f"A warmed: {len([m for m in a_marks if m[0] == 'token'])} tokens, "
          f"a.done={a.done()} engine_error={bool(server.engine_error)}")

    long_prompt = UNIT * args.b_reps
    t_b_sent = time.perf_counter()
    b_marks: list = []
    probe = None
    if args.probe == "tokenize":
        probe = await post_json(args.port, "/tokenize", {"prompt": long_prompt})
    else:
        b = asyncio.create_task(drive("127.0.0.1", args.port,
                                      {"id": "B", "prompt": long_prompt, "max_tokens": 4},
                                      b_marks))
        probe = {"client_ms": None, "body": None, "b_task": b}
    try:
        await asyncio.wait_for(a, timeout=args.timeout)
    except asyncio.TimeoutError:
        trace(f"TIMEOUT on A; engine_error={server.engine_error}")
        a.cancel()
    if args.probe != "tokenize":
        b = probe.pop("b_task")
        try:
            await asyncio.wait_for(b, timeout=args.timeout)
        except asyncio.TimeoutError:
            trace("TIMEOUT on B")
            b.cancel()

    ga, toks = gaps(a_marks)
    stamps = [m[1] for m in toks[1:]]
    before = [g for g, s in zip(ga, stamps) if s <= t_b_sent]
    after = [g for g, s in zip(ga, stamps) if s > t_b_sent]
    b_tok = [r for r in server.tok_ms if r["chars"] == len(long_prompt)]
    b_accept = [m[1] for m in b_marks if m[0] == "accepted"]
    return {
        "mode": args.mode,
        "probe": args.probe,
        "b_reps": args.b_reps,
        "b_prompt_chars": len(long_prompt),
        "engine_error": server.engine_error,
        "tokenize_calls": server.tok_ms,
        "request_a": {
            "tokens": len(toks),
            "gap_before_B_ms": {"n": len(before), "max": round(max(before), 3) if before else None,
                                "median": round(statistics.median(before), 3) if before else None},
            "gap_after_B_ms": {"n": len(after), "max": round(max(after), 3) if after else None,
                               "median": round(statistics.median(after), 3) if after else None},
            "all_gaps_ms": [round(g, 3) for g in ga],
            "gap_stamps_ms": [round((s - t_b_sent) * 1000, 3) for s in stamps],
        },
        "probe_result": {
            "client_ms": probe.get("client_ms"),
            "server_ms": (probe.get("body") or {}).get("server_ms"),
            "tokens": (probe.get("body") or {}).get("tokens") or (b_tok[0]["tokens"] if b_tok else None),
            "tokenize_ms": b_tok[0]["ms"] if b_tok else None,
            "accept_latency_ms": round((b_accept[0] - t_b_sent) * 1000, 3) if b_accept else None,
        },
        "log": server.log,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["inloop", "thread"], required=True)
    p.add_argument("--probe", choices=["tokenize", "generate"], default="tokenize")
    p.add_argument("--out", required=True)
    p.add_argument("--b-reps", type=int, default=1000)
    p.add_argument("--a-tokens", type=int, default=64)
    p.add_argument("--warm-tokens", type=int, default=6)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=16)
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
    engine = Engine(model, block_size=args.block_size, max_batched_tokens=512,
                    max_num_seqs=8, eos_ids=eos)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(scenario(args, engine, tok))
    name = f"hol-{args.probe}-{args.mode}-b{args.b_reps}.json"
    (out / name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "log"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
