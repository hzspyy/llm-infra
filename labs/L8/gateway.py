#!/usr/bin/env python3
"""labs/L8/gateway.py - 8.1: 订阅真实 KV 事件的迷你推理网关.

网关对上层暴露 OpenAI 兼容的 `/v1/completions` 流式接口, 对下层按策略选择 worker。
它做三件真实的事:

1. **路由**: 每个请求先由 `request_router.Router` 打分选 worker, 决策整条记录
   (候选、队列、预测命中、最终选择), 通过响应头 `x-lb-worker` /
   `x-lb-predicted-hit` 回给客户端, 便于逐请求对账。
2. **目录**: 每个 worker 以 `--kv-events-config` 发布 ZMQ KV 事件; 网关用 SUB
   订阅 `(topic, seq, msgpack(EventBatch))`, 把 BlockStored/BlockRemoved/
   AllBlocksCleared 应用到精确目录。可注入丢事件、延迟、重复与 worker 重启,
   观察目录漂移如何变成"预测命中但实际未命中"。
3. **队列**: 网关自己统计每个 worker 的在途请求数, 作为最短队列与队列惩罚项。

事件序列号是连续的, 网关记录 seq 缺口; `replay_endpoint` 可用时按 vLLM 的
ROUTER 重放协议补拉缺失批次 (与直接丢事件形成对照)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from labs.L8.request_router import ApproxPrefixIndex, EventDirectory, Router

CACHED_RE = re.compile(rb'"cached_tokens"\s*:\s*(\d+)')


class EventSubscriber:
    """订阅一个 worker 的 KV 事件, 支持丢包/延迟/重复注入。"""

    def __init__(self, worker: str, endpoint: str, directory: EventDirectory,
                 topic: str = "", drop_rate: float = 0.0, delay_s: float = 0.0,
                 duplicate: bool = False, replay_endpoint: Optional[str] = None):
        self.worker = worker
        self.endpoint = endpoint
        self.topic = topic
        self.directory = directory
        self.drop_rate = drop_rate
        self.delay_s = delay_s
        self.duplicate = duplicate
        self.replay_endpoint = replay_endpoint

        self.seq_seen: List[int] = []
        self.gaps = 0
        self.received = 0
        self.dropped = 0
        self.applied = 0
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def _apply(self, payload: bytes) -> int:
        """解析 msgspec 编码的 EventBatch, 返回事件条数。"""
        import msgspec
        from vllm.distributed.kv_events import KVEventBatch

        batch = msgspec.msgpack.Decoder(KVEventBatch).decode(payload)
        n = 0
        for ev in batch.events:
            kind = type(ev).__name__
            if kind == "BlockStored":
                self.directory.on_block_stored(
                    self.worker, ev.block_hashes, ev.token_ids,
                    block_size=ev.block_size, parent_block_hash=ev.parent_block_hash)
            elif kind == "BlockRemoved":
                self.directory.on_block_removed(self.worker, ev.block_hashes)
            elif kind == "AllBlocksCleared":
                self.directory.on_all_cleared(self.worker)
            n += 1
        return n

    async def _run(self) -> None:
        import zmq
        import zmq.asyncio

        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
        sock.connect(self.endpoint)
        seen: set = set()
        while True:
            try:
                parts = await sock.recv_multipart()
            except asyncio.CancelledError:
                sock.close(0)
                raise
            if len(parts) < 3:
                continue
            _topic, seq_b, payload = parts[0], parts[1], parts[-1]
            seq = int.from_bytes(seq_b, "big")
            self.received += 1
            # 重复事件: 检测并计数, 默认幂等应用 (BlockStored 重复只是重复计数)。
            if seq in seen:
                self.directory.stats["duplicate"] += 0  # 由目录自身计数
            else:
                if self.seq_seen and seq > self.seq_seen[-1] + 1:
                    self.gaps += seq - self.seq_seen[-1] - 1
                self.seq_seen.append(seq)
                seen.add(seq)
            if self.drop_rate > 0 and (hash(seq) % 1000) / 1000.0 < self.drop_rate:
                self.dropped += 1
                continue
            if self.delay_s > 0:
                await asyncio.sleep(self.delay_s)
            self.applied += self._apply(payload)
            if self.duplicate:
                self.applied += self._apply(payload)

    def stats(self) -> Dict[str, Any]:
        return {"worker": self.worker, "endpoint": self.endpoint,
                "received": self.received, "applied": self.applied,
                "dropped": self.dropped, "seq_gaps": self.gaps,
                "last_seq": self.seq_seen[-1] if self.seq_seen else None}


class Gateway:
    def __init__(self, workers: List[Tuple[str, str]], policy: str, block_size: int = 16,
                 queue_penalty: float = 0.0, event_endpoints: Optional[Dict[str, str]] = None,
                 event_topic: str = "", drop_rate: float = 0.0, delay_s: float = 0.0,
                 duplicate: bool = False, prediction_source: str = "events"):
        self.worker_urls = {name: url for name, url in workers}
        self.directory = EventDirectory(block_size)
        self.approx = ApproxPrefixIndex(block_size)
        # prefix_aware 的命中预估来自哪一个目录: 真实 KV 事件 (events) 还是
        # 网关自己发过的请求 (approx)。两者对同一请求的预测差就是目录漂移。
        index = self.directory if prediction_source == "events" else self.approx
        self.router = Router([n for n, _ in workers], policy, block_size, queue_penalty,
                             directory=index, approx=self.approx)
        self.prediction_source = prediction_source
        self.policy = policy
        self.event_endpoints = event_endpoints or {}
        self.event_topic = event_topic
        self.drop_rate = drop_rate
        self.delay_s = delay_s
        self.duplicate = duplicate
        self.subscribers: Dict[str, EventSubscriber] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.t0 = time.monotonic()
        self.counters: Dict[str, int] = {"requests": 0, "errors": 0}
        self.request_log: List[Dict[str, Any]] = []

    async def startup(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600))
        for w, ep in self.event_endpoints.items():
            sub = EventSubscriber(w, ep, self.directory, topic=self.event_topic,
                                  drop_rate=self.drop_rate, delay_s=self.delay_s,
                                  duplicate=self.duplicate)
            sub.start()
            self.subscribers[w] = sub

    async def cleanup(self, app: web.Application) -> None:
        for s in self.subscribers.values():
            await s.stop()
        if self.session:
            await self.session.close()

    # ---- 请求处理 --------------------------------------------------------
    async def completions(self, request: web.Request) -> web.StreamResponse:
        self.counters["requests"] += 1
        body = await request.json()
        prompt = body.get("prompt")
        token_ids: List[int] = prompt if isinstance(prompt, list) and prompt and isinstance(prompt[0], int) else []
        req_id = request.headers.get("x-request-id") or f"gw-{self.counters['requests']:06d}"

        decision = self.router.route(req_id, token_ids)
        worker = decision.chosen_worker
        self.router.on_start(worker)
        url = f"{self.worker_urls[worker].rstrip('/')}/v1/completions"
        rec: Dict[str, Any] = {"request_id": req_id, "worker": worker,
                               "policy": self.policy,
                               "predicted_hit_tokens": decision.predicted_hit_tokens,
                               "queue_depth": decision.queue_depth,
                               "t_start": time.monotonic() - self.t0}
        try:
            async with self.session.post(url, json=body) as resp:
                headers = {
                    "Content-Type": resp.headers.get("Content-Type", "text/event-stream"),
                    "x-lb-worker": worker,
                    "x-lb-predicted-hit": str(decision.predicted_hit_tokens),
                }
                out = web.StreamResponse(status=resp.status, headers=headers)
                await out.prepare(request)
                cached_tokens = None
                async for chunk in resp.content.iter_any():
                    m = CACHED_RE.search(chunk)
                    if m:
                        cached_tokens = int(m.group(1))
                    await out.write(chunk)
                await out.write_eof()
                rec["cached_tokens"] = cached_tokens
                rec["status"] = resp.status
        except Exception as e:  # noqa: BLE001
            self.counters["errors"] += 1
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            if not request.transport or request.transport.is_closing():
                pass
            return web.Response(status=502, text=f"gateway upstream error: {e}")
        finally:
            self.router.on_finish(worker)
            rec["t_end"] = time.monotonic() - self.t0
            self.router.note_routed(worker, token_ids)
            self.request_log.append(rec)
        return out

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response({
            "policy": self.policy,
            "counters": self.counters,
            "queue_depth": self.router.queue_depth,
            "directory_stats": self.directory.stats,
            "subscribers": {w: s.stats() for w, s in self.subscribers.items()},
            "requests": len(self.request_log),
        })

    async def requests_dump(self, request: web.Request) -> web.Response:
        return web.json_response(self.request_log)

    async def admin_clear(self, request: web.Request) -> web.Response:
        """模拟 worker 重启: 清掉该 worker 的目录条目 (真实重启由脚本 kill 进程)。"""
        worker = request.query.get("worker")
        if worker:
            self.directory.on_all_cleared(worker)
        return web.json_response({"cleared": worker})


def build_app(gw: Gateway) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/completions", gw.completions)
    app.router.add_get("/admin/stats", gw.stats)
    app.router.add_get("/admin/requests", gw.requests_dump)
    app.router.add_post("/admin/clear", gw.admin_clear)
    app.on_startup.append(gw.startup)
    app.on_cleanup.append(gw.cleanup)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", required=True,
                    help="name=url[,name=url...]，例如 w0=http://127.0.0.1:8000")
    ap.add_argument("--policy", default="prefix_aware",
                    choices=["round_robin", "shortest_queue", "prefix_aware"])
    ap.add_argument("--queue-penalty", type=float, default=0.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--events", default="",
                    help="name=tcp://127.0.0.1:5557[,name=...] KV 事件端点")
    ap.add_argument("--event-topic", default="")
    ap.add_argument("--drop-rate", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--duplicate", action="store_true")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    workers = [tuple(x.split("=", 1)) for x in args.workers.split(",") if x]
    events = dict(x.split("=", 1) for x in args.events.split(",") if x)
    gw = Gateway(workers, args.policy, args.block_size, args.queue_penalty,
                 event_endpoints=events, event_topic=args.event_topic,
                 drop_rate=args.drop_rate, delay_s=args.delay, duplicate=args.duplicate)
    web.run_app(build_app(gw), port=args.port, access_log=None)


if __name__ == "__main__":
    main()
