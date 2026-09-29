#!/usr/bin/env python3
"""labs/L8/toy_server.py - 行为可控的 SSE 推理服务端 (只依赖标准库).

用于 8.3-A: 在已知真值的服务端上检验压测客户端的三个失效源——
协调遗漏 (coordinated omission)、客户端排队和事件缺失。

服务端模型:
    prefill_delay = prefill_base_s + prompt_len * prefill_per_token_s
    之后按 tpot_mean_s（可加抖动）逐 token 推送, 共 output_len 个 token。

可控项:
    max_concurrency   同时处理的请求数上限, 超出进入等待队列
    queue_capacity    等待队列长度, 超过直接返回 503
    stall_start_s     服务端从启动后第几秒进入全局卡顿
    stall_duration_s  卡顿持续时长; 卡顿期间所有在途请求一起停住
    drop_events       故意丢弃某类事件 ("first_token" / "done"), 检验缺失事件统计

接口:
    POST /generate  {"request_id": str, "prompt_len": int, "output_len": int}
                    -> SSE, 每行 `data: {"type": "token"|"done", "seq": int}`
    GET  /healthz   -> 200

服务端自己的时间线写入 `--server-log` 指定的 JSONL, 供与客户端记录逐请求对账。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional


@dataclasses.dataclass
class ServerConfig:
    max_concurrency: int = 2
    queue_capacity: int = 64
    prefill_base_s: float = 0.030
    prefill_per_token_s: float = 0.00005
    tpot_mean_s: float = 0.015
    tpot_jitter_s: float = 0.0
    stall_start_s: Optional[float] = None
    stall_duration_s: float = 0.0
    drop_events: tuple = ()


class _State:
    def __init__(self, cfg: ServerConfig, server_log: Optional[str]):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.free_slots = threading.Semaphore(cfg.max_concurrency)
        self.waiting = 0
        self.in_flight = 0
        self.t0 = time.monotonic()
        self.stalled = False
        self._log_f = open(server_log, "a", encoding="utf-8") if server_log else None
        if cfg.stall_start_s is not None and cfg.stall_duration_s > 0:
            threading.Thread(target=self._stall, daemon=True).start()

    def _stall(self) -> None:
        time.sleep(self.cfg.stall_start_s or 0.0)
        self.stalled = True
        time.sleep(self.cfg.stall_duration_s)
        self.stalled = False

    def log(self, rec: Dict) -> None:
        if self._log_f:
            with self.lock:
                self._log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._log_f.flush()


def _make_handler(state: _State):
    cfg = state.cfg

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ToyLLM/1.0"

        def log_message(self, *args):  # 静音默认访问日志
            pass

        def _send_json(self, code: int, payload: Dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _chunk(self, data: bytes) -> None:
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                self._send_json(200, {"ok": True, "in_flight": state.in_flight})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self._send_json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"})
                return

            req_id = str(req.get("request_id", "anon"))
            prompt_len = int(req.get("prompt_len", 128))
            output_len = int(req.get("output_len", 16))

            with state.lock:
                if state.waiting >= cfg.queue_capacity:
                    state.log({"request_id": req_id, "server_status": "REJECTED",
                               "rejected_at_s": time.monotonic() - state.t0})
                    self._send_json(503, {"error": "queue full"})
                    return
                state.waiting += 1

            recv_s = time.monotonic()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            queued_start = time.monotonic()
            state.free_slots.acquire()
            with state.lock:
                state.waiting -= 1
                state.in_flight += 1
            admit_s = time.monotonic()

            try:
                while state.stalled:
                    time.sleep(0.005)

                prefill = cfg.prefill_base_s + prompt_len * cfg.prefill_per_token_s
                time.sleep(prefill)

                first_token_s = time.monotonic()
                if "first_token" not in cfg.drop_events:
                    self._chunk(b'data: {"type": "token", "seq": 0}\n\n')
                n_sent = 0 if "first_token" in cfg.drop_events else 1

                for seq in range(1, output_len):
                    while state.stalled:
                        time.sleep(0.005)
                    delay = cfg.tpot_mean_s
                    if cfg.tpot_jitter_s > 0:
                        import random
                        delay += random.uniform(-cfg.tpot_jitter_s, cfg.tpot_jitter_s)
                    time.sleep(max(0.0, delay))
                    self._chunk(b'data: {"type": "token", "seq": %d}\n\n' % seq)
                    n_sent += 1

                finish_s = time.monotonic()
                if "done" not in cfg.drop_events:
                    self._chunk(b'data: {"type": "done"}\n\n')
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

                state.log({
                    "request_id": req_id,
                    "server_status": "SUCCESS",
                    "recv_s": recv_s - state.t0,
                    "admit_s": admit_s - state.t0,
                    "queue_wait_s": admit_s - queued_start,
                    "first_token_s": first_token_s - state.t0,
                    "finish_s": finish_s - state.t0,
                    "tokens_sent": n_sent,
                })
            except (BrokenPipeError, ConnectionResetError):
                state.log({"request_id": req_id, "server_status": "CLIENT_GONE",
                           "recv_s": recv_s - state.t0})
            finally:
                with state.lock:
                    state.in_flight -= 1
                state.free_slots.release()

    return Handler


class ToyServerHandle:
    def __init__(self, cfg: ServerConfig, host: str = "127.0.0.1", port: int = 0,
                 server_log: Optional[str] = None):
        self.cfg = cfg
        self.state = _State(cfg, server_log)
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self.state))
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[0], self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "ToyServerHandle":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description="可控 SSE toy 推理服务端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-concurrency", type=int, default=2)
    ap.add_argument("--queue-capacity", type=int, default=64)
    ap.add_argument("--prefill-base", type=float, default=0.030)
    ap.add_argument("--prefill-per-token", type=float, default=0.00005)
    ap.add_argument("--tpot", type=float, default=0.015)
    ap.add_argument("--stall-start", type=float, default=None)
    ap.add_argument("--stall-duration", type=float, default=0.0)
    ap.add_argument("--drop-events", default="")
    ap.add_argument("--server-log", default=None)
    args = ap.parse_args()

    cfg = ServerConfig(
        max_concurrency=args.max_concurrency,
        queue_capacity=args.queue_capacity,
        prefill_base_s=args.prefill_base,
        prefill_per_token_s=args.prefill_per_token,
        tpot_mean_s=args.tpot,
        stall_start_s=args.stall_start,
        stall_duration_s=args.stall_duration,
        drop_events=tuple(x for x in args.drop_events.split(",") if x),
    )
    h = ToyServerHandle(cfg, args.host, args.port, args.server_log).start()
    print(f"toy server on {h.base_url}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        h.stop()


if __name__ == "__main__":
    main()
