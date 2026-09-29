#!/usr/bin/env python3
"""L5.11 任务 C —— 本地反向代理下的流式行为：缓冲、keep-alive、慢读者、断连。

计划要求「在独立本地代理场景测试 buffer on/off、keep-alive、慢读者与断连；
HTTP/1.1 为基线，支持时补 HTTP/2」「保存原始流和取消传播」。

这里用三个纯 stdlib 组件搭一条最小链路，全部在本机跑（不占 GPU）：

    client ──▶ proxy ──▶ backend(SSE)

  backend  每 50 ms 写一个 SSE 事件，记录每次写入的时刻与是否遇到对端关闭
  proxy    两种模式：stream（收到一段就转发）/ buffer（收完整段再一次性发）
           两种连接：keep-alive（复用后端连接）/ close（每次新建）
  client   原始 socket，记录每个 chunk 的到达时刻；可切换"慢读者"与"中途断连"

四组对照各存原始流（字节 + 到达时刻），并回答四个具体问题：
  1. 代理缓冲会不会把"逐 token 到达"变成"一次性到达"？
  2. keep-alive 关掉后，每请求多了多少连接建立成本？
  3. 慢读者会不会把压力回传到后端（后端发送阻塞）？
  4. 客户端断连后，后端在多久内观察到关闭（取消传播）？

用法：python labs/L5/proxy_stream_audit.py --out results/local/5.11/proxy-stream-<id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EVENTS = 20
EVENT_INTERVAL = 0.05          # 后端每 50 ms 一个事件 → 约 1 s 的总时长
PAYLOAD_BYTES = 16             # 每个事件的数据量；调大才能让慢读者产生回压
BACKEND_PORT = 18111
PROXY_PORT = 18112


class BackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log = None

    def do_GET(self):                                   # noqa: N802
        t0 = time.perf_counter()
        self.log["backend"]["start_abs"] = t0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.log["backend"]["connect"] = t0
        closed_at = None
        try:
            for i in range(EVENTS):
                time.sleep(EVENT_INTERVAL)
                payload = (f"data: {json.dumps({'i': i})}".encode()
                           + b"x" * max(0, PAYLOAD_BYTES - 16) + b"\n\n")
                chunk = b"%x\r\n%s\r\n" % (len(payload), payload)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    closed_at = time.perf_counter()
                    raise
                self.log["backend"]["sends"].append(round(time.perf_counter() - t0, 4))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            # 读端靠 chunked 终止块判断流结束，不靠关连接；
            # 是否真的关连接由请求头决定，这样复用连接的对照才成立。
            self.close_connection = "close" in self.headers.get(
                "Connection", "").lower()
        except (BrokenPipeError, ConnectionResetError):
            closed_at = closed_at or time.perf_counter()
        finally:
            self.log["backend"]["closed_observed_at"] = (
                round(closed_at - t0, 4) if closed_at else None)
            self.log["backend"]["done_at"] = round(time.perf_counter() - t0, 4)

    def log_message(self, *a):
        pass


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log = None
    mode = "stream"
    keepalive = True

    def do_GET(self):                                   # noqa: N802
        t0 = time.perf_counter()
        conn = socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=30)
        if self.keepalive:
            conn.sendall(b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         b"Connection: keep-alive\r\n\r\n")
        else:
            conn.sendall(b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         b"Connection: close\r\n\r\n")
        self.log["proxy"].setdefault("connections", []).append(round(t0, 4))

        header = b""
        while b"\r\n\r\n" not in header:
            header += conn.recv(4096)
        head, rest = header.split(b"\r\n\r\n", 1)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def read_all(buf: bytes) -> bytes:
            """读到 chunked 终止块 0\r\n\r\n 为止，不依赖对端关连接。"""
            while b"0\r\n\r\n" not in buf:
                data = conn.recv(65536)
                if not data:
                    break
                buf += data
            return buf

        if self.mode == "buffer":
            # 攒够整段再发：客户端会看到"首包 = 全部"
            body = read_all(rest)
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.log["proxy"]["client_gone_at"] = round(time.perf_counter() - t0, 4)
            self.log["proxy"]["first_forward_at"] = round(time.perf_counter() - t0, 4)
            self.log["proxy"]["buffered_bytes"] = len(body)
        else:
            self.log["proxy"]["first_forward_at"] = round(time.perf_counter() - t0, 4)
            try:
                if rest:
                    self.wfile.write(rest)
                    self.wfile.flush()
                buf = rest
                while b"0\r\n\r\n" not in buf:
                    data = conn.recv(65536)
                    if not data:
                        break
                    buf += data
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # 客户端走了：把关闭传到后端（取消传播）
                self.log["proxy"]["client_gone_at"] = round(time.perf_counter() - t0, 4)
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        conn.close()

    def log_message(self, *a):
        pass


def start(log, mode, keepalive):
    BackendHandler.log = log
    ProxyHandler.log = log
    ProxyHandler.mode = mode
    ProxyHandler.keepalive = keepalive
    last = None
    for _ in range(20):
        try:
            b = ThreadingHTTPServer(("127.0.0.1", BACKEND_PORT), BackendHandler)
            p = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
            break
        except OSError as e:            # 端口还没从上一轮释放
            last = e
            time.sleep(0.3)
    else:
        raise last
    for s in (b, p):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return b, p


def client_run(slow_reader=False, disconnect_after=None, read_gap=0.3, client_log=None):
    """返回每个 chunk 的到达时刻与原始字节。"""
    s = socket.create_connection(("127.0.0.1", PROXY_PORT), timeout=30)
    s.sendall(b"GET /stream HTTP/1.1\r\nHost: 127.0.0.1\r\n"
              b"Accept: text/event-stream\r\n\r\n")
    t0 = time.perf_counter()
    chunks, raw = [], bytearray()
    s.settimeout(5)
    while True:
        try:
            data = s.recv(65536)
        except socket.timeout:
            break
        if not data:
            break
        chunks.append(round(time.perf_counter() - t0, 4))
        raw += data
        if disconnect_after is not None and len(chunks) >= disconnect_after:
            s.close()
            client_log["disconnect_abs"] = time.perf_counter()
            return chunks, bytes(raw), True
        if slow_reader:
            time.sleep(read_gap)
    s.close()
    return chunks, bytes(raw), False


def one_case(name, mode, keepalive, slow=False, disconnect=None, gap=0.3,
             payload=None):
    global PAYLOAD_BYTES
    if payload is not None:
        PAYLOAD_BYTES = payload
    log = {"backend": {"sends": []}, "proxy": {}}
    b, p = start(log, mode, keepalive)
    try:
        t0 = time.perf_counter()
        chunks, raw, closed = client_run(slow, disconnect, gap,
                                         log.setdefault("client", {}))
        wall = round(time.perf_counter() - t0, 4)
    finally:
        # shutdown() 只停循环，不释放端口；不 server_close() 会撞 TIME_WAIT
        b.shutdown(); p.shutdown()
        b.server_close(); p.server_close()
        time.sleep(0.2)
    n_events = raw.count(b"data: ")
    rec = dict(case=name, proxy_mode=mode, keepalive=keepalive,
               slow_reader=slow, client_disconnect=closed, wall_s=wall,
               client_chunks=len(chunks), events_received=n_events,
               first_chunk_at=chunks[0] if chunks else None,
               last_chunk_at=chunks[-1] if chunks else None,
               chunk_gaps=[round(b - a, 4) for a, b in zip(chunks, chunks[1:])],
               backend=log["backend"], proxy=log["proxy"],
               raw_bytes=len(raw))
    if chunks:
        rec["first_byte_latency_s"] = chunks[0]
        rec["burst"] = len(chunks) <= 2
    # 取消传播：后端观察到对端关闭的绝对时刻 − 客户端真正关闭的绝对时刻
    cl = log.get("client", {})
    be = log["backend"]
    if cl.get("disconnect_abs") and be.get("closed_observed_at_abs") is None \
            and be.get("closed_observed_at") is not None and be.get("start_abs"):
        be["closed_observed_at_abs"] = be["start_abs"] + be["closed_observed_at"]
    if cl.get("disconnect_abs") and be.get("closed_observed_at_abs"):
        rec["cancel_propagation_s"] = round(
            be["closed_observed_at_abs"] - cl["disconnect_abs"], 4)
    if be.get("start_abs") and be.get("sends"):
        rec["backend_last_send_s"] = be["sends"][-1]
    return rec


def connection_setup_cost(n=8):
    """keep-alive 能省下的上限：新建连接 + 首个请求 vs 复用连接再发一次。

    代理那一组只改了请求头里的 Connection，代理本身没有连接池，
    所以它测不出 keep-alive 的收益。这里直接量"建连"这一段的成本，
    给出 keep-alive 在每请求上最多能省多少。
    """
    def roundtrip(conn, first):
        req = (b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
               + (b"Connection: close\r\n" if first else b"Connection: keep-alive\r\n")
               + b"\r\n")
        t0 = time.perf_counter()
        conn.sendall(req)
        buf = b""
        while b"0\r\n\r\n" not in buf:
            d = conn.recv(65536)
            if not d:
                break
            buf += d
        return time.perf_counter() - t0

    fresh, reused = [], []
    for _ in range(n):
        c = socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=10)
        fresh.append(roundtrip(c, True))
        c.close()
    c = socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=10)
    roundtrip(c, False)                       # 先把连接预热
    for _ in range(n):
        reused.append(roundtrip(c, False))
    c.close()
    import statistics as st
    return dict(n=n, fresh_median_ms=round(st.median(fresh) * 1000, 3),
                reused_median_ms=round(st.median(reused) * 1000, 3),
                connect_cost_ms=round((st.median(fresh) - st.median(reused)) * 1000, 3))


def http2_support():
    try:
        import h2  # noqa: F401
        return dict(available=True, how="h2 已安装，可用 hypercorn/httpx 起 HTTP/2 前端")
    except Exception as e:                                   # pragma: no cover
        return dict(available=False, reason=f"h2 未安装：{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = [
        one_case("baseline_stream", "stream", True),
        one_case("proxy_buffer", "buffer", True),
        one_case("keepalive_off", "stream", False),
        one_case("slow_reader_small", "stream", True, slow=True, gap=0.35),
        one_case("slow_reader_large", "stream", True, slow=True, gap=0.35,
                 payload=65536),
        one_case("client_disconnect", "stream", True, disconnect=3),
    ]
    # 建连成本要在后端还活着的时候量，所以单独起一对服务
    log2 = {"backend": {"sends": []}, "proxy": {}}
    b2, p2 = start(log2, "stream", True)
    try:
        conn_cost = connection_setup_cost()
    finally:
        b2.shutdown(); p2.shutdown(); b2.server_close(); p2.server_close()
        time.sleep(0.2)
    out = dict(events_per_run=EVENTS, event_interval_s=EVENT_INTERVAL,
               http2=http2_support(), connection_reuse=conn_cost, cases=cases)
    (args.out / "proxy_stream.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'用例':<20}{'首包 s':>8}{'块数':>6}{'事件':>6}{'墙钟 s':>8}{'后端送完 s':>11}"
          f"{'后端观测关闭 s':>15}{'代理转发首包 s':>16}")
    for c in cases:
        print(f"{c['case']:<20}{str(c.get('first_chunk_at')):>8}{c['client_chunks']:>6}"
              f"{c['events_received']:>6}{c['wall_s']:>8}"
              f"{str(c['backend'].get('done_at')):>11}"
              f"{str(c['backend'].get('closed_observed_at')):>15}"
              f"{str(c['proxy'].get('first_forward_at')):>16}")
    print(f"\n建连成本：新建 {conn_cost['fresh_median_ms']} ms / 复用 "
          f"{conn_cost['reused_median_ms']} ms → keep-alive 上限 "
          f"{conn_cost['connect_cost_ms']} ms 每请求")
    print(f"HTTP/2：{out['http2']}")
    print("\n读法：buffer 模式首包应该接近整段时长、块数塌到 1–2；")
    print("      断连用例要看后端 closed_observed_at —— 那才是取消有没有传下去。")


if __name__ == "__main__":
    main()
