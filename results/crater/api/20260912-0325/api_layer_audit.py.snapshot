#!/usr/bin/env python3
"""5.11 HTTP / 协议层审计：一条请求从 socket 到 token 再回来的每一段。

回答三个问题：
  1. HTTP 层在端到端时延里占多少？——用引擎自己的指标做减法，不靠估算。
  2. SSE 的一个 chunk 到底长什么样？——抓原始字节，把 chunked 与 SSE 两层分帧都解出来。
  3. tokenization 会不会成为瓶颈？——量它的绝对耗时和并发下的串行化。

只用标准库。HTTP 自己用 socket 发，因为要拿到未经解析的字节。
    python api_layer_audit.py --base-url http://127.0.0.1:8000 --model qwen3-1.7b --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mini_sse import SSEDecoder  # noqa: E402

PROMPT = "Explain in detail how a paged KV cache works in an inference engine, step by step."


# ------------------------------------------------------------------ 原始 HTTP
def raw_stream(host, port, path, payload, timeout=120.0):
    """发一个请求，把收到的每一个字节按到达顺序记下来。返回 (头部, 原始字节块, 时间戳)。"""
    body = json.dumps(payload).encode()
    head = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Content-Type: application/json\r\nAccept: text/event-stream\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t_send = time.perf_counter()
    sock.sendall(head + body)
    chunks, stamps = [], []
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
            stamps.append(time.perf_counter() - t_send)
    finally:
        sock.close()
    raw = b"".join(chunks)
    sep = raw.find(b"\r\n\r\n")
    return raw[:sep].decode("latin1"), chunks, stamps, raw, t_send


def dechunk(raw_body: bytes) -> tuple[bytes, list[int]]:
    """解开 HTTP/1.1 chunked 分帧，返回 body 与每个 chunk 的字节长度。"""
    out, sizes, pos = bytearray(), [], 0
    while True:
        eol = raw_body.find(b"\r\n", pos)
        if eol < 0:
            break
        size_field = raw_body[pos:eol].split(b";")[0].strip()
        if not size_field:
            break
        try:
            size = int(size_field, 16)
        except ValueError:
            break
        pos = eol + 2
        if size == 0:
            break
        out += raw_body[pos:pos + size]
        sizes.append(size)
        pos += size + 2
    return bytes(out), sizes


# ------------------------------------------------------------------ 指标
def scrape_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(base_url + "/metrics", timeout=30) as resp:
        text = resp.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        try:
            values[name.strip()] = float(val)
        except ValueError:
            continue
    return values


def hist(values: dict[str, float], name: str) -> tuple[float, float]:
    """返回 (sum, count)。name 是不带 _sum/_count 后缀的指标名。"""
    total = sum(v for k, v in values.items() if k.startswith(name + "_sum"))
    count = sum(v for k, v in values.items() if k.startswith(name + "_count"))
    return total, count


# ------------------------------------------------------------------ 单个请求
def one_request(host, port, path, payload, label=""):
    """发一条流式请求，记录客户端看到的 TTFT / 结束时间，并保留收到的原始字节。"""
    body = json.dumps(payload).encode()
    head = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Content-Type: application/json\r\nAccept: text/event-stream\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    sock = socket.create_connection((host, port), timeout=180.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.perf_counter()
    sock.sendall(head + body)
    buf, header_done, ttft = bytearray(), False, None
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            buf += data
            if not header_done and buf.find(b"\r\n\r\n") >= 0:
                header_done = True
            if ttft is None and b"data:" in bytes(buf):
                ttft = time.perf_counter() - t0
    finally:
        sock.close()
    total = time.perf_counter() - t0
    raw = bytes(buf)
    sep = raw.find(b"\r\n\r\n")
    return {"label": label, "ttft_ms": None if ttft is None else ttft * 1000,
            "wall_ms": total * 1000, "bytes": len(raw),
            "keepalive_comments": raw.count(b": keep-alive"),
            "body": raw[sep + 4:] if sep >= 0 else b""}


# ------------------------------------------------------------------ 实验
def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(args.base_url)
    host, port = parsed.hostname, parsed.port or 80
    path = "/v1/completions"
    result = {"base_url": args.base_url, "model": args.model, "prompt": PROMPT}

    # ---- 1. 原始 SSE 字节流 ------------------------------------------------
    payload = {"model": args.model, "prompt": PROMPT, "max_tokens": 24,
               "temperature": 0.0, "stream": True, "ignore_eos": True}
    header, chunks, stamps, raw, _ = raw_stream(host, port, path, payload)
    raw_body = raw[raw.find(b"\r\n\r\n") + 4:]
    body, chunk_sizes = dechunk(raw_body)
    (out / "sse_raw.bin").write_bytes(raw)
    (out / "sse_body.bin").write_bytes(body)

    parser, events, gaps, prev = SSEDecoder(), [], [], 0.0
    for blob, stamp in zip(chunks, stamps):
        for ev in parser.feed(blob):
            events.append({"at_ms": round(stamp * 1000, 3), "data": ev["data"][:200]})
            gaps.append(stamp - prev)
            prev = stamp
    deltas = [e for e in events if e["data"] != "[DONE]"]
    result["raw_stream"] = {
        "response_header": header,
        "header_bytes": raw.find(b"\r\n\r\n") + 4,
        "total_bytes": len(raw),
        "chunked_frames": len(chunk_sizes),
        "chunk_size_min_max": [min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else None,
        "tcp_recv_calls": len(chunks),
        "body_bytes": len(body),
        "sse_events": len(events),
        "requested_max_tokens": payload["max_tokens"],
        "delta_events": len(deltas),
        "empty_delta_events": sum(1 for e in deltas if not e["data"].strip()),
        "delta_chars_min_max": [min(len(e["data"]) for e in deltas),
                                max(len(e["data"]) for e in deltas)] if deltas else None,
        "bytes_per_delta_event": round(len(body) / len(deltas), 2) if deltas else None,
        "first_event_at_ms": events[0]["at_ms"] if events else None,
        "last_event_at_ms": events[-1]["at_ms"] if events else None,
        "max_gap_ms": round(max(gaps) * 1000, 3) if gaps else None,
        "median_gap_ms": round(statistics.median(gaps) * 1000, 3) if gaps else None,
    }
    (out / "sse_events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))

    # ---- 2. 层间耗时分解 ---------------------------------------------------
    # 引擎指标里的 e2e 从「输入处理开始」算到「最后一个 token 在 engine core 里生成」。
    # 于是 e2e - inference - queue = 前端段（tokenization + 送进 core）；
    # 客户端墙钟 - e2e = 回程段（detokenize + SSE 序列化 + 回写）。
    n = args.requests
    for _ in range(2):                      # 预热，不计入
        one_request(host, port, path, {**payload, "stream": False})
    before = scrape_metrics(args.base_url)
    client = [one_request(host, port, path, payload, label=f"seq{i}") for i in range(n)]
    after = scrape_metrics(args.base_url)

    layers = {}
    for name, key in [("e2e", "vllm:e2e_request_latency_seconds"),
                      ("queue", "vllm:request_queue_time_seconds"),
                      ("inference", "vllm:request_inference_time_seconds"),
                      ("prefill", "vllm:request_prefill_time_seconds"),
                      ("decode", "vllm:request_decode_time_seconds")]:
        s1, c1 = hist(before, key)
        s2, c2 = hist(after, key)
        layers[name] = {"sum_delta_s": s2 - s1, "count_delta": c2 - c1,
                        "mean_ms": round((s2 - s1) / (c2 - c1) * 1000, 3) if c2 > c1 else None}
    walls = [c["wall_ms"] for c in client]
    ttfts = [c["ttft_ms"] for c in client if c["ttft_ms"]]
    e2e, inf, que = layers["e2e"]["mean_ms"], layers["inference"]["mean_ms"], layers["queue"]["mean_ms"]
    result["layers"] = {
        "per_layer": layers,
        "client": {"n": len(walls), "wall_mean_ms": round(statistics.mean(walls), 3),
                   "wall_min_ms": round(min(walls), 3), "wall_max_ms": round(max(walls), 3),
                   "ttft_mean_ms": round(statistics.mean(ttfts), 3) if ttfts else None},
        "derived": {
            "frontend_ms": None if None in (e2e, inf, que) else round(e2e - inf - que, 3),
            "edge_ms": None if e2e is None else round(statistics.mean(walls) - e2e, 3),
            "non_inference_share_pct": None if not e2e else
                round((statistics.mean(walls) - inf) / statistics.mean(walls) * 100, 2),
            "note": "frontend = e2e - inference - queue，进程内前端：tokenization + 序列化 + "
                    "IPC + core 准入。edge = 客户端墙钟 - e2e，进程外：入站解析与中间件 + "
                    "增量 detokenize + SSE 编码 + 回写。两者加 queue 与 inference 恰好等于客户端墙钟。",
        },
    }
    (out / "metrics_before.txt").write_text(json.dumps(before, indent=2, sort_keys=True))
    (out / "metrics_after.txt").write_text(json.dumps(after, indent=2, sort_keys=True))

    # ---- 3. tokenization 成本 ---------------------------------------------
    tok_curve = tokenize_curve(args)
    result["tokenize"] = tok_curve
    (out / "tokenize_curve.json").write_text(json.dumps(tok_curve, ensure_ascii=False, indent=2))

    # ---- 4. 并发下的客户端时延 --------------------------------------------
    result["concurrency"] = concurrency(host, port, path, payload, args, args.base_url)
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def tokenize_curve(args):
    """同一份 tokenizer，量不同长度 prompt 的编码耗时。纯 CPU，不占 GPU。"""
    from transformers import AutoTokenizer
    repo = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    unit = "The paged KV cache stores key and value blocks in a fixed-size pool. "
    out = {"repo": repo, "unit_chars": len(unit), "points": []}
    for reps in (1, 8, 64, 256, 1024):
        text = unit * reps
        n_ids = len(tok.encode(text, add_special_tokens=False))
        samples = []
        for _ in range(7):
            t = time.perf_counter()
            tok.encode(text, add_special_tokens=False)
            samples.append((time.perf_counter() - t) * 1000)
        out["points"].append({"chars": len(text), "tokens": n_ids,
                              "median_ms": round(statistics.median(samples), 4),
                              "min_ms": round(min(samples), 4)})
    # 并发编码：同一进程内顺序调用，看总时间是否等于各次之和
    texts = [unit * 64 for _ in range(16)]
    t = time.perf_counter()
    for x in texts:
        tok.encode(x, add_special_tokens=False)
    out["serial_16x256tokens_ms"] = round((time.perf_counter() - t) * 1000, 3)
    out["note"] = "单进程顺序编码；测的是「多次调用」与「一次长调用」的成本差，未测多线程并行"
    return out


def concurrency(host, port, path, payload, args, base_url):
    import threading
    rows = []
    out = Path(args.out)
    for k in args.concurrency:
        res, lock = [], threading.Lock()

        def worker(i):
            r = one_request(host, port, path, payload, label=f"c{k}-{i}")
            with lock:
                res.append(r)

        before = scrape_metrics(base_url)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
        t = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        span = (time.perf_counter() - t) * 1000
        after = scrape_metrics(base_url)
        walls = sorted(r["wall_ms"] for r in res)
        slowest = max(res, key=lambda r: len(r["body"]))
        (out / f"conc{k}-slowest.bin").write_bytes(slowest["body"])
        per_layer = {}
        for name, key in [("e2e", "vllm:e2e_request_latency_seconds"),
                          ("queue", "vllm:request_queue_time_seconds"),
                          ("inference", "vllm:request_inference_time_seconds"),
                          ("prefill", "vllm:request_prefill_time_seconds"),
                          ("decode", "vllm:request_decode_time_seconds")]:
            s1, c1 = hist(before, key)
            s2, c2 = hist(after, key)
            per_layer[name] = round((s2 - s1) / (c2 - c1) * 1000, 3) if c2 > c1 else None
        wall_mean = statistics.mean(r["wall_ms"] for r in res)
        rows.append({"concurrency": k, "span_ms": round(span, 3),
                     "wall_mean_ms": round(wall_mean, 3),
                     "engine_e2e_ms": per_layer["e2e"],
                     "edge_ms": round(wall_mean - per_layer["e2e"], 3) if per_layer["e2e"] else None,
                     "wall_p50_ms": round(walls[len(walls) // 2], 3),
                     "wall_p95_ms": round(walls[int(len(walls) * 0.95) - 1], 3),
                     "wall_max_ms": round(walls[-1], 3),
                     "ttft_max_ms": round(max(r["ttft_ms"] for r in res if r["ttft_ms"]), 3),
                     "throughput_req_s": round(k / (span / 1000), 3),
                     "bytes_total": sum(r["bytes"] for r in res),
                     "keepalive_comments_total": sum(r["keepalive_comments"] for r in res),
                     "keepalive_comments_per_request": [r["keepalive_comments"] for r in res],
                     "engine_per_request_ms": per_layer,
                     "slowest_saved_as": f"conc{k}-slowest.bin"})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--requests", type=int, default=12)
    p.add_argument("--concurrency", type=int, nargs="*", default=[1, 8, 32])
    run(p.parse_args())


if __name__ == "__main__":
    main()
