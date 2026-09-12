#!/usr/bin/env python3
"""5.12 服务层：embedding 的 batch 扫描与变长输入。

和 5.11 一样用裸 socket 发请求，这样能看到引擎侧指标与客户端时延的差。
两个模式：
  fixed     固定长度，扫并发（batch 由调度器决定，不是客户端指定）
  variable  变长输入，长度取自对数正态分布，用来量 padding 与分桶的影响

    python embedding_serving_audit.py --base-url http://127.0.0.1:8100 \
        --model bge-small --out DIR --mode fixed --length 128 --concurrency 1 8 32 128
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

UNIT = ("The paged KV cache stores key and value blocks in a fixed-size pool "
        "so that sequences can share memory without copying. ")


def make_text(n_tokens, tok_len_hint=4.2):
    """按字符数近似凑长度；真实 token 数以服务端返回的 usage 为准。"""
    reps = max(1, int(n_tokens * tok_len_hint / len(UNIT)))
    return (UNIT * reps)[: int(n_tokens * tok_len_hint)]


def one_embed(host, port, model, text, timeout=180.0):
    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/embeddings", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    dt = (time.perf_counter() - t0) * 1000
    usage = body.get("usage") or {}
    dim = len(body["data"][0]["embedding"]) if body.get("data") else None
    return {"ms": dt, "prompt_tokens": usage.get("prompt_tokens"), "dim": dim}


def scrape(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=30) as r:
        text = r.read().decode()
    vals = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, v = line.rpartition(" ")
        try:
            vals[name.strip()] = float(v)
        except ValueError:
            continue
    return vals


def hist(vals, name):
    s = sum(v for k, v in vals.items() if k.startswith(name + "_sum"))
    c = sum(v for k, v in vals.items() if k.startswith(name + "_count"))
    return s, c


def run_batch(host, port, base_url, model, texts, tag, timeout=180.0):
    """同时发 len(texts) 条请求，返回客户端与引擎两侧的统计。"""
    res, lock = [], threading.Lock()

    def worker(i):
        try:
            r = one_embed(host, port, model, texts[i], timeout)
        except Exception as e:                     # noqa: BLE001
            r = {"ms": None, "error": str(e)[:200]}
        with lock:
            res.append(r)

    before = scrape(base_url)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(texts))]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    span = (time.perf_counter() - t0) * 1000
    after = scrape(base_url)

    ok = [r for r in res if r.get("ms")]
    lats = sorted(r["ms"] for r in ok)
    tokens = sum(r["prompt_tokens"] or 0 for r in ok)
    engine = {}
    for name, key in [("e2e", "vllm:e2e_request_latency_seconds"),
                      ("queue", "vllm:request_queue_time_seconds"),
                      ("inference", "vllm:request_inference_time_seconds"),
                      ("prefill", "vllm:request_prefill_time_seconds")]:
        s1, c1 = hist(before, key)
        s2, c2 = hist(after, key)
        engine[name] = round((s2 - s1) / (c2 - c1) * 1000, 3) if c2 > c1 else None
        engine[name + "_count"] = c2 - c1
    return {"tag": tag, "n_requests": len(texts), "n_ok": len(ok),
            "errors": [r["error"] for r in res if r.get("error")][:3],
            "span_ms": round(span, 3),
            "lat_mean_ms": round(statistics.mean(lats), 3) if lats else None,
            "lat_p50_ms": round(lats[len(lats) // 2], 3) if lats else None,
            "lat_p95_ms": round(lats[int(len(lats) * 0.95) - 1], 3) if lats else None,
            "lat_max_ms": round(lats[-1], 3) if lats else None,
            "prompt_tokens_total": tokens,
            "req_per_s": round(len(ok) / (span / 1000), 3) if span else None,
            "tokens_per_s": round(tokens / (span / 1000), 3) if span else None,
            "embedding_dim": ok[0]["dim"] if ok else None,
            "engine_per_request_ms": engine}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8100")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["fixed", "variable"], required=True)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="*", default=[1, 8, 32, 128])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u = urlparse(args.base_url)
    host, port = u.hostname, u.port or 80
    rng = random.Random(args.seed)

    result = {"base_url": args.base_url, "model": args.model, "mode": args.mode, "rows": []}
    for k in args.concurrency:
        for rep in range(args.repeat):
            if args.mode == "fixed":
                texts = [make_text(args.length) for _ in range(k)]
                tag = f"fixed-l{args.length}-c{k}-r{rep}"
            else:
                lengths = [max(8, int(rng.lognormvariate(4.6, 0.7))) for _ in range(k)]
                lengths = [min(512, x) for x in lengths]
                texts = [make_text(x) for x in lengths]
                tag = f"var-c{k}-r{rep}"
            row = run_batch(host, port, args.base_url, args.model, texts, tag,
                            args.timeout)
            row["requested_length"] = args.length if args.mode == "fixed" else None
            result["rows"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    (out / f"serving-{args.mode}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
