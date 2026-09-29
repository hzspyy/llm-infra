#!/usr/bin/env python3
"""L5.11 补测 · 分词（前端）到底占多少：同一串 token 的两种送法。

前端受限负载下最容易被怀疑的是分词与 detokenize。要把它与 prefill 分开，
最干净的办法是**送同一串 token 的两种形式**：

  A 文本：`{"prompt": "<N 个 token 对应的字符串>"}`  → 服务端要先分词
  B 预分词：`{"prompt": [id, id, …]}`                → 服务端跳过分词

两者进入引擎以后的路径完全相同（同样的 token 数、同样的 prefill、同样的采样），
TTFT 之差就是"分词 + 载荷编解码"的那一段。prompt 长度取 64/1024/4096，
并发 64，max_tokens=1（把设备侧压到最小，让前端占比显出来），每档 3 轮。

用法（先起好 vLLM，见 run_tokenizer_probe.sh）：
    python tokenizer_bottleneck.py --base http://127.0.0.1:8165 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-1.7B"


def one(base, model, payload, timeout=300):
    """发一条请求，返回 (status, 客户端观测的 TTFT ms, 总 ms, 字节)。"""
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/completions", data=raw,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
            status = r.status
    except urllib.error.HTTPError as e:
        return e.code, None, (time.perf_counter() - t0) * 1000, len(raw)
    total = (time.perf_counter() - t0) * 1000
    m = (body.get("choices") or [{}])[0]
    return status, total, total, len(raw)


def burst(base, model, payloads, concurrency):
    """并发发一批，返回各自的完成时间与状态。"""
    results = [None] * len(payloads)
    lock = threading.Lock()
    idx = [0]

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(payloads):
                return
            st, ttft, total, nbytes = one(base, model, payloads[i])
            results[i] = dict(status=st, total_ms=total, bytes=nbytes)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000
    ok = [r for r in results if r and r["status"] == 200]
    return dict(wall_ms=wall, n=len(payloads), n_ok=len(ok),
                p50_ms=(statistics.median(r["total_ms"] for r in ok) if ok else None),
                p99_ms=(sorted(r["total_ms"] for r in ok)[int(0.99 * (len(ok) - 1))]
                        if ok else None),
                bytes_median=(statistics.median(r["bytes"] for r in ok) if ok else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8165")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="64,1024,4096")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    word = "prefix caching "
    rows = []
    for n in (int(x) for x in args.lengths.split(",")):
        # 先造一串 token，再取它对应的文本；文本用 tokenizer 解码回来，
        # 保证两种送法的 token 数一致
        ids = tok.encode(word * (n // 2 + 4), add_special_tokens=False)[:n]
        while len(ids) < n:                      # 保证长度精确
            ids.append(ids[-1])
        text = tok.decode(ids)
        ids_roundtrip = tok.encode(text, add_special_tokens=False)
        for mode in ("text", "ids"):
            reps = []
            for _ in range(args.repeats):
                payloads = []
                for _ in range(args.concurrency):
                    p = ({"model": args.model, "prompt": text,
                          "max_tokens": args.max_tokens, "temperature": 0.0}
                         if mode == "text" else
                         {"model": args.model, "prompt": ids,
                          "max_tokens": args.max_tokens, "temperature": 0.0})
                    payloads.append(p)
                reps.append(burst(args.base, args.model, payloads, args.concurrency))
            row = dict(prompt_tokens=n, mode=mode,
                       text_tokens_roundtrip=len(ids_roundtrip),
                       p50_median_ms=statistics.median(r["p50_ms"] for r in reps
                                                       if r["p50_ms"]),
                       p99_median_ms=statistics.median(r["p99_ms"] for r in reps
                                                       if r["p99_ms"]),
                       wall_median_ms=statistics.median(r["wall_ms"] for r in reps),
                       bytes_median=statistics.median(r["bytes_median"] for r in reps
                                                      if r["bytes_median"]),
                       n_ok=reps[0]["n_ok"])
            rows.append(row)
            print(f"  n={n:<5} {mode:<5} p50 {row['p50_median_ms']:>8.1f} ms  "
                  f"p99 {row['p99_median_ms']:>8.1f} ms  整批 {row['wall_median_ms']:>8.1f} ms  "
                  f"请求体 {row['bytes_median']:>7.0f} B", flush=True)

    report = dict(model=args.model, concurrency=args.concurrency,
                  max_tokens=args.max_tokens, repeats=args.repeats, rows=rows)
    with open(os.path.join(args.out, "tokenizer_bottleneck.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n按长度看「文本 vs 预分词」的 p50 差（即服务端分词 + 载荷解析的成本）：")
    for n in (int(x) for x in args.lengths.split(",")):
        a = next(r for r in rows if r["prompt_tokens"] == n and r["mode"] == "text")
        b = next(r for r in rows if r["prompt_tokens"] == n and r["mode"] == "ids")
        print(f"  n={n:<5} 文本 {a['p50_median_ms']:.1f} ms − 预分词 "
              f"{b['p50_median_ms']:.1f} ms = {a['p50_median_ms'] - b['p50_median_ms']:+.1f} ms")
    print(f"\n写入 {args.out}/tokenizer_bottleneck.json")


if __name__ == "__main__":
    main()
