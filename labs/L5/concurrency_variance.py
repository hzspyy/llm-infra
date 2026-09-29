#!/usr/bin/env python3
"""L5.11 任务 C —— 并发 32 的波动：交错重复 + 客户端 CPU + 引擎排队三份记录。

正文此前记下：并发 32 的三次运行客户端 p50 分别是 131.4 / 105.3 / 105.6 ms，
差幅超 20%，**没有定位到原因**，只保留了方向性结论。计划对这条的要求是
「并发 32 的波动需交错重复、事件循环与连接重用记录解释」。

这个脚本把"交错重复"变成实验设计本身：**每一轮都跑 1/8/32 三档**，
跑 6 轮。这样能分清三件事：

  1. 档位效应（同一轮内 1 vs 8 vs 32）；
  2. 轮次效应/预热（第 0 轮是否系统性偏慢）；
  3. 客户端自身的 CPU 竞争——32 个 Python 线程做 SSE 解析，
     用 process_time 记账，看客户端 CPU 时间是否已经接近墙钟时间。
     若接近，那"波动"里有一部分根本不是服务端的。

连接重用另测：同一档用「每请求新建连接」与「复用一条连接」各跑一遍。
回环上建连成本约 0.97 ms（5.11 任务 C 已量），相对 ~100 ms 的时延约 1%，
所以它**不太可能**解释 25% 的差幅——这一条是用数据把可能性排除掉，而不是猜。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from api_layer_audit import one_request, scrape_metrics, hist      # noqa: E402


def cell(host, port, path, payload, base_url, k, tag):
    res, lock = [], threading.Lock()

    def worker(i):
        cpu0 = time.process_time()
        r = one_request(host, port, path, payload, label=f"{tag}-{i}")
        r["client_cpu_ms"] = (time.process_time() - cpu0) * 1000
        with lock:
            res.append(r)

    before = scrape_metrics(base_url)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    t0 = time.perf_counter()
    cpu_start = time.process_time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    span = (time.perf_counter() - t0) * 1000
    client_cpu = (time.process_time() - cpu_start) * 1000
    after = scrape_metrics(base_url)

    walls = sorted(r["wall_ms"] for r in res)
    per_layer = {}
    for name, key in [("e2e", "vllm:e2e_request_latency_seconds"),
                      ("queue", "vllm:request_queue_time_seconds"),
                      ("inference", "vllm:request_inference_time_seconds")]:
        s1, c1 = hist(before, key)
        s2, c2 = hist(after, key)
        per_layer[name] = round((s2 - s1) / (c2 - c1) * 1000, 3) if c2 > c1 else None
    return dict(level=k, tag=tag, span_ms=round(span, 3),
                wall_p50_ms=round(walls[len(walls) // 2], 3),
                wall_p95_ms=round(walls[int(len(walls) * 0.95) - 1], 3),
                wall_max_ms=round(walls[-1], 3),
                ttft_max_ms=round(max(r["ttft_ms"] for r in res if r["ttft_ms"]), 3),
                throughput_req_s=round(k / (span / 1000), 3),
                client_cpu_ms=round(client_cpu, 3),
                client_cpu_per_req_ms=round(client_cpu / k, 3),
                client_cpu_over_wall=round(client_cpu / span, 3),
                engine=per_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--prompt", default="Explain in detail how a paged KV cache works "
                                        "in an inference engine, step by step.")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from urllib.parse import urlparse
    u = urlparse(args.base_url)
    host, port = u.hostname, u.port or 80
    path = "/v1/completions"
    payload = {"model": args.model, "prompt": args.prompt,
               "max_tokens": args.max_tokens, "temperature": 0.0,
               "stream": True}

    rows = []
    for r in range(args.rounds):
        for k in (1, 8, 32):
            rec = cell(host, port, path, payload, args.base_url, k, f"r{r}c{k}")
            rec["round"] = r
            rows.append(rec)
            print(f"  round {r} conc {k:>2}  p50 {rec['wall_p50_ms']:>7.1f} ms  "
                  f"p95 {rec['wall_p95_ms']:>7.1f}  TTFTmax {rec['ttft_max_ms']:>6.1f}  "
                  f"queue {rec['engine']['queue']} ms  "
                  f"客户端 CPU/请求 {rec['client_cpu_per_req_ms']:>6.2f} ms "
                  f"(占墙钟 {rec['client_cpu_over_wall']:.2f})", flush=True)

    report = dict(rounds=args.rounds, model=args.model, max_tokens=args.max_tokens,
                  rows=rows)
    # 每档的轮间离散
    summary = {}
    for k in (1, 8, 32):
        xs = [r for r in rows if r["level"] == k]
        p50 = [r["wall_p50_ms"] for r in xs]
        summary[k] = dict(p50_min=min(p50), p50_max=max(p50),
                          p50_median=round(statistics.median(p50), 3),
                          spread_pct=round((max(p50) - min(p50)) / statistics.median(p50) * 100, 2),
                          first_round_p50=p50[0],
                          later_rounds_p50=p50[1:],
                          ttft_max=[r["ttft_max_ms"] for r in xs],
                          cpu_over_wall=[r["client_cpu_over_wall"] for r in xs])
    report["summary"] = summary
    (args.out / "concurrency_variance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n各档轮间离散：")
    for k, s in summary.items():
        print(f"  conc {k:>2}: p50 中位 {s['p50_median']} ms，区间 "
              f"[{s['p50_min']}, {s['p50_max']}]，差幅 {s['spread_pct']}%，"
              f"首轮 {s['first_round_p50']} ms，客户端 CPU/墙钟 {s['cpu_over_wall']}")
    print("\n判读：若 conc 32 的首轮明显偏慢、后续轮稳定，则波动是预热/轮次效应；")
    print("      若客户端 CPU/墙钟接近 1，则客户端线程本身是共同瓶颈。")


if __name__ == "__main__":
    main()
