#!/usr/bin/env python3
"""L5.12 任务 C —— 客户端与服务端计时对齐（同请求集、两个时钟、只比时长）。

计划要求「两引擎服务采客户端与服务端事件」且「逐请求分解闭合，禁止以不同样本
中位数相减或以接收耗时冒充网络开销」。既有分段实验把服务器时间当**残差**推断，
这一步改成读服务端自己的直方图：

  1. 客户端：对 N 条相同请求逐条量墙钟，并把序列化与反序列化单独计时；
  2. 服务端：读 `/metrics` 中 `vllm:e2e_request_latency_seconds`、
     `request_inference_time_seconds`、`request_queue_time_seconds`、
     `request_prefill_time_seconds` 四个直方图在**同一批请求**前后的 `_sum/_count` 差；
  3. 对齐：只比同一批 N 条请求的平均时长（直方图给的是 sum/count，即平均），
     不做跨时钟时间戳相减，也不用不同样本的中位数相减；
  4. 残差 = 客户端平均墙钟 − 服务端平均 e2e，包含连接、发送、接收、客户端
     序列化/反序列化与调度抖动，逐项列出已知部分，剩下的明确标为未分解。

用法（先起好服务）：
    python labs/L5/pooling_client_server_align.py --base http://127.0.0.1:8125 \
        --out "$OUT/align" --repeats 40
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.request

TEXT = ("The prefix cache mechanism enables multiple requests to reuse identical "
        "prompt prefixes efficiently.")

METRICS = {
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "success": "vllm:request_success_total",
}


def parse_metrics(text):
    """按行解析 Prometheus 文本：只看以 `<指标名>_sum`/`_count` 开头的行。

    不用正则：直方图的标签组合在不同版本里会变，逐行取最后一个字段最稳。
    同时把未匹配到的指标名记下来，避免把「没有这个指标」和「指标是 0」混为一谈。
    """
    out, seen = {}, {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        head, val = parts[0].strip(), parts[1].strip()
        name = head.split("{", 1)[0]
        try:
            value = float(val)
        except ValueError:
            continue                      # +Inf 等
        seen.setdefault(name, 0.0)
        # 同一指标可能有多个标签组合（例如按 model_name 分），累加
        seen[name] = seen[name] + value
    for key, metric in METRICS.items():
        for suffix in ("_sum", "_count"):
            full = metric + suffix
            out[f"{key}{suffix}"] = seen.get(full)
    return out, seen


def fetch_metrics(base, dest=None):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    if dest is not None:
        dest.write_text(text, encoding="utf-8")
    parsed, seen = parse_metrics(text)
    return parsed, text


def post(base, path, payload):
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=raw,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8125")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--model", default="pooler")
    args = ap.parse_args()
    # 运行脚本已经建好 run 目录并往里写 gpu/进程快照，这里只补写 align.json
    args.out.mkdir(parents=True, exist_ok=True)

    before, _ = fetch_metrics(args.base, args.out / "metrics_before.txt")
    rows = []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        payload = {"model": args.model, "input": TEXT}
        t1 = time.perf_counter()
        status, body = post(args.base, "/pooling", payload)
        t2 = time.perf_counter()
        vec = json.loads(body)
        t3 = time.perf_counter()
        rows.append(dict(i=i, status=status,
                         serialize_ms=(t1 - t0) * 1000,
                         wall_ms=(t2 - t1) * 1000,
                         parse_ms=(t3 - t2) * 1000,
                         dim=len(vec["data"][0]["data"]) if "data" in vec else None))
    after, _ = fetch_metrics(args.base, args.out / "metrics_after.txt")

    delta = {}
    for key in METRICS:
        s0, s1 = before.get(f"{key}_sum"), after.get(f"{key}_sum")
        c0, c1 = before.get(f"{key}_count"), after.get(f"{key}_count")
        delta[key] = dict(sum=(None if s0 is None or s1 is None else s1 - s0),
                          count=(None if c0 is None or c1 is None else c1 - c0),
                          mean_ms=(None if s0 is None or s1 is None or c0 is None or c1 is None
                                   or c1 == c0 else (s1 - s0) / (c1 - c0) * 1000))

    client_wall = [r["wall_ms"] for r in rows]
    client_ser = [r["serialize_ms"] for r in rows]
    client_par = [r["parse_ms"] for r in rows]
    summary = dict(
        requests=len(rows),
        client=dict(wall_mean_ms=statistics.mean(client_wall),
                    wall_median_ms=statistics.median(client_wall),
                    wall_min_ms=min(client_wall), wall_max_ms=max(client_wall),
                    serialize_mean_ms=statistics.mean(client_ser),
                    parse_mean_ms=statistics.mean(client_par)),
        server=delta,
    )
    srv_e2e = delta["e2e"]["mean_ms"]
    if srv_e2e:
        summary["closure"] = dict(
            client_wall_mean_ms=summary["client"]["wall_mean_ms"],
            server_e2e_mean_ms=srv_e2e,
            client_minus_server_ms=summary["client"]["wall_mean_ms"] - srv_e2e,
            known_client_side_ms=summary["client"]["serialize_mean_ms"]
            + summary["client"]["parse_mean_ms"],
            server_inference_mean_ms=delta["inference"]["mean_ms"],
            server_queue_mean_ms=delta["queue"]["mean_ms"],
            server_prefill_mean_ms=delta["prefill"]["mean_ms"],
            server_count_matches_requests=delta["e2e"]["count"] == len(rows),
            note="服务端直方图给的是同一批请求的 sum/count（平均），"
                 "与客户端平均墙钟可比；两者之差含网络往返与调度抖动，未逐项分解。")
    (args.out / "align.json").write_text(
        json.dumps(dict(summary=summary, rows=rows, metrics_after=after),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"请求数 {len(rows)}（服务端计数 {delta['e2e']['count']}）")
    print(f"客户端平均墙钟 {summary['client']['wall_mean_ms']:.3f} ms"
          f"（中位 {summary['client']['wall_median_ms']:.3f}，"
          f"范围 {summary['client']['wall_min_ms']:.3f}–{summary['client']['wall_max_ms']:.3f}）")
    print(f"  其中序列化 {summary['client']['serialize_mean_ms']:.4f} ms，"
          f"反序列化 {summary['client']['parse_mean_ms']:.4f} ms")
    for key, label in [("e2e", "服务端 e2e"), ("queue", "排队"),
                       ("inference", "推理"), ("prefill", "prefill")]:
        d = delta[key]
        if d["mean_ms"] is None:
            print(f"{label:<10} 无数据")
            continue
        print(f"{label:<10} 平均 {d['mean_ms']:>8.3f} ms  "
              f"合计 {d['sum'] * 1000 if d['sum'] else 0:>9.3f} ms  计数 {d['count']}")
    if "closure" in summary:
        c = summary["closure"]
        print(f"\n闭合：客户端平均 {c['client_wall_mean_ms']:.3f} ms − 服务端 e2e 平均 "
              f"{c['server_e2e_mean_ms']:.3f} ms = {c['client_minus_server_ms']:.3f} ms")
        print(f"  其中客户端序列化+反序列化 {c['known_client_side_ms']:.4f} ms；"
              f"其余为网络往返与调度抖动（未分解）")
        print(f"  服务端内部：排队 {c['server_queue_mean_ms']:.3f} + 推理 "
              f"{c['server_inference_mean_ms']:.3f} ms"
              f"（prefill {c['server_prefill_mean_ms']:.3f}）")
        print(f"  服务端计数与请求数一致：{c['server_count_matches_requests']}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
