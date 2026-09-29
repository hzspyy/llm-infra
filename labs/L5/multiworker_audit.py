#!/usr/bin/env python3
"""L5.11 任务 C —— 多 worker / 多 tokenizer 进程：前端能不能吃满并发。

计划里这一格写的是「多 worker 未测」（本章此前都是单进程）。两个引擎各有对应旋钮：

  vLLM   `--api-server-count N`（N 个 API server 进程共享同一个 EngineCore）
  SGLang `--tokenizer-worker-num N` / `--detokenizer-worker-num N`

负载设计成"前端有活干"：prompt 512 token（分词不是免费）、并发 32、每条生成 16 token，
每档 3 轮交错重复。记录 TTFT 的 p50/p99、整批墙钟、单请求总时长与客户端 CPU 占比。

一个可证伪的预期：如果单 API server 的事件循环是瓶颈，那么把 N 从 1 提到 4
应当**主要改善 TTFT 的尾部**（p99），而稳态吞吐变化不大——因为推理段本来就是共享的。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


API = {"path": "/generate", "payload": None}


def one_request(base, text, max_tokens, stream=True):
    t0 = time.perf_counter()
    first = None
    n_chunks = 0
    payload = API["payload"](text, max_tokens)
    with requests.post(base + API["path"], json=payload, stream=True,
                       timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith(b"data:"):
                n_chunks += 1
                if first is None:
                    first = time.perf_counter()
    total = time.perf_counter() - t0
    return dict(ttft_ms=round((first - t0) * 1000, 3) if first else None,
                total_ms=round(total * 1000, 3), chunks=n_chunks)


def cell(base, text, batch, max_tokens):
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch) as ex:
        outs = list(ex.map(lambda _: one_request(base, text, max_tokens),
                           range(batch)))
    wall = (time.perf_counter() - t0) * 1000
    cpu = (time.process_time() - cpu0) * 1000
    ttfts = sorted(o["ttft_ms"] for o in outs if o["ttft_ms"])
    totals = sorted(o["total_ms"] for o in outs)
    return dict(batch=batch, wall_ms=round(wall, 3),
                ttft_p50_ms=round(ttfts[len(ttfts) // 2], 3),
                ttft_p99_ms=round(ttfts[int(len(ttfts) * 0.99) - 1], 3),
                ttft_max_ms=round(ttfts[-1], 3),
                total_p50_ms=round(totals[len(totals) // 2], 3),
                total_p99_ms=round(totals[int(len(totals) * 0.99) - 1], 3),
                client_cpu_per_req_ms=round(cpu / batch, 3),
                client_cpu_over_wall=round(cpu / wall, 3))


def sglang_payload(text, max_tokens):
    return {"text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
            "stream": True}


def openai_payload(model):
    def make(text, max_tokens):
        return {"model": model, "prompt": text, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True}
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--api", choices=["sglang", "openai"], default="sglang")
    ap.add_argument("--served-name", default="m")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.api == "sglang":
        API["path"], API["payload"] = "/generate", sglang_payload
    else:
        # vLLM 的 OpenAI 端点是 /v1/completions，载荷形状与 SGLang 不同；
        # 上一版两边共用 SGLang 形状，vLLM 那两档直接 404 了
        API["path"], API["payload"] = "/v1/completions", openai_payload(args.served_name)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = "Token " * args.prompt_tokens
    ids = tok.encode(text, add_special_tokens=False)

    rows = []
    for rep in range(args.repeats):
        rec = cell(args.base, text, args.batch, args.max_tokens)
        rec.update(rep=rep, label=args.label, prompt_tokens=len(ids))
        rows.append(rec)
        print(f"  [{args.label}] rep{rep}  TTFT p50 {rec['ttft_p50_ms']} ms  "
              f"p99 {rec['ttft_p99_ms']} ms  max {rec['ttft_max_ms']} ms  "
              f"整批 {rec['wall_ms']} ms  单请求 p50 {rec['total_p50_ms']} ms  "
              f"客户端CPU/请求 {rec['client_cpu_per_req_ms']} ms", flush=True)

    summary = dict(
        label=args.label, prompt_tokens=len(ids), batch=args.batch,
        ttft_p50_median=round(statistics.median(r["ttft_p50_ms"] for r in rows), 3),
        ttft_p99_median=round(statistics.median(r["ttft_p99_ms"] for r in rows), 3),
        wall_median=round(statistics.median(r["wall_ms"] for r in rows), 3),
        total_p50_median=round(statistics.median(r["total_p50_ms"] for r in rows), 3))
    (args.out / f"multiworker_{args.label}.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  汇总 {args.label}: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
