#!/usr/bin/env python3
"""L5.10 —— 两引擎对齐：SGLang 的 CUDA Graph 开关对 decode 的影响。

vLLM 侧的图模式在 5.4 量过（FULL_AND_PIECEWISE、捕获尺寸、padding 表）。
任务 D 要求「在两引擎对齐完整路径，比较 eager/graph、驻留/换入」，
这里补 SGLang 一侧：同一个模型、同一批请求，只切
`--cuda-graph-backend-decode full`（默认）与 `disabled`，比较

  * TTFT（首包，流式第一个 chunk）
  * decode 的**逐 token 间隔**（第二个 chunk 起的间隔中位/最大）
  * 总墙钟与吞吐

用流式接口按到达时刻量，不依赖服务端自报指标——这样两引擎可以用同一把尺子。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def stream_one(base, ids, max_new_tokens, timeout=300):
    """返回 (TTFT ms, 逐 token 间隔 ms 列表, 总 ms, 事件数)。"""
    payload = {"input_ids": ids, "sampling_params":
               {"temperature": 0.0, "max_new_tokens": max_new_tokens},
               "stream": True}
    t0 = time.perf_counter()
    marks = []
    with requests.post(base + "/generate", json=payload, stream=True,
                       timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data:"):
                body = line[5:].strip()
                if body and body != b"[DONE]":
                    marks.append((time.perf_counter() - t0) * 1000)
    if not marks:
        return None
    ttft = marks[0]
    gaps = [b - a for a, b in zip(marks, marks[1:])]
    return dict(ttft_ms=round(ttft, 3), n_chunks=len(marks),
                total_ms=round(marks[-1], 3),
                gap_median_ms=round(statistics.median(gaps), 3) if gaps else None,
                gap_max_ms=round(max(gaps), 3) if gaps else None,
                sum_gaps_ms=round(sum(gaps), 3) if gaps else None)


def cell(base, ids, batch, max_new_tokens):
    with ThreadPoolExecutor(max_workers=batch) as ex:
        t0 = time.perf_counter()
        outs = list(ex.map(lambda _: stream_one(base, ids, max_new_tokens),
                           range(batch)))
    wall = (time.perf_counter() - t0) * 1000
    ok = [o for o in outs if o]
    if not ok:
        return dict(batch=batch, error="no streamed output")
    return dict(batch=batch, n_ok=len(ok), wall_ms=round(wall, 3),
                ttft_median_ms=round(statistics.median(o["ttft_ms"] for o in ok), 3),
                ttft_max_ms=round(max(o["ttft_ms"] for o in ok), 3),
                gap_median_ms=round(statistics.median(o["gap_median_ms"] for o in ok
                                                      if o["gap_median_ms"]), 3),
                gap_max_max_ms=round(max(o["gap_max_ms"] for o in ok
                                         if o["gap_max_ms"]), 3),
                per_req_total_median_ms=round(
                    statistics.median(o["total_ms"] for o in ok), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--prompt", default="Explain in detail how a paged KV cache works "
                                        "in an inference engine, step by step.")
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tok.encode(args.prompt, add_special_tokens=False)

    rows = []
    for batch in args.batches:
        for rep in range(args.repeats):
            rec = cell(args.base, ids, batch, args.max_new_tokens)
            rec.update(rep=rep)
            rows.append(rec)
            print(f"  [{args.label}] batch {batch:>2} rep {rep}  "
                  f"TTFT 中位 {rec.get('ttft_median_ms')} ms  "
                  f"逐 token 间隔中位 {rec.get('gap_median_ms')} ms  "
                  f"最大 {rec.get('gap_max_max_ms')} ms  "
                  f"整请求中位 {rec.get('per_req_total_median_ms')} ms", flush=True)

    summary = {}
    for batch in args.batches:
        xs = [r for r in rows if r.get("batch") == batch and "error" not in r]
        if not xs:
            continue
        summary[batch] = dict(
            ttft_median=round(statistics.median(r["ttft_median_ms"] for r in xs), 3),
            gap_median=round(statistics.median(r["gap_median_ms"] for r in xs), 3),
            gap_max=round(max(r["gap_max_max_ms"] for r in xs), 3),
            per_req_total=round(statistics.median(r["per_req_total_median_ms"]
                                                  for r in xs), 3))
    (args.out / f"graph_{args.label}.json").write_text(
        json.dumps(dict(label=args.label, max_new_tokens=args.max_new_tokens,
                        rows=rows, summary=summary), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  汇总 {args.label}: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
