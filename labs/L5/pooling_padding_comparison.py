#!/usr/bin/env python3
"""5.12 padding 正确对照：控制相同 token 数和内容，测量 padding 开销。

实验设计：
1. 固定内容的多个文本（不同长度）
2. 用 tokenizer 预编码，确保 token 数已知
3. 单独请求 vs 批量请求（自动 padding）
4. 测量端到端时延和显存增量
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer


def benchmark_single(base_url: str, model: str, texts: list[str], repeats: int = 10):
    """单独请求，无 padding"""
    results = []

    for r in range(repeats):
        round_start = time.perf_counter()
        for i, text in enumerate(texts):
            payload = json.dumps({"model": model, "input": text}).encode('utf-8')
            req = urllib.request.Request(
                base_url + "/pooling",
                data=payload,
                headers={'Content-Type': 'application/json'}
            )

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as response:
                _ = json.loads(response.read())
            t1 = time.perf_counter()

            results.append({
                "round": r,
                "text_idx": i,
                "latency_ms": (t1 - t0) * 1000,
                "mode": "single"
            })

        round_end = time.perf_counter()
        results.append({
            "round": r,
            "total_latency_ms": (round_end - round_start) * 1000,
            "mode": "single",
            "is_summary": True
        })

    return results


def benchmark_batch(base_url: str, model: str, texts: list[str], repeats: int = 10):
    """批量请求，引擎自动 padding"""
    results = []

    for r in range(repeats):
        payload = json.dumps({"model": model, "input": texts}).encode('utf-8')
        req = urllib.request.Request(
            base_url + "/pooling",
            data=payload,
            headers={'Content-Type': 'application/json'}
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read())
        t1 = time.perf_counter()

        results.append({
            "round": r,
            "latency_ms": (t1 - t0) * 1000,
            "num_texts": len(texts),
            "num_results": len(response_data["data"]),
            "mode": "batch"
        })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 准备固定内容的文本，token 数递增
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    base_text = "The quick brown fox jumps over the lazy dog. "
    texts = []
    token_counts = []

    for length in [10, 20, 40, 80]:
        text = (base_text * (length // len(base_text.split()) + 1))[:length * 5]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        texts.append(text)
        token_counts.append(len(tokens))

    print(f"Prepared {len(texts)} texts with token counts: {token_counts}")
    print(f"Max tokens: {max(token_counts)}, padding overhead: {sum(token_counts)} -> {max(token_counts) * len(texts)}")

    # 运行单独和批量模式
    print("\n[1/2] Running single mode (no padding)...")
    single_results = benchmark_single(args.base_url, args.model, texts, args.repeats)

    print("[2/2] Running batch mode (with padding)...")
    batch_results = benchmark_batch(args.base_url, args.model, texts, args.repeats)

    # 计算统计
    single_summaries = [r for r in single_results if r.get("is_summary")]
    single_latencies = [r["total_latency_ms"] for r in single_summaries]
    batch_latencies = [r["latency_ms"] for r in batch_results]

    summary = {
        "texts": texts,
        "token_counts": token_counts,
        "padding_overhead_tokens": max(token_counts) * len(texts) - sum(token_counts),
        "padding_overhead_ratio": (max(token_counts) * len(texts)) / sum(token_counts),
        "single_mode": {
            "median_ms": sorted(single_latencies)[len(single_latencies)//2],
            "min_ms": min(single_latencies),
            "max_ms": max(single_latencies)
        },
        "batch_mode": {
            "median_ms": sorted(batch_latencies)[len(batch_latencies)//2],
            "min_ms": min(batch_latencies),
            "max_ms": max(batch_latencies)
        },
        "speedup": sorted(single_latencies)[len(single_latencies)//2] / sorted(batch_latencies)[len(batch_latencies)//2]
    }

    out_file = args.out / "padding_comparison.json"
    with out_file.open("w") as f:
        json.dump({
            "summary": summary,
            "single_results": single_results,
            "batch_results": batch_results
        }, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"\nSingle mode (no padding): {summary['single_mode']['median_ms']:.2f} ms")
    print(f"Batch mode (with padding): {summary['batch_mode']['median_ms']:.2f} ms")
    print(f"Speedup: {summary['speedup']:.2f}x")
    print(f"Padding overhead: {summary['padding_overhead_tokens']} tokens ({summary['padding_overhead_ratio']:.2f}x)")


if __name__ == "__main__":
    main()
