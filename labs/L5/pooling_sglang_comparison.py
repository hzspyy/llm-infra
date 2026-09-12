#!/usr/bin/env python3
"""5.12 SGLang pooling 对照：对比 vLLM 与 SGLang 的 pooling 实现。

测试维度：
1. 端到端时延
2. 显存占用
3. API 接口差异
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def benchmark_pooling(base_url: str, model: str, text: str, repeats: int = 20, engine: str = "vllm"):
    """测量 pooling 请求"""
    results = []

    # vLLM 用 /pooling，SGLang 用 /encode
    endpoint = "/encode" if engine == "sglang" else "/pooling"

    for i in range(repeats):
        if engine == "sglang":
            payload = json.dumps({"text": text}).encode('utf-8')
        else:
            payload = json.dumps({"model": model, "input": text}).encode('utf-8')

        req = urllib.request.Request(
            base_url + endpoint,
            data=payload,
            headers={'Content-Type': 'application/json'}
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                response_data = json.loads(response.read())
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000

            # vLLM 返回 {"data": [{"embedding": [...]}]}, SGLang 返回 {"embedding": [...]}
            if engine == "sglang":
                embedding_dim = len(response_data["embedding"])
            else:
                embedding_dim = len(response_data["data"][0]["embedding"])

            results.append({
                "iteration": i,
                "latency_ms": latency_ms,
                "success": True,
                "embedding_dim": embedding_dim
            })

        except Exception as e:
            results.append({
                "iteration": i,
                "error": str(e),
                "success": False
            })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--text", default="The prefix cache mechanism enables reuse.")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Testing {args.engine} at {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Repeats: {args.repeats}")

    results = benchmark_pooling(args.base_url, args.model, args.text, args.repeats, args.engine)

    successful = [r for r in results if r.get("success")]
    if not successful:
        print("All requests failed!")
        return

    latencies = [r["latency_ms"] for r in successful]
    latencies_sorted = sorted(latencies)

    summary = {
        "engine": args.engine,
        "model": args.model,
        "base_url": args.base_url,
        "text": args.text,
        "total_requests": args.repeats,
        "successful_requests": len(successful),
        "latency_ms": {
            "median": latencies_sorted[len(latencies_sorted)//2],
            "p50": latencies_sorted[len(latencies_sorted)//2],
            "p90": latencies_sorted[int(len(latencies_sorted)*0.9)],
            "p99": latencies_sorted[int(len(latencies_sorted)*0.99)],
            "min": min(latencies),
            "max": max(latencies)
        },
        "embedding_dim": successful[0]["embedding_dim"]
    }

    out_file = args.out / f"{args.engine}_results.json"
    with out_file.open("w") as f:
        json.dump({
            "summary": summary,
            "raw_results": results
        }, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"\nLatency (median): {summary['latency_ms']['median']:.2f} ms")
    print(f"Latency (p90): {summary['latency_ms']['p90']:.2f} ms")
    print(f"Embedding dim: {summary['embedding_dim']}")


if __name__ == "__main__":
    main()
