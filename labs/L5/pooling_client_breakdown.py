#!/usr/bin/env python3
"""5.12 客户端残差分段插桩：测量端到端时延的各阶段开销。

分段：
1. 请求构造（序列化）
2. 网络传输到服务器
3. 服务器处理（推理）
4. 网络传输回客户端
5. 响应解析（反序列化）
"""
import argparse
import json
import socket
import time
from pathlib import Path


def benchmark_breakdown(host: str, port: int, model: str, text: str, repeats: int = 30):
    """分段测量客户端到服务器的时延"""
    results = []

    for i in range(repeats):
        # 1. 构造请求（序列化）
        t0 = time.perf_counter()
        payload_dict = {"model": model, "input": text}
        payload = json.dumps(payload_dict).encode('utf-8')
        t1 = time.perf_counter()
        serialize_ms = (t1 - t0) * 1000

        # 2-4. 网络 + 服务器 + 网络（HTTP 往返）
        t2 = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        t3 = time.perf_counter()
        connect_ms = (t3 - t2) * 1000

        # 发送 HTTP 请求
        request = (
            f"POST /pooling HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode('utf-8') + payload

        t4 = time.perf_counter()
        sock.sendall(request)
        t5 = time.perf_counter()
        send_ms = (t5 - t4) * 1000

        # 接收响应
        t6 = time.perf_counter()
        response_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
        t7 = time.perf_counter()
        recv_ms = (t7 - t6) * 1000

        sock.close()

        # 5. 解析响应（反序列化）
        t8 = time.perf_counter()
        # 分离 HTTP 头和 body
        header_end = response_data.find(b"\r\n\r\n")
        if header_end == -1:
            results.append({
                "iteration": i,
                "error": "No HTTP header found",
                "success": False
            })
            continue

        body = response_data[header_end + 4:]
        try:
            response_json = json.loads(body)
            # vLLM pooling 返回 {"data": [{"data": [...]}]}
            embedding_dim = len(response_json["data"][0]["data"])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            results.append({
                "iteration": i,
                "error": str(e),
                "success": False
            })
            continue

        t9 = time.perf_counter()
        deserialize_ms = (t9 - t8) * 1000

        total_ms = (t9 - t0) * 1000
        server_ms = total_ms - serialize_ms - connect_ms - send_ms - recv_ms - deserialize_ms

        results.append({
            "iteration": i,
            "serialize_ms": serialize_ms,
            "connect_ms": connect_ms,
            "send_ms": send_ms,
            "recv_ms": recv_ms,
            "deserialize_ms": deserialize_ms,
            "server_inferred_ms": server_ms,
            "total_ms": total_ms,
            "embedding_dim": embedding_dim,
            "success": True
        })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Testing {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Repeats: {args.repeats}")

    results = benchmark_breakdown(args.host, args.port, args.model, args.text, args.repeats)

    # 统计
    successful = [r for r in results if r.get("success")]
    failed = len(results) - len(successful)

    if not successful:
        print("All requests failed!")
        out_file = args.out / "client_breakdown.json"
        with out_file.open("w") as f:
            json.dump({"results": results, "failed": failed}, f, indent=2)
        return

    def median(values):
        sorted_values = sorted(values)
        return sorted_values[len(sorted_values) // 2]

    summary = {
        "serialize_ms": median([r["serialize_ms"] for r in successful]),
        "connect_ms": median([r["connect_ms"] for r in successful]),
        "send_ms": median([r["send_ms"] for r in successful]),
        "recv_ms": median([r["recv_ms"] for r in successful]),
        "deserialize_ms": median([r["deserialize_ms"] for r in successful]),
        "server_inferred_ms": median([r["server_inferred_ms"] for r in successful]),
        "total_ms": median([r["total_ms"] for r in successful]),
        "num_successful": len(successful),
        "num_failed": failed
    }

    out_file = args.out / "client_breakdown.json"
    with out_file.open("w") as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"\nBreakdown (median):")
    print(f"  Serialize:    {summary['serialize_ms']:.3f} ms")
    print(f"  Connect:      {summary['connect_ms']:.3f} ms")
    print(f"  Send:         {summary['send_ms']:.3f} ms")
    print(f"  Recv:         {summary['recv_ms']:.3f} ms")
    print(f"  Deserialize:  {summary['deserialize_ms']:.3f} ms")
    print(f"  Server (inferred): {summary['server_inferred_ms']:.3f} ms")
    print(f"  Total:        {summary['total_ms']:.3f} ms")
    print(f"\nClient overhead: {summary['serialize_ms'] + summary['connect_ms'] + summary['send_ms'] + summary['recv_ms'] + summary['deserialize_ms']:.3f} ms")


if __name__ == "__main__":
    main()
