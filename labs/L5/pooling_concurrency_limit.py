#!/usr/bin/env python3
"""L5.12 补测 · 池化服务的并发容量边界与高并发残差分解。

两件事合在一个扫描里：

  * **容量边界**：并发按 1/8/32/64/128/256 递增，每档发 4×并发 条请求，记录成功、
    失败（连接被拒/读超时）与客户端 p50/p95/p99；到出现失败为止就是这台服务在这组
    形状下的边界。
  * **残差分解**：每条请求按 socket 分段计时（序列化 / connect / send / recv /
    反序列化），剩下的部分作为"服务端含网络"的推断值；同时按服务端 `/metrics`
    的 histogram 增量算出服务端自己的 e2e/inference/queue/prefill 均值。两者相减
    就是客户端观测里不在服务端计时内的部分（TCP 往返 + 前端入队/出队 + 内核调度），
    高并发时它如何变化是本节的重点。

显存峰值由独立线程用 `nvidia-smi` 轮询采样，避免用客户端进程的 torch 读数。

用法（服务已在 --base 就绪）：
    python pooling_concurrency_limit.py --base http://127.0.0.1:8125 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TEXT = ("Retrieval augmented generation combines a retriever with a generator. "
        "The retriever returns passages and the generator conditions on them. ")
KEYS = ("vllm:e2e_request_latency_seconds", "vllm:request_inference_time_seconds",
        "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds")


def parse_metrics(text: str) -> dict:
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        base = None
        for k in KEYS:
            for suf in ("_sum", "_count"):
                if name == k + suf:
                    base = (k, suf[1:])
        if base is None:
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        d = out.setdefault(base[0], {})
        d[base[1]] = d.get(base[1], 0.0) + val
    return out


def fetch_metrics(base: str) -> dict:
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        return parse_metrics(r.read().decode("utf-8", "replace"))


def gpu_mem_poll(stop: threading.Event, out: list[int]) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            out.append(int(r.stdout.strip().splitlines()[0]))
        except Exception:                                          # noqa: BLE001
            pass
        stop.wait(0.5)


def one_request(host: str, port: int, payload: bytes, timeout: float) -> dict:
    t0 = time.perf_counter()
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        return dict(ok=False, error=f"connect:{exc}", total_ms=(time.perf_counter() - t0) * 1000)
    try:
        t1 = time.perf_counter()
        head = (f"POST /pooling HTTP/1.1\r\nHost: {host}:{port}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                f"Connection: close\r\n\r\n").encode()
        sock.sendall(head + payload)
        t2 = time.perf_counter()
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        t3 = time.perf_counter()
    except OSError as exc:
        sock.close()
        return dict(ok=False, error=f"io:{exc}", total_ms=(time.perf_counter() - t0) * 1000)
    finally:
        try:
            sock.close()
        except OSError:
            pass
    t4 = time.perf_counter()
    body = data.split(b"\r\n\r\n", 1)[-1]
    try:
        dim = len(json.loads(body)["data"][0]["data"])
    except Exception as exc:                                       # noqa: BLE001
        return dict(ok=False, error=f"parse:{exc}", total_ms=(time.perf_counter() - t0) * 1000)
    t5 = time.perf_counter()
    return dict(ok=True, dim=dim,
                send_ms=(t2 - t1) * 1000, recv_ms=(t3 - t2) * 1000,
                deserialize_ms=(t5 - t4) * 1000, total_ms=(t5 - t0) * 1000)


def worker_keepalive(host, port, payload, count, timeout):
    """每线程一条 keep-alive 连接发 count 条，用来和"每请求新建连接"对照。"""
    import http.client
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    rows = []
    for _ in range(count):
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/pooling", payload,
                         {"Content-Type": "application/json"})
            r = conn.getresponse()
            r.read()
            rows.append(dict(ok=True, total_ms=(time.perf_counter() - t0) * 1000,
                             send_ms=None, recv_ms=None, deserialize_ms=None))
        except Exception as exc:                                   # noqa: BLE001
            rows.append(dict(ok=False, error=f"ka:{str(exc)[:60]}",
                             total_ms=(time.perf_counter() - t0) * 1000))
            try:
                conn.close()
            except Exception:                                      # noqa: BLE001
                pass
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.close()
    except Exception:                                              # noqa: BLE001
        pass
    return rows


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, int(q * len(xs)))
    return round(xs[i], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8125")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--levels", default="1,8,32,64,128,256")
    ap.add_argument("--requests-cap", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--text-repeat", type=int, default=3,
                    help="输入文本重复次数；加大它把服务端推到饱和")
    ap.add_argument("--keepalive", action="store_true",
                    help="每线程一条 keep-alive 连接（默认每请求新建连接）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    host = urllib.parse.urlparse(args.base).hostname
    port = urllib.parse.urlparse(args.base).port
    payload = json.dumps({"model": args.model,
                          "input": TEXT * args.text_repeat}).encode()

    # 预热
    one_request(host, port, payload, args.timeout)

    out, rep = [], {}
    out.append(f"L5.12 池化服务并发容量与残差分解 · {args.base}")
    out.append(f"  {'并发':>5}{'请求':>7}{'成功':>6}{'失败':>6}{'QPS':>8}"
               f"{'客户端p50':>11}{'客户端p99':>11}{'服务端e2e':>11}"
               f"{'inference':>11}{'queue':>9}{'残差p50':>10}{'显存峰值MiB':>13}")
    for c in (int(x) for x in args.levels.split(",")):
        n = min(args.requests_cap, 4 * c)
        before = fetch_metrics(args.base)
        mem: list[int] = []
        stop = threading.Event()
        poller = threading.Thread(target=gpu_mem_poll, args=(stop, mem), daemon=True)
        poller.start()
        t0 = time.perf_counter()
        if args.keepalive:
            per = max(1, n // c)
            with ThreadPoolExecutor(max_workers=c) as ex:
                nested = list(ex.map(
                    lambda _: worker_keepalive(host, port, payload, per, args.timeout),
                    range(c)))
            rows = [r for sub in nested for r in sub]
        else:
            with ThreadPoolExecutor(max_workers=c) as ex:
                rows = list(ex.map(
                    lambda _: one_request(host, port, payload, args.timeout), range(n)))
        wall = time.perf_counter() - t0
        stop.set()
        poller.join(timeout=3)
        after = fetch_metrics(args.base)

        ok = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        cli = [r["total_ms"] for r in ok]
        # 服务端只对完成的请求计时，所以它的均值与客户端成功请求同分母
        srv = {}
        for k in KEYS:
            b, a = before.get(k) or {}, after.get(k) or {}
            cnt = (a.get("count", 0.0) - b.get("count", 0.0))
            sm = (a.get("sum", 0.0) - b.get("sum", 0.0))
            srv[k.split(":")[1]] = round(1000 * sm / cnt, 3) if cnt else None
        resid = None
        if srv.get("e2e_request_latency_seconds") and cli:
            resid = round(statistics.median(cli) - srv["e2e_request_latency_seconds"], 3)
        # 客户端分段（只看成功请求）
        seg = {}
        for k in ("send_ms", "recv_ms", "deserialize_ms"):
            vals = [r[k] for r in ok if r.get(k) is not None]
            seg[k] = round(statistics.median(vals), 3) if vals else None
        rec = dict(concurrency=c, requests=len(rows), ok=len(ok), failed=len(bad),
                   mode="keepalive" if args.keepalive else "close",
                   qps=round(len(ok) / wall, 1) if wall else None,
                   errors=sorted({r.get("error", "?").split(":")[0] for r in bad}),
                   wall_s=round(wall, 2),
                   client_p50=pct(cli, 0.5), client_p95=pct(cli, 0.95),
                   client_p99=pct(cli, 0.99),
                   server_ms=srv, residual_p50=resid, segments=seg,
                   gpu_peak_mib=max(mem) if mem else None)
        rep[str(c)] = rec
        out.append(f"  {c:>5}{len(rows):>7}{len(ok):>6}{len(bad):>6}"
                   f"{str(rec['qps']):>8}"
                   f"{str(rec['client_p50']):>11}{str(rec['client_p99']):>11}"
                   f"{str(srv.get('e2e_request_latency_seconds')):>11}"
                   f"{str(srv.get('request_inference_time_seconds')):>11}"
                   f"{str(srv.get('request_queue_time_seconds')):>9}"
                   f"{str(resid):>10}{str(rec['gpu_peak_mib']):>13}")
        if rec["errors"]:
            out.append(f"        错误类型：{rec['errors']}")

    text = "\n".join(out)
    print(text)
    (args.out / "pooling_concurrency.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "pooling_concurrency.json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n写入 {args.out}/pooling_concurrency.txt")


if __name__ == "__main__":
    main()
