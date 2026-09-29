#!/usr/bin/env python3
"""L5.11 任务 C 的 HTTP/2 部分：HTTP/1.1 基线 + 明文 HTTP/2（h2c）对照。

计划写的是「HTTP/1.1 为基线，支持时补 HTTP/2」。SGLang 0.5.19 有
`--enable-http2`（server_args 里还有 `http2_max_concurrent_streams`），
本机 curl 7.81 带 nghttp2，所以这一格可以真的测：

  * 同一个服务上分别用 HTTP/1.1 与 **h2c（prior knowledge）** 发同一批请求；
  * 记录 `%{http_version}`——这是协议协商的直接证据，不靠推断；
  * 流式请求在两种协议下都能到（SSE 走 h2 的 DATA 帧）；
  * 32 条并发：HTTP/1.1 需要多条连接，h2 可以在**一条连接**上开多路复用，
    比较总墙钟与逐请求时间。

Python 侧本机没有 `h2`，httpx 也没有装 http2 依赖，所以客户端一律用 curl 子进程，
并把原始输出落盘。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

PROMPT = ("Explain in detail how a paged KV cache works in an inference "
          "engine, step by step.")
PAYLOAD_TMPL = ('{"text": %s, "sampling_params": {"temperature": 0, '
                '"max_new_tokens": %d}, "stream": %s}')


def curl_version_flag(proto):
    return ["--http2-prior-knowledge"] if proto == "h2c" else ["--http1.1"]


def one(base, proto, max_new_tokens=8, stream=False, timeout=120):
    import json as _json
    payload = PAYLOAD_TMPL % (_json.dumps(PROMPT), max_new_tokens,
                              "true" if stream else "false")
    fmt = "%{http_version} %{time_starttransfer} %{time_total} %{size_download}"
    cmd = (["curl", "-sS", "-o", "/dev/null", "-w", fmt] + curl_version_flag(proto)
           + ["-X", "POST", base + "/generate", "-H", "Content-Type: application/json",
              "--data-binary", payload, "--max-time", str(timeout)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 20)
    if r.returncode != 0:
        return dict(error=r.stderr.strip()[:200], cmd=" ".join(cmd[:6]))
    parts = r.stdout.strip().split()
    if len(parts) < 4:
        return dict(error=f"unexpected curl output: {r.stdout[:120]!r}")
    return dict(http_version=parts[0], starttransfer_s=float(parts[1]),
                total_s=float(parts[2]), bytes=int(parts[3]))


def stream_capture(base, proto, out_file: pathlib.Path, max_new_tokens=16):
    import json as _json
    payload = PAYLOAD_TMPL % (_json.dumps(PROMPT), max_new_tokens, "true")
    cmd = (["curl", "-sS", "-N"] + curl_version_flag(proto)
           + ["-X", "POST", base + "/generate", "-H", "Content-Type: application/json",
              "--data-binary", payload, "--max-time", "120"])
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    wall = time.perf_counter() - t0
    out_file.write_bytes(r.stdout)
    return dict(proto=proto, wall_s=round(wall, 3), bytes=len(r.stdout),
                data_events=r.stdout.count(b"data:"),
                stderr=r.stderr.decode()[:120],
                http_version_hint=("h2" if proto == "h2c" else "1.1"))


def parallel(base, proto, n, out_file: pathlib.Path, max_new_tokens=8):
    """用 curl --parallel 并发 n 条请求。

    注意：curl 的 `--parallel` 下每个传输要用 `--next` 分开，
    否则连写的 `--data-binary` 会被拼到同一个请求体里——上一版就是这么错的，
    服务端返回的是 `JSON decode error: unexpected content after document`。
    请求体写成文件用 `--data-binary @file` 引用，避免参数里带 JSON 的转义问题。
    """
    import json as _json
    payload = PAYLOAD_TMPL % (_json.dumps(PROMPT), max_new_tokens, "false")
    body_file = out_file.with_suffix(".body.json")
    body_file.write_text(payload, encoding="utf-8")
    fmt = "%{http_version} %{time_total} %{size_download}\n"
    groups = []
    for i in range(n):
        if i:
            groups.append("--next")
        # -o/-w 必须**每个传输一份**：全局的只作用于第一个传输，
        # 后面的响应体会直接打到 stdout，解析就乱了
        groups += ["-o", "/dev/null", "-w", fmt,
                   "-X", "POST", base + "/generate",
                   "-H", "Content-Type: application/json",
                   "--data-binary", f"@{body_file}"]
    cmd = (["curl", "-sS"] + curl_version_flag(proto)
           + ["--parallel", "--parallel-max", str(n), "--max-time", "180"] + groups)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall = time.perf_counter() - t0
    out_file.write_text(r.stdout + "\n# stderr:\n" + r.stderr, encoding="utf-8")
    times, versions, sizes, bad = [], [], [], 0
    for line in r.stdout.strip().splitlines():
        p = line.split()
        if len(p) >= 3:
            versions.append(p[0])
            try:
                times.append(float(p[1]))
                sizes.append(int(p[2]))
            except ValueError:
                bad += 1
    import statistics as st
    return dict(proto=proto, n=n, wall_s=round(wall, 3),
                versions=sorted(set(versions)), n_ok=len(times), n_bad=bad,
                per_req_median_s=round(st.median(times), 3) if times else None,
                per_req_max_s=round(max(times), 3) if times else None,
                bytes_min=min(sizes) if sizes else None,
                bytes_max=max(sizes) if sizes else None,
                returncode=r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"base": args.base, "n_parallel": args.n, "results": {}}
    for proto in ("http1.1", "h2c"):
        entry = {}
        entry["single"] = one(args.base, proto)
        entry["stream"] = stream_capture(args.base, proto,
                                         args.out / f"stream-{proto}.txt")
        entry["parallel"] = parallel(args.base, proto, args.n,
                                     args.out / f"parallel-{proto}.txt")
        report["results"][proto] = entry
        print(f"  {proto:<7} 单请求 http_version={entry['single'].get('http_version')} "
              f"总 {entry['single'].get('total_s')} s；流式 "
              f"{entry['stream'].get('data_events')} 个事件；并发 {args.n} 条："
              f"墙钟 {entry['parallel'].get('wall_s')} s，"
              f"逐请求中位 {entry['parallel'].get('per_req_median_s')} s，"
              f"版本集合 {entry['parallel'].get('versions')}", flush=True)

    (args.out / "http2_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    a = report["results"]["http1.1"]["parallel"]
    b = report["results"]["h2c"]["parallel"]
    if a.get("wall_s") and b.get("wall_s"):
        print(f"\n并发 {args.n} 条：HTTP/1.1 {a['wall_s']} s → h2c {b['wall_s']} s，"
              f"差 {a['wall_s'] - b['wall_s']:+.3f} s")


if __name__ == "__main__":
    main()
