#!/usr/bin/env python3
"""0.5-C: 流式输出里 token、文本 delta 与 SSE event 是三种不同的计数。

用原始 socket 发一次 chat completion（stream=true），把服务端回来的字节原样存盘，
然后分三层重建计数：

    HTTP chunk   传输层分帧（chunked transfer-encoding 的长度行）
    SSE event    应用层分帧（`data: ` 开头的行，末尾一条是 [DONE]）
    文本 delta   真正带内容的字段（content / reasoning_content / tool_calls）
    token        usage.completion_tokens，由服务端给出

三种请求分别看三种分段：纯文本、reasoning（think 段）、tool call。
reasoning 与 tool call 由 parser 从同一串 token 里切出来，detokenizer 不认识这些概念。

用法（服务已在跑）：
    python labs/L0/stream_events_probe.py --out-dir <dir> --base-url http://127.0.0.1:8021
整套流程见 labs/L0/run_stream_events.sh。
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

SEP = "-" * 78
MODEL = "Qwen/Qwen3-1.7B"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询某个城市当前的天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市名"}},
                       "required": ["city"]},
    },
}]


def post_stream(base_url: str, body: dict, raw_path: Path) -> tuple[bytes, list[tuple[float, bytes]]]:
    """原始 socket 发 POST，按到达顺序记录每次 recv 的字节与时间。"""
    u = urlparse(base_url)
    payload = json.dumps(body, ensure_ascii=False).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
           f"Accept: text/event-stream\r\nConnection: close\r\n\r\n").encode() + payload
    s = socket.create_connection((u.hostname, u.port), timeout=120)
    s.sendall(req)
    chunks: list[tuple[float, bytes]] = []
    buf = b""
    t0 = time.perf_counter()
    while True:
        b = s.recv(65536)
        if not b:
            break
        chunks.append((time.perf_counter() - t0, b))
        buf += b
    s.close()
    raw_path.write_bytes(buf)
    return buf, chunks


def split_head(raw: bytes) -> tuple[bytes, bytes]:
    i = raw.find(b"\r\n\r\n")
    return raw[:i], raw[i + 4:]


def parse_events(body: bytes) -> tuple[list[dict], int, int]:
    """返回 (SSE 事件列表, HTTP chunk 数, keep-alive 注释行数)。"""
    events, n_http_chunks, comments = [], 0, 0
    pos = 0
    data = b""
    while pos < len(body):
        nl = body.find(b"\r\n", pos)
        if nl == -1:
            break
        size_line = body[pos:nl]
        try:
            size = int(size_line.split(b";")[0], 16)
        except ValueError:
            break
        n_http_chunks += 1
        if size == 0:
            break
        data += body[nl + 2:nl + 2 + size]
        pos = nl + 2 + size + 2
    for line in data.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(b":"):
            comments += 1
            continue
        if line.startswith(b"data: "):
            payload = line[6:]
            if payload == b"[DONE]":
                events.append(dict(kind="done"))
            else:
                events.append(dict(kind="data", obj=json.loads(payload)))
    return events, n_http_chunks, comments


def classify(events: list[dict]) -> dict:
    """把每个 data 事件归类：role / content / reasoning / tool_call / 空 / finish / usage。"""
    rows, counts = [], dict(content=0, reasoning=0, tool_call=0, role_only=0,
                            empty=0, finish=0, usage=0)
    text, reasoning, tool_args = "", "", ""
    tool_names = []
    for i, ev in enumerate(events):
        if ev["kind"] != "data":
            continue
        obj = ev["obj"]
        usage = obj.get("usage")
        choices = obj.get("choices") or []
        if not choices:
            if usage:
                counts["usage"] += 1
                rows.append(dict(i=i, kind="usage", usage=usage))
            continue
        ch = choices[0]
        d = ch.get("delta") or {}
        kinds = []
        # vLLM 0.29.0 的字段名是 reasoning；部分客户端按 reasoning_content 读
        reasoning_delta = d.get("reasoning_content") or d.get("reasoning")
        if d.get("role") and not any([d.get("content"), reasoning_delta, d.get("tool_calls")]):
            kinds.append("role_only")
        if reasoning_delta:
            kinds.append("reasoning")
            reasoning += reasoning_delta
        if d.get("content"):
            kinds.append("content")
            text += d["content"]
        if d.get("tool_calls"):
            kinds.append("tool_call")
            for tc in d["tool_calls"]:
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_names.append(fn["name"])
                if fn.get("arguments"):
                    tool_args += fn["arguments"]
        if ch.get("finish_reason"):
            kinds.append("finish")
        if not kinds:
            kinds.append("empty")
        for k in kinds:
            counts[k] = counts.get(k, 0) + 1
        rows.append(dict(i=i, kinds=kinds, delta=d, finish_reason=ch.get("finish_reason")))
    return dict(counts=counts, rows=rows, text=text, reasoning=reasoning,
                tool_names=tool_names, tool_arguments=tool_args)


def run_case(base_url: str, out: Path, label: str, body: dict) -> dict:
    raw_path = out / f"{label}.sse.bin"
    raw, recvs = post_stream(base_url, body, raw_path)
    head, payload = split_head(raw)
    events, n_http_chunks, comments = parse_events(payload)
    info = classify(events)
    usage = next((r["usage"] for r in info["rows"] if r.get("kind") == "usage"), None)
    n_tokens = usage["completion_tokens"] if usage else None
    n_data = sum(1 for e in events if e["kind"] == "data")
    n_text = info["counts"]["content"] + info["counts"]["reasoning"] + info["counts"]["tool_call"]

    print(f"\n[{label}]")
    print(f"    HTTP 响应头 {len(head)} 字节；body {len(payload)} 字节；"
          f"socket 收到 {len(recvs)} 次")
    print(f"    HTTP chunk 数 {n_http_chunks}   SSE event 数 {len(events)}"
          f"（其中 data {n_data}，[DONE] 1）   keep-alive 注释 {comments}")
    print(f"    completion_tokens {n_tokens}   带内容的 delta {n_text}   "
          f"空 delta {info['counts']['empty']}   role-only {info['counts']['role_only']}")
    print(f"    分类计数 {info['counts']}")
    if info["reasoning"]:
        first_content = next((r["i"] for r in info["rows"]
                              if "content" in r.get("kinds", [])), None)
        print(f"    reasoning 段 {len(info['reasoning'])} 字符，"
              f"{info['counts']['reasoning']} 个 delta：{info['reasoning'][:36]!r}…")
        print(f"    第一个 content delta 出现在第 {first_content} 个 data 事件"
              f"（此前全部是 reasoning）")
    if info["text"]:
        print(f"    content {len(info['text'])} 字符：{info['text'][:40]!r}…")
    if info["tool_names"]:
        print(f"    tool_calls 函数名 {info['tool_names']}  "
              f"arguments 拼接结果 {info['tool_arguments']!r}")
    print(f"    前 6 个 data 事件的 delta：")
    for r in info["rows"][:6]:
        if r.get("kind") == "usage":
            continue
        print(f"      #{r['i']:<3d} {','.join(r['kinds']):<22s}{json.dumps(r['delta'], ensure_ascii=False)[:70]}")
    return dict(label=label, request=body, n_http_chunks=n_http_chunks,
                n_sse_events=len(events), n_data_events=n_data, n_recv=len(recvs),
                completion_tokens=n_tokens, usage=usage, comments=comments,
                counts=info["counts"], text=info["text"], reasoning=info["reasoning"],
                tool_names=info["tool_names"], tool_arguments=info["tool_arguments"],
                rows=info["rows"], header=head.decode(errors="replace"),
                raw_bytes=len(raw), recv_times=[t for t, _ in recvs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8021")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    common = dict(model=args.model, stream=True, temperature=0.0, max_tokens=160,
                  stream_options={"include_usage": True})
    cases = [
        ("plain", dict(common, messages=[{"role": "user", "content": "用一句话说明风是什么。"}],
                       chat_template_kwargs={"enable_thinking": False})),
        ("reasoning", dict(common, max_tokens=600,
                           messages=[{"role": "user", "content": "13 和 17 哪个更大？直接给结论。"}],
                           chat_template_kwargs={"enable_thinking": True})),
        ("tool-call", dict(common, messages=[{"role": "user", "content": "北京现在天气怎么样？"}],
                           tools=TOOLS, tool_choice="auto",
                           chat_template_kwargs={"enable_thinking": False})),
    ]
    print(f"base_url={args.base_url}  model={args.model}\n{SEP}")
    results = [run_case(args.base_url, out, label, body) for label, body in cases]

    print(f"\n{SEP}\n三种计数的对照（同一次请求，三条不同的线）")
    print(f"    {'请求':<12s}{'token':>7s}{'文本 delta':>11s}{'SSE event':>11s}"
          f"{'HTTP chunk':>12s}{'socket 次数':>12s}")
    for r in results:
        print(f"    {r['label']:<12s}{str(r['completion_tokens']):>7s}"
              f"{r['counts']['content'] + r['counts']['reasoning'] + r['counts']['tool_call']:>11d}"
              f"{r['n_sse_events']:>11d}{r['n_http_chunks']:>12d}{r['n_recv']:>12d}")
    print("    三列不相等是常态：token 是模型产出的单位，delta 是文本单位，"
          "event 还要加上 role、finish_reason、usage 这些没有文本的事件。")

    (out / "stream_events.json").write_text(
        json.dumps(dict(model=args.model, base_url=args.base_url, cases=results),
                   ensure_ascii=False, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.5-C", model=args.model, base_url=args.base_url,
        decoding="greedy (temperature=0)", max_tokens=160,
        parsers="--reasoning-parser qwen3 --tool-call-parser hermes",
        host=platform.node(),
        outputs=["stream_events.json", "*.sse.bin", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
