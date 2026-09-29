#!/usr/bin/env python3
"""L5.6 补测 · None 步在真实 SSE 里长什么样。

离线探针（`tool_call_stream_probe.py`）已经把逐 token 喂 parser 的结果记下来了：
136 个 token 里有 13 步 parser 返回 None。但那 13 步在**真实流式响应**里到底发了什么？
客户端看到的是空的 delta、还是根本没有帧、还是帧里带了别的东西？

本脚本对着真实服务发一次流式工具调用请求，把**原始 SSE 字节**落盘，然后：

  1. 逐帧解析 `data:` 行，记录每帧的 JSON 形状（choices[0].delta 的键）；
  2. 用同一段累计文本离线跑 `Hermes2ProToolParser.extract_tool_calls_streaming`，
     得到"这一步 parser 返回 None / delta / finish"的序列；
  3. 把两者按 token 对齐，给出 None 步的实际帧形态与字节开销。

用法（先起好 vLLM，见 run_tool_sse.sh）：
    python tool_call_sse_framing.py --base http://127.0.0.1:8160 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

MODEL = "Qwen/Qwen3-1.7B"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string", "enum": ["c", "f"]},
                           "note": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Use the tools when needed."},
    {"role": "user",
     "content": "What's the weather in Shenzhen right now? "
                "Put a detailed 150-word note about what you checked into the "
                "note field of the tool call."},
]


def stream(base, payload, raw_path):
    """发一次流式请求，返回 (帧文本列表, 原始字节)。"""
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    frames, raw = [], b""
    with urllib.request.urlopen(req, timeout=300) as r:
        buf = b""
        while True:
            chunk = r.read(1)
            if not chunk:
                break
            raw += chunk
            buf += chunk
            while b"\n\n" in buf:
                part, buf = buf.split(b"\n\n", 1)
                frames.append(part.decode(errors="replace"))
    with open(raw_path, "wb") as f:
        f.write(raw)
    return frames, raw


def parse_frames(frames):
    out = []
    for fr in frames:
        for line in fr.splitlines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                out.append(dict(kind="done", bytes=len(line)))
                continue
            try:
                obj = json.loads(body)
            except Exception:                                   # noqa: BLE001
                out.append(dict(kind="unparsed", bytes=len(line), body=body[:80]))
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            out.append(dict(kind="data", bytes=len(line),
                            delta_keys=sorted(delta),
                            content=delta.get("content"),
                            tool_calls=delta.get("tool_calls"),
                            finish_reason=ch.get("finish_reason")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8160")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm.tool_parsers import ToolParserManager
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    parser_cls = ToolParserManager.get_tool_parser("hermes")

    payload = dict(model=args.model, messages=MESSAGES, tools=TOOLS,
                   tool_choice="auto", temperature=0.0,
                   max_tokens=args.max_tokens, stream=True)
    t0 = time.perf_counter()
    frames, raw = stream(args.base, payload, os.path.join(args.out, "stream_raw.bin"))
    wall = (time.perf_counter() - t0) * 1000
    parsed = parse_frames(frames)

    # 离线重放：把**全部**流式文本（正文 + 工具参数的原始片段）按 token 喂 parser。
    # 只拼 content 会漏掉参数区间——而 None 步正是出现在那里。
    def all_text(parsed):
        out = []
        for f in parsed:
            if f["kind"] != "data":
                continue
            if f.get("content"):
                out.append(f["content"])
            for tc in (f.get("tool_calls") or []):
                fn = (tc or {}).get("function") or {}
                if fn.get("arguments"):
                    out.append(fn["arguments"])
        return "".join(out)

    text = all_text(parsed)
    ids = tok.encode(text, add_special_tokens=False)
    parser = parser_cls(tok)
    prev, steps = "", []
    for i in range(len(ids)):
        cur = tok.decode(ids[:i + 1], skip_special_tokens=False)
        res = parser.extract_tool_calls_streaming(
            previous_text=prev, current_text=cur,
            delta_text=tok.decode([ids[i]]),
            previous_token_ids=ids[:i], current_token_ids=ids[:i + 1],
            delta_token_ids=[ids[i]], request=None)
        steps.append("none" if res is None else "delta")
        prev = cur

    content_frames = [f for f in parsed if f["kind"] == "data" and f.get("content")]
    empty_frames = [f for f in parsed if f["kind"] == "data"
                    and not f.get("content") and not f.get("tool_calls")]
    none_steps = sum(1 for s in steps if s == "none")
    report = dict(model=args.model, wall_ms=round(wall, 1),
                  n_frames=len(parsed), n_content_frames=len(content_frames),
                  raw_bytes=len(raw), text=text,
                  parser_steps=steps, n_none_steps=none_steps,
                  frame_kinds={},
                  n_empty_delta_frames=len(empty_frames),
                  empty_delta_bytes=[f["bytes"] for f in empty_frames],
                  content_frame_bytes=[f["bytes"] for f in content_frames])
    for f in parsed:
        k = f["kind"] if f["kind"] != "data" else \
            ("content" if f.get("content") else
             ("tool_calls" if f.get("tool_calls") else "empty-delta"))
        report["frame_kinds"][k] = report["frame_kinds"].get(k, 0) + 1
    report["frames"] = parsed

    with open(os.path.join(args.out, "tool_sse_framing.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  帧 {report['n_frames']} 个（{report['raw_bytes']} 字节），"
          f"其中带 content 的 {report['n_content_frames']} 个，"
          f"帧类型 {report['frame_kinds']}")
    print(f"  离线重放（正文 + 参数共 {len(steps)} 个 token）：parser 返回 None {none_steps} 步"
          f"（{none_steps / max(1, len(steps)):.0%}）")
    print(f"  线上空帧（delta 里既无 content 也无 tool_calls）：{len(empty_frames)} 个，"
          f"字节 {report['empty_delta_bytes']}")
    print(f"  每帧字节：content 帧中位 "
          f"{sorted(report['content_frame_bytes'])[len(report['content_frame_bytes']) // 2]}，"
          f"最大 {max(report['content_frame_bytes'])}")
    print(f"\n写入 {args.out}/tool_sse_framing.json 与 stream_raw.bin")


if __name__ == "__main__":
    main()
