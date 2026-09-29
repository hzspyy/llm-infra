#!/usr/bin/env python3
"""L5.6 任务 E —— tool-call 的流式解析实测。

语法只保证形状；把形状还原成 OpenAI 的 `tool_calls` 是 parser 的事，而流式接口
最难：`extract_tool_calls_streaming`（`vllm/tool_parsers/abstract_tool_parser.py:201`）
拿到的是「累计文本 + 这一步的增量」，要吐出「这一步该发给客户端的 delta」。

本脚本用一次真实生成把这条路径跑通，并给出三件事：

  1. 真实模型的原始输出（含 `<tool_call>` / `</tool_call>` 两个特殊 token）；
  2. 逐 token 喂给 Hermes parser，记录每一步的 delta（哪一步出现参数片段、
     哪一步出现 `finish_reason` 边界），以及哪些步返回 None；
  3. 对拍：把流式 delta 拼接起来，必须与一次性 `extract_tool_calls` 的结果相同；
     两者的 `arguments` 再各自 `json.loads`，与工具 schema 对齐。

用法（crater，serve venv）：
    python labs/L5/tool_call_stream_probe.py --out "$OUT/tool-stream"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L56_MODEL", "Qwen/Qwen3-1.7B")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Use the tools when needed."},
    {"role": "user", "content": "What's the weather in Shenzhen right now?"},
]


def safe_util(reserve_gib=6.0, cap=0.55, floor=0.22):
    import torch
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return max(min(cap, (free / gib - reserve_gib) / (total / gib)), floor)


def build_prompt(tokenizer):
    return tokenizer.apply_chat_template(MESSAGES, tools=TOOLS,
                                         add_generation_prompt=True,
                                         tokenize=False)


def delta_to_dict(dm):
    if dm is None:
        return None
    out = {}
    if getattr(dm, "content", None):
        out["content"] = dm.content
    tcs = getattr(dm, "tool_calls", None)
    if tcs:
        out["tool_calls"] = []
        for tc in tcs:
            item = {"index": tc.index, "id": tc.id, "type": tc.type}
            fn = getattr(tc, "function", None)
            if fn is not None:
                item["function"] = {"name": getattr(fn, "name", None),
                                    "arguments": getattr(fn, "arguments", None)}
            out["tool_calls"].append(item)
    if getattr(dm, "role", None):
        out["role"] = dm.role
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompt = build_prompt(tokenizer)
    llm = LLM(model=MODEL, gpu_memory_utilization=safe_util(), max_model_len=4096,
              enforce_eager=True, enable_prefix_caching=False, disable_log_stats=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    out = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
    raw_text, token_ids = out.text, list(out.token_ids)

    request = ChatCompletionRequest(model=MODEL, messages=MESSAGES, tools=TOOLS)
    parser = Hermes2ProToolParser(tokenizer, tools=None)

    # ---- 一次性解析 ----
    whole = parser.extract_tool_calls(raw_text, request)
    whole_dict = {
        "content": whole.content,
        "tools_called": whole.tools_called,
        "tool_calls": [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (whole.tool_calls or [])
        ],
    }

    # ---- 逐 token 流式解析 ----
    parser_again = Hermes2ProToolParser(tokenizer, tools=None)
    steps, prev_text, parts = [], "", {}
    for i, tid in enumerate(token_ids):
        cur_text = tokenizer.decode(token_ids[:i + 1], skip_special_tokens=False)
        delta_text = cur_text[len(prev_text):]
        delta = parser_again.extract_tool_calls_streaming(
            previous_text=prev_text, current_text=cur_text, delta_text=delta_text,
            previous_token_ids=token_ids[:i], current_token_ids=token_ids[:i + 1],
            delta_token_ids=[tid], request=request)
        d = delta_to_dict(delta)
        steps.append(dict(step=i, token_id=tid,
                          delta_text=delta_text,
                          raw_visible_text=tokenizer.decode([tid], skip_special_tokens=False),
                          delta=d))
        if d and d.get("tool_calls"):
            for tc in d["tool_calls"]:
                fn = tc.get("function") or {}
                if fn.get("arguments"):
                    parts.setdefault(tc["index"], []).append(fn["arguments"])
        prev_text = cur_text

    streamed_args = {i: "".join(v) for i, v in parts.items()}
    whole_args = {i: (whole_dict["tool_calls"][i]["arguments"] if i < len(whole_dict["tool_calls"]) else None)
                  for i in range(max(len(whole_dict["tool_calls"]), len(streamed_args) or 1))}
    reassembly_ok = all(
        json.loads(streamed_args[i]) == json.loads(whole_args[i])
        for i in range(len(whole_dict["tool_calls"]))
    ) if whole_dict["tool_calls"] and streamed_args else False
    valid_json = {}
    for i, a in streamed_args.items():
        try:
            valid_json[i] = json.loads(a)
        except Exception as exc:                                 # noqa: BLE001
            valid_json[i] = f"<{type(exc).__name__}: {exc}>"

    n_delta = sum(1 for s in steps if s["delta"] is not None)
    n_tool_delta = sum(1 for s in steps
                       if s["delta"] and s["delta"].get("tool_calls"))
    report = dict(
        model=MODEL,
        raw_text=raw_text,
        generated_tokens=len(token_ids),
        nonstream=whole_dict,
        stream_steps=steps,
        n_steps_with_delta=n_delta,
        n_steps_with_tool_delta=n_tool_delta,
        streamed_arguments=streamed_args,
        streamed_arguments_parsed=valid_json,
        reassembly_matches_nonstream=reassembly_ok,
        finish_reason=out.finish_reason,
        stop_reason=getattr(out, "stop_reason", None),
    )
    (args.out / "tool_stream.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型输出 {len(token_ids)} 个 token，finish_reason={out.finish_reason}")
    print(f"原始文本：{raw_text!r}")
    print(f"\n一次性解析：tools_called={whole_dict['tools_called']}，"
          f"tool_calls={whole_dict['tool_calls']}")
    print(f"\n逐 token 流式：{len(steps)} 步，其中有 delta 的 {n_delta} 步，"
          f"带 tool_calls 的 {n_tool_delta} 步")
    print(f"  {'step':>5} {'token':>8} {'delta_text':>14}  增量")
    for s in steps:
        if s["delta"] is None and s["step"] % 7:
            continue
        d = json.dumps(s["delta"], ensure_ascii=False) if s["delta"] else "None"
        print(f"  {s['step']:>5} {s['token_id']:>8} {s['delta_text']!r:>14}  {d[:90]}")
    print(f"\n流式拼接的 arguments：{streamed_args}")
    print(f"解析成 JSON：{valid_json}")
    print(f"与一次性解析一致：{reassembly_ok}")
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
