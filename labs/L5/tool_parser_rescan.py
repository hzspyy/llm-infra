#!/usr/bin/env python3
"""L5.6 补测 —— tool-call parser 在超长 arguments 上的重扫代价。

`extract_tool_calls_streaming` 每收到一个 token 就要在**累计文本**上重新定位
`<tool_call>` 区间并尝试解析 JSON：参数越长，单步的重扫成本越高，而这类
"每步都从头扫一遍"的实现给出的端到端代价是二次的。

本脚本不跑模型：把同一段 arguments 文本按 token 切片，模拟流式喂入，
量出

  * 每个 step 的 parser 调用耗时（中位/p99/max）；
  * 总耗时随参数长度（1/4/16/64/256 KB）的增长曲线；
  * 与"每步只处理增量"的理想实现相比多花多少；
  * 解析失败的步数（流式过程中必然出现的 None 步）。

用法（任意机器，CPU 即可）：
    python tool_parser_rescan.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

MODEL = os.environ.get("L56_MODEL", "Qwen/Qwen3-1.7B")
SIZES_KB = (1, 4, 16, 64, 256)


def make_parser():
    from transformers import AutoTokenizer
    from vllm.tool_parsers import ToolParserManager
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cls = ToolParserManager.get_tool_parser("hermes")
    return cls(tok), tok


def run_size(parser, tok, payload_kb):
    """把一条完整 tool_call 文本按 token 切片喂进 streaming parser。"""
    args = {"city": "Shenzhen", "note": "x" * (payload_kb * 1024)}
    body = json.dumps({"name": "get_weather", "arguments": args},
                      ensure_ascii=False)
    text = f"<tool_call>\n{body}\n</tool_call>"
    ids = tok.encode(text, add_special_tokens=False)
    prev = ""
    step_times, none_steps, deltas = [], 0, 0
    for i, tid in enumerate(ids):
        cur = tok.decode(ids[:i + 1], skip_special_tokens=False)
        t0 = time.perf_counter()
        out = parser.extract_tool_calls_streaming(
            previous_text=prev, current_text=cur, delta_text=tok.decode([tid]),
            previous_token_ids=ids[:i], current_token_ids=ids[:i + 1],
            delta_token_ids=[tid], request=None)
        step_times.append((time.perf_counter() - t0) * 1000)
        prev = cur
        if out is None:
            none_steps += 1
        else:
            deltas += 1
    # 一次性解析的参照
    t0 = time.perf_counter()
    whole = parser.extract_tool_calls(text, request=None)
    whole_ms = (time.perf_counter() - t0) * 1000
    return dict(payload_kb=payload_kb, n_tokens=len(ids),
                total_ms=sum(step_times),
                step_median_ms=statistics.median(step_times),
                step_p99_ms=sorted(step_times)[int(0.99 * (len(step_times) - 1))],
                step_max_ms=max(step_times),
                none_steps=none_steps, delta_steps=deltas,
                whole_ms=whole_ms,
                # 增量理想实现的下界：只把新增片段喂进去的解析成本
                incremental_ms=whole_ms,
                quadratic_ratio=sum(step_times) / whole_ms if whole_ms else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parser, tok = make_parser()
    rows = [run_size(parser, tok, kb) for kb in SIZES_KB]
    lines = [f"tool-call parser 重扫代价 · {MODEL} · "
             f"parser={type(parser).__name__}", ""]
    lines.append(f"  {'参数体量':>8}{'token 数':>9}{'总耗时 ms':>11}"
                 f"{'单步中位 ms':>12}{'单步 p99 ms':>12}{'单步 max ms':>12}"
                 f"{'None 步':>8}{'一次性 ms':>10}{'总/一次性':>10}")
    for r in rows:
        lines.append(f"  {r['payload_kb']:>6} KB{r['n_tokens']:>9}{r['total_ms']:>11.1f}"
                     f"{r['step_median_ms']:>12.3f}{r['step_p99_ms']:>12.3f}"
                     f"{r['step_max_ms']:>12.3f}{r['none_steps']:>8}"
                     f"{r['whole_ms']:>10.3f}{r['quadratic_ratio']:>10.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "tool_parser_rescan.json"), "w") as f:
        json.dump(dict(model=MODEL, parser=type(parser).__name__, rows=rows),
                  f, indent=1)
    with open(os.path.join(args.out, "tool_parser_rescan.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
