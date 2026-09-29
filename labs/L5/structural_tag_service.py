#!/usr/bin/env python3
"""L5.6 补测 · structural tag 接到真实服务后的开销与有效率。

CPU 级探针（`structural_tag_probe.py`）只能说明 mask 的合法性形态；这里把它接到
引擎里：同一批 prompt、同一输出长度，跑三组

  A 无约束（自由文本）
  B structural tag（`<tool_call>` 之外自由、之内严格 JSON schema）
  C 整段 JSON schema 约束（对照：约束从第 0 个 token 就开始）

比较 TTFT、TPOT、输出 token 数、是否真的产出可用工具调用，以及引擎侧的
结构化输出统计。批大小 1/8，各重复 3 次取中位。

用法：
    python structural_tag_service.py --out <dir>
"""

import argparse
import json
import os
import statistics
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L56_MODEL", "Qwen/Qwen3-1.7B")
BEGIN, END = "<tool_call>", "</tool_call>"
ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ["get_weather"]},
        "arguments": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "enum": ["Beijing", "Shenzhen"]}},
            "required": ["city"], "additionalProperties": False,
        },
    },
    "required": ["name", "arguments"], "additionalProperties": False,
}
STRUCT_TAG = json.dumps({
    "type": "structural_tag",
    "structures": [{"begin": BEGIN, "schema": ARGS_SCHEMA, "end": END}],
    "triggers": [BEGIN],
})
# 用带 tools 的 chat 模板：模型才会真的吐出 <tool_call> 触发约束区，
# 否则 structural tag 的"区内严格"根本没有机会生效（约束只作用在标签内）。
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"], "additionalProperties": False,
        },
    },
}]
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Use the tools when needed."},
    {"role": "user", "content": "What's the weather in Shenzhen right now?"},
]


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def build_prompt():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    return tok.encode(text, add_special_tokens=False), text


def make_llm(util, backend=None):
    from vllm import LLM
    kw = dict(model=MODEL, gpu_memory_utilization=util, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False,
              disable_log_stats=False, max_num_batched_tokens=4096)
    return LLM(**kw)


def first_json_object(text):
    """从可能带尾随文本的输出里取出第一个完整的 JSON 对象。"""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def run_case(llm, batch, mode, out_len=128, natural_stop=True):
    from vllm import SamplingParams
    # 约束模式下让模型在语法完成时正常停止（ignore_eos=True 会逼它把
    # 剩下的 token 位填满，输出后半段就不再是合法 JSON，测出来的"有效率"是假的）
    sp_kw = dict(max_tokens=out_len, temperature=0.0,
                 ignore_eos=not (natural_stop and mode != "none"))
    if mode in ("structural_tag", "json_schema"):
        # vLLM 0.29.0 要求传 StructuredOutputsParams 对象（dict 会在 verify 时报
        # 'dict' object has no attribute '_backend'）
        from vllm.sampling_params import StructuredOutputsParams
        sp_kw["structured_outputs"] = (
            StructuredOutputsParams(structural_tag=STRUCT_TAG)
            if mode == "structural_tag"
            else StructuredOutputsParams(json=json.dumps(ARGS_SCHEMA)))
    sp = SamplingParams(**sp_kw)
    from vllm import TokensPrompt
    global PROMPT_IDS
    prompts = [TokensPrompt(prompt_token_ids=list(PROMPT_IDS))] * batch
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    texts = [o.outputs[0].text for o in outs]
    toks = [len(o.outputs[0].token_ids) for o in outs]
    oks = []
    triggers = []
    for t in texts:
        triggers.append(BEGIN in t)
        if mode == "structural_tag":
            if BEGIN in t and END in t:
                body = t.split(BEGIN, 1)[1].split(END, 1)[0]
                try:
                    json.loads(body)
                    oks.append(True)
                except Exception:                               # noqa: BLE001
                    oks.append(False)
            else:
                oks.append(False)
        elif mode == "json_schema":
            obj = first_json_object(t)
            try:
                json.loads(obj) if obj else None
                oks.append(bool(obj))
            except Exception:                                   # noqa: BLE001
                oks.append(False)
        else:
            oks.append(None)
    ttfts = [getattr(o.metrics, "first_token_latency", None) for o in outs]
    ttfts = [x * 1000 for x in ttfts if x]
    tpot = []
    for o in outs:
        m = o.metrics
        if m and m.last_token_ts and m.first_token_ts and len(o.outputs[0].token_ids) > 1:
            tpot.append((m.last_token_ts - m.first_token_ts) * 1000
                        / (len(o.outputs[0].token_ids) - 1))
    return dict(mode=mode, batch=batch, wall_ms=wall,
                trigger_ratio=(sum(1 for x in triggers if x) / len(triggers)
                               if mode == "structural_tag" else None),
                ttft_median_ms=statistics.median(ttfts) if ttfts else None,
                tpot_median_ms=statistics.median(tpot) if tpot else None,
                out_tokens_median=statistics.median(toks),
                out_tps=sum(toks) / (wall / 1000),
                valid_ratio=(sum(1 for x in oks if x) / len(oks)
                             if oks[0] is not None else None),
                sample_text=texts[0][:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--util", type=float, default=None)
    ap.add_argument("--forced", action="store_true",
                    help="约束模式也强制生成满 out_len（比 mask 每步成本，不看自然停止）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    util = args.util or safe_util()
    global PROMPT_IDS
    PROMPT_IDS, prompt_text = build_prompt()
    print(f"prompt tokens {len(PROMPT_IDS)}（chat 模板 + tools）", flush=True)
    llm = make_llm(util)
    rows = []
    for batch in (int(x) for x in args.batches.split(",")):
        for mode in ("none", "structural_tag", "json_schema"):
            for _ in range(1):
                run_case(llm, batch, mode, natural_stop=not args.forced)
            reps = [run_case(llm, batch, mode, natural_stop=not args.forced)
                    for _ in range(args.repeats)]
            agg = dict(mode=mode, batch=batch,
                       wall_median_ms=statistics.median(r["wall_ms"] for r in reps),
                       ttft_median_ms=statistics.median(
                           r["ttft_median_ms"] for r in reps if r["ttft_median_ms"]),
                       tpot_median_ms=statistics.median(
                           r["tpot_median_ms"] for r in reps if r["tpot_median_ms"]),
                       out_tokens_median=statistics.median(
                           r["out_tokens_median"] for r in reps),
                       out_tps_median=statistics.median(r["out_tps"] for r in reps),
                       valid_ratio=reps[0]["valid_ratio"],
                       trigger_ratio=reps[0]["trigger_ratio"],
                       wall_range=[min(r["wall_ms"] for r in reps),
                                   max(r["wall_ms"] for r in reps)],
                       sample_text=reps[0]["sample_text"])
            rows.append(agg)
            print(f"  B={batch:<2} {mode:<15} 墙钟中位 {agg['wall_median_ms']:>8.1f} ms"
                  f"（极差 {agg['wall_range'][0]:.0f}-{agg['wall_range'][1]:.0f}）"
                  f" TTFT {agg['ttft_median_ms']:.1f} ms  TPOT {agg['tpot_median_ms']:.2f} ms"
                  f"  输出 token {agg['out_tokens_median']:.0f}"
                  f"  触发 {agg['trigger_ratio']}  有效 {agg['valid_ratio']}",
                  flush=True)
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    with open(os.path.join(args.out, "structural_tag_service.json"), "w") as f:
        json.dump(dict(model=MODEL, util=util, prompt_tokens=len(PROMPT_IDS),
                       prompt_text=prompt_text, schema=ARGS_SCHEMA,
                       structural_tag=STRUCT_TAG, rows=rows), f, indent=1,
                  ensure_ascii=False)
    print(f"\n写入 {args.out}/structural_tag_service.json")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
