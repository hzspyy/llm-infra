#!/usr/bin/env python3
"""L9.2 任务 A/C：工具选择的约束路径，以及内容类型 × 约束/草稿的成本与质量。

两个模式：

``choices``（HTTP，服务端需 `--enable-auto-tool-choice --tool-call-parser hermes`）
    对 ``tool_choice`` ∈ {none, auto, required, named} × 工具数 ∈ {1,4,16,64} 逐格发请求，
    记录模板规模（prompt token）、是否真的产生 tool_calls、流式参数片段数、拼接后的参数是否
    是合法 JSON、选中的工具是否正确、以及完整时间。**语法有效、参数有效、工具选择正确、
    任务成功四件事分开记。**

``content``（进程内 LLM，可选 DFlash 草稿）
    四种内容类型——结构固定 / 参数自由 / 长数字 / 长文本——分别在「不约束」「约束解码」
    与「普通解码」「DFlash 投机」下生成工具参数，比较接受长度、无效输出、完整时间与
    参数正确率。参数正确率按与 prompt 中原始串的逐字符比较计，不看模板可预测性。

用法::

    python labs/L9/tool_pipeline.py choices --base-url http://127.0.0.1:8012/v1 --out "$OUT/choices"
    python labs/L9/tool_pipeline.py content --out "$OUT/content" [--draft-model z-lab/Qwen3-4B-DFlash-b16]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# 进程内 LLM 才能拿到 scheduler 的投机统计（同 5.5 的做法）
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = os.environ.get("L92_MODEL", "Qwen/Qwen3-4B")


# --------------------------------------------------------------------------------------
# 工具集生成
# --------------------------------------------------------------------------------------

def make_tools(n: int) -> list[dict]:
    """生成 n 个结构相似的工具，第一个固定为 calculate，便于 named tool choice。"""
    base = [
        ("calculate", "计算一个算术表达式的精确整数值"),
        ("search_corpus", "在医学文献语料上检索"),
        ("read_file", "读取仓库内一个文件"),
        ("run_tests", "运行仓库的单元测试"),
    ]
    tools = []
    for i in range(n):
        if i < len(base):
            name, desc = base[i]
        else:
            name, desc = f"lookup_{i:02d}", f"查询第 {i} 号知识表（占位工具）"
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "查询内容"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        })
    return tools


CHOICE_PROMPT = "请调用工具查询：珠穆朗玛峰的海拔是多少？只调用一次工具。"


def tool_choice_variants():
    return [
        ("none", "none"),
        ("auto", "auto"),
        ("required", "required"),
        ("named", {"type": "function", "function": {"name": "calculate"}}),
    ]


async def run_choices(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    rows = []
    for label, choice in tool_choice_variants():
        for n_tools in (1, 4, 16, 64):
            tools = make_tools(n_tools)
            for rep in range(args.repeats):
                row = {
                    "tool_choice": label,
                    "n_tools": n_tools,
                    "repeat": rep,
                    "thinking": getattr(args, "thinking", False),
                }
                t0 = time.perf_counter()
                ttft = None
                deltas = 0
                arg_fragments: list[str] = []
                tool_names: list[str] = []
                finish_reason = None
                content_parts: list[str] = []
                usage = None
                error = None
                try:
                    stream = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": CHOICE_PROMPT}],
                        tools=tools,
                        tool_choice=choice,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is None:
                            continue
                        if delta.content:
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000.0
                            content_parts.append(delta.content)
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                        for tc in delta.tool_calls or []:
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000.0
                            deltas += 1
                            fn = getattr(tc, "function", None)
                            if fn is not None and fn.name:
                                tool_names.append(fn.name)
                            if fn is not None and fn.arguments:
                                arg_fragments.append(fn.arguments)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                e2e = (time.perf_counter() - t0) * 1000.0
                joined = "".join(arg_fragments)
                parsed, parse_error = None, None
                if joined:
                    try:
                        parsed = json.loads(joined)
                    except json.JSONDecodeError as exc:
                        parse_error = exc.msg
                cdetails = getattr(usage, "prompt_tokens_details", None) if usage else None
                row.update({
                    "supported": error is None,
                    "error": error,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "cached_tokens": getattr(cdetails, "cached_tokens", None) if cdetails else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "ttft_ms": round(ttft, 3) if ttft else None,
                    "e2e_ms": round(e2e, 3),
                    "arg_deltas": deltas,
                    "args_join_len": len(joined),
                    "args_valid_json": (parsed is not None) if joined else None,
                    "args_parse_error": parse_error,
                    "emitted_tool_call": deltas > 0,
                    "finish_reason": finish_reason,
                    "selected_tool": tool_names[0] if tool_names else None,
                    "selected_tool_known": (tool_names[0] in {t["function"]["name"] for t in tools})
                    if tool_names else None,
                    # named 模式下「工具选择正确」有确定答案；auto/required 下只判是否落在已声明集合内
                    "tool_selection_correct": (tool_names[0] == "calculate")
                    if (tool_names and label == "named") else
                    ((tool_names[0] in {t["function"]["name"] for t in tools}) if tool_names else False),
                    "content_len": sum(len(c) for c in content_parts),
                })
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    await client.close()

    summary = {"config": {"base_url": args.base_url, "model": args.model,
                          "repeats": args.repeats, "max_tokens": args.max_tokens},
               "rows": rows}
    (out / "choices.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    agg = {}
    for r in rows:
        key = f"{r['tool_choice']}|{r['n_tools']}"
        a = agg.setdefault(key, {"supported": 0, "runs": 0, "emitted": 0, "valid_args": 0,
                                 "selected": 0, "prompt_tokens": [], "e2e_ms": [], "deltas": []})
        a["runs"] += 1
        a["supported"] += int(r["supported"])
        a["emitted"] += int(r["emitted_tool_call"])
        a["valid_args"] += int(bool(r["args_valid_json"]))
        a["selected"] += int(bool(r["tool_selection_correct"]))
        if r["prompt_tokens"]:
            a["prompt_tokens"].append(r["prompt_tokens"])
        a["e2e_ms"].append(r["e2e_ms"])
        a["deltas"].append(r["arg_deltas"])
    table = {k: {"runs": v["runs"], "supported": v["supported"], "emitted_tool_call": v["emitted"],
                 "valid_arguments": v["valid_args"], "tool_selection_correct": v["selected"],
                 "prompt_tokens_median": statistics.median(v["prompt_tokens"]) if v["prompt_tokens"] else None,
                 "e2e_ms_median": round(statistics.median(v["e2e_ms"]), 2),
                 "arg_deltas_median": statistics.median(v["deltas"])}
             for k, v in sorted(agg.items())}
    (out / "choices_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# 内容类型 × 约束/草稿
# --------------------------------------------------------------------------------------

LONG_PASSAGE = (
    "The cardiovascular benefits of statins are well documented in large randomized trials. "
    "Patients with elevated low-density lipoprotein cholesterol show a consistent reduction in "
    "major adverse cardiac events when treated with moderate-intensity statin therapy. "
    "However, the absolute risk reduction depends strongly on baseline risk, and the number "
    "needed to treat varies from twenty to over one hundred across primary prevention cohorts."
)

CONTENT_CASES = [
    {
        "name": "fixed_structure",
        "prompt": "把这次运算的参数抽成 JSON（字段 expression 与 precision）：(12+7)*3，precision=8。只输出 JSON。",
        "schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}, "precision": {"type": "integer"}},
            "required": ["expression", "precision"],
            "additionalProperties": False,
        },
        "check": lambda obj: isinstance(obj, dict) and obj.get("precision") == 8
        and isinstance(obj.get("expression"), str),
        "expect": "结构固定：字段名与取值都可枚举",
    },
    {
        "name": "free_arguments",
        "prompt": "用一句你自己的话描述检索意图，输出 JSON，字段 query（字符串，20 字以内）。",
        "schema": {"type": "object", "properties": {"query": {"type": "string", "maxLength": 40}},
                   "required": ["query"], "additionalProperties": False},
        "check": lambda obj: isinstance(obj, dict) and isinstance(obj.get("query"), str) and 0 < len(obj["query"]) <= 40,
        "expect": "参数自由：内容由模型决定",
    },
    {
        "name": "long_number",
        "prompt": ("把下面这个整数原样抄进 JSON 的 value 字段，不要改写、不要加分隔符：\n"
                   "3184729056138745120963451287904512367890\n"
                   "只输出 JSON：{\"value\": <整数>}"),
        "schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                   "required": ["value"], "additionalProperties": False},
        "check": lambda obj: isinstance(obj, dict) and str(obj.get("value")) == "3184729056138745120963451287904512367890",
        "expect": "长数字：高熵 token，草稿很难猜中",
    },
    {
        "name": "long_text",
        "prompt": ("把下面这段文字原样抄进 JSON 的 text 字段（一个字符都不要改）：\n"
                   f"{LONG_PASSAGE}\n只输出 JSON。"),
        "schema": {"type": "object", "properties": {"text": {"type": "string"}},
                   "required": ["text"], "additionalProperties": False},
        "check": lambda obj: isinstance(obj, dict)
        and obj.get("text", "").strip() == LONG_PASSAGE.strip(),
        "expect": "长文本：与 prompt 高度重叠，草稿命中率应当高",
    },
]


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """从模型输出里取第一个完整 JSON 对象（允许前后有解释文字）。"""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1]), None
                    except json.JSONDecodeError as exc:
                        return None, exc.msg
        start = text.find("{", start + 1)
    return None, "no json object"


_ACC = {}


def hook_spec_stats(llm):
    """包装 scheduler.make_spec_decoding_stats，累计草稿与接受数（同 5.5 的手法）。"""
    _ACC.clear()
    _ACC.update(draft_tokens=0, accepted=0)
    core = llm.llm_engine.engine_core
    sched = getattr(core, "engine_core", core).scheduler
    orig = sched.make_spec_decoding_stats

    def wrapped(*a, **kw):
        st = orig(*a, **kw)
        if st is not None:
            _ACC["draft_tokens"] += getattr(st, "num_draft_tokens", 0) or 0
            acc = getattr(st, "num_accepted_tokens", None)
            if acc is None:
                pos = getattr(st, "num_accepted_tokens_per_pos", None)
                acc = sum(pos) if pos else 0
            _ACC["accepted"] += acc
        return st

    sched.make_spec_decoding_stats = wrapped


def run_content(args) -> int:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def safe_util(reserve_gib=4.0, cap=0.5):
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))

    variants = [("free", "none"), ("constrained", "json")]
    rows = []
    for draft_label, spec in (("no_draft", None), ("dflash", args.spec)):
        if spec is None and draft_label == "dflash":
            continue
        llm_kwargs = dict(model=args.model, gpu_memory_utilization=safe_util(),
                          max_model_len=8192, enforce_eager=True,
                          enable_prefix_caching=False, disable_log_stats=(spec is None))
        if spec:
            llm_kwargs["speculative_config"] = spec
        llm = LLM(**llm_kwargs)
        if spec:
            hook_spec_stats(llm)
        try:
            for content in CONTENT_CASES:
                for vlabel, mode in variants:
                    for rep_i in range(args.repeats):
                          sp_kwargs = dict(temperature=0.0, max_tokens=args.max_tokens)
                          if mode == "json":
                              # 0.29 的 structured_outputs 需要 StructuredOutputsParams 对象，
                              # 传 dict 会在 input_processor 校验时报 'dict' has no attribute '_backend'
                              sp_kwargs["structured_outputs"] = StructuredOutputsParams(json=content["schema"])
                          sp = SamplingParams(**sp_kwargs)
                          before = dict(_ACC) if spec else None
                          t0 = time.perf_counter()
                          outs = llm.chat([[{"role": "user", "content": content["prompt"]}]], sp,
                                          use_tqdm=False,
                                          chat_template_kwargs={"enable_thinking": args.thinking})
                          wall_ms = (time.perf_counter() - t0) * 1000.0
                          text = outs[0].outputs[0].text
                          ntok = len(outs[0].outputs[0].token_ids)
                          obj, err = extract_json(text)
                          ok = bool(content["check"](obj)) if obj is not None else False
                          delta_acc = None
                          if spec:
                              delta_acc = {k: _ACC[k] - before[k] for k in _ACC}
                          rows.append({
                              "repeat": rep_i,
                            "draft": draft_label,
                              "content": content["name"],
                              "constraint": vlabel,
                              "structured_outputs": mode,
                              "output_tokens": ntok,
                              "wall_ms": round(wall_ms, 3),
                              "ms_per_token": round(wall_ms / max(1, ntok), 3),
                              "json_parsed": obj is not None,
                              "json_error": err,
                              "argument_correct": ok,
                              "expect": content["expect"],
                              "spec_stats": delta_acc,
                              "raw_head": text[:160],
                          })
                          print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        finally:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:  # noqa: BLE001
                pass
            del llm
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    summary = {
        "config": {"model": args.model, "spec": args.spec, "max_tokens": args.max_tokens},
        "rows": rows,
        "aggregate": aggregate_content(rows),
    }
    (out / "content.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=1))
    return 0



# --------------------------------------------------------------------------------------
# 工具预算：schema 未命中/命中前缀 + 完整工具集 vs 按需发现
# --------------------------------------------------------------------------------------

SEARCH_TOOLS_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_tools",
        "description": "按关键词检索可用的工具，返回匹配工具的调用方式",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "检索关键词"}},
                       "required": ["query"], "additionalProperties": False},
    },
}]

BUDGET_PROMPT = "请调用工具查询第 37 号知识表里的内容。只调用一次工具。"


def _usage_fields(usage) -> tuple:
    if usage is None:
        return None, None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return usage.prompt_tokens, cached


async def _one_call(client, model, messages, tools, choice, max_tokens, timeout):
    """一次非流式调用：返回 (selected_tool, args, prompt_tokens, cached_tokens, ttft_ms, e2e_ms, error)。"""
    t0 = time.perf_counter()
    ttft = None
    selected, args_raw = None, ""
    err = None
    prompt_tokens = cached = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=choice,
            temperature=0.0, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens, cached = _usage_fields(chunk.usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            for tc in delta.tool_calls or []:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                fn = getattr(tc, "function", None)
                if fn is not None and fn.name:
                    selected = fn.name
                if fn is not None and fn.arguments:
                    args_raw += fn.arguments
            if delta.content and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    e2e = (time.perf_counter() - t0) * 1000.0
    return selected, args_raw, prompt_tokens, cached, ttft, e2e, err


async def run_budget(args) -> int:
    """两条测量：schema 的未命中/命中前缀成本；完整工具集 vs 按需发现。"""
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    report: dict = {"config": {"base_url": args.base_url, "model": args.model,
                               "counts": args.counts, "max_tokens": args.max_tokens},
                    "schema_cost": [], "discovery": []}

    # --- 1) schema 的未命中 vs 命中前缀：同一请求连发两次，第二次应命中 ---
    for n in [int(x) for x in args.counts.split(",")]:
        tools = make_tools(n)
        # 同一请求连发 repeats 次：第一次的前缀必然未命中，后续命中；命中状态按实测
        # cached_tokens 标注，而不是按调用次序假设。
        for rep in range(args.repeats):
            sel, raw, pt, cached, ttft, e2e, err = await _one_call(
                client, args.model, [{"role": "user", "content": CHOICE_PROMPT}],
                tools, "auto", args.max_tokens, args.timeout)
            report["schema_cost"].append({
                "n_tools": n, "repeat": rep,
                "cache_state": ("miss" if (cached or 0) == 0 else "hit"),
                "prompt_tokens": pt, "cached_tokens": cached,
                "uncached_tokens": (None if pt is None or cached is None else pt - cached),
                "ttft_ms": round(ttft, 3) if ttft else None,
                "e2e_ms": round(e2e, 3),
                "selected_tool": sel, "error": err,
            })
            print(f"[schema] n={n} rep={rep} cache={report['schema_cost'][-1]['cache_state']} "
                  f"prompt={pt} cached={cached} ttft={report['schema_cost'][-1]['ttft_ms']}", flush=True)

    # --- 2) 完整工具集 vs 按需发现（目标工具在 64 个工具里排第 38 位） ---
    for rep in range(args.repeats):
        # 完整集：一次请求带全部 64 个工具
        tools_full = make_tools(64)
        sel, raw, pt, cached, ttft, e2e, err = await _one_call(
            client, args.model, [{"role": "user", "content": BUDGET_PROMPT}],
            tools_full, "auto", args.max_tokens, args.timeout)
        report["discovery"].append({
            "arm": "full_toolset", "repeat": rep, "round_trips": 1,
            "n_tools_declared": len(tools_full),
            "total_prompt_tokens": pt, "total_cached_tokens": cached,
            "total_ttft_ms": round(ttft, 3) if ttft else None, "total_e2e_ms": round(e2e, 3),
            "final_selected_tool": sel, "final_ok": sel == "lookup_37", "error": err,
        })
        print(f"[discovery] full rep={rep} sel={sel} prompt={pt} e2e={report['discovery'][-1]['total_e2e_ms']}", flush=True)

        # 按需发现：第一轮只给 search_tools，客户端拿到结果后第二轮带上匹配工具
        messages = [{"role": "user", "content": BUDGET_PROMPT}]
        sel1, raw1, pt1, cached1, ttft1, e2e1, err1 = await _one_call(
            client, args.model, messages, SEARCH_TOOLS_TOOL, "auto", args.max_tokens, args.timeout)
        matched = [t for t in make_tools(64) if t["function"]["name"] == "lookup_37"]
        messages2 = messages + [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": sel1 or "search_tools",
                                          "arguments": raw1 or json.dumps({"query": "第 37 号知识表"})}}]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": json.dumps(matched, ensure_ascii=False)},
        ]
        sel2, raw2, pt2, cached2, ttft2, e2e2, err2 = await _one_call(
            client, args.model, messages2, SEARCH_TOOLS_TOOL + matched, "auto",
            args.max_tokens, args.timeout)
        report["discovery"].append({
            "arm": "on_demand", "repeat": rep, "round_trips": 2,
            "n_tools_declared": [len(SEARCH_TOOLS_TOOL), len(SEARCH_TOOLS_TOOL) + len(matched)],
            "round1_selected": sel1,
            "total_prompt_tokens": (pt1 or 0) + (pt2 or 0) if pt1 is not None and pt2 is not None else None,
            "total_cached_tokens": (cached1 or 0) + (cached2 or 0) if cached1 is not None and cached2 is not None else None,
            "total_ttft_ms": round((ttft1 or 0.0) + (ttft2 or 0.0), 3),
            "total_e2e_ms": round(e2e1 + e2e2, 3),
            "final_selected_tool": sel2, "final_ok": sel2 == "lookup_37",
            "error": err1 or err2,
        })
        print(f"[discovery] on_demand rep={rep} r1={sel1} r2={sel2} prompt={report['discovery'][-1]['total_prompt_tokens']} e2e={report['discovery'][-1]['total_e2e_ms']}", flush=True)

    await client.close()
    (out / "tool_budget.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    for n in [int(x) for x in args.counts.split(",")]:
        rows = [r for r in report["schema_cost"] if r["n_tools"] == n]
        pt = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
        print(f"[summary] n_tools={n} prompt_tokens={pt}")
    return 0


def aggregate_content(rows: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for r in rows:
        key = f"{r['draft']}|{r['content']}|{r['constraint']}"
        a = agg.setdefault(key, {"runs": 0, "correct": 0, "valid_json": 0, "tokens": [],
                                 "wall_ms": [], "accepted": 0, "draft_tokens": 0})
        a["runs"] += 1
        a["correct"] += int(r["argument_correct"])
        a["valid_json"] += int(r["json_parsed"])
        a["tokens"].append(r["output_tokens"])
        a["wall_ms"].append(r["wall_ms"])
        if r["spec_stats"]:
            a["accepted"] += r["spec_stats"].get("accepted", 0)
            a["draft_tokens"] += r["spec_stats"].get("draft_tokens", 0)
    return {
        k: {
            "runs": v["runs"],
            "argument_correct": v["correct"],
            "valid_json": v["valid_json"],
            "output_tokens_mean": round(statistics.fmean(v["tokens"]), 1) if v["tokens"] else None,
            "wall_ms_mean": round(statistics.fmean(v["wall_ms"]), 2) if v["wall_ms"] else None,
            "acceptance": round(v["accepted"] / v["draft_tokens"], 4) if v["draft_tokens"] else None,
            "draft_tokens": v["draft_tokens"],
        }
        for k, v in sorted(agg.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.2 工具调用路径与内容类型对照")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("choices")
    p.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=lambda a: asyncio.run(run_choices(a)))

    p = sub.add_parser("budget")
    p.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", default="1,4,16,64")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=lambda a: asyncio.run(run_budget(a)))

    p = sub.add_parser("content")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--spec", type=json.loads, default='{"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": 8}')
    p.add_argument("--thinking", action="store_true", help="内容类型对照默认关 thinking，让参数本身成为变量")
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=run_content)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
