#!/usr/bin/env python3
"""L5.6 补测 · mask apply 在引擎里到底被摊销成了什么。

单独测出来的成本是"每次调用 9–12 µs（Python 侧）/ 1–4 µs（图内设备侧）"，
但那回答不了引擎里的问题：**一步调用几次？随 batch 怎么变？**

源码给出了形状（`vllm/v1/structured_output/utils.py:87 apply_grammar_bitmask`）：
每一步一次，且不是"每条请求一次"——
  * 先按 batch 顺序重排 bitmask（`torch.full((rows, width), -1, pin_memory=True)`
    再逐请求 numpy 行拷贝），
  * 一次性 H2D，
  * 最后调 `xgr.apply_token_bitmask_inplace(logits, bitmask, indices)` 一次作用于整批。
其中 `rows = logits.shape[0]`（batch × (1 + 投机 token)），`width = ceil(vocab/32)`。

所以本脚本量三件事：
  A 每步的调用次数（应当恒为 1）；
  B 每次调用的**主机侧**耗时（重排 + H2D 的封装成本）与 batch 的关系；
  C 引擎每步 device 时间在有/无约束下的差，以及摊到每条请求上的量。

用法（crater）：
    python mask_apply_amortization.py --out <dir> [--batches 1,8,32,64]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L56_MODEL", "Qwen/Qwen3-1.7B")
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string", "enum": ["get_weather"]},
                         "arguments": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}},
          "required": ["name", "arguments"]}

_APPLY = {"calls": 0, "host_us": 0.0, "inner_us": 0.0, "rows": [], "impl": {}}


def first_json_object(text):
    """语法完成后模型会继续写自由文本，所以只校验第一个完整 JSON 对象。"""
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


def patch_apply():
    """两代 model runner 的 apply 入口都挂上计时。

    V2（本机默认，日志里的 `Using V2 Model Runner`）走
    `vllm/v1/worker/gpu/structured_outputs.py:59 StructuredOutputsWorker.apply_grammar_bitmask`，
    里面是一次 Triton kernel 启动 + 两条异步拷贝；
    V1 走 `vllm/v1/structured_output/utils.py:87 apply_grammar_bitmask`（xgrammar 内核）。
    只看其中一个会漏掉真正在跑的那条。
    """
    import numpy as np

    def wrap(obj, name, tag):
        orig = getattr(obj, name)

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            _APPLY["calls"] += 1
            _APPLY["host_us"] += (time.perf_counter() - t0) * 1e6
            if a and hasattr(a[0], "shape"):
                _APPLY["rows"].append(int(a[0].shape[0]))
            _APPLY["impl"][tag] = _APPLY["impl"].get(tag, 0) + 1
            return out
        setattr(obj, name, wrapped)

    from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
    wrap(StructuredOutputsWorker, "apply_grammar_bitmask", "v2_triton")
    import vllm.v1.structured_output.utils as su
    wrap(su, "apply_grammar_bitmask", "v1_xgrammar")


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def make_llm(util):
    from vllm import LLM
    return LLM(model=MODEL, gpu_memory_utilization=util, max_model_len=4096,
               enforce_eager=False, enable_prefix_caching=False,
               disable_log_stats=False, max_num_batched_tokens=4096)


def run_case(llm, batch, constrained, rng, out_len=64):
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams
    sp_kw = dict(max_tokens=out_len, temperature=0.0, ignore_eos=True)
    if constrained:
        sp_kw["structured_outputs"] = StructuredOutputsParams(json=json.dumps(SCHEMA))
    sp = SamplingParams(**sp_kw)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(1000, 60000)
                                              for _ in range(64)]) for _ in range(batch)]
    _APPLY.update(calls=0, host_us=0.0, inner_us=0.0, rows=[], impl={})
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    valid = 0
    for o in outs:
        obj = first_json_object(o.outputs[0].text)
        try:
            json.loads(obj) if obj else None
            valid += int(bool(obj))
        except Exception:                                       # noqa: BLE001
            pass
    return dict(batch=batch, constrained=constrained, wall_ms=wall,
                steps_estimate=batch * out_len, tokens=toks,
                out_tps=toks / (wall / 1000),
                apply_calls=_APPLY["calls"],
                apply_host_us_total=round(_APPLY["host_us"], 1),
                apply_inner_us_total=round(_APPLY["inner_us"], 1),
                apply_rows=sorted(set(_APPLY["rows"])),
                apply_impl=dict(_APPLY["impl"]),
                json_valid=valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    patch_apply()
    import random
    rng = random.Random(20260921)
    util = safe_util()
    llm = make_llm(util)
    rows = []
    try:
        for batch in (int(x) for x in args.batches.split(",")):
            for constrained in (False, True):
                for _ in range(1):
                    run_case(llm, batch, constrained, random.Random(7))
                reps = [run_case(llm, batch, constrained, rng) for _ in range(args.repeats)]
                row = dict(batch=batch, constrained=constrained,
                           wall_median_ms=statistics.median(r["wall_ms"] for r in reps),
                           out_tps_median=statistics.median(r["out_tps"] for r in reps),
                           apply_calls=reps[0]["apply_calls"],
                           apply_host_us_total=reps[0]["apply_host_us_total"],
                           apply_inner_us_total=reps[0]["apply_inner_us_total"],
                           apply_rows=reps[0]["apply_rows"],
                           apply_impl=reps[0]["apply_impl"],
                           json_valid=reps[0]["json_valid"])
                rows.append(row)
                per_step = (row["apply_host_us_total"] / row["apply_calls"]
                            if row["apply_calls"] else None)
                print(f"  B={batch:<3} 约束={str(constrained):<5} "
                      f"墙钟 {row['wall_median_ms']:>8.1f} ms  {row['out_tps_median']:>7.1f} tok/s  "
                      f"apply 调用 {row['apply_calls']:>4}  行数 {row['apply_rows']}  "
                      f"主机 {row['apply_host_us_total']:>8.1f} µs"
                      f"（每步 {('%.1f' % per_step) if per_step is not None else '—'} µs）  "
                      f"实现 {row['apply_impl']}  有效 JSON {row['json_valid']}",
                      flush=True)
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:                                       # noqa: BLE001
            pass

    with open(os.path.join(args.out, "mask_apply_amortization.json"), "w") as f:
        json.dump(dict(model=MODEL, util=util, schema=SCHEMA, rows=rows), f, indent=1)
    print(f"\n写入 {args.out}/mask_apply_amortization.json")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
