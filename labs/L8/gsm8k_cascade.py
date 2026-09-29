#!/usr/bin/env python3
"""labs/L8/gsm8k_cascade.py - 8.1-D: GSM8K 上的真实级联难度路由 (质量-成本).

不做任何概率仿真: 全部数字来自在本机 GPU 上对固定题目跑出的真实生成结果。

流程 (标定与评测严格分开, 顺序固定):
  1. 从 gsm8k test split 取固定前 N 题, 用固定种子切成 calibration / evaluation 两半。
  2. 小模型 (Qwen3-1.7B) 与大模型 (Qwen3-4B) 各自对两半解题, 贪心解码, 记录
     逐 token logprob、输出 token 数、墙钟 GPU 时间。
  3. 用**标定集**选择置信度阈值: 以生成答案的均摊 token logprob 为置信度,
     在"升级率不超过设定上限"的约束下取标定准确率最高的阈值, 冻结后不再改动。
  4. 在**评测集**上给出三种运行方式的准确率与成本:
       all-small / all-large / cascade(小模型判定不确定时升级)
     升级阈值、升级率、错误答案、大模型跑分时长全部计入, 不用平均值掩盖长尾。
  5. 成本以实测 GPU-秒为单位 (小模型全量 + 大模型实际被升级的那部分), 不引入
     外部 API 价格假设; 若需要货币换算, 由正文给出单价, 不写进本脚本的结论。

用法:
    python labs/L8/gsm8k_cascade.py --out-dir results/crater/8.1/<run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROMPT_TEMPLATE = (
    "{question}\n"
    "请一步一步推理, 并在最后一行以 '答案: <数字>' 的形式给出最终结果。"
)

NUM_RE = re.compile(r"答案[:：]\s*\**\s*(-?[\d,]+(?:\.\d+)?)")


def extract_number(text: str) -> Optional[str]:
    """取最后一个 '答案: X'; 没有标记时退回最后一个数字。"""
    ms = NUM_RE.findall(text)
    if ms:
        return ms[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def gold_number(answer_field: str) -> Optional[str]:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    if not m:
        return None
    return m.group(1).replace(",", "")


def build_prompt(tok, question: str) -> str:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_model(model_path: str, prompts: List[str], max_tokens: int, seed: int,
              gpu_util: float = 0.45) -> Tuple[List[Dict], float]:
    """用 vLLM 离线接口跑一批 prompt, 返回逐题结果与墙钟秒数。"""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_util,
              max_model_len=4096, enable_prefix_caching=False, seed=seed)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    t0 = time.monotonic()
    outs = llm.generate(prompts, sp)
    elapsed = time.monotonic() - t0

    results: List[Dict] = []
    for o in outs:
        comp = o.outputs[0]
        lp: List[float] = []
        if comp.logprobs:
            for step in comp.logprobs:
                if step:
                    lp.append(next(iter(step.values())).logprob)
        results.append({
            "text": comp.text,
            "num_tokens": len(comp.token_ids),
            "mean_logprob": (statistics.fmean(lp) if lp else None),
            "min_logprob": (min(lp) if lp else None),
            "finish_reason": comp.finish_reason,
        })
    from vllm import LLM as _LLM  # noqa: F401
    try:
        del llm
    except Exception:
        pass
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return results, elapsed


def score(results: List[Dict], golds: List[Optional[str]]) -> List[bool]:
    ok: List[bool] = []
    for r, g in zip(results, golds):
        pred = extract_number(r["text"])
        ok.append(pred is not None and g is not None and pred == g)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--small", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--large", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400, help="固定取前 n 题")
    ap.add_argument("--calib-frac", type=float, default=0.375)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-escalation", type=float, default=0.35,
                    help="标定阶段允许的最大升级率")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.n, len(ds))
    idx = list(range(n))
    questions = [ds[i]["question"] for i in idx]
    golds = [gold_number(ds[i]["answer"]) for i in idx]
    n_calib = int(n * args.calib_frac)
    calib_idx = list(range(n_calib))
    eval_idx = list(range(n_calib, n))

    tok_small = AutoTokenizer.from_pretrained(args.small)
    prompts = [build_prompt(tok_small, q) for q in questions]

    meta: Dict[str, Any] = {
        "small": args.small, "large": args.large, "n": n,
        "calib": n_calib, "eval": n - n_calib,
        "max_tokens": args.max_tokens, "max_escalation": args.max_escalation,
        "seed": args.seed, "prompt_template": PROMPT_TEMPLATE,
        "dataset_n": len(ds),
    }

    print(f"running small={args.small} on {n} questions ...", flush=True)
    small_res, small_s = run_model(args.small, prompts, args.max_tokens, args.seed)
    print(f"small done in {small_s:.1f}s", flush=True)
    print(f"running large={args.large} on {n} questions ...", flush=True)
    large_res, large_s = run_model(args.large, prompts, args.max_tokens, args.seed)
    print(f"large done in {large_s:.1f}s", flush=True)

    small_ok = score(small_res, golds)
    large_ok = score(large_res, golds)

    # ---- 标定: 冻结阈值 ------------------------------------------------
    calib = []
    for i in calib_idx:
        calib.append({"i": i, "conf": small_res[i]["mean_logprob"], "small_ok": small_ok[i],
                      "large_ok": large_ok[i]})
    confs = sorted({round(c["conf"], 4) for c in calib if c["conf"] is not None})
    sweep = []
    for thr in confs:
        esc = [c for c in calib if c["conf"] is None or c["conf"] < thr]
        rate = len(esc) / len(calib)
        acc = sum(1 for c in calib
                  if (c["large_ok"] if (c["conf"] is None or c["conf"] < thr) else c["small_ok"])) / len(calib)
        sweep.append({"threshold": thr, "escalation_rate": rate, "accuracy": acc})
    feasible = [s for s in sweep if s["escalation_rate"] <= args.max_escalation]
    chosen = max(feasible, key=lambda s: s["accuracy"]) if feasible else min(
        sweep, key=lambda s: abs(s["escalation_rate"] - args.max_escalation))
    threshold = chosen["threshold"]

    # ---- 评测集: 三种方式真实成本 --------------------------------------
    esc_idx = [i for i in eval_idx if small_res[i]["mean_logprob"] is None
               or small_res[i]["mean_logprob"] < threshold]
    # 大模型只对升级子集再跑一遍, 单独计时, 不做时间外推。
    large_sub_res: List[Dict] = []
    large_sub_s = 0.0
    if esc_idx:
        print(f"re-running large on {len(esc_idx)} escalated questions ...", flush=True)
        large_sub_res, large_sub_s = run_model(
            args.large, [prompts[i] for i in esc_idx], args.max_tokens, args.seed)
        print(f"large subset done in {large_sub_s:.1f}s", flush=True)

    sub_ok = score(large_sub_res, [golds[i] for i in esc_idx]) if esc_idx else []
    esc_set = dict(zip(esc_idx, sub_ok))

    def cascade_ok(i: int) -> bool:
        if i in esc_set:
            return esc_set[i]
        return small_ok[i]

    def acc(idxs, fn) -> float:
        return sum(1 for i in idxs if fn(i)) / len(idxs)

    # 成本按"真实跑过的时间"计: 小模型全量 + 大模型被升级子集。
    # 小模型在全量 n 题上跑过一次, 评测集部分按其 token 占比折算墙钟。
    small_tokens_all = sum(r["num_tokens"] for r in small_res)
    small_tokens_eval = sum(small_res[i]["num_tokens"] for i in eval_idx)
    small_s_eval = small_s * (small_tokens_eval / small_tokens_all) if small_tokens_all else 0.0

    large_tokens_all = sum(r["num_tokens"] for r in large_res)
    large_s_eval_all = large_s * (sum(large_res[i]["num_tokens"] for i in eval_idx) / large_tokens_all) \
        if large_tokens_all else 0.0

    summary = {
        "meta": meta,
        "threshold": threshold,
        "calibration": {"chosen": chosen, "sweep": sweep,
                        "small_accuracy": sum(1 for c in calib if c["small_ok"]) / len(calib),
                        "large_accuracy": sum(1 for c in calib if c["large_ok"]) / len(calib)},
        "evaluation": {
            "n": len(eval_idx),
            "all_small": {"accuracy": acc(eval_idx, lambda i: small_ok[i]),
                          "gpu_seconds_on_eval": small_s_eval,
                          "output_tokens": small_tokens_eval},
            "all_large": {"accuracy": acc(eval_idx, lambda i: large_ok[i]),
                          "gpu_seconds_on_eval": large_s_eval_all,
                          "output_tokens": sum(large_res[i]["num_tokens"] for i in eval_idx)},
            "cascade": {
                "accuracy": acc(eval_idx, cascade_ok),
                "escalated": len(esc_idx),
                "escalation_rate": len(esc_idx) / len(eval_idx),
                "gpu_seconds_on_eval": small_s_eval + large_sub_s,
                "small_gpu_seconds": small_s_eval,
                "large_gpu_seconds_escalated": large_sub_s,
                "output_tokens": small_tokens_eval + sum(r["num_tokens"] for r in large_sub_res),
                "wrong_escalated": sum(1 for i in esc_idx if not esc_set.get(i, False)),
                "wrong_not_escalated": sum(1 for i in eval_idx
                                           if i not in esc_set and not small_ok[i]),
            },
        },
        "timing": {"small_all_s": small_s, "large_all_s": large_s, "large_subset_s": large_sub_s},
    }

    (out / "cascade_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐题原始输出: 供复核答案解析与阈值判定。
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "i": i, "split": "calib" if i < n_calib else "eval",
                "question": questions[i], "gold": golds[i],
                "small_pred": extract_number(small_res[i]["text"]),
                "small_ok": small_ok[i], "small_conf": small_res[i]["mean_logprob"],
                "small_tokens": small_res[i]["num_tokens"],
                "large_pred": extract_number(large_res[i]["text"]),
                "large_ok": large_ok[i], "large_tokens": large_res[i]["num_tokens"],
                "escalated": i in esc_set,
                "cascade_ok": cascade_ok(i) if i >= n_calib else None,
                "small_text": small_res[i]["text"][:2000],
                "large_text": large_res[i]["text"][:2000],
            }, ensure_ascii=False) + "\n")

    print(json.dumps(summary["evaluation"], ensure_ascii=False, indent=2))
    print(f"threshold={threshold} -> {out}")


if __name__ == "__main__":
    main()
