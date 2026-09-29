#!/usr/bin/env python3
"""L9.4：reasoning 预算、状态增长与「每正确答案成本」。

四个模式：

``sweep``       固定题目/模板/评分，扫 budget=128/512/2048/8192 与 thinking 开关；保存
                reasoning/final、停止原因、抽取答案与是否正确。截断、超时、未给答案一律
                计入任务结果（当错），不删样本。
``concurrency`` 固定 budget 扫并发 1/4/16，记录每请求 prefill(TTFT)/decode/端到端与长尾；
                另做一组「长短混合」批次，量长输出对短请求的队列影响。
``states``      用流式 chunk 时间戳给出**逐位置 decode 延迟**，比较纯 attention（Qwen3-4B）
                与混合架构（Qwen3.5-4B）随输出长度增长的形状，并按配置算状态字节。
``samples``     相同总 token/GPO 时间预算下比较「一次长推理」与「k 次短采样 + 多数投票」。

用法::

    python labs/L9/reasoning_budget.py sweep --base-url ... --dataset gsm8k --out DIR
    python labs/L9/reasoning_budget.py concurrency --base-url ... --out DIR
    python labs/L9/reasoning_budget.py states --base-url ... --model Qwen/Qwen3.5-4B --out DIR
    python labs/L9/reasoning_budget.py samples --base-url ... --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import threading
import time

GSM8K_GLOB = "/scratch/learn/models/hf/hub/datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet"
MATH500_GLOB = "/scratch/learn/models/hf/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl"

GSM8K_TEMPLATE = (
    "Solve the problem step by step, then give the final numeric answer on its own line "
    "in the form `#### <number>`.\n\nProblem: {q}\n"
)
MATH_TEMPLATE = (
    "Solve the problem step by step. Put the final answer inside \\boxed{{}}. "
    "The final line must be `\\boxed{{answer}}`.\n\nProblem: {q}\n"
)


# --------------------------------------------------------------------------------------
# 数据与评分
# --------------------------------------------------------------------------------------

def load_questions(dataset: str, n: int, seed: int = 0) -> list[dict]:
    import random

    rows: list[dict] = []
    if dataset == "gsm8k":
        import pandas as pd

        path = sorted(glob.glob(GSM8K_GLOB))[0]
        df = pd.read_parquet(path)
        for i, r in df.iterrows():
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"qid": f"gsm8k-{i}", "prompt": GSM8K_TEMPLATE.format(q=r["question"]),
                         "gold": gold})
    elif dataset == "math500":
        path = sorted(glob.glob(MATH500_GLOB))[0]
        with open(path, encoding="utf-8") as fh:
            raw = [json.loads(l) for l in fh]
        for r in raw:
            rows.append({"qid": f"math-{r['unique_id']}", "prompt": MATH_TEMPLATE.format(q=r["problem"]),
                         "gold": r["answer"]})
    else:
        raise ValueError(dataset)
    random.Random(seed).shuffle(rows)
    return rows[:n]


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return s
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.6g}"


def extract_gsm8k(text: str) -> str | None:
    # 千位分隔符先去掉，否则 1,234 会被切成 1 与 234
    cleaned = text.replace(",", "")
    m = re.findall(r"####\s*([^\n]+)", cleaned)
    if m:
        nums = NUM_RE.findall(m[-1])
        if nums:
            return _norm_num(nums[-1])
    nums = NUM_RE.findall(cleaned)
    return _norm_num(nums[-1]) if nums else None


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize_math(s: str) -> str:
    s = s.strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", ""), ("\\ ", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("$", ""), (" ", "")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    return s


def score(dataset: str, text: str, gold: str) -> dict:
    if dataset == "gsm8k":
        pred = extract_gsm8k(text)
        return {"extracted": pred, "correct": pred == _norm_num(gold)}
    pred = extract_boxed(text)
    if pred is None:
        return {"extracted": None, "correct": False}
    ok = normalize_math(pred) == normalize_math(gold)
    if not ok:
        try:
            ok = abs(float(normalize_math(pred)) - float(normalize_math(gold))) < 1e-6
        except (ValueError, TypeError):
            ok = False
    return {"extracted": pred, "correct": bool(ok)}


# --------------------------------------------------------------------------------------
# 功耗采样（nvidia-smi）
# --------------------------------------------------------------------------------------

class PowerMeter:
    """按固定间隔读 nvidia-smi 的瞬时功耗并积分，给出本次运行的焦耳数。"""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()
                if out:
                    self.samples.append((time.perf_counter(), float(out[0])))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if len(self.samples) < 2:
            return {"joules": None, "samples": len(self.samples)}
        joules = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            joules += 0.5 * (p0 + p1) * (t1 - t0)
        return {
            "joules": round(joules, 2),
            "mean_watts": round(joules / (self.samples[-1][0] - self.samples[0][0]), 1),
            "samples": len(self.samples),
            "window_s": round(self.samples[-1][0] - self.samples[0][0], 2),
        }


# --------------------------------------------------------------------------------------
# 单请求执行
# --------------------------------------------------------------------------------------

async def one_request(client, args, prompt, budget, thinking, meter_slice=None):
    """发一条请求，流式记录 TTFT 与（抽样的）chunk 时间戳。"""
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    marks: list[tuple[int, float]] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    finish = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=budget,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            d = ch.delta
            if d is None:
                continue
            piece = getattr(d, "content", None)
            rpiece = getattr(d, "reasoning_content", None)
            if piece or rpiece:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                if rpiece:
                    reasoning_parts.append(rpiece)
                    chunks += 1
                if piece:
                    text_parts.append(piece)
                    chunks += 1
                if chunks % args.mark_every == 0:
                    marks.append((chunks, round((time.perf_counter() - t0) * 1000.0, 3)))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    e2e = (time.perf_counter() - t0) * 1000.0
    return {
        "ttft_ms": round(ttft, 3) if ttft else None,
        "e2e_ms": round(e2e, 3),
        "decode_ms": round(e2e - (ttft or 0.0), 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        "finish_reason": finish,
        "chunks": chunks,
        "marks": marks,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "error": error,
    }


async def run_batch(args, items: list[dict], concurrency: int, meter: PowerMeter | None = None):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def worker(item):
        async with sem:
            r = await one_request(client, args, item["prompt"], item["budget"], item["thinking"])
            r.update({k: item[k] for k in ("qid", "dataset", "budget", "thinking", "tag", "gold") if k in item})
            results.append(r)

    if meter:
        meter.start()
    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in items))
    wall = time.perf_counter() - t0
    energy = meter.stop() if meter else None
    await client.close()
    for r in results:
        if r.get("dataset"):
            sc = score(r["dataset"], r["text"], r["gold"])
            r.update(sc)
    return {"results": results, "wall_s": round(wall, 3), "energy": energy}


# --------------------------------------------------------------------------------------
# 模式实现
# --------------------------------------------------------------------------------------

def cmd_sweep(args) -> int:
    questions = load_questions(args.dataset, args.n, args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for thinking in (True, False):
        for budget in args.budgets:
            items = [{"qid": q["qid"], "dataset": args.dataset, "prompt": q["prompt"],
                      "gold": q["gold"], "budget": budget, "thinking": thinking,
                      "tag": "sweep"} for q in questions]
            meter = PowerMeter()
            batch = asyncio.run(run_batch(args, items, args.concurrency, meter))
            rows = batch["results"]
            n = len(rows)
            correct = sum(1 for r in rows if r.get("correct"))
            truncated = sum(1 for r in rows if r.get("finish_reason") == "length")
            no_answer = sum(1 for r in rows if r.get("extracted") is None)
            errors = sum(1 for r in rows if r.get("error"))
            comp = [r["completion_tokens"] or 0 for r in rows]
            reas = [r["reasoning_tokens"] or 0 for r in rows]
            summary = {
                "thinking": thinking, "budget": budget, "questions": n,
                "correct": correct, "accuracy": round(correct / max(1, n), 4),
                "truncated": truncated, "no_answer": no_answer, "errors": errors,
                "completion_tokens_mean": round(statistics.fmean(comp), 1),
                "reasoning_tokens_mean": round(statistics.fmean(reas), 1),
                "ttft_ms_p50": _q([r["ttft_ms"] for r in rows], 0.5),
                "e2e_ms_p50": _q([r["e2e_ms"] for r in rows], 0.5),
                "wall_s": batch["wall_s"], "energy": batch["energy"],
                "joules_per_correct": round(batch["energy"]["joules"] / correct, 2)
                if batch["energy"] and batch["energy"].get("joules") and correct else None,
                "seconds_per_correct": round(batch["wall_s"] / correct, 3) if correct else None,
            }
            all_rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            with open(out / f"rows-{args.dataset}-th{int(thinking)}-b{budget}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"sweep-{args.dataset}.json").write_text(
        json.dumps({"config": _cfg(args), "rows": all_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(all_rows, ensure_ascii=False, indent=1))
    return 0


def _q(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))], 3)


def _cfg(args) -> dict:
    return {"base_url": args.base_url, "model": args.model, "dataset": getattr(args, "dataset", None),
            "n": getattr(args, "n", None), "budgets": getattr(args, "budgets", None),
            "seed": getattr(args, "seed", None), "temperature": getattr(args, "temperature", None)}


def cmd_concurrency(args) -> int:
    questions = load_questions(args.dataset, args.n, args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for conc in args.concurrency_levels:
        items = [{"qid": q["qid"], "dataset": args.dataset, "prompt": q["prompt"], "gold": q["gold"],
                  "budget": args.budget, "thinking": True, "tag": f"homogeneous-c{conc}"}
                 for q in questions]
        meter = PowerMeter()
        batch = asyncio.run(run_batch(args, items, conc, meter))
        rs = batch["results"]
        rows.append({
            "mode": "homogeneous", "concurrency": conc, "requests": len(rs),
            "correct": sum(1 for r in rs if r.get("correct")),
            "ttft_p50": _q([r["ttft_ms"] for r in rs], 0.5), "ttft_p95": _q([r["ttft_ms"] for r in rs], 0.95),
            "decode_p50": _q([r["decode_ms"] for r in rs], 0.5),
            "e2e_p50": _q([r["e2e_ms"] for r in rs], 0.5), "e2e_p95": _q([r["e2e_ms"] for r in rs], 0.95),
            "throughput_tok_s": round(sum(r["completion_tokens"] or 0 for r in rs) / batch["wall_s"], 1),
            "wall_s": batch["wall_s"], "energy": batch["energy"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    # 长短混合：一半 budget=512、一半 budget=8192，看短请求是否被长请求拖住
    mixed_items = []
    for i, q in enumerate(questions):
        budget = 512 if i % 2 == 0 else 8192
        mixed_items.append({"qid": q["qid"], "dataset": args.dataset, "prompt": q["prompt"], "gold": q["gold"],
                            "budget": budget, "thinking": True, "tag": f"mixed-{budget}"})
    meter = PowerMeter()
    batch = asyncio.run(run_batch(args, mixed_items, args.concurrency_levels[-1], meter))
    rs = batch["results"]
    for tag in ("mixed-512", "mixed-8192"):
        sub = [r for r in rs if r["tag"] == tag]
        rows.append({
            "mode": tag, "concurrency": args.concurrency_levels[-1], "requests": len(sub),
            "correct": sum(1 for r in sub if r.get("correct")),
            "ttft_p50": _q([r["ttft_ms"] for r in sub], 0.5), "ttft_p95": _q([r["ttft_ms"] for r in sub], 0.95),
            "e2e_p50": _q([r["e2e_ms"] for r in sub], 0.5), "e2e_p95": _q([r["e2e_ms"] for r in sub], 0.95),
            "throughput_tok_s": round(sum(r["completion_tokens"] or 0 for r in rs) / batch["wall_s"], 1),
            "wall_s": batch["wall_s"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / "concurrency.json").write_text(
        json.dumps({"config": _cfg(args), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_states(args) -> int:
    """逐位置 decode 延迟。

    两个自变量一起扫：**上下文长度**（用 ``--copies`` 把同一段种子文本复制若干份）与
    **并发**（``--concurrency``）。batch=1 且上下文短时 decode 是权重带宽受限的，
    KV 增长看不出来；只有把 KV 读取抬到与权重读取同量级（大 batch × 长上下文），
    状态增长才会体现在逐 token 斜率上。
    """
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_text = ("Write a long detailed essay about the history of computing, at least "
                 "several hundred words, without stopping. ")
    prompt = seed_text * max(1, args.copies)
    rows = []
    for budget in args.budgets:
        items = [{"prompt": prompt, "budget": budget, "thinking": False, "tag": f"states-{budget}"}
                 for _ in range(max(1, args.concurrency))]
        batch = asyncio.run(run_batch(_clone_args(args, temperature=0.0), items, args.concurrency))
        rs = batch["results"]
        # 逐位置段：按位置对齐后取各请求的平均，减少单请求抖动
        seg_by_pos: dict[tuple[int, int], list[float]] = {}
        for r in rs:
            prev_pos, prev_t = 0, r["ttft_ms"] or 0.0
            for pos, t in r["marks"]:
                seg_by_pos.setdefault((prev_pos, pos), []).append((t - prev_t) / max(1, pos - prev_pos))
                prev_pos, prev_t = pos, t
        seg = [{"from": a, "to": b, "ms_per_token": round(statistics.fmean(v), 3)}
               for (a, b), v in sorted(seg_by_pos.items())]
        rows.append({
            "model": args.model, "budget": budget, "copies": args.copies,
            "concurrency": args.concurrency,
            "requests": len(rs),
            "prompt_tokens": statistics.median([r["prompt_tokens"] or 0 for r in rs]),
            "completion_tokens_mean": round(statistics.fmean([r["completion_tokens"] or 0 for r in rs]), 1),
            "ttft_ms_mean": round(statistics.fmean([r["ttft_ms"] or 0 for r in rs]), 2),
            "e2e_ms_mean": round(statistics.fmean([r["e2e_ms"] for r in rs]), 2),
            "segments": seg,
        })
        print(json.dumps(rows[-1], ensure_ascii=False)[:500], flush=True)
    name = args.model.replace("/", "_")
    fname = f"states-{name}-b{args.concurrency}-copies{args.copies}.json"
    (out / fname).write_text(
        json.dumps({"config": {"model": args.model, "budgets": args.budgets,
                               "copies": args.copies, "concurrency": args.concurrency},
                    "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


class _clone_args:
    def __init__(self, base, **kw):
        self.__dict__.update(base.__dict__)
        self.__dict__.update(kw)


def cmd_samples(args) -> int:
    """相同总输出预算：一次长推理 vs k 次短采样 + 多数投票。"""
    questions = load_questions(args.dataset, args.n, args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # 参照：一次长预算
    for label, k, budget in (("single_long", 1, args.budget), ("k_samples", args.k, args.budget // args.k)):
        items = []
        for q in questions:
            for s in range(k):
                items.append({"qid": q["qid"], "dataset": args.dataset, "prompt": q["prompt"],
                              "gold": q["gold"], "budget": budget, "thinking": True,
                              "tag": f"{label}-s{s}"})
        meter = PowerMeter()
        batch = asyncio.run(run_batch(args, items, args.concurrency, meter))
        rs = batch["results"]
        # 聚合：对每条题的 k 个抽取答案做多数投票
        per_q: dict[str, list[dict]] = {}
        for r in rs:
            per_q.setdefault(r["qid"], []).append(r)
        correct = 0
        for qid, group in per_q.items():
            answers = [r["extracted"] for r in group if r.get("extracted")]
            gold = next(q["gold"] for q in questions if q["qid"] == qid)
            if answers:
                vote = statistics.mode(answers) if len(set(answers)) < len(answers) else answers[0]
                correct += int(score(args.dataset, f"\\boxed{{{vote}}}" if args.dataset == "math500" else f"#### {vote}", gold)["correct"])
        rows.append({
            "label": label, "samples_per_question": k, "budget_per_sample": budget,
            "total_budget_per_question": k * budget,
            "questions": len(questions), "correct": correct,
            "accuracy": round(correct / max(1, len(questions)), 4),
            "completion_tokens_total": sum(r["completion_tokens"] or 0 for r in rs),
            "wall_s": batch["wall_s"], "energy": batch["energy"],
            "joules_per_correct": round(batch["energy"]["joules"] / correct, 2)
            if batch["energy"] and batch["energy"].get("joules") and correct else None,
            "seconds_per_correct": round(batch["wall_s"] / correct, 3) if correct else None,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / f"samples-{args.dataset}.json").write_text(
        json.dumps({"config": _cfg(args), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.4 reasoning 预算")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(base_url="http://127.0.0.1:8013/v1", model="Qwen/Qwen3-4B",
                  temperature=0.0, top_p=1.0, max_tokens=8192, timeout=1800.0)

    p = sub.add_parser("sweep")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--budgets", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mark-every", type=int, default=64)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("concurrency")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mark-every", type=int, default=64)
    p.set_defaults(func=cmd_concurrency)

    p = sub.add_parser("states")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[512, 2048, 4096])
    p.add_argument("--mark-every", type=int, default=32)
    p.add_argument("--copies", type=int, default=1, help="把种子文本复制几份，用来拉长上下文")
    p.add_argument("--concurrency", type=int, default=1)
    p.set_defaults(func=cmd_states)

    p = sub.add_parser("samples")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mark-every", type=int, default=64)
    p.set_defaults(func=cmd_samples)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
