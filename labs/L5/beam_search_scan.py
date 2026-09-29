#!/usr/bin/env python3
"""L5.9 补测 · beam search 的逐 beam 分数、长度惩罚、搜索预算与质量。

上一轮的探针只证明"能跑通、能取出逐 beam 输出"。这一轮把 beam 当成一条
**服务路线**来量四件事：

  1. **逐 beam 分数**：SGLang 在 `meta_info.beam_results[i].meta_info.sequence_score`
     给出每条 beam 自己的分数；顶层 `sequence_score` 只是最好那条。
  2. **长度惩罚**：`length_penalty` 扫 0.0 / 0.6 / 1.0 / 1.2，固定 width=4。
     惩罚改变的是"长序列的相对得分"，因此首先影响选中哪条 beam。
  3. **搜索预算**：`max_new_tokens` 64 / 256 —— beam 的代价是 width × tokens 的
     KV 分叉与合批，`meta_info.completion_tokens` 给的正是"这一步一共算了多少 token"。
  4. **质量**：GSM8K 固定 24 题，按最终数字判对错（与 9.4 同一套提取口径），
     比较贪心与各 beam 配置；同时给出每题耗时与显存侧读数（`/metrics` 采样）。

用法（先起好 SGLang，见 run_beam_scan.sh）：
    python beam_search_scan.py --base http://127.0.0.1:8148 --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import threading
import time

import requests

GSM8K_GLOB = ("/scratch/learn/models/hf/hub/datasets--openai--gsm8k/snapshots/*/"
              "main/test-00000-of-00001.parquet")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def norm_num(s: str) -> str:
    s = str(s).strip().replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def gsm8k_final(text: str) -> str | None:
    cleaned = text.replace(",", "")
    m = re.findall(r"####\s*([^\n]+)", cleaned)
    if m:
        nums = NUM_RE.findall(m[-1])
        if nums:
            return norm_num(nums[-1])
    nums = NUM_RE.findall(cleaned)
    return norm_num(nums[-1]) if nums else None


def load_gsm8k(n: int) -> list[dict]:
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(GSM8K_GLOB))
    if not paths:
        raise SystemExit(f"找不到 GSM8K parquet：{GSM8K_GLOB}")
    t = pq.read_table(paths[-1])
    rows = t.to_pylist()
    out = []
    for r in rows[:n]:
        gold = r["answer"].split("####")[-1].strip()
        out.append(dict(q=r["question"], gold=norm_num(gold)))
    return out


class MetricSampler:
    """按间隔抓 /metrics，取 KV/并发侧读数。"""

    KEYS = ("sglang:token_usage", "sglang:num_running_reqs",
            "sglang:num_queue_reqs", "sglang:cache_hit_rate",
            "sglang:gen_throughput")

    def __init__(self, base, interval=0.1):
        self.base = base
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _read(self):
        try:
            txt = requests.get(self.base + "/metrics", timeout=5).text
        except Exception:                                       # noqa: BLE001
            return {}
        out = {}
        for line in txt.splitlines():
            for k in self.KEYS:
                if line.startswith(k):
                    try:
                        out[k] = float(line.rsplit(" ", 1)[1])
                    except (ValueError, IndexError):
                        pass
        return out

    def start(self):
        def loop():
            while not self._stop.is_set():
                s = self._read()
                if s:
                    self.samples.append(s)
                time.sleep(self.interval)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        peak = {}
        for k in self.KEYS:
            vals = [s[k] for s in self.samples if k in s]
            if vals:
                peak[k] = max(vals)
        return dict(n_samples=len(self.samples), peak=peak)


def submit_chunked(base, tok, questions, sp, chunk, timeout):
    """按 chunk 分批提交：beam 会为每条请求分叉 width 份 KV，
    一次把 n 条全发出去会撞上 KV 池上限（本机 n=96/width≥4 时是 HTTP 500）。"""
    bodies = []
    for i in range(0, len(questions), chunk):
        part = questions[i:i + chunk]
        ids = [tok.encode(q["q"], add_special_tokens=False) for q in part]
        r = requests.post(base + "/generate",
                          json={"input_ids": ids, "sampling_params": sp},
                          timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        b = r.json()
        bodies.extend(b if isinstance(b, list) else [b])
    return bodies, None


def scan(base: str, tok, questions: list[dict], widths, penalties,
         budgets, out_dir, timeout=1800, chunk=24) -> list[dict]:
    rows = []
    # 长度惩罚在本版本不可配置：`BeamGroup` 由 coordinator.py:200 构造时不传
    # length_penalty（默认 1.0），`SamplingParams` 里也没有这个字段。
    # 传进去会直接 500，所以默认只扫宽度与搜索预算。
    plan = ([("width", w, None, b) for b in budgets for w in widths]
            + [("penalty", 4, p, budgets[0]) for p in penalties])
    seen = set()
    for kind, width, pen, budget in plan:
        key = (kind, width, pen, budget)
        if key in seen:
            continue
        seen.add(key)
        sp = dict(temperature=0.0, max_new_tokens=budget, beam_width=width, n=width)
        if kind == "penalty":
            sp["length_penalty"] = pen          # 仅用于记录"本版本拒绝该字段"
        sampler = MetricSampler(base)
        sampler.start()
        t0 = time.perf_counter()
        bodies, err = submit_chunked(base, tok, questions, sp, chunk, timeout)
        wall = time.perf_counter() - t0
        metrics = sampler.stop()
        if err:
            rows.append(dict(kind=kind, width=width, length_penalty=pen,
                             max_new_tokens=budget, error=err))
            print(f"  {kind:<8} w={width} pen={pen} budget={budget}: {err[:80]}")
            continue
        per_q = []
        for i, (b, q) in enumerate(zip(bodies, questions)):
            meta = b.get("meta_info", {})
            beams = [(c.get("meta_info") or {}, c.get("text") or "")
                     for c in (meta.get("beam_results") or [])]
            top_text = b.get("text") or ""
            scored = [(m.get("sequence_score"), t) for m, t in beams]
            if not scored:
                scored = [(meta.get("sequence_score"), top_text)]
            best = max(scored, key=lambda x: (x[0] if x[0] is not None else -1e9))
            per_q.append(dict(i=i, gold=q["gold"],
                              completion_tokens=meta.get("completion_tokens"),
                              top_score=meta.get("sequence_score"),
                              n_beams=len(scored),
                              beam_scores=[s for s, _ in scored],
                              greedy_pred=gsm8k_final(top_text),
                              greedy_ok=gsm8k_final(top_text) == q["gold"],
                              best_pred=gsm8k_final(best[1]),
                              best_ok=gsm8k_final(best[1]) == q["gold"],
                              any_beam_ok=any(gsm8k_final(t) == q["gold"]
                                              for _, t in scored)))
        acc_top = sum(1 for p in per_q if p["greedy_ok"]) / len(per_q)
        acc_best = sum(1 for p in per_q if p["best_ok"]) / len(per_q)
        acc_any = sum(1 for p in per_q if p["any_beam_ok"]) / len(per_q)
        row = dict(kind=kind, width=width, length_penalty=pen,
                   max_new_tokens=budget, wall_s=round(wall, 3),
                   completion_tokens_median=statistics.median(
                       p["completion_tokens"] or 0 for p in per_q),
                   top1_accuracy=acc_top, best_beam_accuracy=acc_best,
                   any_beam_accuracy=acc_any,
                   score_range=[min((s for p in per_q for s in p["beam_scores"]
                                     if s is not None), default=None),
                                max((s for p in per_q for s in p["beam_scores"]
                                     if s is not None), default=None)],
                   metrics=metrics, per_question=per_q)
        rows.append(row)
        print(f"  {kind:<8} w={width} pen={pen} budget={budget}: "
              f"top1 {acc_top:.3f} best {acc_best:.3f} any {acc_any:.3f}  "
              f"墙钟 {wall:.1f}s  completion_tokens 中位 "
              f"{row['completion_tokens_median']:.0f}  峰值 "
              f"{metrics['peak'].get('sglang:token_usage')}", flush=True)
        (out_dir / "partial.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8148")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--widths", default="1,2,4,8")
    ap.add_argument("--penalties", default="",
                    help="本版本不可配置长度惩罚，默认空；给了会被服务端拒绝")
    ap.add_argument("--budgets", default="64,256")
    ap.add_argument("--chunk", type=int, default=24,
                    help="每批提交多少条请求；beam 的 KV 分叉让大批量容易撞上限")
    ap.add_argument("--model", default=os.environ.get(
        "L59_MODEL", "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/"
                      "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"))
    args = ap.parse_args()
    out_dir = __import__("pathlib").Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    questions = load_gsm8k(args.n)
    rows = scan(args.base, tok,
                questions,
                [int(x) for x in args.widths.split(",")],
                [float(x) for x in args.penalties.split(",") if x],
                [int(x) for x in args.budgets.split(",")], out_dir,
                chunk=args.chunk)
    (out_dir / "beam_scan.json").write_text(
        json.dumps(dict(base=args.base, n_questions=len(questions), rows=rows),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out_dir}/beam_scan.json")


if __name__ == "__main__":
    main()
