#!/usr/bin/env python3
"""L5.8 任务 C（背压部分）—— 泊松到达下的有界准入对照。

既有背压实验只放一次性突发（24 条同时到），看不出"到达率 × 队列上限"的交互。
本脚本补上到达过程：

  1. 先用大块池、`max_num_seqs=1` 量单请求服务时间，得到基线服务率；
  2. 用指数间隔生成泊松到达，到达率取基线的 0.3 / 0.6 / 0.9 / 1.1 倍；
  3. 每档各跑 `max_waiting = None / 12 / 4`（无界排队 / 有界准入）；
  4. 记录提供数、接收数、拒绝数、拒绝率、队列峰值、等待时间分布与完成吞吐。

**样本量说明**：每档只有几十条请求，远低于"不足 10000 请求不宣称 p99 稳定"的门槛，
所以这里只报拒绝率、等待中位数与吞吐，p99 只作为同口径的排序参考，不作稳定性结论。

虚拟时钟推进用真实的每步墙钟，所以到达率与引擎速度在同一时间尺度上；
所有配置共用同一串到达时间序列（同一 seed），保证可比。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from engine import Request, State, OutOfBlocks
from failure import ResilientEngine
from model import PagedModel

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
PROMPT = ("Summarize the theory of relativity in one line. "
          "Then explain what a prefix cache does. ") * 1
MODEL = None


def get_model(num_blocks):
    global MODEL
    if MODEL is None or MODEL.num_blocks != num_blocks:
        MODEL = None
        torch.cuda.empty_cache()
        MODEL = PagedModel(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return MODEL


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


def measure_capacity(tok, ids, max_tokens, max_seqs, count, repeats=3):
    """量引擎在本配置下的**容量**：同样并发、足够大的块池、一次性突发。

    用单并发单请求的倒数当基线是错的：引擎在 max_num_seqs 条并发下的容量
    比它高一个量级，那样 0.3–1.1× 全都落在欠载区，永远看不到过载。
    """
    model = get_model(2048)
    rates = []
    for i in range(repeats):
        eng = ResilientEngine(model, block_size=BLOCK_SIZE,
                              max_batched_tokens=256, max_num_seqs=max_seqs,
                              enable_prefix_cache=False, eos_ids=[])
        for j in range(count):
            eng.add(Request(f"c{i}-{j}", ids, max_tokens=max_tokens))
        t0 = time.perf_counter()
        eng.run_until_idle(max_steps=20000)
        dt = time.perf_counter() - t0
        rates.append(count / dt)
    rate = statistics.median(rates)
    return rate, 1.0 / rate


def poisson_arrivals(rate, count, seed):
    """指数间隔的到达时刻，返回累计时间列表。"""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(count):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def run_config(tok, ids, blocks, max_tokens, max_seqs, max_waiting,
               arrivals, count, step_cap=4000):
    model = get_model(blocks)
    eng = ResilientEngine(model, block_size=BLOCK_SIZE, max_batched_tokens=256,
                          max_num_seqs=max_seqs, enable_prefix_cache=False,
                          eos_ids=[], max_waiting=max_waiting)
    t0 = time.perf_counter()
    offered = accepted = 0
    queue_peak = 0
    waits, depths = [], []
    steps = 0
    next_arrival = 0
    while steps < step_cap:
        elapsed = time.perf_counter() - t0
        # 把已经到点的请求交给引擎（有界准入时 try_add 会给出拒绝）
        while next_arrival < count and arrivals[next_arrival] <= elapsed:
            r = Request(f"r{next_arrival}", ids, max_tokens=max_tokens)
            r.arrival = t0 + arrivals[next_arrival]
            offered += 1
            if eng.try_add(r):
                accepted += 1
            next_arrival += 1
        depths.append(len(eng.waiting))
        queue_peak = max(queue_peak, len(eng.waiting))
        if next_arrival >= count and not eng.waiting and not eng.running:
            break
        if not eng.waiting and not eng.running:
            # 没有活可干：跳到下一个到达时刻。空转既会烧 CPU，
            # 也会把 step_cap 白吃掉（到达率低时根本轮不到真正的工作）。
            idle_until = arrivals[next_arrival] if next_arrival < count else None
            if idle_until is None:
                break
            time.sleep(max(0.0, idle_until - (time.perf_counter() - t0)))
            continue
        try:
            eng.step()
        except OutOfBlocks as exc:
            return dict(max_waiting=max_waiting, crashed=str(exc), offered=offered,
                        accepted=accepted, rejected=offered - accepted, steps=steps)
        steps += 1
        for r in list(eng.done.values()):
            if getattr(r, "_waited_logged", False):
                continue
            if r.first_token_at is not None:
                waits.append((r.first_token_at - r.arrival) * 1000)
            r._waited_logged = True
    wall = time.perf_counter() - t0
    done = [r for r in eng.done.values() if r.state is State.FINISHED]
    tokens = sum(len(r.output_ids) for r in done)
    return dict(max_waiting=max_waiting, offered=offered, accepted=accepted,
                rejected=offered - accepted,
                rejection_rate=(offered - accepted) / max(offered, 1),
                queue_peak=queue_peak, max_running=max_seqs,
                completed=len(done), produced_tokens=tokens,
                wall_s=round(wall, 3), steps=steps,
                goodput_tok_s=round(tokens / wall, 1) if wall else None,
                goodput_note="吞吐按整个观察窗计，含等待到达的空闲时间",
                wait_p50_ms=round(statistics.median(waits), 2) if waits else None,
                wait_p99_ms=round(sorted(waits)[int(0.99 * (len(waits) - 1))], 2)
                if len(waits) > 5 else None,
                samples=len(waits), leak=eng.leak_check())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--max-seqs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[0.3, 0.6, 0.9, 1.1],
                    help="到达率相对测量容量的倍数；扫到 2× 可看崩溃边界")
    ap.add_argument("--step-cap", type=int, default=4000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    ids = prompt_ids(tok, PROMPT)

    base_rate, svc = measure_capacity(tok, ids, args.max_tokens, args.max_seqs,
                                      args.count)
    print(f"容量基线（并发 {args.max_seqs}、块池 2048、突发 {args.count} 条）："
          f"{base_rate:.2f} 请求/秒（等效单请求 {svc*1000:.1f} ms）")

    rows = []
    for mult in args.multipliers:
        rate = base_rate * mult
        arrivals = poisson_arrivals(rate, args.count, args.seed)
        for limit in (None, 12, 4):
            r = run_config(tok, ids, args.blocks, args.max_tokens, args.max_seqs,
                           limit, arrivals, args.count,
                           step_cap=args.step_cap)
            r.update(rate_multiplier=mult, rate_req_s=round(rate, 3))
            rows.append(r)
            print(f"  λ={mult:>4}× ({rate:5.2f} req/s)  上限={str(limit):>4}  "
                  f"提供 {r['offered']:>3}  接收 {r['accepted']:>3}  "
                  f"拒绝率 {r['rejection_rate']:>5.1%}  队列峰值 {r.get('queue_peak', 0):>3}  "
                  f"完成 {r.get('completed', 0):>3}  {r.get('goodput_tok_s')} tok/s  "
                  f"等待 p50 {r.get('wait_p50_ms')} ms  p99 {r.get('wait_p99_ms')} ms"
                  f"（样本 {r.get('samples')}）  泄漏 {r.get('leak', {}).get('leaked')}"
                  f"{'  CRASH ' + r['crashed'] if r.get('crashed') else ''}")

    (args.out / "arrival_backpressure.json").write_text(
        json.dumps(dict(service_s=svc, base_rate=base_rate, blocks=args.blocks,
                        count=args.count, max_tokens=args.max_tokens,
                        max_seqs=args.max_seqs, prompt_tokens=len(ids),
                        rows=rows,
                        samples_note="每档请求数远低于 10000，p99 只作同口径排序参考，"
                                     "不作稳定性结论"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读法：上限=None 时超额到达只是排队变慢，拒绝率 0；")
    print("      上限存在时，过载被换成拒绝率，但被接收请求的等待时间与吞吐更稳。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
