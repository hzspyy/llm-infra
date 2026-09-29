#!/usr/bin/env python3
"""L5.7 任务 D —— 新增一种调度策略，记录跨模块改动与故障定位。

计划的验收是「抽象边界通过实际修改验证；不以接口层数或代码量判断架构优劣」，
做法是**真的加一种策略**，然后如实记录改到了哪些地方。

这里加的是**准入优先级**：等待队列里的请求按
`(-priority, arrival)` 出队，而不是先进先出。

关键观察（本次修改的全部内容）：`Engine._schedule()` 的准入顺序**完全由
`self.waiting` 这个 deque 的顺序决定**（`engine.py:273-283`：取 `waiting[0]`、
`popleft()`）。所以换策略不需要动 `_schedule` 的骨架，只要在它被调用前
把等待队列按新顺序排好：

    class PriorityEngine(Engine):
        def _schedule(self):
            if self.admission_policy == "priority" and len(self.waiting) > 1:
                ordered = sorted(self.waiting, key=lambda r: (-r.priority, r.arrival))
                self.waiting.clear(); self.waiting.extend(ordered)
            return super()._schedule()

这就是"抽象边界"的实际形状：**队列顺序是隐式接口**。
它带来两种后果，实验里都会看到：

  * 好处：加策略只碰一个私有方法，不动状态机；
  * 代价：任何直接操作 `self.waiting` 的代码（例如抢占把受害者
    `appendleft` 回队列）都会绕过这个顺序，策略就不是全局一致的——
    这类地方要靠读源码找出来，而不是靠类型检查。

实验：同一批 prompt、同样长度，一半高优先级一半低优先级，
比较两种策略下**两类请求的 TTFT 与完成顺序**，以及总吞吐有没有变化。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from engine import Engine, Request, State                     # noqa: E402
from failure import ResilientEngine                           # noqa: E402
from model import PagedModel                                  # noqa: E402

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
_MODEL = None


def get_model(num_blocks):
    global _MODEL
    if _MODEL is None or _MODEL.num_blocks != num_blocks:
        _MODEL = None
        import torch
        torch.cuda.empty_cache()
        _MODEL = PagedModel(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return _MODEL


class PriorityEngine(Engine):
    """只在准入前把等待队列排成优先级序；其余全部复用基类。"""

    def __init__(self, *a, admission_policy: str = "fcfs", **kw):
        super().__init__(*a, **kw)
        self.admission_policy = admission_policy
        self.reordered = 0

    def _schedule(self):
        if self.admission_policy == "priority" and len(self.waiting) > 1:
            ordered = sorted(self.waiting,
                             key=lambda r: (-getattr(r, "priority", 0), r.arrival))
            if list(self.waiting) != ordered:
                self.reordered += 1
            self.waiting.clear()
            self.waiting.extend(ordered)
        return super()._schedule()


class PriorityResilientEngine(ResilientEngine):
    """同一个重排，但底座是带抢占的引擎——用来验证"策略不是全局一致"。

    抢占（`failure.py:127`）把受害者 `self.waiting.appendleft(victim)` 插到队首，
    这一步**不经过** `_schedule` 里的排序，所以被抢占的请求会插到高优先级等待者前面。
    """

    def __init__(self, *a, admission_policy: str = "fcfs", **kw):
        super().__init__(*a, **kw)
        self.admission_policy = admission_policy
        self.reordered = 0

    def _schedule(self):
        if self.admission_policy == "priority" and len(self.waiting) > 1:
            ordered = sorted(self.waiting,
                             key=lambda r: (-getattr(r, "priority", 0), r.arrival))
            if list(self.waiting) != ordered:
                self.reordered += 1
            self.waiting.clear()
            self.waiting.extend(ordered)
        return super()._schedule()


class PriorityEnqueueOnlyEngine(ResilientEngine):
    """**只在入队时**排序的版本：策略挂在 `add()` 上，`_schedule` 不重排。

    这是"把策略写在入队口"的自然写法。它和 `PriorityResilientEngine`
    只在一点上不同：抢占把受害者 `waiting.appendleft(victim)` 插回队首时，
    没有机会再排一次——于是低优先级受害者会插到高优先级等待者前面。
    两个版本一起跑，就能把"隐式接口"从一句判断变成可观测的差别。
    """

    def __init__(self, *a, admission_policy: str = "fcfs", **kw):
        super().__init__(*a, **kw)
        self.admission_policy = admission_policy
        self.reordered = 0

    def add(self, req):
        super().add(req)
        if self.admission_policy == "priority" and len(self.waiting) > 1:
            ordered = sorted(self.waiting,
                             key=lambda r: (-getattr(r, "priority", 0), r.arrival))
            if list(self.waiting) != ordered:
                self.reordered += 1
            self.waiting.clear()
            self.waiting.extend(ordered)


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


def run(policy, tok, n_low, n_high, blocks, max_tokens, ids, cls=None,
        max_seqs=2):
    global eng_stats
    eng = (cls or PriorityEngine)(get_model(blocks), block_size=BLOCK_SIZE,
                         max_batched_tokens=256, max_num_seqs=max_seqs,
                         enable_prefix_cache=False, eos_ids=[],
                         admission_policy=policy)
    reqs = []
    # 低优先级先到（arrival 更早），高优先级后到——FCFS 下高优先级会排在后面
    t0 = time.perf_counter()
    for i in range(n_low):
        r = Request(f"low{i}", ids, max_tokens=max_tokens)
        r.priority = 0
        r.arrival = t0
        reqs.append(r)
        eng.add(r)
    for i in range(n_high):
        r = Request(f"high{i}", ids, max_tokens=max_tokens)
        r.priority = 10
        r.arrival = t0 + 0.001 * (i + 1)
        reqs.append(r)
        eng.add(r)

    crashed = None
    try:
        eng.run_until_idle(max_steps=20000)
    except Exception as e:              # 块池给太紧时基类会抛 OutOfBlocks
        crashed = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    eng_stats = getattr(eng, "stats", None)

    def ttft(r):
        return (r.first_token_at - r.arrival) * 1000 if r.first_token_at else None

    def collect(prefix):
        return [x for x in (ttft(r) for r in reqs if r.req_id.startswith(prefix))
                if x is not None]

    lows, highs = collect("low"), collect("high")

    def med(xs):
        return round(statistics.median(xs), 2) if xs else None

    def mx(xs):
        return round(max(xs), 2) if xs else None
    order = sorted((r for r in reqs if r.finished_at),
                   key=lambda r: r.finished_at)[:len(reqs)]
    return dict(policy=policy, wall_s=round(wall, 3), crashed=crashed,
                steps=eng.step_index,
                reordered_schedules=eng.reordered,
                n_low_with_token=len(lows), n_high_with_token=len(highs),
                low_ttft_median_ms=med(lows), high_ttft_median_ms=med(highs),
                low_ttft_max_ms=mx(lows), high_ttft_max_ms=mx(highs),
                completion_order=[r.req_id for r in order],
                preempted=getattr(getattr(eng, "stats", None), "preempted", 0),
                finished=sum(1 for r in reqs if r.state is State.FINISHED),
                leak=eng.leak_check()["leaked"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--low", type=int, default=4)
    ap.add_argument("--high", type=int, default=4)
    ap.add_argument("--max-seqs", type=int, default=2)
    ap.add_argument("--prompt-reps", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    ids = prompt_ids(tok, "Continue this story about a lighthouse keeper. " * args.prompt_reps)

    eng_stats = None
    rows = []
    plan = [("fcfs", None), ("priority", None),
            ("fcfs", PriorityResilientEngine),
            ("priority", PriorityResilientEngine),
            ("priority_enqueue_only", PriorityEnqueueOnlyEngine)]
    for policy, cls in plan:
        rec = run(policy, tok, args.low, args.high, args.blocks,
                  args.max_tokens, ids, cls=cls, max_seqs=args.max_seqs)
        rec["engine_class"] = (cls or PriorityEngine).__name__
        rows.append(rec)
        print(f"  {policy:<9} 墙钟 {rec['wall_s']:>6.3f}s  "
              f"低优先级 TTFT 中位 {rec['low_ttft_median_ms']:>8.1f} ms  "
              f"高优先级 TTFT 中位 {rec['high_ttft_median_ms']:>8.1f} ms  "
              f"重排次数 {rec['reordered_schedules']}  "
              f"抢占 {rec['preempted']}  泄漏 {rec['leak']}"
              f"{'  CRASH ' + rec['crashed'] if rec['crashed'] else ''}",
              flush=True)

    out = dict(blocks=args.blocks, max_seqs=args.max_seqs,
               prompt_reps=args.prompt_reps,
               max_tokens=args.max_tokens,
               n_low=args.low, n_high=args.high, rows=rows)
    (args.out / "priority_policy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    fcfs = rows[0]
    prio = rows[1]
    print(f"\n高优先级 TTFT 中位：FCFS {fcfs['high_ttft_median_ms']} ms → "
          f"priority {prio['high_ttft_median_ms']} ms；"
          f"低优先级 {fcfs['low_ttft_median_ms']} → {prio['low_ttft_median_ms']} ms")
    print("读法：优先级策略应当只把高优先级那类的 TTFT 压低，"
          "总墙钟/吞吐大致不变（这是调度重排，不是算力增加）。")


if __name__ == "__main__":
    main()
