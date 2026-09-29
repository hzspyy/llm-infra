#!/usr/bin/env python3
"""L5.3 任务 A · 离散事件调度器：FCFS / decode 优先 / 轮转。

5.3 的正文只讲到「预算怎么分配」的静态函数。修订计划要求把预算 mini
扩展成一个**离散事件调度器**，它持有真实的状态：

    arrival、computed/output tokens、KV 块、deadline、取消

并实现三种策略，用同一条请求轨迹跑，检查饥饿、资源预算与状态不变量。

这里不跑任何模型，也不产生 logits。它回答的是**计数与顺序**问题：
同样的到达与生成长度，不同策略下谁先出第一个 token、谁的间隔最长、
谁被抢占重算、谁在等、什么时候 KV 不够。3.3/5.1 的实测负责回答
「一步要多久」；本文件给出的 step 计数与 token 计数是它们的输入。

用法：
    python mini_scheduler.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: str
    arrival: int
    prompt: int
    output: int
    deadline: int | None = None        # 从到达到首个输出的最大容忍步数
    computed: int = 0                  # 已计算 token（含被抢占前的生成 token）
    generated: int = 0                 # 已产生输出 token
    blocks: list[int] = field(default_factory=list)
    state: str = "waiting"             # waiting / running / finished / cancelled
    first_sched_step: int | None = None
    first_token_step: int | None = None
    finish_step: int | None = None
    preempted: int = 0
    recompute_tokens: int = 0
    cancelled_reason: str | None = None

    @property
    def total_len(self) -> int:
        """被抢占后要重算的长度 = prompt + 已经生成的部分。"""
        return self.prompt + self.generated

    @property
    def remaining(self) -> int:
        return self.total_len - self.computed

    def __repr__(self) -> str:                                # pragma: no cover
        return (f"<{self.rid} {self.state} c={self.computed}/{self.total_len} "
                f"g={self.generated}/{self.output}>")


class BlockPool:
    """按块计数的 KV 池：请求持有块，抢占/结束时归还。"""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.bs = block_size
        self.free = list(range(num_blocks))
        self.owner: dict[int, str] = {}
        self.peak_used = 0

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)

    def blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.bs) if tokens else 0

    def take(self, req: Req, want: int) -> bool:
        """给请求补足到 want 块。不够就返回 False（由调度器决定抢占谁）。"""
        need = want - len(req.blocks)
        if need <= 0:
            return True
        if need > len(self.free):
            return False
        for _ in range(need):
            bid = self.free.pop()
            self.owner[bid] = req.rid
            req.blocks.append(bid)
        self.peak_used = max(self.peak_used, self.used)
        return True

    def release(self, req: Req) -> None:
        for bid in req.blocks:
            self.owner.pop(bid, None)
            self.free.append(bid)
        req.blocks = []


class Scheduler:
    """一个 step 一批：先按策略排候选，再按预算和 KV 容量切分工作量。"""

    def __init__(self, policy: str, budget: int, num_blocks: int,
                 block_size: int = 16, trace: list[Req] | None = None):
        assert policy in ("fcfs", "decode_priority", "round_robin",
                          "prefill_priority")
        self.policy = policy
        self.budget = budget
        self.pool = BlockPool(num_blocks, block_size)
        self.trace = trace or []
        self.waiting: list[Req] = []
        self.running: list[Req] = []
        self.finished: list[Req] = []
        self.cancelled: list[Req] = []
        self.cursor = 0
        self.steps = []
        self.arrivals_done = 0

    # -- 策略 -------------------------------------------------------------
    def candidates(self) -> list[Req]:
        """返回本步的候选顺序。

        * ``fcfs``：先到先服务，running 与 waiting 混在一起按到达时间排。
        * ``decode_priority``：已经进入 decode 的请求全部排在前面
          （原文 Orca 的 iteration-level 调度：decode 每步只花 1 个 token，
          让它先走完再喂 prefill，可以把批凑大）。
        * ``round_robin``：保持 running 优先，但 waiting 队列用游标轮转，
          避免固定顺序造成的饥饿。
        * ``prefill_priority``（反事实对照，不属于计划要求的三条）：所有 prefill
          排在 decode 前面。用它来量化「running 优先」这条规则值多少。
        """
        if self.policy == "fcfs":
            return sorted(self.running + self.waiting,
                          key=lambda r: (r.arrival, r.rid))
        if self.policy == "decode_priority":
            in_decode = [r for r in self.running if r.computed >= r.prompt]
            rest = [r for r in self.running if r.computed < r.prompt] + self.waiting
            return in_decode + sorted(rest, key=lambda r: (r.arrival, r.rid))
        if self.policy == "prefill_priority":
            pre = [r for r in self.running + self.waiting if r.computed < r.prompt]
            dec = [r for r in self.running + self.waiting if r.computed >= r.prompt]
            return (sorted(pre, key=lambda r: (r.arrival, r.rid))
                    + sorted(dec, key=lambda r: (r.arrival, r.rid)))
        # round_robin
        w = self.waiting
        if w:
            self.cursor %= len(w)
            w = w[self.cursor:] + w[:self.cursor]
            self.cursor = (self.cursor + 1) % max(1, len(w))
        return self.running + w

    # -- 一个 step ---------------------------------------------------------
    def step(self, t: int) -> dict:
        # 1. 新到达
        while (self.arrivals_done < len(self.trace)
               and self.trace[self.arrivals_done].arrival <= t):
            self.waiting.append(self.trace[self.arrivals_done])
            self.arrivals_done += 1

        # 2. deadline 取消（还没出第一个 token 的）
        for r in list(self.waiting):
            if (r.deadline is not None and r.first_token_step is None
                    and t - r.arrival > r.deadline):
                self._cancel(r, "deadline")

        # 3. 按预算分配
        budget = self.budget
        plan: list[tuple[Req, int]] = []
        planned: set[int] = set()          # 已进本步计划的请求不能被本步抢占
        order = self.candidates()
        for r in order:
            if budget == 0:
                break
            if r.state in ("finished", "cancelled"):
                continue
            if r.remaining <= 0:
                continue
            want = min(r.remaining, budget)
            # KV：按补齐后的 computed 长度要块
            want_blocks = self.pool.blocks_for(r.computed + want)
            if not self.pool.take(r, want_blocks):
                if not self._make_room(r, want_blocks, planned):
                    continue                       # 抢不动就别调度它
            plan.append((r, want))
            planned.add(id(r))
            budget -= want

        # 4. 执行
        rec = dict(step=t, scheduled={}, phases={}, produced=[], preemptions=0,
                   waiting=len(self.waiting), running=len(self.running))
        for r, n in plan:
            before = r.computed
            if r.first_sched_step is None:
                r.first_sched_step = t
            r.computed += n
            if before < r.total_len <= r.computed:
                if r.first_token_step is None:
                    r.first_token_step = t
                r.generated += 1
                rec["produced"].append(r.rid)
            elif before >= r.total_len:
                r.generated += 1
                rec["produced"].append(r.rid)
            if r.state == "waiting":
                r.state = "running"
                if r in self.waiting:
                    self.waiting.remove(r)
                self.running.append(r)
            rec["scheduled"][r.rid] = n
            rec["phases"][r.rid] = ("prefill" if before < r.total_len
                                    else "decode")
            if r.generated >= r.output:
                self._finish(r, t)
        rec["tokens"] = sum(rec["scheduled"].values())
        rec["blocks_used"] = self.pool.used
        rec["finished"] = len(self.finished)
        self.steps.append(rec)
        self._check_invariants(rec)
        return rec

    # -- 抢占 / 取消 / 完成 ------------------------------------------------
    def _victim(self, exclude: Req, planned: set[int]) -> Req | None:
        """选一个牺牲者。策略不同，选法不同。

        vLLM 默认是 LIFO（抢占最新的请求，让已经跑了很久的先跑完）；
        轮转策略这里改成先让**最老的**走完，用来对照「抢谁」对饥饿的影响。
        已经进入本步计划的请求不能当牺牲者：它的块已经被算进计划了。
        """
        pool = [r for r in self.running
                if r is not exclude and r.computed > 0 and id(r) not in planned]
        if not pool:
            return None
        if self.policy == "round_robin":
            return min(pool, key=lambda r: (r.arrival, r.rid))
        return max(pool, key=lambda r: (r.arrival, r.rid))         # LIFO：抢最新

    def _make_room(self, r: Req, want_blocks: int, planned: set[int]) -> bool:
        """按策略抢占正在跑的请求，直到够块。返回是否成功。"""
        guard = 0
        limit = len(self.trace) + 4
        while want_blocks - len(r.blocks) > len(self.pool.free):
            v = self._victim(r, planned)
            if v is None:
                return False
            self.pool.release(v)
            v.computed = 0                       # KV 被丢掉，必须整体重算
            v.preempted += 1
            v.recompute_tokens += v.total_len
            v.state = "waiting"
            if v in self.running:
                self.running.remove(v)
            self.waiting.append(v)
            guard += 1
            if guard > limit:
                return False
        return self.pool.take(r, want_blocks)

    def _finish(self, r: Req, t: int) -> None:
        r.state = "finished"
        r.finish_step = t
        self.pool.release(r)
        if r in self.running:
            self.running.remove(r)
        if r in self.waiting:
            self.waiting.remove(r)
        self.finished.append(r)

    def _cancel(self, r: Req, reason: str) -> None:
        r.state = "cancelled"
        r.cancelled_reason = reason
        self.pool.release(r)
        if r in self.waiting:
            self.waiting.remove(r)
        if r in self.running:
            self.running.remove(r)
        self.cancelled.append(r)

    # -- 不变量 -----------------------------------------------------------
    def _check_invariants(self, rec: dict) -> None:
        assert rec["tokens"] <= self.budget, "本步 token 超过预算"
        assert self.pool.used <= self.pool.num_blocks, "块使用超过容量"
        for r in self.trace:
            assert r.computed <= r.total_len, f"{r.rid} computed 超过总长"
            assert r.generated <= r.output, f"{r.rid} 生成超过上限"
            if r.state in ("finished", "cancelled"):
                assert not r.blocks, f"{r.rid} 已结束但仍占块"
                assert r not in self.running and r not in self.waiting
            if r.state == "running":
                assert len(r.blocks) * self.pool.bs >= r.computed, \
                    f"{r.rid} 的块装不下已计算长度"

    def run(self, max_steps: int = 40000) -> dict:
        t = 0
        while ((self.arrivals_done < len(self.trace) or self.waiting or self.running)
               and t < max_steps):
            self.step(t)
            t += 1
        return self.report(t)

    def report(self, steps: int) -> dict:
        emit = {r.rid: [] for r in self.trace}
        for s in self.steps:
            for rid in s["produced"]:
                emit[rid].append(s["step"])
        rows = []
        for r in self.trace:
            es = emit.get(r.rid, [])
            gaps = [es[i + 1] - es[i] for i in range(len(es) - 1)]
            rows.append(dict(
                rid=r.rid, arrival=r.arrival, prompt=r.prompt, output=r.output,
                state=r.state, ttft=(r.first_token_step - r.arrival
                                     if r.first_token_step is not None else None),
                queue_wait=(r.first_sched_step - r.arrival
                            if r.first_sched_step is not None else None),
                prefill_steps=((r.first_token_step - r.first_sched_step)
                               if r.first_token_step is not None
                               and r.first_sched_step is not None else None),
                first_token_step=r.first_token_step,
                finish_step=r.finish_step,
                tpot=((r.finish_step - r.first_token_step) / max(1, r.generated - 1)
                      if r.first_token_step is not None and r.finish_step is not None
                      and r.generated > 1 else None),
                max_gap_steps=(max(gaps) if gaps else None),
                preempted=r.preempted, recompute_tokens=r.recompute_tokens,
                cancelled_reason=r.cancelled_reason,
            ))
        waits = [x["ttft"] for x in rows if x["ttft"] is not None]
        qw = [x["queue_wait"] for x in rows if x["queue_wait"] is not None]
        return dict(policy=self.policy, budget=self.budget,
                    blocks=self.pool.num_blocks, steps=steps,
                    peak_blocks=self.pool.peak_used,
                    total_sched_tokens=sum(s["tokens"] for s in self.steps),
                    finished=len(self.finished), cancelled=len(self.cancelled),
                    max_ttft=(max(waits) if waits else None),
                    max_queue_wait=(max(qw) if qw else None),
                    max_finish=max((x["finish_step"] for x in rows
                                    if x["finish_step"] is not None), default=None),
                    requests=rows)


# ---------------------------------------------------------------- 轨迹
def trace_example() -> list[Req]:
    """正文里的两槽位例子：A/B 剩余 2/8，C 中途到达。"""
    return [Req("A", 0, 2, 2), Req("B", 0, 8, 2), Req("C", 2, 4, 2)]


def trace_long_insert(long_len: int = 8192, n_decode: int = 8,
                      decode_prompt: int = 64, decode_out: int = 64,
                      long_out: int = 8) -> list[Req]:
    """与 5.3 真实实验同构：8 条 decode 已在跑，t=8 插入一条长 prompt。"""
    rs = [Req(f"d{i}", 0, decode_prompt, decode_out) for i in range(n_decode)]
    rs.append(Req("LONG", 8, long_len, long_out))
    return rs


def trace_long_first(long_len: int = 8192, n_decode: int = 8,
                     decode_prompt: int = 64, decode_out: int = 64,
                     long_out: int = 8) -> list[Req]:
    """长 prompt 先到（t=0）、8 条 decode 后到（t=2）。

    这是 FCFS 与 decode 优先真正分道扬镳的轨迹：FCFS 会把预算一直给
    更早到达的长 prefill，decode 被压住；decode 优先反过来。
    """
    rs = [Req("LONG", 0, long_len, long_out)]
    rs += [Req(f"d{i}", 2, decode_prompt, decode_out) for i in range(n_decode)]
    return rs


def _i(x):
    """None 友好打印（未完成/未出首 token 记 -1）。"""
    return x if x is not None else -1


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def trace_arrival(rate: float, n: int, prompt: int = 64, output: int = 32) -> list[Req]:
    """确定性泊松式到达（固定种子线性同余，避免抖动）。"""
    rs, t, seed = [], 0, 12345
    for i in range(n):
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        gap = max(1, int(round(1.0 / rate * (0.5 + seed / (1 << 31)))))
        t += gap
        rs.append(Req(f"r{i}", t, prompt, output))
    return rs


def render(title: str, reports: list[dict]) -> str:
    lines = [f"\n=== {title} ==="]
    hdr = (f"  {'策略':<16}{'步数':>6}{'完成':>5}{'取消':>5}{'峰块':>6}"
           f"{'最大TTFT':>10}{'最大完成步':>11}{'抢占':>6}{'重算token':>10}")
    lines.append(hdr)
    for r in reports:
        pre = sum(x["preempted"] for x in r["requests"])
        rec = sum(x["recompute_tokens"] for x in r["requests"])
        lines.append(
            f"  {r['policy']:<16}{r['steps']:>6}{r['finished']:>5}{r['cancelled']:>5}"
            f"{r['peak_blocks']:>6}{r['max_ttft'] if r['max_ttft'] is not None else -1:>10}"
            f"{r['max_finish'] if r['max_finish'] is not None else -1:>11}"
            f"{pre:>6}{rec:>10}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out, rep = [], {}

    policies = ("fcfs", "decode_priority", "round_robin")
    out.append("L5.3-A 离散事件调度器：三条轨迹 × 三种策略（不跑模型，只看计数与顺序）")

    # 1) 两槽位例子：每条请求 1 token/步，预算 1 与 2 各跑一遍
    for budget in (1, 2):
        reports = [Scheduler(p, budget, num_blocks=8,
                             trace=trace_example()).run() for p in policies]
        out.append(render(f"轨迹 1：A(rem2) B(rem8) C(t=2, rem4)，预算 {budget}",
                          reports))
        for r in reports:
            out.append("    " + r["policy"] + " 完成顺序 " +
                       str([x["rid"] for x in sorted(
                           r["requests"], key=lambda q: (q["finish_step"]
                                                         if q["finish_step"] is not None
                                                         else 10 ** 9))]))
        rep[f"example_b{budget}"] = reports

    # 2) 长 prompt 插入：与真实实验同构，扫预算
    lines = ["\n=== 轨迹 2：8 条 decode（prompt 64、生成 64）+ t=8 插入 8192 长 prompt ==="]
    lines.append(f"  {'预算':>6}{'策略':<16}{'步数':>6}{'LONG TTFT':>10}"
                 f"{'LONG 完成':>10}{'d0 TPOT':>9}{'最大TTFT':>10}{'抢占':>5}")
    budget_rows = {}
    for budget in (128, 256, 512, 2048, 8192):
        budget_rows[budget] = {}
        for p in policies:
            r = Scheduler(p, budget, num_blocks=4096,
                          trace=trace_long_insert()).run()
            by = {x["rid"]: x for x in r["requests"]}
            L = by["LONG"]
            lines.append(
                f"  {budget:>6}{p:<16}{r['steps']:>6}{_i(L['ttft']):>10}"
                f"{_i(L['finish_step']):>10}"
                f"{(by['d0']['tpot'] if by['d0']['tpot'] is not None else float('nan')):>9.2f}"
                f"{_i(r['max_ttft']):>10}"
                f"{sum(x['preempted'] for x in r['requests']):>5}")
            budget_rows[budget][p] = r
    out.extend(lines)
    rep["long_insert"] = budget_rows

    # 2b) 长 prompt 先到、decode 后到：FCFS 与 decode 优先真正分开的轨迹
    lines = ["\n=== 轨迹 2b：长 prompt t=0 到达，8 条 decode t=2 到达（预算 512，KV 充足）==="]
    lines.append(f"  {'策略':<16}{'LONG TTFT':>10}{'LONG 完成':>10}"
                 f"{'decode TTFT中位':>16}{'decode 最大间隔':>16}"
                 f"{'decode 完成步中位':>18}{'抢占':>6}")
    longfirst = {}
    for p in policies:
        r = Scheduler(p, budget=512, num_blocks=4096,
                      trace=trace_long_first()).run()
        by = {x["rid"]: x for x in r["requests"]}
        ds = [x for x in r["requests"] if x["rid"] != "LONG"]
        lines.append(
            f"  {p:<16}{_i(by['LONG']['ttft']):>10}{_i(by['LONG']['finish_step']):>10}"
            f"{_i(_med([x['ttft'] for x in ds])):>16}"
            f"{_i(max(x['max_gap_steps'] or 0 for x in ds)):>16}"
            f"{_i(_med([x['finish_step'] for x in ds])):>18}"
            f"{sum(x['preempted'] for x in r['requests']):>6}")
        longfirst[p] = r
    out.extend(lines)
    rep["long_first"] = longfirst

    # 2c) 反事实：把「running 优先」换成「prefill 优先」，量化这条规则值多少
    lines = ["\n=== 轨迹 2c：8 条 decode 先跑，t=8 插入长 prompt（预算 512，KV 充足）==="]
    lines.append("  计划要求的三条策略 + 一条反事实策略 prefill_priority（所有 prefill 排在 decode 前）")
    lines.append(f"  {'策略':<18}{'LONG TTFT':>10}{'LONG 完成':>10}"
                 f"{'decode 最大间隔':>16}{'decode TTFT p90':>17}{'抢占':>6}")
    counterfactual = {}
    for p in policies + ("prefill_priority",):
        r = Scheduler(p, budget=512, num_blocks=4096,
                      trace=trace_long_insert()).run()
        by = {x["rid"]: x for x in r["requests"]}
        ds = [x for x in r["requests"] if x["rid"] != "LONG"]
        ttfts = sorted(x["ttft"] for x in ds if x["ttft"] is not None)
        lines.append(
            f"  {p:<18}{_i(by['LONG']['ttft']):>10}{_i(by['LONG']['finish_step']):>10}"
            f"{_i(max(x['max_gap_steps'] or 0 for x in ds)):>16}"
            f"{_i(ttfts[int(len(ttfts) * 0.9) - 1] if ttfts else None):>17}"
            f"{sum(x['preempted'] for x in r['requests']):>6}")
        counterfactual[p] = r
    out.extend(lines)
    rep["prefill_priority_counterfactual"] = counterfactual

    # 3) 到达率扫描：饥饿 / deadline / 取消。提示长、KV 小，故意制造争用
    SLO_STEPS = 30
    lines = [f"\n=== 轨迹 3：到达过程（prompt 256、生成 64，预算 64，KV 40 块 = 640 token；"
             f"SLO：TTFT ≤ {SLO_STEPS} 步）==="]
    lines.append(f"  {'到达率':>7}{'策略':<16}{'完成':>5}{'取消':>5}{'SLO内':>6}"
                 f"{'最长排队':>9}{'最大TTFT':>10}{'最大完成步':>11}{'抢占':>6}{'重算token':>10}")
    arrival_rows = {}
    for rate in (0.1, 0.2, 0.4):
        arrival_rows[rate] = {}
        for p in policies:
            tr = trace_arrival(rate, 40, prompt=256, output=64)
            for r_ in tr:
                r_.deadline = 200
            r = Scheduler(p, budget=64, num_blocks=40, trace=tr).run()
            arrival_rows[rate][p] = r
            pre = sum(x["preempted"] for x in r["requests"])
            rec = sum(x["recompute_tokens"] for x in r["requests"])
            in_slo = sum(1 for x in r["requests"]
                         if x["ttft"] is not None and x["ttft"] <= SLO_STEPS)
            lines.append(f"  {rate:>7}{p:<16}{r['finished']:>5}{r['cancelled']:>5}"
                         f"{in_slo:>6}"
                         f"{_i(r['max_queue_wait']):>9}"
                         f"{_i(r['max_ttft']):>10}"
                         f"{_i(r['max_finish']):>11}"
                         f"{pre:>6}{rec:>10}")
            reasons = {}
            for x in r["requests"]:
                if x["cancelled_reason"]:
                    reasons[x["cancelled_reason"]] = reasons.get(x["cancelled_reason"], 0) + 1
            if reasons:
                lines.append(f"          取消原因 {reasons}")
    out.extend(lines)
    rep["arrival"] = arrival_rows

    # 4) 饥饿检查：同一轨迹下各策略的 TTFT 分布
    lines = ["\n=== 饥饿检查（轨迹 3）：各策略的 TTFT 分布（步）==="]
    for rate in (0.1, 0.2, 0.4):
        for p in policies:
            waits = sorted(x["ttft"] for x in arrival_rows[rate][p]["requests"]
                           if x["ttft"] is not None)
            if waits:
                lines.append(f"  到达率 {rate}  {p:<16} n={len(waits):>3}  "
                             f"最小 {waits[0]:>4}  中位 {waits[len(waits) // 2]:>4}  "
                             f"p90 {waits[int(len(waits) * 0.9) - 1]:>4}  最大 {waits[-1]:>4}")
    out.extend(lines)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "mini_scheduler.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "mini_scheduler.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\n写入 {args.out}/mini_scheduler.txt 与 mini_scheduler.json")


if __name__ == "__main__":
    main()
