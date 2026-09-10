#!/usr/bin/env python3
"""L5.8：在 5.7 的 nanoserve 上补四条失败路径。

5.7 的 `Engine` 只处理正常结束和显式取消，块池耗尽时直接抛 `OutOfBlocks`。
这里用子类补上：

  抢占  —— 块不够时踢掉最新的请求，把块还回去，它稍后重算
  超时  —— 每条请求带一个 deadline，到点无论在哪个状态都 abort
  背压  —— 等待队列有上限，满了就拒绝，而不是无限排队
  泄漏  —— 一个开关，故意让 abort 忘记释放块，用来验证泄漏检查真的会响

engine.py 不改，5.7 的产物因此保持可复现。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from engine import Engine, OutOfBlocks, Request, State, StepTrace


@dataclass
class FailureStats:
    preempted: int = 0
    recomputed_tokens: int = 0
    timed_out: int = 0
    rejected: int = 0
    aborted: int = 0
    events: list = field(default_factory=list)

    def log(self, kind, **fields):
        self.events.append({"t": round(time.perf_counter(), 6), "kind": kind, **fields})


class ResilientEngine(Engine):
    def __init__(self, *args, max_waiting: int | None = None,
                 leak_on_abort: bool = False, **kw):
        super().__init__(*args, **kw)
        self.max_waiting = max_waiting
        self.leak_on_abort = leak_on_abort
        self.stats = FailureStats()

    # ------------------------------------------------------------ 背压
    def try_add(self, req: Request) -> bool:
        """队列满就拒绝。返回 False 时上层应该回 503，而不是让请求干等。"""
        if self.max_waiting is not None and len(self.waiting) >= self.max_waiting:
            self.stats.rejected += 1
            req.state, req.finish_reason = State.ABORTED, "queue-full"
            req.finished_at = time.perf_counter()
            self.done[req.req_id] = req
            self.stats.log("reject", req=req.req_id, waiting=len(self.waiting),
                           limit=self.max_waiting)
            return False
        self.add(req)
        return True

    # ------------------------------------------------------------ 取消
    def abort(self, req_id: str, reason: str = "aborted") -> bool:
        """加一层统计；leak_on_abort 打开时故意不还块（用来验泄漏检查）。"""
        where = None
        for r in self.waiting:
            if r.req_id == req_id:
                where = r.state.value
                break
        if where is None:
            for r in self.running:
                if r.req_id == req_id:
                    where = r.state.value
                    break
        if where is None:
            return False
        held = 0
        if self.leak_on_abort:
            target = next((r for r in list(self.waiting) + self.running
                           if r.req_id == req_id), None)
            held = len(target.block_table)
            # 只把请求摘出去，块留在原地：这是最常见的一种泄漏写法。
            if target in self.waiting:
                self.waiting.remove(target)
            else:
                self.running.remove(target)
            target.state, target.finish_reason = State.ABORTED, reason
            target.finished_at = time.perf_counter()
            target.block_table = []            # 表清空了，引用计数没减
            self.done[req_id] = target
            ok = True
        else:
            free_before = self.pool.num_free
            ok = super().abort(req_id, reason)
            held = self.pool.num_free - free_before
        self.stats.aborted += 1
        self.stats.log("abort", req=req_id, from_state=where, reason=reason,
                       blocks_returned=held, free=self.pool.num_free,
                       leak_mode=self.leak_on_abort)
        return ok

    # ------------------------------------------------------------ 超时
    def check_deadlines(self, now: float | None = None) -> list[str]:
        now = now or time.perf_counter()
        hit = []
        for r in list(self.waiting) + list(self.running):
            deadline = getattr(r, "deadline", None)
            if deadline is not None and now > deadline:
                state = r.state.value
                self.abort(r.req_id, "timeout")
                self.stats.timed_out += 1
                self.stats.log("timeout", req=r.req_id, from_state=state,
                               waited_ms=round((now - r.arrival) * 1000, 2))
                hit.append(r.req_id)
        return hit

    # ------------------------------------------------------------ 抢占
    def _preempt_one(self) -> str | None:
        """踢掉最新加入的运行请求：它已经算的最少，重算最便宜。"""
        victims = [r for r in self.running if r.state is State.DECODE] or self.running
        if not victims:
            return None
        victim = max(victims, key=lambda r: r.arrival)
        wasted = victim.kv_len
        blocks = len(victim.block_table)
        self._free(victim)
        self.running.remove(victim)
        victim.state = State.WAITING
        victim.num_computed = 0
        victim.output_ids.clear()          # 重算：已生成的 token 一并丢弃
        victim.cached_prefix_blocks = 0
        self.waiting.appendleft(victim)
        self.stats.preempted += 1
        self.stats.recomputed_tokens += wasted
        self.stats.log("preempt", req=victim.req_id, freed_blocks=blocks,
                       discarded_kv_tokens=wasted, free=self.pool.num_free)
        return victim.req_id

    def _running_reserve(self) -> int:
        """正在 decode 的请求，下一步每条最多再要一块。这些块必须留住。"""
        return sum(1 for r in self.running if r.state is State.DECODE)

    def _decode_blocks_needed(self) -> int:
        """这一步真正会跨进新一块的 decode 请求数。"""
        return sum(1 for r in self.running if r.state is State.DECODE
                   and r.kv_len % self.block_size == 0)

    def _grow_block_table(self, req: Request, upto: int) -> None:
        """准入路径上加一道 watermark：不能把正在跑的请求要用的块分光。

        没有这道闸，抢占会变成空转：刚腾出来的块在同一个 step 里
        被 `_schedule()` 的准入立刻吃掉，下一步照样不够。
        vLLM 的 watermark 解决的是同一个问题。
        """
        if req.state is State.WAITING:
            need = ((upto + self.block_size - 1) // self.block_size
                    - len(req.block_table))
            if self.pool.num_free - need < self._running_reserve():
                self.stats.log("admission_blocked", req=req.req_id, need=need,
                               free=self.pool.num_free,
                               reserve=self._running_reserve())
                raise OutOfBlocks("watermark：要给正在跑的请求留块")
        super()._grow_block_table(req, upto)

    def step(self) -> StepTrace:
        """先确认这一步的块够用，不够就抢占——检查发生在任何前向之前。

        5.7 的 step 把「长块表」放在前向前面，所以 OutOfBlocks 会在
        prefill 已经算完之后才抛出来，状态半新半旧。放到这里检查，
        抢占就永远发生在一个干净的边界上。
        """
        while self._decode_blocks_needed() > self.pool.num_free:
            if self._preempt_one() is None:
                break
        try:
            return super().step()
        except OutOfBlocks:
            self.stats.log("out_of_blocks", free=self.pool.num_free,
                           running=len(self.running), waiting=len(self.waiting))
            raise


def attach_deadline(req: Request, seconds: float) -> Request:
    req.deadline = req.arrival + seconds
    return req
