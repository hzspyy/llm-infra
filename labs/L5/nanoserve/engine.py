#!/usr/bin/env python3
"""nanoserve 引擎：请求状态机、块池、调度器。

三件事分开放：
  BlockPool  —— 显存怎么分（块、引用计数、前缀复用）
  Request    —— 一条请求在任何时刻的完整状态
  Scheduler  —— 每一步谁上车、给多少 token 预算

Engine.step() 是唯一的推进函数，跑一次等于引擎的一个 iteration。
每步返回一份 StepTrace，实验脚本靠它把内部状态打出来。
"""
from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch


class State(str, Enum):
    WAITING = "waiting"        # 在队列里，还没拿到块
    PREFILL = "prefill"        # 拿到块了，prompt 还没算完（可能分了好几个 chunk）
    DECODE = "decode"          # prompt 算完了，在一个一个吐 token
    FINISHED = "finished"      # 正常结束：EOS 或到 max_tokens
    ABORTED = "aborted"        # 被取消 / 超时 / 上游断开


class OutOfBlocks(Exception):
    pass


NONE_HASH = hashlib.sha256(b"nanoserve").digest()[:8]


def block_hash(parent: bytes, token_ids: tuple[int, ...]) -> bytes:
    """块哈希必须包含父块哈希，否则会跨前缀错误复用（见 5.2）。"""
    h = hashlib.sha256(parent)
    h.update(b",".join(str(t).encode() for t in token_ids))
    return h.digest()[:8]


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    content_hash: bytes | None = None


class BlockPool:
    """一个块池 + 一张哈希表。分配 = popleft，释放 = 引用计数归零后回到队尾。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free = deque(range(num_blocks))
        self.cached: dict[bytes, int] = {}
        self.stat_alloc = 0
        self.stat_reuse = 0

    @property
    def num_free(self) -> int:
        return len(self.free)

    def _detach(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        if blk.content_hash is not None and self.cached.get(blk.content_hash) == block_id:
            del self.cached[blk.content_hash]
        blk.content_hash = None

    def allocate(self) -> int:
        if not self.free:
            raise OutOfBlocks("块池空了")
        block_id = self.free.popleft()
        self._detach(block_id)
        self.blocks[block_id].ref_count = 1
        self.stat_alloc += 1
        return block_id

    def reuse(self, content_hash: bytes) -> int | None:
        """命中前缀缓存：块还在池子里（可能在空闲链上），引用计数 +1 即可。"""
        block_id = self.cached.get(content_hash)
        if block_id is None:
            return None
        blk = self.blocks[block_id]
        if blk.ref_count == 0:
            self.free.remove(block_id)         # 从空闲链里救回来
        blk.ref_count += 1
        self.stat_reuse += 1
        return block_id

    def publish(self, block_id: int, content_hash: bytes) -> None:
        """块填满之后才登记进哈希表——没填满的块不能被别人复用。"""
        blk = self.blocks[block_id]
        blk.content_hash = content_hash
        self.cached.setdefault(content_hash, block_id)

    def release(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        blk.ref_count -= 1
        if blk.ref_count < 0:
            raise AssertionError(f"block {block_id} 引用计数变成负数")
        if blk.ref_count == 0:
            # 有内容哈希的块留在缓存里，等着被复用；被别人抢走时才真正清掉。
            self.free.append(block_id)

    def snapshot(self) -> dict:
        return {"free": self.num_free, "total": len(self.blocks),
                "cached": len(self.cached),
                "refcounts": {b.block_id: b.ref_count
                              for b in self.blocks if b.ref_count},
                "alloc": self.stat_alloc, "reuse": self.stat_reuse}


@dataclass
class Request:
    req_id: str
    prompt_ids: list[int]
    max_tokens: int = 32
    stop_ids: tuple[int, ...] = ()
    state: State = State.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed: int = 0                    # prompt 里已经算进 KV 的 token 数
    output_ids: list[int] = field(default_factory=list)
    arrival: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None
    cached_prefix_blocks: int = 0
    prefill_chunks: int = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def kv_len(self) -> int:
        """已经写进 KV 的 token 数。

        最新采样出来的那个 token 还没有 KV——它要等下一步被喂进去才写，
        所以 output_ids 里只有最后一个不算数。差这一个位置就会串位。
        """
        if not self.output_ids:
            return self.num_computed
        return self.num_computed + len(self.output_ids) - 1

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class StepTrace:
    step: int
    prefill: list[str] = field(default_factory=list)
    prefill_tokens: int = 0
    decode: list[str] = field(default_factory=list)
    decode_tokens: int = 0
    budget: int = 0
    pool: dict = field(default_factory=dict)
    finished: list[str] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""


class Engine:
    def __init__(self, model, block_size=16, max_batched_tokens=256,
                 max_num_seqs=8, enable_prefix_cache=True, eos_ids=()):
        self.model = model
        self.block_size = block_size
        self.pool = BlockPool(model.num_blocks, block_size)
        self.max_batched_tokens = max_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_cache = enable_prefix_cache
        self.eos_ids = tuple(eos_ids)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.done: dict[str, Request] = {}
        self.step_index = 0
        self.traces: list[StepTrace] = []

    # ---------------------------------------------------------------- 准入
    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def _grow_block_table(self, req: Request, upto: int) -> None:
        """保证 block_table 能装下 upto 个 token。只在需要时分配。"""
        need = (upto + self.block_size - 1) // self.block_size
        while len(req.block_table) < need:
            req.block_table.append(self.pool.allocate())

    def _try_prefix_cache(self, req: Request) -> None:
        """按块滚动哈希，命中就直接接上别人的块，命中的部分不用再算。

        复用来的块是**只读共享**的，所以复用只能停在块边界上。
        如果整段 prompt 都命中，必须退掉最后一块重算——
        否则这一步没有任何 token 要前向，采样就拿不到 logits。
        """
        if not self.enable_prefix_cache:
            return
        parent, hit = NONE_HASH, 0
        for b in range(len(req.prompt_ids) // self.block_size):
            chunk = tuple(req.prompt_ids[b * self.block_size:(b + 1) * self.block_size])
            h = block_hash(parent, chunk)
            block_id = self.pool.reuse(h)
            if block_id is None:
                break
            req.block_table.append(block_id)
            parent, hit = h, hit + 1
        if hit and hit * self.block_size >= len(req.prompt_ids):
            self.pool.release(req.block_table.pop())
            self.pool.stat_reuse -= 1
            hit -= 1
        req.cached_prefix_blocks = hit
        req.num_computed = hit * self.block_size

    def _publish_full_blocks(self, req: Request) -> None:
        if not self.enable_prefix_cache:
            return
        parent = NONE_HASH
        ids = req.all_ids
        for b in range(min(len(req.block_table), req.kv_len // self.block_size)):
            chunk = tuple(ids[b * self.block_size:(b + 1) * self.block_size])
            parent = block_hash(parent, chunk)
            self.pool.publish(req.block_table[b], parent)

    def _free(self, req: Request) -> None:
        for block_id in req.block_table:
            self.pool.release(block_id)
        req.block_table = []

    def abort(self, req_id: str, reason: str = "aborted") -> bool:
        """取消：从队列或运行集里摘掉，释放块。5.8 会沿这条路径继续往上接。"""
        for q in (self.waiting,):
            for r in list(q):
                if r.req_id == req_id:
                    q.remove(r)
                    r.state, r.finish_reason = State.ABORTED, reason
                    r.finished_at = time.perf_counter()
                    self.done[req_id] = r
                    return True
        for r in list(self.running):
            if r.req_id == req_id:
                self.running.remove(r)
                self._free(r)
                r.state, r.finish_reason = State.ABORTED, reason
                r.finished_at = time.perf_counter()
                self.done[req_id] = r
                return True
        return False

    # ---------------------------------------------------------------- 调度
    def _schedule(self):
        """先安排 decode（它们已经占着块了），剩下的预算给 prefill chunk。"""
        budget = self.max_batched_tokens
        decode = [r for r in self.running if r.state is State.DECODE]
        decode = decode[:budget]
        budget -= len(decode)

        prefill: list[tuple[Request, int]] = []
        # 已经在跑但 prompt 没算完的，优先接着算。
        pending = [r for r in self.running if r.state is State.PREFILL]
        while budget > 0 and pending:
            r = pending.pop(0)
            take = min(budget, len(r.prompt_ids) - r.num_computed)
            if take <= 0:
                continue
            prefill.append((r, take))
            budget -= take
        # 再从等待队列里放新的进来。
        while (budget > 0 and self.waiting
               and len(self.running) < self.max_num_seqs):
            r = self.waiting[0]
            try:
                self._try_prefix_cache(r)
                self._grow_block_table(r, len(r.prompt_ids))
            except OutOfBlocks:
                self._free(r)
                r.num_computed, r.cached_prefix_blocks = 0, 0
                break                       # 块不够，这一步就不收新请求了
            self.waiting.popleft()
            r.state = State.PREFILL
            self.running.append(r)
            take = min(budget, len(r.prompt_ids) - r.num_computed)
            prefill.append((r, take))
            budget -= take
        return prefill, decode, budget

    # ---------------------------------------------------------------- 前向
    def _slots(self, req: Request, start: int, count: int) -> list[int]:
        out = []
        for p in range(start, start + count):
            out.append(req.block_table[p // self.block_size] * self.block_size
                       + p % self.block_size)
        return out

    def _gather(self, req: Request, ctx_len: int, width: int) -> list[int]:
        idx = self._slots(req, 0, ctx_len)
        return idx + [idx[-1]] * (width - ctx_len)     # padding 指向已有槽位，靠 mask 屏蔽

    def _run(self, groups: list[tuple[Request, int, int]]):
        """groups: (请求, 本步起始位置, 本步 token 数)。返回每条序列最后一个位置的 logits。"""
        dev = self.model.device
        Q = max(n for _, _, n in groups)
        C = max(start + n for _, start, n in groups)
        B = len(groups)
        input_ids = torch.zeros((B, Q), dtype=torch.long)
        positions = torch.zeros((B, Q), dtype=torch.long)
        slot_map = torch.full((B, Q), -1, dtype=torch.long)
        gather = torch.zeros((B, C), dtype=torch.long)
        ctx_lens = torch.zeros(B, dtype=torch.long)
        last = []
        for b, (req, start, n) in enumerate(groups):
            ids = req.all_ids[start:start + n]
            input_ids[b, :n] = torch.tensor(ids)
            positions[b, :n] = torch.arange(start, start + n)
            slot_map[b, :n] = torch.tensor(self._slots(req, start, n))
            ctx_lens[b] = start + n
            gather[b] = torch.tensor(self._gather(req, start + n, C))
            last.append(n - 1)
        logits = self.model.forward(
            input_ids.to(dev), positions.to(dev), slot_map.to(dev),
            gather.to(dev), ctx_lens.to(dev))
        return logits[torch.arange(B, device=dev),
                      torch.tensor(last, device=dev)]        # [B, V]

    # ---------------------------------------------------------------- 一步
    def step(self) -> StepTrace:
        t0 = time.perf_counter()
        prefill, decode, left = self._schedule()
        trace = StepTrace(step=self.step_index, budget=self.max_batched_tokens)
        self.step_index += 1
        if not prefill and not decode:
            trace.note = "空转"
            trace.pool = self.pool.snapshot()
            self.traces.append(trace)
            return trace

        # ---- prefill 组：可能只是 prompt 的一段 ----
        if prefill:
            groups = [(r, r.num_computed, n) for r, n in prefill]
            logits = self._run(groups)
            trace.prefill = [r.req_id for r, _ in prefill]
            trace.prefill_tokens = sum(n for _, n in prefill)
            for b, (r, n) in enumerate(prefill):
                r.num_computed += n
                r.prefill_chunks += 1
                if r.num_computed >= len(r.prompt_ids):
                    r.state = State.DECODE
                    self._emit(r, logits[b], trace)
                    self._publish_full_blocks(r)

        # ---- decode 组：每条 1 个 token ----
        if decode:
            groups = [(r, r.kv_len, 1) for r in decode]
            for r in decode:
                self._grow_block_table(r, r.kv_len + 1)
            logits = self._run(groups)
            trace.decode = [r.req_id for r in decode]
            trace.decode_tokens = len(decode)
            for b, r in enumerate(decode):
                self._emit(r, logits[b], trace)
                self._publish_full_blocks(r)

        trace.pool = self.pool.snapshot()
        trace.seconds = time.perf_counter() - t0
        self.traces.append(trace)
        return trace

    def _emit(self, req: Request, logits, trace: StepTrace) -> None:
        token = int(logits.argmax())          # 贪心；采样见 5.9
        req.output_ids.append(token)
        if req.first_token_at is None:
            req.first_token_at = time.perf_counter()
        if token in self.eos_ids:
            self._finish(req, "stop", trace)
        elif len(req.output_ids) >= req.max_tokens:
            self._finish(req, "length", trace)

    def _finish(self, req: Request, reason: str, trace: StepTrace) -> None:
        req.state, req.finish_reason = State.FINISHED, reason
        req.finished_at = time.perf_counter()
        self.running.remove(req)
        self._free(req)
        self.done[req.req_id] = req
        trace.finished.append(req.req_id)

    def run_until_idle(self, max_steps=10_000) -> None:
        for _ in range(max_steps):
            if not self.waiting and not self.running:
                return
            self.step()
        raise RuntimeError("超过 max_steps 仍未跑空")

    def leak_check(self) -> dict:
        """所有请求结束后，块池必须回到初始状态。"""
        held = {b.block_id: b.ref_count for b in self.pool.blocks if b.ref_count}
        return {"held_blocks": held, "free": self.pool.num_free,
                "total": len(self.pool.blocks), "leaked": len(held)}
