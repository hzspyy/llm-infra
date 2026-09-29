#!/usr/bin/env python3
"""L5.8 任务 B —— 在 nanoserve 上实现「换出到主机」并给出与重算的同口径对照。

真实引擎的现状先固定下来（`vllm/v1/core/sched/scheduler.py:1405` `_preempt_request`）：
抢占时只做 `_free_request_blocks(request)` 并把 `request.num_computed_tokens = 0`，
即**整段前缀重算**，没有 swap-out / swap-in 这一步；本版本里唯一带 CPU 块的机制是
`vllm/v1/simple_kv_offload/`（显式 KV offload 连接器，不是抢占路径）。
所以「换出」这条路径只能在 nanoserve 上按协议实现并测量——这也正是本章的任务 B 要求。

`SwappingEngine` 在 `ResilientEngine` 之上把抢占从「丢弃」换成「写回主机」：

  * 换出：把该请求块表覆盖的槽位从 `model.k_cache`/`v_cache` 里 gather 出来，
    拷进 **pinned host** 缓冲，然后释放块；`num_computed` 与 `output_ids` 全部保留。
  * 换入：重新分配块，把保存的字节按新块号 scatter 回同样的槽位，
    恢复 `num_computed`，请求直接从 decode 续跑，**不重算**。

受害者策略可换：newest / oldest / longest_kv / shortest_kv。
两类统计都记：换出的请求数、字节、耗时；重算路径的浪费 token 数。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from engine import OutOfBlocks, Request, State
from failure import ResilientEngine


@dataclass
class SwapStats:
    swapped_out: int = 0
    swapped_in: int = 0
    bytes_out: int = 0
    bytes_in: int = 0
    swap_out_s: float = 0.0
    swap_in_s: float = 0.0
    victims: list = field(default_factory=list)


class SwappingEngine(ResilientEngine):
    def __init__(self, *args, victim_policy: str = "newest", **kw):
        super().__init__(*args, **kw)
        self.victim_policy = victim_policy
        self.host: dict[str, dict] = {}          # req_id -> 保存的状态
        self.swap = SwapStats()
        self._pin = torch.cuda.is_available()

    # ------------------------------------------------------------ 受害者选择
    def _pick_victim(self) -> Request | None:
        cands = [r for r in self.running if r.state is State.DECODE] or list(self.running)
        if not cands:
            return None
        pol = self.victim_policy
        if pol == "newest":
            return max(cands, key=lambda r: r.arrival)
        if pol == "oldest":
            return min(cands, key=lambda r: r.arrival)
        if pol == "longest_kv":
            return max(cands, key=lambda r: r.kv_len)
        if pol == "shortest_kv":
            return min(cands, key=lambda r: r.kv_len)
        raise ValueError(f"未知受害者策略 {pol}")

    # ------------------------------------------------------------ 换出 / 换入
    def _block_slots(self, block_table, n_tokens=None):
        """把块表展开成槽位下标。注意不要用 _slots 这个名字——
        `Engine._slots(req, start, count)` 是基类的另一件事。"""
        bs = self.block_size
        n = len(block_table) * bs if n_tokens is None else n_tokens
        out = []
        for b in block_table:
            out.extend(b * bs + i for i in range(bs))
        return out[:n]

    def _swap_out(self, req: Request) -> None:
        """把 KV 写回 pinned host，然后释放块。"""
        dev = self.model.device
        idx = torch.tensor(self._block_slots(req.block_table, req.kv_len), device=dev)
        t0 = time.perf_counter()
        k = self.model.k_cache[:, idx]                       # [L, n, n_kv, hd]
        v = self.model.v_cache[:, idx]
        k_cpu = torch.empty_like(k, device="cpu", pin_memory=self._pin)
        v_cpu = torch.empty_like(v, device="cpu", pin_memory=self._pin)
        k_cpu.copy_(k, non_blocking=True)
        v_cpu.copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nbytes = (k.numel() + v.numel()) * k.element_size()
        self.host[req.req_id] = dict(k=k_cpu, v=v_cpu, kv_len=req.kv_len,
                                     num_computed=req.num_computed,
                                     output_ids=list(req.output_ids),
                                     state=req.state)
        blocks = len(req.block_table)
        self._free(req)
        self.running.remove(req)
        req.state = State.WAITING
        self.waiting.appendleft(req)
        self.swap.swapped_out += 1
        self.swap.bytes_out += nbytes
        self.swap.swap_out_s += dt
        self.swap.victims.append(dict(req=req.req_id, kv_tokens=req.kv_len,
                                      blocks=blocks, bytes=nbytes,
                                      swap_out_ms=round(dt * 1000, 4),
                                      state_at_evict="decode"))
        self.stats.log("swap_out", req=req.req_id, kv_tokens=req.kv_len,
                       blocks=blocks, bytes=nbytes, ms=round(dt * 1000, 4),
                       free=self.pool.num_free)

    def _swap_in(self, req: Request, upto: int) -> None:
        """恢复块与 KV 内容，使请求不重算。

        必须在**任何改动之前**把失败条件判完：块不够就原样抛出，
        不动 `req`、不 pop host 条目。否则调用方的 except 会把
        `num_computed` 归零，而 KV 已经丢了，请求就永远回不来。
        """
        dev = self.model.device
        saved = self.host.get(req.req_id)
        if saved is None:
            super()._grow_block_table(req, upto)
            return
        n_tokens = saved["kv_len"]
        need = (n_tokens + self.block_size - 1) // self.block_size
        have = len(req.block_table)
        # 与准入路径同一道 watermark：不能把正在跑的请求要用的块分光
        # +1：被换入的这条请求自己下一步也要一块，而 _running_reserve()
        # 只数当前已经在跑的 decode 请求，少算它就会在同一个 step 的
        # decode 扩容里把块用光。
        if self.pool.num_free - (need - have) < self._running_reserve() + 1:
            raise OutOfBlocks("watermark：换入要给正在跑的请求留块")
        new_blocks: list[int] = []
        try:
            while have + len(new_blocks) < need:
                new_blocks.append(self.pool.allocate())
        except OutOfBlocks:
            for b in new_blocks:
                self.pool.release(b)
            raise
        req.block_table.extend(new_blocks)
        idx = torch.tensor(self._block_slots(req.block_table, n_tokens), device=dev)
        t0 = time.perf_counter()
        k = saved["k"].to(dev, non_blocking=True)
        v = saved["v"].to(dev, non_blocking=True)
        self.model.k_cache[:, idx] = k
        self.model.v_cache[:, idx] = v
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nbytes = (k.numel() + v.numel()) * k.element_size()
        self.host.pop(req.req_id)
        req.num_computed = saved["num_computed"]
        req.output_ids = saved["output_ids"]
        req.state = State.DECODE
        req.cached_prefix_blocks = 0
        self.swap.swapped_in += 1
        self.swap.bytes_in += nbytes
        self.swap.swap_in_s += dt
        self.stats.log("swap_in", req=req.req_id, kv_tokens=n_tokens,
                       bytes=nbytes, ms=round(dt * 1000, 4),
                       free=self.pool.num_free)

    # ------------------------------------------------------------ 抢占入口
    def _preempt_one(self) -> str | None:
        victim = self._pick_victim()
        if victim is None:
            return None
        self._swap_out(victim)
        self.stats.preempted += 1
        # 换出不走重算，所以 recomputed_tokens 不增加
        return victim.req_id

    # ------------------------------------------------------------ 准入时恢复
    def _grow_block_table(self, req: Request, upto: int) -> None:
        if req.req_id in self.host:
            self._swap_in(req, upto)
            return
        super()._grow_block_table(req, upto)

    def _schedule(self):
        """基类版本会把恢复后的请求强行标成 PREFILL 并排一个 0 token 的组。

        换入的请求 prompt 已经算完（`num_computed == len(prompt_ids)`），
        再按 prefill 排就会得到长度 0 的组，`_run` 直接 IndexError。
        这里只加一处判断：take 为 0 的请求直接进 decode 组，不进 prefill 组。
        其余逻辑与 `Engine._schedule` 一致。
        """
        budget = self.max_batched_tokens
        decode = [r for r in self.running if r.state is State.DECODE]
        decode = decode[:budget]
        budget -= len(decode)

        prefill: list[tuple[Request, int]] = []
        pending = [r for r in self.running if r.state is State.PREFILL]
        while budget > 0 and pending:
            r = pending.pop(0)
            take = min(budget, len(r.prompt_ids) - r.num_computed)
            if take <= 0:
                r.state = State.DECODE
                if r not in decode:
                    decode.append(r)
                continue
            prefill.append((r, take))
            budget -= take
        while (budget > 0 and self.waiting
               and len(self.running) < self.max_num_seqs):
            r = self.waiting[0]
            try:
                self._try_prefix_cache(r)
                self._grow_block_table(r, len(r.prompt_ids))
            except OutOfBlocks:
                self._free(r)
                r.num_computed, r.cached_prefix_blocks = 0, 0
                break
            self.waiting.popleft()
            take = min(budget, len(r.prompt_ids) - r.num_computed)
            if take <= 0:
                # 换入的请求：prompt 已算完，直接续 decode
                r.state = State.DECODE
                self.running.append(r)
                decode.append(r)
                continue
            r.state = State.PREFILL
            self.running.append(r)
            prefill.append((r, take))
            budget -= take
        return prefill, decode, budget
