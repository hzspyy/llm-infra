#!/usr/bin/env python3
"""L5.2 · 自己写一遍：最小可用的分页 KV 分配器 + 前缀缓存。

这个模型演示块引用、前缀查找和淘汰，不执行 attention，也不保存真实 KV。
块粒度的预测用于理解数据结构；它不能替代真实引擎的逐请求命中记录，
也没有模拟生成 logits 所需的末 token 重算。

对照的是 vLLM 的三件核心机制：
    KVCacheBlock.ref_cnt            -> self.ref_cnt
    FreeKVCacheBlockQueue（LRU 双链）-> self.free（这里用 OrderedDict 近似）
    BlockPool.cached_block_hash_to_block -> self.hash_to_block
    hash_block_tokens（滚动哈希）   -> self._rolling_hash

直接跑：
    python mini_block_pool.py            # 跑自测 + 阶梯模拟
"""

from __future__ import annotations

import random
from collections import OrderedDict


class MiniBlockPool:
    """分页 KV 分配器。约 70 行，但三个不变量必须成立：

    1. 一个块只有**填满**才会被哈希、才可复用（复用粒度 = 块）
    2. 块哈希必须**滚动**包含父块哈希（KV 是上下文相关的）
    3. ref_cnt 归零 ≠ 内容失效；真正失效发生在被重新分配时
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.ref_cnt = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        # OrderedDict 当作 LRU 队列：键是 block_id，最早进来的在最前面。
        # 真实实现用手写双向链表，是为了 O(1) 从中间摘除且不分配 Python 对象。
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.hash_to_block: dict[int, int] = {}
        # 统计
        self.stat_queries = 0
        self.stat_hits = 0

    # -- 哈希 --------------------------------------------------------------

    @staticmethod
    def _rolling_hash(parent: int | None, tokens: tuple[int, ...]) -> int:
        """父块哈希进入本块哈希。

        不这样做会有**正确性 bug**：同样 16 个 token 跟在不同前缀后面，
        算出的 K/V 完全不同，误复用会静默产生错误输出。
        """
        return hash((parent, tokens))

    def block_hashes_of(self, token_ids: list[int]) -> list[int]:
        """把一串 token 切成满块并算出每块的滚动哈希。不满的尾块直接丢弃。"""
        out, parent = [], None
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            chunk = tuple(token_ids[i * self.block_size:(i + 1) * self.block_size])
            parent = self._rolling_hash(parent, chunk)
            out.append(parent)
        return out

    # -- 查询 --------------------------------------------------------------

    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        """返回 (可复用的物理块列表, 已被覆盖的 token 数)。

        注意里面那个 break：**前缀一断，后面全部作废**——
        即使后续块内容相同，它们的滚动哈希也已经不同了。
        """
        blocks: list[int] = []
        covered = 0
        for h in self.block_hashes_of(token_ids):
            self.stat_queries += 1
            bid = self.hash_to_block.get(h)
            if bid is None:
                break
            self.stat_hits += 1
            self.ref_cnt[bid] += 1
            self.free.pop(bid, None)      # 命中一个"空闲但仍缓存"的块 → 从空闲队列摘掉
            blocks.append(bid)
            covered += self.block_size
        return blocks, covered

    # -- 分配 / 释放 -------------------------------------------------------

    def alloc(self, n: int) -> list[int]:
        if n > len(self.free):
            raise MemoryError(
                f"需要 {n} 块，只剩 {len(self.free)} —— 真实引擎在这里会去抢占别的请求")
        out = []
        for _ in range(n):
            bid, _ = self.free.popitem(last=False)     # LRU：从头取最久未用的
            old = self.block_hash[bid]
            if old is not None:                        # 复用一个仍带缓存内容的块
                self.hash_to_block.pop(old, None)      # ← 内容此刻才真正失效
                self.block_hash[bid] = None
            self.ref_cnt[bid] = 1
            out.append(bid)
        return out

    def cache_full_blocks(self, blocks: list[int], hashes: list[int]) -> None:
        """prefill 算完后，把满块登记进哈希表，供后续请求复用。"""
        for bid, h in zip(blocks, hashes):
            if self.block_hash[bid] is None:
                self.block_hash[bid] = h
                self.hash_to_block[h] = bid

    def free_blocks(self, blocks: list[int]) -> None:
        """归还。**不清哈希**——块回到空闲队列但内容仍可被命中。

        逆序归还：尾部块排在队列前面，会被优先淘汰。
        因为头部块是前缀，更可能被别的请求复用。
        （vLLM 的实现就是在调用处 reversed() 一下，一行代码 = 一条缓存策略。）
        """
        for bid in reversed(blocks):
            self.ref_cnt[bid] -= 1
            assert self.ref_cnt[bid] >= 0, "ref_cnt 变负了：分配/释放不配对"
            if self.ref_cnt[bid] == 0:
                self.free[bid] = None

    # -- 一次请求的完整流程 ------------------------------------------------

    def serve(self, token_ids: list[int]) -> dict:
        """模拟一次 prefill：查前缀 → 分配缺的块 → 登记 → 返回要重算多少 token。"""
        reused, covered = self.match(token_ids)
        need_tokens = len(token_ids) - covered
        need_blocks = (need_tokens + self.block_size - 1) // self.block_size
        fresh = self.alloc(need_blocks)
        all_hashes = self.block_hashes_of(token_ids)
        self.cache_full_blocks(reused + fresh, all_hashes)
        return {"reused_blocks": len(reused), "recompute_tokens": need_tokens,
                "blocks": reused + fresh}


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def test_rolling_hash_prevents_wrong_reuse() -> None:
    """B 的首块已缓存；必须拒绝复用来自 A 的第二块。"""
    pool = MiniBlockPool(num_blocks=64, block_size=4)
    tail = [900, 901, 902, 903]
    a = [1, 1, 1, 1] + tail
    b = [2, 2, 2, 2] + tail
    ra = pool.serve(a)
    pool.free_blocks(ra["blocks"])
    rb_head = pool.serve(b[:4])
    pool.free_blocks(rb_head["blocks"])
    rb = pool.serve(b)
    assert rb["reused_blocks"] == 1, "只能复用 B 的首块，不能复用 A 的第二块"
    assert pool.block_hashes_of(a)[1] != pool.block_hashes_of(b)[1]
    pool.free_blocks(rb["blocks"])
    print("  ✓ 滚动哈希阻止了跨前缀误复用")


def test_block_granularity() -> None:
    """共享 block_size-1 个 token = 白共享。"""
    bs = 16
    pool = MiniBlockPool(num_blocks=512, block_size=bs)
    base = list(range(1000, 1000 + 256))
    r = pool.serve(base)
    pool.free_blocks(r["blocks"])

    for shared, expect_blocks in [(0, 0), (bs - 1, 0), (bs, 1),
                                  (2 * bs - 1, 1), (2 * bs, 2)]:
        ids = base[:shared] + list(range(9000 + shared, 9000 + 256))
        got = pool.match(ids)[0]
        assert len(got) == expect_blocks, \
            f"共享 {shared} token 应复用 {expect_blocks} 块，实际 {len(got)}"
        pool.free_blocks(got)
    print("  ✓ 复用粒度确实是块：共享 15 个 token 与共享 0 个等价")


def test_free_does_not_invalidate() -> None:
    """ref_cnt 归零后内容仍可命中；被重新分配时才失效。"""
    pool = MiniBlockPool(num_blocks=4, block_size=4)
    a = list(range(100, 116))                  # 4 块
    r = pool.serve(a)
    pool.free_blocks(r["blocks"])
    assert all(c == 0 for c in pool.ref_cnt), "应该全部释放"
    again = pool.match(a)[0]
    assert len(again) == 4, "空闲但仍缓存的块应该还能命中"
    pool.free_blocks(again)

    # 把池子挤满，逼它复用这 4 块 → 旧内容此刻才失效
    b = list(range(500, 516))
    pool.serve(b)
    assert len(pool.match(a)[0]) == 0, "被重新分配后旧内容应已失效"
    print("  ✓ 空闲 ≠ 失效；重新分配时才真正淘汰")


def test_lru_prefers_evicting_tails() -> None:
    """逆序归还使得尾部块先被淘汰，前缀块得以保留。"""
    bs = 4
    pool = MiniBlockPool(num_blocks=6, block_size=bs)
    seq = list(range(200, 200 + 6 * bs))       # 6 块，正好占满
    r = pool.serve(seq)
    pool.free_blocks(r["blocks"])
    # 现在申请 2 块，会淘汰空闲队列最前面的两个 = 原序列的**最后**两块
    pool.alloc(2)
    kept = pool.match(seq)[0]
    assert len(kept) == 4, f"应保留前 4 块前缀，实际 {len(kept)}"
    print("  ✓ 淘汰优先打尾部，前缀被保住")


def simulate_staircase() -> None:
    """并列显示块数预测与历史调用耗时，不把两者当作相互验证。

    这里的块数由模型计算，不是从真实引擎读取的命中数量。
    """
    bs, total = 16, 2048
    pool = MiniBlockPool(num_blocks=4064, block_size=bs)   # 与实测引擎同规格
    rng = random.Random(7)
    base = [rng.randint(1000, 100000) for _ in range(total)]
    r = pool.serve(base)
    pool.free_blocks(r["blocks"])

    print(f"\n  块数预测与历史耗时（block_size={bs}, prompt={total}）")
    print(f"  {'共享前缀':>8} {'预测复用块':>9} {'预测剩余token':>11}   历史调用耗时 ms")
    truth = {0: 36.80, 1: 38.27, 8: 37.19, 15: 36.72, 16: 37.04, 17: 38.42,
             24: 37.37, 31: 37.90, 32: 38.29, 64: 36.11, 128: 34.00,
             256: 32.78, 512: 29.36, 1024: 22.61, 1536: 13.06, 2032: 7.93}
    for shared in sorted(truth):
        ids = base[:shared] + [rng.randint(1000, 100000) for _ in range(total - shared)]
        reused, covered = pool.match(ids)
        pool.free_blocks(reused)
        print(f"  {shared:>8} {len(reused):>9} {total - covered:>11}   {truth[shared]:>10.2f}")

    print("\n  对照真机：共享 0/1/8/15 都是 0 块可复用 → TTFT 全在 36.7–38.3 ms（无趋势）")
    print("            共享 16/17/24/31 都是 1 块可复用 → TTFT 全在 37.0–38.4 ms（无趋势）")
    print("  这些耗时尚不能分辨相邻块边界；需另采实际命中块数或调度 token 数。")


if __name__ == "__main__":
    print("自测：")
    test_rolling_hash_prevents_wrong_reuse()
    test_block_granularity()
    test_free_does_not_invalidate()
    test_lru_prefers_evicting_tails()
    simulate_staircase()
