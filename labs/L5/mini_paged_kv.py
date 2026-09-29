#!/usr/bin/env python3
"""L5.2 任务 A · 带 fork / COW 的分页 KV 状态机，并与「无缓存参照」逐位置对拍。

`mini_block_pool.py` 只验证块计数与哈希不变量：它不保存任何 KV 内容，
所以回答不了三个问题：

  1. 复用块里的**内容**是否真的等于重算出来的内容？
  2. `fork` 之后两条序列分叉写入，父序列会不会被污染？
  3. 丢掉写时复制（COW）会错在第几个位置？

本脚本把「内容」补上。每个位置的值是该位置**全部前缀 token** 的确定性函数
（滚动链 ``v_p = f(v_{p-1}, t_p)``）——这正是 KV 的上下文相关性：
同样一段 token 跟在不同前缀后面，值必须不同。

于是可以逐位置对拍：

    真实路径（分页 + 前缀复用 + fork + COW） 的值
      vs
    无缓存参照（每个位置按自己的 token 前缀从零重算） 的值

块计数错会暴露，内容错也会暴露。值本身是合成的，不代表任何真实模型。

用法：
    python mini_paged_kv.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import OrderedDict

MOD = (1 << 61) - 1
MUL = 1_000_003


def value_step(parent: int, token: int) -> int:
    """该位置的「KV 值」：只看它和它的全部前缀，不看后面。"""
    return (parent * MUL + token + 1) % MOD


def reference_kv(tokens: list[int]) -> list[int]:
    """无缓存参照：每个位置从零重算。"""
    out, v = [], 0
    for t in tokens:
        v = value_step(v, t)
        out.append(v)
    return out


class Seq:
    """一条序列的分页状态：token 历史 + 块表 + 已计算长度。"""

    def __init__(self, sid: str, tokens: list[int] | None = None):
        self.sid = sid
        self.tokens: list[int] = list(tokens or [])
        self.blocks: list[int] = []
        self.num_computed = 0
        self.forked_from: str | None = None

    def __repr__(self) -> str:                                # pragma: no cover
        return f"<Seq {self.sid} tokens={len(self.tokens)} blocks={len(self.blocks)}>"


class PagedKV:
    """分页 KV：物理块池 + 块哈希表 + LRU 空闲链 + 写时复制。

    与 vLLM 的对应关系：

    * ``ref[bid]``            <- ``KVCacheBlock.ref_cnt``
    * ``free``（OrderedDict）  <- ``FreeKVCacheBlockQueue`` 的双向链表
    * ``hash_to_block``       <- ``BlockPool.cached_block_hash_to_block``
    * ``content[bid]``        <- 块里的 K/V 槽位（这里用合成值）
    """

    def __init__(self, num_blocks: int, block_size: int = 16, cow: bool = True):
        self.num_blocks = num_blocks
        self.bs = block_size
        self.cow = cow
        self.ref = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        self.content: list[list[int] | None] = [None] * num_blocks
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.hash_to_block: dict[int, int] = {}
        self.stat = dict(alloc=0, hit_blocks=0, cow_copies=0, evicted=0, released=0)

    # -- 哈希 -------------------------------------------------------------
    @staticmethod
    def _rolling_hash(parent: int | None, chunk: tuple[int, ...]) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(b"\x00" if parent is None else str(parent).encode())
        for t in chunk:
            h.update(str(t).encode())
        return int.from_bytes(h.digest(), "big")

    def block_hashes_of(self, tokens: list[int]) -> list[int]:
        out, parent = [], None
        for i in range(len(tokens) // self.bs):
            chunk = tuple(tokens[i * self.bs:(i + 1) * self.bs])
            parent = self._rolling_hash(parent, chunk)
            out.append(parent)
        return out

    # -- 块池 -------------------------------------------------------------
    def _alloc(self) -> int:
        assert self.free, "没有空闲块：真实引擎在这里会抢占别的请求"
        bid, _ = self.free.popitem(last=False)                # LRU：最久未用的先拿
        old = self.block_hash[bid]
        if old is not None:                                   # 内容在这一刻真正失效
            self.hash_to_block.pop(old, None)
            self.block_hash[bid] = None
        self.content[bid] = []
        self.ref[bid] = 1
        self.stat["alloc"] += 1
        return bid

    def _release(self, bid: int) -> None:
        self.ref[bid] -= 1
        assert self.ref[bid] >= 0, "ref_cnt 变负：分配/释放不配对"
        if self.ref[bid] == 0:
            self.free[bid] = None                             # 内容与哈希保留
            self.stat["released"] += 1

    def release_seq(self, seq: Seq) -> None:
        """逆序归还：尾部块排在空闲队列前面，先被淘汰；前缀块留得更久。"""
        for bid in reversed(seq.blocks):
            self._release(bid)
        seq.blocks = []

    def evict(self, n: int = 1) -> list[int]:
        """显式淘汰：从空闲队列头部取最久未释放的块并清掉缓存身份。"""
        out = []
        for _ in range(n):
            if not self.free:
                break
            bid, _ = self.free.popitem(last=False)
            old = self.block_hash[bid]
            if old is not None:
                self.hash_to_block.pop(old, None)
                self.block_hash[bid] = None
            self.content[bid] = None
            out.append(bid)
            self.stat["evicted"] += 1
        return out

    # -- 前缀匹配 ---------------------------------------------------------
    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        blocks: list[int] = []
        for h in self.block_hashes_of(tokens):
            bid = self.hash_to_block.get(h)
            if bid is None:
                break                                          # 前缀一断，后面全部作废
            self.ref[bid] += 1
            self.free.pop(bid, None)
            blocks.append(bid)
            self.stat["hit_blocks"] += 1
        return blocks, len(blocks) * self.bs

    # -- 写入 -------------------------------------------------------------
    def _cow_tail(self, seq: Seq, bi: int) -> int:
        """尾部不满块被共享（ref>1）时，复制一份私有副本再写。

        ``cow=False`` 时**故意**直接在共享块上写（静默别名），
        只用于 [C] 的负结果：让对拍把污染抓出来。
        """
        old = seq.blocks[bi]
        if self.ref[old] == 1:
            return old
        if not self.cow:
            self.stat["cow_violations"] = self.stat.get("cow_violations", 0) + 1
            return old
        new = self._alloc()
        self.content[new] = list(self.content[old])            # 只拷有效槽位
        self.block_hash[new] = None                            # 副本身份独立
        self._release(old)
        seq.blocks[bi] = new
        self.stat["cow_copies"] += 1
        return new

    def prefill(self, seq: Seq, tokens: list[int], cap_last: bool = True) -> dict:
        """整段 prompt：先查前缀缓存，再为剩余 token 分配块并算值。

        ``cap_last`` 模拟引擎侧 `get_computed_blocks` 的
        ``max_cache_hit_length = num_tokens - 1``：总要算最后一个 token 才能
        得到 logits 产出首个输出。块池本身没有这条规则——关掉它可以看到
        「128 块全命中」与引擎实际命中 127 块的区别。
        """
        assert not seq.blocks and seq.num_computed == 0, "prefill 只能跑在空序列上"
        reused, covered = self.match(tokens)
        if cap_last and covered == len(tokens) and len(tokens) % self.bs == 0:
            keep = (len(tokens) - 1) // self.bs
            for bid in reused[keep:]:
                self._release(bid)
            reused, covered = reused[:keep], keep * self.bs
        seq.blocks = list(reused)
        seq.tokens = list(tokens)
        for p in range(covered, len(tokens)):
            self._write(seq, p, tokens[p])
        seq.num_computed = len(tokens)
        self._cache_full(seq)
        return dict(reused_blocks=len(reused), recompute_tokens=len(tokens) - covered)

    def decode(self, seq: Seq, token: int) -> int:
        """追加一个 token（分叉后的 decode 也走这里）。"""
        p = len(seq.tokens)
        self._write(seq, p, token)
        seq.tokens.append(token)
        seq.num_computed = len(seq.tokens)
        self._cache_full(seq)
        return p

    def _write(self, seq: Seq, pos: int, token: int) -> None:
        bi, off = divmod(pos, self.bs)
        if bi == len(seq.blocks):
            seq.blocks.append(self._alloc())
        elif off == 0:                                          # 满块不动，满块永远不写
            raise AssertionError("pos 落在满块里：调用方算错了已计算长度")
        bid = self._cow_tail(seq, bi)
        # 值链跨块连续：v_p 依赖全部前缀 token，不只是本块内的槽位
        prev = self.value_of(seq, pos - 1) if pos else 0
        self.content[bid].append(value_step(prev, token))
        if self.cow:
            assert len(self.content[bid]) == off + 1, "槽位写入位置与块内偏移不一致"
        else:
            assert len(self.content[bid]) >= off + 1, "别名写入时块内已多出槽位"

    def _cache_full(self, seq: Seq) -> None:
        """只有填满的块才有资格进哈希表（复用粒度 = 块）。"""
        for i, bid in enumerate(seq.blocks):
            if len(self.content[bid]) == self.bs and self.block_hash[bid] is None:
                h = self.block_hashes_of(seq.tokens)[i]
                self.block_hash[bid] = h
                self.hash_to_block[h] = bid

    def fork(self, parent: Seq, sid: str) -> Seq:
        child = Seq(sid)
        child.forked_from = parent.sid
        child.tokens = list(parent.tokens)
        child.num_computed = parent.num_computed
        child.blocks = list(parent.blocks)
        for bid in child.blocks:
            assert self.ref[bid] > 0, "fork 的块必须已在使用中"
            self.ref[bid] += 1
            self.free.pop(bid, None)
        return child

    # -- 观测 -------------------------------------------------------------
    def value_of(self, seq: Seq, pos: int) -> int:
        bid = seq.blocks[pos // self.bs]
        return self.content[bid][pos % self.bs]

    def block_table(self, seq: Seq) -> list[dict]:
        rows = []
        for i, bid in enumerate(seq.blocks):
            rows.append(dict(
                idx=i, block=bid, ref=self.ref[bid],
                valid=len(self.content[bid]) if self.content[bid] is not None else 0,
                full=bool(self.content[bid] is not None
                          and len(self.content[bid]) == self.bs),
                cached=self.block_hash[bid] is not None,
            ))
        return rows

    def content_diff(self, seq: Seq) -> tuple[int, int, int]:
        """与无缓存参照逐位置对拍，返回 (比较位置数, 不一致数, 首个不一致位置)。"""
        ref = reference_kv(seq.tokens)
        n_bad, first = 0, None
        for p in range(len(seq.tokens)):
            if self.value_of(seq, p) != ref[p]:
                n_bad += 1
                if first is None:
                    first = p
        return len(seq.tokens), n_bad, first


# ---------------------------------------------------------------- 场景
def render_table(pool: PagedKV, seq: Seq, limit: int = 6) -> str:
    rows = pool.block_table(seq)
    head = f"  {seq.sid}: tokens={len(seq.tokens)} computed={seq.num_computed} " \
           f"blocks={len(rows)}"
    lines = [head,
             f"    {'idx':>4}{'block':>7}{'ref':>5}{'valid':>7}{'full':>6}{'cached':>8}"]
    for r in rows[:limit]:
        lines.append(f"    {r['idx']:>4}{r['block']:>7}{r['ref']:>5}{r['valid']:>7}"
                     f"{str(r['full']):>6}{str(r['cached']):>8}")
    if len(rows) > limit:
        lines.append(f"    ...（共 {len(rows)} 块，其余省略）")
    return "\n".join(lines)


def scenario_single(out: list[str], rep: dict) -> None:
    out.append("\n[A] 单请求 + 重复请求：复用块的内容是否等于重算")
    bs, n = 16, 4096
    pool = PagedKV(n, bs)
    toks = [1000 + (i * 37) % 50000 for i in range(2048)]
    s1 = Seq("r0")
    st = pool.prefill(s1, toks)
    cmp1 = pool.content_diff(s1)
    out.append(render_table(pool, s1, limit=3))
    out.append(f"    r0 首次：复用 {st['reused_blocks']} 块 / 重算 {st['recompute_tokens']} token；"
               f"内容对拍 {cmp1[0]} 位置，不一致 {cmp1[1]}")
    pool.release_seq(s1)

    s2 = Seq("r1")
    st2 = pool.prefill(s2, toks)
    cmp2 = pool.content_diff(s2)
    out.append(f"    r1 复用：复用 {st2['reused_blocks']} 块 / 重算 {st2['recompute_tokens']} token；"
               f"内容对拍 {cmp2[0]} 位置，不一致 {cmp2[1]}")
    assert cmp1[1] == 0 and cmp2[1] == 0
    assert st2["reused_blocks"] == 2048 // bs - 1, "引擎约定：末块必须重算以产出 logits"
    pool.release_seq(s2)

    # 块池本身没有「留一个 token」的规则：关掉 cap_last 就能看到 128 块全命中
    s3 = Seq("r2")
    st3 = pool.prefill(s3, toks, cap_last=False)
    cmp3 = pool.content_diff(s3)
    out.append(f"    r2 关闭 num_tokens-1 上限（块池原始语义）：复用 {st3['reused_blocks']} 块 / "
               f"重算 {st3['recompute_tokens']} token；内容不一致 {cmp3[1]}")
    rep["A_single"] = dict(prefill=st, repeat=st2, pool_raw=st3,
                           content=dict(first=cmp1, repeat=cmp2, no_cap=cmp3))
    assert st3["reused_blocks"] == 2048 // bs, "块池在全命中时应复用它缓存的全部满块"
    pool.release_seq(s3)


def scenario_shared_prefix(out: list[str], rep: dict) -> None:
    out.append("\n[B] 共享前缀：非对齐边界 + 中段改变 + 相同后缀不同前缀")
    bs, n = 16, 4096
    pool = PagedKV(n, bs)
    base = [7 + (i * 91) % 40000 for i in range(256)]
    s0 = Seq("base")
    pool.prefill(s0, base)
    pool.release_seq(s0)

    rng = random.Random(11)
    rows = []
    for shared in (0, 15, 16, 17, 127, 128, 129):
        ids = base[:shared] + [rng.randint(50000, 99999)
                               for _ in range(256 - shared)]
        s = Seq(f"sh{shared}")
        st = pool.prefill(s, ids)
        cmp_ = pool.content_diff(s)
        rows.append((shared, st["reused_blocks"], st["recompute_tokens"],
                     cmp_[0], cmp_[1]))
        pool.release_seq(s)
    out.append(f"    {'共享前缀':>8}{'复用块':>8}{'重算token':>10}"
               f"{'对拍位置':>9}{'不一致':>7}{'公式⌊p/16⌋':>11}")
    for shared, rb, rc, cp, bad in rows:
        out.append(f"    {shared:>8}{rb:>8}{rc:>10}{cp:>9}{bad:>7}{shared // bs:>11}")
    assert all(r[4] == 0 for r in rows)

    # 相同后缀、不同前缀：中段一改，尾部即使逐 token 相同也不能复用
    mid_a = base[:96] + [11111] + base[97:]
    mid_b = base[:96] + [22222] + base[97:]
    sa, sb = Seq("midA"), Seq("midB")
    pool.prefill(sa, mid_a)
    pool.release_seq(sa)
    stb = pool.prefill(sb, mid_b)
    cmpb = pool.content_diff(sb)
    out.append(f"    中段第 96 个 token 改为不同值：复用块 {stb['reused_blocks']}"
               f"（只到断点前一块 = {96 // bs}），重算 {stb['recompute_tokens']} token，"
               f"内容不一致 {cmpb[1]}")
    rep["B_shared"] = dict(rows=rows, mid_change=dict(st=stb, content=cmpb))
    assert stb["reused_blocks"] == 96 // bs and cmpb[1] == 0


def scenario_fork(out: list[str], rep: dict) -> None:
    out.append("\n[C] fork 后分叉写入：COW 是否保护父序列")
    bs, n = 16, 64
    for align in ("块对齐", "块内偏移"):
        pool = PagedKV(n, bs, cow=True)
        pref = [3 + (i * 13) % 9000 for i in range(70 if align == "块内偏移" else 64)]
        parent = Seq("p")
        pool.prefill(parent, pref)
        before = pool.block_table(parent)
        child = pool.fork(parent, "c")
        shared_before = [r["block"] for r in before]

        pdiff = [1 + (i * 17) % 9000 for i in range(4)]
        cdiff = [60000 + (i * 19) % 9000 for i in range(4)]
        for i in range(4):
            if align == "块对齐":
                pool.decode(parent, pdiff[i])
                pool.decode(child, cdiff[i])
            else:                                   # 子先写（更容易暴露污染）
                pool.decode(child, cdiff[i])
                pool.decode(parent, pdiff[i])

        pc, cc = pool.content_diff(parent), pool.content_diff(child)
        shared_after = [(r["block"], r["ref"]) for r in pool.block_table(parent)
                        if r["block"] in shared_before]
        out.append(f"    {align}：COW 复制 {pool.stat['cow_copies']} 次；"
                   f"父序列内容不一致 {pc[1]}，子序列不一致 {cc[1]}"
                   f"（对拍 {pc[0]}/{cc[0]} 位置）")
        out.append(f"      父表尾：{[(r['block'], r['ref'], r['valid']) for r in pool.block_table(parent)][-2:]}"
                   f"  子表尾：{[(r['block'], r['ref'], r['valid']) for r in pool.block_table(child)][-2:]}")
        rep.setdefault("C_fork", {})[align] = dict(
            cow_copies=pool.stat["cow_copies"],
            parent=pc, child=cc, shared_blocks_before=shared_before,
            shared_refs_after=shared_after)
        assert pc[1] == 0 and cc[1] == 0
        if align == "块内偏移":
            assert pool.stat["cow_copies"] >= 1, "分叉落在不满块里必须发生 COW"

    # 关闭 COW 的负结果：必须被对拍抓到。先写 child 再写 parent，
    # 被污染的是**后写的那条**（它读到的是别人写进共享槽位的值）。
    pool = PagedKV(64, 16, cow=False)
    pref = [3 + (i * 13) % 9000 for i in range(70)]
    parent = Seq("p_nocow")
    pool.prefill(parent, pref)
    child = pool.fork(parent, "c_nocow")
    pool.decode(child, 60001)
    pool.decode(parent, 1)
    pc, cc = pool.content_diff(parent), pool.content_diff(child)
    out.append(f"    关闭 COW：别名写入 {pool.stat.get('cow_violations', 0)} 次；"
               f"后写的父序列内容不一致 {pc[1]} 位置（首个位置 {pc[2]}），"
               f"先写的子序列不一致 {cc[1]} 位置")
    rep["C_no_cow"] = dict(parent=pc, child=cc,
                           cow_violations=pool.stat.get("cow_violations", 0))
    assert pc[1] > 0 or cc[1] > 0, "关闭 COW 后应出现内容分叉，否则本用例无效"


def scenario_eviction(out: list[str], rep: dict) -> None:
    out.append("\n[D] 驱逐顺序与缓存压力：空闲链 LRU 是否先打尾部块")
    bs = 16
    pool = PagedKV(6, bs)
    seq = Seq("s")
    pool.prefill(seq, list(range(200, 200 + 6 * bs)))
    pool.release_seq(seq)
    ev = pool.evict(2)
    kept, _ = pool.match(seq.tokens)
    out.append(f"    6 块池、6 块序列：淘汰 {ev} 后仍可匹配 {len(kept)} 块"
               f"（保留前 {len(kept)} 块前缀，被淘汰的是尾部）")
    rep["D_evict_order"] = dict(evicted=ev, kept=len(kept))
    assert len(kept) == 4, "逆序归还应让尾部块先被淘汰"

    # 缓存压力：同一访问模式、不同池容量，热前缀能否活下来。
    # 用块池原始语义（cap_last=False）：这里要看的是容量与驱逐，不是引擎末块约定。
    rng = random.Random(5)
    hot = [rng.randint(1000, 99999) for _ in range(128)]        # 8 块
    cold = [[rng.randint(1000, 99999) for _ in range(32)] for _ in range(3)]  # 各 2 块
    press = {}
    for nblocks in (32, 16, 12, 10):
        pool = PagedKV(nblocks, bs)
        hits, bad, per_round = 0, 0, []
        for rnd in range(4):
            rh = 0
            for i, b in enumerate([hot] + cold):
                s = Seq(f"p{nblocks}_{rnd}_{i}")
                st = pool.prefill(s, b, cap_last=False)
                hits += st["reused_blocks"]
                rh += st["reused_blocks"]
                bad += pool.content_diff(s)[1]
                pool.release_seq(s)
            per_round.append(rh)
        out.append(f"    池 {nblocks:>2} 块（热前缀 8 块 + 3 条冷请求各 2 块，跑 4 轮）："
                   f"逐轮命中块 {per_round}，内容不一致 {bad}，"
                   f"分配 {pool.stat['alloc']}，归还 {pool.stat['released']}")
        press[nblocks] = dict(hits=hits, per_round=per_round,
                              content_bad=bad, alloc=pool.stat["alloc"])
        assert bad == 0, "淘汰只影响命中率，不能影响内容正确性"
    rep["D_pressure"] = press
    rounds = [press[n]["per_round"][-1] for n in (32, 16, 12, 10)]
    assert rounds[0] == rounds[1] == 14, "装得下工作集时应命中全部 14 块"
    assert rounds[2] < rounds[1] and rounds[3] < rounds[2], \
        "池子装不下时，热前缀应被冷流量逐步挤掉"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    out: list[str] = []
    rep: dict = {}
    out.append("L5.2-A 分页 KV 状态机：fork / COW / 驱逐 与无缓存参照逐位置对拍")
    out.append(f"值函数 v_p = f(v_(p-1), t_p) mod 2^61-1，与 token 前缀一一对应")
    scenario_single(out, rep)
    scenario_shared_prefix(out, rep)
    scenario_fork(out, rep)
    scenario_eviction(out, rep)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "paged_kv.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "paged_kv.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\n写入 {args.out}/paged_kv.txt 与 paged_kv.json")


if __name__ == "__main__":
    main()
