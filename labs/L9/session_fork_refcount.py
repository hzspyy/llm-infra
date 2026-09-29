#!/usr/bin/env python3
"""L9.3 任务 A 的分支部分：会话 fork、写时复制与引用计数。

模型与 vLLM v1 的块管理同构，只保留与「分支」有关的语义：

* 满块以 ``(父块哈希, 本块 token)`` 链式哈希标识，**内容相同且位置相同**才共享；
* 每个块带 ``refcount``：有多少条活着的序列正在用它。``refcount > 0`` 的块不可驱逐；
* ``fork(seq)`` 只增加共享块的引用计数，不复制数据；
* 追加 token：新满块分配新块；若某个序列要往一个 **refcount > 1 的未满块**里追加，
  必须先复制出私有块（写时复制，COW），否则兄弟分支的内容会被改写。

脚本用一条固定调度（父序列 → fork 两个分支 → 一支改写中段 → 两支各自结束）逐事件核对五条不变量，
并给出一个「没有引用计数的缓存」的反例：同样的事件序列下兄弟分支的内容会被改写，用块内容哈希就能抓到。
这条反例与浮点噪声无关——它是可复现的内容不一致，不是数值波动。

用法::

    python labs/L9/session_fork_refcount.py --out out/9.3/fork-refcount
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def block_hash(parent: str | None, tokens: tuple[int, ...]) -> str:
    h = hashlib.sha1()
    h.update((parent or "root").encode())
    h.update(b"|")
    h.update(",".join(str(t) for t in tokens).encode())
    return h.hexdigest()[:12]


class Block:
    __slots__ = ("key", "parent", "tokens", "refcount", "private")

    def __init__(self, key: str, parent: str | None, tokens: tuple[int, ...]):
        self.key = key
        self.parent = parent
        self.tokens = tokens
        self.refcount = 0
        self.private = False       # 未满块按序列私有

    def content_hash(self) -> str:
        return block_hash(self.parent, self.tokens)


class Pool:
    """块池：按内容哈希共享满块，按序列私有未满块；引用计数决定能否回收。"""

    def __init__(self, capacity_blocks: int, block_size: int = 16):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self.blocks: dict[str, Block] = {}
        self.seqs: dict[str, list[str]] = {}          # 序列 → 有序块键
        self.events: list[dict] = []
        self.cow_count = 0
        self.alloc_count = 0
        self.reuse_count = 0

    # -- 基础操作 -------------------------------------------------------------
    def _full_blocks(self, tokens: list[int]) -> list[tuple[int, ...]]:
        n = len(tokens) // self.block_size
        return [tuple(tokens[i * self.block_size:(i + 1) * self.block_size]) for i in range(n)]

    def create(self, seq_id: str, tokens: list[int]) -> None:
        self.seqs[seq_id] = []
        parent = None
        for blk in self._full_blocks(tokens):
            key = block_hash(parent, blk)
            if key in self.blocks:
                self.blocks[key].refcount += 1
                self.reuse_count += 1
            else:
                self.blocks[key] = Block(key, parent, blk)
                self.blocks[key].refcount = 1
                self.alloc_count += 1
            self.seqs[seq_id].append(key)
            parent = key
        self.events.append({"event": "create", "seq": seq_id, "blocks": len(self.seqs[seq_id]),
                            "free_blocks": self.free_blocks()})

    def fork(self, parent_seq: str, child_seq: str) -> None:
        """只加引用计数，不复制数据。"""
        keys = list(self.seqs[parent_seq])
        for k in keys:
            self.blocks[k].refcount += 1
        self.seqs[child_seq] = keys
        self.events.append({"event": "fork", "parent": parent_seq, "child": child_seq,
                            "shared_blocks": len(keys),
                            "refcounts": [self.blocks[k].refcount for k in keys[:4]]})

    def append(self, seq_id: str, tokens: list[int]) -> None:
        keys = self.seqs[seq_id]
        parent = keys[-1] if keys else None
        for blk in self._full_blocks(tokens):
            key = block_hash(parent, blk)
            if key in self.blocks:
                self.blocks[key].refcount += 1
                self.reuse_count += 1
            else:
                self.blocks[key] = Block(key, parent, blk)
                self.blocks[key].refcount = 1
                self.alloc_count += 1
            keys.append(key)
            parent = key
        self.events.append({"event": "append", "seq": seq_id, "added": len(self._full_blocks(tokens)),
                            "free_blocks": self.free_blocks()})

    def copy_on_write(self, seq_id: str, block_key: str) -> str | None:
        """把共享块复制成私有块；返回新键（无需复制时返回 None）。"""
        blk = self.blocks[block_key]
        if blk.refcount <= 1:
            return None
        new_key = block_hash(blk.parent, blk.tokens + (0,))    # 私有一份，内容随后由调用方改写
        self.blocks[new_key] = Block(new_key, blk.parent, blk.tokens)
        self.blocks[new_key].refcount = 1
        self.blocks[new_key].private = True
        idx = self.seqs[seq_id].index(block_key)
        self.seqs[seq_id][idx] = new_key
        blk.refcount -= 1
        self.cow_count += 1
        self.alloc_count += 1
        self.events.append({"event": "copy_on_write", "seq": seq_id, "old": block_key,
                            "new": new_key, "old_refcount": blk.refcount})
        return new_key

    def free(self, seq_id: str) -> None:
        released = 0
        for k in self.seqs.pop(seq_id, []):
            blk = self.blocks[k]
            blk.refcount -= 1
            if blk.refcount == 0 and blk.private:
                del self.blocks[k]
                released += 1
        self.events.append({"event": "free", "seq": seq_id, "released_private": released,
                            "free_blocks": self.free_blocks()})

    # -- 观测 -----------------------------------------------------------------
    def free_blocks(self) -> int:
        return sum(1 for b in self.blocks.values() if b.refcount == 0)

    def evictable(self) -> int:
        return self.free_blocks()

    def live_blocks(self) -> int:
        return sum(1 for b in self.blocks.values() if b.refcount > 0)

    def hits_for(self, tokens: list[int]) -> int:
        """一段前缀能命中多少 token（整块口径）。"""
        parent = None
        hit = 0
        for blk in self._full_blocks(tokens):
            key = block_hash(parent, blk)
            b = self.blocks.get(key)
            if b is None:
                break
            hit += self.block_size
            parent = key
        return hit

    def content_of(self, seq_id: str) -> list[int]:
        out: list[int] = []
        for k in self.seqs[seq_id]:
            out.extend(self.blocks[k].tokens)
        return out


def run_schedule(pool: Pool, with_refcount: bool = True) -> dict:
    """固定调度：父序列 P 建 4 块 → fork 出 B1、B2 → B1 改写第 3 块 → 两支陆续结束。"""
    base = list(range(64))                  # 4 个满块
    pool.create("P", base)
    pool.fork("P", "B1")
    pool.fork("P", "B2")
    third = pool.seqs["P"][2]
    if with_refcount:
        new_key = pool.copy_on_write("B1", third)
        # 分支真正分叉：改写复制出来的私有块，兄弟分支不应受影响
        if new_key is not None:
            pool.blocks[new_key].tokens = tuple(t + 1000 for t in pool.blocks[new_key].tokens)
    else:
        # 反例：不做写时复制，直接改写共享块——兄弟分支看到的内容随之变化
        blk = pool.blocks[third]
        blk.tokens = tuple(t + 1000 for t in blk.tokens)
        pool.events.append({"event": "in_place_write_without_cow", "seq": "B1", "block": third})
    pool.free("P")
    b2_content = pool.content_of("B2")
    b1_content = pool.content_of("B1")
    return {
        "with_refcount": with_refcount,
        "events": pool.events,
        "cow_count": pool.cow_count,
        "alloc_blocks": pool.alloc_count,
        "reused_blocks": pool.reuse_count,
        "b1_third_block": b1_content[32:48],
        "b2_third_block": b2_content[32:48],
        "siblings_differ": b1_content[32:48] != b2_content[32:48],
        "b2_matches_original": b2_content[32:48] == list(range(32, 48)),
        "free_blocks_after_P_exit": pool.free_blocks(),
        "live_blocks_after_P_exit": pool.live_blocks(),
    }


def cmd_demo(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    good = Pool(args.capacity, args.block_size)
    good_res = run_schedule(good, with_refcount=True)
    bad = Pool(args.capacity, args.block_size)
    bad_res = run_schedule(bad, with_refcount=False)

    # 不变量核对
    checks = [
        {"name": "fork_increments_refcount",
         "expected": "第 1 次 fork 后引用计数 2（P+B1），第 2 次后 3（P+B1+B2）",
         "got": {"after_fork1": good_res["events"][1]["refcounts"],
                 "after_fork2": good_res["events"][2]["refcounts"]},
         "match": (all(r == 2 for r in good_res["events"][1]["refcounts"])
                   and all(r == 3 for r in good_res["events"][2]["refcounts"]))},
        {"name": "cow_is_required",
         "expected": "改写共享中段需要一次写时复制",
         "got": good_res["cow_count"],
         "match": good_res["cow_count"] == 1},
        {"name": "sibling_content_preserved_with_cow",
         "expected": "写时复制后 B1 改写自己的私有块，B2 的第 3 块保持原内容",
         "got": {"b1_third": good_res["b1_third_block"][:4], "b2_third": good_res["b2_third_block"][:4],
                 "b2_matches_original": good_res["b2_matches_original"],
                 "siblings_differ": good_res["siblings_differ"]},
         "match": (good_res["b2_matches_original"] and good_res["siblings_differ"]
                   and good_res["b1_third_block"][0] == 1032)},
        {"name": "without_cow_sibling_is_corrupted",
         "expected": "不做写时复制时兄弟分支内容被改写",
         "got": {"b2_third": bad_res["b2_third_block"][:4], "matches_original": bad_res["b2_matches_original"]},
         "match": (not bad_res["b2_matches_original"]) and (not bad_res["siblings_differ"])},
        {"name": "shared_blocks_survive_one_branch_exit",
         "expected": "P 退出后 B1/B2 仍持有共享块（free=0，live>0）",
         "got": {"free": good_res["free_blocks_after_P_exit"],
                 "live": good_res["live_blocks_after_P_exit"]},
         "match": good_res["free_blocks_after_P_exit"] == 0 and good_res["live_blocks_after_P_exit"] > 0},
    ]
    report = {
        "config": {"capacity_blocks": args.capacity, "block_size": args.block_size},
        "with_refcount": good_res,
        "without_refcount": bad_res,
        "checks": checks,
        "all_match": all(c["match"] for c in checks),
        "note": ("两条路径在同样的调度下给出不同的**内容**：不做写时复制时 B2 的中段被 B1 的写入改写，"
                 "这是可复现的内容不一致，与浮点噪声无关；区分二者的办法是对块内容做哈希对拍，"
                 "而不是重复运行看数值是否漂移"),
    }
    (out / "session_fork_refcount.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {c['got']}")
    print("all_match:", report["all_match"])
    print("good events:", json.dumps(good_res["events"], ensure_ascii=False))
    return 0


def cmd_capacity(args) -> int:
    """fork 的扇出对「可驱逐块」的影响：共享前缀越多，可回收的块越少。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in [int(x) for x in args.widths.split(",")]:
        pool = Pool(args.capacity, args.block_size)
        pool.create("P", list(range(64)))
        for i in range(width):
            pool.fork("P", f"B{i}")
        hits = pool.hits_for(list(range(64)))
        refcount_after_forks = pool.blocks[pool.seqs["P"][0]].refcount
        # 父序列退出、再退掉除最后一条之外的所有分支：看共享块何时才可回收
        pool.free("P")
        for i in range(max(0, width - 1)):
            pool.free(f"B{i}")
        rows.append({
            "fanout": width,
            "live_blocks": pool.live_blocks(),
            "free_blocks": pool.free_blocks(),
            "refcount_of_first_block_after_forks": refcount_after_forks,
            "refcount_of_first_block_after_all_but_one_freed":
                pool.blocks[pool.seqs[f"B{width-1}"][0]].refcount,
            # 每条分支命中同一段前缀，命中量不因分支增加而下降
            "hits_per_branch": hits,
        })
    report = {"capacity_blocks": args.capacity, "block_size": args.block_size, "rows": rows,
              "note": ("共享前缀让每条分支都命中同样多的 token，代价是这些块在**所有**分支结束前都不可回收；"
                       "引用计数把「命中收益」与「容量占用」绑定在一起，扇出宽度直接决定可驱逐块数")}
    (out / "fork_capacity.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    for r in rows:
        print(f"fanout={r['fanout']:>2} live={r['live_blocks']:>3} free={r['free_blocks']:>3} "
              f"refcount_after_forks={r['refcount_of_first_block_after_forks']} "
              f"refcount_last_branch={r['refcount_of_first_block_after_all_but_one_freed']} "
              f"hits_per_branch={r['hits_per_branch']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.3 会话 fork / 写时复制 / 引用计数")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo")
    p.add_argument("--out", required=True)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--block-size", type=int, default=16)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("capacity")
    p.add_argument("--out", required=True)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--widths", default="1,2,4,8")
    p.set_defaults(func=cmd_capacity)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
