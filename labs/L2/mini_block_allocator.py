#!/usr/bin/env python3
"""
实验 5：最小块池实现
用 Python 复现 CUDACachingAllocator 的核心逻辑
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class BlockState(Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    PENDING = "pending"

@dataclass
class Block:
    """内存块"""
    id: int
    size: int
    offset: int  # 在 segment 中的偏移
    state: BlockState
    stream_id: Optional[int] = None  # 分配时的流
    event_id: Optional[int] = None   # 释放时记录的 event

@dataclass
class Segment:
    """内存段（从系统分配的大块）"""
    id: int
    size: int
    blocks: List[Block]

class MiniBlockAllocator:
    """最小块池 allocator"""

    def __init__(self, segment_size: int = 2 * 1024 * 1024):
        self.segment_size = segment_size
        self.segments: List[Segment] = []
        self.free_blocks: List[Block] = []
        self.allocated_blocks: Dict[int, Block] = {}  # block_id -> block
        self.pending_blocks: List[Block] = []

        self.next_segment_id = 0
        self.next_block_id = 0
        self.next_event_id = 0

        # 模拟的 CUDA event 完成状态
        self.completed_events = set()

        # 统计
        self.total_allocated = 0
        self.total_reserved = 0
        self.num_cudaMalloc_calls = 0

    def malloc(self, size: int, stream_id: int) -> int:
        """分配内存块，返回 block_id"""

        # 1. 尝试从 free_blocks 找合适的块
        for block in self.free_blocks:
            if block.size >= size:
                # 找到合适的块，复用
                self.free_blocks.remove(block)
                block.state = BlockState.ALLOCATED
                block.stream_id = stream_id
                self.allocated_blocks[block.id] = block
                self.total_allocated += block.size
                return block.id

        # 2. 没有合适的 free block，需要扩展
        # 检查现有 segment 是否有空间
        for segment in self.segments:
            # 计算已用空间
            used = sum(b.size for b in segment.blocks)
            if segment.size - used >= size:
                # 有空间，切出新块
                offset = used
                block = Block(
                    id=self.next_block_id,
                    size=size,
                    offset=offset,
                    state=BlockState.ALLOCATED,
                    stream_id=stream_id
                )
                self.next_block_id += 1
                segment.blocks.append(block)
                self.allocated_blocks[block.id] = block
                self.total_allocated += size
                return block.id

        # 3. 需要分配新 segment
        segment = self._allocate_segment(max(self.segment_size, size))

        # 从新 segment 切出块
        block = Block(
            id=self.next_block_id,
            size=size,
            offset=0,
            state=BlockState.ALLOCATED,
            stream_id=stream_id
        )
        self.next_block_id += 1
        segment.blocks.append(block)
        self.allocated_blocks[block.id] = block
        self.total_allocated += size

        return block.id

    def free(self, block_id: int, current_stream_id: int):
        """释放内存块"""

        block = self.allocated_blocks.get(block_id)
        if not block:
            raise ValueError(f"Block {block_id} not found or already freed")

        # 标记为 pending
        block.state = BlockState.PENDING

        # 记录 event（如果在不同流，需要等待）
        if block.stream_id != current_stream_id:
            block.event_id = self.next_event_id
            self.next_event_id += 1

        # 移动到 pending 队列
        del self.allocated_blocks[block_id]
        self.pending_blocks.append(block)
        self.total_allocated -= block.size

    def synchronize(self):
        """模拟 synchronize：所有 event 完成"""
        # 标记所有 event 为完成
        for block in self.pending_blocks:
            if block.event_id is not None:
                self.completed_events.add(block.event_id)

        # 处理 pending 队列
        self._process_pending()

    def process_events(self, completed_event_ids: List[int]):
        """处理完成的 event"""
        self.completed_events.update(completed_event_ids)
        self._process_pending()

    def _process_pending(self):
        """检查 pending 队列，将已完成的移到 free"""
        remaining = []

        for block in self.pending_blocks:
            # 检查是否可以移到 free
            if block.event_id is None or block.event_id in self.completed_events:
                # 可以复用
                block.state = BlockState.FREE
                block.stream_id = None
                block.event_id = None
                self.free_blocks.append(block)
            else:
                # 仍在等待
                remaining.append(block)

        self.pending_blocks = remaining

    def _allocate_segment(self, size: int) -> Segment:
        """分配新 segment（模拟 cudaMalloc）"""
        segment = Segment(
            id=self.next_segment_id,
            size=size,
            blocks=[]
        )
        self.next_segment_id += 1
        self.segments.append(segment)
        self.total_reserved += size
        self.num_cudaMalloc_calls += 1
        return segment

    def stats(self) -> dict:
        """返回统计信息"""
        return {
            "allocated": self.total_allocated,
            "reserved": self.total_reserved,
            "num_segments": len(self.segments),
            "num_free_blocks": len(self.free_blocks),
            "num_allocated_blocks": len(self.allocated_blocks),
            "num_pending_blocks": len(self.pending_blocks),
            "num_cudaMalloc_calls": self.num_cudaMalloc_calls,
            "fragmentation_pct": (self.total_reserved - self.total_allocated) / self.total_reserved * 100 if self.total_reserved > 0 else 0
        }

def print_stats(allocator: MiniBlockAllocator, label: str):
    """打印统计信息"""
    stats = allocator.stats()
    print(f"{label:50} | Alloc: {stats['allocated']/1024/1024:7.2f} MB | "
          f"Reserved: {stats['reserved']/1024/1024:7.2f} MB | "
          f"Free: {stats['num_free_blocks']:2d} | "
          f"Pending: {stats['num_pending_blocks']:2d} | "
          f"Frag: {stats['fragmentation_pct']:5.1f}%")
    return stats

def experiment_basic():
    """基础分配与释放"""
    print("=" * 110)
    print("实验 5A：基础分配与释放")
    print("=" * 110)

    allocator = MiniBlockAllocator(segment_size=10 * 1024 * 1024)
    results = []
    stream = 0

    # 分配
    b1 = allocator.malloc(4 * 1024 * 1024, stream)
    results.append(print_stats(allocator, "分配 4MB (block 1)"))

    b2 = allocator.malloc(2 * 1024 * 1024, stream)
    results.append(print_stats(allocator, "分配 2MB (block 2)"))

    # 释放
    allocator.free(b1, stream)
    results.append(print_stats(allocator, "释放 block 1"))

    # 同步（pending -> free）
    allocator.synchronize()
    results.append(print_stats(allocator, "synchronize()"))

    # 复用
    b3 = allocator.malloc(4 * 1024 * 1024, stream)
    results.append(print_stats(allocator, "分配 4MB (block 3, 复用)"))

    print()
    return results

def experiment_cross_stream():
    """跨流分配与释放"""
    print("=" * 110)
    print("实验 5B：跨流分配与释放")
    print("=" * 110)

    allocator = MiniBlockAllocator()
    results = []
    stream1 = 1
    stream2 = 2

    # 在 stream1 分配
    b1 = allocator.malloc(4 * 1024 * 1024, stream1)
    results.append(print_stats(allocator, "stream1 分配"))

    # 在 stream2 释放（需要 event）
    allocator.free(b1, stream2)
    results.append(print_stats(allocator, "stream2 释放 (pending)"))

    # event 未完成，无法复用
    b2 = allocator.malloc(4 * 1024 * 1024, stream2)
    results.append(print_stats(allocator, "stream2 分配 (扩展新 segment)"))

    # 同步 stream1
    allocator.process_events([0])  # event 0 完成
    results.append(print_stats(allocator, "stream1 event 完成"))

    # 现在可以复用
    allocator.free(b2, stream2)
    allocator.synchronize()
    b3 = allocator.malloc(4 * 1024 * 1024, stream2)
    results.append(print_stats(allocator, "复用"))

    print()
    return results

def experiment_fragmentation():
    """碎片化场景"""
    print("=" * 110)
    print("实验 5C：碎片化")
    print("=" * 110)

    allocator = MiniBlockAllocator(segment_size=20 * 1024 * 1024)
    results = []
    stream = 0

    sizes = [1, 2, 4, 2, 1, 4, 2]  # MB
    blocks = []

    for i, size_mb in enumerate(sizes):
        b = allocator.malloc(size_mb * 1024 * 1024, stream)
        blocks.append(b)
        results.append(print_stats(allocator, f"分配 {size_mb}MB (block {i+1})"))

    # 释放部分
    for i in [0, 2, 4]:
        allocator.free(blocks[i], stream)
    allocator.synchronize()
    results.append(print_stats(allocator, "释放 block 1, 3, 5"))

    # 尝试分配 8MB（无单个 free block 够大，需扩展）
    b_new = allocator.malloc(8 * 1024 * 1024, stream)
    results.append(print_stats(allocator, "分配 8MB（碎片导致扩展）"))

    print()
    return results

def experiment_variable_sizes():
    """变长分配"""
    print("=" * 110)
    print("实验 5D：变长分配")
    print("=" * 110)

    allocator = MiniBlockAllocator()
    results = []
    stream = 0

    sizes = [1, 3, 2, 5, 2, 1, 4]

    for i, size_mb in enumerate(sizes):
        b = allocator.malloc(size_mb * 1024 * 1024, stream)
        stats = print_stats(allocator, f"迭代 {i+1}: {size_mb}MB")
        results.append({"size_mb": size_mb, **stats})
        allocator.free(b, stream)
        allocator.synchronize()

    print()
    return results



# =====================================================================
# 实验 5E：三个生命周期时刻 + split/merge + 与真实 allocator 对照
#
# 上面的 MiniBlockAllocator 只做"整块分配"，看不到切分与合并。
# 这一节补三件事：
#   1. 每次分配记录 owner stream 与 pending event，释放后仍保留 owner；
#   2. 打印 split（大块切小）与 merge（相邻空闲块合并）事件；
#   3. 把三个时刻分别打印出来，并与 torch 的 memory_snapshot 对账。
#
# 术语与 torch 对齐，不能混用：
#   allocated      已分配出去的请求字节（用户持有引用）
#   reserved       向驱动要来的总字节（含空闲段与空闲块）
#   active         allocated 的块（含被切分后的剩余部分）
#   inactive_split 已释放、但因为是被切出来的"碎片"而无法直接合并的块
#   inactive       已释放、可整块复用的完整块
#   pending        已释放但 GPU 可能还在用（等 event）的块
# =====================================================================

MIN_SPLIT = 512 * 1024          # 剩余部分小于它就不再切，避免碎片过细


class LifecyclePool:
    def __init__(self, segment_size: int = 20 * 1024 * 1024):
        self.segment_size = segment_size
        self.segments: List[Segment] = []
        self.next_block_id = 0
        self.next_event_id = 0
        self.events: List[dict] = []
        self.completed_events = set()

    # ---- 内部工具 ----
    def _new_segment(self, size: int) -> Segment:
        seg = Segment(id=len(self.segments), size=size, blocks=[])
        self.segments.append(seg)
        self.events.append({"op": "cudaMalloc", "segment": seg.id, "bytes": size})
        return seg

    def _insert(self, seg: Segment, block: Block):
        seg.blocks.append(block)
        seg.blocks.sort(key=lambda b: b.offset)

    def _merge(self, seg: Segment):
        """把相邻的 FREE 块合并（真实 allocator 在 free 后做同样的事）。"""
        seg.blocks.sort(key=lambda b: b.offset)
        out = []
        for b in seg.blocks:
            if out and out[-1].state == BlockState.FREE and b.state == BlockState.FREE \
                    and out[-1].offset + out[-1].size == b.offset:
                out[-1].size += b.size
                self.events.append({"op": "merge", "into": out[-1].id,
                                    "absorbed": b.id, "size": out[-1].size})
            else:
                out.append(b)
        seg.blocks = out

    # ---- 分配 ----
    def malloc(self, size: int, stream_id: int) -> Block:
        size = max(size, MIN_SPLIT)
        # 先找能装下的空闲块（best fit）
        best = None
        for seg in self.segments:
            for b in seg.blocks:
                if b.state == BlockState.FREE and b.size >= size:
                    if best is None or b.size < best.size:
                        best = b
        if best is not None:
            seg = next(s for s in self.segments if best in s.blocks)
            if best.size - size >= MIN_SPLIT:
                rest = Block(id=self.next_block_id, size=best.size - size,
                             offset=best.offset + size, state=BlockState.FREE)
                self.next_block_id += 1
                best.size = size
                self._insert(seg, rest)
                self.events.append({"op": "split", "block": best.id,
                                    "kept": size, "rest": rest.id,
                                    "rest_size": rest.size})
            best.state = BlockState.ALLOCATED
            best.stream_id = stream_id
            best.event_id = None
            self.events.append({"op": "reuse", "block": best.id,
                                "size": size, "stream": stream_id})
            return best
        # 没有合适的：开新段（按 20 MiB 向上取整，和真实大块池一致）
        seg = self._new_segment(max(self.segment_size, size))
        blk = Block(id=self.next_block_id, size=size, offset=0,
                    state=BlockState.ALLOCATED, stream_id=stream_id)
        self.next_block_id += 1
        if seg.size - size >= MIN_SPLIT:
            rest = Block(id=self.next_block_id, size=seg.size - size, offset=size,
                         state=BlockState.FREE)
            self.next_block_id += 1
            self._insert(seg, rest)
            self.events.append({"op": "split", "block": blk.id, "kept": size,
                                "rest": rest.id, "rest_size": rest.size})
        self._insert(seg, blk)
        self.events.append({"op": "malloc", "block": blk.id, "size": size,
                            "stream": stream_id})
        return blk

    # ---- 释放：进入 pending，等 event ----
    def free(self, block: Block, current_stream_id: int, gpu_pending: bool):
        """gpu_pending=False 表示当前流就是最后使用它的流，可以不记 event。"""
        block.state = BlockState.PENDING
        if gpu_pending:
            block.event_id = self.next_event_id
            self.next_event_id += 1
        self.events.append({"op": "free", "block": block.id,
                            "owner_stream": block.stream_id,
                            "event": block.event_id})

    def complete(self, event_id: int):
        self.completed_events.add(event_id)
        for seg in self.segments:
            for b in seg.blocks:
                if b.state == BlockState.PENDING and \
                        (b.event_id is None or b.event_id in self.completed_events):
                    b.state = BlockState.FREE
                    self.events.append({"op": "reusable", "block": b.id,
                                        "owner_stream": b.stream_id})
            self._merge(seg)

    # ---- 统计：术语与 torch 对齐 ----
    def stats(self) -> dict:
        allocated = active = inactive_split = inactive = pending = 0
        for seg in self.segments:
            for b in seg.blocks:
                if b.state == BlockState.ALLOCATED:
                    allocated += b.size
                    active += b.size
                elif b.state == BlockState.PENDING:
                    pending += b.size
                elif b.state == BlockState.FREE:
                    if b.offset > 0:          # 被切出来的剩余块
                        inactive_split += b.size
                    else:
                        inactive += b.size
        reserved = sum(s.size for s in self.segments)
        return {"allocated": allocated, "active": active, "reserved": reserved,
                "inactive": inactive, "inactive_split": inactive_split,
                "pending": pending,
                "segments": len(self.segments),
                "cudaMalloc_calls": sum(1 for e in self.events if e["op"] == "cudaMalloc")}


def _mb(x):
    return x / 1024 / 1024


def experiment_lifecycle():
    print("=" * 110)
    print("实验 5E：三个生命周期时刻、split/merge 与真实 allocator 对照")
    print("=" * 110)

    pool = LifecyclePool(segment_size=20 * 1024 * 1024)
    s_main, s_side = 0, 7

    print("\n-- 时刻 1：Python 引用被丢掉，但 GPU 可能还在用 --")
    a = pool.malloc(4 * 1024 * 1024, s_main)
    pool.free(a, s_side, gpu_pending=True)
    st = pool.stats()
    print(f"   free(a) 之后：allocated={_mb(st['allocated']):.2f} MB  "
          f"pending={_mb(st['pending']):.2f} MB  "
          f"inactive_split={_mb(st['inactive_split']):.2f} MB")
    print(f"   块的 owner stream 仍记为 {a.stream_id}，pending event = {a.event_id}；"
          f"此刻复用它是竞态")
    print(f"   最近事件：{pool.events[-1]}")

    print("\n-- 时刻 2：event 完成，allocator 可以复用 --")
    pool.complete(a.event_id)
    st = pool.stats()
    print(f"   complete(event={a.event_id}) 之后：allocated={_mb(st['allocated']):.2f} MB  "
          f"pending={_mb(st['pending']):.2f} MB  "
          f"inactive_split={_mb(st['inactive_split']):.2f} MB")
    print(f"   块状态 = {a.state.value}；只有到这一刻，下一次 malloc 才能拿到它")

    print("\n-- 时刻 3：真正分配给下一个请求（复用 + 切分）--")
    b = pool.malloc(1 * 1024 * 1024, s_main)
    reuse = [e for e in pool.events if e["op"] == "reuse"]
    split = [e for e in pool.events if e["op"] == "split"]
    print(f"   malloc(1 MiB) → block {b.id}，是否复用了刚释放的 block？"
          f"{'是' if b.id == a.id else '否'}")
    for e in reuse[-2:]:
        print(f"   reuse 事件：{e}")
    for e in split[-2:]:
        print(f"   split 事件：{e}")

    print("\n-- merge：相邻空闲块在 event 完成后合并 --")
    pool2 = LifecyclePool(segment_size=20 * 1024 * 1024)
    x = pool2.malloc(4 * 1024 * 1024, s_main)
    y = pool2.malloc(4 * 1024 * 1024, s_main)
    pool2.free(x, s_main, gpu_pending=True)
    pool2.free(y, s_main, gpu_pending=True)
    print(f"   释放两个相邻块后：inactive_split={_mb(pool2.stats()['inactive_split']):.2f} MB")
    for eid in [e["event"] for e in pool2.events
                if e["op"] == "free" and e["event"] is not None]:
        pool2.complete(eid)
    print(f"   全部 event 完成后：inactive={_mb(pool2.stats()['inactive']):.2f} MB  "
          f"inactive_split={_mb(pool2.stats()['inactive_split']):.2f} MB")
    print("   合并事件：" + str([e for e in pool2.events if e["op"] == "merge"]))

    print("\n-- 与真实 allocator 对照（同一组尺寸，torch memory_snapshot）--")
    real = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t1 = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            t2 = torch.empty(1 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            t3 = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            del t2
            segs = torch.cuda.memory_snapshot()
            states = {}
            for s in segs:
                for blk in s["blocks"]:
                    states[blk["state"]] = states.get(blk["state"], 0) + blk["size"]
            real = {"segments": len(segs),
                    "active_allocated_mb": _mb(states.get("active_allocated", 0)),
                    "inactive_split_mb": _mb(states.get("inactive_split", 0)),
                    "inactive_mb": _mb(states.get("inactive", 0)),
                    "reserved_mb": _mb(sum(s["total_size"] for s in segs)),
                    "torch_reserved_mb": _mb(torch.cuda.memory_reserved()),
                    "torch_allocated_mb": _mb(torch.cuda.memory_allocated())}
            del t1, t3
    except Exception as exc:      # 本地无 GPU 时跳过
        print(f"   （跳过：{type(exc).__name__}: {exc}）")

    print("   （两组数字来自不同负载，这里只对齐术语与状态分类，不做性能对比）")
    print(f"   {'口径':<20}{'mini pool':>14}{'torch snapshot':>18}")
    print("   " + "-" * 54)
    mini = pool.stats()
    rows = [("allocated", _mb(mini["allocated"]), real and real["torch_allocated_mb"]),
            ("reserved", _mb(mini["reserved"]), real and real["torch_reserved_mb"]),
            ("active", _mb(mini["active"]), real and real["active_allocated_mb"]),
            ("inactive_split", _mb(mini["inactive_split"]),
             real and real["inactive_split_mb"]),
            ("inactive", _mb(mini["inactive"]), real and real["inactive_mb"]),
            ("segments", mini["segments"], real and real["segments"])]
    for name, m, r in rows:
        print(f"   {name:<20}{m:>14.2f}{('' if r is None else f'{r:>18.2f}')}")

    return {"lifecycle_events": pool.events, "mini_stats": pool.stats(),
            "real_snapshot": real}


if __name__ == "__main__":
    print("Mini Block Allocator 演示")
    print()

    output = {
        "experiment_5a": experiment_basic(),
        "experiment_5b": experiment_cross_stream(),
        "experiment_5c": experiment_fragmentation(),
        "experiment_5d": experiment_variable_sizes(),
        "experiment_5e": experiment_lifecycle(),
    }

    # 保存结果
    with open("mini_block_allocator.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 mini_block_allocator.json")
