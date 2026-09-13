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

if __name__ == "__main__":
    print("Mini Block Allocator 演示")
    print()

    output = {
        "experiment_5a": experiment_basic(),
        "experiment_5b": experiment_cross_stream(),
        "experiment_5c": experiment_fragmentation(),
        "experiment_5d": experiment_variable_sizes(),
    }

    # 保存结果
    with open("mini_block_allocator.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 mini_block_allocator.json")
