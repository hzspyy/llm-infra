#!/usr/bin/env python3
"""
6.0 通信序列记录器。

记录的内容是「哪个 rank 在第几步、对哪个进程组、调用了哪个集合通信、张量 shape/dtype
是什么、调用前后本 rank 持有什么值」。同一个 op 名在不同进程组上是两条独立记录，
所以事件里同时保存 group 标签与进程组成员；否则重建全局视图时无法判断谁和自己配对。

设计要点：
1. 输入全部是确定性生成的小整数，任何 rank 都能在本地算出全局参照，不需要额外通信。
2. Ledger 每记录一步就追加写盘并 flush，进程被 kill 也能留下「最后匹配到的操作」。
3. `global_view()` 把各 rank 的局部事件按 (step, op, group) 对齐，重建全局张量。

被 6.0 的 A/B/C 三个任务与后续 6.1/6.2 复用。
"""

from __future__ import annotations

import json
import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------
# 输入与单进程参照
# --------------------------------------------------------------------------

def op_input(op: str, rank: int, gsize: int, seed: int = 0) -> list[int]:
    """确定性生成 rank 的输入。reduce_scatter 的输入长度等于组内 rank 数。"""
    n = gsize if op == "reduce_scatter" else 4
    rng = random.Random(f"{seed}:{op}:{rank}")
    return [rng.randrange(10) for _ in range(n)]


def reference(op: str, members: list[int], gsize: int, seed: int = 0):
    """纯 Python 单进程参照，整数运算无误差。返回 {rank: 期望输出列表}。"""
    ins = {r: op_input(op, r, gsize, seed) for r in members}
    if op == "all_reduce":
        total = [sum(ins[r][i] for r in members) for i in range(len(ins[members[0]]))]
        return {r: list(total) for r in members}
    if op == "all_gather":
        flat = [v for r in members for v in ins[r]]
        return {r: list(flat) for r in members}
    if op == "reduce_scatter":
        return {members[i]: [sum(ins[r][i] for r in members)] for i in range(len(members))}
    if op == "send_recv":
        # 环：位置 i 发给自己右边的位置，收到左边位置的输入
        return {r: list(ins[members[(members.index(r) - 1) % gsize]]) for r in members}
    raise ValueError(f"未定义参照: {op}")


# --------------------------------------------------------------------------
# 记录器
# --------------------------------------------------------------------------

def snapshot(t: torch.Tensor) -> dict:
    tt = t.detach().to("cpu")
    return {
        "shape": list(tt.shape),
        "dtype": str(tt.dtype).replace("torch.", ""),
        "values": tt.reshape(-1).tolist(),
    }


class Ledger:
    """逐 step 记录本 rank 的集合通信事件，并即时落盘。"""

    def __init__(self, rank, world_size, backend, device, path, seed=0):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.device = str(device)
        self.seed = seed
        self.step = 0
        self.events: list[dict] = []
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def _flush(self, ev):
        self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    @staticmethod
    def members(group, world_size):
        if group is None:
            return list(range(world_size))
        return sorted(dist.get_process_group_ranks(group))

    def record(self, op, group, group_label, inputs, outputs, note="", primary="x"):
        self.step += 1
        mem = self.members(group, self.world_size)
        ev = {
            "rank": self.rank,
            "step": self.step,
            "op": op,
            "group": group_label,
            "group_members": mem,
            "group_size": len(mem),
            "backend": self.backend,
            "device": self.device,
            "input": inputs,
            "output": outputs,
            "primary": primary,
            "note": note,
            "phase": "end",
        }
        self.events.append(ev)
        self._flush(ev)
        return ev

    def call(self, op, group, group_label, tensors: dict, fn, note="", primary="x"):
        """先给输入拍快照，执行 fn 修改张量，再给输出拍快照。

        调用前先往文件里写一条 phase=begin 的占位行：集合通信卡住或进程被杀时，
        这条占位行就是「最后进入的那个操作」，不会因为异常丢失现场。
        """
        before = {k: snapshot(v) for k, v in tensors.items()}
        self._flush({
            "rank": self.rank, "step": self.step + 1, "op": op, "group": group_label,
            "group_members": self.members(group, self.world_size), "backend": self.backend,
            "device": self.device, "input": before, "output": None, "primary": primary,
            "note": note, "phase": "begin",
        })
        fn()
        after = {k: snapshot(v) for k, v in tensors.items()}
        return self.record(op, group, group_label, before, after, note, primary)

    def close(self):
        self._fh.close()


# --------------------------------------------------------------------------
# 进程组
# --------------------------------------------------------------------------

def init_process_group(rank, world_size, backend, timeout_s=30, master_port=29590,
                       master_addr=None):
    """master_addr 为 None 时仍钉在 127.0.0.1（单机多进程的既有行为）。

    跨机档必须显式给 master_addr：每台机器只有一个进程，rank 由启动器传入。
    """
    os.environ["MASTER_ADDR"] = master_addr or "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )
    return dist.group.WORLD


def build_groups(world_size):
    """所有 rank 必须以相同顺序创建全部子组，否则 new_group 自己就会错配。

    world_size<4 时只有 world，无法构造真正的子组对照。
    """
    groups = {"world": None}
    if world_size >= 4:
        pair_handles = [dist.new_group(p) for p in ([0, 1], [2, 3])]
        even = dist.new_group([0, 2])
        return groups, pair_handles, even
    return groups, None, None


def group_for_rank(pair_handles, even, rank, world_size):
    """返回本 rank 能参与的组：{标签: 组句柄}。不属于的组不出现在表里。"""
    out = {"world": None}
    if world_size >= 4:
        if rank < 4:
            out["pairs"] = pair_handles[rank // 2]
        if rank % 2 == 0:
            out["even"] = even
    return out


# --------------------------------------------------------------------------
# 全局视图
# --------------------------------------------------------------------------

def global_view(ledgers: list[dict]) -> list[dict]:
    """把各 rank 的事件按 (step, op, group) 对齐成一行的全局视图。

    每个 rank 的 step 编号只在自己进程内单调，子组调用会让各 rank 的 step 错开；
    因此先按 (op, group, group_members) 分组，再按 step 排序。
    """
    rows = {}
    for lg in ledgers:
        for ev in lg["events"]:
            key = (ev["step"], ev["op"], ev["group"], tuple(ev["group_members"]))
            rows.setdefault(key, {})[ev["rank"]] = ev
    out = []
    for key in sorted(rows):
        step, op, glabel, mem = key
        per_rank = rows[key]
        out.append({
            "step": step,
            "op": op,
            "group": glabel,
            "group_members": list(mem),
            "present": sorted(per_rank),
            "missing": [r for r in mem if r not in per_rank],
            "per_rank": {str(r): {"input": e["input"], "output": e["output"]}
                         for r, e in sorted(per_rank.items())},
        })
    return out


def write_ledgers(out_dir, ledgers, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    allpath = os.path.join(out_dir, "ledger_all.json")
    with open(allpath, "w", encoding="utf-8") as f:
        json.dump(ledgers, f, ensure_ascii=False, indent=2)
    if dist.is_initialized() and dist.get_rank() == 0:
        view = global_view(ledgers)
        with open(os.path.join(out_dir, "global_view.json"), "w", encoding="utf-8") as f:
            json.dump(view, f, ensure_ascii=False, indent=2)
    return allpath
