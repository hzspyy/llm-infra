#!/usr/bin/env python3
"""四种流水调度的事件模拟：空泡、在途激活、权重版本与合法更新点。

调度：
  GPipe        全部 forward 跑完再全部 backward
  1F1B         预热之后 forward 与 backward 交替，稳态在途激活最少
  interleaved  每个设备持有多个虚拟 stage，块更小、预热更短、通信次数翻倍
  ZB-V         把 backward 拆成 B(输入梯度) 与 W(权重梯度)，用 W 填空泡

代价模型：整条模型的一次 forward 记 2×STAGES 个单位，backward 是 forward 的两倍。
每个虚拟 stage 分到 2/chunks 个单位，因此 chunks 变化时总计算量保持不变。
通信忽略。输出是依赖 DAG 下的合法排程，不是任何真实硬件上的耗时。

Usage:
    python labs/L7/pipeline_schedules.py > "$RUN_DIR/pipeline.txt"
"""
from __future__ import annotations

from dataclasses import dataclass

STAGES = 4


@dataclass
class Op:
    vstage: int               # 虚拟 stage 序号，决定依赖顺序
    micro: int
    kind: str                 # F / B / Bi / W
    cost: int
    device: int
    start: int = -1
    end: int = -1

    @property
    def key(self):
        return (self.kind, self.micro, self.vstage)

    def __str__(self):
        return {"F": "F", "B": "B", "Bi": "b", "W": "w"}[self.kind] + str(self.micro)


@dataclass
class Schedule:
    name: str
    ops: list[Op]
    vstages: int
    notes: str = ""
    strict: bool = True       # True：设备严格按给定顺序执行；False：按该顺序做优先级的列表调度


def dependencies(op: Op, ops: dict, vstages: int) -> list[Op]:
    deps = []
    if op.kind == "F" and op.vstage > 0:
        deps.append(ops[("F", op.micro, op.vstage - 1)])
    if op.kind in ("B", "Bi"):
        if op.vstage < vstages - 1:
            key = (op.kind, op.micro, op.vstage + 1)
            if key in ops:
                deps.append(ops[key])
        else:
            deps.append(ops[("F", op.micro, op.vstage)])
    if op.kind == "W":
        deps.append(ops[("Bi", op.micro, op.vstage)])
    return deps


def simulate(sched: Schedule) -> dict:
    """每个设备按给定顺序串行执行；依赖未就绪就等待，等待即空泡。"""
    ops = {op.key: op for op in sched.ops}
    device_time = [0] * STAGES
    queue = {d: [op for op in sched.ops if op.device == d] for d in range(STAGES)}
    progress = True
    while progress:
        progress = False
        for d in range(STAGES):
            if not queue[d]:
                continue
            chosen = None
            for idx, cand in enumerate(queue[d]):
                deps = dependencies(cand, ops, sched.vstages)
                if all(dep.end >= 0 for dep in deps):
                    chosen = (idx, cand, deps)
                    break
                if sched.strict:
                    break
            if chosen is None:
                continue
            idx, op, deps = chosen
            ready = max([device_time[d]] + [dep.end for dep in deps])
            op.start, op.end = ready, ready + op.cost
            device_time[d] = op.end
            queue[d].pop(idx)
            progress = True
    assert all(not q for q in queue.values()), f"{sched.name} 存在无法满足的依赖"

    total = max(op.end for op in sched.ops)
    busy = [sum(op.cost for op in sched.ops if op.device == d) for d in range(STAGES)]
    bubble = [1 - b / total for b in busy]
    inflight = []
    for d in range(STAGES):
        peak = 0
        events = sorted([(op.end, 1) for op in sched.ops
                         if op.device == d and op.kind == "F"]
                        + [(op.end, -1) for op in sched.ops
                           if op.device == d and op.kind in ("B", "Bi")])
        cur = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        inflight.append(peak)
    return {"total": total, "bubble": bubble, "inflight": inflight, "busy": busy}


def gantt(sched: Schedule, width: int = 1) -> str:
    total = max(op.end for op in sched.ops)
    lines = []
    for d in range(STAGES):
        row = [" "] * (total * width)
        for op in sched.ops:
            if op.device != d:
                continue
            span = (op.end - op.start) * width
            cell = str(op)[:span].ljust(span, "·")
            row[op.start * width: op.end * width] = list(cell)
        lines.append(f"  dev{d} |" + "".join(row) + "|")
    return "\n".join(lines)


def _seq_1f1b(m: int, vstages: int, split: bool, costs: dict) -> list[Op]:
    ops: list[Op] = []
    for v in range(vstages):
        device = v % STAGES
        warmup = min(vstages - v - 1, m)
        seq = [Op(v, i, "F", costs["F"], device) for i in range(warmup)]
        f_idx, b_idx = warmup, 0
        while f_idx < m:
            seq.append(Op(v, f_idx, "F", costs["F"], device))
            if split:
                seq.append(Op(v, b_idx, "Bi", costs["Bi"], device))
                seq.append(Op(v, b_idx, "W", costs["W"], device))
            else:
                seq.append(Op(v, b_idx, "B", costs["B"], device))
            f_idx += 1
            b_idx += 1
        while b_idx < m:
            if split:
                seq.append(Op(v, b_idx, "Bi", costs["Bi"], device))
                seq.append(Op(v, b_idx, "W", costs["W"], device))
            else:
                seq.append(Op(v, b_idx, "B", costs["B"], device))
            b_idx += 1
        ops += seq
    return ops


def costs_for(chunks: int) -> dict:
    unit = 2 // chunks
    return {"F": unit, "B": 2 * unit, "Bi": unit, "W": unit}


def gpipe(m: int) -> Schedule:
    c = costs_for(1)
    ops = []
    for v in range(STAGES):
        ops += [Op(v, i, "F", c["F"], v) for i in range(m)]
        ops += [Op(v, i, "B", c["B"], v) for i in range(m - 1, -1, -1)]
    return Schedule("GPipe", ops, STAGES, "所有 F 完成后才开始 B")


def one_f_one_b(m: int) -> Schedule:
    return Schedule("1F1B", _seq_1f1b(m, STAGES, False, costs_for(1)), STAGES,
                    "预热 (stage 数 − 本 stage 号 − 1) 个 F 之后 1F1B 交替")


def interleaved(m: int, chunks: int = 2) -> Schedule:
    """按 Megatron 的 interleaved 1F1B 顺序排每个设备的操作。

    每个设备持有 chunks 个虚拟 stage（vstage = chunk * STAGES + device）。
    第 k 次 forward 属于哪个 chunk、哪个 microbatch，按 Megatron 的分组公式确定：
    每 STAGES 个连续 iteration 属于同一个 chunk，chunks 个 chunk 组成一个 group。
    """
    p_, v = STAGES, chunks
    group = p_ * v
    costs = costs_for(chunks)

    def chunk_of(k: int, forward: bool) -> int:
        cid = (k % group) // p_
        return cid if forward else v - cid - 1

    def micro_of(k: int) -> int:
        return (k // group) * p_ + (k % group) % p_

    total = m * v
    ops: list[Op] = []
    for device in range(p_):
        warmup = min((p_ - device - 1) * 2 + (v - 1) * p_, total)
        seq = []
        for k in range(warmup):
            c = chunk_of(k, True)
            seq.append(Op(c * p_ + device, micro_of(k), "F", costs["F"], device))
        f_k, b_k = warmup, 0
        while f_k < total:
            c = chunk_of(f_k, True)
            seq.append(Op(c * p_ + device, micro_of(f_k), "F", costs["F"], device))
            cb = chunk_of(b_k, False)
            seq.append(Op(cb * p_ + device, micro_of(b_k), "B", costs["B"], device))
            f_k += 1
            b_k += 1
        while b_k < total:
            cb = chunk_of(b_k, False)
            seq.append(Op(cb * p_ + device, micro_of(b_k), "B", costs["B"], device))
            b_k += 1
        ops += seq
    return Schedule(f"interleaved({chunks} chunk)", ops, p_ * v,
                    "每设备持有多个虚拟 stage，块更小、预热更短，stage 间通信次数×chunks",
                    strict=False)


def zb_v(m: int) -> Schedule:
    return Schedule("ZB-V(拆 B/W)", _seq_1f1b(m, STAGES, True, costs_for(1)), STAGES,
                    "B 拆成 Bi（沿流水传播）与 W（可延后），W 用来填空泡")


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    print("代价单位：单 stage 的 F=2、B=4；interleaved 的每个虚拟 stage 减半，总工作量不变")
    for m in (4, 8):
        head(f"microbatch = {m}，设备 = {STAGES}")
        rows = []
        for builder in (gpipe, one_f_one_b, interleaved, zb_v):
            sched = builder(m)
            stats = simulate(sched)
            rows.append((sched, stats))
            print(f"\n[{sched.name}] 总时长={stats['total']}  "
                  f"平均空泡={sum(stats['bubble']) / STAGES:.1%}  "
                  f"各设备在途激活={stats['inflight']}")
            print(f"  {sched.notes}")
            if m == 4:
                print(gantt(sched))
        base = rows[0][1]["total"]
        print(f"\n{'调度':<22}{'总时长':>8}{'相对 GPipe':>12}{'峰值在途激活':>14}"
              f"{'平均空泡':>10}")
        for sched, stats in rows:
            print(f"{sched.name:<22}{stats['total']:>8}"
                  f"{(stats['total'] / base - 1) * 100:>11.1f}%"
                  f"{max(stats['inflight']):>14}"
                  f"{sum(stats['bubble']) / STAGES:>9.1%}")

    head("在途激活：GPipe 与 1F1B 的真正区别")
    for m in (4, 8):
        g, f = simulate(gpipe(m)), simulate(one_f_one_b(m))
        print(f"  m={m}: 总时长 {g['total']} vs {f['total']}（相同）；"
              f"各设备在途激活 {g['inflight']} vs {f['inflight']}")
    print("  两者的空泡结构一样，差别在第一个设备要同时留多少份未回收的激活：")
    print("  GPipe 是 m 份，1F1B 是 stage 数那么多份。m 越大差距越大，")
    print("  这决定了同样的显存能塞下多大的 microbatch 数。")

    head("合法更新点与权重版本")
    sched = one_f_one_b(4)
    simulate(sched)
    last_b = max(op.end for op in sched.ops if op.kind == "B")
    print(f"  1F1B / m=4：最后一个 B 在 t={last_b} 结束，此前所有 F 与 B 用的都是权重版本 V0。")
    print("  optimizer 只能在这之后执行；更新完成后才进入 V1。")
    print("  若某个设备提前更新，它后续 microbatch 的 forward 用 V1，"
          "而这些 microbatch 的 backward 对应的是 V0 的激活，")
    print("  梯度与产生它的前向不再匹配。流水并行里权重版本必须显式记账，")
    print("  因为 in-flight 的 microbatch 天然跨越 forward 与 backward 两端。")

    head("拆 B/W 之后多出来的约束")
    z = zb_v(8)
    stats = simulate(z)
    last_bi = max(op.end for op in z.ops if op.kind == "Bi")
    last_w = max(op.end for op in z.ops if op.kind == "W")
    print(f"  m=8：最后一个 Bi 在 t={last_bi} 结束，最后一个 W 在 t={last_w} 结束，"
          f"总时长 {stats['total']}。")
    print("  Bi 必须按流水顺序传播，W 只依赖本 stage 的 Bi，因此可以推迟到空闲时段。")
    print("  但 optimizer 需要全部 W 完成：能填空泡的是 W 的位置，不是它可以被跳过。")


if __name__ == "__main__":
    main()
