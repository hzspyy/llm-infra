#!/usr/bin/env python3
"""L5.1 · 逐 step 的算子形状与 device kernel 对齐。

`serve_protocol.py --walk` 给出每一步的账（预测的 Q/KV 形状、FLOP、device 时间），
但没有回答"这一步实际发射了哪些算子与 kernel、形状是不是账本预测的那些"。
本文件补上这一层，口径如下：

  * 每个引擎 step 挂一个 NVTX range 与 torch.profiler 的 ``record_function``；
  * 在 range 内 ``torch.cuda.synchronize()``，让这一步的 kernel 落在同一 CPU 区间，
    从而可以按时间区间把事件归到 step（不做顺序切片的近似）；
  * 导出 chrome trace，把 ``cpu_op``（带 ``Input Dims``）与 ``kernel`` 归到 step；
  * 把实测的矩阵乘 M 维与账本预测的本步 token 数逐项对照，输出不一致条目。

nsys 侧的口径见 ``run_step_kernel_align.sh``：同一次运行在
``--cuda-graph-trace=node`` 下展开图内 kernel，再按同一批 NVTX range 归到 step。

用法：
    python step_kernel_align.py --out <dir> [--prompt-len 64] [--steps 4] [--batch 1] [--eager]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch                                                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_protocol import (StepMonitor, model_facts, safe_util,       # noqa: E402
                            shutdown, MODEL)

MATMUL_KEYS = ("mm", "addmm", "linear", "matmul", "bmm", "gemm")


class TracedMonitor(StepMonitor):
    """在 StepMonitor 之上再加 NVTX range 与同步点。"""

    def __init__(self):
        super().__init__()
        self.sync_each_step = True

    def patch(self, llm):
        super().patch(llm)
        core = self.core
        inner = core.step_fn
        mon = self

        def step_fn():
            idx = len(mon.steps)
            name = f"step{idx}"
            torch.cuda.nvtx.range_push(name)
            with torch.profiler.record_function(name):
                out = inner()
                if mon.sync_each_step:
                    torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return out

        core.step_fn = step_fn


# ------------------------------------------------------------------ trace 解析
def load_events(path: str) -> list[dict]:
    with open(path) as f:
        blob = json.load(f)
    return blob["traceEvents"]


def step_ranges(events: list[dict], slack_us: float = 200.0) -> list[tuple[str, float, float]]:
    """取出每个 step 的时间区间（us）。"""
    out = []
    for e in events:
        name = e.get("name", "")
        if e.get("cat") == "user_annotation" and name.startswith("step") \
                and name[4:].isdigit() and "dur" in e:
            out.append((name, e["ts"], e["ts"] + e["dur"] + slack_us))
    out.sort(key=lambda x: int(x[0][4:]))
    return out


def attribute(events: list[dict], ranges, cats) -> dict[str, list[dict]]:
    """把 cat 属于 cats 的事件按 ts 归到 step 区间。"""
    buckets: dict[str, list[dict]] = {name: [] for name, _, _ in ranges}
    for e in events:
        if e.get("cat") not in cats or "ts" not in e or "dur" not in e:
            continue
        ts = e["ts"]
        for name, lo, hi in ranges:
            if lo <= ts <= hi:
                buckets[name].append(e)
                break
    return buckets


def op_dims(e: dict):
    args = e.get("args") or {}
    dims = args.get("Input Dims")
    return dims if isinstance(dims, list) else None


def matmul_rows(ops: list[dict]) -> list[dict]:
    rows = []
    for e in ops:
        name = e.get("name", "")
        if not any(k in name for k in MATMUL_KEYS):
            continue
        dims = op_dims(e)
        if not dims:
            continue
        rows.append(dict(name=name, dims=dims, device_us=float(e.get("dur", 0.0))))
    return rows


def cpu_summary(ops: list[dict], top: int = 6) -> dict:
    """CPU 侧 op 的 dur 之和与最重的几条。

    注意这是逐条 ``cpu_op`` 事件的 dur 相加，嵌套调用（``aten::linear`` 里再调
    ``aten::mm``）会各自计时，因此绝对值偏大；它只用来定位某一步的宿主侧开销
    落在哪一类算子上。
    """
    rows = sorted(ops, key=lambda e: -float(e.get("dur", 0.0)))[:top]
    return dict(cpu_ms=sum(float(e.get("dur", 0.0)) for e in ops) / 1000.0,
                top=[dict(name=e.get("name", "?"),
                          ms=round(float(e.get("dur", 0.0)) / 1000.0, 4)) for e in rows])


def summarize_step(ops: list[dict], kernels: list[dict], top: int = 12) -> dict:
    kagg: dict[str, dict] = {}
    for e in kernels:
        k = kagg.setdefault(e.get("name", "?"), dict(count=0, device_us=0.0))
        k["count"] += 1
        k["device_us"] += float(e.get("dur", 0.0))
    kern_top = sorted(kagg.items(), key=lambda kv: -kv[1]["device_us"])[:top]
    return dict(
        n_kernels=len(kernels),
        kernel_device_ms=sum(float(e.get("dur", 0.0)) for e in kernels) / 1000.0,
        kernel_top=[dict(name=n, count=v["count"], device_ms=round(v["device_us"] / 1000.0, 4))
                    for n, v in kern_top],
        matmuls=matmul_rows(ops),
        cpu=cpu_summary(ops),
    )


def check_shape_alignment(step_rows: list[dict], facts: dict) -> dict:
    """把每步实测的矩阵乘 M 维与账本预测的本步 token 数对照。

    预测：这一步参与矩阵乘的激活行数等于本步调度的 token 数 ``n_sched``。
    实测：矩阵乘调用的第一个入参就是激活（``mm(a, w)``、``linear(x, w)``），
    取它的行数——2D 取第 0 维，3D 取前两维之积。
    """
    hidden = facts["hidden"]
    report = []
    for r in step_rows:
        n = r["sched"]
        m_seen = set()
        for mm in r["measured"]["matmuls"]:
            dims = mm["dims"] or []
            d = dims[0] if dims else None
            if not d or len(d) not in (2, 3):
                continue
            if d[-1] != hidden:             # 第一入参的最后一维不是 hidden：不是激活
                continue
            m_seen.add(d[0] if len(d) == 2 else d[0] * d[1])
        report.append(dict(step=r["step"], phase=r["phase"], sched=n,
                           hist=r["hist"], m_expected=n,
                           m_seen=sorted(m_seen),
                           has_expected=n in m_seen))
    return dict(rows=report,
                matched=sum(1 for x in report if x["has_expected"]),
                total=sum(1 for x in report if x["sched"] > 0))


# ------------------------------------------------------------------ 采集
def collect(llm, mon, facts, prompt_len, steps, batch, trace_path, use_profiler=True):
    from vllm import SamplingParams, TokensPrompt

    ids = [1000 + (i * 7) % 50000 for i in range(prompt_len)]
    prompts = [TokensPrompt(prompt_token_ids=list(ids)) for _ in range(batch)]
    mon.reset()
    sp = SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True)

    if not use_profiler:
        llm.generate(prompts, sp, use_tqdm=False)
        return mon.device_ms()

    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts, record_shapes=True, with_stack=False) as prof:
        llm.generate(prompts, sp, use_tqdm=False)
    prof.export_chrome_trace(trace_path)
    dev = mon.device_ms()
    return dev


def make_engine(facts, eager):
    from vllm import LLM
    return LLM(model=MODEL, gpu_memory_utilization=safe_util(),
               max_model_len=16384, disable_log_stats=False,
               enable_prefix_caching=False, enforce_eager=eager)


def short_report(mon, dev, args) -> tuple[str, dict]:
    """不启用 torch profiler 时的摘要（kernel 归属由 summarize_step_nsys.py 出）。"""
    lines = [f"L5.1 NVTX 标注运行 · prompt={args.prompt_len} batch={args.batch} "
             f"steps={args.steps} {'eager' if args.eager else 'graph'}",
             f"{'step':>4}{'阶段':>12}{'本步token':>9}{'hist':>6}{'墙钟ms':>9}{'device ms':>10}"]
    steps = []
    for i, rec in enumerate(mon.steps):
        n = sum(r["sched"] for r in rec["reqs"].values())
        hist = max((r["after"] for r in rec["reqs"].values()), default=0)
        phases = {r["phase"] for r in rec["reqs"].values()}
        phase = phases.pop() if len(phases) == 1 else "mixed"
        lines.append(f"{i:>4}{phase:>12}{n:>9}{hist:>6}{rec['wall_ms']:>9.3f}"
                     f"{(dev[i] if i < len(dev) else float('nan')):>10.3f}")
        steps.append(dict(step=i, phase=phase, sched=n, hist=hist,
                          wall_ms=rec["wall_ms"],
                          device_ms=dev[i] if i < len(dev) else None))
    return "\n".join(lines), dict(steps=steps, args=vars(args))


def report(mon, dev, facts, trace_path, args) -> tuple[str, dict]:
    events = load_events(trace_path)
    ranges = step_ranges(events)
    ops = attribute(events, ranges, {"cpu_op", "operator"})
    kerns = attribute(events, ranges, {"kernel", "cuda_kernel"})
    lines = []
    step_rows = []
    for i, rec in enumerate(mon.steps):
        name = f"step{i}"
        if name not in ops:
            continue
        n_sched = 0
        phase = "-"
        hist = 0
        for rid, r in rec["reqs"].items():
            n_sched += r["sched"]
            hist = max(hist, r["after"])
            phase = r["phase"] if phase == "-" else (
                "mixed" if phase != r["phase"] else phase)
        measured = summarize_step(ops[name], kerns.get(name, []))
        step_rows.append(dict(step=i, phase=phase, sched=n_sched, hist=hist,
                              device_ms=dev[i] if i < len(dev) else None,
                              measured=measured,
                              wall_ms=rec["wall_ms"],
                              n_reqs=len(rec["reqs"])))
    align = check_shape_alignment(step_rows, facts)
    lines.append(f"L5.1 逐 step kernel 对齐 · {MODEL} · prompt={args.prompt_len} "
                 f"batch={args.batch} steps={args.steps} "
                 f"{'eager' if args.eager else 'graph'}")
    lines.append(f"权重 {facts['weight_bytes'] / 2**30:.2f} GiB  "
                 f"KV {facts['kv_bytes_per_token']} B/token  层数 {facts['layers']}  "
                 f"heads {facts['heads']}/{facts['kv_heads']} head_dim {facts['head_dim']}")
    lines.append("")
    hdr = (f"{'step':>4}{'阶段':>10}{'本步token':>9}{'hist':>6}{'device ms':>10}"
           f"{'kernel数':>9}{'kernel总ms':>11}{'CPU op ms':>11}{'mm调用':>7}"
           f"{'M维实测':>24}{'命中预测':>9}")
    lines.append(hdr)
    for r in step_rows:
        a = next(x for x in align["rows"] if x["step"] == r["step"])
        lines.append(f"{r['step']:>4}{r['phase']:>10}{r['sched']:>9}{r['hist']:>6}"
                     f"{(r['device_ms'] or float('nan')):>10.3f}{r['measured']['n_kernels']:>9}"
                     f"{r['measured']['kernel_device_ms']:>11.3f}"
                     f"{r['measured']['cpu']['cpu_ms']:>11.3f}"
                     f"{len(r['measured']['matmuls']):>7}"
                     f"{str(a['m_seen'])[:24]:>24}{str(a['has_expected']):>9}")
    lines.append("")
    lines.append(f"形状对齐：{align['matched']}/{align['total']} 步出现 M=本步 token 数的矩阵乘输入")
    lines.append("（CPU op ms 是逐条 cpu_op 的 dur 之和，含父子重复，只用于定位宿主侧开销）")
    lines.append("")
    for r in step_rows:
        lines.append(f"— step {r['step']}（{r['phase']}，本步 {r['sched']} token，"
                     f"{r['measured']['n_kernels']} 个 kernel）device kernel top：")
        for k in r["measured"]["kernel_top"]:
            lines.append(f"    {k['device_ms']:>8.3f} ms ×{k['count']:<4} {k['name'][:120]}")
        if r["measured"]["cpu"]["top"]:
            lines.append("    CPU 侧 top op：")
            for c in r["measured"]["cpu"]["top"]:
                lines.append(f"      {c['ms']:>8.3f} ms {c['name'][:90]}")
        if r["measured"]["matmuls"]:
            lines.append("    aten 矩阵乘输入形状：")
            for mm in r["measured"]["matmuls"][:8]:
                lines.append(f"      {mm['name'][:60]:<60} {mm['dims']}")
        lines.append("")
    text = "\n".join(lines)
    blob = dict(model=MODEL, args=vars(args), facts={k: v for k, v in facts.items()},
                steps=[dict(step=r["step"], phase=r["phase"], sched=r["sched"],
                            hist=r["hist"], device_ms=r["device_ms"],
                            wall_ms=r["wall_ms"], n_reqs=r["n_reqs"],
                            n_kernels=r["measured"]["n_kernels"],
                            kernel_device_ms=r["measured"]["kernel_device_ms"],
                            kernel_top=r["measured"]["kernel_top"],
                            cpu=r["measured"]["cpu"],
                            matmuls=[dict(name=m["name"], dims=m["dims"]) for m in
                                     r["measured"]["matmuls"]])
                       for r in step_rows],
                shape_alignment=align)
    return text, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--len2", type=int, default=0,
                    help="第二个请求的长度；与 --prompt-len 一起构成混合批")
    ap.add_argument("--eager", action="store_true", help="enforce_eager（关闭图执行）")
    ap.add_argument("--no-sync", action="store_true",
                    help="不在每步内同步（用 kernel 的 GPU 时间戳归属）")
    ap.add_argument("--no-profiler", action="store_true",
                    help="不启用 torch profiler，只跑 NVTX 标注（供 nsys 外层采集）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    facts = model_facts()
    mon = TracedMonitor()
    mon.sync_each_step = not args.no_sync
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    llm = make_engine(facts, args.eager)
    mon.patch(llm)

    trace_path = os.path.join(args.out, "trace.json")
    dev = collect(llm, mon, facts, args.prompt_len, args.steps, args.batch,
                  trace_path, use_profiler=not args.no_profiler)
    if args.no_profiler:
        text, blob = short_report(mon, dev, args)
        out_name = "nvtx_walk"
    else:
        text, blob = report(mon, dev, facts, trace_path, args)
        out_name = "step_kernel_align"
    with open(os.path.join(args.out, f"{out_name}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, f"{out_name}.json"), "w") as f:
        json.dump(blob, f, indent=1)
    print(text)
    print(f"\n写入 {args.out}/{out_name}.txt")
    shutdown(llm)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
