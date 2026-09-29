#!/usr/bin/env python3
"""L5.4 补测 · `use_inductor_graph_partition` 开启后改变什么。

`PIECEWISE` 图依赖 `torch.compile` 把前向切段。切分发生的位置由
`compilation_config.use_inductor_graph_partition` 决定：

  * False（默认）：在 Dynamo FX 层按 `splitting_ops` 切开，再逐段交给 Inductor；
    编译与融合都只能看到**段内**的图。
  * True：先让 Inductor 跑完全部 pass 与融合，再按同一组 `splitting_ops`
    在 codegen 阶段切分（`config/compilation.py:520-530`），编译能看到整图。

两种方式都产出 `FULL_AND_PIECEWISE` 的图，但段的形状与数量不同。本脚本在
同一负载下分别构造引擎，采：

  * 捕获到的图：按 `BatchExecutionDescriptor` 的 `cg_mode` / `num_tokens` 分组；
  * 每步墙钟与 device 时间（同 5.4 正文口径）；
  * 输出 token 序列（贪心）是否逐位相同。

用法（crater，一次一个进程）：
    python inductor_partition_probe.py --partition 0 --out <dir>
"""

import argparse
import json
import os
import random
import statistics
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L54_MODEL", "Qwen/Qwen3-1.7B")


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def make_llm(partition: bool, util: float):
    from vllm import LLM
    return LLM(model=MODEL, gpu_memory_utilization=util, max_model_len=8192,
               enforce_eager=False, enable_prefix_caching=False,
               disable_log_stats=False, max_num_batched_tokens=8192,
               compilation_config={
                   "use_inductor_graph_partition": partition,
                   "cudagraph_mode": "FULL_AND_PIECEWISE",
               })


def graph_inventory(llm):
    """按 cg_mode 与 num_tokens 统计捕获到的图。"""
    engine = llm.llm_engine.engine_core
    core = getattr(engine, "engine_core", engine)
    runner = core.model_executor.driver_worker.worker.model_runner
    mgr = getattr(runner, "cudagraph_manager", None)
    if mgr is None:
        return dict(error="no cudagraph_manager")
    by_mode = {}
    tokens = {}
    for desc in mgr.graphs:
        mode = getattr(desc.cg_mode, "name", str(desc.cg_mode))
        by_mode[mode] = by_mode.get(mode, 0) + 1
        if "PIECEWISE" in mode or "FULL" in mode:
            tokens.setdefault(mode, []).append(desc.num_tokens)
    return dict(total=len(mgr.graphs), by_mode=by_mode,
                token_counts={k: sorted(set(v)) for k, v in tokens.items()},
                captured_token_counts=sorted(mgr.captured_token_counts()))


class StepTimer:
    def __init__(self, core):
        self.core = core
        self.records = []
        self._sched = None

    def patch(self):
        import vllm.v1.core.sched.scheduler as vsch
        mon = self
        self._orig_schedule = vsch.Scheduler.schedule

        def schedule(self_s, *a, **k):
            out = self._orig_schedule(self_s, *a, **k)
            mon._sched = out
            return out

        vsch.Scheduler.schedule = schedule
        orig = self.core.step_fn

        def step_fn():
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            ev0.record()
            outputs, executed = orig()
            ev1.record()
            self.records.append(dict(
                wall_ms=(time.perf_counter() - t0) * 1000,
                tokens=int(getattr(mon._sched, "total_num_scheduled_tokens", 0))
                if mon._sched is not None else 0,
                ev=(ev0, ev1)))
            return outputs, executed

        self.core.step_fn = step_fn

    def reset(self):
        self.records = []
        torch.cuda.synchronize()

    def device_ms(self):
        torch.cuda.synchronize()
        out = []
        for r in self.records:
            try:
                out.append(r["ev"][0].elapsed_time(r["ev"][1]))
            except Exception:                                   # noqa: BLE001
                out.append(float("nan"))
        return out


def run_case(llm, timer, rng, batch, prompt=128, out=32):
    from vllm import SamplingParams, TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(1000, 60000)
                                              for _ in range(prompt)])
               for _ in range(batch)]
    sp = SamplingParams(max_tokens=out, temperature=0.0, ignore_eos=True)
    timer.reset()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    dev = timer.device_ms()
    steps = [r for r in timer.records]
    tail = [s["wall_ms"] for s in steps[-out:]] or [s["wall_ms"] for s in steps]
    return dict(batch=batch, wall_ms=wall,
                steps=len(steps),
                step_median_ms=statistics.median(tail),
                step_min_ms=min(tail), step_max_ms=max(tail),
                device_ms=sum(x for x in dev if x == x),
                tokens_first=outs[0].outputs[0].token_ids,
                tokens_all=[list(o.outputs[0].token_ids) for o in outs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, required=True, help="0 或 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,8,32")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    util = safe_util()
    llm = make_llm(bool(args.partition), util)
    engine = llm.llm_engine.engine_core
    core = getattr(engine, "engine_core", engine)
    timer = StepTimer(core)
    timer.patch()

    inv = graph_inventory(llm)
    rng = random.Random(20260921)
    cases = []
    for b in (int(x) for x in args.batches.split(",")):
        for _ in range(2):                      # 预热一轮
            run_case(llm, timer, random.Random(7), b)
        cases.append(run_case(llm, timer, rng, b))

    meta = dict(model=MODEL, use_inductor_graph_partition=bool(args.partition),
                gpu_memory_utilization=util, torch=torch.__version__,
                vllm=__import__("vllm").__version__,
                gpu=torch.cuda.get_device_name(0))
    text = [f"use_inductor_graph_partition={bool(args.partition)} · {MODEL} · "
            f"torch {torch.__version__} · vLLM {__import__('vllm').__version__}",
            f"图清单：{json.dumps(inv, ensure_ascii=False)}", ""]
    text.append(f"  {'batch':>5}{'steps':>7}{'步中位 ms':>11}{'最小':>8}{'最大':>8}"
                f"{'总 device ms':>14}{'墙钟 ms':>10}")
    for c in cases:
        text.append(f"  {c['batch']:>5}{c['steps']:>7}{c['step_median_ms']:>11.3f}"
                    f"{c['step_min_ms']:>8.3f}{c['step_max_ms']:>8.3f}"
                    f"{c['device_ms']:>14.1f}{c['wall_ms']:>10.1f}")
    print("\n".join(text))

    tag = f"partition_{args.partition}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(dict(meta=meta, inventory=inv, cases=cases), f, indent=1)
    with open(os.path.join(args.out, f"{tag}.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    print(f"\n写入 {args.out}/{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
