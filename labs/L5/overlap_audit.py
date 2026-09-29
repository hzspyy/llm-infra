#!/usr/bin/env python3
"""L5.4 任务 C —— 执行层的重叠：CPU 准备/提交与 GPU 前向/采样怎样并行。

计划的第三个问题：「CPU 调度、H2D、模型与采样可以怎样重叠」。
本脚本用同一份 decode 负载取四个量：

  [C1] wall/step            不插桩的墙钟差分（正常计时）
  [C2] cpu_process/step     time.process_time 差分（进程真实 CPU 时间）
  [C3] gpu_busy/step        profiler 窗口内 kernel device 时间差分
  [C4] 重叠判据             wall 贴近 max(cpu,gpu) 还是 cpu+gpu

`async_scheduling=False` 就是「关掉一整个重叠阶段」的开关：关掉后
`EngineCore` 不再持有 batch 队列，界面回到一步一等（core.py 的 `step`）。

[C5] 输出缓冲的生命周期：同 seed 下 async on/off 的贪心输出必须逐 token 相同，
     且 gen=8 的输出必须是 gen=32 的前缀 —— 流水线双缓冲不能把采样结果读脏。

用法：
    python overlap_audit.py            # C1-C5
    python overlap_audit.py C1 C4
"""

from __future__ import annotations

import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L54_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)), flush=True)


def safe_util(reserve_gib=6.0, cap=0.55, floor=0.22):
    import torch
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    util = max(min(cap, (free / gib - reserve_gib) / (total / gib)), floor)
    print(f"  [mem] free={free / gib:.2f} GiB -> gpu_memory_utilization={util:.4f}")
    return util


def make_llm(**kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(), max_model_len=4096,
             enable_prefix_caching=False, disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                            # noqa: BLE001
        pass
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def rand_ids(n, rng):
    return [rng.randint(1000, 60000) for _ in range(n)]


def core_of(llm):
    return llm.llm_engine.engine_core.engine_core


def prompts(B, plen, seed):
    from vllm import TokensPrompt
    rng = random.Random(seed)
    return [TokensPrompt(prompt_token_ids=rand_ids(plen, rng)) for _ in range(B)]


def measure(llm, B, plen=64, gen=96, seed=0):
    """一次差分测量：返回 每步的 (wall ms, cpu_process ms)。"""
    from vllm import SamplingParams
    ps = prompts(B, plen, seed)
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spG = SamplingParams(max_tokens=gen + 1, temperature=0.0, ignore_eos=True)
    llm.generate(ps, spG, use_tqdm=False)                    # 预热 + 捕获

    t0, c0 = time.perf_counter(), time.process_time()
    llm.generate(ps, sp1, use_tqdm=False)
    w1, c1 = time.perf_counter() - t0, time.process_time() - c0

    t0, c0 = time.perf_counter(), time.process_time()
    llm.generate(ps, spG, use_tqdm=False)
    wG, cG = time.perf_counter() - t0, time.process_time() - c0

    return ((wG - w1) / gen * 1000, (cG - c1) / gen * 1000)


def profile_gpu(llm, B, plen=64, gen=48, seed=0):
    """profiler 差分出纯 decode 的 GPU 忙时间/步（µs）。"""
    from torch.profiler import ProfilerActivity, profile
    from vllm import SamplingParams
    ps = prompts(B, plen, seed)
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spG = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True)
    llm.generate(ps, spG, use_tqdm=False)

    def window(sp):
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as p:
            llm.generate(ps, sp, use_tqdm=False)
        ka = p.key_averages()
        gpu = sum(e.self_device_time_total for e in ka)
        cpu_launch = sum(e.self_cpu_time_total for e in ka if "Launch" in e.key)
        return gpu, cpu_launch

    g1, l1 = window(sp1)
    gG, lG = window(spG)
    return (gG - g1) / (gen - 1), (lG - l1) / (gen - 1)


CONFIGS = [("async_scheduling=True", dict(async_scheduling=True)),
           ("async_scheduling=False", dict(async_scheduling=False))]


def section_C1(rounds=5, batches=(8, 64)):
    title("[C1] 关掉重叠阶段：async_scheduling on/off 的稳态耗时")
    print("  async_scheduling=True 时 EngineCore 用 batch 队列（core.py:638 "
          "step_with_batch_queue）")
    print("  把 n+1 步的调度/输入准备与 n 步的 GPU 执行并行；False 时回到"
          " 逐步串行（core.py step）。\n")

    data = {}
    struct = {}
    for label, kw in CONFIGS:
        llm = make_llm(**kw)
        ec = core_of(llm)
        bq = getattr(ec, "batch_queue", None)
        struct[label] = dict(
            batch_queue_size=getattr(ec, "batch_queue_size", None),
            has_queue=bq is not None,
            step_fn=type(ec.step_fn).__name__ if hasattr(ec, "step_fn") else None,
            async_flag=llm.llm_engine.vllm_config.scheduler_config.async_scheduling,
        )
        data[label] = {B: [] for B in batches}
        for _ in range(rounds):
            for B in batches:
                w, c = measure(llm, B)
                data[label][B].append((w, c))
        shutdown(llm)

    for label, _ in CONFIGS:
        s = struct[label]
        print(f"  {label:<24} batch_queue_size={s['batch_queue_size']} "
              f"has_queue={s['has_queue']} async_scheduling={s['async_flag']}")

    print(f"\n  每步 wall ms（{rounds} 轮交错的中位数，括号 min–max）与"
          f" 每步 CPU 进程时间")
    print(f"  {'config':<24}{'batch':>6}{'wall ms':>20}{'cpu ms':>20}{'tok/s':>9}")
    med = {}
    for label, _ in CONFIGS:
        for B in batches:
            ws = sorted(x[0] for x in data[label][B])
            cs = sorted(x[1] for x in data[label][B])
            mw = ws[len(ws) // 2]
            mc = cs[len(cs) // 2]
            med[(label, B)] = (mw, mc)
            print(f"  {label:<24}{B:>6}"
                  f"{mw:>14.3f} ({ws[0]:.2f}–{ws[-1]:.2f})"
                  f"{mc:>14.3f} ({cs[0]:.2f}–{cs[-1]:.2f})"
                  f"{B / (mw / 1000):>9.0f}")
    print("\n  加速比（async off / async on，wall 中位数）：")
    for B in batches:
        off = med[("async_scheduling=False", B)][0]
        on = med[("async_scheduling=True", B)][0]
        print(f"    batch={B:<4} off {off:.3f} ms  on {on:.3f} ms  "
              f"on/off = {on / off:.3f}")
    return data, med


def section_C4(med, batches=(8, 64)):
    title("[C4] 重叠判据：wall 贴近 max(cpu,gpu) 还是 cpu+gpu")
    print("  完全串行时 wall ≈ cpu + gpu；完全重叠时 wall ≈ max(cpu, gpu)。")
    print("  overlap = (cpu + gpu - wall) / min(cpu, gpu)，1.0 表示 CPU 完全被藏住。\n")
    print(f"  {'config':<24}{'batch':>6}{'wall':>9}{'cpu':>9}{'gpu':>9}"
          f"{'overlap':>9}  判据")
    for label, kw in CONFIGS:
        llm = make_llm(**kw)
        for B in batches:
            wall = med[(label, B)][0]
            cpu = med[(label, B)][1]
            gpu_us, launch_us = profile_gpu(llm, B)
            gpu = gpu_us / 1000
            ov = (cpu + gpu - wall) / max(min(cpu, gpu), 1e-9)
            verdict = ("CPU 被 GPU 藏住" if ov > 0.8 else
                       "部分重叠" if ov > 0.2 else "基本串行")
            print(f"  {label:<24}{B:>6}{wall:>9.3f}{cpu:>9.3f}{gpu:>9.3f}"
                  f"{ov:>9.2f}  {verdict}")
        shutdown(llm)
    print("\n  注意：cpu 是进程 CPU 时间（time.process_time），gpu 是 kernel")
    print("  device 时间；两者都不是墙钟，只有 wall 是。这一步不引用插桩数作性能值。")


def section_C5():
    title("[C5] 输出缓冲生命周期：流水线的结果必须与串行逐 token 相同")
    from vllm import SamplingParams
    outs = {}
    for label, kw in CONFIGS:
        llm = make_llm(**kw)
        ps = prompts(4, 64, seed=77)
        for gen in (8, 32):
            sp = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True)
            r = llm.generate(ps, sp, use_tqdm=False)
            outs[(label, gen)] = [o.outputs[0].token_ids for o in r]
        shutdown(llm)

    on8 = outs[("async_scheduling=True", 8)]
    off8 = outs[("async_scheduling=False", 8)]
    on32 = outs[("async_scheduling=True", 32)]
    print(f"  async on/off 在 gen=8 上逐 token 相同: {on8 == off8}")
    prefix_ok = all(a == b[:8] for a, b in zip(on8, on32))
    print(f"  gen=8 是 gen=32 的前缀（同 seed、贪心）: {prefix_ok}")
    print("  含义：batch 队列把第 n 步的采样放到第 n+1 步才取回，")
    print("        若采样结果留在被下一轮覆盖的缓冲里，这两项就会失败。")
    print("  结构证据：见 C1 打印的 batch_queue_size 与 has_queue。")


def print_sources():
    """源码行号：运行时定位，避免抄文档。"""
    import inspect
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    print("\n  源码入口：")
    for cls, sym in [(EngineCore, "step"),
                     (EngineCore, "step_with_batch_queue"),
                     (GPUModelRunner, "execute_model"),
                     (GPUModelRunner, "sample_tokens")]:
        try:
            f = getattr(cls, sym)
            print(f"    {cls.__name__}.{sym:<24} "
                  f"{inspect.getsourcefile(f)}:{inspect.getsourcelines(f)[1]}")
        except Exception as exc:                                 # noqa: BLE001
            print(f"    {cls.__name__}.{sym}: {exc}")


def main():
    import torch
    print(f"torch {torch.__version__}  model {MODEL}", flush=True)
    want = [s.upper() for s in sys.argv[1:]] or ["C1", "C4", "C5"]
    if "C4" in want and "C1" not in want:
        # C4 需要 C1 实测的 wall/cpu 中位数，不接受任何占位数值。
        want = ["C1"] + [w for w in want if w != "C4"] + ["C4"]
    if "C1" in want or "C4" in want:
        print_sources()
    med = None
    if "C1" in want:
        _, med = section_C1()
    if "C4" in want:
        section_C4(med)
    if "C5" in want:
        section_C5()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
