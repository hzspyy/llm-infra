#!/usr/bin/env python3
"""L5.4 —— 执行层：CUDA Graph 在引擎里值多少。

2.6b 量过 CUDA Graph 把 CPU 侧 launch 从 14,003 降到 863（16.2×）。
这一章问：**这 16 倍的 launch 减少，在 tok/s 上体现为多少？**

  [A] eager vs CUDA Graph：decode 吞吐随 batch 的对照
  [B] 代价一：捕获时间与显存
  [C] 代价二：只有捕获过的 batch 尺寸能用图
  [D] 什么时候图没有收益

用法：
    python execution_layer.py
"""

import os
import sys
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MB = 1024 * 1024
MODEL = os.environ.get("L54_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def make_llm(**kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(),
             max_model_len=4096, enable_prefix_caching=False,
             disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                         # noqa: BLE001
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def rand_ids(n, rng):
    return [rng.randint(1000, 60000) for _ in range(n)]


def decode_tps(llm, B, gen=64, plen=64, seed=0):
    """量纯 decode 吞吐：用 (gen+1) 与 1 两次调用做差分，扣掉 prefill。"""
    import random
    from vllm import SamplingParams, TokensPrompt
    rng = random.Random(seed)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(plen, rng)) for _ in range(B)]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spG = SamplingParams(max_tokens=gen + 1, temperature=0.0, ignore_eos=True)
    llm.generate(ps, spG, use_tqdm=False)                   # 预热（含图捕获）
    t0 = time.perf_counter(); llm.generate(ps, sp1, use_tqdm=False)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); llm.generate(ps, spG, use_tqdm=False)
    tG = time.perf_counter() - t0
    per_step = (tG - t1) / gen
    return B / per_step, per_step * 1000


# ---------------------------------------------------------------- A
def section_A():
    title("[A] eager vs CUDA Graph：decode 吞吐")

    print("  2.6b 实测：CPU 侧 launch 14,003 -> 863（16.2×），GPU kernel 数不变。")
    print("  launch 省下的是 CPU 时间，所以收益应当在**CPU 提交是瓶颈**的地方最大，")
    print("  也就是 **batch 小、模型小** 的时候（L1.4：每次 launch 2.68 µs）。")

    rows = {}
    for label, kw in [("eager（无图）", dict(enforce_eager=True)),
                      ("CUDA Graph", dict(enforce_eager=False))]:
        llm = make_llm(**kw)
        rows[label] = {}
        for B in [1, 4, 16, 64]:
            tps, ms = decode_tps(llm, B)
            rows[label][B] = (tps, ms)
        shutdown(llm)

    print(f"\n  {'batch':>6} {'eager ms/步':>13} {'图 ms/步':>11} "
          f"{'eager tok/s':>13} {'图 tok/s':>11} {'加速':>8}")
    for B in [1, 4, 16, 64]:
        e_tps, e_ms = rows["eager（无图）"][B]
        g_tps, g_ms = rows["CUDA Graph"][B]
        print(f"  {B:>6} {e_ms:>13.3f} {g_ms:>11.3f} {e_tps:>13.0f} "
              f"{g_tps:>11.0f} {g_tps / e_tps:>7.2f}×")
    e_ms = [rows["eager（无图）"][B][1] for B in [1, 4, 16, 64]]
    g_ms = [rows["CUDA Graph"][B][1] for B in [1, 4, 16, 64]]
    print(f"\n  eager 每步 {min(e_ms):.2f} -> {max(e_ms):.2f} ms（随 batch 增长）")
    print(f"  图   每步 {min(g_ms):.2f} -> {max(g_ms):.2f} ms（基本不随 batch 变）")
    print("  这是本节最清楚的一条：**开图之后每步时间几乎与 batch 无关**，")
    print("  说明这个区间里的 step 时间由 CPU 提交主导，而图把它压成了常数。")
    print()
    print("  ⚠ 加速比**不是单调的**（batch=16 是最低点，64 又回升）。")
    print("  我原本预期它随 batch 单调下降（batch 大 -> GPU 忙 -> CPU 提交被藏住），")
    print("  数据不支持这个说法。**没有查清原因，不要引用这条趋势。**")
    print("  可能的方向：eager 路径在大 batch 上的 python/dispatch 开销、")
    print("  或图捕获尺寸的 padding（见 [C]）。判据：用 2.6 的 nsys 方法")
    print("  分别抓两条路径在 batch=16 和 64 上的 CPU 时间线对比。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 代价一：捕获时间与显存")

    for label, kw in [("eager（无图）", dict(enforce_eager=True)),
                      ("CUDA Graph", dict(enforce_eager=False))]:
        torch.cuda.empty_cache()
        free0, _ = torch.cuda.mem_get_info()
        t0 = time.perf_counter()
        llm = make_llm(**kw)
        t_init = time.perf_counter() - t0
        free1, total = torch.cuda.mem_get_info()
        print(f"  {label:<14} 启动 {t_init:>7.2f} s   "
              f"启动后驱动可见空闲 {free1 / MB / 1024:>6.2f} GiB")
        shutdown(llm)
    print("\n  图捕获发生在启动时：要为每个准备捕获的 batch 尺寸各跑一遍前向并录下来。")
    print("  代价是**启动更慢**，以及**图本身占的显存池**（2.0 §4.3 里")
    print("  vLLM 的 `cudagraph_memory_estimate_applied` 就是这一项）。")
    print("  注意上面两行的空闲显存差不能直接当图的开销 —— ")
    print("  vLLM 会按 gpu_memory_utilization 把剩下的全分给 KV，两者是此消彼长的。")
    print("  **真正被图吃掉的那部分，表现为 KV cache 变小、可容纳的并发变少。**")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 代价二：只有捕获过的尺寸能用图")

    print("  CUDA Graph 把张量地址和形状都固化在图里，所以**每个 batch 尺寸**")
    print("  都要单独捕获一份。vLLM 捕获一组离散的尺寸，")
    print("  运行时把实际 batch **向上取整**到最近的那个捕获尺寸（padding）。")

    llm = make_llm(enforce_eager=False)
    cfg = llm.llm_engine.vllm_config
    try:
        sizes = cfg.compilation_config.cudagraph_capture_sizes
        print(f"\n  本次捕获的尺寸（共 {len(sizes)} 个）：")
        print(f"    {sizes}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  取捕获尺寸失败: {exc}")
        sizes = None

    if sizes:
        print("\n  判据：若 padding 真的有代价，batch=17 的每步时间应当")
        print("  接近 batch=24（它被 pad 成 24），而明显高于 batch=16。")
        print(f"  {'batch':>6} {'pad 到':>8} {'浪费':>7} {'ms/步':>9} {'tok/s':>9}")
        ms = {}
        for B in [8, 16, 17, 24, 32, 33, 40, 48, 64]:
            padded = min([s for s in sorted(sizes) if s >= B], default=B)
            tps, m = decode_tps(llm, B, gen=48)
            ms[B] = m
            print(f"  {B:>6} {padded:>8} {padded / B:>6.2f}× {m:>9.3f} {tps:>9.0f}")
        lo, hi = min(ms.values()), max(ms.values())
        print(f"\n  所有 batch 的每步时间落在 {lo:.3f}–{hi:.3f} ms，"
              f"极差只有 {(hi - lo) / lo:.1%}。")
        print("  **padding 的代价在这个区间里量不出来** —— 因为每步时间由 CPU")
        print("  提交主导（[A] 节已经看到图路径的每步时间几乎与 batch 无关），")
        print("  多算几行 GPU 计算根本不构成瓶颈。")
        print()
        print("  所以 padding 什么时候才要紧？**当 step 变成算力受限时**：")
        print("  batch 大到 GPU 真的忙起来（5.1 算过 batch≈72 越过平衡点），")
        print("  或者序列很长。本 lab 的 batch 上限 64 还没到那里。")
        print("  要看到 padding 的代价，把 batch 扫到几百（见消融 3）。")
    shutdown(llm)


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 什么时候图没有收益")

    print("  图省的是 CPU 提交时间。以下两种情况它帮不上忙：")
    print("    1. prefill —— 形状随 prompt 长度变，且 GPU 本来就很忙")
    print("    2. batch 很大 —— CPU 提交已经被 GPU 执行藏住了")

    rows = {}
    for label, kw in [("eager", dict(enforce_eager=True)),
                      ("CUDA Graph", dict(enforce_eager=False))]:
        import random
        from vllm import SamplingParams, TokensPrompt
        llm = make_llm(**kw)
        rng = random.Random(7)
        out = {}
        for S in [512, 2048]:
            ps = [TokensPrompt(prompt_token_ids=rand_ids(S, rng)) for _ in range(4)]
            sp = SamplingParams(max_tokens=1, temperature=0.0)
            llm.generate(ps, sp, use_tqdm=False)
            t0 = time.perf_counter()
            llm.generate(ps, sp, use_tqdm=False)
            out[S] = (time.perf_counter() - t0) * 1000
        rows[label] = out
        shutdown(llm)

    print(f"\n  prefill（4 条，只生成 1 个 token）")
    print(f"  {'prompt 长度':>11} {'eager ms':>10} {'图 ms':>9} {'加速':>8}")
    for S in [512, 2048]:
        e, g = rows["eager"][S], rows["CUDA Graph"][S]
        print(f"  {S:>11} {e:>10.2f} {g:>9.2f} {e / g:>7.2f}×")
    print("\n  prefill 上基本没有差别 —— 印证 2.6b 的结论：")
    print("  图减少的是**提交次数**，而 prefill 的时间由 GPU 计算主导。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  model {MODEL}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
