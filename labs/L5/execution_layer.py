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


def safe_util(reserve_gib=6.0, cap=0.55, floor=0.22):
    """按当前空闲显存留出余量再折算 gpu_memory_utilization。

    crater 上可能同时有别的 lab 在跑（本 lab 不挤占其它任务），所以预算按
    「当前空闲 - 余量」算；floor 保证权重(约 3.2 GiB)+激活+KV 至少装得下。
    """
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    util = (free / gib - reserve_gib) / (total / gib)
    util = max(min(cap, util), floor)
    print(f"  [mem] free={free / gib:.2f} GiB -> gpu_memory_utilization={util:.4f}")
    return util


def make_llm(**kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=_UTIL_OVERRIDE
             if _UTIL_OVERRIDE is not None else safe_util(),
             max_model_len=4096, enable_prefix_caching=False,
             disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


# 同一节里要连开几个引擎做对照时，必须把 gpu_memory_utilization 钉成同一个值：
# 上一个引擎的显存不一定在下一个引擎取快照前全部归还，按「当前空闲」重算会让
# 后开的引擎拿到更小的预算，KV blocks 那一列就不可比了。
_UTIL_OVERRIDE = None


def pin_util_once():
    global _UTIL_OVERRIDE
    _UTIL_OVERRIDE = safe_util()
    return _UTIL_OVERRIDE


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


# ---------------------------------------------------------------- E
# 五个执行配置：把「torch.compile 分段」与「CUDA Graph」分开，并给重复分布。
EXEC_MODES = [
    ("eager(NONE)", dict(enforce_eager=True)),
    ("COMPILE_ONLY", dict(compilation_config={"cudagraph_mode": "NONE"})),
    ("PIECEWISE", dict(compilation_config={"cudagraph_mode": "PIECEWISE"})),
    ("FULL_DECODE_ONLY", dict(compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})),
    ("FULL_AND_PIECEWISE", {}),
]


def decode_step_ms(llm, B, gen=48, plen=64, seed=0):
    """一轮差分计时：返回 (每步 ms, tok/s)。prefill 由 (gen+1) 与 1 两次调用抵消。"""
    import random
    from vllm import SamplingParams, TokensPrompt
    rng = random.Random(seed)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(plen, rng)) for _ in range(B)]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spG = SamplingParams(max_tokens=gen + 1, temperature=0.0, ignore_eos=True)
    llm.generate(ps, sp1, use_tqdm=False)                   # 预热（含图捕获）
    t0 = time.perf_counter(); llm.generate(ps, sp1, use_tqdm=False)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); llm.generate(ps, spG, use_tqdm=False)
    tG = time.perf_counter() - t0
    per_step = (tG - t1) / gen
    return per_step * 1000, B / per_step


def section_E(rounds=5, batches=(1, 4, 16, 64)):
    title("[E] 同请求对照五个执行配置：启动、捕获、稳态与重复分布")

    print("  五种配置的唯一区别是执行层：")
    print("    eager(NONE)        不编译、不捕获")
    print("    COMPILE_ONLY       torch.compile 分段，无 CUDA Graph")
    print("    PIECEWISE          prefill/decode 都按编译段捕获小图")
    print("    FULL_DECODE_ONLY   decode 整步一张图，prefill 不捕获")
    print("    FULL_AND_PIECEWISE 默认：decode 整步图 + prefill 段图")
    print(f"\n  协议：每个配置一个引擎；batch 轮转 {rounds} 轮交错测量；")
    print("        每轮用 (gen+1) 与 1 次调用的差分扣掉 prefill。")

    data = {}
    meta = {}
    pin_util_once()          # 五个引擎共用同一份预算，KV blocks 才可比
    for label, kw in EXEC_MODES:
        t0 = time.perf_counter()
        llm = make_llm(**kw)
        t_init = time.perf_counter() - t0
        mr = (llm.llm_engine.engine_core.engine_core.model_executor
              .driver_worker.worker.model_runner)
        cgm = mr.cudagraph_manager
        kv = mr.kv_cache_config
        meta[label] = dict(
            init_s=round(t_init, 2),
            graphs=len(cgm.graphs) if cgm is not None else 0,
            blocks=getattr(kv, "num_blocks", None),
            free_after_gib=round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2),
        )
        print(f"  {label:<20} 启动 {t_init:>6.2f}s  捕获图 "
              f"{meta[label]['graphs']:>3}  KV blocks {meta[label]['blocks']}  "
              f"空闲 {meta[label]['free_after_gib']} GiB")
        data[label] = {B: [] for B in batches}
        for _ in range(rounds):
            for B in batches:
                ms, tps = decode_step_ms(llm, B)
                data[label][B].append((ms, tps))
        shutdown(llm)

    print(f"\n  每步耗时 ms（{rounds} 轮交错的中位数，括号内为 min–max）")
    head = f"  {'batch':>6}" + "".join(f"{lab:>22}" for lab, _ in EXEC_MODES)
    print(head)
    med = {}
    for B in batches:
        row = f"  {B:>6}"
        for label, _ in EXEC_MODES:
            xs = sorted(x[0] for x in data[label][B])
            m = xs[len(xs) // 2]
            med[(label, B)] = m
            row += f"{m:>15.3f} ({xs[0]:.2f}–{xs[-1]:.2f})"
        print(row)

    print(f"\n  相对 eager 的加速比（用中位数）")
    print(f"  {'batch':>6}" + "".join(f"{lab:>18}" for lab, _ in EXEC_MODES[1:]))
    for B in batches:
        row = f"  {B:>6}"
        for label, _ in EXEC_MODES[1:]:
            row += f"{med[('eager(NONE)', B)] / med[(label, B)]:>17.2f}×"
        print(row)

    print("\n  非单调检查：加速比是否随 batch 单调变化 ——")
    for label, _ in EXEC_MODES[1:]:
        sp = [med[('eager(NONE)', B)] / med[(label, B)] for B in batches]
        mono = all(a >= b for a, b in zip(sp, sp[1:])) or \
            all(a <= b for a, b in zip(sp, sp[1:]))
        print(f"    {label:<20} " + " ".join(f"{x:.2f}×" for x in sp) +
              ("  单调" if mono else "  **非单调**"))
    print("\n  若同一配置在不同轮之间把顺序反过来，差值仍在 min–max 之内，")
    print("  说明单次采样不足以支撑趋势结论（原 A 节的非单调点即属此类）。")
    return data, meta


# ---------------------------------------------------------------- F
def section_F():
    title("[F] 不同捕获桶：启动、KV 容量与 padding 代价")

    from vllm import SamplingParams, TokensPrompt

    buckets = [
        ("默认 51 个", None),
        ("稀疏 [1,8,64,512]", [1, 8, 64, 512]),
        ("中等 [1,8,16,32,64,128]", [1, 8, 16, 32, 64, 128]),
    ]
    print(f"  {'桶':<24} {'启动 s':>8} {'KV blocks':>10} {'捕获图':>7}  "
          f"{'B=17':>9} {'B=33':>9} {'B=48':>9}")
    pin_util_once()          # 三个桶共用同一份预算，KV blocks 才可比
    rows = {}
    for name, sizes in buckets:
        kw = {} if sizes is None else {
            "compilation_config": {"cudagraph_capture_sizes": sizes}}
        t0 = time.perf_counter()
        llm = make_llm(**kw)
        t_init = time.perf_counter() - t0
        mr = (llm.llm_engine.engine_core.engine_core.model_executor
              .driver_worker.worker.model_runner)
        cgm = mr.cudagraph_manager
        sizes_eff = llm.llm_engine.vllm_config.compilation_config.cudagraph_capture_sizes
        kv = mr.kv_cache_config
        step = {}
        for B in (17, 33, 48):
            ms, _ = decode_step_ms(llm, B, gen=32)
            step[B] = ms
        rows[name] = (t_init, getattr(kv, "num_blocks", None), len(cgm.graphs),
                      step, sizes_eff)
        print(f"  {name:<24} {t_init:>8.2f} {str(getattr(kv, 'num_blocks', None)):>10} "
              f"{len(cgm.graphs):>7}  " +
              " ".join(f"{step[B]:>9.3f}" for B in (17, 33, 48)))
        shutdown(llm)

    print("\n  每个桶把 B=17/33 实际 pad 到哪：")
    for name, (_t, _b, _g, step, sizes_eff) in rows.items():
        pad = {B: min([s for s in sizes_eff if s >= B], default=B) for B in (17, 33, 48)}
        print(f"    {name:<24} " +
              "  ".join(f"B={B}->{pad[B]}({pad[B] / B:.2f}×)" for B in (17, 33, 48)))
    print("\n  判据：稀疏桶的 padding 更大，若每步时间不随之变差，")
    print("  说明这个负载下 GPU 不是瓶颈（见 G 节的 CPU/GPU 分解）。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D,
            "E": section_E, "F": section_F}
# CPU/GPU 归因（原 G 节）改用 nsys：torch profiler 的 device 时间会把 kernel
# 串行化并叠加注入开销，实测给出「每步 GPU 忙 5.8 ms」而墙钟只有 4.4 ms。
# 现在由 labs/L5/run_execution_attribution.sh + summarize_attribution.py 提供。


# ------------------------------------------------- 单配置进程（代价表用）
def cost_one(spec, util):
    """在一个干净进程里只建一个引擎，打印启动时间与容量。

    同进程连续建引擎时，前一个引擎的显存不一定归还（实测空闲显存逐次下降），
    所以「图的代价」必须在各自独立的进程里量，否则 KV blocks 那一列不可比。
    """
    global _UTIL_OVERRIDE
    _UTIL_OVERRIDE = util
    if spec.startswith("bucket:"):
        sizes = [int(x) for x in spec.split(":", 1)[1].split(",")]
        label = "桶 " + str(sizes)
        kw = {"compilation_config": {"cudagraph_capture_sizes": sizes}}
    else:
        label, kw = next((l, k) for l, k in EXEC_MODES if l == spec)
        sizes = None
    free0 = torch.cuda.mem_get_info()[0] / 1024 ** 3
    t0 = time.perf_counter()
    llm = make_llm(**kw)
    t_init = time.perf_counter() - t0
    mr = (llm.llm_engine.engine_core.engine_core.model_executor
          .driver_worker.worker.model_runner)
    cgm = mr.cudagraph_manager
    kv = mr.kv_cache_config
    eff = llm.llm_engine.vllm_config.compilation_config.cudagraph_capture_sizes
    steps = {}
    if sizes is not None:
        for B in (17, 33, 48):
            ms, _ = decode_step_ms(llm, B, gen=32)
            steps[B] = ms
    print(f"COST name={label} spec={spec} util={util:.4f} "
          f"init_s={t_init:.2f} free_before={free0:.2f} "
          f"kv_blocks={getattr(kv, 'num_blocks', None)} "
          f"graphs={len(cgm.graphs) if cgm is not None else 0} "
          f"capture_sizes={len(eff)} "
          f"B17={steps.get(17, float('nan')):.3f} "
          f"B33={steps.get(33, float('nan')):.3f} "
          f"B48={steps.get(48, float('nan')):.3f}", flush=True)
    shutdown(llm)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cost":
        print(f"torch {torch.__version__}  model {MODEL}")
        util = float(os.environ.get("L54_UTIL", "0.35"))
        cost_one(sys.argv[2], util)
        sys.stdout.flush()
        os._exit(0)
    want = [s.upper() for s in sys.argv[1:]] or ["A", "B", "C", "D"]
    print(f"torch {torch.__version__}  model {MODEL}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
