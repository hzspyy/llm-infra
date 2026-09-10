#!/usr/bin/env python3
"""L5.5 —— 投机解码：拿闲着的算力换掉串行的步数。

3.3/5.1 已经证明 decode 是带宽受限的：算力大把闲着。
投机解码就是花那些闲算力：先猜 k 个 token，再用一次前向**并行验证**它们。
猜对了就一次前进 k+1 步，猜错了退回。

  [A] 接受率完全由任务决定（ngram 在能抄的地方好，在要创作的地方差）
  [B] 加速比 vs batch：闲算力被 batch 吃掉之后就没得花了
  [C] 猜几个：num_speculative_tokens 的取舍
  [D] 算一遍账：什么条件下投机才划算

用法：
    python speculative.py
    python speculative.py A B
"""

import os
import sys
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MB = 1024 * 1024
MODEL = os.environ.get("L55_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def make_llm(spec=None, **kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(),
             max_model_len=8192, enforce_eager=True,
             enable_prefix_caching=False,
             # 关键：disable_log_stats=True 时 scheduler 根本不产生
             # spec_decoding_stats，接受率就永远读不到。
             disable_log_stats=(spec is None))
    if spec:
        d["speculative_config"] = spec
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


# 两种任务：一种输出大量抄 prompt，一种完全自由生成
DOC = ("The memory bandwidth of a GPU determines how fast weights and KV cache "
       "can be read. Prefill is compute bound because it processes many tokens "
       "at once. Decode is memory bound because it processes one token at a time. "
       "Continuous batching improves throughput by backfilling finished slots. "
       "Chunked prefill splits a long prompt into smaller pieces so that decode "
       "steps are not blocked. Speculative decoding trades spare compute for "
       "fewer sequential steps. ")

TASK_COPY = ("Repeat the following text exactly, word for word:\n\n"
             + DOC * 3 + "\n\nRepeat it now:\n" + DOC)
TASK_FREE = ("Write an original short story about a lighthouse keeper who "
             "discovers something unusual in the fog. Be creative and do not "
             "repeat any phrase.\n\n")


def run(llm, prompt, n_out, B=1):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
    ps = [prompt] * B
    llm.generate(ps, sp, use_tqdm=False)          # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    return ntok / dt, dt * 1000, ntok


# 接受统计在 scheduler.make_spec_decoding_stats() 里产生，
# 每步产生一次就被消费掉（make_stats 里读到的是 None）。
# 所以要包住那个方法自己累加 —— 和 4.4 钩 router 是同一个手法。
_ACC = {}


def hook_spec_stats(llm):
    _ACC.clear()
    _ACC.update(drafts=0, draft_tokens=0, accepted=0, n=0)
    core = llm.llm_engine.engine_core
    sched = getattr(core, "engine_core", core).scheduler
    orig = sched.make_spec_decoding_stats

    def wrapped(*a, **kw):
        st = orig(*a, **kw)
        if st is not None:
            _ACC["drafts"] += getattr(st, "num_drafts", 0) or 0
            _ACC["draft_tokens"] += getattr(st, "num_draft_tokens", 0) or 0
            acc = getattr(st, "num_accepted_tokens", None)
            if acc is None:
                pos = getattr(st, "num_accepted_tokens_per_pos", None)
                acc = sum(pos) if pos else 0
            _ACC["accepted"] += acc
            _ACC["n"] += 1
        return st

    sched.make_spec_decoding_stats = wrapped


def spec_metrics(llm):
    if not _ACC.get("draft_tokens"):
        return None
    return dict(_ACC)


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 接受率完全由任务决定")

    print("  ngram（prompt lookup）的猜法：在已生成的文本 + prompt 里找")
    print("  和当前后缀匹配的 n-gram，把它后面的几个 token 拿来当草稿。")
    print("  所以它在**输出会抄输入**的任务上很准，在自由创作上基本没用。")

    K = 4
    spec = {"method": "ngram", "num_speculative_tokens": K,
            "prompt_lookup_max": 4, "prompt_lookup_min": 2}

    base = {}
    llm = make_llm()
    for name, pr in [("抄写任务", TASK_COPY), ("自由创作", TASK_FREE)]:
        tps, ms, n = run(llm, pr, 200)
        base[name] = tps
        print(f"\n  [无投机] {name:<8} {tps:>8.1f} tok/s  ({ms:.0f} ms / {n} tok)")
    shutdown(llm)

    llm = make_llm(spec=spec)
    for name, pr in [("抄写任务", TASK_COPY), ("自由创作", TASK_FREE)]:
        hook_spec_stats(llm)
        tps, ms, n = run(llm, pr, 200)
        m = spec_metrics(llm)
        acc = ""
        if m and m.get("draft_tokens"):
            acc = (f"  接受率 {m['accepted'] / m['draft_tokens']:.1%}"
                   f"（{m['accepted']}/{m['draft_tokens']}）")
        print(f"  [投机 k={K}] {name:<8} {tps:>8.1f} tok/s  "
              f"加速 {tps / base[name]:.2f}×{acc}")
    shutdown(llm)

    print("\n  同一个引擎、同一份配置，两个任务的加速比可能相差很远 ——")
    print("  **投机解码的收益不是引擎的属性，是「任务 × 草稿方法」的属性。**")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 加速比 vs batch：闲算力被吃掉之后就没得花了")

    print("  5.1 实测：decode 在 batch<72 时带宽受限，算力闲着。")
    print("  投机就是花那些闲算力。batch 一大，算力不闲了，投机就无利可图。")

    K = 4
    spec = {"method": "ngram", "num_speculative_tokens": K,
            "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    rows = {}
    for label, sp in [("无投机", None), (f"投机 k={K}", spec)]:
        llm = make_llm(spec=sp)
        rows[label] = {}
        for B in [1, 4, 16, 64]:
            tps, ms, n = run(llm, TASK_COPY, 128, B=B)
            rows[label][B] = tps
        shutdown(llm)

    print(f"\n  抄写任务，batch 扫描")
    print(f"  {'batch':>6} {'无投机 tok/s':>14} {'投机 tok/s':>13} {'加速':>8}")
    for B in [1, 4, 16, 64]:
        a, b = rows["无投机"][B], rows[f"投机 k={K}"][B]
        print(f"  {B:>6} {a:>14.1f} {b:>13.1f} {b / a:>7.2f}×")
    print("\n  加速比随 batch 下降 —— 这是投机解码最重要的一条性质。")
    print("  它在**低并发、要低延迟**的场景最有价值，高并发下反而是负担。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 猜几个：num_speculative_tokens 的取舍")

    print("  猜 k 个：验证一次要算 k+1 个位置的前向。")
    print("  接受 a 个（0<=a<=k）就前进 a+1 步。")
    print("  k 越大，猜中时赚得越多，猜不中时浪费越多。")

    rows = []
    llm0 = make_llm()
    base, _, _ = run(llm0, TASK_COPY, 200)
    shutdown(llm0)
    print(f"\n  基准（无投机）{base:.1f} tok/s")
    print(f"  {'k':>4} {'tok/s':>10} {'加速':>8}")
    for K in [1, 2, 4, 8]:
        spec = {"method": "ngram", "num_speculative_tokens": K,
                "prompt_lookup_max": 4, "prompt_lookup_min": 2}
        try:
            llm = make_llm(spec=spec)
            tps, _, _ = run(llm, TASK_COPY, 200)
            rows.append((K, tps))
            print(f"  {K:>4} {tps:>10.1f} {tps / base:>7.2f}×")
            shutdown(llm)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {K:>4}  失败 {str(exc)[:70]}")
            torch.cuda.empty_cache()
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\n  本任务上最好的 k = {best[0]}（{best[1]:.1f} tok/s）。")
        print("  最优 k 取决于接受率：接受率高就该猜多点，低就该少猜。")


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 算一遍账：什么条件下投机才划算")

    print("  记：")
    print("    p  = 每个草稿 token 的接受概率（近似独立）")
    print("    k  = 一次猜几个")
    print("    c  = 验证一步相对普通 decode 一步的耗时倍数（>=1）")
    print()
    print("  一次投机迭代期望前进的 token 数（几何级数）：")
    print("      E[前进] = (1 - p^(k+1)) / (1 - p)")
    print("  每次迭代的成本是 c 步普通 decode。所以")
    print("      加速比 = E[前进] / c")
    print()
    print("  c 为什么 >= 1：验证要一次算 k+1 个位置。decode 是带宽受限的，")
    print("  多算几个位置**几乎不额外花时间**（权重只读一遍）——")
    print("  这正是投机能成立的根本原因。但 c 不会恰好等于 1：")
    print("  草稿本身要花时间（ngram 查表便宜，草稿模型很贵），")
    print("  而且 batch 大时多算的位置会真的占算力（[B] 节）。")

    print(f"\n  {'p':>6}" + "".join(f"{f'k={k}':>9}" for k in [1, 2, 4, 8]))
    for p in [0.3, 0.5, 0.7, 0.8, 0.9]:
        row = f"  {p:>6.1f}"
        for k in [1, 2, 4, 8]:
            adv = (1 - p ** (k + 1)) / (1 - p)
            row += f"{adv:>9.2f}"
        print(row)
    print("  （表里是 E[前进]，还要除以 c 才是加速比）")
    print()
    print("  读法：p=0.5 时猜 8 个只比猜 4 个好一点点（1.99 vs 1.94），")
    print("  但成本 c 随 k 上升 —— 所以低接受率下猜多是亏的。")
    print("  p=0.9 时猜 8 个能前进 5.7 步，值得。")
    print()
    print("  **临界条件：E[前进] > c。** 接受率太低时投机一定亏，")
    print("  因为 c>=1 而 E[前进] 最小是 1。")


# ---------------------------------------------------------------- E
def section_E():
    title("[E] 这个版本支持哪些草稿方法（源码事实）")

    import inspect
    import re
    import vllm.config.speculative as spmod
    src = inspect.getsource(spmod)

    def lit(name):
        m = re.search(rf"\b{name}\s*=\s*Literal\[(.*?)\]", src, re.S)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []

    top = lit("SpeculativeMethod")
    print(f"  SpeculativeMethod（顶层）= {top}")
    for name in ["EagleModelTypes", "MTPModelTypes", "NgramGPUTypes",
                 "DSparkModelTypes", "DFlashModelTypes"]:
        v = lit(name)
        print(f"\n  {name}（{len(v)} 项）")
        print(f"    {v[:8]}{' ...' if len(v) > 8 else ''}")

    print("\n  两点值得注意：")
    print("  1. **MTP 有 26 个模型类型**（deepseek_mtp / qwen3_next_mtp / glm4_moe_mtp /")
    print("     kimi_k3_mtp / gemma4_mtp ...）—— 说明新模型普遍自带 MTP 头。")
    print("  2. **MTP 被归在 EagleModelTypes 里** —— 在 vLLM 里它走的是")
    print("     EAGLE 那条运行时路径，区别在权重从哪来，不在验证机制。")
    print("\n  本 lab 只实测了 ngram（不需要额外权重）。MTP/EAGLE 需要带草稿头的")
    print("  模型，这张卡装不下（DeepSeek-V3 类）—— 见正文「没测的部分」。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D,
            "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  model {MODEL}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
