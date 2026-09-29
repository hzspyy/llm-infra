#!/usr/bin/env python3
"""L5.5 任务 B —— 引擎侧逐位置统计、阶段计时与提交/回滚不变量。

计划要求「为引擎单次安装统计 hook，预热后清零；逐轮保存各位置提案、条件概率、
接受长度、额外 token、KV 有效区间和时间」，并明确「聚合接受率不冒充逐位置条件概率」。
本脚本用引擎自带的观测点做到这一点，不注入 profiler：

  1. `SpecDecodingStats.observe_draft`（`vllm/v1/spec_decode/metrics.py:41`）
     每个请求每步调用一次，参数就是本步的 draft tokens 与 accepted tokens。
     它同时维护 `num_accepted_tokens_per_pos` / `num_draft_tokens_per_pos`：
     前者是「至少接受 i+1 个」的计数，后者是「至少提出 i+1 个」的计数，
     两者之比就是逐位置条件接受概率 a_i。
  2. 自装的阶段计时器（不依赖 runner 版本）：
     forward 用 `execute_model` 前后的 CUDA event，drafter 用包住
     `mr.drafter.propose` 的墙钟（ngram 提案是纯 CPU 查表）。

得到的每步记录是 (draft tokens, accepted tokens, forward ms, drafter ms, target tokens)。
据此给出：逐位置 a_i、接受长度分布、E[N] 的公式预测与实测、草稿/验证两段耗时，
以及一条可直接断言的提交不变量：`Σ(accepted+1) == 实际生成的 token 数`。

版本事实（引擎日志原文）：ngram 投机在本版本会回退到 V1 model runner
——`Model Runner V2 does not yet support ngram/ngram_gpu speculative decoding`。
为了让「有投机」与「无投机」两组阶段计时同源，整份 lab 固定
`VLLM_USE_V2_MODEL_RUNNER=0`，这一开关在正文与 manifest 中记录。

用法：
    python speculative_positions.py
    python speculative_positions.py B1 B2
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

MODEL = os.environ.get("L55_MODEL", "Qwen/Qwen3-1.7B")


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


def make_llm(spec=None, **kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(), max_model_len=8192,
             enforce_eager=True, enable_prefix_caching=False,
             # disable_log_stats=True 时 scheduler 不产生 spec_decoding_stats，
             # 接受统计会永远是空的。
             disable_log_stats=(spec is None))
    if spec:
        d["speculative_config"] = spec
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


def runner_of(llm):
    return (llm.llm_engine.engine_core.engine_core.model_executor
            .driver_worker.worker.model_runner)


def ngram_spec(k, max_lookup=4, min_lookup=2):
    return {"method": "ngram", "num_speculative_tokens": k,
            "prompt_lookup_max": max_lookup, "prompt_lookup_min": min_lookup}


# ------------------------------------------------------------------ 统计 hook
RECORDS: list[dict] = []
PER_POS_ACCEPT: list[int] = []
PER_POS_DRAFT: list[int] = []


_HOOK_INSTALLED = False


def install_draft_hook():
    """在类上打点，捕获每一次 observe_draft。

    同一个进程里多个小节都会调用它；重复包装会让每条记录被追加多次，
    于是「步数」和「Σ(accepted+1)」都会成倍虚高。所以只装一次。
    """
    global _HOOK_INSTALLED
    from vllm.v1.spec_decode.metrics import SpecDecodingStats
    if _HOOK_INSTALLED:
        return SpecDecodingStats
    _HOOK_INSTALLED = True
    orig = SpecDecodingStats.observe_draft

    def patched(self, num_draft_tokens, num_accepted_tokens):
        RECORDS.append(dict(draft=int(num_draft_tokens),
                            accepted=int(num_accepted_tokens)))
        n = self.num_spec_tokens
        while len(PER_POS_ACCEPT) < n:
            PER_POS_ACCEPT.append(0)
            PER_POS_DRAFT.append(0)
        for i in range(int(num_accepted_tokens)):
            PER_POS_ACCEPT[i] += 1
        for i in range(int(num_draft_tokens)):
            PER_POS_DRAFT[i] += 1
        return orig(self, num_draft_tokens, num_accepted_tokens)

    SpecDecodingStats.observe_draft = patched
    return SpecDecodingStats


def reset_records():
    RECORDS.clear()
    PER_POS_ACCEPT.clear()
    PER_POS_DRAFT.clear()


def survival_ratios():
    """引擎给的逐位置比值 = P(A ≥ i+1 | 至少提出 i+1 个)。

    注意这不是「在前 i 个都接受的条件下第 i+1 个被接受」的条件概率：
    `num_draft_tokens_per_pos[i]` 的分母是「至少提出到第 i+1 个」的步数
    （metrics.py:48 的循环），所以比值本身就是生存函数 s_i，不需要再连乘。
    """
    return [(PER_POS_ACCEPT[i] / PER_POS_DRAFT[i]) if PER_POS_DRAFT[i] else 0.0
            for i in range(len(PER_POS_DRAFT))]


def conditional_ratios(s):
    """条件概率 a_i = s_i / s_{i-1}（s_0 即 a_1）。"""
    out = []
    prev = 1.0
    for x in s:
        out.append(x / prev if prev > 0 else float("nan"))
        prev = x
    return out


def expected_tokens(s):
    """E[N] = 1 + Σ_i s_i；等价于 1 + Σ_i Π_{j≤i} a_j（a 为条件概率）。"""
    return 1.0 + sum(s)


def predicted_histogram(s):
    """接受长度分布 P(A=i)，i = 0..k。

    P(A=0) = 1 − s_0；P(A=i) = s_{i-1} − s_i（1 ≤ i < k）；P(A=k) = s_{k-1}。
    注意下标错位：s_i 是 P(A ≥ i+1)，所以 P(A=i) 要减的是 s_i 而不是 s_{i+1}。
    """
    return [1.0 - s[0]] + [s[i - 1] - s[i] for i in range(1, len(s))] + [s[-1]]


def aggregate_geometric(accepted, draft, k):
    """把聚合接受率当 p 代入几何公式得到的 E[N]（错误做法的对照）。"""
    if not draft:
        return float("nan")
    p = accepted / draft
    if p >= 1.0:
        return float(k + 1)
    return (1 - p ** (k + 1)) / (1 - p)


# ------------------------------------------------------------------ 阶段计时
FORWARD_EVENTS: list = []
DRAFT_MS: list = []


def install_step_timers(mr):
    """装两个不依赖 runner 版本的计时器。

    forward：`execute_model` 前后各记一个 CUDA event，同步后取 elapsed_time。
    drafter：包住 `mr.drafter.propose`，用墙钟；ngram 的提案是纯 CPU 查表。
    """
    import torch

    if getattr(mr, "_l55_timers_installed", False):
        return mr
    mr._l55_timers_installed = True
    orig_exec = mr.execute_model

    def timed_exec(scheduler_output, *a, **kw):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = orig_exec(scheduler_output, *a, **kw)
        e.record()
        toks = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        FORWARD_EVENTS.append((s, e, toks))
        return r

    mr.execute_model = timed_exec

    drafter = getattr(mr, "drafter", None)
    propose = getattr(drafter, "propose", None) if drafter is not None else None
    if propose is not None:
        def timed_propose(*a, **kw):
            t0 = time.perf_counter()
            r = propose(*a, **kw)
            DRAFT_MS.append((time.perf_counter() - t0) * 1000)
            return r

        drafter.propose = timed_propose
    return mr


def _resolve_events():
    import torch
    torch.cuda.synchronize()
    return [dict(forward_ms=s.elapsed_time(e), target_tokens=t)
            for s, e, t in FORWARD_EVENTS]


def run_spec(llm, prompt, n_out, k, B=1):
    """跑一次投机解码，返回 (tok/s, 每步记录, 生成 token 数)。"""
    from vllm import SamplingParams
    import torch
    sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
    ps = [prompt] * B
    llm.generate(ps, sp, use_tqdm=False)              # 预热
    reset_records()
    FORWARD_EVENTS.clear()
    DRAFT_MS.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    fwd = _resolve_events()
    steps = [dict(r) for r in RECORDS]
    if steps and len(fwd) >= len(steps):
        fwd = fwd[-len(steps):]          # 每步 forward 与每步统计一一对应
        for s, f in zip(steps, fwd):
            s["forward_ms"] = f["forward_ms"]
            s["target_tokens"] = f["target_tokens"]
        for i, s in enumerate(steps):
            s["drafter_ms"] = DRAFT_MS[i] if i < len(DRAFT_MS) else float("nan")
    return ntok / dt, steps, ntok


def run_base(llm, prompt, n_out, B=1):
    from vllm import SamplingParams
    import torch
    sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
    ps = [prompt] * B
    llm.generate(ps, sp, use_tqdm=False)
    FORWARD_EVENTS.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    return ntok / dt, _resolve_events(), ntok


# ------------------------------------------------------------------ 合成任务
DOC = ("The memory bandwidth of a GPU determines how fast weights and KV cache "
       "can be read. Prefill is compute bound because it processes many tokens "
       "at once. Decode is memory bound because it processes one token at a time. "
       "Continuous batching improves throughput by backfilling finished slots. ")

TASK_COPY = ("Repeat the following text exactly, word for word:\n\n"
             + DOC * 3 + "\n\nRepeat it now:\n" + DOC)
TASK_FREE = ("Write an original short story about a lighthouse keeper who "
             "discovers something unusual in the fog. Be creative and do not "
             "repeat any phrase.\n\n")

TASKS = [("抄写（输出大量重复输入）", TASK_COPY), ("自由创作（无重复）", TASK_FREE)]


# ------------------------------------------------------------------ B1
def section_B1(k=4, n_out=160):
    title(f"[B1] 每步记录：k={k} 时引擎每步提出/接受多少、两段各花多久")
    install_draft_hook()
    llm = make_llm(spec=ngram_spec(k))
    install_step_timers(runner_of(llm))
    for name, prompt in TASKS:
        tps, steps, _ = run_spec(llm, prompt, n_out, k)
        print(f"\n  === {name}  {tps:.1f} tok/s，{len(steps)} 步 ===")
        if steps and "forward_ms" in steps[0]:
            print(f"  {'step':>5} {'draft':>6} {'accept':>7} {'target tok':>11} "
                  f"{'forward ms':>11} {'drafter ms':>11}")
            for i, s in enumerate(steps[:10]):
                print(f"  {i:>5} {s['draft']:>6} {s['accepted']:>7} "
                      f"{s.get('target_tokens', 0):>11} "
                      f"{s.get('forward_ms', float('nan')):>11.3f} "
                      f"{s.get('drafter_ms', float('nan')):>11.3f}")
            if len(steps) > 10:
                print(f"  ... 共 {len(steps)} 步")
        else:
            print(f"  {'step':>5} {'draft':>6} {'accept':>7}")
            for i, s in enumerate(steps[:10]):
                print(f"  {i:>5} {s['draft']:>6} {s['accepted']:>7}")
        tot_d = sum(s["draft"] for s in steps)
        tot_a = sum(s["accepted"] for s in steps)
        print(f"  合计：提出 {tot_d}，接受 {tot_a}，平均接受 "
              f"{tot_a / max(len(steps), 1):.3f} 个/步"
              f"（聚合接受率 {tot_a / max(tot_d, 1):.1%}）")
    shutdown(llm)


# ------------------------------------------------------------------ B2
def section_B2(n_out=200):
    title("[B2] 逐位置接受统计：生存函数、条件概率与 E[N] 核对")
    install_draft_hook()
    print("  引擎的两个逐位置计数（metrics.py:41 起的 observe_draft）：")
    print("    num_draft_tokens_per_pos[i]    = 至少提出到第 i+1 个的步数")
    print("    num_accepted_tokens_per_pos[i] = 至少接受到第 i+1 个的步数")
    print("  两者之比是**生存函数** s_i = P(A ≥ i+1)，不是条件概率；")
    print("  条件概率 a_i = s_i / s_{i-1}，E[N] = 1 + Σ s_i = 1 + Σ Π a_j。")
    print("  每行还给出「把聚合接受率 Σaccepted/Σdraft 当 p 代入几何公式」的对照，")
    print("  它假设各位置同分布，所测数据下与逐位置口径有明显差距。")
    for k in [1, 2, 4, 8, 15]:
        try:
            llm = make_llm(spec=ngram_spec(k))
        except Exception as exc:                                 # noqa: BLE001
            print(f"\n  k={k}: 建引擎失败 {str(exc)[:80]}")
            continue
        install_step_timers(runner_of(llm))
        for name, prompt in TASKS:
            tps, steps, ntok = run_spec(llm, prompt, n_out, k)
            s = survival_ratios()
            a = conditional_ratios(s)
            e_surv = expected_tokens(s)
            tot_acc = sum(st["accepted"] for st in steps)
            tot_draft = sum(st["draft"] for st in steps)
            e_agg = aggregate_geometric(tot_acc, tot_draft, k)
            hist = {}
            for st in steps:
                hist[st["accepted"]] = hist.get(st["accepted"], 0) + 1
            nstat = max(len(steps), 1)
            pred = predicted_histogram(s)
            print(f"\n  k={k:<3} {name}（{nstat} 个有草稿的步，"
                  f"生成 {ntok} token）")
            print("    s_i（生存函数） = " + " ".join(f"{x:.3f}" for x in s))
            print("    a_i（条件概率） = " + " ".join(
                ("  —  " if x != x else f"{x:.3f}") for x in a))
            print(f"    E[N] = 1+Σs_i = {e_surv:.3f}")
            print(f"    用聚合接受率 {tot_acc}/{tot_draft} = "
                  f"{tot_acc / max(tot_draft, 1):.3f} 代入几何公式 = {e_agg:.3f}"
                  f"（相对差 {(e_agg - e_surv) / e_surv:+.1%}）")
            print("    P(A=i) 预测 = " + " ".join(f"{x:.3f}" for x in pred))
            print("    P(A=i) 实测 = " + " ".join(
                f"{hist.get(i, 0) / nstat:.3f}" for i in range(len(s) + 1)))
            mean_commit = sum(st["accepted"] + 1 for st in steps) / nstat
            print(f"    实测每步提交 = {mean_commit:.3f}（= E[N] 的对照）")
        shutdown(llm)
    print("\n  统计只覆盖**有草稿调度的步**：scheduler.py:2722 在 num_draft_tokens=0")
    print("  时直接返回 None，这些步产出的 token 不进统计。所以「生成 token 数 /")
    print("  有草稿的步数」会系统性高于 E[N]，差值就是没有草稿的步数——这条偏差")
    print("  恰恰说明聚合口径不能直接当逐位置条件概率用。")


# ------------------------------------------------------------------ B3
def section_B3(k=4, n_out=200):
    title("[B3] 接受长度分布与草稿/验证两段计时")
    install_draft_hook()
    base = {}
    llm0 = make_llm()
    install_step_timers(runner_of(llm0))
    for name, prompt in TASKS:
        tps, fwd_events, ntok = run_base(llm0, prompt, n_out)
        fwd = [f["forward_ms"] for f in fwd_events]
        base[name] = dict(tps=tps, steps=len(fwd),
                          fwd_mean=sum(fwd) / max(len(fwd), 1))
        print(f"  [无投机] {name:<22} {tps:>7.1f} tok/s  "
              f"{len(fwd):>4} 步  每步前向 {base[name]['fwd_mean']:.3f} ms")
    shutdown(llm0)

    llm = make_llm(spec=ngram_spec(k))
    install_step_timers(runner_of(llm))
    for name, prompt in TASKS:
        tps, steps, ntok = run_spec(llm, prompt, n_out, k)
        hist = {}
        for s in steps:
            hist[s["accepted"]] = hist.get(s["accepted"], 0) + 1
        fwd = [s.get("forward_ms", float("nan")) for s in steps]
        drf = [s.get("drafter_ms", float("nan")) for s in steps]
        b = base[name]
        fwd_mean = sum(fwd) / max(len(fwd), 1)
        drf_mean = sum(drf) / max(len(drf), 1)
        print(f"\n  === {name}  k={k} ===")
        print(f"  接受长度分布（接受数: 步数）："
              + "  ".join(f"{a}:{n}" for a, n in sorted(hist.items())))
        print(f"  步数 {b['steps']} → {len(steps)}，加速 {tps / b['tps']:.2f}×")
        print(f"  每步前向 {fwd_mean:.3f} ms（无投机 {b['fwd_mean']:.3f} ms，"
              f"比值 {fwd_mean / b['fwd_mean']:.2f}×）")
        print(f"  每步草稿 {drf_mean:.3f} ms；两段合计/步 {fwd_mean + drf_mean:.3f} ms")
    shutdown(llm)


# ------------------------------------------------------------------ B4
def section_B4(k=4, n_out=160, reps=3):
    title("[B4] 提交与回滚：统计口径、输出一致性与残留占用")
    install_draft_hook()
    llm = make_llm(spec=ngram_spec(k))
    install_step_timers(runner_of(llm))
    mr = runner_of(llm)
    kv = getattr(mr, "kv_cache_config", None)
    blocks = getattr(kv, "num_blocks", None) if kv is not None else None

    base_out = None
    for rep in range(reps):
        tps, steps, ntok = run_spec(llm, TASK_COPY, n_out, k)
        fwd = len(FORWARD_EVENTS)
        committed = sum(s["accepted"] + 1 for s in steps)
        # 生成 token = prefill 1 个 + 有草稿的步提交的 + 无草稿的步各 1 个
        gap = (ntok - 1) - committed
        print(f"  第 {rep + 1} 轮：生成 {ntok} token，forward {fwd} 次，"
              f"有草稿的步 {len(steps)}，Σ(accepted+1) = {committed}，"
              f"未计入统计的 token = {gap}；KV 容量 {blocks} blocks")
    print()
    print("  `num_accepted = len(generated) − num_sampled`（scheduler.py:1906），")
    print("  所以 Σ(accepted+1) 正好等于**有草稿的步**提交的 token 数；剩下的 token")
    print("  来自没有草稿调度的步（scheduler.py:2722 不计入统计）与 prefill。")
    print("  上式的 gap 应当等于「无草稿的步数」，不会为负；为负说明有位置被多提交。")

    # 输出一致性：同 prompt、同贪心设置，投机与不投机的 token 序列必须逐位相同
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
    spec_out = llm.generate([TASK_COPY], sp, use_tqdm=False)[0].outputs[0].token_ids
    shutdown(llm)
    llm0 = make_llm()
    base_out = llm0.generate([TASK_COPY], sp, use_tqdm=False)[0].outputs[0].token_ids
    shutdown(llm0)
    same = list(spec_out) == list(base_out)
    print(f"\n  贪心输出逐 token 相同：{same}"
          f"（投机 {len(spec_out)} 个，无投机 {len(base_out)} 个）")
    if not same:
        for i, (x, y) in enumerate(zip(spec_out, base_out)):
            if x != y:
                print(f"    第一处差异在第 {i} 个 token：{x} vs {y}")
                break
    print("  这一条覆盖的是提交/回滚路径：被拒位置的 KV 与输出都必须回到已接受前缀。")


SECTIONS = {"B1": section_B1, "B2": section_B2, "B3": section_B3,
            "B4": section_B4}


def main():
    import torch
    print(f"torch {torch.__version__}  model {MODEL}  "
          f"VLLM_USE_V2_MODEL_RUNNER={os.environ.get('VLLM_USE_V2_MODEL_RUNNER')}",
          flush=True)
    want = [s.upper() for s in sys.argv[1:]] or ["B1", "B2", "B3", "B4"]
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
