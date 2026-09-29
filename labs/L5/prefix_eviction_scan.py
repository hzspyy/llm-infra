#!/usr/bin/env python3
"""L5.2 补测 · 前缀缓存的容量拐点与两引擎驱逐策略对照。

两件事，同一个负载生成器：

  * **容量拐点**：固定工作集（互异前缀条数 × 长度），把 KV 池容量从工作集的
    0.375× 扫到 1.5×，逐轮读真实命中量。拐点位置直接说明"缓存装得下"与
    "装不下"的分界，以及装不下时命中如何随容量比例变化。
  * **驱逐策略对照**：把池压到工作集的一部分，跑「热点反复访问 → 冷点冲刷 →
    再问热点」的三段负载。vLLM 0.29.0 的块池是 LRU 双向链表；SGLang 0.5.19 的
    基数树可选 `lru/lfu/slru/priority` 四种策略，同一负载下命中差异就是策略差异。

命中读数都取引擎自己报的量，不由计时反推：

  * vLLM：`KVCacheManager.get_computed_blocks()` 返回的第二个值（命中 token 数）；
  * SGLang：`meta_info['cached_tokens']`。

用法（各自 venv 下运行）：
    python prefix_eviction_scan.py --engine vllm  --mode capacity --out <dir>
    python prefix_eviction_scan.py --engine sglang --mode policy --out <dir> \
        --policies lru,lfu,slru
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L52_MODEL", "Qwen/Qwen3-1.7B")


def gen(rng, n):
    return [rng.randint(1000, 60000) for _ in range(n)]


# ------------------------------------------------------------------ vLLM
class VllmProbe:
    def __init__(self, budget, block_size=16, max_model_len=512, max_batch=512):
        import torch
        from vllm import LLM
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        self.hits: list[int] = []
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        util = min(0.55, max(free / gib - 4.0, 1.0) / (total / gib))
        kw = dict(model=MODEL, max_model_len=max_model_len, disable_log_stats=False,
                  enable_prefix_caching=True, enforce_eager=True,
                  gpu_memory_utilization=util, block_size=block_size,
                  max_num_batched_tokens=max_batch)
        if budget:
            kw["num_gpu_blocks_override"] = int(budget)
        self.llm = LLM(**kw)
        self.budget = budget
        self.max_batch = max_batch

        probe = self
        if not getattr(KVCacheManager, "_l52_probe", False):
            orig = KVCacheManager.get_computed_blocks

            def get_computed_blocks(self, request):
                blocks, n, boundary = orig(self, request)
                VllmProbe._current.hits.append(int(n))
                return blocks, n, boundary

            KVCacheManager.get_computed_blocks = get_computed_blocks
            KVCacheManager._l52_probe = True
        VllmProbe._current = self

    def ask(self, ids, out_len=1):
        from vllm import SamplingParams, TokensPrompt
        self.hits.clear()
        self.llm.generate([TokensPrompt(prompt_token_ids=ids)],
                          SamplingParams(max_tokens=out_len, temperature=0.0,
                                         ignore_eos=True), use_tqdm=False)
        return self.hits[-1] if self.hits else 0

    def close(self):
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:                                          # noqa: BLE001
            pass
        del self.llm
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ SGLang
class SglangProbe:
    def __init__(self, budget, page_size=16, policy=None, mem_fraction=0.5,
                 chunked_prefill_size=8192):
        import sglang as sgl
        kw = dict(model_path=MODEL, tp_size=1, page_size=page_size,
                  disable_radix_cache=False, random_seed=0, log_level="error",
                  mem_fraction_static=mem_fraction,
                  chunked_prefill_size=chunked_prefill_size)
        if budget:
            kw["max_total_tokens"] = int(budget)
        if policy:
            kw["radix_eviction_policy"] = policy
        self.engine = sgl.Engine(**kw)
        self.budget = budget
        self.policy = policy

    def ask(self, ids, out_len=1):
        outs = self.engine.generate(
            input_ids=[ids],
            sampling_params=dict(max_new_tokens=out_len, temperature=0.0,
                                 ignore_eos=True))
        if isinstance(outs, dict):
            outs = [outs]
        meta = (outs[0].get("meta_info") or {}) if outs else {}
        return int(meta.get("cached_tokens") or 0)

    def close(self):
        try:
            self.engine.shutdown()
        except Exception:                                          # noqa: BLE001
            pass
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass


# ------------------------------------------------------------------ 负载
def capacity_scan(factory, budgets, work_ids, rounds, out):
    """逐档扫池容量。

    第 1 轮全 miss 是填充；填充完成后**立刻回访最后一条**，用来区分两种"0 命中"：
    缓存被引擎关掉（回访也 0），还是被后续访问驱逐（回访命中）。
    """
    plen = len(work_ids[0])
    rows = []
    out.append(f"    {'预算':>9}{'工作集/预算':>12}{'第1轮命中%':>11}"
               f"{'末条回访%':>10}{'第2轮命中%':>11}{'第3轮命中%':>11}{'第2轮最小':>10}")
    for b in budgets:
        probe = factory(b)
        per = [[probe.ask(x) for x in work_ids]]
        last_probe = probe.ask(work_ids[-1])
        for _ in range(max(0, rounds - 1)):
            per.append([probe.ask(x) for x in work_ids])
        probe.close()
        pct = [100.0 * statistics.mean(r) / plen for r in per]
        rows.append(dict(budget=b, work=len(work_ids) * plen, prompt=plen,
                         rounds=per, pct=[round(x, 1) for x in pct],
                         last_probe=last_probe,
                         last_probe_pct=round(100.0 * last_probe / plen, 1),
                         r2_min=min(per[1]) if len(per) > 1 else None))
        out.append(f"    {b:>9}{rows[-1]['work'] / b:>12.2f}{pct[0]:>11.1f}"
                   f"{rows[-1]['last_probe_pct']:>10.1f}"
                   + "".join(f"{x:>11.1f}" for x in pct[1:3])
                   + f"{(rows[-1]['r2_min'] if rows[-1]['r2_min'] is not None else -1):>10}")
    return rows


def policy_scan(factory, budgets, hot, cold, out, policy_label=""):
    """热点三访 → 冷点冲刷 → 再问热点与冷点。"""
    plen = len(hot[0])
    rows = []
    for b in budgets:
        probe = factory(b)
        warm = [[probe.ask(x) for x in hot] for _ in range(3)]
        churn = [probe.ask(x) for x in cold]
        hot2 = [probe.ask(x) for x in hot]
        cold2 = [probe.ask(x) for x in cold]
        probe.close()
        rows.append(dict(
            budget=b, policy=policy_label, prompt=plen,
            warm_pct=[round(100.0 * statistics.mean(r) / plen, 1) for r in warm],
            churn_pct=round(100.0 * statistics.mean(churn) / plen, 1),
            hot_probe_pct=round(100.0 * statistics.mean(hot2) / plen, 1),
            cold_probe_pct=round(100.0 * statistics.mean(cold2) / plen, 1),
            hot_probe=hot2, cold_probe=cold2))
        out.append(f"    {policy_label or '-':>8} 预算 {b}：热点前三次命中 "
                   f"{rows[-1]['warm_pct']}%，冷点冲刷 {rows[-1]['churn_pct']}%，"
                   f"回到热点 {rows[-1]['hot_probe_pct']}%，再问冷点 "
                   f"{rows[-1]['cold_probe_pct']}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--mode", choices=["capacity", "policy"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-len", type=int, default=256)
    ap.add_argument("--n-prefixes", type=int, default=40)
    ap.add_argument("--n-hot", type=int, default=8)
    ap.add_argument("--n-cold", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=512,
                    help="vLLM 的 max_num_batched_tokens；预算小时必须一起压低，"
                         "否则调度预留会吃掉整个 KV 池")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--policies", default="lru")
    ap.add_argument("--budgets", default="",
                    help="逗号分隔；留空则按工作集比例自动生成")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免不同预算档互相覆盖")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = random.Random(20260922)
    prefixes = [gen(rng, args.prefix_len) for _ in range(args.n_prefixes)]
    hot = [gen(rng, args.prefix_len) for _ in range(args.n_hot)]
    cold = [gen(rng, args.prefix_len) for _ in range(args.n_cold)]

    out, rep = [], {}
    out.append(f"L5.2 补测 · 前缀缓存容量与驱逐策略 · {args.engine} · {args.mode} · "
               f"prompt {args.prefix_len} token、前缀 {args.n_prefixes} 条")
    t0 = time.time()

    if args.engine == "vllm":
        unit_blocks = args.prefix_len // args.block_size
        work_blocks = args.n_prefixes * unit_blocks
        if args.budgets:
            budgets = [int(x) for x in args.budgets.split(",")]
        else:
            budgets = [int(work_blocks * r) for r in (0.375, 0.625, 0.875, 1.0, 1.125, 1.5)]
        factory = lambda b: VllmProbe(b, block_size=args.block_size,     # noqa: E731
                                      max_batch=args.max_batch)
        out.append(f"  工作集 {work_blocks} 块（{args.n_prefixes}×{unit_blocks}）"
                   f"；预算 {budgets} 块；max_num_batched_tokens={args.max_batch}")
    else:
        work_tokens = args.n_prefixes * args.prefix_len
        if args.budgets:
            budgets = [int(x) for x in args.budgets.split(",")]
        else:
            budgets = [int(work_tokens * r) for r in (0.375, 0.625, 0.875, 1.0, 1.125, 1.5)]
        if args.mode == "policy":
            policies = [p for p in args.policies.split(",") if p]
            factory = lambda b, p=None: SglangProbe(b, page_size=args.page_size,   # noqa: E731
                                                    policy=p)
        else:
            policies = [None]
            factory = lambda b: SglangProbe(b, page_size=args.page_size)           # noqa: E731
        out.append(f"  工作集 {work_tokens} token；预算 {budgets} token；"
                   f"策略 {policies}")

    if args.mode == "capacity":
        rep["capacity"] = capacity_scan(factory, budgets, prefixes, args.rounds, out)
    else:
        rep["policy"] = []
        if args.engine == "vllm":
            rep["policy"].append(policy_scan(factory, budgets, hot, cold, out,
                                             policy_label="lru(块池)"))
        else:
            for p in policies:
                rep["policy"].append(policy_scan(
                    (lambda b, _p=p: factory(b, _p)), budgets, hot, cold, out,
                    policy_label=p))

    rep["meta"] = dict(engine=args.engine, mode=args.mode, args=vars(args),
                       elapsed_s=round(time.time() - t0, 1))
    text = "\n".join(out)
    print(text)
    stem = f"eviction_{args.mode}{('_' + args.tag) if args.tag else ''}"
    with open(os.path.join(args.out, f"{stem}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, f"{stem}.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\n写入 {args.out}/{stem}.txt（{rep['meta']['elapsed_s']} s）")
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
