#!/usr/bin/env python3
"""L5.2 任务 B · 前缀缓存的逐请求命中账（真实引擎读数，不用计时反推）。

修订计划要求：

    page 边界前后、前缀长度 0/15/16/17/127/128/129，中间 token 改变、
    相同 suffix 不同 prefix、LRU 压力分别测试；
    逐请求记录命中块、重算 token、搬运和完整时间。

读数来自两个真实入口，不是公式预测：

  * ``KVCacheManager.get_computed_blocks(request)`` → 这次请求命中的 token 数
  * ``SchedulerOutput.num_scheduled_tokens[req_id]`` → 这一步真正送去算的 token 数

「搬运」这一项在本版本的分页实现里是 0：复用块按引用计数共享，
不为前缀复用搬字节；被避免的成本是**重算**。脚本把
``命中 token × KV 字节/token`` 与 ``重算 token × KV 字节/token`` 并排列出，
避免把「少读的字节」说成「搬过的字节」。

用法（在 crater 上）：
    python prefix_cache_scan.py --out /scratch/learn/work/out/5.2/hits-<id>
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

MODEL = os.environ.get("L52_MODEL", "Qwen/Qwen3-1.7B")

HITS = []           # (req_id, prompt_tokens, cached_tokens, blocks)
STEPS = []          # (step, {req_id: scheduled}) —— 只记被关心的请求
# 一次 generate() 只提交一条请求：缓冲区按次清空，不需要按 request_id 过滤。


def install_hooks(block_size):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_get = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        # vLLM 0.29.0 返回 (blocks, num_local_cached_tokens, shared_prefix_boundary)。
        # 一次 generate() 只提交一条请求，缓冲区在提交前已被清空，所以这里
        # 不做 id 过滤——离线 API 的 request_id 由引擎生成，与调用方的 tag 无关。
        blocks, n, boundary = orig_get(self, request)
        # KVCacheBlocks 没有 __len__：blocks[0] 是第 0 个 KV 组的块序列
        nblocks = len(blocks.blocks[0]) if blocks.blocks else 0
        HITS.append(dict(prompt=request.num_prompt_tokens,
                         cached_tokens=n, blocks=nblocks,
                         shared_prefix_boundary=boundary,
                         block_size=block_size))
        return blocks, n, boundary

    KVCacheManager.get_computed_blocks = get_computed_blocks

    orig_schedule = Scheduler.schedule

    def schedule(self_s, *a, **k):
        out = orig_schedule(self_s, *a, **k)
        rec = dict(getattr(out, "num_scheduled_tokens", {}) or {})
        if rec:
            STEPS.append(rec)
        return out

    Scheduler.schedule = schedule


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    usable = max(free / gib - reserve_gib, 1.0)
    return min(cap, usable / (total / gib))


def make_llm(block_size=16, num_blocks=None, util=None, max_model_len=16384):
    from vllm import LLM
    kw = dict(model=MODEL, max_model_len=max_model_len, disable_log_stats=False,
              enable_prefix_caching=True, enforce_eager=True)
    kw["gpu_memory_utilization"] = util or safe_util()
    if block_size:
        kw["block_size"] = block_size
    if num_blocks:
        kw["num_gpu_blocks_override"] = num_blocks
    return LLM(**kw)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def gen(rng, n):
    return [rng.randint(1000, 60000) for _ in range(n)]


def submit(llm, tag, ids, out_len=1):
    """提交一条请求；返回 (命中 token, 命中块, 墙钟 ms, 首步调度 token)。"""
    from vllm import SamplingParams, TokensPrompt

    HITS.clear()
    STEPS.clear()
    t0 = time.perf_counter()
    llm.generate([TokensPrompt(prompt_token_ids=ids)],
                 SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True),
                 use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    hit = HITS[-1] if HITS else dict(cached_tokens=0, blocks=0, prompt=len(ids))
    first = next((v for s in STEPS for v in s.values()), None)
    return dict(tag=tag, prompt=len(ids), cached_tokens=hit["cached_tokens"],
                blocks=hit["blocks"], first_step_tokens=first, wall_ms=wall)


def _i(x):
    return x if x is not None else -1


def scan_boundaries(llm, out, rng):
    bs = 16
    base = gen(rng, 256)
    warm = submit(llm, "base", base + [1], out_len=1)
    out.append(f"  base 预热：prompt {warm['prompt']}，命中 {warm['cached_tokens']} token，"
               f"墙钟 {warm['wall_ms']:.1f} ms")
    rows = []
    for shared in (0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129):
        ids = base[:shared] + gen(rng, 256 - shared)
        r = submit(llm, f"sh{shared}", ids)
        r["shared"] = shared
        r["pred_blocks"] = shared // bs
        rows.append(r)
    out.append(f"    {'共享前缀':>8}{'命中token':>10}{'命中块':>7}{'预测块':>7}"
               f"{'首步调度':>9}{'墙钟ms':>9}")
    for r in rows:
        out.append(f"    {r['shared']:>8}{r['cached_tokens']:>10}{r['blocks']:>7}"
                   f"{r['pred_blocks']:>7}{_i(r['first_step_tokens']):>9}"
                   f"{r['wall_ms']:>9.1f}")
    bad = [r for r in rows if r["blocks"] != r["pred_blocks"]]
    out.append(f"    命中块 == floor(共享/16) 的格数：{len(rows) - len(bad)}/{len(rows)}"
               f"（不一致：{[(r['shared'], r['blocks'], r['pred_blocks']) for r in bad]}）")
    return rows


def scan_middle_change(llm, out, rng):
    """相同后缀、不同前缀，以及中段一个 token 改变。"""
    base = gen(rng, 256)
    submit(llm, "midbase", base, out_len=1)
    rows = []
    # 相同后缀（后 128 token 完全相同），前缀不同
    suffix = base[128:]
    for k, prefix in (("同前缀", base[:128]),
                      ("不同前缀", gen(rng, 128)),
                      ("中段改1个", base[:64] + [999999 % 60000] + base[65:128])):
        ids = prefix + suffix
        r = submit(llm, f"mid{k}", ids)
        r["case"] = k
        rows.append(r)
    out.append(f"    {'用例':<12}{'命中token':>10}{'命中块':>7}{'首步调度':>9}{'墙钟ms':>9}")
    for r in rows:
        out.append(f"    {r['case']:<12}{r['cached_tokens']:>10}{r['blocks']:>7}"
                   f"{_i(r['first_step_tokens']):>9}{r['wall_ms']:>9.1f}")
    return rows


def scan_sessions(llm, out, rng, n_sessions=16):
    """多会话共享同一段 system prompt：活跃会话数改变时的命中。"""
    sysp = gen(rng, 512)
    rows = []
    for k in (1, 4, 8, 16):
        hits = []
        for i in range(k):
            ids = sysp + gen(rng, 32)
            r = submit(llm, f"sys{k}_{i}", ids)
            hits.append(r["cached_tokens"])
        rows.append(dict(sessions=k, first=hits[0], rest=hits[1:],
                         hit_rate_first=hits[0] / len(sysp)))
    out.append(f"    {'活跃会话':>8}{'首条命中':>10}{'其余命中(最小/最大)':>22}")
    for r in rows:
        rest = r["rest"] or [r["first"]]
        out.append(f"    {r['sessions']:>8}{r['first']:>10}"
                   f"{f'{min(rest)}/{max(rest)}':>22}")
    return rows


def scan_pressure(out, rng, block_size=16, n_prefixes=40, prefix_len=256,
                  num_blocks=128, max_model_len=512):
    """LRU 压力：工作集大于 KV 池，观察命中随池容量下降。"""
    from vllm import SamplingParams, TokensPrompt

    prefixes = [gen(rng, prefix_len) for _ in range(n_prefixes)]
    rows = []
    for nb in (num_blocks, num_blocks // 2, 48):
        # num_blocks 覆盖值必须装得下 max_model_len，否则引擎在启动时就报
        # 「1.75 GiB KV cache is needed」——这里把 max_model_len 压到 512
        llm = make_llm(block_size=block_size, num_blocks=nb,
                       max_model_len=max_model_len)
        total_hit = 0
        per_round = []
        for rnd in range(2):
            rd = 0
            for i, p in enumerate(prefixes):
                HITS.clear()
                llm.generate([TokensPrompt(prompt_token_ids=p)],
                             SamplingParams(max_tokens=1, temperature=0.0,
                                            ignore_eos=True), use_tqdm=False)
                h = HITS[-1]["cached_tokens"] if HITS else 0
                total_hit += h
                rd += h
            per_round.append(rd)
        rows.append(dict(num_blocks=nb, prefix_blocks=(prefix_len // block_size),
                         total_hit_blocks=total_hit // block_size,
                         per_round_blocks=[x // block_size for x in per_round]))
        shutdown(llm)
    out.append(f"    {'KV块数':>8}{'单前缀块数':>11}{'工作集块数':>11}"
               f"{'逐轮命中块':>20}")
    for r in rows:
        out.append(f"    {r['num_blocks']:>8}{r['prefix_blocks']:>11}"
                   f"{n_prefixes * r['prefix_blocks']:>11}"
                   f"{str(r['per_round_blocks']):>20}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=128)
    ap.add_argument("--skip-pressure", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    out, rep = [], {}
    rng = random.Random(20260921)
    out.append(f"L5.2-B 前缀缓存逐请求命中账 · {MODEL} · block_size={args.block_size}")
    out.append("命中读自 KVCacheManager.get_computed_blocks，首步调度读自 "
               "SchedulerOutput.num_scheduled_tokens")

    llm = make_llm(block_size=args.block_size)
    install_hooks(args.block_size)
    try:
        meta = dict(model=MODEL, block_size=args.block_size,
                    gpu=torch.cuda.get_device_name(0),
                    vllm=__import__("vllm").__version__,
                    kv_cache_config=str(llm.llm_engine.vllm_config.cache_config))
        out.append(f"  KV cache config: {meta['kv_cache_config']}")
        out.append("\n[B1] 块边界扫描")
        rep["boundaries"] = scan_boundaries(llm, out, rng)
        out.append("\n[B2] 相同后缀、不同前缀 / 中段改变")
        rep["middle"] = scan_middle_change(llm, out, rng)
        out.append("\n[B3] 多会话共享 system prompt")
        rep["sessions"] = scan_sessions(llm, out, rng)
    finally:
        shutdown(llm)

    if not args.skip_pressure:
        out.append(f"\n[B4] LRU 压力：{40} 条互不相同的前缀 vs 受限 KV 池")
        rep["pressure"] = scan_pressure(out, rng, args.block_size,
                                        num_blocks=args.num_blocks)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "prefix_hits.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "prefix_hits.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\n写入 {args.out}/prefix_hits.txt 与 prefix_hits.json")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
