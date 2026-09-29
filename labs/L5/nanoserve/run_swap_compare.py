#!/usr/bin/env python3
"""L5.8 任务 B 的对照运行：同一份长度分布下，重算 vs 换出，并比较驱逐策略。

同一条请求轨迹、同一个故意开小的块池，跑四组：

  baseline      块池足够大，不发生抢占——它的输出是正确性参照
  recompute     ResilientEngine：块不够就踢掉新请求，前缀重算
  swap-newest   SwappingEngine：换出到 pinned host，受害者=最新加入
  swap-oldest   / swap-longest_kv：换驱逐策略

每组记录：牺牲请求、重算 token、换出/换入字节、恢复时延、每请求结果与完成时延，
并与 baseline 逐 token 比对输出。索引 4 的块池压力由 `--blocks` 控制。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from engine import Request, State, OutOfBlocks
from failure import ResilientEngine
from swap import SwappingEngine
from model import PagedModel

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
MODEL = None


def get_model(num_blocks):
    global MODEL
    if MODEL is None or MODEL.num_blocks != num_blocks:
        MODEL = None
        torch.cuda.empty_cache()
        MODEL = PagedModel(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return MODEL


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


def run_group(cls, tok, prompts, blocks, max_tokens, max_num_seqs,
              arrival_spacing=None, tokens_per_req=None, **kw):
    """arrivals / tokens_per_req 只在非对称轨迹下给出。

    对称轨迹（默认）里 6 条请求的 prompt 与长度完全一样，三种受害者策略
    会选到同一批对象；要让策略产生差别，必须让长度和到达时间都不一致。
    """
    model = get_model(blocks)
    eng = cls(model, block_size=BLOCK_SIZE, max_batched_tokens=256,
              max_num_seqs=max_num_seqs, enable_prefix_cache=False,
              eos_ids=[], **kw)
    reqs = []
    for i, p in enumerate(prompts):
        mt = max_tokens if tokens_per_req is None else tokens_per_req[i]
        r = Request(f"r{i}", prompt_ids(tok, p), max_tokens=mt)
        if arrival_spacing is not None:
            # 每组建自己的到达时间基准：用组外算好的时间戳会把
            # 前面几组的耗时算进本组的时延里。
            r.arrival = time.perf_counter() + arrival_spacing * i
        reqs.append(r)
    for r in reqs:
        eng.add(r)
    t0 = time.perf_counter()
    crashed = None
    try:
        eng.run_until_idle(max_steps=20000)
    except OutOfBlocks as exc:
        crashed = str(exc)
    wall = time.perf_counter() - t0
    lat = [(r.finished_at - r.arrival) * 1000 for r in reqs if r.finished_at]
    return dict(engine=cls.__name__, crashed=crashed, wall_s=round(wall, 4),
                steps=eng.step_index,
                preempted=eng.stats.preempted,
                recomputed_tokens=eng.stats.recomputed_tokens,
                finished=sum(1 for r in reqs if r.state is State.FINISHED),
                produced={r.req_id: list(r.output_ids) for r in reqs},
                prompt_tokens=[len(r.prompt_ids) for r in reqs],
                max_tokens_per_req=[r.max_tokens for r in reqs],
                lat_p50_ms=round(statistics.median(lat), 3) if lat else None,
                lat_p99_ms=round(sorted(lat)[int(0.99 * (len(lat) - 1))], 3) if lat else None,
                leak=eng.leak_check(),
                swap=getattr(eng, "swap", None).__dict__ if getattr(eng, "swap", None) else None), eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-seqs", type=int, default=6)
    ap.add_argument("--asymmetric", action="store_true",
                    help="长度与到达时间都不对称，用来区分受害者策略")
    ap.add_argument("--pattern", default="correlated",
                    choices=["correlated", "anticorrelated", "interleaved"],
                    help="长度顺序与到达顺序的关系：相关 / 反相关 / 交错独立")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)

    filler = "Continue this story about a lighthouse keeper. "
    if args.asymmetric:
        # 长度顺序决定 newest / oldest / longest_kv / shortest_kv 是否选到同一对象：
        #   correlated     —— 后到的更长，"最新"与"最长"是同一个对象（上一轮的轨迹）
        #   anticorrelated —— 先到的最长，"最新"与"最短"是同一个对象
        #   interleaved    —— 长短交错，四个策略两两分开
        orders = {
            "correlated": [1, 2, 4, 8, 12, 16],
            "anticorrelated": [16, 12, 8, 4, 2, 1],
            "interleaved": [16, 1, 12, 2, 8, 4],
        }
        reps = orders[args.pattern]
        prompts = [filler * r for r in reps[:args.requests]]
        tokens_per_req = [24, 32, 48, 64, 80, 96][:args.requests]
        # 只要给策略一个严格的先后顺序即可；偏移取到"未来"会让还没到点
        # 就已被处理的请求算出负时延。
        arrival_spacing = 0.001
    else:
        prompts = [filler * 3] * args.requests
        tokens_per_req = None
        arrival_spacing = None

    groups = [("baseline", ResilientEngine, 256, {}),
              ("recompute", ResilientEngine, args.blocks, {}),
              ("swap-newest", SwappingEngine, args.blocks, {"victim_policy": "newest"}),
              ("swap-oldest", SwappingEngine, args.blocks, {"victim_policy": "oldest"}),
              ("swap-longest_kv", SwappingEngine, args.blocks, {"victim_policy": "longest_kv"}),
              ("swap-shortest_kv", SwappingEngine, args.blocks, {"victim_policy": "shortest_kv"})]
    results, engs = [], {}
    ref = None
    for name, cls, blocks, kw in groups:
        r, eng = run_group(cls, tok, prompts, blocks, args.max_tokens,
                           args.max_seqs, arrival_spacing=arrival_spacing,
                           tokens_per_req=tokens_per_req, **kw)
        r["label"] = name
        r["blocks"] = blocks
        r["pattern"] = args.pattern if args.asymmetric else "equal"
        r["prompt_reps"] = [r_ for r_ in (reps[:args.requests] if args.asymmetric
                                          else [3] * args.requests)]
        results.append(r)
        engs[name] = eng
        if name == "baseline":
            ref = r["produced"]
        print(f"{name:<16} blocks={blocks:<4} wall {r['wall_s']:>7.3f}s  "
              f"steps {r['steps']:>4}  抢占 {r['preempted']:>3}  "
              f"重算 token {r['recomputed_tokens']:>6}  "
              f"p50 {r['lat_p50_ms']} ms  p99 {r['lat_p99_ms']} ms  "
              f"完成 {r['finished']}/{args.requests}  泄漏 {r['leak']['leaked']}")
        if r["swap"]:
            s = r["swap"]
            print(f"{'':16} 换出 {s['swapped_out']} 次 / 换入 {s['swapped_in']} 次  "
                  f"字节 {s['bytes_out']/1e6:.2f} MB 出 / {s['bytes_in']/1e6:.2f} MB 入  "
                  f"耗时 {s['swap_out_s']*1000:.2f} ms 出 / {s['swap_in_s']*1000:.2f} ms 入")

    # 正确性判据随轨迹而变：
    #   对称轨迹（所有请求同 prompt 同长度）→ 批组成在各组间一致，可以要求逐 token 相同；
    #   非对称轨迹 → 不同调度会改变批组成，而 4.2 已实测 batch≥2 起同一请求不再
    #   bit 级可复现，此时只有"每条请求都产出应有数量的 token"是可判定的，
    #   逐 token 相同会被数值差异污染，不能当正确性判据。
    print("\n正确性判据：")
    for r in results:
        if r["label"] == "baseline":
            continue
        if args.asymmetric:
            got = [len(r["produced"].get(k, [])) for k in sorted(ref)]
            want = r["max_tokens_per_req"]
            ok = got == want and r["finished"] == args.requests
            r["token_counts_ok"] = bool(ok)
            print(f"  {r['label']:<16} token 数 {got} == 请求上限 {want} → {ok}"
                  f"（非对称轨迹不比逐 token）")
        else:
            same = r["finished"] == args.requests and all(
                r["produced"].get(k) == v for k, v in ref.items())
            r["matches_baseline"] = bool(same)
            print(f"  {r['label']:<16} 与 baseline 逐 token 相同：{same}")

    model = get_model(args.blocks)
    summary = dict(asymmetric=args.asymmetric, blocks=args.blocks, requests=args.requests,
                   max_tokens=args.max_tokens, max_num_seqs=args.max_seqs,
                   kv_bytes_per_token=model.kv_bytes_per_token(),
                   block_size=BLOCK_SIZE, results=results)
    (args.out / "swap_compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 用实测带宽把字节数折算到 Qwen3-1.7B 的真实几何
    sw = next((r["swap"] for r in results if r["swap"] and r["swap"]["swapped_out"]), None)
    if sw:
        tot_s = sw["swap_out_s"] + sw["swap_in_s"]
        tot_b = sw["bytes_out"] + sw["bytes_in"]
        bw = tot_b / tot_s if tot_s else float("nan")
        print(f"\n实测换出+换入 {tot_b/1e6:.2f} MB / {tot_s*1000:.3f} ms → {bw/1e9:.2f} GB/s")
        print(f"每 token 每序列 KV（本模型几何）{model.kv_bytes_per_token()} bytes")
        print("注：真实 vLLM 0.29.0 的抢占不做换出（scheduler.py:1405 直接把 "
              "num_computed_tokens 归零重算），以上是 nanoserve 上的协议验证。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
