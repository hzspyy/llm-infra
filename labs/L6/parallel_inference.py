#!/usr/bin/env python3
"""
6.2 任务 B/C：真实模型的张量并行与流水并行部署，以及专家并行的合法路径。

任务 B：固定 BF16，对 Qwen3-8B 比较 TP=1/2/4 与 PP=2，采 TTFT、逐 token 时间、
        完成时间、实际输出数与峰值分片；保留小模型多卡减速的现象。
任务 C：对 OLMoE 运行 EP=2/4 的合法路径，记录 token / expert / 位置归属。

刻意关掉 prefix caching 与 CUDA Graph，让不同并行度之间的比较在同一口径上。

用法：
    python parallel_inference.py sweep --model <path> --tp 2 --pp 1 --out <dir>
    python parallel_inference.py ep    --model <path> --ep 2 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

COMBOS = [(1, 128), (8, 128), (1, 2048), (8, 2048), (1, 8192), (8, 8192)]
REDUCED_COMBOS = [(1, 2048), (8, 2048)]


def build_prompt_tokens(seq_len, offset=0):
    """用固定 token id 造正好 seq_len 个 token 的输入，避免分词长度抖动。"""
    return [(1000 + (i + offset) % 5000) for i in range(seq_len)]


def make_engine(model, tp, pp, gpu_util, max_model_len, enable_ep=False):
    from vllm import LLM
    kw = {}
    if enable_ep:
        # 专家并行的 rank 集合与 TP 组一致：EP=k 用 TP=k + enable_expert_parallel
        kw["enable_expert_parallel"] = True
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        **kw,
    )


def run_one(engine, batch, seq_len, max_tokens, tag):
    """逐个 step 推进，记录每个请求的首 token 时刻与总完成时刻。"""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    ids = [f"{tag}-{i}" for i in range(batch)]
    t_add = {}
    t_first = {}
    t_last = {}
    n_out = {}
    out_ids = {}
    step_times = []
    t0 = time.perf_counter()
    for i, rid in enumerate(ids):
        prompt = {"prompt_token_ids": build_prompt_tokens(seq_len, offset=i)}
        engine.add_request(rid, prompt, sp)
        t_add[rid] = time.perf_counter()
    while engine.has_unfinished_requests():
        s0 = time.perf_counter()
        outs = engine.step()
        step_times.append(time.perf_counter() - s0)
        now = time.perf_counter()
        for o in outs:
            rid = o.request_id
            n = len(o.outputs[0].token_ids)
            if n > 0 and rid not in t_first:
                t_first[rid] = now
            n_out[rid] = n
            out_ids[rid] = list(o.outputs[0].token_ids)
            if o.finished:
                t_last[rid] = now
    total = time.perf_counter() - t0
    t_add_last = max(t_add.values())
    ttfts = [t_first[r] - t_add[r] for r in ids if r in t_first]
    tpots = []
    for r in ids:
        if r in t_first and r in t_last and n_out.get(r, 0) > 1:
            tpots.append((t_last[r] - t_first[r]) / (n_out[r] - 1))
    return {
        "batch": batch, "seq_len": seq_len, "max_tokens": max_tokens,
        "total_s": total,
        "queue_s": t_add_last - t0,
        "ttft_s": ttfts, "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "tpot_s": tpots, "tpot_median_s": statistics.median(tpots) if tpots else None,
        "output_tokens": [n_out.get(r, 0) for r in ids],
        "output_token_ids": {r: out_ids.get(r, []) for r in ids},
        "all_completed": len(t_last) == batch,
        "steps": len(step_times),
        "step_median_s": statistics.median(step_times) if step_times else None,
        "throughput_tok_s": sum(n_out.values()) / total if total else None,
    }


def parse_worker_memory(log_text):
    """从 vLLM 自己的日志里取每 rank 的权重与 KV 容量。"""
    weights = [float(m) for m in
               re.findall(r"Model loading took ([\d.]+) GiB memory", log_text)]
    kv = [float(m) for m in
          re.findall(r"Available KV cache memory: ([\d.]+) GiB", log_text)]
    return {"model_loading_GiB_per_worker": weights,
            "kv_cache_GiB_per_worker": kv}


def task_sweep(args):
    import torch
    os.makedirs(args.out, exist_ok=True)
    combos = COMBOS if args.full else REDUCED_COMBOS
    max_len = max(s for _, s in combos) + args.max_tokens + 8
    t_init0 = time.perf_counter()
    llm = make_engine(args.model, args.tp, args.pp, args.gpu_util, max_len)
    engine = llm.llm_engine          # step()/add_request/has_unfinished_requests 都在这一层
    init_s = time.perf_counter() - t_init0
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 2**30
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    else:
        allocated = None

    # 预热：一次小请求把 kernel/图都跑热
    run_one(engine, 1, 128, 8, "warmup")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_after_warmup = torch.cuda.max_memory_allocated() / 2**30
    else:
        peak_after_warmup = None

    results = []
    for batch, seq in combos:
        r = run_one(engine, batch, seq, args.max_tokens, f"b{batch}s{seq}")
        r["tag"] = f"tp{args.tp}_pp{args.pp}_b{batch}_s{seq}"
        results.append(r)
        print(f"  B={batch:<2} S={seq:<5} TTFT={r['ttft_median_s'] * 1e3:8.2f} ms "
              f"TPOT={r['tpot_median_s'] * 1e3:7.3f} ms "
              f"总={r['total_s']:7.3f} s 输出={sum(r['output_tokens'])} "
              f"吞吐={r['throughput_tok_s']:7.1f} tok/s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    payload = {
        "model": args.model, "tp": args.tp, "pp": args.pp,
        "world_size": args.tp * args.pp,
        "max_tokens": args.max_tokens, "gpu_util": args.gpu_util,
        "enforce_eager": True, "prefix_caching": False,
        "engine_init_s": init_s,
        "cuda_allocated_GiB": allocated,
        "cuda_peak_after_warmup_GiB": peak_after_warmup,
        "cuda_peak_GiB": peak,
        "results": results,
    }
    if args.log:
        try:
            payload["vllm_log_memory"] = parse_worker_memory(
                open(args.log, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    try:
        engine.engine_core.shutdown()
    except Exception:
        pass
    del engine, llm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  释放后 allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
              f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    print(f"[sweep tp={args.tp} pp={args.pp}] init={init_s:.1f}s "
          f"peak={peak and round(peak, 2)} GiB")
    return payload


def task_ep(args):
    """专家并行：EP=1 基线 vs TP=k + enable_expert_parallel 的 EP=k。

    vLLM 里专家并行的 rank 集合就是 TP 组，所以 EP=k 用 tensor_parallel_size=k 加
    enable_expert_parallel 打开。输出用贪心解码逐 token 对拍，性能记 TTFT/TPOT。
    """
    import torch
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_experts = (getattr(cfg, "num_experts", None)
                 or getattr(cfg, "num_local_experts", None)
                 or getattr(getattr(cfg, "text_config", None), "num_experts", None))
    n_layers = getattr(cfg, "num_hidden_layers", None)
    top_k = getattr(cfg, "num_experts_per_tok", None)

    rows = []
    baseline = None
    for tag, tp, ep_on in [("ep1", 1, False)] + [(f"ep{ep}", ep, True) for ep in args.ep_list]:
        if n_experts is not None and n_experts % tp:
            rows.append({"tag": tag, "tp": tp, "expert_parallel": ep_on,
                         "supported": False,
                         "reason": f"num_experts={n_experts} 不能被 EP={tp} 整除"})
            print(f"  {tag}: 不可行（{n_experts} 不能被 {tp} 整除）", flush=True)
            continue
        t_init = time.perf_counter()
        llm = make_engine(args.model, tp, 1, args.gpu_util,
                          args.max_tokens + args.seq + 8, enable_ep=ep_on)
        engine = llm.llm_engine
        init_s = time.perf_counter() - t_init
        # 预热必须覆盖测量时用的那个形状：MoE 的 Triton kernel 是按形状 JIT 的，
        # 预热形状不对会让首次测量撞上编译，把 TPOT 抬高两个数量级（实测踩过一次）。
        run_one(engine, args.batch, args.seq, args.max_tokens, f"warm-{tag}")
        r = run_one(engine, args.batch, args.seq, args.max_tokens, tag)
        row = {
            "tag": tag, "tp": tp, "expert_parallel": ep_on, "supported": True,
            "experts_total": n_experts, "layers": n_layers, "top_k": top_k,
            "experts_per_rank": (n_experts // tp) if n_experts else None,
            "engine_init_s": init_s,
            "ttft_median_s": r["ttft_median_s"], "tpot_median_s": r["tpot_median_s"],
            "total_s": r["total_s"], "throughput_tok_s": r["throughput_tok_s"],
            "all_completed": r["all_completed"],
            "output_token_ids": r["output_token_ids"],
        }
        if baseline is None:
            baseline = r["output_token_ids"]
            row["matches_baseline"] = True
        else:
            same = all(row["output_token_ids"].get(k) == v for k, v in baseline.items())
            row["matches_baseline"] = same
            diffs = sum(1 for k, v in baseline.items()
                        if row["output_token_ids"].get(k) != v)
            row["requests_differing"] = diffs
        rows.append(row)
        print(f"  {tag}: TP={tp} EP={'on' if ep_on else 'off'} "
              f"每 rank 专家={row['experts_per_rank']} init={init_s:.1f}s "
              f"TTFT={r['ttft_median_s'] * 1e3:.2f} ms TPOT={r['tpot_median_s'] * 1e3:.3f} ms "
              f"吞吐={r['throughput_tok_s']:.1f} tok/s 与基线一致={row['matches_baseline']}",
              flush=True)
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass
        del engine, llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {"model": args.model, "batch": args.batch, "seq": args.seq,
               "max_tokens": args.max_tokens, "rows": rows}
    with open(os.path.join(args.out, "ep_result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sweep", "ep"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--ep-list", default="2,4")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    a.ep_list = [int(x) for x in a.ep_list.split(",") if x]
    if a.task == "sweep":
        task_sweep(a)
    else:
        task_ep(a)


if __name__ == "__main__":
    sys.exit(main())
