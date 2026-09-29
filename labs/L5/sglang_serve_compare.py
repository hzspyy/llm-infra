#!/usr/bin/env python3
"""L5.1-C / L5.2-D · SGLang 侧对照（同一模型、同一 token 输入）。

两个目的：

  * 5.1-C：同一 B×S 网格上，SGLang 的混批/chunked prefill 与 vLLM 的差异；
  * 5.2-D：RadixCache 的逐请求命中量（``meta_info['cached_tokens']``）
    与 vLLM 块哈希的命中量对照，含 page_size=1 与 16 两种粒度。

计时口径：客户端墙钟 + 引擎自报的 ``e2e_latency``；TTFT 用流式首包，
不把 e2e 差当 TTFT。跨引擎比较只比「同一输入下的整批墙钟与吞吐」。

用法：
    python sglang_serve_compare.py --mode probe --out <dir>
    python sglang_serve_compare.py --mode grid  --out <dir>
    python sglang_serve_compare.py --mode prefix --out <dir> --page-size 16
"""

import argparse
import json
import os
import random
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = os.environ.get("L5_MODEL", "Qwen/Qwen3-1.7B")


def make_engine(page_size=16, chunked_prefill_size=8192, mem_fraction=0.5):
    import sglang as sgl
    return sgl.Engine(model_path=MODEL, tp_size=1,
                      mem_fraction_static=mem_fraction,
                      page_size=page_size,
                      chunked_prefill_size=chunked_prefill_size,
                      disable_radix_cache=False, random_seed=0,
                      log_level="error")


def shutdown(engine):
    try:
        engine.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def ids_of(rng, n):
    return [rng.randint(1000, 60000) for _ in range(n)]


def sp(out_len):
    return dict(max_new_tokens=out_len, temperature=0.0, ignore_eos=True)


def batch_run(engine, prompts, out_len):
    t0 = time.perf_counter()
    outs = engine.generate(input_ids=prompts, sampling_params=sp(out_len))
    wall = (time.perf_counter() - t0) * 1000
    if isinstance(outs, dict):
        outs = [outs]
    metas = [o.get("meta_info", {}) for o in outs]
    e2e = [m.get("e2e_latency") for m in metas]
    comp = [m.get("completion_tokens") for m in metas]
    cached = [m.get("cached_tokens") for m in metas]
    ptok = [m.get("prompt_tokens") for m in metas]
    return dict(wall_ms=wall, e2e_ms=[x * 1000 for x in e2e if x],
                completion_tokens=comp, cached_tokens=cached,
                prompt_tokens=ptok, n=len(outs),
                meta_keys=sorted(metas[0]) if metas else [])


def stream_one(engine, ids, out_len):
    """流式单请求：记录首包与末包时间。"""
    t0 = time.perf_counter()
    first = None
    n = 0
    for chunk in engine.generate(input_ids=[ids], sampling_params=sp(out_len),
                                 stream=True):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        n += 1
    total = (time.perf_counter() - t0) * 1000
    return dict(ttft_ms=first, total_ms=total, chunks=n)


def do_probe(engine, out):
    rng = random.Random(0)
    ids = ids_of(rng, 128)
    r = batch_run(engine, [ids], 8)
    out.append(f"  batch meta_info 键：{r['meta_keys']}")
    out.append(f"  e2e {r['e2e_ms']} ms  completion {r['completion_tokens']} "
               f"cached {r['cached_tokens']} prompt {r['prompt_tokens']}")
    s = stream_one(engine, ids, 16)
    out.append(f"  流式：TTFT {s['ttft_ms']:.1f} ms，总 {s['total_ms']:.1f} ms，"
               f"{s['chunks']} 个 chunk")
    r2 = batch_run(engine, [ids], 8)
    out.append(f"  同一请求再来一次：cached {r2['cached_tokens']} "
               f"（radix 命中应 > 0），e2e {r2['e2e_ms']}")
    return dict(first=r, stream=s, repeat=r2)


def do_grid(engine, out, rng):
    rows = []
    for B in (1, 2, 4, 8, 16, 32, 64):
        prompts = [ids_of(rng, 2048) for _ in range(B)]
        r = batch_run(engine, prompts, 128)
        ttfts = [stream_one(engine, prompts[0], 1)["ttft_ms"]] if B == 1 else []
        out_tok = sum(x for x in r["completion_tokens"] if x)
        rows.append(dict(B=B, S=2048, wall_ms=r["wall_ms"],
                         e2e_median_ms=(statistics.median(r["e2e_ms"])
                                        if r["e2e_ms"] else None),
                         out_tokens=out_tok,
                         out_tps=out_tok / (r["wall_ms"] / 1000),
                         cached=r["cached_tokens"][:3],
                         single_ttft_ms=(ttfts[0] if ttfts else None)))
        out.append(f"  B={B:>3} 墙钟 {r['wall_ms']:>8.1f} ms  输出 {out_tok:>5} token  "
                   f"{rows[-1]['out_tps']:>8.1f} tok/s  "
                   f"e2e 中位 {rows[-1]['e2e_median_ms']}")
    return rows


def do_prefix(engine, out, rng, bs):
    base = ids_of(rng, 256)
    warm = batch_run(engine, [base + [1]], 1)
    out.append(f"  base 预热：cached {warm['cached_tokens']}，"
               f"prompt {warm['prompt_tokens']}，e2e {warm['e2e_ms']}")
    rows = []
    for shared in (0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129):
        ids = base[:shared] + ids_of(rng, 256 - shared)
        r = batch_run(engine, [ids], 1)
        rows.append(dict(shared=shared, cached=r["cached_tokens"][0],
                         prompt=r["prompt_tokens"][0],
                         predicted_blocks=shared // bs,
                         e2e_ms=r["e2e_ms"][0] if r["e2e_ms"] else None))
    out.append(f"    {'共享前缀':>8}{'cached_tokens':>14}{'page':>5}{'预测块':>7}"
               f"{'e2e ms':>9}")
    for r in rows:
        out.append(f"    {r['shared']:>8}{r['cached']:>14}{bs:>5}"
                   f"{r['predicted_blocks']:>7}"
                   f"{(r['e2e_ms'] or float('nan')):>9.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["probe", "grid", "prefix"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    out, rep = [], {}
    rng = random.Random(20260921)
    t0 = time.perf_counter()
    engine = make_engine(args.page_size, args.chunked_prefill_size,
                         args.mem_fraction)
    out.append(f"SGLang 对照 · {MODEL} · page_size={args.page_size} · "
               f"chunked_prefill_size={args.chunked_prefill_size} · "
               f"引擎启动 {time.perf_counter() - t0:.1f} s")
    try:
        if args.mode == "probe":
            rep = do_probe(engine, out)
        elif args.mode == "grid":
            rep["grid"] = do_grid(engine, out, rng)
        else:
            rep["prefix"] = do_prefix(engine, out, rng, args.page_size)
    finally:
        shutdown(engine)

    import sglang
    rep["meta"] = dict(model=MODEL, mode=args.mode, page_size=args.page_size,
                       chunked_prefill_size=args.chunked_prefill_size,
                       mem_fraction_static=args.mem_fraction,
                       sglang=sglang.__version__)
    text = "\n".join(out)
    print(text)
    tag = f"sglang_{args.mode}_p{args.page_size}"
    with open(os.path.join(args.out, tag + ".txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\n写入 {args.out}/{tag}.txt 与 {tag}.json")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
