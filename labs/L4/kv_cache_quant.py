#!/usr/bin/env python3
"""L4.3 修订（任务 D）—— 真实分页接口上的 FP8 KV cache。

同一个引擎、同一批 prompt，只改 kv_cache_dtype（auto=bf16 / fp8），比较：
  [D1] KV cache 容量：同样显存预算下能放多少 token
  [D2] 数值：贪心生成的 token 序列与 top-1 logprob 差
  [D3] 长上下文：S=2048/8192 的 prefill 时间与解码吞吐
两种配置必须分进程执行（vLLM 在进程内不归还显存）。

用法：
    python labs/L4/kv_cache_quant.py --kv auto --outdir out/4.3/run
    python labs/L4/kv_cache_quant.py --kv fp8  --outdir out/4.3/run
"""

import argparse
import glob
import json
import os
import sys
import time

MODEL = "Qwen/Qwen3-1.7B"
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def snap(repo):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="auto", help="auto/fp8/turboquant_4bit_nc/int4_per_token_head/…")
    ap.add_argument("--outdir", default=os.path.expanduser("~/l43_kv"))
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--s-values", default="2048,8192,32768")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 78 + f"\n[D] FP8 KV cache：kv_cache_dtype={args.kv}\n" + "=" * 78,
          flush=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm import LLM, SamplingParams
    rep = {"kv_cache_dtype": args.kv, "batch": args.batch, "model": MODEL,
           "revision": os.path.basename(snap(MODEL))}
    t0 = time.perf_counter()
    llm = LLM(model=snap(MODEL), dtype="bfloat16", kv_cache_dtype=args.kv,
              gpu_memory_utilization=args.util,
              max_model_len=args.max_model_len, enforce_eager=True,
              disable_log_stats=True, enable_prefix_caching=False)
    rep["init_s"] = time.perf_counter() - t0
    cfg = llm.llm_engine.vllm_config.cache_config
    rep["kv_cache_memory_bytes"] = getattr(cfg, "kv_cache_memory", None)
    # 块池字节账：用引擎自报的显存预算与块数算每 token 字节，不依赖内部结构
    nblocks = getattr(cfg, "num_gpu_blocks", None)
    bsize = getattr(cfg, "block_size", None)
    if rep["kv_cache_memory_bytes"] and nblocks and bsize:
        rep["num_gpu_blocks"] = int(nblocks)
        rep["block_size"] = int(bsize)
        rep["kv_bytes_per_token_engine"] = (rep["kv_cache_memory_bytes"]
                                            / (int(nblocks) * int(bsize)))
        mc = llm.llm_engine.model_config.hf_config
        tc = getattr(mc, "text_config", mc)
        rep["kv_bytes_per_token_analytic"] = (2 * tc.num_hidden_layers
                                              * tc.num_key_value_heads
                                              * getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
                                              * (1 if args.kv != "auto" else 2))
        print(f"  块池账：{rep['kv_cache_memory_bytes']/2**30:.2f} GiB / "
              f"({nblocks} 块 × {bsize} token) = "
              f"{rep['kv_bytes_per_token_engine']:.1f} B/token（引擎）  vs  "
              f"{rep['kv_bytes_per_token_analytic']:.1f} B/token（解析：2×层×KV头×head_dim×字节）")
    # 从 KV cache 的 shape 直接算容量
    try:
        m = llm.llm_engine.model_executor.driver_worker.model_runner.model
        kv = getattr(m, "kv_cache", None)
        if kv is None:
            core = llm.llm_engine.engine_core.engine_core
            kv = core.scheduler.kv_cache_manager.block_pool.blocks
            blk = kv[0]
            rep["block_shape"] = list(blk.shape)
            rep["block_bytes"] = int(blk.numel() * blk.element_size())
            rep["num_blocks"] = len(kv)
            rep["kv_bytes_total"] = rep["block_bytes"] * rep["num_blocks"]
    except Exception as e:
        print(f"  （读取 KV 块失败：{type(e).__name__}: {str(e)[:100]}）")
    print(f"  引擎就绪 {rep['init_s']:.1f} s；KV 块 {rep.get('block_shape')} "
          f"× {rep.get('num_blocks')} = {rep.get('kv_bytes_total', 0)/2**30:.2f} GiB")

    tok = __import__("transformers").AutoTokenizer.from_pretrained(snap(MODEL))
    text = ("The history of computing spans many decades. " * 4000)

    out = {}
    s_values = [int(x) for x in args.s_values.split(",") if x]
    for S in s_values:
        ids = tok(text, return_tensors="pt").input_ids[0][:S].tolist()
        prompts = [{"prompt_token_ids": ids}] * args.batch
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
        t0 = time.perf_counter()
        r = llm.generate(prompts, sp)
        ttft = time.perf_counter() - t0
        lp = r[0].outputs[0].logprobs[0]
        top = max(lp.items(), key=lambda kv: kv[1].logprob)
        sp2 = SamplingParams(max_tokens=32, temperature=0.0)
        t0 = time.perf_counter()
        r2 = llm.generate(prompts, sp2)
        dt = time.perf_counter() - t0
        toks = r2[0].outputs[0].token_ids
        out[f"S{S}_B{args.batch}"] = {"tokens": S, "batch": args.batch, "prefill_s": ttft,
                        "first_top1": top[0], "first_logprob": top[1].logprob,
                        "decode32_s": dt,
                        "decode_tok_per_s": len(toks) / max(dt, 1e-9),
                        "token_ids": toks}
        print(f"  S={S} B={args.batch}: prefill {ttft:.3f} s，top-1 {top[0]}（logprob "
              f"{top[1].logprob:.4f}），32 token 解码 {dt:.3f} s "
              f"({args.batch*len(toks)/dt:.1f} tok/s 总 / {len(toks)/dt:.1f} 每请求)")
    rep["runs"] = out
    rep["gpu_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    path = os.path.join(args.outdir, "kv_cache_quant.json")
    if os.path.exists(path):
        data = json.load(open(path))
    else:
        data = {"configs": {}}
    data["configs"][args.kv] = rep
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
