#!/usr/bin/env python3
"""5.12 三点 roofline：同一台机器上，LLM prefill / LLM decode / encoder-only 编码。

三个点量的都是**同一个东西**：一次 transformer body 前向。
不同的是形状——
  decode    每序列 1 个 query token，要读完整段 KV
  prefill   每序列 L 个 query token，因果 mask
  encode    每序列 L 个 query token，双向 mask，之后 mean-pool

FLOPs 与字节量按公式算，耗时用 CUDA event 实测，两者相除得到实际达到的吞吐。
公式（单位：FLOP 与 byte，d = d_model，N = body 参数量，n_l = 层数，B = 序列数）：

  权重项      2 · N · tokens
  attention   causal:     2 · d · n_l · L² · B        （平均只有一半的 L² 被算）
              双向:       4 · d · n_l · L² · B
  decode attn 4 · d · n_l · C · B                     （1 个 query 对 C 个 key）
  权重字节    N · dtype_bytes
  KV 读取     2 · n_l · n_kv_head · d_head · dtype_bytes · C · B

算术强度 = FLOPs / 字节。本章用它判断这个形状落在 roofline 的哪一侧。

只测 body，不含 lm_head / logits：embedding 模型根本没有 lm_head，
把它算进 prefill/decode 会让三个点不可比。lm_head 的量级在正文里单独给。

    python pooling_roofline.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}


def body_stats(model) -> dict:
    """数 body 的参数量，以及影响 KV 字节的几个维度。"""
    n_body = 0
    seen = set()
    for name, p in model.named_parameters():
        # 词嵌入和 lm_head 不计入 body（embedding 是查表，lm_head 只在最后用一次）
        if "embed_tokens" in name or "word_embeddings" in name or name.startswith("lm_head"):
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        n_body += p.numel()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n_kv_head = getattr(cfg, "num_key_value_heads", n_head)
    d_model = cfg.hidden_size
    d_head = getattr(cfg, "head_dim", None) or d_model // n_head
    return {"n_body": n_body, "n_layer": n_layer, "n_head": n_head,
            "n_kv_head": n_kv_head, "d_model": d_model, "d_head": d_head}


def flops_bytes(stats, mode, B, L, C=None, bidirectional=False, dtype_bytes=2):
    """返回 (flops, bytes, 分项)。"""
    d, n_l = stats["d_model"], stats["n_layer"]
    tokens = B * (1 if mode == "decode" else L)
    weights_flops = 2 * stats["n_body"] * tokens
    if mode == "decode":
        attn_flops = 4 * d * n_l * C * B
    elif bidirectional:
        attn_flops = 4 * d * n_l * L * L * B
    else:
        attn_flops = 2 * d * n_l * L * L * B
    flops = weights_flops + attn_flops

    weight_bytes = stats["n_body"] * dtype_bytes
    kv_per_token = 2 * n_l * stats["n_kv_head"] * stats["d_head"] * dtype_bytes
    kv_bytes = kv_per_token * (C if mode == "decode" else L) * B
    total_bytes = weight_bytes + kv_bytes
    return flops, total_bytes, {"weights_flops": weights_flops, "attn_flops": attn_flops,
                                "weight_bytes": weight_bytes, "kv_bytes": kv_bytes,
                                "kv_bytes_per_token": kv_per_token}


def timeit(fn, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


# ------------------------------------------------------------------ 三种形状
def bench_decode(model, stats, B, C, dtype_bytes):
    ids = torch.randint(0, 1000, (B, C), device="cuda")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (B, 1), device="cuda")
    pos = torch.full((B, 1), C, device="cuda", dtype=torch.long)

    def run():
        with torch.no_grad():
            model(nxt, past_key_values=past, position_ids=pos, use_cache=True)

    ms = timeit(run)
    del past, out
    torch.cuda.empty_cache()
    return ms


def bench_prefill(model, stats, B, L, dtype_bytes):
    ids = torch.randint(0, 1000, (B, L), device="cuda")

    def run():
        with torch.no_grad():
            model(ids, use_cache=False)

    return timeit(run)


def bench_encode(model, stats, B, L, dtype_bytes):
    ids = torch.randint(0, 1000, (B, L), device="cuda")
    mask = torch.ones_like(ids)

    def run():
        with torch.no_grad():
            h = model(ids, attention_mask=mask).last_hidden_state
            h.mean(dim=1)

    return timeit(run)


# ------------------------------------------------------------------ padding
def padded_vs_bucketed(model, stats, lengths, dtype_bytes, reps=5):
    """变长输入：padding 到批内最长 vs 按长度分桶后逐桶打包。"""
    max_len = max(lengths)
    vocab = 1000
    # ① 一次性 padding 到全局最长
    ids = torch.zeros((len(lengths), max_len), dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(0, vocab, (n,), device="cuda")
        mask[i, :n] = 1

    def run_padded():
        with torch.no_grad():
            h = model(ids, attention_mask=mask).last_hidden_state
            h.mean(dim=1)

    ms_padded = timeit(run_padded, warmup=2, reps=reps)
    real_tokens = int(sum(lengths))
    padded_tokens = len(lengths) * max_len

    # ② 按长度分桶：同桶内长度接近，再逐桶 padding 到桶内最长
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    buckets = [order[i:i + 8] for i in range(0, len(order), 8)]
    bucket_ms, bucket_tokens = 0.0, 0
    for b in buckets:
        bl = max(lengths[i] for i in b)
        b_ids = torch.zeros((len(b), bl), dtype=torch.long, device="cuda")
        b_mask = torch.zeros_like(b_ids)
        for j, i in enumerate(b):
            b_ids[j, :lengths[i]] = torch.randint(0, vocab, (lengths[i],), device="cuda")
            b_mask[j, :lengths[i]] = 1

        def run_bucket():
            with torch.no_grad():
                h = model(b_ids, attention_mask=b_mask).last_hidden_state
                h.mean(dim=1)

        bucket_ms += timeit(run_bucket, warmup=1, reps=reps)
        bucket_tokens += len(b) * bl
    return {"n_requests": len(lengths), "real_tokens": real_tokens, "max_len": max_len,
            "padded_tokens": padded_tokens,
            "pad_ratio": round(padded_tokens / real_tokens, 3),
            "padded_ms": round(ms_padded, 3),
            "bucketed_ms": round(bucket_ms, 3),
            "bucketed_tokens": bucket_tokens,
            "bucketed_pad_ratio": round(bucket_tokens / real_tokens, 3),
            "speedup": round(ms_padded / bucket_ms, 3) if bucket_ms else None}


# ------------------------------------------------------------------ 主流程
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--hw", default="/scratch/learn/results/hw_crater_5090.json")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cuda": torch.version.cuda, "points": [], "padding": {}, "env": {}}

    hw = {}
    if Path(args.hw).exists():
        hw = json.loads(Path(args.hw).read_text())
    bf16_peak = hw.get("gemm", {}).get("bf16", {}).get("peak_tflops")
    bw = hw.get("memory_bandwidth", {}).get("readonly_gbps")
    result["env"] = {"bf16_gemm_peak_tflops": bf16_peak, "readonly_gbps": bw,
                     "ridge_flop_per_byte": round(bf16_peak * 1e12 / (bw * 1e9), 1)
                     if bf16_peak and bw else None,
                     "hw_source": args.hw}

    from transformers import AutoModel, AutoModelForCausalLM

    # ---------------- causal LM：prefill 与 decode ----------------
    llm_repo = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_repo, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    body = llm.model
    llm_stats = body_stats(body)
    llm_bytes = DTYPE_BYTES[str(body.dtype)]
    result["env"]["llm"] = {"repo": llm_repo, "dtype": str(body.dtype), **llm_stats}

    for L in (512, 1024, 2048):
        B = max(1, 8192 // L)
        ms = bench_prefill(body, llm_stats, B, L, llm_bytes)
        f, b, parts = flops_bytes(llm_stats, "prefill", B, L, dtype_bytes=llm_bytes)
        result["points"].append(rec("prefill", llm_repo, B, L, None, ms, f, b, parts,
                                    bf16_peak, bw, llm_bytes))
    for C in (512, 1024):
        for B in (1, 8, 32, 128):
            ms = bench_decode(body, llm_stats, B, C, llm_bytes)
            f, b, parts = flops_bytes(llm_stats, "decode", B, None, C=C, dtype_bytes=llm_bytes)
            result["points"].append(rec("decode", llm_repo, B, None, C, ms, f, b, parts,
                                        bf16_peak, bw, llm_bytes))
    del llm
    torch.cuda.empty_cache()

    # ---------------- encoder-only ----------------
    enc_repo = os.environ.get("ENC_MODEL", "BAAI/bge-small-en-v1.5")
    enc = AutoModel.from_pretrained(
        enc_repo, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    enc_stats = body_stats(enc)
    enc_bytes = DTYPE_BYTES[str(enc.dtype)]
    result["env"]["encoder"] = {"repo": enc_repo, "dtype": str(enc.dtype), **enc_stats,
                                "native_dtype": "float32",
                                "note": "checkpoint 原为 fp32；为与 LLM 可比，这里统一用 bf16，"
                                        "权重字节因此减半"}

    for L in (128, 256, 512):
        B = max(1, 8192 // L)
        ms = bench_encode(enc, enc_stats, B, L, enc_bytes)
        f, b, parts = flops_bytes(enc_stats, "encode", B, L,
                                  bidirectional=True, dtype_bytes=enc_bytes)
        result["points"].append(rec("encode", enc_repo, B, L, None, ms, f, b, parts,
                                    bf16_peak, bw, enc_bytes))

    # ---------------- 变长输入 ----------------
    import random
    rng = random.Random(512)
    lengths = [min(512, max(8, int(rng.lognormvariate(4.6, 0.7)))) for _ in range(128)]
    result["padding"] = padded_vs_bucketed(enc, enc_stats, lengths, enc_bytes, args.reps)
    result["padding"]["lengths_head"] = lengths[:24]

    (out / "roofline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "points"},
                     ensure_ascii=False, indent=2))
    for r in result["points"]:
        print(f"{r['mode']:8s} B={r['B']:4d} L={str(r['L']):5s} C={str(r['C']):5s} "
              f"{r['ms']:9.3f} ms  AI={r['arithmetic_intensity']:9.2f}  "
              f"{r['achieved_tflops']:8.2f} TFLOP/s  {r['bound']}")


def rec(mode, repo, B, L, C, ms, flops, bytes_, parts, peak, bw, dtype_bytes):
    ai = flops / bytes_
    tflops = flops / (ms * 1e-3) / 1e12
    gbps = bytes_ / (ms * 1e-3) / 1e9
    ridge = (peak * 1e12 / (bw * 1e9)) if peak and bw else None
    bound = "?" if ridge is None else ("compute" if ai >= ridge else "memory")
    return {"mode": mode, "repo": repo, "B": B, "L": L, "C": C, "dtype_bytes": dtype_bytes,
            "ms": round(ms, 3), "flops": flops, "bytes": bytes_, "parts": parts,
            "arithmetic_intensity": round(ai, 3), "achieved_tflops": round(tflops, 3),
            "achieved_gbps": round(gbps, 1),
            "pct_of_peak": round(tflops / peak * 100, 2) if peak else None,
            "bound": bound}


if __name__ == "__main__":
    main()
