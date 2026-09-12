#!/usr/bin/env python3
"""5.12 固定 DynamicCache 的 decode 计时：修正旧实验的缓存增长问题。

旧实验复用了 DynamicCache，预热和重复期间上下文不断增长。
本脚本在每次计时前重置缓存，确保上下文长度固定。

对照 encoder (BERT) 与 decoder (Qwen) 的固定上下文计时。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, DynamicCache


def benchmark_encoder(model, tokenizer, seq_len: int, batch: int, repeats: int):
    """Encoder 无 KV cache，每次独立前向"""
    text = "The prefix cache mechanism. " * (seq_len // 5)
    inputs = tokenizer([text] * batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=seq_len).to("cuda")

    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(**inputs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return {
        "times_ms": times,
        "median_ms": sorted(times)[len(times)//2],
        "actual_tokens": inputs["input_ids"].shape[1]
    }


def benchmark_decoder_fixed_cache(model, tokenizer, context_len: int, batch: int, repeats: int):
    """Decoder 固定 cache 长度的 decode 步计时"""
    # 准备固定长度的上下文
    text = "The prefix cache mechanism. " * (context_len // 5)
    context_inputs = tokenizer([text] * batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=context_len).to("cuda")

    times = []

    for r in range(repeats + 3):  # 3 次预热
        # 每次重新创建 cache
        past_key_values = DynamicCache()

        # Prefill：生成固定长度的 cache
        with torch.no_grad():
            outputs = model(**context_inputs, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values

        # 验证 cache 长度
        actual_cache_len = past_key_values.get_seq_length()

        # Decode：单个 token 前向
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if r >= 3:  # 跳过预热
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            end.record()

            torch.cuda.synchronize()
            times.append({
                "time_ms": start.elapsed_time(end),
                "cache_len": actual_cache_len,
                "batch": batch
            })

    return {
        "times": times,
        "median_ms": sorted([t["time_ms"] for t in times])[len(times)//2],
        "context_len": context_len,
        "actual_cache_len": times[0]["cache_len"] if times else 0
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--decoder", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=256, help="Encoder sequence length")
    parser.add_argument("--context-len", type=int, default=256, help="Decoder context length")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Encoder benchmark
    print(f"[Encoder] Loading {args.encoder}...")
    encoder = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16).to("cuda")
    encoder_tok = AutoTokenizer.from_pretrained(args.encoder)

    print(f"[Encoder] Benchmarking seq_len={args.seq_len}, batch={args.batch}...")
    encoder_result = benchmark_encoder(encoder, encoder_tok, args.seq_len, args.batch, args.repeats)

    del encoder, encoder_tok
    torch.cuda.empty_cache()

    # Decoder benchmark
    print(f"[Decoder] Loading {args.decoder}...")
    decoder = AutoModelForCausalLM.from_pretrained(
        args.decoder, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    decoder_tok = AutoTokenizer.from_pretrained(args.decoder)

    print(f"[Decoder] Benchmarking context_len={args.context_len}, batch={args.batch}...")
    decoder_result = benchmark_decoder_fixed_cache(
        decoder, decoder_tok, args.context_len, args.batch, args.repeats
    )

    # 保存结果
    result = {
        "encoder_model": args.encoder,
        "decoder_model": args.decoder,
        "batch": args.batch,
        "encoder": encoder_result,
        "decoder": decoder_result
    }

    out_file = args.out / "fixed_cache_timing.json"
    with out_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"\nEncoder median: {encoder_result['median_ms']:.3f} ms")
    print(f"Decoder median: {decoder_result['median_ms']:.3f} ms")
    print(f"Decoder cache length: {decoder_result['actual_cache_len']}")


if __name__ == "__main__":
    main()
