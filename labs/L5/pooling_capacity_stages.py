#!/usr/bin/env python3
"""5.12 分阶段容量实测：encoder-only 与 decode 的峰值显存分段对照。

分阶段释放对象并记录实际显存占用，区分权重、激活与持久状态。
使用 torch.cuda.memory_allocated() 和 max_memory_allocated() 读取。

运行前确认 GPU 空闲；单进程执行；输出 JSON 到指定目录。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def measure_stages(model_name: str, mode: str, seq_len: int, batch: int):
    """
    mode: 'encoder' 或 'decode'
    返回各阶段的 allocated 和 max_allocated 字节数
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    stages = {}
    device = "cuda"

    # Stage 0: 初始状态
    stages["init"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    # Stage 1: 加载 tokenizer（CPU）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stages["tokenizer_loaded"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    # Stage 2: 加载模型到 GPU
    if mode == "encoder":
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(device)

    stages["model_loaded"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    # Stage 3: 准备输入
    text = "The prefix cache mechanism. " * (seq_len // 5)
    inputs = tokenizer([text] * batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    stages["inputs_ready"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated(),
        "actual_tokens": input_ids.shape[1],
        "batch_size": input_ids.shape[0]
    }

    # Stage 4: 前向传播
    with torch.no_grad():
        if mode == "encoder":
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # decode 模式：只做 prefill，不生成
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    torch.cuda.synchronize()
    stages["forward_done"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    # Stage 5: 释放输入张量
    del input_ids, attention_mask, inputs
    torch.cuda.empty_cache()
    stages["inputs_freed"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    # Stage 6: 释放模型
    del model
    torch.cuda.empty_cache()
    stages["model_freed"] = {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated()
    }

    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, help="Encoder model name")
    parser.add_argument("--decoder", required=True, help="Decoder model name")
    parser.add_argument("--seq-len", type=int, default=256, help="Target sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[Encoder] {args.encoder}")
    encoder_stages = measure_stages(args.encoder, "encoder", args.seq_len, args.batch)

    # 清理后再测 decoder
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    time.sleep(1)

    print(f"[Decoder] {args.decoder}")
    decoder_stages = measure_stages(args.decoder, "decode", args.seq_len, args.batch)

    result = {
        "encoder_model": args.encoder,
        "decoder_model": args.decoder,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "encoder_stages": encoder_stages,
        "decoder_stages": decoder_stages
    }

    out_file = args.out / "capacity_stages.json"
    with out_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out_file}")

    # 打印关键对比
    print("\n=== Memory Peak (MB) ===")
    enc_peak = encoder_stages["forward_done"]["peak"] / 1024**2
    dec_peak = decoder_stages["forward_done"]["peak"] / 1024**2
    print(f"Encoder forward peak: {enc_peak:.1f} MB")
    print(f"Decoder forward peak: {dec_peak:.1f} MB")


if __name__ == "__main__":
    main()
