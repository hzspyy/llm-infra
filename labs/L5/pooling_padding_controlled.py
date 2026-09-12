#!/usr/bin/env python3
"""5.12 padding 正确对照：相同 token 数、同内容、可控 batch 的对照实验。

控制变量：
- 固定总 token 数
- 固定序列内容（重复使用相同文本）
- 对照 packed（打包寻址）vs padded（矩形 padding）

测量：
- 计算的 token 数
- 实际执行时间
- pooling 结果的数值等价性
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def create_test_sequences(tokenizer, target_tokens: int, num_seqs: int):
    """创建固定总 token 数的测试序列"""
    # 使用简单重复文本确保可复现
    base_text = "The prefix cache mechanism enables multiple requests to reuse identical prompt prefixes. "

    # 计算每个序列的目标长度（变长分布）
    # 使用简单的线性分布：最短 = avg * 0.5, 最长 = avg * 1.5
    avg_len = target_tokens // num_seqs
    min_len = max(10, avg_len // 2)
    max_len = avg_len * 3 // 2

    lengths = []
    remaining = target_tokens
    for i in range(num_seqs - 1):
        # 线性插值
        frac = i / (num_seqs - 1)
        length = int(min_len + (max_len - min_len) * frac)
        lengths.append(length)
        remaining -= length
    lengths.append(max(min_len, remaining))

    # 生成文本
    sequences = []
    for length in lengths:
        # 重复 base_text 直到达到目标长度
        repeated = (base_text * ((length * 6) // len(base_text) + 1))
        encoded = tokenizer.encode(repeated, add_special_tokens=True)[:length]
        text = tokenizer.decode(encoded, skip_special_tokens=False)
        sequences.append(text)

    return sequences, lengths


def benchmark_packed(model, tokenizer, sequences, device="cuda"):
    """无 padding：每个序列独立处理，模拟打包寻址效果"""
    # Tokenize 每个序列，保持原始长度
    all_results = []
    total_tokens = 0

    for seq in sequences:
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
        total_tokens += inputs["input_ids"].shape[1]
        all_results.append(inputs)

    # 预热：处理所有序列
    with torch.no_grad():
        for _ in range(3):
            for inputs in all_results:
                _ = model(**inputs)
    torch.cuda.synchronize()

    # 计时：批量处理所有序列（模拟 packed 的连续处理）
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        pooled = []
        with torch.no_grad():
            for inputs in all_results:
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                # Mean pooling
                pooled.append(hidden.mean(dim=1).squeeze(0))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    pooled_tensor = torch.stack(pooled)

    return {
        "method": "packed",
        "total_tokens": total_tokens,
        "num_sequences": len(sequences),
        "times_ms": times,
        "median_ms": sorted(times)[len(times)//2],
        "pooled_embeddings": pooled_tensor
    }


def benchmark_padded(model, tokenizer, sequences, device="cuda"):
    """Padding 到矩形：最长序列对齐"""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 计算 padding 的 token 数
    total_tokens = attention_mask.sum().item()
    padded_tokens = input_ids.shape[0] * input_ids.shape[1]

    # 预热
    with torch.no_grad():
        for _ in range(3):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    # 计时
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Pooling：对每个序列使用 mask 求均值
    pooled = []
    for i in range(input_ids.shape[0]):
        mask = attention_mask[i].unsqueeze(-1)  # [seq_len, 1]
        seq_hidden = hidden[i] * mask  # [seq_len, hidden]
        seq_sum = seq_hidden.sum(dim=0)
        seq_len = mask.sum()
        pooled.append(seq_sum / seq_len)

    pooled_tensor = torch.stack(pooled)

    return {
        "method": "padded",
        "total_tokens": int(total_tokens),
        "padded_tokens": padded_tokens,
        "waste_ratio": padded_tokens / total_tokens,
        "num_sequences": len(sequences),
        "times_ms": times,
        "median_ms": sorted(times)[len(times)//2],
        "pooled_embeddings": pooled_tensor
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--total-tokens", type=int, default=2048)
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Creating {args.num_sequences} sequences, target total {args.total_tokens} tokens...")
    sequences, target_lengths = create_test_sequences(
        tokenizer, args.total_tokens, args.num_sequences
    )

    # 保存序列
    with (args.out / "sequences.json").open("w") as f:
        json.dump({
            "sequences": sequences,
            "target_lengths": target_lengths
        }, f, indent=2, ensure_ascii=False)

    print("\n[Packed] Benchmarking...")
    packed_result = benchmark_packed(model, tokenizer, sequences)

    print(f"[Padded] Benchmarking...")
    padded_result = benchmark_padded(model, tokenizer, sequences)

    # 检查数值等价性
    packed_emb = packed_result.pop("pooled_embeddings")
    padded_emb = padded_result.pop("pooled_embeddings")

    max_diff = (packed_emb - padded_emb).abs().max().item()
    mean_diff = (packed_emb - padded_emb).abs().mean().item()

    result = {
        "model": args.model,
        "total_tokens_target": args.total_tokens,
        "num_sequences": args.num_sequences,
        "packed": packed_result,
        "padded": padded_result,
        "embedding_diff": {
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "numerically_equal_1e-5": max_diff < 1e-5
        },
        "speedup": padded_result["median_ms"] / packed_result["median_ms"]
    }

    out_file = args.out / "padding_controlled.json"
    with out_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_file}")
    print(f"\nPacked:  {packed_result['total_tokens']} tokens, {packed_result['median_ms']:.3f} ms")
    print(f"Padded:  {padded_result['total_tokens']} tokens ({padded_result['padded_tokens']} with padding)")
    print(f"Waste:   {padded_result['waste_ratio']:.3f}x")
    print(f"Speedup: {result['speedup']:.3f}x (padded / packed)")
    print(f"Embedding diff: max {max_diff:.2e}, mean {mean_diff:.2e}")


if __name__ == "__main__":
    main()
