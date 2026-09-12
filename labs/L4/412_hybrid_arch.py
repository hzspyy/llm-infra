#!/usr/bin/env python3
"""
4.12 混合架构：Attention 与线性递归的融合
对比 RecurrentGemma（temporal attention + recurrent block）与纯 Transformer
的计算复杂度、内存占用和序列长度扩展性。

环境：crater，需要 RecurrentGemma-2B 权重和 vLLM 服务
"""
import torch
import torch.nn.functional as F
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# ============================================================
# 1. 理论复杂度对比
# ============================================================

def theoretical_complexity(seq_len, d_model, num_layers,
                          num_attn_layers, num_recurrent_layers):
    """
    计算混合架构 vs 纯 Transformer 的理论复杂度

    Args:
        seq_len: 序列长度
        d_model: 隐藏维度
        num_layers: 总层数
        num_attn_layers: attention 层数（temporal blocks）
        num_recurrent_layers: 递归层数（recurrent blocks）

    Returns:
        dict: 包含 FLOPs、内存占用等指标
    """
    # Attention 复杂度：O(L^2 * d)
    attn_flops_per_layer = 4 * seq_len * seq_len * d_model  # QK^T + softmax + V
    attn_memory_per_layer = seq_len * seq_len + 2 * seq_len * d_model  # attn_weights + KV

    # 递归复杂度：O(L * d^2) 或 O(L * d)，取决于实现
    # RecurrentGemma 使用 linear recurrence，复杂度为 O(L * d * state_size)
    # 假设 state_size ≈ d
    recurrent_flops_per_layer = 2 * seq_len * d_model * d_model  # 状态更新
    recurrent_memory_per_layer = 2 * d_model  # 只需保存当前状态

    # 混合架构
    hybrid_flops = (num_attn_layers * attn_flops_per_layer +
                   num_recurrent_layers * recurrent_flops_per_layer)
    hybrid_memory = (num_attn_layers * attn_memory_per_layer +
                    num_recurrent_layers * recurrent_memory_per_layer)

    # 纯 Transformer
    transformer_flops = num_layers * attn_flops_per_layer
    transformer_memory = num_layers * attn_memory_per_layer

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "hybrid": {
            "flops": hybrid_flops,
            "memory": hybrid_memory,
            "attn_layers": num_attn_layers,
            "recurrent_layers": num_recurrent_layers
        },
        "transformer": {
            "flops": transformer_flops,
            "memory": transformer_memory,
            "attn_layers": num_layers
        },
        "speedup": {
            "flops_ratio": transformer_flops / hybrid_flops,
            "memory_ratio": transformer_memory / hybrid_memory
        }
    }

# ============================================================
# 2. 从 RecurrentGemma config 读取实际架构
# ============================================================

def analyze_recurrentgemma_architecture(model_path):
    """
    分析 RecurrentGemma 的实际架构配置
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    arch_info = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, 'num_key_value_heads', config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "lru_width": getattr(config, 'lru_width', None),
        "attention_window_size": getattr(config, 'attention_window_size', None),
    }

    # RecurrentGemma 使用 block_types 定义层类型模式
    if hasattr(config, 'block_types'):
        block_types = config.block_types
        arch_info['block_pattern'] = block_types

        # 计算完整 26 层中各类型的数量
        num_layers = config.num_hidden_layers
        pattern_len = len(block_types)
        full_cycles = num_layers // pattern_len
        remainder = num_layers % pattern_len

        # 统计 attention 和 recurrent 层数
        num_attn_in_pattern = block_types.count('attention')
        num_recurrent_in_pattern = block_types.count('recurrent')

        num_attn_layers = full_cycles * num_attn_in_pattern
        num_recurrent_layers = full_cycles * num_recurrent_in_pattern

        # 处理余数部分
        for i in range(remainder):
            if block_types[i] == 'attention':
                num_attn_layers += 1
            elif block_types[i] == 'recurrent':
                num_recurrent_layers += 1
    else:
        # 回退：假设全是 attention
        num_attn_layers = config.num_hidden_layers
        num_recurrent_layers = 0

    arch_info['num_attention_layers'] = num_attn_layers
    arch_info['num_recurrent_layers'] = num_recurrent_layers

    # 展开完整的层序列（用于验证）
    if hasattr(config, 'block_types'):
        expanded_blocks = []
        for i in range(num_layers):
            expanded_blocks.append(block_types[i % pattern_len])
        arch_info['layer_sequence'] = expanded_blocks

    return arch_info

# ============================================================
# 3. 序列长度扫描：理论对比
# ============================================================

def sequence_length_scan(d_model=2560, num_layers=26,
                        num_attn_layers=9, num_recurrent_layers=17):
    """
    扫描不同序列长度下的复杂度差异
    """
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    for seq_len in seq_lengths:
        comp = theoretical_complexity(
            seq_len, d_model, num_layers,
            num_attn_layers, num_recurrent_layers
        )
        results.append(comp)

    return results

# ============================================================
# 4. KV cache 增长对比
# ============================================================

def kv_cache_growth_comparison(max_seq_len=4096, d_model=2560,
                              num_attn_layers=9, num_kv_heads=8):
    """
    对比混合架构 vs 纯 Transformer 的 KV cache 增长

    混合架构只有 attention 层需要 KV cache，递归层只需保存固定大小的状态
    """
    seq_lengths = list(range(1, max_seq_len + 1, 128))

    results = []
    for seq_len in seq_lengths:
        # 纯 Transformer: 所有层都需要 KV cache
        transformer_kv_bytes = (
            26 * 2 * seq_len * num_kv_heads * (d_model // 32) * 2  # bf16
        )

        # 混合架构: 只有 attention 层需要 KV cache
        hybrid_kv_bytes = (
            num_attn_layers * 2 * seq_len * num_kv_heads * (d_model // 32) * 2
        )

        # 递归层的状态内存（固定大小，不随序列长度增长）
        recurrent_state_bytes = 17 * d_model * 2  # 每层一个状态向量

        results.append({
            "seq_len": seq_len,
            "transformer_kv_mb": transformer_kv_bytes / (1024 ** 2),
            "hybrid_kv_mb": hybrid_kv_bytes / (1024 ** 2),
            "recurrent_state_mb": recurrent_state_bytes / (1024 ** 2),
            "hybrid_total_mb": (hybrid_kv_bytes + recurrent_state_bytes) / (1024 ** 2),
            "memory_saving_ratio": transformer_kv_bytes / (hybrid_kv_bytes + recurrent_state_bytes)
        })

    return results

# ============================================================
# 5. 主函数
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 80)

    # RecurrentGemma 模型路径
    model_path = "/scratch/learn/models/hf/models--google--recurrentgemma-2b/snapshots/3620f4ca9c5d16ee56c00180474a3201ec7f734a"

    # 1. 分析架构
    print("\n1. RecurrentGemma 架构分析")
    print("-" * 80)
    arch_info = analyze_recurrentgemma_architecture(model_path)
    print(json.dumps(arch_info, indent=2))

    # 2. 序列长度扫描
    print("\n2. 序列长度扫描：FLOPs 和内存对比")
    print("-" * 80)
    scan_results = sequence_length_scan(
        d_model=arch_info['hidden_size'],
        num_layers=arch_info['num_hidden_layers'],
        num_attn_layers=arch_info['num_attention_layers'],
        num_recurrent_layers=arch_info['num_recurrent_layers']
    )

    print(f"{'Seq Len':<10} {'Hybrid FLOPs':<15} {'Trans FLOPs':<15} {'Speedup':<10} {'Mem Ratio':<10}")
    for result in scan_results:
        print(f"{result['seq_len']:<10} "
              f"{result['hybrid']['flops']/1e9:<15.2f} "
              f"{result['transformer']['flops']/1e9:<15.2f} "
              f"{result['speedup']['flops_ratio']:<10.2f}x "
              f"{result['speedup']['memory_ratio']:<10.2f}x")

    # 3. KV cache 增长对比
    print("\n3. KV Cache 增长对比")
    print("-" * 80)
    kv_results = kv_cache_growth_comparison(
        max_seq_len=4096,
        d_model=arch_info['hidden_size'],
        num_attn_layers=arch_info['num_attention_layers']
    )

    print(f"{'Seq Len':<10} {'Trans KV (MB)':<15} {'Hybrid KV (MB)':<15} {'Hybrid Total (MB)':<18} {'Saving':<10}")
    for i in range(0, len(kv_results), 8):  # 每隔 8 个打印一次
        r = kv_results[i]
        print(f"{r['seq_len']:<10} "
              f"{r['transformer_kv_mb']:<15.2f} "
              f"{r['hybrid_kv_mb']:<15.2f} "
              f"{r['hybrid_total_mb']:<18.2f} "
              f"{r['memory_saving_ratio']:<10.2f}x")

    # 4. 保存完整结果
    output_dir = Path("/scratch/learn/results/hybrid_arch")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "architecture": arch_info,
        "sequence_scan": scan_results,
        "kv_cache_growth": kv_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_file = output_dir / "412_hybrid_complexity.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
