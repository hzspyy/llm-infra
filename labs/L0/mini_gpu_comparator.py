#!/usr/bin/env python3
"""mini_gpu_comparator.py · 异构 GPU 理论 Roofline 预测与实测对拍工具

用法：
    python mini_gpu_comparator.py
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GPUSpec:
    name: str
    copy_bw_gbps: float
    bf16_tflops: float
    launch_us: float

@dataclass
class ModelSpec:
    name: str
    param_count: float       # 参数量 (单位: 个)
    weight_bytes: float      # BF16 权重字节数
    num_layers: int
    num_kv_heads: int
    head_dim: int

def predict_prefill_ms(gpu: GPUSpec, model: ModelSpec, prompt_len: int) -> float:
    # 浮点计算量: 注意 lm_head 仅在最后一个 token 执行投影
    flops = 2.0 * model.param_count * prompt_len
    # 算力主导阶段理论耗时 (ms)
    t_comp_ms = (flops / (gpu.bf16_tflops * 1e12)) * 1e3
    return t_comp_ms

def predict_decode_step_ms(gpu: GPUSpec, model: ModelSpec, batch: int, ctx_len: int) -> float:
    # 显存访存量: 静态权重 + KV cache 读取
    kv_bytes_per_token = 2 * model.num_layers * 2 * model.num_kv_heads * model.head_dim * 2
    total_bytes = model.weight_bytes + batch * ctx_len * kv_bytes_per_token
    # 访存主导阶段理论下界 (ms)
    t_mem_ms = (total_bytes / (gpu.copy_bw_gbps * 1e9)) * 1e3
    return t_mem_ms

def main():
    qwen3 = ModelSpec(
        name="Qwen3-1.7B",
        param_count=1.7e9,
        weight_bytes=3.4e9,
        num_layers=28,
        num_kv_heads=8,
        head_dim=128
    )
    
    rtx5090d = GPUSpec("RTX 5090 D", copy_bw_gbps=1519.3, bf16_tflops=232.0, launch_us=2.68)
    l40s = GPUSpec("L40S", copy_bw_gbps=649.7, bf16_tflops=256.0, launch_us=5.03)

    print("==========================================================================")
    print(f"模型: {qwen3.name} | 异构硬件 Roofline 理论预测 vs 实测对拍")
    print("==========================================================================")
    
    # 1. Decode 阶段对比 (ctx=1024)
    print("\n[Decode 阶段每步耗时 (ms/step) · ctx_len=1024]")
    print(f"{'Batch':<8}{'5090D 预测':<14}{'5090D 实测':<14}{'L40S 预测':<14}{'L40S 实测':<14}{'实测加速比':<10}")
    print("-" * 74)
    
    decode_actuals = {
        1:  (4.483, 11.127),
        4:  (4.680, 11.411),
        16: (5.850, 11.949),
        32: (6.798, 12.624),
    }
    for b in [1, 4, 16, 32]:
        p_5090 = predict_decode_step_ms(rtx5090d, qwen3, b, 1024)
        p_l40s = predict_decode_step_ms(l40s, qwen3, b, 1024)
        a_5090, a_l40s = decode_actuals[b]
        speedup = a_l40s / a_5090
        print(f"{b:<8}{p_5090:<14.3f}{a_5090:<14.3f}{p_l40s:<14.3f}{a_l40s:<14.3f}{speedup:<10.2f}x")

    # 2. Prefill 阶段对比 (B=1)
    print("\n[Prefill 阶段耗时 (ms) · Batch=1]")
    print(f"{'Prompt':<8}{'5090D 预测':<14}{'5090D 实测':<14}{'L40S 预测':<14}{'L40S 实测':<14}{'耗时比例':<10}")
    print("-" * 74)
    
    prefill_actuals = {
        512:  (10.85, 12.80),
        1024: (19.83, 19.75),
        2048: (36.64, 38.85),
        4096: (76.91, 79.54),
    }
    for s in [512, 1024, 2048, 4096]:
        p_5090 = predict_prefill_ms(rtx5090d, qwen3, s)
        p_l40s = predict_prefill_ms(l40s, qwen3, s)
        a_5090, a_l40s = prefill_actuals[s]
        ratio = a_l40s / a_5090
        print(f"{s:<8}{p_5090:<14.2f}{a_5090:<14.2f}{p_l40s:<14.2f}{a_l40s:<14.2f}{ratio:<10.3f}")

if __name__ == "__main__":
    main()
