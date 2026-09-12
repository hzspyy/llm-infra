#!/usr/bin/env python3
"""
5.13 混合架构状态管理实验

检查 RecurrentGemma-2B 的逐层状态形状、前缀复用行为和容量特征。
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def inspect_layer_states(model_path: str) -> dict[str, Any]:
    """检查模型逐层状态结构"""
    print(f"Loading config from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    result = {
        "model_path": model_path,
        "model_type": config.model_type,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "layers": []
    }

    # RecurrentGemma 特有配置
    if hasattr(config, "block_types"):
        result["block_types"] = config.block_types
        print(f"Block types: {config.block_types}")

    # 加载模型检查实际结构（不加载权重，只看架构）
    print("Loading model structure (no weights)...")
    try:
        # 先只看配置，不加载 26 GB 的权重
        for i in range(config.num_hidden_layers):
            layer_info = {"layer_idx": i}

            # RecurrentGemma 的 block_types 定义了每层类型
            if hasattr(config, "block_types") and config.block_types:
                block_type = config.block_types[i % len(config.block_types)]
                layer_info["block_type"] = block_type

                if block_type == "recurrent":
                    # RNN Griffin 层：定长 state
                    # shape: (batch, num_heads, conv_width, head_dim)
                    conv_width = getattr(config, "conv_width", 4)
                    layer_info["state_type"] = "recurrent"
                    head_dim = config.hidden_size // config.num_attention_heads
                    layer_info["state_shape_per_batch"] = [
                        config.num_attention_heads,
                        conv_width,
                        head_dim
                    ]
                    # 字节数：heads * conv_width * head_dim * dtype_bytes
                    # 假设 bfloat16
                    state_bytes = config.num_attention_heads * conv_width * head_dim * 2
                    layer_info["state_bytes_per_batch"] = state_bytes
                    layer_info["grows_with_context"] = False

                elif block_type == "attention":
                    # Local attention 层：KV cache，随 context 增长
                    layer_info["state_type"] = "kv_cache"
                    head_dim = config.hidden_size // config.num_attention_heads
                    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
                    layer_info["state_shape_per_token"] = [
                        2,  # K and V
                        num_kv_heads,
                        head_dim
                    ]
                    # 字节数随 context_len 线性增长
                    bytes_per_token = 2 * num_kv_heads * head_dim * 2
                    layer_info["state_bytes_per_token"] = bytes_per_token
                    layer_info["grows_with_context"] = True
            else:
                # 默认假设是 attention
                layer_info["state_type"] = "kv_cache"
                layer_info["grows_with_context"] = True

            result["layers"].append(layer_info)

    except Exception as e:
        print(f"Warning: Could not load model structure: {e}")
        result["error"] = str(e)

    return result


def test_prefix_reuse(base_url: str, num_requests: int = 5) -> dict[str, Any]:
    """测试前缀复用：混合架构的定长状态能否参与前缀缓存？"""
    print(f"\n=== Testing prefix reuse with {num_requests} requests ===")

    # 共享前缀
    shared_prefix = "The RecurrentGemma architecture combines recurrent layers with local attention. "

    results = []
    for i in range(num_requests):
        prompt = shared_prefix + f"Request {i}: What are the key benefits? "

        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 20,
                "temperature": 0.0
            },
            timeout=30
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            results.append({
                "request_idx": i,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "elapsed_ms": elapsed * 1000,
                "text": data["choices"][0]["text"][:50]
            })
            print(f"Request {i}: {usage.get('prompt_tokens')} tokens, {elapsed*1000:.1f} ms")
        else:
            print(f"Request {i} failed: {resp.status_code}")
            results.append({"request_idx": i, "error": resp.text})

    # 分析：后续请求是否因前缀复用而加速
    if len(results) >= 2:
        first_time = results[0]["elapsed_ms"]
        later_times = [r["elapsed_ms"] for r in results[1:] if "elapsed_ms" in r]
        if later_times:
            avg_later = sum(later_times) / len(later_times)
            speedup = first_time / avg_later if avg_later > 0 else 1.0
            print(f"\nFirst request: {first_time:.1f} ms")
            print(f"Later avg: {avg_later:.1f} ms")
            print(f"Speedup: {speedup:.2f}x")

    return {
        "shared_prefix_tokens": len(shared_prefix.split()),
        "num_requests": num_requests,
        "results": results
    }


def capacity_scan(base_url: str, context_lengths: list[int]) -> dict[str, Any]:
    """扫描不同上下文长度的容量需求"""
    print(f"\n=== Capacity scan across context lengths ===")

    results = []
    for ctx_len in context_lengths:
        # 生成指定长度的 prompt（粗略估计，实际 token 数会有偏差）
        prompt = "Token " * ctx_len

        print(f"\nTesting context length ~{ctx_len} tokens...")
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0
            },
            timeout=60
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            actual_tokens = usage.get("prompt_tokens", 0)
            results.append({
                "target_ctx_len": ctx_len,
                "actual_prompt_tokens": actual_tokens,
                "elapsed_ms": elapsed * 1000
            })
            print(f"  Actual tokens: {actual_tokens}, time: {elapsed*1000:.1f} ms")
        else:
            print(f"  Failed: {resp.status_code}")
            results.append({
                "target_ctx_len": ctx_len,
                "error": resp.text[:200]
            })

    return {"scan": results}


def main():
    model_path = "/scratch/learn/models/hf/models--google--recurrentgemma-2b/snapshots/3620f4ca9c5d16ee56c00180474a3201ec7f734a"
    base_url = "http://localhost:8000"
    output_dir = Path("/scratch/learn/work/out/hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. 检查逐层状态结构
    print("=== Step 1: Inspecting layer states ===")
    layer_states = inspect_layer_states(model_path)
    all_results["layer_states"] = layer_states

    # 统计混合架构的层类型分布
    if "layers" in layer_states:
        recurrent_count = sum(1 for l in layer_states["layers"] if l.get("state_type") == "recurrent")
        attention_count = sum(1 for l in layer_states["layers"] if l.get("state_type") == "kv_cache")
        print(f"\nLayer distribution: {recurrent_count} recurrent, {attention_count} attention")

        # 估算固定状态 vs 可变状态的字节数
        total_recurrent_bytes = sum(
            l.get("state_bytes_per_batch", 0)
            for l in layer_states["layers"]
            if l.get("state_type") == "recurrent"
        )
        total_kv_bytes_per_token = sum(
            l.get("state_bytes_per_token", 0)
            for l in layer_states["layers"]
            if l.get("state_type") == "kv_cache"
        )
        print(f"Recurrent state (fixed): {total_recurrent_bytes:,} bytes per batch")
        print(f"KV cache (grows): {total_kv_bytes_per_token:,} bytes per token")

    # 2. 前缀复用测试
    print("\n=== Step 2: Prefix reuse ===")
    try:
        prefix_result = test_prefix_reuse(base_url, num_requests=5)
        all_results["prefix_reuse"] = prefix_result
    except Exception as e:
        print(f"Prefix reuse test failed: {e}")
        all_results["prefix_reuse"] = {"error": str(e)}

    # 3. 容量扫描
    print("\n=== Step 3: Capacity scan ===")
    try:
        capacity_result = capacity_scan(base_url, context_lengths=[256, 512, 1024, 2048])
        all_results["capacity_scan"] = capacity_result
    except Exception as e:
        print(f"Capacity scan failed: {e}")
        all_results["capacity_scan"] = {"error": str(e)}

    # 保存结果
    output_file = output_dir / "recurrentgemma_state_audit.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Results saved to {output_file} ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())

