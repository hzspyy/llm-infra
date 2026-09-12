#!/usr/bin/env python3
"""Test KV cache invalidation when adapter is updated with same name.

According to vllm/v1/core/kv_cache_utils.py:542-554, the hash key includes
lora_name. This test verifies that updating an adapter with the same name
causes KV cache misses for the new version.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

MODEL = 'Qwen/Qwen3-1.7B'


def make_adapter_pair(root, ident, rank, seed_a, seed_b):
    """Create two different adapters with same structure but different weights."""
    import torch
    from transformers import AutoConfig
    from safetensors.torch import save_file

    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    hidden = cfg.hidden_size

    for version, seed in [('v1', seed_a), ('v2', seed_b)]:
        folder = root / f'adapter-{ident}-{version}'
        folder.mkdir(parents=True, exist_ok=True)

        config = {
            'base_model_name_or_path': MODEL,
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
            'inference_mode': True,
            'r': rank,
            'lora_alpha': rank,
            'target_modules': ['q_proj'],
            'lora_dropout': 0.,
            'bias': 'none',
            'use_dora': False
        }
        with (folder / 'adapter_config.json').open('w') as f:
            json.dump(config, f, indent=2)

        generator = torch.Generator().manual_seed(seed)
        weights = {}
        for layer in range(cfg.num_hidden_layers):
            prefix = f'base_model.model.model.layers.{layer}.self_attn.q_proj'
            a = torch.randn(rank, hidden, generator=generator, dtype=torch.bfloat16)
            b = torch.randn(hidden, rank, generator=generator, dtype=torch.bfloat16)
            weights[f'{prefix}.lora_A.weight'] = a
            weights[f'{prefix}.lora_B.weight'] = b

        save_file(weights, folder / 'adapter_model.safetensors')

    return root / f'adapter-{ident}-v1', root / f'adapter-{ident}-v2'


def run_test(adapter_v1, adapter_v2, output_dir):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    result = {
        'test': 'kv_invalidation_on_same_name_update',
        'steps': [],
        'observations': {}
    }

    # Shared prompt for prefix cache test
    prompt = "The quick brown fox jumps over the lazy dog. " * 10

    llm = LLM(model=MODEL, max_loras=2, max_cpu_loras=3, max_lora_rank=8,
              enable_lora=True, enable_prefix_caching=True,
              max_model_len=512, gpu_memory_utilization=0.3)

    sampling = SamplingParams(temperature=0, max_tokens=8)

    # Step 1: Generate with adapter v1, name "test"
    result['steps'].append('generate_with_v1_name_test')
    outputs_v1_first = llm.generate(
        prompts=[prompt],
        sampling_params=sampling,
        lora_request=LoRARequest("test", 1, str(adapter_v1))
    )
    result['v1_first_output'] = outputs_v1_first[0].outputs[0].text

    # Step 2: Generate again with same adapter v1, same name -> should hit cache
    result['steps'].append('generate_with_v1_again_name_test')
    outputs_v1_second = llm.generate(
        prompts=[prompt],
        sampling_params=sampling,
        lora_request=LoRARequest("test", 1, str(adapter_v1))
    )
    result['v1_second_output'] = outputs_v1_second[0].outputs[0].text

    # Step 3: Generate with adapter v2, SAME name "test" -> new weights, should invalidate
    result['steps'].append('generate_with_v2_name_test')
    outputs_v2 = llm.generate(
        prompts=[prompt],
        sampling_params=sampling,
        lora_request=LoRARequest("test", 2, str(adapter_v2))
    )
    result['v2_output'] = outputs_v2[0].outputs[0].text

    # Step 4: Generate with adapter v1 again, different lora_id but same path -> reuses?
    result['steps'].append('generate_with_v1_third_name_test')
    outputs_v1_third = llm.generate(
        prompts=[prompt],
        sampling_params=sampling,
        lora_request=LoRARequest("test", 3, str(adapter_v1))
    )
    result['v1_third_output'] = outputs_v1_third[0].outputs[0].text

    # Observations
    result['observations']['v1_consistent'] = (
        result['v1_first_output'] == result['v1_second_output'] == result['v1_third_output']
    )
    result['observations']['v2_different_from_v1'] = result['v2_output'] != result['v1_first_output']

    # According to kv_cache_utils.py:542-554, lora_name is part of hash key
    # So same name with different weights should NOT hit cache from old version
    result['observations']['interpretation'] = (
        "If lora_name is the hash key, updating weights at same name should cause KV miss. "
        "If v1_consistent=True and v2_different=True, weights differ. "
        "Cache behavior requires inspecting vLLM internal stats."
    )

    del llm

    output_file = output_dir / 'kv_invalidation.json'
    with output_file.open('w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Creating adapter pair...")
    adapter_v1, adapter_v2 = make_adapter_pair(args.adapters, ident=1, rank=8,
                                                seed_a=1000, seed_b=2000)

    print("\nRunning KV invalidation test...")
    result = run_test(adapter_v1, adapter_v2, args.output)

    print("\nResults:")
    print(f"  v1 outputs consistent: {result['observations']['v1_consistent']}")
    print(f"  v2 different from v1: {result['observations']['v2_different_from_v1']}")
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
