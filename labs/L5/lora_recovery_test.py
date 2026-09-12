#!/usr/bin/env python3
"""Test recovery paths after rank limit violation.

Tests three scenarios:
1. Exception caught, new valid request on same engine -> does it work?
2. Abort the invalid request explicitly -> can engine continue?
3. Restart engine after exception -> fresh state works?
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

MODEL = 'Qwen/Qwen3-1.7B'


def make_test_adapters(root):
    """Create one valid (rank 8) and one invalid (rank 32) adapter."""
    import torch
    from transformers import AutoConfig
    from safetensors.torch import save_file

    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    hidden = cfg.hidden_size

    for ident, rank in [(1, 8), (2, 32)]:
        folder = root / f'adapter-{ident}'
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

        generator = torch.Generator().manual_seed(100 + ident)
        weights = {}
        for layer in range(cfg.num_hidden_layers):
            prefix = f'base_model.model.model.layers.{layer}.self_attn.q_proj'
            a = torch.randn(rank, hidden, generator=generator, dtype=torch.bfloat16)
            b = torch.randn(hidden, rank, generator=generator, dtype=torch.bfloat16)
            weights[f'{prefix}.lora_A.weight'] = a
            weights[f'{prefix}.lora_B.weight'] = b

        save_file(weights, folder / 'adapter_model.safetensors')

    return root / 'adapter-1', root / 'adapter-2'


def run_scenario(scenario, valid_path, invalid_path, output_dir):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    result = {'scenario': scenario, 'success': False, 'error': None, 'steps': []}

    try:
        if scenario == 'continue_same_engine':
            # Create engine with max_lora_rank=16
            result['steps'].append('create_engine')
            llm = LLM(model=MODEL, max_loras=2, max_cpu_loras=3, max_lora_rank=16,
                      enable_lora=True, max_model_len=512, gpu_memory_utilization=0.3)

            # Try invalid adapter (rank 32 > 16)
            result['steps'].append('try_invalid_adapter')
            try:
                outputs = llm.generate(
                    prompts=["Test"],
                    sampling_params=SamplingParams(temperature=0, max_tokens=5),
                    lora_request=LoRARequest("invalid", 1, str(invalid_path))
                )
                result['steps'].append('invalid_adapter_unexpected_success')
            except Exception as e:
                result['steps'].append(f'invalid_adapter_raised: {type(e).__name__}')
                result['error_message'] = str(e)

            # Now try valid adapter
            result['steps'].append('try_valid_adapter_after_exception')
            try:
                outputs = llm.generate(
                    prompts=["Test"],
                    sampling_params=SamplingParams(temperature=0, max_tokens=5),
                    lora_request=LoRARequest("valid", 2, str(valid_path))
                )
                result['steps'].append('valid_adapter_success')
                result['success'] = True
            except Exception as e2:
                result['steps'].append(f'valid_adapter_failed: {type(e2).__name__}')
                result['error'] = f'{type(e2).__name__}: {e2}'

            del llm

        elif scenario == 'restart_engine':
            # First engine with invalid request
            result['steps'].append('create_first_engine')
            llm = LLM(model=MODEL, max_loras=2, max_cpu_loras=3, max_lora_rank=16,
                      enable_lora=True, max_model_len=512, gpu_memory_utilization=0.3)

            result['steps'].append('try_invalid_adapter')
            try:
                outputs = llm.generate(
                    prompts=["Test"],
                    sampling_params=SamplingParams(temperature=0, max_tokens=5),
                    lora_request=LoRARequest("invalid", 1, str(invalid_path))
                )
            except Exception as e:
                result['steps'].append(f'invalid_adapter_raised: {type(e).__name__}')

            result['steps'].append('shutdown_first_engine')
            del llm
            time.sleep(1)

            # Restart with fresh engine
            result['steps'].append('create_second_engine')
            llm = LLM(model=MODEL, max_loras=2, max_cpu_loras=3, max_lora_rank=16,
                      enable_lora=True, max_model_len=512, gpu_memory_utilization=0.3)

            result['steps'].append('try_valid_adapter_on_fresh_engine')
            outputs = llm.generate(
                prompts=["Test"],
                sampling_params=SamplingParams(temperature=0, max_tokens=5),
                lora_request=LoRARequest("valid", 2, str(valid_path))
            )
            result['steps'].append('valid_adapter_success_on_fresh_engine')
            result['success'] = True

            del llm

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'

    output_file = output_dir / f'{scenario}.json'
    with output_file.open('w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Creating test adapters...")
    valid_path, invalid_path = make_test_adapters(args.adapters)

    scenarios = ['continue_same_engine', 'restart_engine']

    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        result = run_scenario(scenario, valid_path, invalid_path, args.output)
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
