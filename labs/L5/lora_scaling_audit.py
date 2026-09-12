#!/usr/bin/env python3
"""L5.10 real vLLM adapter-count sweep, separate routing/profiler evidence.

All adapters are synthetic rank-8 q_proj fixtures. No model quality claim.
Outputs are exclusive-created; existing artifacts must never be overwritten.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
MODEL = "Qwen/Qwen3-1.7B"


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def main(args):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.lora.model_manager import LoRAModelManager, LRUCacheLoRAModelManager
    import vllm.v1.core.sched.scheduler as scheduler
    import vllm.lora.punica_wrapper.punica_gpu as punica
    import vllm.lora.worker_manager as worker

    args.out.mkdir(parents=True, exist_ok=False)
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    versions = {p: md.version(p) for p in ["torch", "vllm", "triton", "transformers", "safetensors"]}
    save(args.out / "environment.json", {"versions": versions, "gpu": torch.cuda.get_device_name(),
        "model": MODEL, "revision": cfg._commit_hash, "cuda": torch.version.cuda,
        "gpu_memory_utilization": os.environ["L510_GPU_MEMORY_UTILIZATION"],
        "cache": {k: os.environ.get(k) for k in ["HF_HOME", "HF_HUB_CACHE", "TRITON_CACHE_DIR"]}})
    sources = args.out / "source"
    sources.mkdir()
    source_rows = []
    for module in [scheduler, punica, worker]:
        src = Path(inspect.getfile(module))
        dest = sources / src.name
        shutil.copyfile(src, dest)
        source_rows.append({"module": module.__name__, "file": str(src),
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    save(args.out / "source_manifest.json", source_rows)

    # Generate small distinct adapters without copying the base model.
    adapters = args.out / "adapters"
    adapters.mkdir()
    manifest = []
    requests = []
    h = cfg.hidden_size
    q = cfg.num_attention_heads * cfg.head_dim
    for ident in range(1, 65):
        folder = adapters / str(ident)
        folder.mkdir()
        adapter_cfg = {"base_model_name_or_path": MODEL, "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "inference_mode": True, "r": 8, "lora_alpha": 8, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "use_dora": False}
        save(folder / "adapter_config.json", adapter_cfg)
        gen = torch.Generator().manual_seed(2000 + ident)
        weights = {}
        for layer in range(cfg.num_hidden_layers):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
            weights[prefix + ".lora_A.weight"] = (torch.randn(8, h, generator=gen) * .03).bfloat16()
            weights[prefix + ".lora_B.weight"] = (torch.randn(q, 8, generator=gen) * .03).bfloat16()
        weightfile = folder / "adapter_model.safetensors"
        save_file(weights, str(weightfile), metadata={"format": "pt"})
        raw = weightfile.read_bytes()
        if ident == 1:
            save(args.out / "adapter1_header.json", json.loads(raw[8:8 + int.from_bytes(raw[:8], "little")]))
        manifest.append({"id": ident, "seed": 2000 + ident, "rank": 8,
            "tensor_bytes": sum(t.numel() * t.element_size() for t in weights.values()),
            "file_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        requests.append(LoRARequest(f"fixture-{ident}", ident, str(folder)))
    save(args.out / "adapter_manifest.json", manifest)

    # All hooks are inactive during ordinary latency measurements.
    telemetry = {"enabled": False, "events": []}
    old_mapping = LoRAModelManager.set_adapter_mapping
    old_activate = LRUCacheLoRAModelManager.activate_adapter
    def mapping(self, value):
        if telemetry["enabled"]:
            counts = Counter(int(v) for v in value.index_mapping)
            telemetry["events"].append({"kind": "mapping", "tokens": len(value.index_mapping),
                "counts": dict(counts), "gpu_slots": list(self.lora_index_to_id)})
        return old_mapping(self, value)
    def activate(self, ident, *a, **kw):
        if not telemetry["enabled"]:
            return old_activate(self, ident, *a, **kw)
        before = list(self.lora_index_to_id)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = old_activate(self, ident, *a, **kw)
        torch.cuda.synchronize()
        telemetry["events"].append({"kind": "activate", "id": ident, "before": before,
            "after": list(self.lora_index_to_id), "wall_ms_sync": (time.perf_counter() - start) * 1000})
        return result
    LoRAModelManager.set_adapter_mapping = mapping
    LRUCacheLoRAModelManager.activate_adapter = activate

    engine_cfg = dict(model=MODEL, enable_lora=True, max_loras=8, max_cpu_loras=64, max_lora_rank=8,
        gpu_memory_utilization=float(os.environ["L510_GPU_MEMORY_UTILIZATION"]), max_model_len=256,
        max_num_seqs=64, max_num_batched_tokens=2048, enforce_eager=True,
        enable_prefix_caching=False, disable_log_stats=True)
    save(args.out / "engine_config.json", engine_cfg)
    llm = LLM(**engine_cfg)
    prompts = ["Write a short paragraph about GPU memory and scheduling."] * 64
    params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    counts = [1, 4, 7, 8, 9, 16, 64]
    timings = []
    def generate(n):
        chosen = [requests[i % n] for i in range(64)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = llm.generate(prompts, params, lora_request=chosen, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        tokens = sum(len(o.outputs[0].token_ids) for o in out)
        assert tokens == 1024, tokens
        assert all(o.outputs[0].finish_reason == "length" for o in out)
        return elapsed, out
    try:
        for n in counts:
            # Two immediate repetitions populate the CPU cache and JIT paths.
            for _ in range(2):
                generate(n)
        for rep in range(5):
            order = counts[rep:] + counts[:rep]
            for n in order:
                # Standardize starting residency to this arm; >8 still thrashes.
                generate(n)
                elapsed, out = generate(n)
                row = {"rep": rep, "adapters": n, "batch": 64, "output_tokens": 1024,
                    "wall_ms": elapsed, "tokens_per_second": 1024000 / elapsed,
                    "request_distribution": dict(Counter((i % n) + 1 for i in range(64)))}
                timings.append(row)
                print(json.dumps({"kind": "timing", **row}), flush=True)
        save(args.out / "timings.json", timings)
        summary = [{"adapters": n, "median_ms": statistics.median(r["wall_ms"] for r in timings if r["adapters"] == n),
            "median_tokens_per_second": statistics.median(r["tokens_per_second"] for r in timings if r["adapters"] == n)} for n in counts]
        save(args.out / "summary.json", summary)

        # Structural runs: deliberately synchronized hooks, never used as performance results.
        for n in [1, 8, 9, 16, 64]:
            generate(n)
            telemetry["enabled"] = True
            telemetry["events"] = []
            _, out = generate(n)
            telemetry["enabled"] = False
            save(args.out / f"routing-{n}.json", telemetry["events"])
            save(args.out / f"tokens-{n}.json", [list(o.outputs[0].token_ids) for o in out])
        # Profiler is separate from ordinary timing arms.
        generate(8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            generate(8)
        prof.export_chrome_trace(str(args.out / "mixed8.trace.json"))
        kernels = Counter()
        durations = Counter()
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels[event.name] += 1
                durations[event.name] += event.device_time_total
        save(args.out / "kernel_summary.json", [{"name": k, "calls": kernels[k], "cuda_us": durations[k]}
             for k in sorted(kernels, key=lambda name: -durations[name])])
        print(json.dumps({"kind": "complete", "summary": summary}), flush=True)
    finally:
        if not (args.out / "timings.json").exists():
            save(args.out / "timings-partial.json", timings)
        llm.llm_engine.engine_core.shutdown()
        LoRAModelManager.set_adapter_mapping = old_mapping
        LRUCacheLoRAModelManager.activate_adapter = old_activate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args())
