#!/usr/bin/env python3
"""蒸馏学生的推理成本（7.7-J）：同一 batch/序列长度下比较教师与学生的前向与解码代价。

只做两件事：固定输入的 prefill 计时（预热 + 多轮取中位数）和固定步数的贪心解码计时，
同时记录参数量、权重字节与峰值显存。它回答"压缩学生换来多少部署成本"，不与训练质量混用。

Usage:
    python labs/L7/teaching_student_cost.py --minimind-src "$SRC/minimind" \
      --teacher "$RUN/sft/best.pt" --student "$RUN/distill-kd/best.pt" \
      --outdir "$RUN/distill-cost"
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


def log(message):
    print(f"[teaching_student_cost] {message}", flush=True)


def load(minimind_src: Path, path: Path, hidden, layers, device):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    config = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_moe=False)
    model = MiniMindForCausalLM(config).to(device)
    if path:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


@torch.no_grad()
def time_prefill(model, batch, seq_len, dtype, device, rounds=5, warmup=2):
    x = torch.randint(3, 6000, (batch, seq_len), device=device)
    times = []
    for i in range(warmup + rounds):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=dtype, enabled=True):
            model(x)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return statistics.median(times)


@torch.no_grad()
def time_decode(model, batch, prompt_len, steps, dtype, device, rounds=3, warmup=1):
    x = torch.randint(3, 6000, (batch, prompt_len), device=device)
    times = []
    for i in range(warmup + rounds):
        cur = x.clone()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                logits = model(cur).logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return statistics.median(times), batch * steps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher-hidden-size", type=int, default=768)
    parser.add_argument("--teacher-num-layers", type=int, default=8)
    parser.add_argument("--student-hidden-size", type=int, default=512)
    parser.add_argument("--student-num-layers", type=int, default=8)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=False)
    device, dtype = "cuda", torch.bfloat16
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "batch": args.batch, "seq_len": args.seq_len, "decode_steps": args.decode_steps,
              "teacher": str(args.teacher), "student": str(args.student), "models": {}}

    for name, path, hidden, layers in (
            ("teacher", args.teacher, args.teacher_hidden_size, args.teacher_num_layers),
            ("student", args.student, args.student_hidden_size, args.student_num_layers)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = load(args.minimind_src, path, hidden, layers, device)
        params = sum(p.numel() for p in model.parameters())
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        prefill_ms = time_prefill(model, args.batch, args.seq_len, dtype, device)
        decode_ms, decode_tokens = time_decode(model, args.batch, args.seq_len // 2,
                                               args.decode_steps, dtype, device)
        entry = {"config": f"{hidden}/{layers}", "params": params,
                 "weight_bytes_fp32": weight_bytes,
                 "weight_bytes_bf16_estimate": params * 2,
                 "prefill_ms": prefill_ms,
                 "prefill_tokens_per_second": args.batch * args.seq_len / prefill_ms * 1000,
                 "decode_total_ms": decode_ms,
                 "decode_ms_per_token": decode_ms / args.decode_steps,
                 "decode_tokens_per_second": decode_tokens / decode_ms * 1000,
                 "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20}
        result["models"][name] = entry
        log(f"{name} {entry['config']}：{params/1e6:.3f}M 参数，prefill {prefill_ms:.2f} ms"
            f"（{entry['prefill_tokens_per_second']:.0f} tok/s），"
            f"decode {entry['decode_tokens_per_second']:.1f} tok/s，"
            f"峰值 {entry['peak_memory_mib']:.0f} MiB")
        del model
        torch.cuda.empty_cache()

    teacher, student = result["models"]["teacher"], result["models"]["student"]
    result["ratios"] = {
        "params": student["params"] / teacher["params"],
        "prefill_ms": student["prefill_ms"] / teacher["prefill_ms"],
        "decode_ms_per_token": student["decode_ms_per_token"] / teacher["decode_ms_per_token"],
        "peak_memory_mib": student["peak_memory_mib"] / teacher["peak_memory_mib"],
    }
    (args.outdir / "student_cost.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    log(f"完成：学生/教师 参数 {result['ratios']['params']:.3f}、"
        f"prefill {result['ratios']['prefill_ms']:.3f}、"
        f"decode {result['ratios']['decode_ms_per_token']:.3f}")


if __name__ == "__main__":
    main()
