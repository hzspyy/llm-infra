#!/usr/bin/env python3
"""L5.4 补测 · 运行时到底挑了哪个 GEMM kernel（对照启发式候选表）。

`cublaslt_heuristic.cu` 打印的是启发式**候选表**；这里打印运行时**实际发射**的
kernel 名，两者并排才能说明「换档」发生在哪里。

形状取 Qwen3-1.7B 的 q_proj：(M=batch, K=2048) x (K=2048, N=2048)，bf16。
每个 M 预热后跑 20 次，用 torch.profiler 取 device kernel 名与次数。

用法（crater）：
    python cublas_choice_probe.py --out <dir>
"""

import argparse
import json
import os
import statistics
import time

import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

N = 2048
K = 2048
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def one(M, reps=20):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize()
    kernels = {}
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.count:
            kernels[ev.key] = ev.count
    del a, b, c
    return dict(M=M, median_us=statistics.median(times), kernels=kernels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [one(m) for m in MS]
    lines = [f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · "
             f"bf16 ({K},) x ({K},{N})，20 次中位与 5 次 profiler 采集", ""]
    lines.append(f"  {'M':>5}{'中位 µs':>10}  实际发射的 device kernel（次数）")
    for r in rows:
        ks = ", ".join(f"{k.split('(')[0][:60]}×{v}"
                       for k, v in sorted(r["kernels"].items(),
                                          key=lambda kv: -kv[1]))
        lines.append(f"  {r['M']:>5}{r['median_us']:>10.1f}  {ks}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "cublas_choice.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "cublas_choice.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n写入 {args.out}/cublas_choice.txt")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
