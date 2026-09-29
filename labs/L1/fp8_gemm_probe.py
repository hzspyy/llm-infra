#!/usr/bin/env python3
"""L1.2 补测：cuBLAS 的 FP8 GEMM 到底跑在哪条指令上。

既有结论里有一个矛盾：cuBLAS 的 FP8 GEMM 折算出 1501 FLOP/clk/SM，
而纯发射基准 `mma.sync.m16n8k32`（f32 累加）的上限只有 1022。
一个真实 GEMM 不可能超过它所用指令的发射上限，所以至少有一个假设是错的。

这个脚本补三件事：

  [1] 时钟。频率不用 NVML 采样，而是在 GEMM 跑的同时，在另一条流上跑一个
      只占 1 个 block 的 clock64 自旋 kernel，用「周期数 / 墙钟」直接算出
      被测区间的真实 SM 频率。
  [2] 吞吐。bf16 与 fp8 两条路径各测 5 轮交错，取中位。
  [3] 指令。用 torch profiler 拿到真实 kernel 名，再和 tensor_core_ladder.cu
      量出的各条指令上限对照，判断 cuBLAS 落在哪一条上。

用法：
    python labs/L1/fp8_gemm_probe.py --out-dir <dir> [--n 8192] [--rounds 5]
需要 GPU 与 torch；首次运行会用 nvcc 编译一个几十行的时钟探针。
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import torch

SEP = "-" * 78

PROBE_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clock_spin(unsigned long long* out, unsigned long long target) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    do {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    } while (t1 - t0 < target);
    if (threadIdx.x == 0) out[0] = t1 - t0;
}

// 只占 1 个 block（170 个 SM 里的 1 个），对被测 GEMM 的扰动可以忽略。
// 返回 (实际自旋周期数, 墙钟毫秒)，相除即被测区间的 SM 频率。
std::vector<double> measure_clock(long long target_cycles) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({1}, opts);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    clock_spin<<<1, 32>>>(reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
                          (unsigned long long)target_cycles);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    double cycles = (double)out.cpu().data_ptr<int64_t>()[0];
    return {cycles, (double)ms};
}
"""

PROBE_HDR = r"""
#include <torch/extension.h>
#include <vector>
std::vector<double> measure_clock(long long target_cycles);
"""


def build_probe(build_dir: Path):
    from torch.utils.cpp_extension import load_inline
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="clock_probe_l12", cpp_sources=PROBE_HDR, cuda_sources=PROBE_SRC,
        functions=["measure_clock"], build_directory=str(build_dir),
        extra_cuda_cflags=["-O3"], verbose=False)


def time_gemm(fn, rounds: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--build-dir", default="/scratch/learn/.cache/l12-clock-probe")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    sm = props.multi_processor_count
    n = args.n
    flop = 2.0 * n * n * n
    print(f"=== {props.name}  SM={sm}  torch={torch.__version__}  "
          f"矩阵 {n}x{n}x{n}  FLOP={flop / 1e12:.3f}T\n")

    probe = build_probe(Path(args.build_dir))

    a_bf16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.t().contiguous().t().to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device="cuda")

    def run_bf16():
        torch.mm(a_bf16, b_bf16)

    def run_fp8():
        torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale,
                         out_dtype=torch.bfloat16)

    # ---- 正确性：fp8 结果与 bf16 参照的相对误差
    ref = torch.mm(a_bf16, b_bf16).float()
    got = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale,
                           out_dtype=torch.bfloat16).float()
    rel = float((ref - got).norm() / ref.norm())
    print(f"[0] 正确性：fp8 vs bf16 相对误差 {rel:.4f}"
          f"（bf16 均值 {float(ref.abs().mean()):.4f} / fp8 均值 {float(got.abs().mean()):.4f}）")
    del ref, got
    torch.cuda.empty_cache()

    # ---- 空闲时的时钟
    idle = probe.measure_clock(2_000_000)
    idle_ghz = idle[0] / (idle[1] * 1e6)
    print(f"\n[1] 时钟探针（1 个 block 的 clock64 自旋）")
    print(f"    空闲时 {idle[0]:.0f} 周期 / {idle[1]:.2f} ms = {idle_ghz:.3f} GHz")

    results = {}
    for label, fn in [("bf16 torch.mm", run_bf16), ("fp8 _scaled_mm", run_fp8)]:
        stop = threading.Event()
        samples: list[float] = []

        def hammer():
            while not stop.is_set():
                fn()

        # 背景线程持续提交 GEMM，主线程在另一条流上量时钟
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        th = threading.Thread(target=hammer, daemon=True)
        th.start()
        time.sleep(0.5)                     # 等频率进入稳态
        for _ in range(5):
            c, ms = probe.measure_clock(4_000_000)
            samples.append(c / (ms * 1e6))
        stop.set()
        th.join()
        torch.cuda.synchronize()
        ghz = statistics.median(samples)

        times = time_gemm(fn, args.rounds)
        ms = statistics.median(times)
        tflops = flop / (ms * 1e-3) / 1e12
        fpc = tflops * 1e12 / (ghz * 1e9 * sm)
        results[label] = dict(ms_median=ms, ms_all=times, tflops=tflops,
                              ghz_during=ghz, ghz_samples=samples,
                              flop_per_clk_sm=fpc)
        print(f"    {label:<16s}GEMM 运行时 {ghz:.3f} GHz"
              f"（{min(samples):.3f}–{max(samples):.3f}）")

    print(f"\n[2] 吞吐与归一化（时钟取上面并发测到的中位数）")
    print(f"    {'路径':<18s}{'ms':>8s}{'TFLOPS':>10s}{'GHz':>8s}{'FLOP/clk/SM':>14s}")
    for label, r in results.items():
        print(f"    {label:<18s}{r['ms_median']:>8.3f}{r['tflops']:>10.1f}"
              f"{r['ghz_during']:>8.3f}{r['flop_per_clk_sm']:>14.0f}")

    # ---- kernel 名
    print(f"\n[3] 真实 kernel 名（torch profiler）")
    names = {}
    from torch.profiler import ProfilerActivity, profile
    for label, fn in [("bf16 torch.mm", run_bf16), ("fp8 _scaled_mm", run_fp8)]:
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.device_time > 0]
        ev.sort(key=lambda e: -e.device_time)
        names[label] = [dict(name=e.key, us=e.device_time, calls=e.count) for e in ev[:3]]
        print(f"    {label}")
        for e in names[label]:
            print(f"      {e['us']:>9.1f} µs  x{e['calls']}  {e['name'][:88]}")

    payload = dict(gpu=props.name, sm=sm, n=n, flop=flop, torch=torch.__version__,
                   rel_err_fp8_vs_bf16=rel, idle_ghz=idle_ghz,
                   results=results, kernels=names, host=platform.node())
    (out / "fp8_gemm_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
