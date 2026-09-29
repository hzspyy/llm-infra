#!/usr/bin/env python3
"""L2.6-A/C · 四种瓶颈的区分实验与一个最小复现。

同一个 profiler 口径下，人为构造四类瓶颈，先写下竞争解释，再用"需要哪个事件"
把它区分开：

  1. CPU 提交受限：几千个微小 kernel，GPU 忙但中间有空隙
  2. 带宽受限：单个大 kernel，GPU 忙且吞吐接近上限
  3. 同步受限：kernel 与 .item() 交替，墙钟远大于 GPU 时间
  4. 占用受限：一个大共享内存 kernel，GPU 满但吞吐上不去

每个场景都记：墙钟、GPU 忙时间（只取 DeviceType.CUDA）、launch 数、
父子事件是否重复累加、以及区分它所需的那个事件。

    python profiling_bottlenecks.py --out-json <path>
"""

import argparse
import json
import pathlib
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile


def measure(fn, iters=5, label="", bytes_moved=None):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0 / iters
    gpu_busy = 0.0
    launches = 0
    total_sum = 0.0
    self_sum = 0.0
    top = []
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            gpu_busy += ev.self_device_time_total / 1000.0
            launches += ev.count
            top.append((ev.key, ev.count, ev.self_device_time_total / 1000.0))
        if ev.key.startswith("aten::") or ev.key.startswith("void "):
            total_sum += ev.cpu_time_total / 1000.0
            self_sum += ev.self_cpu_time_total / 1000.0
    top.sort(key=lambda r: -r[2])
    gpu_busy /= iters              # 与墙钟同口径：都是"每次调用"
    gbs = (bytes_moved / (wall * 1e-3) / 1e9) if bytes_moved else None
    return {"label": label, "wall_ms": wall, "gpu_busy_ms": gpu_busy,
            "gpu_util_pct": 100.0 * gpu_busy / wall if wall else 0.0,
            "bytes_moved": bytes_moved, "achieved_gbs": gbs,
            "cuda_launches": launches,
            "cpu_total_ms": total_sum, "cpu_self_ms": self_sum,
            "double_count_ratio": (total_sum / self_sum) if self_sum else 0.0,
            "top_dev": top[:4]}


def scenario_cpu_submit():
    """几千个微小 kernel：GPU 时间很短，CPU 提交把它们串起来。"""
    x = torch.ones(64, device="cuda")
    def run():
        for _ in range(2000):
            x.add_(1.0)
    return run, "CPU 提交受限：2000 个 add_（每个 64 元素）", "CPU 侧提交时间与 kernel 之间的空隙", None


def scenario_bandwidth():
    """一个足够大的逐元素 kernel，工作集远超 L2。"""
    n = 128 << 20                      # 512 MB
    a = torch.ones(n, device="cuda")
    b = torch.ones(n, device="cuda")
    def run():
        torch.add(a, b, out=b)
    return run, "带宽受限：512 MB 逐元素加", "DRAM 字节数与实测带宽（自算，crater 无计数器）", 3 * n * 4


def scenario_sync():
    """每个 kernel 后面接一次 .item()，强制同步。"""
    x = torch.ones(1 << 20, device="cuda")
    def run():
        for _ in range(20):
            x.add_(1.0)
            _ = x[0].item()                # 阻塞式回读
    return run, "同步受限：20 次 kernel + .item()", "D2H 回读次数与同步等待", None


def scenario_occupancy():
    """占用受限：每条线程申请大块动态共享内存，硬件允许但驻留数低。"""
    from torch.utils.cpp_extension import load_inline
    import os
    build = os.path.join(os.environ.get("LEARN_ROOT", "/scratch/learn"), ".cache", "torchext", "l26")
    os.makedirs(build, exist_ok=True)
    src = r'''
#include <torch/extension.h>
__global__ void smem_heavy(const float* __restrict__ in, float* __restrict__ out,
                           size_t n, size_t iters) {
    extern __shared__ float s[];
    float acc = 0.f;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t k = i; k < n; k += stride) acc += in[k];
    s[threadIdx.x] = acc;
    __syncthreads();
    float v = s[(threadIdx.x + 1) % blockDim.x];
    for (size_t t = 0; t < iters; ++t) v = fmaf(v, 1.0001f, 0.5f);
    if (v == 1234.5f) out[blockIdx.x % 1024] = v;
}
torch::Tensor run_smem(torch::Tensor x, torch::Tensor out, int64_t smem_kb, int64_t iters) {
    size_t n = x.numel();
    cudaFuncSetAttribute(smem_heavy, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    smem_heavy<<<680, 256, (size_t)smem_kb * 1024>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n, (size_t)iters);
    return out;
}
'''
    src = src.replace("    return out;\n}", "    return out;\n}")
    src += r'''
int64_t occupancy(int64_t smem_kb) {
    int bps = 0;
    cudaFuncSetAttribute(smem_heavy, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, smem_heavy, 256, (size_t)smem_kb * 1024);
    return bps;
}
'''
    mod = load_inline(name="l26_smem",
                      cpp_sources="torch::Tensor run_smem(torch::Tensor, torch::Tensor, int64_t, int64_t);\nint64_t occupancy(int64_t);",
                      cuda_sources=src, functions=["run_smem", "occupancy"],
                      build_directory=build, verbose=False)
    x = torch.ones(4 << 20, device="cuda")     # 16 MB，L2 内
    out = torch.zeros(1024, device="cuda")
    occ = int(mod.occupancy(96))
    def run():
        mod.run_smem(x, out, 96, 20000)        # 96 KB/block ⇒ 1 block/SM
    return (run, "占用受限：96 KB 动态共享内存（1 block/SM）",
            "驻留 block/SM（cudaOccupancyMaxActiveBlocksPerMultiprocessor）", 4 << 20, occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    print(f"=== {torch.cuda.get_device_name(0)}  torch {torch.__version__} ===")
    print("口径：墙钟用 perf_counter；GPU 忙时间只累加 DeviceType.CUDA 的 self 时间\n")
    rows = []
    signals = []
    for maker in (scenario_cpu_submit, scenario_bandwidth, scenario_sync, scenario_occupancy):
        fn, label, signal, nbytes, *rest = maker()
        r = measure(fn, iters=args.iters, label=label, bytes_moved=nbytes)
        if rest:
            r["occupancy_blocks_per_sm"] = rest[0]
        r["distinguishing_event"] = signal
        rows.append(r)
        signals.append((label, signal))
        print(f"[{label}]")
        print(f"  墙钟 {r['wall_ms']:9.3f} ms | GPU 忙 {r['gpu_busy_ms']:9.3f} ms "
              f"| GPU 占比 {r['gpu_util_pct']:5.1f}% | CUDA 事件数 {r['cuda_launches']}")
        print(f"  CPU 侧 aten 时间：total {r['cpu_total_ms']:.2f} ms / self {r['cpu_self_ms']:.2f} ms "
              f"⇒ 父子重复累加倍数 {r['double_count_ratio']:.2f}×")
        extra = ""
        if r.get("achieved_gbs"):
            extra = f" | 实测带宽 {r['achieved_gbs']:.1f} GB/s（{r['bytes_moved']/1048576:.0f} MB/次）"
        if r.get("occupancy_blocks_per_sm") is not None:
            extra += f" | 驻留 {r['occupancy_blocks_per_sm']} block/SM"
        print(f"  最重的 GPU 事件：" + ", ".join(f"{k[:60]}×{c} ({t:.3f} ms)" for k, c, t in r['top_dev'][:2]) + extra)
    print()
    print("四类瓶颈各自的判据（先写假设，再指定能区分它的事件）：")
    hypotheses = {
        "CPU 提交受限：2000 个 add_（每个 64 元素）": "假设：GPU 时间很短，墙钟由 CPU 提交决定",
        "带宽受限：512 MB 逐元素加": "假设：墙钟 ≈ GPU 忙，吞吐接近 DRAM 上限",
        "同步受限：20 次 kernel + .item()": "假设：GPU 大部分时间在等 CPU 回读",
        "占用受限：96 KB 动态共享内存（1 block/SM）": "假设：GPU 一直忙但吞吐上不去，因为驻留 warp 太少",
    }
    for lab, sig in signals:
        print(f"  · {lab}")
        print(f"      {hypotheses[lab]}")
        print(f"      区分它需要的事件：{sig}")

    result = {"rows": rows}
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
