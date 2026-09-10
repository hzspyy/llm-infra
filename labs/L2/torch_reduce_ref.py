#!/usr/bin/env python3
"""L2.3 lab · 把自己写的归约阶梯与 PyTorch 官方实现放到同一把尺子上。

自己优化到 100% 带宽之后，还要问一句：官方实现是多少？
如果官方更快，说明还有你没想到的手段；如果一样，说明确实到顶了。
"""
import statistics
import sys

import torch

PEAK_GBS = 1674.0            # L1.1 实测的只读带宽


def bench(fn, iters: int = 30, warm: int = 10) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main() -> None:
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n = mb * 1024 * 1024 // 4
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    nbytes = n * 4
    print(f"=== {torch.cuda.get_device_name(0)}   {mb} MB fp32（{n} 个元素，全为 1.0）")
    print(f"    分母：L1.1 实测只读带宽 {PEAK_GBS:.0f} GB/s\n")
    print("    %-34s %9s %10s %8s  %s" % ("实现", "耗时ms", "GB/s", "占上限", "求和结果"))
    cases = [
        ("torch.sum（fp32 累加）", lambda: x.sum()),
        ("torch.sum(dtype=float64)", lambda: x.sum(dtype=torch.float64)),
        ("torch.linalg.vector_norm", lambda: torch.linalg.vector_norm(x)),
        ("x.mean()", lambda: x.mean()),
    ]
    for name, fn in cases:
        ms = bench(fn)
        v = float(fn().item())
        gbs = nbytes / (ms * 1e-3) / 1e9
        print("    %-34s %9.3f %10.1f %7.1f%%  %.6g" % (name, ms, gbs, gbs / PEAK_GBS * 100, v))

    print(f"\n    正确答案应为 {n}")
    print("    对照 labs/L2/reduce_ladder.cu 的 v6：0.160 ms / 1679 GB/s / 100%")


if __name__ == "__main__":
    main()
