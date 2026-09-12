#!/usr/bin/env python3
"""
测量预热效应

Usage:
    python3 measure_warmup.py > ../../results/local/M2/warmup_effect.txt
"""
import torch
import time


def measure_first_vs_rest():
    """第一次调用 vs 后续调用"""
    print("=== 预热效应：第一次 vs 后续调用 ===\n")

    x = torch.randn(1000, 1000)

    times = []
    for i in range(10):
        start = time.perf_counter()
        y = x @ x
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {i+1:2d}: {elapsed*1e6:8.1f} µs")

    print(f"\n第一次: {times[0]*1e6:.1f} µs (包含初始化)")
    print(f"稳定后 (run 5-10 平均): {sum(times[4:])/6*1e6:.1f} µs")
    print(f"加速比: {times[0]/sum(times[4:])*6:.1f}x\n")


def measure_cache_effect():
    """L2 缓存的影响"""
    print("=== 缓存效应：小工作集 vs 大工作集 ===\n")

    # 小工作集：128x128 FP32 = 64 KB < L2
    x_small = torch.randn(128, 128)

    # 中等工作集：2048x2048 FP32 = 16 MB，可能部分命中 L2
    x_medium = torch.randn(2048, 2048)

    # 大工作集：8192x8192 FP32 = 256 MB > L2 (96 MB on RTX 5090)
    x_large = torch.randn(8192, 8192)

    def time_matmul(x, name):
        # 预热
        for _ in range(5):
            _ = x @ x

        # 测量
        start = time.perf_counter()
        for _ in range(20):
            _ = x @ x
        elapsed = (time.perf_counter() - start) / 20

        size_mb = x.numel() * x.element_size() / 1024**2
        print(f"{name:12s} {size_mb:8.1f} MiB  {elapsed*1e6:10.1f} µs")

    time_matmul(x_small, "小工作集")
    time_matmul(x_medium, "中等工作集")
    time_matmul(x_large, "大工作集")

    print("\n注：本机是 CPU-only，L2 缓存效应不如 GPU 明显")
    print("    在 GPU 上运行时会看到更大的差异\n")


def demonstrate_sync_importance():
    """演示同步的重要性（CPU 版本）"""
    print("=== 同步的重要性 ===\n")

    x = torch.randn(1000, 1000)

    # 预热
    for _ in range(5):
        _ = x @ x

    print("测量 10 次独立样本:")
    times = []
    for i in range(10):
        start = time.perf_counter()
        y = x @ x
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Sample {i+1:2d}: {elapsed*1e6:8.1f} µs")

    import statistics
    median = statistics.median(times)
    mean = statistics.mean(times)

    print(f"\n中位数: {median*1e6:.1f} µs")
    print(f"平均值: {mean*1e6:.1f} µs")
    print(f"标准差: {statistics.stdev(times)*1e6:.1f} µs")


if __name__ == "__main__":
    measure_first_vs_rest()
    measure_cache_effect()
    demonstrate_sync_importance()

    print("\n" + "="*60)
    print("关键要点：")
    print("  1. 丢弃前 3-5 次运行（包含初始化和编译）")
    print("  2. 工作集 > 缓存时才能声称测量了主存带宽")
    print("  3. 采集多个独立样本，报告中位数和分位数")
    print("="*60)
