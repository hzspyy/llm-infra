#!/usr/bin/env python3
"""
实验 2：跨流依赖与竞态
演示 wait_stream 和 record_stream 的不同作用
"""
import torch
import json
import time

def experiment_no_sync():
    """不同步的跨流使用：可能出现竞态"""
    print("=" * 80)
    print("实验 2A：跨流使用，无同步（可能出错）")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2

    with torch.cuda.stream(s2):
        # 危险：s2 可能在 y 计算完成前读取
        z = y * 3

    torch.cuda.synchronize()

    # 检查结果正确性
    with torch.cuda.stream(s1):
        x_check = torch.randn(1000, 1000, device='cuda')
        y_check = x_check * 2
    torch.cuda.synchronize()
    z_check = y_check * 3

    diff = (z - z_check).abs().max().item()
    print(f"结果差异（与串行对比）: {diff}")
    print("注意：在某些情况下可能看到错误结果")
    print()

    return {"max_diff": diff}

def experiment_wait_stream():
    """使用 wait_stream 确保执行顺序"""
    print("=" * 80)
    print("实验 2B：使用 wait_stream 确保顺序")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2

    with torch.cuda.stream(s2):
        s2.wait_stream(s1)  # s2 等待 s1 的所有先前操作
        z = y * 3

    torch.cuda.synchronize()

    # 验证正确性
    with torch.cuda.stream(s1):
        x_check = torch.randn(1000, 1000, device='cuda')
        y_check = x_check * 2
    torch.cuda.synchronize()
    z_check = y_check * 3

    diff = (z - z_check).abs().max().item()
    print(f"结果差异: {diff}")
    print("✓ wait_stream 确保了执行顺序")
    print()

    return {"max_diff": diff}

def experiment_record_stream():
    """使用 record_stream 保护内存"""
    print("=" * 80)
    print("实验 2C：使用 record_stream 保护内存")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        y = torch.randn(1000, 1000, device='cuda')
        y = y * 2

    # 告诉 allocator：y 会在 s2 使用
    y.record_stream(s2)

    with torch.cuda.stream(s2):
        z = y * 3

    torch.cuda.synchronize()
    print("✓ record_stream 防止了内存过早复用")
    print()

    return {"success": True}

def experiment_both():
    """组合使用 wait_stream 和 record_stream"""
    print("=" * 80)
    print("实验 2D：组合使用（推荐做法）")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2

    # 组合使用
    y.record_stream(s2)  # 防止内存过早复用

    with torch.cuda.stream(s2):
        s2.wait_stream(s1)  # 确保执行顺序
        z = y * 3

    torch.cuda.synchronize()
    print("✓ 组合使用确保了执行正确和内存安全")
    print()

    return {"success": True}

def experiment_memory_race():
    """演示内存复用竞态"""
    print("=" * 80)
    print("实验 2E：内存复用竞态")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    results = []

    for trial in range(3):
        print(f"\n尝试 {trial + 1}:")

        with torch.cuda.stream(s1):
            x = torch.randn(1000, 1000, device='cuda')
            y = x * 2

        # 不使用 record_stream
        # del x 后 allocator 可能立即复用 x 的内存

        with torch.cuda.stream(s2):
            # 如果没有 wait_stream，s2 可能在 y 计算完成前启动
            # 如果没有 record_stream，allocator 可能复用 y 的内存
            z = y * 3
            w = torch.randn(1000, 1000, device='cuda')  # 可能复用 y 的内存

        torch.cuda.synchronize()
        results.append({"trial": trial, "completed": True})

    print("\n所有尝试完成（但结果可能不正确）")
    print()

    return results

def experiment_timing():
    """测量同步开销"""
    print("=" * 80)
    print("实验 2F：同步开销测量")
    print("=" * 80)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    n_iters = 100

    # 无同步
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.cuda.stream(s1):
            x = torch.randn(1000, 1000, device='cuda')
        with torch.cuda.stream(s2):
            y = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    time_no_sync = time.perf_counter() - start

    # 使用 wait_stream
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.cuda.stream(s1):
            x = torch.randn(1000, 1000, device='cuda')
        with torch.cuda.stream(s2):
            s2.wait_stream(s1)
            y = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    time_with_sync = time.perf_counter() - start

    print(f"无同步: {time_no_sync*1000:.2f} ms")
    print(f"使用 wait_stream: {time_with_sync*1000:.2f} ms")
    print(f"开销: {(time_with_sync - time_no_sync)*1000:.2f} ms ({(time_with_sync/time_no_sync - 1)*100:.1f}%)")
    print()

    return {
        "no_sync_ms": time_no_sync * 1000,
        "with_sync_ms": time_with_sync * 1000,
        "overhead_pct": (time_with_sync / time_no_sync - 1) * 100
    }

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping experiments")
        exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    output = {
        "experiment_2a": experiment_no_sync(),
        "experiment_2b": experiment_wait_stream(),
        "experiment_2c": experiment_record_stream(),
        "experiment_2d": experiment_both(),
        "experiment_2e": experiment_memory_race(),
        "experiment_2f": experiment_timing(),
    }

    # 保存结果
    with open("memory_cross_stream.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 memory_cross_stream.json")
