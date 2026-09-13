#!/usr/bin/env python3
"""
实验 1：基础内存分配与释放
演示三个生命周期：Python 引用、allocator 可复用、GPU 完成
"""
import torch
import json

def print_memory_stats(label):
    """打印当前内存统计"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{label:30} | Allocated: {allocated:8.2f} MB | Reserved: {reserved:8.2f} MB")
    return {"allocated_mb": allocated, "reserved_mb": reserved}

def experiment_basic_lifecycle():
    """基础生命周期：分配、使用、释放"""
    print("=" * 80)
    print("实验 1A：单次分配与释放")
    print("=" * 80)

    results = {}

    # 初始状态
    torch.cuda.reset_peak_memory_stats()
    results["0_initial"] = print_memory_stats("初始状态")

    # 分配
    x = torch.randn(2000, 2000, device='cuda')
    results["1_after_alloc"] = print_memory_stats("分配 x (2000x2000 fp32)")

    # 使用
    y = x * 2
    results["2_after_compute"] = print_memory_stats("计算 y = x * 2")

    # 删除 Python 引用
    del x
    results["3_after_del_x"] = print_memory_stats("del x")

    # 同步（等待 GPU 完成）
    torch.cuda.synchronize()
    results["4_after_sync"] = print_memory_stats("synchronize()")

    # 删除 y
    del y
    torch.cuda.synchronize()
    results["5_after_del_y"] = print_memory_stats("del y + synchronize()")

    # 清空缓存
    torch.cuda.empty_cache()
    results["6_after_empty_cache"] = print_memory_stats("empty_cache()")

    print()
    return results

def experiment_multiple_allocations():
    """多次分配：观察 allocator 复用"""
    print("=" * 80)
    print("实验 1B：多次分配观察复用")
    print("=" * 80)

    results = []
    torch.cuda.reset_peak_memory_stats()

    for i in range(5):
        x = torch.randn(1000, 1000, device='cuda')
        stats = print_memory_stats(f"第 {i+1} 次分配")
        results.append(stats)
        del x
        torch.cuda.synchronize()

    print()
    return results

def experiment_size_variation():
    """不同大小分配：观察 segment 管理"""
    print("=" * 80)
    print("实验 1C：不同大小分配")
    print("=" * 80)

    results = {}
    torch.cuda.reset_peak_memory_stats()

    sizes = [500, 1000, 2000, 1000, 500]
    for i, size in enumerate(sizes):
        x = torch.randn(size, size, device='cuda')
        stats = print_memory_stats(f"分配 {size}x{size}")
        results[f"alloc_{i}_{size}"] = stats
        del x
        torch.cuda.synchronize()
        stats_after = print_memory_stats(f"释放 {size}x{size}")
        results[f"free_{i}_{size}"] = stats_after

    print()
    return results

def experiment_without_sync():
    """不同步的情况：观察 pending 状态"""
    print("=" * 80)
    print("实验 1D：不调用 synchronize")
    print("=" * 80)

    results = {}
    torch.cuda.reset_peak_memory_stats()

    # 分配并立即删除，不同步
    for i in range(3):
        x = torch.randn(1000, 1000, device='cuda')
        results[f"alloc_{i}"] = print_memory_stats(f"第 {i+1} 次分配")
        del x
        results[f"del_{i}"] = print_memory_stats(f"第 {i+1} 次 del（无 sync）")

    # 最后同步
    torch.cuda.synchronize()
    results["final_sync"] = print_memory_stats("最后 synchronize()")

    print()
    return results

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping experiments")
        exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    output = {
        "experiment_1a": experiment_basic_lifecycle(),
        "experiment_1b": experiment_multiple_allocations(),
        "experiment_1c": experiment_size_variation(),
        "experiment_1d": experiment_without_sync(),
    }

    # 保存结果
    with open("memory_lifecycle_basic.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 memory_lifecycle_basic.json")
