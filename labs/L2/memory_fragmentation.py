#!/usr/bin/env python3
"""
实验 3：动态形状与碎片
演示变长序列如何导致内存碎片积累
"""
import torch
import json
import random

def print_memory_stats(label):
    """打印内存统计"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    print(f"{label:40} | Alloc: {allocated:7.2f} MB | Reserved: {reserved:7.2f} MB | Frag: {fragmentation:5.1f}%")
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "fragmentation_pct": fragmentation
    }

def experiment_uniform_sizes():
    """固定大小分配：无碎片"""
    print("=" * 90)
    print("实验 3A：固定大小分配（基线）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    size = 1000

    for i in range(10):
        x = torch.randn(size, size, device='cuda')
        stats = print_memory_stats(f"迭代 {i+1}: {size}x{size}")
        results.append(stats)
        del x

    print()
    return results

def experiment_varying_sizes():
    """变化大小分配：产生碎片"""
    print("=" * 90)
    print("实验 3B：变化大小分配（产生碎片）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    sizes = [500, 1000, 1500, 2000, 1000, 500, 1500, 1000]

    for i, size in enumerate(sizes):
        x = torch.randn(size, size, device='cuda')
        stats = print_memory_stats(f"迭代 {i+1}: {size}x{size}")
        results.append(stats)
        del x

    print()
    return results

def experiment_random_sizes():
    """随机大小：极端碎片"""
    print("=" * 90)
    print("实验 3C：随机大小分配（极端碎片）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    random.seed(42)

    for i in range(15):
        size = random.randint(500, 2000)
        x = torch.randn(size, size, device='cuda')
        stats = print_memory_stats(f"迭代 {i+1}: {size}x{size}")
        results.append(stats)
        del x

    print()
    return results

def experiment_variable_seq_lens():
    """模拟变长序列（KV cache 场景）"""
    print("=" * 90)
    print("实验 3D：变长序列推理（模拟 KV cache）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    batch_size = 8
    hidden_dim = 4096
    seq_lens = [128, 256, 512, 1024, 512, 256, 128, 256, 512, 1024]

    for i, seq_len in enumerate(seq_lens):
        # 模拟 KV cache: [batch, seq_len, hidden]
        kv_cache = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
        stats = print_memory_stats(f"迭代 {i+1}: seq_len={seq_len}")
        results.append({"seq_len": seq_len, **stats})
        del kv_cache

    print()
    return results

def experiment_with_empty_cache():
    """使用 empty_cache 清理碎片"""
    print("=" * 90)
    print("实验 3E：使用 empty_cache 清理")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    sizes = [500, 1500, 1000, 2000, 1000]

    for i, size in enumerate(sizes):
        x = torch.randn(size, size, device='cuda')
        stats_before = print_memory_stats(f"迭代 {i+1}: {size}x{size} (before del)")
        del x

        # 每次删除后清空缓存
        torch.cuda.empty_cache()
        stats_after = print_memory_stats(f"迭代 {i+1}: after empty_cache")

        results.append({
            "size": size,
            "before_del": stats_before,
            "after_empty_cache": stats_after
        })

    print()
    return results

def experiment_preallocate():
    """预分配策略：避免碎片"""
    print("=" * 90)
    print("实验 3F：预分配最大尺寸（避免碎片）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    max_size = 2000
    sizes = [500, 1000, 1500, 2000, 1000, 500]

    # 预分配最大尺寸
    buffer = torch.zeros(max_size, max_size, device='cuda')
    stats_init = print_memory_stats("预分配 buffer")
    results.append({"stage": "init", **stats_init})

    for i, size in enumerate(sizes):
        # 使用切片，不分配新内存
        x = buffer[:size, :size]
        x.fill_(1.0)
        y = x * 2
        stats = print_memory_stats(f"迭代 {i+1}: 使用 {size}x{size} 切片")
        results.append({"size": size, **stats})

    print()
    return results

def experiment_fragmentation_growth():
    """长期运行：碎片增长趋势"""
    print("=" * 90)
    print("实验 3G：长期碎片增长")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    random.seed(42)

    for i in range(50):
        size = random.randint(500, 1500)
        x = torch.randn(size, size, device='cuda')

        if i % 10 == 0:
            stats = print_memory_stats(f"迭代 {i+1}")
            results.append({"iteration": i+1, **stats})

        del x

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
        "experiment_3a": experiment_uniform_sizes(),
        "experiment_3b": experiment_varying_sizes(),
        "experiment_3c": experiment_random_sizes(),
        "experiment_3d": experiment_variable_seq_lens(),
        "experiment_3e": experiment_with_empty_cache(),
        "experiment_3f": experiment_preallocate(),
        "experiment_3g": experiment_fragmentation_growth(),
    }

    # 保存结果
    with open("memory_fragmentation.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 memory_fragmentation.json")
