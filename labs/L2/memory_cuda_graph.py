#!/usr/bin/env python3
"""
实验 4：CUDA Graph 内存池
演示图池与普通池的隔离
"""
import torch
import json

def print_memory_stats(label):
    """打印内存统计"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{label:50} | Alloc: {allocated:7.2f} MB | Reserved: {reserved:7.2f} MB")
    return {"allocated_mb": allocated, "reserved_mb": reserved}

def experiment_normal_execution():
    """普通执行：每次迭代分配"""
    print("=" * 90)
    print("实验 4A：普通执行（基线）")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    stats_init = print_memory_stats("初始状态")
    results.append({"stage": "init", **stats_init})

    for i in range(5):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2
        z = y + x
        stats = print_memory_stats(f"迭代 {i+1}")
        results.append({"iteration": i+1, **stats})
        del x, y, z

    print()
    return results

def experiment_graph_execution():
    """CUDA Graph 执行：录制时分配一次"""
    print("=" * 90)
    print("实验 4B：CUDA Graph 执行")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []
    stats_init = print_memory_stats("初始状态")
    results.append({"stage": "init", **stats_init})

    # 录制 graph
    print("\n录制 graph:")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2
        z = y + x

    stats_record = print_memory_stats("录制完成")
    results.append({"stage": "recorded", **stats_record})

    # Replay
    print("\nReplay:")
    for i in range(5):
        g.replay()
        stats = print_memory_stats(f"Replay {i+1}")
        results.append({"iteration": i+1, **stats})

    print()
    return results

def experiment_graph_memory_isolation():
    """图池与普通池隔离"""
    print("=" * 90)
    print("实验 4C：图池与普通池隔离")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []

    # 普通分配
    a = torch.randn(500, 500, device='cuda')
    stats_normal = print_memory_stats("普通分配 a (500x500)")
    results.append({"stage": "normal_alloc", **stats_normal})

    # 录制 graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2

    stats_graph = print_memory_stats("录制 graph (1000x1000)")
    results.append({"stage": "graph_recorded", **stats_graph})

    # 再次普通分配
    b = torch.randn(500, 500, device='cuda')
    stats_normal2 = print_memory_stats("再次普通分配 b (500x500)")
    results.append({"stage": "normal_alloc2", **stats_normal2})

    # Replay graph
    g.replay()
    stats_replay = print_memory_stats("Replay graph")
    results.append({"stage": "replay", **stats_replay})

    # 删除普通分配
    del a, b
    torch.cuda.synchronize()
    stats_del_normal = print_memory_stats("删除 a, b")
    results.append({"stage": "del_normal", **stats_del_normal})

    # 删除 graph
    del g, x, y
    torch.cuda.synchronize()
    stats_del_graph = print_memory_stats("删除 graph")
    results.append({"stage": "del_graph", **stats_del_graph})

    print()
    return results

def experiment_multiple_graphs():
    """多个 graph：各自独立池"""
    print("=" * 90)
    print("实验 4D：多个 CUDA Graph")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []

    # Graph 1
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        x1 = torch.randn(800, 800, device='cuda')
        y1 = x1 * 2

    stats_g1 = print_memory_stats("录制 graph1 (800x800)")
    results.append({"stage": "graph1", **stats_g1})

    # Graph 2
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        x2 = torch.randn(1200, 1200, device='cuda')
        y2 = x2 * 3

    stats_g2 = print_memory_stats("录制 graph2 (1200x1200)")
    results.append({"stage": "graph2", **stats_g2})

    # Replay 两个 graph
    g1.replay()
    stats_r1 = print_memory_stats("Replay graph1")
    results.append({"stage": "replay1", **stats_r1})

    g2.replay()
    stats_r2 = print_memory_stats("Replay graph2")
    results.append({"stage": "replay2", **stats_r2})

    print()
    return results

def experiment_graph_with_dynamic():
    """Graph 不支持动态形状"""
    print("=" * 90)
    print("实验 4E：Graph 限制（动态形状）")
    print("=" * 90)

    results = {}

    try:
        g = torch.cuda.CUDAGraph()
        sizes = [500, 1000]  # 动态

        with torch.cuda.graph(g):
            for size in sizes:
                x = torch.randn(size, size, device='cuda')

        results["error"] = None
        results["success"] = True
        print("✓ 录制成功（但每个 size 使用独立内存）")

    except Exception as e:
        results["error"] = str(e)
        results["success"] = False
        print(f"✗ 录制失败: {e}")

    print()
    return results

def experiment_graph_lifetime():
    """Graph 生命周期"""
    print("=" * 90)
    print("实验 4F：Graph 生命周期")
    print("=" * 90)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    results = []

    stats_init = print_memory_stats("初始状态")
    results.append({"stage": "init", **stats_init})

    # 创建并销毁多个 graph
    for i in range(3):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x = torch.randn(1000, 1000, device='cuda')
            y = x * 2

        stats_create = print_memory_stats(f"创建 graph {i+1}")
        results.append({"stage": f"create_{i+1}", **stats_create})

        # 立即销毁
        del g, x, y
        torch.cuda.synchronize()

        stats_destroy = print_memory_stats(f"销毁 graph {i+1}")
        results.append({"stage": f"destroy_{i+1}", **stats_destroy})

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
        "experiment_4a": experiment_normal_execution(),
        "experiment_4b": experiment_graph_execution(),
        "experiment_4c": experiment_graph_memory_isolation(),
        "experiment_4d": experiment_multiple_graphs(),
        "experiment_4e": experiment_graph_with_dynamic(),
        "experiment_4f": experiment_graph_lifetime(),
    }

    # 保存结果
    with open("memory_cuda_graph.json", "w") as f:
        json.dump(output, f, indent=2)

    print("结果已保存到 memory_cuda_graph.json")
