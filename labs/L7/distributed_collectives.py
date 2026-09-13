#!/usr/bin/env python3
"""
7.2 分布式通信原语实验

实验：
1. all-reduce/all-gather/reduce-scatter 通信量与延迟
2. ring/tree 算法通信模式
3. NCCL 拓扑检测与算法选择
4. 单机多卡与跨机通信对比

运行：
  # 单机 4 卡
  torchrun --nproc_per_node=4 distributed_collectives.py --exp=collectives

  # 跨机（两台机器）
  # 机器 0:
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<master_ip> --master_port=29500 \
    distributed_collectives.py --exp=cross_node
  # 机器 1:
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<master_ip> --master_port=29500 \
    distributed_collectives.py --exp=cross_node
"""

import os
import json
import time
import argparse
import torch
import torch.distributed as dist


def init_process_group():
    """初始化分布式进程组"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def benchmark_collective(op_name, tensor_mb, warmup=10, repeat=100):
    """
    测量集合通信延迟

    Args:
        op_name: "all_reduce", "all_gather", "reduce_scatter"
        tensor_mb: 张量大小（MB）
        warmup: 预热次数
        repeat: 测量次数

    Returns:
        {"op": str, "tensor_mb": float, "latency_ms": float, "bandwidth_gbps": float}
    """
    rank, world_size, local_rank = (
        dist.get_rank(),
        dist.get_world_size(),
        torch.cuda.current_device()
    )

    # 根据操作分配张量
    numel = int(tensor_mb * 1024 * 1024 / 4)  # FP32 = 4 bytes

    if op_name == "all_reduce":
        # all-reduce: 每个进程一个张量
        tensor = torch.randn(numel, device=f"cuda:{local_rank}")
    elif op_name == "all_gather":
        # all-gather: 输入张量 + 输出列表
        tensor = torch.randn(numel, device=f"cuda:{local_rank}")
        output_tensors = [
            torch.empty_like(tensor) for _ in range(world_size)
        ]
    elif op_name == "reduce_scatter":
        # reduce-scatter: 输入列表 + 输出张量
        input_tensors = [
            torch.randn(numel, device=f"cuda:{local_rank}")
            for _ in range(world_size)
        ]
        output_tensor = torch.empty(numel, device=f"cuda:{local_rank}")
    else:
        raise ValueError(f"Unknown op: {op_name}")

    # 预热
    for _ in range(warmup):
        if op_name == "all_reduce":
            dist.all_reduce(tensor)
        elif op_name == "all_gather":
            dist.all_gather(output_tensors, tensor)
        elif op_name == "reduce_scatter":
            dist.reduce_scatter(output_tensor, input_tensors)

    torch.cuda.synchronize()
    dist.barrier()

    # 测量
    start = time.perf_counter()
    for _ in range(repeat):
        if op_name == "all_reduce":
            dist.all_reduce(tensor)
        elif op_name == "all_gather":
            dist.all_gather(output_tensors, tensor)
        elif op_name == "reduce_scatter":
            dist.reduce_scatter(output_tensor, input_tensors)

    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    latency_ms = (end - start) / repeat * 1000

    # 计算带宽
    if op_name == "all_reduce":
        # all-reduce 理论通信量：2 * (N-1)/N * S
        # 实际 ring 算法：2 * (N-1)/N * S
        bytes_transferred = 2 * (world_size - 1) / world_size * tensor_mb * 1024 * 1024
    elif op_name == "all_gather":
        # all-gather: (N-1)/N * S
        bytes_transferred = (world_size - 1) / world_size * tensor_mb * 1024 * 1024
    elif op_name == "reduce_scatter":
        # reduce-scatter: (N-1)/N * S
        bytes_transferred = (world_size - 1) / world_size * tensor_mb * 1024 * 1024

    bandwidth_gbps = bytes_transferred / (latency_ms / 1000) / 1e9

    return {
        "op": op_name,
        "tensor_mb": tensor_mb,
        "world_size": world_size,
        "latency_ms": round(latency_ms, 3),
        "bandwidth_gbps": round(bandwidth_gbps, 2),
    }


def exp1_collectives():
    """实验 1：集合通信原语对比"""
    rank, world_size, _ = init_process_group()

    results = []

    # 测试不同大小的张量
    tensor_sizes = [1, 4, 16, 64, 256]  # MB
    ops = ["all_reduce", "all_gather", "reduce_scatter"]

    for tensor_mb in tensor_sizes:
        for op_name in ops:
            if rank == 0:
                print(f"Testing {op_name} with {tensor_mb} MB...")

            result = benchmark_collective(op_name, tensor_mb)
            results.append(result)

    # Rank 0 保存结果
    if rank == 0:
        output_dir = "results/worldvln/7.2"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/collectives.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n实验 1 完成，结果保存至 {output_dir}/collectives.json")
        print(f"world_size={world_size}, 测试了 {len(tensor_sizes)} 种张量大小 × {len(ops)} 种操作")

    dist.destroy_process_group()


def exp2_ring_algorithm():
    """实验 2：ring all-reduce 通信模式"""
    rank, world_size, local_rank = init_process_group()

    # 256 MB 张量
    numel = int(256 * 1024 * 1024 / 4)
    tensor = torch.randn(numel, device=f"cuda:{local_rank}")

    # 预热
    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    # 测量 100 次
    start = time.perf_counter()
    for _ in range(100):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    latency_ms = (end - start) / 100 * 1000

    # Ring all-reduce 理论通信量
    # 分为 reduce-scatter 和 all-gather 两阶段
    # 每阶段: (N-1)/N * S
    # 总计: 2 * (N-1)/N * S
    theoretical_bytes = 2 * (world_size - 1) / world_size * 256 * 1024 * 1024
    bandwidth_gbps = theoretical_bytes / (latency_ms / 1000) / 1e9

    result = {
        "algorithm": "ring_all_reduce",
        "world_size": world_size,
        "tensor_mb": 256,
        "latency_ms": round(latency_ms, 3),
        "theoretical_bytes_mb": round(theoretical_bytes / 1024 / 1024, 2),
        "bandwidth_gbps": round(bandwidth_gbps, 2),
        "note": "2 phases: reduce-scatter + all-gather"
    }

    if rank == 0:
        output_dir = "results/worldvln/7.2"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/ring_algorithm.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n实验 2 完成，结果保存至 {output_dir}/ring_algorithm.json")
        print(f"Ring all-reduce: {latency_ms:.3f} ms, {bandwidth_gbps:.2f} GB/s")

    dist.destroy_process_group()


def exp3_topology_detection():
    """实验 3：NCCL 拓扑检测"""
    rank, world_size, local_rank = init_process_group()

    # 获取当前设备信息
    device_name = torch.cuda.get_device_name(local_rank)
    device_capability = torch.cuda.get_device_capability(local_rank)

    # P2P 连接矩阵
    p2p_matrix = []
    for i in range(world_size):
        if i == rank:
            # 自己到自己：总是可访问
            p2p_matrix.append(1)
        else:
            # 检查是否能 P2P 访问
            can_access = torch.cuda.can_device_access_peer(local_rank, i % torch.cuda.device_count())
            p2p_matrix.append(1 if can_access else 0)

    # 收集所有 rank 的信息
    all_info = [None] * world_size
    info = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": device_name,
        "device_capability": f"{device_capability[0]}.{device_capability[1]}",
        "p2p_accessible": p2p_matrix,
    }

    dist.barrier()
    all_info = [None] * world_size
    dist.all_gather_object(all_info, info)

    if rank == 0:
        output_dir = "results/worldvln/7.2"
        os.makedirs(output_dir, exist_ok=True)

        # 构建 P2P 连接矩阵
        full_p2p_matrix = []
        for i in range(world_size):
            full_p2p_matrix.append(all_info[i]["p2p_accessible"])

        result = {
            "world_size": world_size,
            "devices": [
                {
                    "rank": info["rank"],
                    "local_rank": info["local_rank"],
                    "device_name": info["device_name"],
                    "capability": info["device_capability"],
                }
                for info in all_info
            ],
            "p2p_matrix": full_p2p_matrix,
            "p2p_note": "1 = can access, 0 = cannot access"
        }

        with open(f"{output_dir}/topology.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n实验 3 完成，结果保存至 {output_dir}/topology.json")
        print(f"检测到 {world_size} 个设备")
        print("P2P 连接矩阵:")
        for i, row in enumerate(full_p2p_matrix):
            print(f"  Rank {i}: {row}")

    dist.destroy_process_group()


def exp4_cross_node():
    """实验 4：跨机通信测量"""
    rank, world_size, local_rank = init_process_group()

    # 获取节点信息
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    node_rank = rank // local_world_size

    results = []

    # 测试不同大小
    tensor_sizes = [1, 16, 256]  # MB

    for tensor_mb in tensor_sizes:
        if rank == 0:
            print(f"Testing cross-node with {tensor_mb} MB...")

        result = benchmark_collective("all_reduce", tensor_mb)
        result["node_rank"] = node_rank
        result["local_world_size"] = local_world_size
        results.append(result)

    if rank == 0:
        output_dir = "results/worldvln/7.2"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/cross_node.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n实验 4 完成，结果保存至 {output_dir}/cross_node.json")
        print(f"world_size={world_size}, local_world_size={local_world_size}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["collectives", "ring", "topology", "cross_node"], required=True)
    args = parser.parse_args()

    if args.exp == "collectives":
        exp1_collectives()
    elif args.exp == "ring":
        exp2_ring_algorithm()
    elif args.exp == "topology":
        exp3_topology_detection()
    elif args.exp == "cross_node":
        exp4_cross_node()


if __name__ == "__main__":
    main()
