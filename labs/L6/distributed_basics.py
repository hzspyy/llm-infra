"""
6.0 分布式通信基础实验

演示：
1. 进程组初始化、rank 和 world_size
2. 集合通信的匹配次序要求
3. 异步通信与 buffer 生命周期
4. 常见错误：次序不匹配、提前退出、buffer 重用

使用 Gloo（CPU）和 NCCL（GPU）两种后端对比语义
"""

import os
import sys
import json
import time
import torch
import torch.distributed as dist
from datetime import datetime


def init_process(rank, world_size, backend='gloo'):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialized with backend={backend}, world_size={world_size}")


def experiment_1_basic_collectives(rank, world_size, device='cpu'):
    """实验 1：基础集合通信 - all_reduce, broadcast, gather"""
    print(f"\n[Rank {rank}] === Experiment 1: Basic Collectives ===")

    # 每个 rank 持有不同的张量
    tensor = torch.tensor([rank + 1.0], device=device)
    print(f"[Rank {rank}] Initial tensor: {tensor.item()}")

    # all_reduce: 求和
    tensor_sum = tensor.clone()
    dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After all_reduce(SUM): {tensor_sum.item()}")

    # broadcast: rank 0 广播
    tensor_bcast = torch.tensor([100.0 + rank], device=device)
    dist.broadcast(tensor_bcast, src=0)
    print(f"[Rank {rank}] After broadcast from rank 0: {tensor_bcast.item()}")

    # gather: 收集到 rank 0
    if rank == 0:
        gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.gather(tensor, gather_list, dst=0)
        print(f"[Rank {rank}] Gathered tensors: {[t.item() for t in gather_list]}")
    else:
        dist.gather(tensor, dst=0)

    return {
        'rank': rank,
        'initial': tensor.item(),
        'after_sum': tensor_sum.item(),
        'after_broadcast': tensor_bcast.item()
    }


def experiment_2_order_matching(rank, world_size, device='cpu'):
    """实验 2：通信次序匹配 - 正确和错误的例子"""
    print(f"\n[Rank {rank}] === Experiment 2: Order Matching ===")

    # 正确：所有 rank 调用相同的 collective 序列
    t1 = torch.tensor([rank * 10.0], device=device)
    dist.all_reduce(t1, op=dist.ReduceOp.SUM)

    t2 = torch.tensor([rank * 20.0], device=device)
    dist.all_reduce(t2, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] Correct order - t1: {t1.item()}, t2: {t2.item()}")

    # 演示错误：不同 rank 调用不同次序（注释掉以避免挂起）
    # if rank == 0:
    #     dist.all_reduce(t1)
    #     dist.broadcast(t2, src=0)
    # else:
    #     dist.broadcast(t2, src=0)  # 错误：rank 0 先 reduce，其他 rank 先 broadcast
    #     dist.all_reduce(t1)

    return {'rank': rank, 't1_sum': t1.item(), 't2_sum': t2.item()}


def experiment_3_async_semantics(rank, world_size, device):
    """实验 3：异步语义 - CPU vs GPU"""
    print(f"\n[Rank {rank}] === Experiment 3: Async Semantics on {device} ===")

    tensor = torch.tensor([rank + 1.0], device=device)

    # 记录通信前后的时间
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    sync_time = (end - start) * 1000  # ms

    # 异步调用
    tensor_async = torch.tensor([rank + 1.0], device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start_async = time.perf_counter()
    work = dist.all_reduce(tensor_async, op=dist.ReduceOp.SUM, async_op=True)
    end_async_return = time.perf_counter()

    # 等待完成
    work.wait()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end_async_complete = time.perf_counter()

    return_time = (end_async_return - start_async) * 1000
    complete_time = (end_async_complete - start_async) * 1000

    print(f"[Rank {rank}] Sync call: {sync_time:.3f} ms")
    print(f"[Rank {rank}] Async return: {return_time:.3f} ms, complete: {complete_time:.3f} ms")

    return {
        'rank': rank,
        'device': str(device),
        'sync_time_ms': sync_time,
        'async_return_ms': return_time,
        'async_complete_ms': complete_time,
        'result': tensor_async.item()
    }


def experiment_4_buffer_lifecycle(rank, world_size, device):
    """实验 4：buffer 生命周期 - 演示正确和错误用法"""
    print(f"\n[Rank {rank}] === Experiment 4: Buffer Lifecycle ===")

    # 正确：等待通信完成再重用 buffer
    buffer = torch.zeros(10, device=device)
    buffer[rank] = rank + 1.0

    work = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
    work.wait()  # 等待完成

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    result_correct = buffer.clone()
    print(f"[Rank {rank}] Correct usage - sum: {buffer.sum().item()}")

    # 错误演示（注释掉以避免竞态）：通信未完成就修改 buffer
    # buffer2 = torch.zeros(10, device=device)
    # buffer2[rank] = rank + 1.0
    # work2 = dist.all_reduce(buffer2, async_op=True)
    # buffer2.fill_(999)  # 错误：通信还在进行，修改了 buffer
    # work2.wait()
    # result_wrong = buffer2.clone()

    return {
        'rank': rank,
        'correct_sum': result_correct.sum().item(),
        'buffer_size': buffer.numel()
    }


def experiment_5_timeout_detection(rank, world_size, device='cpu'):
    """实验 5：超时检测 - 某个 rank 不参与通信"""
    print(f"\n[Rank {rank}] === Experiment 5: Timeout Detection ===")

    # 正常通信
    tensor = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Normal communication succeeded: {tensor.item()}")

    # 演示超时（注释掉以避免挂起）：
    # if rank == 0:
    #     print(f"[Rank {rank}] Skipping collective - will cause timeout")
    #     time.sleep(65)  # 等待其他 rank 超时
    # else:
    #     try:
    #         dist.all_reduce(tensor)
    #     except RuntimeError as e:
    #         print(f"[Rank {rank}] Caught timeout: {e}")

    return {'rank': rank, 'result': tensor.item()}


def run_cpu_experiments(rank, world_size):
    """运行 CPU (Gloo) 实验"""
    init_process(rank, world_size, backend='gloo')

    device = torch.device('cpu')
    results = {}
    results['exp1'] = experiment_1_basic_collectives(rank, world_size, device)
    results['exp2'] = experiment_2_order_matching(rank, world_size, device)
    results['exp3'] = experiment_3_async_semantics(rank, world_size, device)
    results['exp4'] = experiment_4_buffer_lifecycle(rank, world_size, device)
    results['exp5'] = experiment_5_timeout_detection(rank, world_size, device)

    # Rank 0 收集结果
    if rank == 0:
        all_results = [None] * world_size
        all_results[0] = results

        for r in range(1, world_size):
            # 简化：只记录 rank 0 的结果
            pass

        output_file = 'results/local/6.0/cpu_gloo_results.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Rank {rank}] Results saved to {output_file}")

    dist.barrier()
    dist.destroy_process_group()


def run_gpu_experiments(rank, world_size):
    """运行 GPU (NCCL) 实验"""
    init_process(rank, world_size, backend='nccl')

    # 每个 rank 使用对应的 GPU
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    results = {}
    results['exp1'] = experiment_1_basic_collectives(rank, world_size, device)
    results['exp2'] = experiment_2_order_matching(rank, world_size, device)
    results['exp3'] = experiment_3_async_semantics(rank, world_size, device)
    results['exp4'] = experiment_4_buffer_lifecycle(rank, world_size, device)
    results['exp5'] = experiment_5_timeout_detection(rank, world_size, device)

    if rank == 0:
        # 检测是否在远程机器
        if os.path.exists('/root/learn'):
            output_file = f'/root/learn/work/out/6.0/gpu_nccl_{world_size}ranks.json'
        else:
            output_file = f'results/worldvln/6.0/gpu_nccl_{world_size}ranks.json'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Rank {rank}] Results saved to {output_file}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp

    if len(sys.argv) < 2:
        print("Usage: python distributed_basics.py [cpu|gpu] [world_size]")
        sys.exit(1)

    mode = sys.argv[1]
    world_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if mode == 'cpu':
        mp.spawn(run_cpu_experiments, args=(world_size,), nprocs=world_size, join=True)
    elif mode == 'gpu':
        if not torch.cuda.is_available():
            print("CUDA not available")
            sys.exit(1)
        if torch.cuda.device_count() < world_size:
            print(f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
            sys.exit(1)
        mp.spawn(run_gpu_experiments, args=(world_size,), nprocs=world_size, join=True)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
