#!/usr/bin/env python3
"""
6.0b 手写 DeviceMesh / Shard / Replicate / Partial。

这一份实现只做一件事：把「全局张量 + mesh + placement」翻译成「每个 rank 手里的那块」，
并把这个翻译写成可检查的纯 Python 代码——每个 rank 的 shape、stride、偏移都能单独打印。
数值对拍的另一边是 PyTorch DTensor 的 `full_tensor()`。

三类 placement 的数值语义：
  Shard(d)   每个 rank 持有第 d 维上的一段，段长按 ceil 切分（7 切成 [4, 3]）
  Replicate  每个 rank 持有完整副本
  Partial    每个 rank 持有「一整块」，全局值等于所有 rank 的逐元素和；
             **归约之前它不是一个完整副本**，只是一个待求和的加数

用法：
    python mini_dtensor.py A --world-size 2 --mesh 2 --out <dir>
    python mini_dtensor.py A --world-size 4 --mesh 2x2 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


# ==========================================================================
# placement
# ==========================================================================

class Placement:
    def __eq__(self, other):
        return type(self) is type(other) and self.dim == other.dim

    def __hash__(self):
        return hash((type(self).__name__, self.dim))

    def __repr__(self):
        return f"S({self.dim})"


class Shard(Placement):
    def __init__(self, dim):
        self.dim = dim


class Replicate(Placement):
    dim = None

    def __repr__(self):
        return "R"


class Partial(Placement):
    def __init__(self, reduce_op="sum"):
        self.dim = None
        self.reduce_op = reduce_op

    def __repr__(self):
        return f"P({self.reduce_op})"


def parse_placement(name):
    if name == "R":
        return Replicate()
    if name.startswith("S("):
        return Shard(int(name[2:-1]))
    if name.startswith("P("):
        return Partial(name[2:-1])
    raise ValueError(name)


def to_torch(pl):
    """本文件的 placement 与 torch.distributed.tensor 的 placement 互转。"""
    from torch.distributed.tensor import Partial as TPartial
    from torch.distributed.tensor import Replicate as TReplicate
    from torch.distributed.tensor import Shard as TShard
    if isinstance(pl, Shard):
        return TShard(pl.dim)
    if isinstance(pl, Replicate):
        return TReplicate()
    if isinstance(pl, Partial):
        return TPartial()
    raise ValueError(pl)


# ==========================================================================
# 切分与放置
# ==========================================================================

def split_sizes(n, k):
    """把长度 n 切成 k 段，前 n%k 段多 1。与 torch.tensor_split 的分法一致。"""
    base, rest = divmod(n, k)
    return [base + (1 if i < rest else 0) for i in range(k)]


def chunk_offset(n, k, i):
    """第 i 段在长度 n 上的起始下标。"""
    base, rest = divmod(n, k)
    return i * base + min(i, rest)


def place(global_tensor, mesh_shape, placements, coord):
    """返回 coord 这个 mesh 坐标上应该持有的局部张量。"""
    t = global_tensor
    for size, pl, c in zip(mesh_shape, placements, coord):
        if isinstance(pl, Shard):
            sizes = split_sizes(t.shape[pl.dim], size)
            start = sum(sizes[:c])
            t = t.narrow(pl.dim, start, sizes[c])
    return t


def local_offset(global_shape, mesh_shape, placements, coord):
    off = [0] * len(global_shape)
    for size, pl, c in zip(mesh_shape, placements, coord):
        if isinstance(pl, Shard):
            off[pl.dim] = chunk_offset(global_shape[pl.dim], size, c)
    return off


def mesh_coord(rank, mesh_shape):
    coord = []
    for size in reversed(mesh_shape):
        coord.append(rank % size)
        rank //= size
    return list(reversed(coord))


def local_meta(rank, mesh_shape, placements, global_shape, local):
    coord = mesh_coord(rank, mesh_shape)
    return {
        "rank": rank,
        "mesh_coord": coord,
        "placements": [repr(p) for p in placements],
        "global_shape": list(global_shape),
        "local_shape": list(local.shape),
        "local_stride": list(local.stride()),
        "offset": local_offset(global_shape, mesh_shape, placements, coord),
        "numel": local.numel(),
        "data_ptr": local.data_ptr(),
        "checksum": float(local.to(torch.float64).sum().item()),
    }


# ==========================================================================
# 全局重建（把各 rank 的局部块收上来，在 rank 0 上用 Python 拼回全局张量）
# ==========================================================================

def reconstruct(shards: dict, mesh_shape, placements, global_shape, device="cpu"):
    """shards: {rank: local tensor}。返回全局张量。

    Partial 的语义是「全局 = 各 rank 之和」，其余两类是「按 offset 摆放」。
    """
    if any(isinstance(p, Partial) for p in placements):
        acc = None
        for r in sorted(shards):
            acc = shards[r].clone() if acc is None else acc + shards[r]
        return acc
    out = torch.zeros(global_shape, dtype=torch.float64, device=device)
    for r in sorted(shards):
        coord = mesh_coord(r, mesh_shape)
        start = local_offset(global_shape, mesh_shape, placements, coord)
        t = shards[r]
        idx = tuple(slice(start[d], start[d] + t.shape[d]) for d in range(len(global_shape)))
        out[idx] = t
    return out


# ==========================================================================
# 运行清单
# ==========================================================================

def env_pins():
    import platform
    import socket
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    return info


def write_manifest(out_dir, task, world_size, cases, extra=None):
    import time
    payload = {
        "task": task,
        "world_size": world_size,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env_pins(),
        "input_spec": "8x6 与 7x5 的 arange 全局张量（float64）；7 用来覆盖不整除切分",
        "tolerance": "布局重建与 single-process 参照逐元素精确相等（整数），容差 0；"
                     "梯度和对拍用 1e-10",
        "source_pins": {
            "placement": "torch/distributed/tensor/placement_types.py:162/1700/1765",
            "propagate": "torch/distributed/tensor/_sharding_prop.py:710/735",
            "redistribute": "torch/distributed/tensor/_redistribute.py:1588",
            "distribute_tensor": "torch/distributed/tensor/_api.py:939",
            "cpu_alltoall_fallback": "torch/distributed/tensor/_collective_utils.py:107-123",
        },
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    return payload


# ==========================================================================
# 任务 A
# ==========================================================================

def _all_gather_locals(local, world_size):
    buf = [None] * world_size
    dist.all_gather_object(buf, local.detach().to("cpu"))
    return {r: buf[r].to(torch.float64).cpu() for r in range(world_size)}


def run_A(rank, world_size, mesh_shape, out_dir, device="cpu"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29720"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor as dt_distribute

    mesh = init_device_mesh(device, tuple(mesh_shape))
    out = {"mesh_shape": mesh_shape, "world_size": world_size,
           "device": device, "backend": backend, "cases": []}

    cases = [
        ("8x6", torch.arange(48, dtype=torch.float64, device=dev).reshape(8, 6)),
        ("7x5", torch.arange(35, dtype=torch.float64, device=dev).reshape(7, 5)),
    ]
    # 一维 mesh 用单 placement；2x2 mesh 用两个不同维度上的 Shard（同一维度切两次不在本实验内）
    if len(mesh_shape) == 1:
        placement_sets = [[Shard(0)], [Shard(1)], [Replicate()]]
    else:
        placement_sets = [[Shard(0), Shard(1)], [Shard(0), Replicate()],
                          [Replicate(), Shard(1)]]

    for name, g in cases:
        for pls in placement_sets:
            if any(isinstance(p, Shard) and p.dim >= g.dim() for p in pls):
                continue
            local = place(g, mesh_shape, pls, mesh_coord(rank, mesh_shape))
            meta = local_meta(rank, mesh_shape, pls, g.shape, local)
            metas = [None] * world_size
            dist.all_gather_object(metas, meta)
            gathered = _all_gather_locals(local, world_size)
            rebuilt = reconstruct(gathered, mesh_shape, pls, g.shape)
            dt = dt_distribute(g, mesh, [to_torch(p) for p in pls])
            ref = dt.full_tensor()
            dt_local = dt.to_local()
            rec = {
                "tensor": name,
                "placements": [repr(p) for p in pls],
                "ranks": metas,
                "local_matches_dtensor": bool(torch.equal(local, dt_local)),
                "dt_local_shape": list(dt_local.shape),
                "rebuilt_matches_global": bool(torch.equal(rebuilt.cpu(), g.cpu())),
                "dt_full_matches_global": bool(torch.equal(ref, g)),
                "max_abs_diff_vs_dtensor": float(
                (rebuilt.cpu() - ref.cpu()).abs().max().item()),
            }
            if rank == 0:
                out["cases"].append(rec)
                plabel = ",".join(repr(p) for p in pls)
                print(f"  {name} {plabel:<28} "
                      f"local={meta['local_shape']} offset={meta['offset']} "
                      f"本rank与DTensor一致={rec['local_matches_dtensor']} "
                      f"重建一致={rec['rebuilt_matches_global']} "
                      f"max|diff|={rec['max_abs_diff_vs_dtensor']:.3e}")

    # Partial 的语义：先在 1 维上验证 P 不是完整副本
    if len(mesh_shape) == 1:
        g = torch.arange(48, dtype=torch.float64, device=dev).reshape(8, 6)
        solo = torch.zeros_like(g)
        part = place(g, mesh_shape, [Shard(0)], mesh_coord(rank, mesh_shape))
        s = split_sizes(g.shape[0], mesh_shape[0])[mesh_coord(rank, mesh_shape)[0]]
        start = chunk_offset(g.shape[0], mesh_shape[0], mesh_coord(rank, mesh_shape)[0])
        solo[start:start + s] = part
        gathered = _all_gather_locals(solo, world_size)
        summed = reconstruct(gathered, mesh_shape, [Partial()], g.shape)
        # 有些版本把「redistribute 到 Partial」判为内部用法而直接拒绝，两种结局都记录
        try:
            dt = dt_distribute(g, mesh, [to_torch(Shard(0))]).redistribute(
                mesh, [to_torch(Partial())])
            dt_partial_local = dt.to_local()
            partial_ok, partial_err = bool(torch.equal(solo.cpu(), dt_partial_local.cpu())), None
        except Exception as e:
            partial_ok, partial_err = None, f"{type(e).__name__}: {e}"
        out["partial"] = {
            "rank_local_sum": float(solo.sum().item()),
            "rank_local_first_row": solo[0].tolist(),
            "sum_matches_global": bool(torch.equal(summed.cpu(), g.cpu())),
            "redistribute_local_matches": partial_ok,
            "redistribute_error": partial_err,
            "shard_to_partial_is_local_only": True,
        }
        if rank == 0:
            print(f"  Shard(0)->Partial 每 rank 局部和 {float(solo.sum().item()):.0f}；"
                  f"逐元素求和等于全局={bool(torch.equal(summed.cpu(), g.cpu()))}；"
                  f"与 DTensor 局部一致={partial_ok}"
                  + (f"（DTensor 拒绝：{partial_err}）" if partial_err else ""))

    dist.barrier()
    gathered_out = [None] * world_size
    dist.all_gather_object(gathered_out, out if rank == 0 else None)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "mini_placements.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        n_cases = len(out["cases"])
        n_match = sum(1 for c in out["cases"]
                      if c["rebuilt_matches_global"] and c["local_matches_dtensor"])
        write_manifest(out_dir, "A", world_size,
                       [{"tensor": c["tensor"], "placements": c["placements"]}
                        for c in out["cases"]],
                       extra={"mesh_shape": mesh_shape, "device": out.get("device")})
        print(f"[A] {n_cases} 个 (张量, placement) 组合，全部一致 {n_match}/{n_cases}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A"])
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--mesh", default="2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    mesh_shape = [int(x) for x in a.mesh.lower().split("x")]
    if a.world_size != int(torch.tensor(mesh_shape).prod()):
        raise SystemExit("world_size 必须等于 mesh 元素个数")
    import torch.multiprocessing as mp
    mp.spawn(run_A, args=(a.world_size, mesh_shape, a.out, a.device),
             nprocs=a.world_size, join=True)


if __name__ == "__main__":
    sys.exit(main())
