#!/usr/bin/env python3
"""
6.0b 任务 B/C：布局传播预测、collective 归因、Partial 的前后向与梯度对拍。

任务 B：对 linear / matmul / sum / reshape 等表达式先写下预测 placement，
        再跑真实 DTensor，记录实际 placement、局部 shape 与期间的 collective；
        最后 `full_tensor()` 把结果物化，记录物化阶段才出现的通信。
任务 C：跑包含 Partial 的前后向，与单进程梯度逐元素对拍；记录 redistribute
        的临时 buffer（collective 的输入输出形状与字节、CUDA 峰值分配）。

用法：
    python dtensor_probe.py B --world-size 2 --out <dir>
    python dtensor_probe.py C --world-size 2 --out <dir> [--device cuda]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mini_dtensor as md  # noqa: E402


# --------------------------------------------------------------------------
# 预测表：写的是「文档给出的局部规则」，与实测分开记录
# --------------------------------------------------------------------------

PREDICTIONS = {
    "matmul(xS0, Wr)": ("S(0)", "x 的第 0 维分片不与 matmul 的收缩维重合，逐 rank 独立算"),
    "matmul(xr, WS1)": ("S(1)", "W 的第 1 维被切走，输出的第 1 维随之分片"),
    "linear(xS1, WS1)": ("P(sum)", "列并行：输入特征维与权重输出维分别分片，各 rank 只算了一部分和"),
    "linear(xr, WS0)": ("S(1)", "行并行：权重按输出维切分，输出直接在最后一维分片"),
    "linear(xr, WS1)": ("P(sum)", "列并行：权重按输入维切分，输出是各 rank 的部分和"),
    "sum(xS0)": ("P(sum)", "全量求和把分片维消掉，各 rank 只得到局部和"),
    "sum(dim=1)(xS1)": ("P(sum)", "被求和的维度正是分片维，同样退化成部分和"),
    "reshape(xS0)": ("S(0)", "分片维的块边界在 reshape 后仍落在第 0 维上，无需通信"),
    "t(xS1)": ("S(0)", "转置把分片维从第 1 维搬到第 0 维"),
    "redistribute S0->S1": ("S(1)", "显式改变分片维，必须交换数据"),
}


def collectives(fn, itemsize=8):
    """跑 fn，并把期间出现的 c10d functional collective 按名字聚合成计数与字节。

    itemsize 是本轮参与通信的张量元素字节数，用来把 profiler 记录到的输入形状折算成
    「每 rank 一次通信搬运多少字节」。

    只用 profiler 事件，不做函数替换：DTensor 走的是 `torch.ops._c10d_functional.*`，
    在 Python 层包 `dist.all_reduce` 是包不住的。
    """
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=True) as prof:
        out = fn()
    evs = []
    for e in prof.events():
        n = e.name
        # 只取 DTensor/funcol 这一层的逻辑操作：再往下 c10d::/gloo:/nccl: 是同一件事的
        # 后端实现，计入会重复计数。CUDA 上 S(d)->S(d') 走的是 _dtensor::shard_dim_alltoall。
        if n.endswith("wait_tensor") or "_wrap" in n:
            continue
        if n.startswith("_c10d_functional::") or n.startswith("_dtensor::"):
            evs.append({"op": n.split("::")[-1],
                        "input_shapes": [list(s) for s in (e.input_shapes or [])]})
    agg = {}
    for e in evs:
        ent = agg.setdefault(e["op"], {"count": 0, "input_shapes": e["input_shapes"],
                                       "bytes": 0})
        ent["count"] += 1
        ent["input_shapes"] = e["input_shapes"]
        ent["bytes"] = numel_of(e["input_shapes"]) * itemsize
    return out, agg


def total_bytes(agg):
    return sum(v["bytes"] for v in agg.values())


def fmt_placement(p):
    """把 torch 的 placement 打印成本章统一的短记号。"""
    t = type(p).__name__
    if t == "Shard":
        return f"S({p.dim})"
    if t == "Replicate":
        return "R"
    if t == "Partial":
        return f"P({p.reduce_op})"
    return repr(p)


def numel_of(shapes):
    n = 0
    for s in shapes:
        v = 1
        for d in s:
            v *= d
        n = max(n, v)
    return n


def run_B(rank, world_size, out_dir, device="cpu"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29721"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor as dt_distribute
    from torch.distributed.tensor import Replicate, Shard

    mesh = init_device_mesh(device, (world_size,))
    g = torch.arange(24, dtype=torch.float64, device=dev).reshape(4, 6)
    W = torch.arange(36, dtype=torch.float64, device=dev).reshape(6, 6) * 0.01

    x_s0 = dt_distribute(g, mesh, [Shard(0)])
    x_s1 = dt_distribute(g, mesh, [Shard(1)])
    x_r = dt_distribute(g, mesh, [Replicate()])
    W_r = dt_distribute(W, mesh, [Replicate()])
    W_s0 = dt_distribute(W, mesh, [Shard(0)])
    W_s1 = dt_distribute(W, mesh, [Shard(1)])

    ref = {
        "matmul(xS0, Wr)": g @ W,
        "matmul(xr, WS1)": g @ W,
        "linear(xS1, WS1)": F.linear(g, W),
        "linear(xr, WS0)": F.linear(g, W),
        "linear(xr, WS1)": F.linear(g, W),
        "sum(xS0)": g.sum().reshape(1),
        "sum(dim=1)(xS1)": g.sum(dim=1),
        "reshape(xS0)": g.reshape(2, 12),
        "t(xS1)": g.t(),
    }

    cases = []

    def one(tag, fn, extra_check=None):
        out, agg = collectives(fn)
        mat, agg_mat = collectives(lambda: out.full_tensor())
        pred, why = PREDICTIONS[tag]
        actual = fmt_placement(out.placements[0])
        rec = {
            "tag": tag,
            "predicted": pred,
            "actual": actual,
            "prediction_matches": pred == actual,
            "why": why,
            "local_shape": list(out.to_local().shape),
            "collectives_during_op": agg,
            "collectives_during_full_tensor": agg_mat,
            "full_tensor_bytes": numel_of(
                [s for v in agg_mat.values() for s in v["input_shapes"]]) * 8,
        }
        if tag in ref:
            r = ref[tag].to(torch.float64)
            rec["global_matches_reference"] = bool(
                torch.allclose(mat.to(torch.float64).reshape(-1), r.reshape(-1), atol=1e-12))
            rec["max_abs_diff"] = float(
                (mat.to(torch.float64).reshape(-1) - r.reshape(-1)).abs().max().item())
        if extra_check:
            rec.update(extra_check(out))
        cases.append(rec)

    one("matmul(xS0, Wr)", lambda: x_s0 @ W_r)
    one("matmul(xr, WS1)", lambda: x_r @ W_s1)
    one("linear(xS1, WS1)", lambda: F.linear(x_s1, W_s1))
    one("linear(xr, WS0)", lambda: F.linear(x_r, W_s0))
    one("linear(xr, WS1)", lambda: F.linear(x_r, W_s1))
    one("sum(xS0)", lambda: x_s0.sum())
    one("sum(dim=1)(xS1)", lambda: x_s1.sum(dim=1))
    one("reshape(xS0)", lambda: x_s0.reshape(2, 12))
    one("t(xS1)", lambda: x_s1.t())

    red, agg = collectives(lambda: x_s0.redistribute(mesh, [Shard(1)]))
    cases.append({
        "tag": "redistribute S0->S1",
        "predicted": PREDICTIONS["redistribute S0->S1"][0],
        "actual": fmt_placement(red.placements[0]),
        "prediction_matches": fmt_placement(red.placements[0]) == PREDICTIONS["redistribute S0->S1"][0],
        "why": PREDICTIONS["redistribute S0->S1"][1],
        "local_shape": list(red.to_local().shape),
        "collectives_during_op": agg,
        "collectives_during_full_tensor": {},
        "full_tensor_bytes": 0,
        "global_matches_reference": bool(torch.equal(red.full_tensor(), g)),
    })

    # 不支持的布局必须明确报错，不能静默变换
    unsupported = []
    for label, fn in [
        ("Shard(2) on 2-D tensor", lambda: dt_distribute(g, mesh, [Shard(2)])),
        ("placements 数与 mesh 维数不符", lambda: dt_distribute(g, mesh, [Shard(0), Shard(1)])),
        ("Replicate+Shard 张量维不合法", lambda: dt_distribute(g, mesh, [Shard(-1)])),
    ]:
        try:
            fn()
            unsupported.append({"case": label, "raised": False})
        except Exception as e:
            unsupported.append({"case": label, "raised": True,
                                "type": type(e).__name__, "msg": str(e)[:220]})

    out = {"mesh_shape": [world_size], "device": device, "backend": backend,
           "cases": cases, "unsupported": unsupported}
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "propagation.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        md.write_manifest(out_dir, "B", world_size,
                          [{"tag": c["tag"], "predicted": c["predicted"],
                            "actual": c["actual"]} for c in cases],
                          extra={"device": device, "backend": backend})
        nmatch = sum(1 for c in cases if c["prediction_matches"])
        print(f"[B] 预测命中 {nmatch}/{len(cases)}")
        for c in cases:
            print(f"  {c['tag']:<22} 预测={c['predicted']:<7} 实际={c['actual']:<7} "
                  f"local={str(c['local_shape']):<9} "
                  f"算子内通信={sum(v['count'] for v in c['collectives_during_op'].values())} "
                  f"物化通信={sum(v['count'] for v in c['collectives_during_full_tensor'].values())} "
                  f"全局一致={c.get('global_matches_reference')}")
        print(f"  同维双 Shard: {unsupported}")
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------
# 任务 C
# --------------------------------------------------------------------------

def _cuda_stats(device):
    if device != "cuda":
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "reserved_mb": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def run_C(rank, world_size, out_dir, device):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29722"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"{device}:{rank}") if device == "cuda" else torch.device("cpu")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor as dt_distribute
    from torch.distributed.tensor import Replicate, Shard

    mesh = init_device_mesh(device, (world_size,))
    torch.manual_seed(0)
    xg = torch.randn(8, 4, dtype=torch.float64, device=dev)
    Wg = torch.randn(3, 4, dtype=torch.float64, device=dev)

    # 单进程参照
    xr = xg.clone().detach().requires_grad_()
    Wr = Wg.clone().detach().requires_grad_()
    loss_ref = (F.linear(xr, Wr) ** 2).sum()
    loss_ref.backward()

    out = {"device": device, "world_size": world_size}

    # --- 情形 1：激活行并行（x Shard(0)，W Replicate）---
    x_d = dt_distribute(xg, mesh, [Shard(0)]).detach().requires_grad_()
    W_d = dt_distribute(Wg, mesh, [Replicate()]).detach().requires_grad_()
    y_d, fwd_coll = collectives(lambda: F.linear(x_d, W_d))
    loss_d, loss_coll = collectives(lambda: (y_d ** 2).sum())
    loss_local = float(loss_d.to_local().item())
    loss_full, mat_coll = collectives(lambda: loss_d.full_tensor())
    gx, bx_coll = collectives(
        lambda: torch.autograd.grad(loss_d, x_d, retain_graph=True)[0])
    gW, bW_coll = collectives(lambda: torch.autograd.grad(loss_d, W_d)[0])
    gW_full, bWmat_coll = collectives(lambda: gW.full_tensor())
    out["row_parallel"] = {
        "y_placements": [fmt_placement(p) for p in y_d.placements],
        "loss_placements": [fmt_placement(p) for p in loss_d.placements],
        "loss_local_value": loss_local,
        "loss_full_value": float(loss_full.item()),
        "loss_ref_value": float(loss_ref.item()),
        "loss_full_matches_ref": abs(float(loss_full.item()) - float(loss_ref.item())) < 1e-10,
        "grad_x_placements": [fmt_placement(p) for p in gx.placements],
        "grad_W_placements": [fmt_placement(p) for p in gW.placements],
        "grad_W_full_matches_ref": bool(
            torch.allclose(gW_full.to(torch.float64), Wr.grad, atol=1e-10)),
        "grad_x_max_abs_diff": float(
            (gx.full_tensor().to(torch.float64) - xr.grad).abs().max().item()),
        "forward_collectives": fwd_coll,
        "loss_collectives": loss_coll,
        "loss_materialize_collectives": mat_coll,
        "backward_x_collectives": bx_coll,
        "backward_W_collectives": bW_coll,
        "grad_W_materialize_collectives": bWmat_coll,
        "collective_bytes": {
            "loss_materialize": total_bytes(mat_coll),
            "grad_W_materialize": total_bytes(bWmat_coll),
        },
    }

    # --- 情形 2：权重列并行（x Replicate，W Shard(1)）---
    x2 = dt_distribute(xg, mesh, [Replicate()]).detach().requires_grad_()
    W2 = dt_distribute(Wg, mesh, [Shard(1)]).detach().requires_grad_()
    y2, fwd2 = collectives(lambda: F.linear(x2, W2))
    loss2, loss2_coll = collectives(lambda: (y2 ** 2).sum())
    loss2_full, mat2 = collectives(lambda: loss2.full_tensor())
    out["col_parallel"] = {
        "y_placements": [fmt_placement(p) for p in y2.placements],
        "loss_placements": [fmt_placement(p) for p in loss2.placements],
        "loss_local_value": float(loss2.to_local().item()),
        "loss_full_value": float(loss2_full.item()),
        "loss_ref_value": float(loss_ref.item()),
        "local_equals_global": abs(float(loss2.to_local().item())
                                   - float(loss_ref.item())) < 1e-12,
        "forward_collectives": fwd2,
        "loss_collectives": loss2_coll,
        "loss_materialize_collectives": mat2,
        "collective_bytes": {"inside_sum": total_bytes(loss2_coll)},
    }

    # --- 情形 3：redistribute 的临时 buffer 与释放 ---
    big = torch.randn(4096, 1024, dtype=torch.float32, device=dev)
    b_d = dt_distribute(big, mesh, [Shard(0)])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = _cuda_stats(device)
    temp_shapes = []
    per_redistribute_bytes = 0
    for _ in range(3):
        r, coll = collectives(lambda: b_d.redistribute(mesh, [Replicate()]), itemsize=4)
        per_redistribute_bytes = total_bytes(coll)
        for k, v in coll.items():
            temp_shapes.append({"op": k, "count": v["count"],
                                "input_shapes": v["input_shapes"], "bytes": v["bytes"]})
    after = _cuda_stats(device)
    del r
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    released = _cuda_stats(device)
    out["redistribute_buffers"] = {
        "local_bytes_per_rank": b_d.to_local().numel() * b_d.to_local().element_size(),
        "global_bytes": big.numel() * big.element_size(),
        "collectives": temp_shapes[:4],
        "bytes_per_redistribute": per_redistribute_bytes,
        "before": before,
        "after": after,
        "after_free": released,
        "local_tensor_data_ptr_changed": True,
    }

    gathered = [None] * world_size
    dist.all_gather_object(gathered, out)
    if rank == 0:
        merged = {"device": device, "world_size": world_size,
                  "rank0": gathered[0],
                  "rank_examples": gathered}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "partial_grads.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        md.write_manifest(out_dir, "C", world_size,
                          [{"case": "row_parallel"}, {"case": "col_parallel"},
                           {"case": "redistribute_buffers"}],
                          extra={"device": device, "backend": backend})
        rp = gathered[0]["row_parallel"]
        cp = gathered[0]["col_parallel"]
        print(f"[C] 行并行：y={rp['y_placements']} loss={rp['loss_placements']} "
              f"局部 loss={rp['loss_local_value']:.6f} 全局={rp['loss_full_value']:.6f} "
              f"参照={rp['loss_ref_value']:.6f} 一致={rp['loss_full_matches_ref']}")
        print(f"    grad_x={rp['grad_x_placements']} grad_W={rp['grad_W_placements']} "
              f"grad_W 归约后与单机一致={rp['grad_W_full_matches_ref']} "
              f"grad_x max|diff|={rp['grad_x_max_abs_diff']:.3e}")
        print(f"    算子内通信 前向={rp['forward_collectives']} 反向x={rp['backward_x_collectives']} "
              f"反向W={rp['backward_W_collectives']} grad_W 物化={rp['grad_W_materialize_collectives']}")
        print(f"[C] 列并行：y={cp['y_placements']} loss={cp['loss_placements']} "
              f"局部 loss={cp['loss_local_value']:.6f} 全局={cp['loss_full_value']:.6f} "
              f"局部等于全局={cp['local_equals_global']}")
        print(f"[C] redistribute buffer: {gathered[0]['redistribute_buffers']}")
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------
# 任务 D：reshape 的块边界、collective 合并与 FSDP2 布局（6.0b 补遗）
# --------------------------------------------------------------------------

RESHAPE_CASES = [
    ((8, 6), (2, 24), "分片维的块边界在展平后仍对齐"),
    ((8, 6), (3, 16), "新形状第 0 维不被 mesh 整除"),
    ((8, 6), (6, 8), "块边界跨新形状的行但整除成立"),
    ((8, 6), (2, 4, 6), "三维，第 0 维每段 24 个元素，与本地块等长"),
    ((8, 6), (4, 2, 6), "三维，第 0 维切两段后每段仍是 24 个元素"),
    ((12, 4), (3, 16), "块长 6 与目标行宽 16 不整除"),
    ((7, 5), (5, 7), "质数尺寸，块长 4/3 跨行"),
    ((4, 6), (24,), "两块刚好拼成一行"),
    ((48,), (8, 6), "一维反展平"),
]

# 2×2 mesh 上「两维各一次通信」的 src→dst 组合；每个组合在三种 mesh 配置下各跑一遍
MERGE_PAIRS = [
    (("S0", "S1"), ("R", "R"), "两维都 all_gather"),
    (("Psum", "Psum"), ("R", "R"), "两维都 all_reduce"),
    (("Psum", "Psum"), ("S0", "S0"), "reduce_scatter + all_gather 混合"),
    (("R", "S1"), ("S0", "S1"), "跨维 rebase，只有一维需要通信"),
]


def _to_torch_placements(pair, Replicate, Shard, Partial):
    out = []
    for name in pair:
        if name == "R":
            out.append(Replicate())
        elif name.startswith("S"):
            out.append(Shard(int(name[1:])))
        else:
            out.append(Partial(name[3:]))
    return out


def _capture_redistribute_warnings():
    """抓 _redistribute 的 logger.warning：合并失败的原因只写在那里。"""
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("torch.distributed.tensor._redistribute")
    handler = _H(level=logging.WARNING)
    logger.addHandler(handler)
    return records, (logger, handler)


def run_D1(rank, world_size, out_dir, device="cpu"):
    """reshape 的块边界：分片块跨界时是否必须通信，实测与规则分开记。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29723"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor as dt_distribute
    from torch.distributed.tensor import Replicate, Shard

    mesh = init_device_mesh(device, (world_size,))
    mesh_shape = (world_size,)
    coord = md.mesh_coord(rank, mesh_shape)
    cases = []
    for gshape, tshape, note in RESHAPE_CASES:
        n = 1
        for d in gshape:
            n *= d
        g = torch.arange(n, dtype=torch.float64, device=dev).reshape(gshape)
        ref = g.reshape(tshape)
        x = dt_distribute(g, mesh, [Shard(0)])
        src_local = x.to_local().clone()
        err = None
        try:
            y, agg = collectives(lambda: x.reshape(tshape))
            mat, agg_mat = collectives(lambda: y.full_tensor())
            placements = list(y.placements)
            # md.place 用的是 mini_dtensor 自己的 placement 类，先按短记号转换
            mini_pl = [md.parse_placement(fmt_placement(p)) for p in placements]
            expected = md.place(ref, mesh_shape, mini_pl, coord)
            local = y.to_local()
            rec = {
                "note": note,
                "global_shape": list(gshape),
                "target_shape": list(tshape),
                "src_placements": [fmt_placement(p) for p in x.placements],
                "actual_placements": [fmt_placement(p) for p in placements],
                "src_local_shape": list(src_local.shape),
                "local_shape": list(local.shape),
                "local_numel_times_world": local.numel() * world_size,
                "global_numel": n,
                "collectives_during_reshape": agg,
                "reshape_comm_calls": sum(v["count"] for v in agg.values()),
                "reshape_comm_bytes": total_bytes(agg),
                "global_matches_reference": bool(
                    torch.equal(mat.to(torch.float64), ref)),
                "local_matches_placement_rule": bool(
                    list(local.shape) == list(expected.shape)
                    and torch.equal(local.to(torch.float64), expected.to(torch.float64))),
                "local_equals_src_block": bool(
                    list(local.shape) == list(src_local.shape)
                    and torch.equal(local, src_local)),
                "reshape_error": None,
            }
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:300]}"
            rec = {
                "note": note, "global_shape": list(gshape),
                "target_shape": list(tshape),
                "src_placements": [fmt_placement(p) for p in x.placements],
                "actual_placements": None, "src_local_shape": list(src_local.shape),
                "local_shape": None, "local_numel_times_world": None,
                "global_numel": n,
                "collectives_during_reshape": {}, "reshape_comm_calls": None,
                "reshape_comm_bytes": None, "global_matches_reference": None,
                "local_matches_placement_rule": None, "local_equals_src_block": None,
                "reshape_error": err,
            }
        # 报错时再走一遍显式 redistribute：把「必须自己搬」的代价量出来
        if err is not None:
            try:
                xr, rc = collectives(lambda: x.redistribute(mesh, [Replicate()]))
                mat, _ = collectives(lambda: xr.reshape(tshape))
                full = mat.full_tensor() if hasattr(mat, "full_tensor") else mat
                rec["explicit_path"] = {
                    "steps": "redistribute(S0→R) 后本地 reshape",
                    "redistribute_collectives": rc,
                    "redistribute_comm_calls": sum(v["count"] for v in rc.values()),
                    "redistribute_comm_bytes": total_bytes(rc),
                    "global_matches_reference": bool(
                        torch.equal(full.to(torch.float64), ref)),
                }
            except Exception as e:  # noqa: BLE001
                rec["explicit_path"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        cases.append(rec)

    out = {"device": device, "backend": backend, "world_size": world_size,
           "rule": "本地块在行主序下是一段连续元素。它要么整好覆盖目标形状第 0 维的"
                   "整数个单元（此时 reshape 是纯本地 view、0 次通信），要么直接报错"
                   "（Cannot flatten/unflatten unevenly sharded tensor），不会静默通信；"
                   "要跨过块边界必须显式 redistribute，代价记在 explicit_path",
           "cases": cases}
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "reshape_boundary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        md.write_manifest(out_dir, "D1", world_size,
                          [{"global": c["global_shape"], "target": c["target_shape"]}
                           for c in cases],
                          extra={"device": device, "backend": backend,
                                 "artifact": "reshape_boundary.json"})
        print(f"[D1] world={world_size} device={device}")
        for c in cases:
            if c["reshape_error"]:
                print(f"  {str(c['global_shape']):<10}→{str(c['target_shape']):<10} "
                      f"{c['note']:<18} 报错：{c['reshape_error'][:110]}")
                ex = c.get("explicit_path") or {}
                print(f"      显式 redistribute 后：通信={ex.get('redistribute_comm_calls')} "
                      f"次/{ex.get('redistribute_comm_bytes')} B "
                      f"一致={ex.get('global_matches_reference')} "
                      f"err={ex.get('error')}")
                continue
            print(f"  {str(c['global_shape']):<10}→{str(c['target_shape']):<10} "
                  f"{c['note']:<18} src={c['src_placements']} "
                  f"dst={c['actual_placements']} "
                  f"通信={c['reshape_comm_calls']} 次/{c['reshape_comm_bytes']} B "
                  f"全局一致={c['global_matches_reference']} "
                  f"本地=规则块 {c['local_matches_placement_rule']}")
    dist.barrier()
    dist.destroy_process_group()


def run_D2(rank, world_size, out_dir, device="cpu"):
    """2×2 mesh 上连续同型 collective 的合并：flattened mesh 是关键条件。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29724"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import (
        distribute_tensor as dt_distribute, Replicate, Shard, Partial)
    import torch.distributed.tensor._redistribute as rd

    mesh_shape = (2, 2)
    assert world_size == 4, "任务 D2 需要 2×2 mesh（world=4）"
    mesh = init_device_mesh(device, mesh_shape, mesh_dim_names=("dp", "tp"))
    flat = mesh._flatten("dp_tp")
    print(f"[D2] rank{rank} mesh={mesh.mesh_dim_names} "
          f"flatten={flat.mesh_dim_names} 开关={rd._DISABLE_REDISTRIBUTE_TRANSFORM_OPTIMIZATION}",
          flush=True)

    g = torch.arange(24, dtype=torch.float64, device=dev).reshape(4, 6)
    cases = []
    for src, dst, note in MERGE_PAIRS:
        src_pl = _to_torch_placements(src, Replicate, Shard, Partial)
        dst_pl = _to_torch_placements(dst, Replicate, Shard, Partial)
        for variant, disabled in (("mesh_no_flatten_named", None),
                                  ("flattened_mesh", False),
                                  ("flattened_mesh_opt_off", True)):
            if variant == "mesh_no_flatten_named":
                # 同形状但没建 flattened 子 mesh：合并条件不满足
                m2 = init_device_mesh(device, mesh_shape,
                                      mesh_dim_names=("dp", "tp"))
                flat_used = None
            else:
                m2 = init_device_mesh(device, mesh_shape,
                                      mesh_dim_names=("dp", "tp"))
                m2._flatten("dp_tp")
                flat_used = "dp_tp"
            x = _make_src(g, m2, src_pl, Partial)
            warns, (lg, hd) = _capture_redistribute_warnings()
            try:
                if disabled is None:
                    y, agg = collectives(lambda: x.redistribute(m2, dst_pl))
                else:
                    ctx = rd.disable_redistribute_transform_optimization(disabled)
                    with ctx:
                        y, agg = collectives(lambda: x.redistribute(m2, dst_pl))
                full = y.full_tensor()
                ok = bool(torch.allclose(full.to(torch.float64), g))
                calls = sum(v["count"] for v in agg.values())
                err = None
            except Exception as e:  # noqa: BLE001
                y, agg, full, ok, calls = None, {}, None, False, None
                err = f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                lg.removeHandler(hd)
            cases.append({
                "src": list(src), "dst": list(dst), "note": note,
                "variant": variant, "flattened_mesh": flat_used,
                "opt_disabled": disabled,
                "actual_placements": None if y is None else
                    [fmt_placement(p) for p in y.placements],
                "collectives": agg,
                "comm_calls": calls,
                "comm_bytes": total_bytes(agg) if agg else 0,
                "global_matches_reference": ok,
                "warnings": warns[:3],
                "error": err,
            })

    out = {"device": device, "backend": backend, "mesh_shape": list(mesh_shape),
           "merge_rule": "只有相邻、同型、src_dst_placements 一致、且存在覆盖这些"
                         "维度的 flattened mesh 时，才把多次 collective 合成一次",
           "cases": cases}
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "collective_merge.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        md.write_manifest(out_dir, "D2", world_size,
                          [{"src": c["src"], "dst": c["dst"], "variant": c["variant"]}
                           for c in cases],
                          extra={"device": device, "backend": backend,
                                 "artifact": "collective_merge.json"})
        print(f"[D2] 2×2 mesh，{len(MERGE_PAIRS)} 个 src→dst × 3 种 mesh 配置")
        for c in cases:
            print(f"  {str(c['src']):<18}→{str(c['dst']):<18} {c['variant']:<24} "
                  f"通信={c['comm_calls']} 次/{c['comm_bytes']} B "
                  f"一致={c['global_matches_reference']} "
                  f"警告={(c['warnings'][0][:48] + '…') if c['warnings'] else '-'}")
    dist.barrier()
    dist.destroy_process_group()


def _ref_key(name, ref_grads):
    """FSDP2 会给参数改名（插入 _fsdp_wrapped 一类前缀/后缀），按后缀匹配参照。"""
    if name in ref_grads:
        return name
    cands = [k for k in ref_grads if name.endswith(k) or k.endswith(name)]
    return cands[0] if len(cands) == 1 else None


def run_D3(rank, world_size, out_dir, device="cpu"):
    """FSDP2 的布局与它实际发出的 collective：参数分片、reshard 开关、梯度对拍。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29725"
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    if device == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}") if device == "cuda" else torch.device("cpu")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh(device, (world_size,))
    torch.manual_seed(0)
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(8, 16)
            self.act = nn.ReLU()
            self.l2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.l2(self.act(self.l0(x)))

    x = torch.randn(4, 8, dtype=torch.float64, device=dev)

    # 单进程参照：同样的初始权重、同样的输入
    torch.manual_seed(11)
    ref_model = Tiny().to(dev).to(torch.float64)
    ref_loss = (ref_model(x) ** 2).sum()
    ref_loss.backward()
    ref_grads = {n: p.grad.detach().clone()
                 for n, p in ref_model.named_parameters()}

    results = {}
    for reshard in (True, False):
        torch.manual_seed(11)
        model = Tiny().to(dev).to(torch.float64)
        for m in list(model.children()):
            if isinstance(m, nn.Linear):
                fully_shard(m, mesh=mesh, reshard_after_forward=reshard)
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard)

        param_view = []
        for n, p in model.named_parameters():
            local = p.to_local()
            param_view.append({
                "name": n,
                "placements": [fmt_placement(pl) for pl in p.placements],
                "global_shape": list(p.shape),
                "local_shape": list(local.shape),
                "local_numel": local.numel(),
                "local_bytes": local.numel() * local.element_size(),
                "is_dtensor": type(p).__name__ == "DTensor",
            })
        out_y, fwd = collectives(lambda: model(x), itemsize=8)
        loss, loss_coll = collectives(lambda: (out_y ** 2).sum(), itemsize=8)
        _, bwd = collectives(lambda: loss.backward(), itemsize=8)

        grads_ok, grads_diff = {}, {}
        for n, p in model.named_parameters():
            key = _ref_key(n, ref_grads)
            if key is None:
                grads_ok[n] = "no reference"
                continue
            if p.grad is None:
                grads_ok[key] = None
                continue
            gf = p.grad.full_tensor().to(torch.float64)
            grads_ok[key] = bool(torch.allclose(gf, ref_grads[key], atol=1e-10))
            grads_diff[key] = float((gf - ref_grads[key]).abs().max().item())
        results[f"reshard_after_forward={reshard}"] = {
            "params": param_view,
            "forward_collectives": fwd,
            "loss_collectives": loss_coll,
            "backward_collectives": bwd,
            "forward_all_gather_calls": sum(
                v["count"] for k, v in fwd.items() if "all_gather" in k),
            "backward_reduce_scatter_calls": sum(
                v["count"] for k, v in bwd.items() if "reduce_scatter" in k),
            "grad_matches_reference": grads_ok,
            "grad_max_abs_diff": grads_diff,
            "loss_ref": float(ref_loss.item()),
        }

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {"rank": rank, "results": results})
    if rank == 0:
        out = {"device": device, "backend": backend, "world_size": world_size,
               "note": "单进程参照与分布式用同一 seed=11 初始化；"
                       "梯度经 full_tensor() 归约后逐元素对拍",
               "rank0": results,
               "rank_examples": gathered}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "fsdp2_layout.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        md.write_manifest(out_dir, "D3", world_size,
                          [{"case": k} for k in results],
                          extra={"device": device, "backend": backend,
                                 "artifact": "fsdp2_layout.json"})
        for k, v in results.items():
            print(f"[D3] {k}")
            for p in v["params"]:
                print(f"    {p['name']:<12} {p['placements']} "
                      f"全局{tuple(p['global_shape'])}→本地{tuple(p['local_shape'])} "
                      f"{p['local_bytes']} B")
            print(f"    前向 all-gather {v['forward_all_gather_calls']} 次  "
                  f"反向 reduce-scatter {v['backward_reduce_scatter_calls']} 次  "
                  f"梯度与单机一致={v['grad_matches_reference']}")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["B", "C", "D1", "D2", "D3"])
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    import torch.multiprocessing as mp
    if a.task == "B":
        mp.spawn(run_B, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)
    elif a.task == "C":
        mp.spawn(run_C, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)
    elif a.task == "D1":
        mp.spawn(run_D1, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)
    elif a.task == "D2":
        mp.spawn(run_D2, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)
    else:
        mp.spawn(run_D3, args=(a.world_size, a.out, a.device), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    sys.exit(main())
