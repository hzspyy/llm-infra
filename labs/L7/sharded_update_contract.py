#!/usr/bin/env python3
"""两 rank 的分片更新契约：状态归属、不等有效数、全局 clip 与通信时序。

同一个模型、同一份数据，用三种方式各做一次更新：
  single   单进程跑完整 global batch，作为参照
  ddp      两 rank 复制参数，梯度 all-reduce
  fsdp2    两 rank 分片参数/梯度/optimizer，前向 all-gather、反向 reduce-scatter

每种方式记录每 rank 的参数/梯度/optimizer 字节、峰值显存、全局梯度范数、
更新后参数与参照的最大差，以及 profiler 抓到的真实 collective 名称与次数。

两 rank 的有效 target 数被故意做成不相等，用来检查 loss 分母与 DDP 默认平均。

Usage（worldvln，envs/serve）:
    python labs/L7/sharded_update_contract.py --mode single --outdir "$RUN/single"
    torchrun --standalone --nproc_per_node=2 labs/L7/sharded_update_contract.py \\
        --mode ddp --reference "$RUN/single/reference.pt" --outdir "$RUN/ddp"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100
VOCAB, DIM, LAYERS, HEADS = 4096, 1024, 8, 8
SEQ, MICRO_BS = 128, 2
LR, WD = 1e-3, 0.01


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.heads, d // self.heads)
        q, k, v = (t_.view(shape).transpose(1, 2) for t_ in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        h = self.norm2(x)
        return x + self.down(F.silu(self.up(h)))


class MiniLM(nn.Module):
    """结构与真实 decoder 一致的小模型：固定 seed 初始化，各 rank 完全相同。"""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block(DIM, HEADS) for _ in range(LAYERS)])
        self.norm = nn.RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def make_shards():
    """两份 microbatch，有效 target 数刻意不等：rank0 少，rank1 多。"""
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, VOCAB, (2 * MICRO_BS, SEQ), generator=gen)
    labels = ids.clone()
    labels[0:MICRO_BS, : SEQ - 8] = IGNORE       # rank0 每条只留 8 个 target
    labels[MICRO_BS:, : SEQ // 2] = IGNORE       # rank1 每条留一半
    return ids, labels


def loss_sum_and_count(model, ids, labels):
    logits = model(ids)
    target = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    total = F.cross_entropy(logits.reshape(-1, VOCAB).float(), target.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    return total, int((target != IGNORE).sum())


def state_bytes(model, opt) -> dict:
    def nbytes(t):
        t = t.to_local() if hasattr(t, "to_local") else t
        return t.numel() * t.element_size()

    params = sum(nbytes(p) for p in model.parameters())
    grads = sum(nbytes(p.grad) for p in model.parameters() if p.grad is not None)
    opt_bytes = sum(nbytes(t) for st in opt.state.values() for t in st.values()
                    if torch.is_tensor(t))
    return {"param_MiB": round(params / 2 ** 20, 2),
            "grad_MiB": round(grads / 2 ** 20, 2),
            "optimizer_MiB": round(opt_bytes / 2 ** 20, 2),
            "sum_MiB": round((params + grads + opt_bytes) / 2 ** 20, 2)}


def full_params(model) -> dict:
    out = {}
    for name, p in model.named_parameters():
        t = p.detach()
        if hasattr(t, "full_tensor"):        # FSDP2 的 DTensor 需要先聚合
            t = t.full_tensor()
        out[name] = t.float().cpu().clone()
    return out


def compare(reference: dict, got: dict) -> dict:
    diffs = {k: float((got[k] - reference[k]).abs().max()) for k in reference}
    worst = max(diffs, key=diffs.get)
    return {"max_abs_diff": diffs[worst], "worst_tensor": worst,
            "tensors_compared": len(diffs)}


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


def run(args) -> dict:
    device = "cuda"
    ids_all, labels_all = make_shards()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    if args.mode != "single":
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        ids = ids_all[rank * MICRO_BS:(rank + 1) * MICRO_BS].to(device)
        labels = labels_all[rank * MICRO_BS:(rank + 1) * MICRO_BS].to(device)
    else:
        ids, labels = ids_all.to(device), labels_all.to(device)

    torch.manual_seed(0)
    model = MiniLM().to(device)
    if args.mode == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], gradient_as_bucket_view=True)
        inner = model.module
    elif args.mode.startswith("fsdp2"):
        from torch.distributed.fsdp import fully_shard
        reshard = args.mode != "fsdp2_noreshard"
        for blk in model.blocks:
            fully_shard(blk, reshard_after_forward=reshard)
        fully_shard(model, reshard_after_forward=reshard)
        inner = model
    else:
        inner = model
    opt = build_optimizer(model)

    # 预热一整步：DDP 要观察到梯度就绪顺序之后才会重建 bucket，
    # 分片实现也要先建好各自的通信 buffer。所有模式做同样的预热，参照仍然可比。
    for _ in range(args.warmup):
        warm_total, warm_valid = loss_sum_and_count(model, ids, labels)
        if args.mode == "single":
            warm_loss = warm_total / warm_valid
        else:
            c = torch.tensor([warm_valid], device=device, dtype=torch.float64)
            dist.all_reduce(c)
            warm_loss = warm_total * world / float(c.item())
        warm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    collectives: dict[str, int] = {}
    prof = None
    if args.profile and args.mode != "single":
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        prof.__enter__()

    mem = {"步开始": round(torch.cuda.memory_allocated() / 2 ** 20, 1)}
    total, local_valid = loss_sum_and_count(model, ids, labels)
    mem["前向后"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
    if args.mode == "single":
        global_valid = local_valid
        loss = total / global_valid
    else:
        counter = torch.tensor([local_valid], device=device, dtype=torch.float64)
        dist.all_reduce(counter)
        global_valid = int(counter.item())
        # DDP/FSDP2 的梯度归约默认取平均，乘回 world_size 才等于全局求和
        loss = total * world / global_valid
    loss.backward()
    # profiler 只覆盖前向与反向：后面聚合梯度用的 full_tensor 本身也会发 all-gather，
    # 留在窗口内会把插桩自身的通信混进机制观测。
    if prof is not None:
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        for evt in prof.key_averages():
            name = evt.key.lower()
            if name.startswith("c10d::") or name.startswith("nccl:"):
                collectives[evt.key[:48]] = int(evt.count)
        prof = None

    mem["反向后"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
    grads = {}
    for name, p in inner.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if hasattr(g, "full_tensor"):
            g = g.full_tensor()
        grads[name] = g.float().cpu().clone()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()
    torch.cuda.synchronize()

    params = full_params(inner)
    record = {
        "mode": args.mode, "rank": rank, "world_size": world,
        "local_valid_targets": local_valid, "global_valid_targets": global_valid,
        "loss": round(float(loss.detach()), 8),
        "grad_norm": round(gnorm, 8),
        "state_bytes": state_bytes(inner, opt),
        "peak_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "allocated_MiB": mem,
        "collectives": collectives,
        "total_params": sum(p.numel() for p in MiniLM().parameters()),
    }

    if args.reference and Path(args.reference).exists():
        ref = torch.load(args.reference, map_location="cpu", weights_only=False)
        record["vs_reference_params"] = compare(ref["params"], params)
        record["vs_reference_grads"] = compare(ref["grads"], grads)
        ref_scale = max(float(t.abs().max()) for t in ref["grads"].values())
        record["vs_reference_grads"]["relative_to_max_grad"] = (
            record["vs_reference_grads"]["max_abs_diff"] / ref_scale)
        record["reference_grad_norm"] = ref["grad_norm"]
        record["reference_global_valid"] = ref["global_valid"]

    if args.mode == "single" and args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"params": params, "grads": grads, "grad_norm": gnorm,
                    "global_valid": global_valid}, out / "reference.pt")

    if rank == 0:
        print(f"[{args.mode}] world={world} 参数 {record['total_params']:,}")
        print(f"  每 rank 状态：{record['state_bytes']}  峰值 {record['peak_MiB']} MiB")
        print(f"  allocated 轨迹：{mem}")
        print(f"  有效 target：本 rank {local_valid}，全局 {global_valid}；"
              f"loss={record['loss']}  全局梯度范数={record['grad_norm']}")
        if "vs_reference_grads" in record:
            g, q = record["vs_reference_grads"], record["vs_reference_params"]
            print(f"  与单进程参照：梯度最大差 {g['max_abs_diff']:.3e}"
                  f"（{g['worst_tensor']}，相对最大梯度 {g['relative_to_max_grad']:.2e}）；"
                  f"更新后参数最大差 {q['max_abs_diff']:.3e}")
            print(f"  参照梯度范数 {record['reference_grad_norm']:.8f}")
        if collectives:
            print(f"  profiler 抓到的 collective：{collectives}")

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank{rank}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode != "single":
        dist.barrier()
        dist.destroy_process_group()
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "ddp", "fsdp2", "fsdp2_noreshard"],
                    default="single")
    ap.add_argument("--reference")
    ap.add_argument("--outdir")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "本 lab 需要 GPU"
    run(args)


if __name__ == "__main__":
    main()
