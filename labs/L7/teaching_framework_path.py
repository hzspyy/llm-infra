#!/usr/bin/env python3
"""教学预训练契约在单卡循环与 FSDP2 框架路径上的同一份实现（7.3-G）。

同一个模型、同一份打包数据、同一条窗口顺序、同一份 lr 与 clip：`--mode plain` 用
MiniMind 自己的单卡循环，`--mode fsdp2` 用 `torch.distributed.fsdp.fully_shard` 包装后
由 torchrun 启动。两条路径都要能对上同一批损失轨迹，并各自走一遍"保存"接口：

  plain  : `torch.save` 全量 state_dict（本项目其他教学入口的格式）
  fsdp2  : `torch.distributed.checkpoint` 的 `save`/`load`（分片元数据 + 重新加载校验）

`--profile` 打开时对中间若干 step 采 profiler，统计 `all_gather` / `reduce_scatter` 的
调用次数，用来区分"框架路径跑了"与"通信真的发生了"。

Usage:
    torchrun --standalone --nproc_per_node=1 labs/L7/teaching_framework_path.py \
      --mode plain --data-dir "$RUN/data-pretrain" --minimind-src "$SRC/minimind" \
      --outdir "$RUN/fw-plain" --max-steps 300
    torchrun --standalone --nproc_per_node=2 labs/L7/teaching_framework_path.py \
      --mode fsdp2 --data-dir "$RUN/data-pretrain" --minimind-src "$SRC/minimind" \
      --outdir "$RUN/fw-fsdp2" --max-steps 300 --profile
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.profiler  # noqa: F401  # 取集合通信调用次数时使用


def log(message):
    print(f"[framework_path] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def gather(tokens, starts, seq_len):
    return np.stack([np.asarray(tokens[s * seq_len:(s + 1) * seq_len]) for s in starts])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["plain", "fsdp2", "compare"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--minimind-src", type=Path, default=None)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-bs", type=int, default=32, help="全局 micro-batch（所有 rank 合计）")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--minimind-commit", default="",
                        help="快照不是 git 仓库时显式记录源码 revision")
    parser.add_argument("--plain-dir", type=Path, help="compare：plain 运行的输出目录")
    parser.add_argument("--fsdp2-dir", type=Path, help="compare：fsdp2 运行的输出目录")
    parser.add_argument("--profile-steps", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "compare":
        if args.plain_dir is None or args.fsdp2_dir is None:
            raise SystemExit("compare 需要 --plain-dir 与 --fsdp2-dir")
        def read_metrics(directory):
            rows = [json.loads(line) for line in
                    (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
            return {row["step"]: row for row in rows}, json.loads(
                (directory / "run.json").read_text())
        plain_rows, plain_run = read_metrics(args.plain_dir)
        fsdp2_rows, fsdp2_run = read_metrics(args.fsdp2_dir)
        steps = sorted(set(plain_rows) & set(fsdp2_rows))
        deltas = [(step, fsdp2_rows[step]["loss"] - plain_rows[step]["loss"]) for step in steps]
        first = next((step for step, delta in deltas if abs(delta) > 1e-3), None)
        worst = max(deltas, key=lambda item: abs(item[1])) if deltas else (None, 0.0)
        result = {
            "section": "framework-path-compare",
            "compared_steps": len(steps),
            "loss_delta_first_gt_1e-3_at_step": first,
            "loss_delta_abs_max": abs(worst[1]), "loss_delta_abs_max_step": worst[0],
            "loss_delta_at_last_step": deltas[-1][1] if deltas else None,
            "plain_final_loss": plain_rows[steps[-1]]["loss"] if steps else None,
            "fsdp2_final_loss": fsdp2_rows[steps[-1]]["loss"] if steps else None,
            "weight_sq_plain": plain_run["weight_sq"], "weight_sq_fsdp2": fsdp2_run["weight_sq"],
            "weight_sq_relative_diff": abs(fsdp2_run["weight_sq"] - plain_run["weight_sq"])
            / abs(plain_run["weight_sq"]),
            "peak_memory_mib_plain": plain_run["peak_memory_mib"],
            "peak_memory_mib_fsdp2_per_rank": fsdp2_run["peak_memory_mib"],
            "collectives_fsdp2": fsdp2_run["collectives"],
            "global_tokens_plain": plain_run["tokens_per_rank"],
            "global_tokens_fsdp2": fsdp2_run["tokens_per_rank"] * fsdp2_run["world_size"],
            "claim_scope": "两条路径的损失轨迹与参数总量；bf16 归约顺序不同，不宣称逐位相同",
        }
        args.outdir.mkdir(parents=True, exist_ok=False)
        (args.outdir / "compare.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.mode != "compare" and (args.data_dir is None or args.minimind_src is None):
        raise SystemExit("plain/fsdp2 模式需要 --data-dir 与 --minimind-src")
    if args.mode == "plain" and world_size != 1:
        raise SystemExit("plain 模式只支持单进程（world_size=1）")
    if args.micro_bs % world_size != 0:
        raise SystemExit(f"micro-bs {args.micro_bs} 不能被 world_size {world_size} 整除")
    per_rank_bs = args.micro_bs // world_size

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mode == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        torch.distributed.init_process_group("nccl")
        mesh = init_device_mesh("cuda", (world_size,))

    sys.path.insert(0, str(args.minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433

    if rank == 0:
        args.outdir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    tokens = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    n_windows = int(len(tokens) // args.seq_len)
    # 两条路径用同一条窗口顺序：同一个 seed、同一个公式，不读磁盘上的旧文件
    order = np.random.default_rng(args.seed).permutation(n_windows)

    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.mode == "fsdp2":
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dtype = torch.bfloat16
    if rank == 0:
        log(f"mode={args.mode} world={world_size} per_rank_bs={per_rank_bs} "
            f"params={n_params/1e6:.2f}M windows={n_windows}")

    metrics_path = args.outdir / "metrics.jsonl"
    run_start, cursor, tokens_seen = time.time(), 0, 0
    collective_counts = Counter()
    step_times = []
    for step in range(args.max_steps):
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # 全局 batch 被切成 world_size 份，rank r 取第 r 段：总量与 plain 路径一致
        starts = []
        for r in range(world_size):
            base = cursor + r * per_rank_bs
            starts.append(order[base:base + per_rank_bs])
        cursor += args.micro_bs
        if cursor + args.micro_bs > len(order):
            cursor = 0
        x = torch.from_numpy(gather(tokens, starts[rank], args.seq_len).astype(np.int64)).to(device)
        optimizer.zero_grad(set_to_none=True)
        profile_now = args.profile and 10 <= step < 10 + args.profile_steps
        if profile_now:
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.autocast("cuda", dtype=dtype, enabled=True):
                    out = model(x, labels=x)
                    loss = out.loss + out.aux_loss
                loss.backward()
            for event in prof.key_averages():
                name = event.key
                if "all_gather" in name or "reduce_scatter" in name or "all_to_all" in name:
                    collective_counts[name] += event.count
        else:
            step_start = time.time()
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                out = model(x, labels=x)
                loss = out.loss + out.aux_loss
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step_times.append(time.time() - step_start if not profile_now else float("nan"))
        tokens_seen += x.numel()
        if rank == 0 and (step % args.log_every == 0 or step + 1 == args.max_steps):
            recent = [value for value in step_times[-args.log_every:] if value == value]
            record = {"step": step + 1, "loss": float(loss), "grad_norm": float(grad_norm),
                      "lr": lr, "world_size": world_size,
                      "tokens_per_rank": tokens_seen,
                      "tokens_per_second_global": (tokens_seen * world_size
                                                   / max(sum(step_times[-args.log_every:]), 1e-9)),
                      "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "step_seconds_median": float(np.median(recent)) if recent else None,
                      "elapsed": time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {record['step']}/{args.max_steps} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm']:.3f} {record['tokens_per_second_global']:.0f} tok/s"
                f" peak {record['peak_memory_mib']:.0f} MiB")

    # 参数总量的平方和：rank 局部求和后 all_reduce，两条路径可直接比较
    local_sq = sum(float((p.detach().float() ** 2).sum()) for p in model.parameters())
    if args.mode == "fsdp2":
        tensor = torch.tensor([local_sq], device=device)
        torch.distributed.all_reduce(tensor)
        weight_sq = float(tensor.item())
    else:
        weight_sq = local_sq

    saved = None
    if args.mode == "plain" and rank == 0:
        path = args.outdir / "plain_last.pt"
        torch.save({"model": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                    "step": args.max_steps, "weight_sq": weight_sq}, path)
        saved = str(path)
    if args.mode == "fsdp2":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (get_model_state_dict,
                                                             get_optimizer_state_dict)
        ckpt_dir = args.outdir / "dcp"
        state = {"model": get_model_state_dict(model),
                 "optimizer": get_optimizer_state_dict(model, optimizer)}
        dcp.save(state, checkpoint_id=ckpt_dir)
        torch.distributed.barrier()
        reloaded = {"model": get_model_state_dict(model),
                    "optimizer": get_optimizer_state_dict(model, optimizer)}
        dcp.load(reloaded, checkpoint_id=ckpt_dir)
        if rank == 0:
            files = sorted(p.name for p in Path(ckpt_dir).rglob("*") if p.is_file())
            saved = {"dcp_dir": str(ckpt_dir), "files": files}

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        summary = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": args.mode, "world_size": world_size, "steps": args.max_steps,
            "global_micro_batch": args.micro_bs, "per_rank_batch": per_rank_bs,
            "sequence_length": args.seq_len, "tokens_per_rank": tokens_seen,
            "global_tokens": tokens_seen * world_size,
            "params": n_params, "weight_sq": weight_sq,
            "elapsed_seconds": time.time() - run_start,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "collectives": dict(collective_counts), "saved": saved,
            "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "minimind_commit": args.minimind_commit or subprocess.run(
                ["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        }
        (args.outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        log(f"结束：weight_sq={weight_sq:.6f} 保存 {saved}")
    if args.mode == "fsdp2":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
