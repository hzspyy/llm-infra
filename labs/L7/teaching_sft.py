#!/usr/bin/env python3
"""教学模型的 SFT 入口（7.5-I）：从 7.1-F 的预训练权重到可对话、可部署的适配模型。

数据是 7.8-I 产出的固定长度张量（input_ids / labels），监督只在 assistant 段。
loss 用"有效回答 token 加权"的 CE：每个 batch 返回 sum 与 count，累计后再相除，
这样长短样本、最后一批不满都不会改变分母口径。

与上游 train_full_sft.py 对齐的部分：lr 1e-5、BF16 autocast、grad clip 1.0、
assistant 段 label 生成规则（见 teaching_data.py）。本项目新增：验证 loss、
grad/update 记录、best 选择、断点续训与阶段产物导出。

Usage:
    python labs/L7/teaching_sft.py \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC/minimind" \
      --init-from "$RUN/pretrain/best.pt" --outdir "$RUN/sft" --max-steps 900
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def log(message):
    print(f"[teaching_sft] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_model_class(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    return MiniMindConfig, MiniMindForCausalLM


def batch_loss(model, x, y, dtype):
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        logits = model(x).logits
    flat_logits = logits[..., :-1, :].reshape(-1, logits.size(-1)).float()
    flat_labels = y[..., 1:].reshape(-1)
    valid = flat_labels != -100
    if not bool(valid.any()):
        return torch.zeros((), device=x.device, requires_grad=True), 0
    loss_sum = F.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="sum")
    return loss_sum, int(valid.sum())


@torch.no_grad()
def evaluate_ce(model, npz_path, micro_bs, dtype, device):
    data = np.load(npz_path)
    inputs, labels = data["input_ids"], data["labels"]
    total, count = 0.0, 0
    model.eval()
    for i in range(0, len(inputs), micro_bs):
        x = torch.from_numpy(inputs[i:i + micro_bs].astype(np.int64)).to(device)
        y = torch.from_numpy(labels[i:i + micro_bs].astype(np.int64)).to(device)
        loss_sum, n = batch_loss(model, x, y, dtype)
        total += float(loss_sum)
        count += n
    model.train()
    return total / max(count, 1), count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--micro-bs", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--keep-snapshots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    MiniMindConfig, MiniMindForCausalLM = load_model_class(args.minimind_src)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    init = torch.load(args.init_from, map_location=device, weights_only=False)
    model.load_state_dict(init["model"])
    log(f"初始化自 {args.init_from}（step={init.get('step')} val={init.get('val_loss')}）")

    train = np.load(args.data_dir / "train.npz")
    train_inputs = torch.from_numpy(train["input_ids"].astype(np.int64))
    train_labels = torch.from_numpy(train["labels"].astype(np.int64))
    n_samples = len(train_inputs)
    steps_per_epoch = max(1, n_samples // args.micro_bs)
    log(f"SFT 训练样本 {n_samples}，每 epoch {steps_per_epoch} step，"
        f"目标 {args.max_steps} step")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    start_step, elapsed_done, best_val = 0, 0.0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        elapsed_done = state.get("elapsed", 0.0)
        best_val = state.get("best_val", float("inf"))
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        log(f"从 {args.resume} 恢复：step={start_step}")

    rng = np.random.default_rng(args.seed + start_step)
    metrics_path = outdir / "metrics.jsonl"
    run_start = time.time()
    step, cursor, answer_tokens = start_step, 0, 0
    epoch_order = rng.permutation(n_samples)
    count_log = 0
    log_start_step = start_step
    model.train()

    def save_checkpoint(name, full):
        payload = {"model": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                   "step": step, "val_loss": last_val, "best_val": best_val,
                   "answer_tokens": answer_tokens, "args": str(vars(args))}
        if full:
            payload |= {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "elapsed": elapsed_done + time.time() - run_start,
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate()}
        torch.save(payload, outdir / name)

    last_val, step_start, reason = None, time.time(), "max_steps"
    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start + elapsed_done) > args.limit_seconds:
            reason = "time_limit"
            break
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_loss, micro_count = 0.0, 0
        for _ in range(args.accum):
            if cursor + args.micro_bs > n_samples:
                epoch_order = rng.permutation(n_samples)
                cursor = 0
            idx = epoch_order[cursor:cursor + args.micro_bs]
            cursor += args.micro_bs
            x = train_inputs[idx].to(device)
            y = train_labels[idx].to(device)
            loss_sum, count = batch_loss(model, x, y, dtype)
            loss = loss_sum / max(count, 1) / args.accum
            scaler.scale(loss).backward()
            micro_loss += float(loss_sum)
            micro_count += count
            answer_tokens += count
            count_log += count
        log_now = (step % args.log_every == 0) or (step + 1 == args.max_steps)
        before = [p.detach().clone() for p in model.parameters()] if log_now else None
        if args.precision == "fp16":
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step += 1
        if log_now:
            upd_sq = sum(float(((p.detach() - b) ** 2).sum()) for p, b in zip(model.parameters(), before))
            del before
            dt = time.time() - step_start
            record = {"step": step, "loss": micro_loss / max(micro_count, 1),
                      "answer_tokens": answer_tokens, "grad_norm": float(grad_norm),
                      "finite": bool(torch.isfinite(grad_norm).item()), "lr": lr,
                      "update_norm": math.sqrt(upd_sq),
                      "answer_tokens_per_second": count_log / max(dt, 1e-9),
                      "log_seconds": dt, "step_seconds": dt / max(step - log_start_step, 1),
                      "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {step}/{args.max_steps} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm']:.3f} lr {lr:.2e} "
                f"{record['answer_tokens_per_second']:.0f} ans-tok/s")
            step_start = time.time()
            log_start_step = step
            count_log = 0

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss, val_tokens = evaluate_ce(model, args.data_dir / "val.npz",
                                               args.micro_bs, dtype, device)
            last_val = val_loss
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "eval": "val", "val_loss": val_loss,
                                         "val_answer_tokens": val_tokens}) + "\n")
            log(f"  eval step {step}: val_loss {val_loss:.4f}（{val_tokens} answer tokens）")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best.pt", full=False)
            save_checkpoint("last.pt", full=True)
            if step % args.save_every == 0:
                save_checkpoint(f"step{step}.pt", full=False)
                snaps = sorted(outdir.glob("step*.pt"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "minimind_commit": subprocess.run(["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "params": sum(p.numel() for p in model.parameters()),
        "init_from": str(args.init_from), "stop_reason": reason, "steps": step,
        "answer_tokens": answer_tokens, "best_val_loss": best_val,
        "elapsed_seconds": elapsed_done + time.time() - run_start,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
    }
    (outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
