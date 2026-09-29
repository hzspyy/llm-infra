#!/usr/bin/env python3
"""教学小语言模型的预算内预训练入口（7.1-F，兼作 7.4-H 与 7.9-I 的载体）。

模型与分词器复用 MiniMind 上游实现（`--minimind-src` 指向固定 commit 的仓库），
本脚本只负责本项目要求的测量层：有效 token、训练/验证 loss、grad norm、
update/weight ratio、LR、吞吐、峰值显存、finite/skip、断点续训与阶段权重导出。

固定下来的训练协议（与上游 train_pretrain.py 一致的部分）：
  * AdamW，LR 为 lr*(0.1+0.45*(1+cos(pi*step/total)))，无 warmup
  * BF16 autocast + FP32 master weight，grad clip 1.0，梯度累积 accumulation_steps
  * 打包窗口 = seq_len token，labels 与 input_ids 相同（模型内部做 next-token shift）

本项目新增的测量与可选改动（正文逐条说明）：
  * 验证/测试 loss 在固定的 hold-out 窗口上算，模型选择以验证 loss 为准
  * `--precision` 与 `--optimizer` 用于限定窗口的精度/优化器对拍（7.9-I）
  * `--limit-seconds` 与 `--max-steps` 共同构成预算停止条件

Usage:
    python labs/L7/teaching_pretrain.py \
      --data-dir "$RUN/data-pretrain" --minimind-src "$SRC/minimind" \
      --outdir "$RUN/pretrain" --max-steps 6000 --micro-bs 32 --accum 1
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
from pathlib import Path

import numpy as np
import torch


def log(message):
    print(f"[teaching_pretrain] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    """上游 trainer_utils.get_lr 的同一公式，便于与 MiniMind 配方对齐。"""
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_model_class(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    return MiniMindConfig, MiniMindForCausalLM


def build_windows(bin_path: Path, seq_len: int):
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_windows = int(len(tokens) // seq_len)
    return tokens, n_windows


def gather(tokens, starts, seq_len):
    return np.stack([np.asarray(tokens[s * seq_len:(s + 1) * seq_len]) for s in starts])


@torch.no_grad()
def evaluate(model, tokens, windows, seq_len, micro_bs, device, dtype):
    if len(windows) == 0:
        return None
    model.eval()
    losses = []
    for i in range(0, len(windows), micro_bs):
        chunk = windows[i:i + micro_bs]
        x = torch.from_numpy(gather(tokens, chunk, seq_len).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            out = model(x, labels=x)
        losses.append(float(out.loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def greedy_sample(model, tokenizer, prompt, max_new_tokens, device, dtype, seq_len):
    model.eval()
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ids = [tokenizer.bos_token_id] + ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = list(ids)
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x[:, -seq_len:]).logits[:, -1, :]
        nxt = int(torch.argmax(logits, dim=-1).item())
        out_ids.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(out_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-bs", type=int, default=32)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--limit-seconds", type=float, default=0.0,
                        help=">0 时到点即停，与 max-steps 共同构成预算停止条件")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--optimizer", choices=["adamw", "fused"], default="adamw")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-windows", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-snapshots", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None,
                        help="从 last.pt 继续；不改变 window_order 与步数口径")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="只加载模型权重（新的微调/消融起点，不恢复 optimizer/RNG）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    assert torch.cuda.is_available(), "本入口需要 GPU"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    torch.backends.cuda.matmul.allow_tf32 = False

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    MiniMindConfig, MiniMindForCausalLM = load_model_class(args.minimind_src)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model", trust_remote_code=True)

    train_tokens, n_train_windows = build_windows(args.data_dir / "train.bin", args.seq_len)
    val_tokens, n_val_windows = build_windows(args.data_dir / "val.bin", args.seq_len)
    test_tokens, n_test_windows = build_windows(args.data_dir / "test.bin", args.seq_len)
    if n_train_windows == 0:
        raise SystemExit("train.bin 不足一个窗口")

    order_path = outdir / "window_order.npy"
    if order_path.exists():
        order = np.load(order_path)
    else:
        order = np.random.default_rng(args.seed).permutation(n_train_windows)
        np.save(order_path, order)

    def fixed_windows(n_windows, count):
        if n_windows <= count:
            return np.arange(n_windows)
        return np.linspace(0, n_windows - 1, count).astype(np.int64)

    val_windows = fixed_windows(n_val_windows, args.eval_windows)
    test_windows = fixed_windows(n_test_windows, args.eval_windows)

    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数 {n_params / 1e6:.2f}M，训练窗口 {n_train_windows}，"
        f"验证/测试窗口 {n_val_windows}/{n_test_windows}")

    opt_cls = torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=args.lr,
                        fused=(args.optimizer == "fused"))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))

    start_step, cursor, tokens_seen, targets_seen, elapsed_done = 0, 0, 0, 0, 0.0
    best_val = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        cursor = state["cursor"]
        tokens_seen = state["tokens_seen"]
        targets_seen = state.get("targets_seen", 0)
        elapsed_done = state.get("elapsed", 0.0)
        best_val = state.get("best_val", float("inf"))
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        log(f"从 {args.resume} 恢复：step={start_step} cursor={cursor} tokens={tokens_seen}")
    elif args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        log(f"只加载模型权重 {args.init_from}（step={state.get('step', -1)}）")

    metrics_path = outdir / "metrics.jsonl"
    samples_path = outdir / "samples.jsonl"
    prompts = ["中国的首都是", "The capital of France is", "def fibonacci(n):",
               "水在标准大气压下的沸点是", "人工智能的发展"]
    run_start = time.time()
    model.train()

    def save_checkpoint(name, full: bool):
        payload = {"model": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                   "step": step, "cursor": cursor, "tokens_seen": tokens_seen,
                   "targets_seen": targets_seen,
                   "val_loss": last_val, "args": vars(args) | {"data_dir": str(args.data_dir),
                                                               "minimind_src": str(args.minimind_src),
                                                               "outdir": str(args.outdir)}}
        if full:
            payload |= {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "elapsed": elapsed_done + time.time() - run_start, "best_val": best_val,
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate()}
        torch.save(payload, outdir / name)

    step = start_step
    last_val = None
    reason = "max_steps"
    step_start = time.time()
    tokens_log = targets_log = 0
    log_start_step = start_step
    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start + elapsed_done) > args.limit_seconds:
            reason = "time_limit"
            break
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        micro_losses = []
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if cursor + args.micro_bs > len(order):      # 一个 epoch 走完就从头重排
                cursor = 0
            starts = order[cursor:cursor + args.micro_bs]
            cursor += args.micro_bs
            x = torch.from_numpy(gather(train_tokens, starts, args.seq_len).astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                out = model(x, labels=x)
                loss = (out.loss + out.aux_loss) / args.accum
            scaler.scale(loss).backward()
            micro_losses.append(float(loss) * args.accum)
            tokens_seen += x.numel()
            targets_seen += int(x.numel() - x.shape[0])
            tokens_log += x.numel()
            targets_log += int(x.numel() - x.shape[0])

        log_now = (step % args.log_every == 0) or (step + 1 == args.max_steps)
        before = None
        if log_now:
            before = [p.detach().clone() for p in model.parameters()]
        if args.precision == "fp16":
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        finite = bool(torch.isfinite(grad_norm).item())
        if args.precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step += 1

        if log_now:
            upd_sq = sum(float(((p.detach() - b) ** 2).sum()) for p, b in zip(model.parameters(), before))
            del before
            w_sq = sum(float((p.detach() ** 2).sum()) for p in model.parameters())
            dt = time.time() - step_start
            record = {
                "step": step, "tokens": tokens_seen, "targets": targets_seen,
                "loss": sum(micro_losses) / len(micro_losses),
                "grad_norm": float(grad_norm), "finite": finite, "lr": lr,
                "update_norm": math.sqrt(upd_sq), "update_to_weight": math.sqrt(upd_sq / max(w_sq, 1e-12)),
                "log_seconds": dt, "step_seconds": dt / max(step - log_start_step, 1),
                "tokens_per_second": tokens_log / max(dt, 1e-9),
                "targets_per_second": targets_log / max(dt, 1e-9),
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "elapsed": elapsed_done + time.time() - run_start,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {step}/{args.max_steps} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm']:.3f} lr {lr:.3e} {record['tokens_per_second']:.0f} tok/s "
                f"peak {record['peak_memory_mib']:.0f} MiB")
            step_start = time.time()
            log_start_step = step
            tokens_log = targets_log = 0

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_tokens, val_windows, args.seq_len,
                                args.micro_bs, device, dtype)
            last_val = val_loss
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "eval": "val", "val_loss": val_loss}) + "\n")
            log(f"  eval step {step}: val_loss {val_loss:.4f}")
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best.pt", full=False)
            save_checkpoint("last.pt", full=True)
            if step % args.save_every == 0:
                save_checkpoint(f"step{step}.pt", full=False)
                snaps = sorted(outdir.glob("step*.pt"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()

        if args.sample_every and step % args.sample_every == 0:
            with samples_path.open("a") as stream:
                for prompt in prompts:
                    text = greedy_sample(model, tokenizer, prompt, args.sample_tokens,
                                         device, dtype, args.seq_len)
                    stream.write(json.dumps({"step": step, "prompt": prompt, "text": text},
                                            ensure_ascii=False) + "\n")

    # 收尾：测试集只在结束时算一次，避免用它做模型选择
    test_loss = evaluate(model, test_tokens, test_windows, args.seq_len, args.micro_bs, device, dtype)
    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "minimind_commit": subprocess.run(
            ["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "params": n_params, "stop_reason": reason, "steps": step,
        "tokens_seen": tokens_seen, "targets_seen": targets_seen,
        "best_val_loss": best_val, "test_loss": test_loss,
        "elapsed_seconds": elapsed_done + time.time() - run_start,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
    }
    (outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，tokens {tokens_seen}，test_loss {test_loss:.4f}")


if __name__ == "__main__":
    main()
