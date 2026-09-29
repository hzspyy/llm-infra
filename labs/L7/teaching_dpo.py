#!/usr/bin/env python3
"""教学模型的 DPO 偏好优化（7.5-J）：同一 SFT 起点上的离线偏好分支。

目标函数与数据装载直接复用上游实现，避免"同名不同义"：

  * `dataset.lm_dataset.DPODataset`：同一条 chat template、同一份 assistant 段 loss mask、
    同一套 chosen/rejected 拼接（一个 batch 内先 chosen 后半 rejected）；
  * `trainer.train_dpo.dpo_loss` / `logits_to_log_probs`：脚本启动时用固定随机 batch 做
    一次数值对拍，对拍不通过直接退出（`--skip-parity` 可显式跳过）。

本项目新增：held-out 偏好 margin 曲线、grad/update/吞吐/峰值记录、最优 checkpoint 选择、
断点续训与导出。margin 的口径与 teaching_preference_eval.py 一致：
`(logπ_c − logπ_r) − (logπ_ref_c − logπ_ref_r)`，reference 固定为 SFT 起点。

Usage:
    python labs/L7/teaching_dpo.py --data-dir "$RUN/data-dpo" \
      --minimind-src "$SRC/minimind" --init-from "$RUN/sft/best.pt" \
      --outdir "$RUN/dpo" --beta 0.15 --lr 4e-8 --max-steps 2000
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
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
    print(f"[teaching_dpo] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_upstream_dpo_loss(minimind_src: Path):
    """从上游 train_dpo.py 原样取出两个损失函数，用于数值对拍。"""
    path = minimind_src / "trainer" / "train_dpo.py"
    spec = importlib.util.spec_from_file_location("upstream_train_dpo", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(minimind_src))
    spec.loader.exec_module(module)
    return module.logits_to_log_probs, module.dpo_loss


def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen = slice(0, batch_size // 2)
    rejected = slice(batch_size // 2, batch_size)
    pi_logratios = policy_log_probs[chosen] - policy_log_probs[rejected]
    ref_logratios = ref_log_probs[chosen] - ref_log_probs[rejected]
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--init-from", type=Path, required=True, help="SFT 阶段 checkpoint")
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--micro-bs", type=int, default=4,
                        help="每份的一半；一个 batch 实为 2×micro-bs 条序列")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=4e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-snapshots", type=int, default=5)
    parser.add_argument("--eval-pairs", type=int, default=128)
    parser.add_argument("--skip-parity", action="store_true")
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
    sys.path.insert(0, str(args.minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    from dataset.lm_dataset import DPODataset  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not args.skip_parity:
        up_logprob, up_loss = load_upstream_dpo_loss(args.minimind_src)
        torch.manual_seed(0)
        logits = torch.randn(4, 7, 13)
        labels = torch.randint(0, 13, (4, 7))
        mask = (torch.rand(4, 7) > 0.3).long()
        with torch.no_grad():
            mine = logits_to_log_probs(logits, labels)
            theirs = up_logprob(logits, labels)
            dl = float((mine - theirs).abs().max())
            ml = float(dpo_loss(mine, mine * 0.5, mask, 0.15))
            tl = float(up_loss(theirs, theirs * 0.5, mask, 0.15))
        if dl != 0 or ml != tl:
            raise SystemExit(f"与上游 train_dpo.py 的损失对拍失败：logprob 差 {dl}，loss 差 {ml-tl}")
        log(f"与上游损失对拍通过：logprob 最大差 {dl}，loss 差 {ml-tl}")

    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model")
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    ref_model = MiniMindForCausalLM(lm_config).to(device)
    init = torch.load(args.init_from, map_location=device, weights_only=False)
    model.load_state_dict(init["model"])
    ref_model.load_state_dict(init["model"])
    ref_model.eval()
    ref_model.requires_grad_(False)
    log(f"策略与 reference 同起自 {args.init_from}（step={init.get('step')}）")

    train_ds = DPODataset(str(args.data_dir / "train.jsonl"), tokenizer, max_length=args.max_len)
    val_ds = DPODataset(str(args.data_dir / "val.jsonl"), tokenizer, max_length=args.max_len)
    log(f"DPO 训练对 {len(train_ds)}，验证对 {len(val_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    start_step, elapsed_done, best_margin = 0, 0.0, float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        elapsed_done = state.get("elapsed", 0.0)
        best_margin = state.get("best_margin", float("-inf"))
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        log(f"从 {args.resume} 恢复：step={start_step}")

    def save_checkpoint(name, full):
        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, outdir / name)
        if full:
            torch.save({"model": {k: v.half().cpu() for k, v in model.state_dict().items()},
                        "step": step, "best_margin": best_margin,
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "elapsed": elapsed_done + time.time() - run_start,
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate(), "args": str(vars(args))},
                       outdir / "last_state.pt")

    @torch.no_grad()
    def margin_of(n_pairs):
        """held-out margin：固定前 n_pairs 对，策略与 reference 的 log-ratio 之差。"""
        model.eval()
        margins = []
        for index in range(min(n_pairs, len(val_ds))):
            batch = val_ds[index]
            x_chosen = batch["x_chosen"].unsqueeze(0).to(device)
            y_chosen = batch["y_chosen"].unsqueeze(0).to(device)
            m_chosen = batch["mask_chosen"].unsqueeze(0).to(device)
            x_rejected = batch["x_rejected"].unsqueeze(0).to(device)
            y_rejected = batch["y_rejected"].unsqueeze(0).to(device)
            m_rejected = batch["mask_rejected"].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                lp_c = float((logits_to_log_probs(model(x_chosen).logits.float(), y_chosen)
                              * m_chosen).sum())
                lp_r = float((logits_to_log_probs(model(x_rejected).logits.float(), y_rejected)
                              * m_rejected).sum())
                lr_c = float((logits_to_log_probs(ref_model(x_chosen).logits.float(), y_chosen)
                              * m_chosen).sum())
                lr_r = float((logits_to_log_probs(ref_model(x_rejected).logits.float(), y_rejected)
                              * m_rejected).sum())
            margins.append((lp_c - lp_r) - (lr_c - lr_r))
        model.train()
        n = max(len(margins), 1)
        return {"pairs": len(margins), "margin_mean": sum(margins) / n,
                "margin_positive_rate": sum(1 for m in margins if m > 0) / n}

    rng = np.random.default_rng(args.seed)
    metrics_path = outdir / "metrics.jsonl"
    run_start = time.time()
    step, cursor, tokens = start_step, 0, 0
    order = rng.permutation(len(train_ds))
    model.train()
    last_margin, step_start, count_log = None, time.time(), 0
    reason = "max_steps"
    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start + elapsed_done) > args.limit_seconds:
            reason = "time_limit"
            break
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if cursor + args.micro_bs > len(train_ds):
            order = rng.permutation(len(train_ds))
            cursor = 0
        idx = order[cursor:cursor + args.micro_bs]
        cursor += args.micro_bs
        batch = [train_ds[int(j)] for j in idx]
        x_chosen = torch.stack([b["x_chosen"] for b in batch]).to(device)
        x_rejected = torch.stack([b["x_rejected"] for b in batch]).to(device)
        y_chosen = torch.stack([b["y_chosen"] for b in batch]).to(device)
        y_rejected = torch.stack([b["y_rejected"] for b in batch]).to(device)
        mask_chosen = torch.stack([b["mask_chosen"] for b in batch]).to(device)
        mask_rejected = torch.stack([b["mask_rejected"] for b in batch]).to(device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        n_tokens = int(mask.sum())
        tokens += n_tokens
        count_log += n_tokens
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            with torch.no_grad():
                ref_logits = ref_model(x).logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            policy_logits = model(x).logits
            policy_log_probs = logits_to_log_probs(policy_logits, y)
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args.beta)
        scaler.scale(loss).backward()
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
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if log_now:
            upd_sq = sum(float(((p.detach() - b) ** 2).sum())
                         for p, b in zip(model.parameters(), before))
            del before, ref_logits, ref_log_probs, policy_logits, policy_log_probs
            dt = time.time() - step_start
            record = {"step": step, "loss": float(loss), "preference_tokens": tokens,
                      "grad_norm": float(grad_norm), "lr": lr,
                      "finite": bool(torch.isfinite(grad_norm).item()),
                      "update_norm": math.sqrt(upd_sq),
                      "preference_tokens_per_second": count_log / max(dt, 1e-9),
                      "log_seconds": dt, "step_seconds": dt / max(args.log_every, 1),
                      "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {step}/{args.max_steps} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm']:.3f} lr {lr:.2e} "
                f"{record['preference_tokens_per_second']:.0f} pref-tok/s")
            step_start = time.time()
            count_log = 0
        if step % args.eval_every == 0 or step == args.max_steps:
            held = margin_of(args.eval_pairs)
            last_margin = held["margin_mean"]
            record = {"step": step, "eval": "held-out", **held,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"  eval step {step}: margin 均值 {held['margin_mean']:.4f}，"
                f"正 margin {held['margin_positive_rate']:.3f}（{held['pairs']} 对）")
            if held["margin_mean"] > best_margin:
                best_margin = held["margin_mean"]
                save_checkpoint("best.pth", full=False)
            save_checkpoint("last.pth", full=True)
            if step % args.save_every == 0:
                save_checkpoint(f"step{step}.pth", full=False)
                snaps = sorted(outdir.glob("step*.pth"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "minimind_commit": subprocess.run(["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "init_from": str(args.init_from), "beta": args.beta, "stop_reason": reason,
        "steps": step, "preference_tokens_consumed": tokens,
        "best_heldout_margin": best_margin, "last_heldout_margin": last_margin,
        "elapsed_seconds": elapsed_done + time.time() - run_start,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "parity_checked": not args.skip_parity,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
    }
    (outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best held-out margin {best_margin:.4f}")


if __name__ == "__main__":
    main()
