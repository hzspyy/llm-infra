#!/usr/bin/env python3
"""教学模型的 LoRA 领域适配（7.5-J）：在 7.5-I 的 SFT 权重上只训练低秩旁路。

复用上游 MiniMind 的三处实现，保证与 `trainer/train_lora.py` 同一目标：

  * `model.model_lora.apply_lora/save_lora/load_lora/merge_lora`：注入哪些模块（所有
    方阵 Linear，rank 由 `--rank` 给定）、B=0 初始化、只存 LoRA 权重、合并方式；
  * `dataset.lm_dataset.SFTDataset`：同一条 chat template、同一份 assistant 段 label 规则；
  * `model.model_minimind.MiniMindForCausalLM(labels=...)` 的 CE。

本项目新增的是测量：逐步 loss/grad/update、LoRA 参数占比、held-out 领域子技能曲线
（`<tool_call>` 出现率与 JSON 可解析率）、最优 checkpoint 选择、断点续训与合并导出。

Usage:
    python labs/L7/teaching_lora.py --data-dir "$RUN/data-lora" \
      --minimind-src "$SRC/minimind" --init-from "$RUN/sft/best.pt" \
      --outdir "$RUN/lora" --max-steps 1000 --rank 16
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


def log(message):
    print(f"[teaching_lora] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_minimind(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    from model.model_lora import apply_lora, load_lora, merge_lora  # noqa: WPS433
    from dataset.lm_dataset import SFTDataset  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433
    return (MiniMindConfig, MiniMindForCausalLM, apply_lora, load_lora, merge_lora,
            SFTDataset, AutoTokenizer)


def lora_param_names(model):
    return [name for name, _ in model.named_parameters() if "lora" in name]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--init-from", type=Path, required=True,
                        help="SFT 阶段 checkpoint（本项目 teaching_sft.py 的 best.pt）")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--micro-bs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-snapshots", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=48)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
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
    (MiniMindConfig, MiniMindForCausalLM, apply_lora, load_lora, merge_lora,
     SFTDataset, AutoTokenizer) = load_minimind(args.minimind_src)

    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model")
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    init = torch.load(args.init_from, map_location=device, weights_only=False)
    model.load_state_dict(init["model"])
    log(f"初始化自 {args.init_from}（step={init.get('step')} val={init.get('val_loss')}）")

    apply_lora(model, rank=args.rank)
    total_params = sum(p.numel() for p in model.parameters())
    names = lora_param_names(model)
    lora_params = []
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
        if param.requires_grad:
            lora_params.append(param)
    lora_count = sum(p.numel() for p in lora_params)
    log(f"LoRA 注入 {len(names)} 个张量 / {lora_count/1e6:.3f}M 参数，占全模型 "
        f"{lora_count/total_params*100:.2f}%（rank={args.rank}）")
    assert names, "apply_lora 未注入任何模块"

    train_ds = SFTDataset(str(args.data_dir / "train.jsonl"), tokenizer, max_length=args.max_len)
    val_ds = SFTDataset(str(args.data_dir / "val.jsonl"), tokenizer, max_length=args.max_len)
    log(f"LoRA 训练样本 {len(train_ds)}，验证样本 {len(val_ds)}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    start_step, elapsed_done, best_val = 0, 0.0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        elapsed_done = state.get("elapsed", 0.0)
        best_val = state.get("best_val", float("inf"))
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        log(f"从 {args.resume} 恢复：step={start_step}")

    def lora_state():
        state = {name: param.detach().half().cpu() for name, param in model.named_parameters()
                 if "lora" in name}
        return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    def save_checkpoint(name, full):
        torch.save(lora_state(), outdir / name)
        if full:
            torch.save({"model": lora_state(), "step": step, "best_val": best_val,
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "elapsed": elapsed_done + time.time() - run_start,
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate(), "args": str(vars(args))},
                       outdir / "last_state.pt")

    @torch.no_grad()
    def evaluate_val():
        model.eval()
        total, count = 0.0, 0
        for i in range(0, len(val_ds), args.micro_bs):
            batch = [val_ds[j] for j in range(i, min(i + args.micro_bs, len(val_ds)))]
            x = torch.stack([b[0] for b in batch]).to(device)
            y = torch.stack([b[1] for b in batch]).to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                out = model(x, labels=y)
            valid = (y[..., 1:] != -100)
            n = int(valid.sum())
            total += float(out.loss) * max(n, 1)
            count += n
        model.train()
        return total / max(count, 1), count

    # 领域子技能曲线：held-out test.jsonl 前 N 条，只判 `<tool_call>` 是否出现与 JSON 可否解析
    test_rows = [json.loads(line) for line in (args.data_dir / "test.jsonl").read_text().splitlines()
                 if line.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import teaching_preference_eval as tpe  # noqa: E402

    @torch.no_grad()
    def evaluate_skill():
        model.eval()
        n = min(args.eval_prompts, len(test_rows))
        emitted = parsed = 0
        for row in test_rows[:n]:
            prefix, tools = tpe.normalize_messages(row["conversations"][:-1])
            prompt = tokenizer.apply_chat_template(prefix, tokenize=False,
                                                   add_generation_prompt=True, tools=tools)
            prompt_ids = [i for i in tokenizer(prompt).input_ids if i != tokenizer.pad_token_id]
            out_ids = tpe.te.generate(model, prompt_ids, args.eval_max_new_tokens,
                                      tokenizer.eos_token_id, device, dtype, args.max_len)
            pred = tokenizer.decode(out_ids)
            if "<tool_call>" in pred:
                emitted += 1
                body = pred.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
                try:
                    json.loads(body)
                    parsed += 1
                except json.JSONDecodeError:
                    pass
        model.train()
        return {"prompts": n, "tool_call_emission_rate": emitted / max(n, 1),
                "tool_call_json_parse_rate": parsed / max(n, 1)}

    rng = np.random.default_rng(args.seed)
    metrics_path = outdir / "metrics.jsonl"
    run_start = time.time()
    step, cursor, answer_tokens = start_step, 0, 0
    order = rng.permutation(len(train_ds))
    model.train()
    last_val, step_start, count_log = None, time.time(), 0
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
        x = torch.stack([b[0] for b in batch]).to(device)
        y = torch.stack([b[1] for b in batch]).to(device)
        valid = (y[..., 1:] != -100)
        count = int(valid.sum())
        answer_tokens += count
        count_log += count
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            out = model(x, labels=y)
        scaler.scale(out.loss).backward()
        log_now = (step % args.log_every == 0) or (step + 1 == args.max_steps)
        before = [p.detach().clone() for p in lora_params] if log_now else None
        if args.precision == "fp16":
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        if args.precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if log_now:
            upd_sq = sum(float(((p.detach() - b) ** 2).sum())
                         for p, b in zip(lora_params, before))
            del before
            dt = time.time() - step_start
            record = {"step": step, "loss": float(out.loss), "answer_tokens": answer_tokens,
                      "grad_norm": float(grad_norm), "lr": lr,
                      "finite": bool(torch.isfinite(grad_norm).item()),
                      "update_norm": math.sqrt(upd_sq),
                      "answer_tokens_per_second": count_log / max(dt, 1e-9),
                      "log_seconds": dt, "step_seconds": dt / max(args.log_every, 1),
                      "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {step}/{args.max_steps} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm']:.3f} lr {lr:.2e} "
                f"{record['answer_tokens_per_second']:.0f} ans-tok/s")
            step_start = time.time()
            count_log = 0
        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss, val_tokens = evaluate_val()
            skill = evaluate_skill()
            last_val = val_loss
            record = {"step": step, "eval": "val", "val_loss": val_loss,
                      "val_answer_tokens": val_tokens, **skill,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"  eval step {step}: val_loss {val_loss:.4f}（{val_tokens} answer tokens）"
                f"、tool_call {skill['tool_call_emission_rate']:.3f}、"
                f"JSON {skill['tool_call_json_parse_rate']:.3f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best.pth", full=False)
            save_checkpoint("last.pth", full=True)
            if step % args.save_every == 0:
                save_checkpoint(f"step{step}.pth", full=False)
                snaps = sorted(outdir.glob("step*.pth"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()

    # 合并导出：base + B@A 写进权重，与未合并推理对拍由 teaching_preference_eval.py 负责
    merge_lora(model, str(outdir / "best.pth"), str(outdir / "merged_best.pth"))
    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "minimind_commit": subprocess.run(["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "total_params": total_params, "lora_params": lora_count,
        "lora_param_share": lora_count / total_params, "lora_tensors": len(names),
        "lora_module_names": names, "rank": args.rank,
        "init_from": str(args.init_from), "stop_reason": reason, "steps": step,
        "answer_tokens_consumed": answer_tokens, "best_val_loss": best_val,
        "last_val_loss": last_val,
        "elapsed_seconds": elapsed_done + time.time() - run_start,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
    }
    (outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
