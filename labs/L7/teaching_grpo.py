#!/usr/bin/env python3
"""可验证奖励的 GRPO 教学分支（7.6-J）：合成算术任务、规则奖励、rollout→reward→update→评测。

任务本身完全可验证：出题时就知道答案，判分只做"从回复里取最终整数"这一件事，不引入奖励
模型。奖励由三部分组成，前两项对应 7.6-B 的可验证奖励与格式约束，第三项沿用上游
`train_grpo.py` 的重复惩罚：

  correctness  最终整数等于 gold                       +1.00
  format       恰好一个 `</think>` 块且总长在 20–800    +0.25 / +0.25
  repetition   3-gram 重复比例，上限 0.5                −rep_penalty

GRPO 的组内基线就是同一 prompt 的 G 条采样奖励的均值与标准差；组内奖励完全相同（零方差）
时不产生优势，这类组单独计数。`--inner-epochs > 1` 会多次使用同一批 rollout，此时才出现
`ratio = exp(logp_current − logp_rollout)` 与 PPO 式裁剪；每步都记录 ratio 分位与裁剪比例。
`--beta-kl` 打开时按 token 加 KL 惩罚（k3 估计），reference 固定为 SFT 起点。

Usage:
    python labs/L7/teaching_grpo.py --build-data "$RUN/rl-data"
    python labs/L7/teaching_grpo.py --eval-only --task-dir "$RUN/rl-data" \
      --minimind-src "$SRC/minimind" --checkpoint "$RUN/sft/best.pt" \
      --split test --outdir "$RUN/rl-eval/base"
    python labs/L7/teaching_grpo.py --task-dir "$RUN/rl-data" --minimind-src "$SRC/minimind" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/rl-grpo" --max-steps 200
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

INT_RE = re.compile(r"-?\d+")
BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
THINK_RE = re.compile(r"</think>")


def log(message):
    print(f"[teaching_grpo] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_model_class(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433
    return MiniMindConfig, MiniMindForCausalLM, AutoTokenizer


def final_integer(text: str):
    boxed = BOXED_RE.findall(text)
    if boxed:
        return int(boxed[-1])
    nums = INT_RE.findall(text)
    return int(nums[-1]) if nums else None


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def score_response(text, gold, format_weight=1.0):
    """返回 (总奖励, 是否答对, 明细)。判分只用最终整数与结构，不读训练 loss。

    `format_weight=0` 时奖励退化为纯可验证项（只有答对得 1），用于把"格式分"从优势里拿掉。
    """
    got = final_integer(text)
    correct = int(got is not None and got == gold)
    think_count = len(THINK_RE.findall(text))
    detail = {"correct": correct, "final_integer": got,
              "format_think": 0.25 if think_count == 1 else 0.0,
              "format_length": 0.25 if 20 <= len(text.strip()) <= 800 else 0.0,
              "rep_penalty": rep_penalty(text)}
    total = (1.0 * correct
             + format_weight * (detail["format_think"] + detail["format_length"]
                                - detail["rep_penalty"]))
    return total, correct, detail


def build_data(task_dir: Path, n_train, n_val, n_test, difficulty, seed, task="arith"):
    task_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    count_len = {"easy": 5, "medium": 8, "hard": 12}[difficulty]

    def make(split, count):
        rows = []
        for i in range(count):
            if task == "arith":
                if difficulty == "easy":
                    a, b = rng.randint(1, 9), rng.randint(1, 9)
                elif difficulty == "medium":
                    a, b = rng.randint(10, 99), rng.randint(10, 99)
                else:
                    a, b = rng.randint(100, 999), rng.randint(10, 99)
                rows.append({"id": f"{split}-{i:05d}", "a": a, "b": b,
                             "prompt": f"请计算 {a}+{b} 的值。", "gold": a + b})
            else:
                # 计数任务：字母表 {A,B,C} 上长度 L 的串，至少一个 A；判分取最终整数
                while True:
                    text = "".join(rng.choice("ABC") for _ in range(count_len))
                    gold = text.count("A")
                    if gold > 0:
                        break
                rows.append({"id": f"{split}-{i:05d}", "string": text, "prompt":
                             f"字符串「{text}」里有几个字母 A？请以「答案：N」结尾。",
                             "gold": gold})
        return rows

    splits = {"train": make("train", n_train), "val": make("val", n_val),
              "test": make("test", n_test)}
    for split, rows in splits.items():
        (task_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    template = ("请计算 {a}+{b} 的值。" if task == "arith"
                else "字符串「{string}」里有几个字母 A？请以「答案：N」结尾。")
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "kind": f"verifiable-{task}", "task": task, "difficulty": difficulty,
                "count_len": count_len if task == "count" else None, "seed": seed,
                "counts": {k: len(v) for k, v in splits.items()},
                "reward": {"correct": 1.0, "format_think": 0.25, "format_length": 0.25,
                           "rep_penalty_cap": 0.5,
                           "rule": "取 \\boxed{N} 或回复中最后一个整数与 gold 比较"},
                "prompt_template": template, "splits_are_disjoint": True}
    (task_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"写出 {task_dir}：{manifest['counts']}，任务 {task}，难度 {difficulty}")


def load_policy(minimind_src, checkpoint, pth, hidden, layers, device):
    MiniMindConfig, MiniMindForCausalLM, _ = load_model_class(minimind_src)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers,
                                               use_moe=False)).to(device)
    step = None
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        step = state.get("step")
    elif pth:
        model.load_state_dict(torch.load(pth, map_location=device, weights_only=False))
    return model, step


def completion_logprobs(model, input_ids, completion_mask, dtype):
    """对 [prompt+completion] 求每个 completion token 的 logprob（当前权重）。

    训练时这一步必须带梯度（ratio 要对当前权重求导），rollout 记录旧策略 logprob 与
    reference 前向则由调用方包在 `torch.no_grad()` 里。
    """
    with torch.autocast("cuda", dtype=dtype, enabled=True):
        logits = model(input_ids).logits
    logprobs = logits[:, :-1, :].float().log_softmax(dim=-1)
    gathered = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return gathered * completion_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--build-data", type=Path, default=None)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--task", choices=["arith", "count"], default="arith",
                        help="arith: 两位数加法；count: 统计串中 A 的个数（长度随难度）")
    parser.add_argument("--minimind-src", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pth", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--micro-bs", type=int, default=16, help="每步的 prompt 数")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--format-weight", type=float, default=1.0,
                        help="格式与重复惩罚的权重；0 = 纯可验证奖励（只有答对得分）")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--inner-epochs", type=int, default=1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--keep-snapshots", type=int, default=3)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.build_data:
        build_data(args.build_data, args.n_train, args.n_val, args.n_test,
                   args.difficulty, args.seed, args.task)
        return
    if args.task_dir is None or args.minimind_src is None:
        raise SystemExit("需要 --task-dir 与 --minimind-src")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    _, _, AutoTokenizer = load_model_class(args.minimind_src)
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def read_split(split):
        return [json.loads(line) for line in
                (args.task_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]

    def chat_prompt(row):
        messages = [{"role": "user", "content": row["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return [i for i in tokenizer(text, add_special_tokens=False).input_ids
                if i != tokenizer.pad_token_id][-args.max_prompt_tokens:]

    @torch.no_grad()
    def batch_rollout(model, rows, sample):
        """返回 prompt/完整序列/完成 mask/logprob/文本/奖励。"""
        prompts = [chat_prompt(row) for row in rows]
        enc = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt").to(device)
        model.eval()
        if sample:
            out = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, num_return_sequences=args.group_size,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        else:
            out = model.generate(**enc, do_sample=False, num_return_sequences=1,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        prompt_len = enc["input_ids"].size(1)
        completions = out[:, prompt_len:]
        # 生成结果里 pad 与 eos 可能是同一个 id，所以按"第一个 eos 之前"算真实完成长度，
        # 不能直接用 (completions != pad) 当 mask。
        lengths = []
        eos_id = tokenizer.eos_token_id
        for row in completions.tolist():
            cut = len(row)
            for pos, tok in enumerate(row):
                if tok == eos_id:
                    cut = pos + 1
                    break
            lengths.append(cut)
        mask_full = torch.zeros_like(out, dtype=torch.long)
        for i, cut in enumerate(lengths):
            mask_full[i, prompt_len:prompt_len + cut] = 1
        # gathered logprob 对齐 input_ids[:, 1:]，所以 mask 也要整体左移一位
        mask = mask_full[:, 1:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        golds = [row["gold"] for row in rows for _ in range(completions.size(0) // len(rows))]
        rewards, corrects, details = [], [], []
        for text, gold in zip(texts, golds):
            total, correct, detail = score_response(text, gold, args.format_weight)
            rewards.append(total)
            corrects.append(correct)
            details.append(detail)
        with torch.no_grad():
            logp = completion_logprobs(model, out, mask, dtype)
        # generate 内部用 inference_mode 建张量，直接带进 autograd 会报
        # "Inference tensors cannot be saved for backward"，因此 clone 成普通张量
        return {"input_ids": out.clone(), "mask": mask.clone(), "texts": texts,
                "rewards": rewards, "corrects": corrects, "details": details,
                "logp": logp, "golds": golds, "lengths": lengths}

    def summarize(rollout):
        r = torch.tensor(rollout["rewards"], device=device)
        return {"reward_mean": float(r.mean()), "reward_std": float(r.std()),
                "accuracy": float(np.mean(rollout["corrects"])),
                "empty_rate": float(np.mean([len(t.strip()) == 0 for t in rollout["texts"]])),
                "mean_length": float(np.mean([len(t) for t in rollout["texts"]]))}

    if args.eval_only:
        if args.outdir is None:
            raise SystemExit("--eval-only 需要 --outdir")
        args.outdir.mkdir(parents=True, exist_ok=False)
        model, step = load_policy(args.minimind_src, args.checkpoint, args.pth,
                                  args.hidden_size, args.num_hidden_layers, device)
        rows = read_split(args.split)
        records, correct_total = [], 0
        for i in range(0, len(rows), args.micro_bs):
            batch = rows[i:i + args.micro_bs]
            rollout = batch_rollout(model, batch, sample=False)
            for row, text, correct, detail in zip(batch, rollout["texts"],
                                                  rollout["corrects"], rollout["details"]):
                correct_total += correct
                records.append({"id": row["id"], "prompt": row["prompt"], "gold": row["gold"],
                                "prediction": text, "correct": correct, "detail": detail})
        result = {"label": "rl-eval", "split": args.split, "rows": len(rows),
                  "accuracy": correct_total / max(len(rows), 1),
                  "checkpoint": str(args.checkpoint or args.pth), "checkpoint_step": step,
                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "task_manifest": json.loads((args.task_dir / "manifest.json").read_text())}
        (args.outdir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (args.outdir / "generations.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        log(f"{args.split} 集 {len(rows)} 题，正确率 {result['accuracy']:.4f}")
        return

    if args.init_from is None or args.outdir is None:
        raise SystemExit("训练需要 --init-from 与 --outdir")
    args.outdir.mkdir(parents=True, exist_ok=True)
    model, init_step = load_policy(args.minimind_src, args.init_from, None,
                                   args.hidden_size, args.num_hidden_layers, device)
    ref_model = None
    if args.beta_kl > 0:
        ref_model, _ = load_policy(args.minimind_src, args.init_from, None,
                                   args.hidden_size, args.num_hidden_layers, device)
        ref_model.eval()
        ref_model.requires_grad_(False)
    train_rows = read_split("train")
    val_rows = read_split("val")
    log(f"训练题 {len(train_rows)}，验证题 {len(val_rows)}，起点 {args.init_from}"
        f"（step={init_step}），组大小 {args.group_size}，micro-bs {args.micro_bs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics_path = args.outdir / "metrics.jsonl"
    run_start = time.time()
    step, cursor = 0, 0
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_rows))
    best_acc = -1.0
    last_val = None
    reason = "max_steps"

    def save_checkpoint(name):
        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, args.outdir / name)

    @torch.no_grad()
    def evaluate_split(rows, limit=None):
        rows = rows if limit is None else rows[:limit]
        correct, lengths = 0, []
        for i in range(0, len(rows), args.micro_bs):
            batch = rows[i:i + args.micro_bs]
            rollout = batch_rollout(model, batch, sample=False)
            correct += sum(rollout["corrects"])
            lengths += [len(t) for t in rollout["texts"]]
        return correct / max(len(rows), 1), float(np.mean(lengths))

    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start) > args.limit_seconds:
            reason = "time_limit"
            break
        if cursor + args.micro_bs > len(train_rows):
            order = rng.permutation(len(train_rows))
            cursor = 0
        idx = order[cursor:cursor + args.micro_bs]
        cursor += args.micro_bs
        batch = [train_rows[int(i)] for i in idx]
        step_start = time.time()
        rollout = batch_rollout(model, batch, sample=True)
        input_ids, mask = rollout["input_ids"], rollout["mask"]
        rewards = torch.tensor(rollout["rewards"], device=device).view(len(batch), args.group_size)
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, unbiased=False, keepdim=True)
        zero_var = int((group_std <= 1e-8).sum())
        advantages = (rewards - group_mean) / (group_std + 1e-8)
        advantages = advantages.view(-1)

        if args.beta_kl > 0:
            with torch.no_grad():
                ref_logp = completion_logprobs(ref_model, input_ids, mask, dtype)
            kl_per_token = (rollout["logp"] - ref_logp)
            kl_penalty = (torch.exp(-kl_per_token) - 1 + kl_per_token)  # k3 估计，>=0
            token_adv = advantages.unsqueeze(1) - args.beta_kl * kl_penalty
            kl_value = float((kl_per_token * mask).sum() / mask.sum().clamp(min=1))
        else:
            token_adv = advantages.unsqueeze(1)
            kl_value = 0.0

        model.train()
        old_logp = rollout["logp"].detach()
        losses, ratios, clip_fracs = [], [], []
        for _ in range(args.inner_epochs):
            new_logp = completion_logprobs(model, input_ids, mask, dtype)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * token_adv
            clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * token_adv
            per_token = -torch.min(unclipped, clipped)
            loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
            lr = get_lr(step, args.max_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss))
            ratios.append(ratio.detach())
            clip_fracs.append(float(((ratio - 1).abs() > args.clip_eps).float().mean()))
            step += 1
        ratio_all = torch.cat([r[mask.bool()] for r in ratios]) if ratios else torch.zeros(1)
        dt = time.time() - step_start
        record = {"step": step, "policy_version": step,
                  "loss": float(np.mean(losses)), "reward_mean": float(rewards.mean()),
                  "reward_std": float(rewards.std()), "group_reward_std_mean": float(group_std.mean()),
                  "zero_variance_groups": zero_var, "groups": len(batch),
                  "accuracy": float(np.mean(rollout["corrects"])),
                  "kl_per_token": kl_value,
                  "ratio_mean": float(ratio_all.mean()), "ratio_max": float(ratio_all.max()),
                  "ratio_min": float(ratio_all.min()), "clip_fraction": float(np.mean(clip_fracs)),
                  "empty_rate": float(np.mean([len(t.strip()) == 0 for t in rollout["texts"]])),
                  "mean_length": float(np.mean([len(t) for t in rollout["texts"]])),
                  "grad_norm": float(grad_norm), "lr": lr,
                  "step_seconds": dt, "completions_per_second": len(rollout["texts"]) / dt,
                  "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                  "elapsed": time.time() - run_start}
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"step {step} reward {record['reward_mean']:.4f}±{record['reward_std']:.4f} "
            f"acc {record['accuracy']:.3f} zero-var {zero_var}/{len(batch)} "
            f"ratio {record['ratio_mean']:.3f} len {record['mean_length']:.0f} "
            f"{record['step_seconds']:.2f}s")
        if step % args.eval_every < args.inner_epochs or step >= args.max_steps:
            acc, mean_len = evaluate_split(val_rows)
            last_val = acc
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "eval": "val", "val_accuracy": acc,
                                         "val_mean_length": mean_len,
                                         "elapsed": time.time() - run_start}) + "\n")
            log(f"  eval step {step}: val 正确率 {acc:.4f}，平均长度 {mean_len:.0f}")
            if acc >= best_acc:
                best_acc = acc
                save_checkpoint("best.pth")
            save_checkpoint("last.pth")
            if step % args.save_every < args.inner_epochs:
                save_checkpoint(f"step{step}.pth")
                snaps = sorted(args.outdir.glob("step*.pth"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()

    test_acc, test_len = evaluate_split(read_split("test"))
    summary = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "minimind_commit": subprocess.run(
                   ["git", "-C", str(args.minimind_src), "rev-parse", "HEAD"],
                   capture_output=True, text=True).stdout.strip(),
               "init_from": str(args.init_from), "init_step": init_step,
               "steps": step, "stop_reason": reason, "group_size": args.group_size,
               "micro_bs": args.micro_bs, "inner_epochs": args.inner_epochs,
               "beta_kl": args.beta_kl, "lr": args.lr, "temperature": args.temperature,
               "format_weight": args.format_weight,
               "max_new_tokens": args.max_new_tokens,
               "best_val_accuracy": best_acc, "last_val_accuracy": last_val,
               "test_accuracy": test_acc, "test_mean_length": test_len,
               "elapsed_seconds": time.time() - run_start,
               "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
               "task_manifest": json.loads((args.task_dir / "manifest.json").read_text())}
    (args.outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best val {best_acc:.4f}，test {test_acc:.4f}")


if __name__ == "__main__":
    main()
