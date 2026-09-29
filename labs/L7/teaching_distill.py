#!/usr/bin/env python3
"""教学模型的蒸馏学生训练（7.7-J）：小教师/学生同词表，软标签在线蒸馏 + top-k 教师缓存。

三个 arms 共用一个循环，尽量保持"等数据、同步数、同起点、同 lr"：

  * `--alpha 0.5 --temperature 1.5`：CE + KL 混合（上游 `train_distillation.py` 的默认口径）
  * `--alpha 1.0`：纯监督 CE，作为等预算直接训练的对照
  * `--build-cache DIR` / `--teacher-cache DIR`：把教师监督离线成 top-k 缓存，训练时不再跑
    教师前向；缓存只覆盖前 `--cache-samples` 条样本，缓存 arm 也只在这些样本上训练。

在线损失与上游 `trainer/train_distillation.py` 的 `distillation_loss` 做数值对拍，对拍不通过
直接退出（`--skip-parity` 可显式跳过）。缓存的 KD 把 top-k 之外的词表合并成一个"其余"桶，
用学生同一支持集上的残差质量补齐。**合并尾部不是数值近似，而是换了一个目标**：KL 对类别
合并不保值，所以缓存 arm 的损失与在线损失不相等；`--build-cache` 会同时记录教师被截断的
质量与"合并尾部 KL / 全词表 KL"的差，缓存与在线两条 arm 的验证曲线另作对照。

Usage:
    python labs/L7/teaching_distill.py --data-dir "$RUN/data-sft" \
      --minimind-src "$SRC/minimind" --teacher-from "$RUN/sft/best.pt" \
      --outdir "$RUN/distill-kd" --student-hidden-size 512 \
      --student-num-layers 8 --alpha 0.5 --temperature 1.5 --max-steps 6000
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
    print(f"[teaching_distill] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def load_model_class(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    return MiniMindConfig, MiniMindForCausalLM


def load_upstream_loss(minimind_src: Path):
    path = minimind_src / "trainer" / "train_distillation.py"
    spec = importlib.util.spec_from_file_location("upstream_train_distill", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(minimind_src))
    spec.loader.exec_module(module)
    return module.distillation_loss


def kd_loss(student_logits, teacher_probs, temperature):
    """与上游一致：log_softmax(student/T) 对 softmax(teacher/T) 的 batchmean KL，再乘 T²。"""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return (temperature ** 2) * kl


def kd_loss_from_topk(student_logits, topk_ids, topk_logits, temperature):
    """top-k 支持集 + "其余"桶上的 KL（学生与教师用同一支持集）。

    这是把尾部合并成一个类别之后的 KL，不是全词表 KL 的近似——类别合并不保持 KL 的值。
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    student_topk = student_log_probs.gather(-1, topk_ids)
    teacher_log_probs = F.log_softmax(topk_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    residual_teacher = (1.0 - teacher_probs.sum(dim=-1)).clamp_min(0.0)
    residual_student = (1.0 - student_topk.exp().sum(dim=-1)).clamp_min(1e-12)
    term = (teacher_probs * (teacher_log_probs - student_topk)).sum(dim=-1)
    term = term + residual_teacher * (residual_teacher.clamp_min(1e-12).log()
                                      - residual_student.log())
    return (temperature ** 2) * term.mean(), float(residual_teacher.mean())


def flat_logits(model, x, dtype):
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        logits = model(x).logits
    head = logits[..., :-1, :]
    return head.reshape(-1, head.size(-1))


def mask_of(labels):
    """与 logits[..., :-1] 对齐的有效位置 mask，形状展平。"""
    return (labels[..., 1:] != -100).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--teacher-from", type=Path, required=True,
                        help="教师 checkpoint（本项目 teaching_sft.py 的 best.pt）")
    parser.add_argument("--teacher-hidden-size", type=int, default=768)
    parser.add_argument("--teacher-num-layers", type=int, default=8)
    parser.add_argument("--student-hidden-size", type=int, default=512)
    parser.add_argument("--student-num-layers", type=int, default=8)
    parser.add_argument("--student-from", type=Path, default=None,
                        help="学生初始化 checkpoint；缺省为随机初始化")
    parser.add_argument("--alpha", type=float, default=0.5, help="总损失 = alpha*CE + (1-alpha)*KD")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--micro-bs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-snapshots", type=int, default=3)
    parser.add_argument("--teacher-cache", type=Path, default=None,
                        help="使用已有的 top-k 教师缓存目录（训练只覆盖缓存中的样本）")
    parser.add_argument("--build-cache", type=Path, default=None,
                        help="构建 top-k 教师缓存到该目录后退出")
    parser.add_argument("--cache-samples", type=int, default=512)
    parser.add_argument("--cache-topk", type=int, default=64)
    parser.add_argument("--subset", type=int, default=0,
                        help="只用训练集前 N 条（0 = 全部）；缓存的 A/B 用同一子集")
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
    MiniMindConfig, MiniMindForCausalLM = load_model_class(args.minimind_src)

    if not args.skip_parity:
        upstream = load_upstream_loss(args.minimind_src)
        torch.manual_seed(0)
        student_probe = torch.randn(5, 17)
        teacher_probe = torch.randn(5, 17)
        mine = float(kd_loss(student_probe, F.softmax(teacher_probe / 1.5, dim=-1).detach(), 1.5))
        theirs = float(upstream(student_probe, teacher_probe, temperature=1.5))
        if abs(mine - theirs) > 1e-9:
            raise SystemExit(f"与上游 distillation_loss 对拍失败：{mine} vs {theirs}")
        log(f"与上游 distillation_loss 对拍通过：{mine:.10f}")

    train = np.load(args.data_dir / "train.npz")
    val = np.load(args.data_dir / "val.npz")
    train_inputs = torch.from_numpy(train["input_ids"].astype(np.int64))
    train_labels = torch.from_numpy(train["labels"].astype(np.int64))
    val_inputs = torch.from_numpy(val["input_ids"].astype(np.int64))
    val_labels = torch.from_numpy(val["labels"].astype(np.int64))
    n_samples = len(train_inputs)
    if args.subset > 0:
        n_samples = min(n_samples, args.subset)
        train_inputs = train_inputs[:n_samples]
        train_labels = train_labels[:n_samples]
        log(f"只使用前 {n_samples} 条训练样本（--subset）")
    log(f"SFT 张量：train {n_samples}、val {len(val_inputs)}，"
        f"有效 answer token {int((train_labels != -100).sum())}")

    def new_model(hidden, layers):
        return MiniMindForCausalLM(
            MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_moe=False)).to(device)

    teacher = None
    if args.alpha < 1.0 and args.teacher_cache is None:
        teacher = new_model(args.teacher_hidden_size, args.teacher_num_layers)
        state = torch.load(args.teacher_from, map_location=device, weights_only=False)
        teacher.load_state_dict(state["model"])
        teacher.eval()
        teacher.requires_grad_(False)
        log(f"教师 {args.teacher_hidden_size}/{args.teacher_num_layers} 载入自 {args.teacher_from}"
            f"（{sum(p.numel() for p in teacher.parameters())/1e6:.3f}M 参数，冻结）")

    student = new_model(args.student_hidden_size, args.student_num_layers)
    if args.student_from:
        state = torch.load(args.student_from, map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        log(f"学生载入自 {args.student_from}")
    student_params = sum(p.numel() for p in student.parameters())
    log(f"学生 {args.student_hidden_size}/{args.student_num_layers}：{student_params/1e6:.3f}M 参数")

    if args.build_cache:
        if args.alpha >= 1.0 or teacher is None:
            raise SystemExit("构建教师缓存需要 alpha < 1 且不使用 --teacher-cache")
        target = args.build_cache
        target.mkdir(parents=True, exist_ok=True)
        ids_out, logits_out, offsets, trunc = [], [], [0], []
        started = time.time()
        n = min(args.cache_samples, n_samples)
        with torch.no_grad():
            for index in range(n):
                x = train_inputs[index:index + 1].to(device)
                y = train_labels[index:index + 1].to(device)
                flat = flat_logits(teacher, x, dtype).float()[mask_of(y)]
                if flat.numel() == 0:
                    offsets.append(offsets[-1])
                    continue
                probs = F.softmax(flat, dim=-1)
                top = probs.topk(args.cache_topk, dim=-1)
                ids_out.append(top.indices.to(torch.uint16 if flat.size(-1) < 65536 else torch.int32))
                logits_out.append(top.values.log().half())
                trunc.append(float((1.0 - top.values.sum(dim=-1)).mean()))
                offsets.append(offsets[-1] + flat.size(0))
        ids = torch.cat(ids_out)
        cached_values = torch.cat(logits_out)
        payload = {"topk_ids": ids, "topk_logits": cached_values,
                   "offsets": torch.tensor(offsets, dtype=torch.int64), "topk": args.cache_topk}
        torch.save(payload, target / "topk.pt")
        # 截断对目标值的影响：同一个随机初始化学生在同一批位置上，全词表 KL 与 top-k KL 之差
        student.eval()
        probe_n = min(8, n)
        online_kd, cached_kd = [], []
        with torch.no_grad():
            for index in range(probe_n):
                x = train_inputs[index:index + 1].to(device)
                y = train_labels[index:index + 1].to(device)
                valid = mask_of(y)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    student_logits = flat_logits(student, x, dtype)[valid].float()
                teacher_full = flat_logits(teacher, x, dtype).float()[valid]
                probs = F.softmax(teacher_full / args.temperature, dim=-1)
                online_kd.append(float(kd_loss(student_logits, probs.detach(), args.temperature)))
                start, end = int(offsets[index]), int(offsets[index + 1])
                if end > start:
                    kd_topk, _ = kd_loss_from_topk(student_logits, ids[start:end].to(device).long(),
                                                   cached_values[start:end].to(device).float(),
                                                   args.temperature)
                    cached_kd.append(float(kd_topk))
        student.train()
        probe = {"probe_samples": probe_n,
                 "online_kd_mean": sum(online_kd) / max(len(online_kd), 1),
                 "cached_kd_mean": sum(cached_kd) / max(len(cached_kd), 1)}
        probe["abs_diff"] = abs(probe["online_kd_mean"] - probe["cached_kd_mean"])
        probe["rel_diff"] = probe["abs_diff"] / max(abs(probe["online_kd_mean"]), 1e-12)
        bytes_used = (ids.numel() * ids.element_size()
                      + cached_values.numel() * cached_values.element_size()
                      + payload["offsets"].numel() * 8)
        manifest = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "samples": n, "positions": int(ids.shape[0]), "topk": args.cache_topk,
            "bytes": bytes_used, "bytes_per_sample": bytes_used / max(n, 1),
            "teacher_from": str(args.teacher_from),
            "teacher_config": f"{args.teacher_hidden_size}/{args.teacher_num_layers}",
            "truncated_mass_mean": sum(trunc) / max(len(trunc), 1),
            "truncated_mass_max": max(trunc) if trunc else None,
            "teacher_forward_seconds": time.time() - started,
            "truncation_probe": probe,
            "student_config": {"hidden_size": args.student_hidden_size,
                               "num_hidden_layers": args.student_num_layers},
            "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        log(f"缓存写出 {target}：{bytes_used/2**20:.1f} MiB，每条 {bytes_used/max(n,1)/1024:.1f} KiB，"
            f"截断质量 mean {manifest['truncated_mass_mean']:.3e}，"
            f"max {manifest['truncated_mass_max']:.3e}；"
            f"KL 对拍 online {probe['online_kd_mean']:.6f} vs top-k {probe['cached_kd_mean']:.6f}"
            f"（相对差 {probe['rel_diff']:.3e}）")
        return

    cache = None
    cache_count = 0
    if args.teacher_cache:
        cache = torch.load(args.teacher_cache / "topk.pt", map_location="cpu", weights_only=False)
        meta = json.loads((args.teacher_cache / "manifest.json").read_text())
        if cache["topk"] != args.cache_topk:
            raise SystemExit("缓存的 top-k 与 --cache-topk 不一致")
        cache_count = int(cache["offsets"].numel() - 1)
        log(f"教师缓存：{meta['samples']} 条样本、top-{cache['topk']}、"
            f"{meta['bytes']/2**20:.1f} MiB，训练只在这 {cache_count} 条上迭代")

    usable = cache_count if cache is not None else n_samples
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    start_step, elapsed_done, best_val = 0, 0.0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        elapsed_done = state.get("elapsed", 0.0)
        best_val = state.get("best_val", float("inf"))
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        log(f"从 {args.resume} 恢复：step={start_step}")

    def save_checkpoint(name, full):
        payload = {"model": {k: v.detach().float().cpu() for k, v in student.state_dict().items()},
                   "step": step, "val_loss": last_val, "best_val": best_val,
                   "alpha": args.alpha, "temperature": args.temperature,
                   "student_config": {"hidden_size": args.student_hidden_size,
                                      "num_hidden_layers": args.student_num_layers}}
        if full:
            payload |= {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "elapsed": elapsed_done + time.time() - run_start,
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate()}
        torch.save(payload, outdir / name)

    @torch.no_grad()
    def evaluate_val():
        student.eval()
        total, count = 0.0, 0
        for i in range(0, len(val_inputs), args.micro_bs):
            x = val_inputs[i:i + args.micro_bs].to(device)
            y = val_labels[i:i + args.micro_bs].to(device)
            flat = flat_logits(student, x, dtype).float()
            valid = mask_of(y)
            total += float(F.cross_entropy(flat[valid], y[..., 1:].reshape(-1)[valid],
                                           reduction="sum"))
            count += int(valid.sum())
        student.train()
        return total / max(count, 1), count

    rng = np.random.default_rng(args.seed)
    metrics_path = outdir / "metrics.jsonl"
    run_start = time.time()
    step, cursor, answer_tokens = start_step, 0, 0
    order = rng.permutation(usable)
    student.train()
    last_val, step_start, count_log = None, time.time(), 0
    reason = "max_steps"
    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start + elapsed_done) > args.limit_seconds:
            reason = "time_limit"
            break
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if cursor + args.micro_bs > usable:
            order = rng.permutation(usable)
            cursor = 0
        idx = sorted(int(j) for j in order[cursor:cursor + args.micro_bs])
        cursor += args.micro_bs
        x = train_inputs[idx].to(device)
        y = train_labels[idx].to(device)
        valid = mask_of(y)
        count = int(valid.sum())
        answer_tokens += count
        count_log += count
        student_logits = flat_logits(student, x, dtype)[valid]
        target = y[..., 1:].reshape(-1)[valid]
        ce = F.cross_entropy(student_logits.float(), target, reduction="mean")
        kd_value, truncated_mass = 0.0, None
        if args.alpha >= 1.0:
            loss = ce
        elif cache is not None:
            positions = torch.cat([torch.arange(int(cache["offsets"][i]), int(cache["offsets"][i + 1]))
                                   for i in idx])
            if positions.numel() != student_logits.size(0):
                raise SystemExit("缓存位置数与当前 batch 的有效位置数不一致，缓存与数据不再对应")
            ids = cache["topk_ids"][positions].to(device).long()
            cached_values = cache["topk_logits"][positions].to(device).float()
            kd, truncated_mass = kd_loss_from_topk(student_logits, ids, cached_values,
                                                  args.temperature)
            kd_value = float(kd)
            loss = args.alpha * ce + (1 - args.alpha) * kd
        else:
            with torch.no_grad():
                teacher_logits = flat_logits(teacher, x, dtype)
                teacher_probs = F.softmax(teacher_logits[valid].float() / args.temperature,
                                          dim=-1).detach()
            kd = kd_loss(student_logits, teacher_probs, args.temperature)
            kd_value = float(kd)
            loss = args.alpha * ce + (1 - args.alpha) * kd
        scaler.scale(loss).backward()
        log_now = (step % args.log_every == 0) or (step + 1 == args.max_steps)
        before = [p.detach().clone() for p in student.parameters()] if log_now else None
        if args.precision == "fp16":
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        if args.precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if log_now:
            upd_sq = sum(float(((p.detach() - b) ** 2).sum())
                         for p, b in zip(student.parameters(), before))
            del before
            dt = time.time() - step_start
            record = {"step": step, "loss": float(loss), "ce": float(ce), "kd": kd_value,
                      "teacher_truncated_mass": truncated_mass, "answer_tokens": answer_tokens,
                      "grad_norm": float(grad_norm), "lr": lr,
                      "finite": bool(torch.isfinite(grad_norm).item()),
                      "update_norm": math.sqrt(upd_sq),
                      "answer_tokens_per_second": count_log / max(dt, 1e-9),
                      "log_seconds": dt, "step_seconds": dt / max(args.log_every, 1),
                      "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "elapsed": elapsed_done + time.time() - run_start}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"step {step}/{args.max_steps} loss {record['loss']:.4f} ce {record['ce']:.4f} "
                f"kd {record['kd']:.4f} gnorm {record['grad_norm']:.3f} lr {lr:.2e} "
                f"{record['answer_tokens_per_second']:.0f} ans-tok/s")
            step_start = time.time()
            count_log = 0
        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss, val_tokens = evaluate_val()
            last_val = val_loss
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "eval": "val", "val_loss": val_loss,
                                         "val_answer_tokens": val_tokens,
                                         "elapsed": elapsed_done + time.time() - run_start}) + "\n")
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
        "arm": "ce-only" if args.alpha >= 1.0 else ("cached-topk" if cache is not None else "online-kd"),
        "alpha": args.alpha, "temperature": args.temperature,
        "teacher_from": str(args.teacher_from),
        "teacher_params": (sum(p.numel() for p in teacher.parameters()) if teacher else None),
        "teacher_cache_samples": cache_count or None,
        "student_params": student_params,
        "student_config": {"hidden_size": args.student_hidden_size,
                           "num_hidden_layers": args.student_num_layers},
        "stop_reason": reason, "steps": step, "answer_tokens_consumed": answer_tokens,
        "best_val_loss": best_val, "last_val_loss": last_val,
        "elapsed_seconds": elapsed_done + time.time() - run_start,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
    }
    (outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
