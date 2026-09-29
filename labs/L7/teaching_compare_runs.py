#!/usr/bin/env python3
"""比较两条教学训练运行的轨迹（7.4-H 恢复对拍 / 7.9-I 限定窗口对照）。

对齐的是同一批 step 上的记录：连续运行与"中断后恢复"的后续窗口应当逐步一致，
限定窗口的精度/优化器改动则给出差异的量级与吞吐/峰值代价。

Usage:
    python labs/L7/teaching_compare_runs.py --run-a RUN/resume/A --run-b RUN/resume/B \
      --ckpt-a RUN/resume/A/last.pt --ckpt-b RUN/resume/B/last.pt \
      --label resume --outdir RUN/resume/compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_metrics(run_dir: Path):
    rows = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("eval"):
            rows[("eval", record["step"])] = record
        else:
            rows[("train", record["step"])] = record
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--ckpt-a", type=Path, default=None)
    parser.add_argument("--ckpt-b", type=Path, default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    a, b = load_metrics(args.run_a), load_metrics(args.run_b)
    common = sorted(set(a) & set(b))
    train_steps = [k for k in common if k[0] == "train"]
    eval_steps = [k for k in common if k[0] == "eval"]
    loss_deltas = [abs(a[k]["loss"] - b[k]["loss"]) for k in train_steps]
    grad_deltas = [abs(a[k]["grad_norm"] - b[k]["grad_norm"]) for k in train_steps]
    tps_a = [a[k]["tokens_per_second"] for k in train_steps]
    tps_b = [b[k]["tokens_per_second"] for k in train_steps]
    result = {
        "label": args.label, "run_a": str(args.run_a), "run_b": str(args.run_b),
        "common_train_steps": len(train_steps),
        "first_common_step": train_steps[0][1] if train_steps else None,
        "last_common_step": train_steps[-1][1] if train_steps else None,
        "max_abs_loss_delta": max(loss_deltas) if loss_deltas else None,
        "mean_abs_loss_delta": sum(loss_deltas) / len(loss_deltas) if loss_deltas else None,
        "max_abs_grad_norm_delta": max(grad_deltas) if grad_deltas else None,
        "tokens_per_second_a": sum(tps_a) / len(tps_a) if tps_a else None,
        "tokens_per_second_b": sum(tps_b) / len(tps_b) if tps_b else None,
        "val_losses": {str(k[1]): [a[k]["val_loss"], b[k]["val_loss"]] for k in eval_steps},
    }

    for name, run_dir in (("a", args.run_a), ("b", args.run_b)):
        summary = json.loads((run_dir / "run.json").read_text())
        result[f"run_{name}_summary"] = {key: summary[key] for key in
                                         ("stop_reason", "steps", "tokens_seen", "targets_seen",
                                          "best_val_loss", "test_loss", "elapsed_seconds",
                                          "peak_memory_mib") if key in summary}

    if args.ckpt_a and args.ckpt_b:
        ca = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
        cb = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
        param_diff = {}
        worst, worst_name = 0.0, None
        differing = 0
        for key in ca["model"]:
            delta = (ca["model"][key].float() - cb["model"][key].float()).abs().max().item()
            param_diff[key] = delta
            if delta > 0:
                differing += 1
            if delta > worst:
                worst, worst_name = delta, key
        result["checkpoint_compare"] = {
            "step_a": ca.get("step"), "step_b": cb.get("step"),
            "cursor_a": ca.get("cursor"), "cursor_b": cb.get("cursor"),
            "tokens_seen_a": ca.get("tokens_seen"), "tokens_seen_b": cb.get("tokens_seen"),
            "targets_seen_a": ca.get("targets_seen"), "targets_seen_b": cb.get("targets_seen"),
            "tensors": len(param_diff), "tensors_differing": differing,
            "max_abs_param_delta": worst, "max_abs_param_delta_tensor": worst_name,
        }

    (args.outdir / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
