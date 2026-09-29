#!/usr/bin/env python3
"""教学项目的成本账（7.11-I）：把各阶段的墙钟、有效 token、峰值与存储折成 GPU 小时。

口径：
  * GPU 小时 = 该进程墙钟 / 3600（单卡独占时段），不把数据准备（CPU）折进去
  * 有效 token 取 run.json 的 targets_seen / answer_tokens，不用输入 token 冒充
  * 机制对拍（resume/ablation）单独列，计入总预算但不计进"项目训练"两阶段
  * 存储分"数据"和"checkpoint"两类；checkpoint 只统计本次运行目录

Usage:
    python labs/L7/teaching_cost_report.py --run-dir "$RUN" --outdir "$RUN/cost"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize_run(run_dir: Path, kind: str):
    summary = read_json(run_dir / "run.json")
    if summary is None:
        return None
    ckpt_bytes = sum(p.stat().st_size for p in run_dir.glob("*.pt"))
    return {
        "kind": kind, "dir": str(run_dir),
        "steps": summary.get("steps"), "tokens": summary.get("targets_seen")
        or summary.get("tokens_seen") or summary.get("answer_tokens"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "gpu_hours": round((summary.get("elapsed_seconds") or 0) / 3600, 4),
        "peak_memory_mib": summary.get("peak_memory_mib"),
        "test_loss": summary.get("test_loss"), "best_val_loss": summary.get("best_val_loss"),
        "stop_reason": summary.get("stop_reason"),
        "checkpoint_bytes": ckpt_bytes,
    }


def dir_bytes(path: Path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--budget-gpu-hours", type=float, default=12.0,
                        help="文本预训练 + SFT 的规划上限")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    projects = [run for run in (
        summarize_run(args.run_dir / "pretrain", "pretrain"),
        summarize_run(args.run_dir / "sft", "sft")) if run]
    mechanisms = [run for run in (
        summarize_run(args.run_dir / "pilot", "pilot"),
        summarize_run(args.run_dir / "resume" / "A", "resume-A"),
        summarize_run(args.run_dir / "resume" / "B", "resume-B"),
        summarize_run(args.run_dir / "resume" / "C", "resume-C"),
        summarize_run(args.run_dir / "ablation" / "bf16-adamw", "ablation-bf16-adamw"),
        summarize_run(args.run_dir / "ablation" / "bf16-fused", "ablation-bf16-fused"),
        summarize_run(args.run_dir / "ablation" / "fp16-adamw", "ablation-fp16-adamw")) if run]

    data_dirs = [d for d in (args.run_dir / "data-pretrain", args.run_dir / "data-sft") if d.exists()]
    data_bytes = sum(dir_bytes(d) for d in data_dirs)
    data_manifests = {d.name: read_json(d / "manifest.json") for d in data_dirs}

    project_hours = sum(run["gpu_hours"] for run in projects)
    result = {
        "run_dir": str(args.run_dir),
        "projects": projects,
        "mechanism_runs": mechanisms,
        "project_gpu_hours": round(project_hours, 4),
        "mechanism_gpu_hours": round(sum(run["gpu_hours"] for run in mechanisms), 4),
        "total_gpu_hours": round(project_hours + sum(run["gpu_hours"] for run in mechanisms), 4),
        "budget_gpu_hours": args.budget_gpu_hours,
        "data_bytes": data_bytes,
        "data_tokens": {name: {split: values["tokens"] for split, values in
                               (manifest or {}).get("splits", {}).items()}
                        for name, manifest in data_manifests.items() if manifest},
        "checkpoint_bytes_total": sum(run["checkpoint_bytes"]
                                      for run in projects + mechanisms),
    }
    (args.outdir / "cost.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    header = (f"{'阶段':<20} {'step':>7} {'有效 token':>14} {'墙钟 s':>10} "
              f"{'GPU 小时':>10} {'峰值 MiB':>10} {'ckpt B':>16}")
    print(header)
    print("-" * len(header))
    for run in projects + mechanisms:
        print(f"{run['kind']:<20} {run['steps'] or 0:>7} {(run['tokens'] or 0):>14,} "
              f"{(run['elapsed_seconds'] or 0):>10.1f} {run['gpu_hours']:>10.4f} "
              f"{(run['peak_memory_mib'] or 0):>10.0f} {run['checkpoint_bytes']:>16,}")
    print(f"\n项目两阶段 GPU 小时 {result['project_gpu_hours']:.4f} / 预算 {args.budget_gpu_hours}"
          f"；机制对拍 {result['mechanism_gpu_hours']:.4f}；合计 {result['total_gpu_hours']:.4f}")
    print(f"数据 {data_bytes:,} B；checkpoint {result['checkpoint_bytes_total']:,} B")


if __name__ == "__main__":
    main()
