#!/usr/bin/env python3
"""阶段权重导出（7.1-F / 7.5-I / 7.4-F）：把本项目的 checkpoint 转成上游可加载的 `.pth`。

本项目在训练中保存的是带 `model`/`optimizer`/`step` 的字典，上游 MiniMind 的
`init_model` 读取的是裸 `state_dict`（`out/<name>_<hidden>.pth`）。两者不是同一个东西：
前者是可继续训练的状态，后者只是权重。导出脚本把差别显式写下来，并落一份阶段清单。

Usage:
    python labs/L7/teaching_export.py --checkpoint "$RUN/pretrain/best.pt" \
      --out "$SRC/minimind/out/pretrain_768.pth" --stage pretrain --outdir "$RUN/export-pretrain"
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--outdir", required=True, type=Path,
                        help="阶段清单目录（新建）")
    parser.add_argument("--half", action="store_true", default=True,
                        help="上游 .pth 用 half 保存（默认开）")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = state["model"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.half() for k, v in model.items()} if args.half else model
    torch.save(payload, args.out)

    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": args.stage,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": state.get("step"),
        "checkpoint_val_loss": state.get("val_loss"),
        "exported": str(args.out),
        "dtype": "float16" if args.half else "float32",
        "tensors": len(payload),
        "bytes": args.out.stat().st_size,
        "excluded_state": [key for key in ("optimizer", "scaler", "torch_rng", "numpy_rng",
                                            "python_rng", "cursor") if key in state],
        "note": "只导出权重；可继续训练的状态（optimizer/RNG/游标）留在 checkpoint 里，"
                "两者用途不同",
    }
    (args.outdir / "export.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
