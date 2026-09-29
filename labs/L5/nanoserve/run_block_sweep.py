#!/usr/bin/env python3
"""L5.7 任务 B 续 —— 块大小 1/16/64 的拐点：分页 kernel 的块内循环效率。

5.7 的消融列表问过"块越小前缀复用越细，代价在哪里"。
块大小同时改变三件事：块表长度与每步 metadata、分页 kernel 的内层循环长度
（`BLOCK_N`）、以及共享前缀能命中的粒度。这个脚本把前两件单独量出来：
固定总槽位数（1024），块大小取 1/16/64，两种执行器各跑同一条轨迹。

块大小只影响寻址与内层循环；`BLOCK_N` 就是块大小，所以块越小 kernel 的
分页循环次数越多（次数 ≈ 上下文/块大小），这正是要看的那条曲线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
TOTAL_SLOTS = 1024
MODELS = {}


def model_for(block_size, cls):
    key = (block_size, cls.__name__)
    if key not in MODELS:
        from transformers import AutoTokenizer  # noqa: F401  (仅用于提示依赖)
        MODELS[key] = cls(REPO, HUB, num_blocks=TOTAL_SLOTS // block_size,
                          block_size=block_size)
    return MODELS[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-reps", type=int, default=2)
    ap.add_argument("--total-slots", type=int, default=1024,
                    help="块池总槽位数；长上下文对照要一起调大")
    ap.add_argument("--max-batched-tokens", type=int, default=256)
    args = ap.parse_args()
    globals()["TOTAL_SLOTS"] = args.total_slots
    args.out.mkdir(parents=True, exist_ok=False)

    from engine import Engine, Request, State
    from paged_engine import PagedEngine, PagedModelExec
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    text = "Continue this story about a lighthouse keeper. " * args.prompt_reps
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))

    rows = []
    for bs in args.block_sizes:
        for tag, cls in [("base", Engine), ("paged", PagedEngine)]:
            model = model_for(bs, PagedModelExec if tag == "paged" else PagedModelExec)
            eng = cls(model, block_size=bs, max_batched_tokens=args.max_batched_tokens,
                      max_num_seqs=args.batch, enable_prefix_cache=False, eos_ids=[])
            reqs = [Request(f"r{i}", ids, max_tokens=args.max_tokens)
                    for i in range(args.batch)]
            for r in reqs:
                eng.add(r)
            t0 = time.perf_counter()
            eng.run_until_idle(max_steps=20000)
            wall = time.perf_counter() - t0
            row = dict(block_size=bs, engine=tag, wall_s=round(wall, 4),
                       steps=eng.step_index, context=len(ids) + args.max_tokens,
                       blocks_per_seq=(len(ids) + args.max_tokens + bs - 1) // bs,
                       paged_steps=getattr(eng, "paged_steps", 0),
                       leak=eng.leak_check()["leaked"],
                       produced={r.req_id: len(r.output_ids) for r in reqs})
            rows.append(row)
            print(f"block_size={bs:>3} {tag:<6} wall {wall:>6.3f}s  "
                  f"steps {eng.step_index:>3}  每序列块数 {row['blocks_per_seq']:>4}  "
                  f"泄漏 {row['leak']}")
        base = next(r for r in rows if r["block_size"] == bs and r["engine"] == "base")
        pg = next(r for r in rows if r["block_size"] == bs and r["engine"] == "paged")
        print(f"           → 分页/基类 = {pg['wall_s'] / base['wall_s']:.3f}")

    (args.out / "block_sweep.json").write_text(
        json.dumps(dict(total_slots=TOTAL_SLOTS, batch=args.batch,
                        max_tokens=args.max_tokens, prompt_tokens=len(ids),
                        rows=rows,
                        note="块大小同时改块表长度与分页 kernel 的内层循环长度；"
                             "总槽位固定 1024，所以块越小块表越长"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读法：块越小，块表与 metadata 越长、kernel 内层循环次数越多；")
    print("      两条曲线一起看才能判断拐点在哪一侧。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
