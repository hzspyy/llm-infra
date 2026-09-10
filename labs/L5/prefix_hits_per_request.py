#!/usr/bin/env python3
"""L5.2 证据补齐 · 逐请求的前缀命中量与调度 token 数。

5.2 的阶梯实验只有**计时**。计时相近不能证明块边界，
所以那一节的块数结论目前只是公式预测（STATUS.md §5 第 2 条）。
这个脚本直接把两个量取出来，和公式预测并排放：

  - `KVCacheManager.get_computed_blocks(request)` 的第二个返回值
    = 这次请求真正命中的 token 数（一定是块对齐的）
  - `SchedulerOutput.num_scheduled_tokens[req_id]`
    = 这一步真正送去算的 token 数

于是「命中了几块」不再靠计时反推，而是直接读数。

用法：
    python prefix_hits_per_request.py --block-size 16
"""

import argparse
import json
import os


HITS = []          # (req_id, num_prompt_tokens, num_cached_tokens)
STEPS = []         # (step, {req_id: num_scheduled_tokens})


def install_hooks():
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_gcb = KVCacheManager.get_computed_blocks

    def gcb(self, request):
        out = orig_gcb(self, request)
        # 返回值第 2 项是命中的 token 数
        HITS.append((request.request_id, request.num_tokens, int(out[1])))
        return out

    KVCacheManager.get_computed_blocks = gcb

    orig_sched = Scheduler.schedule

    def sched(self, *a, **kw):          # 0.29 的 schedule() 带 throttle 参数
        out = orig_sched(self, *a, **kw)
        if out.total_num_scheduled_tokens:
            STEPS.append((len(STEPS), dict(out.num_scheduled_tokens)))
        return out

    Scheduler.schedule = sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    install_hooks()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, enable_prefix_caching=True,
              block_size=args.block_size, gpu_memory_utilization=0.55,
              max_model_len=2048, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

    B = args.block_size

    def run(label, prompt_ids):
        from vllm import TokensPrompt
        HITS.clear(); STEPS.clear()
        llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp, use_tqdm=False)
        n = len(prompt_ids)
        hit = HITS[0][2] if HITS else 0
        sched_first = list(STEPS[0][1].values())[0] if STEPS else 0
        return dict(label=label, prompt_tokens=n, cached_tokens=hit,
                    cached_blocks=hit // B,
                    scheduled_first_step=sched_first,
                    predicted_recompute=n - hit)

    base = tok("The capital of France is Paris. " * 40,
               add_special_tokens=False)["input_ids"]
    rows = []

    print("=== 实验一：完全相同的 prompt 重复三次 ===")
    for i in range(3):
        rows.append(run(f"identical #{i}", list(base)))
        r = rows[-1]
        print(f"  第{i}次  prompt={r['prompt_tokens']:>5}  "
              f"命中 token={r['cached_tokens']:>5} "
              f"(= {r['cached_blocks']} 块 × {B})  "
              f"首步调度 token={r['scheduled_first_step']:>5}")

    print()
    print("=== 实验二：共享前缀 + 不同尾巴，扫过一个块边界 ===")
    print(f"块大小 = {B}。公式预测：命中 = floor(共享长度 / {B}) × {B}，")
    print(f"但完整命中时末位仍要重算（见 5.2 引用的 vLLM 注释）。")
    print(f"  {'共享前缀长度':>12} {'prompt':>7} {'命中token':>9} {'命中块':>7} "
          f"{'首步调度':>9} {'公式预测命中块':>14}")
    for share in range(B * 3 - 2, B * 3 + 6):
        p = list(base[:share]) + [tok.encode(f" zzz{share}",
                                             add_special_tokens=False)[0]] * 8
        r = run(f"share={share}", p)
        rows.append(r)
        print(f"  {share:>12} {r['prompt_tokens']:>7} {r['cached_tokens']:>9} "
              f"{r['cached_blocks']:>7} {r['scheduled_first_step']:>9} "
              f"{share // B:>14}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"\n写入 {args.out}")
    import sys as _s; _s.stdout.flush(); _s.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
