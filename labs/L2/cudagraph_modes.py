#!/usr/bin/env python3
"""L2.6b 证据补齐 · 三种 cudagraph_mode 下真实的 launch 数与 kernel 数。

2.6b 里写过 vLLM v1 默认是 FULL_AND_PIECEWISE，但**没有量过它到底省了多少次
launch**。这个脚本在同一份 decode 负载上跑三种模式，用 nsys 数：

  - CUDA API 侧：cuLaunchKernel 家族 vs cuGraphLaunch 各多少次
  - GPU 侧：实际执行了多少个 kernel

关键点：**图执行不会减少 GPU 上执行的 kernel 数**，它减少的是 CPU 侧的
launch API 调用数。这两个数分开数，才能看出图到底换掉了什么。

用法（由 cudagraph_modes.sh 驱动，每种模式一个干净进程）：
    python cudagraph_modes.py --mode FULL_AND_PIECEWISE
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["NONE", "PIECEWISE", "FULL_AND_PIECEWISE",
                             "FULL", "FULL_DECODE_ONLY"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 见 L2.6 踩坑 5

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enforce_eager=(args.mode == "NONE"),
        compilation_config={"cudagraph_mode": args.mode}
             if args.mode != "NONE" else None,
        gpu_memory_utilization=0.55,
        max_model_len=1024,
        enable_prefix_caching=False,     # 见 L0 踩坑 7：默认开着会污染 prefill
        disable_log_stats=True,
    )

    # 固定长度、互不相同的 prompt，避免任何缓存命中
    import random
    rng = random.Random(20260910)
    prompts = [
        " ".join(str(rng.randint(10000, 99999)) for _ in range(args.prompt_len // 2))
        for _ in range(args.bs)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True)

    llm.generate(prompts, sp)            # 预热 + 触发图捕获
    import torch
    torch.cuda.synchronize()

    # ===== 计数区间：nsys 的 profile 只在这一段打开 =====
    torch.cuda.cudart().cudaProfilerStart()
    out = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    ntok = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"MODE={args.mode} bs={args.bs} gen={args.gen} 实际生成 token={ntok}")
    import sys as _s; _s.stdout.flush(); _s.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
