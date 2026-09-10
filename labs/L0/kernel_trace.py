#!/usr/bin/env python3
"""L0 lab · 调用栈的最后三层：Python → CUDA kernel → SASS。

前面 map_stack.py 停在 Python 边界。这个脚本继续往下：
  1. 用 torch.profiler 抓出**一次 prefill** 和**一次 decode step** 实际下发了哪些 kernel、
     各占多少时间；
  2. 统计每步的 kernel 数量——这个数字直接决定 CUDA Graph 有没有用；
  3. 把其中一个 kernel 的机器码（SASS）dump 出来，让「算子」这个词落到实处。

用法：
    python kernel_trace.py --model /path/to/Qwen3-1.7B --out-dir results/
"""

from __future__ import annotations

import argparse
import random
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# 关键：vLLM 默认把 EngineCore 放在**独立进程**里跑（L0.1 第 6 层，为了绕开 GIL）。
# 那样 torch.profiler 在父进程里什么都抓不到——第一次跑这个脚本就是这么翻车的。
# 关掉多进程，让引擎和 profiler 在同一个解释器里。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def summarize(prof, top: int = 25) -> dict:
    """把 profiler 事件汇总成「kernel 名 → 次数 / 总时长」。

    这里有个必须避开的坑：`key_averages()` 同时包含 **CPU 侧算子记录**和
    **GPU 侧 kernel 记录**。前者的 device_time 是它所有子事件的累加，
    所以 `aten::linear` / `aten::matmul` / `aten::mm` 三层嵌套会把同一段 GPU
    时间报三遍。直接求和会得到 3 倍虚高的总时长。

    正确做法是只保留 device_type == CUDA 的事件——那才是真正下发到 GPU 的 kernel。
    """
    from torch.autograd import DeviceType

    kernels = defaultdict(lambda: {"count": 0, "us": 0.0})
    total_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        dev_us = getattr(evt, "device_time_total", 0.0) or 0.0
        if dev_us <= 0:
            continue
        kernels[evt.key]["count"] += evt.count
        kernels[evt.key]["us"] += dev_us
        total_us += dev_us

    ranked = sorted(kernels.items(), key=lambda kv: -kv[1]["us"])
    return {
        "total_device_us": round(total_us, 1),
        "distinct_kernels": len(kernels),
        "total_launches": sum(v["count"] for v in kernels.values()),
        "top": [
            {"kernel": k, "count": v["count"], "us": round(v["us"], 1),
             "share": round(v["us"] / total_us, 4) if total_us else 0}
            for k, v in ranked[:top]
        ],
    }


def clean_timing(llm, prompt, sp_cls, prompt_len: int, decode_steps: int,
                 repeats: int = 5) -> dict:
    """不开 profiler 的干净计时。

    profiler 会给每个 kernel 加上记录开销，用它测出来的绝对时间是偏高的。
    成分分析用 profiler，绝对时间用这里——两件事必须分开做。
    """
    import statistics
    import time

    def run(max_tokens: int) -> float:
        sp = sp_cls(max_tokens=max_tokens, temperature=0, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1e3

    for _ in range(2):                      # 预热
        run(2)
    t_pf = statistics.median(run(1) for _ in range(repeats))
    t_all = statistics.median(run(decode_steps + 1) for _ in range(repeats))
    return {
        "prefill_ms": round(t_pf, 2),
        "prefill_plus_decode_ms": round(t_all, 2),
        "decode_ms_per_step": round((t_all - t_pf) / decode_steps, 3),
        "repeats": repeats, "prompt_len": prompt_len, "decode_steps": decode_steps,
        "note": "中位数；prefill 用 max_tokens=1，decode 用差分。未开 profiler。",
    }


def dump_sass(match: str, out: Path) -> dict:
    """把**刚才真正跑过的**一个 Triton kernel 反汇编成 SASS。

    Triton 在 JIT 之后会把产物落到 TRITON_CACHE_DIR：同一个 kernel 会有
    .ttir / .ttgir / .llir / .ptx / .cubin 五份，正好是完整的下降链路：
        Triton IR → TritonGPU IR → LLVM IR → PTX（虚拟 ISA）→ cubin（真实机器码）
    这条链在 L2.2 会展开讲。这里只取最后一跳：cubin → SASS。

    注意 PTX 与 SASS 的区别：PTX 是虚拟 ISA，向前兼容、可被驱动 JIT；
    SASS 绑定具体架构（这里是 sm_120），是 GPU 真正执行的指令。
    「装了新卡第一次跑特别慢」通常就是驱动在把 PTX 现场 JIT 成 SASS。
    """
    cache = Path(os.environ.get("TRITON_CACHE_DIR",
                                Path.home() / ".triton" / "cache"))
    if not cache.exists():
        return {"error": f"Triton 缓存目录不存在: {cache}"}

    cubins = sorted(cache.rglob("*.cubin"), key=lambda p: -p.stat().st_size)
    if not cubins:
        return {"error": f"{cache} 下没有 .cubin（本次运行可能没触发 Triton JIT）"}

    picked = next((c for c in cubins if match in c.stem), cubins[0])
    stem_dir = picked.parent
    stages = {ext: p.name for ext in ("ttir", "ttgir", "llir", "ptx", "json")
              for p in stem_dir.glob(f"*.{ext}")}

    try:
        sass = subprocess.run(["nvdisasm", "-c", str(picked)],
                              capture_output=True, text=True, timeout=300).stdout
    except FileNotFoundError:
        return {"error": "nvdisasm 不在 PATH（需要 CUDA toolkit；见 env.sh 里的 CUDA_HOME）"}
    if not sass.strip():
        return {"error": f"nvdisasm 对 {picked.name} 没有输出"}
    out.write_text(sass, encoding="utf-8")

    opcodes: dict[str, int] = defaultdict(int)
    n_inst = 0
    for ln in sass.splitlines():
        s = ln.strip()
        if not s.startswith("/*") or "*/" not in s:
            continue
        rest = s.split("*/", 1)[1].strip()
        tok = rest.split()
        if not tok:
            continue
        # SASS 里 `@!P6  BRA ...` 的第一个 token 是谓词寄存器（predicate），不是指令
        if tok[0].startswith("@"):
            tok = tok[1:]
            if not tok:
                continue
        op = tok[0].split(".")[0]
        if not op or op[0] in "/.":
            continue
        opcodes[op] += 1
        n_inst += 1

    return {
        "cubin": str(picked),
        "cubin_bytes": picked.stat().st_size,
        "n_cubins_in_cache": len(cubins),
        "compilation_stages_present": stages,
        "sass_lines": len(sass.splitlines()),
        "instruction_count": n_inst,
        "top_opcodes": sorted(opcodes.items(), key=lambda kv: -kv[1])[:14],
        "sass_file": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--sass-symbol", default="rms_norm", help="按 cubin 文件名子串挑一个 kernel")
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams
    import vllm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enforce_eager：先关掉 CUDA Graph，才能看见真实的 kernel 下发序列。
    # 打开 graph 后 profiler 只会看到一次 graph launch —— 那本身就是个结论。
    #
    # enable_prefix_caching=False：**这一行是踩坑换来的**。vLLM V1 默认开前缀缓存，
    # 而本脚本会用同一个 prompt 反复跑，第二次起整段 prefill 直接命中缓存，
    # 测出来的 prefill 只有 6 ms —— 比 roofline 下界还快，物理上不可能。
    # 想测「一次真实的 prefill」，必须关掉它（或每次换 prompt，本脚本两者都做）。
    llm = LLM(model=args.model, enforce_eager=True, gpu_memory_utilization=0.6,
              max_model_len=8192, disable_log_stats=True,
              enable_prefix_caching=False)

    tok = llm.get_tokenizer()
    rng = random.Random(0)

    def make_prompt() -> dict:
        """每次生成不同的 token 序列，双保险地避开任何前缀复用。"""
        lo, hi = 1000, min(100000, tok.vocab_size - 1)
        return {"prompt_token_ids": [rng.randint(lo, hi) for _ in range(args.prompt_len)]}

    prompt = make_prompt()

    result: dict = {"vllm": vllm.__version__, "torch": torch.__version__,
                    "model": args.model, "prompt_len": args.prompt_len}

    # 预热：第一次跑包含 JIT、autotune、显存分配，不能计入
    llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

    # --- 0. 干净计时（不开 profiler）---
    result["timing"] = clean_timing(llm, prompt, SamplingParams,
                                    args.prompt_len, args.decode_steps)

    # --- 1. prefill：只出 1 个 token，这一步几乎全是 prefill ---
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_pf:
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    result["prefill"] = summarize(prof_pf)

    # --- 2. prefill + N 步 decode，减去上面的 prefill 得到 decode 的边际成本 ---
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_dec:
        llm.generate([prompt], SamplingParams(max_tokens=args.decode_steps + 1,
                                              temperature=0, ignore_eos=True),
                     use_tqdm=False)
    both = summarize(prof_dec)
    result["prefill_plus_decode"] = both
    result["decode_derived"] = {
        "steps": args.decode_steps,
        "device_us_per_step": round(
            (both["total_device_us"] - result["prefill"]["total_device_us"])
            / args.decode_steps, 1),
        "launches_per_step": round(
            (both["total_launches"] - result["prefill"]["total_launches"])
            / args.decode_steps, 1),
        "note": "用「prefill+N步」减「仅prefill」得到 decode 的边际值，"
                "避开了把启动开销算进去",
    }

    # --- 3. SASS ---
    result["sass"] = dump_sass(args.sass_symbol,
                               out_dir / f"sass_{args.sass_symbol}.txt")

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    (out_dir / "kernel_trace.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
