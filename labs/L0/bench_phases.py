#!/usr/bin/env python3
"""L0 lab · 分相位基准：把 prefill 和 decode 分开测，再跟 roofline 下界比。

大部分 benchmark 只报一个「吞吐」数字，那没法回答任何问题。
本脚本分别测：
  A. prefill 时间 vs prompt 长度   → TTFT 的来源，受**算力**限制
  B. decode 每步时间 vs batch      → TPOT 的来源，受**带宽**限制
  C. 各自达到 roofline 下界的比例  → 差距该去哪一层找

方法学（L0 就要立规矩，M2 会展开）：
  - 预热两次，丢掉（首次含 JIT、显存分配、时钟爬升）
  - 每个点重复 N 次取中位数，不取平均（避免个别抖动主导）
  - 关掉 prefix caching，每次换随机 prompt（否则 prefill 会被缓存命中）
  - 关掉 CUDA Graph（enforce_eager）以隔离变量；graph 的收益单独在 L5.4 测
  - 用差分把 prefill 从 decode 里减掉

用法：
    python bench_phases.py --model /path --hw results/hw_x.json --out results/bench_x.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def load_arch(model_dir: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ledger import Arch, decode_step, prefill_flops, roofline_tpot_ms  # noqa: E402
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = cfg["text_config"]
    return Arch.from_hf(cfg, Path(model_dir).name), decode_step, prefill_flops, roofline_tpot_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hw", required=True, help="probe_hw.py 的输出，用来算 roofline 下界")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-lens", default="512,1024,2048,4096")
    ap.add_argument("--batches", default="1,4,16,32")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ctx-for-decode", type=int, default=1024)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    arch, decode_step, prefill_flops, roofline = load_arch(args.model)
    hw = json.loads(Path(args.hw).read_text())
    rng = random.Random(1234)

    prompt_lens = [int(x) for x in args.prompt_lens.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    max_len = max(max(prompt_lens), args.ctx_for_decode) + args.decode_steps + 16

    llm = LLM(model=args.model, enforce_eager=True,
              gpu_memory_utilization=args.gpu_frac, max_model_len=max_len,
              disable_log_stats=True, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    lo, hi = 1000, min(100000, tok.vocab_size - 1)

    def prompts(n: int, length: int):
        """每次都是新的随机 token 串——双保险地避开任何前缀复用。"""
        return [{"prompt_token_ids": [rng.randint(lo, hi) for _ in range(length)]}
                for _ in range(n)]

    def run(ps, max_tokens: int) -> float:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate(ps, sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1e3

    def median_of(fn) -> float:
        for _ in range(2):                       # 预热并丢弃
            fn()
        return statistics.median(fn() for _ in range(args.repeats))

    result = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "host": hw["device"]["host"], "gpu": hw["device"]["name"],
        "model": Path(args.model).name,
        "hw_roofline": {"bw_gbps": hw["memory_bandwidth"]["copy_gbps"],
                        "bf16_tflops": hw["gemm"]["bf16"]["peak_tflops"],
                        "launch_us": hw["launch_overhead"]["eager_us_per_kernel"]},
        "config": vars(args),
    }

    # ---- A. prefill vs prompt 长度 ----
    print(f"\n{'prompt':>7} {'实测ms':>9} {'下界ms':>9} {'达标率':>7} {'有效TFLOPS':>11}")
    rows = []
    for L in prompt_lens:
        ms = median_of(lambda L=L: run(prompts(1, L), 1))
        pf = prefill_flops(arch, L)
        lb = pf["total"] / (hw["gemm"]["bf16"]["peak_tflops"] * 1e12) * 1e3
        eff = pf["total"] / (ms * 1e-3) / 1e12
        print(f"{L:>7} {ms:>9.2f} {lb:>9.2f} {lb / ms:>6.0%} {eff:>11.1f}")
        rows.append({"prompt_len": L, "ms": round(ms, 2), "lower_bound_ms": round(lb, 2),
                     "frac_of_roofline": round(lb / ms, 4),
                     "effective_tflops": round(eff, 1),
                     "tflop": round(pf["total"] / 1e12, 3)})
    result["prefill"] = rows

    # ---- B. decode vs batch ----
    C = args.ctx_for_decode
    print(f"\n上下文 {C}，每条序列生成 {args.decode_steps} token")
    print(f"{'batch':>6} {'每步ms':>9} {'下界ms':>9} {'达标率':>7} {'tok/s':>9} {'有效GB/s':>10}")
    rows = []
    for B in batches:
        def one(B=B):
            ps = prompts(B, C)
            t_pf = run(ps, 1)
            ps2 = prompts(B, C)
            t_all = run(ps2, args.decode_steps + 1)
            return (t_all - t_pf) / args.decode_steps
        ms = median_of(one)
        st = decode_step(arch, C, B)
        rf = roofline(st, hw)
        tps = B / (ms * 1e-3)
        eff_bw = st["bytes_total"] / (ms * 1e-3) / 1e9
        print(f"{B:>6} {ms:>9.3f} {rf['t_lower_bound_ms']:>9.3f} "
              f"{rf['t_lower_bound_ms'] / ms:>6.0%} {tps:>9.0f} {eff_bw:>10.0f}")
        rows.append({"batch": B, "ms_per_step": round(ms, 3),
                     "lower_bound_ms": round(rf["t_lower_bound_ms"], 3),
                     "frac_of_roofline": round(rf["t_lower_bound_ms"] / ms, 4),
                     "tokens_per_s": round(tps, 1),
                     "effective_bw_gbps": round(eff_bw, 1),
                     "arithmetic_intensity": round(st["arithmetic_intensity"], 2)})
    result["decode"] = rows

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
