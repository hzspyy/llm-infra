#!/usr/bin/env python3
"""L5.2 lab · 把「分页」和「前缀复用」这两件事测出来。

三个实验，每个都对应一段代码里的具体机制：

  A. 前缀缓存开/关的 TTFT 差 —— 量化 `get_computed_blocks()` 的价值
  B. **阶梯实验**：把第一个不同的 token 放在位置 p，扫 p，看 TTFT 怎么变。
     如果前缀是按 **块**（默认 16 token）为粒度复用的，
     块数预测呈阶梯；计时是否能分辨块边界，需要单独验证。
  C. 块分配与碎片：直接读引擎暴露的 KV cache 使用率与命中率指标

用法：
    python prefix_probe.py --model /path --out results/prefix_probe.json
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


def make_engine(model: str, *, prefix_caching: bool, max_len: int, block_size: int | None):
    from vllm import LLM
    kw = dict(model=model, enforce_eager=True, gpu_memory_utilization=0.35,
              max_model_len=max_len, disable_log_stats=False,
              enable_prefix_caching=prefix_caching)
    if block_size:
        kw["block_size"] = block_size
    return LLM(**kw)


def ttft_ms(llm, token_ids: list[int], repeats: int = 5) -> float:
    """只生成 1 个 token，用整次调用耗时近似 TTFT。

    离线 API 没有真正的 SSE 首 token 时刻，但 max_tokens=1 时
    「整次调用」包含 prompt 前向、首 token 采样和请求处理开销，
    首 token 来自 prompt 前向的 logits，不应额外算成一次 decode 前向。
    用于**相对比较**足够；绝对 TTFT 要用在线压测（L8.3）。
    """
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0)
    p = {"prompt_token_ids": token_ids}
    llm.generate([p], sp, use_tqdm=False)          # 预热并把前缀写进缓存
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        llm.generate([p], sp, use_tqdm=False)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def cache_metrics(llm) -> dict:
    """把引擎自己报的 KV cache 指标读出来——不要自己猜命中率。"""
    try:
        ms = llm.get_metrics()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    out = {}
    for m in ms:
        name = getattr(m, "name", "")
        if "prefix_cache" in name or "kv_cache" in name or "cache_usage" in name:
            v = getattr(m, "value", None)
            if v is None and hasattr(m, "sum"):     # Counter/Vector 型
                v = {"sum": m.sum, "count": getattr(m, "count", None)}
            out[name] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=None,
                    help="不给则用引擎默认（vLLM 通常是 16）")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(7)
    max_len = args.prefix_len + 64

    result = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model": Path(args.model).name, "config": vars(args),
    }

    # ---------- A. 开 / 关前缀缓存 ----------
    from vllm import LLM  # noqa: F401  (确保 vllm 可用后再建引擎)
    llm_on = make_engine(args.model, prefix_caching=True, max_len=max_len,
                         block_size=args.block_size)
    tok = llm_on.get_tokenizer()
    lo, hi = 1000, min(100000, tok.vocab_size - 1)
    base = [rng.randint(lo, hi) for _ in range(args.prefix_len)]

    cfg = llm_on.llm_engine.vllm_config
    block_size = cfg.cache_config.block_size
    result["engine"] = {
        "block_size": block_size,
        "num_gpu_blocks": cfg.cache_config.num_gpu_blocks,
        "kv_bytes_per_block": None,
    }
    print(f"块大小 = {block_size} token；GPU 块数 = {cfg.cache_config.num_gpu_blocks}")

    # 冷启动：全新前缀（缓存里没有）
    cold = [rng.randint(lo, hi) for _ in range(args.prefix_len)]
    from vllm import SamplingParams
    t0 = time.perf_counter()
    llm_on.generate([{"prompt_token_ids": cold}],
                    SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    ms_cold = (time.perf_counter() - t0) * 1e3

    # 热：同一个前缀再跑（缓存全命中）
    ms_hot = ttft_ms(llm_on, cold, args.repeats)

    print(f"\n[A] 前缀 {args.prefix_len} token")
    print(f"    冷（首次，无命中） {ms_cold:8.2f} ms")
    print(f"    热（完全命中）     {ms_hot:8.2f} ms   加速 {ms_cold / ms_hot:.1f}×")
    result["A_hit_vs_miss"] = {"cold_ms": round(ms_cold, 2), "hot_ms": round(ms_hot, 2),
                               "speedup": round(ms_cold / ms_hot, 2)}
    result["A_metrics"] = cache_metrics(llm_on)

    # ---------- B. 阶梯实验 ----------
    # 关键：必须测**首次**调用。第一版这里调了带预热的 ttft_ms()，
    # 结果预热那一次就把完整 prompt 也写进了缓存，后面每次都 100% 命中，
    # 测出来是一条水平线（7.7~7.9 ms 不随共享前缀变化）——完全没有信息量。
    # 正确做法：base 先进缓存，然后每个探测点用**全新的随机后缀**只跑一次，
    # 重复 N 次时每次都换后缀，取中位数。
    from vllm import SamplingParams
    sp1 = SamplingParams(max_tokens=1, temperature=0)
    ttft_ms(llm_on, base, 2)                      # 把 base 的 KV 写进缓存
    print(f"\n[B] 共享前缀长度 → 首次 TTFT（块大小 {block_size}，prompt 总长 {args.prefix_len}）")
    print(f"    {'共享前缀':>8} {'可复用块':>9} {'需重算token':>11} {'首次TTFT ms':>12}")
    rows = []
    probes = []
    for blk in (0, 1, 2, 4, 8, 16, 32, 64, 96, 127):
        b = blk * block_size
        if b > args.prefix_len - block_size:
            continue
        probes += [b, b + 1, b + block_size // 2, b + block_size - 1]
    probes = sorted({p for p in probes if 0 <= p < args.prefix_len})

    n_rep = max(3, args.repeats - 2)
    for shared in probes:
        samples = []
        for _ in range(n_rep):
            ids = base[:shared] + [rng.randint(lo, hi)
                                   for _ in range(args.prefix_len - shared)]
            t0 = time.perf_counter()
            llm_on.generate([{"prompt_token_ids": ids}], sp1, use_tqdm=False)
            samples.append((time.perf_counter() - t0) * 1e3)
        ms = statistics.median(samples)
        full_blocks = shared // block_size          # 理论预测，不是实际命中计数
        recompute = args.prefix_len - full_blocks * block_size
        print(f"    {shared:>8} {full_blocks:>9} {recompute:>11} {ms:>12.2f}")
        rows.append({"shared_prefix_tokens": shared, "full_blocks_reusable": full_blocks,
                     "tokens_to_recompute": recompute, "ttft_ms": round(ms, 2)})
    result["B_staircase"] = rows
    result["B_note"] = ("每个探测点用全新随机后缀只跑一次（不预热），重复取中位数；"
                        "块数和剩余 token 数由公式预测，并非逐请求实测；"
                        "本脚本不据此判定已经观测到块边界。")
    result["B_evidence_kind"] = {
        "full_blocks_reusable": "predicted",
        "tokens_to_recompute": "predicted",
        "ttft_ms": "measured_offline_call_duration",
    }

    # 释放引擎 A。vLLM 的 KV 池不会因为 del 就还给 CUDA——必须显式关掉引擎核心，
    # 否则第二个引擎启动时会报 "Free memory ... is less than desired"（真踩过）。
    try:
        llm_on.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm_on
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # ---------- C. 关掉前缀缓存做对照 ----------
    llm_off = make_engine(args.model, prefix_caching=False, max_len=max_len,
                          block_size=args.block_size)
    ms_off = ttft_ms(llm_off, base, args.repeats)
    print(f"\n[C] 关掉前缀缓存，同样 prompt 重复跑：{ms_off:8.2f} ms")
    print(f"    对照 [A] 的热命中 {ms_hot:.2f} ms ⇒ 前缀缓存省了 {(1 - ms_hot / ms_off):.0%}")
    result["C_no_prefix_cache"] = {"ms": round(ms_off, 2),
                                   "saving_vs_hot": round(1 - ms_hot / ms_off, 4)}

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
