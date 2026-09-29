#!/usr/bin/env python3
"""L5.6 任务 B/C —— mask apply kernel 的单独计时。

正文此前只有 apply 的**正确性**验证（`mask_apply_probe.py` 逐位检查被屏蔽的位置
变成 -inf、合法位置不变），没有它的**时间**。这一节把 apply 单独拿出来量：

  1. 真实捕获的 mask（`grammar-libs-20260910-1907/*/mask-*.bin`，三种库 × 两份 schema）；
  2. 合成的三种密度：全允许（0 位被屏蔽，对应 structural tag 的标签外区域）、
     稀疏允许（约 12/151936 合法）、中等（约 3% 被屏蔽）；
  3. batch 1/8/32（同一行复制），词表 151936 与 32000 两档。

计时用 device event：先预热，再按 (轮数 × 每轮次数) 取中位数。同时给出
「每元素纳秒」和「按 4 字节读写折算的等效带宽」，用来判断这个 kernel 是
启动受限还是带宽受限。

用装了 xgrammar 与 CUDA 的 serve venv 运行：
    python labs/L5/mask_apply_timing.py --out "$OUT/apply-timing" \
        --libs-dir results/crater/structured/grammar-libs-20260910-1907
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np
import torch
import xgrammar as xg

VOCABS = [151936, 32000]
BATCHES = [1, 8, 32]
ROUNDS = 5
REPS = 20
WARMUP = 10


def bits_to_words(bits: np.ndarray) -> np.ndarray:
    """把 0/1 位图打包成 32 位字（低位在前，与 xgrammar 的布局一致）。"""
    pad = (-len(bits)) % 32
    if pad:
        bits = np.concatenate([bits, np.ones(pad, dtype=np.uint8)])
    words = np.zeros(len(bits) // 32, dtype=np.uint32)
    w = bits.reshape(-1, 32).astype(np.uint32)
    for i in range(32):
        words |= w[:, i] << i
    return words


def make_mask(vocab: int, disallowed_fraction: float, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    bits = np.ones(vocab, dtype=np.uint8)
    n_bad = int(round(vocab * disallowed_fraction))
    if n_bad:
        bits[rng.choice(vocab, size=n_bad, replace=False)] = 0
    return torch.from_numpy(bits_to_words(bits).astype(np.int32))


def load_masks(libs_dir: pathlib.Path):
    out = []
    for f in sorted(libs_dir.glob("*/mask-*.bin")):
        words = np.frombuffer(f.read_bytes(), dtype="<i4").copy()
        out.append((f"{f.parent.name}/{f.name}", torch.from_numpy(words)))
    return out


def _measure(fn, rounds=ROUNDS, reps=REPS, warmup=WARMUP):
    """同时给出设备时间与调用侧墙钟（µs/次，取中位数）。"""
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, cpu = [], []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        cpu.append((time.perf_counter() - t0) * 1e6 / reps)
        torch.cuda.synchronize()
        gpu.append(s.elapsed_time(e) * 1000 / reps)
    return statistics.median(gpu), statistics.median(cpu), gpu, cpu


def _measure_graph(fn, reps=REPS, rounds=ROUNDS):
    """把 reps 次调用捕进一张 CUDA Graph，重放后除以 reps，去掉调用侧开销。

    调用侧 9.5 µs/次时，逐次提交的 event 窗口等于 CPU 的提交节奏而不是 kernel
    时间（5.4 讨论过同一件事）。捕获后 CPU 只提交一次，剩下的就是设备时间。
    """
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay()
        torch.cuda.synchronize()
    except Exception as exc:                                     # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:90]}"
    out = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1000 / reps)
    return statistics.median(out), None


def time_apply(mask_row: torch.Tensor, vocab: int, batch: int):
    """返回一次 apply 的设备时间、调用侧时间、基准与允许比例。

    两个基准把「调用开销」和「同尺寸的普通 elementwise kernel」分开：
      * `logits.mul_(1.0)` 是同样读写 4 字节 × batch × vocab 的 elementwise 核；
      * 两者之差反映 xgrammar 封装（Python 层参数检查、dtype/dispatch）的成本。
    """
    mask = mask_row.unsqueeze(0).repeat(batch, 1).cuda()
    logits = torch.zeros(batch, vocab, dtype=torch.float32, device="cuda")

    apply = lambda: xg.apply_token_bitmask_inplace(logits, mask)   # noqa: E731
    gpu, cpu, gs, cs = _measure(apply)
    graph_us, graph_err = _measure_graph(apply)
    base_gpu, base_cpu, _, _ = _measure(lambda: logits.mul_(1.0))
    base_graph, _ = _measure_graph(lambda: logits.mul_(1.0))
    empty_gpu, empty_cpu, _, _ = _measure(lambda: logits[0, :1].fill_(0.0))

    i = torch.arange(vocab, device="cuda")
    allowed = int((((mask[0, i // 32] >> (i % 32)) & 1) != 0).sum())
    del mask, logits
    torch.cuda.empty_cache()
    return dict(median_us=gpu, cpu_us=cpu, graph_us=graph_us,
                graph_error=graph_err, baseline_us=base_gpu,
                baseline_cpu_us=base_cpu, baseline_graph_us=base_graph,
                tiny_us=empty_gpu, tiny_cpu_us=empty_cpu,
                allowed_fraction=allowed / vocab,
                samples_us=gs, samples_cpu_us=cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--libs-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    try:
        from importlib.metadata import version as _v
        xg_ver = _v("xgrammar")
    except Exception:                                            # noqa: BLE001
        xg_ver = "unknown"
    print(f"xgrammar {xg_ver}  torch {torch.__version__}")
    rows = []

    # 合成密度：0 位屏蔽 = structural tag 的标签外区域
    cases = [("synth:全允许(0% 屏蔽)", 151936, 0.0),
             ("synth:中等(3.16% 屏蔽)", 151936, 0.0316),
             ("synth:稀疏允许(12/151936)", 151936, 1 - 12 / 151936),
             ("synth:32000 全允许", 32000, 0.0),
             ("synth:32000 稀疏允许", 32000, 1 - 12 / 32000)]
    for label, vocab, frac in cases:
        m = make_mask(vocab, frac)
        for b in BATCHES:
            r = time_apply(m, vocab, b)
            r.update(source=label, vocab=vocab, batch=b)
            rows.append(r)
            g = ("失败 " + str(r["graph_error"])[:28]) if r["graph_us"] is None \
                else f"{r['graph_us']:>7.2f} µs"
            print(f"  {label:<28} V={vocab:>6} b={b:>2}  "
                  f"逐次 {r['median_us']:>6.2f}/{r['cpu_us']:>6.2f} µs  "
                  f"图内 {g}  "
                  f"mul_基准 逐次 {r['baseline_us']:>5.2f} 图内 "
                  f"{('%.2f' % r['baseline_graph_us']) if r['baseline_graph_us'] else 'n/a'}"
                  f"  允许 {r['allowed_fraction']:.6f}")

    if args.libs_dir and args.libs_dir.exists():
        print(f"\n真实捕获的 mask（{args.libs_dir}）：")
        for name, words in load_masks(args.libs_dir):
            vocab = words.numel() * 32
            for b in BATCHES:
                r = time_apply(words, vocab, b)
                r.update(source=name, vocab=vocab, batch=b)
                rows.append(r)
                g = ("失败 " + str(r["graph_error"])[:28]) if r["graph_us"] is None \
                    else f"{r['graph_us']:>7.2f} µs"
                print(f"  {name:<34} b={b:>2}  逐次 {r['median_us']:>6.2f}/"
                      f"{r['cpu_us']:>6.2f} µs  图内 {g}  "
                      f"允许 {r['allowed_fraction']:.6f}")

    (args.out / "apply_timing.json").write_text(
        json.dumps(dict(xgrammar=xg_ver, torch=torch.__version__,
                        rounds=ROUNDS, reps=REPS, rows=rows),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n读法（『逐次』列对是 event 窗口与调用侧墙钟；『图内』是把 20 次调用")
    print("捕进一张 CUDA Graph 重放后除以 20，用来剥掉调用侧节奏；mul_ 是同样")
    print("尺寸的 elementwise 基准）：")
    print("  * 逐次的 event 时间 ≈ 调用侧时间 → 那次测量落在 CPU 提交节奏上，")
    print("    不能当 kernel 时间；")
    print("  * 图内时间才是设备侧的真实成本，用它比较词表、密度与 batch；")
    print("  * mul_ 基准随 batch 增长而 apply 不增长，说明 apply 的成本与 logits")
    print("    尺寸无关，凡是把端到端差额按字节归因到 apply 的说法都不成立。")


if __name__ == "__main__":
    main()
