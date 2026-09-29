#!/usr/bin/env python3
"""L2.4-C · 用 Qwen3-1.7B 的真实维度扫 cuBLAS GEMM，并区分驻留与轮转权重。

维度来自模型自己的 config.json（hidden=2048、intermediate=6144、heads 16/8、head_dim 128）。
对每个形状量 bf16 GEMM 的 TFLOPS：
  resident  同一份权重反复用（权重留在 L2）
  rotating  轮流用 8 份权重（总字节远超 96 MB L2，权重每次都从 DRAM 读）

    python gemm_shapes.py --config <config.json> --out-json <path>
"""

import argparse
import json
import pathlib
import sys

import torch


def make_shapes(c):
    h = c["hidden_size"]
    i = c["intermediate_size"]
    nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    q_dim = nh * hd
    kv_dim = nkv * hd
    return [
        ("q_proj", 1, h, q_dim), ("q_proj", 8, h, q_dim), ("q_proj", 128, h, q_dim),
        ("q_proj", 512, h, q_dim), ("q_proj", 2048, h, q_dim),
        ("o_proj", 2048, q_dim, h),
        ("gate_up_proj", 2048, h, 2 * i),
        ("down_proj", 1, i, h), ("down_proj", 8, i, h), ("down_proj", 128, i, h),
        ("down_proj", 512, i, h), ("down_proj", 2048, i, h),
        ("square4096", 4096, 4096, 4096),
    ]


def bench(a, b, rotating_pool=None, iters=20, warmup=3):
    def once(i):
        if rotating_pool is None:
            torch.mm(a, b)
        else:
            torch.mm(a, rotating_pool[i % len(rotating_pool)])
    for k in range(warmup):
        once(k)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters):
        once(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    print(f"=== {torch.cuda.get_device_name(0)}  bf16  config 1ddb5b89  "
          f"hidden={cfg['hidden_size']} inter={cfg['intermediate_size']} ===")
    print(f"{'层':<14}{'M':>6}{'K':>7}{'N':>7}{'FLOP(G)':>10}"
          f"{'驻留ms':>10}{'TFLOPS':>9}{'轮转ms':>10}{'TFLOPS':>9}")
    rows = []
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    for name, M, K, N in make_shapes(cfg):
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        flop = 2.0 * M * K * N
        t_res = bench(a, b)
        pool = None
        if K * N * 2 * 8 > l2:            # 8 份权重大于 L2 才做轮转
            pool = [torch.randn(K, N, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
        t_rot = bench(a, b, pool) if pool else float("nan")
        rec = {"name": name, "M": M, "K": K, "N": N, "flop": flop,
               "resident_ms": t_res, "resident_tflops": flop / (t_res * 1e-3) / 1e12,
               "rotating_ms": None if pool is None else t_rot,
               "rotating_tflops": None if pool is None else flop / (t_rot * 1e-3) / 1e12,
               "weight_mb": K * N * 2 / 1048576}
        rows.append(rec)
        print(f"{name:<14}{M:>6}{K:>7}{N:>7}{flop/1e9:>10.1f}"
              f"{t_res:>10.4f}{rec['resident_tflops']:>9.1f}"
              + (f"{t_rot:>10.4f}{rec['rotating_tflops']:>9.1f}" if pool else f"{'-':>10}{'-':>9}"))
        del a, b, pool
        torch.cuda.empty_cache()
    print("  对照：L1.2 的 tensor core bf16 上限 251.9 TFLOPS；"
          f"本卡 L2 = {l2/1048576:.0f} MB")
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
        print(f"JSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
