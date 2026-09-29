#!/usr/bin/env python3
"""L2.0c task C · 真实变长分配轨迹、碎片定义与 expandable_segments 对照。

负载：200 个请求、长度各不相同（固定 seed），分批到来；每个请求分配
"KV + 激活"两块变长张量，完成时释放。释放顺序与分配顺序不同，
于是产生真实的可复用空洞 —— 而不是只分配不释放的假碎片。

三种运行方式（各自独立进程，因为分配器配置必须早于 CUDA 初始化）：

    python alloc_trace.py --mode default      # 默认分配器
    python alloc_trace.py --mode expandable   # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python alloc_trace.py --mode graph        # 固定形状步骤 + CUDA Graph 捕获/replay
    python alloc_trace.py --mode compare      # 依次起三个子进程并打印对照表

两个口径分开记：
    active     我们自己的存活张量字节数（逐 step 与 memory_allocated 对账）
    reserved   向驱动要来的总字节
碎片判据取 snapshot 里的 inactive_split（段内已释放但切碎的空闲），
而不是直接用 reserved - allocated —— 后者大部分是可复用的缓存。
"""

import argparse
import json
import os
import subprocess
import sys

N_REQ = 200
MAX_BATCH = 8
HEADS, HEAD_DIM, HIDDEN = 8, 64, 2048          # KV 与激活的形状参数

# 固定的 200 个请求长度：长短交错，覆盖 16 到 4096
LENGTHS_INTERLEAVED = ([16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                        96, 192, 384, 768, 1536, 3072, 48] * 13)[:N_REQ]
# 同样的长度集合、按升序到达：用于对照"规则尺寸"与"交错尺寸"
LENGTHS_REGULAR = sorted(LENGTHS_INTERLEAVED)


def lengths_for(kind):
    return LENGTHS_REGULAR if kind == "regular" else LENGTHS_INTERLEAVED


def batches(lengths, max_batch):
    out = []
    for i in range(0, len(lengths), max_batch):
        out.append(lengths[i:i + max_batch])
    return out


def snapshot_summary():
    """从真实 allocator snapshot 里取段与块的状态分布。"""
    import torch
    segs = torch.cuda.memory_snapshot()
    states = {}
    total = 0
    for s in segs:
        total += s["total_size"]
        for b in s["blocks"]:
            states[b["state"]] = states.get(b["state"], 0) + b["size"]
    return {"segments": len(segs), "segment_total_mb": total / 2**20,
            "active_mb": states.get("active_allocated", 0) / 2**20,
            "active_split_mb": states.get("active_split", 0) / 2**20,
            "inactive_split_mb": states.get("inactive_split", 0) / 2**20,
            "inactive_mb": states.get("inactive", 0) / 2**20,
            "free_blocks": sum(1 for s in segs for b in s["blocks"]
                               if b["state"] != "active_allocated")}


def run_trace(mode, out_path, order="interleaved"):
    import random

    import torch

    rng = random.Random(20260913)
    torch.manual_seed(20260913)
    dev = "cuda"
    trace = []
    pending = []                   # [finish_step, kv, act]
    queue = list(lengths_for(order))

    def live_bytes():
        return sum(t.numel() * t.element_size()
                   for _f, kv, act in pending for t in (kv, act))

    peak = {"active": 0, "reserved": 0, "allocated": 0, "inactive_split": 0}
    step = 0
    while queue or pending:
        chunk, queue = queue[:MAX_BATCH], queue[MAX_BATCH:]
        for L in chunk:
            kv = torch.empty(2 * HEADS * L * HEAD_DIM, dtype=torch.float16, device=dev)
            act = torch.empty(L * HIDDEN, dtype=torch.float16, device=dev)
            kv.fill_(1.0)
            act.fill_(2.0)
            # 完成时刻与到达顺序无关：长短请求交错释放
            pending.append([step + rng.randint(1, 10), kv, act])
        kept = []
        for fin, kv, act in pending:
            if fin <= step:
                del kv, act
            else:
                kept.append([fin, kv, act])
        pending = kept

        if step % 20 == 19:        # 周期性的临时大工作区
            big = torch.empty(24 * 2**20, dtype=torch.uint8, device=dev)
            big.fill_(3)
            del big

        own_mb = live_bytes() / 2**20
        allocated_mb = torch.cuda.memory_allocated() / 2**20
        reserved_mb = torch.cuda.memory_reserved() / 2**20
        snap = snapshot_summary()
        rec = {
            "step": step, "live_tensors": len(pending) * 2,
            "active_mb": allocated_mb,          # 以 allocator 计数为准
            "own_sum_mb": own_mb,               # 自有账本，作为交叉核对
            "reserved_mb": reserved_mb,
            "cached_free_mb": reserved_mb - allocated_mb,
            "segments": snap["segments"],
            "free_blocks": snap["free_blocks"],
            "inactive_split_mb": snap["inactive_split_mb"],
            "inactive_mb": snap["inactive_mb"],
            "num_alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0),
        }
        trace.append(rec)
        peak["active"] = max(peak["active"], rec["active_mb"])
        peak["reserved"] = max(peak["reserved"], rec["reserved_mb"])
        peak["allocated"] = max(peak["allocated"], rec["active_mb"])
        peak["inactive_split"] = max(peak["inactive_split"], rec["inactive_split_mb"])
        step += 1

    stats = torch.cuda.memory_stats()
    del pending
    summary = {
        "mode": mode,
        "conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"),
        "order": order,
        "total_requested_mb": sum(6144 * L for L in lengths_for(order)) / 2**20,
        "steps": len(trace), "requests": N_REQ,
        "peak_active_mb": peak["active"], "peak_allocated_mb": peak["allocated"],
        "peak_reserved_mb": peak["reserved"],
        "peak_inactive_split_mb": peak["inactive_split"],
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "segments": trace[-1]["segments"],
        "own_sum_max_residual_mb": max(abs(r["own_sum_mb"] - r["active_mb"])
                                       for r in trace),
        "own_trace_mb": [round(r["own_sum_mb"], 3) for r in trace],
        "peak_cached_free_mb": max(r["cached_free_mb"] for r in trace),
        "peak_free_blocks": max(r["free_blocks"] for r in trace),
        "active_trace_mb": [round(r["active_mb"], 3) for r in trace],
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "trace": trace}, f, ensure_ascii=False, indent=2)
    return summary


def run_graph(out_path):
    """固定形状步骤：不捕获 vs 捕获一次 replay 200 次。

    记录两套数：峰值（max_memory_*）与结束后的占用（memory_*）。
    图有自己的内存池，捕获后静态输入/输出与图内工作区都从池里出。
    """
    import torch

    def step(t):
        a = t @ t
        b = a * 2.0
        return b.sum()

    x = torch.randn(1024, 1024, device="cuda")

    # ---- 不捕获 ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(200):
        step(x)
    torch.cuda.synchronize()
    eager = {
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "end_reserved_mb": torch.cuda.memory_reserved() / 2**20,
        "end_allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "retries": torch.cuda.memory_stats().get("num_alloc_retries", 0),
    }

    # ---- 捕获一次 + replay ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(x)
    torch.cuda.current_stream().wait_stream(s)

    static_in = x.clone()
    static_out = torch.zeros((), device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out.copy_(step(static_in))
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    graph = {
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "end_reserved_mb": torch.cuda.memory_reserved() / 2**20,
        "end_allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "retries": torch.cuda.memory_stats().get("num_alloc_retries", 0),
    }

    out = {"eager": eager, "graph": graph,
           "note": "固定形状步骤 200 次；peak_* 取 max_memory_*，end_* 取循环结束时的 memory_*"}
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def compare(out_path="alloc_compare.json"):
    here = os.path.abspath(__file__)
    py = sys.executable
    results = {}
    jobs = [("default", "", "interleaved"),
            ("expandable", "expandable_segments:True", "interleaved"),
            ("regular", "", "regular")]
    for mode, conf, order in jobs:
        out = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                           f"alloc_{mode}.json")
        env = dict(os.environ)
        if conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = conf
        r = subprocess.run([py, here, "--mode", mode, "--order", order, "--out", out],
                           env=env, capture_output=True, text=True)
        print(f"--- {mode} order={order} rc={r.returncode}")
        print("\n".join(r.stdout.splitlines()[-6:]))
        if r.returncode != 0:
            print(r.stderr[-800:])
        results[mode] = json.load(open(out))

    rows = ["default", "expandable", "regular"]
    print()
    print(f"{'配置':<14} {'peak active':>12} {'peak reserved':>14} "
          f"{'reserved/active':>16} {'inactive_split':>15} {'segments':>9}")
    print("-" * 96)
    for k in rows:
        s = results[k]["summary"]
        ratio = (s['peak_reserved_mb'] / s['peak_active_mb']
                 if s['peak_active_mb'] else 0)
        print(f"{k:<14} {s['peak_active_mb']:>10.2f} MB {s['peak_reserved_mb']:>12.2f} MB "
              f"{ratio:>15.2f}x {s['peak_inactive_split_mb']:>13.2f} MB "
              f"{s['segments']:>9}")
    print("  注：default 与 expandable 的活跃轨迹逐 step 相同，可直接比 reserved；")
    print("      regular 的峰值 active 本身不同（升序到达会让大请求同时驻留），")
    print("      它只用来对照'尺寸顺序'对 reserved/active 比值的影响，不能直接比 reserved。")
    da = results["default"]["summary"]
    ep = results["expandable"]["summary"]
    print()
    rg = results["regular"]["summary"]
    print(f"请求总量：default {da['total_requested_mb']:.1f} MB | expandable "
          f"{ep['total_requested_mb']:.1f} MB | regular {rg['total_requested_mb']:.1f} MB "
          f"（三者必须相同）")
    print(f"自有账本（存活张量字节）逐 step 相同: "
          f"{da['own_trace_mb'] == ep['own_trace_mb']}（default vs expandable，"
          f"{len(da['own_trace_mb'])} 步）")
    print(f"allocator 的 active 轨迹逐 step 相同: "
          f"{da['active_trace_mb'] == ep['active_trace_mb']}")
    print(f"自有账本与 allocator 的最大残差：default "
          f"{da['own_sum_max_residual_mb']:.2f} MB，expandable "
          f"{ep['own_sum_max_residual_mb']:.2f} MB")
    print("（默认分配器会把部分请求按块取整：本负载里一个 2 MiB 请求占了 3 MiB 的 "
          "active 块；expandable 下残差为 0）")
    with open(out_path, "w") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"JSON -> {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="default",
                    choices=["default", "expandable", "regular", "graph", "compare"])
    ap.add_argument("--out", default="alloc_trace.json")
    ap.add_argument("--order", default="interleaved",
                    choices=["interleaved", "regular"])
    args = ap.parse_args()

    if args.mode == "compare":
        compare(args.out)
        return 0
    if args.mode == "graph":
        run_graph(args.out)
        return 0
    if args.mode == "expandable":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.mode == "regular":
        args.order = "regular"
    import torch
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  "
          f"conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(default)')}")
    s = run_trace(args.mode, args.out, args.order)
    print(f"  peak active {s['peak_active_mb']:.1f} MB | peak reserved "
          f"{s['peak_reserved_mb']:.1f} MB | peak cached-free "
          f"{s['peak_cached_free_mb']:.1f} MB | inactive_split "
          f"{s['peak_inactive_split_mb']:.2f} MB | retries {s['num_alloc_retries']} "
          f"| segments {s['segments']} | 自有账本残差 "
          f"{s['own_sum_max_residual_mb']:.2f} MB")
    print(f"  JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
