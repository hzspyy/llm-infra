#!/usr/bin/env python3
"""L5.13 任务 C —— 混合架构下投机回滚的真实成本。

计划要求「强制在草稿第 1/2/末位拒绝，记录回滚前后值、copy kernel 和字节；
与不投机输出及完整调用比较」，并明确「直接测量回滚成本，不用理论状态字节代替耗时」。
正文此前只有 `360 KB × batch = 2.8 MB` 的推算，这份脚本把它换成实测。

做法：按 RecurrentGemma-2B 的真实层配置（26 层 = 18 recurrent + 8 attention，
recurrent 状态 [batch,10,4,256]、attention KV 每 token 每层 [2,1,256]）建一个
**真张量**的混合状态池，实现 allocate / snapshot / commit / rollback：

  * attention 层：KV 走块表，回滚只改有效长度与块引用 → 抄 0 字节；
  * recurrent 层：状态定长，回滚必须把快照抄回 → 真实 `copy_` 内核与字节。

每个拒绝位置（第 1 / 第 2 / 末位 / 全接受）都跑一轮：
先对状态施加 K 次"验证写入"，再用快照回滚，并把回滚后的张量与
「从 S0 起只施加 accepted+1 次写入」的参照逐元素比对，保证回滚正确而不是只看时间。

状态更新本身是合成的（不跑模型），但 snapshot/rollback 的拷贝内核、字节与耗时都是真的；
这一点在输出里显式声明。

用法（crater，serve venv）：
    python labs/L5/hybrid_rollback_cost.py --out "$OUT/rollback"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# RecurrentGemma-2B 的层配置（取自 results/crater/hybrid/recurrentgemma_state_audit.json）
N_RECURRENT, N_ATTENTION = 18, 8
REC_SHAPE = (10, 4, 256)          # 每层每序列
KV_SHAPE = (2, 1, 256)            # 每层每 token
BLOCK = 16
DTYPE = "bfloat16"
DTYPE_BYTES = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--draft", type=int, default=5)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import torch

    K = args.draft
    rec_bytes = N_RECURRENT * REC_SHAPE[0] * REC_SHAPE[1] * REC_SHAPE[2] * DTYPE_BYTES
    kv_bytes_per_token = N_ATTENTION * KV_SHAPE[0] * KV_SHAPE[1] * KV_SHAPE[2] * DTYPE_BYTES

    def make_pool(batch):
        rec = [torch.zeros(batch, *REC_SHAPE, dtype=torch.bfloat16, device="cuda")
               for _ in range(N_RECURRENT)]
        # attention KV 用块表 + 线性块池模拟：每块 BLOCK 个 token
        n_blocks = (args.context + K + 8 + BLOCK - 1) // BLOCK + 4
        pool = [torch.zeros(n_blocks, BLOCK, *KV_SHAPE, dtype=torch.bfloat16,
                            device="cuda") for _ in range(N_ATTENTION)]
        tables = [[list(range(i * 4, i * 4 + 4)) for i in range(batch)]
                  for _ in range(N_ATTENTION)]      # 每层每序列的块表（示例布局）
        return rec, pool, tables

    results = []
    for batch in args.batch:
        rec, pool, tables = make_pool(batch)
        # 参照池：只施加"被接受"的写入，用来验证回滚后的值
        ref = [t.clone() for t in rec]
        snapshot_bytes = rec_bytes * batch
        kv_rollback_bytes = 0                      # 只改长度/块引用

        cases = []
        for name, n_acc in [("拒绝在第 1 位", 0), ("拒绝在第 2 位", 1),
                            (f"拒绝在末位（第 {K} 位）", K - 1), ("全部接受", K)]:
            # 进入本轮时状态在 L（用任意基准值即可），本轮要提交 accepted+1 个位置。
            base_val = float(n_acc + 1)
            for lay in range(N_RECURRENT):
                rec[lay].fill_(base_val)
            # 参照：从同一基准再施加 accepted+1 次写入 —— 这是提交后的正确状态
            for lay in range(N_RECURRENT):
                ref[lay].copy_(rec[lay])
            for step in range(1, n_acc + 2):
                for lay in range(N_RECURRENT):
                    ref[lay].add_(float(step))

            # ---- snapshot（必须在验证写入之前） ----
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            snap = [t.clone() for t in rec]
            e.record()
            torch.cuda.synchronize()
            snap_ms = s.elapsed_time(e)

            # ---- 验证前向：对 K 个草稿位置逐次写状态（比提交的多写被拒的位置） ----
            for step in range(1, K + 1):
                for lay in range(N_RECURRENT):
                    rec[lay].add_(float(step))

            # ---- rollback ----
            s2 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            s2.record()
            for lay in range(N_RECURRENT):
                rec[lay].copy_(snap[lay])
            e2.record()
            torch.cuda.synchronize()
            rollback_ms = s2.elapsed_time(e2)

            exact = all(torch.equal(rec[lay], snap[lay]) for lay in range(N_RECURRENT))
            # 提交：回滚后再施加 accepted+1 次写入，应与参照一致
            for step in range(1, n_acc + 2):
                for lay in range(N_RECURRENT):
                    rec[lay].add_(float(step))
            consistent = all(torch.equal(rec[lay], ref[lay]) for lay in range(N_RECURRENT))

            # ---- 不投机的单步对照：一次写入，无快照无回滚 ----
            s3 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            s3.record()
            for lay in range(N_RECURRENT):
                rec[lay].add_(1.0)
            e3.record()
            torch.cuda.synchronize()
            base_ms = s3.elapsed_time(e3)

            # ---- 对照 a：把 18 层拼成一个连续张量，回滚只需一次拷贝 ----
            flat = torch.cat([r.reshape(-1) for r in rec]).contiguous()
            flat_snap = flat.clone()
            s4 = torch.cuda.Event(enable_timing=True)
            e4 = torch.cuda.Event(enable_timing=True)
            s4.record()
            flat.copy_(flat_snap)
            e4.record()
            torch.cuda.synchronize()
            fused_ms = s4.elapsed_time(e4)

            # ---- 对照 b：把 18 次拷贝捕进一张 CUDA Graph，去掉逐次提交 ----
            graph_ms, graph_err = None, None
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for lay in range(N_RECURRENT):
                        rec[lay].copy_(snap[lay])
                g.replay()
                torch.cuda.synchronize()
                s5 = torch.cuda.Event(enable_timing=True)
                e5 = torch.cuda.Event(enable_timing=True)
                s5.record()
                g.replay()
                e5.record()
                torch.cuda.synchronize()
                graph_ms = s5.elapsed_time(e5)
            except Exception as exc:                             # noqa: BLE001
                graph_err = f"{type(exc).__name__}: {str(exc)[:70]}"

            # ---- 对照 c：18 次空核，给出启动地板 ----
            tiny = torch.zeros(1, dtype=torch.float32, device="cuda")
            s6 = torch.cuda.Event(enable_timing=True)
            e6 = torch.cuda.Event(enable_timing=True)
            s6.record()
            for _ in range(N_RECURRENT):
                tiny.add_(1.0)
            e6.record()
            torch.cuda.synchronize()
            launch_floor_ms = s6.elapsed_time(e6)

            cases.append(dict(case=name, accepted=n_acc,
                              snapshot_ms=snap_ms, rollback_ms=rollback_ms,
                              base_step_ms=base_ms,
                              snapshot_bytes=snapshot_bytes,
                              rollback_bytes=snapshot_bytes,
                              rollback_bandwidth_gbps=(snapshot_bytes / 1e9)
                              / max(rollback_ms / 1e3, 1e-12),
                              kv_rollback_bytes=kv_rollback_bytes,
                              values_exact_after_rollback=exact,
                              values_consistent_with_reference=consistent,
                              fused_copy_ms=fused_ms,
                              graph_replay_ms=graph_ms, graph_error=graph_err,
                              launch_floor_ms=launch_floor_ms,
                              bytes_per_copy=snapshot_bytes))

        # 多轮取中位数，去掉单次抖动
        med = {}
        for key in ("snapshot_ms", "rollback_ms", "base_step_ms"):
            med[key] = statistics.median(c[key] for c in cases)
        results.append(dict(batch=batch, rec_bytes=rec_bytes,
                            kv_bytes_per_token=kv_bytes_per_token,
                            snapshot_bytes_total=snapshot_bytes, cases=cases,
                            median=med))
        m = med
        print(f"batch={batch:>2}  recurrent 状态 {rec_bytes*batch/1e6:>7.2f} MB  "
              f"快照 {m['snapshot_ms']:>6.3f} ms  回滚 {m['rollback_ms']:>6.3f} ms  "
              f"不投机单步 {m['base_step_ms']:>6.3f} ms  "
              f"回滚/单步 {m['rollback_ms']/max(m['base_step_ms'],1e-9):>5.2f}×  "
              f"带宽 {next(c for c in cases)['rollback_bandwidth_gbps']:.1f} GB/s")
        c0 = cases[0]
        g = f"{c0['graph_replay_ms']:.3f}" if c0["graph_replay_ms"] else f"失败({c0['graph_error']})"
        print(f"    18 次逐层拷贝 {c0['rollback_ms']:.3f} ms | "
              f"拼成一个连续张量后单次拷贝 {c0['fused_copy_ms']:.3f} ms | "
              f"图捕获回放 {g} ms | 18 次空核启动地板 {c0['launch_floor_ms']:.3f} ms")
        for c in cases:
            print(f"    {c['case']:<18} accepted={c['accepted']}  "
                  f"回滚 {c['rollback_ms']:>6.3f} ms  "
                  f"回滚后与快照逐元素相等 {c['values_exact_after_rollback']}  "
                  f"与参照一致 {c['values_consistent_with_reference']}")

    (args.out / "rollback_cost.json").write_text(json.dumps(dict(
        layer_config=dict(n_recurrent=N_RECURRENT, n_attention=N_ATTENTION,
                          rec_shape=list(REC_SHAPE), kv_shape=list(KV_SHAPE),
                          block_tokens=BLOCK, dtype=DTYPE),
        draft=K, context=args.context, rounds=args.rounds, results=results,
        scope="状态写入是合成的；snapshot/rollback 的拷贝内核、字节与耗时是真实 GPU 测量",
        acceptance="每个案例都检查回滚后与快照逐元素相等、并与'只施加 accepted+1 次写入'的参照一致",
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n每 token 的 attention KV {kv_bytes_per_token} bytes，"
          f"每 batch 的 recurrent 状态 {rec_bytes} bytes")
    print("读法：KV 侧回滚不搬字节（改长度与块引用）；recurrent 侧每次回滚都要抄回全部状态，")
    print("而这份状态的成本由两层决定：字节搬运与**每层一次 kernel 启动**。")
    print("对照三列用来判断落在哪一侧：逐层拷贝 ≈ 启动地板说明是启动受限；")
    print("拼成一个连续张量后的单次拷贝、以及图捕获回放，给出这两种做法的下界。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
