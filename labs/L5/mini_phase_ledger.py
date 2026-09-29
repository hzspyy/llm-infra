#!/usr/bin/env python3
"""L5.1 mini 实现 · 阶段判定与逐 step 读写账。

`labs/L5/prefill_decode.py` 用公式算 roofline；`serve_protocol.py` 从真实引擎
采逐 step 事件。本文件把两者接起来，只做两件事：

  1. **阶段判定**：给定 (本步之前已计算 token, 本步之后, prompt 长度)，
     判定这一步属于 prefill / prefill-chunk / prefill-last / decode。
     判据只用调度器的两个计数，不用调用耗时差。
  2. **逐 step 账**：给定阶段、本步调度的 token 数与历史长度，算出
     每层 Q/K/V 形状、权重读字节、KV 读写字节、FLOP 与算术强度。

再用它核对已有工件里的不变量：

  * 每个 decode 步的调度 token 数必须是 1；
  * prefill-last 必须满足 before < prompt <= after；
  * 一条请求各步调度 token 之和 = prompt + 生成 token 数；
  * 首输出所在步必须就是 prefill-last（首 token 由这一前向的 logits 产生）。

用法：
    python mini_phase_ledger.py --results <results/crater/5.1> [--out <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

CFG = dict(hidden=2048, inter=6144, layers=28, heads=16, kv_heads=8,
           head_dim=128, vocab=151936, dtype_bytes=2)


def params(cfg=CFG) -> int:
    H, I, L = cfg["hidden"], cfg["inter"], cfg["layers"]
    nq, nkv, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    return L * (H * nq * hd + 2 * H * nkv * hd + nq * hd * H + 3 * H * I) \
        + cfg["vocab"] * H


def weight_bytes(cfg=CFG) -> int:
    return params(cfg) * cfg["dtype_bytes"]


def kv_bytes_per_token(cfg=CFG) -> int:
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * cfg["dtype_bytes"]


def classify(before: int, after: int, prompt: int) -> str:
    """只用两个计数判定阶段。"""
    if after < prompt and before == 0:
        return "prefill"
    if after < prompt:
        return "prefill-chunk"
    if before < prompt <= after:
        return "prefill-last"
    return "decode"


def step_ledger(phase: str, n_sched: int, hist: int, cfg=CFG) -> dict:
    """一步的读写账。hist 是这一步之后的已计算长度（= KV 里的历史位置数）。"""
    nq, nkv, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    flops = 2 * params(cfg) * n_sched
    wb = weight_bytes(cfg)
    kv_read = hist * kv_bytes_per_token(cfg)
    kv_write = n_sched * kv_bytes_per_token(cfg)
    return dict(
        phase=phase, sched=n_sched, hist=hist,
        q_shape=(1, nq, n_sched, hd), kv_shape=(1, nkv, hist, hd),
        weight_bytes=wb, kv_read_bytes=kv_read, kv_write_bytes=kv_write,
        flops=flops, intensity=flops / (wb + kv_read),
    )


def roofline_table(balance: float, out: list[str]) -> None:
    out.append(f"权重模型平衡点 {balance:.1f} FLOP/byte（bf16 线性层的简化模型）")
    out.append(f"  {'相':<9}{'batch':>6}{'序列':>7}{'FLOP':>12}{'权重字节':>14}"
               f"{'算术强度':>10}{'落点':>8}")
    rows = [("prefill", 1, 2048), ("prefill", 1, 4096), ("decode", 1, 1),
            ("decode", 8, 1), ("decode", 64, 1), ("decode", 256, 1)]
    for phase, b, s in rows:
        n = b * s
        flops = 2 * params() * n
        wb = weight_bytes()
        inten = flops / wb                     # 简化：只算权重读取
        out.append(f"  {phase:<9}{b:>6}{s:>7}{flops:>12.3e}{wb:>14.3e}"
                   f"{inten:>10.1f}{'算力' if inten > balance else '带宽':>8}")
    out.append("")


def check_invariants(results_dir: str, out: list[str]) -> dict:
    """用 mini 的判据核对真实工件的逐 step 记录。"""
    tot = dict(files=0, steps=0, reqs=0, bad_phase=0, bad_decode=0,
               bad_sum=0, bad_first=0, exact_sum=0, delta=[], phase_counts={})
    for path in sorted(glob.glob(os.path.join(results_dir, "serve-*", "serve_*.json"))):
        with open(path) as f:
            blob = json.load(f)
        tot["files"] += 1
        for run in blob["runs"]:
            # 每条请求：各步调度 token 之和 == prompt + 生成
            per_req = {}
            for st in run["steps"]:
                for rid, r in st["reqs"].items():
                    mine = classify(r["before"], r["after"], r["prompt"])
                    tot["steps"] += 1
                    tot["phase_counts"][mine] = tot["phase_counts"].get(mine, 0) + 1
                    if mine != r["phase"]:
                        tot["bad_phase"] += 1
                    if mine == "decode" and r["sched"] != 1:
                        tot["bad_decode"] += 1
                    if mine == "prefill-last" and not (r["before"] < r["prompt"]
                                                       <= r["after"]):
                        tot["bad_phase"] += 1
                    per_req.setdefault(rid, 0)
                    per_req[rid] += r["sched"]
            for req in run["requests"]:
                tot["reqs"] += 1
                # 请求 id 在 step 记录里带 uuid 后缀，这里按前缀匹配
                cand = [k for k in per_req
                        if k.rsplit("-", 1)[0] == req["request_id"]]
                sched_sum = sum(per_req[k] for k in cand)
                # 首个输出由 prefill 那次前向产生，所以不额外耗一个 decode 前向：
                # 没有抢占时，各步调度 token 之和 = prompt + 生成 - 1
                base = req["prompt_tokens"] + req["out_tokens"] - 1
                tot["delta"].append(sched_sum - base)
                if sched_sum < base:
                    tot["bad_sum"] += 1
                if sched_sum == base:
                    tot["exact_sum"] += 1
                if req["first_emit_step"] != (req["prefill_done_step"] or 0) + 1 \
                        and req["step_lag"] != 1:
                    tot["bad_first"] += 1
    from collections import Counter
    tot["delta_hist"] = dict(Counter(tot.pop("delta")))
    out.append(f"不变量核对：{tot['files']} 个工件目录、{tot['steps']} 个 (step, 请求) 条目、"
               f"{tot['reqs']} 条请求")
    out.append(f"  阶段计数 {tot['phase_counts']}")
    out.append(f"  阶段判定不一致 {tot['bad_phase']}、decode 步调度 token≠1 {tot['bad_decode']}")
    out.append(f"  各步调度 token 之和：等于 prompt+生成-1 的 {tot['exact_sum']} 条，"
               f"低于下界的 {tot['bad_sum']} 条")
    out.append(f"  超出下界的量（抢占重算会把它推高）："
               f"{ {k: v for k, v in sorted(tot['delta_hist'].items()) if k} }")
    out.append(f"  首输出不在 prefill-last 后继步 {tot['bad_first']}")
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/crater/5.1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.results, "..", "..", "local", "5.1",
                                       "mini-phase-ledger")
    os.makedirs(out_dir, exist_ok=True)
    out: list[str] = []
    out.append(f"L5.1 mini：阶段判定 + 逐 step 读写账 · 模型配置 {CFG}")
    out.append(f"params {params() / 1e9:.2f}B  权重 {weight_bytes() / 2**30:.2f} GiB  "
               f"KV {kv_bytes_per_token()} B/token")
    out.append("")

    bal = 232000 / 1608.6            # 0.3 的实测算力与带宽
    roofline_table(bal, out)

    out.append("单请求逐 step 账（prompt=64，历史长度取这一步之后的 computed）")
    out.append(f"  {'step':>4}{'阶段':>13}{'本步token':>9}{'Q(每层)':>17}"
               f"{'K/V读取(每层)':>18}{'KV读 B':>10}{'FLOP':>12}{'强度':>8}")
    hist = 0
    for i, n in enumerate([64, 1, 1, 1]):
        phase = "prefill-last" if i == 0 else "decode"
        hist += n
        L = step_ledger(phase, n, hist)
        out.append(f"  {i:>4}{phase:>13}{n:>9}{str(L['q_shape']):>17}"
                   f"{str(L['kv_shape']):>18}{L['kv_read_bytes']:>10}"
                   f"{L['flops']:>12.3e}{L['intensity']:>8.3f}")
    out.append("")

    inv = check_invariants(args.results, out)
    text = "\n".join(out)
    print(text)
    with open(os.path.join(out_dir, "mini_phase_ledger.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "mini_phase_ledger.json"), "w") as f:
        json.dump(dict(cfg=CFG, params=params(), weight_bytes=weight_bytes(),
                       kv_bytes_per_token=kv_bytes_per_token(),
                       invariants=inv), f, indent=1)
    print(f"\n写入 {out_dir}/mini_phase_ledger.txt")


if __name__ == "__main__":
    main()
