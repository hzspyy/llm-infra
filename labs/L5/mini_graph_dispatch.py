#!/usr/bin/env python3
"""L5.4 工程实现 · 自己写一遍 —— 一个可运行的 mini 图分派器。

vLLM 0.29.0 的 `CudaGraphManager`（`vllm/v1/worker/gpu/cudagraph_utils.py:111`）
把「用哪张图」拆成两步：初始化时按 `cudagraph_capture_sizes` 构出候选表，
运行时 `dispatch()` 只做一次查表加一次兼容性判断。本脚本用 ~90 行复现这套
规则，并用真实引擎观测到的分派结果逐条核对。

不依赖 GPU，可离线运行：
    python labs/L5/mini_graph_dispatch.py
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

# vLLM 0.29.0 在 RTX 5090 D 上的默认捕获尺寸（51 个），取自引擎的
# compilation_config.cudagraph_capture_sizes。
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
                 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
MAX_CAPTURE = 512
MAX_NUM_REQS = 256          # scheduler_config.max_num_seqs
DECODE_QUERY_LEN = 1        # 无投机时的 uniform decode 查询长度


@dataclass(frozen=True)
class Desc:
    mode: str               # "FULL" / "PIECEWISE"
    num_tokens: int
    num_reqs: int | None    # None = 不要求请求数（PIECEWISE 可吃任意请求数）
    uniform: int | None     # None = 不要求每个请求的 query 长度一致


def build_capture_descs(capture_sizes, mode="FULL_AND_PIECEWISE"):
    """按 CudaGraphManager._init_candidates 的规则构造捕获描述符。"""
    descs = []
    mixed = "PIECEWISE" if "PIECEWISE" in mode else None
    decode = "FULL" if "FULL" in mode else None
    max_decode_tokens = MAX_NUM_REQS * DECODE_QUERY_LEN

    for n in sorted(capture_sizes):
        if decode is not None and n <= max_decode_tokens:
            descs.append(Desc(decode, n, min(n, MAX_NUM_REQS), DECODE_QUERY_LEN))
        if mixed is not None:
            descs.append(Desc(mixed, n, None, None))
    return descs


def build_candidates(descs, max_size=MAX_CAPTURE):
    """候选表：num_tokens -> 可服务它的描述符，按优先级排序。

    vLLM 里的优先级来自描述符按 num_tokens 升序的区间填充，FULL 排在
    PIECEWISE 前面（先试整步图，再退到段图）。
    """
    table: dict[int, list[Desc]] = {}
    for n in range(0, max_size + 1):
        cands = [d for d in descs if d.num_tokens >= n]
        cands.sort(key=lambda d: (d.num_tokens, 0 if d.mode == "FULL" else 1))
        table[n] = cands
    return table


def compatible(desc: Desc, num_reqs: int, num_tokens: int,
               uniform_token_count: int | None) -> bool:
    """对应源码里的 _is_compatible。"""
    if desc.uniform is not None and desc.uniform != uniform_token_count:
        return False
    if desc.num_reqs is not None and desc.num_reqs < num_reqs:
        return False
    return desc.num_tokens >= num_tokens


class MiniDispatcher:
    def __init__(self, capture_sizes=CAPTURE_SIZES, mode="FULL_AND_PIECEWISE"):
        self.descs = build_capture_descs(capture_sizes, mode)
        self.table = build_candidates(self.descs)
        self.pad = {}
        sizes = sorted(capture_sizes)
        for n in range(0, MAX_CAPTURE + 1):
            i = bisect.bisect_left(sizes, n)
            self.pad[n] = sizes[i] if i < len(sizes) else None

    def dispatch(self, num_tokens, num_reqs=None, uniform_token_count=None):
        if num_tokens > MAX_CAPTURE or num_tokens not in self.table:
            return Desc("NONE", num_tokens, num_reqs, None)
        num_reqs = num_reqs if num_reqs is not None else num_tokens
        for d in self.table[num_tokens]:
            if compatible(d, num_reqs, num_tokens, uniform_token_count):
                return d
        return Desc("NONE", num_tokens, num_reqs, None)


# 真实引擎在同一台机器上观测到的分派结果（results/crater/engine/
# graph-dispatch-20260913/audit.txt 的 B 节），用来核对 mini 实现。
OBSERVED = [
    # (num_tokens, uniform_token_count, 期望 mode, 期望图 token 数)
    (512, None, "PIECEWISE", 512),      # 8 条 64-token prompt 的 prefill
    (8, 1, "FULL", 8),
    (15, 1, "FULL", 16),
    (16, 1, "FULL", 16),
    (17, 1, "FULL", 24),
    (23, 1, "FULL", 24),
    (24, 1, "FULL", 24),
    (25, 1, "FULL", 32),
    (33, 1, "FULL", 40),
    (40, 1, "FULL", 40),
    (257, None, "PIECEWISE", 272),
    (511, None, "PIECEWISE", 512),
    (513, None, "NONE", 513),
    (2048, None, "NONE", 2048),
    (3000, None, "NONE", 3000),
    (3512, None, "NONE", 3512),
]


def main():
    d = MiniDispatcher()
    n_full = sum(1 for x in d.descs if x.mode == "FULL")
    n_pw = sum(1 for x in d.descs if x.mode == "PIECEWISE")
    print(f"捕获尺寸 {len(CAPTURE_SIZES)} 个，max_cudagraph_capture_size="
          f"{MAX_CAPTURE}，max_num_seqs={MAX_NUM_REQS}")
    print(f"mini 构出描述符：FULL {n_full} 个（num_tokens ≤ {MAX_NUM_REQS}），"
          f"PIECEWISE {n_pw} 个")
    print("真实引擎在同机同配置下捕获 FULL 35 张、PIECEWISE 51 张。")

    print(f"\n  {'num_tokens':>10} {'uniform':>8} {'mini mode':>11} {'mini 图':>8} "
          f"{'实测 mode':>11} {'实测图':>8}  一致")
    bad = 0
    for nt, uni, mode, gt in OBSERVED:
        got = d.dispatch(nt, uniform_token_count=uni)
        ok = (got.mode == mode and got.num_tokens == gt)
        bad += 0 if ok else 1
        print(f"  {nt:>10} {str(uni):>8} {got.mode:>11} {got.num_tokens:>8} "
              f"{mode:>11} {gt:>8}  {'✓' if ok else '✗'}")
    print(f"\n  逐条核对：不一致 {bad} / {len(OBSERVED)}")

    print("\n  桶边界两侧的 padding（mini 的 pad 表）：")
    print(f"  {'batch':>6} {'pad 到':>7} {'浪费':>7}")
    for b in [15, 16, 17, 23, 24, 25, 32, 33, 40, 48, 64]:
        p = d.pad.get(b)
        print(f"  {b:>6} {str(p):>7} "
              f"{(p / b if p else float('nan')):>6.2f}×")
    print("\n  三条规则决定上面的表：")
    print("    1) num_tokens 超过 max_cudagraph_capture_size(512) -> NONE；")
    print("    2) 有同号的捕获尺寸就用它，否则向上取最近的桶（padding）；")
    print("    3) FULL 要求每个请求恰好 1 个 token 且请求数不超过桶大小，")
    print("       否则退到 PIECEWISE（它不限制请求数），再退到 NONE。")


if __name__ == "__main__":
    main()
