#!/usr/bin/env python3
"""labs/L8/scale_out_budget.py - 8.7-C: 扩容预算 (真实轨迹 + 实测参数的离散事件模拟).

**这是模拟, 不是实测**: 到达过程取自 8.3 的真实逐请求记录, 服务能力与扩容滞后取自
8.2/8.3/8.7 的实测量, 但"什么时候扩容、扩容后队列怎么变"是在模型里推的。正文与
STATUS 必须按这个口径引用它。

模型:

  * 到达: 直接读 8.3 的 `records_*.jsonl`, 取 `planned_arrival_s` 作为到达时刻;
  * 容量: 单副本的服务率 μ 取自 8.3 实测的参考容量 R_ref (req/s); 副本数 N 线性叠加
    容量上限, 这是乐观假设 (真实系统有批处理与显存上限, 见正文的边界说明);
  * 排队时延: 用 8.3 实测的 (利用率, true TTFT p50) 曲线做分段线性插值, 超过最高
    实测点后按队列积压量线性外推, 并在输出里标注哪些点落在实测范围内;
  * 扩容: 队列长度超过阈值 K 时申请新副本, 新副本经过 T_ready 才计入容量;
    队列降到 K_low 以下持续 T_cool 秒则回收一个副本 (回收立即生效);
  * 预热: T_ready=0 代表"预热的备用副本", 用来量化预热池买到的那部分 SLO。

输出: 每个策略的 SLO 达标率、被扩容滞后拖累的请求数、峰值队列、副本秒 (含空闲副本秒)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.request_metrics import read_jsonl  # noqa: E402

# 8.3 实测: 到达率倍数 -> true TTFT p50 (秒), 以及参考容量
MEASURED_CURVE = [(0.3, 0.0602), (0.6, 0.0717), (0.9, 0.1100), (1.1, 7.8420)]


def interp_delay(util: float) -> Tuple[float, bool]:
    """由利用率插值出排队时延; 返回 (秒, 是否落在实测范围内)。"""
    if util <= MEASURED_CURVE[0][0]:
        return MEASURED_CURVE[0][1], True
    for (x0, y0), (x1, y1) in zip(MEASURED_CURVE, MEASURED_CURVE[1:]):
        if util <= x1:
            w = (util - x0) / (x1 - x0)
            return y0 + w * (y1 - y0), True
    # 超出最高实测点: 队列以 (util-1) 的速度无界增长, 用超出量线性外推
    x_last, y_last = MEASURED_CURVE[-1]
    return y_last + (util - x_last) * 10.0, False


def simulate(arrivals: List[float], mu: float, duration: float, *,
             threshold: int, low: int, cool: float, t_ready: float,
             max_replicas: int = 8, step: float = 0.1,
             slo_ttft: float = 1.0) -> Dict[str, Any]:
    """离散事件模拟: 每 step 秒推进一次, 统计排队与容量。"""
    arrivals = sorted(arrivals)
    ai = 0
    n = len(arrivals)
    inflight = 0.0          # 当前占用的并发数 (用 Little 定律折算: 占用 = 到达率 × 时延)
    capacity = mu            # 单副本容量上限, 单位 req/s
    replicas = 1
    provisioning: List[float] = []   # 每个待就绪副本的剩余时间
    t = 0.0
    peak_queue = 0.0
    wait_budget = 0.0        # 用于计算队列长度 (未服务请求数)
    pending = 0.0
    breaches = 0
    provisioned = 0
    replica_seconds = 0.0
    idle_replica_seconds = 0.0
    delay_sum = 0.0
    delay_n = 0
    out_of_measured_range = 0
    cool_since: float = -1.0
    log: List[Dict[str, Any]] = []

    while t < duration + 120 and (ai < n or pending > 0.5 or provisioning):
        # 到达
        while ai < n and arrivals[ai] <= t:
            pending += 1.0
            ai += 1
        # 副本就绪
        for i in range(len(provisioning) - 1, -1, -1):
            provisioning[i] -= step
            if provisioning[i] <= 0:
                provisioning.pop(i)
                replicas += 1
                capacity = mu * replicas
        # 排队时延 (利用率 = 到达负荷 / 容量)
        util = pending / max(capacity * step, 1e-9) * step  # 待服务量/每秒容量
        util_ratio = pending / max(capacity, 1e-9)
        delay, in_range = interp_delay(min(util_ratio, 3.0))
        if not in_range:
            out_of_measured_range += 1
        # 服务: 这一步最多处理 capacity*step 个请求
        served = min(pending, capacity * step)
        pending -= served
        peak_queue = max(peak_queue, pending)
        replica_seconds += replicas * step
        if pending < 0.5:
            idle_replica_seconds += replicas * step
        # 记录被拖累的请求: 这一步被服务的请求按当前时延判定
        if served > 0 and delay > slo_ttft:
            breaches += int(round(served))
        delay_sum += delay * served
        delay_n += served
        # 扩容决策
        if pending >= threshold and not provisioning and replicas + len(provisioning) < max_replicas:
            provisioning.append(t_ready)
            provisioned += 1
            log.append({"t_s": round(t, 1), "event": "provision", "pending": round(pending, 1),
                        "replicas": replicas, "t_ready": t_ready})
            cool_since = -1.0
        elif pending <= low:
            if cool_since < 0:
                cool_since = t
            elif t - cool_since >= cool and replicas > 1:
                replicas -= 1
                capacity = mu * replicas
                log.append({"t_s": round(t, 1), "event": "drain_replica",
                            "replicas": replicas})
                cool_since = -1.0
        else:
            cool_since = -1.0
        t += step

    mean_delay = (delay_sum / delay_n) if delay_n else None
    return {
        "arrivals": n, "replicas_final": replicas, "provisioned": provisioned,
        "peak_queue": round(peak_queue, 1), "t_ready_s": t_ready,
        "breached_requests": breaches,
        "breach_ratio": breaches / n if n else None,
        "mean_queue_delay_s": mean_delay,
        "replica_seconds": round(replica_seconds, 1),
        "idle_replica_seconds": round(idle_replica_seconds, 1),
        "steps_out_of_measured_range": out_of_measured_range,
        "log": log,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="8.3 的 records_*.jsonl")
    ap.add_argument("--ref-qps", type=float, default=10.317, help="8.3 实测参考容量")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="4,16,64")
    ap.add_argument("--ready-times", default="0,27.1,39.4,55.5",
                    help="就绪时间 (秒): 0=预热池; 27.1/39.4 来自 8.7 冷/热实测; 55.5 来自 8.2 崩溃重启")
    args = ap.parse_args()

    recs = read_jsonl(args.trace)
    arrivals = [r.planned_arrival_s for r in recs]
    t0 = min(arrivals)
    arrivals = [a - t0 for a in arrivals]
    duration = max(arrivals) + 60.0
    print(f"trace: {len(arrivals)} 请求, 时长 {max(arrivals):.1f}s, "
          f"平均到达率 {len(arrivals)/max(arrivals):.2f} qps")

    results: List[Dict[str, Any]] = []
    for thr in [int(x) for x in args.thresholds.split(",")]:
        for tr in [float(x) for x in args.ready_times.split(",")]:
            r = simulate(arrivals, args.ref_qps, duration, threshold=thr,
                         low=1, cool=10.0, t_ready=tr)
            r["policy"] = {"threshold": thr, "t_ready_s": tr}
            r.pop("log")
            results.append(r)
            print(f"thr={thr:3d} T_ready={tr:5.1f}s -> 违规 {r['breached_requests']:5d}"
                  f" ({100*(r['breach_ratio'] or 0):5.1f}%) 峰值队列 {r['peak_queue']:6.1f}"
                  f" 扩容 {r['provisioned']} 次 副本秒 {r['replica_seconds']:8.1f}"
                  f" 空闲副本秒 {r['idle_replica_seconds']:8.1f}")

    Path(args.out).write_text(json.dumps({
        "note": "模拟: 到达取自 8.3 真实记录; 容量与排队时延曲线取自 8.3 实测; 就绪时间取自 8.2/8.7 实测",
        "trace": args.trace, "ref_qps": args.ref_qps,
        "measured_curve": MEASURED_CURVE, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()