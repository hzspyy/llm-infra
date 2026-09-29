#!/usr/bin/env python3
"""labs/L8/admission_budget.py - 8.4-D: 按任务预算的准入控制与可复算容量边界.

在混部(两个引擎同一张卡)之上加一层**应用侧准入**: 每个任务类各有一个令牌桶
(每秒允许进入的 prompt token 数) 与一个在途请求上限。请求到达时先问控制器:

  令牌不足  -> 拒绝, 计成 REJECTED(admission), 不进引擎
  在途已满  -> 同上
  通过      -> 发到对应引擎, 完成/失败都按类记账

同一份到达清单在若干组 (gen 预算, emb 预算) 下重放, 每类分别报告接受率、TTFT
分位与 SLO 达标率。容量边界就是"两类都满足各自 SLO 的最大预算组合"。控制器是
应用层的, 因此它**只能决定谁进来**, 不能改变进来之后两类共享算力的事实——这正是
边界曲线要说明的: 准入能把混合负载拉回 SLO, 代价是拒绝。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx  # noqa: E402

from labs.L8.load_generator import RequestSpec, StreamingClient  # noqa: E402
from labs.L8.mixed_workload import (  # noqa: E402
    EMB_SNAP,
    GEN_SNAP,
    GpuSampler,
    Server,
    build_servers,
    make_prompt_ids,
    poisson_arrivals,
)

SLO_TTFT_S = 1.0
SLO_GEN_TOTAL_S = 8.0
SLO_EMB_TOTAL_S = 1.0


class AdmissionController:
    """按类别的令牌桶 + 在途上限。"""

    def __init__(self, budgets: Dict[str, Dict[str, float]]):
        self.budgets = budgets
        self.tokens: Dict[str, float] = {}
        self.last: Dict[str, float] = {}
        self.in_flight: Dict[str, int] = {}
        self.rejected: Dict[str, int] = {}

    def _refill(self, cls: str, now: float) -> None:
        b = self.budgets[cls]
        last = self.last.get(cls, now)
        self.tokens[cls] = min(b["tokens_per_s"], self.tokens.get(cls, b["tokens_per_s"])
                               + (now - last) * b["tokens_per_s"])
        self.last[cls] = now

    def check(self, cls: str, prompt_tokens: int, now: float) -> Tuple[bool, str]:
        self._refill(cls, now)
        b = self.budgets[cls]
        if self.in_flight.get(cls, 0) >= b["max_inflight"]:
            self.rejected[cls] = self.rejected.get(cls, 0) + 1
            return False, "inflight"
        if self.tokens[cls] < prompt_tokens:
            self.rejected[cls] = self.rejected.get(cls, 0) + 1
            return False, "tokens"
        self.tokens[cls] -= prompt_tokens
        self.in_flight[cls] = self.in_flight.get(cls, 0) + 1
        return True, ""

    def done(self, cls: str) -> None:
        self.in_flight[cls] = max(0, self.in_flight.get(cls, 0) - 1)


async def controlled_gen(port: int, model: str, arrivals: List[float], prompt_ids: List[int],
                         output_len: int, timeout_s: float, ctrl: AdmissionController,
                         t0: float) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        cli = StreamingClient(f"http://127.0.0.1:{port}", "openai", model, timeout_s, t0,
                              client)

        async def one(i: int, arr: float):
            await asyncio.sleep(max(0.0, t0 + arr - time.monotonic()))
            now = time.monotonic()
            ok, why = ctrl.check("gen", len(prompt_ids), now)
            if not ok:
                return {"id": f"gen-{i}", "class": "gen", "planned": arr,
                        "status": "REJECTED", "reason": why, "ttft": None,
                        "finish": now - t0}
            try:
                spec = RequestSpec(f"gen-{i}", arr, len(prompt_ids), output_len, prompt_ids)
                r = await cli.run(spec)
                return {"id": r.request_id, "class": "gen", "planned": arr,
                        "send": None if r.actual_send_s is None else r.actual_send_s - t0,
                        "finish": None if r.finish_s is None else r.finish_s - t0,
                        "ttft": r.observed_ttft_s, "tpot": r.tpot_s, "status": r.status,
                        "tokens": len(r.token_timestamps_s)}
            finally:
                ctrl.done("gen")

        recs = list(await asyncio.gather(*[one(i, a) for i, a in enumerate(arrivals)]))
    return recs


async def controlled_emb(port: int, model: str, arrivals: List[float], prompt_ids: List[int],
                         timeout_s: float, ctrl: AdmissionController,
                         t0: float) -> List[Dict[str, Any]]:
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def one(i: int, arr: float):
            await asyncio.sleep(max(0.0, t0 + arr - time.monotonic()))
            now = time.monotonic()
            ok, why = ctrl.check("emb", len(prompt_ids), now)
            if not ok:
                return {"id": f"emb-{i}", "class": "emb", "planned": arr,
                        "status": "REJECTED", "reason": why, "ttft": None,
                        "finish": now - t0}
            try:
                send = time.monotonic()
                status, err = "SUCCESS", None
                try:
                    r = await client.post(f"http://127.0.0.1:{port}/v1/embeddings",
                                          json={"model": model, "input": prompt_ids})
                    if r.status_code >= 500 or r.status_code in (429, 503):
                        status, err = "REJECTED", f"HTTP {r.status_code}"
                    elif r.status_code != 200:
                        status, err = "ERROR", f"HTTP {r.status_code}"
                except (httpx.TimeoutException, httpx.HTTPError) as e:
                    status = "TIMEOUT" if isinstance(e, httpx.TimeoutException) else "ERROR"
                    err = type(e).__name__
                finish = time.monotonic()
                return {"id": f"emb-{i}", "class": "emb", "planned": arr,
                        "send": send - t0, "finish": finish - t0,
                        "ttft": finish - send, "status": status, "error": err,
                        "tokens": len(prompt_ids)}
            finally:
                ctrl.done("emb")

        return list(await asyncio.gather(*[one(i, a) for i, a in enumerate(arrivals)]))


def summarize(recs: List[Dict[str, Any]], cls: str) -> Dict[str, Any]:
    ok = [r for r in recs if r["status"] in ("SUCCESS", "TRUNCATED")]
    lats = sorted(r["finish"] - r["planned"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    def q(xs, p):
        if not xs:
            return None
        return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1)))], 4)
    slo_total = SLO_GEN_TOTAL_S if cls == "gen" else SLO_EMB_TOTAL_S
    attain = (sum(1 for r in ok if (r["finish"] - r["planned"]) <= slo_total
                  and (r["ttft"] is None or r["ttft"] <= SLO_TTFT_S)) / len(recs)
              if recs else None)
    return {
        "offered": len(recs), "ok": len(ok),
        "rejected_admission": sum(1 for r in recs if r["status"] == "REJECTED"
                                  and r.get("reason") in ("tokens", "inflight")),
        "statuses": {s: sum(1 for r in recs if r["status"] == s)
                     for s in {r["status"] for r in recs}},
        "accept_rate": round(len(ok) / len(recs), 4) if recs else None,
        "latency_p50": q(lats, 0.5), "latency_p95": q(lats, 0.95),
        "latency_p99": q(lats, 0.99),
        "ttft_p50": q(ttfts, 0.5), "ttft_p99": q(ttfts, 0.99),
        "slo_attainment_including_rejects": None if attain is None else round(attain, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen-port", type=int, default=18091)
    ap.add_argument("--emb-port", type=int, default=18092)
    ap.add_argument("--gen-rate", type=float, default=4.0)
    ap.add_argument("--emb-rate", type=float, default=8.0)
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--gen-prompt-len", type=int, default=512)
    ap.add_argument("--emb-prompt-len", type=int, default=512)
    ap.add_argument("--gen-output-len", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--gen-budgets", default="1,2,4",
                    help="gen 令牌桶 (prompt token/s) 的扫描点")
    ap.add_argument("--emb-budgets", default="4,8,16",
                    help="emb 令牌桶扫描点")
    ap.add_argument("--gen-inflight", type=int, default=8)
    ap.add_argument("--emb-inflight", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-snap", default=GEN_SNAP)
    ap.add_argument("--emb-snap", default=EMB_SNAP)
    ap.add_argument("--gen-gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--emb-gpu-mem-util", type=float, default=0.12)
    ap.add_argument("--gen-max-model-len", type=int, default=4096)
    ap.add_argument("--emb-max-model-len", type=int, default=2048)
    ap.add_argument("--intervention", default="none")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.share_emb = args.emb_rate / (args.gen_rate + args.emb_rate)
    args.gen_prompt_ids = make_prompt_ids(args.gen_prompt_len, args.seed + 1)
    args.emb_prompt_ids = make_prompt_ids(args.emb_prompt_len, args.seed + 2)
    servers = build_servers(args, out_dir, True, True, "none")
    for s in servers:
        s.start()
    gen_srv = next(s for s in servers if s.name == "gen")
    emb_srv = next(s for s in servers if s.name == "emb")
    gen_arr = poisson_arrivals(args.gen_rate, args.duration_s, args.seed)
    emb_arr = poisson_arrivals(args.emb_rate, args.duration_s, args.seed + 9)

    rows: List[Dict[str, Any]] = []
    try:
        for gb in [float(x) for x in args.gen_budgets.split(",")]:
            for eb in [float(x) for x in args.emb_budgets.split(",")]:
                ctrl = AdmissionController({
                    "gen": {"tokens_per_s": gb * args.gen_prompt_len,
                            "max_inflight": args.gen_inflight},
                    "emb": {"tokens_per_s": eb * args.emb_prompt_len,
                            "max_inflight": args.emb_inflight},
                })
                sampler = GpuSampler(0.5)
                sampler.start()
                t0 = time.monotonic()
                try:
                    async def both():
                        return await asyncio.gather(
                            controlled_gen(gen_srv.port, "gen", gen_arr,
                                           args.gen_prompt_ids, args.gen_output_len,
                                           args.timeout_s, ctrl, t0),
                            controlled_emb(emb_srv.port, "emb", emb_arr,
                                           args.emb_prompt_ids, args.timeout_s, ctrl, t0))
                    gen_recs, emb_recs = asyncio.run(both())
                finally:
                    samples = sampler.stop()
                row = {
                    "gen_budget_qps": gb, "emb_budget_qps": eb,
                    "gen": summarize(gen_recs, "gen"), "emb": summarize(emb_recs, "emb"),
                    "gpu_peak_mib": max((s["mem_mib"] for s in samples), default=None),
                    "gpu_mean_util_pct": round(sum(s["util_pct"] for s in samples)
                                               / max(1, len(samples)), 2),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False)[:420], flush=True)
    finally:
        for s in servers:
            s.stop()
    # 容量边界是一族阈值曲线, 不是一个最优点: 对每个达标率阈值取"总接受量"最大的组合
    boundaries = []
    for th in (0.5, 0.8, 0.9, 0.99):
        feas = [r for r in rows
                if (r["gen"]["slo_attainment_including_rejects"] or 0) >= th
                and (r["emb"]["slo_attainment_including_rejects"] or 0) >= th]
        b = max(feas, key=lambda r: r["gen"]["ok"] + r["emb"]["ok"]) if feas else None
        boundaries.append({
            "min_attainment": th,
            "best": None if b is None else {
                "gen_budget_qps": b["gen_budget_qps"], "emb_budget_qps": b["emb_budget_qps"],
                "gen_ok": b["gen"]["ok"], "emb_ok": b["emb"]["ok"],
                "gen_accept_rate": b["gen"]["accept_rate"],
                "emb_accept_rate": b["emb"]["accept_rate"]},
        })
    best = next((x["best"] for x in boundaries if x["min_attainment"] == 0.99), None)
    summary = {
        "offered": {"gen_rate": args.gen_rate, "emb_rate": args.emb_rate,
                    "gen_prompt_len": args.gen_prompt_len,
                    "emb_prompt_len": args.emb_prompt_len,
                    "duration_s": args.duration_s},
        "slo": {"gen_total_s": SLO_GEN_TOTAL_S, "emb_total_s": SLO_EMB_TOTAL_S,
                "ttft_s": SLO_TTFT_S},
        "rows": rows,
        "capacity_boundary": best,
        "boundaries_by_threshold": boundaries,
        "note": "SLO 在扫描前冻结; 接受率与拒绝都进分母; 边界只覆盖本次两类负载与这张卡",
    }
    (out_dir / "admission_budget.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("capacity boundary:", json.dumps(best, ensure_ascii=False)[:400] if best else "无")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
