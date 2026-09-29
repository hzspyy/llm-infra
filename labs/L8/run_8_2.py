#!/usr/bin/env python3
"""labs/L8/run_8_2.py - 8.2 的真实进程级实验驱动.

两台机器上都不存在集群控制面, 因此这里只做**容器内能真实复现**的部分:

  A 段 单 worker 的冷/热启动时间线: 三级探针 (pid → /v1/models → 真实补全) 把
       ALLOCATED/LOADING/WARMING/READY 分开, 冷启动用全新的 VLLM_CACHE_ROOT。
  B 段 两个真实 replica + 持续重放: 先 SIGKILL 打掉一个 (崩溃), 再做一次
       滚动更新 (停止接新请求 → 等在途归零 → SIGTERM → 起新进程 → 等 READY)。
       客户端对失败的请求用同一 request_id 重试, 逐请求记 attempts 与最终状态,
       用来区分"静默丢失"与"重试后成功"。

调度器/集群控制面 (K8s、Slurm、Ray 集群、MIG/MPS、跨节点 gang scheduling) 不在
本机能力范围内, 只做固定版本源码与官方文档的映射, 不臆造实测。

用法:
    python labs/L8/run_8_2.py --mode A --out-dir <dir> [--cold]
    python labs/L8/run_8_2.py --mode B --out-dir <dir> --workers w0=2:19000,w1=3:19001
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.worker_lifecycle import (  # noqa: E402
    READY,
    STOPPED,
    WorkerSupervisor,
    gpu_free_mib,
)

MODEL = "Qwen/Qwen3-1.7B"


# --------------------------------------------------------------------------
# A 段: 冷/热启动时间线
# --------------------------------------------------------------------------
def run_mode_a(out: Path, model: str, gpu: int, port: int, cold: bool,
               python: str, vllm: str, cache_root: Optional[str] = None,
               label: str = "worker_a") -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    # 显式给 cache_root 才能区分冷/热: 同一个目录的第二次运行才是"热编译缓存"。
    root = Path(cache_root) if cache_root else out / ("cache_cold" if cold else "cache_warm")
    root.mkdir(parents=True, exist_ok=True)
    sup = WorkerSupervisor(label, gpu, port, model, out,
                           gpu_memory_utilization=0.35, max_model_len=4096,
                           cache_root=str(root), python=python, vllm=vllm)
    sup.start()
    if sup.state == READY:
        # 就绪后再打一次补全, 量"首个真实请求"的额外代价 (编译/捕获可能推迟到这一步)
        t_ready = sup._now()
        ok = False
        import urllib.request
        body = json.dumps({"model": model, "prompt": [9707, 11, 220],
                           "max_tokens": 8, "temperature": 0.0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        t_req0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                ok = r.status == 200
        except Exception as e:  # noqa: BLE001
            sup.event("first_request_error", error=f"{type(e).__name__}: {e}")
        sup.event("first_request_done", ok=ok, latency_s=time.monotonic() - t_req0)
        sup.event("ready_marker", t_ready_s=t_ready)
    summary = sup.phase_summary()
    sup.stop()
    sup.dump()
    summary["cold_cache"] = cold
    summary["cache_root"] = str(root)
    summary["gpu_free_after_stop_mib"] = gpu_free_mib(gpu)
    (out / f"summary_A_{label}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# B 段: 双副本 + 持续重放 + 崩溃/滚动更新
# --------------------------------------------------------------------------
@dataclasses.dataclass
class ReplayRecord:
    request_id: str
    planned_s: float
    attempts: int = 0
    first_attempt_s: Optional[float] = None
    final_s: Optional[float] = None
    status: str = "PENDING"
    workers_tried: List[int] = dataclasses.field(default_factory=list)
    output_tokens: int = 0
    error: Optional[str] = None
    waited_for_worker_s: float = 0.0   # 没有副本接单时的等待时间（不是丢失）


class Cluster:
    """极简副本注册表: 只记录哪些副本此刻愿意接新请求。"""

    def __init__(self, n: int):
        self.accepting = {i: True for i in range(n)}
        self.inflight = {i: 0 for i in range(n)}
        self.assigned = {i: 0 for i in range(n)}
        self._rr = 0

    def pick(self) -> Optional[int]:
        cands = [i for i in range(len(self.accepting)) if self.accepting[i]]
        if not cands:
            return None
        # 轮转而不是最短队列: 目的不是最优调度, 而是保证崩溃时确实有在途请求
        self._rr = (self._rr + 1) % len(cands)
        return cands[self._rr]

    def pending(self, i: int) -> int:
        return self.inflight[i]


async def replay_request(client: httpx.AsyncClient, url: str, model: str,
                         req: ReplayRecord, cluster: Cluster, worker: int,
                         prompt_len: int, output_len: int,
                         max_attempts: int = 3, timeout_s: float = 60.0) -> None:
    cluster.inflight[worker] += 1
    cluster.assigned[worker] += 1
    body = {
        "model": model,
        "prompt": [9707] * prompt_len,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        for attempt in range(1, max_attempts + 1):
            req.attempts = attempt
            req.workers_tried.append(worker)
            if req.first_attempt_s is None:
                req.first_attempt_s = time.monotonic()
            try:
                async with client.stream("POST", f"{url}/v1/completions", json=body,
                                         timeout=timeout_s) as r:
                    if r.status_code != 200:
                        await r.aread()
                        req.error = f"HTTP {r.status_code}"
                        raise RuntimeError(req.error)
                    usage = None
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload in ("[DONE]", ""):
                            continue
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                    req.output_tokens = (usage or {}).get("completion_tokens", 0)
                    req.status = "SUCCESS"
                    return
            except Exception as e:  # noqa: BLE001
                req.error = f"{type(e).__name__}: {e}"
                # 换一个仍然接单的副本重试, 保持同一个 request_id (幂等键)
                nxt = cluster.pick()
                if nxt is None or attempt == max_attempts:
                    req.status = "FAILED"
                    return
                worker = nxt
                await asyncio.sleep(0.05)
    finally:
        cluster.inflight[worker] -= 1
        req.final_s = time.monotonic()


async def run_mode_b(out: Path, workers: List[Tuple[str, int, int]], model: str,
                     rate: float, duration: float, crash_at: float,
                     roll_at: float, prompt_len: int, output_len: int,
                     cold: bool, python: str, vllm: str) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    sups: List[WorkerSupervisor] = []
    for name, gpu, port in workers:
        cache_root = out / "cache"
        cache_root.mkdir(exist_ok=True)
        sups.append(WorkerSupervisor(name, gpu, port, model, out,
                                     gpu_memory_utilization=0.35, max_model_len=4096,
                                     cache_root=str(cache_root), python=python, vllm=vllm))
    # 两个副本并发启动: 串行会把启动时间叠加成两倍, 掩盖真实的扩容延迟。
    loop = asyncio.get_event_loop()
    await asyncio.gather(*[loop.run_in_executor(None, s.start) for s in sups])
    cluster = Cluster(len(sups))
    urls = [f"http://127.0.0.1:{p}" for _, _, p in workers]

    rng = random.Random(0)
    n_planned = int(rate * duration)
    arrivals: List[float] = []
    t = 0.0
    for _ in range(n_planned):
        t += -__import__("math").log(max(1e-12, rng.random())) / rate
        arrivals.append(t)
    records = [ReplayRecord(request_id=f"r{i:05d}", planned_s=a)
               for i, a in enumerate(arrivals)]

    t0 = time.monotonic()
    events: List[Dict[str, Any]] = []

    def log_event(kind: str, **f: Any) -> None:
        rec = {"t_s": time.monotonic() - t0, "event": kind, **f}
        events.append(rec)
        print(f"[B] {rec['t_s']:7.3f}s {kind} {f}", flush=True)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        async def scheduled(rec: ReplayRecord) -> None:
            now = time.monotonic() - t0
            if rec.planned_s > now:
                await asyncio.sleep(rec.planned_s - now)
            w = cluster.pick()
            if w is None:
                # 没有任何副本接单时, 真实客户端/网关会等而不是立刻失败。
                # 这里等待有上限, 等不到才记 FAILED —— "静默丢失"必须与"等待"分开。
                t_wait = time.monotonic()
                while time.monotonic() - t_wait < 180.0:
                    await asyncio.sleep(0.05)
                    w = cluster.pick()
                    if w is not None:
                        break
                rec.waited_for_worker_s = time.monotonic() - t_wait
                if w is None:
                    rec.status = "FAILED"
                    rec.error = "no accepting worker within 180s"
                    return
            await replay_request(client, urls[w], model, rec, cluster, w,
                                 prompt_len, output_len)

        tasks = [asyncio.create_task(scheduled(r)) for r in records]

        # 周期性采样总在途数, 用来确认崩溃/滚动更新时确实有请求在途。
        max_inflight = {"v": 0}

        async def sample_inflight() -> None:
            while True:
                max_inflight["v"] = max(max_inflight["v"], sum(cluster.inflight.values()))
                await asyncio.sleep(0.05)

        # 崩溃: 直接 SIGKILL 一个副本 (不排空), 之后重启它
        async def crash_phase() -> None:
            await asyncio.sleep(crash_at)
            victim = 0
            cluster.accepting[victim] = False
            log_event("crash_begin", worker=sups[victim].name,
                      inflight=cluster.pending(victim))
            await loop.run_in_executor(None, sups[victim].crash)
            log_event("crash_done", worker=sups[victim].name,
                      inflight=cluster.pending(victim))
            sups[victim].name = f"{sups[victim].name}_restart"
            await loop.run_in_executor(None, sups[victim].start)
            cluster.accepting[victim] = True
            log_event("crash_restart_ready", worker=sups[victim].name)

        # 滚动更新: 排空 → SIGTERM → 起新进程 → READY
        async def roll_phase() -> None:
            await asyncio.sleep(roll_at)
            victim = 1 if len(sups) > 1 else 0
            cluster.accepting[victim] = False
            log_event("roll_drain_begin", worker=sups[victim].name,
                      inflight=cluster.pending(victim))
            waited = await loop.run_in_executor(
                None, sups[victim].drain, (lambda v=victim: cluster.pending(v)))
            log_event("roll_drain_done", worker=sups[victim].name, waited_s=waited)
            await loop.run_in_executor(None, sups[victim].stop)
            log_event("roll_stopped", worker=sups[victim].name)
            sups[victim].name = f"{sups[victim].name}_roll"
            await loop.run_in_executor(None, sups[victim].start)
            cluster.accepting[victim] = True
            log_event("roll_new_ready", worker=sups[victim].name)

        sampler = asyncio.create_task(sample_inflight())
        await asyncio.gather(*tasks, crash_phase(), roll_phase())
        sampler.cancel()
    window = time.monotonic() - t0

    for s in sups:
        s.stop()
        s.dump()

    counts: Dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    retried = [r for r in records if r.attempts > 1]
    summary = {
        "planned": len(records),
        "rate_qps": rate,
        "window_s": window,
        "crash_at_s": crash_at,
        "roll_at_s": roll_at,
        "counts": counts,
        "retried_requests": len(retried),
        "retry_attempts_hist": {str(k): sum(1 for r in records if r.attempts == k)
                                for k in sorted({r.attempts for r in records})},
        "max_inflight": max_inflight["v"],
        "waiting_requests": sum(1 for r in records if r.waited_for_worker_s > 0),
        "max_worker_wait_s": max((r.waited_for_worker_s for r in records), default=0.0),
        "failed_requests": [dataclasses.asdict(r) for r in records if r.status != "SUCCESS"][:50],
        "gpu_free_after": {s.name: gpu_free_mib(s.gpu_index) for s in sups},
        "worker_phases": [s.phase_summary() for s in sups],
        "worker_segments": [seg for s in sups for seg in s.phase_segments()],
    }
    with (out / "replay_records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")
    with (out / "cluster_events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "summary_B.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("planned", "counts", "retried_requests", "retry_attempts_hist")},
                     ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=19000)
    ap.add_argument("--workers", default="w0=2:19000,w1=3:19001")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--cache-root", default=None,
                    help="显式指定编译/图缓存目录; 同一目录第二次运行即热缓存")
    ap.add_argument("--label", default="worker_a", help="A 段本次运行的标签")
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--crash-at", type=float, default=25.0)
    ap.add_argument("--roll-at", type=float, default=60.0)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--python", default="python")
    ap.add_argument("--vllm", default="vllm")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.mode == "A":
        s = run_mode_a(out, args.model, args.gpu, args.port, args.cold,
                       args.python, args.vllm, cache_root=args.cache_root,
                       label=args.label)
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        wl = []
        for item in args.workers.split(","):
            name, rest = item.split("=")
            gpu, port = rest.split(":")
            wl.append((name, int(gpu), int(port)))
        asyncio.run(run_mode_b(out, wl, args.model, args.rate, args.duration,
                               args.crash_at, args.roll_at, args.prompt_len,
                               args.output_len, args.cold, args.python, args.vllm))


if __name__ == "__main__":
    main()
