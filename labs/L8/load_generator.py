#!/usr/bin/env python3
"""labs/L8/load_generator.py - 开环/闭环负载发生器与逐请求事件采集.

同一份客户端代码同时支持两类后端, 以保证 8.3-A 的 toy server 与 8.3-B 的真实引擎
走完全相同的计时路径:

    --backend toy      POST /generate          (labs/L8/toy_server.py)
    --backend openai   POST /v1/completions    (vLLM / SGLang 的 OpenAI 兼容接口)

时间轴 (全部为同一进程的 time.monotonic, 单位秒):
    planned_arrival -> actual_send -> first_byte -> first_token -> tokens -> finish

开环模式严格按预生成的到达清单派发; 事件循环被阻塞、连接池排队或服务端背压导致的
延后都记在 `client_queue_delay = actual_send - planned_arrival` 里, 这是协调遗漏的
直接度量。闭环模式固定并发, 每个 worker 收到完整响应后再发下一个请求, 计划时间等于
实际时间, 因此**看不到**系统过载时的排队真相——这正是对比实验要展示的差别。

用法:
    python labs/L8/load_generator.py --backend toy --mode compare-omission
    python labs/L8/load_generator.py --backend openai --mode open \\
        --base-url http://127.0.0.1:8000 --rate 4 --duration 120 \\
        --prompt-len 2048 --output-len 128 --out-dir results/local/8.3/engine
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.request_metrics import (  # noqa: E402
    STATUS_ABORTED,
    STATUS_ERROR,
    STATUS_REJECTED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_TRUNCATED,
    RequestRecord,
    TEACHING_SLOS,
    correct_coordinated_omission,
    evaluate,
    summarize,
    write_jsonl,
)


# --------------------------------------------------------------------------
# 到达过程
# --------------------------------------------------------------------------
def generate_arrivals(
    mode: str,
    rate_qps: float,
    duration_s: float,
    seed: int = 0,
    burst_size: int = 20,
    burst_gap_s: float = 2.0,
) -> List[float]:
    """生成计划到达时刻 (相对 t0, 升序)。

    poisson: 指数间隔; uniform: 等间隔; burst: 每 burst_gap_s 在同一时刻投入
    burst_size 个请求 (检验突发下的排队与协调遗漏)。
    """
    rng = random.Random(seed)
    arrivals: List[float] = []
    if mode == "poisson":
        t = 0.0
        while True:
            t += -math.log(max(1e-12, rng.random())) / rate_qps
            if t > duration_s:
                break
            arrivals.append(t)
    elif mode == "uniform":
        t = 0.0
        dt = 1.0 / rate_qps
        while t <= duration_s:
            arrivals.append(t)
            t += dt
    elif mode == "burst":
        t = 0.0
        while t <= duration_s:
            for _ in range(burst_size):
                arrivals.append(min(t, duration_s))
            t += burst_gap_s
    else:
        raise ValueError(f"unknown arrival mode {mode}")
    return sorted(arrivals)


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------
@dataclasses.dataclass
class RequestSpec:
    request_id: str
    planned_arrival_s: float
    prompt_len: int
    output_len: int
    prompt_ids: Optional[List[int]] = None


class StreamingClient:
    """对一个 SSE 端点发单个请求并记录全生命周期。"""

    def __init__(self, base_url: str, backend: str, model: str,
                 timeout_s: float, t0: float, client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.model = model
        self.timeout_s = timeout_s
        self.t0 = t0
        self.client = client

    def _now(self) -> float:
        return time.monotonic()

    async def run(self, spec: RequestSpec) -> RequestRecord:
        rec = RequestRecord(
            request_id=spec.request_id,
            prompt_len=spec.prompt_len,
            requested_output_len=spec.output_len,
            planned_arrival_s=self.t0 + spec.planned_arrival_s,
        )
        actual_send = self._now()
        rec.actual_send_s = actual_send   # 先记账: 后面任何失败都留在这里

        if self.backend == "toy":
            url = f"{self.base_url}/generate"
            body: Dict[str, Any] = {
                "request_id": spec.request_id,
                "prompt_len": spec.prompt_len,
                "output_len": spec.output_len,
            }
        else:
            url = f"{self.base_url}/v1/completions"
            body = {
                "model": self.model,
                "prompt": spec.prompt_ids,
                "max_tokens": spec.output_len,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

        deadline = actual_send + self.timeout_s
        try:
            # 把 request_id 放进请求头: 网关用它做自己的决策日志主键, 这样
            # 客户端记录与网关侧 (worker、真实 cached_tokens) 才能逐请求 join。
            async with self.client.stream("POST", url, json=body,
                                          headers={"x-request-id": spec.request_id}) as resp:
                # 网关会回传它选的 worker 与预测命中, 记进 metadata 供逐请求对账。
                for hdr, key in (("x-lb-worker", "worker"),
                                 ("x-lb-predicted-hit", "predicted_hit_tokens")):
                    if hdr in resp.headers:
                        rec.metadata[key] = resp.headers[hdr]
                if resp.status_code >= 500 or resp.status_code in (429, 503):
                    await resp.aread()
                    rec.status = STATUS_REJECTED
                    rec.error = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:
                    await resp.aread()
                    rec.status = STATUS_ERROR
                    rec.error = f"HTTP {resp.status_code}"
                else:
                    await self._consume(resp, rec, deadline)
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            rec.status = STATUS_TIMEOUT
            rec.error = "client timeout"
        except httpx.HTTPError as e:
            rec.status = STATUS_ERROR
            rec.error = f"{type(e).__name__}: {e}"
        except asyncio.CancelledError:
            rec.status = STATUS_ABORTED
            rec.error = "cancelled"
            raise
        finally:
            if rec.finish_s is None:
                rec.finish_s = self._now()
        if rec.status == STATUS_SUCCESS and rec.num_tokens < spec.output_len:
            rec.status = STATUS_TRUNCATED
        return rec

    async def _consume(self, resp: httpx.Response, rec: RequestRecord, deadline: float) -> None:
        saw_done = False
        async for line in resp.aiter_lines():
            if self._now() > deadline:
                rec.status = STATUS_TIMEOUT
                rec.error = "deadline exceeded mid-stream"
                return
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if rec.first_byte_s is None:
                rec.first_byte_s = self._now()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if self.backend == "toy":
                if obj.get("type") == "done":
                    saw_done = True
                elif obj.get("type") == "token":
                    ts = self._now()
                    if rec.first_token_s is None:
                        rec.first_token_s = ts
                    rec.token_timestamps_s.append(ts)
                continue

            usage = obj.get("usage")
            if usage:
                rec.server_reported_prompt_tokens = usage.get("prompt_tokens")
                rec.server_reported_output_tokens = usage.get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and choices[0].get("finish_reason"):
                saw_done = True
            if not choices:
                continue
            text = choices[0].get("text")
            if text:
                ts = self._now()
                if rec.first_token_s is None:
                    rec.first_token_s = ts
                rec.token_timestamps_s.append(ts)
        rec.status = STATUS_SUCCESS
        if not saw_done:
            # 结束标记缺失: 计数可能仍是完整的, 但协议层没有给出"正常结束"的证据。
            rec.metadata["done_missing"] = True


# --------------------------------------------------------------------------
# 驱动器
# --------------------------------------------------------------------------
async def run_open_loop(
    base_url: str,
    arrivals: Sequence[float],
    prompt_len: int,
    output_len: int,
    backend: str = "openai",
    model: str = "",
    timeout_s: float = 60.0,
    max_connections: int = 256,
    prompt_ids: Optional[List[List[int]]] = None,
    prompt_lens: Optional[List[int]] = None,
) -> List[RequestRecord]:
    """开环: 按计划时刻异步派发, 不等待前序请求完成。"""
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    records: List[RequestRecord] = []
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        cli = StreamingClient(base_url, backend, model, timeout_s, t0, client)

        async def scheduled(idx: int, planned: float):
            now = time.monotonic() - t0
            if planned > now:
                await asyncio.sleep(planned - now)
            spec = RequestSpec(
                request_id=f"open-{idx:05d}",
                planned_arrival_s=planned,
                prompt_len=(prompt_lens[idx] if prompt_lens else prompt_len),
                output_len=output_len,
                prompt_ids=(prompt_ids[idx % len(prompt_ids)] if prompt_ids else None),
            )
            try:
                records.append(await cli.run(spec))
            except asyncio.CancelledError:
                pass

        tasks = [asyncio.create_task(scheduled(i, p)) for i, p in enumerate(arrivals)]
        await asyncio.gather(*tasks, return_exceptions=True)
    return records


async def run_closed_loop(
    base_url: str,
    concurrency: int,
    total_requests: int,
    prompt_len: int,
    output_len: int,
    backend: str = "openai",
    model: str = "",
    timeout_s: float = 60.0,
    prompt_ids: Optional[List[List[int]]] = None,
    stop_after_s: Optional[float] = None,
    t0_monotonic: Optional[float] = None,
) -> List[RequestRecord]:
    """闭环: 固定并发 worker, 每个 worker 收到响应后才发下一个。

    stop_after_s: 到达该墙钟时间后不再派发新请求 (正在处理的仍会完成)。用于
    "固定时长饱和吞吐"的基线测量。
    """
    t0 = t0_monotonic if t0_monotonic is not None else time.monotonic()
    records: List[RequestRecord] = []
    counter = 0
    lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)

    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        cli = StreamingClient(base_url, backend, model, timeout_s, t0, client)

        async def worker(wid: int):
            nonlocal counter
            while True:
                async with lock:
                    if counter >= total_requests:
                        return
                    if stop_after_s is not None and (time.monotonic() - t0) >= stop_after_s:
                        return
                    idx = counter
                    counter += 1
                spec = RequestSpec(
                    request_id=f"closed-{wid}-{idx:05d}",
                    planned_arrival_s=time.monotonic() - t0,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    prompt_ids=(prompt_ids[idx % len(prompt_ids)] if prompt_ids else None),
                )
                records.append(await cli.run(spec))

        await asyncio.gather(*[worker(w) for w in range(concurrency)])
    return records


# --------------------------------------------------------------------------
# 8.3-A: toy server 上的三个失效源
# --------------------------------------------------------------------------
async def compare_omission(out_dir: Path) -> None:
    """在 toy server 上隔离演示协调遗漏、客户端排队与事件缺失。

    场景: 服务单请求 ~0.32s (prefill 0.036s + 19 步 decode), 单次 2s 全局卡顿。
    闭环用 2 并发, 未过载时可持续 ~6.25 QPS; 开环按 5 QPS 派发, 稳态利用率 0.8。
    两者服务端配置完全相同, 差别只在到达过程是否受客户端"发完才发下一个"约束。
    """
    from labs.L8.toy_server import ServerConfig, ToyServerHandle

    out_dir.mkdir(parents=True, exist_ok=True)
    stall_start, stall_dur = 4.0, 2.0
    concurrency, duration, rate = 2, 12.0, 5.0
    prompt_len, output_len = 128, 20
    # 未过载时的可持续服务间隔: 单请求时长 / 并发数。
    service_s = 0.030 + prompt_len * 0.00005 + (output_len - 1) * 0.015
    expected_interval = service_s / concurrency

    def run_cfg():
        return ServerConfig(max_concurrency=concurrency, stall_start_s=stall_start,
                            stall_duration_s=stall_dur)

    # --- 闭环 (传统压测口径) -------------------------------------------
    srv = ToyServerHandle(run_cfg(), server_log=str(out_dir / "server_closed.jsonl")).start()
    try:
        t0 = time.monotonic()
        closed = await run_closed_loop(
            srv.base_url, concurrency=concurrency, total_requests=75,
            prompt_len=prompt_len, output_len=output_len, backend="toy", timeout_s=30.0,
        )
        closed_win = time.monotonic() - t0
    finally:
        srv.stop()
    closed_summary = evaluate(closed, "closed-loop", closed_win)
    corrected = correct_coordinated_omission(
        [r.observed_ttft_s for r in closed], expected_interval)

    # --- 开环 (计划到达口径) ------------------------------------------
    arrivals = generate_arrivals("poisson", rate, duration, seed=7)
    srv = ToyServerHandle(run_cfg(), server_log=str(out_dir / "server_open.jsonl")).start()
    try:
        t0 = time.monotonic()
        open_recs = await run_open_loop(
            srv.base_url, arrivals, prompt_len=prompt_len, output_len=output_len,
            backend="toy", timeout_s=30.0,
        )
        open_win = time.monotonic() - t0
    finally:
        srv.stop()
    open_summary = evaluate(open_recs, "open-loop", open_win, reserved_events=arrivals)

    # --- 缺失事件: 服务端丢掉结束标记 ----------------------------------
    arrivals_gap = generate_arrivals("uniform", 5.0, 4.0, seed=3)
    srv = ToyServerHandle(ServerConfig(max_concurrency=4, drop_events=("done",)),
                          server_log=str(out_dir / "server_gap.jsonl")).start()
    try:
        gap_recs = await run_open_loop(
            srv.base_url, arrivals_gap, prompt_len=prompt_len, output_len=8,
            backend="toy", timeout_s=10.0,
        )
    finally:
        srv.stop()
    gap_summary = evaluate(gap_recs, "missing-events", 4.0, reserved_events=arrivals_gap)

    write_jsonl(str(out_dir / "records_closed.jsonl"), closed)
    write_jsonl(str(out_dir / "records_open.jsonl"), open_recs)
    write_jsonl(str(out_dir / "records_gap.jsonl"), gap_recs)
    corrected_stats = summarize(corrected)

    # 闭环的"漏记到达": 客户端在卡顿期间停发, 它的完成时间线上会出现一段空档。
    # 空档长度乘以未过载时的可持续到达率, 就是这段时间里没有被任何记录覆盖的请求数。
    finishes = sorted(r.finish_s for r in closed if r.finish_s is not None)
    gaps = [b - a for a, b in zip(finishes, finishes[1:])]
    max_gap = max(gaps) if gaps else 0.0
    sustainable_rate = 1.0 / expected_interval
    omitted = {
        "max_completion_gap_s": max_gap,
        "sustainable_rate_qps": sustainable_rate,
        "implied_omitted_requests": max_gap * sustainable_rate,
        "closed_loop_window_s": closed_win,
        "closed_loop_completed": len(closed),
        "closed_loop_mean_rate_qps": len(closed) / closed_win if closed_win else None,
    }

    (out_dir / "summary.json").write_text(json.dumps(
        {"scenario": {"stall_start_s": stall_start, "stall_duration_s": stall_dur,
                      "closed_concurrency": concurrency, "open_rate_qps": rate,
                      "duration_s": duration, "service_s": service_s,
                      "expected_interval_s": expected_interval},
         "closed_loop": closed_summary.to_dict(),
         "closed_loop_co_corrected_ttft": corrected_stats,
         "closed_loop_omitted_load": omitted,
         "open_loop": open_summary.to_dict(),
         "missing_events": gap_summary.to_dict()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    def show(s, title):
        print(f"\n--- {title} ---")
        print(f"planned={s.planned_requests} counts={s.counts}")
        print(f"observed TTFT p50/p99 = {s.latency['observed_ttft']['p50']} / {s.latency['observed_ttft']['p99']}")
        print(f"true     TTFT p50/p99 = {s.latency['true_ttft']['p50']} / {s.latency['true_ttft']['p99']}")
        print(f"client queue p99      = {s.client_queue['p99']}")
        print(f"goodput(SLO-2s/100ms) = {s.goodput['SLO-2s/100ms']['ratio']}")
        print(f"event gaps            = {s.event_gaps}")

    show(closed_summary, "closed-loop (传统口径)")
    show(open_summary, "open-loop (计划到达口径)")
    show(gap_summary, "missing-events (服务端丢 done 标记)")
    print(f"\n闭环协调遗漏修正 (expected_interval={expected_interval*1000:.1f}ms): "
          f"p50={corrected_stats['p50']*1000:.1f}ms p99={corrected_stats['p99']*1000:.1f}ms "
          f"n={corrected_stats['n']} (原始样本 {len(closed)})")
    print(f"闭环完成时间线最大空档 = {max_gap*1000:.0f}ms; 按可持续 {sustainable_rate:.2f} QPS 折算, "
          f"卡顿期间约有 {omitted['implied_omitted_requests']:.1f} 个请求从未被任何记录覆盖")
    print(f"结果写入 {out_dir}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="L8 负载发生器")
    ap.add_argument("--mode", choices=["open", "closed", "compare-omission"], default="compare-omission")
    ap.add_argument("--backend", choices=["toy", "openai"], default="toy")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--arrival", choices=["poisson", "uniform", "burst"], default="poisson")
    ap.add_argument("--rate", type=float, default=4.0, help="开环到达率 QPS")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--total-requests", type=int, default=200)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="results/local/8.3/gen")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OpenAI 兼容端点不认「只给长度」的请求：prompt 必须是文本或 token id 列表。
    # 这里按 seed 合成固定 token id，保证同一 seed 的到达清单与输入可复现。
    prompt_ids = None
    if args.backend == "openai":
        rng = random.Random(args.seed)
        prompt_ids = [[rng.randint(1000, 60000) for _ in range(args.prompt_len)]]


    if args.mode == "compare-omission":
        asyncio.run(compare_omission(out_dir))
        return

    if args.mode == "open":
        arrivals = generate_arrivals(args.arrival, args.rate, args.duration, seed=args.seed)
        recs = asyncio.run(run_open_loop(
            args.base_url, arrivals, args.prompt_len, args.output_len,
            backend=args.backend, model=args.model, timeout_s=args.timeout,
            prompt_ids=prompt_ids,
        ))
        summary = evaluate(recs, args.tag, args.duration, reserved_events=arrivals)
    else:
        t0 = time.monotonic()
        recs = asyncio.run(run_closed_loop(
            args.base_url, args.concurrency, args.total_requests,
            args.prompt_len, args.output_len,
            backend=args.backend, model=args.model, timeout_s=args.timeout,
            prompt_ids=prompt_ids,
        ))
        summary = evaluate(recs, args.tag, time.monotonic() - t0)

    write_jsonl(str(out_dir / f"records_{args.tag}.jsonl"), recs)
    (out_dir / f"summary_{args.tag}.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary.to_dict()[k] for k in
                      ("tag", "planned_requests", "counts", "goodput", "throughput")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
