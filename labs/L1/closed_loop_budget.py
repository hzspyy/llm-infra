#!/usr/bin/env python3
"""L1.7 lab · 实时闭环的延迟预算：输入年龄、排队、抖动与过期动作。

闭环系统的关键量不是「推理多久」，而是**决策时刻手上的输入有多旧**。
这个 lab 把一条最小闭环拆成可观测的五个事件：

    采样 sample → 入队 enqueue → 推理开始 infer_start → 推理结束 infer_end
                → 动作取用 action_use（或丢弃 action_drop）

三种运行模式：

  run     固定时间戳重放 + 周期消费者，扫描频率/队列容量/策略；
          抖动注入 5/20/50 ms；可对接本地或远程推理后端
  agent   把一个「端侧推理 agent」跑成 HTTP 服务，用自己的时钟打点，
          并支持注入时钟偏移（--clock-offset-ms）以检验校准
  calibrate  只跑 NTP 式的偏移估计，报告 offset ± RTT/2

后端三选一：
  --backend sleep   纯延迟模型（可复现，用来验证时序逻辑本身）
  --backend http    真实调用 llama-server /completion（同机或远程）
  --backend agent   通过 agent 协议调用（含跨设备时钟校准）

用法：
    # 本地验证时序逻辑
    python closed_loop_budget.py --mode run --backend sleep --latency-ms 40
    # 真实端侧推理：agent 在 Jetson 上，harness 通过 agent 协议驱动
    python closed_loop_budget.py --mode agent --port 19080 --backend http \
        --endpoint http://127.0.0.1:8080/completion --clock-offset-ms 250
    python closed_loop_budget.py --mode run --backend agent \
        --agent-url http://127.0.0.1:19080 --freq 10 --queue 4 --policy latest
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

T0 = time.monotonic()


def now_ms() -> float:
    """所有模式共用的单调时钟（毫秒）。跨设备时各进程有各自的 T0。"""
    return (time.monotonic() - T0) * 1000.0


# ------------------------------------------------------------------ event log
class EventLog:
    def __init__(self, path: Path | None):
        self.path = path
        self.fh = open(path, "a") if path else None

    def write(self, ev: dict) -> None:
        if not self.fh:
            return
        ev = {"t": round(ev.get("t", now_ms()), 3), **ev}
        self.fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def close(self):
        if self.fh:
            self.fh.close()


# ------------------------------------------------------------------ backends
@dataclass
class InferenceResult:
    text: str
    latency_ms: float
    remote_start: float | None = None
    remote_end: float | None = None


class SleepBackend:
    """纯延迟模型：只用来验证排队与过期策略，不代表任何真实硬件。"""

    name = "sleep"

    def __init__(self, latency_ms: float, jitter_ms: float = 0.0, seed: int = 0):
        self.latency = latency_ms
        self.jitter = jitter_ms
        self.rng = random.Random(seed)

    def infer(self, prompt: str, n_predict: int = 32) -> InferenceResult:
        lat = self.latency + (self.rng.uniform(-self.jitter, self.jitter)
                              if self.jitter else 0.0)
        lat = max(0.0, lat)
        time.sleep(lat / 1000.0)
        return InferenceResult(text="x" * n_predict, latency_ms=lat)


class HttpBackend:
    """真实推理：llama-server 的 /completion，非流式，拿服务端 timings。"""

    name = "http"

    def __init__(self, endpoint: str, n_predict: int = 32, timeout: float = 600.0):
        self.endpoint = endpoint
        self.n_predict = n_predict
        self.timeout = timeout

    def infer(self, prompt: str, n_predict: int | None = None) -> InferenceResult:
        payload = json.dumps({"prompt": prompt,
                              "n_predict": n_predict or self.n_predict,
                              "temperature": 0.0, "stream": False,
                              "cache_prompt": False}).encode()
        req = urllib.request.Request(self.endpoint, data=payload,
                                     headers={"Content-Type": "application/json"})
        t0 = now_ms()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read().decode())
        return InferenceResult(text=body.get("content", ""),
                               latency_ms=now_ms() - t0)


class AgentBackend:
    """通过 agent 协议调用远程推理，并在启动时做一次时钟校准。"""

    name = "agent"

    def __init__(self, url: str, n_predict: int = 32):
        self.url = url.rstrip("/")
        self.n_predict = n_predict
        self.offset_ms = 0.0
        self.rtt_ms = 0.0

    def _post(self, path: str, payload: dict, timeout: float = 600.0) -> dict:
        req = urllib.request.Request(self.url + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def calibrate(self, rounds: int = 9) -> dict:
        """NTP 式偏移估计：offset = ((t2-t1) + (t3-t4)) / 2，精度受 RTT/2 限制。"""
        samples = []
        for _ in range(rounds):
            t1 = now_ms()
            r = self._post("/ping", {"t1": t1}, timeout=10)
            t4 = now_ms()
            t2, t3 = r["t2"], r["t3"]
            off = ((t2 - t1) + (t3 - t4)) / 2.0
            samples.append({"offset_ms": off, "rtt_ms": t4 - t1})
        best = min(samples, key=lambda s: s["rtt_ms"])       # 取 RTT 最小的一次
        self.offset_ms = best["offset_ms"]
        self.rtt_ms = best["rtt_ms"]
        return {"offset_ms": round(self.offset_ms, 3),
                "rtt_ms": round(self.rtt_ms, 3),
                "samples": [{"offset_ms": round(s["offset_ms"], 2),
                             "rtt_ms": round(s["rtt_ms"], 2)} for s in samples],
                "precision_bound_ms": round(self.rtt_ms / 2, 3)}

    def to_local(self, remote_t: float) -> float:
        """把 agent 时钟域的时间戳换到本地时钟域。"""
        return remote_t - self.offset_ms

    def infer(self, prompt: str, n_predict: int | None = None) -> InferenceResult:
        r = self._post("/infer", {"prompt": prompt,
                                  "n_predict": n_predict or self.n_predict})
        return InferenceResult(text=r.get("text", ""),
                               latency_ms=r["latency_ms"],
                               remote_start=r.get("t_start"),
                               remote_end=r.get("t_end"))


def make_backend(args) -> object:
    if args.backend == "sleep":
        return SleepBackend(args.latency_ms, args.latency_jitter_ms, args.seed)
    if args.backend == "http":
        if not args.endpoint:
            raise SystemExit("--backend http 需要 --endpoint")
        return HttpBackend(args.endpoint, args.n_predict)
    if args.backend == "agent":
        if not args.agent_url:
            raise SystemExit("--backend agent 需要 --agent-url")
        b = AgentBackend(args.agent_url, args.n_predict)
        if args.no_calibrate:
            # 故意不校准：offset 保持 0，用来展示「时间戳不同源就直接相减」的后果
            b.calibrated = {"offset_ms": 0.0, "rtt_ms": None,
                            "precision_bound_ms": None, "skipped": True}
        else:
            b.calibrated = b.calibrate(args.calibrate_rounds)
        return b
    raise SystemExit(f"未知 backend {args.backend}")


# ------------------------------------------------------------------ agent 端
@dataclass
class Input:
    sid: int
    t_sampled: float                       # 本地时钟域
    t_sampled_remote: float | None = None  # agent 时钟域（若经 agent 采样）
    t_remote_start: float | None = None
    t_remote_end: float | None = None
    payload: str = ""
    t_enqueue: float = 0.0
    t_infer_start: float = 0.0
    t_infer_end: float = 0.0
    t_use: float | None = None
    t_drop: float | None = None
    drop_reason: str | None = None
    age_ms_at_use: float | None = None


class Agent:
    """端侧 agent：用自己的时钟给推理打点，并支持注入时钟偏移。

    `clock_offset_ms` 模拟「设备与主控的单调时钟不同源」。校准的目标是把
    这个偏移估计出来，否则把 agent 的时间戳直接和本地时间戳相减，
    得到的输入年龄会整体偏掉。
    """

    def __init__(self, backend, port: int, clock_offset_ms: float = 0.0):
        self.backend = backend
        self.offset = clock_offset_ms
        self.port = port
        self.server = None

    def remote_now(self) -> float:
        return now_ms() + self.offset

    def serve(self):
        agent = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, obj: dict):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/ping":
                    t2 = agent.remote_now()
                    self._send({"t1": req.get("t1"), "t2": t2,
                                "t3": agent.remote_now()})
                elif self.path == "/infer":
                    t_start = agent.remote_now()
                    r = agent.backend.infer(req.get("prompt", ""),
                                            req.get("n_predict"))
                    t_end = agent.remote_now()
                    self._send({"text": r.text,
                                "latency_ms": round(t_end - t_start, 3),
                                "t_start": t_start, "t_end": t_end})
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):  # 静音
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        print(f"agent 监听 :{self.port}  注入时钟偏移 {self.offset:+.1f} ms", flush=True)
        self.server.serve_forever()


# ------------------------------------------------------------------ 策略
def pick(queue: list[Input], policy: str, now: float, max_age_ms: float | None):
    """返回 (选中的输入, 被丢弃的输入列表)。"""
    if not queue:
        return None, []
    dropped = []
    if policy == "fifo":
        chosen = queue[0]
        dropped = []
    elif policy == "latest":
        chosen = queue[-1]
        dropped = queue[:-1]
    elif policy == "drop-stale":
        fresh = [x for x in queue if max_age_ms is None
                 or (now - x.t_sampled) <= max_age_ms]
        dropped = [x for x in queue if x not in fresh]
        if not fresh:
            return None, dropped
        chosen = fresh[-1]
        dropped += fresh[:-1]
    else:
        raise SystemExit(f"未知策略 {policy}")
    for d in dropped:
        d.t_drop = now
        d.drop_reason = policy
    return chosen, dropped


# ------------------------------------------------------------------ 重放与闭环
@dataclass
class Metrics:
    periods: int = 0
    applied: int = 0
    deadline_miss: int = 0
    no_input: int = 0
    stale_used: int = 0
    ages: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    drops: dict = field(default_factory=dict)


def run_loop(args, backend, log: EventLog) -> dict:
    period = 1000.0 / args.freq
    max_age = args.max_age_ms
    rng = random.Random(args.seed)
    metrics = Metrics()
    queue: list[Input] = []
    lock = threading.Lock()
    stop = threading.Event()
    inflight: dict = {}
    samples_seen = 0

    # --- 输入线程：按固定节奏产生样本，可注入抖动 ---
    def producer():
        nonlocal samples_seen
        i = 0
        t_start = now_ms()
        while not stop.is_set():
            ideal = t_start + i * period
            jitter = (rng.uniform(-args.jitter_ms, args.jitter_ms)
                      if args.jitter_ms else 0.0)
            t_sample = ideal + jitter
            delay = t_sample - now_ms()
            if delay > 0:
                time.sleep(delay / 1000.0)
            inp = Input(sid=i, t_sampled=now_ms(), payload=f"{args.prompt_prefix}{i}")
            inp.t_enqueue = now_ms()
            with lock:
                queue.append(inp)
                depth = len(queue)
                samples_seen += 1
                if args.queue and depth > args.queue:
                    # 背压：队列满时丢最旧的输入
                    old = queue.pop(0)
                    old.t_drop = now_ms()
                    old.drop_reason = "queue_full"
                    metrics.drops["queue_full"] = metrics.drops.get("queue_full", 0) + 1
                    log.write({"event": "drop", "sid": old.sid, "reason": "queue_full",
                               "t": old.t_drop})
            log.write({"event": "enqueue", "sid": inp.sid, "queue": depth,
                       "jitter_ms": round(jitter, 2)})
            i += 1

    # --- 推理线程：把选中的输入送进 backend ---
    def worker(chosen: Input):
        chosen.t_infer_start = now_ms()
        log.write({"event": "infer_start", "sid": chosen.sid,
                   "queue_age_ms": round(chosen.t_infer_start - chosen.t_sampled, 2)})
        try:
            r = backend.infer(chosen.payload)
            chosen.t_infer_end = now_ms()
            metrics.latencies.append(r.latency_ms)
            log.write({"event": "infer_end", "sid": chosen.sid,
                       "latency_ms": round(r.latency_ms, 2)})
            if isinstance(backend, AgentBackend) and r.remote_start is not None:
                # agent 的时间戳在自己的时钟域里，换算到本地域后才能和
                # 采样时刻相减；不换算就会整段偏移（校准要解决的问题）。
                ls, le = backend.to_local(r.remote_start), backend.to_local(r.remote_end)
                log.write({"event": "agent_span_local", "sid": chosen.sid,
                           "offset_ms": round(backend.offset_ms, 3),
                           "remote_start_local": round(ls, 2),
                           "remote_end_local": round(le, 2),
                           "local_start": round(chosen.t_infer_start, 2),
                           "local_end": round(chosen.t_infer_end, 2),
                           "start_skew_ms": round(ls - chosen.t_infer_start, 2),
                           "end_skew_ms": round(le - chosen.t_infer_end, 2)})
        except Exception as e:  # noqa: BLE001
            chosen.t_infer_end = now_ms()
            log.write({"event": "infer_error", "sid": chosen.sid, "error": repr(e)})
        finally:
            inflight.pop(chosen.sid, None)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    t_end = now_ms() + args.seconds * 1000.0
    chunk_left = 0
    chunk_input: Input | None = None

    while now_ms() < t_end:
        period_start = now_ms()
        deadline = period_start + args.deadline_ms
        metrics.periods += 1

        # 控制周期内决定：是否有可用的动作
        with lock:
            chosen, dropped = (None, [])
            if chunk_left <= 0:
                chosen, dropped = pick(queue, args.policy, period_start, max_age)
                if chosen is not None:
                    queue.remove(chosen)
            depth = len(queue)

        for d in dropped:
            metrics.drops[d.drop_reason] = metrics.drops.get(d.drop_reason, 0) + 1
            log.write({"event": "drop", "sid": d.sid, "reason": d.drop_reason,
                       "t": d.t_drop})

        if chosen is not None:
            inflight[chosen.sid] = chosen
            th = threading.Thread(target=worker, args=(chosen,), daemon=True)
            th.start()
            # 等到 deadline：能等到就取用，等不到就记一次 deadline miss
            th.join(timeout=max(0.0, (deadline - now_ms()) / 1000.0))
            if th.is_alive():
                metrics.deadline_miss += 1
                log.write({"event": "deadline_miss", "sid": chosen.sid,
                           "t": deadline})
                # 动作还没算完：这一周期不取用；chunk 计数归零，
                # 下一周期重新按策略取输入。注意**不能 continue**，
                # 否则会跳过周期对齐、把控制节奏打乱（死循环式空转）。
                chosen = None
                chunk_left = 0
            else:
                chunk_input = chosen
                chunk_left = args.chunk

        if chunk_left > 0 and chunk_input is not None:
            age = now_ms() - chunk_input.t_sampled
            metrics.ages.append(age)
            chunk_input.t_use = now_ms()
            chunk_input.age_ms_at_use = age
            if args.max_age_ms and age > args.max_age_ms:
                metrics.stale_used += 1
                log.write({"event": "action_stale", "sid": chunk_input.sid,
                           "age_ms": round(age, 2),
                           "limit_ms": args.max_age_ms})
            metrics.applied += 1
            log.write({"event": "action_use", "sid": chunk_input.sid,
                       "age_ms": round(age, 2), "chunk_left": chunk_left})
            chunk_left -= 1
        else:
            metrics.no_input += 1
            log.write({"event": "action_none", "t": period_start})

        # 与下一个周期对齐
        next_t = period_start + period
        sleep_ms = next_t - now_ms()
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    stop.set()
    prod.join(timeout=2)
    ages = metrics.ages
    lat = metrics.latencies

    def pct(xs, q):
        if not xs:
            return None
        ys = sorted(xs)
        return round(ys[min(len(ys) - 1, int(len(ys) * q))], 2)

    return {
        "config": {k: getattr(args, k) for k in
                   ("freq", "queue", "policy", "jitter_ms", "chunk",
                    "max_age_ms", "deadline_ms", "seconds", "backend")},
        "periods": metrics.periods,
        "applied": metrics.applied,
        "no_input": metrics.no_input,
        "deadline_miss": metrics.deadline_miss,
        "effective_action_rate": round(metrics.applied / max(metrics.periods, 1), 3),
        "deadline_miss_rate": round(metrics.deadline_miss / max(metrics.periods, 1), 3),
        "stale_used": metrics.stale_used,
        "input_age_ms": {"p50": pct(ages, 0.5), "p95": pct(ages, 0.95),
                         "p99": pct(ages, 0.99),
                         "max": round(max(ages), 2) if ages else None},
        "infer_latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
                             "max": round(max(lat), 2) if lat else None,
                             "mean": round(statistics.mean(lat), 2) if lat else None},
        "drops": metrics.drops,
        "samples": samples_seen,
    }


def matrix(args, backend) -> list[dict]:
    """三条轴：抖动、频率×队列×策略、action chunk。

    固定其它量、每次只动一条轴，避免把三者的交互混成一个数字。
    """
    runs: list[dict] = []

    def one(tag: str, **over) -> dict:
        ns = argparse.Namespace(**{**args.__dict__, **over})
        log = EventLog(Path(args.events.replace(".jsonl", f"_{tag}.jsonl"))
                       if args.events else None)
        try:
            r = run_loop(ns, backend, log)
        finally:
            log.close()
        r["tag"] = tag
        runs.append(r)
        ia = r["input_age_ms"]
        print(f"    {tag:<34} 有效 {r['effective_action_rate']:.3f}  "
              f"miss {r['deadline_miss_rate']:.3f}  "
              f"年龄 p50/p95 {ia['p50']}/{ia['p95']} ms  drops {r['drops']}")
        return r

    print("[轴 1] 抖动注入（freq=10, queue=4, latest）")
    for j in (0, 5, 20, 50):
        one(f"jitter{j:>02}", jitter_ms=j, freq=10, queue=4, policy="latest")

    print("[轴 2] 频率 × 队列容量 × 策略（jitter=20 ms）")
    for f in (5, 10, 20):
        for q in (1, 4, 16):
            for p in ("fifo", "latest", "drop-stale"):
                one(f"f{f:>02}_q{q:>02}_{p}", freq=f, queue=q, policy=p, jitter_ms=20)

    print("[轴 3] action chunk（freq=10, queue=4, latest, jitter=20 ms）")
    for c in (1, 4, 8):
        one(f"chunk{c}", chunk=c, freq=10, queue=4, policy="latest", jitter_ms=20)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["run", "agent", "calibrate", "matrix"])
    ap.add_argument("--backend", default="sleep", choices=["sleep", "http", "agent"])
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--agent-url", default=None)
    ap.add_argument("--port", type=int, default=19080)
    ap.add_argument("--clock-offset-ms", type=float, default=0.0)
    ap.add_argument("--calibrate-rounds", type=int, default=9)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="不校准时钟（用于展示未校准的偏差）")
    ap.add_argument("--latency-ms", type=float, default=40.0)
    ap.add_argument("--latency-jitter-ms", type=float, default=0.0)
    ap.add_argument("--freq", type=float, default=10.0)
    ap.add_argument("--queue", type=int, default=4)
    ap.add_argument("--policy", default="latest",
                    choices=["fifo", "latest", "drop-stale"])
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--max-age-ms", type=float, default=None)
    ap.add_argument("--deadline-ms", type=float, default=80.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--prompt-prefix", default="控制器输入 #")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", default=None, help="事件 JSONL 输出路径")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {"measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "mode": args.mode, "args": args.__dict__.copy()}

    if args.mode == "agent":
        if args.backend == "agent":
            raise SystemExit("agent 进程本身不能再用 agent backend")
        Agent(make_backend(args), args.port, args.clock_offset_ms).serve()
        return

    if args.mode == "calibrate":
        b = AgentBackend(args.agent_url, args.n_predict)
        res["calibration"] = b.calibrate(args.calibrate_rounds)
        print(f"offset={res['calibration']['offset_ms']} ms  "
              f"RTT={res['calibration']['rtt_ms']} ms  "
              f"（注入 {args.clock_offset_ms:+.1f} ms，"
              f"精度上界 ±{res['calibration']['precision_bound_ms']} ms）")
    elif args.mode == "matrix":
        backend = make_backend(args)
        if isinstance(backend, AgentBackend):
            res["calibration"] = backend.calibrated
            print(f"时钟校准 offset={backend.calibrated['offset_ms']} ms "
                  f"（±{backend.calibrated['precision_bound_ms']} ms）")
        res["runs"] = matrix(args, backend)
    else:
        backend = make_backend(args)
        if isinstance(backend, AgentBackend):
            res["calibration"] = backend.calibrated
            print(f"时钟校准 offset={backend.calibrated['offset_ms']} ms "
                  f"（±{backend.calibrated['precision_bound_ms']} ms）")
        log = EventLog(Path(args.events) if args.events else None)
        try:
            res.update(run_loop(args, backend, log))
        finally:
            log.close()
        r = res
        print(f"周期 {r['periods']}  有效动作率 {r['effective_action_rate']:.3f}  "
              f"deadline miss {r['deadline_miss']} ({r['deadline_miss_rate']:.3f})  "
              f"输入年龄 p50/p95/p99 {r['input_age_ms']['p50']}/"
              f"{r['input_age_ms']['p95']}/{r['input_age_ms']['p99']} ms  "
              f"丢弃 {r['drops']}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
