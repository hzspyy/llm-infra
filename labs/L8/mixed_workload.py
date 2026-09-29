#!/usr/bin/env python3
"""labs/L8/mixed_workload.py - 8.4-A/B/C: 两类任务在同一张卡上的基线、混部与干预.

两类任务:

  gen  Qwen3-1.7B 文本生成 (vLLM generate, /v1/completions 流式)
  emb  Qwen3-Embedding-0.6B 向量化 (vLLM pooling, /v1/embeddings 非流式)

三个子命令:

  baseline  每个任务**单独**跑一遍到达清单, 建立 CPU/GPU/时延/SLO 基线
  mixed     两个引擎同时跑同一份到达清单, 用 --share-emb 控制向量化那一类的负载份额
  intervene 在混部基础上改一个变量 (--intervention): 向量化串行化、生成关前缀缓存、
            或时间分片 (两个类轮流独占), 用来把"相关"变成"因果"

每个请求记录到达/发送/完成/状态; 全程按固定间隔采样 nvidia-smi 的显存占用与利用率,
因此混部的资源时间线与逐请求时延可以对齐分析。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx  # noqa: E402

from labs.L8.load_generator import RequestSpec, StreamingClient  # noqa: E402
from labs.L8.request_metrics import (  # noqa: E402
    STATUS_SUCCESS,
    STATUS_TRUNCATED,
    TEACHING_SLOS,
    evaluate,
    summarize,
    write_jsonl,
)

GEN_SNAP = "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EMB_SNAP = ("/scratch/learn/models/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")


# --------------------------------------------------------------------------
# 引擎进程
# --------------------------------------------------------------------------
class Server:
    def __init__(self, python: str, name: str, model: str, port: int, log: Path,
                 runner: str = "auto", max_model_len: int = 4096,
                 gpu_mem_util: float = 0.32, extra: Optional[List[str]] = None,
                 prefix_caching: bool = True):
        self.python = python
        self.name = name
        self.model = model
        self.port = port
        self.log = log
        self.runner = runner
        self.max_model_len = max_model_len
        self.gpu_mem_util = gpu_mem_util
        self.extra = extra or []
        self.prefix_caching = prefix_caching
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: List[str] = []

    def start(self) -> float:
        cmd = [self.python, "-m", "vllm.entrypoints.openai.api_server",
               "--model", self.model, "--served-model-name", self.name,
               "--host", "127.0.0.1", "--port", str(self.port),
               "--max-model-len", str(self.max_model_len),
               "--gpu-memory-utilization", str(self.gpu_mem_util),
               "--runner", self.runner] + self.extra
        if self.prefix_caching:
            cmd.append("--enable-prefix-caching")
        self.cmd = cmd
        f = open(self.log, "ab")
        self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                     env=dict(os.environ), start_new_session=True)
        t0 = time.monotonic()
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} 退出 code={self.proc.returncode}, 见 {self.log}")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return time.monotonic() - t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.name} 未就绪")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait(timeout=30)
        self.proc = None
        time.sleep(2)


class GpuSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: List[Dict[str, Any]] = []
        self._stop_flag = threading.Event()
        self.t0 = 0.0

    def run(self) -> None:
        self.t0 = time.monotonic()
        while not self._stop_flag.is_set():
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
                mem, util = [x.strip() for x in q.split(",")]
                self.samples.append({"t": round(time.monotonic() - self.t0, 3),
                                     "mem_mib": float(mem), "util_pct": float(util)})
            except Exception:  # noqa: BLE001
                pass
            self._stop_flag.wait(self.interval_s)

    def stop(self) -> List[Dict[str, Any]]:
        self._stop_flag.set()
        self.join(timeout=5)
        return self.samples


# --------------------------------------------------------------------------
# 到达清单
# --------------------------------------------------------------------------
def poisson_arrivals(rate: float, duration_s: float, seed: int) -> List[float]:
    if rate <= 0:
        return []
    rng = random.Random(seed)
    out, t = [], 0.0
    while True:
        t += -math.log(max(1e-12, rng.random())) / rate
        if t > duration_s:
            break
        out.append(round(t, 6))
    return out


def make_prompt_ids(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 90000) for _ in range(n)]


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------
async def run_gen(port: int, model: str, arrivals: List[float], prompt_ids: List[int],
                  output_len: int, timeout_s: float) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    recs: List[Dict[str, Any]] = []
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        cli = StreamingClient(f"http://127.0.0.1:{port}", "openai", model, timeout_s, t0,
                              client)

        async def one(i: int, arr: float):
            await asyncio.sleep(max(0.0, t0 + arr - time.monotonic()))
            spec = RequestSpec(f"gen-{i}", arr, len(prompt_ids), output_len, prompt_ids)
            r = await cli.run(spec)
            return {"id": r.request_id, "class": "gen", "planned": arr,
                    "send": None if r.actual_send_s is None else r.actual_send_s - t0,
                    "finish": None if r.finish_s is None else r.finish_s - t0,
                    "ttft": r.observed_ttft_s, "tpot": r.tpot_s, "status": r.status,
                    "tokens": len(r.token_timestamps_s)}

        recs = list(await asyncio.gather(*[one(i, a) for i, a in enumerate(arrivals)]))
    return recs


async def run_emb(port: int, model: str, arrivals: List[float], prompt_ids: List[int],
                  timeout_s: float) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def one(i: int, arr: float):
            await asyncio.sleep(max(0.0, t0 + arr - time.monotonic()))
            send = time.monotonic()
            status, err = "SUCCESS", None
            try:
                r = await client.post(f"http://127.0.0.1:{port}/v1/embeddings",
                                      json={"model": model, "input": prompt_ids})
                if r.status_code >= 500 or r.status_code in (429, 503):
                    status = "REJECTED"
                    err = f"HTTP {r.status_code}"
                elif r.status_code != 200:
                    status = "ERROR"
                    err = f"HTTP {r.status_code}"
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                status = "TIMEOUT" if isinstance(e, httpx.TimeoutException) else "ERROR"
                err = type(e).__name__
            finish = time.monotonic()
            return {"id": f"emb-{i}", "class": "emb", "planned": arr,
                    "send": send - t0, "finish": finish - t0,
                    "ttft": finish - send, "tpot": None, "status": status,
                    "tokens": len(prompt_ids), "error": err}

        return list(await asyncio.gather(*[one(i, a) for i, a in enumerate(arrivals)]))


def summarize_class(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in recs if r["status"] in (STATUS_SUCCESS, STATUS_TRUNCATED)]
    lats = sorted(r["finish"] - r["planned"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    def q(xs, p):
        if not xs:
            return None
        return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1)))], 4)
    return {
        "n": len(recs), "ok": len(ok),
        "statuses": {s: sum(1 for r in recs if r["status"] == s)
                     for s in {r["status"] for r in recs}},
        "latency_p50": q(lats, 0.5), "latency_p95": q(lats, 0.95),
        "latency_p99": q(lats, 0.99),
        "ttft_p50": q(ttfts, 0.5), "ttft_p99": q(ttfts, 0.99),
        "throughput_per_s": round(len(ok) / max(1e-9, max([r["finish"] for r in ok],
                                                          default=0.0)), 3),
    }


# --------------------------------------------------------------------------
# 场景执行
# --------------------------------------------------------------------------
def run_scenario(args, out_dir: Path, tag: str, servers: List[Server],
                 gen_arr: List[float], emb_arr: List[float]) -> Dict[str, Any]:
    sampler = GpuSampler(0.5)
    sampler.start()
    t_start = time.monotonic()
    gen_recs: List[Dict[str, Any]] = []
    emb_recs: List[Dict[str, Any]] = []
    try:
        async def both():
            tasks = []
            if servers:
                gen_srv = next((s for s in servers if s.name == "gen"), None)
                emb_srv = next((s for s in servers if s.name == "emb"), None)
                if gen_srv:
                    tasks.append(run_gen(gen_srv.port, "gen", gen_arr, args.gen_prompt_ids,
                                         args.gen_output_len, args.timeout_s))
                if emb_srv:
                    tasks.append(run_emb(emb_srv.port, "emb", emb_arr, args.emb_prompt_ids,
                                         args.timeout_s))
            return await asyncio.gather(*tasks)
        results = asyncio.run(both())
        for r in results:
            if r and r[0]["class"] == "gen":
                gen_recs = r
            elif r:
                emb_recs = r
    finally:
        samples = sampler.stop()
    res = {
        "tag": tag, "gen_rate": args.gen_rate, "emb_rate": args.emb_rate,
        "share_emb": args.share_emb, "duration_s": args.duration_s,
        "wall_s": round(time.monotonic() - t_start, 2),
        "gen": summarize_class(gen_recs) if gen_recs else None,
        "emb": summarize_class(emb_recs) if emb_recs else None,
        "gpu_peak_mib": max((s["mem_mib"] for s in samples), default=None),
        "gpu_mean_util_pct": round(sum(s["util_pct"] for s in samples) / max(1, len(samples)), 2),
        "gpu_samples": len(samples),
        "servers": [{"name": s.name, "port": s.port, "cmd": s.cmd} for s in servers],
    }
    (out_dir / f"{tag}_records.json").write_text(json.dumps(
        {"gen": gen_recs, "emb": emb_recs}, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{tag}_gpu.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in samples), encoding="utf-8")
    return res


def build_servers(args, out_dir: Path, want_gen: bool, want_emb: bool,
                  intervention: str) -> List[Server]:
    servers: List[Server] = []
    if want_gen:
        extra: List[str] = []
        pc = True
        if intervention == "gen_noprefix":
            pc = False
        servers.append(Server(args.python, "gen", args.gen_snap, args.gen_port,
                              out_dir / "gen.log", gpu_mem_util=args.gen_gpu_mem_util,
                              max_model_len=args.gen_max_model_len, extra=extra,
                              prefix_caching=pc))
    if want_emb:
        extra = []
        if intervention == "emb_seq":
            extra += ["--max-num-seqs", "1", "--max-num-batched-tokens", "512"]
        servers.append(Server(args.python, "emb", args.emb_snap, args.emb_port,
                              out_dir / "emb.log", runner="pooling",
                              gpu_mem_util=args.emb_gpu_mem_util,
                              max_model_len=args.emb_max_model_len, extra=extra))
    return servers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "mixed", "intervene"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen-snap", default=GEN_SNAP)
    ap.add_argument("--emb-snap", default=EMB_SNAP)
    ap.add_argument("--gen-port", type=int, default=18091)
    ap.add_argument("--emb-port", type=int, default=18092)
    ap.add_argument("--gen-gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--emb-gpu-mem-util", type=float, default=0.12)
    ap.add_argument("--gen-max-model-len", type=int, default=4096)
    ap.add_argument("--emb-max-model-len", type=int, default=2048)
    ap.add_argument("--gen-rate", type=float, default=4.0)
    ap.add_argument("--emb-rate", type=float, default=8.0)
    ap.add_argument("--share-emb", type=float, default=0.5,
                    help="混部时向量化那一类占总负载的份额 (0/0.25/0.5/0.75/1)")
    ap.add_argument("--shares", default="0,0.25,0.5,0.75,1.0",
                    help="mixed 模式下依次重放的份额列表; 引擎只启动一次")
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--gen-prompt-len", type=int, default=512)
    ap.add_argument("--emb-prompt-len", type=int, default=512)
    ap.add_argument("--gen-output-len", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--intervention", default="none",
                    choices=["none", "emb_seq", "gen_noprefix", "time_slice"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.gen_prompt_ids = make_prompt_ids(args.gen_prompt_len, args.seed + 1)
    args.emb_prompt_ids = make_prompt_ids(args.emb_prompt_len, args.seed + 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 总负载固定, 份额只改变两类各自的比例
    args.total_rate = args.gen_rate + args.emb_rate
    emb_rate = args.total_rate * args.share_emb
    gen_rate = args.total_rate * (1 - args.share_emb)
    args.gen_rate, args.emb_rate = gen_rate, emb_rate

    summary: Dict[str, Any] = {"mode": args.mode, "share_emb": args.share_emb,
                               "cases": []}
    if args.mode == "baseline":
        for name, want_gen, want_emb in (("gen_only", True, False), ("emb_only", False, True)):
            servers = build_servers(args, out_dir, want_gen, want_emb, "none")
            for s in servers:
                s.start()
            try:
                r = run_scenario(args, out_dir, name,
                                 servers,
                                 poisson_arrivals(args.gen_rate if want_gen else 0.0,
                                                  args.duration_s, args.seed),
                                 poisson_arrivals(args.emb_rate if want_emb else 0.0,
                                                  args.duration_s, args.seed + 9))
            finally:
                for s in servers:
                    s.stop()
            summary["cases"].append(r)
            print(name, json.dumps({k: r[k] for k in ("gen", "emb", "gpu_peak_mib",
                                                      "gpu_mean_util_pct")},
                                   ensure_ascii=False), flush=True)
    elif args.mode == "mixed":
        servers = build_servers(args, out_dir, True, True, "none")
        for s in servers:
            s.start()
        try:
            shares = [float(x) for x in args.shares.split(",")]
            for share in shares:
                total = args.gen_rate + args.emb_rate   # build_servers 之后已被改写, 这里用原始总率
                # args.gen_rate/emb_rate 在 main 开头被按首个 share 改写过, 因此用 --total-rate 还原
                total = args.total_rate
                gen_rate = total * (1 - share)
                emb_rate = total * share
                gen_arr = poisson_arrivals(gen_rate, args.duration_s, args.seed)
                emb_arr = poisson_arrivals(emb_rate, args.duration_s, args.seed + 9)
                if args.intervention == "time_slice":
                    half = args.duration_s / 2
                    gen_arr = [t for t in gen_arr if t < half]
                    emb_arr = [t - half for t in emb_arr if t >= half]
                case_args = argparse.Namespace(**vars(args))
                case_args.share_emb = share
                case_args.gen_rate, case_args.emb_rate = gen_rate, emb_rate
                r = run_scenario(case_args, out_dir,
                                 f"mixed_{args.intervention}_s{share}", servers,
                                 gen_arr, emb_arr)
                summary["cases"].append(r)
                print(f"share={share}", json.dumps(
                    {k: r[k] for k in ("gen", "emb", "gpu_peak_mib", "gpu_mean_util_pct")},
                    ensure_ascii=False)[:600], flush=True)
        finally:
            for s in servers:
                s.stop()
    else:
        for iv in ("emb_seq", "gen_noprefix", "time_slice"):
            servers = build_servers(args, out_dir, True, True, iv)
            for s in servers:
                s.start()
            try:
                gen_arr = poisson_arrivals(args.gen_rate, args.duration_s, args.seed)
                emb_arr = poisson_arrivals(args.emb_rate, args.duration_s, args.seed + 9)
                if iv == "time_slice":
                    half = args.duration_s / 2
                    gen_arr = [t for t in gen_arr if t < half]
                    emb_arr = [t - half for t in emb_arr if t >= half]
                r = run_scenario(args, out_dir, f"iv_{iv}", servers, gen_arr, emb_arr)
            finally:
                for s in servers:
                    s.stop()
            summary["cases"].append(r)
            print(iv, json.dumps({k: r[k] for k in ("gen", "emb", "gpu_peak_mib")},
                                 ensure_ascii=False), flush=True)
    (out_dir / f"summary_{args.mode}_{args.intervention}_{args.share_emb}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
