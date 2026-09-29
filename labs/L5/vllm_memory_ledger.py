#!/usr/bin/env python3
"""L5.12 补测 · vLLM 启动账本：workspace / 图池 / KV 各占多少。

5.12 讨论非生成服务的容量时只用了"分阶段峰值"（encoder vs decoder）。
这里补的是**同一个 vLLM 进程内**的四段账——它们都由引擎自己在启动日志里报出来：

  1. 权重：`Model loading took X GiB memory`
  2. 非 torch 占用（含 workspace/通信缓冲）：`Actual usage is X GiB for consumed memory`
  3. CUDA 图池：`CUDA graph pool memory: X GiB (actual), Y GiB (estimated)`
  4. KV：`Available KV cache memory: X GiB`（与 `GPU KV cache size: N tokens` 互相核对）

四段之和与"请求预算"（`gpu_memory_utilization × 显存总量`）的差就是引擎没有单列的部分
（激活峰值、采样器工作区、碎片与未归还缓存）。本脚本在 2–3 个 utilization 上各起一次服务，
把日志解析成表，并在真实流量期间用 `nvidia-smi` 独立采一条"整卡常驻"的上下界。

用法（crater）：
    python vllm_memory_ledger.py --out <dir> --utilizations 0.30,0.45
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time

MODEL_SNAP_GLOB = "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"

PATTERNS = {
    "total_gib": re.compile(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)"),
    "desired": re.compile(r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)"),
    "consumed": re.compile(r"Actual usage is ([\d.]+) GiB for consumed memory"),
    "weights": re.compile(r"Model loading took ([\d.]+) GiB memory"),
    "graph_actual": re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
    "graph_est": re.compile(r"\(actual\), ([\d.]+) GiB \(estimated\)"),
    "kv_avail": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+) tokens"),
    "kv_usage": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request"),
}


def nvidia_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    total, used = (int(x) for x in out.split(","))
    return total, used


class Sampler:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    _, used = nvidia_free()
                    self.samples.append(used)
                except Exception:                               # noqa: BLE001
                    pass
                time.sleep(self.interval)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return dict(n=len(self.samples),
                    min_mib=min(self.samples) if self.samples else None,
                    max_mib=max(self.samples) if self.samples else None)


def parse_log(path):
    text = open(path, errors="replace").read()
    got = {}
    for name, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            got[name] = float(m.group(1).replace(",", "")) if "tokens" not in name \
                else int(m.group(1).replace(",", ""))
    if "total_gib" in got:
        got["total_gib"] = float(PATTERNS["total_gib"].search(text).group(2))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--utilizations", default="0.30,0.45")
    ap.add_argument("--port", type=int, default=8170)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import glob
    snap = sorted(glob.glob(MODEL_SNAP_GLOB))[-1]
    rows = []
    for util in (float(x) for x in args.utilizations.split(",")):
        tag = f"util{util}"
        log = os.path.join(args.out, f"server_{tag}.log")
        free = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        srv = subprocess.Popen(
            ["/scratch/learn/envs/serve/bin/python", "-m",
             "vllm.entrypoints.openai.api_server",
             "--model", snap, "--served-model-name", "m",
             "--port", str(args.port), "--host", "127.0.0.1",
             "--max-model-len", "4096", "--gpu-memory-utilization", str(util),
             "--dtype", "bfloat16", "--no-enable-prefix-caching"],
            stdout=open(log, "w"), stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"})
        ready = False
        for _ in range(300):
            try:
                subprocess.run(["curl", "-sf",
                                f"http://127.0.0.1:{args.port}/v1/models"],
                               check=True, capture_output=True)
                ready = True
                break
            except Exception:                                   # noqa: BLE001
                time.sleep(1)
        sampler = Sampler()
        sampler.start()
        if ready:
            # 一点真实流量，看常驻上界
            for _ in range(2):
                subprocess.run(["curl", "-s", "-o", "/dev/null", "-X", "POST",
                                f"http://127.0.0.1:{args.port}/v1/completions",
                                "-H", "Content-Type: application/json",
                                "-d", json.dumps({"model": "m", "prompt": "hello " * 200,
                                                  "max_tokens": 8, "temperature": 0})],
                               capture_output=True)
        time.sleep(3)
        env_stats = sampler.stop()
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        time.sleep(5)
        stats = parse_log(log)
        total_mib, used_now = nvidia_free()
        budget_mib = stats.get("desired", 0) * 1024 if stats.get("desired") else None
        ledger = dict(utilization=util, ready=ready, free_before_mib=int(free),
                      budget_gib=stats.get("desired"),
                      weights_gib=stats.get("weights"),
                      consumed_gib=stats.get("consumed"),
                      graph_actual_gib=stats.get("graph_actual"),
                      graph_estimated_gib=stats.get("graph_est"),
                      kv_available_gib=stats.get("kv_avail"),
                      kv_tokens=stats.get("kv_tokens"),
                      kv_max_concurrency_tokens=stats.get("kv_usage"),
                      resident_during=env_stats, total_gib=stats.get("total_gib"))
        named = sum(stats.get(k, 0) or 0 for k in
                    ("weights", "graph_actual", "kv_avail"))
        ledger["named_sum_gib"] = round(named, 3)
        if budget_mib:
            ledger["unnamed_gib"] = round(stats["desired"] - named, 3)
        rows.append(ledger)
        print(f"  util={util}: 预算 {stats.get('desired')} GiB = 权重 "
              f"{stats.get('weights')} + 图池 {stats.get('graph_actual')}（估计 "
              f"{stats.get('graph_est')}）+ KV {stats.get('kv_avail')}，"
              f"未单列 {ledger.get('unnamed_gib')} GiB；"
              f"KV {stats.get('kv_tokens')} token；"
              f"服务期常驻 {env_stats.get('min_mib')}–{env_stats.get('max_mib')} MiB",
              flush=True)
        (open(os.path.join(args.out, "ledger.json"), "w")
         .write(json.dumps(rows, ensure_ascii=False, indent=2)))

    print(f"\n写入 {args.out}/ledger.json")


if __name__ == "__main__":
    main()
