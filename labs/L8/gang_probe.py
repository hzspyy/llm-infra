#!/usr/bin/env python3
"""labs/L8/gang_probe.py - 8.2-D: TP=2 的"整组分配"与"部分分配"探针 (真实进程).

测的是三件事, 全部是本机可复现的进程级事实:

  1. 两张卡都空闲时, TP=2 worker 从进程启动到**可服务**要多久。用来证明
     "拿到了两块 GPU"与"模型就绪"之间隔着一大段加载与编译时间。
  2. 只把一张卡放进可见集合、却要求 TP=2 时, 引擎报什么错、多久报错。
     这是"部分分配"的最直接后果: 请求方以为拿到 2 张卡, 实际集合里只有 1 张。
  3. 两张卡都可见, 但其中一张已经被别的进程占用了一部分显存时, 引擎是
     启动失败还是降级运行。

这里不做调度: 没有集群控制面, gang scheduling 的语义用固定版本材料映射
(见 results/*/8.2_refs/coscheduling_README.md), 只把"分配"与"就绪"的差别测实。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def gpu_state() -> List[Dict[str, Any]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        text=True, timeout=15)
    rows = []
    for line in out.splitlines():
        idx, used, total = [p.strip() for p in line.split(",")]
        rows.append({"index": int(idx),
                     "used_mib": int(used.split()[0]),
                     "total_mib": int(total.split()[0])})
    return rows


def wait_ready(port: int, proc: subprocess.Popen, deadline_s: float) -> Dict[str, Any]:
    """轮询 /health 直到可服务; 同时监控进程是否提前退出。"""
    import urllib.request
    t0 = time.monotonic()
    rc: Optional[int] = None
    while time.monotonic() - t0 < deadline_s:
        rc = proc.poll()
        if rc is not None:
            return {"ready": False, "exit_code": rc, "elapsed_s": time.monotonic() - t0}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return {"ready": True, "elapsed_s": time.monotonic() - t0}
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    return {"ready": False, "exit_code": None, "elapsed_s": time.monotonic() - t0,
            "note": "timeout"}


def run_case(name: str, visible: str, tp: int, out: Path, vllm: str,
             model: str, port: int, deadline_s: float,
             gpu_mem_util: float = 0.40) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{name}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible
    cmd = [vllm, "serve", model, "--port", str(port),
           "--tensor-parallel-size", str(tp),
           "--gpu-memory-utilization", str(gpu_mem_util),
           "--max-model-len", "4096"]
    before = gpu_state()
    t_start = time.monotonic()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        res = wait_ready(port, proc, deadline_s)
        started_s = time.monotonic() - t_start
        # 出错时把日志里第一行真正的错误抓出来, 便于正文引用原文
        tail = ""
        if not res.get("ready"):
            try:
                text = log.read_text(encoding="utf-8", errors="replace").splitlines()
                hits = [ln for ln in text if any(k in ln for k in
                        ("Error", "error", "Traceback", "RuntimeError", "ValueError",
                         "not enough", "CUDA_VISIBLE", "world_size"))]
                tail = "\n".join(hits[:12])
            except Exception:  # noqa: BLE001
                pass
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
    after = gpu_state()
    rec = {
        "case": name,
        "cuda_visible_devices": visible,
        "tensor_parallel_size": tp,
        "deadline_s": deadline_s,
        "process_started_to_result_s": started_s,
        "ready": res.get("ready"),
        "ready_elapsed_s": res.get("elapsed_s"),
        "exit_code": res.get("exit_code"),
        "gpu_before": before,
        "gpu_after": after,
        "error_excerpt": tail,
        "log": str(log),
    }
    print(json.dumps({k: rec[k] for k in ("case", "cuda_visible_devices",
                                          "tensor_parallel_size", "ready",
                                          "ready_elapsed_s", "exit_code")},
                     ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--vllm", default="vllm")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--port", type=int, default=19200)
    ap.add_argument("--deadline", type=float, default=420.0)
    ap.add_argument("--cases", default="both_visible,one_visible",
                    help="both_visible: 两卡可见跑 TP=2; one_visible: 只给一张卡却要 TP=2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    results: List[Dict[str, Any]] = []

    for case in args.cases.split(","):
        if case == "both_visible":
            results.append(run_case("tp2_both_visible", args.gpus, 2, out, args.vllm,
                                    args.model, args.port, args.deadline))
        elif case == "one_visible":
            results.append(run_case("tp2_one_visible", gpus[0], 2, out, args.vllm,
                                    args.model, args.port + 1, min(args.deadline, 180.0)))
        elif case == "tp1":
            results.append(run_case("tp1_single_gpu", gpus[0], 1, out, args.vllm,
                                    args.model, args.port + 2, args.deadline))
        else:
            raise SystemExit(f"unknown case {case}")

    (out / "gang_probe.json").write_text(
        json.dumps({"gpu_after_all": gpu_state(), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out/'gang_probe.json'}")


if __name__ == "__main__":
    main()
