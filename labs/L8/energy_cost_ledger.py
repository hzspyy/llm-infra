#!/usr/bin/env python3
"""labs/L8/energy_cost_ledger.py - 8.5: 真实功率采样、能量积分与成本账本.

测量口径 (读数与价格假设严格分开):

  * 用 NVML (pynvml) 以固定间隔采 `power.draw` / 显存 / 利用率, 原始序列逐条落盘;
  * idle 段与 workload 段分别积分: 梯形积分给总能量, 增量能量 = 工作段能量 − idle
    平均功率 × 工作段时长;
  * token 数取服务端 usage 的 `completion_tokens` 之和, 只统计真正完成的请求;
  * 时延与 SLO 达标率复用 8.3 的 `request_metrics` 与冻结门槛, 于是"每千有效输出
    token 成本"落在同一口径上。

只测 GPU: 采样对象是单张卡的 `power.draw`, 因此结论只覆盖该卡的能耗, **不能**外推
到整机、机房或 PUE。价格与电价不是读数, 单独写进 `price_assumptions.json`, 并做
敏感度扫描。

用法 (服务端已就绪):
    python labs/L8/energy_cost_ledger.py --base-url http://127.0.0.1:8000 \
        --out-dir <dir> --gpu 0 --idle-s 60 --window 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.load_generator import (  # noqa: E402
    generate_arrivals,
    run_closed_loop,
    run_open_loop,
)
from labs.L8.request_metrics import TEACHING_SLOS, evaluate, write_jsonl  # noqa: E402


class PowerSampler:
    """以固定间隔采 NVML 读数; 采样分辨率与传感器自身的平均窗口分开记录。"""

    def __init__(self, gpu_index: int, interval_s: float, out_path: Path):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.out_path = out_path
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop.is_set():
            try:
                t = time.monotonic()
                pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0        # mW -> W
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                try:
                    clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                except Exception:  # noqa: BLE001
                    clk = None
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:  # noqa: BLE001
                    temp = None
                self.samples.append({
                    "t_s": t, "power_w": pw, "mem_used_mib": mem.used / 2**20,
                    "util_gpu_pct": util.gpu, "sm_clock_mhz": clk, "temp_c": temp,
                })
            except Exception as e:  # noqa: BLE001
                self.samples.append({"t_s": time.monotonic(), "error": str(e)})
            self._stop.wait(self.interval_s)
        pynvml.nvmlShutdown()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self.out_path.open("w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


def integrate_energy(samples: List[Dict[str, Any]], t0: float, t1: float) -> Dict[str, Any]:
    """对 [t0, t1] 做梯形积分。返回总能量与功率统计。"""
    seg = [s for s in samples if "power_w" in s and t0 <= s["t_s"] <= t1]
    if len(seg) < 2:
        return {"samples": len(seg), "energy_j": None}
    e = 0.0
    for a, b in zip(seg, seg[1:]):
        dt = b["t_s"] - a["t_s"]
        e += 0.5 * (a["power_w"] + b["power_w"]) * dt
    powers = [s["power_w"] for s in seg]
    return {
        "samples": len(seg),
        "window_s": seg[-1]["t_s"] - seg[0]["t_s"],
        "energy_j": e,
        "power_mean_w": sum(powers) / len(powers),
        "power_min_w": min(powers),
        "power_max_w": max(powers),
        "mem_peak_mib": max(s["mem_used_mib"] for s in seg),
        "util_mean_pct": sum(s["util_gpu_pct"] for s in seg) / len(seg),
    }


async def run_case_energy(base_url: str, out: Path, name: str, sampler: PowerSampler,
                          gpu: int, model: str, kind: str, window: float,
                          concurrency: int = 1, rate: float = 4.0,
                          prompt_len: int = 2048, output_len: int = 128,
                          seed: int = 0, timeout_s: float = 120.0) -> Dict[str, Any]:
    """跑一个配置并把它窗口内的功率积分出来。"""
    out.mkdir(parents=True, exist_ok=True)
    n = int(max(1, math.floor(rate * window))) if kind == "open" else 10 ** 7
    prompt = [seed + 1000 + (i % 997) for i in range(prompt_len)]
    prompts = [[seed + 1000 + ((i * 37 + j) % 997) for j in range(prompt_len)]
               for i in range(256)]

    t_start = time.monotonic()
    # prompt_ids 以"循环池"方式使用: 驱动器按 idx % len(prompt_ids) 取, 所以这里只给
    # 一个有界的提示池, 不需要按请求数展开 (否则 10^7 个 2048 token 提示会撑爆内存)。
    if kind == "open":
        arrivals = generate_arrivals("poisson", rate, window, seed=seed)
        records = await run_open_loop(
            base_url, arrivals, prompt_len, output_len, model=model,
            timeout_s=timeout_s, prompt_ids=prompts)
        summary = evaluate(records, name, window, reserved_events=arrivals)
    else:
        records = await run_closed_loop(
            base_url, concurrency=concurrency, total_requests=n,
            prompt_len=prompt_len, output_len=output_len, model=model,
            timeout_s=timeout_s, prompt_ids=prompts,
            stop_after_s=window, t0_monotonic=t_start)
        win = time.monotonic() - t_start
        summary = evaluate(records, name, win)
    t_end = time.monotonic()

    energy = integrate_energy(sampler.samples, t_start, t_end)
    tokens = sum(r.num_tokens for r in records)
    energy["tokens"] = tokens
    energy["requests"] = len(records)
    energy["case"] = name
    energy["kind"] = kind
    energy["concurrency"] = concurrency if kind != "open" else None
    energy["rate_qps"] = rate if kind == "open" else None
    energy["wall_s"] = t_end - t_start
    energy["summary"] = summary.to_dict()
    write_jsonl(str(out / f"records_{name}.jsonl"), records)
    (out / f"energy_{name}.json").write_text(
        json.dumps(energy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] wall={energy['wall_s']:.1f}s E={energy['energy_j']:.1f}J "
          f"Pmean={energy['power_mean_w']:.1f}W tokens={tokens} "
          f"reqs={len(records)}", flush=True)
    return energy


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--interval", type=float, default=0.1, help="功率采样间隔 (秒)")
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrencies", default="1,8,32")
    ap.add_argument("--open-mults", default="0.3,0.9")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = PowerSampler(args.gpu, args.interval, out / "power_all.jsonl")
    sampler.start()
    await asyncio.sleep(2.0)

    cases: List[Dict[str, Any]] = []

    # 1) idle: 服务已就绪但完全没有请求
    t0 = time.monotonic()
    await asyncio.sleep(args.idle_s)
    t1 = time.monotonic()
    idle = integrate_energy(sampler.samples, t0, t1)
    idle.update({"case": "idle", "kind": "idle", "tokens": 0, "requests": 0,
                 "wall_s": t1 - t0})
    cases.append(idle)
    print(f"[idle] {idle['window_s']:.1f}s Pmean={idle['power_mean_w']:.1f}W "
          f"E={idle['energy_j']:.1f}J", flush=True)

    # 2) 闭环并发 1/8/32: 并发数就是引擎侧批大小的直接来源
    concurrency_energies: Dict[int, Dict[str, Any]] = {}
    for c in [int(x) for x in args.concurrencies.split(",")]:
        e = await run_case_energy(args.base_url, out, f"closed_c{c}", sampler, args.gpu,
                                  args.model, "closed", args.window, concurrency=c,
                                  prompt_len=args.prompt_len, output_len=args.output_len)
        concurrency_energies[c] = e
        cases.append(e)

    # 3) 用最大并发档的完成速率当参考容量, 再扫到达率
    ref = concurrency_energies[max(concurrency_energies)]
    r_ref = ref["requests"] / ref["wall_s"]
    for mult in [float(x) for x in args.open_mults.split(",")]:
        e = await run_case_energy(args.base_url, out, f"open_{mult}x", sampler, args.gpu,
                                  args.model, "open", args.window,
                                  rate=mult * r_ref,
                                  prompt_len=args.prompt_len, output_len=args.output_len)
        e["rate_reference_qps"] = r_ref
        cases.append(e)

    sampler.stop()

    # ---- 能量账: 增量能量 = 工作段 − idle 平均功率 × 时长 ----
    idle_p = idle["power_mean_w"] or 0.0
    ledger = []
    for e in cases:
        if e.get("energy_j") is None:
            continue
        dur = e.get("window_s") or e.get("wall_s")
        inc = e["energy_j"] - idle_p * dur
        tok = e.get("tokens") or 0
        gp = e.get("summary", {}).get("goodput", {})
        attained_tokens = None
        if gp:
            # 各 SLO 档下的"有效输出 token": 达标请求的输出 token 之和
            vals = {}
            for name, g in gp.items():
                vals[name] = g.get("attained")
            attained_tokens = vals
        ledger.append({
            "case": e["case"], "kind": e["kind"],
            "concurrency": e.get("concurrency"), "rate_qps": e.get("rate_qps"),
            "window_s": dur, "energy_j": e["energy_j"], "power_mean_w": e["power_mean_w"],
            "incremental_energy_j": inc,
            "tokens": tok, "requests": e.get("requests"),
            "j_per_token_total": (e["energy_j"] / tok) if tok else None,
            "j_per_token_incremental": (inc / tok) if tok else None,
            "tokens_per_j_incremental": (tok / inc) if inc and inc > 0 else None,
            "mem_peak_mib": e.get("mem_peak_mib"),
            "ttft_p50": e.get("summary", {}).get("latency", {}).get("true_ttft", {}).get("p50"),
            "ttft_p99": e.get("summary", {}).get("latency", {}).get("true_ttft", {}).get("p99"),
            "tpot_p50": e.get("summary", {}).get("latency", {}).get("tpot", {}).get("p50"),
            "slo_attained": attained_tokens,
        })
    result = {
        "sampling": {"interval_s": args.interval, "gpu_index": args.gpu,
                     "note": "NVML power.draw 自带平均窗口, 采样间隔不等于传感器窗口"},
        "idle": idle,
        "r_ref_qps": r_ref,
        "cases": cases,
        "ledger": ledger,
    }
    (out / "energy_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ledger, ensure_ascii=False, indent=2))
    print(f"-> {out/'energy_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
