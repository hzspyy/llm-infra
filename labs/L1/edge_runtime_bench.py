#!/usr/bin/env python3
"""L1.6 lab · 端侧推理栈的完整测量：后端确认、延迟扫描、困惑度与持续运行。

对象是 llama.cpp 的 `llama-server`（GGUF + CUDA backend），测量五种东西：

  info     实际选中的后端与显存/内存占用（从服务端日志里抓，不靠猜）
  scan     prompt=128/2048/8192 × 并发=1/2/4，输出固定 128 token
  ppl      quantize 前后的困惑度（同一份固定语料、同一 chunk 数）
  sustain  同一配置持续运行，按秒采 tegrastats（温度/功耗/频率/内存），
           同时收逐请求延迟，报告冷机段与热稳态、能量/有效输出、deadline miss

只用标准库：HTTP 用 urllib，并发用线程，指标来自 llama-server 自己的
`timings` 字段 + 客户端墙钟，避免再加一层依赖。

用法：
    python edge_runtime_bench.py --mode info --labels f16=... --out info.json
    python edge_runtime_bench.py --mode scan --labels f16=...,q4km=... --out scan.json
    python edge_runtime_bench.py --mode ppl  --labels f16=...,q4km=... --corpus wiki.test.raw
    python edge_runtime_bench.py --mode sustain --minutes 30 --labels q4km=... \
        --prompt-tokens 2048 --deadline-ms 3000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# 一段固定文本：按 token 近似切成指定长度（中英混合，避免纯英文下的分词偏差）
PASSAGE = (
    "在推理系统的工程实践里，延迟与吞吐从来不是同一个问题。"
    "KV cache 的容量决定了并发度，而并发度又反过来改变每一步的算术强度。"
    "The roofline model tells us which side we are on: memory bound or compute bound. "
    "端侧设备把这件事推到极端：内存与显存共享同一块物理内存，"
    "带宽只有数据中心卡的几分之一，而散热与功耗预算先于算力成为约束。"
    "因此同一份模型在 Orin 上的最优 batch、上下文长度与量化位宽，"
    "往往和数据中心上的结论相反，必须实测而不是外推。 "
)


def sh(cmd: list[str], timeout: int = 60) -> dict:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(cmd), "rc": r.returncode,
            "out": r.stdout[-4000:], "err": r.stderr[-4000:]}


BENCH_ROW = re.compile(
    r"\|\s*([\w.\- ]+?)\s*\|\s*([\d.]+ \w+)\s*\|\s*([\d.]+ \w+)\s*\|\s*(\w+)\s*\|"
    r"\s*(\d+)\s*\|\s*(\w+)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|")


def bench_probe(binary: str, model: str, ngl: int, threads: int) -> dict:
    """跑一次 llama-bench：它是唯一在标准输出里直接给出 `backend` 列的工具，
    并附带 `ggml_cuda_init` 的设备与统一内存总量。"""
    if not Path(binary).exists():
        return {"error": f"{binary} 不存在（跳过）"}
    r = sh([binary, "-m", model, "-ngl", str(ngl), "-p", "128", "-n", "32",
            "-r", "2", "-t", str(threads)], timeout=3600)
    text = r["out"] + r["err"]
    devs = re.findall(r"Device (\d+): ([^,]+), compute capability ([\d.]+), "
                      r"VMM: (\w+), VRAM: (\d+) MiB", text)
    rows = [{"name": m[0], "size": m[1], "params": m[2], "backend": m[3],
             "ngl": int(m[4]), "test": m[5], "t_s": float(m[6]), "sd": float(m[7])}
            for m in BENCH_ROW.findall(text)]
    return {"cmd": r["cmd"], "rc": r["rc"],
            "devices": [{"index": int(d[0]), "name": d[1], "cc": d[2],
                         "vmm": d[3], "vram_mib": int(d[4])} for d in devs],
            "rows": rows, "git": (re.search(r"build: (\w+)", text).group(1)
                                  if re.search(r"build: (\w+)", text) else None),
            "raw_tail": text[-800:]}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Server:
    """拉起 llama-server，抓启动日志里的后端证据，退出时确保回收。"""

    BACKEND_PATTERNS = [
        ("cuda_init", r"ggml_cuda_init|CUDA0"),
        ("layers_offloaded", r"offloaded\s+(\d+)/(\d+)\s+layers to GPU"),
        ("model_buffer", r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB"),
        ("kv_buffer", r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB"),
        ("compute_buffer", r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB"),
        ("flash_attn", r"flash_attn|FLASH_ATTN"),
        ("cpu_only", r"using CPU backend only|no usable GPU"),
    ]

    def __init__(self, binary: str, model: str, ctx: int, parallel: int,
                 ngl: int, threads: int, fa: str = "auto", log: Path | None = None,
                 extra: list[str] | None = None):
        self.port = free_port()
        cmd = [binary, "-m", model, "-c", str(ctx), "-ngl", str(ngl),
               "-np", str(parallel), "-t", str(threads), "--host", "127.0.0.1",
               "--port", str(self.port), "-fa", fa]
        if extra:
            cmd += extra
        self.cmd = cmd
        self.log = log or Path("/tmp/llama-server.log")
        self.proc: subprocess.Popen | None = None
        self.backend: dict = {}

    def __enter__(self):
        self.fh = open(self.log, "wb")
        self.proc = subprocess.Popen(self.cmd, stdout=self.fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server 提前退出，日志见 {self.log}")
            try:
                http_json(f"http://127.0.0.1:{self.port}/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            raise TimeoutError("llama-server 启动超时")
        self.fh.flush()
        self.backend = self.detect_backend()
        return self

    def detect_backend(self) -> dict:
        """服务端日志给的是「服务进程用了什么」；后端名还要靠 llama-bench 的表格。

        这个 build 的 llama-server 默认不打印设备与 offload 细节，
        所以 bench_probe() 会另跑一次 llama-bench 拿 `backend` 列与显存设备信息。
        """
        text = Path(self.log).read_text(errors="replace")
        found = {}
        for name, pat in self.BACKEND_PATTERNS:
            m = re.search(pat, text)
            if m:
                found[name] = m.groups() if m.groups() else True
        found["cuda_lines"] = len(re.findall(r"CUDA0", text))
        found["ctx_per_slot"] = (re.search(r"n_ctx_slot = (\d+)", text).group(1)
                                 if re.search(r"n_ctx_slot = (\d+)", text) else None)
        found["log"] = str(self.log)
        return found

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.fh.close()
        return False

    # ---- 一次流式请求：客户端墙钟 + 服务端 timings ----
    def completion(self, prompt: str, n_predict: int) -> dict:
        payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                   "stream": True, "cache_prompt": False}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion",
                                     data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft = None
        text = []
        timings = {}
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                ev = json.loads(body)
                if ev.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(ev["content"])
                if ev.get("timings"):
                    timings = ev["timings"]
        total = time.perf_counter() - t0
        return {"ttft_ms": None if ttft is None else round(ttft * 1e3, 2),
                "total_ms": round(total * 1e3, 2),
                "text_chars": len("".join(text)),
                "server": timings}


def build_prompt(target_tokens: int, chars_per_token: float = 1.6) -> str:
    """把固定文本重复到大约 target_tokens 个 token。

    真实 token 数由服务端 `timings.prompt_n` 回读，超出/不足都记录在案，
    不假装它正好等于目标值。
    """
    reps = max(1, int(target_tokens / (len(PASSAGE) / chars_per_token)))
    return "".join(PASSAGE for _ in range(reps))


def run_request(srv: Server, prompt: str, n_predict: int, out: list, idx: int,
                deadline_ms: float | None) -> None:
    try:
        r = srv.completion(prompt, n_predict)
        r["idx"] = idx
        r["deadline_miss"] = (deadline_ms is not None
                              and r["total_ms"] > deadline_ms)
        out[idx] = r
    except Exception as e:  # noqa: BLE001
        out[idx] = {"idx": idx, "error": repr(e)}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return round(ys[min(len(ys) - 1, int(len(ys) * q))], 2)


def scan_case(srv: Server, prompt_tokens: int, concurrency: int, n_predict: int,
              repeats: int, deadline_ms: float | None) -> dict:
    prompt = build_prompt(prompt_tokens)
    rows: list[dict] = []
    for _ in range(repeats):
        out: list = [None] * concurrency
        t0 = time.perf_counter()
        ts = [threading.Thread(target=run_request,
                               args=(srv, prompt, n_predict, out, i, deadline_ms))
              for i in range(concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        gen = sum((r or {}).get("server", {}).get("predicted_n", 0)
                  for r in out if r and "server" in r)
        prompt_n = next((r["server"].get("prompt_n") for r in out
                         if r and r.get("server")), None)
        rows.append({
            "concurrency": concurrency, "wall_ms": round(wall * 1e3, 2),
            "requested_prompt_tokens": prompt_tokens, "server_prompt_n": prompt_n,
            "generated_tokens": gen,
            "aggregate_tok_s": round(gen / wall, 2) if wall > 0 else None,
            "ttft_ms": [r["ttft_ms"] for r in out if r and r.get("ttft_ms")],
            "total_ms": [r["total_ms"] for r in out if r and "total_ms" in r],
            "errors": [r for r in out if r and "error" in r],
            "deadline_miss": sum(1 for r in out if r and r.get("deadline_miss")),
            "per_request": out,
        })
    ttfts = [x for row in rows for x in row["ttft_ms"]]
    totals = [x for row in rows for x in row["total_ms"]]
    return {
        "concurrency": concurrency,
        "repeats": repeats,
        "requests": sum(len(r["total_ms"]) for r in rows),
        "ttft_ms": {"p50": percentile(ttfts, 0.5), "p95": percentile(ttfts, 0.95),
                    "max": round(max(ttfts), 2) if ttfts else None},
        "total_ms": {"p50": percentile(totals, 0.5), "p95": percentile(totals, 0.95),
                     "max": round(max(totals), 2) if totals else None},
        "aggregate_tok_s": round(statistics.median(
            [r["aggregate_tok_s"] for r in rows if r["aggregate_tok_s"]]), 2),
        "errors": sum(len(r["errors"]) for r in rows),
        "deadline_miss": sum(r["deadline_miss"] for r in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------- tegrastats
# 功耗轨的名字随平台而变：Xavier/NX 用 VDD_IN / VDD_CPU_GPU_CV / VDD_SOC，
# Orin 用 VIN_SYS_5V0 / VDD_GPU_SOC / VDD_CPU_CV。两种都收，读不到就是 None，
# 而不是把别的轨当成总功耗。
TEGRA_RE = {
    "ram": r"RAM (\d+)/(\d+)MB",
    "gr3d_pct": r"GR3D_FREQ (\d+)%",
    "gr3d_mhz": r"GR3D_FREQ \d+%@\[(\d+)\]",
    "tj_c": r"tj@(\d+(?:\.\d+)?)C",
    "gpu_c": r"gpu@(\d+(?:\.\d+)?)C",
    "cpu_c": r"cpu@(\d+(?:\.\d+)?)C",
    "vdd_in_mw": r"(?:VDD_IN|VIN_SYS_5V0) (\d+)mW",
    "vdd_gpu_soc_mw": r"(?:VDD_CPU_GPU_CV|VDD_GPU_SOC) (\d+)mW",
    "vdd_cpu_cv_mw": r"(?:VDD_CPU_CV) (\d+)mW",
    "vdd_soc_mw": r"(?<!_VDD_)VDD_SOC (\d+)mW",
}


def parse_tegrastats(line: str) -> dict:
    out = {}
    for k, pat in TEGRA_RE.items():
        m = re.search(pat, line)
        if not m:
            continue
        g = m.groups()
        if k == "ram":
            out["ram_used_mb"], out["ram_total_mb"] = int(g[0]), int(g[1])
        elif k == "cpu_pct":
            out[k] = g[0]
        else:
            try:
                out[k] = int(g[0])
            except ValueError:
                out[k] = float(g[0])
    m = re.search(r"CPU \[([^\]]+)\]", line)
    if m:
        pcts = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
        out["cpu_max_pct"] = max(pcts) if pcts else None
        out["cpu_mean_pct"] = round(sum(pcts) / len(pcts), 1) if pcts else None
        out["cpu_cores"] = len(pcts)
    return out


class TegraLogger(threading.Thread):
    def __init__(self, interval_ms: int = 1000):
        super().__init__(daemon=True)
        self.interval = interval_ms
        self.samples: list[dict] = []
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    def run(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                                         stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.samples.append({"error": "tegrastats 不存在"})
            return
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            s = parse_tegrastats(line)
            s["t"] = time.time()
            self.samples.append(s)

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def tegrastats_summary(samples: list[dict]) -> dict:
    ok = [s for s in samples if "ram_used_mb" in s]
    if not ok:
        return {"samples": 0, "error": samples[0].get("error") if samples else "无样本"}
    out = {"samples": len(ok)}
    for key in ("ram_used_mb", "ram_total_mb", "tj_c", "gpu_c", "cpu_c",
                "vdd_in_mw", "vdd_gpu_soc_mw", "vdd_cpu_cv_mw", "gr3d_pct",
                "cpu_max_pct"):
        vals = [float(s[key]) for s in ok if isinstance(s.get(key), (int, float))]
        if vals:
            out[key] = {"first": round(vals[0], 1), "last": round(vals[-1], 1),
                        "mean": round(statistics.mean(vals), 1),
                        "max": round(max(vals), 1)}
    # Orin 上没有单条「整机功耗」读数：tegrastats 给的是三条轨
    # （VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0）。报告里把三条轨分别列出，
    # 同时给出**轨和**并显式标注这是轨和而不是单轨读数。
    if all(k in out for k in ("vdd_gpu_soc_mw", "vdd_cpu_cv_mw", "vdd_in_mw")):
        total = [s["vdd_gpu_soc_mw"] + s.get("vdd_cpu_cv_mw", 0) + s["vdd_in_mw"]
                 for s in ok if "vdd_gpu_soc_mw" in s and "vdd_in_mw" in s]
        out["power_total_mw"] = {"mean": round(statistics.mean(total), 1),
                                 "max": round(max(total), 1),
                                 "rails": ["vdd_gpu_soc_mw", "vdd_cpu_cv_mw",
                                           "vdd_in_mw"],
                                 "note": "轨和，不是单轨读数"}
        energy_j = 0.0
        prev = None
        for s in ok:
            if "vdd_gpu_soc_mw" not in s or "vdd_in_mw" not in s:
                continue
            p = s["vdd_gpu_soc_mw"] + s.get("vdd_cpu_cv_mw", 0) + s["vdd_in_mw"]
            if prev is not None:
                energy_j += (p + prev[1]) / 2 * (s["t"] - prev[0]) / 1000.0
            prev = (s["t"], p)
        out["energy_j"] = round(energy_j, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["info", "scan", "ppl", "sustain"])
    ap.add_argument("--labels", required=True,
                    help="名称=路径[,名称=路径]，如 f16=/m/Qwen3-F16.gguf")
    ap.add_argument("--server-bin", default="llama-server")
    ap.add_argument("--cli-bin", default="llama-cli")
    ap.add_argument("--ppl-bin", default="llama-perplexity")
    ap.add_argument("--bench-bin", default="llama-bench")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt-tokens", default="128,2048,8192")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--deadline-ms", type=float, default=None)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = dict(kv.split("=", 1) for kv in args.labels.split(","))
    threads = args.threads or len(os.sched_getaffinity(0))
    logdir = Path(args.logdir or ".") 
    logdir.mkdir(parents=True, exist_ok=True)
    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "mode": args.mode,
        "threads": threads, "ctx": args.ctx, "ngl": args.ngl,
        "arg_max": args.__dict__.copy(), "models": {},
    }
    res["device"] = {
        "l4t": Path("/etc/nv_tegra_release").read_text(errors="replace").strip()
        if Path("/etc/nv_tegra_release").exists() else None,
        "power_mode": sh(["nvpmodel", "-q"])["out"].strip() if
        sh(["bash", "-lc", "command -v nvpmodel"])["rc"] == 0 else None,
        "jetson_clocks": sh(["bash", "-lc",
                             "jetson_clocks --show 2>/dev/null | head -20"])["out"],
    }

    for label, model in labels.items():
        entry: dict = {"model": model, "size_bytes": Path(model).stat().st_size}
        print(f"\n===== {label}: {model} "
              f"({entry['size_bytes'] / 2**30:.2f} GiB)")

        if args.mode == "info":
            with Server(args.server_bin, model, args.ctx, args.parallel, args.ngl,
                        threads, log=logdir / f"server_{label}.log") as srv:
                entry["backend"] = srv.backend
                entry["startup_cmd"] = srv.cmd
                r = srv.completion(build_prompt(64), 16)
                entry["smoke"] = r
                print(f"    backend: {srv.backend}")
                print(f"    冒烟请求 TTFT {r['ttft_ms']} ms，总计 {r['total_ms']} ms")
            entry["bench"] = bench_probe(args.bench_bin, model, args.ngl, threads)
            for row in entry["bench"].get("rows", []):
                print(f"    llama-bench backend={row['backend']} ngl={row['ngl']} "
                      f"{row['test']} {row['t_s']} ± {row['sd']} t/s")
            for d in entry["bench"].get("devices", []):
                print(f"    device {d['index']}: {d['name']} cc={d['cc']} "
                      f"VRAM={d['vram_mib']} MiB VMM={d['vmm']}")

        elif args.mode == "scan":
            entry["cases"] = []
            with Server(args.server_bin, model, args.ctx, args.parallel, args.ngl,
                        threads, log=logdir / f"server_{label}.log") as srv:
                entry["backend"] = srv.backend
                for pt in [int(x) for x in args.prompt_tokens.split(",")]:
                    for conc in [int(x) for x in args.concurrency.split(",")]:
                        if conc > args.parallel:
                            continue
                        c = scan_case(srv, pt, conc, args.n_predict, args.repeats,
                                      args.deadline_ms)
                        c["prompt_tokens_target"] = pt
                        entry["cases"].append(c)
                        print(f"    prompt≈{pt:>5} 并发{conc}  "
                              f"TTFT p50/p95 {c['ttft_ms']['p50']}/"
                              f"{c['ttft_ms']['p95']} ms  "
                              f"总计 p95 {c['total_ms']['p95']} ms  "
                              f"聚合 {c['aggregate_tok_s']} tok/s  "
                              f"实际 prompt_n={c['rows'][0].get('server_prompt_n')}")

        elif args.mode == "ppl":
            if not args.corpus:
                raise SystemExit("--mode ppl 需要 --corpus")
            r = sh([args.ppl_bin, "-m", model, "-f", args.corpus, "-c", "512",
                    "--chunks", str(args.chunks), "-ngl", str(args.ngl),
                    "-t", str(threads)], timeout=7200)
            m = re.search(r"Final estimate: PPL = ([\d.]+)", r["out"] + r["err"])
            entry["perplexity"] = float(m.group(1)) if m else None
            entry["ppl_cmd"] = r["cmd"]
            entry["ppl_tail"] = (r["out"] + r["err"])[-1500:]
            print(f"    PPL = {entry['perplexity']}（chunks={args.chunks}）")

        elif args.mode == "sustain":
            tg = TegraLogger(1000)
            tg.start()
            entry["cases"] = []
            with Server(args.server_bin, model, args.ctx, args.parallel, args.ngl,
                        threads, log=logdir / f"server_{label}.log") as srv:
                entry["backend"] = srv.backend
                t_end = time.time() + args.minutes * 60
                rounds = 0
                while time.time() < t_end:
                    c = scan_case(srv, int(args.prompt_tokens.split(",")[0]),
                                  1, args.n_predict, 1, args.deadline_ms)
                    c["round"] = rounds
                    c["at_s"] = round(time.time() - (t_end - args.minutes * 60), 1)
                    entry["cases"].append(c)
                    rounds += 1
                    if rounds % 10 == 0:
                        print(f"    {c['at_s']:>6.0f}s  "
                              f"total p50 {c['total_ms']['p50']} ms  "
                              f"tok/s {c['aggregate_tok_s']}")
            tg.stop()
            entry["tegrastats"] = tegrastats_summary(tg.samples)
            entry["tegrastats_series"] = [
                {k: v for k, v in s.items() if k != "cpu_pct"} for s in tg.samples[::5]]
            all_tot = [x["total_ms"]["p50"] for x in entry["cases"]]
            n = max(1, len(all_tot) // 6)
            cold, hot = all_tot[:n], all_tot[-n:]
            # 输出 token 总数要从每轮的 rows 里取，不能用 tok/s 相加——
            # 速率相加得到的是「每秒 token 的平方」，量纲不对。
            out_tokens = sum(r.get("generated_tokens", 0)
                             for x in entry["cases"] for r in x["rows"])
            entry["cold_vs_hot"] = {
                "cold_p50_ms": round(statistics.median(cold), 2),
                "hot_p50_ms": round(statistics.median(hot), 2),
                "slowdown": round(statistics.median(hot) / statistics.median(cold), 3),
                "window_rounds": n,
            }
            entry["output_tokens"] = out_tokens
            if entry["tegrastats"].get("energy_j") and out_tokens:
                entry["energy_per_token_j"] = round(
                    entry["tegrastats"]["energy_j"] / out_tokens, 4)
                entry["tokens_per_joule"] = round(
                    out_tokens / entry["tegrastats"]["energy_j"], 4)
            entry["deadline_miss_total"] = sum(x["deadline_miss"] for x in entry["cases"])
            entry["rounds"] = rounds
            print(f"    冷机 p50 {entry['cold_vs_hot']['cold_p50_ms']} ms → "
                  f"热稳态 {entry['cold_vs_hot']['hot_p50_ms']} ms "
                  f"({entry['cold_vs_hot']['slowdown']}×)")
            print(f"    tegrastats: {entry['tegrastats']}")
            print(f"    能量/输出 token: {entry.get('energy_per_token_j')} J，"
                  f"deadline miss {entry['deadline_miss_total']}/{rounds}")

        res["models"][label] = entry

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
