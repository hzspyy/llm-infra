#!/usr/bin/env python3
"""labs/L8/kv_offload_engine.py - 8.6-B/C 的真实引擎侧: vLLM 自带的 KV offloading 层级.

mini store (`tiered_kv_store.py`) 给出逐元素、逐状态的受控证据; 这一份脚本回答"接进
真实调度器与 attention 之后, 取回是否真的比重算便宜、以及重启后缓存还算不算数":

  none     不启用 KV 连接器: GPU 前缀缓存被驱逐后只能重算
  cpu      OffloadingConnector + CPUOffloadingSpec: 驱逐到主机内存
  tiering  OffloadingConnector + TieringOffloadingSpec: CPU 主层 + fs 次级层 (NVMe)

同一组前缀在每个配置下走三段:
  warm   首次请求 (prefill, 把块算出来)
  flush  另一组等量前缀, 把 warm 的 GPU 块挤掉 (触发 offload)
  reuse  再次请求 warm 前缀, 量 TTFT 与 /metrics 增量

对 tiering 配置可再加一次 `--restart`: 杀掉引擎后用同一个 fs root_dir 重启, 再请求
同一组前缀, 用来检验"服务重启后缓存状态是否还能用"。

用法:
    python labs/L8/kv_offload_engine.py --mode tiering --out-dir /path/out --restart
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx  # noqa: E402

from labs.L8.load_generator import RequestSpec, StreamingClient  # noqa: E402
from labs.L8.request_metrics import RequestRecord  # noqa: E402

SNAP = ("/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
METRIC_FILTERS = ("kv_offload", "prefix_cache", "kv_cache_usage", "external_prefix_cache",
                  "num_requests", "request_success", "time_to_first_token")


# --------------------------------------------------------------------------
# 前缀构造
# --------------------------------------------------------------------------
def make_prefixes(n: int, length: int, seed: int = 0, vocab: int = 100000) -> List[List[int]]:
    """确定性伪随机 token 序列: 前缀之间首个 block 不同, 因此不会互相命中。"""
    import random
    rng = random.Random(seed)
    return [[rng.randrange(1000, vocab) for _ in range(length)] for _ in range(n)]


# --------------------------------------------------------------------------
# 引擎进程
# --------------------------------------------------------------------------
def build_kv_transfer_config(mode: str, cpu_bytes: int, fs_dir: Path,
                             block_size: int = 16) -> Optional[str]:
    if mode == "none":
        return None
    extra: Dict[str, Any] = {
        "cpu_bytes_to_use": cpu_bytes,
        "block_size": block_size,
        "eviction_policy": "lru",
    }
    if mode == "cpu":
        extra["spec_name"] = "CPUOffloadingSpec"
    elif mode == "tiering":
        extra["spec_name"] = "TieringOffloadingSpec"
        extra["secondary_tiers"] = [{
            "type": "fs", "root_dir": str(fs_dir),
            "n_read_threads": 8, "n_write_threads": 8,
        }]
    else:
        raise ValueError(mode)
    return json.dumps({
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }, ensure_ascii=False)


class Engine:
    def __init__(self, args, mode: str, fs_dir: Path, log_path: Path):
        self.args = args
        self.mode = mode
        self.fs_dir = fs_dir
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: List[str] = []

    def start(self) -> None:
        cmd = [
            self.args.python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.args.snap,
            "--served-model-name", "qwen3-1.7b",
            "--host", "127.0.0.1", "--port", str(self.args.port),
            "--max-model-len", str(self.args.max_model_len),
            "--gpu-memory-utilization", str(self.args.gpu_mem_util),
            "--enable-prefix-caching",
        ]
        cfg = build_kv_transfer_config(self.mode, int(self.args.cpu_gib * 2 ** 30),
                                       self.fs_dir, self.args.block_size)
        if cfg:
            cmd += ["--kv-transfer-config", cfg]
        self.cmd = cmd
        log = open(self.log_path, "ab")
        env = dict(os.environ)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        self.wait_ready()

    def wait_ready(self, timeout_s: float = 600) -> float:
        t0 = time.monotonic()
        url = f"http://127.0.0.1:{self.args.port}/health"
        while time.monotonic() - t0 < timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"引擎退出, code={self.proc.returncode}, 见 {self.log_path}")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return time.monotonic() - t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise TimeoutError("引擎未在超时内就绪")

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
        # 等显存归还
        time.sleep(3)


def scrape_metrics(port: int) -> Dict[str, float]:
    """只保留与本实验相关的计数器/仪表, 避免整份 /metrics 进工件。"""
    out: Dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}  # type: ignore[dict-item]
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(f in name for f in METRIC_FILTERS):
            try:
                out[line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return out


def metric_delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a) | set(b)
    return {k: round(b.get(k, 0.0) - a.get(k, 0.0), 4) for k in sorted(keys)
            if b.get(k, 0.0) - a.get(k, 0.0) != 0}


# --------------------------------------------------------------------------
# 请求
# --------------------------------------------------------------------------
async def send_phase(port: int, model: str, prefixes: List[List[int]], tag: str,
                     output_len: int, timeout_s: float) -> List[Dict[str, Any]]:
    """顺序发一批请求并记录逐请求事件。"""
    t0 = time.monotonic()
    out: List[Dict[str, Any]] = []
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        cli = StreamingClient(f"http://127.0.0.1:{port}", "openai", model, timeout_s, t0, client)
        for i, ids in enumerate(prefixes):
            spec = RequestSpec(request_id=f"{tag}-{i}", planned_arrival_s=0.0,
                               prompt_len=len(ids), output_len=output_len, prompt_ids=ids)
            rec: RequestRecord = await cli.run(spec)
            out.append({
                "request_id": rec.request_id, "prompt_len": rec.prompt_len,
                "status": rec.status, "error": rec.error,
                "ttft_s": None if rec.observed_ttft_s is None else round(rec.observed_ttft_s, 4),
                "e2e_s": None if rec.finish_s is None else round(rec.finish_s - rec.actual_send_s, 4),
                "server_prompt_tokens": rec.server_reported_prompt_tokens,
                "tokens": len(rec.token_timestamps_s),
            })
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if r["ttft_s"] is not None]
    ttfts = sorted(r["ttft_s"] for r in ok)
    e2es = sorted(r["e2e_s"] for r in ok if r["e2e_s"] is not None)
    def q(xs, p):
        if not xs:
            return None
        i = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
        return round(xs[i], 4)
    return {
        "n": len(records), "n_ok": len(ok),
        "ttft_min": q(ttfts, 0.0), "ttft_p50": q(ttfts, 0.5), "ttft_max": q(ttfts, 1.0),
        "e2e_p50": q(e2es, 0.5),
        "server_prompt_tokens": sorted({r["server_prompt_tokens"] for r in ok}),
        "statuses": {s: sum(1 for r in records if r["status"] == s)
                     for s in {r["status"] for r in records}},
    }


def run_config(args, mode: str, warm: List[List[int]], flush: List[List[int]],
                     do_restart: bool) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_dir = out_dir / f"fs_tier_{mode}"
    if args.reset_fs and fs_dir.exists():
        shutil.rmtree(fs_dir)
    fs_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"engine_{mode}.log"
    if log_path.exists():
        log_path.unlink()
    res: Dict[str, Any] = {
        "mode": mode, "out_dir": str(out_dir), "fs_dir": str(fs_dir),
        "max_model_len": args.max_model_len, "gpu_mem_util": args.gpu_mem_util,
        "cpu_gib": args.cpu_gib, "block_size": args.block_size,
        "prefix_len": args.prefix_len, "num_prefixes": args.num_prefixes,
        "num_flush": args.num_flush, "output_len": args.output_len,
    }
    eng = Engine(args, mode, fs_dir, log_path)
    t_start = time.monotonic()
    eng.start()
    res["engine_state"] = {
        "kv_transfer_config": build_kv_transfer_config(
            mode, int(args.cpu_gib * 2 ** 30), fs_dir, args.block_size),
        "cmd": eng.cmd, "ready_s": round(time.monotonic() - t_start, 2),
    }
    try:
        phases: Dict[str, Any] = {}
        for name, prefixes in (("warm", warm), ("flush", flush), ("reuse", warm),
                               ("reuse_again", warm)):
            m0 = scrape_metrics(args.port)
            recs = asyncio.run(send_phase(args.port, "qwen3-1.7b", prefixes, name,
                                          args.output_len, args.timeout_s))
            m1 = scrape_metrics(args.port)
            phases[name] = {"records": recs, "summary": summarize(recs),
                            "metrics_delta": metric_delta(m0, m1)}
        res["phases"] = phases
        res["gpu_blocks_seen"] = {k: v for k, v in scrape_metrics(args.port).items()
                                  if "kv_cache_usage" in k}
        if do_restart:
            res["restart"] = {}
            eng.stop()
            res["restart"]["stop_s"] = 0.0
            fs_files = list(fs_dir.rglob("*.bin"))
            res["restart"]["fs_files_before"] = len(fs_files)
            res["restart"]["fs_bytes_before"] = sum(f.stat().st_size for f in fs_files)
            t1 = time.monotonic()
            eng.start()
            res["restart"]["ready_s"] = round(time.monotonic() - t1, 2)
            m0 = scrape_metrics(args.port)
            recs = asyncio.run(send_phase(args.port, "qwen3-1.7b", warm, "reuse_restart",
                                          args.output_len, args.timeout_s))
            m1 = scrape_metrics(args.port)
            res["restart"]["phase"] = {"records": recs, "summary": summarize(recs),
                                      "metrics_delta": metric_delta(m0, m1)}
    finally:
        eng.stop()
        tail = log_path.read_text(errors="replace").splitlines()[-120:]
        (out_dir / f"engine_{mode}.tail.log").write_text("\n".join(tail), encoding="utf-8")
    (out_dir / f"result_{mode}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="none,cpu,tiering")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--snap", default=SNAP)
    ap.add_argument("--port", type=int, default=18086)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--cpu-gib", type=float, default=8.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--prefix-len", type=int, default=4096)
    ap.add_argument("--num-prefixes", type=int, default=8)
    ap.add_argument("--num-flush", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--reset-fs", action="store_true", default=True)
    ap.add_argument("--keep-fs", dest="reset_fs", action="store_false")
    ap.add_argument("--restart", action="store_true",
                    help="tiering 配置结束后重启引擎, 检验 fs 层恢复")
    args = ap.parse_args()
    warm = make_prefixes(args.num_prefixes, args.prefix_len, seed=11)
    flush = make_prefixes(args.num_flush, args.prefix_len, seed=99)
    all_res = {}
    for mode in args.modes.split(","):
        print(f"=== mode={mode} ({time.strftime('%H:%M:%S')})", flush=True)
        r = run_config(args, mode, warm, flush,
                       do_restart=(args.restart and mode == "tiering"))
        all_res[mode] = {k: r[k] for k in ("mode", "engine_state", "gpu_blocks_seen",
                                           "restart") if k in r}
        for name, ph in r.get("phases", {}).items():
            all_res[mode][name] = ph["summary"]
            print(f"  {name}: {json.dumps(ph['summary'], ensure_ascii=False)}", flush=True)
            print(f"    metrics_delta: {json.dumps(ph['metrics_delta'], ensure_ascii=False)[:400]}",
                  flush=True)
        if "restart" in r:
            print(f"  restart: files={r['restart']['fs_files_before']} "
                  f"bytes={r['restart']['fs_bytes_before']} "
                  f"ready={r['restart']['ready_s']}s "
                  f"{json.dumps(r['restart']['phase']['summary'], ensure_ascii=False)}",
                  flush=True)
    (Path(args.out_dir) / "summary.json").write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
