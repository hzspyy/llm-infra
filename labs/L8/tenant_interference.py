#!/usr/bin/env python3
"""labs/L8/tenant_interference.py - 8.8-B/A 的真实引擎侧实验.

两个子命令:

  interference  同一张卡上两种部署方式的对照:
                  shared    一个 vLLM 实例同时服务租户 A (交互) 与 B (长 prompt 批式),
                            两者各用一个 LoRA adapter
                  isolated  两个 vLLM 实例各服务一个租户 (进程隔离)
                两档都按同一份到达清单发请求, 分租户记录 TTFT/TPOT/拒绝/资源,
                并在运行中采样显卡的显存与利用率。

  identity      adapter revision 进不进缓存键: 引擎开前缀缓存并把适配器加载成 rev,
                同一段 prompt 先在 rev(A 权重) 上建立缓存, 再用
                `/v1/load_lora_adapter {load_inplace:true}` 就地换成 B 权重 (同一个
                name), 然后分别用**同一个 cache_salt** (可能命中旧 KV) 与**新的
                cache_salt** (必然重算) 请求同一段 prompt, 比较首个 token 的
                top-5 logprob。

  说明: revB 的权重是把公开 adapter 的 lora_B 取负得到的合成权重, 只用于让两份
  revision 的函数确实不同; 它不代表任何训练结果。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx  # noqa: E402

from labs.L8.load_generator import RequestSpec, StreamingClient  # noqa: E402
from labs.L8.request_metrics import RequestRecord  # noqa: E402

SNAP = ("/scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots/"
        "1cfa9a7208912126459214e8b04321603b3df60c")
LORA = "/scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots"

METRIC_FILTERS = ("prefix_cache", "kv_cache_usage", "num_requests", "request_success",
                  "time_to_first_token", "lora")


# --------------------------------------------------------------------------
# 引擎进程
# --------------------------------------------------------------------------
class Server:
    def __init__(self, python: str, port: int, log_path: Path, model: str = SNAP,
                 served: str = "qwen3-4b", loras: Tuple[str, ...] = (),
                 max_model_len: int = 8192, gpu_mem_util: float = 0.42,
                 runtime_lora: bool = False, scheduling: str = "fcfs"):
        self.python = python
        self.port = port
        self.log_path = log_path
        self.model = model
        self.served = served
        self.loras = loras
        self.max_model_len = max_model_len
        self.gpu_mem_util = gpu_mem_util
        self.runtime_lora = runtime_lora
        self.scheduling = scheduling
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: List[str] = []

    def start(self) -> float:
        cmd = [self.python, "-m", "vllm.entrypoints.openai.api_server",
               "--model", self.model, "--served-model-name", self.served,
               "--host", "127.0.0.1", "--port", str(self.port),
               "--max-model-len", str(self.max_model_len),
               "--gpu-memory-utilization", str(self.gpu_mem_util),
               "--enable-prefix-caching"]
        if self.loras:
            cmd += ["--enable-lora", "--max-lora-rank", "16"]
            cmd += ["--lora-modules", *self.loras]
        if self.runtime_lora:
            cmd += ["--scheduling-policy", self.scheduling]
        self.cmd = cmd
        env = dict(os.environ)
        if self.runtime_lora:
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        t0 = time.monotonic()
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"引擎退出 code={self.proc.returncode}, 见 {self.log_path}")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return time.monotonic() - t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise TimeoutError("引擎未就绪")

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
        time.sleep(3)


def scrape_metrics(port: int) -> Dict[str, float]:
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


def gpu_sample() -> Dict[str, Any]:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        used, total, util = [x.strip() for x in q.split(",")]
        return {"mem_used_mib": float(used), "mem_total_mib": float(total),
                "util_pct": float(util)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --------------------------------------------------------------------------
# B 段: 干扰对照
# --------------------------------------------------------------------------
def make_ids(n: int, seed: int) -> List[int]:
    import random
    rng = random.Random(seed)
    return [rng.randrange(1000, 90000) for _ in range(n)]


async def send_one(port: int, model: str, spec: RequestSpec, timeout_s: float,
                   client: httpx.AsyncClient, t0: float):
    cli = StreamingClient(f"http://127.0.0.1:{port}", "openai", model, timeout_s, t0, client)
    return await cli.run(spec)


async def tenant_load(port: int, model: str, tenant: str, n: int, interval_s: float,
                      prompt_ids: List[int], output_len: int, timeout_s: float,
                      t0: float, sampler: List[Dict[str, Any]]) -> List[RequestRecord]:
    recs: List[RequestRecord] = []
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def one(i: int):
            await asyncio.sleep(i * interval_s)
            spec = RequestSpec(f"{tenant}-{i}", 0.0, len(prompt_ids), output_len,
                               prompt_ids)
            return await send_one(port, model, spec, timeout_s, client, t0)
        recs = list(await asyncio.gather(*[one(i) for i in range(n)]))
    return recs


def summarize(recs: List[RequestRecord]) -> Dict[str, Any]:
    ttfts = sorted(r.observed_ttft_s for r in recs if r.observed_ttft_s is not None)
    tpots = sorted(r.tpot_s for r in recs if getattr(r, "tpot_s", None) is not None)
    def q(xs, p):
        if not xs:
            return None
        return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1)))], 4)
    return {
        "n": len(recs),
        "statuses": {s: sum(1 for r in recs if r.status == s)
                     for s in {r.status for r in recs}},
        "ttft_p50": q(ttfts, 0.5), "ttft_p95": q(ttfts, 0.95), "ttft_p99": q(ttfts, 0.99),
        "tpot_p50": q(tpots, 0.5), "tpot_p99": q(tpots, 0.99),
        "n_ttft": len(ttfts),
    }


async def run_interference_mode(args, mode: str, out_dir: Path) -> Dict[str, Any]:
    t0 = time.monotonic()
    a_ids = make_ids(args.a_prompt_len, seed=1)
    b_ids = make_ids(args.b_prompt_len, seed=2)
    res: Dict[str, Any] = {"mode": mode, "a_prompt_len": args.a_prompt_len,
                           "b_prompt_len": args.b_prompt_len,
                           "a_count": args.a_count, "b_count": args.b_count,
                           "interval_s": args.interval_s}
    servers: List[Server] = []
    try:
        if mode == "shared":
            s = Server(args.python, args.port, out_dir / "shared.log", loras=(
                f"rev-a={args.lora_a}", f"rev-b={args.lora_b}"),
                gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len)
            servers.append(s)
            ready = s.start()
            res["servers"] = [{"port": s.port, "ready_s": round(ready, 2),
                               "cmd": s.cmd}]
            m0 = scrape_metrics(s.port)
            gpu0 = gpu_sample()
            tasks = [
                tenant_load(s.port, "rev-a", "A", args.a_count, args.interval_s,
                            a_ids, args.a_output_len, args.timeout_s, t0, []),
                tenant_load(s.port, "rev-b", "B", args.b_count, args.interval_s,
                            b_ids, args.b_output_len, args.timeout_s, t0, []),
            ]
            a_recs, b_recs = await asyncio.gather(*tasks)
            gpu1 = gpu_sample()
            m1 = scrape_metrics(s.port)
        else:
            s1 = Server(args.python, args.port, out_dir / "isolated_a.log",
                        loras=(f"rev-a={args.lora_a}",),
                        gpu_mem_util=args.isolated_gpu_mem_util,
                        max_model_len=args.max_model_len)
            s2 = Server(args.python, args.port + 1, out_dir / "isolated_b.log",
                        loras=(f"rev-b={args.lora_b}",),
                        gpu_mem_util=args.isolated_gpu_mem_util,
                        max_model_len=args.max_model_len)
            servers += [s1, s2]
            r1 = s1.start()
            r2 = s2.start()
            res["servers"] = [{"port": s1.port, "ready_s": round(r1, 2)},
                              {"port": s2.port, "ready_s": round(r2, 2)}]
            res["two_engines_ready_s"] = round(max(r1, r2), 2)
            m0 = scrape_metrics(s1.port)
            gpu0 = gpu_sample()
            a_recs, b_recs = await asyncio.gather(
                tenant_load(s1.port, "rev-a", "A", args.a_count, args.interval_s,
                            a_ids, args.a_output_len, args.timeout_s, t0, []),
                tenant_load(s2.port, "rev-b", "B", args.b_count, args.interval_s,
                            b_ids, args.b_output_len, args.timeout_s, t0, []),
            )
            gpu1 = gpu_sample()
            m1 = scrape_metrics(s1.port)
        res["gpu_before"] = gpu0
        res["gpu_after"] = gpu1
        res["tenant_A"] = summarize(a_recs)
        res["tenant_B"] = summarize(b_recs)
        res["metrics_delta"] = {k: round(v - m0.get(k, 0.0), 3)
                                for k, v in m1.items() if not k.startswith("_")}
    finally:
        for s in servers:
            s.stop()
    (out_dir / f"interference_{mode}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# --------------------------------------------------------------------------
# A 段: adapter revision 与缓存身份
# --------------------------------------------------------------------------
def prepare_revisions(out_dir: Path, lora_snap: Path) -> Tuple[Path, Path]:
    """revA = 公开 adapter; revB = 把 lora_B 取负的合成权重 (只用于制造差异)。"""
    import torch
    from safetensors.torch import load_file, save_file
    src = sorted(lora_snap.glob("*/"))[0]
    a_dir = out_dir / "adapter_revA"
    b_dir = out_dir / "adapter_revB"
    for d in (a_dir, b_dir):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(src, d)
    ws = load_file(str(b_dir / "adapter_model.safetensors"))
    flipped = {k: (-v if "lora_B" in k else v) for k, v in ws.items()}
    save_file({k: v.contiguous() for k, v in flipped.items()},
              str(b_dir / "adapter_model.safetensors"))
    return a_dir, b_dir


def completion_logprobs(port: int, model: str, prompt_ids: List[int], salt: str,
                        top: int = 5) -> Dict[str, Any]:
    body = {"model": model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0.0,
            "logprobs": top, "cache_salt": salt}
    with urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                   data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}),
            timeout=300) as r:
        data = json.loads(r.read().decode())
    ch = data["choices"][0]
    lp = ch.get("logprobs") or {}
    return {"token": (lp.get("tokens") or [None])[0],
            "top_logprobs": (lp.get("top_logprobs") or [{}])[0],
            "finish_reason": ch.get("finish_reason")}


def post_json(url: str, body: Dict[str, Any], timeout: float = 300) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode()
    try:
        return json.loads(txt) if txt.strip() else {}
    except json.JSONDecodeError:
        return {"raw": txt[:200]}


def cmd_identity(args) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_dir, b_dir = prepare_revisions(out_dir, Path(args.lora_snap))
    s = Server(args.python, args.port, out_dir / "identity.log",
               loras=(f"rev={a_dir}",), gpu_mem_util=args.gpu_mem_util,
               max_model_len=args.max_model_len, runtime_lora=True)
    res: Dict[str, Any] = {"adapter_revA": str(a_dir), "adapter_revB": str(b_dir),
                           "note": "revB 由 revA 的 lora_B 取负得到, 只用于制造函数差异"}
    prompt = make_ids(args.prompt_len, seed=7)
    try:
        res["ready_s"] = round(s.start(), 2)
        res["cmd"] = s.cmd
        m0 = scrape_metrics(s.port)
        # 1) revA 冷计算 (salt s1) 与再算一次 (salt s2) -> 确定性
        r1 = completion_logprobs(s.port, "rev", prompt, "s1")
        r1b = completion_logprobs(s.port, "rev", prompt, "s2")
        res["revA_salt_s1"] = r1
        res["revA_salt_s2"] = r1b
        res["revA_deterministic_across_salts"] = r1["token"] == r1b["token"]
        # 2) 用 s1 再请求两次把这条前缀的缓存坐热
        r1c = completion_logprobs(s.port, "rev", prompt, "s1")
        r1d = completion_logprobs(s.port, "rev", prompt, "s1")
        m1 = scrape_metrics(s.port)
        res["revA_salt_s1_warm"] = r1c
        res["prefix_cache_metrics_after_warm"] = {
            k: v for k, v in m1.items() if "prefix_cache" in k}
        res["prefix_cache_delta_warm"] = {
            k: round(v - m0.get(k, 0.0), 3) for k, v in m1.items()
            if "prefix_cache" in k}
        # 3) 同名就地换权重
        res["load_inplace_response"] = post_json(
            f"http://127.0.0.1:{s.port}/v1/load_lora_adapter",
            {"lora_name": "rev", "lora_path": str(b_dir), "load_inplace": True})
        # 4) 同一个 salt (=s1) 请求: 若 revision 不进键, 这里会命中 revA 的 KV
        r2_cached = completion_logprobs(s.port, "rev", prompt, "s1")
        res["revB_same_salt_after_inplace"] = r2_cached
        # 5) 新 salt: 必然重算, 得到真正的 revB 结果
        r2_fresh = completion_logprobs(s.port, "rev", prompt, "s3")
        res["revB_fresh_salt"] = r2_fresh
        res["revisions_differ"] = r2_fresh["token"] != r1["token"]
        res["stale_kv_reused"] = (r2_cached["token"] == r1["token"]
                                  and r2_cached["token"] != r2_fresh["token"])
        # 6) 换一个 name 重新加载同一份 revB 权重: 键不同 -> 必然重算
        res["load_new_name_response"] = post_json(
            f"http://127.0.0.1:{s.port}/v1/load_lora_adapter",
            {"lora_name": "rev-next", "lora_path": str(b_dir), "load_inplace": False})
        r3 = completion_logprobs(s.port, "rev-next", prompt, "s1")
        res["revB_new_name_same_salt"] = r3
        res["new_name_matches_fresh_revB"] = r3["token"] == r2_fresh["token"]
        res["metrics_final"] = {k: v for k, v in scrape_metrics(s.port).items()
                                if "prefix_cache" in k}
    finally:
        s.stop()
        tail = (out_dir / "identity.log").read_text(errors="replace").splitlines()[-80:]
        (out_dir / "identity.tail.log").write_text("\n".join(tail), encoding="utf-8")
    (out_dir / "identity.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["interference", "identity"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=18088)
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--lora-snap", default=LORA)
    ap.add_argument("--lora-a", default="")
    ap.add_argument("--lora-b", default="")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.42)
    ap.add_argument("--isolated-gpu-mem-util", type=float, default=0.32)
    ap.add_argument("--if-modes", default="shared,isolated",
                    help="interference 模式下要跑的部署方式")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--a-prompt-len", type=int, default=1024)
    ap.add_argument("--b-prompt-len", type=int, default=6144)
    ap.add_argument("--a-output-len", type=int, default=32)
    ap.add_argument("--b-output-len", type=int, default=64)
    ap.add_argument("--a-count", type=int, default=60)
    ap.add_argument("--b-count", type=int, default=12)
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()
    if args.mode == "interference":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        a_dir, b_dir = prepare_revisions(out, Path(args.lora_snap))
        args.lora_a = args.lora_a or str(a_dir)
        args.lora_b = args.lora_b or str(b_dir)
        summary = {}
        for mode in args.if_modes.split(","):
            print(f"=== interference mode={mode}", flush=True)
            r = asyncio.run(run_interference_mode(args, mode, out))
            summary[mode] = r
            print(f"  A {json.dumps(r['tenant_A'], ensure_ascii=False)}", flush=True)
            print(f"  B {json.dumps(r['tenant_B'], ensure_ascii=False)}", flush=True)
        (out / "interference_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        res = cmd_identity(args)
        print(json.dumps({k: res[k] for k in res if k.startswith(("rev", "stale", "new_name",
                                                                  "prefix_cache_delta"))},
                         ensure_ascii=False)[:1200])
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
