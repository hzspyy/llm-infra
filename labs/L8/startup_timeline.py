#!/usr/bin/env python3
"""labs/L8/startup_timeline.py - 8.7-A/B: 服务就绪账本的逐段测量.

把"从零到一个能服务的 worker"拆成可采样的段, 每段都有独立证据。阶段标记直接来自
vLLM 自己的日志行 (不用模糊匹配), 时间戳取本进程的单调时钟, 所以能与 /proc 采样对齐:

    Initializing a V1 LLM engine            引擎进程起来
    Time spent downloading weights ...      权重解析/取回 (本机上这一步走 NFS)
    Loading weights took ...                safetensors 反序列化 + H2D
    Model loading took ... GiB memory        权重阶段合计
    torch.compile took ... s                编译
    Graph capturing finished in ... secs    图捕获
    (一次真实补全成功)                       READY

冷态不靠"换个文件名"认定, 也不 drop_caches。两件事同时做:

  * `--evict-weights` 用 `posix_fadvise(POSIX_FADV_DONTNEED)` 只驱逐该模型的权重页
    (定点驱逐, 不影响机器上其它文件), 这是本机能拿到的真实冷读条件;
  * /proc/<pid>/io 的 read_bytes 给证据: 冷读时它应接近权重体积 (3.78 GiB), 热读时接近 0。

无法证明冷态时如实标记, 不假设。

用法:
    python labs/L8/startup_timeline.py --out-dir <dir> --gpu 0 --port 19300 \
        --cache-root <dir> --label cold [--evict-weights] [--model-path <snapshot>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.worker_lifecycle import ProbeClient, gpu_free_mib  # noqa: E402

# 精确日志标记 -> 事件名。顺序即事件顺序; 每个标记只取**首次出现**。
# 注意 "Graph capturing finished" 在 PIECEWISE 与 FULL 两个阶段各打印一次,
# 取首次出现才对应用户可感知的捕获耗时 (FULL 只占几秒)。
MARKERS = [
    ("Initializing a V1 LLM engine", "engine_init"),
    ("Loading model from scratch", "model_load_scratch"),
    ("Time spent downloading weights", "weights_fetch_done"),
    ("Loading weights took", "weights_loaded"),
    ("Model loading took", "model_loaded"),
    ("torch.compile took", "compile_done"),
    ("Capturing CUDA graphs (PIECEWISE)", "capture_start"),
    ("Graph capturing finished", "graph_capture_done"),
]
FETCH_RE = re.compile(r"Time spent downloading weights for .*?: ([\d.]+) seconds")
WEIGHTS_RE = re.compile(r"Loading weights took ([\d.]+) seconds")
MODEL_RE = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
COMPILE_RE = re.compile(r"torch.compile took ([\d.]+) s in total")
CAPTURE_RE = re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB")
FS_RE = re.compile(r"Filesystem type for checkpoints: (\S+)\. Checkpoint size: ([\d.]+) GiB\. "
                   r"Available RAM: ([\d.]+) GiB")


def proc_io(pid: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            k, v = line.split(":")
            out[k.strip()] = int(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        rest = stat[stat.rindex(")") + 2:].split()
        out["minflt"] = int(rest[7])
        out["majflt"] = int(rest[9])
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                out["rss_kb"] = int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return out


def evict_paths(paths: List[Path]) -> Dict[str, Any]:
    """对给定文件做定点页缓存驱逐 (POSIX_FADV_DONTNEED), 返回字节数与文件数。"""
    total = 0
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                sz = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total += sz
                n += 1
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001
            continue
    return {"files": n, "bytes": total, "gib": round(total / 2**30, 3)}


def measure_import(python: str, modules: List[str]) -> Dict[str, Any]:
    code = ("import json,time\n"
            f"mods={modules!r}\n"
            "out=[]\n"
            "for m in mods:\n"
            "    t=time.monotonic()\n"
            "    __import__(m)\n"
            "    out.append((m, time.monotonic()-t))\n"
            "print(json.dumps(out))\n")
    t0 = time.monotonic()
    raw = subprocess.check_output([python, "-c", code], text=True, timeout=900)
    total = time.monotonic() - t0
    per = json.loads(raw.strip().splitlines()[-1])
    return {"total_s": total, "per_module_s": {m: round(s, 3) for m, s in per}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=19300)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--model-path", default=None,
                    help="显式本地快照路径; 给定时 --evict-weights 会驱逐它的权重文件")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--vllm", default="vllm")
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--evict-weights", action="store_true")
    ap.add_argument("--skip-import", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_root)
    fresh = not cache.exists() or not any(cache.iterdir())
    cache.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "label": args.label, "gpu_index": args.gpu, "model": args.model,
        "cache_root": str(cache), "cache_root_was_empty": fresh,
        "gpu_free_before_mib": gpu_free_mib(args.gpu),
    }

    if args.evict_weights:
        if not args.model_path:
            raise SystemExit("--evict-weights 需要 --model-path")
        files = [p for p in Path(args.model_path).rglob("*") if p.is_file()]
        ev = evict_paths(files)
        result["eviction"] = ev
        print(f"[{args.label}] page-cache evict: {ev}", flush=True)

    if not args.skip_import:
        imp = measure_import(args.python, ["torch", "vllm", "transformers", "flashinfer"])
        result["import"] = imp
        print(f"[{args.label}] import total={imp['total_s']:.1f}s {imp['per_module_s']}",
              flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VLLM_CACHE_ROOT"] = str(cache)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    log_path = out / f"startup_{args.label}.log"
    # `vllm serve` 只接受一种模型指定方式: 位置参数与 `--model` 不能同时给,
    # 否则直接报 unrecognized arguments 并退出 (本轮第一次运行就踩到)。
    cmd = [args.vllm, "serve",
           *( ["--model", args.model_path] if args.model_path else [args.model] ),
           "--port", str(args.port),
           "--gpu-memory-utilization", str(args.gpu_mem_util),
           "--max-model-len", str(args.max_model_len)]

    samples: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    def ev(kind: str, **f: Any) -> None:
        events.append({"t_s": time.monotonic() - t0, "event": kind, **f})
        print(f"[{args.label}] {time.monotonic()-t0:7.2f}s {kind} {f}", flush=True)

    ev("process_start")
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        io0 = proc_io(proc.pid)
        served_name = args.model_path or args.model
        probe = ProbeClient(args.port, served_name, args.gpu)
        seen: Dict[str, bool] = {name: False for _, name in MARKERS}
        read_at: Dict[str, int] = {}
        deadline = time.monotonic() + 1800
        next_probe = time.monotonic() + 1.0
        ready = False
        saw_http_ready = False
        text = ""
        _log_pos = 0
        while time.monotonic() < deadline:
            if not Path(f"/proc/{proc.pid}").exists():
                ev("process_exit_early", rc=proc.poll())
                break
            # 增量读日志: 只读新追加的字节, 在累计缓冲里找**首次出现**的标记。
            # 只读尾部窗口会漏掉早期标记 (PIECEWISE 捕获的完成行会被 FULL 的覆盖)。
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as _lf:
                    _lf.seek(_log_pos)
                    text = text + _lf.read()
                    _log_pos = _lf.tell()
            except Exception:  # noqa: BLE001
                pass
            for marker, name in MARKERS:
                if seen[name] or marker not in text:
                    continue
                seen[name] = True
                io = proc_io(proc.pid)
                read_at[name] = io.get("read_bytes", 0)
                extra: Dict[str, Any] = {}
                if name == "weights_fetch_done":
                    m = FETCH_RE.search(text)
                    extra["fetch_s"] = float(m.group(1)) if m else None
                    f = FS_RE.search(text)
                    if f:
                        extra.update({"fs": f.group(1), "ckpt_gib": float(f.group(2)),
                                      "avail_ram_gib": float(f.group(3))})
                elif name == "weights_loaded":
                    m = WEIGHTS_RE.search(text)
                    extra["load_s"] = float(m.group(1)) if m else None
                elif name == "model_loaded":
                    m = MODEL_RE.search(text)
                    if m:
                        extra.update({"model_gib": float(m.group(1)),
                                      "model_load_s": float(m.group(2))})
                elif name == "compile_done":
                    m = COMPILE_RE.search(text)
                    extra["compile_s"] = float(m.group(1)) if m else None
                elif name == "graph_capture_done":
                    m = CAPTURE_RE.search(text)
                    if m:
                        extra.update({"capture_s": int(m.group(1)),
                                      "capture_gib": float(m.group(2))})
                ev(name, io_read_bytes_delta=io.get("read_bytes", 0) - io0.get("read_bytes", 0),
                   **extra)
            if time.monotonic() >= next_probe:
                next_probe = time.monotonic() + 2.0
                p = probe.probe(time.monotonic() - t0, proc.pid)
                samples.append({**p.to_dict(), **proc_io(proc.pid)})
                if p.http_ready and not saw_http_ready:
                    # HTTP 层一可用就查一次补全端点的状态码: 服务名与请求里的 model
                    # 不一致时它返回 404, 探针永远不会成功, 必须当场暴露而不是空转到超时。
                    saw_http_ready = True
                    code, _ = probe._http(
                        "POST", "/v1/completions",
                        json.dumps({"model": served_name, "prompt": [9707],
                                    "max_tokens": 1}).encode(), timeout=30.0)
                    ev("serving_probe_status", http_status=code)
                    if code == 404:
                        ev("fail_serving_404", note="model 名与服务名不一致; 本次运行作废")
                        break
                if p.serving:
                    ready = True
                    ev("ready", io=proc_io(proc.pid), gpu_used_mib=p.gpu_used_mib)
                    break
            time.sleep(args.interval)

        first: Dict[str, Any] = {}
        if ready:
            body = json.dumps({"model": served_name, "prompt": [9707] * 64,
                               "max_tokens": 16, "temperature": 0.0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/completions",
                                         data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            t_req = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    first = {"ok": r.status == 200, "latency_s": time.monotonic() - t_req}
            except Exception as e:  # noqa: BLE001
                first = {"ok": False, "error": f"{type(e).__name__}: {e}",
                         "latency_s": time.monotonic() - t_req}
            ev("first_request", **first)

        io1 = proc_io(proc.pid)
        result["io_first"] = io0
        result["io_last"] = io1
        result["io_delta"] = {k: io1.get(k, 0) - io0.get(k, 0) for k in set(io0) | set(io1)}
        result["read_bytes_mib"] = result["io_delta"].get("read_bytes", 0) / 2**20
        result["rchar_mib"] = result["io_delta"].get("rchar", 0) / 2**20
        result["majflt"] = result["io_delta"].get("majflt", 0)
        result["minflt"] = result["io_delta"].get("minflt", 0)
        result["read_bytes_at_marker"] = read_at
        result["ready"] = ready
        result["events"] = events
        result["phases"] = []
        prev_t, prev_name = 0.0, "process_start"
        for e in events:
            result["phases"].append({"from": prev_name, "to": e["event"],
                                     "duration_s": round(e["t_s"] - prev_t, 3)})
            prev_t, prev_name = e["t_s"], e["event"]

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    result["gpu_free_after_mib"] = gpu_free_mib(args.gpu)

    with (out / f"samples_{args.label}.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (out / f"startup_{args.label}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result.get(k) for k in
                      ("label", "ready", "read_bytes_mib", "rchar_mib", "majflt",
                       "minflt", "eviction")}, ensure_ascii=False))
    print(json.dumps(result["phases"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()