#!/usr/bin/env python3
"""labs/L8/worker_lifecycle.py - 真实 worker 进程的六阶段生命周期与探针 (8.2-A).

这里监督的是**真实进程**，不是仿真: 用 subprocess 起 vLLM, 用 /proc 判断进程树，
用三个互相独立的探针区分三种"看起来活着"的状态:

    pid_alive   进程还在（/proc/<pid> 存在）        —— 最弱
    http_ready  /v1/models 返回 200                 —— 中间
    serving     一次 1 token 的真实补全返回 200 且有 token —— 最强, 才算 READY

状态机:

    ALLOCATED -> LOADING -> WARMING -> READY -> DRAINING -> STOPPED

    ALLOCATED 选定 GPU 且确认显存空闲（GPU lease 的本地等价物）
    LOADING   进程已起、权重在加载（进程树 + 日志标记）
    WARMING   引擎编译/捕获（CPU 与显存都在动，但还不能服务）
    READY     三个探针全部通过
    DRAINING  不再接新请求，等在途请求归零
    STOPPED   进程退出且显存回到基线

每个事件与每次探测都写进 JSONL，字段是单调时钟秒与进程树快照，供逐阶段复算。
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 状态常量
ALLOCATED = "ALLOCATED"
LOADING = "LOADING"
WARMING = "WARMING"
READY = "READY"
DRAINING = "DRAINING"
STOPPED = "STOPPED"


def gpu_free_mib(index: int) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(index):
            try:
                used = int(parts[1].split()[0])
                total = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
                    text=True, timeout=10).splitlines()[index].split()[0])
                return total - used
            except Exception:  # noqa: BLE001
                return None
    return None


def process_tree(pid: int) -> List[Dict[str, Any]]:
    """返回以 pid 为根的进程树快照（(pid, ppid, comm, rss_kb)）。"""
    rows: List[Tuple[int, int, str, int]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            rest = stat[stat.rindex(")") + 2:].split()
            ppid = int(rest[1])
            rss = 0
            try:
                for line in (entry / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
            except Exception:  # noqa: BLE001
                pass
            rows.append((int(entry.name), ppid, comm, rss))
        except Exception:  # noqa: BLE001
            continue
    children: Dict[int, List[int]] = {}
    for p, pp, _, _ in rows:
        children.setdefault(pp, []).append(p)
    by_pid = {p: (pp, c, r) for p, pp, c, r in rows}
    out: List[Dict[str, Any]] = []

    def walk(node: int, depth: int) -> None:
        if node not in by_pid:
            return
        pp, c, r = by_pid[node]
        out.append({"pid": node, "ppid": pp, "comm": c, "rss_kb": r, "depth": depth})
        for ch in children.get(node, []):
            walk(ch, depth + 1)

    walk(pid, 0)
    return out


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


@dataclasses.dataclass
class Probe:
    t_s: float
    pid_alive: bool
    http_ready: bool
    serving: bool
    gpu_used_mib: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class ProbeClient:
    """对 worker 做三级探测。只依赖标准库, 不引入额外依赖。"""

    def __init__(self, port: int, model: str, gpu_index: int):
        self.base = f"http://127.0.0.1:{port}"
        self.model = model
        self.gpu_index = gpu_index

    def _http(self, method: str, path: str, body: Optional[bytes] = None,
              timeout: float = 5.0) -> Tuple[Optional[int], bytes]:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + path, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
        except Exception:  # noqa: BLE001
            return None, b""

    def http_ready(self) -> bool:
        code, _ = self._http("GET", "/v1/models", timeout=3.0)
        return code == 200

    def serving(self) -> bool:
        body = json.dumps({
            "model": self.model,
            "prompt": [9707],
            "max_tokens": 1,
            "temperature": 0.0,
        }).encode()
        code, raw = self._http("POST", "/v1/completions", body, timeout=30.0)
        if code != 200:
            return False
        try:
            obj = json.loads(raw)
            return bool(obj.get("choices"))
        except Exception:  # noqa: BLE001
            return False

    def gpu_used_mib(self) -> Optional[int]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                text=True, timeout=10)
        except Exception:  # noqa: BLE001
            return None
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.gpu_index):
                return int(parts[1].split()[0])
        return None

    def probe(self, t_s: float, pid: int) -> Probe:
        return Probe(t_s=t_s, pid_alive=pid_alive(pid),
                     http_ready=self.http_ready(), serving=self.serving(),
                     gpu_used_mib=self.gpu_used_mib())


class WorkerSupervisor:
    """起一个真实 vLLM worker 并把六阶段事件写进 JSONL。"""

    def __init__(self, name: str, gpu_index: int, port: int, model: str,
                 out_dir: Path, gpu_memory_utilization: float = 0.35,
                 max_model_len: int = 4096, extra_args: Optional[List[str]] = None,
                 cache_root: Optional[str] = None, python: str = "python",
                 vllm: str = "vllm"):
        self.name = name
        self.gpu_index = gpu_index
        self.port = port
        self.model = model
        self.out_dir = out_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []
        self.cache_root = cache_root
        self.python = python
        self.vllm = vllm

        self.events: List[Dict[str, Any]] = []
        self.probes: List[Probe] = []
        self.state = ALLOCATED
        self.proc: Optional[subprocess.Popen] = None
        self.t0 = time.monotonic()
        self.log_path = out_dir / f"{name}.log"

    # ---- 记录 -----------------------------------------------------------
    def _now(self) -> float:
        return time.monotonic() - self.t0

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"name": self.name, "t_s": self._now(), "event": kind,
               "state": self.state, **fields}
        if self.proc is not None and pid_alive(self.proc.pid):
            rec["tree"] = process_tree(self.proc.pid)
        self.events.append(rec)
        print(f"[{self.name}] {self._now():7.3f}s {self.state:9s} {kind} "
              f"{ {k: v for k, v in fields.items() if k != 'tree'} }", flush=True)

    # ---- 状态迁移 -------------------------------------------------------
    def start(self) -> None:
        free = gpu_free_mib(self.gpu_index)
        self.event("gpu_lease", gpu_index=self.gpu_index, free_mib=free)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        if self.cache_root:
            env["VLLM_CACHE_ROOT"] = self.cache_root
            env["TORCHINDUCTOR_CACHE_DIR"] = f"{self.cache_root}/torchinductor"
            env["TRITON_CACHE_DIR"] = f"{self.cache_root}/triton"
        cmd = [self.vllm, "serve", self.model,
               "--port", str(self.port),
               "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--max-model-len", str(self.max_model_len),
               *self.extra_args]
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        self.state = LOADING
        self.event("process_started", pid=self.proc.pid, cmd=" ".join(cmd),
                   cache_root=self.cache_root)

        # 轮询: 每 100 ms 看进程, 每 250 ms 做一次 HTTP 探测。
        # 用日志里的标记把 LOADING 与 WARMING 分开, 这是引擎自己报告的阶段。
        saw_weights = saw_warming = False
        deadline = time.monotonic() + 900.0
        next_http = time.monotonic()
        while time.monotonic() < deadline:
            if not pid_alive(self.proc.pid):
                self.state = STOPPED
                self.event("process_exited_early", returncode=self.proc.poll())
                return
            if not saw_weights and "Loading weights took" in self._log_tail():
                saw_weights = True
                self.state = WARMING
                self.event("weights_loaded")
            if not saw_warming and "Capturing CUDA graphs" in self._log_tail():
                saw_warming = True
                self.event("graph_capture_started")
            if time.monotonic() >= next_http:
                next_http = time.monotonic() + 0.25
                p = ProbeClient(self.port, self.model, self.gpu_index).probe(self._now(),
                                                                            self.proc.pid)
                self.probes.append(p)
                if p.serving:
                    self.state = READY
                    self.event("ready", http_ready=p.http_ready, serving=p.serving,
                               gpu_used_mib=p.gpu_used_mib)
                    return
            time.sleep(0.1)
        self.event("ready_timeout")

    def _log_tail(self, n: int = 4000) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[-n:]
        except Exception:  # noqa: BLE001
            return ""

    def drain(self, inflight, deadline_s: float = 120.0) -> float:
        """不再接新请求, 等在途请求归零。返回实际等待时间。"""
        self.state = DRAINING
        t_start = self._now()
        self.event("drain_start", inflight=inflight())
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            if inflight() <= 0:
                break
            time.sleep(0.05)
        waited = self._now() - t_start
        self.event("drain_done", waited_s=waited, inflight=inflight())
        return waited

    def stop(self, grace_s: float = 30.0) -> Dict[str, Any]:
        if self.proc is None:
            return {}
        base = gpu_free_mib(self.gpu_index)
        self.event("stop_signal", signal="SIGTERM")
        self.proc.send_signal(signal.SIGTERM)
        try:
            rc = self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            self.event("stop_sigkill")
            self.proc.kill()
            rc = self.proc.wait(timeout=30)
        # 等显存真正归还 (SIGTERM 返回后驱动可能还要几百毫秒)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            free = gpu_free_mib(self.gpu_index)
            if free is not None and base is not None and free >= base - 256:
                break
            time.sleep(0.2)
        self.state = STOPPED
        out = {"returncode": rc, "exit_s": self._now(),
               "gpu_free_before_mib": base, "gpu_free_after_mib": gpu_free_mib(self.gpu_index)}
        self.event("stopped", **out)
        try:
            self.log_f.close()
        except Exception:  # noqa: BLE001
            pass
        return out

    def crash(self) -> Dict[str, Any]:
        """模拟 worker 崩溃: 直接 SIGKILL, 不做排空。"""
        if self.proc is None:
            return {}
        self.event("crash_sigkill")
        self.proc.kill()
        rc = self.proc.wait(timeout=30)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            if not pid_alive(self.proc.pid):
                break
            time.sleep(0.1)
        self.state = STOPPED
        out = {"returncode": rc, "exit_s": self._now()}
        self.event("crashed", **out)
        return out

    # ---- 落盘 -----------------------------------------------------------
    def dump(self) -> None:
        with open(self.out_dir / f"{self.name}_lifecycle.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(self.out_dir / f"{self.name}_probes.jsonl", "w", encoding="utf-8") as f:
            for p in self.probes:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    def phase_segments(self) -> List[Dict[str, Any]]:
        """按每次 `gpu_lease` 切段后逐段统计。

        同一个 supervisor 会经历"首次启动 → 崩溃 → 重启"或"启动 → 排空 → 滚动更新 → 新进程"
        多次启动; 只看第一条 `gpu_lease` 与第一条 `ready` 会把重启算成 117 s 而不是 55 s。
        """
        segs: List[Dict[str, Any]] = []
        starts = [i for i, e in enumerate(self.events) if e["event"] == "gpu_lease"]
        for si, start_i in enumerate(starts):
            end_i = starts[si + 1] if si + 1 < len(starts) else len(self.events)
            ev = self.events[start_i:end_i]

            def t_of(kind: str) -> Optional[float]:
                for e in ev:
                    if e["event"] == kind:
                        return e["t_s"]
                return None

            lease, started = t_of("gpu_lease"), t_of("process_started")
            weights, ready = t_of("weights_loaded"), t_of("ready")
            t_lo = lease if lease is not None else 0.0
            t_hi = end_i and (self.events[end_i]["t_s"] if end_i < len(self.events) else None)
            probes = [p for p in self.probes
                      if p.t_s >= t_lo and (t_hi is None or p.t_s < t_hi)]
            segs.append({
                "name": self.name,
                "segment": si,
                "gpu_lease_s": lease,
                "process_start_s": started,
                "weights_loaded_s": weights,
                "ready_s": ready,
                "lease_to_process_s": None if None in (lease, started) else started - lease,
                "process_to_weights_s": None if None in (started, weights) else weights - started,
                "weights_to_ready_s": None if None in (weights, ready) else ready - weights,
                "process_to_ready_s": None if None in (started, ready) else ready - started,
                "total_to_ready_s": None if None in (lease, ready) else ready - lease,
                "ready_reached": ready is not None,
                "probe_counts": {
                    "pid_alive_only": sum(1 for p in probes
                                          if p.pid_alive and not p.http_ready and not p.serving),
                    "http_ready_not_serving": sum(1 for p in probes
                                                  if p.pid_alive and p.http_ready and not p.serving),
                    "serving": sum(1 for p in probes if p.serving),
                    "total": len(probes),
                    "pid_alive_only_seconds": 0.25 * sum(
                        1 for p in probes if p.pid_alive and not p.http_ready and not p.serving),
                },
            })
        return segs

    def phase_summary(self) -> Dict[str, Any]:
        """兼容旧调用: 返回**最后一次**启动的段 (重启场景下才是有意义的那次)。"""
        segs = self.phase_segments()
        return segs[-1] if segs else {"name": self.name, "probe_counts": {}}
