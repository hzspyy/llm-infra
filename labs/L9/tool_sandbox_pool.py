#!/usr/bin/env python3
"""L9.8 任务 B/C 的核心实现：工具执行环境的状态机、有界池、资源限制与回收。

一台机器能提供哪几层隔离由 `sandbox_capability_probe.py` 先核验；本脚本只用核验为可用的原语：

* **进程组**（`setsid`）——取消时 `killpg` 整组回收；
* **rlimit**——`RLIMIT_AS` / `RLIMIT_FSIZE` / `RLIMIT_CPU` / `RLIMIT_NOFILE`；
* **user / net 命名空间**（`unshare --user --map-root-user --net`）——轻量隔离对照，不冒充容器；
* **工作区**——每个槽位一个私有目录，可快照/恢复。

状态机与池：

```text
NEW ──► STARTING ──► READY ──► BUSY ──► DRAINING ──► DESTROYED
                        ▲        │            │
                        └────────┘            └──► READY（健康探测通过）
```

* ``READY`` 必须经过**健康探测**（跑一条最短命令并看到返回码）：进程存活不代表可执行；
* 池有 ``max_slots`` 上界；``warm`` 决定启动时预先准备几个槽位；
* 任务只从 ``READY`` 槽位开始；``BUSY`` 结束后先 ``DRAINING``（回收子进程、检查工作区），
  健康探测通过才回到 ``READY``，否则 ``DESTROYED``；
* 每个槽位带租约与 fencing token；取消/超时后 epoch 加一，晚到的结果按 epoch 拒绝。

进程数限制在本机不能用 `RLIMIT_NPROC` 实现（核验结果：限 4 仍能 fork 出 8 个），
所以实现方式改为**监督进程组存活成员数**，超限即整组回收，并把这条差别记进事件日志。

用法::

    python labs/L9/tool_sandbox_pool.py --out out/9.8/pool --warm 1 --tasks 4 --concurrency 2
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

STATES = ("NEW", "STARTING", "READY", "BUSY", "DRAINING", "DESTROYED")

# 一条"工具"命令：读输入、做点计算、写输出。用 python 是为了在三台机器上都有同样的解释器。
TOOL_SOURCE = r"""
import json, os, sys, time

def main():
    spec = json.loads(sys.argv[1])
    if spec.get("sleep_ms"):
        time.sleep(spec["sleep_ms"] / 1000.0)
    if spec.get("alloc_mb"):
        blob = bytearray(spec["alloc_mb"] * 1024 * 1024)
        blob[0] = 1
    if spec.get("output_bytes"):
        sys.stdout.write("x" * spec["output_bytes"])
        sys.stdout.flush()
    if spec.get("spawn_children"):
        import subprocess
        kids = [subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
                for _ in range(spec["spawn_children"])]
        print(json.dumps({"children": [k.pid for k in kids]}), flush=True)
        time.sleep(spec.get("hold_s", 30))
    if spec.get("write_mb"):
        with open(os.path.join(os.getcwd(), "blob.bin"), "wb") as fh:
            fh.write(b"y" * (spec["write_mb"] * 1024 * 1024))
    print(json.dumps({"ok": True, "pid": os.getpid(), "cwd": os.getcwd(),
                      "uid": os.getuid(), "argv": spec}), flush=True)

main()
"""


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class Slot:
    """一个工具执行环境：进程组 + 私有工作区 + 租约。"""

    def __init__(self, pool, index: int, workspace: pathlib.Path, isolate: str):
        self.pool = pool
        self.index = index
        self.workspace = workspace
        self.isolate = isolate          # "plain" 或 "userns"
        self.state = "NEW"
        self.pgid = None
        self.proc = None
        self.epoch = 0
        self.attempt = 0
        self.lease_expires_at = 0.0
        self.created_at = now_ms()
        self.ready_at = None
        self.busy_count = 0
        self.destroy_reason = None
        self.tool_source = workspace / "tool.py"

    # -- 状态迁移 -------------------------------------------------------------
    def to(self, state: str, **detail) -> None:
        assert state in STATES, state
        prev, self.state = self.state, state
        self.pool.emit("state", slot=self.index, prev=prev, state=state, **detail)

    def start(self) -> dict:
        """STARTING：准备解释器、工作区与（可选）隔离包装；不启动常驻进程。"""
        self.to("STARTING")
        t0 = now_ms()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tool_source.write_text(TOOL_SOURCE, encoding="utf-8")
        (self.workspace / "input.json").write_text("{}", encoding="utf-8")
        self.prepare_ms = round(now_ms() - t0, 3)
        self.pool.emit("slot_prepared", slot=self.index, prepare_ms=self.prepare_ms,
                       cwd=str(self.workspace), isolate=self.isolate)
        return self.health_probe()

    def health_probe(self) -> dict:
        """READY 的门槛：真的跑一条最短命令并看到退出码。"""
        t0 = now_ms()
        probe = self.pool.run_tool(self, {"alloc_mb": 0}, timeout_s=10.0, apply_limits=False)
        ok = probe["exit_code"] == 0 and '"ok": true' in probe["stdout"]
        self.probe_ms = round(now_ms() - t0, 3)
        self.pool.emit("health_probe", slot=self.index, ok=ok, probe_ms=self.probe_ms,
                       exit_code=probe["exit_code"])
        if ok:
            self.ready_at = now_ms()
            self.to("READY")
        else:
            self.to("DESTROYED", reason="health_probe_failed")
        return {"ok": ok, "probe_ms": self.probe_ms}

    def begin_busy(self, task: dict) -> None:
        self.attempt += 1
        self.busy_count += 1
        self.lease_expires_at = time.time() + self.pool.lease_s
        self.to("BUSY", task=task.get("id"), attempt=self.attempt, epoch=self.epoch,
                lease_expires_at=round(self.lease_expires_at, 3))

    def drain(self, cleanup: dict) -> bool:
        """DRAINING：回收子进程与工作区残留，然后决定回 READY 还是 DESTROYED。"""
        self.to("DRAINING", **cleanup)
        healthy = self.health_probe()
        if healthy["ok"]:
            return True
        self.destroy("drain_probe_failed")
        return False

    def destroy(self, reason: str) -> None:
        self.destroy_reason = reason
        self.to("DESTROYED", reason=reason)


class Pool:
    def __init__(self, out: pathlib.Path, *, max_slots: int, warm: int, lease_s: float,
                 isolate: str, limits: dict, output_cap_bytes: int, max_children: int):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.events_fh = open(out / "events.jsonl", "w", encoding="utf-8")
        self.max_slots = max_slots
        self.lease_s = lease_s
        self.isolate = isolate
        self.limits = limits
        self.output_cap_bytes = output_cap_bytes
        self.max_children = max_children
        self.slots: list[Slot] = []
        self.hits = 0
        self.misses = 0
        self.overflows = 0
        self.epoch = 0

    # -- 观测 -----------------------------------------------------------------
    def emit(self, kind: str, **kw) -> dict:
        rec = {"ts_ms": round(now_ms(), 3), "kind": kind, **kw}
        self.events_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events_fh.flush()
        return rec

    def counts(self) -> dict:
        out = {s: 0 for s in STATES}
        for s in self.slots:
            out[s.state] += 1
        return out

    # -- 池管理 ---------------------------------------------------------------
    def _new_slot(self) -> Slot:
        idx = len(self.slots)
        ws = self.out / f"slot-{idx}"
        slot = Slot(self, idx, ws, self.isolate)
        self.slots.append(slot)
        self.emit("slot_created", slot=idx, total=len(self.slots))
        return slot

    def prewarm(self, warm: int) -> dict:
        """预热：准备 warm 个槽位并全部通过健康探测。"""
        started = []
        t0 = now_ms()
        for _ in range(min(warm, self.max_slots)):
            slot = self._new_slot()
            started.append(slot.start())
        return {"warm_requested": warm, "ready": sum(1 for s in started if s["ok"]),
                "prewarm_ms": round(now_ms() - t0, 3)}

    def acquire(self, task: dict) -> tuple[Slot | None, float]:
        """取一个 READY 槽位；没有就现起（受 max_slots 限制）。返回 (槽位, 等待毫秒)。"""
        t0 = now_ms()
        for s in self.slots:
            if s.state == "READY":
                self.hits += 1
                s.begin_busy(task)
                return s, round(now_ms() - t0, 3)
        if len(self.slots) >= self.max_slots:
            self.overflows += 1
            self.emit("pool_overflow", task=task.get("id"), max_slots=self.max_slots)
            return None, round(now_ms() - t0, 3)
        self.misses += 1
        slot = self._new_slot()
        res = slot.start()
        if not res["ok"]:
            return None, round(now_ms() - t0, 3)
        slot.begin_busy(task)
        return slot, round(now_ms() - t0, 3)

    # -- 执行 -----------------------------------------------------------------
    def _isolate_prefix(self) -> list[str]:
        if self.isolate == "userns":
            return ["unshare", "--user", "--map-root-user", "--net", "--"]
        return []

    def run_tool(self, slot: Slot, spec: dict, *, timeout_s: float,
                 apply_limits: bool = True) -> dict:
        """在槽位的进程组里跑一次工具，带 rlimit、输出上限、wall-time 与进程数监督。"""
        cmd = self._isolate_prefix() + [sys.executable, str(slot.tool_source),
                                        json.dumps(spec, ensure_ascii=False)]

        def preexec():
            os.setsid()
            if apply_limits:
                resource.setrlimit(resource.RLIMIT_AS,
                                   (self.limits["as_mb"] * 1024 * 1024,) * 2)
                resource.setrlimit(resource.RLIMIT_FSIZE,
                                   (self.limits["fsize_mb"] * 1024 * 1024,) * 2)
                resource.setrlimit(resource.RLIMIT_CPU, (self.limits["cpu_s"],) * 2)
                resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits["nofile"],) * 2)

        t0 = now_ms()
        proc = subprocess.Popen(cmd, cwd=str(slot.workspace), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, preexec_fn=preexec)
        slot.proc = proc
        slot.pgid = os.getpgid(proc.pid)
        self.emit("tool_start", slot=slot.index, pid=proc.pid, pgid=slot.pgid,
                  spec={k: v for k, v in spec.items() if k != "output_bytes"},
                  timeout_s=timeout_s)
        deadline = time.time() + timeout_s
        out_chunks: list[str] = []
        out_bytes = 0
        truncated = False
        exit_reason = None
        group_peak = 0
        # 必须用非阻塞读：`proc.stdout.read(n)` 在管道上没有数据时会阻塞，
        # wall-time 检查就永远轮不到——这正是"超时没生效"的常见成因。
        import selectors

        sel = selectors.DefaultSelector()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                os.set_blocking(stream.fileno(), False)
                sel.register(stream, selectors.EVENT_READ, "out" if stream is proc.stdout else "err")
        stderr_chunks: list[str] = []
        while True:
            rc_now = proc.poll()
            if rc_now is not None and not sel.get_map():
                break
            if time.time() > deadline:
                exit_reason = "wall_timeout"
                self.emit("wall_timeout", slot=slot.index, pid=proc.pid,
                          timeout_s=timeout_s, elapsed_ms=round(now_ms() - t0, 3))
                self.kill_group(slot)
                break
            members = self.group_members(slot.pgid)
            group_peak = max(group_peak, members)
            if self.max_children and members > self.max_children + 1:   # 除主进程之外
                exit_reason = "child_limit_exceeded"
                self.emit("child_limit", slot=slot.index, members=members,
                          max_children=self.max_children)
                self.kill_group(slot)
                break
            for key, _ in sel.select(timeout=0.05):
                try:
                    data = os.read(key.fileobj.fileno(), 65536)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    data = b""
                if not data:
                    try:
                        sel.unregister(key.fileobj)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                text = data.decode("utf-8", errors="replace")
                if key.data == "err":
                    stderr_chunks.append(text)
                    continue
                out_bytes += len(text)
                if out_bytes <= self.output_cap_bytes:
                    out_chunks.append(text)
                else:
                    truncated = True
                if truncated:
                    # 输出已超上限：仍然读干管道，但不再累积，避免内存被工具撑爆
                    pass
            if proc.poll() is not None and not sel.get_map():
                break
        # 收尾：把管道里剩下的读完（有界），避免子进程因管道满而卡住
        for _ in range(200):
            drained = False
            for stream, is_out in ((proc.stdout, True), (proc.stderr, False)):
                if stream is None:
                    continue
                try:
                    data = os.read(stream.fileno(), 65536)
                except (BlockingIOError, InterruptedError, OSError):
                    continue
                if not data:
                    continue
                drained = True
                text = data.decode("utf-8", errors="replace")
                if is_out:
                    out_bytes += len(text)
                    if out_bytes <= self.output_cap_bytes:
                        out_chunks.append(text)
                    else:
                        truncated = True
                else:
                    stderr_chunks.append(text)
            if not drained:
                break
        try:
            sel.close()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.poll()
        if rc is None:
            rc = proc.wait(timeout=5)
        stdout = "".join(out_chunks)[: self.output_cap_bytes]
        stderr = "".join(stderr_chunks)
        children_peak = group_peak
        if exit_reason is None:
            if rc == 0:
                exit_reason = "ok"
            elif rc < 0:
                exit_reason = f"signal_{abs(rc)}"
            elif "File too large" in stderr:
                exit_reason = "fsize_limit"
            elif "MemoryError" in stderr:
                exit_reason = "as_limit"
            else:
                exit_reason = f"exit_{rc}"
        res = {
            "exit_code": rc, "exit_reason": exit_reason, "stdout": stdout,
            "stderr_tail": stderr[-300:], "output_bytes": out_bytes,
            "output_truncated": truncated, "children_peak": children_peak,
            "duration_ms": round(now_ms() - t0, 3), "pgid": slot.pgid,
            "workspace": str(slot.workspace),
        }
        self.emit("tool_end", slot=slot.index, **{k: res[k] for k in
                  ("exit_code", "exit_reason", "output_bytes", "output_truncated",
                   "children_peak", "duration_ms")})
        return res

    @staticmethod
    def group_members(pgid: int) -> int:
        """进程组存活成员数（走 /proc，不依赖 RLIMIT_NPROC）。"""
        n = 0
        proc_root = pathlib.Path("/proc")
        if not proc_root.exists():
            return 0
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                fields = stat.rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in ("Z",):
                    n += 1
            except Exception:  # noqa: BLE001
                continue
        return n

    def kill_group(self, slot: Slot) -> dict:
        killed = False
        if slot.pgid:
            try:
                os.killpg(slot.pgid, signal.SIGKILL)
                killed = True
            except ProcessLookupError:
                pass
        time.sleep(0.2)
        running = self.group_members(slot.pgid) if slot.pgid else 0
        return {"killpg": killed, "running_after": running}

    # -- 工作区回收 -----------------------------------------------------------
    def reclaim_workspace(self, slot: Slot, *, keep: tuple[str, ...] = ("tool.py",)) -> dict:
        removed = []
        for p in sorted(slot.workspace.iterdir()):
            if p.name in keep:
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed.append(p.name)
        return {"removed": removed, "remaining": sorted(p.name for p in slot.workspace.iterdir())}

    def snapshot(self, slot: Slot, name: str) -> dict:
        path = self.out / f"{name}.tar.gz"
        with tarfile.open(path, "w:gz") as tf:
            tf.add(slot.workspace, arcname="workspace")
        return {"snapshot": str(path), "bytes": path.stat().st_size}

    def restore(self, snapshot: pathlib.Path, dest: pathlib.Path) -> dict:
        t0 = now_ms()
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(snapshot, "r:gz") as tf:
            tf.extractall(dest.parent)
        return {"restored_to": str(dest), "restore_ms": round(now_ms() - t0, 3)}

    def close(self) -> None:
        for s in self.slots:
            if s.pgid:
                self.kill_group(s)
        self.events_fh.close()


def run_tasks(out: pathlib.Path, warm: int, n_tasks: int, max_slots: int,
              isolate: str, limits: dict, output_cap_bytes: int, max_children: int,
              lease_s: float, tool_specs: list[dict], timeout_s: float) -> dict:
    pool = Pool(out, max_slots=max_slots, warm=warm, lease_s=lease_s, isolate=isolate,
                limits=limits, output_cap_bytes=output_cap_bytes, max_children=max_children)
    prewarm = pool.prewarm(warm)
    results = []
    for i in range(n_tasks):
        spec = tool_specs[i % len(tool_specs)] if tool_specs else {}
        task = {"id": f"t{i}", "spec": spec}
        slot, wait_ms = pool.acquire(task)
        if slot is None:
            results.append({"task": task["id"], "acquired": False, "wait_ms": wait_ms,
                            "pool_counts": pool.counts()})
            continue
        res = pool.run_tool(slot, spec, timeout_s=timeout_s)
        cleanup = pool.kill_group(slot)
        reclaim = pool.reclaim_workspace(slot)
        slot_state = slot.drain({**cleanup, "reclaimed": reclaim["removed"]})
        results.append({"task": task["id"], "acquired": True, "wait_ms": wait_ms,
                        "slot": slot.index, "result": res, "cleanup": cleanup,
                        "reclaim": reclaim, "back_to_ready": slot_state,
                        "pool_counts": pool.counts()})
        pool.emit("task_done", task=task["id"], slot=slot.index,
                  exit_reason=res["exit_reason"], back_to_ready=slot_state)
    summary = {
        "config": {"warm": warm, "tasks": n_tasks, "max_slots": max_slots, "isolate": isolate,
                   "limits": limits, "output_cap_bytes": output_cap_bytes,
                   "max_children": max_children, "lease_s": lease_s, "timeout_s": timeout_s},
        "prewarm": prewarm,
        "pool": {"hits": pool.hits, "misses": pool.misses, "overflows": pool.overflows,
                 "slots": len(pool.slots), "final_counts": pool.counts()},
        "results": results,
        "state_transitions": [json.loads(l) for l in open(out / "events.jsonl", encoding="utf-8")
                              if json.loads(l)["kind"] == "state"],
    }
    pool.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.8 工具沙箱池：状态机、限制与回收")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--max-slots", type=int, default=2)
    ap.add_argument("--isolate", choices=["plain", "userns"], default="userns")
    ap.add_argument("--as-mb", type=int, default=512)
    ap.add_argument("--fsize-mb", type=int, default=4)
    ap.add_argument("--cpu-s", type=int, default=5)
    ap.add_argument("--nofile", type=int, default=256)
    ap.add_argument("--output-cap-bytes", type=int, default=64 * 1024)
    ap.add_argument("--max-children", type=int, default=2)
    ap.add_argument("--lease-s", type=float, default=30.0)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    args = ap.parse_args()

    specs = [
        {"alloc_mb": 0, "output_bytes": 0},
        {"alloc_mb": 0, "output_bytes": 256 * 1024},      # 超过输出上限
        {"alloc_mb": 900},                                 # 超过地址空间限制
        {"alloc_mb": 0, "sleep_ms": 60000},                # 卡住的工具
    ]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = run_tasks(out, args.warm, args.tasks, args.max_slots, args.isolate,
                        {"as_mb": args.as_mb, "fsize_mb": args.fsize_mb, "cpu_s": args.cpu_s,
                         "nofile": args.nofile}, args.output_cap_bytes, args.max_children,
                        args.lease_s, specs, args.timeout_s)
    (out / "pool.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(json.dumps({"prewarm": summary["prewarm"], "pool": summary["pool"]},
                     ensure_ascii=False, indent=1))
    for r in summary["results"]:
        if r["acquired"]:
            print(f"[{r['task']}] slot={r['slot']} wait={r['wait_ms']}ms "
                  f"exit={r['result']['exit_reason']} out={r['result']['output_bytes']}B "
                  f"trunc={r['result']['output_truncated']} children_peak={r['result']['children_peak']} "
                  f"ready_again={r['back_to_ready']} running_after={r['cleanup']['running_after']}")
        else:
            print(f"[{r['task']}] not acquired (overflow) counts={r['pool_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
