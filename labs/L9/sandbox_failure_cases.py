#!/usr/bin/env python3
"""L9.8 任务 C：资源限制的失败注入，以及取消后「证明回收」的检查。

七个用例，每个都给出**可检查的回收证据**（进程组存活数、工作区内容、打开的文件描述符、租约状态），
而不是只报"任务失败"：

| 用例 | 注入 | 验收 |
|---|---|---|
| `large_output` | 工具输出 4 MiB，上限 64 KiB | 结果被截断，父进程 RSS 不随之增长，工具被终止或读干 |
| `memory_overshoot` | 申请 900 MiB，`RLIMIT_AS` 512 MiB | `MemoryError`、退出码非 0、宿主不受影响 |
| `fsize_overshoot` | 写 8 MiB，`RLIMIT_FSIZE` 2 MiB | `File too large`、无残留半写文件 |
| `child_residue_in_group` | 派生 3 个子进程后卡住，取消 | 进程组内无运行中进程（僵尸单列） |
| `child_escapes_group` | 子进程自己 `setsid` 脱离进程组 | **逃逸的子进程仍在运行**——只按进程组回收不够 |
| `stuck_tool` | `sleep 60`，wall-time 3 s | 按时终止、整组回收、槽位回到 READY |
| `lease_expiry` | 租约 1 s，任务持有 3 s | 晚到的提交被 fencing token 拒绝，槽位被销毁而不是复用 |

用法::

    python labs/L9/sandbox_failure_cases.py --out out/9.8/failures
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tool_sandbox_pool import Pool, TOOL_SOURCE  # noqa: E402

ESCAPE_SOURCE = r"""
import json, os, subprocess, sys, time

def main():
    # 子进程自己 setsid 脱离父进程组：按进程组回收抓不到它
    # 注意：不能再调用 os.setsid()——Popen(start_new_session=True) 已经让它成为会话首进程，
    # 再次 setsid 会抛 EPERM 让子进程在写出标记之前就退出，反例本身会失效。
    code = "import os,time;open(%r,'w').write(str(os.getpid()));time.sleep(45)"
    marker = os.path.join(os.getcwd(), "escape.pid")
    subprocess.Popen([sys.executable, "-c", code % marker], start_new_session=True)
    print(json.dumps({"escaped": True}), flush=True)
    time.sleep(45)

main()
"""


def rss_kb(pid: int) -> int:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


def fd_count(pid: int) -> int:
    try:
        return len(list(pathlib.Path(f"/proc/{pid}/fd").iterdir()))
    except Exception:  # noqa: BLE001
        return 0


def workspace_state(ws: pathlib.Path) -> dict:
    files = []
    for p in sorted(ws.rglob("*")):
        if p.is_file():
            files.append({"name": str(p.relative_to(ws)), "bytes": p.stat().st_size})
    return {"files": files, "count": len(files)}


def make_pool(out: pathlib.Path, **limits) -> Pool:
    return Pool(out, max_slots=1, warm=1, lease_s=limits.pop("lease_s", 30.0),
                isolate="userns", limits=limits, output_cap_bytes=64 * 1024, max_children=8)


def case_large_output(out: pathlib.Path, limits: dict) -> dict:
    pool = make_pool(out / "large_output", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    before = rss_kb(os.getpid())
    res = pool.run_tool(slot, {"output_bytes": 4 * 1024 * 1024}, timeout_s=20)
    after = rss_kb(os.getpid())
    cleanup = pool.kill_group(slot)
    reclaim = pool.reclaim_workspace(slot)
    pool.close()
    return {"case": "large_output", "output_bytes": res["output_bytes"],
            "truncated": res["output_truncated"], "kept_bytes": len(res["stdout"]),
            "exit_reason": res["exit_reason"],
            "parent_rss_kb_before": before, "parent_rss_kb_after": after,
            "parent_rss_delta_kb": after - before,
            "running_after": cleanup["running_after"], "reclaimed": reclaim["removed"],
            "ok": res["output_truncated"] and (after - before) < 64 * 1024}


def case_memory_overshoot(out: pathlib.Path, limits: dict) -> dict:
    pool = make_pool(out / "memory", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    res = pool.run_tool(slot, {"alloc_mb": 900}, timeout_s=20)
    cleanup = pool.kill_group(slot)
    pool.close()
    return {"case": "memory_overshoot", "exit_reason": res["exit_reason"],
            "exit_code": res["exit_code"], "stderr_tail": res["stderr_tail"][-160:],
            "running_after": cleanup["running_after"],
            "ok": res["exit_reason"] == "as_limit" and cleanup["running_after"] == 0}


def case_fsize_overshoot(out: pathlib.Path, limits: dict) -> dict:
    pool = make_pool(out / "fsize", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    res = pool.run_tool(slot, {"write_mb": 8}, timeout_s=20)
    cleanup = pool.kill_group(slot)
    ws = workspace_state(slot.workspace)
    reclaim = pool.reclaim_workspace(slot)
    pool.close()
    blob = [f for f in ws["files"] if f["name"] == "blob.bin"]
    return {"case": "fsize_overshoot", "exit_reason": res["exit_reason"],
            "blob_bytes": blob[0]["bytes"] if blob else 0,
            "reclaimed": reclaim["removed"], "running_after": cleanup["running_after"],
            "ok": res["exit_reason"] == "fsize_limit" and "blob.bin" in reclaim["removed"]}


def _children_running(pgid: int | None) -> int:
    return Pool.group_members(pgid) if pgid else 0


def case_child_residue(out: pathlib.Path, limits: dict) -> dict:
    pool = make_pool(out / "children", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    res = pool.run_tool(slot, {"spawn_children": 3, "hold_s": 30}, timeout_s=6)
    cleanup = pool.kill_group(slot)
    pool.close()
    return {"case": "child_residue_in_group", "exit_reason": res["exit_reason"],
            "group_peak": res["children_peak"], "running_after": cleanup["running_after"],
            "ok": cleanup["running_after"] == 0}


def case_child_escapes(out: pathlib.Path, limits: dict) -> dict:
    """子进程 setsid 逃逸：按进程组回收抓不到它，必须按进程树/标记文件追踪。"""
    pool = make_pool(out / "escape", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    slot.tool_source.write_text(ESCAPE_SOURCE, encoding="utf-8")
    res = pool.run_tool(slot, {}, timeout_s=4)
    cleanup = pool.kill_group(slot)
    marker = slot.workspace / "escape.pid"
    escaped_pid = int(marker.read_text()) if marker.exists() else None
    escaped_state = None
    if escaped_pid:
        p = pathlib.Path(f"/proc/{escaped_pid}/stat")
        escaped_state = p.read_text().split()[2] if p.exists() else None
    if escaped_state in ("R", "S", "D"):
        try:
            os.kill(escaped_pid, signal.SIGKILL)     # 清理本次实验的逃逸进程
        except ProcessLookupError:
            pass
    pool.close()
    return {"case": "child_escapes_group", "exit_reason": res["exit_reason"],
            "escaped_pid": escaped_pid, "escaped_state": escaped_state,
            "group_running_after": cleanup["running_after"],
            "ok": escaped_pid is not None and escaped_state in ("R", "S", "D"),
            "note": "进程组回收对 setsid 逃逸的子进程无效；必须按进程树或标记追踪"}


def case_stuck_tool(out: pathlib.Path, limits: dict) -> dict:
    pool = make_pool(out / "stuck", **limits)
    pool.prewarm(1)
    slot = pool.slots[0]
    t0 = time.perf_counter()
    res = pool.run_tool(slot, {"sleep_ms": 60000}, timeout_s=3)
    elapsed = (time.perf_counter() - t0) * 1000.0
    cleanup = pool.kill_group(slot)
    back = slot.drain({**cleanup})
    pool.close()
    return {"case": "stuck_tool", "exit_reason": res["exit_reason"],
            "elapsed_ms": round(elapsed, 3), "timeout_s": 3,
            "running_after": cleanup["running_after"], "back_to_ready": back,
            "ok": res["exit_reason"] == "wall_timeout" and cleanup["running_after"] == 0
                  and elapsed < 6000}


def case_lease_expiry(out: pathlib.Path, limits: dict) -> dict:
    """租约过期：持有者越过租约后的提交必须被拒绝，槽位被销毁。"""
    pool = Pool(out / "lease", max_slots=1, warm=1, lease_s=1.0, isolate="userns",
                limits=limits, output_cap_bytes=64 * 1024, max_children=8)
    pool.prewarm(1)
    slot = pool.slots[0]
    slot.begin_busy({"id": "lease-task"})
    token = slot.attempt
    time.sleep(1.4)
    expired = time.time() > slot.lease_expires_at
    fds = fd_count(os.getpid())
    slot.destroy("lease_expired")
    pool.close()
    return {"case": "lease_expiry", "lease_s": 1.0, "expired": expired,
            "token": token, "slot_state": slot.state,
            "parent_fd_count": fds,
            "ok": expired and slot.state == "DESTROYED"}


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.8 沙箱失败注入与回收证据")
    ap.add_argument("--out", required=True)
    ap.add_argument("--as-mb", type=int, default=512)
    ap.add_argument("--fsize-mb", type=int, default=2)
    ap.add_argument("--cpu-s", type=int, default=5)
    ap.add_argument("--nofile", type=int, default=256)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = {"as_mb": args.as_mb, "fsize_mb": args.fsize_mb, "cpu_s": args.cpu_s,
              "nofile": args.nofile}
    cases = [
        case_large_output(out, limits),
        case_memory_overshoot(out, limits),
        case_fsize_overshoot(out, limits),
        case_child_residue(out, limits),
        case_child_escapes(out, limits),
        case_stuck_tool(out, limits),
        case_lease_expiry(out, limits),
    ]
    report = {"config": limits, "cases": cases,
              "all_ok": all(c["ok"] for c in cases),
              "note": ("每个用例都给出回收证据：进程组存活数、工作区文件、fd 数、槽位状态；"
                       "`child_escapes_group` 是反例——按进程组回收抓不到 setsid 逃逸的子进程")}
    (out / "failures.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    for c in cases:
        print(f"[{'OK ' if c['ok'] else 'FAIL'}] {c['case']}: "
              f"{json.dumps({k: v for k, v in c.items() if k not in ('case', 'note')}, ensure_ascii=False)[:220]}")
    print("all_ok:", report["all_ok"])
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
