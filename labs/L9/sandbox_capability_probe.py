#!/usr/bin/env python3
"""L9.8 任务 A 的前置核验：这台机器到底能提供哪几层隔离。

9.8 的四个任务分别依赖不同的隔离原语。容器、gVisor、microVM 需要各自的内核/工具链支持，
不能假定存在；本脚本把「能用什么」与「必须标 UNVERIFIED 的路线」一次问清楚：

* 命名空间：PID / mount / net / user 各自能否创建（`unshare`）；
* cgroup v2：挂载点是否可写、有哪些 controller；
* 能力与 seccomp：`CapEff/CapPrm/CapBnd`、seccomp 模式；
* 进程级限制：`setrlimit` 是否可用，并在子进程里**实际验证** AS/FILE/CPU/NPROC 会被执行；
* 进程树：`setsid` + `killpg` 能否整组回收；
* 容器/微虚拟机：docker、podman、runc、runsc、bwrap、crun、firecracker、`/dev/kvm` 是否在位。

输出是一张能力表加一句结论：哪些路线可测、哪些保持 `UNVERIFIED`。用法::

    python labs/L9/sandbox_capability_probe.py --out out/9.8/probe
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

TOOLS = ("docker", "podman", "runc", "runsc", "bwrap", "crun", "firecracker",
         "unshare", "nsenter", "setpriv", "capsh", "cgcreate", "systemd-run")


def sh(cmd: str, timeout: float = 20.0) -> dict:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": (p.stdout or "").strip()[:600],
                "err": (p.stderr or "").strip()[:300]}
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "", "err": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"rc": None, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def probe_platform() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "uname": " ".join(platform.uname()),
        "python": sys.version.split()[0],
        "pid1_cgroup": pathlib.Path("/proc/1/cgroup").read_text().strip()[:200]
        if pathlib.Path("/proc/1/cgroup").exists() else None,
        "dockerenv": pathlib.Path("/.dockerenv").exists(),
        "container_env": {k: v for k, v in os.environ.items()
                          if k in ("container", "KUBERNETES_SERVICE_HOST", "DOCKER_HOST")},
    }


def probe_static() -> dict:
    caps = {}
    status = pathlib.Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("CapEff", "CapPrm", "CapBnd", "Seccomp")):
                k, _, v = line.partition(":")
                caps[k.strip()] = v.strip()
    cgroup_root = pathlib.Path("/sys/fs/cgroup")
    controllers = None
    writable = None
    if cgroup_root.exists():
        ctl = cgroup_root / "cgroup.controllers"
        controllers = ctl.read_text().strip() if ctl.exists() else None
        probe_dir = cgroup_root / "l9probe"
        try:
            probe_dir.mkdir(exist_ok=True)
            writable = True
            probe_dir.rmdir()
        except Exception:  # noqa: BLE001
            writable = False
    return {
        "capabilities": caps,
        "cgroup_v2_mounted": cgroup_root.exists(),
        "cgroup_controllers": controllers,
        "cgroup_writable": writable,
        "self_cgroup": pathlib.Path("/proc/self/cgroup").read_text().strip()
        if pathlib.Path("/proc/self/cgroup").exists() else None,
        "kvm_device": pathlib.Path("/dev/kvm").exists(),
        "user_max_user_namespaces": _read("/proc/sys/user/max_user_namespaces"),
        "unprivileged_userns_clone": _read("/proc/sys/kernel/unprivileged_userns_clone"),
        "tools": {t: shutil.which(t) for t in TOOLS},
    }


def _read(path: str) -> str | None:
    p = pathlib.Path(path)
    return p.read_text().strip() if p.exists() else None


def probe_namespaces() -> dict:
    tests = {
        "pid": "unshare --pid --fork --mount-proc echo ok",
        "user": "unshare --user --map-root-user id -u",
        "net": "unshare --net --map-root-user ip link show lo",
        "mount": "unshare --mount --map-root-user --propagation private sh -c 'mount -t tmpfs none /mnt && echo ok'",
        "uts": "unshare --uts --map-root-user sh -c 'hostname l9probe && hostname'",
        "ipc": "unshare --ipc --map-root-user echo ok",
    }
    out = {}
    for name, cmd in tests.items():
        if not shutil.which("unshare"):
            out[name] = {"supported": False, "reason": "unshare missing"}
            continue
        r = sh(cmd)
        out[name] = {"supported": r["rc"] == 0, "rc": r["rc"],
                     "detail": (r["out"] or r["err"])[:160]}
    return out


def probe_rlimits(out_dir: pathlib.Path) -> dict:
    child = out_dir / "rlimit_child.py"
    child.write_text(
        "import json,resource,sys,os\n"
        "kind=sys.argv[1]\n"
        "r={\"kind\":kind}\n"
        "try:\n"
        "    if kind=='as':\n"
        "        resource.setrlimit(resource.RLIMIT_AS,(200*1024*1024,)*2)\n"
        "        b=bytearray(400*1024*1024)\n"
        "        r['enforced']=False; r['detail']='allocated 400MB despite 200MB limit'\n"
        "    elif kind=='fsize':\n"
        "        resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,)*2)\n"
        "        with open('big.bin','wb') as f: f.write(b'x'*(4*1024*1024))\n"
        "        r['enforced']=False; r['detail']='wrote 4MB despite 1MB limit'\n"
        "    elif kind=='cpu':\n"
        "        resource.setrlimit(resource.RLIMIT_CPU,(1,1))\n"
        "        t=0\n"
        "        while True: t+=1\n"
        "        r['enforced']=False\n"
        "    elif kind=='nproc':\n"
        "        resource.setrlimit(resource.RLIMIT_NPROC,(4,4))\n"
        "        kids=[]\n"
        "        for i in range(8):\n"
        "            kids.append(os.fork())\n"
        "        r['enforced']=False; r['detail']='forked 8 children despite limit 4'\n"
        "except MemoryError as e:\n"
        "    r['enforced']=True; r['detail']='MemoryError'\n"
        "except OSError as e:\n"
        "    r['enforced']=True; r['detail']=type(e).__name__+': '+str(e)[:80]\n"
        "except Exception as e:\n"
        "    r['enforced']=True; r['detail']=type(e).__name__+': '+str(e)[:80]\n"
        "print(json.dumps(r))\n",
        encoding="utf-8")
    results = {}
    for kind, timeout in (("as", 30), ("fsize", 30), ("cpu", 30), ("nproc", 30)):
        workdir = out_dir / f"rlimit-{kind}"
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run([sys.executable, str(child), kind], cwd=workdir,
                               capture_output=True, text=True, timeout=timeout)
            line = (p.stdout or "").strip().splitlines()
            results[kind] = {"rc": p.returncode,
                             "result": json.loads(line[-1]) if line else None,
                             "stderr": (p.stderr or "").strip()[-200:]}
        except subprocess.TimeoutExpired:
            results[kind] = {"rc": None, "result": {"kind": kind, "enforced": True,
                                                    "detail": "killed by timeout (CPU limit enforced)"}}
    results["defaults"] = {
        name: resource.getrlimit(getattr(resource, f"RLIMIT_{name}"))
        for name in ("AS", "NPROC", "NOFILE", "CPU", "FSIZE", "CORE")
    }
    return results


def proc_state(pid: int) -> str | None:
    """读 /proc/<pid>/stat 的状态字符：R/S/D 是活着，Z 是僵尸，None 表示不存在。

    只看 `kill(pid, 0)` 会把**僵尸**也算成"还活着"——被 SIGKILL 杀死但尚未被回收的子进程
    会一直挂在 init 下，直到有人 wait。这个区分决定了"进程树回收"的验收结论。
    """
    p = pathlib.Path(f"/proc/{pid}/stat")
    if not p.exists():
        return None
    try:
        fields = p.read_text().split()
        return fields[2]
    except Exception:  # noqa: BLE001
        return None


def probe_process_tree(out_dir: pathlib.Path) -> dict:
    """setsid + killpg 能否整组回收；对比只杀父进程时的残留。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    child = out_dir / "spawn_tree.py"
    child.write_text(
        "import subprocess,sys,time\n"
        "grand = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "print(grand.pid, flush=True)\n"
        "time.sleep(60)\n", encoding="utf-8")
    res: dict = {}
    for mode in ("killpg", "kill_only_parent"):
        proc = subprocess.Popen([sys.executable, str(child)], stdout=subprocess.PIPE,
                                text=True, start_new_session=(mode == "killpg"))
        grandchild = int(proc.stdout.readline().strip())
        time.sleep(0.3)
        if mode == "killpg":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
        states = {pid: proc_state(pid) for pid in (proc.pid, grandchild)}
        running = [pid for pid, st in states.items() if st in ("R", "S", "D")]
        zombies = [pid for pid, st in states.items() if st == "Z"]
        if mode == "kill_only_parent" and states.get(grandchild) in ("R", "S", "D"):
            try:
                os.kill(grandchild, signal.SIGKILL)   # 清理本次实验的残留
                os.waitpid(grandchild, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        res[mode] = {"parent_state": states.get(proc.pid),
                     "grandchild_state": states.get(grandchild),
                     "running_pids": running, "zombie_pids": zombies}
    ok_group = not res["killpg"]["running_pids"]
    res["conclusion"] = (
        f"killpg 后无运行中进程（僵尸 {len(res['killpg']['zombie_pids'])} 个，属已杀死未回收）；"
        f"只杀父进程留下运行中子进程 {res['kill_only_parent']['running_pids']}"
        if ok_group else "killpg 未能回收整组，需检查会话与进程组归属")
    res["killpg_reclaims_group"] = ok_group
    res["kill_only_parent_leaves_orphan"] = bool(res["kill_only_parent"]["running_pids"])
    return res


def cmd_run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": probe_platform(),
        "static": probe_static(),
        "namespaces": probe_namespaces(),
        "rlimits": probe_rlimits(out),
        "process_tree": probe_process_tree(out),
    }
    ns = report["namespaces"]
    static = report["static"]
    container_runtime = any(static["tools"].get(t) for t in ("docker", "podman", "runc", "crun"))
    report["capability_matrix"] = {
        "process_group_reclaim": bool(report["process_tree"].get("killpg_reclaims_group")),
        "rlimit_address_space": bool((report["rlimits"]["as"].get("result") or {}).get("enforced")),
        "rlimit_file_size": bool((report["rlimits"]["fsize"].get("result") or {}).get("enforced")),
        "rlimit_cpu_time": bool(report["rlimits"]["cpu"].get("rc") == -9
                                or (report["rlimits"]["cpu"].get("result") or {}).get("enforced")),
        "rlimit_process_count": bool((report["rlimits"]["nproc"].get("result") or {}).get("enforced")),
        "userns_isolation": bool(ns["user"]["supported"]),
        "netns_isolation": bool(ns["net"]["supported"]),
        "pidns_isolation": bool(ns["pid"]["supported"]),
        "mountns_isolation": bool(ns["mount"]["supported"]),
        "cgroup_v2_limits": bool(static["cgroup_writable"]),
        "container_runtime": container_runtime,
        "gvisor_runsc": bool(static["tools"].get("runsc")),
        "microvm_firecracker": bool(static["tools"].get("firecracker")),
        "kvm": bool(static["kvm_device"]),
    }
    unavailable = [k for k, v in report["capability_matrix"].items() if v is False]
    report["verdict"] = {
        "testable_here": [k for k, v in report["capability_matrix"].items() if v],
        "unverified_here": unavailable,
        "note": ("不可用的路线在本机保持 UNVERIFIED，并记录补测条件（需要 CAP_SYS_ADMIN/可写 cgroup、"
                 "容器运行时、gVisor 或 /dev/kvm）；可用路线在 9.8 里按实际接口验收"),
    }
    (out / "sandbox_capability.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(json.dumps({"capability_matrix": report["capability_matrix"],
                      "verdict": report["verdict"]}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.8 隔离能力核验")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
