#!/usr/bin/env python3
"""
6.0 / 6.1 跨机实验的启动器（在本地 macOS 上运行，只用标准库）。

crater 与 crater2 在同一 /24 网段上互通、各有一张空闲 GPU，因此可以把 6.0/6.1
任务书里「跨机 TCP transport」那一档真正跑起来，而不是继续留在 UNVERIFIED。
两台机器都没有 RDMA（无 /dev/infiniband），所以走的就是 NCCL 的 NET/Socket 路径。

本脚本负责：
  1. 把 labs/L6 的源文件推到两台机器（tar over ssh，crater2 没有 rsync）；
  2. 在两端各自发现同一 /24 上的真实网卡名，交给 NCCL_SOCKET_IFNAME；
  3. 并行启动 rank0（crater）与 rank1（crater2），收集退出码与 stdout；
  4. 把两侧原始工件回收到 results/<机器>/<章节>/<run_id>/；
  5. 在本地做多 rank 归并与 NCCL 算法选择解析。

用法：
    /Volumes/data/venvs/llm-infra/bin/python labs/L6/run_crosshost.py \
        sweep --run-id 20260922-crosshost --sizes 4096,262144,16777216 \
        --ops all_reduce,all_gather
    /Volumes/data/venvs/llm-infra/bin/python labs/L6/run_crosshost.py p2p \
        --run-id 20260922-p2p --bytes 8388608 --iters 20
    /Volumes/data/venvs/llm-infra/bin/python labs/L6/run_crosshost.py completion \
        --run-id 20260922-completion --bytes 67108864 --trials 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_LABS = os.path.join(ROOT, "labs", "L6")
HOSTS = ["crater", "crater2"]
REMOTE_ROOT = "/scratch/learn"
REMOTE_LABS = f"{REMOTE_ROOT}/work/labs/L6"
REMOTE_PY = f"{REMOTE_ROOT}/envs/serve/bin/python"
RECV_DIR = os.path.join(ROOT, "results")

# 任务 → 章节目录（跨机完成语义按 STATUS 的归属记在 6.1）
TASK_CHAPTER = {"sweep": "6.1", "completion": "6.1", "p2p": "6.0",
                "socket": "6.1"}

ALGO_RE = re.compile(
    r"NCCL INFO (\w+): (\d+) Bytes -> Algo (\w+) proto (\w+) "
    r"channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")
OP_MAP = {"allreduce": "all_reduce", "allgather": "all_gather",
          "reducescatter": "reduce_scatter", "broadcast": "broadcast",
          "send": "sendrecv", "recv": "sendrecv"}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def ssh(host, script, timeout=None):
    return subprocess.run(["ssh", host, script], text=True, capture_output=True,
                          timeout=timeout)


def net_probe(host):
    """返回该机器上参与 /24 互连的 (ifname, ip)。"""
    script = ("ip -4 -o addr show | awk '$2!=\"lo\"{print $2, $4}'")
    p = ssh(host, script, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"{host} 网卡枚举失败: {p.stderr.strip()}")
    cands = []
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            cands.append((parts[0], parts[1].split("/")[0]))
    return cands


CONTAINER_IF_PREFIXES = ("eth0", "cali", "docker", "veth", "flannel", "cni", "br-")


def _if_rank(name):
    """排序用：容器/overlay 网卡排在真实主机网卡之后。"""
    return (any(name.startswith(p) for p in CONTAINER_IF_PREFIXES), name)


def pick_link(probes, ifname=None):
    """挑出两台机器上属于同一 /24 的那对网卡。

    两台机器上都有两个候选：容器 overlay（eth0, 10.233.x）与主机网卡
    （net1, 192.168.x）。默认优先真实主机网卡，可用 --ifname 显式指定，
    两条链路都会被记录，便于对照。
    """
    hits = []
    for i in range(len(HOSTS)):
        for j in range(i + 1, len(HOSTS)):
            for name_i, ip_i in probes[HOSTS[i]]:
                for name_j, ip_j in probes[HOSTS[j]]:
                    if ip_i.split(".")[:3] != ip_j.split(".")[:3]:
                        continue
                    if ifname and not (name_i == ifname and name_j == ifname):
                        continue
                    hits.append((HOSTS[i], name_i, ip_i, HOSTS[j], name_j, ip_j))
    if not hits:
        raise RuntimeError(f"两台机器没有可用共同 /24（ifname={ifname}）: {probes}")
    hits.sort(key=lambda h: (h[0] != HOSTS[0], h[3] != HOSTS[0],
                             _if_rank(h[1]), _if_rank(h[4])))
    return hits[0], hits


def push_labs():
    files = [f for f in sorted(os.listdir(LOCAL_LABS))
             if f.endswith((".py", ".sh"))]
    names = " ".join(shlex.quote(f) for f in files)
    for host in HOSTS:
        cmd = (f"tar -cf - -C {shlex.quote(LOCAL_LABS)} {names} | "
               f"ssh {host} 'mkdir -p {REMOTE_LABS} && tar -xf - -C {REMOTE_LABS}'")
        p = sh(cmd, timeout=300)
        if p.returncode != 0:
            raise RuntimeError(f"推送 labs 到 {host} 失败: {p.stderr.strip()}")
    print(f"[push] {len(files)} 个文件 → {HOSTS}", flush=True)


def build_remote_cmd(rank, host, task, args, master_addr, master_port,
                     ifname, out_dir, nccl_debug):
    inner = [
        f"source {REMOTE_ROOT}/env.sh >/dev/null 2>&1",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "export CUDA_VISIBLE_DEVICES=0",
        f"export NCCL_SOCKET_IFNAME={shlex.quote(ifname)}",
        "export NCCL_IB_DISABLE=1",
        "export NCCL_DEBUG=INFO",
        "export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,TUNING",
        f"export NCCL_DEBUG_FILE={shlex.quote(nccl_debug)}",
        f"mkdir -p {shlex.quote(out_dir)}",
        f"cd {REMOTE_LABS}",
        " ".join([
            REMOTE_PY, "crosshost_comm.py", task,
            "--rank", str(rank),
            "--world-size", str(len(HOSTS)),
            "--master-addr", shlex.quote(master_addr),
            "--master-port", str(master_port),
            "--out", shlex.quote(out_dir),
        ] + args),
    ]
    return "; ".join(inner)


def run_task(task, run_id, args, master_port=None, timeout=5400, ifname=None):
    probes = {h: net_probe(h) for h in HOSTS}
    link, hits = pick_link(probes, ifname)
    a_host, a_if, a_ip, b_host, b_if, b_ip = link
    master_addr = a_ip
    if master_port is None:
        master_port = 29921
    chapter = TASK_CHAPTER[task]
    print(f"[net] {a_host}({a_ip},{a_if}) ↔ {b_host}({b_ip},{b_if}) "
          f"master={master_addr}:{master_port}", flush=True)
    all_links = [{"rank0_host": h[0], "rank0_if": h[1], "rank0_ip": h[2],
                  "rank1_host": h[3], "rank1_if": h[4], "rank1_ip": h[5],
                  "used": (h == link)} for h in hits]

    procs, outs = {}, {}
    for rank, host, ifname in ((0, a_host, a_if), (1, b_host, b_if)):
        remote_out = f"{REMOTE_ROOT}/work/out/{chapter}/{run_id}"
        nccl_debug = f"{remote_out}/nccl_debug_%h.log"
        cmd = build_remote_cmd(rank, host, task, args, master_addr, master_port,
                               ifname, remote_out, nccl_debug)
        local_log = open(f"/Volumes/data/artifacts/llm-infra/scratch/"
                         f"crosshost_{task}_{run_id}_rank{rank}.log", "w",
                         encoding="utf-8")
        outs[rank] = local_log
        procs[rank] = subprocess.Popen(["ssh", host, cmd], stdout=local_log,
                                       stderr=subprocess.STDOUT, text=True)
        print(f"[launch] rank{rank} on {host}", flush=True)

    rc = {}
    for rank, p in procs.items():
        try:
            rc[rank] = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc[rank] = "timeout"
    for f in outs.values():
        f.close()
    print(f"[exit] {rc}", flush=True)

    # 回收：两台机器各自的原始工件进各自的 results/<机器>/ 目录
    for rank, host in ((0, a_host), (1, b_host)):
        dest = os.path.join(RECV_DIR, host, chapter)
        os.makedirs(dest, exist_ok=True)
        remote_out = f"{REMOTE_ROOT}/work/out/{chapter}"
        cmd = (f"ssh {host} 'tar -cf - -C {shlex.quote(remote_out)} "
               f"{shlex.quote(run_id)}' | tar -xf - -C {shlex.quote(dest)}")
        p = sh(cmd, timeout=1800)
        if p.returncode != 0:
            print(f"[collect:{host}] 失败: {p.stderr.strip()}", flush=True)
        else:
            print(f"[collect:{host}] → {dest}/{run_id}", flush=True)
        # 本机启动日志一并留档，便于看真实退出过程
        src = f"/Volumes/data/artifacts/llm-infra/scratch/" \
              f"crosshost_{task}_{run_id}_rank{rank}.log"
        if os.path.exists(src):
            with open(src, encoding="utf-8", errors="replace") as fin, \
                    open(os.path.join(dest, run_id, f"launch.rank{rank}.log"),
                         "w", encoding="utf-8") as fout:
                fout.write(fin.read())

    write_run_config(task, run_id, chapter, a_host, b_host, master_addr,
                     master_port, args, rc, all_links)
    return merge(task, run_id, chapter, a_host, b_host)


def write_run_config(task, run_id, chapter, a_host, b_host, master_addr,
                     master_port, args, rc, all_links=None):
    cfg = {
        "task": task,
        "run_id": run_id,
        "chapter": chapter,
        "hosts": {"rank0": a_host, "rank1": b_host},
        "master_addr": master_addr,
        "master_port": master_port,
        "transport": "NCCL over TCP sockets (NET/Socket)；两机均无 /dev/infiniband",
        "launcher_args": args,
        "remote_python": REMOTE_PY,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank_exit_codes": {str(k): v for k, v in rc.items()},
        "launcher": "labs/L6/run_crosshost.py",
        "worker": "labs/L6/crosshost_comm.py",
        "candidate_links": all_links or [],
    }
    dest = os.path.join(RECV_DIR, a_host, chapter, run_id)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_algo_log(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ALGO_RE.search(line)
            if not m:
                continue
            name, nbytes, algo, proto, lo, hi = m.groups()
            op = OP_MAP.get(name.lower())
            if not op:
                continue
            out.setdefault(op, {})[int(nbytes)] = {
                "algo": algo, "proto": proto, "channels": int(hi) - int(lo) + 1}
    return out


def busbw_factor(op, nranks):
    if op == "all_reduce":
        return 2 * (nranks - 1) / nranks
    if op in ("all_gather", "reduce_scatter"):
        return (nranks - 1) / nranks
    if op in ("broadcast", "sendrecv"):
        return 1.0
    raise ValueError(op)


def _rank_dirs(chapter, run_id, a_host, b_host):
    return [os.path.join(RECV_DIR, h, chapter, run_id) for h in (a_host, b_host)]


def merge_sweep(chapter, run_id, a_host, b_host):
    dirs = _rank_dirs(chapter, run_id, a_host, b_host)
    per_rank = {}
    algo = {}
    for rank, d in enumerate(dirs):
        path = os.path.join(d, f"sweep.rank{rank}.json")
        per_rank[rank] = load(path)["points"] if os.path.exists(path) else []
        for name in os.listdir(d):
            if name.startswith("nccl_debug"):
                for op, dd in parse_algo_log(os.path.join(d, name)).items():
                    algo.setdefault(op, {}).update(dd)
    nranks = len(dirs)
    merged = []
    for i, p0 in enumerate(per_rank[0]):
        op, nb = p0["op"], p0["per_rank_bytes"]
        meds, err = [], None
        for r in range(nranks):
            q = per_rank[r][i] if i < len(per_rank[r]) else {"error": "missing"}
            if "error" in q:
                err = q["error"]
            else:
                meds.append(q["t_s_median"])
        if err:
            merged.append({"op": op, "per_rank_bytes": nb, "error": err})
            continue
        t_max = max(meds)
        merged.append({
            "op": op,
            "per_rank_bytes": nb,
            "t_s_per_rank": meds,
            "t_s_min_rank": min(meds),
            "t_s_max_rank": t_max,
            "t_s_spread": (max(meds) - min(meds)) / statistics.median(meds),
            "algbw_GBps_max_rank": nb / t_max / 1e9,
            "busbw_GBps_max_rank": busbw_factor(op, nranks) * nb / t_max / 1e9,
            "selected": algo.get(op, {}).get(nb),
        })
    payload = {
        "kind": "crosshost_collective_sweep",
        "chapter": chapter,
        "run_id": run_id,
        "hosts": {"rank0": a_host, "rank1": b_host},
        "world_size": nranks,
        "definition": "M = 每 rank 输入字节；algbw = M/t；busbw = M*系数/t，"
                      "系数见 collective_costs.busbw_factor",
        "transport": "NET/Socket（两机均无 RDMA 设备）",
        "algo_selection": {op: {str(k): v for k, v in sorted(d.items())}
                           for op, d in algo.items()},
        "points": merged,
    }
    dest = os.path.join(RECV_DIR, a_host, chapter, run_id, "crosshost_sweep.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[merge] → {dest}", flush=True)
    for pt in merged:
        if "error" in pt:
            print(f"   {pt['op']:<15} {pt['per_rank_bytes']:>10} ERROR {pt['error'][:60]}")
        else:
            sel = pt["selected"] or {}
            print(f"   {pt['op']:<15} {pt['per_rank_bytes']:>10} "
                  f"t={pt['t_s_max_rank'] * 1e3:9.3f} ms  "
                  f"algbw={pt['algbw_GBps_max_rank']:6.3f}  "
                  f"busbw={pt['busbw_GBps_max_rank']:6.3f} GB/s  "
                  f"{sel.get('algo', '-'):<5}/{sel.get('proto', '-'):<6} "
                  f"ch={sel.get('channels', '-')}")
    return payload


def merge_p2p(chapter, run_id, a_host, b_host):
    dirs = _rank_dirs(chapter, run_id, a_host, b_host)
    ranks = []
    for rank, d in enumerate(dirs):
        path = os.path.join(d, f"p2p_batching.rank{rank}.json")
        if os.path.exists(path):
            ranks.append(load(path))
    api = ranks[0]["api"] if ranks else {}
    variants = []
    if ranks:
        names = [r["variant"] for r in ranks[0]["records"]]
        for name in names:
            rows = []
            for r in ranks:
                for rec in r["records"]:
                    if rec["variant"] == name:
                        rows.append(rec)
            base = rows[0]
            entry = {
                "variant": name,
                "doc": base.get("doc"),
                "ranks": [{"rank": r["rank"],
                           "t_s_median": r.get("t_s_median"),
                           "cpu_enqueue_ms_median": r.get("cpu_enqueue_ms_median"),
                           "correct": r.get("correct"),
                           "received": r.get("received"),
                           "error": r.get("error")} for r in rows],
            }
            ok = [r for r in rows if r.get("t_s_median") is not None]
            if ok:
                entry["t_s_max_rank"] = max(r["t_s_median"] for r in ok)
                entry["cpu_enqueue_ms_max_rank"] = max(
                    r["cpu_enqueue_ms_median"] for r in ok)
                entry["all_correct"] = all(r.get("correct") for r in ok)
            variants.append(entry)
    payload = {
        "kind": "crosshost_p2p_batching",
        "chapter": chapter,
        "run_id": run_id,
        "hosts": {"rank0": a_host, "rank1": b_host},
        "world_size": len(dirs),
        "payload_bytes": ranks[0]["payload_bytes"] if ranks else None,
        "iters": ranks[0]["iters"] if ranks else None,
        "api_surface": api,
        "variants": variants,
    }
    dest = os.path.join(RECV_DIR, a_host, chapter, run_id, "crosshost_p2p.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[merge] → {dest}", flush=True)
    for v in variants:
        t = v.get("t_s_max_rank")
        if t is None:
            print(f"   {v['variant']:<20} ERROR "
                  f"{str(v['ranks'][0].get('error'))[:70]}")
        else:
            print(f"   {v['variant']:<20} t={t * 1e3:8.3f} ms  "
                  f"cpu={v['cpu_enqueue_ms_max_rank']:7.3f} ms  "
                  f"correct={v['all_correct']}")
    return payload


def merge_completion(chapter, run_id, a_host, b_host):
    dirs = _rank_dirs(chapter, run_id, a_host, b_host)
    ranks = []
    for rank, d in enumerate(dirs):
        path = os.path.join(d, f"stream_semantics.rank{rank}.json")
        if os.path.exists(path):
            ranks.append(load(path))
    summary = {}
    if ranks:
        for v in ranks[0]["summary"]:
            rows = [r["summary"][v] for r in ranks]
            summary[v] = {
                "trials_per_rank": [r["trials"] for r in rows],
                "correct_observed_trials": [r["correct_observed_trials"] for r in rows],
                "correct_final_trials": [r["correct_final_trials"] for r in rows],
                "cpu_return_ms_median": [r["cpu_return_ms_median"] for r in rows],
                "sync_done_ms_median": [r["sync_done_ms_median"] for r in rows],
                "observed_values": sorted({round(x, 3)
                                           for r in rows for x in r["observed_values"]})[:8],
            }
    payload = {
        "kind": "crosshost_completion_semantics",
        "chapter": chapter,
        "run_id": run_id,
        "hosts": {"rank0": a_host, "rank1": b_host},
        "world_size": len(dirs),
        "buffer_bytes": ranks[0]["buffer_bytes"] if ranks else None,
        "expected_value": ranks[0]["expected_value"] if ranks else None,
        "summary": summary,
    }
    dest = os.path.join(RECV_DIR, a_host, chapter, run_id, "crosshost_completion.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[merge] → {dest}", flush=True)
    for v, s in summary.items():
        print(f"   {v:<20} 读时正确 {s['correct_observed_trials']}/{s['trials_per_rank']}  "
              f"全同步后 {s['correct_final_trials']}/{s['trials_per_rank']}  "
              f"CPU {s['cpu_return_ms_median']} ms  设备 {s['sync_done_ms_median']} ms  "
              f"读到 {s['observed_values']}")
    return payload


def merge(task, run_id, chapter, a_host, b_host):
    if task == "sweep":
        return merge_sweep(chapter, run_id, a_host, b_host)
    if task == "p2p":
        return merge_p2p(chapter, run_id, a_host, b_host)
    if task == "socket":
        return merge_socket(chapter, run_id, a_host, b_host)
    return merge_completion(chapter, run_id, a_host, b_host)


def merge_socket(chapter, run_id, a_host, b_host):
    dirs = _rank_dirs(chapter, run_id, a_host, b_host)
    ranks = []
    for rank, d in enumerate(dirs):
        path = os.path.join(d, f"socket.rank{rank}.json")
        if os.path.exists(path):
            ranks.append(load(path))
    payload = {
        "kind": "crosshost_socket_reference",
        "chapter": chapter,
        "run_id": run_id,
        "hosts": {"rank0": a_host, "rank1": b_host},
        "world_size": len(dirs),
        "payload_bytes": ranks[0]["payload_bytes"] if ranks else None,
        "reps": ranks[0]["reps"] if ranks else None,
        "ranks": ranks,
    }
    dest = os.path.join(RECV_DIR, a_host, chapter, run_id, "crosshost_socket.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[merge] → {dest}", flush=True)
    for r in ranks:
        print(f"   rank{r['rank']} {r['direction']:<12} "
              f"t={r['t_s_median'] * 1e3:8.3f} ms  "
              f"{r['gbps_median']:.4f} GB/s", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sweep", "completion", "p2p", "socket"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--master-port", type=int, default=29921)
    ap.add_argument("--sizes", default="")
    ap.add_argument("--ops", default="")
    ap.add_argument("--bytes", type=int, default=64 << 20)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5400.0)
    ap.add_argument("--ifname", default=None,
                    help="强制两台机器使用同一网卡名；默认优先真实主机网卡")
    ap.add_argument("--isolate", action="store_true",
                    help="sweep 在每轮计时前插 barrier，切断跨轮流水")
    ap.add_argument("--variants", default="",
                    help="p2p 只跑列出的变体（逗号分隔）")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    args = []
    if a.task == "sweep":
        if a.sizes:
            args += ["--sizes", a.sizes]
        if a.ops:
            args += ["--ops", a.ops]
        if a.isolate:
            args += ["--isolate"]
    elif a.task == "p2p":
        args += ["--bytes", str(a.bytes), "--iters", str(a.trials)]
        if a.variants:
            args += ["--variants", a.variants]
    else:
        args += ["--bytes", str(a.bytes), "--trials", str(a.trials)]

    os.makedirs("/Volumes/data/artifacts/llm-infra/scratch", exist_ok=True)
    if not a.no_push:
        push_labs()
    return run_task(a.task, a.run_id, args, a.master_port, a.timeout, a.ifname)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
