#!/usr/bin/env python3
"""
6.0 三个任务的驱动脚本。

    python collective_contracts.py A --backend gloo --world-size 4 --out <dir>
    python collective_contracts.py B --backend gloo --case order_mismatch --out <dir>
    python collective_contracts.py C --world-size 2 --out <dir>

任务 A：把 world 与子组上的四类集合通信跑一遍，逐步列出本 rank 的值与重建出的全局值，
        并与纯 Python 单进程参照逐元素对拍。
任务 B：四类失败注入（op 次序错配、shape 错配、单 rank 提前退出、组错配），
        每种都在独立子进程里跑，带超时与强制回收，保存原始堆栈和退出码。
任务 C：通信流写、计算流读同一 buffer 的四种完成语义对照（只等 CPU 返回、
        过早覆盖、正确 event、正确等待），数值与完成时间都记录。

本文件只做实验编排与记录；序列记录器的实现在 comm_sequence.py。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comm_sequence as cs  # noqa: E402


# ==========================================================================
# 运行清单
# ==========================================================================

def env_pins():
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    return info


def write_manifest(out_dir, task, backend, world_size, seed, cases, extra=None):
    payload = {
        "task": task,
        "backend": backend,
        "world_size": world_size,
        "seed": seed,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env_pins(),
        "input_spec": "确定性小整数列表，length=组大小（reduce_scatter）或 4（其余）；"
                      "由 (seed, op, rank) 生成，任何 rank 可本地重建全局参照",
        "tolerance": "集合通信结果逐元素精确相等（整数），容差 0",
        "timing_boundary": "wall/perf_counter 由 CPU 侧打点；C 任务的设备完成时间以 "
                           "torch.cuda.synchronize 返回为准",
        "source_pins": {
            "repo": "https://github.com/pytorch/pytorch",
            "nccl": "v2.13.0 torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
            "gloo": "v2.14.0 torch/csrc/distributed/c10d/ProcessGroupGloo.cpp",
            "work": "v2.14.0 torch/csrc/distributed/c10d/Work.cpp",
        },
        "outputs": "见同目录 verify.json / failure.json / stream_semantics.json 与各 rank 日志",
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    return payload


# ==========================================================================
# 任务 A
# ==========================================================================

def _device_for(backend, rank):
    if backend == "nccl":
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(dev)
        return dev
    return torch.device("cpu")


# torch 2.14 起 *_tensor 系列改名，两套名字在 2.13/2.14 上签名相同
ALL_GATHER_INTO = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
REDUCE_SCATTER = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor


def _i64(values, device):
    return torch.tensor(values, dtype=torch.int64, device=device)


def task_A(rank, world_size, backend, out_dir, seed):
    device = _device_for(backend, rank)
    cs.init_process_group(rank, world_size, backend, timeout_s=60)
    groups, pair_handles, even = cs.build_groups(world_size)
    mine = cs.group_for_rank(pair_handles, even, rank, world_size)
    ledger = cs.Ledger(rank, world_size, backend, device,
                       os.path.join(out_dir, f"ledger_rank{rank}.jsonl"), seed)
    t_start = time.perf_counter()

    # ---- 1. world all_reduce ----
    x = _i64(cs.op_input("all_reduce", rank, world_size, seed), device)
    ledger.call("all_reduce", None, "world", {"x": x},
                lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM))

    # ---- 2. world all_gather ----
    g_in = _i64(cs.op_input("all_gather", rank, world_size, seed), device)
    g_out = torch.zeros(world_size * g_in.numel(), dtype=torch.int64, device=device)
    ledger.call("all_gather", None, "world", {"in": g_in, "out": g_out},
                lambda: ALL_GATHER_INTO(g_out, g_in), primary="out")

    # ---- 3. world reduce_scatter ----
    rs_in = _i64(cs.op_input("reduce_scatter", rank, world_size, seed), device)
    rs_out = torch.zeros(1, dtype=torch.int64, device=device)
    ledger.call("reduce_scatter", None, "world", {"in": rs_in, "out": rs_out},
                lambda: REDUCE_SCATTER(rs_out, rs_in, op=dist.ReduceOp.SUM), primary="out")

    # ---- 4. world send/recv 环 ----
    # NCCL 的未批量化 P2P 在同一对 rank 上共用一个 2-rank 通信子，配对靠序号，
    # 两端都「先发后收」会把 seq0 的 send 对 send，必须用 batch_isend_irecv 成对提交。
    # Gloo 靠 tag 匹配，不批量化也接受「先发后收」。
    if world_size > 1:
        snd = _i64(cs.op_input("send_recv", rank, world_size, seed), device)
        rcv = torch.full_like(snd, -1)
        if backend == "nccl":
            ops = [dist.P2POp(dist.isend, snd, (rank + 1) % world_size),
                   dist.P2POp(dist.irecv, rcv, (rank - 1) % world_size)]
            pair = dist.batch_isend_irecv(ops)
            note = "batch_isend_irecv 成对提交"
        else:
            pair = (dist.isend(snd, dst=(rank + 1) % world_size),
                    dist.irecv(rcv, src=(rank - 1) % world_size))
            note = "未批量化 isend/irecv"
        ledger.call("send_recv", None, "world", {"send": snd, "recv": rcv},
                    lambda: [w.wait() for w in pair], primary="recv", note=note)

    # ---- 5. 子组 all_reduce：同一个 op 名，不同进程组 ----
    if "pairs" in mine:
        members = cs.Ledger.members(mine["pairs"], world_size)
        p = _i64(cs.op_input("all_reduce", rank, len(members), seed + 1), device)
        ledger.call("all_reduce", mine["pairs"], "pairs", {"x": p},
                    lambda: dist.all_reduce(p, op=dist.ReduceOp.SUM, group=mine["pairs"]))
    if "even" in mine:
        members = cs.Ledger.members(mine["even"], world_size)
        e = _i64(cs.op_input("all_reduce", rank, len(members), seed + 2), device)
        ledger.call("all_reduce", mine["even"], "even", {"x": e},
                    lambda: dist.all_reduce(e, op=dist.ReduceOp.SUM, group=mine["even"]))

    elapsed = time.perf_counter() - t_start
    dist.barrier()
    ledger.close()

    # ---- 收齐各 rank 的 ledger 并对拍 ----
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {"rank": rank,
                                      "events": ledger.events,
                                      "elapsed_s": elapsed})
    summary = None
    if rank == 0:
        summary = verify_and_write(out_dir, gathered, world_size, backend, seed)

    dist.barrier()
    dist.destroy_process_group()
    return summary


def verify_and_write(out_dir, gathered, world_size, backend, seed):
    """把每个事件与单进程参照对拍，首个差异单独列出。"""
    checks, mismatches = [], []
    for lg in gathered:
        for ev in lg["events"]:
            if ev["op"] == "send_recv" and len(ev["group_members"]) < 2:
                continue
            if ev["primary"] not in ev["output"]:
                # barrier 这类无张量输出的控制操作只记录进程组归属，不做数值对拍
                checks.append({
                    "rank": ev["rank"], "step": ev["step"], "op": ev["op"],
                    "group": ev["group"], "group_members": ev["group_members"],
                    "primary": None, "expected": None, "actual": None,
                    "match": None,
                })
                continue
            exp = cs.reference(ev["op"], ev["group_members"],
                               len(ev["group_members"]), seed if ev["group"] == "world" else
                               (seed + 1 if ev["group"] == "pairs" else seed + 2))
            got = ev["output"][ev["primary"]]["values"]
            row = {
                "rank": ev["rank"], "step": ev["step"], "op": ev["op"],
                "group": ev["group"], "group_members": ev["group_members"],
                "primary": ev["primary"],
                "expected": exp[ev["rank"]], "actual": got,
                "match": exp[ev["rank"]] == got,
            }
            checks.append(row)
            if row["match"] is False:
                mismatches.append(row)
    view = cs.global_view(gathered)
    payload = {
        "backend": backend,
        "world_size": world_size,
        "checks": checks,
        "n_checks": len(checks),
        "n_mismatch": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "elapsed_s": {str(g["rank"]): g["elapsed_s"] for g in gathered},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ledger_all.json"), "w", encoding="utf-8") as f:
        json.dump(gathered, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "global_view.json"), "w", encoding="utf-8") as f:
        json.dump(view, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_manifest(out_dir, "A", backend, world_size, seed, cases=[
        {"rank": c["rank"], "step": c["step"], "op": c["op"], "group": c["group"],
         "group_members": c["group_members"], "expected": c["expected"]} for c in checks])
    print(f"[A] {len(checks)} 项对拍，不一致 {len(mismatches)}")
    for row in checks:
        print(f"  rank{row['rank']} step{row['step']:<2} {row['op']:<15} "
              f"group={row['group']:<6} members={row['group_members']} "
              f"期望={row['expected']} 实得={row['actual']} "
              f"{'OK' if row['match'] else 'MISMATCH'}")
    return payload


# ==========================================================================
# 任务 B：失败注入
# ==========================================================================

def _bdev(backend, rank):
    """B 的注入现场在哪个设备上：NCCL 必须用 CUDA 张量，Gloo 用 CPU。"""
    return _device_for(backend, rank)


def worker_order_mismatch(rank, world_size, backend, out_dir):
    """rank 0 先 all_reduce 再 broadcast，rank 1 反过来。"""
    cs.init_process_group(rank, world_size, backend, timeout_s=20, master_port=29601)
    dev = _bdev(backend, rank)
    lg = cs.Ledger(rank, world_size, backend, dev,
                   os.path.join(out_dir, f"rank{rank}.jsonl"))
    a = torch.tensor([rank + 1], dtype=torch.int64, device=dev)
    b = torch.tensor([100 + rank], dtype=torch.int64, device=dev)
    if rank == 0:
        lg.call("all_reduce", None, "world", {"x": a},
                lambda: dist.all_reduce(a, op=dist.ReduceOp.SUM), note="rank0 第 1 步")
        lg.call("broadcast", None, "world", {"x": b},
                lambda: dist.broadcast(b, src=0), note="rank0 第 2 步")
    else:
        lg.call("broadcast", None, "world", {"x": b},
                lambda: dist.broadcast(b, src=0), note="rank1 第 1 步")
        lg.call("all_reduce", None, "world", {"x": a},
                lambda: dist.all_reduce(a, op=dist.ReduceOp.SUM), note="rank1 第 2 步")
    dist.destroy_process_group()


def worker_shape_mismatch(rank, world_size, backend, out_dir):
    """rank 0 用 4 个元素，rank 1 用 8 个元素做同一个 all_reduce。"""
    cs.init_process_group(rank, world_size, backend, timeout_s=20, master_port=29602)
    dev = _bdev(backend, rank)
    lg = cs.Ledger(rank, world_size, backend, dev,
                   os.path.join(out_dir, f"rank{rank}.jsonl"))
    n = 4 if rank == 0 else 8
    x = torch.arange(n, dtype=torch.int64, device=dev) + 1
    lg.call("all_reduce", None, "world", {"x": x},
            lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), note=f"本 rank numel={n}")
    dist.destroy_process_group()


def worker_early_exit(rank, world_size, backend, out_dir):
    """rank 0 只做 barrier 就退出；其余 rank 继续调 all_reduce。"""
    cs.init_process_group(rank, world_size, backend, timeout_s=20, master_port=29603)
    dev = _bdev(backend, rank)
    lg = cs.Ledger(rank, world_size, backend, dev,
                   os.path.join(out_dir, f"rank{rank}.jsonl"))
    lg.call("barrier", None, "world", {}, lambda: dist.barrier(), note="第 1 步：全体到达")
    if rank == 0:
        lg.close()
        os._exit(7)
    x = torch.tensor([rank + 1], dtype=torch.int64, device=dev)
    lg.call("all_reduce", None, "world", {"x": x},
            lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), note="第 2 步：rank0 已不在")
    dist.destroy_process_group()


def worker_group_mismatch(rank, world_size, backend, out_dir):
    """rank 0 在 size=1 的子组上 all_reduce，rank 1 在 world 上 all_reduce。"""
    cs.init_process_group(rank, world_size, backend, timeout_s=20, master_port=29604)
    dev = _bdev(backend, rank)
    solo = [dist.new_group([r]) for r in range(world_size)]
    lg = cs.Ledger(rank, world_size, backend, dev,
                   os.path.join(out_dir, f"rank{rank}.jsonl"))
    x = torch.tensor([rank + 1], dtype=torch.int64, device=dev)
    if rank == 0:
        lg.call("all_reduce", solo[0], "solo", {"x": x},
                lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM, group=solo[0]),
                note="rank0 在自己的 size=1 子组里")
    else:
        lg.call("all_reduce", None, "world", {"x": x},
                lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM),
                note="rank1 等 world 里的 rank0")
    dist.destroy_process_group()


def worker_p2p_send_first(rank, world_size, backend, out_dir):
    """两端都「先 isend 再 irecv」。

    NCCL 在同一对 rank 上为未批量化 P2P 建一个共用的 2-rank 通信子，配对按序号：
    两端的 seq0 都是 send，谁也没在 seq0 上 recv，RECV 永远等不到。
    Gloo 按 tag 匹配，同样的调用次序能完成。
    """
    cs.init_process_group(rank, world_size, backend, timeout_s=20, master_port=29605)
    dev = _device_for(backend, rank)
    lg = cs.Ledger(rank, world_size, backend, dev,
                   os.path.join(out_dir, f"rank{rank}.jsonl"))
    snd = _i64([1, 2, 3, 4], dev) + rank * 10
    rcv = torch.full_like(snd, -1)
    a = dist.isend(snd, dst=(rank + 1) % world_size)
    b = dist.irecv(rcv, src=(rank - 1) % world_size)
    lg.call("send_recv", None, "world", {"send": snd, "recv": rcv},
            lambda: [a.wait(), b.wait()], primary="recv",
            note="两端都先 isend 再 irecv")
    dist.destroy_process_group()


CASES = {
    "order_mismatch": (worker_order_mismatch, 2),
    "shape_mismatch": (worker_shape_mismatch, 2),
    "early_exit": (worker_early_exit, 2),
    "group_mismatch": (worker_group_mismatch, 2),
    "p2p_send_first": (worker_p2p_send_first, 2),
}


def task_B(case, backend, out_dir, parent_timeout=75):
    """在独立子进程里跑一种失败注入，超时后强制回收整组进程。"""
    fn, world_size = CASES[case]
    os.makedirs(out_dir, exist_ok=True)
    env0 = dict(os.environ)
    env0.pop("RANK", None)
    env0.pop("WORLD_SIZE", None)
    procs = []
    for r in range(world_size):
        env = dict(env0)
        env["RANK"] = str(r)
        env["WORLD_SIZE"] = str(world_size)
        if backend != "nccl":
            env["CUDA_VISIBLE_DEVICES"] = ""
        log = open(os.path.join(out_dir, f"rank{r}.log"), "w", encoding="utf-8")
        procs.append((r, subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "B",
             "--case", case, "--backend", backend, "--out", out_dir,
             "--worker-rank", str(r)], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True), log))

    t0 = time.perf_counter()
    timed_out = False
    deadline = t0 + parent_timeout
    while time.perf_counter() < deadline:
        if all(p.poll() is not None for _, p, _ in procs):
            break
        time.sleep(0.25)
    else:
        timed_out = True
    elapsed = time.perf_counter() - t0

    if timed_out:
        for _, p, _ in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for _, p, _ in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    records = []
    for r, p, log in procs:
        log.close()
        txt = open(os.path.join(out_dir, f"rank{r}.log"), encoding="utf-8").read()
        last = None
        completed = None
        lgpath = os.path.join(out_dir, f"rank{r}.jsonl")
        if os.path.exists(lgpath):
            with open(lgpath, encoding="utf-8") as f:
                lines = [json.loads(x) for x in f if x.strip()]
            ends = [l for l in lines if l.get("phase") == "end"]
            if ends:
                completed = {"step": ends[-1]["step"], "op": ends[-1]["op"],
                             "group": ends[-1]["group"], "note": ends[-1]["note"]}
            if lines:
                last = {"step": lines[-1]["step"], "op": lines[-1]["op"],
                        "group": lines[-1]["group"], "note": lines[-1]["note"],
                        "phase": lines[-1].get("phase")}
        records.append({
            "rank": r,
            "returncode": p.returncode,
            "last_recorded_op": last,
            "last_completed_op": completed,
            "log_tail": txt[-4000:],
            "traceback_lines": [l for l in txt.splitlines()
                                if any(k in l for k in
                                       ("Error", "error", "Timeout", "terminating",
                                        "enforce fail", "what():"))][-10:],
        })

    payload = {
        "case": case,
        "backend": backend,
        "world_size": world_size,
        "parent_timeout_s": parent_timeout,
        "hung_until_parent_timeout": timed_out,
        "wall_s": elapsed,
        "ranks": records,
    }
    with open(os.path.join(out_dir, "failure.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_manifest(out_dir, f"B:{case}", backend, world_size, 0,
                   cases=[{"case": case, "world_size": world_size,
                           "parent_timeout_s": parent_timeout,
                           "worker_pg_timeout_s": 20}],
                   extra={"hung_until_parent_timeout": timed_out, "wall_s": elapsed,
                          "returncodes": {str(r["rank"]): r["returncode"] for r in records}})
    print(f"[B:{case}] wall={elapsed:.1f}s 父进程超时={timed_out} "
          f"退出码={[r['returncode'] for r in records]}")
    for r in records:
        print(f"  rank{r['rank']} rc={r['returncode']} "
              f"最后进入={r['last_recorded_op']} 最后完成={r['last_completed_op']}")
        for l in r["traceback_lines"]:
            print(f"    {l.strip()[:160]}")
    return payload


# ==========================================================================
# 任务 C：通信流与计算流的完成语义
# ==========================================================================

C_VARIANTS = [
    "cpu_return_only",      # 只等 CPU 侧返回
    "event_without_wait",   # 事件记在入队点，没先 work.wait()
    "premature_overwrite",  # 不等完成就改写同一 buffer
    "event_after_wait",     # 正确 event：work.wait() 之后在 s_comm 上记事件
    "wait_correct",         # 正确等待：同一流上 wait 之后再读
]


def _one_trial(variant, rank, world_size, buf, streams, expected, trials_so_far):
    s_comm, s_comp, s_bad = streams
    rec = {"variant": variant, "rank": rank}
    # 输入在通信流上产生，保证 NCCL 读到的是本次 trial 的填充值
    with torch.cuda.stream(s_comm):
        buf.fill_(float(rank + 1))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s_comm):
        work = dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=True)
        ev_enqueue = torch.cuda.Event()
        ev_enqueue.record(s_comm)
    rec["cpu_return_ms"] = (time.perf_counter() - t0) * 1e3

    if variant == "cpu_return_only":
        # 只看到 CPU 侧返回就在默认流上读；默认流与 s_comm 之间没有任何依赖
        rec["observed"] = float(buf[0].item())
    elif variant == "event_without_wait":
        # 事件只标在「调用已入队」这一点上，s_comm 还没和 NCCL 结束事件建立依赖
        s_comp.wait_event(ev_enqueue)
        with torch.cuda.stream(s_comp):
            rec["observed"] = float(buf[0].item())
    elif variant == "premature_overwrite":
        # 不等待通信完成，立刻在第三条流上改写同一 buffer
        with torch.cuda.stream(s_bad):
            buf.zero_()
        rec["observed"] = float(buf[0].item())
    elif variant == "event_after_wait":
        # wait() 让 s_comm 阻塞在 NCCL 结束事件上，此后记的事件才代表通信完成
        with torch.cuda.stream(s_comm):
            work.wait()
            ev_done = torch.cuda.Event()
            ev_done.record(s_comm)
        rec["wait_return_ms"] = (time.perf_counter() - t0) * 1e3
        s_comp.wait_event(ev_done)
        with torch.cuda.stream(s_comp):
            rec["observed"] = float(buf[0].item())
    elif variant == "wait_correct":
        with torch.cuda.stream(s_comm):
            work.wait()
            rec["wait_return_ms"] = (time.perf_counter() - t0) * 1e3
            rec["observed"] = float(buf[0].item())

    torch.cuda.synchronize()
    rec["sync_done_ms"] = (time.perf_counter() - t0) * 1e3
    rec["final"] = float(buf[0].item())
    rec["all_equal"] = bool(torch.all(buf == buf[0]).item())
    rec["expected"] = expected
    rec["correct_observed"] = (abs(rec["observed"] - expected) < 1e-6 and rec["all_equal"])
    rec["correct_final"] = (abs(rec["final"] - expected) < 1e-6 and rec["all_equal"])
    return rec


def task_C(rank, world_size, out_dir, nbytes, trials, device=None,
           master_addr=None, master_port=29590):
    # device/master_addr 为空时保持单机多进程的既有行为（cuda:rank、127.0.0.1）；
    # 跨机档由启动器传入 cuda:0 与真实的 MASTER_ADDR。
    device = torch.device(device) if device else torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    cs.init_process_group(rank, world_size, "nccl", timeout_s=120,
                          master_addr=master_addr, master_port=master_port)
    n = nbytes // 4
    buf = torch.zeros(n, dtype=torch.float32, device=device)
    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    s_bad = torch.cuda.Stream()
    expected = float(world_size * (world_size + 1) // 2)

    all_recs = []
    for variant in C_VARIANTS:
        for _ in range(trials):
            dist.barrier()
            torch.cuda.synchronize()
            rec = _one_trial(variant, rank, world_size, buf,
                             (s_comm, s_comp, s_bad), expected, len(all_recs))
            all_recs.append(rec)
    dist.barrier()

    gathered = [None] * world_size
    dist.all_gather_object(gathered, all_recs)
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        flat = [r for g in gathered for r in g]
        summary = {}
        for v in C_VARIANTS:
            rs = [r for r in flat if r["variant"] == v]
            summary[v] = {
                "trials": len(rs),
                "correct_observed_trials": sum(1 for r in rs if r["correct_observed"]),
                "correct_final_trials": sum(1 for r in rs if r["correct_final"]),
                "observed_values": [r["observed"] for r in rs],
                "final_values": [r["final"] for r in rs],
                "all_equal_trials": sum(1 for r in rs if r["all_equal"]),
                "cpu_return_ms_median": statistics.median(r["cpu_return_ms"] for r in rs),
                "sync_done_ms_median": statistics.median(r["sync_done_ms"] for r in rs),
            }
        payload = {
            "world_size": world_size,
            "buffer_bytes": nbytes,
            "expected_value": expected,
            "trials_per_variant": trials,
            "summary": summary,
            "records": flat,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "stream_semantics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        write_manifest(out_dir, "C", "nccl", world_size, 0, cases=[
            {"variant": v, "trials": trials, "buffer_bytes": nbytes,
             "expected_value": expected} for v in C_VARIANTS],
            extra={"buffer_bytes": nbytes, "trials_per_variant": trials})
        for v in C_VARIANTS:
            s = summary[v]
            print(f"[C] {v:<20} 读时正确 {s['correct_observed_trials']}/{s['trials']}  "
                  f"全同步后正确 {s['correct_final_trials']}/{s['trials']}  "
                  f"CPU 返回中位 {s['cpu_return_ms_median']:.3f} ms  "
                  f"设备完成中位 {s['sync_done_ms_median']:.3f} ms  "
                  f"读到的值 {sorted(set(round(x, 3) for x in s['observed_values']))[:6]}  "
                  f"终值 {sorted(set(round(x, 3) for x in s['final_values']))[:6]}")
        return payload
    return None


# ==========================================================================
# 入口
# ==========================================================================

def spawn(fn, world_size, args):
    import torch.multiprocessing as mp
    mp.spawn(fn, args=args, nprocs=world_size, join=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A", "B", "C"])
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case", default="order_mismatch")
    ap.add_argument("--worker-rank", type=int, default=-1)
    ap.add_argument("--parent-timeout", type=float, default=75.0)
    ap.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()

    if a.task == "A":
        spawn(task_A, a.world_size, (a.world_size, a.backend, a.out, a.seed))
    elif a.task == "B":
        if a.worker_rank >= 0:
            fn, ws = CASES[a.case]
            try:
                fn(a.worker_rank, ws, a.backend, a.out)
            except Exception:
                traceback.print_exc()
                sys.exit(1)
            return
        for case in CASES:
            task_B(case, a.backend, os.path.join(a.out, case), a.parent_timeout)
    elif a.task == "C":
        spawn(task_C, a.world_size, (a.world_size, a.out, a.bytes, a.trials))


if __name__ == "__main__":
    main()
