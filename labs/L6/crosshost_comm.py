#!/usr/bin/env python3
"""
6.0 / 6.1 跨机通信 worker（crater ↔ crater2，NET/Socket transport）。

每台机器只起一个进程、固定用本机 cuda:0；全局 rank 与 MASTER_ADDR 由启动器
（run_crosshost.py）通过 ssh 传入。三个任务分别补 6.0 与 6.1 的任务书缺口：

  sweep       集合通信按消息大小扫描。6.1-B/D 的跨机档：β、NCCL 自己选中的
              算法/协议/通道，以及实际走到的 transport（NET/Socket vs SHM/direct）。
  completion  buffer 完成语义五种写法。6.0-C 的跨机复测——单机上被证明正确的
              写法在跨机 transport 上是否仍成立，Work.wait 到底等到了什么。
  p2p         P2P 批量化路径对照。6.0 补遗：逐条同步 isend/irecv、逐条异步加
              两次 wait、batch_isend_irecv、以及 Python 侧可达的两条 coalescing
              路径（_coalescing_manager 与 _start/_end_coalescing）。

本文件只做本机一次测量并落盘；ssh 编排、结果回收与多 rank 归并在
run_crosshost.py。各 rank 各写一份 JSON，归并留给启动器（跨机时 rank 之间
看不到对方的目录）。

用法（由启动器调用，不手工执行）：
    python crosshost_comm.py sweep --rank 0 --world-size 2 \
        --master-addr 192.168.105.100 --master-port 29921 --out <本机目录>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collective_costs as cc          # noqa: E402
import collective_contracts as ccon    # noqa: E402

DEVICE = "cuda:0"
OPS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "sendrecv"]


# ==========================================================================
# 进程组与清单
# ==========================================================================

def init_pg(rank, world_size, master_addr, master_port, timeout_s=180):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=timeout_s))
    torch.cuda.set_device(0)
    return torch.device(DEVICE)


def env_pins(rank, world_size, master_addr, master_port):
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": ".".join(str(x) for x in torch.cuda.nccl.version()),
        "gpu": torch.cuda.get_device_name(0),
        "rank": rank,
        "world_size": world_size,
        "master_addr": master_addr,
        "master_port": master_port,
        "nccl_env": {k: v for k, v in sorted(os.environ.items())
                     if k.startswith("NCCL_")},
    }
    return info


def write_json(out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[write] {path}", flush=True)
    return path


def write_manifest(out_dir, task, rank, world_size, master_addr, master_port,
                   cases, extra=None):
    payload = {
        "task": task,
        "backend": "nccl",
        "rank": rank,
        "world_size": world_size,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env_pins(rank, world_size, master_addr, master_port),
        "cases": cases,
        "timing_boundary": "device-event 包住整个调用；CPU 入队时间用 perf_counter "
                           "分别记录，两者不混用",
        "outputs": "本目录下 sweep.rank*.json / stream_semantics.rank*.json / "
                   "p2p_batching.rank*.json，由 run_crosshost.py 回收并归并",
    }
    if extra:
        payload.update(extra)
    return write_json(out_dir, f"manifest.rank{rank}.json", payload)


# ==========================================================================
# 任务 sweep：跨机集合通信成本
# ==========================================================================

def task_sweep(rank, world_size, out_dir, master_addr, master_port, sizes, ops,
               isolate=False):
    init_pg(rank, world_size, master_addr, master_port)
    points = []
    for op in ops:
        for nb in sizes:
            try:
                r = cc._bench_one(rank, world_size, op, nb, device=DEVICE,
                                  isolate=isolate)
                points.append(r)
                if rank == 0:
                    print(f"  {op:<15} {cc.human(nb):>8} "
                          f"t={r['t_s_median'] * 1e3:9.3f} ms", flush=True)
            except Exception as e:  # noqa: BLE001
                points.append({"rank": rank, "op": op, "per_rank_bytes": nb,
                               "error": f"{type(e).__name__}: {e}"})
                if rank == 0:
                    print(f"  {op:<15} {cc.human(nb):>8} ERROR {e}", flush=True)
        dist.barrier()
    write_json(out_dir, f"sweep.rank{rank}.json",
               {"rank": rank, "world_size": world_size, "isolate": isolate,
                "points": points})
    write_manifest(out_dir, "sweep", rank, world_size, master_addr, master_port,
                   cases=[{"op": op, "per_rank_bytes": nb} for op in ops for nb in sizes],
                   extra={"sizes": sizes, "ops": ops, "isolate": isolate,
                          "buffer_dtype": "float32",
                          "per_rank_bytes_definition": "M = 每 rank 的输入字节数"})
    dist.barrier()
    dist.destroy_process_group()


# ==========================================================================
# 任务 completion：跨机完成语义（复用 6.0-C 的五种写法）
# ==========================================================================

def task_completion(rank, world_size, out_dir, master_addr, master_port,
                    nbytes, trials):
    ccon.task_C(rank, world_size, out_dir, nbytes, trials,
                device=DEVICE, master_addr=master_addr, master_port=master_port)
    # task_C 落盘的文件名与单机档一致；跨机时再加一份带 rank 的副本供归并
    src = os.path.join(out_dir, "stream_semantics.json")
    if os.path.exists(src):
        os.replace(src, os.path.join(out_dir, f"stream_semantics.rank{rank}.json"))


# ==========================================================================
# 任务 p2p：批量化路径对照
# ==========================================================================

P2P_VARIANTS = [
    "async_isend_irecv",     # 两条都先入队再逐个 wait：环上唯一正确的「逐条」写法
    "batch_isend_irecv",     # P2POp 列表一次提交（6.1 扫描 sendrecv 用的就是它）
    "coalescing_async",      # _coalescing_manager(device, async_ops=True) + cm.wait()
    "coalescing_sync",       # _coalescing_manager(device, async_ops=False)
    "sync_send_recv_ring",   # 对称阻塞 send/recv：环上会死锁，放最后且只跑 1 轮
]

P2P_DOC = {
    "async_isend_irecv": "isend 与 irecv 都先入队，再逐个 wait()——逐条路径的正确形式",
    "batch_isend_irecv": "P2POp 列表一次提交，返回 Work 列表",
    "coalescing_async": "Python 侧 _coalescing_manager(device, async_ops=True)，"
                        "内部调 group._start_coalescing/_end_coalescing",
    "coalescing_sync": "同上但 async_ops=False，_end_coalescing 内部同步",
    "sync_send_recv_ring": "两条阻塞 send/recv 顺序调用；对称环上双方都在等对端"
                           "先收，超时后由 NCCL watchdog 报错",
}

# 不预热、只跑 1 轮的变体：预热只会把同一个失败重复好几遍
P2P_FAILFAST = {"sync_send_recv_ring"}


def _p2p_call(variant, rank, world_size, snd, rcv, dev):
    peer_to = (rank + 1) % world_size
    peer_from = (rank - 1) % world_size
    if variant == "async_isend_irecv":
        w1 = dist.isend(snd, peer_to)
        w2 = dist.irecv(rcv, peer_from)
        for w in (w1, w2):
            w.wait()
        return None
    if variant == "batch_isend_irecv":
        ops = [dist.P2POp(dist.isend, snd, peer_to),
               dist.P2POp(dist.irecv, rcv, peer_from)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        return None
    if variant == "coalescing_async":
        with dc._coalescing_manager(device=dev, async_ops=True) as cm:
            dist.isend(snd, peer_to)
            dist.irecv(rcv, peer_from)
        cm.wait()
        return cm
    if variant == "coalescing_sync":
        with dc._coalescing_manager(device=dev, async_ops=False):
            dist.isend(snd, peer_to)
            dist.irecv(rcv, peer_from)
        return None
    if variant == "sync_send_recv_ring":
        dist.send(snd, peer_to)
        dist.recv(rcv, peer_from)
        return None
    raise ValueError(variant)


def _p2p_trial(variant, rank, world_size, snd, rcv, dev, iters=20):
    peer_from = (rank - 1) % world_size
    expected = float((peer_from % world_size) + 1)
    warmup = 0 if variant in P2P_FAILFAST else 3
    for _ in range(warmup):
        _p2p_call(variant, rank, world_size, snd, rcv, dev)
    torch.cuda.synchronize()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    cpu_ms = []
    for i in range(iters):
        rcv.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        starts[i].record()
        _p2p_call(variant, rank, world_size, snd, rcv, dev)
        ends[i].record()
        cpu_ms.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]
    ok = bool(torch.all(rcv == expected).item())
    return {
        "variant": variant,
        "rank": rank,
        "iters": iters,
        "t_s_median": statistics.median(times),
        "t_s": times,
        "cpu_enqueue_ms_median": statistics.median(cpu_ms),
        "cpu_enqueue_ms": cpu_ms,
        "received": float(rcv[0].item()),
        "expected": expected,
        "correct": ok,
        "all_equal": bool(torch.all(rcv == rcv[0]).item()),
    }


def task_p2p(rank, world_size, out_dir, master_addr, master_port, nbytes, iters,
             variants=None):
    dev = init_pg(rank, world_size, master_addr, master_port)
    elems = max(1, nbytes // 4)
    snd = torch.full((elems,), float(rank + 1), dtype=torch.float32, device=dev)
    rcv = torch.zeros(elems, dtype=torch.float32, device=dev)

    # coalescing 的 Python 入口在版本间会变，先记录本环境实际有什么
    pg = dc._get_default_group()
    api = {
        "has_coalescing_manager": hasattr(dc, "_coalescing_manager"),
        "has_start_coalescing": hasattr(pg, "_start_coalescing"),
        "has_end_coalescing": hasattr(pg, "_end_coalescing"),
        "pg_type": type(pg).__name__,
        "end_coalescing_returns": None,
    }
    try:
        pg._start_coalescing(dev)
        w = pg._end_coalescing(dev)
        api["end_coalescing_returns"] = type(w).__name__ if w is not None else None
        if w is not None:
            w.wait()
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        api["end_coalescing_returns"] = f"{type(e).__name__}: {e}"

    recs = []
    selected = list(variants) if variants else list(P2P_VARIANTS)
    for variant in selected:
        dist.barrier()
        v_iters = 1 if variant in P2P_FAILFAST else iters
        t0 = time.perf_counter()
        try:
            rec = _p2p_trial(variant, rank, world_size, snd, rcv, dev, v_iters)
        except Exception as e:  # noqa: BLE001
            rec = {"variant": variant, "rank": rank, "iters": v_iters,
                   "error": f"{type(e).__name__}: {e}",
                   "time_to_error_s": time.perf_counter() - t0}
        else:
            rec["doc"] = P2P_DOC[variant]
        recs.append(rec)
        if rank == 0:
            if "error" in rec:
                print(f"  [D] {variant:<20} ERROR({rec['time_to_error_s']:.1f}s) "
                      f"{rec['error'][:90]}", flush=True)
            else:
                print(f"  [D] {variant:<20} device={rec['t_s_median'] * 1e3:8.3f} ms  "
                      f"cpu={rec['cpu_enqueue_ms_median']:7.3f} ms  "
                      f"正确={rec['correct']}", flush=True)
    try:
        dist.barrier()
    except Exception:  # noqa: BLE001
        # 死锁变体触发 NCCL 错误后，末尾这道 barrier 本身也可能失败；记录但不掩盖
        api["trailing_barrier_failed"] = True
    write_json(out_dir, f"p2p_batching.rank{rank}.json",
               {"rank": rank, "world_size": world_size, "payload_bytes": nbytes,
                "iters": iters, "api": api, "records": recs})
    write_manifest(out_dir, "p2p", rank, world_size, master_addr, master_port,
                   cases=[{"variant": v,
                           "iters": 1 if v in P2P_FAILFAST else iters}
                          for v in selected],
                   extra={"api_surface": api, "payload_bytes": nbytes,
                          "variants_doc": P2P_DOC,
                          "variants_selected": selected,
                          "failfast_variants": sorted(P2P_FAILFAST),
                          "note": "sync_send_recv_ring 在跨机 NET/Socket 上触发 CUDA "
                                  "illegal memory access，进程在写盘前中止（见 "
                                  "20260922-p2p-batching 的启动日志），因此本档用 "
                                  "--variants 显式排除该变体"})
    dist.destroy_process_group()


# ==========================================================================
# 任务 socket：裸 TCP 参照（不经 NCCL、不经 GPU）
# ==========================================================================

def _socket_send(payload, addr, reps):
    """rank0：向 rank1 反复发同一块数据，记录真实字节与耗时。"""
    import socket as _s
    times = []
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as srv:
        srv.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", addr[1]))
        srv.listen(1)
        conn, peer = srv.accept()[:2]
        with conn:
            conn.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)
            total = 0
            for _ in range(reps):
                t0 = time.perf_counter()
                conn.sendall(payload)
                # 让对端确认收齐再进入下一轮，否则测到的是发送缓冲
                conn.recv(8)
                times.append(time.perf_counter() - t0)
                total += len(payload)
    return times, total, peer


def _socket_recv(nbytes, addr, reps, retries=20):
    """rank1：收满 nbytes×reps，逐轮回一个 8 字节确认。"""
    import socket as _s
    times = []
    got = 0
    # 两端进程是各自拉起的，rank0 可能还没 bind 完；连接拒绝要重试而不是直接失败
    for attempt in range(retries):
        try:
            s = _s.create_connection((addr[0], addr[1]), timeout=30)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)
    with s:
        s.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)
        buf = memoryview(bytearray(1 << 20))
        for _ in range(reps):
            t0 = time.perf_counter()
            left = nbytes
            while left > 0:
                n = s.recv_into(buf, min(len(buf), left))
                if n == 0:
                    raise RuntimeError("对端提前关闭")
                left -= n
            times.append(time.perf_counter() - t0)
            got += nbytes
            s.sendall(b"ok")
    return times, got


def task_socket(rank, world_size, out_dir, master_addr, master_port, nbytes,
                reps):
    """两台机器之间一个方向的有效载荷吞吐，作为集合通信的链路参照。"""
    if world_size != 2:
        raise ValueError("socket 参照只做 2 rank")
    if rank == 0:
        payload = bytes(nbytes)
        times, total, peer = _socket_send(payload, (master_addr, master_port),
                                          reps)
    else:
        times, total = _socket_recv(nbytes, (master_addr, master_port), reps)
        peer = None
    med = statistics.median(times)
    out = {
        "rank": rank,
        "peer": peer[0] if peer else None,
        "direction": "rank0→rank1" if rank == 0 else "rank1←rank0",
        "payload_bytes": nbytes,
        "reps": reps,
        "t_s_median": med,
        "t_s": times,
        "gbps_median": nbytes / med / 1e9,
        "bytes_total": total,
    }
    write_json(out_dir, f"socket.rank{rank}.json", out)
    write_manifest(out_dir, "socket", rank, world_size, master_addr,
                   master_port,
                   cases=[{"payload_bytes": nbytes, "reps": reps}],
                   extra={"note": "裸 TCP 单向流；每轮 sendall 后等对端回 8 字节"
                                  "确认，确保计入的是数据真正送达的时间",
                          "payload_bytes": nbytes, "reps": reps})
    print(f"[socket] rank{rank} {nbytes} B × {reps} t_median="
          f"{med * 1e3:.3f} ms → {nbytes / med / 1e9:.3f} GB/s", flush=True)


# ==========================================================================
# 入口
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sweep", "completion", "p2p", "socket"])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--master-addr", required=True)
    ap.add_argument("--master-port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="")
    ap.add_argument("--ops", default="")
    ap.add_argument("--bytes", type=int, default=64 << 20)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--iters", type=int, default=0,
                    help="p2p 的计时轮数；0 表示用 --trials")
    ap.add_argument("--isolate", action="store_true",
                    help="sweep 在每轮计时前插 barrier，切断跨轮流水")
    ap.add_argument("--variants", default="",
                    help="p2p 只跑列出的变体（逗号分隔）；用于显式排除会让进程在"
                         "写盘前中止的 sync_send_recv_ring")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else cc.SIZES
    ops = [x for x in a.ops.split(",") if x] if a.ops else OPS
    variants = [x for x in a.variants.split(",") if x] if a.variants else None

    print(f"[{a.task}] rank={a.rank}/{a.world_size} master={a.master_addr}:"
          f"{a.master_port} host={socket.gethostname()}", flush=True)
    if a.task == "sweep":
        task_sweep(a.rank, a.world_size, a.out, a.master_addr, a.master_port,
                   sizes, ops, isolate=a.isolate)
    elif a.task == "completion":
        task_completion(a.rank, a.world_size, a.out, a.master_addr,
                        a.master_port, a.bytes, a.trials)
    elif a.task == "p2p":
        task_p2p(a.rank, a.world_size, a.out, a.master_addr, a.master_port,
                 a.bytes, a.iters or a.trials, variants=variants)
    elif a.task == "socket":
        task_socket(a.rank, a.world_size, a.out, a.master_addr, a.master_port,
                    a.bytes, a.trials)


if __name__ == "__main__":
    main()
