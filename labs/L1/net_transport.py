#!/usr/bin/env python3
"""L1.5 lab · 传输后端对照：同一份数据，换 transport / 块大小 / 并发。

任务 B 要求在**项目端点**上比较 TCP 与可用传输后端，并让「实际选中的
transport、链路和 buffer 注册」可追踪。本环境的硬约束（正文有记录）：

  - GPU 机器之间没有互通网络（crater 在 192.168.105.0/24，worldvln 在
    192.168.26.0/24，两者互不可达），跨机 TCP 对照在这个环境里做不了；
  - worldvln 的 RoCE 端口（irdma0/1）是 ACTIVE，但没有 dev 头文件、
    没有 libibverbs.so 开发软链、也没有任何 Python RDMA 绑定，
    无法注册 buffer、无法建立 QP。

于是这里做能做的严格对照：TCP over loopback / TCP over 容器网卡 / UDS，
块大小 64 KiB–64 MiB，并发 1/4/16。

两个工程要点（第一版都踩了）：
  1. **收发两端必须在不同进程**。同一进程里用两个线程，GIL 会把
     「Python 层协议开销」混进「链路吞吐」，测出来只有 0.3–3 GB/s。
     这里用 fork：服务端在子进程，客户端在父进程。
  2. **每块一次系统调用**。块头（长度 + 序号）和数据预先拼成一帧，
     每块只调用一次 sendall；服务端只对抽样的块算哈希，
     否则 hashlib 本身就会成为瓶颈。

用法：
    python net_transport.py --out net_transport.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HDR = 16          # 8 字节长度 + 8 字节块序号


def sh(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception:  # noqa: BLE001
        return ""


def local_ipv4() -> str:
    """取容器自己 eth0 的地址（只让内核选源地址，不探测外部主机）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))       # TEST-NET-1
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class TcpEndpoint:
    def __init__(self, host: str):
        self.host = host
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, 0))
        self.srv.listen(128)
        self.port = self.srv.getsockname()[1]

    def connect(self) -> socket.socket:
        c = socket.create_connection((self.host, self.port), timeout=120)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        return c

    def accept(self) -> socket.socket:
        c, _ = self.srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        return c

    def close(self):
        self.srv.close()

    def describe(self, sock: socket.socket) -> dict:
        return {"transport": self.name, "local": f"{sock.getsockname()}",
                "peer": f"{sock.getpeername()}",
                "family": int(sock.family), "type": int(sock.type),
                "tcp_nodelay": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY),
                "sndbuf": sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                "rcvbuf": sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)}

    @property
    def name(self) -> str:
        return f"tcp/{self.host}"


class UnixEndpoint(TcpEndpoint):
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(str(self.path))
        self.srv.listen(128)
        self.port = 0

    def connect(self) -> socket.socket:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(str(self.path))
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        return c

    def accept(self) -> socket.socket:
        # 必须自己实现：父类的 accept() 会设 TCP_NODELAY，
        # 对 AF_UNIX 套接字调用 IPPROTO_TCP 选项会直接抛 OSError，
        # 子进程里的 accept 线程挂掉、客户端只看到 Broken pipe（第一次跑就是这样）。
        c, _ = self.srv.accept()
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        return c

    def describe(self, sock: socket.socket) -> dict:
        return {"transport": self.name, "family": int(sock.family),
                "type": int(sock.type),
                "sndbuf": sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                "rcvbuf": sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)}

    @property
    def name(self) -> str:
        return "unix"


# ------------------------------------------------------------------ protocol
def recv_exact_into(sock: socket.socket, view: memoryview, n: int) -> int:
    got = 0
    while got < n:
        k = sock.recv_into(view[got:n])
        if k == 0:
            raise ConnectionResetError(f"对端提前关闭（{got}/{n}）")
        got += k
    return got


def server_conn(ep, total_bytes: int, block: int, sample_idx: set, idx: int) -> dict:
    """收满 total_bytes，校验块序号，只对抽样的块算哈希，最后发完成事件。"""
    c = ep.accept()
    buf = bytearray(HDR + block)
    view = memoryview(buf)
    got = 0
    nblocks = 0
    samples: dict[str, str] = {}
    try:
        while got < total_bytes:
            recv_exact_into(c, view, HDR)
            ln = int.from_bytes(buf[0:8], "little")
            bi = int.from_bytes(buf[8:16], "little")
            if bi != nblocks:
                raise RuntimeError(f"块序号错乱：期望 {nblocks}，收到 {bi}")
            recv_exact_into(c, view[HDR:HDR + ln], ln)
            if nblocks in sample_idx:
                samples[str(nblocks)] = hashlib.sha256(buf[HDR:HDR + ln]).hexdigest()
            got += ln
            nblocks += 1
        done = json.dumps({"conn": idx, "bytes": got, "blocks": nblocks,
                           "samples": samples}).encode()
        c.sendall(len(done).to_bytes(8, "little") + done)
        return json.loads(done)
    finally:
        c.close()


def server_child(ep, conc: int, total_bytes: int, block: int, sample_idx: set,
                 wfd: int) -> None:
    results: list = [None] * conc
    ts = [threading.Thread(target=lambda i=i: results.__setitem__(
        i, _safe(server_conn, ep, total_bytes, block, sample_idx, i)))
        for i in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=1200)
    os.write(wfd, json.dumps(results).encode())


def _safe(fn, *a) -> dict:
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def make_frames(total_bytes: int, block: int, pattern: bytes) -> tuple[list, dict]:
    """预先拼好每一帧：16 字节头（长度 + 序号）+ 数据。每块一次 sendall。

    注意 `pattern[:block]` 只在 block ≤ 1 MiB 时正确；block 更大时必须把
    1 MiB 的 pattern 重复拼到 block 长度，否则帧头声明的长度和实际字节数
    不一致，接收端会一直等（第一次跑就是这样在大块上超时的）。
    """
    nblocks = total_bytes // block
    if block <= len(pattern):
        payload = pattern[:block]
    else:
        reps = block // len(pattern)
        payload = pattern * reps + pattern[:block - reps * len(pattern)]
    frames = [block.to_bytes(8, "little") + i.to_bytes(8, "little") + payload
              for i in range(nblocks)]
    return frames, {str(i): hashlib.sha256(frames[i][HDR:]).hexdigest()
                    for i in (0, nblocks - 1, nblocks // 2)}


def client_conn(ep, frames: list, local_hashes: dict, idx: int) -> dict:
    c = ep.connect()
    info = ep.describe(c)
    t0 = time.perf_counter()
    total = 0
    for f in frames:
        c.sendall(f)
        total += len(f) - HDR
    hdr = bytearray(8)
    recv_exact_into(c, memoryview(hdr), 8)
    ln = int.from_bytes(hdr, "little")
    body = bytearray(ln)
    recv_exact_into(c, memoryview(body), ln)
    dt = time.perf_counter() - t0
    done = json.loads(bytes(body))
    ok = all(done["samples"].get(k) == v for k, v in local_hashes.items())
    c.close()
    return {"conn": idx, "bytes": total, "seconds": round(dt, 4),
            "gbps": round(total / dt / 1e9, 2),
            "server_bytes": done["bytes"], "server_blocks": done["blocks"],
            "content_ok": ok, **info}


def run_case(ep, total_bytes: int, block: int, conc: int, pattern: bytes) -> dict:
    frames, local_hashes = make_frames(total_bytes, block, pattern)
    sample_idx = {int(k) for k in local_hashes}
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # 子进程：服务端
        os.close(rfd)
        try:
            server_child(ep, conc, total_bytes, block, sample_idx, wfd)
        except BaseException as e:                 # noqa: BLE001
            # 子进程里的异常必须回传，否则父进程只能看到「对端关闭」这种二手症状
            try:
                os.write(wfd, json.dumps({"child_error": repr(e)}).encode())
            except OSError:
                pass
        finally:
            os._exit(0)
    os.close(wfd)
    clients = [None] * conc
    ts = [threading.Thread(target=lambda i=i: clients.__setitem__(
        i, _safe(client_conn, ep, frames, local_hashes, i))) for i in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=1200)
    chunks = []
    while True:
        b = os.read(rfd, 1 << 20)
        if not b:
            break
        chunks.append(b)
    os.close(rfd)
    _, status = os.waitpid(pid, 0)
    try:
        srv = json.loads(b"".join(chunks))
    except Exception:  # noqa: BLE001
        srv = []
    ok = [c for c in clients if c and c.get("content_ok")]
    gbps = [c["gbps"] for c in clients if c and "gbps" in c]
    return {
        "block_kib": block // 1024, "concurrency": conc,
        "per_conn_bytes_mib": total_bytes // 2**20,
        "client_ok": len(ok), "conn_total": conc,
        "content_ok": len(ok) == conc,
        "server_bytes_ok": all(s and s.get("bytes") == total_bytes for s in srv)
                           if srv else False,
        "aggregate_gbps": round(sum(gbps), 2),
        "per_conn_gbps": [round(g, 2) for g in gbps],
        "first_conn_info": clients[0] if clients else {},
        "child_exit": os.waitstatus_to_exitcode(status),
    }


def rdma_probe() -> dict:
    """可用性判定一律看内核侧与注册能力，不看「库存在」。"""
    out = {}
    out["infiniband_devices"] = sorted(os.listdir("/sys/class/infiniband")) \
        if os.path.isdir("/sys/class/infiniband") else []
    out["ib_uverbs"] = sorted(os.listdir("/dev/infiniband")) \
        if os.path.isdir("/dev/infiniband") else []
    out["memory_peers"] = sorted(os.listdir("/sys/kernel/mm/memory_peers")) \
        if os.path.isdir("/sys/kernel/mm/memory_peers") else []
    out["rdma_link"] = sh(["rdma", "link", "show"])
    out["libs"] = [p for p in ("libibverbs.so", "librdmacm.so", "libibverbs.so.1",
                               "librdmacm.so.1", "libmlx5.so.1")
                   if Path("/usr/lib/x86_64-linux-gnu", p).exists()]
    out["headers"] = [p for p in ("infiniband/verbs.h", "rdma/rdma_cma.h")
                      if Path("/usr/include", p).exists()]
    out["tools"] = {t: bool(sh(["bash", "-lc", f"command -v {t}"]))
                    for t in ("ibv_devinfo", "ib_write_bw", "rxe_cfg")}
    try:
        import importlib.util
        out["python_bindings"] = {m: importlib.util.find_spec(m) is not None
                                  for m in ("pyverbs", "rdma", "ibverbs")}
    except Exception:  # noqa: BLE001
        out["python_bindings"] = {}
    out["rdma_usable"] = bool(out["ib_uverbs"]) and bool(out["libs"]) and bool(out["headers"])
    out["verdict"] = ("可发起 RDMA" if out["rdma_usable"] else
                      "无 /dev/infiniband、无 verbs 头文件：无法注册 buffer，"
                      "本环境未启用 RDMA；只保留源码级对照")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--total-mib", type=int, default=64,
                    help="每个连接的字节数")
    ap.add_argument("--only", default=None, help="只跑指定 transport（调试用）")
    args = ap.parse_args()
    if not args.work:
        args.work = str(Path(os.environ.get("LEARN_ROOT", ".")) / "work" / "net")
    Path(args.work).mkdir(parents=True, exist_ok=True)

    total_bytes = args.total_mib << 20
    pattern = os.urandom(1 << 20)
    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "per_conn_mib": args.total_mib,
        "local_ipv4": local_ipv4(),
        "cases": [], "rdma": rdma_probe(),
    }
    print(f"=== 传输后端对照（每连接 {args.total_mib} MiB；fork 出独立服务端进程）")
    print(f"    本机地址 {res['local_ipv4']}")
    print(f"    {'transport':>14} {'block':>9} {'并发':>4} {'聚合GB/s':>9} "
          f"{'单连接GB/s':>22} {'内容':>5} {'字节':>5}")

    cases = [
        ("tcp-loopback", lambda: TcpEndpoint("127.0.0.1")),
        ("tcp-eth0", lambda: TcpEndpoint(res["local_ipv4"])),
        ("unix", lambda: UnixEndpoint(Path(args.work) / "net.sock")),
    ]
    for name, mk in cases:
        if args.only and name != args.only:
            continue
        for block in (64 << 10, 1 << 20, 16 << 20, 64 << 20):
            if block > total_bytes:
                continue
            for conc in (1, 4, 16):
                if conc > 1 and block > (1 << 20):
                    continue        # 大块只跑单连接，控制总时长
                ep = mk()
                try:
                    r = run_case(ep, total_bytes, block, conc, pattern)
                finally:
                    ep.close()
                r["endpoint"] = name
                res["cases"].append(r)
                print(f"    {name:>14} {r['block_kib']:>7}KiB {conc:>4} "
                      f"{r['aggregate_gbps']:>9.2f} "
                      f"{str(r['per_conn_gbps']):>22} {str(r['content_ok']):>5} "
                      f"{str(r['server_bytes_ok']):>5}")

    print("\n[RDMA 可用性]")
    for k in ("infiniband_devices", "ib_uverbs", "memory_peers", "libs", "headers"):
        print(f"    {k:22s} {res['rdma'][k]}")
    print(f"    rdma link              {res['rdma']['rdma_link'].splitlines()[:2]}")
    print(f"    tools                  {res['rdma']['tools']}")
    print(f"    python bindings        {res['rdma']['python_bindings']}")
    print(f"    ⇒ {res['rdma']['verdict']}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
