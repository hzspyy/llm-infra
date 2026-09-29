#!/usr/bin/env python3
"""L1.5 lab · 传输适配器：注册 / 传输 / 等待 / 释放，以及失败注入。

任务 C 要求为 6.4/8.6 提供一个「注册、传输、等待、释放」的小型适配器，并
注入**接收端中断**与**重复完成事件**，验证内容哈希、句柄生命周期与失败清理
一致，同时报告注册成本是否被摊销。

为什么需要一个适配器而不是直接 socket：
传输库（NIXL、Mooncake、NCCL、RDMA verbs）的差别在注册与完成语义上——
「注册一块 buffer 得到一个句柄，传输时引用句柄，完成后触发一次完成事件，
用完显式释放」。把这条生命周期显式写出来，才能在换后端时只替换实现。

本脚本用 TCP 实现同一套接口，并提供三个必须能过的检查：
  1. happy path：内容哈希一致、句柄全部释放、没有 fd/线程泄漏
  2. 重复完成：接收端同一个 transfer id 发两次完成事件 → 适配器只处理一次
  3. 接收端中断：传输中途杀掉接收端 → 发送端有界失败，句柄与连接被清理

用法：
    python transfer_adapter.py --out transfer_adapter.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HDR = 16          # 8 字节 transfer id + 8 字节 payload 长度


def open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


class TransferAdapter:
    """注册 → 传输 → 等待 → 释放。

    句柄 = 注册时分配的 id。`wait()` 消费完成事件：**同一个 transfer id
    的重复完成事件只处理一次**（真实传输库在重传/超时重试时会出现这种情况，
    上层如果不做幂等，就会把一块数据算了两次或提前释放句柄）。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, server: bool = False):
        self.handles: dict[int, dict] = {}
        self.completed: dict[int, dict] = {}
        self._next_id = 1
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.sock: socket.socket | None = None
        self.leaks: list[str] = []
        if server:
            self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.srv.bind((host, port))
            self.srv.listen(8)
            self.port = self.srv.getsockname()[1]
        else:
            self.srv = None
            self.port = port

    # ---- 生命周期 ----
    def register(self, name: str, nbytes: int) -> int:
        """注册一块 buffer，返回句柄。成本记在 register_us 里。"""
        t0 = time.perf_counter()
        with self.lock:
            h = self._next_id
            self._next_id += 1
            self.handles[h] = {"name": name, "bytes": nbytes,
                               "register_us": round((time.perf_counter() - t0) * 1e6, 2)}
        self.events.append({"op": "register", "handle": h, "name": name, "bytes": nbytes})
        return h

    def transfer(self, h: int, payload: bytes, repeat_done: bool = False) -> None:
        """引用句柄把数据推给对端，并等待完成事件。"""
        assert h in self.handles, f"句柄 {h} 不存在或已释放"
        if self.sock is None:
            self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        self.events.append({"op": "transfer", "handle": h, "bytes": len(payload)})
        self.sock.sendall(h.to_bytes(8, "little") + len(payload).to_bytes(8, "little")
                          + payload)
        self.wait(h)
        if repeat_done:
            # 注入：服务端会再发一次同一个 transfer id 的完成事件
            self._recv_event(duplicate=True)

    def _recv_exact(self, n: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("尚未建立连接")
        buf = bytearray()
        while len(buf) < n:
            b = self.sock.recv(n - len(buf))
            if not b:
                raise ConnectionResetError("接收端在完成前关闭了连接")
            buf += b
        return bytes(buf)

    def _recv_event(self, duplicate: bool = False) -> None:
        """完成事件 = 8 字节长度 + JSON。

        必须是**长度前缀的定长读取**：直接 `recv(256)` 可能一次读到两个
        完成事件（重复注入时尤其容易），也可能只读到半个 JSON——
        第一版就是这样把接收端卡死的。
        """
        n = int.from_bytes(self._recv_exact(8), "little")
        ev = json.loads(self._recv_exact(n))
        tid = int(ev["tid"])
        with self.lock:
            if tid in self.completed:
                self.events.append({"op": "duplicate_done_ignored", "handle": tid})
                return
            self.completed[tid] = {"bytes": ev["bytes"], "sha256": ev["sha256"],
                                   "duplicate_seen": duplicate}
        self.events.append({"op": "done", "handle": tid, "bytes": ev["bytes"]})

    def wait(self, h: int, timeout: float = 30.0) -> dict:
        t0 = time.perf_counter()
        while h not in self.completed:
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"等待句柄 {h} 的完成事件超时")
            self._recv_event()
        return self.completed[h]

    def release(self, h: int) -> None:
        self.handles.pop(h, None)
        self.completed.pop(h, None)
        self.events.append({"op": "release", "handle": h})

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None
        if self.srv:
            self.srv.close()

    # ---- 服务端 ----
    def serve(self, n_expect: int, dup_ids: set[int], fail_after: int | None,
              ready: threading.Event | None = None) -> dict:
        ready and ready.set()
        conn, _ = self.srv.accept()
        got_total = 0
        n = 0
        results = []
        try:
            while n < n_expect:
                hdr = self._recv_exact_sock(conn, HDR)
                if hdr is None:
                    results.append({"error": "对端提前关闭"})
                    break
                tid = int.from_bytes(hdr[0:8], "little")
                ln = int.from_bytes(hdr[8:16], "little")
                body = self._recv_exact_sock(conn, ln)
                if body is None:
                    results.append({"error": "payload 截断"})
                    break
                digest = hashlib.sha256(body).hexdigest()
                got_total += ln
                n += 1
                results.append({"handle": tid, "bytes": ln, "sha256": digest})
                if fail_after is not None and n >= fail_after:
                    conn.close()                     # 注入：接收端中断
                    return {"interrupted_after": n, "results": results,
                            "bytes": got_total}
                ev = json.dumps({"tid": tid, "bytes": ln,
                                 "sha256": digest}).encode()
                conn.sendall(len(ev).to_bytes(8, "little") + ev)
                if tid in dup_ids:
                    conn.sendall(len(ev).to_bytes(8, "little") + ev)   # 重复完成
        finally:
            try:
                conn.close()
            except OSError:
                pass
        return {"results": results, "bytes": got_total}

    @staticmethod
    def _recv_exact_sock(sock: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            b = sock.recv(n - len(buf))
            if not b:
                return None
            buf += b
        return bytes(buf)


def run_happy(dup_ids: set[int]) -> dict:
    """正例：4 块数据，注册/传输/等待/释放，校验哈希与句柄清理。"""
    payloads = [os.urandom(1 << 20) for _ in range(4)]
    srv = TransferAdapter(server=True)
    out = {}

    def serve():
        out["server"] = srv.serve(len(payloads), dup_ids, fail_after=None)

    t = threading.Thread(target=serve)
    t.start()
    cli = TransferAdapter(port=srv.port)
    fds0 = open_fds()
    handles = [cli.register(f"buf{i}", len(p)) for i, p in enumerate(payloads)]
    local_hashes = {h: hashlib.sha256(p).hexdigest() for h, p in zip(handles, payloads)}
    t0 = time.perf_counter()
    for h, p in zip(handles, payloads):
        cli.transfer(h, p, repeat_done=(h in dup_ids))
    elapsed = time.perf_counter() - t0
    ok = all(cli.completed[h]["sha256"] == local_hashes[h] for h in handles)
    for h in handles:
        cli.release(h)
    cli.close()
    # 必须先等服务端线程收尾再读它的结果：客户端拿到最后一个完成事件就返回了，
    # 此时服务端可能还差几条语句没执行完（第一版在这里 KeyError）。
    t.join(timeout=30)
    server_ok = all(r.get("sha256") == local_hashes[r["handle"]]
                    for r in out.get("server", {}).get("results", []))
    srv.close()
    return {
        "case": "happy_path", "n": len(payloads),
        "content_ok": ok, "server_content_ok": server_ok,
        "handles_after_release": len(cli.handles),
        "completed_after_release": len(cli.completed),
        "elapsed_ms": round(elapsed * 1e3, 2),
        "fd_before": fds0, "fd_after": open_fds(),
        "events": cli.events,
        "server_result": out.get("server"),
    }


def run_duplicate() -> dict:
    """注入重复完成事件：同一个 transfer id 的完成事件发两次。"""
    payload = os.urandom(1 << 20)
    srv = TransferAdapter(server=True)
    out = {}

    def serve():
        out["server"] = srv.serve(1, {1}, fail_after=None)

    t = threading.Thread(target=serve)
    t.start()
    cli = TransferAdapter(port=srv.port)
    h = cli.register("dup", len(payload))
    cli.transfer(h, payload, repeat_done=True)
    time.sleep(0.2)
    dup_events = [e for e in cli.events if e["op"] == "duplicate_done_ignored"]
    ok = (cli.completed[h]["sha256"] == hashlib.sha256(payload).hexdigest()
          and len(dup_events) == 1)
    cli.release(h)
    cli.close()
    t.join(timeout=30)
    srv.close()
    return {"case": "duplicate_completion", "content_ok": bool(ok),
            "duplicates_ignored": len(dup_events),
            "handles_after_release": len(cli.handles),
            "events": cli.events, "server_result": out.get("server")}


def run_interrupt() -> dict:
    """注入接收端中断：收完第 2 块就关连接，发送端必须**有界失败并清理**。"""
    payloads = [os.urandom(1 << 20) for _ in range(6)]
    srv = TransferAdapter(server=True)
    out = {}

    def serve():
        out["server"] = srv.serve(len(payloads), set(), fail_after=2)

    t = threading.Thread(target=serve)
    t.start()
    cli = TransferAdapter(port=srv.port)
    fds0 = open_fds()
    handles = [cli.register(f"b{i}", len(p)) for i, p in enumerate(payloads)]
    err = None
    done = 0
    t0 = time.perf_counter()
    try:
        for h, p in zip(handles, payloads):
            cli.transfer(h, p)
            done += 1
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    elapsed = time.perf_counter() - t0
    for h in handles:
        cli.release(h)
    cli.close()
    t.join(timeout=30)
    srv.close()
    return {"case": "receiver_interrupt", "transfers_completed": done,
            "error": err, "bounded_ms": round(elapsed * 1e3, 2),
            "handles_after_release": len(cli.handles),
            "fd_before": fds0, "fd_after": open_fds(),
            "server_result": out.get("server")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=== 传输适配器：注册 / 传输 / 等待 / 释放 + 失败注入")
    res = {"measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    happy = run_happy(dup_ids={2})
    dup = run_duplicate()
    itr = run_interrupt()
    res.update({"happy_path": happy, "duplicate_completion": dup,
                "receiver_interrupt": itr})

    reg_us = [e.get("register_us", 0) for e in happy["events"] if e["op"] == "register"]
    print(f"\n[正例] {happy['n']} 块 1 MiB：内容一致={happy['content_ok']}，"
          f"服务端一致={happy['server_content_ok']}，"
          f"未释放句柄={happy['handles_after_release']}，"
          f"fd {happy['fd_before']}→{happy['fd_after']}，"
          f"{happy['elapsed_ms']} ms")
    print(f"       注册成本：{happy['n']} 次注册共 {sum(reg_us):.2f} µs，"
          f"平均 {sum(reg_us)/max(1,len(reg_us)):.2f} µs/次")
    print("       注意边界：本适配器的「注册」只是句柄登记，成本接近零；"
          "真实传输库的注册要锁页、建 DMA 映射，那部分本环境（无 RDMA）测不到，")
    print("       正文按 UNVERIFIED 处理，不能把这里的 0 µs 当成通用结论。")
    print(f"[重复完成] 内容一致={dup['content_ok']}，"
          f"被忽略的重复事件={dup['duplicates_ignored']}，"
          f"未释放句柄={dup['handles_after_release']}")
    print(f"[接收端中断] 完成 {itr['transfers_completed']} 块后失败：{itr['error']}")
    print(f"             有界结束 {itr['bounded_ms']} ms，"
          f"未释放句柄={itr['handles_after_release']}，"
          f"fd {itr['fd_before']}→{itr['fd_after']}，"
          f"服务端记录={itr['server_result'].get('interrupted_after')} 块后中断")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
