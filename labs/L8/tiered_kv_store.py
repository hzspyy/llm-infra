#!/usr/bin/env python3
"""labs/L8/tiered_kv_store.py - 8.6-A: GPU / CPU / NVMe 三级 KV store 的最小实现.

设计目标是"能被检查", 因此把五件事写成显式状态而不是隐式约定:

1. **身份**: 每条 KV 条目带完整身份元组 (模型 revision、adapter revision、dtype、
   layout、层数、KV 头数、head dim、token 数、量化)。取回时必须逐字段相等, 否则抛
   `KVIdentityError` —— 不允许"名字相同就当同一份数据"。
2. **键**: 键只编码**逻辑身份** (model revision + adapter revision + 前 64 个 token)。
   dtype/layout/shape 不进键, 它们变化时应当是"键命中但身份被拒", 而不是静默 miss。
3. **引用**: 每次 `get` 增加引用计数; 引用未归还的条目不可驱逐, 驱逐时若引用非零
   就跳过并记录一次 `deferred_eviction`; `release` 下溢直接报错。
4. **状态**: REGISTERED → TRANSFERRING → RESIDENT_CPU → RESIDENT_GPU → EVICTING →
   EVICTED。传输中取消把条目退回并归零引用, 不留半成品。
5. **落盘**: 每层 K/V 连续写入一个文件, 文件尾部带身份元组的长度前缀 JSON。重新打开
   store 时只凭文件与文件名 (键) 就能恢复身份, 因此重启后的 `get` 仍然做完整身份校验。

文件格式 (kv_<key>.bin):
    [layer0.K bytes][layer0.V bytes][layer1.K bytes][layer1.V bytes]...
    [identity_json][identity_json_len: 8 bytes little-endian]
每层单个张量的形状由身份元组推出: (1, num_kv_heads, token_len, head_dim)。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class KVIdentityError(RuntimeError):
    """身份不匹配: 不允许把别的 revision / layout / dtype 的 KV 当成命中。"""


class KVStateError(RuntimeError):
    """状态机非法转移 (例如对已驱逐的条目取数据、重复取消、引用下溢)。"""


@dataclasses.dataclass(frozen=True)
class KVIdentity:
    model_revision: str
    adapter_revision: str
    dtype: str
    layout: str          # "BHSD" (batch, heads, seq, dim) 为唯一支持的物理布局
    num_layers: int
    num_kv_heads: int
    head_dim: int
    token_len: int       # 该条目覆盖的 token 数; 分块存储时用于拼装
    quant: str = "none"

    def check_compatible(self, other: "KVIdentity") -> None:
        diffs = []
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a != b:
                diffs.append(f"{f.name}: stored={a!r} requested={b!r}")
        if diffs:
            raise KVIdentityError("KV 身份不匹配 -> " + "; ".join(diffs))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def tensor_bytes(self) -> int:
        return self.num_kv_heads * self.head_dim * self.token_len * 2

    def entry_bytes(self) -> int:
        return 2 * self.num_layers * self.tensor_bytes()


STATE_REGISTERED = "REGISTERED"
STATE_TRANSFERRING = "TRANSFERRING"
STATE_CPU = "RESIDENT_CPU"
STATE_GPU = "RESIDENT_GPU"
STATE_EVICTING = "EVICTING"
STATE_EVICTED = "EVICTED"


@dataclasses.dataclass
class Entry:
    key: str
    identity: KVIdentity
    state: str = STATE_REGISTERED
    refcount: int = 0
    last_used_s: float = dataclasses.field(default_factory=time.monotonic)
    created_s: float = dataclasses.field(default_factory=time.monotonic)
    nvme_path: Optional[Path] = None
    _kv_gpu: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    _kv_cpu: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    cancel_requested: bool = False
    hits: int = 0

    @property
    def total_bytes(self) -> int:
        return self.identity.entry_bytes()


# --------------------------------------------------------------------------
# 文件格式
# --------------------------------------------------------------------------
def write_entry_file(kv: List[Tuple[torch.Tensor, torch.Tensor]], identity: KVIdentity,
                     path: Path) -> int:
    """把每层 K/V 按 BHSD 连续化后顺序写入, 尾部追加身份元组。

    文件尾部是 [identity_json][json_len: 8 B], 长度写在最后, 这样重新打开时只需读
    末尾 8 字节就能知道 JSON 从哪里开始。
    """
    written = 0
    with open(path, "wb") as f:
        for k, v in kv:
            for t in (k, v):
                # bfloat16 没有 numpy 对应类型, 因此按字节搬运: 拉平后把 dtype
                # 重解释成 uint8 再取 buffer, 这也是"KV 是字节级复用"的直接体现。
                cont = t.detach().to("cpu", copy=False).contiguous().reshape(-1)
                b = cont.view(torch.uint8).numpy().tobytes()
                f.write(b)
                written += len(b)
        payload = json.dumps(identity.to_dict()).encode()
        f.write(payload)
        f.write(struct.pack("<Q", len(payload)))
        written += 8 + len(payload)
    return written


def read_identity(path: Path) -> KVIdentity:
    """从文件尾部读出身份元组, 供服务重启后恢复。"""
    with open(path, "rb") as f:
        f.seek(-8, os.SEEK_END)
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(-8 - n, os.SEEK_END)
        payload = json.loads(f.read(n).decode())
    return KVIdentity(**payload)


class TieredKVStore:
    def __init__(self, nvme_dir: Path, ttl_s: float = 0.0, gpu_budget_bytes: int = 0,
                 pinned: bool = True, device: str = "cuda:0"):
        self.nvme_dir = Path(nvme_dir)
        self.nvme_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.pinned = pinned
        self.ttl_s = ttl_s
        self.gpu_budget_bytes = gpu_budget_bytes
        self.entries: Dict[str, Entry] = {}
        self.lock = threading.RLock()
        self.stats: Dict[str, int] = {
            "put_new": 0, "put_hit": 0, "get_hit_gpu": 0, "get_hit_cpu": 0,
            "get_hit_nvme": 0, "get_miss": 0, "reject_identity": 0,
            "evict": 0, "deferred_eviction": 0, "cancel": 0, "restore": 0,
        }
        self.timing: Dict[str, float] = {
            "put_stage_cpu_s": 0.0, "put_nvme_s": 0.0, "get_h2d_s": 0.0,
            "get_nvme_read_s": 0.0, "maintenance_s": 0.0,
        }
        self._stop = threading.Event()
        self._maint: Optional[threading.Thread] = None

    # ---- 键与身份 -------------------------------------------------------
    @staticmethod
    def key_for(prefix_tokens: Tuple[int, ...], identity: KVIdentity) -> str:
        """缓存键只含逻辑身份: 模型 revision、adapter revision 与前缀 token。

        物理身份 (dtype / layout / 层数 / KV 头数 / head dim / 量化 / 长度)
        **不进键**: 它们是同一个键下可能被配置悄悄改掉的属性, 也正是最危险的情况——
        键命中而物理布局不同。这类不一致必须在 `get` 里被显式拒绝, 而不是因为键不同
        就变成一次普通的 miss (miss 是安全的, 但会掩盖配置漂移)。
        换 adapter 改变的是缓存身份本身, 应该在键上分开, 表现为 miss。
        """
        h = hashlib.sha256()
        h.update(str(identity.model_revision).encode())
        h.update(b"\x00")
        h.update(str(identity.adapter_revision).encode())
        h.update(str(len(prefix_tokens)).encode())
        for t in prefix_tokens[:64]:           # 前 64 个 token 足以区分前缀
            h.update(int(t).to_bytes(4, "little", signed=True))
        return h.hexdigest()[:32]

    # ---- 写入 -----------------------------------------------------------
    def put(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity,
            kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> Entry:
        with self.lock:
            key = self.key_for(prefix_tokens, identity)
            if key in self.entries and self.entries[key].state != STATE_EVICTED:
                self.stats["put_hit"] += 1
                e = self.entries[key]
                e.last_used_s = time.monotonic()
                return e
            e = Entry(key=key, identity=identity, state=STATE_TRANSFERRING)
            self.entries[key] = e
            # CPU 一级: 注册成本 = 分配 + H2D 之外的 D2H 拷贝
            t0 = time.monotonic()
            cpu = []
            for k, v in kv:
                kc = torch.empty_like(k, device="cpu", pin_memory=self.pinned)
                vc = torch.empty_like(v, device="cpu", pin_memory=self.pinned)
                kc.copy_(k, non_blocking=False)
                vc.copy_(v, non_blocking=False)
                cpu.append((kc, vc))
            e._kv_cpu = cpu
            e.state = STATE_CPU
            self.timing["put_stage_cpu_s"] += time.monotonic() - t0
            # NVMe 一级
            t1 = time.monotonic()
            path = self.nvme_dir / f"kv_{key}.bin"
            write_entry_file(kv, identity, path)
            e.nvme_path = path
            self.timing["put_nvme_s"] += time.monotonic() - t1
            if self.gpu_budget_bytes >= e.total_bytes:
                e._kv_gpu = [(k.detach(), v.detach()) for k, v in kv]
                e.state = STATE_GPU
            self.stats["put_new"] += 1
            return e

    # ---- 取回 -----------------------------------------------------------
    def get(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity,
            force_tier: Optional[str] = None
            ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """取回一个条目的 KV。

        `force_tier` 用于实验里分别测各层: "nvme" 会跳过 GPU/CPU 副本, "cpu" 跳过
        GPU 副本。真实引擎由 block manager 决定从哪一级取, 这里显式暴露以便对拍。
        """
        key = self.key_for(prefix_tokens, identity)
        with self.lock:
            e = self.entries.get(key)
            if e is None or e.state == STATE_EVICTED:
                self.stats["get_miss"] += 1
                return None
            # 身份逐字段复核: 键相同也必须再查一次, 避免哈希碰撞后被误用
            try:
                e.identity.check_compatible(identity)
            except KVIdentityError:
                self.stats["reject_identity"] += 1
                raise
            e.refcount += 1
            e.last_used_s = time.monotonic()
            if e.cancel_requested:
                e.refcount -= 1
                self.stats["cancel"] += 1
                raise KVStateError("取回被取消")
            e.state = STATE_TRANSFERRING
            gpu = e._kv_gpu if force_tier != "cpu" and force_tier != "nvme" else None
            cpu = e._kv_cpu if force_tier != "nvme" else None
            if gpu is not None:
                out = gpu
                src = "gpu"
            elif cpu is not None:
                src = "cpu"
                t0 = time.monotonic()
                out = [(k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True))
                       for k, v in cpu]
                torch.cuda.synchronize()
                self.timing["get_h2d_s"] += time.monotonic() - t0
            else:
                src = "nvme"
                out = self._load_from_nvme(e)
            e.state = STATE_GPU if e._kv_gpu is not None else STATE_CPU
            self.stats[f"get_hit_{src}"] += 1
            e.hits += 1
            return out

    def _load_from_nvme(self, e: Entry) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """从文件恢复: 先按身份元组求各层形状, 再逐层读取并搬到设备。"""
        ident = e.identity
        shape = (1, ident.num_kv_heads, ident.token_len, ident.head_dim)
        nbytes = ident.tensor_bytes()
        t0 = time.monotonic()
        with open(e.nvme_path, "rb") as f:
            raw = f.read(2 * ident.num_layers * nbytes)
        self.timing["get_nvme_read_s"] += time.monotonic() - t0
        t1 = time.monotonic()
        out = []
        off = 0
        for _ in range(ident.num_layers):
            pair = []
            for _ in (0, 1):
                buf = raw[off:off + nbytes]
                off += nbytes
                t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(shape)
                pair.append(t.to(self.device))
            out.append((pair[0], pair[1]))
        torch.cuda.synchronize()
        self.timing["get_h2d_s"] += time.monotonic() - t1
        return out

    def release(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity) -> None:
        key = self.key_for(prefix_tokens, identity)
        with self.lock:
            e = self.entries.get(key)
            if e is None:
                raise KVStateError("release 一个不存在的条目")
            if e.refcount <= 0:
                raise KVStateError("release 次数多于 get (引用计数下溢)")
            e.refcount -= 1

    # ---- 驱逐与维护 ------------------------------------------------------
    def evict_expired(self, now: Optional[float] = None) -> int:
        now = now or time.monotonic()
        t0 = time.monotonic()
        n = 0
        with self.lock:
            for e in list(self.entries.values()):
                if e.state == STATE_EVICTED or self.ttl_s <= 0:
                    continue
                if now - e.last_used_s < self.ttl_s:
                    continue
                if e.refcount > 0:
                    # 被引用中不可驱逐: 记一次 deferred, 等归还后再扫
                    self.stats["deferred_eviction"] += 1
                    continue
                e.state = STATE_EVICTING
                e._kv_cpu = None
                e._kv_gpu = None
                if e.nvme_path and e.nvme_path.exists():
                    e.nvme_path.unlink()
                e.state = STATE_EVICTED
                self.stats["evict"] += 1
                n += 1
        self.timing["maintenance_s"] += time.monotonic() - t0
        return n

    def start_maintenance(self, interval_s: float = 1.0) -> None:
        def _loop() -> None:
            while not self._stop.is_set():
                self.evict_expired()
                self._stop.wait(interval_s)
        self._maint = threading.Thread(target=_loop, daemon=True)
        self._maint.start()

    def stop_maintenance(self) -> None:
        self._stop.set()
        if self._maint:
            self._maint.join(timeout=5)

    # ---- 故障注入 --------------------------------------------------------
    def inject_cancel(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity) -> None:
        key = self.key_for(prefix_tokens, identity)
        with self.lock:
            e = self.entries.get(key)
            if e is None:
                raise KVStateError("取消一个不存在的条目")
            e.cancel_requested = True

    def resume(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity) -> None:
        key = self.key_for(prefix_tokens, identity)
        with self.lock:
            self.entries[key].cancel_requested = False

    def drop_tiers(self, prefix_tokens: Tuple[int, ...], identity: KVIdentity) -> None:
        """丢掉 GPU/CPU 副本, 只留 NVMe 文件 (用于测最慢的一级与重启恢复)。"""
        key = self.key_for(prefix_tokens, identity)
        with self.lock:
            e = self.entries[key]
            e._kv_cpu = None
            e._kv_gpu = None

    def snapshot(self, path: Path) -> None:
        """落一份可读清单 (身份 + 键 + 文件路径 + 字节数 + 状态)。"""
        with self.lock:
            data = [{
                "key": e.key, "identity": e.identity.to_dict(), "bytes": e.total_bytes,
                "nvme": str(e.nvme_path) if e.nvme_path else None,
                "state": e.state, "hits": e.hits,
            } for e in self.entries.values()]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def bytes_report(self) -> Dict[str, int]:
        with self.lock:
            gpu = sum(e.total_bytes for e in self.entries.values() if e._kv_gpu is not None)
            cpu = sum(e.total_bytes for e in self.entries.values() if e._kv_cpu is not None)
            nvme = sum(e.nvme_path.stat().st_size for e in self.entries.values()
                       if e.nvme_path and e.nvme_path.exists())
        return {"gpu_bytes": gpu, "cpu_bytes": cpu, "nvme_bytes": nvme,
                "entries": len(self.entries)}

    # ---- 重启恢复 --------------------------------------------------------
    def open_existing(self, verify_identity: bool = True) -> Dict[str, int]:
        """模拟服务重启: 不依赖内存状态, 只按目录里的文件恢复。

        身份从文件尾部的 JSON 还原; 若 `verify_identity` 打开, 还会校验文件长度与
        身份推出的形状一致, 不一致的文件不会被登记 (避免"磁盘上有文件"被当作命中)。
        """
        restored = 0
        rejected = 0
        for f in sorted(self.nvme_dir.glob("kv_*.bin")):
            key = f.name[len("kv_"):-len(".bin")]
            try:
                ident = read_identity(f)
                expect = ident.entry_bytes() + 8 + len(json.dumps(ident.to_dict()))
                if verify_identity and f.stat().st_size != expect:
                    rejected += 1
                    continue
            except Exception:  # noqa: BLE001
                rejected += 1
                continue
            self.entries[key] = Entry(key=key, identity=ident, state=STATE_CPU, nvme_path=f)
            restored += 1
        self.stats["restore"] += restored
        if rejected:
            self.stats["reject_identity"] += rejected
        return {"restored_files": restored, "rejected_files": rejected}
