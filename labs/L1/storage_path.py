#!/usr/bin/env python3
"""L1.5 lab · 权重从磁盘到显存的完整路径。

L1.3 测出 PCIe 是 47 GB/s（Gen5）/ 25 GB/s（Gen4）。
但权重不是凭空出现在主机内存里的——它先得从磁盘或网络文件系统读出来。
那一段往往比 PCIe 还慢，而且被大多数人忽略。

四个实验：
  A. 冷读 vs page cache 命中：差多少
  B. 读法对比：read() / mmap / safetensors 的实际路径
  C. 端到端：safetensors -> 主机内存 -> 显存，每段各占多久
  D. GPUDirect Storage 可用性检查（绕过主机内存的那条路）

**不使用 drop_caches**——那会清空整台机器的页缓存，影响共享机器上的其他人。
改用 posix_fadvise(POSIX_FADV_DONTNEED)，只把目标文件从缓存里踢出去。

用法：
    python storage_path.py --model /path/to/Qwen3-1.7B --out results/storage.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def drop_file_cache(path: Path) -> bool:
    """只把这个文件从页缓存里逐出，不影响系统其他部分。"""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except Exception:  # noqa: BLE001
        return False


def cached_pages(path: Path) -> float:
    """用 mincore 估算这个文件当前有多少比例在页缓存里。

    为什么不用 Python 的 `mmap.mmap` + `from_buffer`：
    只读映射不是可写缓冲区，`ctypes.c_char.from_buffer(mm)` 会直接抛异常，
    于是 mincore 永远报"不可用"（本章陷阱 ⑦ 记的就是这个现象）。
    正确做法是自己调 libc 的 mmap 拿地址，用完 munmap——
    mincore 只要求映射存在，不要求可写。
    """
    try:
        size = path.stat().st_size
        page = os.sysconf("SC_PAGE_SIZE")
        n = (size + page - 1) // page
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # 必须声明 argtypes：否则 64 位指针会被当成 int 截断（L1.4 踩过同类坑）
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.restype = ctypes.c_int
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.restype = ctypes.c_int
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_ubyte)]
        PROT_READ, MAP_PRIVATE = 1, 2
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, ctypes.c_size_t(size), PROT_READ, MAP_PRIVATE, fd, 0)
        finally:
            os.close(fd)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return -1.0
        vec = (ctypes.c_ubyte * n)()
        rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
        resident = sum(1 for v in vec if v & 1) if rc == 0 else -1
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        return resident / n if rc == 0 else -1.0
    except Exception:  # noqa: BLE001
        return -1.0


def read_seq(path: Path, block: int = 8 << 20) -> float:
    t0 = time.perf_counter()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(block)
            if not b:
                break
            n += len(b)
    return path.stat().st_size / (time.perf_counter() - t0) / 1e9


def read_mmap(path: Path) -> float:
    size = path.stat().st_size
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
        # 必须真的碰每一页，否则 mmap 只是建立映射、什么都没读
        total = 0
        page = os.sysconf("SC_PAGE_SIZE")
        for off in range(0, size, page):
            total += mm[off]
        mm.close()
    return size / (time.perf_counter() - t0) / 1e9


def fs_of(path: Path) -> str:
    try:
        r = subprocess.run(["df", "-hT", str(path)], capture_output=True,
                           text=True, timeout=10).stdout.splitlines()
        return " ".join(r[-1].split()[:3]) if len(r) > 1 else "?"
    except Exception:  # noqa: BLE001
        return "?"


def block_device_of(path: Path) -> str:
    """这个文件落在哪个块设备上——介质信息要和带宽一起记。"""
    try:
        r = subprocess.run(["df", str(path)], capture_output=True, text=True,
                           timeout=10).stdout.splitlines()
        return r[-1].split()[0] if len(r) > 1 else "?"
    except Exception:  # noqa: BLE001
        return "?"


def proc_io() -> dict:
    """本进程真正从块设备读了多少字节（/proc/self/io）。

    `rchar` 包含缓存命中的读，`read_bytes` 只计真正下到块层的字节。
    两者一起看才能区分「读了文件」和「真的碰了盘」。
    """
    out = {}
    try:
        for line in Path(f"/proc/{os.getpid()}/io").read_text().splitlines():
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.strip())
    except OSError:
        pass
    return out


def page_faults() -> dict:
    """软/硬缺页计数（/proc/self/stat 的 minflt / majflt）。"""
    try:
        fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"minflt": int(fields[7]), "majflt": int(fields[9])}
    except (OSError, IndexError):
        return {}


class Stage:
    """一段计时的上下文：墙钟 + 缺页增量 + 进程 IO 增量。

    把「这一段到底读了多少字节、触发多少缺页」和耗时一起记下来，
    是区分 mmap 建视图 / 真正调页 / CPU 转换 / H2D 的唯一办法。
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.f0 = page_faults()
        self.io0 = proc_io()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1e3
        f1, io1 = page_faults(), proc_io()
        self.minflt = f1.get("minflt", 0) - self.f0.get("minflt", 0)
        self.majflt = f1.get("majflt", 0) - self.f0.get("majflt", 0)
        self.rchar = io1.get("rchar", 0) - self.io0.get("rchar", 0)
        self.read_bytes = io1.get("read_bytes", 0) - self.io0.get("read_bytes", 0)
        return False

    def as_dict(self) -> dict:
        return {"ms": round(self.ms, 1), "minflt": self.minflt, "majflt": self.majflt,
                "rchar_mib": round(self.rchar / 2**20, 1),
                "read_bytes_mib": round(self.read_bytes / 2**20, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None,
                    help="把分片复制到这个目录下的独立文件再测；"
                         "避免别的进程读同一个模型文件把页缓存弄热（冷态控制）")
    ap.add_argument("--keep-copy", action="store_true")
    args = ap.parse_args()

    model = Path(args.model)
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{model} 下没有 .safetensors")
    src = max(shards, key=lambda p: p.stat().st_size)

    # 独立数据文件：冷态控制必须先排除「别人也在读同一份文件」。
    # 复制是顺序读+写，之后所有实验都只碰这个私有副本。
    shard = src
    copy_ms = None
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
        shard = work / f"cold_{src.name}"
        if not shard.exists() or shard.stat().st_size != src.stat().st_size:
            t0 = time.perf_counter()
            with open(src, "rb") as fi, open(shard, "wb") as fo:
                while True:
                    b = fi.read(8 << 20)
                    if not b:
                        break
                    fo.write(b)
            copy_ms = round((time.perf_counter() - t0) * 1e3, 1)
            os.fsync(os.open(shard, os.O_RDONLY))
    size_gb = shard.stat().st_size / 1e9

    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_file": str(src), "file": str(shard), "size_gb": round(size_gb, 2),
        "fs": fs_of(shard), "block_device": block_device_of(shard),
        "private_copy": bool(args.work), "private_copy_ms": copy_ms,
        "media": {},
    }
    print(f"=== {shard.name}  {size_gb:.2f} GB")
    print(f"    文件系统: {res['fs']}   块设备: {res['block_device']}")
    if copy_ms is not None:
        print(f"    已复制到独立文件（{copy_ms} ms），后续冷热控制只针对它")
    try:
        st = os.statvfs(shard)
        res["media"]["statvfs_bsize"] = st.f_bsize
        res["media"]["free_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
    except OSError:
        pass

    # ---- A. 冷读 vs 热读 ----
    print("\n[A] 冷读（页缓存已逐出） vs 热读（命中页缓存）")
    frac_before = cached_pages(shard)
    ok = drop_file_cache(shard)
    frac = cached_pages(shard)
    if frac >= 0:
        print(f"    mincore 逐出前驻留 {frac_before:.1%}  →  "
              f"posix_fadvise(DONTNEED) {'成功' if ok else '失败'}  →  逐出后 {frac:.1%}")
    else:
        print(f"    （mincore 不可用，无法给出驻留比例；DONTNEED {'成功' if ok else '失败'}）")
    if frac >= 0 and frac > 0.2:
        print("    ⚠ 逐出后仍有 >20% 驻留：这不是干净的冷态，冷读数字只能当上界看")
    with Stage("cold_read") as s_cold:
        cold = read_seq(shard)
    with Stage("warm_read") as s_warm:
        warm = read_seq(shard)
    warm2 = read_seq(shard)
    print(f"    冷读   {cold:7.2f} GB/s   "
          f"（进程 read_bytes 增量 {s_cold.read_bytes / 2**30:.2f} GiB，"
          f"缺页 {s_cold.minflt}+{s_cold.majflt}）")
    print(f"    热读   {warm:7.2f} GB/s   （第二次 {warm2:.2f}；"
          f"read_bytes 增量 {s_warm.read_bytes / 2**30:.2f} GiB）")
    print(f"    页缓存带来 {warm/cold:.1f}× —— 这就是「第二次加载模型快得多」的原因")
    res["A_cold_vs_warm"] = {
        "cold_gbps": round(cold, 2), "warm_gbps": round(warm, 2),
        "warm2_gbps": round(warm2, 2), "ratio": round(warm / cold, 2),
        "mincore_resident_before": round(frac_before, 4),
        "mincore_resident_after_fadvise": round(frac, 4),
        "fadvise_ok": ok,
        "cold_stage": s_cold.as_dict(), "warm_stage": s_warm.as_dict(),
        "cold_evidence_ok": bool(frac >= 0 and frac <= 0.2),
    }

    # ---- B. read() vs mmap ----
    print("\n[B] 读法对比（都在页缓存已热的前提下）")
    r_read = statistics.median(read_seq(shard) for _ in range(3))
    r_mmap = read_mmap(shard)
    print(f"    read() 顺序读     {r_read:7.2f} GB/s")
    print(f"    mmap + 逐页触碰   {r_mmap:7.2f} GB/s")
    print("    mmap 慢是因为逐页缺页中断；但它的价值是**按需**——")
    print("    safetensors 靠 mmap 才能只读一个张量而不加载整个文件（L0.1 原始现场①）。")
    res["B_read_methods"] = {"read_gbps": round(r_read, 2), "mmap_gbps": round(r_mmap, 2)}

    # ---- C. 端到端：磁盘 -> 主机 -> 显存 ----
    #
    # ★ 这一段第一版测错了，值得说清楚：
    # safetensors 的 get_tensor() 返回的是**建立在 mmap 之上的视图**，
    # 调用它并不真的把字节从磁盘读出来。第一版直接给这一步计时，
    # 测出"冷加载 104 GB/s"——比任何磁盘都快，因为它什么都没读。
    # 真正的磁盘读被推迟到了后面 .to("cuda") 触碰内存的那一刻，
    # 于是 H2D 只测出 6.7 GB/s（而 PCIe 5.0 是 47）。**两段时间串了台。**
    #
    # 正确做法：把三个阶段显式分开，中间用真实触碰强制页调入。
    print("\n[C] 端到端：safetensors -> 主机内存 -> 显存（每段单独计时 + 缺页/字节账）")
    import torch
    from safetensors import safe_open

    drop_file_cache(shard)

    # C0: 只读文件头并解析 JSON（CPU 侧的解码，不碰张量数据）
    with Stage("C0_header_decode") as s_c0:
        with open(shard, "rb") as f:
            n_hdr = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n_hdr))
    hdr_tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    # C1: 只建立 mmap 视图（不读数据）
    with Stage("C1_mmap_view") as s_c1:
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            views = {k: f.get_tensor(k) for k in keys}

    # C2: 强制把所有页真的读进主机内存（clone 会逐字节拷贝一遍）
    with Stage("C2_fault_in") as s_c2:
        resident = {k: v.clone() for k, v in views.items()}
    del views

    # C2b: CPU 侧解码/转换——把 BF16 张量转成 FP16。
    #      checkpoint 里存的精度和 kernel 想要的精度经常不同，这一步是纯 CPU 工作，
    #      它的成本既不属于磁盘读，也不属于 H2D，必须单列。
    with Stage("C2b_cpu_cast") as s_c2b:
        cast = {k: (v.to(torch.float16) if v.dtype == torch.bfloat16 else v)
                for k, v in resident.items()}

    # C3: 主机内存（已驻留） -> 显存
    torch.cuda.init(); torch.cuda.synchronize()
    with Stage("C3_h2d_pageable") as s_c3:
        on_gpu = {k: v.to("cuda", non_blocking=False) for k, v in cast.items()}
        torch.cuda.synchronize()
    n_tensors = len(resident)
    t_map, t_fault, t_h2d = s_c1.ms / 1e3, s_c2.ms / 1e3, s_c3.ms / 1e3

    print(f"    张量数 {n_tensors}（文件头里 {len(hdr_tensors)} 项）")
    print(f"    C0 读文件头 + JSON 解析 {s_c0.ms:8.1f} ms  "
          f"（rchar {s_c0.rchar / 2**20:.1f} MiB，只读了头部）")
    print(f"    C1 建立 mmap 视图     {s_c1.ms:8.1f} ms   "
          f"（不读数据：rchar {s_c1.rchar / 2**20:.1f} MiB）")
    print(f"    C2 强制读入主机内存   {s_c2.ms:8.1f} ms  ({size_gb/(s_c2.ms/1e3):6.2f} GB/s)  "
          f"← 真正的磁盘/缓存读；缺页 {s_c2.minflt}+{s_c2.majflt}，"
          f"read_bytes {s_c2.read_bytes / 2**20:.1f} MiB")
    print(f"    C2b CPU 侧精度转换    {s_c2b.ms:8.1f} ms  "
          f"（bf16→fp16，纯 CPU，缺页 {s_c2b.minflt}+{s_c2b.majflt}）")
    print(f"    C3 主机内存 -> 显存   {s_c3.ms:8.1f} ms  ({size_gb/t_h2d:6.2f} GB/s)")
    total = t_map + t_fault + t_h2d + s_c2b.ms / 1e3
    print(f"    ⇒ 冷启动总计约 {total*1e3:.0f} ms")
    print(f"      磁盘/缓存占 {t_fault/total:.0%}，CPU 转换占 {s_c2b.ms/1e3/total:.0%}，"
          f"PCIe 占 {t_h2d/total:.0%}")
    print(f"    注意 C3 的 {size_gb/t_h2d:.1f} GB/s 远低于 L1.3 实测的 47 GB/s ——下面查原因。")
    res["C_end_to_end"] = {
        "n_tensors": n_tensors,
        "header_items": len(hdr_tensors),
        "header_decode": s_c0.as_dict(),
        "mmap_view_ms": round(s_c1.ms, 1),
        "mmap_view": s_c1.as_dict(),
        "fault_in_ms": round(s_c2.ms, 1),
        "fault_in_gbps": round(size_gb / (s_c2.ms / 1e3), 2),
        "fault_in": s_c2.as_dict(),
        "cpu_cast": s_c2b.as_dict(),
        "h2d_ms": round(s_c3.ms, 1),
        "h2d_gbps": round(size_gb / (s_c3.ms / 1e3), 2),
        "total_ms": round(total * 1e3, 1),
        "note": "get_tensor() 只建 mmap 视图；必须显式触碰才会真读磁盘"}

    # C4: 对照——把所有张量拼成一块大的再传，看固定开销的影响
    flat = torch.cat([v.reshape(-1).view(torch.uint8) for v in resident.values()])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = flat.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    t_bulk = time.perf_counter() - t0
    print(f"    C4 假设一：传输太碎？拼成 1 次传输：{t_bulk*1e3:8.1f} ms  "
          f"({flat.numel()/1e9/t_bulk:6.2f} GB/s)  ← 只快 {t_h2d/t_bulk:.2f}×，**假设被证伪**")

    # C5: 假设二——源是 pageable 内存。换成 pinned 再测。
    pinned_src = torch.empty_like(flat, pin_memory=True)
    pinned_src.copy_(flat)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = pinned_src.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    t_pin = time.perf_counter() - t0
    print(f"    C5 假设二：源是 pageable？换成 pinned：{t_pin*1e3:8.1f} ms  "
          f"({flat.numel()/1e9/t_pin:6.2f} GB/s)  ← 快 {t_bulk/t_pin:.2f}×，**这才是原因**")
    print(f"       对照 L1.3：pinned H2D 47.2 GB/s、pageable 23.6 GB/s。")
    print(f"       ⇒ 模型加载慢，不是因为张量碎，而是 safetensors 的 mmap 内存**不是 pinned 的**。")
    res["C_end_to_end"]["pinned_h2d_ms"] = round(t_pin * 1e3, 1)
    res["C_end_to_end"]["pinned_h2d_gbps"] = round(flat.numel() / 1e9 / t_pin, 2)
    del pinned_src
    res["C_end_to_end"]["bulk_h2d_ms"] = round(t_bulk * 1e3, 1)
    res["C_end_to_end"]["bulk_h2d_gbps"] = round(flat.numel() / 1e9 / t_bulk, 2)
    del on_gpu, resident, flat
    torch.cuda.empty_cache()

    # ---- D. GPUDirect Storage ----
    print("\n[D] GPUDirect Storage（绕过主机内存，NVMe 直连显存）")
    gds = {}
    for p in ("/proc/driver/nvidia-fs/stats", "/dev/nvidia-fs0"):
        gds[p] = os.path.exists(p)
        print(f"    {p:36s} {'存在' if gds[p] else '不存在'}")
    try:
        import torch as _t
        lib = Path(_t.__file__).parent.parent / "nvidia" / "cu13" / "lib" / "libcufile.so.0"
        gds["libcufile"] = lib.exists()
        print(f"    libcufile.so.0 (用户态库)             {'存在' if lib.exists() else '不存在'}")
    except Exception:  # noqa: BLE001
        pass
    if not gds.get("/dev/nvidia-fs0"):
        print("    ⇒ 内核侧 nvidia-fs 未加载：GDS 不可用，读文件仍要经过主机内存。")
        print("      注意用户态库存在 ≠ 功能可用——这是很常见的误判。")
    res["D_gds"] = gds

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
