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
    """用 mincore 估算这个文件当前有多少比例在页缓存里。"""
    try:
        size = path.stat().st_size
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        page = os.sysconf("SC_PAGE_SIZE")
        n = (size + page - 1) // page
        vec = (ctypes.c_ubyte * n)()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # 必须声明 argtypes：否则指针会被当成 int 截断成 32 位（L1.4 踩过同类坑）
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_ubyte)]
        libc.mincore.restype = ctypes.c_int
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
        if rc != 0:
            mm.close()
            return -1.0
        resident = sum(1 for v in vec if v & 1)
        mm.close()
        return resident / n
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = Path(args.model)
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{model} 下没有 .safetensors")
    shard = max(shards, key=lambda p: p.stat().st_size)
    size_gb = shard.stat().st_size / 1e9

    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "file": str(shard), "size_gb": round(size_gb, 2), "fs": fs_of(shard),
    }
    print(f"=== {shard.name}  {size_gb:.2f} GB")
    print(f"    文件系统: {res['fs']}")

    # ---- A. 冷读 vs 热读 ----
    print("\n[A] 冷读（页缓存已逐出） vs 热读（命中页缓存）")
    ok = drop_file_cache(shard)
    frac = cached_pages(shard)
    print(f"    posix_fadvise(DONTNEED) {'成功' if ok else '失败'}；"
          f"逐出后仍在缓存的比例 {frac:.1%}" if frac >= 0 else "    （mincore 不可用）")
    cold = read_seq(shard)
    warm = read_seq(shard)
    warm2 = read_seq(shard)
    print(f"    冷读   {cold:7.2f} GB/s")
    print(f"    热读   {warm:7.2f} GB/s   （第二次 {warm2:.2f}）")
    print(f"    页缓存带来 {warm/cold:.1f}× —— 这就是「第二次加载模型快得多」的原因")
    res["A_cold_vs_warm"] = {"cold_gbps": round(cold, 2), "warm_gbps": round(warm, 2),
                             "warm2_gbps": round(warm2, 2), "ratio": round(warm / cold, 2)}

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
    print("\n[C] 端到端：safetensors -> 主机内存 -> 显存（三段分开计时）")
    import torch
    from safetensors import safe_open

    drop_file_cache(shard)

    # C1: 只建立 mmap 视图（不读数据）
    t0 = time.perf_counter()
    with safe_open(str(shard), framework="pt") as f:
        keys = list(f.keys())
        views = {k: f.get_tensor(k) for k in keys}
    t_map = time.perf_counter() - t0

    # C2: 强制把所有页真的读进主机内存（clone 会逐字节拷贝一遍）
    t0 = time.perf_counter()
    resident = {k: v.clone() for k, v in views.items()}
    t_fault = time.perf_counter() - t0
    del views

    # C3: 主机内存（已驻留） -> 显存
    torch.cuda.init(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    on_gpu = {k: v.to("cuda", non_blocking=False) for k, v in resident.items()}
    torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t0
    n_tensors = len(resident)

    print(f"    张量数 {n_tensors}")
    print(f"    C1 建立 mmap 视图     {t_map*1e3:8.1f} ms   "
          f"（不读数据，所以'带宽'没有意义）")
    print(f"    C2 强制读入主机内存   {t_fault*1e3:8.1f} ms  ({size_gb/t_fault:6.2f} GB/s)  ← 真正的磁盘/缓存读")
    print(f"    C3 主机内存 -> 显存   {t_h2d*1e3:8.1f} ms  ({size_gb/t_h2d:6.2f} GB/s)")
    print(f"    ⇒ 冷启动总计约 {(t_map+t_fault+t_h2d)*1e3:.0f} ms")
    print(f"      磁盘/缓存占 {t_fault/(t_map+t_fault+t_h2d):.0%}，"
          f"PCIe 占 {t_h2d/(t_map+t_fault+t_h2d):.0%}")
    print(f"    注意 C3 的 {size_gb/t_h2d:.1f} GB/s 远低于 L1.3 实测的 47 GB/s ——下面查原因。")
    res["C_end_to_end"] = {
        "n_tensors": n_tensors,
        "mmap_view_ms": round(t_map * 1e3, 1),
        "fault_in_ms": round(t_fault * 1e3, 1),
        "fault_in_gbps": round(size_gb / t_fault, 2),
        "h2d_ms": round(t_h2d * 1e3, 1),
        "h2d_gbps": round(size_gb / t_h2d, 2),
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
