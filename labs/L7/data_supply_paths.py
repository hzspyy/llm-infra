#!/usr/bin/env python3
"""供数路径：三种存储布局的结构代价、真实 DataLoader 的重叠与背压、坏样本与游标。

第 1 节把同一份 2000 条样本写成 Parquet / tar shard / 预分词 mmap，
比较"取一条随机样本要读多少字节、解多少行"以及需要哪些辅助索引。
第 2 节用真实 torch DataLoader（注入读取延迟）量 worker 与 prefetch 的重叠和 straggler。
第 3 节注入坏样本、缺文件、重复 ID 与 worker 崩溃，检查 consumed/acknowledged 游标。

数据文件写在 --workdir 指定的目录（默认走学习盘），运行结束后可直接删除。

Usage:
    python labs/L7/data_supply_paths.py --workdir "$RUN_DIR/supply" > "$RUN_DIR/supply.txt"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import struct
import tarfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

N_SAMPLES = 2000
ROW_GROUP = 256
VOCAB = 50000


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def make_corpus(seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(N_SAMPLES):
        n = max(16, int(rng.lognormvariate(0, 0.5) * 256))
        out.append({"id": f"doc_{i:05d}",
                    "source": rng.choice(["web", "book", "code"]),
                    "tokens": [rng.randrange(VOCAB) for _ in range(n)]})
    return out


# --------------------------------------------------------------------------
# 1. 三种存储布局
# --------------------------------------------------------------------------

def write_parquet(corpus, path: Path):
    table = pa.table({"id": [d["id"] for d in corpus],
                      "source": [d["source"] for d in corpus],
                      "tokens": pa.array([d["tokens"] for d in corpus],
                                         type=pa.list_(pa.uint16()))})
    pq.write_table(table, path, row_group_size=ROW_GROUP, compression="zstd")


def write_tar(corpus, path: Path, index_path: Path):
    """写 tar，再完整扫一遍成员头建外部索引——tar 自身没有目录结构。"""
    with tarfile.open(path, "w") as tar:
        for d in corpus:
            payload = json.dumps(d).encode()
            info = tarfile.TarInfo(name=f"{d['id']}.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    index = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            index[member.name.removesuffix(".json")] = [member.offset_data, member.size]
    index_path.write_text(json.dumps(index))
    return len(index)


def write_mmap(corpus, bin_path: Path, idx_path: Path):
    offsets = [0]
    with bin_path.open("wb") as stream:
        for d in corpus:
            stream.write(np.asarray(d["tokens"], dtype=np.uint16).tobytes())
            offsets.append(offsets[-1] + len(d["tokens"]) * 2)
    with idx_path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(corpus)))
        stream.write(np.asarray(offsets, dtype=np.int64).tobytes())


def probe_storage(work: Path, corpus):
    parquet = work / "corpus.parquet"
    tar_path, tar_index = work / "corpus.tar", work / "corpus.index.json"
    bin_path, idx_path = work / "corpus.bin", work / "corpus.idx"
    write_parquet(corpus, parquet)
    write_tar(corpus, tar_path, tar_index)
    write_mmap(corpus, bin_path, idx_path)

    target = 1337                                   # 固定的随机取样目标
    pf = pq.ParquetFile(parquet)
    rg = target // ROW_GROUP
    meta = pf.metadata.row_group(rg)
    pq_rows = meta.num_rows
    pq_compressed = sum(meta.column(c).total_compressed_size for c in range(meta.num_columns))

    index = json.loads(tar_index.read_text())
    offset, size = index[corpus[target]["id"]]
    with tar_path.open("rb") as stream:
        member = os.pread(stream.fileno(), size, offset)
    tar_bytes = size
    # 没有索引时必须顺着 512 字节的头一路跳过去
    headers_scanned = target + 1

    with idx_path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        offsets = np.frombuffer(stream.read(count * 8 + 8), dtype=np.int64)
    start, end = int(offsets[target]), int(offsets[target + 1])
    with bin_path.open("rb") as stream:
        raw = os.pread(stream.fileno(), end - start, start)
    mmap_bytes = end - start

    print(f"目标样本 {corpus[target]['id']}，{len(corpus[target]['tokens'])} 个 token"
          f"（原始 {len(corpus[target]['tokens']) * 2} 字节）")
    print("\n格式          | 文件大小 | 随机取 1 条需读 | 需解码的行数 | 辅助索引 | 变长字段")
    print(f"Parquet       | {parquet.stat().st_size:8d} | {pq_compressed:15d} | {pq_rows:12d} |"
          f" 文件内 footer | 原生支持 list")
    print(f"tar shard     | {tar_path.stat().st_size:8d} | {tar_bytes:15d} | {1:12d} |"
          f" 外部 {tar_index.stat().st_size} B | 每条一个成员")
    print(f"预分词 mmap   | {bin_path.stat().st_size:8d} | {mmap_bytes:15d} | {1:12d} |"
          f" 外部 {idx_path.stat().st_size} B | 只存定宽 token")
    print(f"\n没有外部索引时，tar 要顺序跳过 {headers_scanned} 个 512 字节头才能定位这一条；"
          "\nParquet 的最小读取单位是 row group（本例 {} 行），要拿 1 条就得解 {} 条。".format(ROW_GROUP, pq_rows))
    decoded = json.loads(member)
    assert decoded["id"] == corpus[target]["id"]
    assert np.frombuffer(raw, dtype=np.uint16).tolist() == corpus[target]["tokens"]
    print("三条路径取到的内容一致（tar 的 JSON 与 mmap 的 uint16 已逐元素核对）。")

    t0 = time.perf_counter()
    total_tokens = 0
    for rg_i in range(pf.metadata.num_row_groups):
        total_tokens += sum(len(x) for x in pf.read_row_group(rg_i, columns=["tokens"])["tokens"].to_pylist())
    seq_parquet = time.perf_counter() - t0
    t0 = time.perf_counter()
    arr = np.fromfile(bin_path, dtype=np.uint16)
    seq_mmap = time.perf_counter() - t0
    assert total_tokens == arr.size
    print(f"\n顺序全量扫描 {arr.size} 个 token：Parquet 解码 {seq_parquet * 1e3:.1f} ms，"
          f"mmap 直读 {seq_mmap * 1e3:.1f} ms")
    print("这两个数是本机页缓存已热的 CPU 解码开销，不是存储带宽；"
          "跨机/对象存储的读取代价要在目标环境单独测。")
    return {"parquet": parquet, "bin": bin_path, "idx": idx_path}


# --------------------------------------------------------------------------
# 2. 真实 DataLoader 的重叠与背压
# --------------------------------------------------------------------------

class DelayedDataset(Dataset):
    """每条样本注入固定读取延迟；straggler_index 上的样本慢 10 倍。"""

    def __init__(self, n: int, read_ms: float, straggler_index: int | None = None):
        self.n, self.read_ms, self.straggler_index = n, read_ms, straggler_index

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        delay = self.read_ms * (10 if i == self.straggler_index else 1)
        time.sleep(delay / 1000.0)
        return torch.full((8,), float(i))


def build_loader(dataset, workers, prefetch, batch_size=8, persistent=False):
    kwargs = {"batch_size": batch_size, "num_workers": workers}
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
        kwargs["persistent_workers"] = persistent
    return DataLoader(dataset, **kwargs)


def drain(loader, compute_ms):
    """返回 (worker 启动耗时, 稳态循环耗时, 每步等待列表)。启动与稳态分开计时。"""
    t0 = time.perf_counter()
    it = iter(loader)                       # spawn 平台在这里创建 worker 进程
    startup = (time.perf_counter() - t0) * 1000
    waits, t_loop = [], time.perf_counter()
    while True:
        t1 = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            break
        waits.append((time.perf_counter() - t1) * 1000)
        time.sleep(compute_ms / 1000.0)
    return startup, (time.perf_counter() - t_loop) * 1000, waits


def stats(startup, loop_ms, waits):
    s = sorted(waits)
    return {"startup_ms": startup, "loop_ms": loop_ms, "steps": len(waits),
            "wait_p50": s[len(s) // 2], "wait_p99": s[int(len(s) * 0.99)],
            "wait_max": s[-1], "wait_sum": sum(s)}


def supply_experiments(read_ms, compute_ms, n):
    batch = 8
    print(f"每条样本读取 {read_ms} ms（batch={batch} 即纯读取 {read_ms * batch} ms），"
          f"每步计算 {compute_ms} ms，共 {n // batch} 步")
    print(f"本机 multiprocessing 启动方式：{torch.multiprocessing.get_start_method()}；"
          "worker>0 的行先跑一个 epoch 预热（结果丢弃），只报第 2 个 epoch\n")
    print("workers | prefetch | 第 2 epoch 启动 ms | 稳态循环 ms | 等待 p50 ms | p99 ms | 等待占稳态")
    for workers, prefetch in ((0, 0), (2, 2), (2, 6), (4, 2)):
        loader = build_loader(DelayedDataset(n, read_ms), workers, prefetch, batch,
                              persistent=workers > 0)
        if workers > 0:
            drain(loader, compute_ms)                  # 预热：付掉 spawn 与 import 的一次性成本
        r = stats(*drain(loader, compute_ms))
        print(f"{workers:7d} | {prefetch:8d} | {r['startup_ms']:18.1f} | {r['loop_ms']:11.1f} |"
              f" {r['wait_p50']:11.2f} | {r['wait_p99']:6.1f} | {r['wait_sum'] / r['loop_ms']:10.1%}")
    steps = n // batch
    print(f"\n计算本身的下限是 {steps * compute_ms:.0f} ms；"
          f"workers=0 的预期是 {steps} × ({compute_ms} + {read_ms * batch}) = "
          f"{steps * (compute_ms + read_ms * batch):.0f} ms。")
    print("多 worker 把读取藏进上一步的计算里，等待 p50 掉到 1 ms 以下，稳态循环收敛到计算下限；"
          "\n代价是内存里多驻留 workers×prefetch×batch 条样本。")

    section("2b. worker 启动是每个 epoch 一次的固定成本")
    for persistent in (False, True):
        loader = build_loader(DelayedDataset(n, read_ms), 2, 2, batch, persistent=persistent)
        first = drain(loader, compute_ms)
        second = drain(loader, compute_ms)
        print(f"  persistent_workers={persistent}: 第 1 个 epoch 的 iter() {first[0]:.1f} ms，"
              f"第 2 个 epoch {second[0]:.1f} ms，两个 epoch 的循环 "
              f"{first[1]:.0f} / {second[1]:.0f} ms")
    print("  spawn 平台上 iter() 只负责创建进程就返回，子进程 import torch 的开销落在"
          "\n  最初几个 batch 的等待和整机 CPU 竞争上；短 epoch 的墙钟对照会因此把"
          "\n  「多 worker」误判成更慢。启动与稳态必须分开计时。")

    section("2c. straggler：一条样本慢 10 倍")
    for workers in (0, 2):
        loader = build_loader(DelayedDataset(n, read_ms, straggler_index=n // 2),
                              workers, 2, batch, persistent=workers > 0)
        if workers > 0:
            drain(loader, compute_ms)
        r = stats(*drain(loader, compute_ms))
        print(f"  workers={workers}: 稳态循环 {r['loop_ms']:.1f} ms，"
              f"等待 p50 {r['wait_p50']:.2f} ms，p99 {r['wait_p99']:.1f} ms，"
              f"最大 {r['wait_max']:.1f} ms")
    print("  单条慢样本不改变 p50，只抬高尾部；平均吞吐掩盖它，逐步等待时间才能定位。")


# --------------------------------------------------------------------------
# 3. 坏样本、重复、worker 崩溃与游标
# --------------------------------------------------------------------------

class FaultyDataset(Dataset):
    """在固定下标注入解码失败与缺文件。"""

    BAD_DECODE = {13, 57}
    MISSING = {31}

    def __init__(self, n: int, policy: str):
        self.n, self.policy = n, policy
        self.rejected = []

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i in self.BAD_DECODE or i in self.MISSING:
            reason = "decode_error" if i in self.BAD_DECODE else "missing_file"
            if self.policy == "raise":
                raise RuntimeError(f"sample {i}: {reason}")
            self.rejected.append({"index": i, "reason": reason})
            if self.policy == "skip":
                return None                      # 交给 collate 丢掉
            return torch.zeros(8)                # placeholder：保持形状与步数
        return torch.full((8,), float(i))


def skip_collate(items):
    kept = [x for x in items if x is not None]
    return torch.stack(kept) if kept else None


def fault_experiments():
    n, batch = 80, 8
    print("注入：解码失败 {13, 57}、缺文件 {31}，共 3 条坏样本，数据集 80 条、batch 8")
    for policy, collate in (("skip", skip_collate), ("placeholder", None)):
        ds = FaultyDataset(n, policy)
        loader = DataLoader(ds, batch_size=batch, num_workers=0,
                            collate_fn=collate if collate else None)
        shapes = [tuple(b.shape) for b in loader if b is not None]
        print(f"\n  policy={policy}: 产出 {len(shapes)} 个 batch，形状 {sorted(set(shapes))}，"
              f"记录 rejected {len(ds.rejected)} 条 {ds.rejected}")
    print("\n  skip 让含坏样本的 batch 变短（8→7），步数不变但每步有效元素数变了；"
          "\n  placeholder 保持形状，代价是要把占位样本从 loss mask 里排除，否则它成为真实监督。")

    section("3b. 两个 rank 各自静默跳过时会发生什么")
    per_rank = {0: list(range(0, n, 2)), 1: list(range(1, n, 2))}
    bad = FaultyDataset.BAD_DECODE | FaultyDataset.MISSING
    for rank, ids in per_rank.items():
        kept = [i for i in ids if i not in bad]
        print(f"  rank {rank}: 分到 {len(ids)} 条，跳过 {len(ids) - len(kept)} 条，"
              f"剩 {len(kept)} 条 → {len(kept) // batch} 个完整 batch（余 {len(kept) % batch}）")
    print("  两个 rank 的完整 batch 数不同时，先跑完的 rank 会在下一次 collective 上等到超时；"
          "\n  坏样本必须在全局层面统一处理（同步跳过整步，或补齐到同样的步数），不能各 rank 自行 continue。")

    section("3c. worker 抛异常时的真实报错")
    ds = FaultyDataset(n, "raise")
    try:
        for _ in DataLoader(ds, batch_size=batch, num_workers=2):
            pass
    except Exception as exc:                            # noqa: BLE001 - 这里要的就是原文
        text = str(exc).strip().splitlines()
        print(f"  {type(exc).__name__}: {text[0]}")
        if len(text) > 1:
            print(f"  ... 末行: {text[-1]}")
    print("  worker 进程里的异常被重新抛回主进程，整个 epoch 终止；"
          "\n  想跳过就必须在 __getitem__ 里显式处理，DataLoader 没有「忽略坏样本」的开关。")

    section("3d. consumed 与 acknowledged 是两个游标")
    prefetched, committed = 0, 0
    timeline = []
    for step in range(6):
        prefetched += batch * 2                      # 预取跑在前面
        if step < 4:                                 # 第 4、5 步的更新还没提交
            committed += batch
        timeline.append((step, prefetched, committed))
    for step, p, c in timeline:
        print(f"  step {step}: 已读取到样本 {p}，已提交到样本 {c}，在途 {p - c}")
    last = timeline[-1]
    print(f"\n  从 checkpoint 恢复要回到 acknowledged={last[2]}，"
          f"于是样本 [{last[2]}, {last[1]}) 共 {last[1] - last[2]} 条会被重新消费。")
    print("  把 prefetch 指针写进 checkpoint 会漏掉这批样本；"
          "把它们当成「已训练」则统计数据消耗时多算。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="写小型数据文件的目录（学习盘上），运行后可删")
    parser.add_argument("--read-ms", type=float, default=4.0)
    parser.add_argument("--compute-ms", type=float, default=30.0)
    parser.add_argument("--loader-samples", type=int, default=320)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    section("1. 三种存储布局：取一条随机样本的结构代价")
    corpus = make_corpus()
    probe_storage(args.workdir, corpus)

    section("2. 真实 DataLoader：worker、prefetch 与重叠")
    supply_experiments(args.read_ms, args.compute_ms, args.loader_samples)

    section("3. 坏样本、缺文件与游标")
    fault_experiments()


if __name__ == "__main__":
    main()
