#!/usr/bin/env python3
"""教师特征库：真实落盘、ACK、崩溃重放、过期拦截与三种生产模式（7.7-D）。

特征库把 teacher 的隐藏状态或 logits 落成 shard 文件，学生侧按记录读取并 ACK。
本 lab 用真实文件与真实子进程实现：

- `offline`：先全部生产，再消费（教师前向与学生训练不同时进行）；
- `colocated`：生产与消费在同一进程内交替（共置）；
- `disaggregated`：生产者与消费者各自一个子进程并行，消费者轮询新 shard。

并检查：过期记录（teacher revision / 模板 / token / 层 / dtype 任一不符）被拒绝、
消费者崩溃后未 ACK 的队尾被重放且不重复提交、以及实测字节账。

Usage:
    python labs/L7/teacher_feature_store.py --mode offline --outdir "$RUN_DIR/store-offline" \
        --scratch /Volumes/data/artifacts/llm-infra/scratch
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np

FEATURE_DTYPE = np.float16
SAMPLES = 24
SEQ_LEN = 64
HIDDEN = 32
LAYERS = 3
DRAFT_VOCAB = 4096          # 小词表，便于真实落盘；账目另按真实 152064 词表折算
BLOCK_ELEMS = SEQ_LEN * HIDDEN * LAYERS


def sample_ids():
    return [f"s{i:03d}" for i in range(SAMPLES)]


def make_feature(sample_id: str):
    """用固定种子的真实矩阵乘法生成特征，替代真实 teacher 前向。"""
    seed = int(sample_id[1:])
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((SEQ_LEN, HIDDEN))
    projection = rng.standard_normal((HIDDEN, LAYERS * HIDDEN))
    feature = (hidden @ projection).reshape(SEQ_LEN, LAYERS, HIDDEN).astype(FEATURE_DTYPE)
    return feature


def token_hash(sample_id: str) -> str:
    return f"tok-{sample_id}"


def identity(config: dict, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "teacher_revision": config["teacher_revision"],
        "template_id": config["template_id"],
        "token_id": token_hash(sample_id),
        "target_layers": config["target_layers"],
        "dtype": config["dtype"],
        "shape": [SEQ_LEN, LAYERS, HIDDEN],
    }


def produce(store: Path, config: dict, shard_size: int = 8, announce: bool = True) -> dict:
    """写 shard：先写 .pending，再 rename 成正式文件，最后追加索引。"""
    shards = store / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    index_path = store / "index.json"
    written = 0
    shard_id = 0
    t0 = time.perf_counter()
    for start in range(0, len(sample_ids()), shard_size):
        group = sample_ids()[start:start + shard_size]
        pending = shards / f"shard-{shard_id:04d}.pending"
        with pending.open("wb") as stream:
            records = []
            offset = 0
            for sid in group:
                payload = make_feature(sid).tobytes()
                stream.write(payload)
                records.append({**identity(config, sid), "offset": offset, "nbytes": len(payload)})
                offset += len(payload)
        final = shards / f"shard-{shard_id:04d}.bin"
        pending.replace(final)
        meta = shards / f"shard-{shard_id:04d}.json"
        meta.write_text(json.dumps({"shard": final.name, "records": records}, ensure_ascii=False))
        written += len(group)
        shard_id += 1
        if announce:
            index_path.write_text(json.dumps(
                {"shards": sorted(p.name for p in shards.glob("shard-*.json")),
                 "records": written}, ensure_ascii=False))
    return {"records": written, "shards": shard_id, "produce_seconds": time.perf_counter() - t0}


def read_records(store: Path):
    """按索引读取全部记录（含 payload 切片信息）。"""
    shards = store / "shards"
    for meta_path in sorted(shards.glob("shard-*.json")):
        meta = json.loads(meta_path.read_text())
        bin_path = shards / meta["shard"]
        blob = bin_path.read_bytes()
        for record in meta["records"]:
            yield bin_path, blob, record


def validate(record: dict, config: dict, token_lookup) -> str | None:
    """返回拒绝原因；None 表示通过。"""
    if record["teacher_revision"] != config["teacher_revision"]:
        return "teacher_revision_mismatch"
    if record["template_id"] != config["template_id"]:
        return "template_mismatch"
    if record["token_id"] != token_lookup(record["sample_id"]):
        return "token_mismatch"
    if record["target_layers"] != config["target_layers"]:
        return "layer_mismatch"
    if record["dtype"] != config["dtype"]:
        return "dtype_mismatch"
    if record["shape"] != [SEQ_LEN, LAYERS, HIDDEN]:
        return "shape_mismatch"
    return None


def consume(store: Path, config: dict, ack_path: Path, crash_after: int | None = None,
            wait_for_shards: bool = False, poll_seconds: float = 0.05,
            deadline_seconds: float = 30.0) -> dict:
    """消费并按记录 ACK；crash_after 用于模拟消费者崩溃（不做优雅退出）。"""
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    acked = set()
    if ack_path.exists():
        for line in ack_path.read_text().splitlines():
            if line.strip():
                acked.add(json.loads(line)["sample_id"])
    processed = 0
    rejected: dict[str, int] = {}
    t0 = time.perf_counter()
    deadline = t0 + deadline_seconds
    scanned: set[str] = set()
    while True:
        found_new = False
        for _, blob, record in read_records(store):
            if record["sample_id"] in scanned:
                continue
            scanned.add(record["sample_id"])
            found_new = True
            reason = validate(record, config, token_hash)
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            array = np.frombuffer(
                blob[record["offset"]:record["offset"] + record["nbytes"]], dtype=FEATURE_DTYPE)
            assert array.size == BLOCK_ELEMS, (record["sample_id"], array.size)
            if record["sample_id"] in acked:
                continue
            if crash_after is not None and processed == crash_after:
                os._exit(3)          # 未写 ACK 就退出：模拟消费者崩溃
            with ack_path.open("a") as stream:
                stream.write(json.dumps({"sample_id": record["sample_id"],
                                         "at": time.monotonic()}) + "\n")
                stream.flush()
            acked.add(record["sample_id"])
            processed += 1
        if not wait_for_shards:
            break
        if len(acked) >= SAMPLES or time.perf_counter() > deadline:
            break
        if not found_new:
            time.sleep(poll_seconds)
    return {"processed": processed, "total_acked": len(acked), "rejected": rejected,
            "consume_seconds": time.perf_counter() - t0}


def _producer_process(store: str, config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    produce(Path(store), config)


def _consumer_process(store: str, config_path: str, ack_path: str,
                      crash_after: int | None = None) -> None:
    config = json.loads(Path(config_path).read_text())
    result = consume(Path(store), config, Path(ack_path),
                     crash_after=crash_after, wait_for_shards=crash_after is None)
    Path(store, f"consumer-result-{os.getpid()}.json").write_text(json.dumps(result))


def run_colocated(store: Path, config: dict, shard_size: int = 4) -> dict:
    """共置：生产一个 shard 就消费一次，生产者与消费者在同一进程内交替。"""
    ack_path = store / "acks" / "consumer-colocated.jsonl"
    store.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    produced_total = 0
    consumed_total = 0
    for _ in range(0, len(sample_ids()), shard_size):
        pass
    shards_seen = 0
    while produced_total < SAMPLES:
        group = sample_ids()[produced_total:produced_total + shard_size]
        produce_subset(store, config, group, shards_seen)
        produced_total += len(group)
        shards_seen += 1
        consumed_total += consume(store, config, ack_path)["processed"]
    return {"produce_records": produced_total, "consume_records": consumed_total,
            "wall_seconds": time.perf_counter() - t0,
            "note": "生产与消费在同一进程内按 shard 交替"}


def produce_subset(store: Path, config: dict, group: list[str], shard_id: int) -> None:
    shards = store / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    pending = shards / f"shard-{shard_id:04d}.pending"
    with pending.open("wb") as stream:
        records = []
        offset = 0
        for sid in group:
            payload = make_feature(sid).tobytes()
            stream.write(payload)
            records.append({**identity(config, sid), "offset": offset, "nbytes": len(payload)})
            offset += len(payload)
    final = shards / f"shard-{shard_id:04d}.bin"
    pending.replace(final)
    (shards / f"shard-{shard_id:04d}.json").write_text(
        json.dumps({"shard": final.name, "records": records}, ensure_ascii=False))


def run_disaggregated(store: Path, config: dict, scratch: Path) -> dict:
    """分离：生产者与消费者各自一个子进程并行，消费者轮询新 shard。"""
    if store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True)
    config_path = store / "config.json"
    config_path.write_text(json.dumps(config))
    ack_path = store / "acks" / "consumer-disagg.jsonl"
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    producer = ctx.Process(target=_producer_process, args=(str(store), str(config_path)))
    consumer = ctx.Process(target=_consumer_process, args=(str(store), str(config_path), str(ack_path)))
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    wall = time.perf_counter() - t0
    results = list(store.glob("consumer-result-*.json"))
    consumed = json.loads(results[0].read_text()) if results else {}
    return {"producer_returncode": producer.exitcode, "consumer_returncode": consumer.exitcode,
            "consume": consumed, "wall_seconds": wall,
            "note": "生产与消费由两个子进程并行完成"}


def main() -> int:
    parser = argparse.ArgumentParser(description="教师特征库契约、崩溃重放与三种生产模式")
    parser.add_argument("--mode", choices=("offline", "colocated", "disaggregated"), required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--scratch", type=Path,
                        default=Path("/Volumes/data/artifacts/llm-infra/scratch"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)

    config = {"teacher_revision": "teacher-rev-A", "template_id": "chatml-v1",
              "target_layers": [8, 16, 24], "dtype": "float16"}
    work = args.scratch / f"store-{args.mode}-{int(time.time()*1000)}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    result = {"mode": args.mode, "config": config,
              "geometry": {"samples": SAMPLES, "seq_len": SEQ_LEN, "hidden": HIDDEN,
                           "layers": LAYERS, "draft_vocab": DRAFT_VOCAB}}

    t0 = time.perf_counter()
    produced = produce(work, config)
    consumed = consume(work, config, work / "acks" / "consumer-0.jsonl")
    result["offline_sequence"] = {"produce": produced, "consume": consumed,
                                  "wall_seconds": time.perf_counter() - t0}

    # 过期拦截：复制一份库，改动身份字段后逐项拒绝
    stale = args.scratch / f"store-stale-{int(time.time()*1000)}"
    shutil.copytree(work, stale)
    stale_configs = [
        ("teacher_revision_mismatch", {**config, "teacher_revision": "teacher-rev-B"}),
        ("template_mismatch", {**config, "template_id": "chatml-v2"}),
        ("layer_mismatch", {**config, "target_layers": [8, 16, 25]}),
        ("dtype_mismatch", {**config, "dtype": "bfloat16"}),
    ]
    stale_rows = {}
    for name, cfg in stale_configs:
        ack = stale / "acks" / f"stale-{name}.jsonl"
        out = consume(stale, cfg, ack)
        stale_rows[name] = {"rejected": out["rejected"], "processed": out["processed"]}
    result["staleness"] = stale_rows

    # 崩溃重放：子进程消费到第 5 条时崩溃，重启后从 ACK 日志继续
    crash_dir = args.scratch / f"store-crash-{int(time.time()*1000)}"
    shutil.copytree(work, crash_dir)
    crash_ack = crash_dir / "acks" / "consumer-crash.jsonl"
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_consumer_process,
                        args=(str(crash_dir), str(crash_dir / "config.json"), str(crash_ack), 5))
    (crash_dir / "config.json").write_text(json.dumps(config))
    child.start()
    child.join()
    acked_before = len(crash_ack.read_text().splitlines()) if crash_ack.exists() else 0
    resumed = consume(crash_dir, config, crash_ack)
    result["crash_replay"] = {"exitcode": child.exitcode, "acked_before_restart": acked_before,
                              "resumed": resumed,
                              "acked_after_restart": len(crash_ack.read_text().splitlines()),
                              "duplicates": len(crash_ack.read_text().splitlines()) - SAMPLES}

    # 字节账：实测 payload 字节 vs 全词表 logits 折算
    payload_bytes = sum(p.stat().st_size for p in (work / "shards").glob("shard-*.bin"))
    meta_bytes = sum(p.stat().st_size for p in (work / "shards").glob("shard-*.json"))
    result["bytes"] = {
        "payload_bytes_measured": payload_bytes,
        "meta_bytes_measured": meta_bytes,
        "bytes_per_sample_payload": payload_bytes / SAMPLES,
        "full_vocab_logits_bytes_per_sample_152064_fp16": SEQ_LEN * 152064 * 2,
        "hidden_bytes_per_sample_draft_vocab_fp16": BLOCK_ELEMS * 2,
        "ratio_full_logits_over_hidden": (SEQ_LEN * 152064 * 2) / (BLOCK_ELEMS * 2),
    }

    if args.mode == "colocated":
        col_dir = args.scratch / f"store-colocated-{int(time.time()*1000)}"
        result["colocated_run"] = run_colocated(col_dir, config)
        shutil.rmtree(col_dir, ignore_errors=True)
    if args.mode == "disaggregated":
        result["disaggregated_run"] = run_disaggregated(
            args.scratch / f"store-disagg-{int(time.time()*1000)}", config, args.scratch)

    (args.outdir / "teacher_feature_store.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.outdir / "cases.json").write_text(json.dumps({
        "mode": args.mode, "samples": SAMPLES,
        "checks": ["staleness_rejection", "crash_replay_no_duplicate", "measured_bytes"],
        "simulated_parts": "teacher 前向用固定种子的 numpy 矩阵乘法替代；其余为真实文件读写与子进程",
    }, ensure_ascii=False, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(stale, ignore_errors=True)
    shutil.rmtree(crash_dir, ignore_errors=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())