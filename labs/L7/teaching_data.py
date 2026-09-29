#!/usr/bin/env python3
"""教学项目的数据产品（7.8-I）：从公开语料生成可复现的预训练与 SFT 数据集。

这个脚本不是"再写一个 dataloader"，而是把 7.8 的小型 contract 接到真实语料上：

  pretrain  公开 pretrain jsonl -> 长度/字符/复读过滤 -> reservoir 采样 ->
            精确去重 + MinHash band 近似去重 -> 按内容哈希划分 train/val/test ->
            项目 tokenizer 编码 -> 每条文档 BOS/EOS 后拼成 uint16 分片，并留下
            文档边界偏移，供打包窗口的跨文档统计与按文档评测使用。

  sft       公开 SFT jsonl -> 确定性对话模板 -> 逐 token label（只监督 assistant 段）
            -> 固定长度 int32 张量 + 每个样本的有效回答 token 账。

  dpo       公开偏好 jsonl -> 结构检查 + 有限词表粗筛 -> reservoir -> 按内容哈希划分
            train/val/test；不做长度归一化，也不改写 chosen/rejected 文本。

边界（写进 manifest，也写进正文）：
  * 语料是上游公布的再清洗文件，不是原始网页；本脚本只做本课程声明的过滤与去重，
    去重口径是"入选 reservoir 内"，不是对上游全库的重新清洗。
  * 划分按文档内容哈希，train/val/test 文档隔离，但不是上游原始划分。
  * 预训练打包允许同一窗口内跨文档 attention，脚本给出跨文档窗口的实测占比；
    SFT 不打包，逐样本 padding 并只在 assistant 段计 loss。

Usage:
    python labs/L7/teaching_data.py pretrain \
      --raw "$DATA/pretrain_t2t_mini.jsonl" --tokenizer "$SRC/minimind/model" \
      --outdir "$RUN/data-pretrain" --budget-tokens 80000000 --val-tokens 2000000 \
      --source-revision 1e6e909
    python labs/L7/teaching_data.py sft \
      --raw "$DATA/sft_t2t_mini.jsonl" --tokenizer "$SRC/minimind/model" \
      --outdir "$RUN/data-sft" --max-samples 8000 --max-len 768
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

REPLACEMENT = "\ufffd"
_POLY = np.uint64(131)
_MASK32 = np.uint64(0xFFFFFFFF)


def log(message):
    print(f"[teaching_data] {message}", flush=True)


def norm_text(text: str) -> str:
    return " ".join(text.split())


# 公开偏好语料里混有少量不适合公开讲义的成人内容。这里只做一次可复现的有限词表粗筛，
# 规则、命中数与丢弃量都写进 manifest；它不是完整的内容安全审核，也不改写保留的样本。
_EXPLICIT_RE = re.compile(
    r"(?i)\b(anal|porn|pornographic|nude|naked|orgasm|penis|vagina|blowjob|"
    r"handjob|hentai|nsfw|escort|incest|rape|fetish|milf|xxx)\b")


def explicit_hit(record) -> bool:
    text = " ".join(str(message.get("content") or "")
                    for side in ("chosen", "rejected") for message in record.get(side) or [])
    return bool(_EXPLICIT_RE.search(text))


def content_hash(text: str, key: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest(), "big")


def json_reader(path: Path, block_size: int = 1 << 26):
    """按 record batch 流式读 JSON Lines；字段缺失的行由调用方兜底。"""
    import pyarrow.json as pajson
    return pajson.open_json(str(path), read_options=pajson.ReadOptions(block_size=block_size))


class Shingler:
    """n-gram 哈希 + (b, r) MinHash LSH：任一 band 的 r 行签名全等即判为候选重复。

    隐含阈值约 (1/b)^(1/r)：datatrove 默认 b=14, r=8 对应 0.719。
    r=1 时碰撞概率是 1-(1-J)^b，同样 b 下阈值低得多，会大量误删。
    """

    def __init__(self, n, limit, bands, rows, seed):
        self.n, self.limit, self.n_bands, self.rows, self.seed = n, limit, bands, rows, seed
        self.word_ids = {}
        self.seeds = np.asarray(
            [zlib.crc32(f"{seed}:{b}:{r}".encode()) for b in range(bands) for r in range(rows)],
            dtype=np.uint64)

    def _ids(self, words):
        cache = self.word_ids
        if len(cache) > 2_000_000:
            cache.clear()
        out = np.empty(len(words), dtype=np.uint64)
        for i, word in enumerate(words):
            value = cache.get(word)
            if value is None:
                value = zlib.crc32(word.encode())
                cache[word] = value
            out[i] = value
        return out

    def shingles(self, norm):
        words = norm.split()
        if len(words) < self.n:
            return None
        ids = self._ids(words)
        span = ids.shape[0] - self.n + 1
        h = np.zeros(span, dtype=np.uint64)
        for k in range(self.n):
            h = h * _POLY + ids[k:k + span]
            h &= _MASK32
        if span > self.limit:
            h = h[:: max(1, span // self.limit)]
        return h

    def bands(self, h):
        mixed = ((h[None, :] ^ self.seeds[:, None]) * np.uint64(0x9E3779B1)) & _MASK32
        signature = mixed.min(axis=1).reshape(self.n_bands, self.rows)
        keys = []
        for b in range(self.n_bands):
            folded = 0
            for value in signature[b].tolist():
                folded = (folded * 1_000_003 + value) & 0xFFFFFFFFFFFFFFFF
            keys.append((b << 64) | folded)
        return keys


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------

def cmd_pretrain(args):
    from transformers import AutoTokenizer

    raw = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tokenizer)

    rng = random.Random(args.seed)
    counts = Counter()
    reservoir = []                      # (text, norm)
    n_seen = 0
    reader = json_reader(raw)
    for batch in reader:
        column = batch.column("text")
        for i in range(batch.num_rows):
            text = column[i].as_py()
            n_seen += 1
            if not text:
                counts["empty"] += 1
                continue
            norm = norm_text(text)
            if not norm:
                counts["empty"] += 1
                continue
            if REPLACEMENT in norm:
                counts["replacement_char"] += 1
                continue
            if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in norm):
                counts["control_char"] += 1
                continue
            if len(norm) < args.min_chars:
                counts["too_short"] += 1
                continue
            if len(norm) > args.max_chars:
                counts["too_long"] += 1
                continue
            item = (text, norm)
            if len(reservoir) < args.reservoir_docs:
                reservoir.append(item)
            else:
                j = rng.randrange(n_seen)
                if j < args.reservoir_docs:
                    reservoir[j] = item
    counts["docs_scanned"] = n_seen
    log(f"扫描 {n_seen} 条文档，reservoir {len(reservoir)} 条")

    shingler = Shingler(args.shingle_n, args.shingle_limit, args.near_dup_bands,
                        args.near_dup_rows, args.seed)
    seen_exact, near_keys = set(), set()
    held = []
    for text, norm in reservoir:
        sh = shingler.shingles(norm)
        if sh is not None and sh.shape[0] >= 8:
            unique_ratio = len(np.unique(sh)) / sh.shape[0]
            if 1.0 - unique_ratio > args.degenerate_ratio:
                counts["degenerate"] += 1
                continue
            keys = shingler.bands(sh)
            if any(key in near_keys for key in keys):
                counts["near_dup"] += 1
                continue
            near_keys.update(keys)
        digest = content_hash(norm, b"exact")
        if digest in seen_exact:
            counts["exact_dup"] += 1
            continue
        seen_exact.add(digest)
        split_roll = content_hash(norm, b"split") % 1000
        if split_roll < args.val_permille:
            split = "val"
        elif split_roll < args.val_permille + args.test_permille:
            split = "test"
        else:
            split = "train"
        held.append((text, split))
    counts["reservoir_kept"] = len(held)

    order = rng.sample(range(len(held)), len(held))
    budgets = {"train": args.budget_tokens, "val": args.val_tokens, "test": args.val_tokens}
    chosen, used = [], Counter()
    for idx in order:
        text, split = held[idx]
        if used[split] >= budgets[split]:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=args.seq_len - 2).input_ids
        if len(ids) < args.min_tokens:
            counts["too_few_tokens"] += 1
            continue
        tokens = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        used[split] += len(tokens)
        chosen.append((idx, split, tokens))
        if all(used[s] >= budgets[s] for s in budgets):
            break
    chosen.sort(key=lambda row: row[0])
    log(f"入选文档 {len(chosen)} 条，token 账 {dict(used)}")

    summary = {}
    for split in ("train", "val", "test"):
        rows = [(idx, tokens) for idx, sp, tokens in chosen if sp == split]
        flat = np.empty(sum(len(t) for _, t in rows), dtype=np.uint16)
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        pos = 0
        for i, (_, tokens) in enumerate(rows):
            flat[pos:pos + len(tokens)] = np.asarray(tokens, dtype=np.uint16)
            pos += len(tokens)
            offsets[i + 1] = pos
        flat.tofile(outdir / f"{split}.bin")
        np.save(outdir / f"{split}.offsets.npy", offsets)
        lens = np.diff(offsets)
        windows = int(len(flat) // args.seq_len)
        boundary_windows = boundary_pairs = 0
        if windows:
            starts = np.arange(windows, dtype=np.int64) * args.seq_len
            ends = starts + args.seq_len
            left = np.searchsorted(offsets, starts, side="right") - 1
            right = np.searchsorted(offsets, ends, side="left")
            boundary_windows = int(np.count_nonzero(right > left + 1))
            boundary_pairs = int(np.sum(np.maximum(right - left - 1, 0)))
        summary[split] = {
            "docs": len(rows),
            "tokens": int(len(flat)),
            "bytes": int(flat.nbytes),
            "doc_tokens_p50": int(np.percentile(lens, 50)) if len(lens) else 0,
            "doc_tokens_p90": int(np.percentile(lens, 90)) if len(lens) else 0,
            "doc_tokens_p99": int(np.percentile(lens, 99)) if len(lens) else 0,
            "windows": windows,
            "windows_spanning_docs": boundary_windows,
            "window_cross_doc_fraction": round(boundary_windows / windows, 6) if windows else 0.0,
            "cross_doc_boundaries": boundary_pairs,
        }
        log(f"  {split}: {summary[split]['docs']} docs / {len(flat)} tokens / "
            f"{windows} windows，跨文档窗口 {boundary_windows}")

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "kind": "pretrain",
        "source": {"raw": str(raw), "bytes": raw.stat().st_size, "revision": args.source_revision,
                   "license_note": args.license_note},
        "tokenizer": {"path": str(args.tokenizer), "vocab_size": vocab_size,
                      "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                      "pad": tokenizer.pad_token_id},
        "params": {"seq_len": args.seq_len, "seed": args.seed, "reservoir_docs": args.reservoir_docs,
                   "min_chars": args.min_chars, "max_chars": args.max_chars,
                   "min_tokens": args.min_tokens,
                   "budget_tokens": {"train": args.budget_tokens, "val": args.val_tokens,
                                     "test": args.val_tokens},
                   "val_permille": args.val_permille, "test_permille": args.test_permille,
                   "shingle_n": args.shingle_n, "shingle_limit": args.shingle_limit,
                   "near_dup_bands": args.near_dup_bands, "near_dup_rows": args.near_dup_rows,
                   "degenerate_ratio": args.degenerate_ratio},
        "counters": dict(sorted(counts.items())),
        "dedup_scope": (f"入选 reservoir 内（exact + MinHash LSH b={args.near_dup_bands} "
                        f"r={args.near_dup_rows}，隐含阈值 "
                        f"{(1.0 / args.near_dup_bands) ** (1.0 / args.near_dup_rows):.3f}）"),
        "splits": summary,
        "files": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"写出 {outdir}/manifest.json")


# ---------------------------------------------------------------------------
# sft
# ---------------------------------------------------------------------------

def build_sft_example(tokenizer, conversations, max_len: int, template_policy: str):
    """复刻上游 SFTDataset 的 assistant 段监督，但把随机模板改成确定性选择。"""
    messages, tools = [], None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False, tools=tools)
    if template_policy == "deterministic":
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    ids = tokenizer(prompt).input_ids[:max_len]
    labels = [-100] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(ids):
                if ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), len(ids))):
                labels[j] = ids[j]
            i = end + len(eos_id) if end < len(ids) else len(ids)
        else:
            i += 1
    return ids, labels


def cmd_sft(args):
    from transformers import AutoTokenizer

    raw = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    counts = Counter()
    reservoir = []
    n_seen = 0
    # SFT 行的可选字段（reasoning_content/tools/tool_calls）在不同行之间不一致，
    # pyarrow 的 JSON 流式读取会因 schema 变化报 unexpected field，这里逐行 json 解析。
    with raw.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            conversations = obj.get("conversations")
            n_seen += 1
            if not conversations:
                counts["empty"] += 1
                continue
            if len(reservoir) < args.reservoir:
                reservoir.append(conversations)
            else:
                j = rng.randrange(n_seen)
                if j < args.reservoir:
                    reservoir[j] = conversations
    counts["samples_scanned"] = n_seen
    log(f"扫描 {n_seen} 条会话，reservoir {len(reservoir)} 条")

    rows, previews = [], []
    for idx, conversations in enumerate(reservoir):
        ids, labels = build_sft_example(tokenizer, conversations, args.max_len, args.template_policy)
        answer = sum(1 for x in labels if x != -100)
        if answer < args.min_answer_tokens:
            counts["dropped_no_supervision"] += 1
            continue
        if len(ids) == args.max_len and ids[-1] != tokenizer.eos_token_id:
            counts["truncated"] += 1
        split_roll = content_hash(json.dumps(conversations, ensure_ascii=False), b"sft-split") % 1000
        if split_roll < args.val_permille:
            split = "val"
        elif split_roll < args.val_permille + args.test_permille:
            split = "test"
        else:
            split = "train"
        rows.append((idx, split, ids, labels, answer))
        if len(previews) < 5:
            answer_ids = [x for x in labels if x != -100]
            previews.append({
                "reservoir_index": idx, "split": split, "prompt_tokens": len(ids) - answer,
                "answer_tokens": answer,
                "prompt_tail": tokenizer.decode(ids[max(0, len(ids) - answer - 40):len(ids) - answer]),
                "supervision_head": tokenizer.decode(answer_ids[:40]),
            })
    for split in ("train", "val", "test"):
        sel = [r for r in rows if r[1] == split]
        if split == "train" and len(sel) > args.max_samples:
            sel = rng.sample(sel, args.max_samples)
        sel.sort(key=lambda r: r[0])
        inputs = np.full((len(sel), args.max_len), tokenizer.pad_token_id, dtype=np.int32)
        targets = np.full((len(sel), args.max_len), -100, dtype=np.int32)
        for i, (_, _, ids, labels, _) in enumerate(sel):
            inputs[i, :len(ids)] = ids
            targets[i, :len(ids)] = labels
        np.savez_compressed(outdir / f"{split}.npz", input_ids=inputs, labels=targets)
        answer_total = int(sum(r[4] for r in sel))
        counts[f"{split}_samples"] = len(sel)
        counts[f"{split}_answer_tokens"] = answer_total
        counts[f"{split}_prompt_tokens"] = int(sum(len(r[2]) - r[4] for r in sel))
        log(f"  {split}: {len(sel)} samples / {answer_total} answer tokens")

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(), "kind": "sft",
        "source": {"raw": str(raw), "bytes": raw.stat().st_size, "revision": args.source_revision,
                   "license_note": args.license_note},
        "tokenizer": {"path": str(args.tokenizer), "vocab_size": len(tokenizer)},
        "params": {"max_len": args.max_len, "max_samples": args.max_samples, "seed": args.seed,
                   "reservoir": args.reservoir, "template_policy": args.template_policy,
                   "val_permille": args.val_permille, "test_permille": args.test_permille,
                   "min_answer_tokens": args.min_answer_tokens},
        "counters": dict(sorted(counts.items())),
        "supervision": "只对 assistant 段（BOS+assistant\\n 到 EOS+\\n）置 label，其余 -100",
        "files": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (outdir / "preview.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in previews))
    log(f"写出 {outdir}/manifest.json")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_of(record, val_permille, test_permille):
    roll = content_hash(json.dumps(record, ensure_ascii=False), b"posttrain-split") % 1000
    if roll < val_permille:
        return "val"
    if roll < val_permille + test_permille:
        return "test"
    return "train"


def cmd_lora(args):
    """LoRA 领域适配的数据产品：从 SFT 语料里抽出"带工具调用"的子集。

    上游 `apply_lora` 只给方阵 Linear（本结构里是 q_proj/o_proj）加旁路，领域选择也因此
    要落在一个能独立评测的子技能上；这里选 tool-call，因为 7.5-I 的 SFT 恰好在这一项上
    没达标（0.02 对参考 0.14），适配前后有可直接比较的判据。
    """
    raw = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    counts = Counter()
    reservoir = []
    n_seen = n_tool = 0
    with raw.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            n_seen += 1
            conversations = record.get("conversations") or []
            has_tool = any(m.get("tool_calls") for m in conversations)
            has_sys_tools = any(m.get("role") == "system" and m.get("tools") for m in conversations)
            if not (has_tool or has_sys_tools):
                continue
            n_tool += 1
            if len(reservoir) < args.reservoir:
                reservoir.append(record)
            else:
                j = rng.randrange(n_tool)
                if j < args.reservoir:
                    reservoir[j] = record
    counts["samples_scanned"] = n_seen
    counts["tool_samples"] = n_tool
    counts["reservoir"] = len(reservoir)
    log(f"扫描 {n_seen} 条，含工具调用/工具 schema 的 {n_tool} 条，reservoir {len(reservoir)} 条")

    buckets = {"train": [], "val": [], "test": []}
    for record in reservoir:
        buckets[_split_of(record, args.val_permille, args.test_permille)].append(record)
    if len(buckets["train"]) > args.max_samples:
        buckets["train"] = rng.sample(buckets["train"], args.max_samples)
    for split, rows in buckets.items():
        counts[f"{split}_samples"] = len(rows)
        write_jsonl(outdir / f"{split}.jsonl", rows)
        log(f"  {split}: {len(rows)} 条")

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(), "kind": "lora",
        "source": {"raw": str(raw), "bytes": raw.stat().st_size, "revision": args.source_revision,
                   "license_note": args.license_note},
        "selection": "conversations 中任一条带 tool_calls，或 system 消息带 tools",
        "params": {"reservoir": args.reservoir, "max_samples": args.max_samples, "seed": args.seed,
                   "val_permille": args.val_permille, "test_permille": args.test_permille},
        "counters": dict(sorted(counts.items())),
        "files": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"写出 {outdir}/manifest.json")


def cmd_dpo(args):
    """偏好优化的数据产品：把公开偏好对按内容哈希切成 train/val/test。"""
    raw = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    counts = Counter()
    reservoir = []
    n_seen = 0
    with raw.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            n_seen += 1
            if not record.get("chosen") or not record.get("rejected"):
                counts["missing_side"] += 1
                continue
            if explicit_hit(record):
                counts["filtered_explicit"] += 1
                continue
            if len(reservoir) < args.reservoir:
                reservoir.append(record)
            else:
                j = rng.randrange(n_seen)
                if j < args.reservoir:
                    reservoir[j] = record
    counts["pairs_scanned"] = n_seen
    counts["reservoir"] = len(reservoir)
    log(f"扫描 {n_seen} 对，reservoir {len(reservoir)} 对")

    buckets = {"train": [], "val": [], "test": []}
    for record in reservoir:
        buckets[_split_of(record, args.val_permille, args.test_permille)].append(record)
    if len(buckets["train"]) > args.max_pairs:
        buckets["train"] = rng.sample(buckets["train"], args.max_pairs)
    for split, rows in buckets.items():
        counts[f"{split}_pairs"] = len(rows)
        write_jsonl(outdir / f"{split}.jsonl", rows)
        log(f"  {split}: {len(rows)} 对")

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(), "kind": "dpo",
        "source": {"raw": str(raw), "bytes": raw.stat().st_size, "revision": args.source_revision,
                   "license_note": args.license_note},
        "params": {"reservoir": args.reservoir, "max_pairs": args.max_pairs, "seed": args.seed,
                   "val_permille": args.val_permille, "test_permille": args.test_permille},
        "filters": {"explicit_terms": _EXPLICIT_RE.pattern,
                    "rule": "chosen/rejected 任一 content 命中词表即整对丢弃；粗筛，不是完整内容审核"},
        "counters": dict(sorted(counts.items())),
        "files": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"写出 {outdir}/manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--raw", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--source-revision", default="unknown")
    p.add_argument("--license-note", default="见上游数据集页与 MiniMind README 的数据来源说明")
    p.add_argument("--budget-tokens", type=int, default=80_000_000)
    p.add_argument("--val-tokens", type=int, default=2_000_000)
    p.add_argument("--reservoir-docs", type=int, default=600_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--min-chars", type=int, default=64)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--min-tokens", type=int, default=16)
    p.add_argument("--val-permille", type=int, default=25)
    p.add_argument("--test-permille", type=int, default=25)
    p.add_argument("--shingle-n", type=int, default=5)
    p.add_argument("--shingle-limit", type=int, default=128)
    p.add_argument("--near-dup-bands", type=int, default=14)
    p.add_argument("--near-dup-rows", type=int, default=8)
    p.add_argument("--degenerate-ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_pretrain)

    s = sub.add_parser("sft")
    s.add_argument("--raw", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--source-revision", default="unknown")
    s.add_argument("--license-note", default="见上游数据集页与 MiniMind README 的数据来源说明")
    s.add_argument("--max-samples", type=int, default=8000)
    s.add_argument("--max-len", type=int, default=768)
    s.add_argument("--reservoir", type=int, default=12000)
    s.add_argument("--min-answer-tokens", type=int, default=4)
    s.add_argument("--val-permille", type=int, default=50)
    s.add_argument("--test-permille", type=int, default=50)
    s.add_argument("--template-policy", choices=["deterministic", "upstream"], default="deterministic")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sft)

    l = sub.add_parser("lora")
    l.add_argument("--raw", required=True)
    l.add_argument("--outdir", required=True)
    l.add_argument("--source-revision", default="unknown")
    l.add_argument("--license-note", default="见上游数据集页与 MiniMind README 的数据来源说明")
    l.add_argument("--reservoir", type=int, default=10000)
    l.add_argument("--max-samples", type=int, default=8000)
    l.add_argument("--val-permille", type=int, default=100)
    l.add_argument("--test-permille", type=int, default=100)
    l.add_argument("--seed", type=int, default=42)
    l.set_defaults(func=cmd_lora)

    d = sub.add_parser("dpo")
    d.add_argument("--raw", required=True)
    d.add_argument("--outdir", required=True)
    d.add_argument("--source-revision", default="unknown")
    d.add_argument("--license-note", default="见上游数据集页与 MiniMind README 的数据来源说明")
    d.add_argument("--reservoir", type=int, default=12000)
    d.add_argument("--max-pairs", type=int, default=8000)
    d.add_argument("--val-permille", type=int, default=40)
    d.add_argument("--test-permille", type=int, default=40)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_dpo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
