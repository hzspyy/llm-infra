#!/usr/bin/env python3
"""0.4-C: BPE 与 Unigram 的搜索目标不同，批处理入口在哪里。

两件事分开讲：

  1. 算法。BPE 按 merges 的固定顺序贪心合并，走一条路，不比较别的切法；
     Unigram 给每个 piece 一个 log 概率，在所有合法切分里用 Viterbi 取总分最高的一条。
     本脚本用纯 Python 重写 Unigram 的 Viterbi，和真实 tokenizer 逐 id 对拍，
     再打印同一个歧义串的前 5 条候选切分及其分数——BPE 没有这样的候选表。
  2. 工程。tokenizers 是 Rust 实现，Python 侧只是绑定；批处理走 encode_batch，
     内部按 rayon 并行。源码入口从解包的 tokenizers 源码里 grep 出真实行号，
     再测一次 batch 与逐条编码的吞吐——这是 tokenization 的速度，和模型速度无关。

用法：
    python labs/L0/unigram_vs_bpe.py --out-dir <dir> \
        [--bpe-model Qwen/Qwen3-1.7B] [--unigram-model google/mt5-small] \
        [--src-root /path/to/tokenizers-0.23.2]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import time
from pathlib import Path

SEP = "-" * 78

AMBIGUOUS = [
    ("zh-bridge", "上海市长江大桥"),
    ("zh-plain", "风吹过山谷"),
    ("en-token", "tokenization"),
    ("en-intl", "internationalization"),
]

# 在解包的 tokenizers 源码里定位的符号：(标签, 相对路径, 正则)
SYMBOLS = [
    ("encode_batch", "tokenizers/src/tokenizer/mod.rs", r"pub fn encode_batch\b"),
    ("encode_batch_fast", "tokenizers/src/tokenizer/mod.rs", r"pub fn encode_batch_fast\b"),
    ("maybe_par_bridge", "tokenizers/src/utils/parallelism.rs", r"fn maybe_par_bridge\b"),
    ("parallelism_env", "tokenizers/src/utils/parallelism.rs", r"ENV_VARIABLE"),
    ("bpe_tokenize", "tokenizers/src/models/bpe/model.rs", r"fn tokenize_with_cache\b"),
    ("bpe_merge_word", "tokenizers/src/models/bpe/model.rs", r"fn merge_word\b"),
    ("bpe_model_tokenize", "tokenizers/src/models/bpe/model.rs", r"fn tokenize\(&self"),
    ("unigram_encode", "tokenizers/src/models/unigram/model.rs", r"pub fn encode\b"),
    ("unigram_viterbi", "tokenizers/src/models/unigram/lattice.rs", r"pub fn viterbi\b"),
    ("unigram_nbest", "tokenizers/src/models/unigram/lattice.rs", r"pub fn nbest\b"),
    ("bytelevel_pre", "tokenizers/src/pre_tokenizers/byte_level.rs", r"fn pre_tokenize\b"),
    ("metaspace_pre", "tokenizers/src/pre_tokenizers/metaspace.rs", r"fn pre_tokenize\b"),
    ("split_pre", "tokenizers/src/pre_tokenizers/split.rs", r"fn pre_tokenize\b"),
]


# --------------------------------------------------------------- 结构对照

def spec_of(tok) -> dict:
    return json.loads(tok.backend_tokenizer.to_str())


def section_structure(bpe_tok, uni_tok, bpe_name, uni_name) -> dict:
    print(f"[0] 两份 tokenizer.json 的结构\n{SEP}")
    out = {}
    for label, name, tok in [("BPE", bpe_name, bpe_tok), ("Unigram", uni_name, uni_tok)]:
        spec = spec_of(tok)
        m = spec["model"]
        pre = spec.get("pre_tokenizer") or {}
        pre_types = [p["type"] for p in pre.get("pretokenizers", [])] or [pre.get("type")]
        entry = dict(
            name=name, model_type=m["type"], vocab=len(m["vocab"]),
            n_merges=len(m.get("merges", []) or []),
            has_scores=isinstance(m["vocab"][0], list) if isinstance(m["vocab"], list) else False,
            normalizer=(spec.get("normalizer") or {}).get("type"),
            pre_tokenizers=pre_types,
            decoder=(spec.get("decoder") or {}).get("type"),
            unk=m.get("unk_id", m.get("unk_token")),
        )
        out[label] = entry
        print(f"    {label:<8s}{name}")
        print(f"        model.type   {entry['model_type']}   vocab {entry['vocab']:,}"
              f"   merges {entry['n_merges']:,}   带分数 {entry['has_scores']}")
        print(f"        normalizer   {entry['normalizer']}")
        print(f"        pre_tokenizer{pre_types}")
        print(f"        decoder      {entry['decoder']}   unk {entry['unk']}")
    print("\n    BPE 的 model 里是有序 merges，没有任何分数；")
    print("    Unigram 的 model 里是 (piece, log 概率) 列表，没有 merges。")
    print("    两者的搜索目标因此不同：一个按学习顺序合并，一个最大化整条切分的总分。")
    return out


# ------------------------------------------------------- Unigram 的 Viterbi

class UnigramScorer:
    """纯 Python 重写 tokenizers 的 Unigram 编码：Metaspace 之后按 piece 跑 Viterbi。"""

    def __init__(self, tok):
        spec = spec_of(tok)
        m = spec["model"]
        self.pieces: dict[str, tuple[int, float]] = {}
        for i, entry in enumerate(m["vocab"]):
            piece, score = entry
            self.pieces.setdefault(piece, (i, float(score)))
        self.unk_id = m.get("unk_id", 2)
        self.unk_score = min(s for _, s in self.pieces.values())
        self.fuse_unk = bool(m.get("fuse_unk", False))
        self.byte_fallback = bool(m.get("byte_fallback", False))
        self.max_len = max(len(p) for p in self.pieces)
        self.backend = tok.backend_tokenizer

    def prepare(self, text: str) -> list[str]:
        normed = self.backend.normalizer.normalize_str(text) if self.backend.normalizer else text
        return [p for p, _ in self.backend.pre_tokenizer.pre_tokenize_str(normed)]

    def viterbi(self, piece: str) -> tuple[list[str], float]:
        n = len(piece)
        best = [(-math.inf, -1, None)] * (n + 1)
        best[0] = (0.0, -1, None)
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_len), end):
                if best[start][0] == -math.inf:
                    continue
                sub = piece[start:end]
                hit = self.pieces.get(sub)
                score = hit[1] if hit else (self.unk_score - 10.0 if end - start == 1 else None)
                if score is None:
                    continue
                cand = best[start][0] + score
                if cand > best[end][0]:
                    best[end] = (cand, start, sub)
        out, pos = [], n
        while pos > 0:
            _, prev, sub = best[pos]
            out.append(sub)
            pos = prev
        return list(reversed(out)), best[n][0]

    def nbest(self, piece: str, k: int = 5) -> list[tuple[list[str], float]]:
        """k-best 动态规划：每个位置保留前 k 条路径。"""
        n = len(piece)
        table: list[list[tuple[float, list[str]]]] = [[] for _ in range(n + 1)]
        table[0] = [(0.0, [])]
        for end in range(1, n + 1):
            cands = []
            for start in range(max(0, end - self.max_len), end):
                sub = piece[start:end]
                hit = self.pieces.get(sub)
                score = hit[1] if hit else (self.unk_score - 10.0 if end - start == 1 else None)
                if score is None:
                    continue
                for prev_score, path in table[start]:
                    cands.append((prev_score + score, path + [sub]))
            cands.sort(key=lambda x: -x[0])
            table[end] = cands[:k]
        return [(p, s) for s, p in table[n]]

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in self.prepare(text):
            for sub in self.viterbi(piece)[0]:
                hit = self.pieces.get(sub)
                ids.append(hit[0] if hit else self.unk_id)
        return ids


def bpe_merge_trace(tok, text: str, limit: int = 12) -> list[dict]:
    """把 BPE 的合并顺序打出来：它只走一条路。"""
    spec = spec_of(tok)
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
              for m in spec["model"]["merges"]]
    rank = {m: i for i, m in enumerate(merges)}
    backend = tok.backend_tokenizer
    normed = backend.normalizer.normalize_str(text) if backend.normalizer else text
    trace = []
    for piece, _ in backend.pre_tokenizer.pre_tokenize_str(normed):
        syms = list(piece)
        while len(syms) >= 2 and len(trace) < limit:
            pairs = set(zip(syms, syms[1:]))
            best = min(pairs, key=lambda p: rank.get(p, 1 << 30))
            if best not in rank:
                break
            merged, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:
                    merged.append(syms[i] + syms[i + 1])
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            syms = merged
            trace.append(dict(piece=piece, pair=list(best), rank=rank[best],
                              result="".join(s + "|" for s in syms)))
    return trace


def section_search(bpe_tok, uni_tok) -> dict:
    print(f"\n[1] 同一个歧义串：Unigram 有候选表，BPE 没有\n{SEP}")
    scorer = UnigramScorer(uni_tok)
    out = {}
    for name, text in AMBIGUOUS:
        real = uni_tok(text, add_special_tokens=False)["input_ids"]
        mine = scorer.encode(text)
        pieces = scorer.prepare(text)
        print(f"\n    {name}  {text!r}")
        print(f"      Metaspace 预切分 {pieces}")
        print(f"      手写 Viterbi 与真实 Unigram 逐 id "
              f"{'一致' if mine == real else '不一致'}（{len(mine)} vs {len(real)} 个 id）")
        rows = []
        for piece in pieces:
            cands = scorer.nbest(piece, 5)
            for rank_i, (path, score) in enumerate(cands):
                rows.append(dict(piece=piece, rank=rank_i, path=path, score=round(score, 4)))
                print(f"      候选 {rank_i}  总分 {score:>10.3f}  {' | '.join(path)}")
            if len(cands) == 1:
                print("      只有这一条合法切分：该串的多字 piece 不在词表里，只能逐字。")
        bpe_ids = bpe_tok(text, add_special_tokens=False)["input_ids"]
        trace = bpe_merge_trace(bpe_tok, text)
        print(f"      BPE 切成 {len(bpe_ids)} 个 token："
              f"{[bpe_tok.decode([i]) for i in bpe_ids]}")
        for t in trace[:4]:
            print(f"        合并 rank={t['rank']:<7d}{t['pair']}  ->  {t['result']}")
        out[name] = dict(text=text, unigram_real=real, unigram_mine=mine,
                         match=mine == real, pieces=pieces, nbest=rows,
                         bpe_ids=bpe_ids, bpe_tokens=[bpe_tok.decode([i]) for i in bpe_ids],
                         bpe_trace=trace)
    matched = sum(1 for v in out.values() if v["match"])
    print(f"\n    手写 Viterbi 对拍：{matched}/{len(out)} 条与真实 Unigram 完全相同。")
    print("    Unigram 的候选表里，第 2 名和第 1 名的分差就是这次切分的「把握」；")
    print("    BPE 只有一条合并路径，没有第 2 名，也没有分数可比。")
    return out


# ------------------------------------------------------------ 源码入口

def section_source(src_root: Path | None) -> dict:
    print(f"\n[2] tokenizers 的批处理与预切分入口（Rust）\n{SEP}")
    if src_root is None or not src_root.exists():
        print("    未提供 --src-root，跳过；不凭记忆写行号。")
        return dict(error="src-root missing")
    out = {}
    for label, rel, pattern in SYMBOLS:
        f = src_root / rel
        if not f.exists():
            out[label] = dict(file=rel, error="file missing")
            continue
        hit = None
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if re.search(pattern, line):
                hit = (i, line.strip())
                break
        if hit:
            out[label] = dict(file=rel, line=hit[0], text=hit[1])
            print(f"    {label:<18s}{rel}:{hit[0]}  {hit[1][:60]}")
        else:
            out[label] = dict(file=rel, error="pattern not found", pattern=pattern)
            print(f"    {label:<18s}{rel}  未匹配 {pattern}")
    return out


# ------------------------------------------------------------ 批处理吞吐

def section_batch(bpe_tok, texts: list[str], repeats: int = 5) -> dict:
    print(f"\n[3] batch 与逐条编码的吞吐（CPU，与模型速度无关）\n{SEP}")
    backend = bpe_tok.backend_tokenizer
    n_tokens = sum(len(backend.encode(t, add_special_tokens=False).ids) for t in texts)
    for _ in range(2):                      # 预热
        backend.encode_batch([t for t in texts[:64]])
    res = {}
    for label, fn in [
        ("encode_batch", lambda: backend.encode_batch(texts)),
        ("逐条 encode", lambda: [backend.encode(t) for t in texts]),
        ("transformers __call__", lambda: bpe_tok(texts)),
    ]:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        res[label] = dict(median_s=med, all_s=times,
                          tokens_per_s=n_tokens / med, texts=len(texts), tokens=n_tokens)
        print(f"    {label:<24s}中位 {med * 1e3:>8.2f} ms   "
              f"{n_tokens / med / 1e6:>6.2f} M token/s")
    speedup = res["逐条 encode"]["median_s"] / res["encode_batch"]["median_s"]
    res["speedup_batch_over_loop"] = speedup
    print(f"\n    {len(texts)} 条、{n_tokens:,} 个 token；encode_batch 比逐条快 {speedup:.2f}×。")
    print("    这条差距来自 Rust 侧的 rayon 并行与一次跨语言调用，不涉及任何模型计算。")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bpe-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--unigram-model", default="google/mt5-small")
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--batch-texts", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    bpe_tok = AutoTokenizer.from_pretrained(args.bpe_model)
    uni_tok = AutoTokenizer.from_pretrained(args.unigram_model)
    print(f"tokenizers={tokenizers.__version__}  transformers={transformers.__version__}\n")

    structure = section_structure(bpe_tok, uni_tok, args.bpe_model, args.unigram_model)
    search = section_search(bpe_tok, uni_tok)
    source = section_source(Path(args.src_root) if args.src_root else None)

    from bpe_corpus30 import CORPUS
    pool = [t for _, t in CORPUS]
    texts = [pool[i % len(pool)] for i in range(args.batch_texts)]
    batch = section_batch(bpe_tok, texts)

    (out / "unigram_vs_bpe.json").write_text(json.dumps(dict(
        structure=structure, search=search, source=source, batch=batch,
    ), ensure_ascii=False, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.4-C", bpe_model=args.bpe_model, unigram_model=args.unigram_model,
        tokenizers=tokenizers.__version__, transformers=transformers.__version__,
        src_root=args.src_root, batch_texts=args.batch_texts,
        timing="CPU 墙钟，预热 2 次，重复 5 次取中位", host=platform.node(),
        outputs=["unigram_vs_bpe.json", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
