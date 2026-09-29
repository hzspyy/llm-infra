#!/usr/bin/env python3
"""0.4-A: 30 条固定文本穿过手写 BPE，再逐层对照真实 tokenizer。

固定语料覆盖中英文、组合音标、emoji（含 ZWJ 与肤色修饰符）、各类空白、
特殊 token 字面量、数字与代码。每条文本都打印 UTF-8 字节、预切分、
合并顺序与最终 id，然后回答一个问题：**手写版和真实 tokenizer 的差异出在哪一层**。

分层的做法是把真实 tokenizer 拆成四层，逐层换回手写实现：

    L1 normalizer      NFC 规范化
    L2 pre_tokenizer   一条正则 + ByteLevel 字节映射
    L3 model           merges 表（有序）+ vocab
    L4 added_tokens    特殊 token，在前三层之前先切出来

最后一步用真实 merges 驱动手写的合并循环，和真实 tokenizer 逐 id 对拍：
算法相同，差的只是表和这几层包装。

用法：
    python labs/L0/bpe_corpus30.py --out-dir results/local/0.4/<run>
    python labs/L0/bpe_corpus30.py --out-dir <dir> --hf-model Qwen/Qwen3-1.7B
第二种需要 transformers 与已缓存的模型；不给 --hf-model 时只跑手写部分。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe_from_scratch import get_pair_counts, merge_pair, pre_tokenize, show_token  # noqa: E402

SEP = "-" * 78

# 30 条固定文本。id 用于在正文和工件里引用，顺序不变。
CORPUS: tuple[tuple[str, str], ...] = (
    ("zh-simp", "风吹过山谷，云落在水面。"),
    ("zh-trad", "風吹過山谷，雲落在水面。"),
    ("zh-en-mix", "把 KV cache 放进显存，decode 每步只读一次权重。"),
    ("zh-rare", "龘齉鱻鼟"),
    ("en-lower", "the wind comes from the sea"),
    ("en-title", "The Wind Comes From The Sea"),
    ("en-contraction", "don't stop; it's the model's turn"),
    ("en-long-word", "internationalization"),
    ("nfc-composed", "café résumé naïve"),
    ("nfd-decomposed", "cafe\u0301 re\u0301sume\u0301 nai\u0308ve"),
    ("nfc-hangul", "한국어"),
    ("nfd-hangul", "\u1112\u1161\u11ab\u1100\u116e\u11a8\u110b\u1165"),
    ("emoji-plain", "🌊🔥🌙"),
    ("emoji-zwj", "👨‍👩‍👧‍👦 一家四口"),
    ("emoji-skin", "👍🏽 ok"),
    ("emoji-flag", "🇨🇳🇯🇵"),
    ("ws-indent", "def f():\n    return 1\n"),
    ("ws-run", "a" + " " * 20 + "b"),
    ("ws-tab", "col1\tcol2\tcol3"),
    ("ws-crlf", "line1\r\nline2\r\n"),
    ("ws-nbsp", "1 000 元"),
    ("special-im-start", "<|im_start|>user"),
    ("special-in-text", "文本里出现 <|endoftext|> 字面量"),
    ("num-pi", "3.14159265358979"),
    ("num-date", "2026-09-13"),
    ("num-long", "1234567890"),
    ("code-py", "for i in range(10): print(i)"),
    ("code-json", '{"name": "qwen", "n": 3}'),
    ("url", "https://example.com/path?q=1&r=2"),
    ("math-unicode", "∀x∈ℝ: x²≥0"),
)
assert len(CORPUS) == 30, len(CORPUS)

N_MERGES = 200


def esc(s: str) -> str:
    """把不可见字符显示出来，其余原样。"""
    out = []
    for c in s:
        if c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif unicodedata.category(c) in ("Cc", "Cf") or c == " ":
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    return "".join(out)


# --------------------------------------------------------------- [0] 语料本身

def section_corpus() -> list[dict]:
    print(f"[0] 固定语料：30 条，每条的字符数、码点数与 UTF-8 字节数\n{SEP}")
    print(f"    {'id':<17s}{'字符':>4s}{'码点':>5s}{'字节':>5s}{'NFC?':>6s}  文本")
    rows = []
    for name, text in CORPUS:
        nfc = unicodedata.normalize("NFC", text)
        row = dict(
            id=name, text=text, chars=len(text), codepoints=len(text),
            bytes=len(text.encode("utf-8")), nfc_stable=(nfc == text),
            nfc_bytes=len(nfc.encode("utf-8")),
        )
        rows.append(row)
        mark = "是" if row["nfc_stable"] else "否"
        print(f"    {name:<17s}{row['chars']:>4d}{row['codepoints']:>5d}"
              f"{row['bytes']:>5d}{mark:>6s}  {esc(text)[:40]}")
    unstable = [r["id"] for r in rows if not r["nfc_stable"]]
    print(f"\n    NFC 下会变形的有 {len(unstable)} 条：{unstable}")
    print("    这几条进真实 tokenizer 前先被 normalizer 改写，字节数因此变化。")
    return rows


# ------------------------------------------------------- [1] 字符 → UTF-8 字节

def section_utf8() -> list[dict]:
    print(f"\n[1] 逐字符的 UTF-8 展开（挑 4 条）\n{SEP}")
    picked = ["zh-simp", "nfd-decomposed", "emoji-zwj", "emoji-flag"]
    table = dict(CORPUS)
    out = []
    for name in picked:
        text = table[name]
        print(f"    {name}  {esc(text)}")
        items = []
        for c in text[:8]:
            b = c.encode("utf-8")
            items.append(dict(char=c, cp=f"U+{ord(c):04X}", nbytes=len(b),
                              hex=" ".join(f"{x:02x}" for x in b),
                              cat=unicodedata.category(c)))
            print(f"        {esc(c):<4s} U+{ord(c):04X}  {len(b)} 字节  "
                  f"{' '.join(f'{x:02x}' for x in b):<12s} {unicodedata.category(c)}")
        if len(text) > 8:
            print(f"        …（共 {len(text)} 个码点）")
        out.append(dict(id=name, chars=items, total_codepoints=len(text)))
    print("\n    👨‍👩‍👧‍👦 本身是 7 个码点（4 个人形 + 3 个 U+200D 零宽连接符），共 25 字节；")
    print("    emoji-flag 的每面旗是 2 个 Regional Indicator 码点，共 8 字节。")
    print("    组合音标 nfd-decomposed 的 'e' 和 U+0301 是两个码点，显示成一个字符。")
    return out


# ------------------------------------------------------------- [2] 预切分

def section_pretok() -> list[dict]:
    print(f"\n[2] 手写预切分的结果（按字符类别分块）\n{SEP}")
    rows = []
    for name, text in CORPUS:
        pieces = pre_tokenize(text)
        rows.append(dict(id=name, n=len(pieces), pieces=pieces))
    for name in ["en-lower", "ws-indent", "num-pi", "special-im-start", "zh-en-mix"]:
        r = next(x for x in rows if x["id"] == name)
        shown = " | ".join(esc(p) for p in r["pieces"][:12])
        print(f"    {name:<17s}{r['n']:>3d} 块  {shown}")
    print("\n    手写版按「字母/数字/空白/其它」分类，空格并入后一块。")
    print("    它不认识 <|im_start|> 这类特殊 token，也不按 Unicode 属性判断类别。")
    return rows


# --------------------------------------------------------------- [3] 训练

def section_train(n_merges: int) -> tuple[list[tuple[int, int]], dict[int, bytes], list[dict]]:
    print(f"\n[3] 在这 30 条文本上训练 {n_merges} 轮合并\n{SEP}")
    chunks: list[list[int]] = []
    for _, text in CORPUS:
        for piece in pre_tokenize(text):
            chunks.append(list(piece.encode("utf-8")))
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    total0 = sum(len(c) for c in chunks)
    print(f"    初始：256 个字节 token，{len(chunks)} 个预切分块，共 {total0} 个符号")
    print(f"    {'轮':>4s}{'合并的对':>26s}{'次数':>6s}{'新 id':>7s}  {'新符号':<22s}{'剩余符号':>9s}")
    trace = []
    for step in range(n_merges):
        counts = get_pair_counts(chunks)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:
            print(f"    第 {step} 轮最高频只出现 {freq} 次，停止训练")
            break
        new_id = 256 + len(merges)
        chunks = merge_pair(chunks, pair, new_id)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(pair)
        left, right = show_token(vocab[pair[0]]), show_token(vocab[pair[1]])
        rest = sum(len(c) for c in chunks)
        rec = dict(step=step, pair=[int(pair[0]), int(pair[1])], freq=int(freq),
                   new_id=new_id, symbol=show_token(vocab[new_id]),
                   utf8_complete=_decodable(vocab[new_id]), symbols_left=rest)
        trace.append(rec)
        if step < 14:
            print(f"    {step:>4d}{left + ' + ' + right:>26s}{freq:>6d}{new_id:>7d}  "
                  f"{show_token(vocab[new_id]):<22s}{rest:>9d}")
    print(f"    …（共 {len(merges)} 轮，词表 {len(vocab)}）")
    partial = [t for t in trace if not t["utf8_complete"]]
    print(f"\n    {len(partial)}/{len(trace)} 个新符号不是完整 UTF-8 序列——它们是半个字符。")
    print("    汉字要两次合并（3 字节）、emoji 要三次（4 字节）才能凑成一个 token。")
    return merges, vocab, trace


def _decodable(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# --------------------------------------------------------------- [4] 编码

def mini_encode(text: str, rank: dict[tuple[int, int], int]) -> list[int]:
    out: list[int] = []
    for piece in pre_tokenize(text):
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: rank.get(p, 1 << 30))
            if best not in rank:
                break
            ids = merge_pair([ids], best, 256 + rank[best])[0]
        out.extend(ids)
    return out


def section_encode(merges, vocab) -> list[dict]:
    print(f"\n[4] 用学到的表编码这 30 条\n{SEP}")
    rank = {p: i for i, p in enumerate(merges)}
    rows = []
    print(f"    {'id':<17s}{'字节':>5s}{'token':>6s}{'字节/token':>11s}  前 8 个 token")
    for name, text in CORPUS:
        ids = mini_encode(text, rank)
        toks = [show_token(vocab[i]) for i in ids]
        nb = len(text.encode("utf-8"))
        rows.append(dict(id=name, n_bytes=nb, n_tokens=len(ids), ids=ids, tokens=toks,
                         bytes_per_token=round(nb / max(len(ids), 1), 3),
                         roundtrip=b"".join(vocab[i] for i in ids) == text.encode("utf-8")))
        print(f"    {name:<17s}{nb:>5d}{len(ids):>6d}{nb / max(len(ids), 1):>11.2f}  "
              f"{' | '.join(toks[:8])[:52]}")
    bad = [r["id"] for r in rows if not r["roundtrip"]]
    print(f"\n    解码回原字节：{'全部一致' if not bad else '失败 ' + str(bad)}")
    print("    字节级 BPE 没有 OOV：没学到的字退回逐字节，仍然可逆。")
    return rows


# ---------------------------------------------- [5] 与真实 tokenizer 分层对照

def section_layers(hf_model: str, out: Path, enc: list[dict]) -> dict:
    print(f"\n[5] 和真实 tokenizer 分层对照  （{hf_model}）\n{SEP}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model)
    backend = tok.backend_tokenizer
    spec = json.loads(backend.to_str())
    model = spec["model"]
    vocab: dict[str, int] = model["vocab"]
    raw_merges = model["merges"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in raw_merges]
    rank = {m: i for i, m in enumerate(merges)}
    added = {t["content"]: t["id"] for t in spec["added_tokens"]}
    print(f"    vocab {len(vocab):,}   merges {len(merges):,}   added_tokens {len(added)}")
    print(f"    normalizer      {json.dumps(spec['normalizer'], ensure_ascii=False)}")
    print(f"    pre_tokenizer   {spec['pre_tokenizer']['type']}"
          f" -> {[p['type'] for p in spec['pre_tokenizer']['pretokenizers']]}")

    # ---- L1 normalizer
    print(f"\n    L1 normalizer（{spec['normalizer']['type']}）")
    l1 = []
    for name, text in CORPUS:
        normed = backend.normalizer.normalize_str(text)
        l1.append(dict(id=name, changed=normed != text,
                       bytes_before=len(text.encode("utf-8")),
                       bytes_after=len(normed.encode("utf-8"))))
    changed = [r for r in l1 if r["changed"]]
    print(f"       {len(changed)}/30 条被改写：" +
          ", ".join(f"{r['id']}({r['bytes_before']}→{r['bytes_after']} 字节)" for r in changed))
    print("       手写版没有这一层，这几条的输入字节从这里就已经不同。")

    # ---- L2 pre_tokenizer
    print("\n    L2 pre_tokenizer（正则 + ByteLevel）")
    l2 = []
    for name, text in CORPUS:
        normed = backend.normalizer.normalize_str(text)
        real = backend.pre_tokenizer.pre_tokenize_str(normed)
        real_spans = [span for _, span in real]
        mine, pos = [], 0
        for piece in pre_tokenize(normed):
            mine.append((pos, pos + len(piece)))
            pos += len(piece)
        l2.append(dict(id=name, real_pieces=[p for p, _ in real], real_spans=real_spans,
                       mine_spans=mine, same=real_spans == mine))
    same = sum(1 for r in l2 if r["same"])
    print(f"       切分边界与手写版一致的有 {same}/30 条")
    for name in ["num-pi", "en-contraction", "special-im-start"]:
        r = next(x for x in l2 if x["id"] == name)
        print(f"       {name:<17s}真实 {len(r['real_pieces']):>2d} 块 "
              f"{' | '.join(r['real_pieces'][:8])}")
    print("       数字被 \\p{N} 逐位切开，缩写 's 单独成块——手写版都做不到。")

    # ---- L3 用真实 merges 驱动手写的合并循环
    print("\n    L3 model：把真实 merges 装进手写的合并循环")

    def bpe_piece(piece: str) -> list[int]:
        syms = list(piece)
        while len(syms) >= 2:
            pairs = set(zip(syms, syms[1:]))
            best = min(pairs, key=lambda p: rank.get(p, 1 << 30))
            if best not in rank:
                break
            i, merged = 0, []
            while i < len(syms):
                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:
                    merged.append(syms[i] + syms[i + 1])
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            syms = merged
        return [vocab[s] for s in syms]

    def hand_encode(text: str, use_added: bool) -> list[int]:
        segments: list[tuple[str, bool]] = [(text, False)]
        if use_added and added:
            for content in sorted(added, key=len, reverse=True):
                nxt: list[tuple[str, bool]] = []
                for seg, is_special in segments:
                    if is_special or content not in seg:
                        nxt.append((seg, is_special))
                        continue
                    parts = seg.split(content)
                    for k, part in enumerate(parts):
                        if k:
                            nxt.append((content, True))
                        if part:
                            nxt.append((part, False))
                segments = nxt
        ids: list[int] = []
        for seg, is_special in segments:
            if is_special:
                ids.append(added[seg])
                continue
            normed = backend.normalizer.normalize_str(seg)
            for piece, _ in backend.pre_tokenizer.pre_tokenize_str(normed):
                ids.extend(bpe_piece(piece))
        return ids

    l3 = []
    for name, text in CORPUS:
        real = tok(text, add_special_tokens=False)["input_ids"]
        mine_added = hand_encode(text, use_added=True)
        mine_plain = hand_encode(text, use_added=False)
        l3.append(dict(id=name, n_real=len(real), real=real,
                       hand_with_added=mine_added, hand_without_added=mine_plain,
                       match_with_added=mine_added == real,
                       match_without_added=mine_plain == real))
    ok = sum(1 for r in l3 if r["match_with_added"])
    ok_plain = sum(1 for r in l3 if r["match_without_added"])
    print(f"       手写合并循环 + 真实表：{ok}/30 条与真实 tokenizer 逐 id 相同")
    print(f"       去掉 L4（不先切特殊 token）：{ok_plain}/30 条相同")
    diff = [r for r in l3 if not r["match_without_added"]]
    for r in diff[:3]:
        print(f"       {r['id']:<17s}真实 {r['n_real']} 个 id，"
              f"不切特殊 token 时 {len(r['hand_without_added'])} 个")
    bad = [r["id"] for r in l3 if not r["match_with_added"]]
    print(f"       仍不一致：{bad if bad else '无'}")

    # ---- 规模对照：小语料 merges 不能冒充完整词表
    print("\n    小语料 merges（90 轮）与完整词表（15 万 merges）的差距")
    print(f"       {'id':<17s}{'字节':>5s}{'真实':>6s}{'手写':>6s}"
          f"{'真实 字节/token':>16s}{'手写':>8s}")
    mini = {r["id"]: r for r in enc}
    scale = []
    for name, text in CORPUS:
        real = tok(text, add_special_tokens=False)["input_ids"]
        nb = len(text.encode("utf-8"))
        n_mini = mini[name]["n_tokens"]
        scale.append(dict(id=name, n_bytes=nb, n_real=len(real), n_mini=n_mini,
                          bpt_real=round(nb / len(real), 3), bpt_mini=round(nb / n_mini, 3)))
    for name in ["zh-simp", "en-lower", "num-pi", "code-py", "emoji-zwj"]:
        r = next(x for x in scale if x["id"] == name)
        print(f"       {name:<17s}{r['n_bytes']:>5d}{r['n_real']:>6d}{r['n_mini']:>6d}"
              f"{r['bpt_real']:>16.2f}{r['bpt_mini']:>8.2f}")
    tot_real = sum(r["n_real"] for r in scale)
    tot_mini = sum(r["n_mini"] for r in scale)
    tot_bytes = sum(r["n_bytes"] for r in scale)
    print(f"       {'30 条合计':<17s}{tot_bytes:>5d}{tot_real:>6d}{tot_mini:>6d}"
          f"{tot_bytes / tot_real:>16.2f}{tot_bytes / tot_mini:>8.2f}")
    payload = dict(model=hf_model, vocab_size=len(vocab), n_merges=len(merges),
                   added_tokens=len(added), normalizer=spec["normalizer"]["type"],
                   layer1=l1, layer2=l2, layer3=l3, real_counts=scale)
    (out / "layers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--n-merges", type=int, default=N_MERGES)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = section_corpus()
    utf8 = section_utf8()
    pretok = section_pretok()
    merges, vocab, trace = section_train(args.n_merges)
    enc = section_encode(merges, vocab)
    layers = section_layers(args.hf_model, out, enc) if args.hf_model else None

    (out / "mini_bpe.json").write_text(json.dumps(dict(
        corpus=corpus, utf8=utf8, pretokenize=pretok, merge_trace=trace,
        encoded=enc, n_merges=len(merges), vocab_size=len(vocab),
    ), ensure_ascii=False, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.4-A", corpus_size=len(CORPUS), n_merges_requested=args.n_merges,
        n_merges_learned=len(merges), hf_model=args.hf_model,
        python=platform.python_version(), host=platform.node(),
        note="纯 Python，无随机性；对拍口径是整数 id 精确相等",
        outputs=["mini_bpe.json", "layers.json", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
