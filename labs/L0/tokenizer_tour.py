#!/usr/bin/env python3
"""L0.4 lab · 真实 tokenizer 的原始现场：文件里到底存了什么。

bpe_from_scratch.py 手写了算法，这个脚本看真实模型的分词器文件。
两件事要对上：手写版学到的东西，和真实 tokenizer 文件里存的东西，是同一类东西。

按顺序打印：
    [0] 分词器由哪几个文件构成，各多大
    [1] tokenizer.json 的顶层结构 —— 它是一份完整的配置，不是词表而已
    [2] 词表片段 —— 直接看那些奇怪的 token 长什么样
    [3] merges 表片段 —— 和手写版的 merges 是同一个东西
    [4] 特殊 token —— 它们不是学出来的，是硬加进去的
    [5] 同一句话的切分 —— 中文 / 英文 / 数字 / 代码，token 效率对比
    [6] chat template —— 一个 Jinja 模板，展开后往序列里插了什么
    [7] padding / attention_mask / position_ids —— 一个 batch 真正喂进去的样子

用法（需要已缓存的模型）：
    python tokenizer_tour.py [--model Qwen/Qwen3-1.7B] [--out-dir results/raw]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SEP = "-" * 78


def show_bytes(s: str, limit: int = 12) -> str:
    b = s.encode("utf-8")[:limit]
    return " ".join(f"{x:02x}" for x in b) + ("…" if len(s.encode("utf-8")) > limit else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out-dir", default="raw")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # ---------------------------------------------------------------- [0]
    print(f"[0] 分词器由哪几个文件构成  （{args.model}）")
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.txt"]))
    print(f"    本地路径 {local}")
    rows = []
    for f in sorted(local.iterdir()):
        if f.is_file():
            rows.append((f.name, f.stat().st_size))
    for name, size in rows:
        print(f"    {name:<34s} {size:>12,d} 字节")
    print("\n    tokenizer.json 是主文件（词表 + merges + 规则，一份自包含的配置）；")
    print("    tokenizer_config.json 放特殊 token 与 chat template；")
    print("    vocab.json / merges.txt 是旧格式，有些模型两种都带。")
    print(SEP)

    # ---------------------------------------------------------------- [1]
    tj_path = local / "tokenizer.json"
    tj = json.loads(tj_path.read_text(encoding="utf-8"))
    print("[1] tokenizer.json 的顶层结构")
    print(f"    文件大小 {tj_path.stat().st_size/1e6:.1f} MB")
    for k in tj:
        v = tj[k]
        if isinstance(v, dict):
            desc = f"dict，键 = {list(v.keys())[:6]}"
        elif isinstance(v, list):
            desc = f"list，{len(v)} 项"
        else:
            desc = repr(v)[:60]
        print(f"    {k:<22s} {desc}")
    print(f"\n    normalizer     = {json.dumps(tj.get('normalizer'), ensure_ascii=False)[:100]}")
    print(f"    pre_tokenizer  = {json.dumps(tj.get('pre_tokenizer'), ensure_ascii=False)[:300]}")
    print("\n    pre_tokenizer 里那条正则，就是 bpe_from_scratch.py 里 pre_tokenize() 的真身。")
    print(SEP)

    # ---------------------------------------------------------------- [2]
    vocab = tj["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}
    print("[2] 词表片段")
    print(f"    词表大小 {len(vocab):,}")
    print(f"    {'id':>8s}  {'token（repr）':<24s} {'字节'}")
    for i in [0, 1, 2, 100, 101, 1000, 5000, 50000, 100000, len(vocab) - 1]:
        if i in inv:
            t = inv[i]
            print(f"    {i:>8d}  {t!r:<24s} {show_bytes(t)}")
    print("\n    注意那些 'Ġ' 开头的 token：Ġ 是空格的可打印替身（U+0120）。")
    print("    字节级 BPE 把 0..255 每个字节映射到一个可打印字符，")
    print("    这样词表才能存成 JSON。空格 0x20 被映射成 Ġ。")
    print(f"    验证：'Ġthe' 的 id = {vocab.get('Ġthe')}，' the' 编码结果 = {tok.encode(' the')}")
    print(SEP)

    # ---------------------------------------------------------------- [3]
    merges = tj["model"].get("merges", [])
    print("[3] merges 表")
    print(f"    共 {len(merges):,} 条。前 8 条与最后 4 条：")
    for i in list(range(8)) + list(range(len(merges) - 4, len(merges))):
        m = merges[i]
        m = " ".join(m) if isinstance(m, list) else m
        print(f"    {i:>8d}  {m}")
        if i == 7:
            print(f"    {'...':>8s}")
    print("\n    格式是「左 右」两个已有 token，合并成一个新 token。")
    print("    顺序就是学习顺序，编码时必须按这个顺序尝试——和手写版完全一致。")
    print(SEP)

    # ---------------------------------------------------------------- [4]
    print("[4] 特殊 token")
    added = tj.get("added_tokens", [])
    print(f"    added_tokens 共 {len(added)} 个（这些不是学出来的，是硬加的）：")
    for a in added[:14]:
        print(f"    {a['id']:>8d}  {a['content']!r:<24s} special={a.get('special')}")
    if len(added) > 14:
        print(f"    ... 还有 {len(added)-14} 个")
    print(f"\n    tok.eos_token = {tok.eos_token!r} (id {tok.eos_token_id})")
    print(f"    tok.pad_token = {tok.pad_token!r} (id {tok.pad_token_id})")
    print("    生成停止的判据之一就是采到了 eos_token_id（L0.5）。")
    print(SEP)

    # ---------------------------------------------------------------- [5]
    print("[5] 同一段内容，不同语言/形态的 token 效率")
    samples = [
        ("中文", "风吹过山谷，云落在水面。"),
        ("英文", "The wind blows over the valley."),
        ("数字", "3.14159265358979"),
        ("代码", "for i in range(10):\n    print(i)"),
        ("重复空格", "a" + " " * 20 + "b"),
        ("emoji", "🌊🌊🌊"),
    ]
    print(f"    {'类型':<8s} {'字符数':>6s} {'字节数':>6s} {'token 数':>8s} {'字节/token':>10s}")
    for tag, text in samples:
        ids = tok.encode(text)
        nb = len(text.encode("utf-8"))
        print(f"    {tag:<8s} {len(text):>6d} {nb:>6d} {len(ids):>8d} {nb/len(ids):>10.2f}")
    print("\n    逐 token 看中文那一句：")
    ids = tok.encode(samples[0][1])
    for i, t in enumerate(ids):
        piece = tok.convert_ids_to_tokens([t])[0]
        print(f"      {i:>3d}  id={t:<8d} token={piece!r:<14s} decode={tok.decode([t])!r}")
    print(SEP)

    # ---------------------------------------------------------------- [6]
    print("[6] chat template：一个 Jinja 模板")
    cfg = json.loads((local / "tokenizer_config.json").read_text(encoding="utf-8"))
    tmpl = cfg.get("chat_template") or getattr(tok, "chat_template", None)
    if tmpl:
        print(f"    模板长度 {len(tmpl)} 字符。前 600 字符：\n")
        for line in tmpl[:600].splitlines():
            print(f"      {line}")
        print("      ...")
    msgs = [{"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "风是什么？"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"\n    展开后的完整字符串（{len(rendered)} 字符），原样打印：")
    print("    " + "─" * 60)
    for line in rendered.splitlines():
        print(f"    │{line}")
    print("    " + "─" * 60)
    # 直接编码展开后的字符串即可——模板本身只是字符串拼接，没有别的魔法。
    # （apply_chat_template(tokenize=True) 在不同 transformers 版本返回类型不一致，
    #   这里避开它，顺便让「模板 = 拼字符串」这件事更清楚。）
    ids2 = tok.encode(rendered, add_special_tokens=False)
    print(f"\n    编码成 {len(ids2)} 个 token。前 12 个：")
    for i, t in enumerate(ids2[:12]):
        print(f"      {i:>3d}  id={t:<8d} {tok.convert_ids_to_tokens([t])[0]!r}")
    print("\n    模板做的事：插入角色标记、把多轮消息串起来、"
          "在末尾放上「该模型说话了」的提示。")
    print("    add_generation_prompt=True 就是最后那一段——没有它模型不知道该接话。")
    print(f"\n    对照：不加 add_generation_prompt 时，末尾少了 "
          f"{len(rendered) - len(tok.apply_chat_template(msgs, tokenize=False)):d} 个字符")
    print(SEP)

    # ------------------------------------------------ [5b] 跨 tokenizer 对照
    print("[5b] 换一个 tokenizer，同样的文本")
    print("    「中文 token 效率低」这个说法要看是谁的 tokenizer。")
    others = ["gpt2", "meta-llama/Llama-3.2-1B"]
    rows = []
    for name in [args.model] + others:
        try:
            t2 = AutoTokenizer.from_pretrained(name)
        except Exception as exc:                          # noqa: BLE001
            rows.append((name, None, str(exc)[:40]))
            continue
        cells = []
        for tag, text in samples[:4]:
            nb = len(text.encode("utf-8"))
            cells.append(nb / len(t2.encode(text)))
        rows.append((name, len(t2), cells))
    print(f"    {'tokenizer':<26s} {'词表':>8s} " +
          " ".join(f"{tag:>8s}" for tag, _ in samples[:4]))
    for name, size, cells in rows:
        if size is None:
            print(f"    {name:<26s} {'(不可用)':>8s}  {cells}")
            continue
        print(f"    {name:<26s} {size:>8,d} " +
              " ".join(f"{c:>8.2f}" for c in cells) + "   字节/token")
    print("\n    单位是「字节/token」，越大越省。")
    print(SEP)

    # ---------------------------------------------------------------- [7]
    print("[7] 一个 batch 真正喂进去的样子")
    batch = ["风是什么？", "云为什么会下雨呢？请详细说明。"]
    for side in ("right", "left"):
        tok.padding_side = side
        enc = tok(batch, padding=True, return_tensors=None)
        print(f"\n    padding_side = {side}")
        for i, (ids3, mask) in enumerate(zip(enc["input_ids"], enc["attention_mask"])):
            toks = [tok.convert_ids_to_tokens([t])[0] for t in ids3]
            print(f"      样本 {i}  len={len(ids3)}")
            print(f"        input_ids      {ids3}")
            print(f"        attention_mask {mask}")
            print(f"        tokens         {toks}")
        # position_ids 通常由模型按 mask 生成；这里手工算出来对照
        import itertools
        for i, mask in enumerate(enc["attention_mask"]):
            pos = list(itertools.accumulate(mask))
            pos = [p - 1 if m else 0 for p, m in zip(pos, mask)]
            print(f"      样本 {i} 的 position_ids（按 mask 累加）: {pos}")
    print("\n    decoder-only 模型生成时必须用 **left padding**：")
    print("    生成是从序列末尾接着往下写的，右 padding 会让模型从 <pad> 后面开始写。")
    print("    训练时用右 padding 没问题，因为 loss 会用 mask 把 padding 位置排除。")
    print(SEP)

    # 落盘原始材料
    (out / "tokenizer_structure.json").write_text(
        json.dumps({k: (list(v.keys()) if isinstance(v, dict) else
                        (f"<list {len(v)}>" if isinstance(v, list) else v))
                    for k, v in tj.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "vocab_excerpt.txt").write_text(
        "\n".join(f"{i}\t{inv[i]!r}" for i in sorted(inv)[:200]), encoding="utf-8")
    (out / "merges_excerpt.txt").write_text(
        "\n".join(" ".join(m) if isinstance(m, list) else str(m) for m in merges[:200]),
        encoding="utf-8")
    if tmpl:
        (out / "chat_template.jinja").write_text(tmpl, encoding="utf-8")
    (out / "chat_rendered.txt").write_text(rendered, encoding="utf-8")
    print(f"原始材料写入 {out.absolute()}")


if __name__ == "__main__":
    main()
