#!/usr/bin/env python3
"""0.5-B: 增量 detokenize 与停止条件，逐字节与引擎对拍。

两个阶段：

  detok  构造在 token 边界上被切开的字节序列（罕见汉字、ZWJ emoji、肤色修饰符、
         组合音标），比较三种做法：逐 token decode、手写字节缓冲、
         vLLM 实际使用的 tokenizers DecodeStream（`v1/engine/detokenizer.py:185`）。
  stop   用真实 vLLM 生成，构造三种停止情形：停止串横跨两个 token、
         include_stop_str_in_output 的两种取值、EOS 与 max_tokens 同时满足。
         每种情形记录 finish_reason、stop_reason、token 数与输出字节。

用法：
    python labs/L0/detok_stop_probe.py detok --out-dir <dir>
    python labs/L0/detok_stop_probe.py stop  --out-dir <dir>
detok 阶段只要 tokenizers/transformers；stop 阶段需要 GPU 与 vLLM。
"""
from __future__ import annotations

import argparse
import json
import platform
import unicodedata
from pathlib import Path

SEP = "-" * 78
MODEL = "Qwen/Qwen3-1.7B"

CASES = [
    ("rare-cjk", "𩸽"),
    ("emoji-zwj", "👨‍👩‍👧‍👦"),
    ("emoji-skin", "👍🏽"),
    ("nfd-accent", "café"),
    ("cjk-sentence", "风吹过山谷"),
    ("mixed", "答案是 𩸽 和 👍🏽。"),
]


def hexs(b: bytes, limit: int = 12) -> str:
    return " ".join(f"{x:02x}" for x in b[:limit]) + ("…" if len(b) > limit else "")


# ------------------------------------------------------- 手写增量 detokenizer

class ByteBufferDetokenizer:
    """按字节缓冲的增量解码：凑不齐完整 UTF-8 序列就先不吐字。"""

    def __init__(self, id_to_bytes):
        self.id_to_bytes = id_to_bytes
        self.buf = b""

    def step(self, token_id: int) -> str:
        self.buf += self.id_to_bytes[token_id]
        out = ""
        # 尽可能多地解出完整字符，剩下的留在缓冲里
        for cut in range(len(self.buf), 0, -1):
            try:
                out = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            self.buf = self.buf[cut:]
            return out
        return ""


def build_id_to_bytes(tok):
    """反查字节级映射表，拿到每个 token 的真实字节（不能走 decode，会被替换符污染）。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    table = {}
    for token, tid in tok.get_vocab().items():
        try:
            table[tid] = bytes(char_to_byte[c] for c in token)
        except KeyError:
            table[tid] = token.encode("utf-8")     # added_tokens 等非字节级条目
    return table


# --------------------------------------------------------------- detok 阶段

def run_detok(out: Path) -> None:
    import tokenizers
    import tokenizers.decoders
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    backend = tok.backend_tokenizer
    id_to_bytes = build_id_to_bytes(tok)
    print(f"model={MODEL}  tokenizers={tokenizers.__version__}\n")
    rows = []
    for name, text in CASES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        per_token = [tok.decode([i]) for i in ids]
        naive = "".join(per_token)

        mini = ByteBufferDetokenizer(id_to_bytes)
        mini_steps = [mini.step(i) for i in ids]
        mini_text = "".join(mini_steps)

        stream = tokenizers.decoders.DecodeStream(skip_special_tokens=False)
        engine_steps = [stream.step(backend, i) or "" for i in ids]
        engine_text = "".join(engine_steps)

        whole = tok.decode(ids)
        row = dict(id=name, text=text, n_tokens=len(ids), ids=ids,
                   token_bytes=[hexs(id_to_bytes[i]) for i in ids],
                   per_token_decode=per_token, naive_text=naive,
                   mini_steps=mini_steps, mini_text=mini_text,
                   engine_steps=engine_steps, engine_text=engine_text,
                   whole_decode=whole,
                   naive_ok=naive == text, mini_ok=mini_text == text,
                   engine_ok=engine_text == text, mini_matches_engine=mini_steps == engine_steps,
                   nfc_text=unicodedata.normalize("NFC", text),
                   mini_ok_vs_nfc=mini_text == unicodedata.normalize("NFC", text))
        rows.append(row)
        print(f"[{name}] {text!r}  ->  {len(ids)} 个 token")
        print(f"    {'i':>3s} {'id':>8s}  {'字节':<14s}{'单独 decode':<14s}"
              f"{'手写缓冲吐出':<16s}引擎 DecodeStream 吐出")
        for i, tid in enumerate(ids):
            print(f"    {i:>3d} {tid:>8d}  {hexs(id_to_bytes[tid]):<14s}"
                  f"{per_token[i]!r:<14s}{mini_steps[i]!r:<16s}{engine_steps[i]!r}")
        print(f"    逐 token 拼接 {naive!r}  {'一致' if row['naive_ok'] else '与原文不同'}")
        print(f"    手写缓冲     {mini_text!r}  {'一致' if row['mini_ok'] else '与原文不同'}")
        print(f"    引擎 stream  {engine_text!r}  {'一致' if row['engine_ok'] else '与原文不同'}")
        if not row["mini_ok"] and row["mini_ok_vs_nfc"]:
            print("    与原文不同但与 NFC 规范化后的文本相同："
                  "tokenizer 的 normalizer 在编码阶段就改写了输入（见 0.4 的 L1 层），"
                  "这不是 detokenizer 的问题。")
        print(f"    手写与引擎逐步相同：{row['mini_matches_engine']}\n")

    bad_naive = [r["id"] for r in rows if not r["naive_ok"]]
    same = [r["id"] for r in rows if r["mini_matches_engine"]]
    print(SEP)
    print(f"    逐 token decode 出错的用例：{bad_naive}")
    print(f"    手写缓冲与引擎逐步一致的用例：{len(same)}/{len(rows)}  {same}")
    diff = [r["id"] for r in rows if not r["mini_matches_engine"]]
    print("    引擎的 FastIncrementalDetokenizer 走的就是这个 DecodeStream"
          "（vllm/v1/engine/detokenizer.py:185）。")
    if diff:
        print(f"    逐步吐字不同但最终文本相同的用例：{diff}"
              "——两种实现的 flush 时机不同，落到流式接口上就是「这一步的 delta 为空」。")
    (out / "detok.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")


# ---------------------------------------------------------------- stop 阶段

def run_stop(out: Path) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    id_to_bytes = build_id_to_bytes(tok)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话说明风是什么。"}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=False) if "enable_thinking" in tok.chat_template else None
    if prompt is None:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "用一句话说明风是什么。"}],
            add_generation_prompt=True, tokenize=False)
    print(f"vllm={vllm.__version__}  torch={torch.__version__}  "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"prompt（{len(tok(prompt, add_special_tokens=False)['input_ids'])} token）"
          f"{prompt!r}\n")

    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.35,
              enforce_eager=True, enable_prefix_caching=False)
    payload = dict(model=MODEL, vllm=vllm.__version__, prompt=prompt, cases=[])

    def gen(params, label):
        o = llm.generate([prompt], params)[0].outputs[0]
        rec = dict(label=label, text=o.text, n_tokens=len(o.token_ids),
                   token_ids=list(o.token_ids), finish_reason=o.finish_reason,
                   stop_reason=o.stop_reason,
                   params={k: v for k, v in dict(
                       max_tokens=params.max_tokens, temperature=params.temperature,
                       stop=params.stop,
                       include_stop_str_in_output=params.include_stop_str_in_output,
                       min_tokens=params.min_tokens).items()})
        payload["cases"].append(rec)
        return rec

    # ---- 基线：一直生成到自然结束
    head("[1] 基线：不设 stop，跑到 EOS")
    base = gen(SamplingParams(temperature=0.0, max_tokens=256), "baseline")
    print(f"    finish_reason={base['finish_reason']}  stop_reason={base['stop_reason']}  "
          f"{base['n_tokens']} 个 token")
    print(f"    文本 {base['text']!r}")
    eos_in_ids = tok.eos_token_id in base["token_ids"]
    print(f"    token_ids 里是否含 eos({tok.eos_token_id})：{eos_in_ids}；"
          f"文本里是否含 {tok.eos_token!r}：{tok.eos_token in base['text']}")
    print("    token 计数含 eos，文本不含：SamplingParams 默认 skip_special_tokens=True，"
          "两个计数因此差 1 个 token。")

    # ---- 找一个横跨两个 token 的停止串
    head("[2] 停止串横跨两个 token")
    pieces = [tok.decode([i]) for i in base["token_ids"]]
    span, cut_at = None, None
    for i in range(len(pieces) - 1):
        a, b = pieces[i], pieces[i + 1]
        if len(a) >= 2 and len(b) >= 2:
            cand = a[-2:] + b[:2]
            if base["text"].count(cand) == 1:
                span, cut_at = cand, i
                break
    print(f"    第 {cut_at} 和第 {cut_at + 1} 个 token 是 {pieces[cut_at]!r} {pieces[cut_at + 1]!r}，"
          f"取跨界 4 个字符作停止串 {span!r}")
    print(f"    这个串不等于任何单个 token 的文本："
          f"{all(span != p for p in pieces)}，只能在拼出的文本上判断")
    r_excl = gen(SamplingParams(temperature=0.0, max_tokens=256, stop=[span]),
                 "stop-span-exclude")
    r_incl = gen(SamplingParams(temperature=0.0, max_tokens=256, stop=[span],
                                include_stop_str_in_output=True), "stop-span-include")
    for r in (r_excl, r_incl):
        print(f"    {r['label']:<20s}{r['n_tokens']:>4d} token  "
              f"finish={r['finish_reason']}  stop_reason={r['stop_reason']!r}")
        print(f"      文本尾部 {r['text'][-24:]!r}")
    print(f"    两者 token 数差 {r_incl['n_tokens'] - r_excl['n_tokens']}，"
          f"文本长度差 {len(r_incl['text']) - len(r_excl['text'])} 字符")
    print("    排除停止串时引擎会把最后一个 token 整个跳过解码"
          "（detokenizer.py:107-113），文本按字符位置截断。")

    # ---- 手写缓冲与引擎输出逐字节对拍
    head("[3] 手写增量 detokenizer 与引擎输出逐字节对拍")
    specials = set(tok.all_special_ids)
    mini_all = ByteBufferDetokenizer(id_to_bytes)
    text_all = "".join(mini_all.step(i) for i in base["token_ids"])
    mini_skip = ByteBufferDetokenizer(id_to_bytes)
    text_skip = "".join(mini_skip.step(i) for i in base["token_ids"] if i not in specials)
    print(f"    引擎文本          {len(base['text'].encode()):>4d} 字节  "
          f"{hexs(base['text'].encode(), 14)}")
    print(f"    手写（全部 token）{len(text_all.encode()):>4d} 字节  "
          f"{hexs(text_all.encode(), 14)}  相同：{text_all == base['text']}")
    print(f"    手写（跳过特殊）  {len(text_skip.encode()):>4d} 字节  "
          f"{hexs(text_skip.encode(), 14)}  相同：{text_skip == base['text']}")
    print(f"    缓冲残留 {len(mini_skip.buf)} 字节。两者的差正是 eos 的 "
          f"{len(tok.eos_token.encode())} 字节——对拍前必须先对齐 skip_special_tokens。")
    payload["mini_vs_engine"] = dict(
        same_all=text_all == base["text"], same_skip_special=text_skip == base["text"],
        engine_bytes=len(base["text"].encode()), mini_all_bytes=len(text_all.encode()),
        mini_skip_bytes=len(text_skip.encode()), leftover=len(mini_skip.buf))

    # ---- EOS 与 max_tokens 同时满足
    head("[4] EOS 与 max_tokens 同时满足时，finish_reason 是哪一个")
    n = base["n_tokens"]
    for label, mt in [("max_tokens=n-1", n - 1), ("max_tokens=n", n), ("max_tokens=n+1", n + 1)]:
        r = gen(SamplingParams(temperature=0.0, max_tokens=mt), f"eos-vs-length:{label}")
        print(f"    {label:<16s}(={mt:>3d})  实际 {r['n_tokens']:>3d} token  "
              f"finish_reason={r['finish_reason']:<8s} stop_reason={r['stop_reason']!r}")
    print(f"    自然结束长度是 {n}；两条件同时满足那一行的 finish_reason 就是引擎的优先级。")

    # ---- min_tokens 与 stop 的相互作用
    head("[5] min_tokens 会让停止串失效")
    r = gen(SamplingParams(temperature=0.0, max_tokens=256, stop=[span], min_tokens=40),
            "stop-span-min-tokens")
    print(f"    min_tokens=40 时 {r['n_tokens']} token  finish={r['finish_reason']}  "
          f"stop_reason={r['stop_reason']!r}")
    print("    停止串只在 num_output_tokens > min_tokens 之后才检查"
          "（detokenizer.py:130），min_tokens 之前出现的匹配被忽略。")

    (out / "stop.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    del llm


def head(t: str) -> None:
    print(f"\n{t}\n{SEP}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["detok", "stop"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "detok":
        run_detok(out)
    else:
        run_stop(out)
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.5-B", phase=args.phase, model=MODEL, cases=[c[0] for c in CASES],
        decoding="greedy (temperature=0)", host=platform.node(),
        outputs=[f"{args.phase}.json", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
