#!/usr/bin/env python3
"""L4.9 修订（任务 B/C，替代模型）—— 分块重放与整段评测。

主模型 `Qwen/Qwen3-ASR-0.6B` 在本环境无法加载（权重键 `thinker.` 前缀与 HF 类
不匹配，且包索引只提供 transformers ≤5.17.0），因此 B/C 用计划里列出的**结构对照
模型 `openai/whisper-small`** 做，所有结论都标注为替代模型的结果。

[B] 可重放输入分块：把同一段音频按 1/2/4 s 的前缀依次送入，记录
    每次的转写、与最终整段转写的最长公共前缀（稳定前缀）与重新编码次数；
    区分「重新编码累计音频」与「真流式增量编码」
[C] LibriSpeech test-clean 的固定样本：整段与分块两种方式的 WER/CER、
    batch=1/4 的 RTF、峰值显存与重复/漏字

用法：
    python labs/L4/asr_stream_bench.py --outdir out/4.9/20260913-asr B
    python labs/L4/asr_stream_bench.py --outdir out/4.9/20260913-asr C --n 24
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import time

import numpy as np
import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
SUBSTITUTE = "openai/whisper-small"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def load_samples(n):
    """LibriSpeech test-clean：直接读 parquet 字节 + soundfile 解码（绕开 torchcodec）。"""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    p = hf_hub_download("openslr/librispeech_asr", "all/test.clean/0000.parquet",
                        repo_type="dataset")
    tbl = pq.read_table(p)
    out = []
    for i in range(min(n, tbl.num_rows)):
        row = tbl.slice(i, 1).to_pylist()[0]
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), always_2d=True)
        out.append({"id": row.get("id", f"ls_{i}"), "x": x.mean(axis=1).astype(np.float32),
                    "sr": sr, "text": row["text"].strip()})
    return out


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", str(t).lower()).split())


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return dp[len(h)] / max(1, len(r)), len(r)


def load_model():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    d = snap(SUBSTITUTE)
    proc = WhisperProcessor.from_pretrained(d)
    model = WhisperForConditionalGeneration.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def transcribe(proc, model, audios, max_new_tokens=200):
    ins = proc(audio=audios, sampling_rate=16000, return_tensors="pt",
               padding="max_length").to("cuda")
    with torch.no_grad():
        feats = ins["input_features"].to(torch.bfloat16)
        out = model.generate(feats, max_new_tokens=max_new_tokens)
    return [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True)]


def section_B(args):
    title("[B] 分块重放与稳定前缀（替代模型 whisper-small）")
    print(f"  替代模型 {SUBSTITUTE}：Qwen3-ASR 在本环境无法加载（见 STATUS）")
    proc, model = load_model()
    s = max(load_samples(8), key=lambda x: len(x["x"]))     # 取最长的一条
    x, truth = s["x"], s["text"]
    print(f"  样本 {s['id']}：{len(x)/16000:.2f} s；参考转写 {truth[:60]!r}")
    t0 = time.perf_counter()
    full = transcribe(proc, model, [x])[0]
    t_full = time.perf_counter() - t0
    print(f"  整段：{full[:70]!r}  {t_full*1e3:.0f} ms")

    def lcp(a, b):
        n = 0
        for u, v in zip(a, b):
            if u != v:
                break
            n += 1
        return a[:n]

    rows = []
    for chunk_s in (1, 2, 4):
        chunk = int(chunk_s * 16000)
        recs = []
        for end in range(chunk, len(x) + 1, chunk):
            t0 = time.perf_counter()
            txt = transcribe(proc, model, [x[:end]])[0]
            recs.append({"end_s": end / 16000, "text": txt,
                         "ms": (time.perf_counter() - t0) * 1e3,
                         "stable_len": len(lcp(txt, full))})
        first_stable = next((r["end_s"] for r in recs
                             if r["stable_len"] >= max(10, len(full) * 0.5)), None)
        rows.append({"chunk_s": chunk_s, "n_reencodes": len(recs),
                     "total_ms": sum(r["ms"] for r in recs),
                     "last_equals_full": recs[-1]["text"] == full,
                     "stable_prefix_last": lcp(recs[-1]["text"], full)[:60],
                     "first_stable_end_s": first_stable,
                     "records": recs})
        print(f"  chunk={chunk_s}s：重新编码 {len(recs)} 次，"
              f"总耗时 {sum(r['ms'] for r in recs):.0f} ms（整段 {t_full*1e3:.0f} ms），"
              f"最后一次与整段相同={recs[-1]['text']==full}，"
              f"稳定前缀首次达到一半长度于 {first_stable} s")
    print("  读法：每次送入的都是**累计音频** → 每块都重新编码了之前的内容；")
    print("  这不是增量流式，只是「切片重放 + 伪流式 transcript」；")
    print("  末块转写正确也不能证明中间前缀稳定（看 stable_prefix 序列）。")
    SUMMARY["B"] = {"model": SUBSTITUTE, "sample": s["id"],
                    "duration_s": len(x) / 16000, "truth": truth, "full": full,
                    "full_ms": t_full * 1e3, "rows": rows}
    del model
    torch.cuda.empty_cache()


def section_C(args):
    title("[C] 整段 vs 分块：WER、RTF 与 batch（替代模型 whisper-small）")
    proc, model = load_model()
    samples = load_samples(args.n)
    durs = sorted(len(s["x"]) / 16000 for s in samples)
    print(f"  样本 {len(samples)} 条（LibriSpeech test-clean 前 {len(samples)} 条）："
          f"时长 {durs[0]:.2f}–{durs[-1]:.2f} s，合计 {sum(durs):.1f} s")

    def run(batch, chunk_s=None):
        outs, times = [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(samples), batch):
            grp = samples[i:i + batch]
            if chunk_s is None:
                audios = [g["x"] for g in grp]
                parts = [None] * len(grp)
            else:
                audios, parts = [], []
                for g in grp:
                    n = len(g["x"]) // int(chunk_s * 16000)
                    audios.append(g["x"][:n * int(chunk_s * 16000)])
                    parts.append(n)
            t0 = time.perf_counter()
            txts = transcribe(proc, model, audios)
            times.append(time.perf_counter() - t0)
            for g, t in zip(grp, txts):
                outs.append((g, t))
        eff = [wer(g["text"], t) for g, t in outs]
        audio_s = sum(durs)
        return {"batch": batch, "chunk_s": chunk_s,
                "wer": sum(e[0] for e in eff) / len(eff),
                "words": sum(e[1] for e in eff),
                "total_s": sum(times), "rtf": sum(times) / audio_s,
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                "pairs": [{"id": g["id"], "ref": g["text"][:50], "hyp": t[:50],
                           "wer": e[0]} for (g, t), e in zip(outs, eff)]}

    results = []
    for bs in (1, 4):
        r = run(bs)
        results.append(r)
        print(f"  batch={bs} 整段：WER {r['wer']:.3f}（{r['words']} 词） "
              f"总耗时 {r['total_s']:.1f} s  RTF {r['rtf']:.3f}  峰值 {r['peak_mib']:.0f} MiB")
    r = run(1, chunk_s=5)
    results.append(r)
    print(f"  batch=1 每 5 s 切一段：WER {r['wer']:.3f}  RTF {r['rtf']:.3f}")
    for p in results[0]["pairs"][:4]:
        print(f"    {p['id']}: WER {p['wer']:.2f}  ref {p['ref'][:38]!r}  hyp {p['hyp'][:38]!r}")
    SUMMARY["C"] = {"model": SUBSTITUTE, "n": len(samples),
                    "durations": durs, "results": results}
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["B"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l49b"))
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    SUMMARY["env"] = {"torch": torch.__version__,
                      "transformers": __import__("transformers").__version__,
                      "gpu": torch.cuda.get_device_name(0)}
    for s in [x.upper() for x in args.sections]:
        if s == "B":
            section_B(args)
        elif s == "C":
            section_C(args)
    path = os.path.join(args.outdir, "asr_stream_bench.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
