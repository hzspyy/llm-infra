#!/usr/bin/env python3
"""L4.10 修订（任务 B 的时序部分，替代模型）—— 首可播放与 RTF。

主例 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 在本环境无法运行（transformers 5.17 没有
`Qwen3TTSTokenizerV2Model`/talker 类，也没有 vLLM-Omni/SGLang-Omni），因此这里用
**非自回归的替代 TTS `facebook/mms-tts-eng`（VITS）** 量「文本→波形」的时序，
所有数字都标注为替代模型结果。

它回答的是 4.10-B 的前半部分：
  [B1] 不同长度文本的生成耗时、音频时长与 RTF
  [B2] 首可播放的语义：非 AR 模型一次前向出整段 → 首可播放 = 全部生成时间
       （与 AR talker 的「逐帧 RTF≈1」是完全不同的形状）
  [B3] 用实测值替换 4.11 会话模拟里「talker 每帧 80 ms」的假设值

用法：
    python labs/L4/tts_stream_bench.py --outdir out/4.10/20260913-tts
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
SUBSTITUTE = "facebook/mms-tts-eng"
TEXTS = {
    "短（10 词）": "Hello, this is a short sentence for timing.",
    "中（30 词）": ("The history of speech synthesis spans several decades, from "
                "formant synthesizers to neural vocoders, and each generation "
                "changed what latency and quality meant in practice."),
    "长（80 词）": ("In a streaming speech system the first packet and the first "
                "playable audio are different quantities. The first packet "
                "measures how fast bytes reach the client, while the first "
                "playable audio measures how long the client must buffer before "
                "the sound card can start. When generation is slower than "
                "playback the buffer drains and the listener hears a gap. This "
                "is why real time factor and buffer thresholds have to be "
                "reported together with quality metrics."),
}
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def snap(repo):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/l410b"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    title("[B] 替代模型 mms-tts-eng（VITS）：生成耗时、RTF 与首可播放")
    print(f"  替代模型 {SUBSTITUTE}：Qwen3-TTS 在本环境没有可用的类（见 STATUS）")
    from transformers import VitsModel, AutoTokenizer
    d = snap(SUBSTITUTE)
    tok = AutoTokenizer.from_pretrained(d)
    model = VitsModel.from_pretrained(d, dtype=torch.float32).to("cuda").eval()
    sr = model.config.sampling_rate
    print(f"  采样率 {sr} Hz")

    def synth(text):
        ins = tok(text, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = model(**ins).waveform
        torch.cuda.synchronize()
        return wav.reshape(-1).float().cpu().numpy(), time.perf_counter() - t0

    # 预热
    synth("warm up")
    rows = []
    for name, text in TEXTS.items():
        ts = []
        for _ in range(args.repeats):
            wav, dt = synth(text)
            ts.append(dt)
        dt = float(np.median(ts))
        audio_s = len(wav) / sr
        rows.append({"text": name, "chars": len(text), "audio_s": audio_s,
                     "gen_s": dt, "rtf": dt / audio_s,
                     "first_playable_s": dt,
                     "ms_per_s_audio": dt / audio_s * 1e3})
        print(f"  {name:<12} 字符 {len(text):>3}  音频 {audio_s:5.2f} s  "
              f"生成 {dt*1e3:7.1f} ms  RTF {dt/audio_s:6.3f}  "
              f"每秒音频 {dt/audio_s*1e3:6.1f} ms")
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  峰值显存 {peak:.0f} MiB；{args.repeats} 次取中位数")

    print("\n  读法：VITS 是**非自回归**模型——一次前向出整段波形，")
    print("  因此「首可播放」= 全部生成时间，中间没有任何可播片段；")
    print("  而 AR talker（Qwen3-TTS 的路线）逐帧生成、RTF≈1，首可播放≈第一帧的时间，")
    print("  之后靠缓冲维持。**两种形状的 RTF 与首可播放含义完全不同**，不能互相套用。")
    ref = 80.0
    per_s = float(np.mean([r["ms_per_s_audio"] for r in rows]))
    print(f"\n  与 4.11 会话模拟的假设对照：那里假设 talker 每帧 80 ms"
          f"（= 每秒音频 1000 ms，RTF 1.0）；")
    print(f"  本替代模型实测每秒音频 {per_s:.1f} ms（RTF {per_s/1000:.3f}）。")
    print("  差距来自架构：AR talker 的每帧成本由解码步决定，非 AR vocoder 由一次前向决定。")

    SUMMARY["B_substitute"] = {"model": SUBSTITUTE, "sampling_rate": sr,
                               "rows": rows, "peak_mib": peak,
                               "ms_per_s_audio_mean": per_s,
                               "note": "非自回归：首可播放 = 全部生成时间；不能与 AR talker 的逐帧形状互换"}
    path = os.path.join(args.outdir, "tts_stream_bench.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
