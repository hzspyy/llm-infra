#!/usr/bin/env python3
"""L4.10 —— 语音生成、codec 与播放（任务 A：codec 账 + 打包/解包 + 拼接队列）。

[A1] 从 talker / speech_tokenizer 配置算码本账：码本数、码本大小、帧率、
     bit/frame、码率、与 PCM 的压缩比、token↔样本↔时长 的换算
[A2] 手写码本打包/解包（11 bit × 16 码本 → 22 字节/帧），两种独立实现互相对拍
[A3] 拼接队列：按 4/8/16 帧的 chunk 输出 PCM，核对有效采样点、边界不连续（爆音）
     与丢帧造成的间隙
[A4] 边界：空流、单帧、非整 chunk 长度、越界码值、静音帧

模型侧的 encode/decode 需要 `Qwen3TTSTokenizerV2Model`，本环境的 transformers
5.17 没有该类（记录为阻塞，见正文与 STATUS）。

用法：
    python labs/L4/tts_codec_probe.py --outdir out/4.10/run
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo=REPO):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


# ---------------------------------------------------------------- A1
def section_A1(args):
    title("[A1] codec 账：码本、帧率、码率与长度换算")
    d = snap()
    cfg = json.load(open(d + "/config.json"))
    st = json.load(open(d + "/speech_tokenizer/config.json"))
    talker = cfg["talker_config"]
    cp = talker["code_predictor_config"]
    n_q = st["encoder_valid_num_quantizers"]
    cb_size = st["decoder_config"]["codebook_size"]
    sr = st["output_sample_rate"]
    down = st["encode_downsample_rate"]
    up = st["decode_upsample_rate"]
    frame_samples = down
    frame_s = frame_samples / sr
    fps = 1.0 / frame_s
    bits_per_code = math.log2(cb_size)
    bits_per_frame = n_q * bits_per_code
    bytes_per_frame = math.ceil(bits_per_frame / 8)
    bitrate = bits_per_frame * fps
    pcm_bitrate = sr * 16

    print(f"  tokenizer：{st['architectures'][0]}，input/output {st['input_sample_rate']}/"
          f"{st['output_sample_rate']} Hz，downsample {down}，upsample {up}")
    print(f"  码本：{n_q} 个（RVQ），每个 {cb_size} 项（{bits_per_code:.0f} bit），"
          f"decoder codebook_dim {st['decoder_config']['codebook_dim']}、"
          f"latent_dim {st['decoder_config']['latent_dim']}")
    print(f"  talker：num_code_groups {cp['num_code_groups']}、"
          f"code predictor {cp['num_hidden_layers']} 层、vocab {cp['vocab_size']}")
    print(f"  一帧 = {frame_samples} 采样 = {frame_s*1000:.1f} ms → 帧率 {fps:.2f} Hz"
          f"（模型名里的 12Hz 即此）")
    print(f"  一帧 = {n_q} × {bits_per_code:.0f} = {bits_per_frame:.0f} bit = "
          f"{bytes_per_frame} 字节")
    print(f"  码流码率 = {bitrate:.0f} bit/s = {bitrate/8:.0f} B/s；"
          f"PCM({sr} Hz/16 bit) = {pcm_bitrate} bit/s → 压缩 {pcm_bitrate/bitrate:.0f}×")
    rows = []
    for seconds in (1, 10, 60):
        frames = seconds * fps
        rows.append({"seconds": seconds, "frames": frames,
                     "samples": int(round(frames * frame_samples)),
                     "codec_bytes": int(round(frames * bytes_per_frame)),
                     "pcm_bytes": int(round(seconds * sr * 2))})
        print(f"  {seconds:>3} s → {frames:8.1f} 帧 → "
              f"{int(round(frames*frame_samples)):>9} 采样；"
              f"码流 {int(round(frames*bytes_per_frame)):>7} B vs "
              f"PCM {int(round(seconds*sr*2)):>8} B")
    SUMMARY["A1"] = {"arch": st["architectures"][0], "n_quantizers": n_q,
                     "codebook_size": cb_size, "sample_rate": sr,
                     "downsample": down, "upsample": up,
                     "frame_samples": frame_samples, "frame_ms": frame_s * 1000,
                     "fps": fps, "bits_per_frame": bits_per_frame,
                     "bytes_per_frame": bytes_per_frame, "bitrate": bitrate,
                     "pcm_bitrate": pcm_bitrate, "compression": pcm_bitrate / bitrate,
                     "num_code_groups": cp["num_code_groups"], "rows": rows,
                     "speaker_encoder": cfg.get("speaker_encoder_config")}
    return SUMMARY["A1"]


# ---------------------------------------------------------------- A2
def pack_frame_shift(codes, bits=11, n=16):
    """实现一：位累加器（大端序，低位在前）。"""
    acc = 0
    for i, c in enumerate(codes):
        if not (0 <= c < (1 << bits)):
            raise ValueError(f"码值越界: {c}")
        acc |= (int(c) & ((1 << bits) - 1)) << (i * bits)
    nbytes = (n * bits + 7) // 8
    return acc.to_bytes(nbytes, "little")


def unpack_frame_shift(buf, bits=11, n=16):
    acc = int.from_bytes(buf, "little")
    mask = (1 << bits) - 1
    return [(acc >> (i * bits)) & mask for i in range(n)]


def pack_frame_bits(codes, bits=11, n=16):
    """实现二：numpy unpackbits（独立实现，用于互相对拍）。"""
    arr = np.array(codes, dtype=np.uint32)
    out = np.zeros(n * bits, dtype=np.uint8)
    for i in range(bits):
        out[i::bits] = (arr >> i) & 1
    return np.packbits(out, bitorder="little").tobytes()


def unpack_frame_bits(buf, bits=11, n=16):
    b = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    b = b[:n * bits].reshape(n, bits)
    return [int(sum(int(b[i, j]) << j for j in range(bits))) for i in range(n)]


def section_A2(args):
    title("[A2] 手写码本打包/解包（11 bit × 16 → 22 字节/帧）")
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 2048, size=(64, 16))
    ok = True
    for f in frames:
        a = pack_frame_shift(list(f))
        b = pack_frame_bits(list(f))
        if a != b:
            ok = False
        if list(f) != unpack_frame_shift(a) or list(f) != unpack_frame_bits(a):
            ok = False
    print(f"  64 帧随机码：位移实现与 numpy 实现逐字节相同 = {ok}；"
          f"解包往返全部一致 = {ok}")
    pkt = pack_frame_shift(list(frames[0]))
    print(f"  第一帧码 {frames[0][:6].tolist()}… → {len(pkt)} 字节 {pkt.hex()}")
    print(f"  解包回来 {unpack_frame_shift(pkt)[:6]}")
    edge = {}
    for tag, codes in (("全 0", [0] * 16), ("全 2047", [2047] * 16),
                       ("混合边界", [0, 2047] * 8)):
        p = pack_frame_shift(codes)
        edge[tag] = {"bytes": len(p), "roundtrip": unpack_frame_shift(p) == codes}
        print(f"  {tag:<10} {len(p)} 字节，往返一致 {edge[tag]['roundtrip']}")
    try:
        pack_frame_shift([2048] + [0] * 15)
        edge["越界 2048"] = {"result": "未报错"}
    except ValueError as e:
        edge["越界 2048"] = {"error": str(e)}
        print(f"  越界码值 2048 → ValueError: {e}")
    # 位浪费：11 bit 的实际利用率
    print(f"  位宽账：11 bit 可表示 2048 个码值，刚好用满（log2(2048)=11.000）；"
          f"16 码本 × 11 = 176 bit，按字节对齐后 22 字节（浪费 0 bit）")
    SUMMARY["A2"] = {"two_impls_equal": ok, "frame_bytes": len(pkt), "edges": edge}
    return SUMMARY["A2"]


# ---------------------------------------------------------------- A3
def section_A3(args):
    title("[A3] 拼接队列：chunk=4/8/16 帧与边界不连续")
    a1 = SUMMARY["A1"]
    sr, frame_samples = int(a1["sample_rate"]), int(a1["frame_samples"])
    n_frames = 40
    # 用 220 Hz 正弦填充每一帧，保证帧内连续；叠一个每帧幅度的包络
    t = np.arange(n_frames * frame_samples) / sr
    wave_full = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    rows = []
    for chunk in (4, 8, 16):
        out, joins = [], []
        pos = 0
        for start in range(0, n_frames, chunk):
            blk = wave_full[start * frame_samples:(start + chunk) * frame_samples]
            if pos > 0 and len(out):
                joins.append(float(abs(blk[0] - out[-1][-1])))
            out.append(blk)
            pos += len(blk)
        pcm = np.concatenate(out)
        maxj = max(joins) if joins else 0.0
        rows.append({"chunk_frames": chunk, "chunks": len(out),
                     "samples": len(pcm),
                     "expected_samples": n_frames * frame_samples,
                     "join_discontinuity": maxj})
        print(f"  chunk={chunk:>2} 帧：{len(out)} 个 chunk，"
              f"{len(pcm)} 采样（期望 {n_frames*frame_samples}），"
              f"拼接处最大跳变 {maxj:.2e}")
    # 丢帧：抽掉中间一帧，看间隙造成的跳变与时长损失
    keep = [i for i in range(n_frames) if i != n_frames // 2]
    pcm_drop = np.concatenate([wave_full[i * frame_samples:(i + 1) * frame_samples]
                               for i in keep])
    gap_jump = abs(wave_full[(n_frames // 2 + 1) * frame_samples] -
                   wave_full[(n_frames // 2 - 1 + 1) * frame_samples])
    print(f"  丢 1 帧：长度 {len(pcm_drop)} 采样（少 {frame_samples} = "
          f"{frame_samples/sr*1000:.0f} ms），间隙处跳变 {gap_jump:.2e}"
          f"（连续信号该处应≈0）")
    # 非整 chunk 长度
    rem = n_frames % 16
    print(f"  非整 chunk：{n_frames} 帧按 16 分块会剩 {rem} 帧（{rem*frame_samples} 采样），"
          f"队列必须支持尾包不足一块")
    SUMMARY["A3"] = {"rows": rows, "drop_gap_jump": float(gap_jump),
                     "samples_per_frame": frame_samples,
                     "n_frames": n_frames, "remainder_frames": int(rem)}
    return SUMMARY["A3"]


# ---------------------------------------------------------------- A4
def section_A4(args):
    title("[A4] 边界：空流、单帧、静音帧、越界码值")
    a1 = SUMMARY["A1"]
    sr, frame_samples = int(a1["sample_rate"]), int(a1["frame_samples"])
    edges = {}
    edges["空流"] = {"frames": 0, "samples": 0, "bytes": 0}
    print(f"  空流：0 帧 → 0 采样、0 字节（播放端应立即结束，而不是等超时）")
    edges["单帧"] = {"frames": 1, "samples": frame_samples,
                     "bytes": int(a1["bytes_per_frame"])}
    print(f"  单帧：{frame_samples} 采样 = {frame_samples/sr*1000:.0f} ms，"
          f"{int(a1['bytes_per_frame'])} 字节（首包可播放的最小单位）")
    silence = np.zeros(frame_samples * 4, dtype=np.float32)
    edges["静音帧"] = {"samples": len(silence), "rms": float(np.sqrt((silence**2).mean()))}
    print(f"  静音 4 帧：RMS {edges['静音帧']['rms']:.1f}"
          f"（码流里静音仍占 4×22=88 字节——码率与内容无关）")
    for bad in (-1, 2048):
        try:
            pack_frame_shift([bad] + [0] * 15)
            edges[f"越界 {bad}"] = {"result": "未报错"}
        except ValueError as e:
            edges[f"越界 {bad}"] = {"error": str(e)}
            print(f"  越界码值 {bad} → ValueError（打包前必须校验范围）")
    tool = {}
    for name in ("vllm", "sglang"):
        try:
            mod = __import__(name)
            tool[name] = getattr(mod, "__version__", "?")
        except Exception as e:
            tool[name] = f"不可用（{type(e).__name__}）"
    try:
        import transformers
        has = [n for n in dir(transformers) if "TTS" in n]
        tool["transformers"] = f"{transformers.__version__}；TTS 相关类 {has or '无'}"
    except Exception as e:
        tool["transformers"] = str(e)
    print(f"  运行时检查：{tool}")
    print("  → 本环境没有 Qwen3TTSTokenizerV2Model / vLLM-Omni / SGLang-Omni，"
          "模型侧 encode/decode 无法运行（见正文）")
    SUMMARY["A4"] = {"edges": edges, "runtimes": tool}
    return SUMMARY["A4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/l410_out"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    SUMMARY["env"] = {"python": sys.version.split()[0],
                      "numpy": np.__version__}
    section_A1(args)
    section_A2(args)
    section_A3(args)
    section_A4(args)
    path = os.path.join(args.outdir, "tts_codec_probe.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
