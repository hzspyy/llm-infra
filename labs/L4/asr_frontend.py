#!/usr/bin/env python3
"""L4.9 —— 语音输入与流式 ASR。

[A] 16 kHz PCM → 分帧 → log-mel 特征 → 有效长度/mask/音频 token 的长度传播；
    与官方 processor 的特征逐元素对拍；重采样、空音频、截断、静音、错误采样率反例
[B] 可重放输入分块：chunk=1/2/4 s、feed step=20/100 ms，逐块记录累计音频、
    回退文本与稳定前缀；区分「重新编码累计音频」与「伪流式 transcript」
[C] 小规模质量—成本：整段 vs 分块、batch=1/4 的 WER/CER、RTF、峰值与首个稳定转写

用法：
    python labs/L4/asr_frontend.py --outdir out/4.9/run A
    python labs/L4/asr_frontend.py --outdir out/4.9/run B
    python labs/L4/asr_frontend.py --outdir out/4.9/run C
"""

import argparse
import glob
import io
import json
import os
import sys
import time
import wave

import numpy as np
import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = "Qwen/Qwen3-ASR-0.6B"
CONTRAST = "openai/whisper-small"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo=REPO):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


# ---------------------------------------------------------------- 手写前端
def resample_linear(x, sr_in, sr_out):
    """线性插值重采样（手写，用于验证重采样规则）。"""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def frame_signal(x, frame_len, hop):
    """分帧：返回 [n_frames, frame_len] 与有效长度。"""
    if len(x) < frame_len:
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad))
        return x.reshape(1, -1), 1, pad
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx], n, 0


def logmel(x, sr=16000, n_fft=400, hop=160, n_mels=80, fmin=0, fmax=8000,
           mel_scale="htk", periodic=True, fb_override=None):
    """手写 log-mel：Hann 窗 → rfft → mel 滤波器组 → log10。"""
    win = torch.hann_window(n_fft, periodic=periodic)
    spec = torch.stft(torch.from_numpy(x).float(), n_fft, hop_length=hop,
                      win_length=n_fft, window=win, center=True,
                      return_complex=True).abs() ** 2
    fb = fb_override if fb_override is not None else \
        mel_filterbank(sr, n_fft, n_mels, fmin, fmax, scale=mel_scale)
    m = fb @ spec
    return torch.log10(m.clamp_min(1e-10))


def whisper_norm(m):
    """Whisper 的 log-mel 归一化：抬底 8 个数量级，再线性压到约 [-1,1]。"""
    m = m.clamp_min(m.max() - 8.0)
    return (m + 4.0) / 4.0


def hz_to_mel(f):
    """HTK mel（本章第一版用的就是这个）。"""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def hz_to_mel_slaney(f):
    """Slaney mel：1 kHz 以下线性、以上对数（Whisper/transformers 用的就是它）。"""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3.0                      # 200/3 Hz 一个 mel
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    if np.ndim(f) == 0:
        if f >= min_log_hz:
            return min_log_mel + np.log(f / min_log_hz) / logstep
        return mels
    hi = f >= min_log_hz
    mels[hi] = min_log_mel + np.log(f[hi] / min_log_hz) / logstep
    return mels


def mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = f_sp * m
    hi = m >= min_log_mel
    hz[hi] = min_log_hz * np.exp(logstep * (m[hi] - min_log_mel))
    return hz


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax, scale="htk"):
    """三角滤波器组；scale="slaney" 时用 Slaney mel 刻度并做 Slaney 归一化。"""
    n_freqs = n_fft // 2 + 1
    if scale == "slaney":
        mels = np.linspace(hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax), n_mels + 2)
        hz = mel_to_hz_slaney(mels)
    else:
        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, min(c, n_freqs)):
            fb[i, k] = (k - l) / (c - l)
        for k in range(c, min(r, n_freqs)):
            fb[i, k] = (r - k) / (r - c)
        if scale == "slaney":
            fb[i] *= 2.0 / (hz[i + 2] - hz[i])       # Slaney 归一化
    return torch.from_numpy(fb)


def load_wav_bytes(b):
    with wave.open(io.BytesIO(b), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2:
        x /= 32768.0
    elif sw == 4:
        x /= 2147483648.0
    else:
        x = (x - 128.0) / 128.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr, ch


def load_audio(path_or_bytes, want_sr=16000):
    """读 WAV（stdlib）或 FLAC/任意（soundfile），单声道，返回 (x, sr, ch)。"""
    if isinstance(path_or_bytes, bytes):
        try:
            return load_wav_bytes(path_or_bytes)
        except Exception:
            import soundfile as sf
            x, sr = sf.read(io.BytesIO(path_or_bytes), always_2d=True)
            return x.mean(axis=1).astype(np.float32), sr, x.shape[1]
    import soundfile as sf
    info = sf.info(path_or_bytes)
    x, sr = sf.read(path_or_bytes, always_2d=True)
    return x.mean(axis=1).astype(np.float32), sr, info.channels


def get_samples(n=3, outdir=None):
    """取真实语音：LibriSpeech dummy（FLAC 字节，用 soundfile 解码）。

    datasets 5.x 的音频解码依赖 torchcodec/ffmpeg（本机没有），所以直接读 parquet。
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    pq_path = hf_hub_download("hf-internal-testing/librispeech_asr_dummy",
                              "clean/validation-00000-of-00001.parquet",
                              repo_type="dataset")
    tbl = pq.read_table(pq_path)
    out = []
    for i in range(min(n, tbl.num_rows)):
        row = tbl.slice(i, 1).to_pylist()[0]
        b = row["audio"]["bytes"]
        x, sr = sf.read(io.BytesIO(b), always_2d=True)
        ch = x.shape[1]
        x = x.mean(axis=1).astype(np.float32)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            sf.write(os.path.join(outdir, f"librispeech_dummy_{i}.wav"), x, sr)
        out.append({"id": f"ls_dummy_{i}", "x": x, "sr": sr, "channels": ch,
                    "text": row["text"].strip(), "bytes": len(b)})
    return out


# ---------------------------------------------------------------- A
def section_A(args, ctx=None):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    title("[A] 波形到特征：长度传播、mask 与官方对拍")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    pp = json.load(open(d + "/preprocessor_config.json"))
    print(f"  processor 配置：{json.dumps(pp, ensure_ascii=False)[:400]}")
    samples = get_samples(n=args.n_samples, outdir=os.path.join(args.outdir, "audio"))
    print(f"  取到 {len(samples)} 条真实语音（LibriSpeech dummy，16 kHz 单声道）")
    rows = []
    for s in samples:
        x, sr = s["x"], s["sr"]
        dur = len(x) / sr
        x16 = resample_linear(x, sr, 16000)
        # 关键：官方先补零到 30 s（480000 采样）再算 STFT，帧数与对齐都随之变化
        n_pad = pp["n_samples"]
        x_pad = np.pad(x16, (0, max(0, n_pad - len(x16))))[:n_pad]
        feats_htk = logmel(x_pad, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                           hop=pp["hop_length"], mel_scale="htk")[..., :pp["nb_max_frames"]]
        feats = logmel(x_pad, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                       hop=pp["hop_length"], mel_scale="slaney")[..., :pp["nb_max_frames"]]
        feats_sym = logmel(x_pad, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                           hop=pp["hop_length"], mel_scale="slaney",
                           periodic=False)[..., :pp["nb_max_frames"]]
        # 对照：滤波器组直接用 transformers 的 mel_filter_bank（其余仍用本章实现）
        hf_fb = None
        try:
            from transformers.audio_utils import mel_filter_bank as _hfb
            hf_fb = torch.from_numpy(_hfb(num_frequency_bins=pp["n_fft"] // 2 + 1,
                                          num_mel_filters=pp["feature_size"],
                                          min_frequency=0.0, max_frequency=8000.0,
                                          sampling_rate=16000, norm="slaney",
                                          mel_scale="slaney")).float().T
            feats_hfb = logmel(x_pad, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                               hop=pp["hop_length"], periodic=False,
                               fb_override=hf_fb)[..., :pp["nb_max_frames"]]
            feats_hfb_per = logmel(x_pad, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                                   hop=pp["hop_length"], periodic=True,
                                   fb_override=hf_fb)[..., :pp["nb_max_frames"]]
        except Exception as e:
            feats_hfb = None
            print(f"  （取官方滤波器组失败：{type(e).__name__}: {str(e)[:80]}）")
        # 对照：不补零直接算（帧数只有有效长度），看残差是否来自补零位置
        feats_nopad = logmel(x16, n_mels=pp["feature_size"], n_fft=pp["n_fft"],
                             hop=pp["hop_length"], mel_scale="slaney",
                             periodic=False)
        feats_raw_theory = int(np.ceil(len(x16) / pp["hop_length"]))
        off = proc.feature_extractor(raw_speech=[x], sampling_rate=sr,
                                     return_tensors="pt")
        official = off["input_features"]
        n_frames = feats.shape[-1]
        # 官方特征固定补到 3000 帧（chunk_length=30 s / nb_max_frames=3000），
        # 只在有效帧上对拍；padding 区的取值另算
        valid = min(feats.shape[-1], official.shape[-1])
        d_raw = float((feats.unsqueeze(0)[..., :valid] -
                       official[..., :valid]).abs().max().item())
        d_htk = float((whisper_norm(feats_htk).unsqueeze(0)[..., :valid] -
                       official[..., :valid]).abs().max().item())
        # 把残差分成「有效帧」与「补零帧」两段，定位差异来源
        mine_n = whisper_norm(feats).unsqueeze(0)
        v = feats_raw_theory
        d_sym = float((whisper_norm(feats_sym).unsqueeze(0)[..., :valid] -
                       official[..., :valid]).abs().max().item())
        mn = min(feats_nopad.shape[-1], valid)
        d_nopad = float((whisper_norm(feats_nopad).unsqueeze(0)[..., :mn] -
                         official[..., :mn]).abs().max().item())
        d_hfb_per = None if feats_hfb is None else float(
            (whisper_norm(feats_hfb_per).unsqueeze(0)[..., :valid] -
             official[..., :valid]).abs().max().item())
        d_hfb = None if feats_hfb is None else float(
            (whisper_norm(feats_hfb).unsqueeze(0)[..., :valid] -
             official[..., :valid]).abs().max().item())
        if hf_fb is not None:
            mine_fb = mel_filterbank(16000, pp["n_fft"], pp["feature_size"], 0, 8000,
                                     scale="slaney")
            n_diff = int((mine_fb - hf_fb).abs().max(dim=1).values.gt(1e-9).sum().item())
            print(f"  滤波器组对照：我的 slaney fb 与官方 mel_filter_bank 有 "
                  f"{n_diff}/{pp['feature_size']} 行不同，最大元素差 "
                  f"{(mine_fb - hf_fb).abs().max().item():.3e}")
        d_valid = float((mine_n[..., :v] - official[..., :v]).abs().max().item())
        d_pad = float((mine_n[..., v:valid] - official[..., v:valid]).abs().max().item()) if valid > v else None
        dmax = float((whisper_norm(feats).unsqueeze(0)[..., :valid] -
                      official[..., :valid]).abs().max().item())
        rows.append({"id": s["id"], "sr": sr, "channels": s["channels"],
                     "my_vs_official_max_abs_diff": dmax,
                     "raw_logmel_vs_official_max_abs_diff": d_raw,
                     "htk_scale_max_abs_diff": d_htk,
                     "symmetric_window_max_abs_diff": d_sym,
                     "no_pad_max_abs_diff": d_nopad,
                     "official_filterbank_max_abs_diff": d_hfb,
                     "official_filterbank_periodic_max_abs_diff": d_hfb_per,
                     "valid_region_max_abs_diff": d_valid,
                     "padded_region_max_abs_diff": d_pad,
                     "samples": len(x), "duration_s": dur,
                     "my_mel_frames": feats_raw_theory,
                     "official_shape": list(official.shape),
                     "official_frames": int(official.shape[-1]),
                     "official_pad_value": float(official[0, 0, -1].item()),
                     "my_pad_value": float(feats[..., -1].mean().item()),
                     "text": s["text"][:60], "wav_bytes": s["bytes"]})
        print(f"  {s['id']}: {sr} Hz / {s['channels']} ch / {len(x)} 采样 / "
              f"{dur:.2f} s → 我 {n_frames} 帧，官方 {tuple(official.shape)}，"
              f"max|diff| {dmax}")
    m = rows[0]
    print(f"\n  长度传播（第一条）：采样 {m['samples']} → 帧 {m['my_mel_frames']}"
          f"（官方 {m['official_frames']}）")
    print(f"  公式：帧数 ≈ ceil(采样数 / hop)（hop=160 = 10 ms），"
          f"理论 {int(np.ceil(m['samples']/160))}")
    SUMMARY["A"] = {"preprocessor": pp, "rows": rows}

    sub("A2 边界与非法输入")
    edges = {}

    def case(tag, fn):
        try:
            r = fn()
            edges[tag] = {"result": r}
            print(f"  {tag:<26} {r}")
        except Exception as e:
            edges[tag] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
            print(f"  {tag:<26} {type(e).__name__}: {str(e)[:110]}")

    x0 = samples[0]["x"]
    case("错误采样率 8k（不重采样）", lambda: f"直接送 8k："
         f"{tuple(proc.feature_extractor(raw_speech=[x0], sampling_rate=8000, return_tensors='pt')['input_features'].shape)}"
         f"（长度翻倍，但内容是原波形的错位解释）")
    case("重采样到 16k（正确做法）", lambda: f"{tuple(proc.feature_extractor(raw_speech=[resample_linear(x0, 16000, 16000)], sampling_rate=16000, return_tensors='pt')['input_features'].shape)}")
    case("空音频（0 采样）", lambda: f"{tuple(proc.feature_extractor(raw_speech=[np.zeros(0, dtype=np.float32)], sampling_rate=16000, return_tensors='pt')['input_features'].shape)}")
    case("纯静音 1 s", lambda: f"{tuple(proc.feature_extractor(raw_speech=[np.zeros(16000, dtype=np.float32)], sampling_rate=16000, return_tensors='pt')['input_features'].shape)}")
    case("截断到 0.2 s", lambda: f"{tuple(proc.feature_extractor(raw_speech=[x0[:3200]], sampling_rate=16000, return_tensors='pt')['input_features'].shape)}")
    def mask_info(audio, sr=16000):
        o = proc.feature_extractor(raw_speech=[audio], sampling_rate=sr,
                                   return_tensors="pt")
        m = o.get("attention_mask")
        f = o["input_features"]
        if m is None:
            return f"features {tuple(f.shape)}，无 attention_mask"
        keep = int(m.sum().item())
        return (f"features {tuple(f.shape)}，attention_mask 有效**采样**数 {keep}"
                f"（输入 {len(audio)} 采样；上限 30 s = 480000 采样）")

    case("超长（拼接 60 s）", lambda: mask_info(np.tile(x0, 60)))
    case("空音频的 mask", lambda: mask_info(np.zeros(0, dtype=np.float32)))
    case("静音 1 s 的 mask", lambda: mask_info(np.zeros(16000, dtype=np.float32)))
    case("截断 0.2 s 的 mask", lambda: mask_info(x0[:3200]))
    case("31 s（超过 30 s 上限）", lambda: mask_info(np.tile(x0, 6)[:16000 * 31]))
    SUMMARY["A_edges"] = edges
    return ctx


# ---------------------------------------------------------------- B
def section_B(args, ctx=None):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    title("[B] 输入分块与稳定前缀")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    if ctx is None:
        ctx = Qwen3ASRForConditionalGeneration.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx
    samples = get_samples(n=1)
    x = samples[0]["x"]
    truth = samples[0]["text"]
    print(f"  音频 {len(x)/16000:.2f} s；参考转写：{truth[:70]}")

    def transcribe(audio):
        ins = proc(text=[""] * 1, audio=[audio], sampling_rate=16000,
                   return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ins, max_new_tokens=200, do_sample=False)
        dt = time.perf_counter() - t0
        txt = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        return txt, dt

    full, t_full = transcribe(x)
    print(f"  整段：{t_full*1e3:.0f} ms  {full[:70]!r}")
    rows = []
    for chunk_s in (1, 2, 4):
        chunk = int(chunk_s * 16000)
        step = chunk                       # 每满一个 chunk 喂一次（重放累计音频）
        prefixes, times = [], []
        for end in range(step, len(x) + 1, step):
            txt, dt = transcribe(x[:end])
            prefixes.append(txt)
            times.append(dt)
        # 稳定前缀：与最终整段结果的最长公共前缀
        def lcp(a, b):
            n = 0
            for u, v in zip(a, b):
                if u != v:
                    break
                n += 1
            return a[:n]
        stables = [lcp(p, full) for p in prefixes]
        rows.append({"chunk_s": chunk_s, "n_chunks": len(prefixes),
                     "final_chunked": prefixes[-1] if prefixes else "",
                     "times_s": times,
                     "stable_prefix_chars": [len(s) for s in stables],
                     "stable_prefix_last": stables[-1][:60] if stables else "",
                     "matches_full": (prefixes[-1] == full) if prefixes else False})
        print(f"  chunk={chunk_s}s：喂 {len(prefixes)} 次，"
              f"最后一次与整段相同={rows[-1]['matches_full']}，"
              f"稳定前缀字符数 {rows[-1]['stable_prefix_chars']}")
    print("\n  读法：每次喂入的都是**累计音频**，因此每次都在重新编码整段之前的内容；")
    print("  这不是缓存增量式的流式，只是「切片重放 + 伪流式 transcript」。")
    print(f"  整段耗时 {t_full*1e3:.0f} ms；分块总耗时分别为 "
          f"{[round(sum(r['times_s'])*1e3) for r in rows]} ms（重复计算的代价）")
    SUMMARY["B"] = {"truth": truth, "full": full, "full_s": t_full, "rows": rows}
    del model
    torch.cuda.empty_cache()
    return None


# ---------------------------------------------------------------- C
def section_C(args, ctx=None):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    title("[C] 整段 vs 分块、batch 与 WER/CER")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    if ctx is None:
        ctx = Qwen3ASRForConditionalGeneration.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx
    samples = get_samples(n=args.n_samples)
    print(f"  样本 {len(samples)} 条（真实语音 + 参考转写）")

    def norm(t):
        import re
        t = t.lower()
        t = re.sub(r"[^a-z0-9' ]+", " ", t)
        return " ".join(t.split())

    def wer(ref, hyp):
        r, h = norm(ref).split(), norm(hyp).split()
        dp = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            prev, dp[0] = dp[0], i
            for j, hw in enumerate(h, 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                            prev + (rw != hw))
                prev = cur
        return (dp[len(h)] / max(1, len(r)), len(r))

    def run(batch_size, chunk_s=None):
        outs, times = [], []
        for i in range(0, len(samples), batch_size):
            grp = samples[i:i + batch_size]
            audios = []
            for s in grp:
                a = s["x"]
                if chunk_s:                     # 分块=每 chunk_s 秒喂一次累计音频
                    cut = min(len(a), int(chunk_s * 16000) * max(1, len(a) // int(chunk_s * 16000)))
                    a = a[:cut]
                audios.append(a)
            ins = proc(text=[""] * len(audios), audio=audios,
                       sampling_rate=16000, return_tensors="pt",
                       padding=True).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**ins, max_new_tokens=200, do_sample=False)
            dt = time.perf_counter() - t0
            txts = proc.batch_decode(out, skip_special_tokens=True)
            for s, t in zip(grp, txts):
                outs.append((s, t.strip()))
            times.append(dt)
            peak = torch.cuda.max_memory_allocated() / 2**20
        eff = [wer(s["text"], t) for s, t in outs]
        audio_s = sum(len(s["x"]) / 16000 for s in samples)
        return {"batch": batch_size, "chunk_s": chunk_s,
                "wer": sum(e[0] for e in eff) / len(eff),
                "words": sum(e[1] for e in eff),
                "total_s": sum(times), "rtf": sum(times) / audio_s,
                "audio_s": audio_s, "peak_mib": peak,
                "pairs": [{"id": s["id"], "ref": s["text"][:50], "hyp": t[:50],
                           "wer": e[0]} for (s, t), e in zip(outs, eff)]}

    results = []
    for bs in (1, 4):
        r = run(bs)
        results.append(r)
        print(f"  batch={bs} 整段：WER {r['wer']:.3f}（{r['words']} 词） "
              f"总耗时 {r['total_s']*1e3:.0f} ms  RTF {r['rtf']:.3f}  "
              f"峰值 {r['peak_mib']:.0f} MiB")
    r = run(1, chunk_s=2)
    results.append(r)
    print(f"  batch=1 chunk=2s：WER {r['wer']:.3f}  总耗时 {r['total_s']*1e3:.0f} ms  "
          f"RTF {r['rtf']:.3f}")
    for p in results[0]["pairs"]:
        print(f"    {p['id']}: WER {p['wer']:.2f}  ref {p['ref'][:40]!r}  "
              f"hyp {p['hyp'][:40]!r}")
    SUMMARY["C"] = {"n": len(samples), "results": results}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l49_out"))
    ap.add_argument("--n-samples", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A"]
    SUMMARY["env"] = {"torch": torch.__version__,
                      "transformers": __import__("transformers").__version__,
                      "gpu": torch.cuda.get_device_name(0)}
    ctx = None
    for s in want:
        if s == "A":
            ctx = section_A(args, ctx)
        elif s == "B":
            ctx = section_B(args, ctx)
        elif s == "C":
            ctx = section_C(args, ctx)
    path = os.path.join(args.outdir, "asr_frontend.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
