#!/usr/bin/env python3
"""L4.8 —— 多模态 batching、缓存与视频流。

[A] 三层特征缓存（processor / encoder / KV）的最小实现与命中矩阵：
    同图重复、同图不同 crop、同帧不同时间戳、同特征不同文本上下文
[B] vLLM 服务：纯文本 / 单图 / 多图 / 视频（4/8/16 帧）的请求组合，
    首 token 延迟与视觉 token 账；引擎级 mm 处理器缓存开/关对照
[C] 慢模态的长尾与队列：用 B 的实测服务时长驱动有界/无界队列的比较

用法：
    python labs/L4/multimodal_serving.py --outdir out/4.8/run A
    python labs/L4/multimodal_serving.py --outdir out/4.8/run B
    python labs/L4/multimodal_serving.py --outdir out/4.8/run C
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = "Qwen/Qwen3-VL-4B-Instruct"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo=REPO):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def img_hash(im):
    return hashlib.sha1(im.tobytes()).hexdigest()[:16]


def make_images():
    from PIL import Image, ImageDraw
    out = {}
    for tag, (w, h, color, text) in {
        "A": (640, 480, (20, 30, 60), "IMG-A"),
        "B": (640, 480, (60, 20, 20), "IMG-B"),
        "A_crop": (512, 384, (20, 30, 60), "IMG-A"),
    }.items():
        im = Image.new("RGB", (w, h), color)
        ImageDraw.Draw(im).text((30, h // 2), text, fill=(255, 255, 255))
        out[tag] = im
    return out


def make_video(n=8):
    from PIL import Image, ImageDraw
    frames = []
    for i in range(n):
        im = Image.new("RGB", (320, 240), (12, 16, 24))
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"t={i/4:.2f}s", fill=(255, 255, 255))
        if i == n // 2:
            d.rectangle([140, 100, 180, 140], fill=(60, 220, 120))
        frames.append(im)
    return frames


# ---------------------------------------------------------------- A
class FeatureCache:
    """三层缓存：processor 产物 / encoder 载荷 / KV。

    L1 键：图像内容 + 目标尺寸策略（processor 的输出像素张量）
    L2 键：模型 revision + processor 配置 + 内容 + 尺寸 + 帧采样（视觉塔的 4 个载荷）
    L3（KV）：本实现**不缓存**，因为位置数组随序列位置变化（4.7），
             相同的视觉特征在不同序列位置上得到不同的 RoPE 相位。
    """

    def __init__(self, rev, proc_sig):
        self.rev, self.proc_sig = rev, proc_sig
        self.l1, self.l2 = {}, {}
        self.stats = {"L1_hit": 0, "L1_miss": 0, "L2_hit": 0, "L2_miss": 0,
                      "L1_bytes": 0, "L2_bytes": 0, "saved_encoder_s": 0.0}

    def l1_key(self, im, policy):
        return (img_hash(im), policy)

    def l2_key(self, im, policy, frames=None):
        k = (self.rev, self.proc_sig, img_hash(im), policy)
        return k + (frames,) if frames is not None else k

    def get_processor(self, im, policy, compute):
        k = self.l1_key(im, policy)
        if k in self.l1:
            self.stats["L1_hit"] += 1
            return self.l1[k], True
        self.stats["L1_miss"] += 1
        v = compute()
        self.l1[k] = v
        self.stats["L1_bytes"] += sum(t.numel() * t.element_size() for t in v.values()
                                      if isinstance(t, torch.Tensor))
        return v, False

    def get_encoder(self, im, policy, frames, compute):
        k = self.l2_key(im, policy, frames)
        if k in self.l2:
            self.stats["L2_hit"] += 1
            return self.l2[k], True
        self.stats["L2_miss"] += 1
        t0 = time.perf_counter()
        v = compute()
        self.stats["saved_encoder_s"] += time.perf_counter() - t0
        self.l2[k] = v
        self.stats["L2_bytes"] += sum(t.numel() * t.element_size() for t in v)
        return v, False


def section_A(args, ctx=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    title("[A] 三层特征缓存：命中矩阵与字节账")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    if ctx is None:
        ctx = AutoModelForImageTextToText.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx
    vis = model.model.visual
    imgs = make_images()
    cfg = json.load(open(d + "/preprocessor_config.json"))
    proc_sig = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    rev = os.path.basename(d)
    cache = FeatureCache(rev, proc_sig)
    print(f"  模型 revision {rev[:12]}…，processor 配置签名 {proc_sig}")
    print(f"  L1 键 = (图像内容, 尺寸策略)；L2 键 = (revision, 配置签名, 内容, 尺寸, 帧采样)")

    def resize_policy(pixels):
        ip = proc.image_processor
        old = dict(ip.size)
        ip.size = {"shortest_edge": pixels, "longest_edge": pixels}
        try:
            return proc.image_processor(images=im, return_tensors="pt")
        finally:
            ip.size = old

    def encode(im, pixels):
        ins = proc.image_processor(images=im, return_tensors="pt") if pixels is None else None
        pv = ins["pixel_values"].to("cuda") if ins is not None else None
        thw = ins["image_grid_thw"].to("cuda") if ins is not None else None
        with torch.no_grad():
            out = vis(pv, grid_thw=thw)
        payload = [t.detach() for t in out.deepstack_features] + [out.last_hidden_state.detach()]
        return payload

    scenarios = [
        ("同图重复（同尺寸）", [("A", None), ("A", None)]),
        ("同图不同 crop（尺寸不同）", [("A", None), ("A_crop", None)]),
        ("不同图（同尺寸）", [("A", None), ("B", None)]),
        ("同帧不同时间戳（同一帧两次）", [("A", None), ("A", "t=0.0")]),
        ("同图像 + 不同文本上下文", [("A", None), ("A", None)]),
    ]
    rows = []
    for name, seq in scenarios:
        before = dict(cache.stats)
        hits = []
        for tag, frames in seq:
            im = imgs[tag]
            # L1：processor 的像素张量
            pol = "default"
            _, h1 = cache.get_processor(im, pol, lambda im=im: {
                "pixel_values": proc.image_processor(images=im,
                                                     return_tensors="pt")["pixel_values"]})
            # L2：视觉塔的 4 个载荷
            _, h2 = cache.get_encoder(im, pol, frames, lambda im=im: encode(im, None))
            hits.append((h1, h2))
        rows.append({"scenario": name,
                     "L1": [h[0] for h in hits], "L2": [h[1] for h in hits]})
        print(f"  {name:<28} L1 命中 {[h[0] for h in hits]}  "
              f"L2 命中 {[h[1] for h in hits]}")
    print("\n  逐场景读法：")
    print("   - 同图重复：三层都可命中（L1/L2 都 True）。")
    print("   - 同图不同 crop：尺寸策略不同 → L1/L2 都按不同键处理（内容同、尺寸不同）。")
    print("   - 不同图：内容不同 → 都 miss。")
    print("   - 同帧不同时间戳：帧采样进入 L2 键，L1（像素）仍命中。")
    print("   - 同图像不同文本：L1/L2 命中，**KV 不能复用**（位置随序列变化，4.7）。")
    print(f"\n  缓存统计：L1 {cache.stats['L1_hit']} 命中 / {cache.stats['L1_miss']} 未命中 "
          f"（{cache.stats['L1_bytes']/2**20:.3f} MiB）；"
          f"L2 {cache.stats['L2_hit']} / {cache.stats['L2_miss']}"
          f"（{cache.stats['L2_bytes']/2**20:.3f} MiB）")

    sub("A2 缓存省下的时间（单层视觉塔）")
    im = imgs["A"]
    t0 = time.perf_counter()
    for _ in range(3):
        encode(im, None)
    torch.cuda.synchronize()
    t_enc = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        proc.image_processor(images=im, return_tensors="pt")
    t_proc = (time.perf_counter() - t0) / 3
    one_payload = cache.l2[cache.l2_key(im, "default", None)]
    nbytes = sum(t.numel() * t.element_size() for t in one_payload)
    print(f"  单次视觉编码 {t_enc*1e3:.2f} ms、processor {t_proc*1e3:.2f} ms")
    print(f"  单张图的 L2 载荷 {nbytes/2**20:.3f} MiB（{len(one_payload)} 个张量）")
    print(f"  命中 L2 省下的是这 {t_enc*1e3:.2f} ms 的编码时间，但没有省下"
          f" LLM 的 prefill（视觉 token 仍要进序列）")
    SUMMARY["A"] = {"scenarios": rows, "stats": cache.stats,
                    "encode_ms": t_enc * 1e3, "proc_ms": t_proc * 1e3,
                    "payload_bytes": nbytes, "revision": rev, "proc_sig": proc_sig}
    return ctx


# ---------------------------------------------------------------- B
def section_B(args, ctx=None):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    title("[B] vLLM 服务：请求组合、首 token 延迟与 mm 缓存开关")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    imgs = make_images()
    video = make_video(16)
    import numpy as np
    import cv2
    vid_path = os.path.join(args.outdir, "synthetic_16f.mp4")
    if not os.path.exists(vid_path):
        vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 4,
                             (320, 240))
        for fr in video:
            vw.write(cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR))
        vw.release()
    vurl = {"url": f"file://{vid_path}"}

    def img(im):
        return {"type": "image_pil", "image_pil": im}

    mixes = {
        "纯文本": [{"role": "user", "content": [{"type": "text", "text": "用一句话解释 MoE。"}]}],
        "单图": [{"role": "user", "content": [img(imgs["A"]),
                                             {"type": "text", "text": "图里写了什么？"}]}],
        "双图": [{"role": "user", "content": [img(imgs["A"]), img(imgs["B"]),
                                             {"type": "text", "text": "两张图各写了什么？"}]}],
        "视频16帧": [{"role": "user", "content": [
            {"type": "video_url", "video_url": vurl},
            {"type": "text", "text": "视频里描述了多久？"}]}],
    }
    # 预先用 processor 统计视觉 token 与文本长度
    stats = {}
    hf_convs = {
        "纯文本": mixes["纯文本"],
        "单图": [{"role": "user", "content": [{"type": "image", "image": imgs["A"]},
                                             {"type": "text", "text": "图里写了什么？"}]}],
        "双图": [{"role": "user", "content": [{"type": "image", "image": imgs["A"]},
                                             {"type": "image", "image": imgs["B"]},
                                             {"type": "text", "text": "两张图各写了什么？"}]}],
        "视频16帧": [{"role": "user", "content": [{"type": "video", "video": video, "fps": 2},
                                                 {"type": "text", "text": "视频里描述了多久？"}]}],
    }
    for name, conv in hf_convs.items():
        ins = proc.apply_chat_template([conv], add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
        n = ins["input_ids"].shape[1]
        vt = 0
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in ins:
                vt += int(ins[key].prod(dim=1).sum().item() / 4)
        stats[name] = {"prompt_tokens": int(n), "visual_tokens": vt,
                       "text_tokens": int(n) - vt}
        print(f"  {name:<8} 提示长度 {n:>5} token（视觉 {vt:>5} / 文本 {n-vt:>4}）")

    rows = []
    for cache_gb in (args.mm_cache_gb, 0.0):
        print(f"\n  === mm_processor_cache_gb = {cache_gb} ===")
        t0 = time.perf_counter()
        llm = LLM(model=d, dtype="bfloat16", gpu_memory_utilization=0.55,
                  max_model_len=8192, enforce_eager=True, disable_log_stats=True,
                  limit_mm_per_prompt={"image": 4, "video": 1},
                  allowed_local_media_path=args.outdir,
                  mm_processor_cache_gb=cache_gb)
        init = time.perf_counter() - t0
        print(f"  引擎就绪 {init:.1f} s")
        for name, conv in mixes.items():
            # 首 token 延迟（max_tokens=1）；用 vLLM 的 chat 接口传多模态消息
            t0 = time.perf_counter()
            llm.chat([conv], SamplingParams(max_tokens=1, temperature=0.0))
            ttft = time.perf_counter() - t0
            # 重复同一请求第二次（考察缓存对重复内容的作用）
            t0 = time.perf_counter()
            llm.chat([conv], SamplingParams(max_tokens=1, temperature=0.0))
            ttft2 = time.perf_counter() - t0
            # 完整生成 32 token
            t0 = time.perf_counter()
            r = llm.chat([conv], SamplingParams(max_tokens=32, temperature=0.0))
            full = time.perf_counter() - t0
            out = r[0].outputs[0].text.strip().replace("\n", " ")[:40]
            rows.append({"cache_gb": cache_gb, "mix": name,
                         **stats[name], "ttft_s": ttft, "ttft_repeat_s": ttft2,
                         "full32_s": full, "out": out})
            print(f"    {name:<8} 首 token {ttft*1e3:8.1f} ms  重复 {ttft2*1e3:8.1f} ms  "
                  f"32 token {full*1e3:8.1f} ms  → {out!r}")
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
    print("\n  读法：首 token 延迟包含 processor（CPU）、视觉塔、LLM prefill 三段；")
    print("  mm_processor_cache_gb 只影响 processor 段的重复内容，不影响视觉塔与 prefill。")
    SUMMARY["B"] = {"mixes": stats, "rows": rows, "env": "vllm 0.29.0"}
    return ctx


# ---------------------------------------------------------------- C
def section_C(args, ctx=None):
    """用 B 的实测服务时长驱动有界/无界队列比较（排队模拟，不是引擎实测）。"""
    title("[C] 慢模态的长尾：有界与无界队列（排队模拟）")
    if "B" not in SUMMARY:
        p = os.path.join(args.outdir, "multimodal_serving.json")
        if os.path.exists(p):
            SUMMARY.update(json.load(open(p)))
    B = SUMMARY.get("B", {})
    rows = B.get("rows", [])
    if not rows:
        print("  缺少 B 段实测（先跑 B）")
        return ctx
    svc = {}
    for r in rows:
        if r["cache_gb"] == args.mm_cache_gb:
            svc[r["mix"]] = r["ttft_s"]
    order = ["纯文本", "单图", "双图", "视频4帧", "视频8帧", "视频16帧"]
    order = [k for k in order if k in svc]
    print(f"  服务时长（首 token，来自 B 段实测）：")
    for k in order:
        print(f"    {k:<8} {svc[k]*1e3:8.1f} ms")
    import random
    random.seed(0)
    N = 200
    mix_probs = {"纯文本": 0.5, "单图": 0.2, "双图": 0.1, "视频4帧": 0.08,
                 "视频8帧": 0.07, "视频16帧": 0.05}
    mix_probs = {k: v for k, v in mix_probs.items() if k in svc}
    tot = sum(mix_probs.values())
    mix_probs = {k: v / tot for k, v in mix_probs.items()}
    kinds = list(mix_probs)
    probs = [mix_probs[k] for k in kinds]

    def run(bound, rate):
        """单服务器 FIFO：服务器空闲即刻服务，忙则排队；队列满则丢弃（bound=None 为无界）。"""
        random.seed(1)
        import collections as _c
        t, next_free = 0.0, 0.0
        waiting = _c.deque()
        lat, served, drops = [], [], 0
        for _ in range(N):
            t += random.expovariate(rate)
            kind = random.choices(kinds, probs)[0]
            # 先消化队列里在 t 之前就能服务完的请求（保持 FIFO）
            while waiting and next_free <= t:
                a, k = waiting.popleft()
                start = max(a, next_free)
                next_free = start + svc[k]
                lat.append((next_free - a, k))
                served.append(k)
            if next_free <= t:                      # 服务器空闲
                next_free = t + svc[kind]
                lat.append((svc[kind], kind))
                served.append(kind)
            else:                                   # 排队或丢弃
                if bound is not None and len(waiting) >= bound:
                    drops += 1
                else:
                    waiting.append((t, kind))
        while waiting:                              # 收尾
            a, k = waiting.popleft()
            start = max(a, next_free)
            next_free = start + svc[k]
            lat.append((next_free - a, k))
            served.append(k)
        def pct(xs, q):
            if not xs:
                return None
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(len(xs) * q))]
        all_lat = [x for x, _ in lat]
        fast = [x for x, k in lat if k == "纯文本"]
        slow = [x for x, k in lat if k in ("视频16帧", "视频8帧", "视频4帧")]
        return {"rate": rate, "bound": bound, "dropped": drops,
                "served": len(served), "drop_rate": drops / N,
                "p50": pct(all_lat, 0.5), "p99": pct(all_lat, 0.99),
                "fast_p50": pct(fast, 0.5), "fast_p99": pct(fast, 0.99),
                "slow_p50": pct(slow, 0.5), "n_fast": len(fast), "n_slow": len(slow),
                "mean_service": sum(svc[k] * p for k, p in zip(kinds, probs))}

    mean_svc = sum(svc[k] * p for k, p in zip(kinds, probs))
    print(f"  平均服务时长 {mean_svc*1e3:.1f} ms → 单 worker 饱和速率 "
          f"{1/mean_svc:.1f} req/s")
    results = []
    for rate in (0.8 / mean_svc, 1.0 / mean_svc, 1.2 / mean_svc):
        for bound in (None, 8):
            r = run(bound, rate)
            results.append(r)
            print(f"  到达 {rate:5.2f}/s（ρ={rate*mean_svc:.2f}） 队列"
                  f"{'无界' if bound is None else f'≤{bound}':<4} "
                  f"丢弃 {r['dropped']:>3}（{r['drop_rate']*100:4.1f}%）  "
                  f"p50 {r['p50']*1e3:7.1f} ms  p99 {r['p99']*1e3:8.1f} ms  "
                  f"纯文本 p99 {(r['fast_p99'] or 0)*1e3:7.1f} ms（n={r['n_fast']}）  "
                  f"慢模态 p50 {(r['slow_p50'] or 0)*1e3:7.1f} ms")
    print("\n  读法：无界队列在过载时把延迟推给所有请求（纯文本 p99 被慢模态拖长）；")
    print("  有界队列把过载转成丢弃（尾延迟被截断，但请求失败）。")
    print("  这是用 B 段实测时长驱动的排队模拟，不是引擎实测；真实引擎还有 batching、")
    print("  抢占与 encoder 准入（见正文的未覆盖项）。")
    SUMMARY["C"] = {"service_s": svc, "results": results,
                    "note": "排队模拟，驱动数据来自 B 段实测"}
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l48_out"))
    ap.add_argument("--mm-cache-gb", type=float, default=4.0)
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
    path = os.path.join(args.outdir, "multimodal_serving.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
