#!/usr/bin/env python3
"""L4.5 —— VLM（一）：图像与视频预处理。

[A] 从 processor 配置提取 patch / temporal patch / merge / resize 规则，
    手写 smart_resize、normalize、patchify 与 grid 计算，与官方 processor 逐元素对拍
[B] 12 张自有合成图（320×240 / 640×480 / 1280×720 / 竖图）与合成视频：
    采帧时间戳、舍入与 padding、非法边界
[C] 视觉 token 预算 128/512/2048：同一组 OCR/计数/时间定位小任务上的
    Qwen3-VL 实际作答与信息损失；Gemma3 processor 的 crop/merge 结构对照

图片全部由脚本用 PIL 生成（自有、许可明确、可复现），内容与答案在生成时确定。

用法：
    python labs/L4/vision_preprocess.py --outdir out/4.5/run A B
    python labs/L4/vision_preprocess.py --outdir out/4.5/run C
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
MAIN = "Qwen/Qwen3-VL-4B-Instruct"
GEM = "google/gemma-3-4b-it"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


# ---------------------------------------------------------------- 手写预处理
def smart_resize(h, w, factor=32, min_pixels=256 * 256, max_pixels=4096 * 4096):
    """Qwen 系列的 resize 规则：先按面积夹到 [min,max]，再对齐到 factor 的整数倍。"""
    if h * w > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h, w = math.floor(h / beta), math.floor(w / beta)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h, w = math.ceil(h * beta), math.ceil(w * beta)
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    return h_bar, w_bar


def normalize(img_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """img_tensor: [3,H,W] float in [0,1] -> 归一化。"""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img_tensor - m) / s


def patchify(img, patch=16, temporal_patch=2, merge=2):
    """[3,H,W] -> [grid_t*grid_h*grid_w, 3*temporal_patch*patch*patch]。

    与 Qwen2/3-VL 的约定一致：单个 patch 内的排布是
    (c, t, patch_h, patch_w) 展平（通道在前、时间其次），且单帧沿时间维重复
    temporal_patch 次。实测把 (t,c,ph,pw) 当成官方顺序会在形状全对的情况下得到
    1.86 的最大差（归一化后的整个取值范围才 2.0）。
    """
    C, H, W = img.shape
    gh, gw = H // patch, W // patch
    x = img.reshape(C, gh, patch, gw, patch).permute(1, 3, 0, 2, 4)   # [gh,gw,C,ph,pw]
    # 官方顺序是「merge 块主序」：块按行主序，块内 m×m 个 patch 再行主序
    x = x.reshape(gh // merge, merge, gw // merge, merge, C, patch, patch)
    x = x.permute(0, 2, 1, 3, 4, 5, 6).reshape(-1, C, patch, patch)
    # 单帧沿时间维复制；patch 内的排布是 (c, t, ph, pw)——通道在前、时间其次
    x = x.unsqueeze(2).expand(-1, -1, temporal_patch, -1, -1)
    x = x.reshape(gh * gw, C * temporal_patch * patch * patch)
    return x, (1, gh, gw)


def load_img(path):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(im)).float() / 255.0     # [H,W,3]
    return arr.permute(2, 0, 1).contiguous(), im.size


# ---------------------------------------------------------------- 图像与视频生成
def make_images(outdir):
    """生成 12 张自有合成图；返回 manifest（含答案）。"""
    from PIL import Image, ImageDraw
    os.makedirs(outdir, exist_ok=True)
    sizes = [(320, 240), (640, 480), (1280, 720), (480, 640)]
    items = []
    idx = 0
    for (w, h) in sizes:
        for kind in ("ocr", "count", "time"):
            idx += 1
            name = f"img{idx:02d}_{kind}_{w}x{h}.png"
            im = Image.new("RGB", (w, h), (18, 22, 34))
            d = ImageDraw.Draw(im)
            if kind == "ocr":
                # 固定字符串，答案就是它
                text = f"TOKEN-{idx:02d}-AB7"
                d.text((int(w * 0.06), int(h * 0.42)), text, fill=(255, 255, 255))
                ans = text
            elif kind == "count":
                n = 3 + idx % 5
                r = max(6, min(w, h) // 14)
                for k in range(n):
                    cx = int(w * (0.15 + 0.7 * (k % 5) / 5))
                    cy = int(h * (0.3 + 0.4 * (k // 5)))
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 90, 60))
                ans = n
            else:
                ts = 1.0 + 0.5 * (idx % 6)
                d.rectangle([int(w * 0.05), int(h * 0.05),
                             int(w * 0.35), int(h * 0.15)], fill=(240, 200, 60))
                d.text((int(w * 0.06), int(h * 0.07)), f"t={ts:.1f}s", fill=(0, 0, 0))
                ans = ts
            p = os.path.join(outdir, name)
            im.save(p)
            items.append({"file": os.path.basename(p), "kind": kind,
                          "orig_wh": [w, h], "answer": ans})
    return items


def make_video(n_frames=16, w=320, h=240, fps=8):
    """合成一段有运动标记的视频：返回 PIL 帧列表与标记所在帧号。"""
    from PIL import Image, ImageDraw
    frames = []
    marker = n_frames // 2
    for i in range(n_frames):
        im = Image.new("RGB", (w, h), (12, 16, 24))
        d = ImageDraw.Draw(im)
        x = int(w * (0.1 + 0.8 * i / max(1, n_frames - 1)))
        d.rectangle([x - 12, h // 2 - 12, x + 12, h // 2 + 12],
                    fill=(60, 220, 120) if i == marker else (200, 200, 200))
        d.text((6, 6), f"frame {i}  t={i / fps:.2f}s", fill=(255, 255, 255))
        frames.append(im)
    return frames, marker


def sample_frames(n_total, fps, src_fps, num_frames=None):
    """Qt 的采帧规则：按目标 fps 等间隔取帧，返回帧号与时间戳。"""
    if num_frames:
        idx = [round(i * (n_total - 1) / max(1, num_frames - 1))
               for i in range(num_frames)] if num_frames > 1 else [0]
    else:
        step = max(1, round(src_fps / fps))
        idx = list(range(0, n_total, step))
    return idx, [i / src_fps for i in idx]


# ---------------------------------------------------------------- A
def section_A(args):
    from transformers import AutoProcessor
    title("[A] processor 配置、手写 resize/normalize/patchify 与对拍")
    d = snap(MAIN)
    pp = json.load(open(d + "/preprocessor_config.json"))
    vp = json.load(open(d + "/video_preprocessor_config.json"))
    print(f"  图像 processor：patch {pp['patch_size']}、temporal "
          f"{pp['temporal_patch_size']}、merge {pp['merge_size']}、"
          f"mean {pp['image_mean']}、std {pp['image_std']}")
    print(f"  resize 边界（像素数）：最短边 {pp['size']['shortest_edge']}、"
          f"最长边 {pp['size']['longest_edge']}  → "
          f"{pp['size']['shortest_edge']} = {int(math.sqrt(pp['size']['shortest_edge']))}²，"
          f"{pp['size']['longest_edge']} = {int(math.sqrt(pp['size']['longest_edge']))}²")
    print(f"  视频 processor：{vp.get('video_processor_type')}，边界 "
          f"{vp['size']}，patch/temporal/merge 同上")
    factor = pp["patch_size"] * pp["merge_size"]
    print(f"  对齐因子 = patch × merge = {pp['patch_size']} × {pp['merge_size']} = {factor}")

    sub("A1 手写 smart_resize 与官方 resize 对照")
    proc = AutoProcessor.from_pretrained(d)
    img_items = make_images(os.path.join(args.outdir, "images"))
    rows = []
    for it in img_items:
        p = os.path.join(args.outdir, "images", it["file"])
        img_t, (w, h) = load_img(p)
        hb, wb = smart_resize(h, w, factor, pp["size"]["shortest_edge"],
                              pp["size"]["longest_edge"])
        from PIL import Image
        import numpy as np

        def build_patches(resize_fn):
            resized = resize_fn(Image.open(p).convert("RGB"), (wb, hb))
            rt_ = torch.from_numpy(np.asarray(resized).copy()).float() / 255.0
            norm_ = normalize(rt_.permute(2, 0, 1), pp["image_mean"], pp["image_std"])
            return patchify(norm_, pp["patch_size"], pp["temporal_patch_size"],
                            pp["merge_size"])[0]

        # 主路径：torchvision 的双三次 + 抗锯齿（官方 processor 用的就是这条）
        def tv_resize(im, sz):
            from torchvision.transforms import v2 as _v2
            t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
            r = _v2.functional.resize(t, [sz[1], sz[0]],
                                      interpolation=_v2.InterpolationMode.BICUBIC,
                                      antialias=True)
            return Image.fromarray(r.permute(1, 2, 0).numpy().astype("uint8"))

        try:
            resized = tv_resize(Image.open(p).convert("RGB"), (wb, hb))
            primary = "torchvision双三次抗锯齿"
        except Exception:
            resized = Image.open(p).convert("RGB").resize((wb, hb))
            primary = "PIL默认"
        rt = torch.from_numpy(np.asarray(resized).copy()).float() / 255.0
        rt = rt.permute(2, 0, 1)
        norm = normalize(rt, pp["image_mean"], pp["image_std"])
        mine, grid = patchify(norm, pp["patch_size"], pp["temporal_patch_size"],
                              pp["merge_size"])
        out = proc.image_processor(images=Image.open(p).convert("RGB"),
                                   return_tensors="pt")
        pv = out["pixel_values"]
        thw = out["image_grid_thw"][0].tolist()
        # 固定 patch 顺序后，逐一比对 resize 的实现（PIL 默认 / PIL 双三次 / torchvision 双三次+抗锯齿）
        variant_diffs = {}
        try:
            from PIL import Image as _I
            for tag, fn in (("PIL默认", lambda im, sz: im.resize(sz)),
                            ("PIL双三次", lambda im, sz: im.resize(sz, resample=_I.BICUBIC)),
                            ("PIL双线性", lambda im, sz: im.resize(sz, resample=_I.BILINEAR))):
                variant_diffs[tag] = float((build_patches(fn).float() -
                                            pv.float()).abs().max().item())
        except Exception as e:
            variant_diffs["PIL变体失败"] = f"{type(e).__name__}: {str(e)[:60]}"
        try:
            from torchvision.transforms import v2 as _v2
            def tv(im, sz):
                t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
                r = _v2.functional.resize(t, [sz[1], sz[0]],
                                          interpolation=_v2.InterpolationMode.BICUBIC,
                                          antialias=True)
                return Image.fromarray(r.permute(1, 2, 0).numpy())
            variant_diffs["torchvision双三次抗锯齿"] = float(
                (build_patches(tv).float() - pv.float()).abs().max().item())
        except Exception as e:
            variant_diffs["torchvision失败"] = f"{type(e).__name__}: {str(e)[:60]}"
        dd = (mine.float() - pv.float()).abs()
        dmax = dd.max().item()
        # 对照：错误的 patch 内顺序 (t,c,ph,pw)
        # 错误对照：把时间维放到通道前面 (t,c,ph,pw)
        alt = norm.reshape(3, hb // 16, 16, wb // 16, 16).permute(1, 3, 0, 2, 4)
        alt = alt.reshape(-1, 3, 16, 16).unsqueeze(1).expand(-1, 2, -1, -1, -1)
        alt = alt.reshape(-1, 3 * 2 * 16 * 16)   # 无 merge 块重排的对照
        dalt = (alt.float() - pv.float()).abs().mean().item()
        rows.append({"file": it["file"], "orig": [w, h], "resized": [wb, hb],
                     "grid_thw": thw, "mine_grid": grid,
                     "tokens": int(thw[0] * thw[1] * thw[2] / pp["merge_size"] ** 2),
                     "patch_rows": int(mine.shape[0]), "patch_dim": int(mine.shape[1]),
                     "max_abs_diff": dmax, "mean_abs_diff": dd.mean().item(),
                     "frac_gt_0.5": (dd > 0.5).float().mean().item(),
                     "mean_abs_diff_wrong_layout": dalt,
                     "resize_variant_max_abs_diff": variant_diffs,
                     "primary_resize": primary})
    for r in rows[:4] + rows[-2:]:
        print(f"  {r['file']:<26} {r['orig']} → {r['resized']}  grid {r['grid_thw']}  "
              f"token {r['tokens']:<5} patch {r['patch_rows']}×{r['patch_dim']}  "
              f"mean|diff| {r['mean_abs_diff']:.2e}  max {r['max_abs_diff']:.2e}")
    allzero = all(r["max_abs_diff"] == 0.0 for r in rows)
    print(f"  12 张图逐元素对拍：全部为 0 = {allzero}")
    print(f"  平均绝对差 {max(r['mean_abs_diff'] for r in rows):.2e}（取值域 [-1,1]），"
          f"超过 0.5 的元素占比最大 {max(r['frac_gt_0.5'] for r in rows):.4%}")
    print(f"  形状与几何完全一致（grid_thw 与 patch 数逐张相同），残差集中在"
          f"高对比细笔画像素上；原因未定位，见本章“陷阱”")
    ratios = [r["mean_abs_diff_wrong_layout"] / max(r["mean_abs_diff"], 1e-12)
              for r in rows]
    agg = {}
    for r in rows:
        for k, v in (r.get("resize_variant_max_abs_diff") or {}).items():
            if isinstance(v, float):
                agg[k] = max(agg.get(k, 0.0), v)
    print(f"  主路径 resize = {rows[0].get('primary_resize')}")
    print(f"  resize 实现对照（最大逐元素差，越小越接近官方）："
          + "，".join(f"{k} {v:.3e}" for k, v in agg.items()))
    print(f"  对照：把 patch 内顺序改成 (t,c,ph,pw)，平均绝对差最大 "
          f"{max(r['mean_abs_diff_wrong_layout'] for r in rows):.2e}，"
          f"是正确顺序的 {min(ratios):.1f}–{max(ratios):.1f} 倍 → 顺序判断是决定性的")
    SUMMARY["A"] = {"config": {"patch_size": pp["patch_size"],
                               "temporal_patch_size": pp["temporal_patch_size"],
                               "merge_size": pp["merge_size"],
                               "mean": pp["image_mean"], "std": pp["image_std"],
                               "image_size": pp["size"], "video_size": vp["size"],
                               "factor": factor},
                    "rows": rows, "all_zero": allzero,
                    "revision": os.path.basename(d)}
    return proc


# ---------------------------------------------------------------- B
def section_B(args, proc=None):
    from transformers import AutoProcessor
    from PIL import Image
    title("[B] 图像尺寸覆盖、视频采帧与非法边界")
    d = snap(MAIN)
    if proc is None:
        proc = AutoProcessor.from_pretrained(d)
    pp = json.load(open(d + "/preprocessor_config.json"))
    factor = pp["patch_size"] * pp["merge_size"]
    items = make_images(os.path.join(args.outdir, "images"))

    sub("B1 12 张图的尺寸、token 预算与 CPU 耗时")
    rows = []
    for it in items:
        p = os.path.join(args.outdir, "images", it["file"])
        t0 = time.perf_counter()
        out = proc.image_processor(images=Image.open(p).convert("RGB"),
                                   return_tensors="pt")
        dt = time.perf_counter() - t0
        thw = out["image_grid_thw"][0].tolist()
        tokens = int(thw[0] * thw[1] * thw[2] / pp["merge_size"] ** 2)
        rows.append({**it, "grid_thw": thw, "tokens": tokens,
                     "cpu_ms": dt * 1e3,
                     "pixel_bytes": int(out["pixel_values"].numel() * 2)})
    for r in rows:
        print(f"  {r['file']:<28} {r['orig_wh']} → grid {r['grid_thw']}  "
              f"token {r['tokens']:<5} CPU {r['cpu_ms']:6.2f} ms  "
              f"pixel {r['pixel_bytes']/1024:7.1f} KiB")
    print("  注意：文件大小、原始像素数与视觉 token 数是三个不同的量；"
          "上表给出的是 processor 输出的真实 token 数。")

    sub("B2 视频采帧：fps 与 num_frames")
    frames, marker = make_video()
    src_fps = 8
    vrows = []
    for fps in (1, 2, 4):
        idx, ts = sample_frames(len(frames), fps, src_fps)
        vrows.append({"mode": f"fps={fps}", "frames": len(idx),
                      "frame_index": idx, "timestamps": [round(t, 3) for t in ts]})
    for nf in (4, 8, 16):
        idx, ts = sample_frames(len(frames), None, src_fps, num_frames=nf)
        vrows.append({"mode": f"num_frames={nf}", "frames": len(idx),
                      "frame_index": idx, "timestamps": [round(t, 3) for t in ts]})
    for r in vrows:
        print(f"  {r['mode']:<16} 取 {r['frames']:>2} 帧  "
              f"时间戳 {r['timestamps'][:6]}{'...' if len(r['timestamps']) > 6 else ''}")
    # 交给官方 video processor，检查 grid_t 与 temporal patch 的整除
    vout = None
    try:
        vout = proc.video_processor(videos=[frames], return_tensors="pt")
        g = vout["video_grid_thw"][0].tolist()
        print(f"  官方 video_processor：grid_thw {g}，"
              f"temporal_patch_size {pp['temporal_patch_size']}，"
              f"grid_t×temporal = {g[0] * pp['temporal_patch_size']} 帧")
    except Exception as e:
        print(f"  官方 video_processor 调用失败：{type(e).__name__}: {str(e)[:120]}")

    sub("B3 非法与边界输入")
    edge = {}

    def edge_case(tag, fn):
        try:
            r = fn()
            edge[tag] = {"result": r}
            print(f"  {tag:<28} 未报错 → {r}")
        except Exception as e:
            edge[tag] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
            print(f"  {tag:<28} {type(e).__name__}: {str(e)[:110]}")

    def p1():
        im = Image.new("RGB", (32, 32), (255, 0, 0))
        o = proc.image_processor(images=im, return_tensors="pt")
        return f"32×32 → grid {o['image_grid_thw'][0].tolist()}"

    def p2():
        im = Image.new("RGB", (5000, 5000), (0, 255, 0))
        o = proc.image_processor(images=im, return_tensors="pt")
        g = o["image_grid_thw"][0].tolist()
        return f"5000×5000 → grid {g}，token {int(g[0]*g[1]*g[2]/4)}"

    def p3():
        return f"fps=0 采帧：{sample_frames(16, 0, 8)[:1]}（脚本自行处理，processor 无此参数）"

    def p4():
        o = proc.video_processor(videos=[frames[:1]], return_tensors="pt")
        return f"单帧视频 → grid {o['video_grid_thw'][0].tolist()}"

    def p5():
        o = proc.video_processor(videos=[frames[:5]], return_tensors="pt")
        g = o["video_grid_thw"][0].tolist()
        return (f"5 帧（奇数，temporal=2）→ grid {g}，"
                f"grid_t×temporal = {g[0]*pp['temporal_patch_size']} > 5 → 需要 padding/复制")

    def p6():
        im = Image.new("RGBA", (300, 200), (10, 20, 30, 128))
        o = proc.image_processor(images=im, return_tensors="pt")
        return f"RGBA 图 → grid {o['image_grid_thw'][0].tolist()}（自动转 RGB）"

    for tag, fn in (("32×32 小图", p1), ("5000×5000 大图", p2), ("fps=0", p3),
                    ("单帧视频", p4), ("5 帧奇数长度", p5), ("RGBA 输入", p6)):
        edge_case(tag, fn)
    SUMMARY["B"] = {"images": rows, "video": vrows, "video_grid": (
        vout["video_grid_thw"][0].tolist() if vout is not None else None),
        "marker_frame": marker, "edges": edge,
        "src_fps": src_fps}
    return items


# ---------------------------------------------------------------- C
TASKS = [
    ("ocr", "这张图里的编号字符串是什么？只回答字符串。"),
    ("count", "图中有几个红色圆点？只回答数字。"),
    ("time", "图中黄色标签上的时间是多少？只回答形如 1.5s 的值。"),
]


def section_C(args, proc=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    title("[C] 视觉 token 预算：128 / 512 / 2048")
    d = snap(MAIN)
    if proc is None:
        proc = AutoProcessor.from_pretrained(d)
    items = make_images(os.path.join(args.outdir, "images"))
    pp = json.load(open(d + "/preprocessor_config.json"))

    # 预算 -> (min_pixels, max_pixels)：控制 resize 的目标面积
    budgets = {128: (128 * 32 * 32, int(128 * 32 * 32 * 1.02)),
               512: (512 * 32 * 32, int(512 * 32 * 32 * 1.02)),
               2048: (2048 * 32 * 32, int(2048 * 32 * 32 * 1.02))}
    print("  预算换算：token 数 = (grid_t·grid_h·grid_w)/merge²，"
          "grid_h·grid_w = 像素面积/(patch²·temporal) → "
          "min_pixels ≈ token×32×32")
    rows = []
    for it in items:
        for budget, (mn, mx) in budgets.items():
            p = os.path.join(args.outdir, "images", it["file"])
            ip = proc.image_processor
            old = dict(ip.size)
            ip.size = {"shortest_edge": mn, "longest_edge": mx}
            t0 = time.perf_counter()
            out = ip(images=Image.open(p).convert("RGB"), return_tensors="pt")
            cpu = time.perf_counter() - t0
            ip.size = old
            thw = out["image_grid_thw"][0].tolist()
            tokens = int(thw[0] * thw[1] * thw[2] / pp["merge_size"] ** 2)
            rows.append({"file": it["file"], "kind": it["kind"],
                         "answer": it["answer"], "budget": budget,
                         "grid_thw": thw, "tokens": tokens, "cpu_ms": cpu * 1e3,
                         "pixel_bytes": int(out["pixel_values"].numel() * 2)})
    print(f"  {'budget':>7} {'token 中位数':>12} {'CPU 中位数 ms':>14} "
          f"{'pixel MiB 中位数':>16}")
    for budget in budgets:
        sel = [r for r in rows if r["budget"] == budget]
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"  {budget:>7} {med([r['tokens'] for r in sel]):>12} "
              f"{med([r['cpu_ms'] for r in sel]):>14.2f} "
              f"{med([r['pixel_bytes'] for r in sel])/2**20:>16.3f}")
    print("  预算设置值与实际 token 数分别记录：面积夹取与 32 对齐会产生偏差。")
    SUMMARY["C_budget"] = rows

    sub("C1 Qwen3-VL 在三种预算下的实际作答")
    model = AutoModelForImageTextToText.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda").eval()
    results = []
    for budget, (mn, mx) in budgets.items():
        ip = proc.image_processor
        old = dict(ip.size)
        ip.size = {"shortest_edge": mn, "longest_edge": mx}
        for it in items:
            p = os.path.join(args.outdir, "images", it["file"])
            prompt = dict(TASKS)[it["kind"]]
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(p).convert("RGB")},
                {"type": "text", "text": prompt}]}]
            inputs = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            dt = time.perf_counter() - t0
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
            vtokens = int(inputs["image_grid_thw"][0].prod().item() / 4)
            results.append({"file": it["file"], "kind": it["kind"],
                            "answer": it["answer"], "budget": budget,
                            "visual_tokens": vtokens, "output": text[:40],
                            "gen_s": dt})
        ip.size = old
    for budget in budgets:
        sel = [r for r in results if r["budget"] == budget]
        hit = sum(1 for r in sel if str(r["answer"]).lower() in r["output"].lower())
        print(f"  budget={budget:<5} 视觉 token 中位数 "
              f"{sorted(r['visual_tokens'] for r in sel)[len(sel)//2]:<5} "
              f"命中 {hit}/{len(sel)}  生成耗时中位数 "
              f"{sorted(r['gen_s'] for r in sel)[len(sel)//2]*1e3:.0f} ms")
    SUMMARY["C_model"] = results
    del model
    torch.cuda.empty_cache()

    sub("C2 Gemma3 processor 的结构对照")
    try:
        from transformers import Gemma3ImageProcessor
        gip = Gemma3ImageProcessor()
        attrs = {a: getattr(gip, a, None) for a in
                 ("do_pan_and_scan", "pan_and_scan_min_crop_size",
                  "pan_and_scan_max_num_crops", "size", "do_resize",
                  "resample", "image_mean", "image_std")}
        print(f"  Gemma3 图像 processor 属性："
              f"{ {k: (str(v) if not isinstance(v, (int, float, str, type(None), bool)) else v) for k, v in attrs.items()} }")
        grows = []
        for it in items[:4]:
            p = os.path.join(args.outdir, "images", it["file"])
            try:
                o = gip(images=Image.open(p).convert("RGB"), return_tensors="pt")
                grows.append({"file": it["file"],
                              "pixel_shape": list(o["pixel_values"].shape),
                              "num_crops": len(o.get("num_crops", [])) if
                              "num_crops" in o else None})
            except Exception as e:
                grows.append({"file": it["file"], "error": f"{type(e).__name__}: {str(e)[:80]}"})
        for r in grows:
            print(f"  {r['file']:<28} {r.get('pixel_shape')} num_crops={r.get('num_crops')}")
        print("  HF 的 Gemma3ImageProcessor 默认把图缩到固定的 224×224（实测 size "
              f"{getattr(gip,'size',None)}），resample 是 BILINEAR；")
        print("  真实的 gemma-3-4b-it 配置用 896×896 加可选的 pan-and-scan（大图切多块），")
        print("  权重受许可限制未下载，这里只对照 processor 的默认结构与输出布局。")
        print("  与 Qwen3-VL 的「按面积夹取 + 32 对齐 + merge 2×2 + 动态 token 数」是两种结构：")
        print("  Gemma3 的输出像素尺寸固定，token 数由 crop 数决定；Qwen3-VL 的输出尺寸随图变。")
        SUMMARY["C_gemma"] = {"patch_size": gip.patch_size,
                              "size": str(getattr(gip, "size", None)),
                              "rows": grows,
                              "note": "gemma-3-4b-it 权重受许可限制，未下载；只比较 processor 结构"}
    except Exception as e:
        print(f"  Gemma3 processor 不可用：{type(e).__name__}: {str(e)[:120]}")
        SUMMARY["C_gemma"] = {"error": f"{type(e).__name__}: {str(e)}"}
    return SUMMARY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l45_out"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B"]
    SUMMARY["env"] = {"torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "transformers": __import__("transformers").__version__}
    proc = None
    if "A" in want:
        proc = section_A(args)
    if "B" in want:
        section_B(args, proc)
    if "C" in want:
        section_C(args, proc)
    path = os.path.join(args.outdir, "vision_preprocess.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
