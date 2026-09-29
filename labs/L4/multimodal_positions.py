#!/usr/bin/env python3
"""L4.7 —— 多模态序列、位置与 mask。

[A] 手工构造「文本—图1—文本—图2」与 8 帧视频：打印模板、占位符、grid、
    embedding 替换位置、position_ids 与 mask；用独立实现与官方 get_rope_index 逐元素对拍
[B] 变体与失败输入：图像顺序、padding、视频 fps、cache_position 增量解码；
    占位符数量错误、丢帧、错误 RoPE section 各有可定位的反例
[C] 时间定位任务上的受控干预：改文本时间戳 vs 打乱 temporal 位置

用法：
    python labs/L4/multimodal_positions.py --outdir out/4.7/run A B
    python labs/L4/multimodal_positions.py --outdir out/4.7/run C
"""

import argparse
import glob
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


# ---------------------------------------------------------------- 独立实现
def my_vision_positions(grid_thw, merge=2, include_temporal=True):
    """独立实现：每个**合并后的视觉 token** 一个 (t,h,w)。

    注意不是每个 patch 一个位置：merge m×m 把 m² 个 patch 合成一个 LLM token，
    所以位置数组的长度是 (h/m)·(w/m)·t，与 <|image_pad|> 的数量相等。
    """
    out = []
    for t, h, w in grid_thw.tolist():
        gh, gw = h // merge, w // merge
        hb = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
        wb = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
        if include_temporal:
            tb = torch.arange(t).repeat_interleave(gh * gw)
            out.append(torch.stack([tb, hb.repeat(t), wb.repeat(t)], dim=-1))
        else:
            out.append(torch.stack([hb, wb], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def my_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None,
                  video_grid_thw=None, attention_mask=None, merge=2,
                  swap_hw=False, shuffle_temporal=False):
    """独立实现的 get_rope_index：文本段 arange+current_pos，视觉段用网格位置+current_pos。"""
    import itertools
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    img_it = iter(image_grid_thw) if image_grid_thw is not None else None
    vid_it = iter(video_grid_thw) if video_grid_thw is not None else None
    pos = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long)
    deltas = []
    for b in range(input_ids.shape[0]):
        tt = mm_token_type_ids[b]
        ids = input_ids[b]
        if attention_mask is not None:
            tt = tt[attention_mask[b].bool()]
            ids = ids[attention_mask[b].bool()]
        groups = [(k, len(list(g))) for k, g in itertools.groupby(tt.tolist())]
        cur, parts = 0, []
        for kind, length in groups:
            if kind == 0:
                parts.append(torch.arange(length).view(1, -1).expand(3, -1) + cur)
                cur += length
            else:
                grid = next(img_it if kind == 1 else vid_it)
                vp = my_vision_positions(grid.view(1, 3), merge)          # [N,3]
                vp = vp.t().contiguous()                                   # [3,N]
                if swap_hw:
                    vp = vp[[0, 2, 1]]
                if shuffle_temporal:
                    vp = vp.clone()
                    vp[0] = vp[0].flip(0)
                parts.append(vp + cur)
                cur += max(int(grid[1]), int(grid[2])) // merge
        llm = torch.cat(parts, dim=1)
        if attention_mask is not None:
            pos[:, b, attention_mask[b].bool()] = llm
        else:
            pos[:, b] = llm
        deltas.append(llm.max() + 1 - len(ids))
    return pos, torch.tensor(deltas).unsqueeze(1)


# ---------------------------------------------------------------- 素材
def make_images():
    from PIL import Image, ImageDraw
    a = Image.new("RGB", (320, 240), (20, 30, 60))
    ImageDraw.Draw(a).text((30, 100), "IMG-A", fill=(255, 255, 255))
    b = Image.new("RGB", (640, 480), (60, 20, 20))
    ImageDraw.Draw(b).text((30, 200), "IMG-B", fill=(255, 255, 255))
    return a, b


def make_video(n=8, w=320, h=240, fps=4, marker=4):
    from PIL import Image, ImageDraw
    frames = []
    for i in range(n):
        im = Image.new("RGB", (w, h), (12, 16, 24))
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"t={i/fps:.2f}s", fill=(255, 255, 255))
        if i == marker:
            d.rectangle([w // 2 - 20, h // 2 - 20, w // 2 + 20, h // 2 + 20],
                        fill=(60, 220, 120))
        frames.append(im)
    return frames, marker


def show_runs(tt):
    """把 mm_token_type_ids 压成 (类型, 长度) 游程。"""
    import itertools
    names = {0: "text", 1: "image", 2: "video"}
    return [(names[k], len(list(g))) for k, g in itertools.groupby(tt.tolist())]


# ---------------------------------------------------------------- A
def section_A(args, ctx=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    title("[A] 模板、占位符、grid、position_ids 与独立实现对拍")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    img_a, img_b = make_images()
    conv = [{"role": "user", "content": [
        {"type": "text", "text": "第一张："},
        {"type": "image", "image": img_a},
        {"type": "text", "text": "第二张："},
        {"type": "image", "image": img_b},
        {"type": "text", "text": "它们分别写了什么？"}]}]
    inputs = proc.apply_chat_template(conv, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
    ids = inputs["input_ids"]
    tt = inputs["mm_token_type_ids"] if "mm_token_type_ids" in inputs else None
    print(f"  序列长度 {ids.shape[1]}；含 mm_token_type_ids = {tt is not None}")
    if tt is not None:
        runs = show_runs(tt[0])
        print(f"  分段游程：{runs}")
    thw = inputs["image_grid_thw"]
    print(f"  image_grid_thw {thw.tolist()} → 视觉 token "
          f"{int(thw.prod(dim=1).sum().item()/4)}")
    pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_pad = int((ids[0] == pad_id).sum())
    print(f"  <|image_pad|> 出现 {n_pad} 次（应与视觉 token 数相等："
          f"{n_pad == int(thw.prod(dim=1).sum().item()/4)}）")
    dec = proc.tokenizer.decode(ids[0], skip_special_tokens=False)
    print(f"  模板片段（前 220 字符）：{dec[:220]}")

    if ctx is None:
        ctx = AutoModelForImageTextToText.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx
    with torch.no_grad():
        official, deltas = model.model.get_rope_index(
            ids, tt, image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"))
    mine, d2 = my_rope_index(ids, tt, inputs.get("image_grid_thw"),
                             inputs.get("video_grid_thw"),
                             inputs.get("attention_mask"))
    eq = torch.equal(official, mine)
    print(f"  官方 position_ids {tuple(official.shape)}，deltas {deltas.tolist()}")
    print(f"  独立实现逐元素相同：{eq}（最大差 "
          f"{(official-mine).abs().max().item() if not eq else 0}）")
    n = ids.shape[1]
    print(f"  {'位置':>5} {'token':>7} {'类型':>6} {'t':>4} {'h':>4} {'w':>4}")
    for i in list(range(6)) + list(range(n - 6, n)):
        t = int(tt[0, i]) if tt is not None else -1
        print(f"  {i:>5} {int(ids[0, i]):>7} {t:>6} "
              f"{int(official[0,0,i]):>4} {int(official[1,0,i]):>4} {int(official[2,0,i]):>4}")
    SUMMARY["A"] = {"seq_len": int(n), "runs": show_runs(tt[0]) if tt is not None else None,
                    "grid_thw": thw.tolist(), "image_pad": n_pad,
                    "official_equals_mine": bool(eq),
                    "deltas": deltas.tolist(),
                    "position_ids": official[0].tolist(),
                    "mm_token_type_ids": tt[0].tolist() if tt is not None else None}
    return ctx


# ---------------------------------------------------------------- B
def section_B(args, ctx=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    title("[B] 变体、增量解码与失败输入")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    img_a, img_b = make_images()
    if ctx is None:
        ctx = AutoModelForImageTextToText.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx

    def build(conv, **kw):
        ins = proc.apply_chat_template(conv, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt", **kw)
        return ins

    def pos(ins, **kw):
        with torch.no_grad():
            return model.model.get_rope_index(
                ins["input_ids"], ins["mm_token_type_ids"],
                image_grid_thw=ins.get("image_grid_thw"),
                video_grid_thw=ins.get("video_grid_thw"),
                attention_mask=ins.get("attention_mask"), **kw)

    sub("B1 图像顺序与位置")
    convs = {}
    for tag, order in (("A,B", [img_a, img_b]), ("B,A", [img_b, img_a])):
        convs[tag] = [{"role": "user", "content": [
            {"type": "text", "text": "看图："},
            {"type": "image", "image": order[0]},
            {"type": "image", "image": order[1]}]}]
    outs = {}
    for tag, c in convs.items():
        ins = build(c)
        p, dl = pos(ins)
        span = []
        tt = ins["mm_token_type_ids"][0]
        i = 0
        while i < len(tt):
            if tt[i] != 0:
                j = i
                while j < len(tt) and tt[j] == tt[i]:
                    j += 1
                span.append((i, j, int(p[1, 0, i].item()), int(p[2, 0, i].item()),
                             int(p[2, 0, j - 1].item())))
                i = j
            else:
                i += 1
        outs[tag] = {"grid": ins["image_grid_thw"].tolist(), "spans": span,
                     "deltas": dl.tolist()}
        print(f"  {tag}: grid {ins['image_grid_thw'].tolist()}  "
              f"视觉段 (起,止,h0,w0,w末) {span}")
    print("  顺序改变会改变哪张图落在哪个位置区间，但两段的 h/w 取值域只由各自 grid 决定。")

    sub("B2 padding 与 attention_mask")
    for side in ("right", "left"):
        proc.tokenizer.padding_side = side
        conv = [[{"role": "user", "content": [{"type": "image", "image": img_a},
                                              {"type": "text", "text": "hi"}]}],
                [{"role": "user", "content": [{"type": "text",
                                               "text": "a much longer question here"}]}]]
        ins = build(conv, padding=True)
        p, dl = pos(ins)
        am = ins["attention_mask"]
        print(f"  padding={side:<5} 形状 {tuple(ins['input_ids'].shape)} "
              f"有效长度 {am.sum(1).tolist()} deltas {[int(x) for x in dl.reshape(-1)]}")
        pad_pos = p[0, 0, am[0] == 0]
        print(f"    被 mask 的位置上 position_ids 取值："
              f"{pad_pos[:4].tolist() if pad_pos.numel() else '（无）'}"
              f" → 必须在 attention 里被屏蔽，取值本身不参与计算")
    proc.tokenizer.padding_side = "right"

    sub("B3 视频：fps 与时间戳")
    frames, marker = make_video()
    for fps in (2, 4):
        conv = [{"role": "user", "content": [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": "标记出现在第几秒？"}]}]
        ins = build(conv)
        p, dl = pos(ins)
        grid = ins["video_grid_thw"].tolist()
        tt = ins["mm_token_type_ids"][0]
        runs = show_runs(tt)
        print(f"  fps={fps}: video_grid_thw {grid} 游程 {runs} "
              f"deltas {[int(x) for x in dl.reshape(-1)]}")
        ids = ins["input_ids"][0]
        ts_tokens = [int(x) for x in ids if 150000 < int(x) < 160000][:6]
        print(f"    文本里出现的时间戳 token（前 6 个 id）：{ts_tokens}")

    sub("B4 增量解码的 cache_position")
    conv = [{"role": "user", "content": [{"type": "image", "image": img_a},
                                         {"type": "text", "text": "描述这张图"}]}]
    ins = build(conv)
    p, dl = pos(ins)
    L = ins["input_ids"].shape[1]
    print(f"  prefill 长度 {L}，position_ids 末位列 "
          f"{[int(p[i,0,-1]) for i in range(3)]}，deltas {[int(x) for x in dl.reshape(-1)]}")
    with torch.no_grad():
        out = model(**ins.to("cuda"), use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    past = out.past_key_values
    with torch.no_grad():
        out2 = model(nxt, past_key_values=past, use_cache=True,
                     cache_position=torch.tensor([L], device="cuda"))
    print(f"  解码一步后 cache 长度 {out2.past_key_values.get_seq_length()}；"
          f"期望 {L + 1}（cache_position 必须接着 prefill 的末尾，而不是从 0 开始）")

    sub("B5 三类失败输入")
    fails = {}

    def case(tag, fn):
        try:
            r = fn()
            fails[tag] = {"result": r}
            print(f"  {tag:<28} 未报错 → {r}")
        except Exception as e:
            fails[tag] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
            print(f"  {tag:<28} {type(e).__name__}: {str(e)[:110]}")

    def p1():
        conv = [{"role": "user", "content": [{"type": "text", "text": "没有图"}]}]
        ins = build(conv)
        with torch.no_grad():
            model(**ins.to("cuda"))
        return "文本请求正常"

    def p2():
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "看图：<|image_pad|>"},
            {"type": "image", "image": img_a}]}]
        ins = build(conv)
        with torch.no_grad():
            model(**ins.to("cuda"))
        return "手写占位符未报错"

    def p3():
        ins = build([{"role": "user", "content": [{"type": "image", "image": img_a},
                                                  {"type": "text", "text": "hi"}]}])
        ins["image_grid_thw"] = ins["image_grid_thw"] + 0
        ins["image_grid_thw"][:, 0] = 4          # 谎报 grid_t
        with torch.no_grad():
            model(**ins.to("cuda"))
        return "谎报 grid_t 未报错"

    def p4():
        ins = build([{"role": "user", "content": [{"type": "image", "image": img_a},
                                                  {"type": "text", "text": "hi"}]}])
        p, _ = pos(ins)
        bad = p.clone()
        bad[1], bad[2] = p[2].clone(), p[1].clone()      # 交换 h/w 两段（错误 RoPE section）
        with torch.no_grad():
            model(**ins.to("cuda"), position_ids=bad.to("cuda"))
        return "交换 h/w 后未报错（数值会变，需靠对拍发现）"

    for tag, fn in (("纯文本请求", p1), ("手写 <|image_pad|>", p2),
                    ("谎报 image_grid_thw", p3), ("交换 h/w 位置段", p4)):
        case(tag, fn)
    SUMMARY["B"] = {"order": outs, "video_fps": [2, 4],
                    "prefill_len": int(L), "decode_cache_len": int(out2.past_key_values.get_seq_length()),
                    "failures": fails}
    return ctx


# ---------------------------------------------------------------- C
def section_C(args, ctx=None):
    """时间定位：改进后的任务设计 + 三条受控条件（内容不变只改一条通道）。

    任务：32 帧 / 每帧角落写着自己的时间戳 / 第 20 帧画绿色方块（=5.0 s）。
    条件一（基线）：帧与 fps 都按 4 fps 原样。
    条件二（只改文本时间戳）：同样的 32 帧，但按 8 fps 解释——帧完全不变，
            写进文本的时间戳减半（5.0 → 2.5 s）。
    条件三（只打乱 temporal 位置）：帧与文本都不变，只把视频段的 t 分量反转。
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    title("[C] 时间定位：内容不变、只改一条通道")
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    if ctx is None:
        ctx = AutoModelForImageTextToText.from_pretrained(
            d, dtype=torch.bfloat16, device_map="cuda").eval()
    model = ctx
    task = getattr(args, "task", "timestamp")
    n, marker, fps = 32, 20, 4
    if task == "conflict":
        # 决定性问题：帧序与帧内时间戳互相矛盾时，模型信哪一个？
        from PIL import Image, ImageDraw
        base_frames = []
        for i in range(n):
            im = Image.new("RGB", (320, 240), (12, 16, 24))
            d_ = ImageDraw.Draw(im)
            d_.text((8, 8), f"t={i/fps:.2f}s", fill=(255, 255, 255))
            if i == 4:
                d_.ellipse([60, 100, 120, 160], fill=(220, 60, 60))
            if i == 28:
                d_.ellipse([200, 100, 260, 160], fill=(60, 120, 220))
            base_frames.append(im)
        swapped = list(base_frames)
        swapped[4], swapped[28] = swapped[28], swapped[4]   # 帧序交换：蓝块现在排在前面
        truth = "红色"
        q = ("红色方块和蓝色方块哪个先出现（按画面里的时间戳判断）？"
             "只回答「红色」或「蓝色」。")
        print(f"  冲突设计：帧内写着各自的时间戳；交换第 4 与第 28 帧后，"
              f"蓝块（t=7.00）排在红块（t=1.00）之前")
        rows = {}
        for tag, frames_used, fps_meta in (("基线（原帧序）", base_frames, fps),
                                           ("交换帧序（蓝先出现）", swapped, fps),
                                           ("交换帧序+时间戳减半", swapped, 8)):
            conv = [{"role": "user", "content": [
                {"type": "video", "video": frames_used, "fps": fps_meta},
                {"type": "text", "text": q}]}]
            ins = proc.apply_chat_template(conv, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**ins, max_new_tokens=8, do_sample=False)
            txt = proc.batch_decode(gen[:, ins["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
            rows[tag] = {"answer": txt[:30], "red": "红色" in txt, "blue": "蓝色" in txt}
            print(f"  {tag}：{txt[:30]!r}")
        verdict = ("交换帧序后若仍答红 → 模型信帧内时间戳；若改答蓝 → 信序列顺序"
                   if rows["基线（原帧序）"]["red"] else
                   "基线未命中，无法解释")
        print(f"  判定：{verdict}")
        SUMMARY["C"] = {"task": "conflict", "truth": truth, "question": q,
                        "rows": rows, "verdict": verdict,
                        "baseline_ok": rows["基线（原帧序）"]["red"]}
        return ctx
    if task == "order":
        # 事件先后：红色在第 4 帧、蓝色在第 28 帧，帧内不写任何文字
        from PIL import Image, ImageDraw
        frames = []
        for i in range(n):
            im = Image.new("RGB", (320, 240), (12, 16, 24))
            d_ = ImageDraw.Draw(im)
            if i == 4:
                d_.ellipse([60, 100, 120, 160], fill=(220, 60, 60))
            if i == 28:
                d_.ellipse([200, 100, 260, 160], fill=(60, 120, 220))
            frames.append(im)
        truth = "红色"
        q = "红色方块和蓝色方块哪个先出现？只回答「红色」或「蓝色」。"
        print(f"  {n} 帧 {fps} fps：红色在第 4 帧（1.0 s）、蓝色在第 28 帧（7.0 s），帧内无文字")
        rows = {}
        for tag, fps_meta, mode in (("基线(fps=4)", fps, "normal"),
                                    ("只改文本时间戳(fps=8)", 8, "normal"),
                                    ("只反转 temporal 位置", fps, "flip_temporal")):
            conv = [{"role": "user", "content": [
                {"type": "video", "video": frames, "fps": fps_meta},
                {"type": "text", "text": q}]}]
            ins = proc.apply_chat_template(conv, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda")
            kw = {}
            if mode != "normal":
                with torch.no_grad():
                    pos, _ = model.model.get_rope_index(
                        ins["input_ids"], ins["mm_token_type_ids"],
                        image_grid_thw=ins.get("image_grid_thw"),
                        video_grid_thw=ins.get("video_grid_thw"),
                        attention_mask=ins.get("attention_mask"))
                bad = pos.clone()
                m = (ins["mm_token_type_ids"][0] == 2)
                bad[0, 0, m] = pos[0, 0, m].flip(0)
                kw["position_ids"] = bad
            with torch.no_grad():
                gen = model.generate(**ins, max_new_tokens=8, do_sample=False, **kw)
            txt = proc.batch_decode(gen[:, ins["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
            rows[tag] = {"answer": txt[:30], "hit": truth in txt}
            print(f"  {tag}：{txt[:30]!r}  命中(含「{truth}」)={truth in txt}")
        SUMMARY["C"] = {"task": "order", "truth": truth, "question": q, "rows": rows,
                        "baseline_ok": rows["基线(fps=4)"]["hit"],
                        "verdict": ("基线命中：可比较另两条是否改变答案（预期 fps 干预不变、位置反转会翻转）"
                                    if rows["基线(fps=4)"]["hit"] else
                                    "基线未命中：该任务设计下模型无法判断先后，干预不可解释")}
        print(f"  判定：{SUMMARY['C']['verdict']}")
        return ctx
    frame_text = args.frame_text
    if frame_text == "on":
        frames, _ = make_video(n=n, fps=fps, marker=marker)
    else:
        # 去掉帧内文字：时间信息只存在于元数据时间戳里（消除视觉/元数据冗余）
        from PIL import Image, ImageDraw
        frames = []
        for i in range(n):
            im = Image.new("RGB", (320, 240), (12, 16, 24))
            d_ = ImageDraw.Draw(im)
            if i == marker:
                d_.rectangle([140, 100, 180, 140], fill=(60, 220, 120))
            frames.append(im)
    truth = marker / fps
    print(f"  {n} 帧，标称 {fps} fps，绿色方块在第 {marker} 帧 = {truth:.2f} s；"
          f"帧内文字 = {frame_text}"
          f"（{'每帧写着 t=…s，视觉与元数据冗余' if frame_text == 'on' else '帧内无文字，时间只能来自元数据时间戳'}）")
    q = "绿色方块出现在第几秒？只回答一个数字（例如 5.0）。"

    def ask(fps_meta, position_mode="normal"):
        conv = [{"role": "user", "content": [
            {"type": "video", "video": frames, "fps": fps_meta},
            {"type": "text", "text": q}]}]
        ins = proc.apply_chat_template(conv, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
        kw = {}
        if position_mode != "normal":
            with torch.no_grad():
                pos, _ = model.model.get_rope_index(
                    ins["input_ids"], ins["mm_token_type_ids"],
                    image_grid_thw=ins.get("image_grid_thw"),
                    video_grid_thw=ins.get("video_grid_thw"),
                    attention_mask=ins.get("attention_mask"))
            bad = pos.clone()
            m = (ins["mm_token_type_ids"][0] == 2)
            bad[0, 0, m] = pos[0, 0, m].flip(0)
            kw["position_ids"] = bad
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**ins, max_new_tokens=12, do_sample=False, **kw)
        dt = time.perf_counter() - t0
        txt = proc.batch_decode(gen[:, ins["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        return {"answer": txt[:40], "gen_ms": dt * 1e3,
                "hit": truth_str in txt or (fps_meta != fps and alt_str in txt)}

    truth_str = f"{truth:.1f}"
    alt_str = f"{truth * fps / 8:.1f}"        # 文本时间戳减半后的值（2.5）

    rows = {}
    a = ask(fps)
    rows["baseline(fps=4)"] = a
    print(f"  基线：{a['answer']!r}（正确答案 {truth_str}，命中 "
          f"{'✓' if truth_str in a['answer'] else '✗'}）  {a['gen_ms']:.0f} ms")
    b = ask(8)
    rows["text_timestamps_x0.5(fps=8)"] = b
    print(f"  只改文本时间戳（帧不变，fps 4→8）：{b['answer']!r}"
          f"（若跟随文本应给出 {alt_str}，若跟随画面应仍为 {truth_str}）")
    c = ask(fps, position_mode="flip_temporal")
    rows["temporal_positions_flipped"] = c
    print(f"  只打乱 temporal 位置：{c['answer']!r}")
    hits = {k: (truth_str in v["answer"]) for k, v in rows.items()}
    print(f"  命中判定（答案含 {truth_str}）：{hits}")
    baseline_ok = hits["baseline(fps=4)"]
    verdict = ("基线未命中，三个条件都无法解释 → 该任务设计仍不成立，需再改"
               if not baseline_ok else
               "基线命中；比较另两条是否改变答案，才能把因果归到具体通道")
    print(f"  判定：{verdict}")
    SUMMARY["C"] = {"frame_text": frame_text, "n_frames": n, "marker_frame": marker, "fps": fps,
                    "truth_s": truth, "question": q, "rows": rows, "hits": hits,
                    "baseline_ok": baseline_ok, "verdict": verdict}
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l47_out"))
    ap.add_argument("--task", choices=["timestamp", "order", "conflict"], default="timestamp",
                    help="timestamp=时间定位；order=事件先后（无帧内文字）")
    ap.add_argument("--frame-text", choices=["on", "off"], default="on",
                    help="帧内是否写时间戳（off = 消除视觉/元数据冗余）")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B"]
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
    path = os.path.join(args.outdir, "multimodal_positions.json")
    prev = {}
    if os.path.exists(path):
        prev = json.load(open(path))
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
