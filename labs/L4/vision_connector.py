#!/usr/bin/env python3
"""L4.6 —— 视觉编码器与 connector。

[A] 逐层特征清单：patch embedding、每个 block、merger、DeepStack 注入点；
    手写 patch merger（LayerNorm+MLP，不是"一个 linear"）与官方逐元素对拍
[B] 固定图片扫分辨率 × batch=1/2/4/8：processor / ViT / merger / LLM prefill
    分段计时与峰值显存，检查每阶段资源闭合
[C] DeepStack：记录 [5,11,17] 三个注入点的载荷（形状/dtype/字节），
    做"去掉 DeepStack"的分析性消融（对照原模型的任务命中与成本），
    并给出 4.8 特征缓存所需的完整载荷清单

用法：
    python labs/L4/vision_connector.py --outdir out/4.6/run A B
    python labs/L4/vision_connector.py --outdir out/4.6/run C
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
REPO = "Qwen/Qwen3-VL-4B-Instruct"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo=REPO):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def load():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    model = AutoModelForImageTextToText.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda").eval()
    return d, proc, model


# ---------------------------------------------------------------- A
def section_A(args, ctx=None):
    title("[A] 视觉编码器与 connector：逐层特征清单")
    d, proc, model = ctx or load()
    vis = model.model.visual
    print(f"  {REPO} 的 vision tower：depth {vis.config.depth}、hidden "
          f"{vis.config.hidden_size}、intermediate {vis.config.intermediate_size}、"
          f"heads {vis.config.num_heads}")
    print(f"  输出维度 {vis.config.out_hidden_size}（送入 LLM 的 hidden）、"
          f"spatial_merge {vis.config.spatial_merge_size}、"
          f"DeepStack 注入层 {vis.config.deepstack_visual_indexes}")
    print(f"  patch_embed: {type(vis.patch_embed).__name__}，"
          f"merger: {type(vis.merger).__name__}（use_postshuffle_norm="
          f"{vis.merger.use_postshuffle_norm}），"
          f"deepstack_merger_list {len(vis.deepstack_merger_list)} 个"
          f"（use_postshuffle_norm={vis.deepstack_merger_list[0].use_postshuffle_norm}）")
    print("  merger 的模块树（证明它不是单个 linear）：")
    for n, m in vis.merger.named_children():
        extra = f" {tuple(m.weight.shape)}" if hasattr(m, "weight") else ""
        print(f"    merger.{n:<12} {type(m).__name__}{extra}")

    from PIL import Image
    img = Image.new("RGB", (640, 480), (20, 24, 36))
    from PIL import ImageDraw
    ImageDraw.Draw(img).text((40, 200), "CONNECTOR", fill=(255, 255, 255))
    inputs = proc(images=[img], text=["describe"], return_tensors="pt").to("cuda")
    pv, thw = inputs["pixel_values"], inputs["image_grid_thw"]
    print(f"  输入：pixel_values {tuple(pv.shape)}，image_grid_thw "
          f"{thw[0].tolist()} → {int(thw.prod().item()/4)} 个视觉 token")

    caps = {}
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                caps[name] = t.detach()
        return fn

    handles.append(vis.patch_embed.register_forward_hook(hook("patch_embed")))
    for i in (0, 5, 11, 17, 23):
        handles.append(vis.blocks[i].register_forward_hook(hook(f"block_{i}")))
    handles.append(vis.merger.register_forward_hook(hook("merger")))
    for j, m in enumerate(vis.deepstack_merger_list):
        handles.append(m.register_forward_hook(hook(f"deepstack_merger_{j}")))
    with torch.no_grad():
        vis_out = vis(pv, grid_thw=thw)
    for h in handles:
        h.remove()
    print(f"  {'阶段':<22} {'形状':<24} {'dtype':<10} {'L2 均值':>9} {'元素字节':>10}")
    inv = []
    for k, v in caps.items():
        row = {"stage": k, "shape": list(v.shape),
               "dtype": str(v.dtype).replace("torch.", ""),
               "l2_mean": v.float().norm(dim=-1).mean().item(),
               "bytes": int(v.numel() * v.element_size())}
        inv.append(row)
        print(f"  {k:<22} {str(tuple(v.shape)):<24} {row['dtype']:<10} "
              f"{row['l2_mean']:>9.3f} {row['bytes']/2**20:>10.3f} MiB")
    ds = [f for f in vis_out.deepstack_features]
    print(f"  tower 直接返回：last_hidden_state {tuple(vis_out.last_hidden_state.shape)}，"
          f"deepstack_features {len(ds)} × {tuple(ds[0].shape)}")

    sub("A2 手写 patch merger 与官方对拍")
    mg = vis.merger
    hs = vis.config.hidden_size * vis.config.spatial_merge_size ** 2
    x = (torch.randn(1200, vis.config.hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        y_off = mg(x)

    def my_merger_plain(t, m):
        """use_postshuffle_norm=False：先在 1024 维上 LayerNorm，再拼 2×2。"""
        return m.linear_fc2(m.act_fn(m.linear_fc1(m.norm(t).reshape(-1, 4 * t.shape[-1]))))

    def my_merger_post(t, m):
        """use_postshuffle_norm=True：先拼 2×2，再在 4096 维上 LayerNorm。"""
        z = t.reshape(-1, 4 * t.shape[-1])
        return m.linear_fc2(m.act_fn(m.linear_fc1(m.norm(z))))

    with torch.no_grad():
        y_my = my_merger_plain(x, mg)
    d_my = (y_off.float() - y_my.float()).abs().max().item()
    print(f"  主 merger（use_postshuffle_norm=False）：输入 [1200,1024] → 输出 "
          f"{tuple(y_off.shape)}")
    print(f"    手写 norm→reshape→fc1→GELU→fc2：max|diff| {d_my:.3e}")
    try:
        with torch.no_grad():
            my_merger_post(x, mg)
        print("    用错误顺序（先拼再 norm）未报错")
    except Exception as e:
        print(f"    用错误顺序（先拼再 norm）：{type(e).__name__}: {str(e)[:80]}")
        print("    → 两类 merger 的 norm 维度不同（1024 vs 4096），顺序写错会在形状上直接报错")
    dm = vis.deepstack_merger_list[0]
    with torch.no_grad():
        z_off = dm(x)
        z_my = my_merger_post(x, dm)
    print(f"  DeepStack merger（use_postshuffle_norm=True）："
          f"手写 max|diff| {(z_off.float()-z_my.float()).abs().max().item():.3e}")
    print(f"  若把 connector 简化成单个 linear（{hs}→{vis.config.out_hidden_size}）：")
    lin = torch.nn.Linear(hs, vis.config.out_hidden_size).cuda().to(torch.bfloat16)
    with torch.no_grad():
        y_lin = lin(x.reshape(-1, hs))
    print(f"    与官方输出的相对差 "
          f"{((y_off.float()-y_lin.float()).norm()/y_off.float().norm()).item():.3f}"
          f"（量级 = 1 说明完全不相关）")
    print(f"  tower 的返回字段：{type(vis_out).__name__} → "
          f"{[k for k in vis_out.keys() if vis_out[k] is not None]}")
    SUMMARY["A"] = {"inventory": inv,
                    "merger_plain_max_abs_diff": d_my,
                    "deepstack_merger_max_abs_diff": (z_off.float()-z_my.float()).abs().max().item(),
                    "tower_output_fields": [k for k in vis_out.keys() if vis_out[k] is not None],
                    "config": {"depth": vis.config.depth,
                               "hidden": vis.config.hidden_size,
                               "out_hidden": vis.config.out_hidden_size,
                               "merge": vis.config.spatial_merge_size,
                               "deepstack_indexes": list(vis.config.deepstack_visual_indexes)},
                    "mini_merger_max_abs_diff": (y_off.float()-y_my.float()).abs().max().item(),
                    "single_linear_rel": ((y_off.float()-y_lin.float()).norm()/y_off.float().norm()).item(),
                    "visual_tokens": int(thw.prod().item()/4)}
    return ctx if ctx else (d, proc, model)


# ---------------------------------------------------------------- B
def section_B(args, ctx=None):
    from PIL import Image, ImageDraw
    title("[B] processor / ViT / merger / LLM prefill 的分段成本")
    d, proc, model = ctx or load()
    vis = model.model.visual
    base = Image.new("RGB", (1280, 720), (16, 20, 30))
    ImageDraw.Draw(base).text((60, 320), "STAGE TIMING", fill=(255, 255, 255))
    resolutions = {"small_320x240": (320, 240), "mid_640x480": (640, 480),
                   "large_1280x720": (1280, 720)}
    # 预热一次，避免把首次调用的 JIT/缓存冷启动算进第一格
    warm = base.resize((640, 480))
    wi = proc.apply_chat_template(
        [[{"role": "user", "content": [{"type": "image", "image": warm},
                                       {"type": "text", "text": "warmup"}]}]],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**wi)
    torch.cuda.synchronize()
    rows = []
    for name, (w, h) in resolutions.items():
        img = base.resize((w, h))
        for B in (1, 2, 4, 8):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            iput = proc.image_processor(images=[img] * B, return_tensors="pt")
            t_proc = time.perf_counter() - t0
            convs = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "describe"}]}] for _ in range(B)]
            inputs = proc.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True).to("cuda")
            pv, thw = inputs["pixel_values"], inputs["image_grid_thw"]
            assert iput["pixel_values"].shape[0] == pv.shape[0], "图像块数不一致"
            # ViT（含 merger）
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                vout = vis(pv, grid_thw=thw)
            torch.cuda.synchronize()
            t_vit = time.perf_counter() - t0
            # 单独计时 merger
            t0 = time.perf_counter()
            with torch.no_grad():
                merged = vis.merger(vout.last_hidden_state)
            torch.cuda.synchronize()
            t_merger = time.perf_counter() - t0
            # 完整请求
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()
            t_full = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20
            vtokens = int(thw.prod(dim=1).sum().item() / 4)   # 逐图求和
            rows.append({"res": name, "wh": [w, h], "batch": B,
                         "visual_tokens_total": vtokens,
                         "proc_s": t_proc, "vit_s": t_vit, "merger_s": t_merger,
                         "full_s": t_full, "llm_s": t_full - t_vit,
                         "peak_mib": peak})
            print(f"  {name:<15} B={B:<2} token {vtokens:<6} "
                  f"processor {t_proc*1e3:7.2f} ms  ViT {t_vit*1e3:7.2f}  "
                  f"merger {t_merger*1e3:6.2f}  LLM {max(t_full-t_vit,0)*1e3:8.2f}  "
                  f"完整 {t_full*1e3:8.2f} ms  峰值 {peak:7.0f} MiB")
    # 每阶段最优 batch
    print("\n  各阶段随 batch 的变化（同一分辨率 640×480）：")
    for stage in ("proc_s", "vit_s", "merger_s", "full_s"):
        sel = {r["batch"]: r[stage] for r in rows if r["res"] == "mid_640x480"}
        base1 = sel[1]
        print(f"    {stage:<9} B=1 {base1*1e3:7.2f} ms → B=8 {sel[8]*1e3:7.2f} ms "
              f"（{base1/sel[8]:.2f}× 加速，理论上限 8×）")
    SUMMARY["B"] = rows
    return ctx if ctx else (d, proc, model)


# ---------------------------------------------------------------- C
def section_C(args, ctx=None):
    from PIL import Image, ImageDraw
    title("[C] DeepStack 注入点与消融")
    d, proc, model = ctx or load()
    vis = model.model.visual
    idxs = list(vis.config.deepstack_visual_indexes)
    layers = model.model.language_model.layers
    print(f"  DeepStack 注入层（vision 侧）{idxs} → LLM 侧前 "
          f"{len(idxs)} 层（_deepstack_process 按 index 取用）")

    img = Image.new("RGB", (640, 480), (18, 22, 34))
    ImageDraw.Draw(img).text((40, 200), "DEEPSTACK-42", fill=(255, 255, 255))
    inputs = proc(images=[img], text=["图中写了什么？"], return_tensors="pt").to("cuda")
    pv, thw = inputs["pixel_values"], inputs["image_grid_thw"]
    with torch.no_grad():
        vout = vis(pv, grid_thw=thw)
    payload = [{"name": f"deepstack_{j}_layer{idx}", "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "bytes": int(t.numel() * t.element_size())}
               for j, (idx, t) in enumerate(zip(idxs, vout.deepstack_features))]
    payload.append({"name": "main_merger", "shape": list(vout.last_hidden_state.shape),
                    "dtype": str(vout.last_hidden_state.dtype).replace("torch.", ""),
                    "bytes": int(vout.last_hidden_state.numel() *
                                 vout.last_hidden_state.element_size())})
    print(f"  {'载荷':<26} {'形状':<22} {'dtype':<9} {'字节':>10}")
    total = 0
    for p in payload:
        total += p["bytes"]
        print(f"  {p['name']:<26} {str(tuple(p['shape'])):<22} {p['dtype']:<9} "
              f"{p['bytes']/2**20:>9.3f} MiB")
    print(f"  单张 640×480 图的视觉特征总载荷 {total/2**20:.3f} MiB"
          f"（4.8 的 feature cache 需要按这份清单做键）")

    sub("C2 分析性消融：去掉 DeepStack 注入")
    originals = {id(m): m.forward for m in vis.deepstack_merger_list}

    def zero_forward(self, x):
        y = originals[id(self)](x)
        return torch.zeros_like(y)

    TASKS = [("read", "图中的编号字符串是什么？只回答字符串。"),
             ("count", "图中有几个红色圆点？只回答数字。")]
    test_imgs = []
    n_ablate = getattr(args, "n_ablate", 6)
    for i in range(n_ablate):
        im = Image.new("RGB", (640, 480), (18, 22, 34))
        dr = ImageDraw.Draw(im)
        if i % 2 == 0:
            s = f"TOKEN-{i:02d}-AB7"
            dr.text((60, 220), s, fill=(255, 255, 255))
            test_imgs.append((im, "read", s))
        else:
            n = 3 + i % 4
            for k in range(n):
                cx, cy = 90 + 70 * k, 240
                dr.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(220, 90, 60))
            test_imgs.append((im, "count", n))

    def run_all():
        out = []
        for im, kind, ans in test_imgs:
            prompt = dict(TASKS)[kind]
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
            ins = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**ins, max_new_tokens=12, do_sample=False)
            dt = time.perf_counter() - t0
            txt = proc.batch_decode(gen[:, ins["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
            out.append({"kind": kind, "answer": ans, "output": txt[:30],
                        "hit": str(ans).lower() in txt.lower(), "s": dt})
        return out

    base = run_all()
    for m in vis.deepstack_merger_list:
        m.forward = zero_forward.__get__(m, type(m))
    try:
        ablated = run_all()
    finally:
        for m in vis.deepstack_merger_list:
            m.forward = originals[id(m)]
    hb = sum(r["hit"] for r in base)
    ha = sum(r["hit"] for r in ablated)
    print(f"  原模型 {hb}/{len(base)} 命中，平均 {sum(r['s'] for r in base)/len(base)*1e3:.0f} ms")
    print(f"  去掉 DeepStack 注入 {ha}/{len(ablated)} 命中，"
          f"平均 {sum(r['s'] for r in ablated)/len(ablated)*1e3:.0f} ms")
    for a, b in zip(base, ablated):
        print(f"    {a['kind']:<6} 答案 {str(a['answer']):<12} 原 {a['output'][:14]:<16}"
              f" {'✓' if a['hit'] else '✗'} | 消融 {b['output'][:14]:<16}"
              f" {'✓' if b['hit'] else '✗'}")
    print("  这是分析性消融：把 deepstack merger 的输出置零，其余保持官方配置；")
    print("  修改后的模型不能称为官方配置，结论只用于说明注入点的作用。")
    SUMMARY["C"] = {"payload": payload, "total_bytes": total,
                    "deepstack_indexes": idxs,
                    "baseline": base, "ablated": ablated,
                    "baseline_hits": hb, "ablated_hits": ha,
                    "n": len(base)}
    return ctx if ctx else (d, proc, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B"])
    ap.add_argument("--n-ablate", type=int, default=6,
                    help="DeepStack 消融的样本数（read/count 各一半）")
    ap.add_argument("--outdir", default=os.path.expanduser("~/l46_out"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B"]
    SUMMARY["env"] = {"torch": torch.__version__,
                      "transformers": __import__("transformers").__version__,
                      "gpu": torch.cuda.get_device_name(0)}
    ctx = None
    if "A" in want:
        ctx = section_A(args, ctx)
    if "B" in want:
        ctx = section_B(args, ctx)
    if "C" in want:
        ctx = section_C(args, ctx)
    path = os.path.join(args.outdir, "vision_connector.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
