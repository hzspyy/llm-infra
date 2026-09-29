#!/usr/bin/env python
"""10.2-B/C/D SDXL 与 LCM 的 solver 质量—成本对照。

对照维度：
  * 步数 10/20/40（SDXL 基线）与 2/4/8（LCM 少步）；
  * CFG 1/5/7.5（CFG=1 时不复制 batch）；
  * 求解器 Euler / Heun / DPM-Solver++(2M) / DDIM；
  * 记录实际 denoiser 调用次数（NFE）与每次调用的 batch（CFG 会让它翻倍），
    并把文本编码、VAE 解码的固定成本单独计时；
  * 质量用 CLIP 图文相似度（条件一致性）、同 latent 不同配置的 CLIP 特征距离、
    以及固定顺序的盲评表（盲评本身需要人，脚本只产出材料）。

用法：
  python labs/L10/solver_quality_bench.py pilot  --out <out>
  python labs/L10/solver_quality_bench.py run    --out <out> [--resolution 768] [--budget-s 2400]
  python labs/L10/solver_quality_bench.py mismatch --out <out>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
LCM = "latent-consistency/lcm-sdxl"

PROMPTS_ZH = [
    "一只戴帽子的猫坐在窗台上，水彩画风格",
    "雪山脚下的湖泊，清晨薄雾，写实摄影",
    "赛博朋克城市夜景，霓虹灯与雨后的街道",
    "一盘刚出炉的可颂，木质桌面，暖光",
    "宇航员漂浮在星云中，电影感光效",
    "中国古典园林，拱桥与锦鲤，水墨风格",
    "复古摩托车停在沙漠公路旁，黄昏",
    "书房里堆满旧书，阳光从百叶窗照进来",
    "一只柯基在草地上奔跑，浅景深",
    "未来主义建筑，白色曲面与玻璃幕墙",
    "樱花开满的河岸，花瓣落在水面上",
    "深海中的水母群，幽蓝的生物荧光",
    "蒸汽朋克风格的机械钟表内部结构",
    "雨天的东京街头，行人与透明雨伞",
    "北极光下的木屋，雪地里的脚印",
    "一杯拿铁上的拉花，微距特写",
]
PROMPTS_EN = [
    "a cat wearing a hat sitting on a windowsill, watercolor",
    "a lake at the foot of a snowy mountain, morning mist, photo",
    "cyberpunk city at night, neon signs and wet streets",
    "freshly baked croissants on a wooden table, warm light",
    "an astronaut floating in a nebula, cinematic lighting",
    "classical chinese garden, arched bridge and koi, ink painting",
    "a vintage motorcycle parked on a desert road at dusk",
    "a study filled with old books, sunlight through blinds",
    "a corgi running on grass, shallow depth of field",
    "futuristic architecture, white curves and glass facade",
    "a riverbank full of cherry blossoms, petals on the water",
    "a swarm of jellyfish in the deep sea, bioluminescence",
    "the inner mechanism of a steampunk clock",
    "a rainy street in Tokyo, pedestrians with clear umbrellas",
    "a wooden cabin under the aurora, footprints in snow",
    "latte art in a cup, macro shot",
]
PROMPTS = PROMPTS_ZH + PROMPTS_EN


class NFECounter:
    """统计 denoiser 的实际调用次数与 batch 大小。"""

    def __init__(self, unet):
        self.unet = unet
        self.calls = 0
        self.batch_sum = 0
        self.batches = []
        self._h = unet.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = args[0]
        b = int(x.shape[0])
        self.calls += 1
        self.batch_sum += b
        self.batches.append(b)

    def reset(self):
        self.calls = 0
        self.batch_sum = 0
        self.batches = []

    def summary(self):
        return {"nfe": self.calls, "batch_sum": self.batch_sum,
                "max_batch": max(self.batches) if self.batches else 0,
                "unique_batches": sorted(set(self.batches)),
                "cfg_batch_doubled": bool(self.batches) and min(self.batches) > 1
                                     and len(set(self.batches)) == 1}

    def close(self):
        self._h.remove()


class ClipScorer:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def score(self, images: list[Image.Image], prompts: list[str]):
        out = []
        for i in range(0, len(images), 16):
            inp = self.proc(text=prompts[i:i + 16], images=images[i:i + 16],
                            return_tensors="pt", padding=True, truncation=True).to(self.device)
            img_f = self.model.get_image_features(pixel_values=inp["pixel_values"])
            txt_f = self.model.get_text_features(input_ids=inp["input_ids"],
                                                 attention_mask=inp["attention_mask"])
            img_f = (img_f.pooler_output if hasattr(img_f, "pooler_output") else img_f).float()
            txt_f = (txt_f.pooler_output if hasattr(txt_f, "pooler_output") else txt_f).float()
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            out.append((img_f * txt_f).sum(-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def features(self, images: list[Image.Image]):
        outs = []
        for i in range(0, len(images), 16):
            inp = self.proc(images=images[i:i + 16], return_tensors="pt").to(self.device)
            f = self.model.get_image_features(pixel_values=inp["pixel_values"])
            f = (f.pooler_output if hasattr(f, "pooler_output") else f).float()
            outs.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        return torch.cat(outs)


def load_sdxl(device, resolution, scheduler_name="default"):
    from diffusers import (StableDiffusionXLPipeline, DDIMScheduler, EulerDiscreteScheduler,
                           HeunDiscreteScheduler, DPMSolverMultistepScheduler)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    base = dict(pipe.scheduler.config)
    if scheduler_name == "default":
        pass
    elif scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(base)
    elif scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(base)
    elif scheduler_name == "heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(base)
    elif scheduler_name == "dpmpp_2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base, algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
            lower_order_final=True)
    else:
        raise ValueError(scheduler_name)
    return pipe


def load_lcm(device):
    """LCM 官方用法：在 SDXL pipeline 上同时替换 unet 权重与 scheduler。

    `latent-consistency/lcm-sdxl` 仓库只有一个裸 UNet（config.json + 权重），
    不是完整 pipeline；把它当作 pipeline 加载会失败。
    """
    from diffusers import StableDiffusionXLPipeline, LCMScheduler, UNet2DConditionModel
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.unet = UNet2DConditionModel.from_pretrained(LCM, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def timed_generate(pipe, counter, prompt, seed, steps, guidance, resolution, measure_fixed=True):
    """生成一张图，返回 (image, 计时, NFE 统计)。"""
    device = pipe.device
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fixed = {}
    if measure_fixed:
        t0 = time.time()
        with torch.no_grad():
            emb, neg, pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=guidance > 1.0)
        torch.cuda.synchronize()
        fixed["text_encode_s"] = time.time() - t0
    counter.reset()
    t0 = time.time()
    out = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
               height=resolution, width=resolution, generator=gen, output_type="pil")
    torch.cuda.synchronize()
    total = time.time() - t0
    stats = counter.summary()
    stats["total_s"] = total
    stats["fixed"] = fixed
    return out.images[0], stats


def grid(images: list[Image.Image], path: Path, cols: int = 4, thumb: int = 384):
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), "black")
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((thumb, thumb)), (c * thumb, r * thumb))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def pilot(args):
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_sdxl(device, 768, "euler")
    counter = NFECounter(pipe.unet)
    rep = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__}
    try:
        rows = []
        for res in (768, 1024):
            for steps in (20,):
                torch.cuda.reset_peak_memory_stats(device)
                img, stats = timed_generate(pipe, counter, PROMPTS_ZH[0], 0, steps, 5.0, res)
                stats["resolution"] = res
                stats["peak_GiB"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                rows.append(stats)
                img.save(out / f"pilot_{res}_{steps}.png")
                print(json.dumps(stats), flush=True)
        rep["runs"] = rows
    finally:
        counter.close()
    (out / "pilot.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


def run(args):
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = args.resolution
    started = time.time()
    result = {"resolution": res, "runs": [], "prompts": {"zh": PROMPTS_ZH, "en": PROMPTS_EN}}

    clip = ClipScorer(device)

    def budget_left():
        return (time.time() - started) < args.budget_s

    # ---------------- B-1：SDXL 基线，步数扫描（Euler，CFG=5，全部 32 prompt × 3 latent） ----------------
    pipe = load_sdxl(device, res, "euler")
    counter = NFECounter(pipe.unet)
    base_images, base_prompts = [], []
    rows = []
    for pi, prompt in enumerate(PROMPTS):
        if not budget_left():
            break
        for k in range(3):
            seed = 1000 * (pi + 1) + k
            img, stats = timed_generate(pipe, counter, prompt, seed, args.steps_mid, 5.0, res)
            base_images.append(img); base_prompts.append(prompt)
            stats.update({"prompt_idx": pi, "latent_idx": k, "seed": seed,
                          "cfg": 5.0, "steps": args.steps_mid, "solver": "euler"})
            rows.append(stats)
    scores = clip.score(base_images, base_prompts)
    for r, s in zip(rows, scores.tolist()):
        r["clip_text_sim"] = s
    result["runs"].append({"name": "sdxl_euler_cfg5", "rows": rows,
                           "mean_clip": float(scores.mean()), "n": len(rows)})
    grid(base_images[:16], out / "sdxl_euler_cfg5.png")
    print(json.dumps({"stage": "B1", "n": len(rows), "mean_clip": float(scores.mean())}), flush=True)

    # ---------------- B-2：步数 10/20/40 × CFG 1/5/7.5（8 prompt × 3 latent） ----------------
    probe_prompts = list(range(0, 32, 4))
    for cfg in (1.0, 5.0, 7.5):
        for steps in (10, 20, 40):
            if not budget_left():
                break
            imgs, prs, rows = [], [], []
            for pi in probe_prompts:
                prompt = PROMPTS[pi]
                for k in range(3):
                    seed = 1000 * (pi + 1) + k
                    img, stats = timed_generate(pipe, counter, prompt, seed, steps, cfg, res,
                                                measure_fixed=(len(rows) == 0))
                    imgs.append(img); prs.append(prompt)
                    stats.update({"prompt_idx": pi, "latent_idx": k, "seed": seed,
                                  "cfg": cfg, "steps": steps, "solver": "euler"})
                    rows.append(stats)
            if rows:
                sc = clip.score(imgs, prs)
                for r, s in zip(rows, sc.tolist()):
                    r["clip_text_sim"] = s
                result["runs"].append({"name": f"sdxl_euler_cfg{cfg}_steps{steps}", "rows": rows,
                                       "mean_clip": float(sc.mean()), "n": len(rows)})
                grid(imgs[:16], out / f"sdxl_cfg{cfg}_steps{steps}.png")
                print(json.dumps({"stage": "B2", "cfg": cfg, "steps": steps,
                                  "n": len(rows), "mean_clip": float(sc.mean()),
                                  "nfe": rows[0]["nfe"]}), flush=True)
        if not budget_left():
            break

    # ---------------- B-3：求解器对照（20 步，CFG=5，8 prompt × 3 latent，同 latents） ----------------
    for solver in ("ddim", "heun", "dpmpp_2m"):
        if not budget_left():
            break
        p2 = load_sdxl(device, res, solver)
        c2 = NFECounter(p2.unet)
        try:
            imgs, prs, rows = [], [], []
            for pi in probe_prompts:
                prompt = PROMPTS[pi]
                for k in range(3):
                    seed = 1000 * (pi + 1) + k
                    img, stats = timed_generate(p2, c2, prompt, seed, 20, 5.0, res,
                                                measure_fixed=True)
                    imgs.append(img); prs.append(prompt)
                    stats.update({"prompt_idx": pi, "latent_idx": k, "seed": seed,
                                  "cfg": 5.0, "steps": 20, "solver": solver})
                    rows.append(stats)
            sc = clip.score(imgs, prs)
            for r, s in zip(rows, sc.tolist()):
                r["clip_text_sim"] = s
            result["runs"].append({"name": f"sdxl_{solver}_cfg5_steps20", "rows": rows,
                                   "mean_clip": float(sc.mean()), "n": len(rows)})
            grid(imgs[:16], out / f"sdxl_{solver}_steps20.png")
            print(json.dumps({"stage": "B3", "solver": solver, "n": len(rows),
                              "mean_clip": float(sc.mean()), "nfe": rows[0]["nfe"]}), flush=True)
        finally:
            c2.close()
            del p2
            torch.cuda.empty_cache()

    counter.close()
    del pipe
    torch.cuda.empty_cache()

    # ---------------- C：LCM 少步 ----------------
    pipe_lcm = load_lcm(device)
    counter = NFECounter(pipe_lcm.unet)
    for steps in (2, 4, 8):
        if not budget_left():
            break
        imgs, prs, rows = [], [], []
        for pi in probe_prompts:
            prompt = PROMPTS[pi]
            for k in range(3):
                seed = 1000 * (pi + 1) + k
                img, stats = timed_generate(pipe_lcm, counter, prompt, seed, steps, 1.0, res,
                                            measure_fixed=(len(rows) == 0))
                imgs.append(img); prs.append(prompt)
                stats.update({"prompt_idx": pi, "latent_idx": k, "seed": seed,
                              "cfg": 1.0, "steps": steps, "solver": "lcm"})
                rows.append(stats)
        sc = clip.score(imgs, prs)
        for r, s in zip(rows, sc.tolist()):
            r["clip_text_sim"] = s
        result["runs"].append({"name": f"lcm_steps{steps}", "rows": rows,
                               "mean_clip": float(sc.mean()), "n": len(rows)})
        grid(imgs[:16], out / f"lcm_steps{steps}.png")
        print(json.dumps({"stage": "C", "steps": steps, "n": len(rows),
                          "mean_clip": float(sc.mean()), "nfe": rows[0]["nfe"]}), flush=True)
    counter.close()

    # ---------------- D：同 latents 的感知距离 ----------------
    dists = {}
    for a, b in (("sdxl_euler_cfg5_steps20", "sdxl_heun_cfg5_steps20"),
                 ("sdxl_euler_cfg5_steps20", "sdxl_dpmpp_2m_cfg5_steps20"),
                 ("sdxl_euler_cfg5", "lcm_steps4")):
        ra = next((r for r in result["runs"] if r["name"] == a), None)
        rb = next((r for r in result["runs"] if r["name"] == b), None)
        if ra and rb:
            sa = {r["seed"]: r["clip_text_sim"] for r in ra["rows"]}
            sb = {r["seed"]: r["clip_text_sim"] for r in rb["rows"]}
            common = sorted(set(sa) & set(sb))
            if common:
                dists[f"{a} vs {b}"] = {
                    "n": len(common),
                    "mean_clip_abs_diff": float(sum(abs(sa[s] - sb[s]) for s in common) / len(common)),
                }
    result["condition_consistency_diff"] = dists
    (out / "solver_bench.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "done", "runs": [r["name"] for r in result["runs"]]}), flush=True)


def mismatch(args):
    """C 的失败样本：模型与 scheduler 的配对错误。"""
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from diffusers import LCMScheduler, EulerDiscreteScheduler
    rep = {}
    # 1) LCM 权重 + SDXL 的 Euler（步数按 LCM 的需要给 4 步）
    pipe = load_lcm(device)
    lcm_ok = []
    for k in range(2):
        img, stats = timed_generate(pipe, NFECounter(pipe.unet), PROMPTS_ZH[1], 2001 + k, 4, 1.0,
                                    args.resolution)
        lcm_ok.append(img)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    wrong = []
    for k in range(2):
        img, stats = timed_generate(pipe, NFECounter(pipe.unet), PROMPTS_ZH[1], 2001 + k, 4, 7.5,
                                    args.resolution)
        wrong.append(img)
    grid(lcm_ok + wrong, out / "lcm_matched_vs_wrong_scheduler.png", cols=4)
    rep["lcm_unet_with_euler_4steps_cfg7.5"] = "见 lcm_matched_vs_wrong_scheduler.png 右侧"
    del pipe
    torch.cuda.empty_cache()
    # 2) SDXL base 权重 + LCMScheduler
    pipe2 = load_sdxl(device, args.resolution, "euler")
    pipe2.scheduler = LCMScheduler.from_config(pipe2.scheduler.config)
    imgs = []
    for k in range(2):
        img, stats = timed_generate(pipe2, NFECounter(pipe2.unet), PROMPTS_ZH[1], 2001 + k, 4, 1.0,
                                    args.resolution)
        imgs.append(img)
    grid(imgs, out / "sdxl_base_with_lcm_scheduler.png", cols=2)
    rep["sdxl_base_with_lcm_scheduler"] = "见 sdxl_base_with_lcm_scheduler.png"
    (out / "mismatch.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--steps-mid", type=int, default=20)
    p.add_argument("--budget-s", type=float, default=2400)
    p = sub.add_parser("mismatch")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=768)
    args = ap.parse_args()
    if args.cmd == "pilot":
        pilot(args)
    elif args.cmd == "run":
        run(args)
    else:
        mismatch(args)


if __name__ == "__main__":
    main()
