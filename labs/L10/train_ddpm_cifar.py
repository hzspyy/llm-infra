#!/usr/bin/env python
"""10.1-G 小型 DDPM 完整教学项目：CIFAR-10 随机初始化小 UNet。

要求（见 docs/plans/pending.md#c-10-1 的 G 与训练共同任务）：
  数据划分 → 随机初始化小 UNet → DDPM 训练 + EMA → 验证 → checkpoint/恢复 →
  导出 → 固定噪声采样 → 独立误差/多样性/记忆检查 → 阶段样本与 NFE/延迟/资源成本。

训练与验证/测试严格分开：训练只用 train split，最终评测只用 test split，训练图像不进入评测。

确定性：数据顺序由 (seed, epoch) 决定，扩散噪声由 (seed, step) 决定，
因此「中断后继续」与「不中断」必须给出逐位相同的参数轨迹；本脚本用 --stop-at 支持该对拍。

用法示例：
  python train_ddpm_cifar.py pilot --out /path/out --steps 100
  python train_ddpm_cifar.py train --out /path/out --steps 20000 --tag eps
  python train_ddpm_cifar.py train --out /path/out --steps 20000 --tag eps --stop-at 5000
  python train_ddpm_cifar.py train --out /path/out --steps 20000 --tag eps --resume
  python train_ddpm_cifar.py eval --out /path/out --tag eps --n 4096 --ddim-steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


# --------------------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------------------
def load_cifar(out_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (train_uint8 [N,3,32,32], test_uint8, test_labels)。缓存为 .pt。"""
    cache = out_dir / "data"
    cache.mkdir(parents=True, exist_ok=True)
    f_train, f_test = cache / "cifar10_train.pt", cache / "cifar10_test.pt"
    if f_train.exists() and f_test.exists():
        d = torch.load(f_train, map_location="cpu")
        t = torch.load(f_test, map_location="cpu")
        return d["x"], t["x"], t["y"]
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10")
    def conv(split):
        xs = torch.stack([torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1)
                          for im in split["img"]])
        ys = torch.tensor(split["label"], dtype=torch.long)
        return xs, ys
    xtr, _ = conv(ds["train"])
    xte, yte = conv(ds["test"])
    torch.save({"x": xtr}, f_train)
    torch.save({"x": xte, "y": yte}, f_test)
    return xtr, xte, yte


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div(255.0).mul(2.0).sub(1.0)


# --------------------------------------------------------------------------------------
# 模型与调度器
# --------------------------------------------------------------------------------------
def build_unet(base: int = 64, attention: bool = True):
    from diffusers import UNet2DModel
    if attention:
        down = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    else:
        down = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
        up = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(base, base * 2, base * 4),
        down_block_types=down,
        up_block_types=up,
        attention_head_dim=base // 4 if attention else None,
        norm_num_groups=8,
    )


def build_scheduler(prediction_type: str = "epsilon", clipped: bool = True):
    from diffusers import DDPMScheduler
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type=prediction_type,
        clip_sample=clipped and prediction_type in ("epsilon", "sample"),
        clip_sample_range=1.0,
    )


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}

    def copy_to(self, model):
        sd = model.state_dict()
        out = {k: (self.shadow[k].to(sd[k].dtype) if sd[k].dtype.is_floating_point else sd[k])
               for k in sd}
        model.load_state_dict(out)


# --------------------------------------------------------------------------------------
# 数据顺序与噪声（可精确恢复）
# --------------------------------------------------------------------------------------
def epoch_perm(n: int, seed: int, epoch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    return torch.randperm(n, generator=g)


def batch_indices(n: int, seed: int, step: int, batch: int) -> tuple[torch.Tensor, int, int]:
    """返回第 step 步（0-based）的样本索引与该步所在 epoch/偏移。"""
    per_epoch = n // batch
    epoch = step // per_epoch
    off = (step % per_epoch) * batch
    perm = epoch_perm(n, seed, epoch)
    return perm[off:off + batch], epoch, off


def step_generator(seed: int, step: int, device) -> torch.Generator:
    g = torch.Generator(device=device if device.type == "cpu" else "cpu")
    g.manual_seed((seed * 1000003 + step) % (2 ** 31 - 1))
    return g


def randn_like_t(shape, seed, step, device, dtype):
    """按 (seed, step) 生成确定噪声，设备无关（先在 CPU 生成再搬运）。"""
    g = torch.Generator(device="cpu").manual_seed((seed * 1000003 + step) % (2 ** 31 - 1))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------------------
@torch.no_grad()
def sample_ddim(model, scheduler, n, ddim_steps, device, seed, nfe_only=False, batch=128):
    """确定性 DDIM 采样；返回 [n,3,32,32] 与 NFE。"""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_config(scheduler.config)
    sched.set_timesteps(ddim_steps)
    xs = []
    model.eval()
    for i in range(0, n, batch):
        k = min(batch, n - i)
        x = randn_like_t((k, 3, 32, 32), seed + i, 0, device, torch.float32)
        for t in sched.timesteps:
            eps = model(x, t.to(device)).sample
            x = sched.step(eps, t, x, eta=0.0).prev_sample
        xs.append(x.clamp(-1, 1))
    return torch.cat(xs, 0), len(sched.timesteps)


@torch.no_grad()
def sample_ddpm(model, scheduler, n, steps, device, seed, batch=128):
    """祖先采样（随机）；用于与 DDIM 对照。"""
    sched = scheduler
    sched.set_timesteps(steps)
    xs = []
    model.eval()
    for i in range(0, n, batch):
        k = min(batch, n - i)
        x = randn_like_t((k, 3, 32, 32), seed + i, 0, device, torch.float32)
        for j, t in enumerate(sched.timesteps):
            eps = model(x, t.to(device)).sample
            x = sched.step(eps, t, x, generator=torch.Generator(device="cpu").manual_seed(seed + i + j)).prev_sample
        xs.append(x.clamp(-1, 1))
    return torch.cat(xs, 0), len(sched.timesteps)


def save_grid(x: torch.Tensor, path: Path, cols: int = 8):
    """把 [-1,1] 的 32x32 图拼成 PNG 网格。"""
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
    n = x.shape[0]
    rows = (n + cols - 1) // cols
    grid = torch.zeros(rows * 32, cols * 32, 3, dtype=torch.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = x[i].permute(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid.numpy()).save(path)


# --------------------------------------------------------------------------------------
# 特征与指标（CLIP 特征上的 Fréchet 距离，不是 FID；另做多样性与记忆检查）
# --------------------------------------------------------------------------------------
def _as_tensor(out):
    """transformers 5.x 的 get_image_features/get_text_features 返回带 pooler_output 的输出对象。"""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"unsupported feature output: {type(out)}")


class ClipFeat:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def features(self, x_minus1: torch.Tensor, batch: int = 256) -> torch.Tensor:
        outs = []
        for i in range(0, x_minus1.shape[0], batch):
            imgs = ((x_minus1[i:i + batch].clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
            pil = [Image.fromarray(im.transpose(1, 2, 0)) for im in imgs]
            inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
            f = _as_tensor(self.model.get_image_features(**inputs))
            outs.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def zero_shot_hist(self, x_minus1: torch.Tensor, batch: int = 256) -> list:
        prompts = [f"a photo of a {c}" for c in CIFAR_CLASSES]
        text = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        tf = _as_tensor(self.model.get_text_features(**text))
        tf = F.normalize(tf.float(), dim=-1).cpu()
        feats = self.features(x_minus1, batch)
        pred = (feats @ tf.t()).argmax(-1)
        return torch.bincount(pred, minlength=10).tolist()


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    c1 = torch.cov(a.t()) if a.shape[0] > a.shape[1] else torch.cov(a.t())
    c2 = torch.cov(b.t())
    # 用特征维度上的协方差（512x512），数值上先做 Cholesky 分解对称化
    diff = mu1 - mu2
    eps = 1e-6 * torch.eye(c1.shape[0])
    s1 = torch.linalg.cholesky(c1 + eps)
    s2 = torch.linalg.cholesky(c2 + eps)
    # tr((C1 C2)^{1/2}) 用奇异值求和
    sv = torch.linalg.svdvals(s1 @ s2)
    return float(diff.dot(diff) + torch.diagonal(c1).sum() + torch.diagonal(c2).sum() - 2 * sv.sum())


def pairwise_cos_dist_mean(f: torch.Tensor, max_n: int = 512, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(f.shape[0], generator=g)[:max_n]
    s = f[idx]
    d = 1 - s @ s.t()
    n = s.shape[0]
    return float(d.sum() / (n * (n - 1)))


def nn_distance_stats(query: torch.Tensor, ref: torch.Tensor, max_q: int = 1000, max_r: int = 10000) -> dict:
    """query 到 ref 的最近邻 L2 距离分位数（展平像素空间，ref 抽样以控内存）。"""
    nq = min(query.shape[0], max_q)
    nr = min(ref.shape[0], max_r)
    q = query[:nq].reshape(nq, -1).float().cpu()
    r = ref[:nr].reshape(nr, -1).float().cpu()
    d = torch.cdist(q, r)
    vals = d.min(dim=1).values
    return {
        "p05": float(vals.quantile(0.05)),
        "p50": float(vals.quantile(0.50)),
        "p95": float(vals.quantile(0.95)),
        "mean": float(vals.mean()),
    }


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    xtr, xte, yte = load_cifar(out)
    n = xtr.shape[0]
    model = build_unet(args.base, attention=not args.no_attention).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    scheduler = build_scheduler(args.prediction_type, clipped=not args.no_clip)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    start_step = 0
    state_path = out / f"state_{args.tag}.pt"
    if args.resume and state_path.exists():
        sd = torch.load(state_path, map_location=device)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        ema.load_state_dict(sd["ema"])
        start_step = sd["step"]
        print(f"[resume] from step {start_step}", flush=True)

    accum = 0
    t0 = time.time()
    first_losses = []
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step = start_step
    while step < args.steps:
        idx, epoch, off = batch_indices(n, args.seed, step, args.batch)
        x0 = normalize(xtr[idx]).to(device, non_blocking=True)
        if args.flip:
            g = step_generator(args.seed, step, device)
            flip_mask = torch.rand(x0.shape[0], generator=g) < 0.5
            x0[flip_mask] = torch.flip(x0[flip_mask], dims=[-1])
        noise = randn_like_t(x0.shape, args.seed + 777, step, device, x0.dtype)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.shape[0],),
                          generator=step_generator(args.seed + 1, step, device)).to(device)
        xt = scheduler.add_noise(x0, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and bool(args.amp))):
            pred = model(xt, t).sample
            loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)
        first_losses.append(float(loss.detach()))
        step += 1

        if step % args.log_every == 0 or step == args.steps:
            recent = sum(first_losses[-args.log_every:]) / len(first_losses[-args.log_every:])
            el = time.time() - t0
            print(json.dumps({"step": step, "loss": recent, "epoch": epoch, "offset": off,
                              "steps_per_s": (step - start_step) / el,
                              "peak_GiB": torch.cuda.max_memory_allocated(device) / 2**30}), flush=True)

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "step": step,
                        "args": vars(args)}, state_path)
            print(f"[save] {state_path} step={step}", flush=True)

        if step in set(args.snapshot_steps):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "step": step, "args": vars(args)},
                       out / f"state_{args.tag}_{step}.pt")
            print(f"[snapshot] step={step}", flush=True)

        if args.stop_at and step >= args.stop_at:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "step": step, "args": vars(args)}, state_path)
            print(f"[stop-at] {step}", flush=True)
            break

    wall = time.time() - t0
    # 最终保存：模型 + EMA + 导出 pipeline
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(),
                "step": step, "args": vars(args)}, state_path)
    ema_model = build_unet(args.base, attention=not args.no_attention).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema.copy_to(ema_model)
    torch.save({"model": ema_model.state_dict(), "step": step}, out / f"ema_{args.tag}.pt")

    # 导出：diffusers pipeline（UNet + scheduler 配置），用 from_pretrained 直接加载
    export_dir = out / f"export_{args.tag}"
    ema_model.save_pretrained(export_dir / "unet")
    scheduler.save_pretrained(export_dir / "scheduler")
    (export_dir / "export_info.json").write_text(json.dumps({
        "prediction_type": args.prediction_type,
        "num_train_timesteps": scheduler.config.num_train_timesteps,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "clip_sample": bool(scheduler.config.clip_sample),
        "params": n_params, "trained_steps": step, "amp": args.amp, "ema_decay": args.ema_decay,
        "data": "uoft-cs/cifar10 train split (50000), test split held out for evaluation",
    }, indent=2), encoding="utf-8")

    summary = {
        "tag": args.tag, "steps": step, "params": n_params, "batch": args.batch,
        "lr": args.lr, "amp": args.amp, "prediction_type": args.prediction_type,
        "ema_decay": args.ema_decay, "flip": args.flip, "grad_clip": args.grad_clip,
        "wall_s": wall, "steps_per_s": (step - start_step) / max(wall, 1e-9),
        "peak_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_loss_mean_last100": float(np.mean(first_losses[-100:])),
        "tokens_or_images": step * args.batch,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "python": platform.python_version(),
        "seed": args.seed, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / f"train_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# --------------------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    xtr, xte, yte = load_cifar(out)
    model = build_unet(args.base, attention=not args.no_attention)
    ck_path = Path(args.ckpt) if args.ckpt else out / (f"ema_{args.tag}.pt" if args.use_ema else f"model_{args.tag}.pt")
    if ck_path.exists():
        blob = torch.load(ck_path, map_location="cpu")
        if args.use_ema and "ema" in blob:                 # 训练中保存的 state 里有 EMA 影子权重
            model.load_state_dict(blob["ema"]["shadow"])
        elif "model" in blob:
            model.load_state_dict(blob["model"])
        else:
            model.load_state_dict(blob)
    else:
        st = torch.load(out / f"state_{args.tag}.pt", map_location="cpu")
        model.load_state_dict(st["ema"]["shadow"] if args.use_ema else st["model"])
    model = model.to(device).eval()
    scheduler = build_scheduler(args.prediction_type, clipped=not args.no_clip)

    g_all = torch.Generator().manual_seed(args.seed)
    # 1) 生成样本（DDIM 确定性）
    t_gen0 = time.time()
    gen, nfe = sample_ddim(model, scheduler, args.n, args.ddim_steps, device, args.seed)
    t_gen = time.time() - t_gen0
    # 2) 同 latents 的祖先采样对照（少量，用于比较两种采样器）
    anc, nfe_a = sample_ddpm(model, scheduler, min(256, args.n), args.ddim_steps, device, args.seed)
    # 3) 参考集：test split（训练从未见过）
    ridx = torch.randperm(xte.shape[0], generator=g_all)[:args.n]
    real = normalize(xte[ridx])

    clip = ClipFeat(device)
    fg = clip.features(gen)
    fr = clip.features(real)
    ftr = clip.features(normalize(xtr[torch.randperm(xtr.shape[0], generator=g_all)[:2048]]))
    fd = frechet_distance(fg, fr)
    fd_ref = frechet_distance(ftr, fr)          # train↔test 的特征距离标尺
    metrics = {
        "n": args.n, "ddim_steps": args.ddim_steps, "nfe": nfe,
        "gen_wall_s": t_gen, "ms_per_image": t_gen / args.n * 1000,
        "clip_fd_gen_vs_test": fd,
        "clip_fd_train_vs_test": fd_ref,
        "diversity_gen": pairwise_cos_dist_mean(fg),
        "diversity_test": pairwise_cos_dist_mean(fr),
        "nn_gen_to_train": nn_distance_stats(gen, normalize(xtr)),
        "nn_gen_to_test": nn_distance_stats(gen, real),
        "nn_test_to_train": nn_distance_stats(real, normalize(xtr)),
        "zero_shot_hist_gen": clip.zero_shot_hist(gen),
        "zero_shot_hist_test": clip.zero_shot_hist(real),
        "ancestral_vs_ddim_mean_abs": float((anc - gen[:anc.shape[0]]).abs().mean()),
        "ddpm_ancestral_nfe": nfe_a,
        "prediction_type": args.prediction_type,
        "use_ema": args.use_ema,
        "gpu": torch.cuda.get_device_name(device),
    }
    if args.wrong_prediction_type:
        sched_wrong = build_scheduler(args.wrong_prediction_type, clipped=not args.no_clip)
        bad, _ = sample_ddim(model, sched_wrong, 64, args.ddim_steps, device, args.seed)
        metrics["mismatched_prediction_type"] = {
            "used": args.wrong_prediction_type,
            "mean_abs_vs_correct": float((bad - gen[:64]).abs().mean()),
            "mean_abs_pixel": float(bad.abs().mean()),
        }
        save_grid(bad[:64], out / f"mismatch_{args.wrong_prediction_type}_{args.tag}.png")

    # 阶段样本与输出
    stem = args.out_name or args.tag
    save_grid(gen[:64], out / f"samples_{stem}_{args.ddim_steps}steps.png")
    save_grid(anc[:64], out / f"samples_ancestral_{stem}_{args.ddim_steps}steps.png")
    suffix = args.out_name or f"{args.tag}"
    (out / f"eval_{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


# --------------------------------------------------------------------------------------
# NFE / 延迟扫描
# --------------------------------------------------------------------------------------
def scan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    model = build_unet(args.base, attention=not args.no_attention)
    ck = Path(args.ckpt) if args.ckpt else out / f"ema_{args.tag}.pt"
    blob = torch.load(ck, map_location="cpu")
    model.load_state_dict(blob.get("model", blob))
    model = model.to(device).eval()
    scheduler = build_scheduler(args.prediction_type, clipped=not args.no_clip)
    rows = []
    # 预热：第一次采样含 cuDNN/内核选择与显存分配，不能计入延迟
    for _ in range(3):
        sample_ddim(model, scheduler, args.n, 10, device, args.seed)
    torch.cuda.synchronize()
    for steps in args.steps_list:
        torch.cuda.synchronize()
        t0 = time.time()
        _, nfe = sample_ddim(model, scheduler, args.n, steps, device, args.seed)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rows.append({"ddim_steps": steps, "nfe": nfe, "wall_s": dt,
                     "ms_per_image": dt / args.n * 1000,
                     "peak_GiB": torch.cuda.max_memory_allocated(device) / 2**30})
        print(json.dumps(rows[-1]), flush=True)
    (out / f"scan_{args.tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "train"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=100)
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--base", type=int, default=64)
        p.add_argument("--ema-decay", type=float, default=0.999)
        p.add_argument("--prediction-type", default="epsilon", choices=["epsilon", "sample", "v_prediction"])
        p.add_argument("--tag", default="eps")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--flip", type=int, default=1)
        p.add_argument("--amp", type=int, default=1)
        p.add_argument("--no-clip", type=int, default=0)
        p.add_argument("--no-attention", type=int, default=0)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--log-every", type=int, default=50)
        p.add_argument("--save-every", type=int, default=2000)
        p.add_argument("--stop-at", type=int, default=0)
        p.add_argument("--resume", type=int, default=0)
        p.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)

    p = sub.add_parser("scan")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="eps")
    p.add_argument("--ckpt", default="")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--prediction-type", default="epsilon")
    p.add_argument("--no-attention", type=int, default=0)
    p.add_argument("--no-clip", type=int, default=0)
    p.add_argument("--steps-list", type=int, nargs="+", default=[5, 10, 20, 50, 100, 250])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval")
    p.add_argument("--out", required=True)
    p.add_argument("--out-name", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--tag", default="eps")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--prediction-type", default="epsilon")
    p.add_argument("--wrong-prediction-type", default="")
    p.add_argument("--use-ema", type=int, default=1)
    p.add_argument("--no-clip", type=int, default=0)
    p.add_argument("--no-attention", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd in ("pilot", "train"):
        train(args)
    elif args.cmd == "compare":
        a = torch.load(args.a, map_location="cpu")
        b = torch.load(args.b, map_location="cpu")
        rep = {"step_a": a["step"], "step_b": b["step"], "groups": {}}
        for grp in ("model", "ema"):
            sa, sb = (a[grp]["shadow"] if grp == "ema" else a[grp]), (b[grp]["shadow"] if grp == "ema" else b[grp])
            mx = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
            rep["groups"][grp] = {"max_abs_diff": mx,
                                  "allclose_0": bool(all(torch.equal(sa[k], sb[k]) for k in sa))}
        pm = [float((pa.float() - pb.float()).abs().max())
              for pa, pb in zip(a["opt"]["state"].values(), b["opt"]["state"].values())
              for pa, pb in [(pa.get("exp_avg"), pb.get("exp_avg"))] if pa is not None]
        rep["optimizer_max_abs_diff"] = max(pm) if pm else None
        print(json.dumps(rep, indent=2))
    elif args.cmd == "scan":
        scan(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
