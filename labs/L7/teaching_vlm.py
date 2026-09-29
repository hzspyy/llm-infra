#!/usr/bin/env python3
"""视觉适配的最小两阶段训练（7.10-J）：冻结 CLIP 视觉塔 + 随机 projector + 教学 LLM。

结构对照 [MiniMind-V](https://github.com/jingyaogong/minimind-v)：同样是"冻结视觉塔 →
projector → 把视觉 token 的 embedding 替换进语言模型序列"。差异写在正文里：本实现用
CLIP ViT-B/32（224²，1 个 CLS + 49 个 patch，取 patch），MiniMind-V 用 SigLIP2-base-p32-256
（64 个 patch）；两者 projector 都是 LayerNorm + Linear + GELU + Linear。

四个阶段分开，便于逐段检查可训练参数：

  data   合成形状图片与问答/描述文本，按图片内容哈希切分 train/val/test（图像级隔离）
  align  只训练 projector（视觉塔与 LLM 冻结），目标是图像描述文本
  sft    projector + LLM 联合训练，目标是问答的回答 token
  eval   留出问答精确匹配、三种图像条件（原图 / 换图 / 空图）

Usage:
    python labs/L7/teaching_vlm.py --mode data --outdir "$RUN/vlm-data"
    python labs/L7/teaching_vlm.py --mode align --data-dir "$RUN/vlm-data" \
      --minimind-src "$SRC/minimind" --vision-path "$CLIP" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/vlm-align"
    python labs/L7/teaching_vlm.py --mode sft --data-dir "$RUN/vlm-data" \
      --minimind-src "$SRC/minimind" --vision-path "$CLIP" --init-from "$RUN/sft/best.pt" \
      --align-from "$RUN/vlm-align/best.pt" --outdir "$RUN/vlm-sft"
    python labs/L7/teaching_vlm.py --mode eval --data-dir "$RUN/vlm-data" \
      --minimind-src "$SRC/minimind" --vision-path "$CLIP" \
      --ckpt "$RUN/vlm-sft/best.pt" --split test --outdir "$RUN/vlm-eval/sft"
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COLORS = {"红色": (220, 60, 60), "绿色": (60, 170, 80), "蓝色": (60, 90, 220)}
SHAPES = ("圆形", "方块", "三角")
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def log(message):
    print(f"[teaching_vlm] {message}", flush=True)


def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def draw_sample(rng):
    """在 224×224 画布上按 3×3 网格放形状，返回 (PIL 图, 计数表, 三个被提问的组合)。

    被提问的三个组合的计数固定取 {0,1,2} 的一个排列，其余组合只放干扰形状。这样每张图的
    三个问答答案恰好是 0/1/2，恒定答案的准确率上限是 1/3 左右——如果答案分布集中在 0，
    "总答 0"就能拿到接近 60% 的分数，指标会退化成先验测量而不是计数测量。
    """
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (224, 224), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    cells = list(range(9))
    rng.shuffle(cells)
    all_combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(all_combos)
    queried = all_combos[:3]
    distractors = all_combos[3:]
    counts = {(c, s): 0 for c, s in all_combos}
    plan = []
    for combo, count in zip(queried, rng.sample([0, 1, 2], 3)):
        plan += [combo] * count
    plan += distractors[:rng.randint(2, 5)]
    rng.shuffle(plan)
    for cell, (color, shape) in zip(cells[:len(plan)], plan):
        row, col = divmod(cell, 3)
        cx, cy = 37 + col * 74 + rng.randint(-6, 6), 37 + row * 74 + rng.randint(-6, 6)
        r = rng.randint(15, 22)
        if shape == "圆形":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color])
        elif shape == "方块":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color])
        else:
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])
        counts[(color, shape)] += 1
    return image, counts, queried


def build_data(outdir: Path, n_train, n_val, n_test, seed):
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "images").mkdir()
    rng = random.Random(seed)
    buckets = {"train": [], "val": [], "test": []}
    targets = {"train": n_train, "val": n_val, "test": n_test}
    seen = set()
    while any(len(buckets[k]) < targets[k] for k in buckets):
        image, counts, queried = draw_sample(rng)
        digest = hashlib.blake2b(image.tobytes(), digest_size=8).hexdigest()
        if digest in seen:                      # 图像级去重，避免同一张图进两个 split
            continue
        seen.add(digest)
        split = "train"                          # 按哈希前缀分区，图像级隔离
        bucket = int(digest[:4], 16) % 100
        if bucket < 12:
            split = "val"
        elif bucket < 24:
            split = "test"
        if len(buckets[split]) >= targets[split]:
            continue
        name = f"{digest}.png"
        image.save(outdir / "images" / name)
        present = [f"{c}{s}有 {n} 个" for (c, s), n in counts.items() if n > 0]
        caption = "这张图是一个形状图案。" + "，".join(present) + "。"
        for color, shape in queried:
            buckets[split].append({
                "id": f"{split}-{len(buckets[split]):05d}", "image": name, "image_sha": digest,
                "question": f"图中有几个{color}{shape}？", "answer": str(counts[(color, shape)]),
                "caption": caption, "counts": {f"{c}{s}": n for (c, s), n in counts.items()},
            })
    for split, rows in buckets.items():
        (outdir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "kind": "synthetic-shape-qa", "seed": seed,
                "images": len(seen), "counts": {k: len(v) for k, v in buckets.items()},
                "resolution": 224, "colors": list(COLORS), "shapes": list(SHAPES),
                "split_rule": "按图像内容 blake2b 哈希前缀分区（<12 val、<24 test），图像级隔离",
                "caption_template": "这张图是一个形状图案。<组合>有 n 个…。",
                "answer_balance": "每张图被提问的三个组合计数是 {0,1,2} 的一个排列，"
                                  "恒定答案的准确率上限约 1/3",
                "question_template": "图中有几个{颜色}{形状}？",
                "boundary": "合成图案语料，不是公开 VQA 数据集；只用于验证 projector 与冻结策略是否真的使用图像"}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"写出 {outdir}：{manifest['counts']}，图片 {len(seen)} 张")


class VLM(nn.Module):
    """冻结 CLIP 视觉塔 + 可训练 projector + MiniMind LLM（按阶段决定是否冻结）。"""

    def __init__(self, minimind_src: Path, vision_path: Path, hidden=768, layers=8,
                 device="cuda"):
        super().__init__()
        sys.path.insert(0, str(minimind_src))
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
        from transformers import CLIPVisionModel, CLIPImageProcessor  # noqa: WPS433
        self.device = device
        self.config = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_moe=False)
        self.llm = MiniMindForCausalLM(self.config).to(device)
        self.vision = CLIPVisionModel.from_pretrained(vision_path).to(device).eval()
        for p in self.vision.parameters():
            p.requires_grad = False
        self.processor = CLIPImageProcessor.from_pretrained(vision_path)
        vision_dim = self.vision.config.hidden_size
        with torch.no_grad():                     # 用一次哑前向读出 patch 网格大小
            probe = self.vision(pixel_values=torch.zeros(1, 3, 224, 224, device=device))
        self.image_token_len = probe.last_hidden_state.size(1) - 1
        self.projector = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden)).to(device)
        self._override = None
        # 视觉 token 的 embedding 由 projector 提供：不新增子模块，state_dict 键与 MiniMind
        # 保持一致。override 只被消费一次，生成时的后续解码步回到普通 embedding。
        embed = self.llm.model.embed_tokens

        def patched_forward(input_ids):
            if self._override is not None:
                override, self._override = self._override, None
                return override
            return F.embedding(input_ids, embed.weight)

        embed.forward = patched_forward

    def trainable_groups(self, stage):
        for p in self.llm.parameters():
            p.requires_grad = stage == "sft"
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True

    def encode(self, pixel_values):
        with torch.no_grad():
            out = self.vision(pixel_values=pixel_values).last_hidden_state
        patches = out[:, 1:, :] if out.size(1) == self.image_token_len + 1 else out
        return self.projector(patches.float().to(self.projector[0].weight.dtype))

    def set_vision_embeds(self, pixel_values, text_ids):
        feats = self.encode(pixel_values)
        text_embeds = F.embedding(text_ids, self.llm.model.embed_tokens.weight)
        self._override = torch.cat([feats, text_embeds], dim=1)

    def full_input_ids(self, text_ids, pad_id):
        """视觉 token 的位置也要占位：RoPE 按 input_ids 的宽度分片，必须传全宽 id。"""
        image_ids = torch.full((text_ids.size(0), self.image_token_len), pad_id,
                               dtype=text_ids.dtype, device=text_ids.device)
        return torch.cat([image_ids, text_ids], dim=1)

    def forward(self, pixel_values, text_ids, attention_mask, labels, pad_id=0):
        self.set_vision_embeds(pixel_values, text_ids)
        return self.llm(input_ids=self.full_input_ids(text_ids, pad_id),
                        attention_mask=attention_mask, labels=labels)

    def save(self, path):
        torch.save({"llm": {k: v.half().cpu() for k, v in self.llm.state_dict().items()},
                    "projector": {k: v.float().cpu() for k, v in self.projector.state_dict().items()},
                    "vision_path": str(self.vision.name_or_path)}, path)

    def load(self, path):
        """VLM checkpoint 载入 LLM 与 projector；只给 MiniMind checkpoint 时 projector 保持随机。

        后一种情形就是"随机 projector + 已 SFT 的 LLM"未适配基线。
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "llm" in state:
            self.llm.load_state_dict(state["llm"])
            self.projector.load_state_dict(state["projector"])
        elif "model" in state:
            self.llm.load_state_dict(state["model"])
        else:
            self.llm.load_state_dict(state)


def shuffled_images(rows):
    """为每条样本指定"另一张图"的文件名：按图像分组后整体轮换，保证换的是不同图片。

    同一条样本的 3 个问答共用一张图，若只是把行下标平移 1，2/3 的样本拿到的还是原图，
    对照会失效。
    """
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_sha"], []).append(row["image"])
    keys = list(by_image)
    rotated = keys[1:] + keys[:1]
    mapping = {}
    for src, dst in zip(keys, rotated):
        for image in by_image[src]:
            mapping.setdefault(image, by_image[dst][0])
    return {row["id"]: mapping[row["image"]] for row in rows}


def make_batch(vlm, tokenizer, rows, image_dir, mode, max_text_len, device, target="qa",
               image_override=None):
    from PIL import Image
    prompts, answers, images = [], [], []
    for i, row in enumerate(rows):
        if target == "caption":
            question, answer_text = "描述这张图。", row["caption"]
        else:
            question, answer_text = row["question"], row["answer"]
        prompt = f"<|im_start|>user\n问题：{question}<|im_end|>\n" + ASSISTANT_PREFIX
        prompts.append(tokenizer(prompt, add_special_tokens=False).input_ids[:max_text_len])
        answers.append(tokenizer(f"{answer_text}<|im_end|>\n",
                                 add_special_tokens=False).input_ids)
        if mode == "blank":
            images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
        elif mode == "shuffle":
            images.append(Image.open(image_dir / image_override[i]).convert("RGB"))
        else:
            images.append(Image.open(image_dir / row["image"]).convert("RGB"))
    width = min(max(len(p) + len(a) for p, a in zip(prompts, answers)), max_text_len)
    text_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        p, a = p[:width], a[:max(0, width - len(p))]
        text_ids[i, :len(p)] = torch.tensor(p)
        text_ids[i, len(p):len(p) + len(a)] = torch.tensor(a)
        labels[i, len(p):len(p) + len(a)] = torch.tensor(a)
    pixels = vlm.processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    mask = torch.cat([torch.ones((len(rows), vlm.image_token_len), dtype=torch.long),
                      (text_ids != tokenizer.pad_token_id).long()], dim=1).to(device)
    full_labels = torch.cat([torch.full((len(rows), vlm.image_token_len), -100, dtype=torch.long),
                             labels], dim=1).to(device)
    return pixels, text_ids.to(device), mask, full_labels


def first_digit_match(pred, gold):
    digits = "".join(ch for ch in pred if ch.isdigit())
    return bool(digits) and digits[0] == gold


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["data", "align", "sft", "eval"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--minimind-src", type=Path, default=None)
    parser.add_argument("--vision-path", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None, help="MiniMind SFT checkpoint")
    parser.add_argument("--align-from", type=Path, default=None, help="对齐阶段产出的 projector")
    parser.add_argument("--ckpt", type=Path, default=None, help="评测用的 VLM checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-text-len", type=int, default=96)
    parser.add_argument("--micro-bs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--limit-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-snapshots", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    if args.mode == "data":
        build_data(args.outdir, args.n_train, args.n_val, args.n_test, args.seed)
        return
    if args.data_dir is None or args.minimind_src is None or args.vision_path is None:
        raise SystemExit("需要 --data-dir、--minimind-src 与 --vision-path")

    sys.path.insert(0, str(args.minimind_src))
    from transformers import AutoTokenizer  # noqa: WPS433
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model")
    vlm = VLM(args.minimind_src, args.vision_path, args.hidden_size, args.num_hidden_layers,
              device=device)
    if args.mode in ("align", "sft") and args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        vlm.llm.load_state_dict(state["model"])
        log(f"LLM 初始化自 {args.init_from}（step={state.get('step')}）")
    if args.mode == "sft" and args.align_from:
        align = torch.load(args.align_from, map_location=device, weights_only=False)
        vlm.projector.load_state_dict(align["projector"])
        log(f"projector 初始化自 {args.align_from}")
    if args.mode == "eval":
        vlm.load(args.ckpt)
        log(f"评测权重 {args.ckpt}")

    def read_split(split):
        return [json.loads(line) for line in
                (args.data_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]

    image_dir = args.data_dir / "images"
    pad_id = tokenizer.pad_token_id

    @torch.no_grad()
    def evaluate(rows, mode="normal"):
        vlm.eval()
        correct = 0
        override = shuffled_images(rows) if mode == "shuffle" else None
        for i in range(0, len(rows), args.micro_bs):
            chunk = rows[i:i + args.micro_bs]
            chunk_override = [override[row["id"]] for row in chunk] if override else None
            pixels, text_ids, mask, _ = make_batch(vlm, tokenizer, chunk, image_dir, mode,
                                                   args.max_text_len, device,
                                                   image_override=chunk_override)
            full_ids = vlm.full_input_ids(text_ids, pad_id)
            vlm.set_vision_embeds(pixels, text_ids)
            generated = vlm.llm.generate(input_ids=full_ids, attention_mask=mask,
                                         max_new_tokens=args.max_new_tokens, do_sample=False,
                                         pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id)
            for row, out in zip(chunk, generated):
                pred = tokenizer.decode(out[full_ids.size(1):], skip_special_tokens=True)
                correct += int(first_digit_match(pred, row["answer"]))
        return correct / max(len(rows), 1)

    if args.mode == "eval":
        args.outdir.mkdir(parents=True, exist_ok=False)
        rows = read_split(args.split)
        result = {"label": f"vlm-{args.split}", "ckpt": str(args.ckpt), "rows": len(rows),
                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        for mode in ("normal", "shuffle", "blank"):
            acc = evaluate(rows, mode)
            result[f"accuracy_{mode}"] = acc
            log(f"{args.split} · {mode}：{acc:.4f}（{len(rows)} 题）")
        (args.outdir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return

    stage = args.mode
    vlm.trainable_groups(stage)
    trainable = [p for p in vlm.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    log(f"阶段 {stage}：可训练 {trainable_params/1e6:.3f}M"
        f"（projector {sum(p.numel() for p in vlm.projector.parameters())/1e6:.3f}M，"
        f"视觉 token {vlm.image_token_len} 个）")
    rows = read_split("train")
    val_rows = read_split("val")
    target = "caption" if stage == "align" else "qa"
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.outdir / "metrics.jsonl"
    run_start, step, cursor = time.time(), 0, 0
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    best_acc, last_val, reason = -1.0, None, "max_steps"

    def save_checkpoint(name):
        vlm.save(args.outdir / name)

    while step < args.max_steps:
        if args.limit_seconds > 0 and (time.time() - run_start) > args.limit_seconds:
            reason = "time_limit"
            break
        if cursor + args.micro_bs > len(rows):
            order = rng.permutation(len(rows))
            cursor = 0
        idx = order[cursor:cursor + args.micro_bs]
        cursor += args.micro_bs
        batch = [rows[int(i)] for i in idx]
        step_start = time.time()
        pixels, text_ids, mask, labels = make_batch(vlm, tokenizer, batch, image_dir, "normal",
                                                    args.max_text_len, device, target)
        lr = get_lr(step, args.max_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        vlm.train()
        out = vlm(pixels, text_ids, mask, labels, pad_id=pad_id)
        loss = out.loss + out.aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        step += 1
        with metrics_path.open("a") as stream:
            stream.write(json.dumps({"step": step, "stage": stage, "loss": float(loss), "lr": lr,
                                     "grad_norm": float(grad_norm),
                                     "step_seconds": time.time() - step_start,
                                     "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                                     "elapsed": time.time() - run_start}) + "\n")
        if step % 50 == 0 or step == 1:
            log(f"step {step}/{args.max_steps} loss {float(loss):.4f} gnorm {float(grad_norm):.3f} "
                f"lr {lr:.2e} {time.time() - step_start:.2f}s")
        if step % args.eval_every == 0 or step == args.max_steps:
            acc = evaluate(val_rows)
            last_val = acc
            with metrics_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "eval": "val", "val_accuracy": acc,
                                         "elapsed": time.time() - run_start}) + "\n")
            log(f"  eval step {step}: val 精确匹配 {acc:.4f}")
            if acc >= best_acc:
                best_acc = acc
                save_checkpoint("best.pt")
            save_checkpoint("last.pt")
            if step % args.save_every == 0:
                save_checkpoint(f"step{step}.pt")
                snaps = sorted(args.outdir.glob("step*.pt"), key=lambda p: int(p.stem[4:]))
                for old in snaps[:-args.keep_snapshots]:
                    old.unlink()
    summary = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "stage": stage, "target": target, "steps": step, "stop_reason": reason,
               "trainable_params": trainable_params,
               "projector_params": sum(p.numel() for p in vlm.projector.parameters()),
               "llm_params": sum(p.numel() for p in vlm.llm.parameters()),
               "vision_params": sum(p.numel() for p in vlm.vision.parameters()),
               "image_token_len": vlm.image_token_len,
               "vision_path": str(args.vision_path), "best_val_accuracy": best_acc,
               "last_val_accuracy": last_val, "elapsed_seconds": time.time() - run_start,
               "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
               "data_manifest": json.loads((args.data_dir / "manifest.json").read_text())}
    (args.outdir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log(f"结束：{reason}，step {step}，best val {best_acc:.4f}")


if __name__ == "__main__":
    main()
