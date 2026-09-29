#!/usr/bin/env python3
"""留出评测（7.1-F / 7.5-I / 7.4-H 共用）：同一协议下比较随机初始化、预训练与 SFT 权重。

三组证据分开记：
  * 语言建模：预训练 val/test 窗口上的 next-token CE（说"预训练相对随机初始化"）
  * 监督任务：SFT test 分片上的 answer-token 加权 CE 与贪心生成的 exact/F1
  * 遗忘：同一个 SFT 权重在预训练 hold-out 上的 CE 相对 base 的变化

生成严格贪心、固定题号、固定长度上限，属于"确定性重复"，不是独立重采样；
因此 exact/F1 的差异只在同一批题上比较，样本量写进输出。

Usage:
    python labs/L7/teaching_eval.py --minimind-src "$SRC/minimind" \
      --checkpoint "$RUN/pretrain/best.pt" --pretrain-data-dir "$RUN/data-pretrain" \
      --outdir "$RUN/eval/pretrain" --label pretrain
    python labs/L7/teaching_eval.py --minimind-src "$SRC/minimind" \
      --checkpoint "$RUN/sft/best.pt" --sft-data-dir "$RUN/data-sft" \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/sft" --label sft
    python labs/L7/teaching_eval.py --minimind-src "$SRC/minimind" --random-init \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/random" --label random-init
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PUNCT = re.compile(r"[\s，。！？、；：\"'（）()\[\]{}<>,.!?;:·…—-]+")


def log(message):
    print(f"[teaching_eval] {message}", flush=True)


def normalize(text: str) -> str:
    return PUNCT.sub("", text).lower()


def char_f1(pred: str, ref: str) -> float:
    p, r = Counter(pred), Counter(ref)
    common = sum((p & r).values())
    if not p or not r:
        return 0.0
    precision = common / sum(p.values())
    recall = common / sum(r.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def load_model_class(minimind_src: Path):
    sys.path.insert(0, str(minimind_src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433
    return MiniMindConfig, MiniMindForCausalLM


@torch.no_grad()
def lm_ce(model, data_dir: Path, split: str, seq_len: int, micro_bs: int, device, dtype):
    tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    n_windows = int(len(tokens) // seq_len)
    windows = np.linspace(0, n_windows - 1, min(n_windows, 64)).astype(np.int64)
    total, count = 0.0, 0
    model.eval()
    for i in range(0, len(windows), micro_bs):
        chunk = windows[i:i + micro_bs]
        x = np.stack([np.asarray(tokens[s * seq_len:(s + 1) * seq_len]) for s in chunk]).astype(np.int64)
        x = torch.from_numpy(x).to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x).logits
        flat = logits[..., :-1, :].reshape(-1, logits.size(-1)).float()
        target = x[..., 1:].reshape(-1)
        total += float(F.cross_entropy(flat, target, reduction="sum"))
        count += int(target.numel())
    model.train()
    return total / max(count, 1), count


@torch.no_grad()
def sft_ce(model, npz_path: Path, micro_bs: int, device, dtype):
    data = np.load(npz_path)
    inputs, labels = data["input_ids"], data["labels"]
    total, count = 0.0, 0
    model.eval()
    for i in range(0, len(inputs), micro_bs):
        x = torch.from_numpy(inputs[i:i + micro_bs].astype(np.int64)).to(device)
        y = torch.from_numpy(labels[i:i + micro_bs].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x).logits
        flat = logits[..., :-1, :].reshape(-1, logits.size(-1)).float()
        target = y[..., 1:].reshape(-1)
        valid = target != -100
        total += float(F.cross_entropy(flat[valid], target[valid], reduction="sum"))
        count += int(valid.sum())
    model.train()
    return total / max(count, 1), count


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, eos_id, device, dtype, seq_len):
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x[:, -seq_len:]).logits[:, -1, :]
        nxt = int(torch.argmax(logits, dim=-1).item())
        if nxt == eos_id:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pth", type=Path, default=None,
                        help="裸 state_dict（上游导出格式 / 后训练分支的 best.pth）")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pretrain-data-dir", type=Path, default=None)
    parser.add_argument("--sft-data-dir", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--micro-bs", type=int, default=8)
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.checkpoint is None and args.pth is None and not args.random_init:
        raise SystemExit("需要 --checkpoint、--pth 或 --random-init")

    args.outdir.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    MiniMindConfig, MiniMindForCausalLM = load_model_class(args.minimind_src)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model", trust_remote_code=True)

    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(lm_config).to(device)
    ckpt_step = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ckpt_step = state.get("step")
    elif args.pth:
        model.load_state_dict(torch.load(args.pth, map_location=device, weights_only=False))
    model.eval()

    result = {"label": args.label,
              "checkpoint": str(args.checkpoint or args.pth) if (args.checkpoint or args.pth) else None,
              "random_init": args.random_init, "checkpoint_step": ckpt_step,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    if args.pretrain_data_dir:
        for split in ("val", "test"):
            ce, count = lm_ce(model, args.pretrain_data_dir, split, args.seq_len,
                              args.micro_bs, device, dtype)
            result[f"pretrain_{split}_ce"] = ce
            result[f"pretrain_{split}_tokens"] = count
            log(f"预训练 {split} CE {ce:.4f}（{count} targets）")

    generations = []
    if args.sft_data_dir:
        ce, count = sft_ce(model, args.sft_data_dir / "test.npz", args.micro_bs, device, dtype)
        result["sft_test_ce"] = ce
        result["sft_test_answer_tokens"] = count
        log(f"SFT test CE {ce:.4f}（{count} answer tokens）")

        data = np.load(args.sft_data_dir / "test.npz")
        inputs, labels = data["input_ids"], data["labels"]
        n = len(inputs)
        idxs = np.linspace(0, n - 1, min(n, args.n_prompts)).astype(np.int64)
        exact = f1s = contained = 0
        for idx in idxs:
            ids = inputs[idx].tolist()
            lab = labels[idx].tolist()
            sup = [i for i, value in enumerate(lab) if value != -100]
            if not sup:
                continue
            first = sup[0]
            prompt_ids = [i for i in ids[:first] if i != tokenizer.pad_token_id]
            ref_ids = [lab[i] for i in sup if lab[i] != -100]
            ref = tokenizer.decode(ref_ids)
            out_ids = generate(model, prompt_ids, args.max_new_tokens,
                               tokenizer.eos_token_id, device, dtype, args.max_len)
            pred = tokenizer.decode(out_ids)
            np_, nr = normalize(pred), normalize(ref)
            exact += int(np_ == nr and len(nr) > 0)
            contained += int(len(nr) > 0 and nr in np_)
            f1s += char_f1(np_, nr)
            generations.append({"index": int(idx), "prompt": tokenizer.decode(prompt_ids),
                                "reference": ref, "prediction": pred,
                                "exact": np_ == nr and len(nr) > 0,
                                "contained": len(nr) > 0 and nr in np_,
                                "char_f1": char_f1(np_, nr)})
        result |= {"sft_prompts": len(generations), "sft_exact_match": exact / max(len(generations), 1),
                   "sft_contained": contained / max(len(generations), 1),
                   "sft_char_f1": f1s / max(len(generations), 1)}
        log(f"SFT 生成：{len(generations)} 题，exact {exact}，"
            f"contained {contained}，char-F1 {result['sft_char_f1']:.4f}")

    (args.outdir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if generations:
        (args.outdir / "generations.jsonl").write_text(
            "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in generations))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
