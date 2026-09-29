#!/usr/bin/env python3
"""SmolLM2-360M 上的一次完整训练步，以及三条累积路径的真实模型对照。

在真实 tokenizer、真实 causal LM 上重放 7.0b 的分母问题：
  P1 直接大 batch                      model(labels=...) 对 4 条一起前向
  P2 累积 + 全局分母                    逐条前向，传 num_items_in_batch=N_total
  P3 累积 + 每份自己的均值再除以 M        逐条 model(labels=...) 后 /M
三条路径从同一份初始权重各做一次 AdamW 更新，比较梯度与参数。

不写 checkpoint：保存与恢复的实测在 7.4。

Usage（crater，envs/serve）:
    python labs/L7/training_step_smollm.py --outdir "$RUN_DIR/one-step"
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE = -100
LR, WD = 1e-4, 0.01

TEXTS = [
    "The first example sentence for gradient accumulation test.",
    "The second example with different content and length.",
    "Third sentence is about machine learning and deep learning.",
    "Fourth and final sentence concludes the micro batch set.",
]


def encode(tokenizer, texts, width):
    enc = tokenizer(texts, padding="max_length", max_length=width, truncation=True,
                    return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = IGNORE
    return enc["input_ids"], enc["attention_mask"], labels


def valid_count(labels):
    """transformers v5 的对齐：右补一个 ignore 再左移，有效数 = 每条 token 数 − 1。"""
    shifted = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    return int((shifted != IGNORE).sum())


def grad_snapshot(model):
    return {name: p.grad.detach().float().cpu().clone()
            for name, p in model.named_parameters() if p.grad is not None}


def param_snapshot(model):
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()}


def max_diff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_path(model, init_state, path, ids, attn, labels, device):
    """从同一初始权重执行一次完整更新，返回 (梯度快照, 参数快照, 记录)。"""
    model.load_state_dict(init_state)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    opt.zero_grad(set_to_none=True)
    n_total = valid_count(labels)
    record = {"path": path, "n_total": n_total, "micro": []}
    if path == "P1":
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                    labels=labels.to(device))
        record["loss"] = out.loss.item()
        out.loss.backward()
    else:
        running = 0.0
        for i in range(ids.shape[0]):
            row_ids = ids[i:i + 1].to(device)
            row_attn = attn[i:i + 1].to(device)
            row_labels = labels[i:i + 1].to(device)
            n_i = valid_count(labels[i:i + 1])
            if path == "P2":
                out = model(input_ids=row_ids, attention_mask=row_attn, labels=row_labels,
                            num_items_in_batch=n_total)
                scaled = out.loss
            else:
                out = model(input_ids=row_ids, attention_mask=row_attn, labels=row_labels)
                scaled = out.loss / ids.shape[0]
            scaled.backward()
            running += scaled.item()
            record["micro"].append({"index": i, "n": n_i, "reported": out.loss.item(),
                                    "contribution": scaled.item()})
        record["loss"] = running
    record["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
    grads = grad_snapshot(model)
    opt.step()
    return grads, param_snapshot(model), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    init_state = copy.deepcopy(model.state_dict())

    print(f"model={args.model}  device={device}  dtype={next(model.parameters()).dtype}")
    print(f"transformers={transformers.__version__}  torch={torch.__version__}")
    print(f"loss 入口：LlamaForCausalLM.forward → self.loss_function → ForCausalLMLoss "
          f"→ fixed_cross_entropy（num_items_in_batch 为 None 时 reduction='mean'）")
    if device == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")

    section("1. batch 与有效 target 数")
    ids, attn, labels = encode(tokenizer, TEXTS, args.width)
    per_sample = []
    for i, text in enumerate(TEXTS):
        n_tok = int(attn[i].sum())
        n_valid = valid_count(labels[i:i + 1])
        per_sample.append({"index": i, "text": text, "tokens": n_tok, "targets": n_valid})
        print(f"  [{i}] tokens={n_tok:3d}  有效 target={n_valid:3d}  «{text}»")
    n_total = valid_count(labels)
    print(f"  padding 到 {args.width}，合计有效 target N_total={n_total}")

    section("2. 一次完整更新的逐项读数（P1）")
    model.load_state_dict(init_state)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    opt.zero_grad(set_to_none=True)
    out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                labels=labels.to(device))
    shifted = F.pad(labels, (0, 1), value=IGNORE)[..., 1:].to(device)
    per_token = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]).float(),
                                shifted.reshape(-1), ignore_index=IGNORE, reduction="none")
    keep = shifted.reshape(-1) != IGNORE
    manual = per_token[keep].sum() / n_total
    print(f"  model.loss = {out.loss.item():.8f}")
    print(f"  手算 Σ/N   = {manual.item():.8f}   差 {abs(out.loss - manual).item():.3e}")
    print(f"  逐 token loss（前 10 个有效位置）: "
          f"{[round(v, 6) for v in per_token[keep][:10].tolist()]}")
    out.loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    before = param_snapshot(model)
    opt.step()
    after = param_snapshot(model)
    print(f"  clip 前 global grad norm = {norm:.6f}（max_norm=1.0，"
          f"{'裁剪' if norm > 1 else '未裁剪'}）")
    print(f"  一次 AdamW 更新后参数最大变化 = {max_diff(before, after):.3e}")
    print(f"  有 optimizer 状态的张量 {sum(1 for p in model.parameters() if opt.state.get(p))}"
          f"/{len(list(model.parameters()))}")
    del before, after, opt, out, per_token
    if device == "cuda":
        torch.cuda.empty_cache()

    section("3. 三条累积路径")
    results, grads, params = {}, {}, {}
    for path in ("P1", "P2", "P3"):
        g, p, rec = run_path(model, init_state, path, ids, attn, labels, device)
        grads[path], params[path], results[path] = g, p, rec
        if rec["micro"]:
            print(f"\n[{path}] 逐 microbatch：")
            for m in rec["micro"]:
                print(f"    i={m['index']} N={m['n']} model.loss={m['reported']:.6f} "
                      f"贡献={m['contribution']:.6f}")
        print(f"[{path}] 累计 loss = {rec['loss']:.8f}   grad 全局范数 = {rec['grad_norm']:.6f}")
        if device == "cuda":
            torch.cuda.empty_cache()

    scale = max(float(v.abs().max()) for v in grads["P1"].values())
    print(f"\n  P1 梯度最大绝对值 = {scale:.6f}")
    print("  对照            | 梯度最大差   | 相对梯度量级 | 一次更新后参数最大差")
    for path in ("P2", "P3"):
        gd = max_diff(grads["P1"], grads[path])
        print(f"  P1 ↔ {path}         | {gd:.3e}    | {gd / scale:.3e}    |"
              f" {max_diff(params['P1'], params[path]):.3e}")
    weighted = sum(m["reported"] * m["n"] for m in results["P3"]["micro"]) / n_total
    print(f"\n  P3 的四个 model.loss 按有效数加权平均 = {weighted:.8f}，"
          f"P1 的 loss = {results['P1']['loss']:.8f}，差 {abs(weighted - results['P1']['loss']):.2e}")
    print(f"  P3 的等权平均 = {sum(m['reported'] for m in results['P3']['micro']) / 4:.8f}")
    print("  梯度是这里唯一有分辨力的量：P2 与 P1 差 1e-5 量级（fp32 归约顺序不同），"
          "P3 与 P1 差 1e-1 量级。")
    print(f"  参数差不能用来判断：m/v 从零起步时 Adam 的首步更新约等于 lr·sign(g)={LR:.0e}，"
          "\n  只要梯度符号不同，参数差就被钉在 2·lr 附近，与两条路径的真实差距无关。")

    summary = {"model": args.model, "device": device,
               "transformers": transformers.__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "width": args.width, "n_total": n_total, "samples": per_sample,
               "paths": results,
               "grad_max_diff": {p: max_diff(grads["P1"], grads[p]) for p in ("P2", "P3")},
               "param_max_diff": {p: max_diff(params["P1"], params[p]) for p in ("P2", "P3")},
               "claim_scope": "单次机制对照，不含吞吐、收敛或质量结论"}
    (args.outdir / "one_step.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\n摘要写入 {args.outdir / 'one_step.json'}")


if __name__ == "__main__":
    main()
