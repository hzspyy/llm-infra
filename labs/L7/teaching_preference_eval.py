#!/usr/bin/env python3
"""后训练分支的独立评测（7.5-J）：偏好 log-ratio、领域子技能、通用能力与合并对拍。

三个分支共用一套口径：

  * 偏好：按上游 `DPODataset` 的同一约定构造 chosen/rejected 序列，逐 token log-prob
    只在 assistant 段求和，margin = (logπ_c − logπ_r) − (logπ_ref_c − logπ_ref_r)。
    reference 固定为 SFT 起点，所以 dpo 权重上的 margin 是"相对起点"的净变化。
  * 领域子技能：LoRA 用 held-out 的 tool-call 集，指标是生成里是否出现 `<tool_call>`
    以及其中的 JSON 能否解析——不是训练 loss。
  * 通用能力：SFT test CE（回答 token 加权）与预训练 val CE（遗忘）。
  * 合并对拍：base+adapter 与 B@A 合并进权重后的 logits 最大差。

`--merge-check` 只在 LoRA 分支有意义。

Usage:
    python labs/L7/teaching_preference_eval.py --minimind-src SRC \
      --checkpoint RUN/sft/best.pt --dpo-data-dir RUN/data-dpo \
      --sft-data-dir RUN/data-sft --pretrain-data-dir RUN/data-pretrain \
      --label dpo --outdir RUN/eval/dpo
    python labs/L7/teaching_preference_eval.py --minimind-src SRC \
      --checkpoint RUN/sft/best.pt --lora RUN/out/lora_teaching_768.pth --merge-check \
      --lora-data-dir RUN/data-lora --sft-data-dir RUN/data-sft \
      --pretrain-data-dir RUN/data-pretrain --label lora --outdir RUN/eval/lora
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teaching_eval as te  # noqa: E402


def log(message):
    print(f"[teaching_preference_eval] {message}", flush=True)


def load_base(minimind_src, checkpoint, pth, hidden_size, num_hidden_layers, device):
    config_cls, model_cls = te.load_model_class(minimind_src)
    config = config_cls(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=False)
    model = model_cls(config).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        step = state.get("step")
    else:
        state = torch.load(pth, map_location=device, weights_only=False)
        model.load_state_dict(state)
        step = None
    return model, step


def attach_lora(model, lora_path, minimind_src):
    sys.path.insert(0, str(minimind_src))
    from model.model_lora import apply_lora, load_lora  # noqa: WPS433
    before = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    apply_lora(model)
    adapted = [name for name, module in model.named_modules() if hasattr(module, "lora")]
    load_lora(model, lora_path)
    params = sum(p.numel() for name, p in model.named_parameters() if "lora" in name)
    return {"linear_modules": len(before), "adapted_modules": len(adapted),
            "adapted_names": sorted(adapted), "lora_params": params}


@torch.no_grad()
def merge_lora_inplace(model):
    """把 B@A 加进权重并摘掉旁路，返回合并的模块数。"""
    merged = 0
    for module in model.modules():
        if hasattr(module, "lora"):
            delta = module.lora.B.weight.data @ module.lora.A.weight.data
            module.weight.data += delta
            module.forward = (lambda x, m=module: F.linear(x, m.weight, m.bias))
            merged += 1
    return merged


@torch.no_grad()
def seq_logprob(model, ids, mask, device, dtype):
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    m = torch.tensor([mask[1:]], dtype=torch.float32, device=device)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        logits = model(x).logits
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float((logp * m).sum())


def normalize_messages(conversations):
    """复刻上游 SFTDataset.create_chat_prompt：解出 system 里的 tools 与字符串形式的 tool_calls。

    语料里 tool_calls 多以 JSON 字符串存储，直接交给模板会让 tool_call.function 变成
    Undefined，模板在 tojson 时抛错；这一步同时解释了"为什么评测脚本必须和训练脚本
    用同一套字段规范化"。
    """
    messages, tools = [], None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = (json.loads(message["tools"]) if isinstance(message["tools"], str)
                     else message["tools"])
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return messages, tools


def build_pair(tokenizer, conversations, max_len, bos_id, eos_id):
    """复刻上游 DPODataset：整段会话 + add_generation_prompt=False，mask 覆盖所有 assistant 段。"""
    messages, tools = normalize_messages(conversations)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False, tools=tools)
    prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    ids = tokenizer(prompt).input_ids[:max_len]
    mask = [0] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(ids) and ids[end:end + len(eos_id)] != eos_id:
                end += 1
            for j in range(start, min(end + len(eos_id), len(ids))):
                mask[j] = 1
            i = end + len(eos_id) if end < len(ids) else len(ids)
        else:
            i += 1
    return ids, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minimind-src", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pth", type=Path, default=None, help="裸 state_dict（上游导出格式）")
    parser.add_argument("--reference", type=Path, default=None,
                        help="偏好评测的 reference 权重；默认为同一个 --checkpoint")
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--merge-check", action="store_true")
    parser.add_argument("--dpo-data-dir", type=Path, default=None)
    parser.add_argument("--lora-data-dir", type=Path, default=None)
    parser.add_argument("--sft-data-dir", type=Path, default=None)
    parser.add_argument("--pretrain-data-dir", type=Path, default=None)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--micro-bs", type=int, default=8)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--n-prompts", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()
    if args.checkpoint is None and args.pth is None:
        raise SystemExit("需要 --checkpoint 或 --pth")

    args.outdir.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_src / "model", trust_remote_code=True)
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    model, step = load_base(args.minimind_src, args.checkpoint, args.pth,
                            args.hidden_size, args.num_hidden_layers, device)
    result = {"label": args.label, "checkpoint": str(args.checkpoint or args.pth),
              "checkpoint_step": step, "created_utc":
              datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    if args.lora:
        info = attach_lora(model, args.lora, args.minimind_src)
        result["lora"] = info
        log(f"LoRA：{info['adapted_modules']} 个模块（{info['lora_params']/1e6:.3f}M 参数，"
            f"占 {info['lora_params']/sum(p.numel() for p in model.parameters())*100:.2f}%）")
    model.eval()

    # ---- 通用能力 ----
    if args.pretrain_data_dir:
        ce, count = te.lm_ce(model, args.pretrain_data_dir, "val", args.seq_len,
                             args.micro_bs, device, dtype)
        result["pretrain_val_ce"] = ce
        result["pretrain_val_tokens"] = count
        log(f"预训练 val CE {ce:.4f}（{count} targets）")
    if args.sft_data_dir:
        ce, count = te.sft_ce(model, args.sft_data_dir / "test.npz", args.micro_bs, device, dtype)
        result["sft_test_ce"] = ce
        result["sft_test_answer_tokens"] = count
        log(f"SFT test CE {ce:.4f}（{count} answer tokens）")

    # ---- LoRA：领域子技能 ----
    if args.lora_data_dir:
        rows = [json.loads(line) for line in
                (args.lora_data_dir / "test.jsonl").read_text().splitlines() if line.strip()]
        idxs = list(range(len(rows)))[: args.n_prompts]
        emitted = json_ok = 0
        f1s = 0.0
        gens = []
        for idx in idxs:
            conversations = rows[idx]["conversations"]
            prefix, tools = normalize_messages(conversations[:-1])
            prompt_ids = tokenizer.apply_chat_template(prefix, tokenize=False,
                                                       add_generation_prompt=True, tools=tools)
            prompt_ids = [i for i in tokenizer(prompt_ids).input_ids if i != tokenizer.pad_token_id]
            out_ids = te.generate(model, prompt_ids, args.max_new_tokens,
                                  tokenizer.eos_token_id, device, dtype, args.max_len)
            pred = tokenizer.decode(out_ids)
            ref = conversations[-1].get("content") or ""
            has_call = "<tool_call>" in pred
            emitted += int(has_call)
            if has_call:
                body = pred.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
                try:
                    json.loads(body)
                    json_ok += 1
                except json.JSONDecodeError:
                    pass
            f1s += te.char_f1(te.normalize(pred), te.normalize(ref))
            gens.append({"index": idx, "prompt": tokenizer.decode(prompt_ids)[-300:],
                         "reference": ref[:400], "prediction": pred[:400],
                         "tool_call_emitted": has_call,
                         "char_f1": te.char_f1(te.normalize(pred), te.normalize(ref))})
        n = max(len(idxs), 1)
        result |= {"lora_prompts": len(idxs),
                   "last_turn_reference_with_tool_call": 0,
                   "tool_call_emission_rate": emitted / n,
                   "tool_call_json_parse_rate": json_ok / n, "lora_char_f1": f1s / n}
        log(f"末轮对照：{emitted}/{len(idxs)} 出现 <tool_call>，其中 {json_ok} 个 JSON 可解析，"
            f"char-F1 {f1s / n:.4f}（该 split 的末轮参考全是 content）")
        (args.outdir / "lora_generations.jsonl").write_text(
            "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gens))

        # 真正的工具调用能力：把对话切在第一个带 tool_calls 的 assistant 轮之前
        cut = []
        for idx in range(len(rows)):
            conversations = rows[idx]["conversations"]
            for pos, message in enumerate(conversations):
                if message.get("tool_calls"):
                    cut.append((idx, pos))
                    break
            if len(cut) >= args.n_prompts:
                break
        tc_emit = tc_json = tc_name = 0
        tc_gens = []
        for idx, pos in cut:
            conversations = rows[idx]["conversations"]
            prefix, tools = normalize_messages(conversations[:pos])
            prompt = tokenizer.apply_chat_template(prefix, tokenize=False,
                                                   add_generation_prompt=True, tools=tools)
            prompt_ids = [i for i in tokenizer(prompt).input_ids if i != tokenizer.pad_token_id]
            out_ids = te.generate(model, prompt_ids, args.max_new_tokens,
                                  tokenizer.eos_token_id, device, dtype, args.max_len)
            pred = tokenizer.decode(out_ids)
            wanted = []
            for call in normalize_messages([conversations[pos]])[0][0]["tool_calls"]:
                call = call.get("function", call)
                wanted.append(str(call.get("name")))
            has_call = "<tool_call>" in pred
            tc_emit += int(has_call)
            parsed_name = None
            if has_call:
                body = pred.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
                try:
                    parsed_name = json.loads(body)["name"]
                    tc_json += 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            tc_name += int(parsed_name in wanted) if parsed_name else 0
            tc_gens.append({"index": idx, "cut_at": pos,
                            "prompt": tokenizer.decode(prompt_ids)[-300:],
                            "reference_tool_calls": wanted, "prediction": pred[:400],
                            "tool_call_emitted": has_call, "parsed_name": parsed_name})
        m = max(len(cut), 1)
        result |= {"toolcall_prompts": len(cut), "toolcall_emission_rate": tc_emit / m,
                   "toolcall_json_parse_rate": tc_json / m, "toolcall_name_match_rate": tc_name / m}
        log(f"工具调用切点：{len(cut)} 题（切在首个含 tool_calls 的 assistant 轮之前），"
            f"出现 <tool_call> {tc_emit}、JSON 可解析 {tc_json}、函数名命中 {tc_name}")
        (args.outdir / "toolcall_generations.jsonl").write_text(
            "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in tc_gens))

    # ---- DPO：偏好 log-ratio ----
    if args.dpo_data_dir:
        ref_path = args.reference if args.reference else args.checkpoint
        ref_model, _ = load_base(args.minimind_src, ref_path, None,
                                 args.hidden_size, args.num_hidden_layers, device)
        ref_model.eval()
        rows = [json.loads(line) for line in
                (args.dpo_data_dir / "test.jsonl").read_text().splitlines() if line.strip()]
        idxs = list(range(len(rows)))[: args.n_pairs]
        margins, policy_ratios, ref_ratios, records = [], [], [], []
        for idx in idxs:
            pair = rows[idx]
            c_ids, c_mask = build_pair(tokenizer, pair["chosen"], args.max_len, bos_id, eos_id)
            r_ids, r_mask = build_pair(tokenizer, pair["rejected"], args.max_len, bos_id, eos_id)
            lp_c = seq_logprob(model, c_ids, c_mask, device, dtype)
            lp_r = seq_logprob(model, r_ids, r_mask, device, dtype)
            lr_c = seq_logprob(ref_model, c_ids, c_mask, device, dtype)
            lr_r = seq_logprob(ref_model, r_ids, r_mask, device, dtype)
            policy_ratio = lp_c - lp_r
            ref_ratio = lr_c - lr_r
            margin = policy_ratio - ref_ratio
            margins.append(margin)
            policy_ratios.append(policy_ratio)
            ref_ratios.append(ref_ratio)
            records.append({"index": idx, "policy_logratio": policy_ratio,
                            "reference_logratio": ref_ratio, "margin": margin,
                            "margin_positive": margin > 0})
        n = max(len(margins), 1)
        result |= {
            "dpo_pairs": len(margins),
            "policy_logratio_mean": sum(policy_ratios) / n,
            "reference_logratio_mean": sum(ref_ratios) / n,
            "margin_mean": sum(margins) / n,
            "margin_positive_rate": sum(1 for m in margins if m > 0) / n,
            "margin_min": min(margins) if margins else None,
            "margin_max": max(margins) if margins else None,
        }
        log(f"偏好 margin 均值 {result['margin_mean']:.4f}，正 margin 比例 "
            f"{result['margin_positive_rate']:.3f}（{len(margins)} 对）")
        (args.outdir / "dpo_pairs.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    # ---- 合并对拍 ----
    if args.lora and args.merge_check:
        sample = next((json.loads(line) for line in
                       (args.lora_data_dir / "test.jsonl").read_text().splitlines() if line.strip()), None)
        prefix, tools = normalize_messages(sample["conversations"][:-1])
        prompt_ids = tokenizer.apply_chat_template(prefix, tokenize=False,
                                                   add_generation_prompt=True, tools=tools)
        prompt_ids = [i for i in tokenizer(prompt_ids).input_ids if i != tokenizer.pad_token_id][:args.max_len]
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            unmerged = model(x).logits.float()
        merged_count = merge_lora_inplace(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            merged = model(x).logits.float()
        result["merge_parity"] = {
            "modules_merged": merged_count,
            "max_abs_logit_diff": float((unmerged - merged).abs().max()),
            "mean_abs_logit_diff": float((unmerged - merged).abs().mean()),
        }
        log(f"合并对拍：{merged_count} 个模块，logits 最大差 "
            f"{result['merge_parity']['max_abs_logit_diff']:.3e}")

    (args.outdir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
