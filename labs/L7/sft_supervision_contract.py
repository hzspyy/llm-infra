#!/usr/bin/env python3
"""SFT 的监督契约：从真实会话到逐 token 的 loss mask。

用 SmolLM3 后训练配方里点名的那几个 smoltalk2 split，各取几条真实样本：
  A 数据形态   角色序列、thinking 开关、工具调用、长度分布
  B 模板与掩码 apply_chat_template → 逐段定位 assistant → 只对回答计 loss
  C 数值对拍   手写逐 token CE 与 model(labels=...) 的 loss 对齐
  D 反例       截断吃掉答案、pad=EOS 被屏蔽、packing 不隔离文档

样本通过 datasets-server 的 rows 接口按需取，只下几十条，不拉整个数据集。

Usage（crater，envs/serve）:
    python labs/L7/sft_supervision_contract.py --outdir "$RUN_DIR/sft"
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "HuggingFaceTB/smoltalk2"
SFT_SPLITS = [
    ("SFT", "smoltalk_smollm3_everyday_conversations_no_think"),
    ("SFT", "hermes_function_calling_v1_no_think"),
    ("SFT", "smoltalk_smollm3_smol_magpie_ultra_no_think"),
    ("SFT", "OpenThoughts3_1.2M_think"),
]
PREF_SPLIT = ("Preference", "llama_3.1_tulu_3_8b_preference_mixture_no_think")
MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
# chat template 属于 instruct 侧的 tokenizer，base 仓库里没有它
TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
IGNORE = -100


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fetch_rows(config: str, split: str, n: int) -> list[dict]:
    url = ("https://datasets-server.huggingface.co/rows?"
           + urllib.parse.urlencode({"dataset": DATASET, "config": config,
                                     "split": split, "offset": 0, "length": n}))
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    return [r["row"] for r in data["rows"]]


# ------------------------------------------------------------------ A 数据形态
def section_a(tok, samples: dict) -> dict:
    head("A 真实样本的形态：角色、thinking 开关、工具调用与长度")
    summary = {}
    print(f"{'split':<44}{'条数':>5}{'角色序列':<34}{'thinking':<10}{'token 数':>10}")
    for (config, split), rows in samples.items():
        roles = []
        lens = []
        thinking = set()
        for row in rows:
            msgs = row.get("messages") or []
            roles.append("→".join(m["role"][0] for m in msgs))
            kw = row.get("chat_template_kwargs") or {}
            thinking.add(bool(kw.get("enable_thinking", False)))
            text = tok.apply_chat_template(msgs, tokenize=False, **kw)
            lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
        print(f"{split[:42]:<44}{len(rows):>5}{roles[0][:32]:<34}"
              f"{str(sorted(thinking)):<10}{min(lens):>5}-{max(lens):<5}")
        summary[split] = {"roles": roles, "token_lengths": lens,
                          "thinking": sorted(thinking)}
    tool_rows = samples[("SFT", "hermes_function_calling_v1_no_think")]
    tool_msg = next((m for row in tool_rows for m in row["messages"]
                     if m["role"] not in ("user", "assistant", "system")), None)
    print(f"\n  工具调用样本里出现的非 user/assistant 角色："
          f"{tool_msg['role'] if tool_msg else '无'}")
    if tool_msg:
        print(f"  它的内容前 120 字：{tool_msg['content'][:120]!r}")
    print("  这类消息是模型的输入而不是要学的输出，loss mask 必须把它排除。")
    sys_rows = [m for row in samples[("SFT", "smoltalk_smollm3_everyday_conversations_no_think")]
                for m in row["messages"] if m["role"] == "system"]
    print(f"  everyday_conversations 里的 system 消息数：{len(sys_rows)}；"
          f"系统提示由 chat_template_kwargs 的 custom_instructions 注入")
    return summary


# ------------------------------------------------------------------ B 掩码
def build_masked(tok, messages: list[dict], kwargs: dict) -> dict:
    """逐段渲染，定位每条 assistant 消息在 token 序列里的区间。"""
    ids_full = tok(tok.apply_chat_template(messages, tokenize=False, **kwargs),
                   add_special_tokens=False)["input_ids"]
    labels = [IGNORE] * len(ids_full)
    spans = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tok(tok.apply_chat_template(messages[:i], tokenize=False,
                                             add_generation_prompt=True, **kwargs),
                     add_special_tokens=False)["input_ids"]
        upto = tok(tok.apply_chat_template(messages[:i + 1], tokenize=False, **kwargs),
                   add_special_tokens=False)["input_ids"]
        start, end = len(prefix), len(upto)
        if ids_full[:start] != prefix[:start]:
            start = next((j for j in range(len(prefix), -1, -1)
                          if ids_full[:j] == prefix[:j]), start)
        for j in range(start, min(end, len(ids_full))):
            labels[j] = ids_full[j]
        spans.append((start, min(end, len(ids_full))))
    return {"input_ids": ids_full, "labels": labels, "spans": spans}


def section_b(tok, sample) -> dict:
    head("B 从模板到 loss mask：只有回答参与 loss")
    msgs = sample["messages"]
    kw = sample.get("chat_template_kwargs") or {}
    rendered = tok.apply_chat_template(msgs, tokenize=False, **kw)
    print(f"  渲染后的前 240 字：\n    {rendered[:240]!r}")
    built = build_masked(tok, msgs, kw)
    ids, labels, spans = built["input_ids"], built["labels"], built["spans"]
    valid = sum(1 for v in labels if v != IGNORE)
    print(f"\n  总 token {len(ids)}，assistant 区间 {spans}，"
          f"计入 loss 的 token {valid}（{valid / len(ids):.1%}）")
    first = spans[0]
    print(f"  第一个 assistant 区间解码：{tok.decode(ids[first[0]:first[1]])[:160]!r}")
    print(f"  它前面一个 token 是：{tok.decode([ids[first[0] - 1]])!r}"
          f"（生成提示的结尾，属于输入而不是目标）")
    print("\n  三类角色的处理：system/user/tool 的内容进输入、不进 loss；")
    print("  assistant 的内容既进输入（作为后续轮次的上下文）也进 loss。")
    return built


# ------------------------------------------------------------------ C 数值对拍
def section_c(model, built: dict, device: str) -> dict:
    head("C 手写逐 token CE 与 model(labels=...) 对拍")
    ids = torch.tensor([built["input_ids"]], device=device)
    labels = torch.tensor([built["labels"]], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
        logits = out.logits.float()
    shifted = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    mask = shifted != IGNORE
    manual_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 shifted.reshape(-1), ignore_index=IGNORE,
                                 reduction="sum")
    n_valid = int(mask.sum())
    manual_mean = float(manual_sum / n_valid)
    hf = float(out.loss)
    print(f"  有效 target {n_valid}；手写 sum/N = {manual_mean:.8f}；"
          f"model.loss = {hf:.8f}；差 {abs(manual_mean - hf):.3e}")
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                shifted.reshape(-1), ignore_index=IGNORE,
                                reduction="none").reshape(shifted.shape)
    top = torch.topk(per_token[mask], k=min(5, n_valid))
    idx = mask.nonzero()[top.indices[:, ] if top.indices.dim() > 1 else top.indices]
    print(f"  逐 token loss 的最大 5 个值：{[round(float(v), 3) for v in top.values]}")
    print("  对齐规则：labels 右补一个 ignore 再左移，第 t 个 logit 预测第 t+1 个 token；")
    print("  因此每条 assistant 区间的第一个 token 由生成提示的最后一个位置预测。")
    return {"n_valid": n_valid, "manual": manual_mean, "hf": hf}


# ------------------------------------------------------------------ D 反例
def section_d(tok, model, built: dict, device: str) -> dict:
    head("D 三个会静默改变监督范围的反例")
    ids, labels = built["input_ids"], built["labels"]
    rows = []

    cut = built["spans"][0][0] + 2                       # 截在第一个答案刚开头
    trunc_labels = labels[:cut]
    valid = sum(1 for v in trunc_labels if v != IGNORE)
    rows.append(("max_length 截在答案开头", f"{cut} token", valid,
                 "答案几乎全被截掉，这条样本只贡献 2 个 target"))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id
    eos_in_labels = sum(1 for v in labels if v == eos_id)
    masked_eos = [IGNORE if v == pad_id else v for v in labels]
    valid_masked = sum(1 for v in masked_eos if v != IGNORE)
    rows.append((f"pad_token_id={pad_id} 与 eos 相同时按 pad 屏蔽",
                 f"原有 {sum(1 for v in labels if v != IGNORE)} 个 target",
                 valid_masked,
                 f"{eos_in_labels} 个 EOS 目标被一起屏蔽，模型学不到停止"))

    # 让拼接点落在两段 assistant 之间：前一段以答案结尾，后一段以答案开头
    s0, e0 = built["spans"][0]
    s1, e1 = built["spans"][1]
    a_ids, a_labels = ids[s0:e0], labels[s0:e0]
    b_ids, b_labels = ids[s1:e1], labels[s1:e1]
    packed_labels = a_labels + b_labels
    boundary = len(a_labels)
    cross = packed_labels[boundary] != IGNORE
    rows.append(("packing 后不重置文档边界", f"拼接位置 {boundary}",
                 sum(1 for v in packed_labels if v != IGNORE),
                 f"位置 {boundary} 的 target "
                 f"{'仍然计入' if cross else '是 ignore'}，"
                 f"由上一条样本的最后一个位置预测"))

    print(f"  {'反例':<34}{'设置':<24}{'有效 target':>12}  后果")
    for name, setting, valid_n, effect in rows:
        print(f"  {name:<34}{setting:<24}{valid_n:>12}  {effect}")
    print("\n  三个反例的共同点：loss 依然算得出来，曲线也不会报警，")
    print("  只有把有效 target 数和它们的位置打印出来才看得见。")
    print("  packing 还要同时处理 attention mask 与 position_ids，见 7.8。")
    return {"cases": [{"name": n, "setting": s, "valid": v, "effect": e}
                      for n, s, v, e in rows]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--outdir")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer={args.tokenizer} vocab={tok.vocab_size} "
          f"eos={tok.eos_token_id} pad={tok.pad_token_id}")

    samples = {}
    for config, split in SFT_SPLITS:
        samples[(config, split)] = fetch_rows(config, split, args.rows)
    pref = fetch_rows(*PREF_SPLIT, args.rows)
    print(f"共取 {sum(len(v) for v in samples.values())} 条会话 + {len(pref)} 条偏好对")

    report = {"model": args.model, "splits": {}}
    report["splits"] = section_a(tok, samples)

    head("A' 偏好对的结构")
    row = pref[0]
    print(f"  字段：{sorted(row)}")
    for key in ("chosen", "rejected"):
        msgs = row[key]
        print(f"  {key}：{len(msgs)} 条消息，角色 "
              f"{'→'.join(m['role'][0] for m in msgs)}，"
              f"最后一条长度 {len(msgs[-1]['content'])} 字")
    same_prefix = row["chosen"][:-1] == row["rejected"][:-1]
    print(f"  chosen 与 rejected 的前缀完全相同：{same_prefix}")
    print("  DPO 要求两条序列共享同一个 prompt；前缀不同就等于比较了两个不同的问题。")
    report["preference"] = {"fields": sorted(row), "shared_prefix": same_prefix}

    sample = samples[("SFT", "smoltalk_smollm3_everyday_conversations_no_think")][0]
    built = section_b(tok, sample)
    report["mask"] = {"n_tokens": len(built["input_ids"]), "spans": built["spans"],
                      "n_valid": sum(1 for v in built["labels"] if v != IGNORE)}

    print(f"\n加载 {args.model} 做数值对拍（bf16，只前向）……")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()
    report["loss_check"] = section_c(model, built, device)
    report["counterexamples"] = section_d(tok, model, built, device)

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "sft_supervision.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "samples.json").write_text(
            json.dumps({f"{c}/{s}": r for (c, s), r in samples.items()}
                       | {"preference": pref}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n结果与固定样本写入 {out}")


if __name__ == "__main__":
    main()
