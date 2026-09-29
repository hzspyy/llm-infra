#!/usr/bin/env python3
"""0.4-B: 模板、重复特殊 token、padding 与 position_ids 对 logits 的实际影响。

同一段有效文本，换三种模板（plain / chat / chat+tools）、换两种 padding、
再把特殊 token 重复加一遍，逐项核对 input_ids、attention_mask、position_ids
与最后一个有效位置的 logits。

批内数值本身就有抖动（见 4.2 batch 不变性），所以先测噪声地板：
同一条 prompt 单独跑 vs 放进等长 batch 跑，两者的 logits 最大差值。
padding 造成的差值只有明显超过这个地板才算真实影响。

用法：
    python labs/L0/template_padding_probe.py --out-dir <dir> [--model Qwen/Qwen3-1.7B]
需要 GPU 与已缓存的模型。
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

SEP = "-" * 78
SEED = 0

USER_A = "风是什么？"
USER_B = "用一句话说明 KV cache 为什么能省算力，并举一个例子。"
SYSTEM = "你是一个助手。"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询某地天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def esc(s: str) -> str:
    return s.replace("\n", "\\n")


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.float(), b.float()
    return dict(
        max_abs=float((a32 - b32).abs().max()),
        argmax_a=int(a32.argmax()), argmax_b=int(b32.argmax()),
        argmax_same=bool(a32.argmax() == b32.argmax()),
        top5_a=[int(i) for i in a32.topk(5).indices],
        top5_b=[int(i) for i in b32.topk(5).indices],
    )


# ------------------------------------------------------------- [1] 三种模板

def section_templates(tok) -> dict:
    print(f"[1] 同一条消息，三种模板\n{SEP}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_A}]
    plain = USER_A
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    chat_no_gen = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    tool = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)

    rows = {}
    for name, text in [("plain", plain), ("chat", chat),
                       ("chat_no_gen", chat_no_gen), ("chat_tools", tool)]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        rows[name] = dict(chars=len(text), bytes=len(text.encode("utf-8")),
                          n_tokens=len(ids), ids=ids, text=text)
        print(f"    {name:<13s}{len(text):>6d} 字符 {len(text.encode('utf-8')):>6d} 字节"
              f"{len(ids):>6d} token")

    print(f"\n    有效文本只有 {len(USER_A)} 个字符；chat 模板把它变成 "
          f"{rows['chat']['n_tokens']} 个 token。")
    print("    chat 比 plain 多出来的 token（逐个）：")
    plain_ids = rows["plain"]["ids"]
    chat_ids = rows["chat"]["ids"]
    i = chat_ids.index(plain_ids[0]) if plain_ids[0] in chat_ids else 0
    head, tail = chat_ids[:i], chat_ids[i + len(plain_ids):]
    for pos, tid in enumerate(head):
        print(f"      前缀 {pos:>2d}  id={tid:<8d}{esc(tok.decode([tid]))!r}")
    for pos, tid in enumerate(tail):
        print(f"      后缀 {pos:>2d}  id={tid:<8d}{esc(tok.decode([tid]))!r}")
    gen_delta = rows["chat"]["n_tokens"] - rows["chat_no_gen"]["n_tokens"]
    print(f"\n    add_generation_prompt 贡献 {gen_delta} 个 token："
          f"{esc(chat[len(chat_no_gen):])!r}")
    tool_delta = rows["chat_tools"]["n_tokens"] - rows["chat"]["n_tokens"]
    print(f"    tools 把 {len(TOOLS)} 个函数的 JSON schema 拼进 system 消息，"
          f"多 {tool_delta} 个 token。")
    print(f"    tools 段原文前 220 字符：{esc(tool[:220])!r}")
    rows["deltas"] = dict(generation_prompt_tokens=gen_delta, tools_tokens=tool_delta,
                          chat_minus_plain=rows["chat"]["n_tokens"] - rows["plain"]["n_tokens"],
                          prefix_ids=head, suffix_ids=tail)
    return rows


# ----------------------------------------------------- [2] 重复加特殊 token

def section_double_special(tok, model_name: str) -> dict:
    print(f"\n[2] 特殊 token 被加了两次会发生什么  （{model_name}）\n{SEP}")
    has_template = getattr(tok, "chat_template", None) is not None
    print(f"    add_bos_token={getattr(tok, 'add_bos_token', None)}  "
          f"bos={tok.bos_token!r}  eos={tok.eos_token!r}  chat_template={has_template}")
    out = dict(model=model_name, has_chat_template=has_template)

    # 纯文本路径：add_special_tokens 会不会自己补 BOS/EOS
    plain = tok(USER_A, add_special_tokens=True)["input_ids"]
    plain_no = tok(USER_A, add_special_tokens=False)["input_ids"]
    out.update(plain_add_special=plain, plain_no_special=plain_no)
    print(f"    tok(text, add_special_tokens=True)   {len(plain):>3d} token  {plain}")
    print(f"    tok(text, add_special_tokens=False)  {len(plain_no):>3d} token  {plain_no}")
    if tok.bos_token_id is not None:
        manual = [tok.bos_token_id] + plain
        out["manual_double_bos"] = manual
        print(f"    先手工补 BOS 再让 tokenizer 补一次：{manual[:4]} …  "
              f"开头出现两个 {tok.bos_token!r}")

    if has_template:
        msgs = [{"role": "user", "content": USER_A}]
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        once = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True)["input_ids"]
        twice = tok(rendered, add_special_tokens=True)["input_ids"]
        no_special = tok(rendered, add_special_tokens=False)["input_ids"]
        out.update(rendered=rendered, apply_chat_template_ids=list(once),
                   retokenize_add_special=twice, retokenize_no_special=no_special,
                   template_path_same=list(once) == twice)
        print(f"\n    apply_chat_template(tokenize=True)        {len(once)} token")
        print(f"    先 render 再 tok(add_special_tokens=True)  {len(twice)} token"
              f"  {'一致' if out['template_path_same'] else '不一致'}")
        print(f"    先 render 再 tok(add_special_tokens=False) {len(no_special)} token")
        eos = tok.eos_token
        dup = tok(rendered + eos + eos, add_special_tokens=False)["input_ids"]
        out["dup_eos_tail"] = dup[-6:]
        print(f"\n    手工在末尾重复两个 {eos!r}：尾部 id 变成 {dup[-6:]}")
        print(f"    解码：{esc(tok.decode(dup[-6:]))!r}")
        print("    这串 id 在训练语料里不存在，属于分布外输入；不会报错，只会静默变差。")
    return out


# -------------------------------------------- [3] padding / mask / position

def build_batch(tok, texts: list[str], side: str) -> dict:
    tok.padding_side = side
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    mask = enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)     # 正确写法：按 mask 累加
    naive = torch.arange(mask.shape[1]).unsqueeze(0).expand_as(mask)  # 错误写法
    return dict(input_ids=enc["input_ids"], attention_mask=mask,
                position_ids=pos, naive_position_ids=naive)


def section_padding(tok, texts: list[str]) -> dict:
    print(f"\n[3] 左/右 padding 的三个张量\n{SEP}")
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    print(f"    两条有效长度：{lens}，pad 到 {max(lens)}；pad token = "
          f"{tok.pad_token!r} (id {tok.pad_token_id})")
    out = {}
    for side in ("left", "right"):
        b = build_batch(tok, texts, side)
        ids, mask, pos = b["input_ids"], b["attention_mask"], b["position_ids"]
        out[side] = {k: v.tolist() for k, v in b.items()}
        print(f"\n    padding_side = {side}   样本 0（短的那条，有效 {lens[0]}）")
        print(f"      input_ids      {ids[0].tolist()}")
        print(f"      attention_mask {mask[0].tolist()}")
        print(f"      position_ids   {pos[0].tolist()}")
        print(f"      naive arange   {b['naive_position_ids'][0].tolist()}")
    out["lens"] = lens
    print("\n    左 padding 下最后一列一定是有效 token，右 padding 下不是。")
    return out


# ------------------------------------------------------------- [4] logits

@torch.no_grad()
def last_logits(model, ids, mask, pos, index):
    o = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
              position_ids=pos.to(model.device))
    return o.logits[torch.arange(ids.shape[0]), index]


def section_logits(tok, model, texts: list[str]) -> dict:
    print(f"\n[4] 同一条有效文本，padding 方式不同，最后一个有效位置的 logits\n{SEP}")
    res = {}
    single = []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        single.append(last_logits(model, ids, mask, pos, torch.tensor([ids.shape[1] - 1]))[0].cpu())

    # 噪声地板：同一条 prompt 单独跑 vs 放进等长 batch（无 padding）
    enc = tok([texts[0], texts[0]], return_tensors="pt", add_special_tokens=False)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    dup = last_logits(model, ids, mask, pos,
                      torch.tensor([ids.shape[1] - 1] * 2)).cpu()
    floor = diff_stats(single[0], dup[0])
    res["batch_noise_floor"] = floor
    print(f"    噪声地板（等长 batch，无 padding）：max|Δ| = {floor['max_abs']:.3e}，"
          f"argmax {'相同' if floor['argmax_same'] else '不同'}")

    cases = {}
    for side in ("left", "right"):
        b = build_batch(tok, texts, side)
        ids, mask, pos = b["input_ids"], b["attention_mask"], b["position_ids"]
        last_valid = mask.shape[1] - 1 if side == "left" else mask.sum(-1) - 1
        idx = torch.full((ids.shape[0],), last_valid) if side == "left" else last_valid
        got = last_logits(model, ids, mask, pos, idx).cpu()
        cases[f"{side}_correct"] = [diff_stats(single[i], got[i]) for i in range(len(texts))]
        # 错误写法一：无视 mask 直接取最后一列
        tail = last_logits(model, ids, mask, pos,
                          torch.full((ids.shape[0],), ids.shape[1] - 1)).cpu()
        cases[f"{side}_take_last_column"] = [diff_stats(single[i], tail[i]) for i in range(len(texts))]
        # 错误写法二：position_ids 用 arange
        naive = last_logits(model, ids, mask, b["naive_position_ids"], idx).cpu()
        cases[f"{side}_naive_position"] = [diff_stats(single[i], naive[i]) for i in range(len(texts))]

    print(f"\n    {'口径':<26s}{'样本':>4s}{'max|Δ| vs 单条':>16s}{'argmax':>8s}  top1 解码")
    for key, stats in cases.items():
        for i, s in enumerate(stats):
            print(f"    {key:<26s}{i:>4d}{s['max_abs']:>16.3e}"
                  f"{'同' if s['argmax_same'] else '异':>8s}  "
                  f"{tok.decode([s['argmax_b']])!r}")
    res["cases"] = cases
    res["reference_top1"] = [tok.decode([int(x.float().argmax())]) for x in single]
    res["tolerance"] = dict(
        rule="max|Δlogits| 与噪声地板同量级即视为一致",
        floor_max_abs=floor["max_abs"],
        basis="同一 prompt 单独跑 vs 等长无 padding batch 跑，同 dtype 下的批内重排差异")
    ratio = {k: [round(s["max_abs"] / floor["max_abs"], 2) for s in v] for k, v in cases.items()}
    res["ratio_to_floor"] = ratio
    print(f"\n    与噪声地板的倍数（≈1 表示看不出 padding 的影响）：")
    for k, v in ratio.items():
        print(f"      {k:<26s}{v}")
    print(f"    单条参照的 top1：{res['reference_top1']}")
    return res


@torch.no_grad()
def section_generate(tok, model, texts: list[str]) -> dict:
    print(f"\n[5] 两种 padding 各贪心生成 16 个 token\n{SEP}")
    out = {}
    for side in ("left", "right"):
        tok.padding_side = side
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        gen = model.generate(input_ids=enc["input_ids"].to(model.device),
                             attention_mask=enc["attention_mask"].to(model.device),
                             max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts_out = [tok.decode(row, skip_special_tokens=False) for row in new]
        out[side] = dict(new_ids=new.tolist(), text=texts_out)
        for i, t in enumerate(texts_out):
            print(f"    {side:<6s}样本 {i}  {esc(t)!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bos-model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"model={args.model}  transformers={transformers.__version__}  "
          f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}\n")

    templates = section_templates(tok)
    double = section_double_special(tok, args.model)
    try:
        tok2 = AutoTokenizer.from_pretrained(args.bos_model)
        double_bos = section_double_special(tok2, args.bos_model)
    except Exception as e:  # 缓存里没有这个模型时跳过，不伪造结果
        double_bos = dict(error=f"{type(e).__name__}: {e}")
        print(f"    [skip] {args.bos_model}: {e}")

    texts = [tok.apply_chat_template([{"role": "user", "content": u}],
                                     add_generation_prompt=True, tokenize=False)
             for u in (USER_A, USER_B)]
    padding = section_padding(tok, texts)

    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    model.eval()
    logits = section_logits(tok, model, texts)
    gen = section_generate(tok, model, texts)

    payload = dict(templates=templates, double_special=double, double_special_bos=double_bos,
                   padding=padding, logits=logits, generate=gen)
    (out / "template_padding.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        task="0.4-B", model=args.model, bos_model=args.bos_model,
        dtype=args.dtype, device=torch.cuda.get_device_name(0),
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        transformers=transformers.__version__, torch=torch.__version__,
        user_texts=[USER_A, USER_B], system=SYSTEM, tools=TOOLS,
        decoding="greedy (do_sample=False)", tolerance=logits["tolerance"],
        host=platform.node(), outputs=["template_padding.json", "stdout.txt"],
    ), ensure_ascii=False, indent=1) + "\n")
    print(f"\n工件写入 {out}")


if __name__ == "__main__":
    main()
