#!/usr/bin/env python3
"""L5.10 任务 D —— SGLang 侧挂真实 LoRA：与 vLLM 侧同一套判据。

vLLM 侧已经量过 base vs adapter（§H：每轮 4/4 不同、首个分岔位置三轮恒定的
[9,4,7,28]）。这一轮在 SGLang 上做同一件事，好让"两引擎对齐完整路径"这句
有两侧的数据：

  * 同一个 Qwen3-1.7B、同一批 prompt、同一份 rank-8 adapter（q_proj + v_proj）；
  * 三轮**交错**重复 base / adapter，因为上一轮已经证明这台机器上
    "同一配置重复运行之间也会分岔"，单次对比没有意义；
  * 判据是首 token 的 top-k logprob 差（只在两边 token 相同的上下文里比）
    与分岔位置，不是"文本是否相同"。

SGLang 的 LoRA 走 `--enable-lora --lora-paths pub=<dir>`，请求里用
`sampling_params.lora_path="pub"`；它会把 `[q_proj, v_proj]` 归一成
`qkv_proj` 再送进分段 kernel（`srt/lora/utils.py`）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import requests

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France, and why is it famous?",
    "Summarize the theory of relativity in two sentences.",
]


def make_adapter(root: pathlib.Path, model: str, rank: int = 8, seed: int = 4242):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, local_files_only=True)
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
    q_out = cfg.num_attention_heads * head_dim
    kv_out = cfg.num_key_value_heads * head_dim
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": model, "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": rank, "lora_alpha": rank, "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0, "bias": "none", "use_dora": False,
    }, indent=2))
    gen = torch.Generator().manual_seed(seed)
    w = {}
    for layer in range(cfg.num_hidden_layers):
        pre = f"base_model.model.model.layers.{layer}.self_attn"
        for mod, out in (("q_proj", q_out), ("v_proj", kv_out)):
            w[f"{pre}.{mod}.lora_A.weight"] = torch.randn(rank, hidden, generator=gen,
                                                          dtype=torch.bfloat16)
            w[f"{pre}.{mod}.lora_B.weight"] = torch.randn(out, rank, generator=gen,
                                                          dtype=torch.bfloat16)
    save_file(w, root / "adapter_model.safetensors")
    return root


def ask(base, ids, lora, max_new_tokens=24):
    # lora_path 是 /generate 的**顶层**字段（GenerateReqInput.lora_path），
    # 放进 sampling_params 会 500：TypeError: Unexpected keyword argument 'lora_path'
    payload = {"input_ids": ids,
               "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
               "return_logprob": True, "logprob_start_len": 0,
               # 不显式要 top-k 时只回被选中 token 的 logprob，两边没法逐项比
               "top_logprobs_num": 5}
    if lora:
        payload["lora_path"] = lora
    t0 = time.perf_counter()
    r = requests.post(base + "/generate", json=payload, timeout=300)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dict(error=f"HTTP {r.status_code}: {r.text[:160]}", wall_s=dt)
    b = r.json()
    meta = b.get("meta_info", {})
    olp = meta.get("output_token_logprobs") or []
    first = {}
    if olp:
        # 首 token 位置的 top-k 不给，只能拿到被选中 token 的 logprob；
        # 用 return_logprob 的 top_logprobs 字段（SGLang 在 meta_info 里给）
        first = {int(x[1]): float(x[0]) for x in olp[:1]}
    top = meta.get("output_top_logprobs") or []
    topk = {}
    if top:
        for item in top[0]:
            # SGLang 这里给的是 (logprob, token_id) 或 (logprob, token_id, text)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                topk[int(item[1])] = float(item[0])
            elif isinstance(item, dict):
                tid = item.get("token_id") or item.get("id")
                if tid is not None:
                    topk[int(tid)] = float(item.get("logprob", 0.0))
    return dict(wall_s=round(dt, 3), token_ids=olp and [int(x[1]) for x in olp] or [],
                chosen_first_logprob=first,
                first_topk=topk,
                finish_reason=str(meta.get("finish_reason")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--adapter-root", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--lora-name", default="pub")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    make_adapter(args.adapter_root, args.model)
    ids_list = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]

    rounds = []
    for r in range(args.rounds):
        row = {}
        for key, lora in (("base", None), ("adapter", args.lora_name)):
            row[key] = [ask(args.base, ids, lora) for ids in ids_list]
        rounds.append(row)
        print(f"  round {r}: base 首 token "
              f"{[x.get('token_ids', [None])[:1] for x in row['base']]}  "
              f"adapter 首 token "
              f"{[x.get('token_ids', [None])[:1] for x in row['adapter']]}", flush=True)

    def cmp(a, b):
        """同一 prompt 下 base 与 adapter 的差异（只看两边都有的 top-k）。"""
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x.get("error") or y.get("error"):
                out.append(dict(prompt=i, error=x.get("error") or y.get("error")))
                continue
            tx, ty = x.get("token_ids", []), y.get("token_ids", [])
            first_diff = next((j for j, (m, n) in enumerate(zip(tx, ty)) if m != n),
                              None)
            dx, dy = x.get("first_topk") or {}, y.get("first_topk") or {}
            common = sorted(set(dx) & set(dy))
            maxdiff = (round(max(abs(dx[t] - dy[t]) for t in common), 5)
                       if common else None)
            out.append(dict(prompt=i, len_x=len(tx), len_y=len(ty),
                            first_diff=first_diff,
                            topk_common=len(common),
                            max_abs_logprob_diff=maxdiff,
                            argmax_same=(max(dx, key=dx.get) == max(dy, key=dy.get)
                                         if dx and dy else None)))
        return out

    report = dict(label=args.label, model=args.model, lora_name=args.lora_name,
                  prompts=PROMPTS, rounds=rounds,
                  per_round_base_vs_adapter=[cmp(r["base"], r["adapter"])
                                             for r in rounds],
                  per_round_base_vs_base=[cmp(rounds[i]["base"], rounds[j]["base"])
                                          for i, j in [(0, 1), (0, 2), (1, 2)]
                                          if len(rounds) > j],
                  latency_median_s=dict(
                      base=round(statistics.median(
                          x["wall_s"] for r in rounds for x in r["base"] if "wall_s" in x), 3),
                      adapter=round(statistics.median(
                          x["wall_s"] for r in rounds for x in r["adapter"] if "wall_s" in x), 3)))
    (args.out / f"sglang_lora_{args.label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for r, cells in enumerate(report["per_round_base_vs_adapter"]):
        n_diff = sum(1 for c in cells if c.get("first_diff") is not None)
        errs = [c.get("error") for c in cells if c.get("error")]
        ok = [c for c in cells if not c.get("error")]
        md = max((c["max_abs_logprob_diff"] or 0) for c in ok) if ok else None
        print(f"  round {r}: base≠adapter {n_diff}/{len(cells)}；首个分岔 "
              f"{[c.get('first_diff') for c in ok]}；max|Δlogprob| {md}"
              f"{'  ERR ' + str(errs) if errs else ''}")
    print(f"  耗时中位：{report['latency_median_s']}")


if __name__ == "__main__":
    main()
