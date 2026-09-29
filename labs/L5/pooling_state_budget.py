#!/usr/bin/env python3
"""L5.12 任务 B —— 同一 checkpoint 上的三态显存账：池化 / 建缓存 / 定长 decode。

计划要求在**同一个 checkpoint 内**比较「首轮 / 固定长度 decode / 池化任务」，
并按参数、激活、KV 分段采内存，且「缓存长度固定且每轮可核查」。
既有实验比的是两个不同大小的模型（BERT 类 encoder 对 Qwen 类 decoder），
那条 32× 的差值混着架构、参数量和上下文三个变量，不能用来说明「池化省状态」。

本脚本只用一个 checkpoint（Qwen3-Reranker-0.6B，因果 LM，可同时做
pooling 与自回归），跑三种模式并逐项分开记：

  pooling        一次前向、`use_cache=False`，取末位 hidden → 一个分数
  prefill_cache  同一次前向但 `use_cache=True`，把返回的 KV 张量字节数记下来
  decode_fixed   建好 L 长度的缓存后逐 token 解码 N 步；每步前后断言
                 `get_seq_length()` 恰好递增长度，缓存长度固定可核查

每个 (seq_len, batch) 都重开一次，避免上一模式的峰值污染下一模式。

用法（crater，serve venv）：
    python labs/L5/pooling_state_budget.py --out "$OUT/state-budget"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = os.environ.get(
    "L512_STATE_MODEL",
    "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/"
    "snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473")

QUERY = "Which mechanism lets multiple requests reuse an identical prompt prefix?"
DOC = ("Prefix caching reuses computed key and value blocks for matching prompt "
       "prefixes. A paged cache stores key and value tensors in fixed-size blocks. ")


def gib(n):
    return round(n / 1024 ** 3, 4)


def kv_geometry(cfg):
    """从 config 取 KV 的形状参数，供公式与实测对照。"""
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return dict(layers=cfg.num_hidden_layers, kv_heads=heads, head_dim=head_dim,
                dtype_bytes=2)          # bf16


def kv_bytes_per_token(cfg):
    g = kv_geometry(cfg)
    return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * g["dtype_bytes"]


def cache_bytes(past):
    """DynamicCache 里所有层 K/V 张量的实际字节数。"""
    total = 0
    try:
        for layer in past.layers:
            for t in (layer.keys, layer.values):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total
    except AttributeError:
        for k, v in getattr(past, "key_cache", {}).items() if hasattr(past, "key_cache") else []:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--decode-steps", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                        padding_side="left")

    def make_prompt(target_len: int) -> str:
        """把文档重复到足够长，让序列真正达到目标长度（否则截断不起作用）。"""
        doc = DOC
        while len(tok.encode(doc, add_special_tokens=False)) < target_len:
            doc += DOC
        return tok.apply_chat_template(
            [{"role": "query", "content": QUERY},
             {"role": "document", "content": doc}], tokenize=False)

    prompts = [make_prompt(max(args.seq_lens))] * max(args.batches)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    geo = kv_geometry(cfg)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    per_token = kv_bytes_per_token(cfg)

    rows = []
    for seq_len in args.seq_lens:
        for batch in args.batches:
            text = prompts[:batch]
            enc = tok(text, return_tensors="pt", padding=True, truncation=True,
                      max_length=seq_len, add_special_tokens=False).to("cuda")
            L = int(enc["input_ids"].shape[1])

            # 预热一次同形状前向：首次调用会额外分配 SDPA workspace，不进稳态账。
            with torch.inference_mode():
                _ = model(**enc, use_cache=False)
            torch.cuda.synchronize()

            # ---- 模式 1：池化（无缓存） ----
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.inference_mode():
                out = model.model(**enc, use_cache=False).last_hidden_state[:, -1, :]
                _ = out[:, 0].float().sum()      # 保持与其它模式一致的读取动作
            torch.cuda.synchronize()
            pooling_peak = torch.cuda.max_memory_allocated()
            pooling_act = pooling_peak - base

            # ---- 模式 2：建缓存（同一次前向，但保留 KV） ----
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base2 = torch.cuda.memory_allocated()
            with torch.inference_mode():
                out2 = model(**enc, use_cache=True)
            torch.cuda.synchronize()
            prefill_peak = torch.cuda.max_memory_allocated()
            past = out2.past_key_values
            prefill_kv = cache_bytes(past)
            cache_len_after_prefill = int(past.get_seq_length())
            del out2, past
            torch.cuda.empty_cache()

            # ---- 模式 3：定长 decode，每步核对缓存长度 ----
            with torch.inference_mode():
                pref = model(**enc, use_cache=True)
            past = pref.past_key_values
            del pref
            torch.cuda.reset_peak_memory_stats()
            base3 = torch.cuda.memory_allocated()
            nxt = enc["input_ids"][:, -1:]
            steps, lens = [], []
            for i in range(args.decode_steps):
                expect_in = cache_len_after_prefill + i
                assert int(past.get_seq_length()) == expect_in, (
                    f"缓存长度不是固定递推: {past.get_seq_length()} != {expect_in}")
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                pos = torch.full((batch, 1), expect_in, dtype=torch.long, device="cuda")
                s.record()
                with torch.inference_mode():
                    o = model(input_ids=nxt, position_ids=pos,
                              past_key_values=past, use_cache=True)
                e.record()
                torch.cuda.synchronize()
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
                lens.append(int(past.get_seq_length()))
                steps.append(dict(step=i, cache_len_before=expect_in,
                                  cache_len_after=int(past.get_seq_length()),
                                  kv_bytes=cache_bytes(past),
                                  ms=s.elapsed_time(e)))
                del o
            decode_peak = torch.cuda.max_memory_allocated()
            decode_act = decode_peak - base3
            del past
            torch.cuda.empty_cache()

            kv_growth = ((steps[-1]["kv_bytes"] - prefill_kv) / len(steps)
                         if steps else float("nan"))
            row = dict(
                seq_len_requested=seq_len, batch=batch, actual_tokens=L,
                pooling=dict(activation_peak_bytes=pooling_act, peak=gib(pooling_peak)),
                prefill=dict(kv_bytes=prefill_kv, peak=gib(prefill_peak),
                             cache_len=cache_len_after_prefill,
                             kv_formula_bytes=per_token * L * batch,
                             kv_formula_match=abs(prefill_kv - per_token * L * batch)
                             <= max(4096, 0.01 * per_token * L * batch)),
                decode=dict(activation_peak_bytes=decode_act, peak=gib(decode_peak),
                            steps=lens,
                            step_ms_median=sorted(x["ms"] for x in steps)[len(steps) // 2],
                            kv_growth_per_step_bytes=kv_growth,
                            kv_growth_formula_bytes=per_token * batch),
                kv_per_token_formula=per_token,
            )
            rows.append(row)
            print(f"L={L:>5} B={batch:>2}（请求 {seq_len}）  "
                  f"pooling 激活 {pooling_act/1e6:>8.2f} MB  "
                  f"prefill KV {prefill_kv/1e6:>8.2f} MB（公式 {per_token*L*batch/1e6:.2f}，"
                  f"吻合 {row['prefill']['kv_formula_match']}）  "
                  f"decode 峰值 +{decode_act/1e6:>7.2f} MB  每步 "
                  f"{row['decode']['step_ms_median']:.3f} ms  "
                  f"KV 每步增长 {kv_growth/1e6:.3f} MB（公式 "
                  f"{per_token*batch/1e6:.3f}）")
            print(f"      缓存长度轨迹 {lens}（每步 +1 已断言）")

    (args.out / "state_budget.json").write_text(json.dumps(dict(
        model=MODEL, torch=torch.__version__, transformers=transformers.__version__,
        dtype="bfloat16", kv_geometry=geo, param_bytes=param_bytes,
        param_gib=gib(param_bytes), kv_bytes_per_token=per_token,
        decode_steps=args.decode_steps, rows=rows,
        acceptance="KV 公式与实测相对差 ≤1%；每步缓存长度增量恰为 1；"
                   "pooling 与 decode 的激活峰值在同一 checkpoint 下分别报告"),
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n权重字节 {param_bytes/1e6:.2f} MB（bf16），"
          f"每 token 每序列 KV {per_token} bytes"
          f"（{geo['layers']} 层 × {geo['kv_heads']} KV head × {geo['head_dim']} dim × 2 × 2）")
    print("读法：三种模式共用同一份权重，差别只在持久状态与激活峰值——")
    print("  pooling 没有 KV；prefill/decode 的 KV 按每 token 常数线性增长。")
    print("旧的 32× 容量比来自两个不同大小的模型，混了架构与参数量，不能引用。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
