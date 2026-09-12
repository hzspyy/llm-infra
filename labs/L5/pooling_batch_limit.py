#!/usr/bin/env python3
"""5.12 batch 上限：为什么 embedding 服务能上大得多的 batch。

两件事分开做：
  (a) 解析账本 —— 只读 HF config，按公式算「权重 + KV + 激活」三项，
      给出在给定显存预算下 decode 与 encode 各自的 batch 上限。
  (b) 实测上限 —— 在 GPU 上把 batch 一路加大，记录最后一个不 OOM 的档位。

(a) 和 (b) 摆在一起才能说明问题：(a) 告诉你上限由哪一项决定，(b) 告诉你
框架实际用掉了多少（激活项在账本里是最难估的，只能实测）。

    python pooling_batch_limit.py --out DIR --budget-gb 30
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DT = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}


def cfg_of(repo: str, hub: str) -> dict:
    import glob
    hits = glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*", "config.json"))
    if not hits:
        raise FileNotFoundError(repo)
    return json.loads(Path(sorted(hits)[-1]).read_text())


def arch(cfg: dict) -> dict:
    d = cfg["hidden_size"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)
    d_head = cfg.get("head_dim") or d // n_head
    n_l = cfg["num_hidden_layers"]
    d_ffn = cfg.get("intermediate_size", 4 * d)
    vocab = cfg["vocab_size"]
    tied = bool(cfg.get("tie_word_embeddings", False))
    # 只支持本章已固定的架构，未知结构拒绝套用近似公式。
    kind = cfg["model_type"]
    attn = 2 * d * (n_head * d_head) + 2 * d * (n_kv * d_head)
    if kind == "qwen2":
        ffn = 3 * d * d_ffn
        # Q/K/V bias、两处 RMSNorm，以及最终 RMSNorm。
        n_body = n_l * (attn + ffn + n_head * d_head + 2 * n_kv * d_head + 2 * d) + d
        extra = 0
        head = 0 if tied else vocab * d
    elif kind == "bert" and cfg.get("architectures") == ["BertModel"]:
        ffn = 2 * d * d_ffn
        # 四个 attention bias、FFN bias、两处 LayerNorm。
        n_body = n_l * (attn + ffn + 4 * d + d_ffn + d + 4 * d)
        # 位置/类型嵌入、embedding LayerNorm、BertModel pooler。
        extra = (cfg["max_position_embeddings"] + cfg["type_vocab_size"]) * d + 2 * d + d * d + d
        head = 0
    else:
        raise ValueError(f"unsupported architecture: {kind}")
    return {"n_layer": n_l, "d_model": d, "n_head": n_head, "n_kv_head": n_kv,
            "d_head": d_head, "d_ffn": d_ffn, "vocab": vocab, "tied": tied,
            "n_body_est": n_body, "embedding_params": vocab * d,
            "extra_params": extra, "lm_head_params": head,
            "parameter_count": n_body + vocab * d + extra + head}


def ledger(a: dict, mode: str, B: int, L: int, dtype_bytes=2, act_per_token=16):
    """返回三项字节。act_per_token 是每个 token 的激活字节数（按 d_model 的倍数估）。"""
    weights = a["parameter_count"] * dtype_bytes
    if mode == "decode":
        kv = 2 * a["n_layer"] * a["n_kv_head"] * a["d_head"] * dtype_bytes * L * B
        act = act_per_token * a["d_model"] * dtype_bytes * B          # 1 个 token/序列
    else:
        kv = 0                                                        # encoder-only 不留 KV
        act = act_per_token * a["d_model"] * dtype_bytes * B * L
    return {"weights": weights, "kv": kv, "act": act, "total": weights + kv + act}


def max_batch(a: dict, mode: str, L: int, budget: int, dtype_bytes=2, act_per_token=16):
    if ledger(a, mode, 0, L, dtype_bytes, act_per_token)["total"] > budget:
        return 0
    lo, hi = 0, 1
    while ledger(a, mode, hi, L, dtype_bytes, act_per_token)["total"] <= budget and hi < 1 << 20:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ledger(a, mode, mid, L, dtype_bytes, act_per_token)["total"] <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ------------------------------------------------------------------ 实测
def measure_limit(forward, L, cap=4096):
    """从 1 开始翻倍加 batch，返回最后一个成功的 B 与失败时的 B。"""
    import torch
    ok, fail = 0, None
    B = 1
    while B <= cap:
        try:
            forward(B)
            torch.cuda.synchronize()
            ok = B
        except torch.cuda.OutOfMemoryError:
            fail = B
            break
        finally:
            torch.cuda.empty_cache()
        B *= 2
    return {"max_ok_batch": ok, "first_oom_batch": fail, "length": L}


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--budget-gb", type=float, default=30.0)
    p.add_argument("--hub", default=os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub"))
    p.add_argument("--llm", default=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    p.add_argument("--enc", default=os.environ.get("ENC_MODEL", "BAAI/bge-small-en-v1.5"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "batch_limit.json").exists():
        raise FileExistsError(out / "batch_limit.json")
    budget = int(args.budget_gb * (1 << 30))

    llm_a = arch(cfg_of(args.llm, args.hub))
    enc_a = arch(cfg_of(args.enc, args.hub))
    res = {"budget_bytes": budget, "llm": {"repo": args.llm, **llm_a},
           "encoder": {"repo": args.enc, **enc_a}, "analytic": {}, "measured": {}}

    for label, a, mode, L in [("llm_decode_ctx1024", llm_a, "decode", 1024),
                              ("llm_decode_ctx4096", llm_a, "decode", 4096),
                              ("encoder_len512", enc_a, "encode", 512)]:
        mb = max_batch(a, mode, L, budget)
        res["analytic"][label] = {
            "max_batch": mb, "length": L,
            "at_max": ledger(a, mode, mb, L),
            "at_1": ledger(a, mode, 1, L)}

    # ---- 实测：把 batch 一路加大，看框架实际能到哪 ----
    from transformers import AutoModel, AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    body = llm.model

    def fwd_decode(B):
        ids = torch.randint(0, 1000, (B, 1024), device="cuda")
        with torch.no_grad():
            out = body(ids, use_cache=True)
        nxt = torch.randint(0, 1000, (B, 1), device="cuda")
        pos = torch.full((B, 1), 1024, device="cuda", dtype=torch.long)
        try:
            with torch.no_grad():
                body(nxt, past_key_values=out.past_key_values, position_ids=pos, use_cache=True)
        finally:
            del out
    # decode 的 KV 是随 B 一起建的，所以「建 cache」和「走一步」要一起算
    res["measured"]["llm_decode_ctx1024"] = measure_limit(fwd_decode, 1024)
    del fwd_decode, body, llm
    torch.cuda.empty_cache()

    enc = AutoModel.from_pretrained(
        args.enc, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()

    def fwd_enc(B):
        ids = torch.randint(0, 1000, (B, 512), device="cuda")
        mask = torch.ones_like(ids)
        with torch.no_grad():
            enc(ids, attention_mask=mask).last_hidden_state.mean(dim=1)

    res["measured"]["encoder_len512"] = measure_limit(fwd_enc, 512)

    (out / "batch_limit.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
