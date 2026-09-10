#!/usr/bin/env python3
"""L0 lab · 资源账本：把一个模型翻译成字节数、FLOP 数和时间下界。

这套材料里几乎每一个优化，本质上都是在改这张账单的某一项：
  - 量化改的是「权重字节」这一行
  - GQA / MLA / KV 量化改的是「每 token KV 字节」这一行
  - FlashAttention 改的是「attention 中间结果的访存」这一行
  - continuous batching 改的是「权重字节被多少个请求摊薄」这一行
  - 投机解码改的是「每次搬完权重能产出几个 token」这一行

所以先学会算这张账，后面每一章才有位置放。

用法：
    python ledger.py --config path/to/config.json --ctx 4096 --batch 1,8,32
    python ledger.py --config ... --hw hw_crater.json      # 带上实测 roofline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "fp4": 0.5, "int4": 0.5}


# ---------------------------------------------------------------------------

@dataclass
class Arch:
    """从 HF config.json 里抽出真正影响账单的那几个数。"""
    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_head: int
    d_ffn: int
    vocab: int
    tied_embeddings: bool = True
    # MoE
    n_expert: int = 0
    n_expert_active: int = 0
    d_ffn_shared: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, cfg: dict, name: str = "") -> "Arch":
        d = cfg["hidden_size"]
        n_head = cfg["num_attention_heads"]
        d_head = cfg.get("head_dim") or d // n_head
        return cls(
            name=name or cfg.get("_name_or_path", "?"),
            n_layer=cfg["num_hidden_layers"],
            d_model=d,
            n_head=n_head,
            n_kv_head=cfg.get("num_key_value_heads", n_head),
            d_head=d_head,
            d_ffn=cfg.get("intermediate_size", 4 * d),
            vocab=cfg["vocab_size"],
            tied_embeddings=cfg.get("tie_word_embeddings", False),
            n_expert=cfg.get("num_experts", cfg.get("n_routed_experts", 0)) or 0,
            n_expert_active=cfg.get("num_experts_per_tok", 0) or 0,
            d_ffn_shared=cfg.get("moe_intermediate_size", 0) or 0,
        )

    # -- 参数量 ------------------------------------------------------------

    @property
    def p_attn_per_layer(self) -> int:
        """q/k/v/o 四个投影。注意 k、v 用的是 n_kv_head——GQA 省的就是这里。"""
        q = self.d_model * self.n_head * self.d_head
        k = self.d_model * self.n_kv_head * self.d_head
        v = k
        o = self.n_head * self.d_head * self.d_model
        return q + k + v + o

    @property
    def p_ffn_per_layer(self) -> int:
        """SwiGLU 有三个矩阵：gate、up、down。"""
        if self.n_expert:
            d_e = self.d_ffn_shared or self.d_ffn
            return self.n_expert * 3 * self.d_model * d_e
        return 3 * self.d_model * self.d_ffn

    @property
    def p_ffn_active_per_layer(self) -> int:
        """MoE 下每个 token 实际点亮的 FFN 参数。稠密模型两者相等。"""
        if self.n_expert:
            d_e = self.d_ffn_shared or self.d_ffn
            return self.n_expert_active * 3 * self.d_model * d_e
        return self.p_ffn_per_layer

    @property
    def p_embed(self) -> int:
        return self.vocab * self.d_model * (1 if self.tied_embeddings else 2)

    @property
    def p_total(self) -> int:
        per_layer = self.p_attn_per_layer + self.p_ffn_per_layer
        return self.n_layer * per_layer + self.p_embed

    @property
    def p_body(self) -> int:
        """28 层主干里，每个 token 都要过一遍的线性层参数（不含 lm_head）。"""
        return self.n_layer * (self.p_attn_per_layer + self.p_ffn_active_per_layer)

    @property
    def p_head(self) -> int:
        """lm_head：隐状态 → 词表 logits 的那个大矩阵。"""
        return self.vocab * self.d_model

    @property
    def p_linear_active(self) -> int:
        """decode 一步要**读一遍**的线性层参数总量（主干 + lm_head）。

        注意这是「字节」口径：decode 每步都要把 lm_head 整个读进来。
        FLOP 口径在 prefill 时不同——见 prefill_flops() 的注释。
        """
        return self.p_body + self.p_head

    # -- KV cache ----------------------------------------------------------

    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        """K 和 V 各一份，所以有那个 2。这是 decode 阶段增长最快的一项。"""
        return 2 * self.n_layer * self.n_kv_head * self.d_head * BYTES[kv_dtype]


# ---------------------------------------------------------------------------

def prefill_flops(a: Arch, seq: int) -> dict:
    """一次 prefill 的 FLOP 拆分。

    这里有个容易高估 18% 的坑：**lm_head 只作用在最后一个 token 上**。
    prefill 的目的是把 prompt 的 KV 填好并产出第一个 token，
    中间那 S-1 个位置的 logits 没人要，引擎不会算（vLLM 只对
    `logits_indices` 指定的位置做 lm_head）。
    如果按 `p_linear_active * seq` 算，会把 lm_head 多算 S-1 倍——
    对 Qwen3-1.7B、S=4096 来说，凭空多出 2.55 TFLOP（总量的 18%）。
    """
    linear_body = 2 * a.p_body * seq
    linear_head = 2 * a.p_head * 1
    # QK^T 与 AV 各一次：2 * S^2 * n_head * d_head * 2(乘加)，causal 掩掉一半
    attn = a.n_layer * 2 * (2 * seq * seq * a.n_head * a.d_head) * 0.5
    return {"linear": linear_body + linear_head, "linear_body": linear_body,
            "linear_head": linear_head, "attention": attn,
            "total": linear_body + linear_head + attn}


def decode_step(a: Arch, ctx: int, batch: int, *,
                w_dtype: str = "bf16", kv_dtype: str = "bf16") -> dict:
    """decode 一步（每条序列出 1 个 token）的 FLOP 与字节。

    关键不对称：
      - 权重字节与 batch **无关**（一整批共享同一份权重）
      - KV 字节与 batch × 上下文长度 **成正比**
    这个不对称正是 continuous batching 有效、而超长上下文又会把收益吃掉的原因。
    """
    # decode 时每条序列都要出 logits，所以 lm_head 对每个 batch 元素都算一次
    flops_linear = 2 * (a.p_body + a.p_head) * batch
    flops_attn = a.n_layer * 2 * (2 * ctx * a.n_head * a.d_head) * batch
    flops = flops_linear + flops_attn

    bytes_w = a.p_linear_active * BYTES[w_dtype]
    bytes_kv = batch * ctx * a.kv_bytes_per_token(kv_dtype)
    total_bytes = bytes_w + bytes_kv

    return {
        "flops": flops,
        "bytes_weight": bytes_w,
        "bytes_kv": bytes_kv,
        "bytes_total": total_bytes,
        "arithmetic_intensity": flops / total_bytes,
    }


def roofline_tpot_ms(step: dict, hw: dict) -> dict:
    """把字节数和 FLOP 数换算成时间下界。

    memory 时间 = 搬运字节 / 实测显存带宽
    compute 时间 = FLOP / 实测稠密 GEMM 吞吐
    实际耗时 >= max(两者)。取不到这个下界，说明有别的开销（launch、同步、
    kernel 效率、CPU 调度），那就是要去 profile 的地方。
    """
    bw = hw["memory_bandwidth"]["copy_gbps"] * 1e9
    tflops = hw["gemm"]["bf16"]["peak_tflops"] * 1e12
    t_mem = step["bytes_total"] / bw
    t_cmp = step["flops"] / tflops
    return {
        "t_memory_ms": t_mem * 1e3,
        "t_compute_ms": t_cmp * 1e3,
        "t_lower_bound_ms": max(t_mem, t_cmp) * 1e3,
        "bound_by": "memory" if t_mem > t_cmp else "compute",
        "machine_balance_flops_per_byte": tflops / bw,
    }


# ---------------------------------------------------------------------------

def human_bytes(n: float) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def human_count(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def report(a: Arch, ctx: int, batches: list[int], hw: dict | None,
           w_dtype: str, kv_dtype: str) -> dict:
    out: dict = {"model": a.name, "ctx": ctx, "w_dtype": w_dtype, "kv_dtype": kv_dtype}

    print(f"\n{'=' * 78}")
    print(f"  {a.name}   L={a.n_layer} d={a.d_model} heads={a.n_head}/{a.n_kv_head}(kv) "
          f"d_head={a.d_head} ffn={a.d_ffn} vocab={a.vocab}")
    print(f"{'=' * 78}")

    # --- 参数账 ---
    p_attn = a.n_layer * a.p_attn_per_layer
    p_ffn = a.n_layer * a.p_ffn_per_layer
    print("\n[1] 参数账")
    print(f"  attention 投影      {human_count(p_attn):>10}  ({p_attn / a.p_total:5.1%})")
    print(f"  FFN                 {human_count(p_ffn):>10}  ({p_ffn / a.p_total:5.1%})")
    print(f"  embedding{'(tied)' if a.tied_embeddings else '+lm_head':<10} "
          f"{human_count(a.p_embed):>10}  ({a.p_embed / a.p_total:5.1%})")
    print(f"  {'总计':<19} {human_count(a.p_total):>10}"
          f"   权重占显存 @{w_dtype}: {human_bytes(a.p_total * BYTES[w_dtype])}")
    out["params"] = {"attn": p_attn, "ffn": p_ffn, "embed": a.p_embed,
                     "total": a.p_total,
                     "weight_bytes": a.p_total * BYTES[w_dtype]}

    # --- KV 账 ---
    kv_tok = a.kv_bytes_per_token(kv_dtype)
    kv_nogqa = 2 * a.n_layer * a.n_head * a.d_head * BYTES[kv_dtype]
    print("\n[2] KV cache 账")
    print(f"  每 token           {human_bytes(kv_tok)}   "
          f"(若不用 GQA 会是 {human_bytes(kv_nogqa)}，省了 {kv_nogqa / kv_tok:.1f}×)")
    print(f"  {ctx} token 上下文  {human_bytes(kv_tok * ctx)}  / 每条序列")
    out["kv"] = {"bytes_per_token": kv_tok, "bytes_at_ctx": kv_tok * ctx,
                 "gqa_saving": kv_nogqa / kv_tok}

    # --- prefill ---
    pf = prefill_flops(a, ctx)
    print(f"\n[3] prefill {ctx} token 的 FLOP")
    print(f"  线性层(主干)       {pf['linear_body'] / 1e12:8.2f} TFLOP  ({pf['linear_body'] / pf['total']:5.1%})")
    print(f"  lm_head(仅末位)    {pf['linear_head'] / 1e12:8.4f} TFLOP  ({pf['linear_head'] / pf['total']:5.2%})"
          f"   ← 只算最后一个 token，不是 S 个")
    print(f"  attention 打分     {pf['attention'] / 1e12:8.2f} TFLOP  ({pf['attention'] / pf['total']:5.1%})"
          f"   ← 随 S² 增长，长上下文时会反超")
    print(f"  {'总计':<18} {pf['total'] / 1e12:8.2f} TFLOP")
    out["prefill"] = pf
    if hw:
        t = pf["total"] / (hw["gemm"]["bf16"]["peak_tflops"] * 1e12)
        print(f"  → 用实测 {hw['gemm']['bf16']['peak_tflops']} TFLOPS 算，"
              f"理论下界 {t * 1e3:.1f} ms（TTFT 不可能低于它）")
        out["prefill"]["t_lower_bound_ms"] = t * 1e3

    # --- 显存预算：真正决定 batch 上限的东西 ---
    kv_budget = None
    max_seqs = None
    if hw:
        total = hw["device"]["total_mem_gb"] * 1e9
        util = 0.90                      # vLLM 默认 gpu_memory_utilization
        overhead = 1.5e9                 # 激活、通信缓冲、CUDA context 的粗估
        kv_budget = total * util - a.p_total * BYTES[w_dtype] - overhead
        max_seqs = int(kv_budget / (kv_tok * ctx))
        print(f"\n[3.5] 显存预算（{hw['device']['name']}, {total / 1e9:.1f} GB, util=0.90）")
        print(f"  权重 {human_bytes(a.p_total * BYTES[w_dtype])}"
              f" + 杂项 ~{human_bytes(overhead)} ⇒ 留给 KV 约 {human_bytes(kv_budget)}")
        print(f"  在 {ctx} token 上下文下，最多同时驻留 ≈ {max_seqs} 条序列。")
        print("  这才是 batch 的物理天花板——不是调度器想开多大就开多大。")
        out["kv_budget_bytes"] = kv_budget
        out["max_concurrent_seqs"] = max_seqs

    # --- decode ---
    print(f"\n[4] decode 一步（上下文 {ctx}）")
    hdr = f"  {'batch':>6} {'FLOP':>10} {'权重字节':>10} {'KV字节':>10} {'强度':>8}"
    if hw:
        hdr += f" {'访存ms':>8} {'算力ms':>8} {'下界ms':>8} {'瓶颈':>7} {'tok/s':>9} {'装得下':>7}"
    print(hdr)
    rows = []
    for b in batches:
        st = decode_step(a, ctx, b, w_dtype=w_dtype, kv_dtype=kv_dtype)
        line = (f"  {b:>6} {st['flops'] / 1e9:>9.1f}G {human_bytes(st['bytes_weight']):>10}"
                f" {human_bytes(st['bytes_kv']):>10} {st['arithmetic_intensity']:>7.1f}")
        row = {"batch": b, **st}
        if hw:
            rf = roofline_tpot_ms(st, hw)
            fits = st["bytes_kv"] <= kv_budget
            line += (f" {rf['t_memory_ms']:>8.2f} {rf['t_compute_ms']:>8.2f}"
                     f" {rf['t_lower_bound_ms']:>8.2f} {rf['bound_by']:>7}"
                     f" {b / (rf['t_lower_bound_ms'] * 1e-3):>9.0f}"
                     f" {'是' if fits else '✗ 放不下':>7}")
            row["roofline"] = rf
            row["fits_in_memory"] = fits
        print(line)
        rows.append(row)
    out["decode"] = rows

    if hw:
        mb = hw["gemm"]["bf16"]["peak_tflops"] * 1e12 / (hw["memory_bandwidth"]["copy_gbps"] * 1e9)
        print(f"\n  机器平衡点 (machine balance) = {mb:.0f} FLOP/byte。")
        print("  decode 的算术强度远低于它 ⇒ 天然 memory-bound：GPU 绝大部分算力在空转，")
        print("  时间几乎全花在把权重和 KV 从显存搬进片上。这是所有 decode 优化的出发点。")
        out["machine_balance"] = mb

    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="HF config.json 路径")
    ap.add_argument("--name", default="")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", default="1,8,32,128")
    ap.add_argument("--w-dtype", default="bf16", choices=list(BYTES))
    ap.add_argument("--kv-dtype", default="bf16", choices=list(BYTES))
    ap.add_argument("--hw", default=None, help="probe_hw.py 产出的 JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    # 多模态 config 把语言塔放在 text_config 里
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = cfg["text_config"]
    arch = Arch.from_hf(cfg, args.name or Path(args.config).parent.name)
    hw = json.loads(Path(args.hw).read_text()) if args.hw else None
    batches = [int(x) for x in args.batch.split(",")]

    res = report(arch, args.ctx, batches, hw, args.w_dtype, args.kv_dtype)
    if hw:
        res["hw_source"] = {"host": hw["device"]["host"], "gpu": hw["device"]["name"],
                            "measured_at": hw["measured_at"]}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
