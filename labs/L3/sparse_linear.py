#!/usr/bin/env python3
"""L3.4 —— 稀疏、线性与混合：哪些能测，哪些只能算。

这一层的东西大多还在变动，很多没有可用实现。所以本 lab 严格分两类：
**能在这张卡上直接测的**，和**只能按定义算的**。

  [A] attention sink：从真实模型里把注意力权重取出来看
  [B] 滑动窗口：KV 有上界之后，decode 的账变成什么样
  [C] 线性 attention：O(S) 的代价与它换掉的东西
  [D] RoPE 与长上下文外推：频率表长什么样，scaling 改了什么

用法：
    python sparse_linear.py
    python sparse_linear.py A D
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

MB = 1024 * 1024
MODEL = os.environ.get("L34_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


# ---------------------------------------------------------------- A
def section_A():
    title("[A] attention sink：真实模型的注意力权重长什么样")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

    text = ("The capital of France is Paris. The capital of Japan is Tokyo. "
            "Machine learning models process sequences of tokens one at a time.")
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    S = ids.shape[1]
    print(f"  模型 {MODEL}，输入 {S} 个 token")
    print(f"  前 8 个 token: {[tok.decode([i]) for i in ids[0, :8].tolist()]}")

    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attn = out.attentions          # tuple: 每层 [B, H, S, S]
    print(f"  拿到 {len(attn)} 层，每层 shape {tuple(attn[0].shape)}")

    sub("每一层里，最后一个 query 分给「第 0 个 token」多少注意力")
    print(f"  如果注意力是均匀的，应该是 1/{S} = {1 / S:.4f}")
    print(f"  {'层':>4} {'给 token 0':>12} {'给最近 8 个':>12} {'其余':>10} {'token0 是均匀的几倍':>18}")
    for li in [0, 1, 2, len(attn) // 2, len(attn) - 2, len(attn) - 1]:
        a = attn[li][0, :, -1, :].float()        # [H, S] 最后一行
        p0 = a[:, 0].mean().item()
        recent = a[:, -8:].sum(-1).mean().item()
        rest = 1.0 - p0 - recent
        print(f"  {li:>4} {p0:>12.4f} {recent:>12.4f} {rest:>10.4f} "
              f"{p0 * S:>17.1f}×")

    sub("把第 0 个 token 的注意力份额按层×头铺开（挑一层）")
    li = len(attn) // 2
    a0 = attn[li][0, :, -1, 0].float()
    print(f"  第 {li} 层，各头给 token 0 的份额：")
    print("   ", " ".join(f"{v:.3f}" for v in a0.tolist()))
    print(f"  最大 {a0.max().item():.3f}  最小 {a0.min().item():.3f}  "
          f"中位数 {a0.median().item():.3f}")

    sub("那个 token 0 是什么")
    print(f"  token id {ids[0, 0].item()} = {tok.decode([ids[0, 0].item()])!r}")
    print("  它通常没有语义上的重要性。attention sink 的解释是：")
    print("  softmax 强制每行和为 1，当某个头「这一步不想关注任何东西」时，")
    print("  它需要一个地方倾倒这份概率质量，于是选了位置最固定的第一个 token。")
    print("  这条解释在文献里有，但**本 lab 只测到了现象，没有验证机制**。")

    sub("对滑动窗口的直接后果")
    print("  如果窗口把 token 0 划出去，那份质量无处可去，")
    print("  会被迫分给窗口内的 token —— 这就是 StreamingLLM 要单独保留")
    print("  前几个 token（sink token）的原因。")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 滑动窗口：KV 有了上界")

    print("  full attention：KV 随上下文线性增长，无上界。")
    print("  滑动窗口（SWA）：只保留最近 W 个 token 的 KV，上界就是 W。")
    print("  代价：窗口外的信息完全看不到（除非靠层数堆出感受野）。")

    L, hkv, D = 28, 8, 128
    per_tok = 2 * L * hkv * D * 2
    print(f"\n  以 Qwen3-1.7B 的配置算（28 层, 8 KV 头, D=128, bf16）")
    print(f"  每 token 每层 KV = {per_tok // L} B，全模型 {per_tok} B")
    print(f"  {'上下文':>9} {'full KV':>12} {'W=4096':>12} {'W=1024':>12}")
    for ctx in [4096, 16384, 65536, 262144, 1048576]:
        full = per_tok * ctx / MB
        w4 = per_tok * min(ctx, 4096) / MB
        w1 = per_tok * min(ctx, 1024) / MB
        print(f"  {ctx:>9} {full:>10.1f}MB {w4:>10.1f}MB {w1:>10.1f}MB")
    print("\n  上下文 1M 时 full 要 114 GB，W=4096 只要 448 MB —— 差 256 倍。")
    print("  更关键的是 **W 那两列不再增长**：显存与每步耗时都变成常数。")

    sub("实测：decode 每步耗时 vs 上下文")
    B, H = 8, 8
    print(f"  B={B} H={H} D={D} bf16")
    print(f"  {'上下文':>9} {'full ms':>10} {'W=4096 ms':>12} {'加速':>8}")
    for ctx in [4096, 16384, 65536, 131072]:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
        try:
            k = torch.randn(B, H, ctx, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, H, ctx, D, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f"  {ctx:>9}  显存不够"); torch.cuda.empty_cache(); continue
        t_full = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
        w = min(ctx, 4096)
        kw, vw = k[:, :, -w:].contiguous(), v[:, :, -w:].contiguous()
        t_win = timeit(lambda: F.scaled_dot_product_attention(q, kw, vw))
        print(f"  {ctx:>9} {t_full:>10.4f} {t_win:>12.4f} {t_full / t_win:>7.2f}×")
        del k, v, kw, vw, q
        torch.cuda.empty_cache()
    print("\n  W 固定之后，decode 每步耗时不随上下文增长 —— 这是 SWA 的全部卖点。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 线性 attention：把 softmax 换掉之后")

    print("  标准 attention：O = softmax(QK^T/√d) V，必须先算 QK^T（S×S）。")
    print("  线性 attention：把 softmax 换成一个可分解的核 φ(q)·φ(k)^T，于是")
    print("      O_i = φ(q_i) · Σ_{j<=i} φ(k_j) v_j^T  /  (φ(q_i) · Σ_{j<=i} φ(k_j))")
    print("  括号里那两个和是**可以增量维护的状态**（一个 d×d 矩阵 + 一个 d 向量）。")
    print("  于是：不需要 S×S，decode 时状态大小与上下文长度**无关**。")

    def linear_attn(Q, K, V):
        """最简单的 φ = elu(x)+1 版本，因果。O(S·d²)。"""
        phi_q = F.elu(Q) + 1
        phi_k = F.elu(K) + 1
        # 累积状态：KV[i] = Σ_{j<=i} φ(k_j) v_j^T,  Z[i] = Σ_{j<=i} φ(k_j)
        kv = torch.einsum("bhsd,bhse->bhsde", phi_k, V).cumsum(dim=2)
        z = phi_k.cumsum(dim=2)
        num = torch.einsum("bhsd,bhsde->bhse", phi_q, kv)
        den = torch.einsum("bhsd,bhsd->bhs", phi_q, z).clamp(min=1e-6)
        return num / den.unsqueeze(-1)

    sub("复杂度对照（prefill，B=1 H=8 D=64）")
    B, H, D = 1, 8, 64
    print(f"  {'S':>7} {'标准 ms':>10} {'线性 ms':>10} {'标准/线性':>10} "
          f"{'标准峰值MB':>11} {'线性峰值MB':>11}")
    for S in [512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t_std = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True), n=10)
        m_std = (torch.cuda.max_memory_allocated() - base) / MB
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t_lin = timeit(lambda: linear_attn(Q, K, V), n=10)
        m_lin = (torch.cuda.max_memory_allocated() - base) / MB
        print(f"  {S:>7} {t_std:>10.4f} {t_lin:>10.4f} {t_std / t_lin:>9.2f}× "
              f"{m_std:>10.1f} {m_lin:>10.1f}")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  注意：这个朴素线性实现用 cumsum 物化了 S 份 d×d 状态，")
    print("  所以它的显存反而更大、也没有更快。**真实的线性 attention 用**")
    print("  **分块递推，不物化中间状态。** 这里量的是「朴素写法」，")
    print("  用来说明复杂度优势不会自动变成性能优势。")

    sub("线性 attention 换掉了什么：精确检索")
    print("  做一个最简单的检索任务：序列里埋一个 key-value 对，最后去查它。")
    S, D = 256, 64
    torch.manual_seed(0)
    Q = torch.zeros(1, 1, S, D, device="cuda")
    K = torch.randn(1, 1, S, D, device="cuda")
    V = torch.randn(1, 1, S, D, device="cuda")
    target = 37
    Q[0, 0, -1] = K[0, 0, target] * 8.0        # 让最后一个 query 明确指向 target
    std = F.scaled_dot_product_attention(Q, K, V, is_causal=True)[0, 0, -1]
    lin = linear_attn(Q, K, V)[0, 0, -1]
    want = V[0, 0, target]
    print(f"  埋在位置 {target}，最后一个 query 指向它")
    print(f"  标准 attention 取回的向量 与目标的余弦相似度: "
          f"{F.cosine_similarity(std, want, dim=0).item():.4f}")
    print(f"  线性 attention 取回的向量 与目标的余弦相似度: "
          f"{F.cosine_similarity(lin, want, dim=0).item():.4f}")
    print("\n  softmax 的指数放大让最大的那一项主导，可以做到近乎精确的检索；")
    print("  线性核没有这个放大，状态是所有 (k,v) 的加权和，信息被摊平了。")
    print("  这就是纯线性架构在「大海捞针」类任务上吃亏的原因，")
    print("  也是混合架构（几层 full attention + 多层线性/SSM）流行的原因。")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] RoPE 与长上下文外推")

    import json
    import glob
    cfgs = glob.glob(os.path.expanduser(
        "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/*/config.json"))
    D, base, trained = 128, 1000000.0, 32768
    if cfgs:
        c = json.load(open(cfgs[0]))
        D = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
        base = float(c.get("rope_theta", 10000.0))
        trained = c.get("max_position_embeddings", 32768)
        print(f"  从 config 读到：head_dim={D}  rope_theta={base:g}  "
              f"max_position_embeddings={trained}")
        print(f"  rope_scaling = {c.get('rope_scaling')}")

    inv = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
    period = 2 * math.pi / inv
    print(f"\n  RoPE 把 head_dim 拆成 {D // 2} 对，第 i 对的角频率 "
          f"= 1/base^(2i/d)")
    print(f"  {'第几对':>6} {'角频率':>14} {'周期(token)':>16} {'说明'}")
    for i in [0, 1, 8, 16, 32, D // 2 - 2, D // 2 - 1]:
        note = ""
        if period[i].item() > trained:
            note = f"周期 > 训练长度 {trained}，从未转满一圈"
        print(f"  {i:>6} {inv[i].item():>14.3e} {period[i].item():>16.1f}  {note}")

    print(f"\n  关键：周期大于训练长度的那些维度，在训练时**从未见过完整的一圈**。")
    print(f"  推理时如果位置超过 {trained}，它们会进入训练中没出现过的角度区间 ——")
    print("  这就是长上下文外推失败的直接原因，不是「模型记不住」。")

    sub("三种 scaling 改的是什么")
    S_FACTOR = 4.0
    # PI：位置除以 s，等价于所有角频率除以 s，所有周期 × s
    inv_pi = inv / S_FACTOR
    # NTK / base 放大：base -> base * s^(d/(d-2))
    b_ntk = base * (S_FACTOR ** (D / (D - 2)))
    inv_ntk = 1.0 / (b_ntk ** (torch.arange(0, D, 2).float() / D))

    print(f"  放大倍数 s = {S_FACTOR:g}（想把 {trained} 撑到 {int(trained * S_FACTOR)}）")
    print(f"  {'方案':<10} {'最高频周期(i=0)':>18} {'中间(i=32)':>14} "
          f"{'最低频周期(i=63)':>20}")
    for name, iv in [("原始", inv), ("PI", inv_pi), ("NTK", inv_ntk)]:
        pp = 2 * math.pi / iv
        print(f"  {name:<10} {pp[0].item():>18.2f} {pp[32].item():>14.1f} "
              f"{pp[-1].item():>20.1f}")

    print(f"\n  只看最低频那一列，PI 与 NTK **完全一样**（都 ×{S_FACTOR:g}）——")
    print("  NTK 的 base 放大公式本来就是这么设计的，让最低频恰好对上 PI。")
    print("  差别全在最高频那一列：")
    print(f"    PI  把 i=0 的周期从 {(2 * math.pi / inv)[0].item():.2f} "
          f"拉到 {(2 * math.pi / inv_pi)[0].item():.2f}（×{S_FACTOR:g}）")
    print(f"    NTK 把 i=0 的周期从 {(2 * math.pi / inv)[0].item():.2f} "
          f"拉到 {(2 * math.pi / inv_ntk)[0].item():.2f}"
          f"（×{((2 * math.pi / inv_ntk)[0] / (2 * math.pi / inv)[0]).item():.3f}，几乎不变）")
    print("\n  高频维度负责区分「相邻几个 token」。PI 把它也拉长 4 倍，")
    print("  等于让模型分不清近处的位置；NTK 保住高频，只拉低频。")
    print("  YaRN 在此基础上再按维度分段处理，并加一个注意力温度修正。")
    print("\n  **本 lab 只算了频率，没有测任何模型质量。**")
    print("  哪种 scaling 更好必须跑长上下文评测（大海捞针、长文困惑度），")
    print("  频率表只能说明它们改了什么，不能说明改得好不好。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
