#!/usr/bin/env python3
"""L3.1 —— attention 的内存墙：把 S×S 存下来要多少代价。

三件事：
  [A] 峰值显存：朴素实现 vs SDPA，随 S 增长；朴素在哪个 S 上 OOM
  [B] 时间与有效带宽，以及算术强度的推导 vs 实测
  [C] PyTorch 到底选了哪个 attention 后端

用法：
    python attention_memory.py
"""

import os
import sys

import torch
import torch.nn.functional as F

MB = 1024 * 1024


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def naive_attention(Q, K, V, causal=True):
    """教科书写法：把 S×S 的打分矩阵完整算出来。"""
    D = Q.shape[-1]
    s = (Q @ K.transpose(-1, -2)) * (D ** -0.5)     # [B,H,S,S]  <- 这一行是问题所在
    if causal:
        S = Q.shape[-2]
        mask = torch.ones(S, S, dtype=torch.bool, device=Q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)                    # 又一个 [B,H,S,S]
    return p @ V


def timeit(fn, n=10, warmup=3):
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


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / MB


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 峰值显存：那个 S×S 到底有多大")

    B, H, D = 1, 32, 128
    dtype = torch.bfloat16
    print(f"  B={B} H={H} D={D} dtype={dtype}  （约等于一个 7B 模型的一层）")
    print(f"\n  {'S':>7} {'Q/K/V 合计':>12} {'S×S 一份':>12} "
          f"{'朴素峰值':>12} {'SDPA 峰值':>12} {'倍数':>7}")
    for S in [512, 1024, 2048, 4096, 8192, 16384]:
        qkv_mb = 3 * B * H * S * D * 2 / MB
        ss_mb = B * H * S * S * 2 / MB
        Q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        K = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        V = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        try:
            pn = peak_mb(lambda: naive_attention(Q, K, V))
            pn_s = f"{pn:>10.1f}"
        except torch.cuda.OutOfMemoryError:
            pn, pn_s = float("inf"), "      OOM"
            torch.cuda.empty_cache()
        pf = peak_mb(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        ratio = f"{pn / pf:>6.1f}×" if pn != float("inf") else "     —"
        print(f"  {S:>7} {qkv_mb:>10.1f}MB {ss_mb:>10.1f}MB "
              f"{pn_s}MB {pf:>10.1f}MB {ratio}")
        del Q, K, V
        torch.cuda.empty_cache()

    print("\n  S×S 一份的大小 = B·H·S²·2 字节，随 S **平方**增长。")
    print("  Q/K/V 合计只随 S 线性增长。S 一大，S×S 就成了唯一的大头。")
    print("  SDPA 的峰值基本只有 Q/K/V + 输出，因为它从不把 S×S 写出去。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 访存量推导与算术强度")

    print("  单头、序列 S、头维 D、bf16（2 字节）。只数必须过 DRAM 的量。")
    print()
    print("  FLOP（两次矩阵乘）：")
    print("      QK^T : 2·S²·D        PV : 2·S²·D        合计 4·S²·D")
    print()
    print("  朴素实现的 DRAM 流量（保守估计，只算 S×S 那几趟）：")
    print("      写 s        S²·2")
    print("      读 s（softmax 求 max/和）  S²·2 × 2 趟")
    print("      写 p        S²·2")
    print("      读 p（PV）  S²·2")
    print("      ≈ 6·S²·2 = 12·S² 字节，再加 Q/K/V 的 3·S·D·2")
    print()
    print("  分块实现：S×S 从不落 DRAM，只有 Q/K/V/O 各过一遍")
    print("      ≈ 4·S·D·2 = 8·S·D 字节")
    print()
    print("  算术强度 = FLOP / 字节：")
    print("      朴素  ≈ 4S²D / (12S² + 6SD)  --S 大时-->  D/3     （**与 S 无关**）")
    print("      分块  ≈ 4S²D / (8SD)         =            S/2     （**随 S 增长**）")
    print()
    print("  这就是内存墙的全部内容：朴素实现的算术强度被钉死在 D/3 附近，")
    print("  D=128 时约 43 FLOP/byte，而 crater 的机器平衡点是 152.7（L0.2）——")
    print("  它落在 roofline 的带宽一侧，而且再长的序列也走不出去。")

    B, H, D = 1, 32, 128
    print(f"\n  {'S':>7} {'朴素 ms':>10} {'SDPA ms':>10} {'加速':>7} "
          f"{'SDPA TFLOP/s':>13} {'朴素强度':>9} {'分块强度':>9}")
    for S in [512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        # 满 attention: QK^T 2S²D + PV 2S²D = 4S²D（每头）。causal 只需一半。
        flops = 4 * B * H * S * S * D / 2
        try:
            tn = timeit(lambda: naive_attention(Q, K, V))
            tn_s, sp = f"{tn:>8.3f}", None
        except torch.cuda.OutOfMemoryError:
            tn, tn_s = float("inf"), "     OOM"
            torch.cuda.empty_cache()
        tf = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        sp = f"{tn / tf:>6.1f}×" if tn != float("inf") else "     —"
        ai_naive = 4 * S * S * D / (12 * S * S + 6 * S * D)
        ai_flash = 4 * S * S * D / (8 * S * D)
        print(f"  {S:>7} {tn_s} {tf:>10.3f} {sp} {flops / tf / 1e9:>13.1f} "
              f"{ai_naive:>9.1f} {ai_flash:>9.1f}")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  「朴素强度」这一列几乎不随 S 变化，「分块强度」随 S 线性增长。")
    print("  推导预测的东西，在加速比那一列上看得见。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] SDPA 到底选了哪个后端")

    from torch.nn.attention import SDPBackend, sdpa_kernel
    B, H, S, D = 1, 32, 4096, 128
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    names = {SDPBackend.FLASH_ATTENTION: "FLASH_ATTENTION",
             SDPBackend.EFFICIENT_ATTENTION: "EFFICIENT_ATTENTION",
             SDPBackend.MATH: "MATH（就是朴素实现）",
             SDPBackend.CUDNN_ATTENTION: "CUDNN_ATTENTION"}
    print(f"  B={B} H={H} S={S} D={D} bf16 causal")
    print(f"  {'后端':<28} {'ms':>9} {'峰值 MB':>10}")
    for be, nm in names.items():
        try:
            with sdpa_kernel(be):
                t = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
                p = peak_mb(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
            print(f"  {nm:<28} {t:>9.3f} {p:>10.1f}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  {nm:<28} 不可用: {str(exc).splitlines()[0][:60]}")
            torch.cuda.empty_cache()

    sub("默认会选谁")
    t = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
    print(f"  不指定后端  {t:>9.3f} ms   （和上表哪一行最接近，就是默认选的那个）")
    print("\n  MATH 后端就是本 lab 的 naive_attention —— 它是**参照实现**，")
    print("  在任何硬件上都能跑，也用来给别的后端对答案。")
    del Q, K, V
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    print(f"L2 {p.L2_cache_size / MB:.0f} MiB  显存 {p.total_memory / MB / 1024:.1f} GiB")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
