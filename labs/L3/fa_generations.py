#!/usr/bin/env python3
"""L3.2 —— 每一代 FlashAttention 解决的具体矛盾，能测的那几条。

不复述论文。只测三件在这张卡上测得出来的事：

  [A] 并行度：kernel 到底沿哪些维度切分？（FA1→FA2 的核心改动）
  [B] 因果 mask 省掉了多少？（理想是一半，实际呢）
  [C] 广度：SDPA / FlashInfer 在同一批形状上的对照

FA3 的机制（TMA、warp specialization、wgmma 异步）是 Hopper(sm_90) 专有的，
这张卡是 sm_120，**测不了**。正文对 FA3 只作源码与文档层面的陈述。

用法：
    python fa_generations.py
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


def attn_flops(B, H, S, D, causal):
    f = 4.0 * B * H * S * S * D
    return f / 2 if causal else f


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 并行度：kernel 沿哪些维度切分")

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  这张卡有 {sm} 个 SM。")
    print("  FA1 只沿 (batch × head) 切分：并行单元数 = B·H。")
    print("  B·H 小于 SM 数时，一部分 SM 无事可做，和序列多长无关。")
    print("  FA2 增加了沿**查询块**的切分：并行单元数 = B·H·⌈S/块⌉。")
    print()
    print("  判据：固定 S，把 B·H 从远小于 SM 数扫到远大于。")
    print("  如果只沿 B·H 切分，达到的 TFLOP/s 会在 B·H≈SM 数处才饱和；")
    print("  如果也沿序列切分，B·H 很小时就已经接近饱和。")

    S, D = 4096, 128
    print(f"\n  S={S} D={D} bf16 causal")
    print(f"  {'B·H':>6} {'占 SM 比例':>10} {'ms':>9} {'TFLOP/s':>10} {'相对峰值':>9}")
    peak = 232.0                       # L1.2 实测 bf16 上限
    for bh in [1, 2, 4, 8, 16, 32, 64, 128, 170, 256, 512]:
        try:
            Q = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
            K = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
            V = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f"  {bh:>6}  显存不够，跳过")
            torch.cuda.empty_cache()
            continue
        t = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        tf = attn_flops(bh, 1, S, D, True) / t / 1e9
        print(f"  {bh:>6} {bh / sm:>9.2f}× {t:>9.3f} {tf:>10.1f} {tf / peak:>8.1%}")
        del Q, K, V
        torch.cuda.empty_cache()

    sub("对照：固定总工作量，把并行度从 B·H 挪到 S")
    print("  保持 B·H·S² 不变（总 FLOP 相同），只改 B·H 与 S 的分配。")
    print("  只沿 B·H 切分的实现会在「B·H 小、S 大」那一端明显变慢。")
    print(f"  {'B·H':>6} {'S':>7} {'总 FLOP(G)':>12} {'ms':>9} {'TFLOP/s':>10}")
    base_bh, base_s = 256, 2048
    total = base_bh * base_s * base_s
    for bh in [4, 16, 64, 256]:
        S2 = int((total / bh) ** 0.5)
        S2 = (S2 // 128) * 128
        try:
            Q = torch.randn(bh, 1, S2, D, device="cuda", dtype=torch.bfloat16)
            K = torch.randn(bh, 1, S2, D, device="cuda", dtype=torch.bfloat16)
            V = torch.randn(bh, 1, S2, D, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f"  {bh:>6} {S2:>7}  显存不够，跳过")
            torch.cuda.empty_cache()
            continue
        t = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        fl = attn_flops(bh, 1, S2, D, True)
        print(f"  {bh:>6} {S2:>7} {fl / 1e9:>12.1f} {t:>9.3f} {fl / t / 1e9:>10.1f}")
        del Q, K, V
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 因果 mask 省掉了多少")

    print("  因果 attention 只需算下三角，理想省一半（时间比 0.50）。")
    print("  但块粒度下，对角线上那一排块只有一半有效，整块仍要算。")
    print("  设 N = S/Bk 个 key 块，query 块 i 要算 (i+1) 个 key 块：")
    print("      因果块数 = N(N+1)/2，非因果 = N²  ->  比值 = (N+1)/(2N) = 0.5 + 1/(2N)")
    print("  所以块粒度带来的额外开销只有 1/(2N)，S 越大越可以忽略。")
    D, BK = 128, 128
    print(f"\n  {'S':>7} {'非因果 ms':>11} {'因果 ms':>10} {'实测比':>8} "
          f"{'块粒度预测':>11} {'差额':>8}")
    for S in [512, 1024, 2048, 4096, 8192, 16384]:
        bh = max(1, 2 ** 22 // (S * D) )      # 控制显存
        try:
            Q = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
            K = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
            V = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        t_full = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=False))
        t_caus = timeit(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        N = max(1, S // BK)
        pred = 0.5 + 1.0 / (2 * N)
        meas = t_caus / t_full
        print(f"  {S:>7} {t_full:>11.3f} {t_caus:>10.3f} "
              f"{meas:>8.3f} {pred:>11.3f} {meas - pred:>+8.3f}")
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  实测比一路下降，方向和块粒度预测一致 —— 说明整块在未来的块**确实被跳过了**。")
    print("  但差额没有随 S 收敛到 0：S 大时块粒度只该贡献 0.004，实测仍高出 0.16 左右。")
    print("  所以还有别的原因。最可能的是**负载不均**：因果下 query 块 i 的工作量")
    print("  正比于 i+1，最后一个块是第一个块的 N 倍。如果块按简单顺序分给 SM，")
    print("  整个 kernel 要等最重的那些块结束。")
    print("  这是一个**假设**，本轮没有验证。要证实它，用 2.6 那个 %smid+%globaltimer")
    print("  的自制 profiler 记录每个 block 的起止时间，看尾部是否被少数重块拖住。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 广度：同一批形状上的几种实现")

    from torch.nn.attention import SDPBackend, sdpa_kernel
    D = 128
    shapes = [(8, 4096), (32, 2048), (4, 8192)]
    print(f"  {'B·H':>5} {'S':>7} {'SDPA-FLASH':>12} {'SDPA-CUDNN':>12} "
          f"{'SDPA-EFF':>10} {'FlashInfer':>12}")
    for bh, S in shapes:
        Q = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(bh, 1, S, D, device="cuda", dtype=torch.bfloat16)
        row = [f"  {bh:>5} {S:>7}"]
        for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                   SDPBackend.EFFICIENT_ATTENTION]:
            try:
                with sdpa_kernel(be):
                    t = timeit(lambda: F.scaled_dot_product_attention(
                        Q, K, V, is_causal=True))
                row.append(f"{t:>12.3f}")
            except Exception:                                 # noqa: BLE001
                row.append(f"{'不可用':>12}")
                torch.cuda.empty_cache()
        # FlashInfer 的 prefill 接口用 [S, H, D] 布局
        try:
            import flashinfer
            q = Q.squeeze(1).transpose(0, 1).contiguous().view(S, bh, D)
            k = K.squeeze(1).transpose(0, 1).contiguous().view(S, bh, D)
            v = V.squeeze(1).transpose(0, 1).contiguous().view(S, bh, D)
            t = timeit(lambda: flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=True))
            row.append(f"{t:>12.3f}")
        except Exception as exc:                              # noqa: BLE001
            row.append(f"{'失败':>12}")
            if bh == shapes[0][0]:
                print(f"    (FlashInfer: {str(exc).splitlines()[0][:90]})")
            torch.cuda.empty_cache()
        print(" ".join(row))
        del Q, K, V
        torch.cuda.empty_cache()
    print("\n  单次测量差 10% 以内不构成谁更快的结论（见 M2）。")
    print("  这张表的用处是：确认它们都存在、都能跑、量级一致。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  sm_{p.major}{p.minor}  SM {p.multi_processor_count}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
