#!/usr/bin/env python3
"""L3.3 —— decode 侧 attention：完全不同的一道题。

prefill 时 Q 有 S 行，attention 是 S×S 的二次问题（3.1/3.2）。
decode 时 Q 只有 1 行，那个二次项直接消失，剩下的是
「读完整个 KV cache，只为算一个 token」—— 一道纯访存题。

  [A] 算术强度：decode 为什么必然是访存受限
  [B] GQA / MQA：把 KV 读取量按比例砍掉
  [C] 并行度：单请求长上下文 decode 为什么需要 split-K
  [D] 分页布局要付多少代价
  [E] 广度：SDPA 与 FlashInfer 的 decode 路径

用法：
    python decode_attention.py
    python decode_attention.py B C
"""

import os
import sys

import torch
import torch.nn.functional as F

MB = 1024 * 1024
# decode attention 只读不写 KV，所以参照应当用**只读**带宽而不是 copy 带宽。
# L1.1 实测 crater: copy 1519.3 GB/s（含读+写两份流量），readonly 1608.6 GB/s。
PEAK_BW = 1608.6        # 只读带宽，GB/s
PEAK_COPY = 1519.3
PEAK_TF = 232.0         # L1.2 实测 bf16 上限 TFLOP/s
L2_MB = 96.0            # 这张卡的 L2；工作集小于它时测的是 L2 不是 DRAM


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def timeit(fn, n=30, warmup=10):
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
    title("[A] decode 的算术强度：为什么它必然是访存受限")

    print("  一步 decode，单头：Q 是 1×D，K/V 各是 S×D。")
    print("    FLOP: qK^T 2SD + pV 2SD = 4SD")
    print("    字节: 读 K/V = 2·S·D·2 = 4SD（bf16）；Q 与 O 只有 O(D)，忽略")
    print("    算术强度 = 4SD / 4SD = **1 FLOP/byte**")
    print()
    print(f"  而这张卡的机器平衡点是 {PEAK_TF * 1e12 / (PEAK_BW * 1e9):.1f} FLOP/byte。")
    print("  1 远小于它，**而且与 S、D 都无关** —— 再怎么调也走不出访存一侧。")
    print("  batch 能帮忙：B 条请求共享一次权重读取，但 KV 是各自的，读取量线性增长。")
    print("  所以 decode attention 的上限就是：把 KV cache 按带宽读一遍要多久。")

    D, H = 128, 32
    print(f"\n  H={H} D={D} bf16，逐个 (batch, kv_len) 量实际带宽")
    print(f"  参照：只读带宽 {PEAK_BW} GB/s。L2={L2_MB:.0f} MiB —— "
          f"工作集小于它时量的是 L2，不是 DRAM。")
    print(f"  {'batch':>6} {'kv_len':>8} {'KV 大小':>10} {'在 L2?':>7} {'ms':>9} "
          f"{'GB/s':>9} {'占只读峰值':>10}")
    for B, S in [(1, 1024), (1, 4096), (1, 16384), (1, 65536),
                 (8, 4096), (32, 4096), (128, 4096), (256, 4096)]:
        try:
            q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:>6} {S:>8}  显存不够"); torch.cuda.empty_cache(); continue
        kv_bytes = 2 * B * H * S * D * 2
        t = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
        gbs = kv_bytes / t / 1e6
        in_l2 = "是" if kv_bytes / MB <= L2_MB else "否"
        print(f"  {B:>6} {S:>8} {kv_bytes / MB:>8.1f}MB {in_l2:>6} {t:>9.4f} "
              f"{gbs:>9.1f} {gbs / PEAK_BW:>9.1%}")
        del q, k, v
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def section_B():
    title("[B] GQA / MQA：直接砍 KV 的读取量")

    print("  MHA：每个 query 头有自己的 K/V 头     -> KV 头数 = H")
    print("  GQA：若干 query 头共享一组 K/V        -> KV 头数 = H/g")
    print("  MQA：所有 query 头共享一组 K/V        -> KV 头数 = 1")
    print()
    print("  decode 是访存受限（[A] 节），KV 读取量按比例下降，时间就该按比例下降。")

    # B 取 32：让 KV 头数=1 时的 KV（128 MB）也大于 L2（96 MiB），
    # 否则小 KV 那几档量的是 L2 带宽，会得到超线性的假加速。
    B, S, D, HQ = 32, 8192, 128, 32
    print(f"\n  B={B} kv_len={S} D={D} query 头={HQ} bf16")
    print(f"  {'KV 头数':>8} {'名称':>6} {'KV 大小':>10} {'在 L2?':>7} {'ms':>9} "
          f"{'GB/s':>9} {'相对 MHA':>9} {'字节比':>8}")
    base = None
    for hkv in [32, 16, 8, 4, 2, 1]:
        q = torch.randn(B, HQ, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, hkv, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, hkv, S, D, device="cuda", dtype=torch.bfloat16)
        kv_bytes = 2 * B * hkv * S * D * 2
        # enable_gqa 让 SDPA 直接吃不同的 KV 头数，不用手工 expand
        t = timeit(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
        if base is None:
            base = t
        name = "MHA" if hkv == HQ else ("MQA" if hkv == 1 else "GQA")
        in_l2 = "是" if kv_bytes / MB <= L2_MB else "否"
        print(f"  {hkv:>8} {name:>6} {kv_bytes / MB:>8.1f}MB {in_l2:>6} {t:>9.4f} "
              f"{kv_bytes / t / 1e6:>9.1f} {base / t:>8.2f}× {HQ / hkv:>7.0f}×")
        del q, k, v
        torch.cuda.empty_cache()
    print("\n  「相对 MHA」应当与最后一列「字节比」一致 —— 一致就说明")
    print("  时间确实由 KV 读取量决定，GQA 的收益是线性的、可预测的。")
    print("  若某一行明显超过字节比，先查它是不是已经装进 L2 了。")

    sub("换算成一个真实模型的 KV cache")
    print("  Qwen3-1.7B: 28 层, 16 query 头, 8 KV 头, head_dim 128 (GQA g=2)")
    L, hkv, D2 = 28, 8, 128
    for ctx in [4096, 32768, 131072]:
        per_tok = 2 * L * hkv * D2 * 2                # K和V, bf16
        print(f"    上下文 {ctx:>7}: 每 token {per_tok} B, 单请求 KV = "
              f"{per_tok * ctx / MB:>8.1f} MB")
    print("    若是 MHA（16 个 KV 头），上面每个数字都要乘 2。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 并行度：单请求长上下文 decode 为什么需要 split-K")

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  这张卡 {sm} 个 SM。decode 时 Q 只有 1 行：")
    print("  沿 batch×head 切分能得到的并行单元只有 B·H 个。")
    print("  B=1、H=32 时只有 32 个单元 —— 一多半 SM 闲着，无论上下文多长。")
    print()
    print("  FlashDecoding 的做法：**再沿 KV 长度切分**（split-K），")
    print("  每段各自算局部的 m/l/O，最后用 3.1 的合并公式归约到一起。")
    print("  判据：固定 B·H 很小，把 kv_len 拉长。")
    print("  若没有 split-K，带宽利用率会随 kv_len 停在低位；有的话会随之上升。")

    D = 128
    for B, H in [(1, 8), (1, 32)]:
        print(f"\n  B={B} H={H}  (B·H={B * H}, 占 SM 的 {B * H / sm:.0%})")
        print(f"  {'kv_len':>8} {'KV 大小':>10} {'在 L2?':>7} {'ms':>9} "
              f"{'GB/s':>9} {'占只读峰值':>10}")
        for S in [1024, 4096, 16384, 65536, 262144]:
            try:
                q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            except torch.cuda.OutOfMemoryError:
                print(f"  {S:>8}  显存不够"); torch.cuda.empty_cache(); continue
            kv = 2 * B * H * S * D * 2
            t = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
            gbs = kv / t / 1e6
            in_l2 = "是" if kv / MB <= L2_MB else "否"
            print(f"  {S:>8} {kv / MB:>8.1f}MB {in_l2:>6} {t:>9.4f} {gbs:>9.1f} "
                  f"{gbs / PEAK_BW:>9.1%}")
            del q, k, v
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 分页布局要付多少代价")

    print("  vLLM 的 KV cache 不是每条序列一段连续内存，而是若干固定大小的块，")
    print("  用 block table 记录逻辑块->物理块（5.2）。好处是不用预留最大长度；")
    print("  代价是读 KV 时要先查表再 gather。这里量一下 gather 本身有多贵。")

    B, H, D, S, BLK = 8, 8, 8192, 128, 16
    S, D = 8192, 128
    nblk = S // BLK
    print(f"\n  B={B} H={H} kv_len={S} D={D} 块大小={BLK} -> 每条序列 {nblk} 块")

    # 连续布局
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
    t_cont = timeit(lambda: F.scaled_dot_product_attention(q, k, v))

    # 分页布局：物理块池 + 打乱的 block table
    pool_k = torch.randn(B * nblk, H, BLK, D, device="cuda", dtype=torch.bfloat16)
    pool_v = torch.randn(B * nblk, H, BLK, D, device="cuda", dtype=torch.bfloat16)
    perm = torch.randperm(B * nblk, device="cuda")
    table = perm.view(B, nblk)

    def paged():
        # gather 出每条序列的 KV，再走同一个 attention
        kk = pool_k[table]                      # [B, nblk, H, BLK, D]
        vv = pool_v[table]
        kk = kk.permute(0, 2, 1, 3, 4).reshape(B, H, S, D)
        vv = vv.permute(0, 2, 1, 3, 4).reshape(B, H, S, D)
        return F.scaled_dot_product_attention(q, kk, vv)

    t_paged = timeit(paged)
    kv_bytes = 2 * B * H * S * D * 2
    print(f"\n  {'布局':<26} {'ms':>9} {'GB/s':>9} {'相对连续':>9}")
    print(f"  {'连续':<26} {t_cont:>9.4f} {kv_bytes / t_cont / 1e6:>9.1f} {1.0:>8.2f}×")
    print(f"  {'分页 + 显式 gather':<22} {t_paged:>9.4f} "
          f"{kv_bytes / t_paged / 1e6:>9.1f} {t_cont / t_paged:>8.2f}×")
    print("\n  注意：这里的 gather 是**显式物化**成连续张量再算，")
    print("  是分页代价的**上界**。真实的 paged attention kernel 在 kernel 内部")
    print("  按 block table 直接寻址，不产生这份拷贝 —— 5.2 会看真实实现。")
    del k, v, pool_k, pool_v
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- E
def section_E():
    title("[E] 广度：SDPA 与 FlashInfer 的 decode 路径")

    D = 128
    print(f"  {'B':>4} {'H':>4} {'Hkv':>4} {'kv_len':>8} {'SDPA ms':>9} "
          f"{'FlashInfer ms':>14} {'比值':>7}")
    for B, H, HKV, S in [(1, 32, 8, 8192), (8, 32, 8, 8192),
                         (32, 32, 8, 4096), (1, 32, 8, 65536)]:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
        t_sdpa = timeit(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
        t_fi = None
        try:
            import flashinfer
            # FlashInfer 的单请求 decode 接口：q [H,D], k/v [S,Hkv,D]
            if B == 1:
                q2 = q.view(H, D)
                k2 = k.view(HKV, S, D).transpose(0, 1).contiguous()
                v2 = v.view(HKV, S, D).transpose(0, 1).contiguous()
                t_fi = timeit(lambda: flashinfer.single_decode_with_kv_cache(q2, k2, v2))
        except Exception as exc:                              # noqa: BLE001
            print(f"    (FlashInfer: {str(exc).splitlines()[0][:80]})")
            torch.cuda.empty_cache()
        fi_s = f"{t_fi:>14.4f}" if t_fi else f"{'—':>14}"
        ratio = f"{t_sdpa / t_fi:>7.2f}×" if t_fi else f"{'—':>7}"
        print(f"  {B:>4} {H:>4} {HKV:>4} {S:>8} {t_sdpa:>9.4f} {fi_s} {ratio}")
        del q, k, v
        torch.cuda.empty_cache()
    print("\n  FlashInfer 的批量接口需要先 plan（建 block table 与调度），")
    print("  单次调用的对比会低估它在服务里的价值。5.2/5.3 会在真实引擎里看。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  SM {p.multi_processor_count}")
    print(f"参照：只读 {PEAK_BW} / copy {PEAK_COPY} GB/s, bf16 {PEAK_TF} TFLOP/s "
          f"(L1.1/L1.2 实测)  L2 {L2_MB:.0f} MiB")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
