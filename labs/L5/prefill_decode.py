#!/usr/bin/env python3
"""L5.1 —— prefill 与 decode 的系统性质。

3.1/3.3 从 attention 的角度看过这两相。这里从**整个引擎**的角度看：
它们对硬件的要求完全相反，所以调度、批处理、甚至部署方式都不同。

  [A] 两相的 roofline 位置：算力受限 vs 带宽受限
  [B] batch 对两相的影响完全不同
  [C] 一个请求的时间构成：TTFT vs TPOT
  [D] 长度不齐时的浪费：为什么需要 continuous batching（5.3）

用法：
    python prefill_decode.py
"""

import os
import sys
import time

import torch

# 模型都已在本地缓存；强制离线，避免 vLLM/transformers 每次去连 Hub
# （连不上时会直接抛 httpx.ConnectError，即使文件就在本地）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MB = 1024 * 1024
PEAK_TF = 232.0      # L1.2 实测 bf16
PEAK_BW = 1608.6     # L1.1 实测只读带宽
MODEL = os.environ.get("L51_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def safe_util(reserve_gib=4.0, cap=0.55):
    """按**当前空闲显存**算一个安全的 gpu_memory_utilization。

    crater 是共享机器，可能有别人的作业在跑（本轮就撞上过一次
    loopserve 的 lightnav_replay_gate 占了 14 GB）。
    写死 0.55 会在别人占了一半时直接启动失败，
    更糟的是在别人**之后**才启动时把对方挤掉。
    所以按空闲量算，并额外留 reserve_gib 给对方增长。
    """
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    usable = max(free / gib - reserve_gib, 1.0)
    util = min(cap, usable / (total / gib))
    print(f"  [显存] 空闲 {free / gib:.1f}/{total / gib:.1f} GiB，"
          f"留 {reserve_gib:.0f} GiB 给他人 -> gpu_memory_utilization={util:.2f}")
    return util


def make_llm(**kw):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(),
             max_model_len=8192, enforce_eager=True,
             enable_prefix_caching=False, disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                         # noqa: BLE001
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def rand_prompt(tok, n, rng):
    ids = [rng.randint(1000, 60000) for _ in range(n)]
    return ids


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 两相在 roofline 上的位置")

    import json, glob
    hub = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
    d = sorted(glob.glob(f"{hub}/models--{MODEL.replace('/', '--')}/snapshots/*"))[0]
    c = json.load(open(d + "/config.json"))
    H, I, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    nq, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim", H // nq)
    V = c["vocab_size"]
    P = L * (H * nq * hd + 2 * H * nkv * hd + nq * hd * H + 3 * H * I) + V * H

    print(f"  {MODEL}: {P / 1e9:.2f}B 参数，权重 bf16 = {P * 2 / MB / 1024:.2f} GiB")
    print(f"  机器平衡点 = {PEAK_TF * 1e12 / (PEAK_BW * 1e9):.1f} FLOP/byte")
    print(f"\n  {'相':<8} {'batch':>6} {'序列':>7} {'FLOP':>14} {'权重字节':>14} "
          f"{'算术强度':>10} {'落在哪一侧':>12}")
    bal = PEAK_TF * 1e12 / (PEAK_BW * 1e9)
    for name, B, S in [("prefill", 1, 2048), ("prefill", 1, 4096),
                       ("decode", 1, 1), ("decode", 8, 1),
                       ("decode", 64, 1), ("decode", 256, 1)]:
        flop = 2 * P * B * S
        wbytes = P * 2                 # 权重只读一遍，与 batch 无关
        ai = flop / wbytes
        print(f"  {name:<8} {B:>6} {S:>7} {flop:>14,.0f} {wbytes:>14,} "
              f"{ai:>10.1f} {'算力' if ai > bal else '带宽':>12}")
    print(f"\n  **prefill 天然在算力一侧，decode 在带宽一侧。**")
    print(f"  decode 要靠 batch 把权重读取摊薄：batch≈{bal / 2:.0f} 时越过平衡点。")
    print("  （这里只算权重，没算 KV —— KV 的账见 3.3，它不随 batch 摊薄。）")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] batch 对两相的影响完全不同")

    import random
    from vllm import SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(0)
    llm = make_llm()

    sub("prefill：加 batch 几乎不提高吞吐（已经算力受限）")
    S = 1024
    print(f"  每条 prompt {S} token，只生成 1 个 token")
    print(f"  {'batch':>6} {'总耗时 ms':>11} {'每条 ms':>10} "
          f"{'prompt tok/s':>13} {'相对 batch=1':>13}")
    base = None
    for B in [1, 2, 4, 8, 16, 32]:
        ps = [TokensPrompt(prompt_token_ids=rand_prompt(tok, S, rng))
              for _ in range(B)]
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate(ps, sp, use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate(ps, sp, use_tqdm=False)
        dt = (time.perf_counter() - t0) * 1000
        tp = B * S / (dt / 1000)
        if base is None:
            base = tp
        print(f"  {B:>6} {dt:>11.1f} {dt / B:>10.1f} {tp:>13.0f} "
              f"{tp / base:>12.2f}×")

    sub("decode：加 batch 吞吐几乎线性上涨（带宽被摊薄）")
    print(f"  每条 prompt 128 token，生成 64 token；只看 decode 部分")
    print(f"  {'batch':>6} {'总耗时 ms':>11} {'每步 ms':>10} "
          f"{'output tok/s':>13} {'相对 batch=1':>13}")
    base = None
    for B in [1, 2, 4, 8, 16, 32, 64]:
        ps = [TokensPrompt(prompt_token_ids=rand_prompt(tok, 128, rng))
              for _ in range(B)]
        sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        sp2 = SamplingParams(max_tokens=65, temperature=0.0, ignore_eos=True)
        llm.generate(ps, sp2, use_tqdm=False)
        t0 = time.perf_counter(); llm.generate(ps, sp1, use_tqdm=False)
        t_1 = time.perf_counter() - t0
        t0 = time.perf_counter(); llm.generate(ps, sp2, use_tqdm=False)
        t_65 = time.perf_counter() - t0
        per_step = (t_65 - t_1) / 64 * 1000
        tp = B / (per_step / 1000)
        if base is None:
            base = tp
        print(f"  {B:>6} {t_65 * 1000:>11.1f} {per_step:>10.3f} {tp:>13.0f} "
              f"{tp / base:>12.2f}×")
    print("\n  两张表放一起就是引擎所有调度决策的出发点：")
    print("  **prefill 不需要凑批，decode 极其需要凑批。**")
    shutdown(llm)


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 一个请求的时间构成：TTFT 与 TPOT")

    import random
    from vllm import SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(1)
    llm = make_llm()

    print("  TTFT = time to first token（含排队 + prefill）")
    print("  TPOT = time per output token（decode 每步）")
    print(f"\n  {'prompt 长度':>11} {'生成':>6} {'TTFT ms':>10} {'TPOT ms':>10} "
          f"{'总时长 ms':>11} {'TTFT 占比':>10}")
    for S in [128, 512, 2048, 4096]:
        for G in [64]:
            p = [TokensPrompt(prompt_token_ids=rand_prompt(tok, S, rng))]
            sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            spG = SamplingParams(max_tokens=G + 1, temperature=0.0, ignore_eos=True)
            llm.generate(p, spG, use_tqdm=False)
            t0 = time.perf_counter(); llm.generate(p, sp1, use_tqdm=False)
            ttft = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter(); llm.generate(p, spG, use_tqdm=False)
            tot = (time.perf_counter() - t0) * 1000
            tpot = (tot - ttft) / G
            print(f"  {S:>11} {G:>6} {ttft:>10.2f} {tpot:>10.3f} "
                  f"{tot:>11.1f} {ttft / tot:>9.1%}")
    print("\n  prompt 越长，TTFT 越大而 TPOT 基本不变（3.3 说过 decode 由 KV 决定，")
    print("  这里 KV 还很小）。所以「长 prompt + 短输出」的负载 TTFT 占主导，")
    print("  「短 prompt + 长输出」的负载 TPOT 占主导。**两者要用不同的指标衡量。**")
    shutdown(llm)


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 长度不齐的浪费")

    import random
    from vllm import SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(2)
    llm = make_llm()

    print("  静态批处理：一批一起开始、一起结束，最长的那条决定整批时长。")
    print("  连续批处理（continuous batching）：谁结束谁就退出，空位立刻补新请求。")
    print("  这里量一下「长度不齐」造成的浪费有多大。")

    B = 16
    for label, lens in [("长度一致（都生成 64）", [64] * B),
                        ("长度不齐（1..128 均匀）",
                         [rng.randint(1, 128) for _ in range(B)])]:
        ps = [TokensPrompt(prompt_token_ids=rand_prompt(tok, 128, rng))
              for _ in range(B)]
        sps = [SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
               for n in lens]
        llm.generate(ps, sps, use_tqdm=False)
        t0 = time.perf_counter()
        outs = llm.generate(ps, sps, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"\n  {label}")
        print(f"    实际生成 {ntok} token，耗时 {dt * 1000:.1f} ms，"
              f"吞吐 {ntok / dt:.0f} tok/s")
        print(f"    最长 {max(lens)}，若静态批处理要按最长算："
              f"{B * max(lens)} 个 step 的位置")
        print(f"    利用率 = {ntok} / {B * max(lens)} = "
              f"{ntok / (B * max(lens)):.1%}")
    print("\n  读这两行要小心，它们说的是两件事：")
    print("  1. 「利用率」是**静态批处理**会浪费掉的比例（不齐时只有约一半）。")
    print("  2. 但两行的**吞吐**也差了近一倍，而 vLLM 已经是连续批处理 ——")
    print("     原因是这里**没有排队的新请求可以补位**。批在跑的过程中不断有请求")
    print("     结束，活跃数从 16 一路掉到 1，最后几步是在按 batch=1 的速度跑。")
    print("  **连续批处理消除的是「等最慢的那条」，不是「没活干」。**")
    print("  它的收益要在有队列的开环压测下才看得到（5.3 / L8.3）。")

    sub("补位实验：给它一个队列")
    ps = [TokensPrompt(prompt_token_ids=rand_prompt(tok, 128, rng))
          for _ in range(64)]
    lens = [rng.randint(1, 128) for _ in range(64)]
    sps = [SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
           for n in lens]
    llm.generate(ps[:4], sps[:4], use_tqdm=False)
    t0 = time.perf_counter()
    outs = llm.generate(ps, sps, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  一次提交 64 条（长度同样 1..128 不齐）")
    print(f"    实际生成 {ntok} token，耗时 {dt * 1000:.1f} ms，"
          f"吞吐 {ntok / dt:.0f} tok/s")
    print(f"    对照上面 16 条不齐时的 1710 tok/s —— 有更多请求可调度时，")
    print(f"    早结束的请求空出的位置立刻被后面的补上，吞吐回来了。")
    shutdown(llm)


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  model {MODEL}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
