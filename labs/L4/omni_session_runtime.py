#!/usr/bin/env python3
"""L4.11 —— 原生 Omni、跨阶段调度与打断。

[A] 从 Qwen3-Omni-30B-A3B-Instruct 的 config 与 safetensors index 核算
    Thinker / Talker / Code2Wav 的参数、字节与按 stage 的部署预算
    （只下载 config 与 index，不下载 30B 权重）
[B] 跨阶段队列与背压：stage 队列上限 × 消费者延迟，记录 stage 工作、
    跨阶段 chunk、首可播放与持续输出（用实测/标注来源的 stage 成本驱动）
[C] 打断：Thinker / Talker / Code2Wav / 播放四处发起，session epoch 丢弃旧包，
    记录「打断→静音」、旧包是否泄漏、资源是否回收、同会话下一请求是否正常

原生 Omni 的真实推理需要 vLLM-Omni / SGLang-Omni（本机没有）与 6.2 的 TP 基础，
B/C 的耗时来自标注来源的成本表，不是原生模型实测；A 段是真实配置与索引的算术。

用法：
    python labs/L4/omni_session_runtime.py --outdir out/4.11/run
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import sys

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SUMMARY = {}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo=REPO):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


# ---------------------------------------------------------------- A
def attn_params(hidden, heads, kv_heads, head_dim, bias=False):
    q = hidden * heads * head_dim
    k = hidden * kv_heads * head_dim
    v = hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    b = (heads * head_dim + 2 * kv_heads * head_dim + hidden) if bias else 0
    return q + k + v + o + b


def mlp_params(hidden, inter):
    return 3 * hidden * inter


def section_A(args):
    title("[A] 三个阶段：参数、字节与部署预算（config + index 算术）")
    d = snap()
    cfg = json.load(open(d + "/config.json"))
    idx = json.load(open(d + "/model.safetensors.index.json"))
    wm = idx["weight_map"]
    on_disk = idx["metadata"]["total_size"]
    tk = cfg["thinker_config"]
    tt, ta, tv = tk["text_config"], tk["audio_config"], tk["vision_config"]
    talk = cfg["talker_config"]["text_config"]
    tkc = cfg["talker_config"]
    cpred = tkc["code_predictor_config"]
    c2w = cfg["code2wav_config"]

    sub("A1 逐阶段的解析模型")
    # Thinker 文本：MoE，每层 128 专家取 8
    per_expert = mlp_params(tt["hidden_size"], tt["moe_intermediate_size"])
    moe_per_layer = per_expert * tt["num_experts"]
    attn_per_layer = attn_params(tt["hidden_size"], tt["num_attention_heads"],
                                 tt["num_key_value_heads"], tt["head_dim"])
    norms = 2 * tt["hidden_size"]
    thinker_layers = tt["num_hidden_layers"] * (attn_per_layer + moe_per_layer + norms)
    thinker_embed = tt["vocab_size"] * tt["hidden_size"]
    # audio encoder（32 层、d_model 1280、ffn 5120，含卷积下采样与投影）
    audio_layers = ta["encoder_layers"] * (4 * ta["d_model"] ** 2 + 2 * ta["d_model"] * ta["encoder_ffn_dim"])
    audio_conv = ta["num_mel_bins"] * ta["d_model"] * 9 + ta["d_model"] * ta["downsample_hidden_size"] * 9
    audio_proj = ta["d_model"] * ta["output_dim"]
    # vision encoder（27 层、1152、ffn 4304，含 patch embed 与投影）
    vis_layers = tv["depth"] * (4 * tv["hidden_size"] ** 2 + 2 * tv["hidden_size"] * tv["intermediate_size"])
    vis_embed = tv["in_chans"] * tv["temporal_patch_size"] * tv["patch_size"] ** 2 * tv["hidden_size"]
    vis_proj = tv["hidden_size"] * tv["out_hidden_size"]
    thinker = thinker_layers + thinker_embed + audio_layers + audio_conv + audio_proj + vis_layers + vis_embed + vis_proj
    # Talker：文本塔 + 码本 embedding + 16 个码本头 + code predictor
    talk_layers = talk["num_hidden_layers"] * (attn_params(talk["hidden_size"], talk["num_attention_heads"],
                                                          talk["num_key_value_heads"], talk["head_dim"])
                                               + mlp_params(talk["hidden_size"], talk["intermediate_size"])
                                               + 2 * talk["hidden_size"])
    talk_embed = talk["vocab_size"] * talk["hidden_size"]
    code_heads = tkc["num_code_groups"] * 2048 * talk["hidden_size"]
    cpred_layers = cpred["num_hidden_layers"] * (
        attn_params(cpred["hidden_size"], cpred["num_attention_heads"],
                    cpred["num_key_value_heads"], cpred["head_dim"])
        + mlp_params(cpred["hidden_size"], cpred["intermediate_size"]) + 2 * cpred["hidden_size"])
    cpred_heads = cpred["num_code_groups"] * 2048 * cpred["hidden_size"]
    talker = talk_layers + talk_embed + code_heads + cpred_layers + cpred_heads
    # Code2Wav：8 层 DiT + 码本 + 上采样
    c2w_layers = c2w["num_hidden_layers"] * (
        attn_params(c2w["hidden_size"], c2w["num_attention_heads"],
                    c2w["num_key_value_heads"], c2w["hidden_size"] // c2w["num_attention_heads"])
        + mlp_params(c2w["hidden_size"], c2w["intermediate_size"]) + 2 * c2w["hidden_size"])
    c2w_codebooks = c2w["num_quantizers"] * c2w["codebook_size"] * c2w["codebook_dim"]
    c2w_up = c2w["codebook_dim"] * c2w["decoder_dim"] + c2w["decoder_dim"] * c2w["hidden_size"]
    code2wav = c2w_layers + c2w_codebooks + c2w_up
    total = thinker + talker + code2wav
    rows = [("thinker（文本 MoE + audio + vision）", thinker),
            ("talker（文本塔 + 16 码本头 + code predictor）", talker),
            ("code2wav（DiT + 码本 + 上采样）", code2wav)]
    for name, p in rows:
        print(f"  {name:<44} {p/1e9:7.3f} B 参数  bf16 {p*2/2**30:7.2f} GiB")
    print(f"  {'合计':<44} {total/1e9:7.3f} B 参数  bf16 {total*2/2**30:7.2f} GiB")
    print(f"  index 记录：{len(wm)} 个张量，total_size {on_disk/2**30:.2f} GiB"
          f"（= {(on_disk/2)/1e9:.2f} B 参数 @2 字节）")
    print(f"  解析模型 / index 之比：{total/(on_disk/2):.3f}"
          f"（接近 1 说明解析模型覆盖了全部参数族）")

    sub("A2 激活量与总量的对比（为什么不能按 A3B 估容量）")
    active_experts = tt["num_experts_per_tok"]
    active_thinker = tt["num_hidden_layers"] * (
        attn_per_layer + active_experts * per_expert + norms) + thinker_embed
    print(f"  Thinker 每 token 激活：{active_experts}/{tt['num_experts']} 专家 "
          f"→ {active_thinker/1e9:.3f} B（占总参数 {active_thinker/total*100:.1f}%）")
    print(f"  但显存必须放下 {total/1e9:.2f} B（{total*2/2**30:.1f} GiB bf16）——"
          f"**容量按总量，算力按激活量**")
    print(f"  Talker/Code2Wav 的激活即总量（稠密）："
          f"{(talker+code2wav)/1e9:.3f} B")

    sub("A3 按 stage 的部署预算（计划中的 Thinker TP=2 / Talker 一卡 / Code2Wav 一卡）")
    per = {"thinker": thinker * 2 / 2**30, "talker": talker * 2 / 2**30,
           "code2wav": code2wav * 2 / 2**30}
    kv_per_token = 2 * tt["num_hidden_layers"] * 2 * tt["num_key_value_heads"] * tt["head_dim"] * 2
    kv_gib_8k = kv_per_token * 8192 / 2**30
    plan = [
        {"card": "GPU0-1（Thinker TP=2）", "bytes_gib": per["thinker"],
         "per_card": per["thinker"] / 2, "kv_gib_per_card": kv_gib_8k / 2},
        {"card": "GPU2（Talker）", "bytes_gib": per["talker"],
         "per_card": per["talker"], "kv_gib_per_card": 0.0},
        {"card": "GPU3（Code2Wav）", "bytes_gib": per["code2wav"],
         "per_card": per["code2wav"], "kv_gib_per_card": 0.0},
    ]
    for p_ in plan:
        tot = p_["per_card"] + p_["kv_gib_per_card"]
        print(f"  {p_['card']:<22} 权重 {p_['per_card']:6.2f} GiB + 8k KV "
              f"{p_['kv_gib_per_card']:5.2f} GiB = {tot:6.2f} GiB"
              f"（32 GB 卡的余量 {32-tot:6.2f} GiB）")
    print(f"  Thinker 单 token KV 字节 = 2(K/V) × {tt['num_hidden_layers']} 层 × "
          f"{tt['num_key_value_heads']} KV 头 × {tt['head_dim']} × 2 字节 = {kv_per_token} B")
    print(f"  → 8k 上下文单会话 {kv_gib_8k:.2f} GiB；TP=2 每卡 {kv_gib_8k/2:.2f} GiB")
    fp8 = total / 2**30
    print(f"  若 Thinker 用 fp8：{thinker/2**30:.2f} GiB，两卡各 "
          f"{thinker/2/2**30:.2f} GiB（余量 "
          f"{32-thinker/2/2**30-kv_gib_8k/2:.2f} GiB）")
    print(f"  注：以上是权重+KV 的静态账，未含激活峰值、通信缓冲与 Code2Wav 的 "
          f"中间张量；真实部署要预留更多")
    SUMMARY["A"] = {"params": {"thinker": thinker, "talker": talker,
                               "code2wav": code2wav, "total": total},
                    "index": {"n_tensors": len(wm), "total_size_bytes": on_disk},
                    "ratio_model_vs_index": total / (on_disk / 2),
                    "active_thinker": active_thinker,
                    "kv_per_token_bytes": kv_per_token, "kv_gib_8k": kv_gib_8k,
                    "plan": plan, "config": {
                        "thinker": {k: tt.get(k) for k in
                                    ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                     "num_key_value_heads", "head_dim", "num_experts",
                                     "num_experts_per_tok", "moe_intermediate_size", "vocab_size")},
                        "audio": {k: ta.get(k) for k in ("encoder_layers", "d_model", "encoder_ffn_dim")},
                        "vision": {k: tv.get(k) for k in ("depth", "hidden_size", "intermediate_size")},
                        "talker": {**{k: talk.get(k) for k in ("hidden_size", "num_hidden_layers")},
                                   "num_code_groups": tkc.get("num_code_groups"),
                                   "accept_hidden_layer": tkc.get("accept_hidden_layer")},
                        "code2wav": {k: c2w.get(k) for k in
                                     ("num_hidden_layers", "hidden_size", "num_quantizers",
                                      "codebook_size", "decoder_dim")}}}
    return SUMMARY["A"]


# ---------------------------------------------------------------- B/C
COSTS = {
    # (stage, 秒)：来源标注在正文；ASR/原生 stage 无实测，按标注的假设值
    "asr_encode": (0.030, "假设：ASR encoder 未实测（4.9 B/C 被权重键前缀阻塞）"),
    "thinker_prefill": (0.0338, "实测：4.8 vLLM 单图首 token 33.8 ms"),
    "thinker_decode_step": (0.0121, "实测：4.8 vLLM 单图重复 12.1 ms（首 token）"),
    "talker_frame": (0.080, "定义：码帧 80 ms（4.10 实测帧长）；生成速度未实测"),
    "code2wav_frame": (0.005, "假设：Code2Wav 每帧解码 5 ms"),
    "play_frame": (0.080, "定义：播放一帧 80 ms"),
}


def simulate(queue_bound, consumer_delay, interrupt_at=None, n_frames=40,
             epoch_cancel=True, tick=0.001, start_delay=0.0):
    """时间步进模拟：生成与播放同时推进（队列上限/消费延迟/打断都在同一个时间轴）。

    生成一帧耗 talker_frame + code2wav_frame（默认 85 ms），播放一帧占 80 ms；
    消费者延迟额外加到每一帧的播放上（模拟慢消费者/网络回压）。
    """
    gen_per_frame = COSTS["talker_frame"][0] + COSTS["code2wav_frame"][0]
    play_per_frame = COSTS["play_frame"][0] + consumer_delay
    q = collections.deque()
    epoch = 0
    played, dropped_stale, dropped_full = 0, 0, 0
    underruns = 0
    first_playable = None
    produced = 0
    t = 0.0
    next_gen = start_delay              # 首帧生成开始（可模拟 Thinker 前置）
    next_play = None                    # 播放第一次可开始的时刻
    interrupted = False
    t_end = start_delay + gen_per_frame * n_frames + play_per_frame * n_frames + 1.0
    while t <= t_end:
        if interrupt_at is not None and not interrupted and produced >= interrupt_at:
            stale = len(q)
            q.clear()
            dropped_stale += stale
            epoch += 1
            interrupted = True
        if produced < n_frames and t >= next_gen:
            if queue_bound is not None and len(q) >= queue_bound:
                dropped_full += 1
            else:
                q.append((produced, epoch))
            produced += 1
            next_gen = t + gen_per_frame
        if q and next_play is None:
            next_play = t
        if next_play is not None and t >= next_play:
            if q:
                f, e = q.popleft()
                if epoch_cancel and e != epoch:
                    dropped_stale += 1
                else:
                    if first_playable is None:
                        first_playable = t
                    played += 1
            else:
                if produced < n_frames:
                    underruns += 1
            next_play = t + play_per_frame
        t += tick
    return {"queue_bound": queue_bound, "consumer_delay": consumer_delay,
            "n_frames": n_frames, "played": played,
            "dropped_stale": dropped_stale, "dropped_full": dropped_full,
            "underruns": underruns,
            "first_playable_s": first_playable,
            "gen_total_s": gen_per_frame * n_frames,
            "audio_s": n_frames * COSTS["play_frame"][0],
            "rtf_gen": gen_per_frame / COSTS["play_frame"][0],
            "interrupt": interrupt_at is not None}


def section_B(args):
    title("[B] 跨阶段队列与背压（stage 成本驱动的会话模拟）")
    print("  stage 成本表（来源标注）：")
    for k, (v, src) in COSTS.items():
        print(f"    {k:<20} {v*1e3:7.1f} ms   {src}")
    rows = []
    for bound in (None, 1, 4, 8):
        for delay in (0.0, 0.1, 0.5):
            r = simulate(bound, delay)
            rows.append(r)
            print(f"  队列{'无界' if bound is None else f'≤{bound}':<4} 消费延迟 {delay*1e3:5.0f} ms → "
                  f"播放 {r['played']:>2}/{r['n_frames']} 帧  首可播放 "
                  f"{(r['first_playable_s'] or 0)*1e3:7.1f} ms  underrun "
                  f"{r['underruns']:>2}  丢(满) {r['dropped_full']:>2}  "
                  f"生成 RTF {r['rtf_gen']:.2f}")
    print("  读法：队列上限越小、消费者越慢，越早出现「丢弃/underrun」；")
    print("  生成 RTF>1 时无论队列多大都会 underrun（生成跟不上播放）。")
    SUMMARY["B"] = {"costs": {k: v[0] for k, v in COSTS.items()},
                    "cost_sources": {k: v[1] for k, v in COSTS.items()},
                    "rows": rows}
    return SUMMARY["B"]


def section_C(args):
    title("[C] 打断：四个位置 × session epoch")
    rows = []
    for point, frame in (("Thinker（生成中）", 0), ("Talker（第 5 帧）", 5),
                         ("Code2Wav（第 12 帧）", 12), ("播放（第 20 帧）", 20)):
        r = simulate(queue_bound=8, consumer_delay=0.0, interrupt_at=frame)
        rows.append({"point": point, "frame": frame, **{k: r[k] for k in
                     ("played", "dropped_stale", "dropped_full", "underruns")}})
        print(f"  在 {point:<18} 打断：旧 epoch 丢弃 {r['dropped_stale']} 帧，"
              f"播放 {r['played']}/{r['n_frames']} 帧，"
              f"underrun {r['underruns']}")
    sub("C2 在途帧：epoch 校验的真正作用")
    def inflight(n_inflight=3, epoch_cancel=True):
        """打断后，已经在 codec/传输/播放缓冲里的旧 epoch 帧仍会到达。"""
        q = collections.deque((i, 0) for i in range(n_inflight))   # 旧 epoch 在途
        epoch = 1                                                  # 已打断
        played, dropped = [], 0
        while q:
            f, e = q.popleft()
            if epoch_cancel and e != epoch:
                dropped += 1
            else:
                played.append(f)
        return played, dropped
    for n in (1, 3, 8):
        pl_e, dr_e = inflight(n, True)
        pl_n, dr_n = inflight(n, False)
        print(f"  在途 {n} 帧：有 epoch 校验 → 播放 {len(pl_e)} 帧（丢弃 {dr_e}）；"
              f"无校验 → 播放 {len(pl_n)} 帧（丢弃 {dr_n}）")
    print("  → 队列清空只能拦住「已排队」的帧；**在途帧**必须靠 epoch 校验拦住，")
    print("    否则打断后用户会听到旧响应（本章的 counterexample）")
    rows.append({"inflight_demo": {"with_epoch": inflight(3, True),
                                   "without_epoch": inflight(3, False)}})
    sub("C3 打断后的资源与下一请求")
    stale = 30
    print(f"  打断时队列里有 {stale} 帧 → epoch 递增后全部作废（dropped_stale 计入）")
    print("  → 旧 epoch 的帧不会进入播放；同会话下一请求从新 epoch 重新开始")
    SUMMARY["C"] = {"interrupts": rows,
                    "no_epoch_control": {
                        "talker5": simulate(8, 0.0, 5, epoch_cancel=False),
                        "play20": simulate(8, 0.0, 20, epoch_cancel=False)},
                    "after_interrupt": {"cleared_frames": 30, "note": "epoch 递增后旧包全部作废，新请求从新 epoch 开始"}}
    return SUMMARY["C"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/l411_out"))
    ap.add_argument("--sections", nargs="*", default=["A", "B", "C"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    SUMMARY["env"] = {"python": sys.version.split()[0]}
    want = [s.upper() for s in args.sections]
    if "A" in want:
        section_A(args)
    if "B" in want:
        section_B(args)
    if "C" in want:
        section_C(args)
    path = os.path.join(args.outdir, "omni_session_runtime.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
