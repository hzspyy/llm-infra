#!/usr/bin/env python3
"""L4.2 修订 —— 数值系统与确定性。

对应修订任务：
  [A] 格式枚举与精度分层：位模式全枚举（可表示数、次正规、溢出、舍入），
      把「输入格式 / 乘法精度 / 累加精度」三类误差分开测量；极值、抵消、
      长归约对 FP64 参照；给出平均/最大误差与误差比的定义
  [B] 配对 batch 不变性：固定 200 条输入，batch=1/2/8/32、位置轮换与 padding 下
      保存 top-5 logits、top-2 margin、argmax 翻转与贪心生成差异
  [C] 后端与确定性范围：eager/compile × TF32 开关 × deterministic 开关 × 重复运行，
      把第一处差异定位到层，并与引擎 logprob 对拍

误差定义（全章统一）：
  平均绝对误差 = mean(|x_i − r_i|)；最大绝对误差 = max(|x_i − r_i|)；
  最大相对误差 = 最大绝对误差 / max(|r_i|)；误差比 = 最大绝对误差 / 平均绝对误差。

用法：
    python labs/L4/numerics_determinism.py --outdir out/4.2/run A B C
    python labs/L4/numerics_determinism.py --outdir out/4.2/run C --skip-vllm
"""

import argparse
import glob
import json
import math
import os
import sys
import time

SUMMARY = {"sections": {}}
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


def _err(x, ref):
    import torch
    d = (x.double() - ref.double()).abs()
    denom = ref.double().abs().max().clamp_min(1e-30)
    mean_abs = d.mean().item()
    max_abs = d.max().item()
    ratio = float("inf") if mean_abs == 0 else max_abs / mean_abs
    return mean_abs, max_abs, (max_abs / denom.item()), ratio


# ------------------------------------------------------------------ A
# (名称, 符号位, 指数位, 尾数位)。float32 不参与位模式全枚举（2^32 个编码）。
FORMATS = [
    ("bfloat16", 1, 8, 7),
    ("float16", 1, 5, 10),
    ("float8_e4m3fn", 1, 4, 3),
    ("float8_e5m2", 1, 5, 2),
]


def _enumerate(dtype, sbits, ebits, mbits):
    """按位模式枚举一个格式的全部编码。"""
    import torch
    total = sbits + ebits + mbits
    codes = torch.arange(1 << total, dtype=torch.int32)
    if total == 16:
        raw = codes.to(torch.int16).view(dtype)
    else:
        raw = codes.to(torch.uint8).view(dtype)
    vals = raw.float()
    exp = (codes >> mbits) & ((1 << ebits) - 1)
    man = codes & ((1 << mbits) - 1)
    inf_mask = (exp == (1 << ebits) - 1) & (man == 0)
    nan_mask = (exp == (1 << ebits) - 1) & (man != 0)
    zero_mask = (exp == 0) & (man == 0)
    sub_mask = (exp == 0) & (man != 0)
    normal_mask = (exp > 0) & (exp < (1 << ebits) - 1)
    # inf/nan 按解码后的数值判定，而不是按指数字段：fp8_e4m3fn 的
    # 「指数全 1 且尾数 0」编码也是 nan（fn = finite only，没有 inf 编码）。
    is_nan = torch.isnan(vals)
    is_inf = torch.isinf(vals)
    is_zero = (vals == 0)
    n_nan, n_inf, n_zero = int(is_nan.sum()), int(is_inf.sum()), int(is_zero.sum())
    n_sub = int(sub_mask.sum())
    cat = {"zero": n_zero, "subnormal": n_sub,
           "normal": int(vals.numel() - n_nan - n_inf - n_zero - n_sub),
           "inf": n_inf, "nan": n_nan}
    binades = {}
    for e in range(1, (1 << ebits) - 1):
        n = int((exp == e).sum())
        if n:
            binades[e] = n
    return {"vals": vals, "cat": cat, "exp": exp, "man": man,
            "sub_mask": sub_mask, "binades": binades,
            "finite": torch.isfinite(vals)}


def section_A(outdir):
    import torch
    rep = {}
    title("[A] 格式枚举与精度分层")

    sub("A1 位模式全枚举：每个格式能表示多少个数")
    print(f"  {'格式':<14} {'位':>3} {'零':>4} {'次正规':>7} {'正规':>7} "
          f"{'inf':>4} {'nan':>4} {'最大有限':>12} {'最小正规':>11} {'最小次正规':>11}")
    enum = {}
    for name, s_, e_, m_ in FORMATS:
        dt = getattr(torch, name)
        info = _enumerate(dt, s_, e_, m_)
        finite = info["vals"][info["finite"]]
        fmax = finite.abs().max().item()
        norm = info["vals"][(info["exp"] > 0) & (info["exp"] < (1 << e_) - 1)]
        nmin = norm.abs().min().item() if norm.numel() else float("nan")
        subv = info["vals"][info["sub_mask"]]
        smin = subv.abs().min().item() if subv.numel() else 0.0
        c = info["cat"]
        print(f"  {name:<14} {s_ + e_ + m_:>3} {c['zero']:>4} {c['subnormal']:>7} "
              f"{c['normal']:>7} {c['inf']:>4} {c['nan']:>4} {fmax:>12.5g} "
              f"{nmin:>11.4g} {smin:>11.4g}")
        enum[name] = {"bits": s_ + e_ + m_, "categories": c, "max_finite": fmax,
                      "min_normal": nmin, "min_subnormal": smin,
                      "binades": len(info["binades"]),
                      "values_per_binade": sorted(set(info["binades"].values()))}
    fi32 = torch.finfo(torch.float32)
    print(f"  {'float32':<14} {32:>3} {'—':>4} {'—':>7} {'—':>7} {'1':>4} {'—':>4} "
          f"{fi32.max:>12.5g} {fi32.tiny:>11.4g} {'—':>11}")
    enum["float32"] = {"bits": 32, "max_finite": fi32.max,
                       "min_normal": fi32.tiny, "eps": fi32.eps,
                       "enumerated": False}
    rep["enumeration"] = enum
    print("  fp8_e4m3fn 没有 inf 编码（fn = finite only），溢出按 nan 处理；")
    print("  fp8_e5m2 与 fp16 有 inf。次正规的个数等于尾数空间大小：")
    print("  每个指数从一个 binade 到下一个之间不是跳变，而是逐格填满。")

    sub("A2 舍入：ties-to-even、截断对照与溢出/次正规")
    eps16 = torch.finfo(torch.bfloat16).eps
    bf16_one = torch.tensor([1.0]).bfloat16()
    # bf16 在 [1,2) 上的间距就是 eps=2^-7；用 1+eps 得到下一个可表示数
    nxt = torch.tensor([1.0 + eps16]).bfloat16()
    spacing = (nxt.float() - bf16_one.float()).item()
    mid = bf16_one.float() + spacing / 2
    print(f"  bf16 在 1.0 附近：相邻间距 {spacing:.8g}（= eps），可表示 "
          f"{bf16_one.float().item():.8g} / {nxt.float().item():.8g}")
    print(f"  两数中点 {mid.item():.8g} 舍入到 {mid.bfloat16().float().item():.8g}"
          f"（ties-to-even 落到尾数最低位为偶的那个）")
    torch.manual_seed(0)
    x = torch.randn(200000)
    trunc = (x.view(torch.int32) & ~0xFFFF).view(torch.float32).bfloat16()
    rne = x.bfloat16()
    diff = (trunc.float() != rne.float()).float().mean().item()
    worst = (trunc.float() - x).abs().max().item()
    print(f"  20 万个随机 fp32：截断低 16 位与库转换不同的比例 {diff:.4f}，"
          f"截断的最大绝对误差 {worst:.6g}")
    print("  库转换是 round-to-nearest-even，截断会系统性地偏向零，"
          "两者在长链计算里积累成不同的偏差。")
    cases = [65504.0, 65519.0, 65520.0, 65536.0, 1e5, 6.104e-5, 5.96e-8, 3e-8, 1e-8]
    print(f"\n  {'输入':>12} {'fp16':>14} {'bf16':>14} {'fp8_e4m3fn':>14} {'fp8_e5m2':>14}")
    rows = []
    for v in cases:
        t = torch.tensor([v], dtype=torch.float32)
        row = [t.half().float().item(), t.bfloat16().float().item(),
               t.to(torch.float8_e4m3fn).float().item(),
               t.to(torch.float8_e5m2).float().item()]
        rows.append({"value": v, "fp16": row[0], "bf16": row[1],
                     "fp8_e4m3fn": row[2], "fp8_e5m2": row[3]})
        print(f"  {v:>12.6g} {row[0]:>14.6g} {row[1]:>14.6g} {row[2]:>14.6g} "
              f"{row[3]:>14.6g}")
    sub_bits = {}
    for v in (5.96e-8, 1e-8):
        b = torch.tensor([v], dtype=torch.float32).half().view(torch.int16).item() & 0xFFFF
        sub_bits[f"{v:.3g}"] = format(b, "016b")
    print(f"  最小次正规 fp16 = 2^-24 的位模式 {sub_bits['5.96e-08']}；"
          f"1e-8 的位模式 {sub_bits['1e-08']}（全零 = 下溢成 0）")
    rep["rounding"] = {"bf16_spacing_at_1": spacing, "midpoint": mid.item(),
                       "midpoint_rounds_to": mid.bfloat16().float().item(),
                       "trunc_vs_rne_diff_fraction": diff,
                       "trunc_max_abs_error": worst,
                       "overflow_and_subnormal": rows,
                       "fp16_subnormal_bits": sub_bits}

    sub("A3 三类误差分离：输入格式 / 乘法 / 累加")
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n, dtype=torch.float64)
    w = torch.randn(n, dtype=torch.float64) * 0.02
    ref = (x * w).sum()

    def quant(v, dt):
        return v.float().to(dt).float().double()

    input_only = {}
    for name in ("bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"):
        dt = getattr(torch, name)
        input_only[name] = _err(quant(x, dt) * quant(w, dt), x * w)
    print("  ① 只把输入量化到目标格式（乘法与累加在 FP64）——输入格式的贡献")
    print(f"  {'输入格式':<14} {'平均绝对误差':>14} {'最大绝对误差':>14} "
          f"{'最大相对误差':>14} {'误差比':>10}")
    for k, v in input_only.items():
        print(f"  {k:<14} {v[0]:>14.4e} {v[1]:>14.4e} {v[2]:>14.4e} {v[3]:>10.2f}")

    def seq_acc(v, dt):
        acc = torch.zeros((), dtype=dt)
        for i in range(v.numel()):
            acc = (acc + v[i].to(dt)).to(dt)
        return acc.double()

    acc_only = {}
    for name in ("float32", "bfloat16", "float16"):
        dt = getattr(torch, name)
        acc_only[name] = _err(seq_acc(x * w, dt), ref)
    print("\n  ② 只把累加降精度（输入保持 FP64，逐项顺序累加）——累加的贡献")
    for k, v in acc_only.items():
        print(f"  {k:<14} {v[0]:>14.4e} {v[1]:>14.4e} {v[2]:>14.4e} {v[3]:>10.2f}")

    both = {}
    for name in ("bfloat16", "float16"):
        dt = getattr(torch, name)
        both[name] = _err(seq_acc(quant(x, dt) * quant(w, dt), dt), ref)
    print("\n  ③ 输入与累加同时降到同一格式")
    for k, v in both.items():
        print(f"  {k:<14} {v[0]:>14.4e} {v[1]:>14.4e} {v[2]:>14.4e} {v[3]:>10.2f}")
    print("\n  ①与②分开才知道该升级哪一端：输入格式属于表示误差，"
          "降累加精度属于算法误差，两者不能互相替代。")
    rep["error_split"] = {"input_only": input_only, "accumulate_only": acc_only,
                          "both": both, "reference": ref.item(), "n_terms": n}

    sub("A4 抵消与长归约")
    a = torch.tensor([1e8, 1.0, -1e8])
    cancel = {}
    for name in ("float64", "float32", "bfloat16", "float16"):
        dt = getattr(torch, name)
        seq = torch.zeros((), dtype=dt)
        for i in range(a.numel()):
            seq = (seq + a[i].to(dt)).to(dt)
        kahan = torch.zeros((), dtype=dt)
        c = torch.zeros((), dtype=dt)
        for i in range(a.numel()):
            y = a[i].to(dt) - c
            t = kahan + y
            c = (t - kahan) - y
            kahan = t
        # Neumaier：按量级决定补偿量归到哪一侧
        s = torch.zeros((), dtype=dt)
        comp = torch.zeros((), dtype=dt)
        for i in range(a.numel()):
            xv = a[i].to(dt)
            t = s + xv
            if s.abs() >= xv.abs():
                comp = comp + ((s - t) + xv)
            else:
                comp = comp + ((xv - t) + s)
            s = t
        neumaier = (s + comp)
        cancel[name] = {"sequential": seq.double().item(),
                        "kahan": kahan.double().item(),
                        "neumaier": neumaier.double().item()}
        print(f"  [1e8, 1, -1e8] {name:<10} 顺序 {seq.item():>10.4g}   "
              f"Kahan {kahan.item():>10.4g}   Neumaier {neumaier.item():>10.4g}")
    print("  精确值是 1.0。顺序累加把 1.0 吞掉；Kahan 在这种量级悬殊的抵消里"
          "也救不回来（补偿量本身又被吞掉），Neumaier 按量级选择补偿方向才拿回来。"
          "fp16 连 1e8 都表示不了，直接 inf。")

    torch.manual_seed(1)
    m = 1 << 20
    big = torch.randn(m)
    exact = big.double().sum().item()
    blocks = 4096
    print(f"\n  2^20 项求和，FP64 参照 = {exact:.10f}（块内两两、块间顺序，块 {blocks}）")
    print(f"  {'累加格式':<12} {'结果':>20} {'平均绝对误差':>14} {'最大绝对误差':>14}")
    long_red = {}
    for name in ("float32", "bfloat16", "float16"):
        dt = getattr(torch, name)
        acc = torch.zeros((), dtype=dt)
        for i in range(0, m, blocks):
            acc = (acc + big[i:i + blocks].sum(dtype=dt)).to(dt)
        e = _err(acc.double().reshape(1),
                 torch.tensor([exact], dtype=torch.float64))
        long_red[name] = {"value": acc.double().item(), "mean_abs": e[0],
                          "max_abs": e[1]}
        print(f"  {name:<12} {acc.double().item():>20.6f} {e[0]:>14.4e} {e[1]:>14.4e}")
    rep["cancellation"] = cancel
    rep["long_reduction"] = {"terms": m, "block": blocks, "reference": exact,
                             "formats": long_red}
    SUMMARY["sections"]["A"] = rep
    return rep


# ------------------------------------------------------------------ B
PROMPTS = [
    "The capital of France is", "Water boils at",
    "def fibonacci(n):", "The largest planet in the solar system is",
    "In 1969, humans first landed on", "The chemical symbol for gold is",
    "Photosynthesis converts light into", "The speed of light in vacuum is",
    "A binary search runs in", "The Great Wall of China is located in",
    "The first law of thermodynamics states that", "Shakespeare wrote",
    "The boiling point of water at sea level is", "DNA stands for",
    "The capital of Japan is", "Machine learning models are trained by",
    "The largest ocean on Earth is", "A prime number is",
    "The theory of relativity was proposed by", "The human genome contains",
]


def _prompts(n):
    """确定性生成 n 条长度 16–48 的 prompt（不依赖外部语料）。"""
    return [f"[{i:03d}] {PROMPTS[i % len(PROMPTS)]}" + " and" * (i % 7)
            for i in range(n)]


def _load_model(repo, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=getattr(torch, dtype)).cuda().eval()
    return tok, model


def _topk(logits, k=5):
    import torch
    v, i = torch.topk(logits.float(), k)
    return v.tolist(), i.tolist()


def section_B(outdir):
    import torch
    rep = {}
    title("[B] 配对 batch 不变性：200 条输入 × batch/位置/padding")
    repo = os.environ.get("L42_MODEL", "Qwen/Qwen3-1.7B")
    tok, model = _load_model(repo)
    texts = _prompts(200)
    enc = [tok(t, return_tensors="pt").input_ids[0] for t in texts]
    print(f"  模型 {repo}，输入 {len(texts)} 条，长度 "
          f"{min(len(e) for e in enc)}–{max(len(e) for e in enc)} token，"
          f"左 padding（保证最后一个位置始终是真实 token）")

    ref = {}
    for i, ids in enumerate(enc):
        with torch.no_grad():
            lg = model(ids.unsqueeze(0).cuda()).logits[0, -1].float().cpu()
        v, idx = _topk(lg)
        ref[i] = {"top5_v": v, "top5_i": idx, "argmax": idx[0],
                  "margin": v[0] - v[1], "logits": lg}
    print("  参照完成：batch=1 逐条；记录 top-5 与 top-2 margin")

    def batched(idx_list):
        maxlen = max(len(enc[i]) for i in idx_list)
        ids = torch.full((len(idx_list), maxlen), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx_list), maxlen), dtype=torch.long)
        for r, i in enumerate(idx_list):
            ids[r, maxlen - len(enc[i]):] = enc[i]
            att[r, maxlen - len(enc[i]):] = 1
        with torch.no_grad():
            lg = model(ids.cuda(), attention_mask=att.cuda()).logits[:, -1]
        return lg.float().cpu()

    def compare(idx_list):
        lg = batched(idx_list)
        rows = []
        for r, i in enumerate(idx_list):
            r0 = ref[i]
            v, k = _topk(lg[r])
            rows.append({"sample": i, "position": r,
                         "max_abs_diff": (lg[r] - r0["logits"]).abs().max().item(),
                         "bitwise_equal": bool(torch.equal(lg[r], r0["logits"])),
                         "argmax_same": bool(k[0] == r0["argmax"]),
                         "ref_margin": r0["margin"], "margin": v[0] - v[1],
                         "top5_v": v, "top5_i": k})
        return rows

    results = {}
    order = list(range(len(texts)))
    for B in (1, 2, 8, 32):
        rows = []
        t0 = time.perf_counter()
        for s in range(0, len(order), B):
            rows += compare(order[s:s + B])
        dt = time.perf_counter() - t0
        flips = [r for r in rows if not r["argmax_same"]]
        bit = sum(1 for r in rows if r["bitwise_equal"])
        print(f"  batch={B:<3} 前向 {math.ceil(len(order) / B):>3} 次（{dt:5.1f} s）  "
              f"逐位相同 {bit:>3}/{len(rows)}  翻转 {len(flips):>2}  "
              f"max|diff| {max(r['max_abs_diff'] for r in rows):.3e}  "
              f"翻转样本参照 margin 最小 "
              f"{min((r['ref_margin'] for r in flips), default=float('nan')):.4f}")
        results[f"batch_{B}"] = rows

    sub("位置轮换：样本在批次里的位次是否影响结果")
    pos_rows = []
    probe = list(range(32))
    for pos in range(32):
        rotated = probe[pos:] + probe[:pos]
        r = compare(rotated)
        pos_rows.append({"probe_sample": rotated[0], "position": pos,
                         "max_abs_diff": r[0]["max_abs_diff"],
                         "bitwise_equal": r[0]["bitwise_equal"],
                         "argmax_same": r[0]["argmax_same"]})
    print(f"  32 个位次：逐位相同 {sum(p['bitwise_equal'] for p in pos_rows)}/32，"
          f"max|diff| {min(p['max_abs_diff'] for p in pos_rows):.3e}–"
          f"{max(p['max_abs_diff'] for p in pos_rows):.3e}")

    sub("padding：变长输入左 padding 到同一长度")
    sub_idx = list(range(8))
    padded = batched(sub_idx)
    pad_rows = []
    for r, i in enumerate(sub_idx):
        v, k = _topk(padded[r])
        pad_rows.append({"sample": i,
                         "max_abs_diff": (padded[r] - ref[i]["logits"]).abs().max().item(),
                         "bitwise_equal": bool(torch.equal(padded[r], ref[i]["logits"])),
                         "argmax_same": bool(k[0] == ref[i]["argmax"])})
    print(f"  左 padding 到 {max(len(enc[i]) for i in sub_idx)}：逐位相同 "
          f"{sum(p['bitwise_equal'] for p in pad_rows)}/8，翻转 "
          f"{sum(not p['argmax_same'] for p in pad_rows)}，"
          f"max|diff| {max(p['max_abs_diff'] for p in pad_rows):.3e}")

    sub("固定长度对照：所有输入左 padding 到同一长度，只改 batch 组成")
    L = max(len(e) for e in enc)
    ids_L = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
    att_L = torch.zeros((len(enc), L), dtype=torch.long)
    for r, e in enumerate(enc):
        ids_L[r, L - len(e):] = e
        att_L[r, L - len(e):] = 1
    refL = []
    for i in range(len(enc)):
        with torch.no_grad():
            lg = model(ids_L[i:i + 1].cuda(), attention_mask=att_L[i:i + 1].cuda())
        refL.append(lg.logits[0, -1].float().cpu())
    fixed = {}
    for B in (1, 2, 8, 32):
        rows = []
        for s in range(0, len(enc), B):
            sl = slice(s, s + B)
            with torch.no_grad():
                lg = model(ids_L[sl].cuda(), attention_mask=att_L[sl].cuda())
            lg = lg.logits[:, -1].float().cpu()
            for j, i in enumerate(range(s, min(s + B, len(enc)))):
                rows.append({"sample": i,
                             "max_abs_diff": (lg[j] - refL[i]).abs().max().item(),
                             "bitwise_equal": bool(torch.equal(lg[j], refL[i])),
                             "argmax_same": bool(lg[j].argmax().item()
                                                 == refL[i].argmax().item())})
        fixed[f"batch_{B}"] = rows
        print(f"  统一长度 {L}，batch={B:<3} 逐位相同 "
              f"{sum(r['bitwise_equal'] for r in rows):>3}/{len(rows)}  "
              f"翻转 {sum(not r['argmax_same'] for r in rows):>2}  "
              f"max|diff| {max(r['max_abs_diff'] for r in rows):.3e}")
    print("  这一组里所有序列的形状完全相同，唯一的变量是 batch 组成："
          "差异仍出现，说明它不来自 padding，而来自 GEMM 形状改变了归约切分。")

    sub("贪心生成：batch=1 与 batch=8 的 token 序列")
    n_new = 16
    gen_rows = []
    for i in range(8):
        ids = enc[i].unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
        gen_rows.append(out[0, len(enc[i]):].tolist())
    maxlen = max(len(enc[i]) for i in range(8))
    ids = torch.full((8, maxlen), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((8, maxlen), dtype=torch.long)
    for r, i in enumerate(range(8)):
        ids[r, maxlen - len(enc[i]):] = enc[i]
        att[r, maxlen - len(enc[i]):] = 1
    with torch.no_grad():
        out = model.generate(ids.cuda(), attention_mask=att.cuda(),
                             max_new_tokens=n_new, do_sample=False)
    batch8 = [out[r, maxlen:].tolist() for r in range(8)]
    same = sum(1 for a, b in zip(gen_rows, batch8) if a == b)
    first_diff = [next((j for j in range(n_new) if a[j] != b[j]), None)
                  for a, b in zip(gen_rows, batch8)]
    print(f"  8 条样本 × {n_new} token：序列完全相同 {same}/8，"
          f"首个分歧位置 {first_diff}")

    sub("翻转与 top-2 margin 的关系")
    all_rows = results["batch_32"]
    flips = [r for r in all_rows if not r["argmax_same"]]
    nf = [r for r in all_rows if r["argmax_same"]]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")

    def pctl(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

    print(f"  batch=32：翻转 {len(flips)} 条，参照 margin 中位数 "
          f"{med([r['ref_margin'] for r in flips]):.4f}；未翻转 {len(nf)} 条，"
          f"中位数 {med([r['ref_margin'] for r in nf]):.4f}")
    flip_by_margin = {}
    for thr in (0.01, 0.1, 0.5, 1.0):
        risky = [r for r in all_rows if r["ref_margin"] < thr]
        f = sum(1 for r in risky if not r["argmax_same"])
        flip_by_margin[str(thr)] = {"n": len(risky), "flips": f}
        print(f"  margin < {thr:<5}: {len(risky):>3} 条中翻转 {f}"
              f"（{f / max(1, len(risky)):.1%}）")
    print("\n  比翻转计数更稳的判据是「决策余量与偏差量级」：")
    for tag, rows in (("变长批次 batch=32", all_rows),
                      ("统一长度 batch=32", fixed["batch_32"])):
        dmax = max(r["max_abs_diff"] for r in rows)
        dm = [r["max_abs_diff"] for r in rows]
        refm = [r["ref_margin"] for r in results["batch_1"]]
        inside = sum(1 for m in refm if m < dmax)
        print(f"  {tag}: logits 偏差 p50 {pctl(dm, 0.5):.3e} / p90 {pctl(dm, 0.9):.3e} "
              f"/ max {dmax:.3e}")
        print(f"     参照 margin < {dmax:.3f}（即落在偏差量级内）的样本 "
              f"{inside}/{len(refm)}（{inside / len(refm):.1%}）——"
              f"这些样本的 top-1 无法由本配置保证")
    for r in flips:
        print(f"  翻转样本 #{r['sample']}：参照 margin {r['ref_margin']:.6f}，"
              f"参照 top-5 {r['top5_v'][:2]}，偏差 {r['max_abs_diff']:.3e}")

    rep["config"] = {"repo": repo, "dtype": "bfloat16", "n_samples": len(texts),
                     "padding_side": "left", "prompts": texts,
                     "generation_tokens": n_new,
                     "torch_flags": {
                         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "cudnn_deterministic": torch.backends.cudnn.deterministic}}
    rep["batch"] = results
    rep["fixed_len"] = {"length": L, "batches": fixed}
    rep["position"] = pos_rows
    rep["padding"] = pad_rows
    rep["generation"] = {"batch1": gen_rows, "batch8": batch8,
                         "identical": same, "first_divergence": first_diff}
    rep["flip_by_margin"] = {"flips": len(flips), "non_flips": len(nf),
                             "flip_margin_median": med([r["ref_margin"] for r in flips]),
                             "nonflip_margin_median": med([r["ref_margin"] for r in nf]),
                             "by_threshold": flip_by_margin,
                             "margin_below_deviation": {
                                 "batch32_variable_len": sum(
                                     1 for m in (r["ref_margin"] for r in results["batch_1"])
                                     if m < max(x["max_abs_diff"] for x in all_rows)),
                                 "batch32_fixed_len": sum(
                                     1 for m in (r["ref_margin"] for r in results["batch_1"])
                                     if m < max(x["max_abs_diff"] for x in fixed["batch_32"]))}}
    del model
    torch.cuda.empty_cache()
    SUMMARY["sections"]["B"] = rep
    return rep


# ------------------------------------------------------------------ C
def section_C(outdir, skip_vllm=False):
    import torch
    rep = {}
    title("[C] 后端与确定性：范围、第一处差异与 logprob 对拍")
    repo = os.environ.get("L42_MODEL", "Qwen/Qwen3-1.7B")
    tok, model = _load_model(repo)

    def set_flags(tf32, det):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.use_deterministic_algorithms(det, warn_only=True)

    set_flags(False, False)
    text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids.cuda()

    def logits():
        with torch.no_grad():
            return model(ids).logits[0, -1].float().cpu()

    sub("C1 同配置重复：位级确定性")
    for tag, tf32, det in (("tf32_off", False, False), ("tf32_on", True, False),
                           ("det_on", False, True)):
        set_flags(tf32, det)
        vals = [logits() for _ in range(5)]
        same = all(torch.equal(vals[0], v) for v in vals[1:])
        spread = max((v - vals[0]).abs().max().item() for v in vals[1:])
        print(f"  {tag:<10} 5 次逐位相同 {same}  最大差 {spread:.3e}")
        rep.setdefault("repeat", {})[tag] = {"bitwise_same": bool(same),
                                             "max_abs_diff": spread}
    set_flags(False, False)
    base = logits()

    sub("C2 TF32 与 deterministic 对 logits 与 argmax 的影响")
    diag = {}
    for tag, tf32, det in (("tf32_on", True, False), ("det_on", False, True)):
        set_flags(tf32, det)
        v = logits()
        d = (v - base).abs().max().item()
        margin = v.topk(2).values.diff().abs().item()
        print(f"  {tag:<10} vs 基线 max|diff| {d:.3e}  argmax 相同 "
              f"{v.argmax().item() == base.argmax().item()}  top-2 margin {margin:.4f}")
        diag[tag] = {"max_abs_diff": d,
                     "argmax_same": bool(v.argmax().item() == base.argmax().item()),
                     "top2_margin": margin}
    set_flags(False, False)
    rep["switches"] = diag
    print("  bf16 模型上这两个开关的差异都是 0：TF32 只作用于 fp32 矩阵乘，"
          "deterministic 只改原子类算子的实现。")
    sub("C2b TF32 的真实作用范围：fp32 矩阵乘")
    a32 = torch.randn(2048, 2048, device="cuda")
    b32 = torch.randn(2048, 2048, device="cuda")
    ref64 = (a32.double() @ b32.double())
    tf32_probe = {}
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        r = (a32 @ b32).double()
        d = (r - ref64).abs()
        rel = (d.max() / ref64.abs().max()).item()
        tf32_probe[f"allow_tf32_{tf32}"] = {"max_abs": d.max().item(),
                                            "max_rel_to_max": rel}
        print(f"  allow_tf32={tf32!s:<5} fp32 2048³ 矩阵乘 vs FP64："
              f"max|diff| {d.max().item():.3e}  相对最大 {rel:.3e}")
    torch.backends.cuda.matmul.allow_tf32 = False
    rep["tf32_probe"] = tf32_probe
    print("  TF32 把 fp32 输入压到 10 位尾数再算，相对误差到 1e-3 量级；"
          "模型用 bf16 权重时这条路径根本不参与，开关自然无效。")

    sub("C3 eager vs torch.compile")
    try:
        comp = torch.compile(model, mode="default", dynamic=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            v_comp = comp(ids).logits[0, -1].float().cpu()
        compile_s = time.perf_counter() - t0
        d = (v_comp - base).abs().max().item()
        reps = []
        for _ in range(4):
            with torch.no_grad():
                reps.append(comp(ids).logits[0, -1].float().cpu())
        print(f"  首次调用（含编译）{compile_s:.1f} s；max|diff| vs eager {d:.3e}  "
              f"argmax 相同 {v_comp.argmax().item() == base.argmax().item()}")
        print(f"  编译后 4 次重复逐位相同 "
              f"{all(torch.equal(reps[0], r) for r in reps[1:])}")
        rep["compile"] = {
            "max_abs_diff": d,
            "argmax_same": bool(v_comp.argmax().item() == base.argmax().item()),
            "compile_s": compile_s,
            "repeat_bitwise_same": bool(all(torch.equal(reps[0], r) for r in reps[1:]))}
    except Exception as e:
        import traceback
        traceback.print_exc()
        rep["compile"] = {"error": f"{type(e).__name__}: {e}"}

    sub("C4 第一处差异定位到层")
    texts = _prompts(40)
    enc = [tok(t, return_tensors="pt").input_ids[0] for t in texts[:32]]
    L = max(len(e) for e in enc)

    def pack(idx_list):
        ids = torch.full((len(idx_list), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx_list), L), dtype=torch.long)
        for r, i in enumerate(idx_list):
            ids[r, L - len(enc[i]):] = enc[i]
            att[r, L - len(enc[i]):] = 1
        return ids.cuda(), att.cuda()

    hooks = {}

    def make_hook(i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hooks.setdefault(i, []).append(h.detach().float().cpu())
        return fn

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.layers)]

    def capture(idx_list):
        hooks.clear()
        ids_c, att_c = pack(idx_list)
        with torch.no_grad():
            model(ids_c, attention_mask=att_c)
        return {i: v[0] for i, v in hooks.items()}

    set_flags(False, False)
    single = capture([0])              # batch=1，长度与后面完全一致
    set_flags(True, False)
    tf32_on = capture([0])
    set_flags(False, False)
    many = capture(list(range(32)))    # batch=32，同一长度
    for h in handles:
        h.remove()
    growth, first = [], None
    for i in sorted(single):
        d = (single[i][0] - many[i][0]).abs().max().item()
        growth.append({"layer": i, "max_abs_diff": d,
                       "hidden_scale": single[i].abs().max().item()})
        if first is None and d > 0:
            first = i
    print(f"  batch=1 vs batch=32（同一长度 L={L}，只看第 0 行）："
          f"第一处非零差异在第 {first} 层")
    print(f"  逐层最大差 {growth[0]['max_abs_diff']:.3e} → "
          f"{growth[-1]['max_abs_diff']:.3e}；隐藏状态量级 "
          f"{growth[0]['hidden_scale']:.2f} → {growth[-1]['hidden_scale']:.2f}")
    z = max((single[i] - tf32_on[i]).abs().max().item() for i in single)
    print(f"  TF32 关 vs 开在同一路径上的逐层最大差：{z:.3e}（bf16 模型，符合预期）")
    rep["layer_diff"] = {"length": L, "first_layer_batch1_vs_32": first,
                         "growth": growth, "tf32_on_layer_max_diff": z}

    if not skip_vllm:
        sub("C5 与引擎 logprob 对拍")
        try:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            from vllm import LLM, SamplingParams
            llm = LLM(model=snap(repo), dtype="bfloat16",
                      gpu_memory_utilization=0.45, max_model_len=1024,
                      enforce_eager=True, disable_log_stats=True)
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
            outs = llm.generate(texts[:5], sp)
            rows = []
            maxdiff = 0.0
            for t, o in zip(texts[:5], outs):
                e = tok(t, return_tensors="pt").input_ids[0]
                with torch.no_grad():
                    lg = model(e.unsqueeze(0).cuda()).logits[0].float().cpu()
                lp = torch.log_softmax(lg[-1], dim=-1)
                # vLLM 的第一个输出 token 的 logprobs 就是「给定整条 prompt 的
                # 下一个 token 分布」；prompt_logprobs 是逐 prompt token 的，
                # 两者不是一回事。
                ol = o.outputs[0].logprobs[0]
                diffs = []
                for tid, lob in ol.items():
                    diffs.append(abs(lob.logprob - lp[tid].item()))
                vllm_top = max(ol.items(), key=lambda kv: kv[1].logprob)
                hf_top = int(lp.argmax())
                maxdiff = max(maxdiff, max(diffs) if diffs else 0.0)
                rows.append({"prompt": t, "vllm_argmax": vllm_top[0],
                             "hf_argmax": hf_top,
                             "argmax_same": bool(vllm_top[0] == hf_top),
                             "top5_max_abs_logprob_diff": max(diffs) if diffs else None,
                             "vllm_top5": {int(k): v.logprob for k, v in ol.items()},
                             "hf_top5_logprob": [lp[int(k)].item() for k in ol]})
                print(f"  {t[:26]:<28} vllm top-1 {vllm_top[0]:>7} / hf {hf_top:>7}  "
                      f"相同 {vllm_top[0] == hf_top}  "
                      f"top-5 logprob 最大差 {max(diffs) if diffs else float('nan'):.4f}")
            print(f"  5 条 prompt 的 top-5 logprob 最大绝对差 {maxdiff:.4f}")
            rep["logprob"] = {"rows": rows, "max_abs_logprob_diff": maxdiff}
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del llm
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            rep["logprob"] = {"error": f"{type(e).__name__}: {e}"}

    del model
    torch.cuda.empty_cache()
    SUMMARY["sections"]["C"] = rep
    return rep


# ------------------------------------------------------------------ BI
def section_BI(outdir, bi):
    """引擎侧的 batch 不变模式：VLLM_BATCH_INVARIANT=0/1 需分进程对照。"""
    import torch
    os.environ["VLLM_BATCH_INVARIANT"] = str(bi)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    rep = {"modes": {}}
    title(f"[BI] 引擎 batch 不变模式：VLLM_BATCH_INVARIANT={bi}")
    repo = os.environ.get("L42_MODEL", "Qwen/Qwen3-1.7B")
    util = float(os.environ.get("L42_UTIL", "0.6"))
    from vllm import LLM, SamplingParams
    texts = _prompts(32)
    llm = LLM(model=snap(repo), dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=1024, enforce_eager=True, disable_log_stats=True)
    bi_active = bool(getattr(__import__("vllm").envs, "VLLM_BATCH_INVARIANT", False))
    print(f"  vllm.envs.VLLM_BATCH_INVARIANT = {bi_active}，"
          f"gpu_memory_utilization = {util}")
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    solo = [llm.generate([t], sp)[0] for t in texts[:8]]
    batched = llm.generate(texts, sp)
    rows = []
    for i, (s, b) in enumerate(zip(solo, batched[:8])):
        st = s.outputs[0]
        bt = b.outputs[0]
        sl, bl = st.logprobs[0], bt.logprobs[0]
        top = max(sl.items(), key=lambda kv: kv[1].logprob)[0]
        d = (abs(sl[top].logprob - bl[top].logprob)
             if top in bl else None)
        rows.append({"sample": i,
                     "solo_token": st.token_ids[0],
                     "batched_token": bt.token_ids[0],
                     "token_same": bool(st.token_ids[0] == bt.token_ids[0]),
                     "top1_logprob_abs_diff": d})
    same = sum(1 for r in rows if r["token_same"])
    diffs = [r["top1_logprob_abs_diff"] for r in rows
             if r["top1_logprob_abs_diff"] is not None]
    dmax = max(diffs) if diffs else None
    print(f"  逐条前向 vs 32 条同批前向：贪心 token 相同 {same}/8；"
          f"top-1 logprob 最大绝对差 "
          f"{'n/a' if dmax is None else f'{dmax:.6f}'}")
    rep["modes"][f"bi_{bi}"] = {"env_value": bi_active, "token_same": same,
                                "n": len(rows),
                                "top1_logprob_max_abs_diff": dmax, "rows": rows}
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    SUMMARY["sections"]["BI"] = rep
    return rep


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l42_out"))
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--bi", type=int, choices=[0, 1], default=1,
                    help="BI 段使用的 VLLM_BATCH_INVARIANT 取值")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B", "C"]
    env = {"python": sys.version.split()[0], "HF_HOME": os.environ.get("HF_HOME")}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    for m in ("transformers", "vllm"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    SUMMARY["env"] = env
    for s in want:
        if s == "C":
            section_C(args.outdir, args.skip_vllm)
        elif s == "BI":
            section_BI(args.outdir, args.bi)
        else:
            {"A": section_A, "B": section_B}[s](args.outdir)
    SUMMARY["outdir"] = args.outdir
    path = os.path.join(args.outdir, "numerics_determinism.json")
    if os.path.exists(path):
        try:
            old = json.load(open(path))
            new_secs = dict(SUMMARY["sections"])
            oldBI = (old.get("sections", {}).get("BI") or {}).get("modes") or {}
            newBI = (new_secs.get("BI") or {}).get("modes")
            if newBI:
                new_secs["BI"] = {"modes": {**oldBI, **newBI}}
            SUMMARY.update({"env": {**old.get("env", {}), **env},
                            "sections": {**old.get("sections", {}), **new_secs},
                            "runs": old.get("runs", []) + [list(want)]})
        except Exception as e:
            print(f"  (合并旧结果失败: {type(e).__name__}: {e})")
    else:
        SUMMARY["runs"] = [list(want)]
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
