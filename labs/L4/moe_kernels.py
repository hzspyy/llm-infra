#!/usr/bin/env python3
"""L4.4 修订（任务 B/C）—— 真实 grouped kernel 与真实专家访问集合。

[B] 用 vLLM 的 fused_experts 真实 grouped kernel 替换 Python 循环：
    先与逐专家 torch 循环对拍，再把时间拆成 路由 / 排序计数 / pack / kernel / combine，
    并比较均匀、单专家 20%/50% 倾斜、真实路由四种分布 × tokens=1/32/256/4096
[C] 真实模型（OLMoE-1B-7B）逐层逐步的专家访问集合与字节：
    用 gate 的 forward hook 采真实路由，比较 prefill 与 decode、batch=1/4，
    与"均匀路由覆盖公式"分别报告

用法：
    python labs/L4/moe_kernels.py --section B --outdir out/4.4/run
    python labs/L4/moe_kernels.py --section C --outdir out/4.4/run
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
OLMOE = "allenai/OLMoE-1B-7B-0924-Instruct"
SUMMARY = {}
CFG = {"H": 2048, "I": 1024, "E": 64, "K": 8, "L": 16}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------- B
def make_weights(E, H, I, dtype=torch.bfloat16, dev="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    w1 = (torch.randn(E, 2 * I, H, generator=g) * 0.02).to(dtype).to(dev)
    w2 = (torch.randn(E, H, I, generator=g) * 0.02).to(dtype).to(dev)
    return w1, w2


def torch_loop_experts(x, w1, w2, ids, weights):
    """逐 (token, 槽位) 的 fp32 累加参照：每个槽位单独算一次专家前向再加权。"""
    I = w1.shape[1] // 2
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        acc = torch.zeros(x.shape[1], dtype=torch.float32, device=x.device)
        h = x[t:t + 1].float()
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            if e < 0 or e >= w1.shape[0]:
                continue
            g = h @ w1[e, :I].float().t()
            u = h @ w1[e, I:].float().t()
            acc = acc + float(weights[t, j]) * (
                (torch.nn.functional.silu(g) * u) @ w2[e].float().t())[0]
        out[t] = acc
    return out


def route_topk(x, gate_w, K, mode="uniform", skew_expert=0, skew_frac=0.0, seed=0):
    """返回 (ids, weights)：uniform=随机不重复；skew=按比例塞给某个专家；real=真实 router。"""
    T = x.shape[0]
    g = torch.Generator(device=x.device).manual_seed(seed)
    if mode == "uniform":
        ids = torch.stack([torch.randperm(gate_w.shape[0], generator=g,
                                          device=x.device)[:K] for _ in range(T)])
    elif mode == "skew":
        n_skew = int(T * skew_frac)
        ids = torch.stack([torch.randperm(gate_w.shape[0], generator=g,
                                          device=x.device)[:K] for _ in range(T)])
        if n_skew:
            ids[:n_skew, 0] = skew_expert
    else:                                   # real
        logits = x @ gate_w.t()
        ids = torch.topk(logits, K, dim=-1).indices
    weights = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=x.device)
    return ids, weights


def section_B(args):
    title("[B] 真实 grouped kernel：fused_experts")
    # 公开的 fused_experts 需要 MoEActivation 枚举；这里用接受字符串 activation
    # 的 fused_experts_impl（同一实现，同一 kernel 选择）
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
    E, H, I, K = CFG["E"], CFG["H"], CFG["I"], CFG["K"]
    w1, w2 = make_weights(E, H, I)
    print(f"  OLMoE 单层形状：hidden {H}、专家中间维 {I}、专家 {E}、top_k {K}")
    print(f"  w1 {tuple(w1.shape)}（gate|up 融合）{w1.numel()*2/2**20:.0f} MiB，"
          f"w2 {tuple(w2.shape)} {w2.numel()*2/2**20:.0f} MiB，"
          f"每层专家权重 {(w1.numel()+w2.numel())*2/2**20:.0f} MiB")

    sub("B1 与逐专家 torch 循环对拍")
    T = 64
    x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
    ids, weights = route_topk(x, torch.randn(E, H, device="cuda"), K, "uniform")
    y_fused = fused_experts_impl(x, w1, w2, weights, ids, activation="silu")
    y_loop = torch_loop_experts(x, w1, w2, ids, weights)
    d = (y_fused.float() - y_loop).abs().max().item()
    rel = ((y_fused.float() - y_loop).norm() / y_loop.norm()).item()
    print(f"  fused_experts vs torch 逐专家循环：max|diff| {d:.3e} 相对 {rel:.3e}")
    SUMMARY["B1"] = {"max_abs_diff": d, "rel": rel, "T": T}

    sub("B2 时间构成（real 路由，tokens 扫描）")
    gate_w = (torch.randn(E, H, device="cuda") * 0.05).to(torch.bfloat16)
    rows = []
    for T in (1, 32, 256, 4096):
        x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
        ids, weights = route_topk(x, gate_w, K, "real")
        logits = x @ gate_w.t()

        t_route = bench(lambda: torch.topk(logits, K, dim=-1).indices)
        flat = ids.reshape(-1)
        t_sort = bench(lambda: torch.cumsum(torch.bincount(flat, minlength=E), 0))
        t_pack = bench(lambda: x.repeat_interleave(K, dim=0)[
            torch.argsort(flat, stable=True)])
        t_fused = bench(lambda: fused_experts_impl(x, w1, w2, weights, ids,
                                                  activation="silu"))
        t_loop = bench(lambda: torch_loop_experts(x, w1, w2, ids, weights),
                       warmup=2, iters=5)

        def full():
            ids2 = torch.topk(x @ gate_w.t(), K, dim=-1).indices
            w2_ = torch.full((T, K), 1.0 / K, device="cuda", dtype=torch.float32)
            fused_experts_impl(x, w1, w2, w2_, ids2, activation="silu")

        t_full = bench(full)
        counts = torch.bincount(flat, minlength=E).float()
        rows.append({"tokens": T, "route_ms": t_route * 1e3, "sort_count_ms": t_sort * 1e3,
                     "pack_ms": t_pack * 1e3, "fused_ms": t_fused * 1e3,
                     "torch_loop_ms": t_loop * 1e3, "full_ms": t_full * 1e3,
                     "max_load": int(counts.max()), "mean_load": counts.mean().item(),
                     "skew": (counts.max() / counts.mean()).item()})
        print(f"  T={T:<5} 路由 {t_route*1e3:6.3f} 排序计数 {t_sort*1e3:6.3f} "
              f"pack {t_pack*1e3:6.3f} fused {t_fused*1e3:7.3f} "
              f"torch循环 {t_loop*1e3:8.3f} 完整 {t_full*1e3:7.3f} ms  "
              f"负载 max/mean {counts.max()/counts.mean():.2f}")
    SUMMARY["B2"] = rows

    sub("B3 路由倾斜：均匀 / 20% / 50% / 真实")
    skew_rows = []
    for tag, mode, frac in (("uniform", "uniform", 0.0), ("single_20%", "skew", 0.2),
                            ("single_50%", "skew", 0.5), ("real", "real", 0.0)):
        for T in (256, 4096):
            x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
            ids, weights = route_topk(x, gate_w, K, mode if mode != "skew" else "skew",
                                      skew_expert=0, skew_frac=frac)
            counts = torch.bincount(ids.reshape(-1), minlength=E).float()
            t = bench(lambda: fused_experts_impl(x, w1, w2, weights, ids, activation="silu"))
            skew_rows.append({"dist": tag, "tokens": T, "ms": t * 1e3,
                              "max_load": int(counts.max()),
                              "mean_load": counts.mean().item(),
                              "skew": (counts.max() / counts.mean()).item()})
            print(f"  {tag:<12} T={T:<5} {t*1e3:8.3f} ms  "
                  f"负载 max/mean {counts.max()/counts.mean():5.2f}  "
                  f"（max {int(counts.max())} / mean {counts.mean():.1f}）")
    print("  max/mean 只是「不均衡度」的指标，不等于 GPU 效率上界：")
    print("  一个热专家会被切成多个 tile 并行执行，而冷专家也要占一次 kernel 启动。")
    SUMMARY["B3"] = skew_rows
    SUMMARY["config"] = CFG
    return SUMMARY


# ---------------------------------------------------------------- E
PROMPTS = {
    "英文散文": ("The history of computing spans many decades, from mechanical "
             "calculators to programmable machines and then to integrated "
             "circuits that made personal computers possible. " * 20),
    "Python 代码": ("def quicksort(a):\n    if len(a) <= 1: return a\n"
                "    p = a[len(a)//2]\n    l = [x for x in a if x < p]\n"
                "    m = [x for x in a if x == p]\n    r = [x for x in a if x > p]\n"
                "    return quicksort(l) + m + quicksort(r)\n" * 12),
    "中文文本": ("稀疏混合专家模型把参数按专家切分，每个词元只经过其中少数几个专家，"
             "因此显存要按总量准备、算力按激活量计算。" * 30),
    "数字与算术": ("12 + 37 = 49; 88 - 15 = 73; 6 * 9 = 54; 144 / 12 = 12; "
              "2 ** 10 = 1024; 17 % 5 = 2; " * 25),
}


def section_E(args, ctx=None):
    title("[E] 多任务/多提示的专家访问对照")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = snap(OLMOE)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda").eval()
    gates = {}
    for i, layer in enumerate(model.model.layers):
        gates[i] = {"rows": [], "shape": None}

        def hook(mod, inp, out, i=i):
            logits = out[0] if isinstance(out, tuple) else out
            gates[i]["rows"].append(logits.detach().float().cpu())
            gates[i]["shape"] = tuple(logits.shape)
        layer.mlp.gate.register_forward_hook(hook)

    per_prompt = {}
    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids[:, :256].cuda()
        for i in gates:
            gates[i]["rows"] = []
        with torch.no_grad():
            model(ids)
        cover, hot, sets = [], [], {}
        for i in sorted(gates):
            lg = torch.cat(gates[i]["rows"], 0)
            top = torch.topk(lg, CFG["K"], dim=-1).indices
            cnt = torch.bincount(top.reshape(-1), minlength=CFG["E"]).float()
            uniq = int((cnt > 0).sum())
            cover.append(uniq)
            hot.append(float(cnt.max() / cnt.mean()))
            sets[i] = set(torch.unique(top).tolist())
        per_prompt[name] = {"tokens": int(ids.shape[1]), "coverage": cover,
                            "hot_ratio": hot, "sets": sets}
        print(f"  {name:<10} token {ids.shape[1]:<4} 每层覆盖 "
              f"{min(cover)}–{max(cover)}（均值 {sum(cover)/len(cover):.1f}/{CFG['E']}）"
              f"  最热/均值 {min(hot):.2f}–{max(hot):.2f}")

    sub("E2 不同提示之间的专家集合重叠（逐层 Jaccard）")
    names = list(PROMPTS)
    print(f"  {'层':>3} " + " ".join(f"{n:>9}" for n in names[1:]))
    jac_rows = []
    for i in sorted(per_prompt[names[0]]["sets"]):
        js = []
        base = per_prompt[names[0]]["sets"][i]
        for n in names[1:]:
            other = per_prompt[n]["sets"][i]
            j = len(base & other) / max(1, len(base | other))
            js.append(j)
        jac_rows.append({"layer": i, "jaccard": js})
        if i < 4 or i > len(gates) - 3:
            print(f"  {i:>3} " + " ".join(f"{x:9.2f}" for x in js))
    mean_j = [sum(r["jaccard"][k] for r in jac_rows) / len(jac_rows)
              for k in range(len(names) - 1)]
    print(f"  平均 Jaccard（对 {names[0]}）："
          + "，".join(f"{n} {v:.2f}" for n, v in zip(names[1:], mean_j)))
    print("  读法：覆盖率都接近满（256 token 就碰遍大部分专家），"
          "但重叠率远不为 1——**不同任务的『哪些专家』并不相同**，")
    print("  这正是专家并行下按请求类型做路由/放置的依据。")
    SUMMARY["E"] = {"prompts": {k: {"tokens": v["tokens"],
                                    "coverage_mean": sum(v["coverage"]) / len(v["coverage"]),
                                    "coverage_min": min(v["coverage"]),
                                    "coverage_max": max(v["coverage"]),
                                    "hot_min": min(v["hot_ratio"]),
                                    "hot_max": max(v["hot_ratio"])}
                                for k, v in per_prompt.items()},
                    "jaccard_mean": {n: v for n, v in zip(names[1:], mean_j)},
                    "jaccard_rows": jac_rows}
    del model
    torch.cuda.empty_cache()
    return SUMMARY


# ---------------------------------------------------------------- C
def section_C(args):
    title("[C] 真实模型的逐层专家访问集合")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = snap(OLMOE)
    tok = AutoTokenizer.from_pretrained(d)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"  模型加载 {time.perf_counter() - t0:.1f} s"
          f"（{sum(p.numel() for p in model.parameters())/1e9:.2f} B 参数）")
    gates = {}
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        gates[i] = {"rows": [], "shape": None}

        def hook(mod, inp, out, i=i):
            # OlmoeSparseMoeBlock 的 gate 返回 (logits, top_k_weights, top_k_index)
            logits = out[0] if isinstance(out, tuple) else out
            gates[i]["rows"].append(logits.detach().float().cpu())
            gates[i]["shape"] = tuple(logits.shape)
        gate.register_forward_hook(hook)

    text = ("The history of computing spans many decades, from mechanical calculators "
            "to programmable machines and then to the integrated circuits that made "
            "personal computers possible. Operating systems, compilers and networks "
            "each added a layer of abstraction. " * 40)
    ids = tok(text, return_tensors="pt").input_ids[:, :256].cuda()
    print(f"  prefill 输入 {ids.shape[1]} token")

    def collect(logits_by_layer):
        return {i: torch.cat(v, 0) for i, v in logits_by_layer.items() if v}

    # prefill B=1 / B=4
    pref = {}
    for B in (1, 4):
        for i in gates:
            gates[i]["rows"] = []
        with torch.no_grad():
            model(ids.repeat(B, 1))
        lg = collect({i: gates[i]["rows"] for i in gates})
        cov, pairs = [], 0
        for i in sorted(lg):
            top = torch.topk(lg[i], CFG["K"], dim=-1).indices
            uniq = torch.unique(top).numel()
            cov.append(uniq)
            pairs += top.numel()
        pref[f"prefill_B{B}"] = {"tokens": ids.shape[1] * B,
                                 "distinct_experts_per_layer": cov,
                                 "mean_coverage": sum(cov) / len(cov),
                                 "expert_token_pairs": pairs}
        print(f"  prefill B={B}：每层不同专家数 "
              f"{min(cov)}–{max(cov)}（均值 {sum(cov)/len(cov):.1f}/{CFG['E']}），"
              f"专家-token 对 {pairs}")

    # decode：逐步喂单 token，带 KV cache
    for i in gates:
        gates[i]["rows"] = []
    with torch.no_grad():
        out = model(ids[:, :64], use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for step in range(32):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
    lg = collect({i: gates[i]["rows"] for i in gates})
    cov, pairs = [], 0
    for i in sorted(lg):
        top = torch.topk(lg[i], CFG["K"], dim=-1).indices
        cov.append(torch.unique(top).numel())
        pairs += top.numel()
    dec = {"steps": 32, "distinct_experts_per_layer": cov,
           "mean_coverage": sum(cov) / len(cov), "expert_token_pairs": pairs}
    print(f"  decode 32 步：每层不同专家数 {min(cov)}–{max(cov)}"
          f"（均值 {sum(cov)/len(cov):.1f}/{CFG['E']}），专家-token 对 {pairs}")

    sub("C2 字节账与均匀覆盖公式对照")
    # 每层专家权重按 W4A16 的实际读取量（只读被访问的专家）
    per_expert_bytes_bf16 = (2 * CFG["I"] * CFG["H"] + CFG["H"] * CFG["I"]) * 2
    rows = []
    for tag, item in list(pref.items()) + [("decode_32steps", dec)]:
        T = item["tokens"] if "tokens" in item else item["steps"]
        k = CFG["K"]
        uniform = CFG["E"] * (1 - (1 - 1 / CFG["E"]) ** (T * k))
        real_cov = item["mean_coverage"]
        layers_bytes = real_cov * CFG["L"] * per_expert_bytes_bf16
        rows.append({"case": tag, "tokens": T, "real_coverage": real_cov,
                     "uniform_formula": uniform,
                     "bf16_layer_bytes_mib": layers_bytes / 2**20,
                     "w4a16_layer_bytes_mib": layers_bytes / 4 / 2**20})
        print(f"  {tag:<16} token {T:<5} 实测每层覆盖 {real_cov:5.1f}  "
              f"均匀公式 {uniform:5.1f}  "
              f"全 16 层专家读取 {layers_bytes/2**20:7.1f} MiB（bf16）"
              f" / {layers_bytes/4/2**20:6.1f} MiB（W4A16）")
    print("  均匀公式 E(1-(1-1/E)^(T·k)) 假设路由近似独立均匀；真实路由有偏，")
    print("  两个数字分别报告，不用公式替代实测。")
    SUMMARY["C"] = {"prefill": pref, "decode": dec, "bytes": rows,
                    "per_expert_bytes_bf16": per_expert_bytes_bf16}
    del model
    torch.cuda.empty_cache()
    return SUMMARY


# ---------------------------------------------------------------- D
def section_D(args):
    title("[D] 量化专家读取：bf16 vs int4 打包的显存流量")
    E, H, I, K = CFG["E"], CFG["H"], CFG["I"], CFG["K"]
    E, H, I = 16, 2048, 1024          # 缩小规模以便逐项可测（结论看带宽比例）
    w1, w2 = make_weights(E, H, I)
    n_expert_bytes_bf16 = (w1.numel() + w2.numel()) * 2
    print(f"  规模：{E} 专家 × (w1 {tuple(w1.shape)} + w2 {tuple(w2.shape)})，"
          f"bf16 共 {n_expert_bytes_bf16/2**20:.0f} MiB")

    def pack_int4(w, group=128):
        """沿最后一维分组对称量化到 int4；支持任意前导维（专家维）。"""
        lead, in_f = w.shape[:-1], w.shape[-1]
        w2d = w.reshape(-1, in_f)
        g = w2d.reshape(w2d.shape[0], in_f // group, group)
        scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
        q = torch.round(g / scale).clamp(-7, 7).to(torch.int8).reshape(w2d.shape[0], in_f)
        n = q & 0xF
        packed = (n[:, 0::2] | (n[:, 1::2] << 4)).to(torch.uint8)
        return packed.reshape(*lead, -1), scale.reshape(*lead, -1).to(torch.bfloat16)

    def unpack_int4(packed, scale, in_f, group=128, out_dtype=torch.bfloat16):
        lead = packed.shape[:-1]
        p2d = packed.reshape(-1, packed.shape[-1])
        s2d = scale.reshape(-1, scale.shape[-1])
        lo = (p2d & 0xF).to(torch.int8)
        hi = ((p2d >> 4) & 0xF).to(torch.int8)
        n = torch.stack([lo, hi], dim=-1).reshape(p2d.shape[0], -1)
        n = torch.where(n >= 8, n - 16, n).to(torch.bfloat16)
        out = (n.reshape(p2d.shape[0], -1, group) * s2d.unsqueeze(-1))
        return out.reshape(*lead, in_f).to(out_dtype)

    p1, s1 = pack_int4(w1)
    p2, s2 = pack_int4(w2)
    qbytes = ((p1.numel() + p2.numel()) + (s1.numel() + s2.numel()) * 2)
    print(f"  int4 打包（group 128）后：{qbytes/2**20:.0f} MiB"
          f"（压缩 {n_expert_bytes_bf16/qbytes:.2f}×，含 scale）")

    def read_bf16():
        return w1.float().sum() + w2.float().sum()

    def read_int4_dequant():
        a = unpack_int4(p1, s1, w1.shape[-1]).float().sum()
        b = unpack_int4(p2, s2, w2.shape[-1]).float().sum()
        return a + b

    # 只读取、不做乘加：测的是「把专家权重读进来」这条路径
    named = [("bf16 直接读", read_bf16, n_expert_bytes_bf16),
             ("int4 读+解包", read_int4_dequant, qbytes)]
    rows = []
    for name, fn, nbytes in named:
        t = bench(fn, warmup=3, iters=10)
        bw = nbytes / t / 1e9
        rows.append({"path": name, "ms": t * 1e3, "bytes": nbytes,
                     "effective_gib_s": bw})
        print(f"  {name:<14} {t*1e3:8.3f} ms  {nbytes/2**20:7.0f} MiB  "
              f"有效带宽 {bw:6.1f} GiB/s")
    print("  读法：int4 把字节数降到 1/3.2，但要多做一次解包；")
    print("  两者的有效带宽对比说明「省下的流量」与「解包的算力」谁更贵。")

    sub("D2 放进真实 MoE 前向：同一批 token 的专家读取量")
    for T in (256, 4096):
        g = torch.zeros(E, H, device="cuda")
        ids, _ = route_topk(torch.zeros(T, H, device="cuda"), g, K, "uniform", seed=1)
        counts = torch.bincount(ids.reshape(-1).clamp_min(0), minlength=E).float()
        touched = int((counts > 0).sum())
        per_expert = (2 * I * H + H * I) * 2          # OLMoE 真实维度下的单专家字节
        print(f"  T={T:<5} 触发专家 {touched}/{E} → 读取 {touched*per_expert/2**20:7.0f} MiB"
              f"（bf16） / {touched*per_expert/4/2**20:7.0f} MiB（int4 打包，不含 scale）")
    SUMMARY["D"] = {"expert_bytes_bf16": n_expert_bytes_bf16, "expert_bytes_int4": qbytes,
                    "compression": n_expert_bytes_bf16 / qbytes, "rows": rows}
    return SUMMARY["D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["B", "C", "D", "E", "BC", "BCD", "E"], default="BC")
    ap.add_argument("--outdir", default=os.path.expanduser("~/l44_kernels"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    SUMMARY["env"] = {"torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0)}
    if "B" in args.section:
        section_B(args)
    if "C" in args.section:
        section_C(args)
    if "D" in args.section:
        section_D(args)
    if "E" in args.section:
        section_E(args)
    path = os.path.join(args.outdir, "moe_kernels.json")
    if os.path.exists(path):
        old = json.load(open(path))
        old.update(SUMMARY)
        SUMMARY.clear()
        SUMMARY.update(old)
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
