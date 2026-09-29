#!/usr/bin/env python3
"""L5.10 任务 D —— 公开已训练 adapter 在真实引擎里的加载与输出对照。

上一轮只做了 CPU 侧产物核对。这一轮把 `trl-lib/Qwen3-4B-LoRA`（r=8、alpha=8、
targets=q_proj/v_proj）与匹配基座 `Qwen/Qwen3-4B` 一起放进 vLLM 的离线 API，
在**冻结 revision / 模板 / 输入**的前提下回答四个可证伪的问题：

  1. 启用 LoRA（但不使用）会不会改变 base 的输出？——这是本章的核心不变量"base 行不得误加增量"。
  2. 挂上 adapter 后输出是否真的变了？（若完全不变，说明 adapter 被静默忽略）
  3. 把 adapter_config 的 alpha 从 8 改成 16（缩放 1.0 → 2.0），输出是否随之改变？
     ——若不变，说明缩放没有真正进入 kernel。
  4. 同一个权重目录换一个 lora_name 请求，输出是否一致？——名称与权重是两件事。
  5. 带 adapter 的 decode 相对 base 多花多少时间。

不合并 dense 权重（合并对照见任务 A 的 mini 实现与 merged 逐行参照）。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# 4.2 的结论：默认配置下 batch 组成会改变数值结果，于是"同一配置重复运行"本身
# 就不可复现。VLLM_BATCH_INVARIANT=1 让结果与 batch 组成无关（必须在建引擎前设置）。
if os.environ.get("L510_BATCH_INVARIANT") == "1":
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

BASE = "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots"
ADAPTER = ("/scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots/"
           "036d6b7a5b589ea27bb9a855386e0ce1e281fa75")

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France, and why is it famous?",
    "Summarize the theory of relativity in two sentences.",
]


def base_snapshot(root: pathlib.Path) -> str:
    snaps = sorted((root).iterdir())
    return str(snaps[0])


def make_alpha_variant(run_root: pathlib.Path, alpha: int, src: str) -> str:
    """复制 adapter_config，改 alpha，权重用符号链接指向原文件。"""
    dst = run_root / f"adapter-alpha{alpha}"
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((pathlib.Path(src) / "adapter_config.json").read_text())
    cfg["lora_alpha"] = alpha
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    w = dst / "adapter_model.safetensors"
    if not w.exists():
        w.symlink_to(pathlib.Path(src) / "adapter_model.safetensors")
    return str(dst)


def build(model: str, enable_lora: bool):
    from vllm import LLM
    kw = dict(model=model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.55, enforce_eager=True,
              disable_log_stats=True)
    if enable_lora:
        kw.update(enable_lora=True, max_lora_rank=8, max_loras=1,
                  max_cpu_loras=2)
    return LLM(**kw)


def run_all(llm, prompts, sp, lora_request=None, label=""):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=lora_request, use_tqdm=False)
    dt = time.perf_counter() - t0
    recs = []
    for o in outs:
        r = o.outputs[0]
        recs.append(dict(text=r.text,
                         token_ids=list(r.token_ids),
                         finish_reason=str(r.finish_reason)))
    return dict(label=label, wall_s=round(dt, 3), outputs=recs)


def compare(a, b):
    diffs = []
    for i, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
        ta, tb = x["token_ids"], y["token_ids"]
        if ta == tb:
            diffs.append(None)
            continue
        j = next((k for k, (m, n) in enumerate(zip(ta, tb)) if m != n),
                 min(len(ta), len(tb)))
        diffs.append(dict(prompt=i, first_diff=j, a=ta[j] if j < len(ta) else None,
                          b=tb[j] if j < len(tb) else None,
                          len_a=len(ta), len_b=len(tb)))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--base-root", default=BASE)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="传给 SamplingParams 的 seed；配合 temperature=0 固定采样路径")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    model = base_snapshot(pathlib.Path(args.base_root))
    alpha16 = make_alpha_variant(args.out, 16, args.adapter)
    sp_kw = dict(temperature=0.0, max_tokens=64)
    if args.seed is not None:
        sp_kw["seed"] = args.seed
    sp = SamplingParams(**sp_kw)

    report = dict(model=model, adapter=args.adapter, alpha16_dir=alpha16,
                  prompts=PROMPTS, max_tokens=64, seed=args.seed,
                  batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT"),
                  rounds=args.rounds)

    # ---- 第一台引擎：开 LoRA。
    # 三组交错重复：单次顺序跑会把"跑的先后"混进耗时差里，
    # 也会让"换 name 是否改变输出"分不清是稳定差异还是抖动。
    llm = build(model, enable_lora=True)
    variants = {
        "base": (None, "base(lora开启但不使用)"),
        "adapter": (LoRARequest("pub", 1, args.adapter), "adapter"),
        "other_name": (LoRARequest("another-name", 2, args.adapter), "同名权重换name"),
        "alpha16": (LoRARequest("a16", 3, alpha16), "alpha=16"),
    }
    rounds = []
    for r in range(args.rounds):
        row = {}
        for key in ("base", "adapter", "other_name", "alpha16"):
            req, label = variants[key]
            row[key] = run_all(llm, PROMPTS, sp, req, f"{label}#r{r}")
        rounds.append(row)
        print(f"  round {r}: " + "  ".join(f"{k}={row[k]['wall_s']}s"
                                          for k in row), flush=True)
    report["lora_enabled"] = {k: rounds[-1][k] for k in rounds[-1]}
    report["rounds"] = rounds
    report["per_round_checks"] = [
        dict(round=r,
             base_vs_adapter=compare(row["base"], row["adapter"]),
             adapter_vs_other_name=compare(row["adapter"], row["other_name"]),
             adapter_vs_alpha16=compare(row["adapter"], row["alpha16"]))
        for r, row in enumerate(rounds)]
    report["latency_per_round"] = [
        {k: row[k]["wall_s"] for k in row} for row in rounds]
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- 第二台引擎：关 LoRA，作 base 的参照 ----
    llm2 = build(model, enable_lora=False)
    report["lora_disabled"] = dict(base=run_all(llm2, PROMPTS, sp, None, "base(未开LoRA)"))
    del llm2
    gc.collect()

    L = report["lora_enabled"]
    import statistics as _st
    report["latency_median"] = {
        k: round(_st.median(row[k]["wall_s"] for row in rounds), 3)
        for k in ("base", "adapter", "other_name", "alpha16")}
    report["checks"] = dict(
        lora_enabled_vs_disabled_base=compare(L["base"], report["lora_disabled"]["base"]),
        base_vs_adapter=compare(L["base"], L["adapter"]),
        adapter_vs_other_name=compare(L["adapter"], L["other_name"]),
        adapter_vs_alpha16=compare(L["adapter"], L["alpha16"]),
    )
    report["latency"] = {k: v["wall_s"] for k, v in L.items()}
    stable = all(any(x for x in pc["adapter_vs_other_name"])
                 for pc in report["per_round_checks"])
    report["adapter_name_difference_stable_across_rounds"] = stable
    # 运行间可复现性：同一配置在相邻两轮之间是否逐 token 相同
    if len(rounds) >= 2:
        report["round_reproducibility"] = {
            k: [compare(rounds[r][k], rounds[r + 1][k]) for r in range(len(rounds) - 1)]
            for k in ("base", "adapter", "other_name", "alpha16")}
        all_same = all(
            all(x is None for x in diffs)
            for per_key in report["round_reproducibility"].values()
            for diffs in per_key)
        report["all_rounds_reproducible"] = all_same

    (args.out / "public_adapter.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def verdict(d):
        n_diff = sum(1 for x in d if x)
        return f"{n_diff}/{len(d)} 条不同" + ("" if n_diff == 0 else
                                            f"（首个分岔在 {d[[i for i,x in enumerate(d) if x][0]]}）")
    print("启用 LoRA 但不用 adapter vs 未开 LoRA：", verdict(report["checks"]["lora_enabled_vs_disabled_base"]))
    print("base vs adapter          ：", verdict(report["checks"]["base_vs_adapter"]))
    print("adapter vs 换 lora_name  ：", verdict(report["checks"]["adapter_vs_other_name"]))
    print("adapter(alpha=8) vs 16   ：", verdict(report["checks"]["adapter_vs_alpha16"]))
    print("耗时（中位，3 轮交错）：",
          json.dumps(report["latency_median"], ensure_ascii=False))
    print("换 name 的差异是否每轮都出现：",
          report["adapter_name_difference_stable_across_rounds"])


if __name__ == "__main__":
    main()
