#!/usr/bin/env python3
"""L5.10 任务 D 的另一半 —— merged dense 与 unmerged 服务的对照。

任务 D 要求「用公开已训练 adapter＋匹配 base，在冻结当前 revision/模板/输入后
检查 merged/unmerged 与服务输出」。上一轮做了服务侧挂 adapter（unmerged），
这一轮补 merged：用 PEFT 的 `merge_and_unload()` 生成一份合并权重，
再把两份都用同一个 vLLM 跑同一批 prompt，比较**首 token 的 logprob 分布**。

为什么比 logprob 而不是比文本：
5.10 的公开 adapter 实验已经证明这台引擎的贪心输出在重复运行之间都不可复现，
所以"文本相同/不同"不能作为合并是否正确的判据；而首 token 的分布是同一上下文下
可以直接相减的量，差多少就是合并误差（含 bf16 舍入）。

合并本身带一个自检：对第 0 层的 q_proj 验证
    merged_w - base_w == (alpha/r) * B @ A
若这个等式不成立，说明合并脚本写错了，后面的比较都不用看。
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

BASE = ("/scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots/"
        "1cfa9a7208912126459214e8b04321603b3df60c")
ADAPTER = ("/scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots/"
           "036d6b7a5b589ea27bb9a855386e0ce1e281fa75")

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of France, and why is it famous?",
    "Summarize the theory of relativity in two sentences.",
]


def weight_check(check_path: pathlib.Path):
    """把 bf16 存储的舍入也算进去，重新核对落盘 merged 权重（可离线复跑）。"""
    import torch
    from safetensors import safe_open
    merged_file = pathlib.Path("/scratch/learn/models/merged/Qwen3-4B-lora-trl-lib/"
                               "model.safetensors")
    out = {}
    with safe_open(f"{BASE}/model-00001-of-00003.safetensors", framework="pt") as f:
        base = {"q_proj": f.get_tensor("model.layers.0.self_attn.q_proj.weight").float(),
                "v_proj": f.get_tensor("model.layers.0.self_attn.v_proj.weight").float()}
    with safe_open(pathlib.Path(ADAPTER) / "adapter_model.safetensors", framework="pt") as f:
        ab = {}
        for mod in ("q_proj", "v_proj"):
            pre = f"base_model.model.model.layers.0.self_attn.{mod}.lora_"
            ab[mod] = (f.get_tensor(pre + "A.weight").float(),
                       f.get_tensor(pre + "B.weight").float())
    with safe_open(merged_file, framework="pt") as f:
        merged = {"q_proj": f.get_tensor("model.layers.0.self_attn.q_proj.weight").float(),
                  "v_proj": f.get_tensor("model.layers.0.self_attn.v_proj.weight").float()}
    for mod in ("q_proj", "v_proj"):
        a, b = ab[mod]
        want = (base[mod] + b @ a).to(torch.bfloat16).float()
        out[mod] = dict(
            max_abs_diff_vs_bf16_rounded_merge=float((want - merged[mod]).abs().max()),
            max_abs_diff_vs_float32_merge=float(
                ((base[mod] + b @ a) - merged[mod]).abs().max()),
            base_abs_max=float(base[mod].abs().max()),
            delta_abs_max=float((b @ a).abs().max()),
            bf16_ulp_at_base_scale=float(base[mod].abs().max() * 2 ** -8))
    check_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return out


def merge(out_dir: pathlib.Path):
    """PEFT 合并 + 自检 + 落盘。返回自检结果。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, device_map="cuda")
    base_w = None
    check = {}
    with torch.no_grad():
        base_w = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    peft_model = PeftModel.from_pretrained(model, ADAPTER)
    merged = peft_model.merge_and_unload()
    with torch.no_grad():
        merged_w = merged.model.layers[0].self_attn.q_proj.weight.detach()
        delta = (merged_w.float() - base_w.float())
        # 从 adapter 文件直接算 (alpha/r) * B @ A 作参照
        from safetensors import safe_open
        a = b = None
        with safe_open(pathlib.Path(ADAPTER) / "adapter_model.safetensors",
                       framework="pt", device="cpu") as f:
            for k in f.keys():
                if k == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                    a = f.get_tensor(k).to("cuda").float()
                if k == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                    b = f.get_tensor(k).to("cuda").float()
        expect = (b @ a)
        check = dict(layer=0, module="q_proj",
                     max_abs_diff=float((delta - expect).abs().max()),
                     delta_abs_max=float(delta.abs().max()),
                     expect_abs_max=float(expect.abs().max()),
                     shape=list(delta.shape))
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    for name in ("generation_config.json",):
        src = pathlib.Path(BASE) / name
        if src.exists() and not (out_dir / name).exists():
            shutil.copy(src, out_dir / name)
    del model, peft_model, merged
    gc.collect()
    try:
        import torch as _t
        _t.cuda.empty_cache()
    except Exception:
        pass
    return check, out_dir


def run_llm(model_path: str, lora=None):
    from vllm import LLM, SamplingParams
    kw = dict(model=model_path, dtype="bfloat16", max_model_len=1024,
              gpu_memory_utilization=0.30, enforce_eager=True,
              disable_log_stats=True)
    if lora is not None:
        kw.update(enable_lora=True, max_lora_rank=8, max_loras=1)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=20)
    req = None
    if lora is not None:
        from vllm.lora.request import LoRARequest
        req = LoRARequest("pub", 1, lora)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp, lora_request=req, use_tqdm=False)
    wall = time.perf_counter() - t0
    recs = []
    for o in outs:
        r = o.outputs[0]
        first = r.logprobs[0] if r.logprobs else {}
        recs.append(dict(
            token_ids=list(r.token_ids),
            text=r.text,
            first_token_topk={int(k): float(v.logprob) for k, v in first.items()},
            finish_reason=str(r.finish_reason)))
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return dict(wall_s=round(wall, 3), outputs=recs)


def compare_first_token(a, b, k=10):
    """同一 prompt 的首 token 分布：取两边 top-k 的并集逐个比 logprob。"""
    rows = []
    for i, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
        dx, dy = x["first_token_topk"], y["first_token_topk"]
        ids = sorted(set(dx) | set(dy),
                     key=lambda t: -max(dx.get(t, -1e9), dy.get(t, -1e9)))[:k]
        diffs = [abs(dx.get(t, -1e9) - dy.get(t, -1e9)) for t in ids]
        ta, tb = x["token_ids"], y["token_ids"]
        first_diff = next((j for j, (m, n) in enumerate(zip(ta, tb)) if m != n),
                          None)
        rows.append(dict(prompt=i, tokens_overlap=len(set(dx) & set(dy)),
                         max_abs_logprob_diff=round(max(diffs), 6),
                         argmax_same=max(dx, key=dx.get) == max(dy, key=dy.get),
                         gen_first_diff_index=first_diff,
                         len_a=len(ta), len_b=len(tb)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--merged-dir", type=pathlib.Path,
                    default=pathlib.Path("/scratch/learn/models/merged/Qwen3-4B-lora-trl-lib"))
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="只重跑合并权重自检，不再起引擎")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        out = weight_check(args.out / "merge_weight_check.json")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    report = dict(base=BASE, adapter=ADAPTER, merged_dir=str(args.merged_dir),
                  prompts=PROMPTS)
    if args.skip_merge and (args.merged_dir / "config.json").exists():
        report["merge_check"] = dict(skipped=True)
    else:
        check, _ = merge(args.merged_dir)
        report["merge_check"] = check
        print(f"合并自检：max|(merged-base) - B@A| = {check['max_abs_diff']:.3e} "
              f"（delta 量级 {check['delta_abs_max']:.3e}）")

    print("跑 merged …")
    merged = run_llm(str(args.merged_dir))
    print(f"  {merged['wall_s']} s")
    print("跑 unmerged（base + adapter）…")
    unmerged = run_llm(BASE, lora=ADAPTER)
    print(f"  {unmerged['wall_s']} s")

    report["merged"] = merged
    report["unmerged"] = unmerged
    report["comparison"] = compare_first_token(merged, unmerged)
    (args.out / "merge_reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n首 token 分布对比（merged vs unmerged）：")
    for row in report["comparison"]:
        print(f"  prompt {row['prompt']}: 重叠 top-k {row['tokens_overlap']} 个，"
              f"max|Δlogprob| {row['max_abs_logprob_diff']:.3e}，"
              f"argmax 相同 {row['argmax_same']}，"
              f"生成序列首个分岔位置 {row['gen_first_diff_index']}")
    print(f"\n耗时：merged {merged['wall_s']} s，unmerged {unmerged['wall_s']} s")


if __name__ == "__main__":
    main()
