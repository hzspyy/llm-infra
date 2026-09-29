#!/usr/bin/env python3
"""L5.10 任务 C —— 同名热更新：把 name / id / path / 权重版本四个量分开。

旧实验（`lora_kv_invalidation_test.py`）把四步里的 `lora_int_id` 从 1 改到 3，
同时又把同名不同权重的 adapter 换进来，最后得出结论的方向也不成立
（正文已注明"相反结论也未成立"）。计划对这一条的要求是：

  「分开同一已加载 ID 的权重替换与更换 adapter ID，并与独立引擎重启对照」
  「记录缓存命中或实际 KV 状态，与关闭缓存的新版本参照比较」
  「不从生成文本相同或不同推断 KV 是否失效」

所以这里把身份四个量固定成一个序列来分别动：

  name    adapter 的逻辑名（进入 prefix cache key 的就是它）
  id      LoRARequest 里的 lora_int_id
  path    adapter 目录路径（引擎按它去读权重）
  weights 目录里的权重内容

序列（prefix caching 打开）：
  S1  name=A id=1 path=P1 权重 v1        —— 基线
  S2  同 S1                              —— 应当命中前缀缓存
  S3  把 P1 的权重**原地换成 v2**，仍 name=A id=1 path=P1  —— 同名热更新
  S4  name=A id=1 path=P2（v2 的副本）    —— 换 path 但同名同 id
  S5  name=B id=2 path=P1（权重已变 v2）  —— 换 name 与 id
  S6  在独立的新引擎上重跑 S4             —— 与"同进程恢复"分开

每一步都记录：输出 token、以及 `llm.get_metrics()` 里 prefix cache 的
queries/hits 计数增量（这才是"缓存命中"的直接证据，不是文本比较）。
另外用 prefix caching **关闭**的引擎跑同一序列的 S1/S3/S4 作为参照。
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
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 10


def make_adapter(root: pathlib.Path, ident: str, version: str, seed: int, rank=8):
    import torch
    from transformers import AutoConfig
    from safetensors.torch import save_file

    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    hidden = cfg.hidden_size
    folder = root / f"adapter-{ident}-{version}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": MODEL, "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": rank, "lora_alpha": rank, "target_modules": ["q_proj"],
        "lora_dropout": 0.0, "bias": "none", "use_dora": False,
    }, indent=2))
    gen = torch.Generator().manual_seed(seed)
    weights = {}
    for layer in range(cfg.num_hidden_layers):
        p = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        weights[f"{p}.lora_A.weight"] = torch.randn(rank, hidden, generator=gen,
                                                    dtype=torch.bfloat16)
        weights[f"{p}.lora_B.weight"] = torch.randn(hidden, rank, generator=gen,
                                                    dtype=torch.bfloat16)
    save_file(weights, folder / "adapter_model.safetensors")
    return folder


def cache_counters(llm):
    """prefix cache 的 queries / hits（vLLM 的 Metric 对象带 name 与 value）。"""
    out = {}
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if "prefix_cache" in name:
                out[name] = getattr(m, "value", None)
    except Exception as e:                                    # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def build(prefix_caching: bool, util: float, max_cpu_loras: int = 3,
          max_loras: int = 2):
    from vllm import LLM
    return LLM(model=MODEL, max_loras=max_loras, max_cpu_loras=max_cpu_loras,
               max_lora_rank=8,
               enable_lora=True, enable_prefix_caching=prefix_caching,
               max_model_len=512, gpu_memory_utilization=util,
               enforce_eager=True, disable_log_stats=False)


def split_hot_update(args, adapters, p1, p2):
    """把 S3 的两处成因分开：adapter 常驻缓存 vs 前缀缓存复用旧 KV。

    三种配置各跑一遍 S1→S2→（可选挤掉 CPU adapter 缓存）→原地热更新→再请求：
      * cache_on_cpu3  ：复现 S3（两者的效果叠加）；
      * cache_off_cpu3 ：关掉前缀缓存——若仍返回 v1，说明旧值来自 adapter 常驻缓存；
      * cache_on_cpu1  ：把 max_cpu_loras 压到 1 并用第二个 adapter 挤掉它，
                         强制重读磁盘——若这时变 v2，说明常驻缓存是主因。
    """
    v1_backup = adapters / "adapter-1-v1-backup"
    if v1_backup.exists():
        shutil.rmtree(v1_backup)
    shutil.copytree(p1, v1_backup)                 # 每次都从干净的 v1 开始

    rows = []
    configs = [("cache_on_cpu3", True, 3, False),
               ("cache_off_cpu3", False, 3, False),
               ("cache_on_cpu1_evict", True, 1, True)]
    for label, prefix_on, cpu_loras, evict in configs:
        shutil.copy(v1_backup / "adapter_model.safetensors",
                    p1 / "adapter_model.safetensors")
        llm = build(prefix_on, args.util, max_cpu_loras=cpu_loras,
                    max_loras=1 if evict else 2)
        prev = cache_counters(llm)
        recs = []
        for lab, name, lid, path in [("S1_v1_first", "A", 1, p1),
                                     ("S2_v1_again", "A", 1, p1)]:
            rec, prev = step(llm, name, lid, path, prev)
            rec["step"] = lab
            recs.append(rec)
        if evict:
            rec, prev = step(llm, "TMP", 9, p2, prev)
            rec["step"] = "evict_with_other_adapter"
            recs.append(rec)
        shutil.copy(p2 / "adapter_model.safetensors",
                    p1 / "adapter_model.safetensors")   # 原地热更新 v1 → v2
        rec, prev = step(llm, "A", 1, p1, prev)
        rec["step"] = "S3_hot_update_same_path"
        recs.append(rec)
        rows.append(dict(config=label, prefix_caching=prefix_on,
                         max_cpu_loras=cpu_loras,
                         evict_before_update=evict, steps=recs))
        print(f"  {label:<20} S3 输出 {rec['token_ids'][:4]} "
              f"命中增量 {rec['metrics_delta']}")
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                      # noqa: BLE001
            pass
    return rows


def step(llm, name, lid, path, prev):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    t0 = time.perf_counter()
    out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8),
                       lora_request=LoRARequest(name, lid, str(path)),
                       use_tqdm=False)
    wall = time.perf_counter() - t0
    metrics = cache_counters(llm)
    delta = {k: (metrics.get(k) - prev.get(k) if isinstance(metrics.get(k), (int, float))
                 and isinstance(prev.get(k), (int, float)) else None)
             for k in metrics if k != "error"}
    return dict(name=name, lora_id=lid, path=str(path),
                token_ids=list(out[0].outputs[0].token_ids),
                text=out[0].outputs[0].text, wall_s=round(wall, 3),
                metrics=metrics, metrics_delta=delta), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--util", type=float, default=0.30)
    ap.add_argument("--mode", choices=["identity", "split"], default="identity",
                    help="identity = 原 S1–S6 身份矩阵；split = 只拆 S3 的两处成因")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adapters = args.out / "adapters"
    p1 = make_adapter(adapters, "1", "v1", seed=1000)
    p2 = make_adapter(adapters, "1", "v2", seed=2000)
    v2_copy = adapters / "adapter-1-v2copy"
    if v2_copy.exists():
        shutil.rmtree(v2_copy)
    shutil.copytree(p2, v2_copy)                 # 同权重、不同 path

    if args.mode == "split":
        report = dict(mode="split", prompt=PROMPT, prompt_repeat=10, max_tokens=8,
                      adapter_paths=dict(p1=str(p1), p2=str(p2)),
                      configs=split_hot_update(args, adapters, p1, p2))
        (args.out / "lora_hot_update_split.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"\n写入 {args.out}/lora_hot_update_split.json")
        os._exit(0)

    report = dict(prompt=PROMPT, prompt_repeat=10, max_tokens=8,
                  adapter_paths=dict(p1=str(p1), p2=str(p2), v2copy=str(v2_copy)),
                  cache_on=[], cache_off=[], hot_update={})

    # ---------- prefix caching 打开 ----------
    llm = build(True, args.util)
    prev = cache_counters(llm)
    for label, name, lid, path in [
        ("S1_v1_first", "A", 1, p1),
        ("S2_v1_again", "A", 1, p1),
    ]:
        rec, prev = step(llm, name, lid, path, prev)
        rec["step"] = label
        report["cache_on"].append(rec)
        print(f"  {label:<14} {rec['token_ids'][:4]} 命中增量 {rec['metrics_delta']}")

    # S3：原地热更新（同一 path，权重从 v1 换成 v2）
    shutil.copy(p2 / "adapter_model.safetensors", p1 / "adapter_model.safetensors")
    rec, prev = step(llm, "A", 1, p1, prev)
    rec["step"] = "S3_hot_update_same_path"
    report["cache_on"].append(rec)
    print(f"  {'S3_hot_update':<14} {rec['token_ids'][:4]} 命中增量 {rec['metrics_delta']}")

    # S4/S5
    for label, name, lid, path in [
        ("S4_new_path_same_name_id", "A", 1, v2_copy),
        ("S5_same_path_new_name_id", "B", 2, p1),
    ]:
        rec, prev = step(llm, name, lid, path, prev)
        rec["step"] = label
        report["cache_on"].append(rec)
        print(f"  {label:<14} {rec['token_ids'][:4]} 命中增量 {rec['metrics_delta']}")

    report["cache_on_final"] = cache_counters(llm)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---------- prefix caching 关闭（参照）----------
    llm2 = build(False, args.util)
    prev2 = cache_counters(llm2)
    for label, name, lid, path in [("R1_v2_copy_cache_off", "A", 1, v2_copy)]:
        rec, prev2 = step(llm2, name, lid, path, prev2)
        rec["step"] = label
        report["cache_off"].append(rec)
        print(f"  {label:<20} {rec['token_ids'][:4]}")
    del llm2
    gc.collect()

    # ---------- 独立引擎重启后重跑 S4 ----------
    llm3 = build(True, args.util)
    prev3 = cache_counters(llm3)
    rec, _ = step(llm3, "A", 1, v2_copy, prev3)
    rec["step"] = "S6_fresh_engine_same_request"
    report["hot_update"]["fresh_engine"] = rec
    print(f"  {'S6_fresh_engine':<20} {rec['token_ids'][:4]}")
    del llm3

    # ---------- 判定 ----------
    tok = {r["step"]: r["token_ids"] for r in report["cache_on"]}
    v2_ref = report["cache_off"][0]["token_ids"]
    v1_ref = tok.get("S1_v1_first")
    s3 = tok.get("S3_hot_update_same_path")
    s4 = tok.get("S4_new_path_same_name_id")
    report["verdict"] = dict(
        v1_reference=v1_ref, v2_reference=v2_ref,
        S2_hits_prefix_cache=bool(
            (report["cache_on"][1]["metrics_delta"] or {}).get(
                "vllm:prefix_cache_hits") or
            (report["cache_on"][1]["metrics_delta"] or {}).get(
                "vllm:prefix_cache_hits_total")),
        S3_hot_update_matches_v1=s3 == v1_ref,
        S3_hot_update_matches_v2=s3 == v2_ref,
        S4_new_path_matches_v2=s4 == v2_ref,
        S6_fresh_engine_matches_v2=report["hot_update"]["fresh_engine"]["token_ids"] == v2_ref,
    )
    (args.out / "lora_cache_identity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n判定：")
    for k, v in report["verdict"].items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
