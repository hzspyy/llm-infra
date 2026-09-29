#!/usr/bin/env python3
"""只读配方检查器：把真实 YAML/TOML/JSON 规范化成统一契约，再逐条检查。

不导入、不执行任何 Python 配置代码；YAML 走 safe_load。
检查分三层：
  1. 类型与取值   microbatch / accumulation / dp 必须是正整数
  2. 一致性       global batch = micro × accum × dp；分片整除；权重与路径对齐
  3. 契约         loss 分母、恢复所需状态、adapter target、显存预算

推导出的 token 数是配置算术，不是已经跑过的训练进度。

Usage:
    python labs/L7/recipe_inspector.py --config <recipe.yaml> [--config ...] \\
        --outdir "$RUN_DIR/recipes"
    python labs/L7/recipe_inspector.py --self-test        # 只跑反例套件
"""
from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import yaml


# ------------------------------------------------------------------ 读取
def read_config(path: Path) -> dict:
    if path.suffix in (".yaml", ".yml"):
        value = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
    elif path.suffix == ".toml":
        value = tomllib.loads(path.read_text())
    else:
        raise ValueError("只接受 YAML / JSON / TOML；不执行任何 Python 配置")
    if not isinstance(value, dict):
        raise ValueError("配置根必须是映射")
    return value


# ------------------------------------------------------------------ 规范化
def normalize(value: dict, name: str) -> dict:
    if "tokens" in value and "parallelism" in value:
        return _normalize_nanotron(value, name)
    if "model_name_or_path" in value:
        return _normalize_handbook(value, name)
    if value.get("kind") == "normalized_contract":
        return value
    raise ValueError(f"{name}: 无法识别的配方结构")


def _normalize_nanotron(v: dict, name: str) -> dict:
    tok, mesh = v["tokens"], v["parallelism"]
    model = v["model"]["model_config"]
    opt = v["optimizer"]
    sched = opt["learning_rate_scheduler"]
    stages = []
    for stage in v.get("data_stages", []):
        data = stage["data"]["dataset"]
        weights = data.get("dataset_weights") or []
        folders = data.get("dataset_folder") or []
        stages.append({"name": stage["name"],
                       "start_training_step": stage["start_training_step"],
                       "sources": len(folders), "weights": weights,
                       "weight_sum": sum(weights) if weights else None})
    gbs = tok["micro_batch_size"] * tok["batch_accumulation_per_replica"] * mesh["dp"]
    return {
        "kind": "nanotron", "name": name,
        "micro_batch_size": tok["micro_batch_size"],
        "accumulation": tok["batch_accumulation_per_replica"],
        "dp": mesh["dp"], "tp": mesh["tp"], "pp": mesh["pp"],
        "cp": mesh.get("context_parallel_size", 1),
        "ep": mesh.get("expert_parallel_size", 1),
        "global_batch_size": gbs,
        "sequence_length": tok["sequence_length"],
        "tokens_per_step": gbs * tok["sequence_length"],
        "train_steps": tok["train_steps"],
        "param_dtype": v["model"]["dtype"],
        "accumulate_grad_in_fp32": opt.get("accumulate_grad_in_fp32"),
        "clip_grad": opt.get("clip_grad"),
        "weight_decay": opt.get("weight_decay"),
        "wd_exclude": opt.get("weight_decay_exclude_named_params"),
        "optimizer_name": opt["optimizer_factory"]["name"],
        "betas": [opt["optimizer_factory"].get("adam_beta1"),
                  opt["optimizer_factory"].get("adam_beta2")],
        "eps": opt["optimizer_factory"].get("adam_eps"),
        "fused": opt["optimizer_factory"].get("torch_adam_is_fused"),
        "lr": sched["learning_rate"], "lr_unit": "per optimizer step",
        "warmup_steps": sched.get("lr_warmup_steps"),
        "decay_style": sched.get("lr_decay_style"),
        "decay_start": sched.get("lr_decay_starting_step"),
        "decay_steps": sched.get("lr_decay_steps"),
        "zero_stage": opt.get("zero_stage"),
        "checkpoint_interval": v["checkpoints"].get("checkpoint_interval"),
        "resume_checkpoint_path": v["checkpoints"].get("resume_checkpoint_path"),
        "load_optimizer": v["checkpoints"].get("load_optimizer"),
        "load_lr_scheduler": v["checkpoints"].get("load_lr_scheduler"),
        "heads": model["num_attention_heads"], "kv_heads": model["num_key_value_heads"],
        "hidden": model["hidden_size"], "layers": model["num_hidden_layers"],
        "vocab": model["vocab_size"], "tie_embeddings": model.get("tie_word_embeddings"),
        "data_stages": stages,
    }


def _normalize_handbook(v: dict, name: str) -> dict:
    mixture = v.get("dataset_mixture") or {}
    return {
        "kind": "alignment_handbook", "name": name,
        "base_model": v["model_name_or_path"], "model_revision": v.get("model_revision"),
        "micro_batch_size": v.get("per_device_train_batch_size"),
        "accumulation": v.get("gradient_accumulation_steps"),
        "dp": None,                      # 这类 YAML 不声明 world size
        "global_batch_size": None,
        "param_dtype": v.get("torch_dtype"),
        "max_length": v.get("max_length"),
        "epochs": v.get("num_train_epochs"),
        "lr": v.get("learning_rate"), "lr_unit": "per optimizer step",
        "lr_scheduler": v.get("lr_scheduler_type"),
        "warmup_ratio": v.get("warmup_ratio"),
        "loss_type": v.get("loss_type", "SFT CE"), "beta": v.get("beta"),
        "packing": v.get("packing"), "padding_free": v.get("padding_free"),
        "average_tokens_across_devices": v.get("average_tokens_across_devices"),
        "gradient_checkpointing": v.get("gradient_checkpointing"),
        "save_strategy": v.get("save_strategy"), "save_steps": v.get("save_steps"),
        "datasets": [d.get("id") for d in mixture.get("datasets", [])],
        "dataset_configs": [d.get("config") for d in mixture.get("datasets", [])],
        "mixture_seed": mixture.get("seed"),
    }


# ------------------------------------------------------------------ 检查
def _positive_int(value, field: str, errors: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field} 不是数值：{value!r}")
        return
    if isinstance(value, float) and not float(value).is_integer():
        errors.append(f"{field} 不是整数：{value}")
        return
    if value <= 0:
        errors.append(f"{field} 必须为正：{value}")


def validate(r: dict) -> list[str]:
    errors: list[str] = []
    for field in ("micro_batch_size", "accumulation", "dp", "tp", "pp", "cp"):
        _positive_int(r.get(field), field, errors)

    if r.get("dp") and r.get("global_batch_size"):
        expected = r["micro_batch_size"] * r["accumulation"] * r["dp"]
        if r["global_batch_size"] != expected:
            errors.append(f"global batch {r['global_batch_size']} ≠ "
                          f"micro×accum×dp={expected}")
    if r.get("heads") and r.get("tp"):
        if r["heads"] % r["tp"]:
            errors.append(f"注意力头 {r['heads']} 不能被 tp={r['tp']} 整除")
        if r.get("kv_heads") and r["heads"] % r["kv_heads"]:
            errors.append(f"注意力头 {r['heads']} 不能被 KV 头 {r['kv_heads']} 整除")
        if r.get("hidden") and r["hidden"] % r["heads"]:
            errors.append(f"hidden {r['hidden']} 不能被头数 {r['heads']} 整除")
    if r.get("layers") and r.get("pp") and r["layers"] % r["pp"]:
        errors.append(f"层数 {r['layers']} 不能被 pp={r['pp']} 整除")
    if r.get("lr") is not None and not (0 < r["lr"] < 1):
        errors.append(f"学习率 {r['lr']} 超出常见区间，确认单位")
    if r.get("decay_start") is not None and r.get("train_steps"):
        if r["decay_start"] > r["train_steps"]:
            errors.append(f"衰减起点 {r['decay_start']} 超过 train_steps {r['train_steps']}"
                          "（多阶段续训时这通常是正常的，需按阶段核对）")
    for stage in r.get("data_stages", []):
        if stage["weights"] and stage["sources"] != len(stage["weights"]):
            errors.append(f"阶段 {stage['name']}：{stage['sources']} 个数据源对 "
                          f"{len(stage['weights'])} 个权重")
        if stage["weights"] and (any(not math.isfinite(w) or w < 0 for w in stage["weights"])
                                 or stage["weight_sum"] <= 0):
            errors.append(f"阶段 {stage['name']}：权重非法")

    c = r.get("validation_contract", {})
    if "valid_target_count" in c and c.get("loss_denominator") != c["valid_target_count"]:
        errors.append(f"loss 分母 {c.get('loss_denominator')} ≠ 有效 target 数 "
                      f"{c['valid_target_count']}")
    if c.get("resume_same_training"):
        missing = sorted(set(c["required_states"]) - set(c["available_states"]))
        if missing:
            errors.append(f"严格恢复缺状态：{', '.join(missing)}")
    for target in c.get("adapter_targets", []):
        if target not in c.get("model_module_names", []):
            errors.append(f"adapter target 在模型里不存在：{target}")
    if "memory_components" in c:
        missing = sorted(set(c["required_memory_components"]) - set(c["memory_components"]))
        if missing:
            errors.append(f"预算漏项：{', '.join(missing)}")
        if sum(c["memory_components"].values()) > c["memory_budget_bytes"]:
            errors.append(f"预算超限：{sum(c['memory_components'].values())} > "
                          f"{c['memory_budget_bytes']}")
    return errors


# ------------------------------------------------------------------ 反例
BASELINE = {
    "kind": "normalized_contract", "name": "baseline",
    "micro_batch_size": 2, "accumulation": 4, "dp": 8, "tp": 2, "pp": 2, "cp": 1,
    "global_batch_size": 64, "heads": 32, "kv_heads": 8, "hidden": 4096,
    "layers": 32, "lr": 2e-4,
    "validation_contract": {
        "valid_target_count": 1792, "loss_denominator": 1792,
        "resume_same_training": True,
        "required_states": ["model", "optimizer", "scheduler", "rng", "data_cursor"],
        "available_states": ["model", "optimizer", "scheduler", "rng", "data_cursor"],
        "adapter_targets": ["q_proj", "v_proj"],
        "model_module_names": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "memory_components": {"params": 2, "grads": 2, "optimizer": 8, "activation": 6},
        "required_memory_components": ["params", "grads", "optimizer", "activation"],
        "memory_budget_bytes": 24,
    },
}

FAULTS = [
    ("microbatch 为 0", lambda r: r.update(micro_batch_size=0)),
    ("accumulation 为 -1", lambda r: r.update(accumulation=-1)),
    ("dp 为 1.5", lambda r: r.update(dp=1.5)),
    ("dp 写成字符串", lambda r: r.update(dp="8")),
    ("global batch 与三项乘积不符", lambda r: r.update(global_batch_size=60)),
    ("tp 不能整除注意力头", lambda r: r.update(tp=5)),
    ("pp 不能整除层数", lambda r: r.update(pp=5)),
    ("loss 分母写成 batch 内 token 总数",
     lambda r: r["validation_contract"].update(loss_denominator=2048)),
    ("恢复缺少 data_cursor 与 rng",
     lambda r: r["validation_contract"].update(
         available_states=["model", "optimizer", "scheduler"])),
    ("adapter target 模型里没有",
     lambda r: r["validation_contract"].update(adapter_targets=["gate_proj"])),
    ("预算漏掉 activation",
     lambda r: r["validation_contract"].update(
         memory_components={"params": 2, "grads": 2, "optimizer": 8})),
    ("预算超限", lambda r: r["validation_contract"].update(
        memory_components={"params": 2, "grads": 2, "optimizer": 8, "activation": 20})),
]


def self_test() -> list[dict]:
    import copy
    assert validate(BASELINE) == [], validate(BASELINE)
    rows = []
    print(f"基线配置通过全部检查\n")
    print(f"{'注入的错误':<32}{'检查器输出'}")
    for name, edit in FAULTS:
        broken = copy.deepcopy(BASELINE)
        edit(broken)
        errs = validate(broken)
        rows.append({"fault": name, "errors": errs})
        print(f"{name:<32}{errs[0] if errs else '未被发现 ← 检查器漏洞'}")
        if len(errs) > 1:
            for extra in errs[1:]:
                print(f"{'':<32}{extra}")
    missed = [r["fault"] for r in rows if not r["errors"]]
    print(f"\n{len(rows) - len(missed)}/{len(rows)} 个反例被拒绝"
          + (f"；漏掉：{missed}" if missed else ""))
    return rows


# ------------------------------------------------------------------ 主流程
def summarize(r: dict) -> str:
    if r["kind"] == "nanotron":
        return (f"  micro={r['micro_batch_size']} accum={r['accumulation']} dp={r['dp']} "
                f"tp={r['tp']} pp={r['pp']} → GBS={r['global_batch_size']} "
                f"seq={r['sequence_length']} → {r['tokens_per_step']:,} token/step\n"
                f"  dtype={r['param_dtype']} accumulate_grad_in_fp32="
                f"{r['accumulate_grad_in_fp32']} clip={r['clip_grad']} "
                f"zero_stage={r['zero_stage']}\n"
                f"  lr={r['lr']} warmup={r['warmup_steps']} decay={r['decay_style']} "
                f"从 step {r['decay_start']} 起 {r['decay_steps']} 步\n"
                f"  数据阶段：" + "；".join(
                    f"{s['name']}（step {s['start_training_step']} 起，{s['sources']} 个源）"
                    for s in r["data_stages"]))
    return (f"  base={r['base_model']} revision={r['model_revision']} "
            f"dtype={r['param_dtype']}\n"
            f"  micro={r['micro_batch_size']} accum={r['accumulation']} "
            f"dp 未声明 → GBS 需要启动命令才能确定\n"
            f"  loss={r['loss_type']} beta={r['beta']} max_length={r['max_length']} "
            f"packing={r['packing']} padding_free={r['padding_free']}\n"
            f"  lr={r['lr']} {r['lr_scheduler']} warmup_ratio={r['warmup_ratio']} "
            f"average_tokens_across_devices={r['average_tokens_across_devices']}\n"
            f"  数据集：{len(r['datasets'])} 个子集，来自 "
            f"{sorted(set(r['datasets']))}；前三个 config："
            f"{r['dataset_configs'][:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--outdir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    report = {"recipes": [], "faults": []}
    for path in args.config:
        p = Path(path)
        try:
            r = normalize(read_config(p), p.name)
        except Exception as exc:
            print(f"\n[解析失败] {p.name}: {type(exc).__name__}: {exc}")
            report["recipes"].append({"name": p.name, "parse_error": str(exc)})
            continue
        errs = validate(r)
        print(f"\n[{'通过' if not errs else '有问题'}] {p.name}（{r['kind']}）")
        print(summarize(r))
        for e in errs:
            print(f"  · {e}")
        report["recipes"].append({"name": p.name, "normalized": r, "errors": errs})

    if args.self_test or not args.config:
        print(f"\n{'=' * 78}\n反例套件\n{'=' * 78}")
        report["faults"] = self_test()

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "recipe_inspection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化结果写入 {out}/recipe_inspection.json")


if __name__ == "__main__":
    main()
