#!/usr/bin/env python3
"""L5.10 任务 E —— 公开 LoRA 产物的清单与 adapter→loader→kernel 路径核对。

计划要求「追踪 PEFT 的保存、base 引用、target_modules、rank/alpha、额外训练模块和
tokenizer/template；给出 adapter→loader→kernel 的产物清单」，
并明确「base_model_name_or_path 不是精确 revision」。

本脚本只用 **CPU** 读产物与配置，不加载 4B 权重：
  1. 列全产物（含大小），区分"adapter 本体"与"随包带的 tokenizer/template"；
  2. 读 `adapter_model.safetensors` 头部，核对 A/B 的形状与 rank、统计参数量与字节；
  3. 读 base 的 config，核对 target_modules 是否真在基座里、共有多少处；
  4. 分别调用两个引擎自己的 PEFT 解析路径，看它们各取了哪些字段、有没有拒绝；
  5. 记录边界：base 只给名字不给 revision，训练数据与超参不在产物里。

用法：python lora_artifact_audit.py --adapter DIR --base DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys


def file_inventory(root: pathlib.Path):
    out = []
    for p in sorted(root.iterdir()):
        if p.name == ".gitattributes":
            continue
        real = p.resolve()
        out.append(dict(name=p.name, bytes=real.stat().st_size))
    return out


def safetensors_header(path: pathlib.Path):
    """只读头部：8 字节长度 + JSON。不加载任何权重。"""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--tag", default="run",
                    help="输出文件名后缀，便于在两个 venv 里各跑一次")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inv = file_inventory(args.adapter)
    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    head = safetensors_header(args.adapter / "adapter_model.safetensors")

    # ---- A/B 形状与参数量 ----
    tensors = []
    total_elems = 0
    for name, meta in sorted(head.items()):
        shape = meta["shape"]
        elems = 1
        for d in shape:
            elems *= d
        total_elems += elems
        tensors.append(dict(name=name, shape=shape, dtype=meta["dtype"],
                            elements=elems))
    # rank 是二维张量的**较小**维度：A 是 [r, in]，B 是 [out, r]
    rank_observed = sorted({min(t["shape"]) for t in tensors if len(t["shape"]) == 2})
    # 每个 target module 在每个 transformer 层有一对 A/B
    # 名字形如 base_model.model.model.layers.<i>.self_attn.q_proj.lora_A.weight
    # split 后层号在下标 4，不是 2
    pair_layers = sorted({int(n.split(".")[4]) for n in head
                          if n.startswith("base_model.model.model.layers.")})

    # ---- base 侧核对 ----
    base_cfg = json.loads((args.base / "config.json").read_text())
    n_layers = base_cfg["num_hidden_layers"]
    targets = cfg["target_modules"]
    if isinstance(targets, str):
        targets = [targets]
    # 从 base 的 state_dict 索引读不到（不加载权重），用 config 的层数×每层模块数核对
    expected = {t: n_layers for t in targets}

    # ---- 两个引擎各自的解析路径 ----
    vllm_parse = None
    try:
        from vllm.lora.peft_helper import PEFTHelper
        h = PEFTHelper.from_local_dir(
            str(args.adapter),
            max_position_embeddings=base_cfg.get("max_position_embeddings"))
        # 直接 dump 它自己的 dataclass 字段，别猜属性名
        import dataclasses
        vllm_parse = {f.name: getattr(h, f.name)
                      for f in dataclasses.fields(h)}
        vllm_parse["scaling_lora_alpha_over_r"] = (
            h.lora_alpha / h.r if h.r else None)
    except Exception as e:                                   # pragma: no cover
        vllm_parse = dict(error=f"{type(e).__name__}: {e}")

    sgl_parse = None
    try:
        from sglang.srt.lora.utils import get_normalized_target_modules
        sgl_parse = dict(
            # 该函数返回 set，JSON 需要 list
            normalized_target_modules=sorted(get_normalized_target_modules(targets)),
            note="SGLang 在 LoRAAdapter 里用 lora_alpha/r 作 scaling（lora.py:71）")
    except Exception as e:                                   # pragma: no cover
        sgl_parse = dict(error=f"{type(e).__name__}: {e}")

    # training_args.bin：训练侧的元信息（数据/超参是否随产物发布）
    training_args = None
    try:
        import io
        import pickle
        import torch

        class _Stub:
            """产物里的 pickle 引用了训练侧库（trl 等）。缺库时用占位类顶住，
            这样至少能读出它带的字段，而不是整个文件读不了。"""
            def __init__(self, *a, **kw):
                pass

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return type(name, (_Stub,), {})

        with (args.adapter / "training_args.bin").open("rb") as f:
            ta = _Unpickler(f).load()
        d = ta.to_dict() if hasattr(ta, "to_dict") else dict(vars(ta))
        keep = ("output_dir", "model_name_or_path", "dataset_name",
                "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                "max_seq_length", "lora_r", "lora_alpha", "lora_dropout",
                "target_modules", "bf16", "gradient_accumulation_steps",
                "report_to", "run_name", "seed")
        training_args = {k: d.get(k) for k in keep if k in d}
        training_args["_all_keys"] = sorted(d.keys())
    except Exception as e:
        training_args = dict(error=f"{type(e).__name__}: {e}")

    report = dict(
        adapter_dir=str(args.adapter), base_dir=str(args.base),
        training_args=training_args,
        revision_hint=dict(
            base_model_name_or_path=cfg["base_model_name_or_path"],
            revision_field=cfg.get("revision"),
            note="revision 为 null ⇒ 只给了仓库名，没有精确 base commit；"
                 "训练数据与训练超参不在产物里（training_args.bin 是二进制）"),
        inventory=inv, inventory_bytes=sum(f["bytes"] for f in inv),
        peft_config={k: cfg[k] for k in ("peft_type", "r", "lora_alpha",
                                         "target_modules", "bias", "use_dora",
                                         "use_rslora", "task_type",
                                         "modules_to_save", "inference_mode")},
        tensor_summary=dict(count=len(tensors), total_elements=total_elems,
                            bytes=sum(t["elements"] * (2 if t["dtype"] == "BF16"
                                                       else 4) for t in tensors),
                            rank_observed=rank_observed,
                            layers_covered=[pair_layers[0], pair_layers[-1],
                                            len(pair_layers)],
                            sample=tensors[:2] + tensors[-2:]),
        base_config=dict(num_hidden_layers=n_layers,
                         hidden_size=base_cfg["hidden_size"],
                         num_attention_heads=base_cfg["num_attention_heads"],
                         architectures=base_cfg.get("architectures")),
        expected_pairs_per_target=expected,
        vllm_peft_helper=vllm_parse,
        sglang_target_module_normalization=sgl_parse,
        tokenizer_files=[f["name"] for f in inv
                         if "tokeniz" in f["name"] or "vocab" in f["name"]
                         or "merges" in f["name"] or "template" in f["name"]],
    )
    (args.out / f"lora_artifact_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"产物 {len(inv)} 个文件，共 {report['inventory_bytes']} 字节")
    print(f"base 引用：{cfg['base_model_name_or_path']}（revision={cfg.get('revision')}）")
    print(f"peft：r={cfg['r']} alpha={cfg['lora_alpha']} "
          f"targets={cfg['target_modules']} bias={cfg['bias']}")
    print(f"safetensors：{len(tensors)} 个张量 / {total_elems} 元素 / "
          f"{report['tensor_summary']['bytes']} 字节，rank 观测 {rank_observed}，"
          f"覆盖层 {pair_layers[0]}..{pair_layers[-1]}（{len(pair_layers)} 层）")
    print(f"base 层数 {n_layers} ⇒ 每个 target 期望 {n_layers} 对 A/B")
    print(f"vLLM PEFTHelper：{json.dumps(vllm_parse, ensure_ascii=False)}")
    print(f"SGLang 归一化：{json.dumps(sgl_parse, ensure_ascii=False)}")
    tk = sum(f["bytes"] for f in inv
             if f["name"] in report["tokenizer_files"]
             or f["name"] in ("added_tokens.json", "special_tokens_map.json"))
    print(f"随包带的 tokenizer/template：{report['tokenizer_files']}"
          f"（合计 {tk:,} 字节，adapter 本体 {report['tensor_summary']['bytes']:,} 字节）")
    print(f"training_args：{json.dumps(training_args, ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
