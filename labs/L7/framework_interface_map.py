#!/usr/bin/env python3
"""教学训练循环到四种生产框架的接口映射，以及 Puro-Megatron 的扩展点（7.3-G）。

只读仓库内的源码快照，不导入任何框架、不执行框架代码。每个阶段记录两件事：

  * 教学 loop（`labs/L7/teaching_pretrain.py`）里承担该阶段的语句与其行号；
  * 四种框架（PyTorch FSDP2、DeepSpeed、Megatron-LM/Puro-Megatron、TorchTitan）里对应的
    符号定义位置 `file:line` 与签名行。

符号找不到时不猜，写进 `missing`；行号来自快照文件本身，因此换源码版本会直接失配，
这也正是它要暴露的事情。

Usage:
    python labs/L7/framework_interface_map.py \
      --snapshot-root results/local/7.3 --outdir "$RUN_DIR/framework-map"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 训练循环的九个阶段，以及每个阶段在四种框架里的候选符号（按顺序第一个命中者胜出）
STAGES = (
    ("init", "进程组与设备", ["init_process_group", "initialize_distributed", "init_distributed"]),
    ("build", "模型构造", ["_model_builder", "get_model", "GPTModel(", "build_model",
                          "model_cls("]),
    ("parallelize", "并行/包装", ["fully_shard", "DeepSpeedEngine", "DistributedDataParallel",
                                  "parallelize_module", "parallelize_llama"]),
    ("make_batch", "取数", ["get_batch", "build_train_valid_test_data_iterators", "next(iter",
                            "dataloader", "DistributedSampler"]),
    ("forward_loss", "前向与损失", ["def forward", "def train_step", "_forward_backward",
                                    "cross_entropy", "def loss_func"]),
    ("backward", "反向", ["def backward", "loss.backward", "_backward", "backward_step"]),
    ("optimizer", "优化器与更新", ["class AdamW", "class Muon", "class DistributedOptimizer",
                                   "optimizer.step", "optimizers.step", "def step"]),
    ("save", "保存", ["dcp.save", "async_save", "def save", "save_checkpoint",
                      "class Checkpointer"]),
)

FRAMEWORKS = {
    "pytorch_fsdp2": ["pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_state.py",
                      "pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py",
                      "pytorch/torch/distributed/fsdp/_fully_shard/_fully_shard.py",
                      "pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_api.py"],
    "deepspeed": ["deepspeed/deepspeed/runtime/engine.py",
                  "deepspeed/deepspeed/runtime/zero/stage3.py",
                  "deepspeed/deepspeed/runtime/zero/parameter_offload.py"],
    "megatron": ["megatron/megatron/training/training.py",
                 "megatron/megatron/core/distributed/distributed_data_parallel.py",
                 "megatron/megatron/core/optimizer/distrib_optimizer.py"],
    "torchtitan": ["torchtitan/train.py", "torchtitan/trainer.py", "torchtitan/fsdp.py",
                   "torchtitan/parallel_dims.py", "torchtitan/activation_checkpoint.py",
                   "torchtitan/dcp.py"],
    "torch_dcp": ["pytorch/torch/distributed/checkpoint/state_dict_saver.py"],
}

# 教学 loop 每个阶段对应的源码特征（按顺序第一个命中者胜出）
TEACHING_MARKERS = {
    "init": ["torch.manual_seed", "device = \"cuda\"", "assert torch.cuda.is_available()"],
    "build": ["model = MiniMindForCausalLM(lm_config)", "lm_config = MiniMindConfig("],
    "parallelize": ["model = MiniMindForCausalLM(lm_config).to(device)"],
    "make_batch": ["starts = order[cursor:cursor + args.micro_bs]",
                   "x = torch.from_numpy(gather(train_tokens"],
    "forward_loss": ["out = model(x, labels=x)"],
    "backward": ["scaler.scale(loss).backward()"],
    "optimizer": ["optimizer = opt_cls(model.parameters()",
                  "grad_norm = torch.nn.utils.clip_grad_norm_",
                  "optimizer.step()"],
    "save": ["torch.save(payload, outdir / name)"],
}

PURO = {
    "revision": "a7b80e873a0b5e1820ae425b9abd1b9ec578dc5c",
    "upstream": "NVIDIA Megatron-LM core_v0.16.0 (3bec9aa97dda898d16ff5a89bac0ed2b6682b172)",
    "probes": [
        {"file": "puro/emerging_optimizers.py", "symbol": "_is_hyperball_matrix",
         "role": "MuonH 参数路由谓词：2 维、非 embedding/输出、非 attention 输出门、非 bias、非 router/gate"},
        {"file": "puro/emerging_optimizers.py", "symbol": "_muon_hyperball_param_overrides_factory",
         "role": "路由表：MuonH 矩阵 wd_mult=0，其余（非线性/embedding/attention 门/router）交给 scalar 优化器"},
        {"file": "puro/emerging_optimizers.py", "symbol": "get_emerging_optimizer_param_overrides",
         "role": "muon 与 muon_hyperball 分别返回不同的路由表"},
        {"file": "puro/emerging_optimizers.py", "symbol": "_is_nonlinear_or_embedding",
         "role": "标量优化器侧的判定：embedding/输出参数或非 2 维"},
        {"file": "puro/training_config.py", "symbol": "reset_opt_param_scheduler_progress",
         "role": "阶段切换/恢复时是否重置学习率调度进度"},
        {"file": "puro/training_config.py", "symbol": "reset_train_dataloader_progress",
         "role": "阶段切换/恢复时是否重置数据游标"},
        {"file": "puro/megatron_checkpointing.py", "symbol": "reset_opt_param_scheduler_progress",
         "role": "恢复路径里真正读取该开关的位置"},
        {"file": "puro/megatron_training.py", "symbol": "reset_train_dataloader_progress",
         "role": "训练循环里重置数据游标的位置"},
    ],
}


def locate_symbol(text: str, symbol: str):
    """返回符号首次出现的 `行号: 该行内容`；优先匹配定义行。"""
    lines = text.splitlines()
    pattern = re.compile(r"^\s*(def|class)\s+" + re.escape(symbol.rstrip("(")))
    for index, line in enumerate(lines, 1):
        if pattern.match(line):
            return index, line.strip()
    for index, line in enumerate(lines, 1):
        if symbol in line:
            return index, line.strip()
    return None, None


def build_mapping(snapshot_root: Path) -> dict:
    teaching_path = snapshot_root / "20260923-puro-and-mapping" / "source" / "teaching" / \
        "teaching_pretrain.py"
    teaching_text = teaching_path.read_text(encoding="utf-8")
    roots = ["20260914-review-fixes/source", "20260915-frameworks/source",
             "20260923-puro-and-mapping/source"]
    framework_text = {}
    for name, files in FRAMEWORKS.items():
        framework_text[name] = []
        for relative in files:
            text = None
            for root in roots:
                candidate = snapshot_root / root / relative
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8")
                    break
            framework_text[name].append((relative, text))

    rows, missing = [], []
    for stage, meaning, symbols in STAGES:
        row = {"stage": stage, "meaning": meaning, "teaching": {}, "frameworks": {}}
        markers = TEACHING_MARKERS[stage]
        for marker in markers:
            for index, line in enumerate(teaching_text.splitlines(), 1):
                if marker in line:
                    row["teaching"] = {"marker": marker,
                                       "line": index,
                                       "file": "labs/L7/teaching_pretrain.py",
                                       "code": line.strip()}
                    break
            if row["teaching"]:
                break
        if not row["teaching"]:
            missing.append(f"teaching:{stage}")
        for name, blobs in framework_text.items():
            hit = None
            for relative, text in blobs:
                if text is None:
                    missing.append(f"{name}:{relative}")
                    continue
                for symbol in symbols:
                    line_no, code = locate_symbol(text, symbol)
                    if line_no:
                        hit = {"symbol": symbol, "file": relative, "line": line_no, "code": code}
                        break
                if hit:
                    break
            row["frameworks"][name] = hit
            if hit is None:
                missing.append(f"{name}:{stage}")
        rows.append(row)

    puro_section = {"revision": PURO["revision"], "upstream": PURO["upstream"], "probes": []}
    for probe in PURO["probes"]:
        text = (snapshot_root / "20260923-puro-and-mapping" / "source" / probe["file"]).read_text(
            encoding="utf-8")
        line_no, code = locate_symbol(text, probe["symbol"])
        puro_section["probes"].append(probe | {"line": line_no, "code": code})
        if line_no is None:
            missing.append(f"puro:{probe['symbol']}")

    return {"section": "framework-interface-map", "snapshot_root": str(snapshot_root),
            "stages": rows, "puro_megatron": puro_section, "missing": missing,
            "claim_scope": "全部行号来自仓库内快照文件；未导入或运行任何框架，"
                           "因此这是接口位置对照，不是行为等价或性能结论"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    result = build_mapping(args.snapshot_root)
    (args.outdir / "framework_interface_map.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stages": len(result["stages"]), "missing": result["missing"],
                      "puro_probes": [(p["symbol"], p["line"]) for p in
                                      result["puro_megatron"]["probes"]]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
