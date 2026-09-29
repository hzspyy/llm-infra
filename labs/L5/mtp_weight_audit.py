#!/usr/bin/env python3
"""L5.5 任务 D · Qwen3.5-4B 的 MTP 权重与配置核对（只读，不需要 GPU）。

计划要「模型族声明不代替具体 checkpoint 支持」，所以把声明与产物逐项对齐：

  1. `config.json` 里与多 token 预测有关的字段（`text_config.mtp_num_hidden_layers`、
     `mtp_use_dedicated_embeddings`）与主干层配置（层数、`layer_types`、
     `full_attention_interval`）；
  2. 权重清单里 `mtp.*` 张量的**逐个形状**（直接从 safetensors header 读，不加载权重）；
  3. 主干层号的范围，用来确认 MTP 层不在主干层号里、是外加的一层。

用法：
    python mtp_weight_audit.py --model <snapshot 目录> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def snapshot_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    hub = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
    import glob
    snaps = sorted(glob.glob(f"{hub}/models--{model.replace('/', '--')}/snapshots/*"))
    if not snaps:
        raise SystemExit(f"没有找到 {model} 的快照")
    return snaps[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    snap = snapshot_dir(args.model)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tc = cfg.get("text_config") or {}
    out: list[str] = []
    rep: dict = dict(snapshot=os.path.basename(snap), model=args.model)

    out.append(f"L5.5-D MTP 权重与配置核对 · {args.model} · snapshot {rep['snapshot']}")
    out.append(f"  architectures = {cfg.get('architectures')}  model_type = {cfg.get('model_type')}")
    keys = ("num_hidden_layers", "layer_types", "full_attention_interval",
            "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads")
    for k in keys:
        if k in tc:
            v = tc[k]
            out.append(f"  text_config.{k} = {str(v)[:180]}")

    idx_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        weight_map = idx["weight_map"]
    else:
        weight_map = {}
        for fn in os.listdir(snap):
            if fn.endswith(".safetensors"):
                for name in read_safetensors_header(os.path.join(snap, fn)):
                    if name != "__metadata__":
                        weight_map[name] = fn

    layers = sorted({int(m.group(1)) for n in weight_map
                     for m in [re.search(r"layers\.(\d+)\.", n)] if m})
    mtp = sorted(n for n in weight_map if re.match(r"mtp\.", n))
    rep["n_tensors"] = len(weight_map)
    rep["layer_range"] = [min(layers), max(layers)] if layers else None
    rep["n_layer_ids"] = len(layers)
    rep["mtp_tensors"] = mtp

    out.append("")
    out.append(f"  权重清单 {len(weight_map)} 个张量；主干层号 {min(layers)}–{max(layers)}"
               f"（{len(layers)} 层）")
    types = tc.get("layer_types") or []
    if types:
        from collections import Counter
        out.append(f"  layer_types 计数：{dict(Counter(types))}")
    out.append(f"  mtp.* 张量 {len(mtp)} 个，逐个形状（读 safetensors header，不加载）：")

    headers: dict[str, dict] = {}
    shapes = {}
    for name in mtp:
        fn = weight_map[name]
        if fn not in headers:
            headers[fn] = read_safetensors_header(os.path.join(snap, fn))
        meta = headers[fn].get(name) or {}
        shapes[name] = dict(dtype=meta.get("dtype"), shape=meta.get("shape"))
        out.append(f"    {name:<58} {meta.get('dtype'):>6} {meta.get('shape')}")
    rep["mtp_shapes"] = shapes

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "mtp_weight_audit.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "mtp_weight_audit.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
