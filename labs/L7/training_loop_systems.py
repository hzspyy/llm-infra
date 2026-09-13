#!/usr/bin/env python3
"""
7.1 训练循环系统视角实验

在 7.0b 的基础上，增加：
1. DataLoader 流水：多 worker、prefetch、pin_memory
2. Activation checkpointing：保存 vs 重算
3. 混合精度：autocast + GradScaler
4. 完整内存曲线：各阶段内存追踪
5. 关键路径分析：数据等待 vs GPU 等待

运行：
  python training_loop_systems.py

环境要求：
  - crater RTX 5090 D
  - PyTorch 2.13.0+cu130
  - transformers >= 4.40.0
  - 24GB+ 显存
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
from pathlib import Path
from collections import defaultdict
import gc


class SimpleTextDataset(Dataset):
    """简单文本数据集，用于测试 DataLoader 流水"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 模拟 CPU 侧的 tokenization
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: 与 input_ids 相同，但 padding 位置为 -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def measure_memory():
    """测量当前 GPU 内存"""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    return


def exp1_dataloader_pipeline(output_dir):
    """
    实验 1: DataLoader 流水与重叠

    对照：
    - num_workers=0 (无 prefetch)
    - num_workers=2, prefetch_factor=2
    - num_workers=4, prefetch_factor=4
    - pin_memory=True/False
    """
    print("\n" + "="*70)
    print("实验 1: DataLoader 流水与重叠")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolLM2-360M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 准备较大的数据集（模拟真实训练）
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Deep neural networks require substantial computational resources.",
        "Training large language models demands careful optimization.",
    ] * 25  # 100 样本

    dataset = SimpleTextDataset(texts, tokenizer, max_length=128)

    configs = [
        {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "label": "单线程"},
        {"num_workers": 2, "prefetch_factor": 2, "pin_memory": False, "label": "2worker-nopin"},
        {"num_workers": 2, "prefetch_factor": 2, "pin_memory": True, "label": "2worker-pin"},
        {"num_workers": 4, "prefetch_factor": 4, "pin_memory": True, "label": "4worker-pin"},
    ]

    results = []

    for cfg in configs:
        print(f"\n配置: {cfg['label']}")
        print(f"  num_workers={cfg['num_workers']}, prefetch_factor={cfg['prefetch_factor']}, pin_memory={cfg['pin_memory']}")

        loader_kwargs = {
            "batch_size": 4,
            "shuffle": False,
            "num_workers": cfg["num_workers"],
            "pin_memory": cfg["pin_memory"],
        }
        if cfg["prefetch_factor"] is not None and cfg["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]

        dataloader = DataLoader(dataset, **loader_kwargs)

        # 预热
        for _ in range(2):
            for batch in dataloader:
                _ = {k: v.to(device) for k, v in batch.items()}
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # 正式计时
        times = {"data_prep": [], "forward": [], "backward": [], "total": []}

        start_total = time.perf_counter()

        for step_idx, batch in enumerate(dataloader):
            if step_idx >= 10:  # 只跑 10 步
                break

            t0 = time.perf_counter()

            # 数据 H2D
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times["data_prep"].append(t1 - t0)

            # 前向
            outputs = model(**batch_gpu)
            loss = outputs.loss
            t2 = time.perf_counter()
            times["forward"].append(t2 - t1)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()
            times["backward"].append(t3 - t2)

            times["total"].append(t3 - t0)

        end_total = time.perf_counter()

        result = {
            "config": cfg["label"],
            "num_workers": cfg["num_workers"],
            "pin_memory": cfg["pin_memory"],
            "data_prep_ms": sum(times["data_prep"]) / len(times["data_prep"]) * 1000,
            "forward_ms": sum(times["forward"]) / len(times["forward"]) * 1000,
            "backward_ms": sum(times["backward"]) / len(times["backward"]) * 1000,
            "total_per_step_ms": sum(times["total"]) / len(times["total"]) * 1000,
            "steps": len(times["total"]),
            "wall_time_s": end_total - start_total,
        }

        results.append(result)

        print(f"  数据准备: {result['data_prep_ms']:.2f} ms")
        print(f"  前向:     {result['forward_ms']:.2f} ms")
        print(f"  反向:     {result['backward_ms']:.2f} ms")
        print(f"  总计:     {result['total_per_step_ms']:.2f} ms/step")
        print(f"  墙上时间: {result['wall_time_s']:.2f} s")

    # 保存结果
    output_path = Path(output_dir) / "dataloader_pipeline.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    return results


def exp2_activation_checkpointing(output_dir):
    """
    实验 2: Activation Checkpointing

    对照：
    - 无 checkpointing（保存所有激活）
    - 有 checkpointing（重算部分激活）

    记录：
    - 前向峰值内存
    - 反向峰值内存
    - 计算时间
    """
    print("\n" + "="*70)
    print("实验 2: Activation Checkpointing")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolLM2-360M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备输入
    texts = ["The quick brown fox jumps over the lazy dog."] * 4
    encoded = tokenizer(texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    results = []

    for use_checkpoint in [False, True]:
        label = "有checkpointing" if use_checkpoint else "无checkpointing"
        print(f"\n配置: {label}")

        # 重新加载模型（清理状态）
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        model.train()

        # 启用 gradient checkpointing
        if use_checkpoint:
            model.gradient_checkpointing_enable()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 预热
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        gc.collect()

        # 正式计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        mem_before = measure_memory()

        t0 = time.perf_counter()

        optimizer.zero_grad()

        # 前向
        mem_after_forward_start = measure_memory()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        mem_after_forward = measure_memory()

        # 反向
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        mem_after_backward = measure_memory()

        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t3 = time.perf_counter()

        result = {
            "config": label,
            "use_checkpoint": use_checkpoint,
            "forward_ms": (t1 - t0) * 1000,
            "backward_ms": (t2 - t1) * 1000,
            "optimizer_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "mem_before_mb": mem_before.get("allocated_mb", 0),
            "mem_after_forward_mb": mem_after_forward.get("allocated_mb", 0),
            "mem_after_backward_mb": mem_after_backward.get("allocated_mb", 0),
            "mem_peak_forward_mb": mem_after_forward.get("max_allocated_mb", 0),
            "mem_peak_backward_mb": mem_after_backward.get("max_allocated_mb", 0),
        }

        results.append(result)

        print(f"  前向:     {result['forward_ms']:.2f} ms")
        print(f"  反向:     {result['backward_ms']:.2f} ms")
        print(f"  优化器:   {result['optimizer_ms']:.2f} ms")
        print(f"  总计:     {result['total_ms']:.2f} ms")
        print(f"  前向后内存: {result['mem_after_forward_mb']:.1f} MB")
        print(f"  反向后内存: {result['mem_after_backward_mb']:.1f} MB")
        print(f"  峰值内存:   {result['mem_peak_backward_mb']:.1f} MB")

        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # 保存结果
    output_path = Path(output_dir) / "activation_checkpointing.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    return results


def exp3_mixed_precision(output_dir):
    """
    实验 3: 混合精度训练

    对照：
    - FP32（基线）
    - FP16 + GradScaler
    - BF16（RTX 5090 支持）

    记录：
    - 前向/反向时间
    - 内存占用
    - loss 与梯度范数
    - GradScaler 的 scale 因子
    """
    print("\n" + "="*70)
    print("实验 3: 混合精度训练")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolLM2-360M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备输入
    texts = ["The quick brown fox jumps over the lazy dog."] * 4
    encoded = tokenizer(texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    configs = [
        {"dtype": torch.float32, "use_amp": False, "amp_dtype": None, "label": "FP32"},
        {"dtype": torch.float32, "use_amp": True, "amp_dtype": torch.float16, "label": "FP16+scaler"},
        {"dtype": torch.float32, "use_amp": True, "amp_dtype": torch.bfloat16, "label": "BF16"},
    ]

    results = []

    for cfg in configs:
        print(f"\n配置: {cfg['label']}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=cfg["dtype"],  # 模型参数始终 FP32
            device_map=device
        )
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler() if cfg["use_amp"] and cfg["amp_dtype"] == torch.float16 else None

        # 预热
        for _ in range(2):
            optimizer.zero_grad()
            if cfg["use_amp"]:
                with autocast(dtype=cfg["amp_dtype"]):  # autocast 只影响前向计算
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # 正式计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.perf_counter()

        optimizer.zero_grad()

        if cfg["use_amp"]:
            with autocast(dtype=cfg["amp_dtype"]):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()

        # 计算梯度范数（在 unscale 之前，梯度是 scaled 的）
        if scaler:
            # 先 unscale 梯度，再计算范数
            scaler.unscale_(optimizer)
            grad_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None))
            scaler.step(optimizer)
            scaler.update()
            scale_value = scaler.get_scale()
        else:
            grad_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            scale_value = 1.0

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t3 = time.perf_counter()

        mem = measure_memory()

        result = {
            "config": cfg["label"],
            "dtype": str(cfg["dtype"]),
            "use_amp": cfg["use_amp"],
            "amp_dtype": str(cfg["amp_dtype"]) if cfg["amp_dtype"] else None,
            "forward_ms": (t1 - t0) * 1000,
            "backward_ms": (t2 - t1) * 1000,
            "optimizer_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "scaler_scale": scale_value,
            "mem_allocated_mb": mem.get("allocated_mb", 0),
            "mem_peak_mb": mem.get("max_allocated_mb", 0),
        }

        results.append(result)

        print(f"  前向:       {result['forward_ms']:.2f} ms")
        print(f"  反向:       {result['backward_ms']:.2f} ms")
        print(f"  优化器:     {result['optimizer_ms']:.2f} ms")
        print(f"  总计:       {result['total_ms']:.2f} ms")
        print(f"  Loss:       {result['loss']:.6f}")
        print(f"  梯度范数:   {result['grad_norm']:.6f}")
        print(f"  Scale:      {result['scaler_scale']:.1f}")
        print(f"  峰值内存:   {result['mem_peak_mb']:.1f} MB")

        del model, optimizer, scaler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # 保存结果
    output_path = Path(output_dir) / "mixed_precision.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    return results


def exp4_memory_timeline(output_dir):
    """
    实验 4: 完整内存曲线

    记录训练步各阶段的内存：
    - 模型加载后
    - 数据 H2D 后
    - 前向后
    - 反向后
    - 优化器后
    - 清理后
    """
    print("\n" + "="*70)
    print("实验 4: 完整内存曲线")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolLM2-360M"

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gc.collect()

    timeline = []

    # 阶段 0: 初始状态
    mem = measure_memory()
    timeline.append({"stage": "0_initial", "desc": "初始状态", **mem})
    print(f"阶段 0 - 初始状态: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 1: 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.train()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "1_model_loaded", "desc": "模型加载后", **mem})
    print(f"阶段 1 - 模型加载: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 2: 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "2_optimizer_created", "desc": "优化器创建后", **mem})
    print(f"阶段 2 - 优化器: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 3: 数据准备与 H2D
    texts = ["The quick brown fox jumps over the lazy dog."] * 4
    encoded = tokenizer(texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "3_data_h2d", "desc": "数据 H2D 后", **mem})
    print(f"阶段 3 - 数据 H2D: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 4: 前向
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "4_forward", "desc": "前向后", **mem})
    print(f"阶段 4 - 前向: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 5: 反向
    loss.backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "5_backward", "desc": "反向后", **mem})
    print(f"阶段 5 - 反向: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 6: 优化器更新
    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "6_optimizer_step", "desc": "优化器更新后", **mem})
    print(f"阶段 6 - 优化器: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 7: 释放中间变量
    del outputs, loss

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "7_after_del", "desc": "释放中间变量后", **mem})
    print(f"阶段 7 - 释放变量: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 8: zero_grad
    optimizer.zero_grad()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "8_zero_grad", "desc": "zero_grad 后", **mem})
    print(f"阶段 8 - zero_grad: {mem.get('allocated_mb', 0):.1f} MB")

    # 阶段 9: 释放数据
    del input_ids, attention_mask, labels

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem = measure_memory()
    timeline.append({"stage": "9_data_freed", "desc": "释放数据后", **mem})
    print(f"阶段 9 - 释放数据: {mem.get('allocated_mb', 0):.1f} MB")

    # 保存结果
    output_path = Path(output_dir) / "memory_timeline.json"
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    return timeline


def main():
    """运行所有实验"""
    print("7.1 训练循环系统视角实验")
    print("="*70)

    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 创建输出目录
    output_dir = Path("results/crater/7.1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行实验
    results = {}

    try:
        results["exp1_dataloader"] = exp1_dataloader_pipeline(output_dir)
    except Exception as e:
        print(f"\n实验 1 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["exp2_checkpointing"] = exp2_activation_checkpointing(output_dir)
    except Exception as e:
        print(f"\n实验 2 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["exp3_mixed_precision"] = exp3_mixed_precision(output_dir)
    except Exception as e:
        print(f"\n实验 3 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["exp4_memory_timeline"] = exp4_memory_timeline(output_dir)
    except Exception as e:
        print(f"\n实验 4 失败: {e}")
        import traceback
        traceback.print_exc()

    # 保存汇总
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "experiments": list(results.keys()),
        }, f, indent=2)

    print("\n" + "="*70)
    print("所有实验完成")
    print(f"结果目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
