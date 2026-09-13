#!/usr/bin/env python3
"""
SmolLM3 3B 完整训练步实验

验证：
1. 真实模型的 labels 移位、loss mask、有效 token 归一化
2. 梯度累积 vs 直接 batch
3. AdamW 优化器状态
4. Checkpoint 保存与恢复

运行：
  python training_step_smollm.py

环境要求：
  - transformers >= 4.40.0
  - torch >= 2.0.0
  - 单卡 24GB+ 显存（可调整 batch_size）
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import json
from pathlib import Path


def prepare_batch(tokenizer, texts, max_length=512):
    """
    准备训练 batch：tokenization + padding + labels 移位

    Returns:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        labels: [batch_size, seq_len]，padding 位置为 -100
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Labels: 右移一位（decoder-only 模型）
    # input:  [A, B, C, D, pad, pad]
    # labels: [B, C, D, pad, pad, pad]  但 padding 设为 -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # padding 位置不计算 loss

    # 注意：Hugging Face 的 forward 内部会自动处理移位
    # 这里我们手动展示移位逻辑

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def compute_loss_breakdown(model, batch, device):
    """
    计算逐 token loss 与统计信息

    Returns:
        total_loss: 标量
        per_token_losses: [num_valid_tokens]
        valid_token_count: int
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 前向
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    total_loss = outputs.loss

    # 手动计算逐 token loss（验证）
    logits = outputs.logits  # [B, L, V]

    # Shift for decoder-only（与 labels 对齐）
    shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
    shift_labels = labels[..., 1:].contiguous()      # [B, L-1]

    # 逐 token loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    per_token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # 过滤出有效 token 的 loss
    valid_mask = (shift_labels.view(-1) != -100)
    valid_losses = per_token_losses[valid_mask]

    # 有效 token 数
    valid_count = valid_mask.sum().item()

    # 验证：手动计算的平均 loss 应该等于 model 输出的 loss
    manual_loss = valid_losses.sum() / valid_count

    print(f"  Model loss:  {total_loss.item():.6f}")
    print(f"  Manual loss: {manual_loss.item():.6f}")
    print(f"  差异:        {abs(total_loss.item() - manual_loss.item()):.9f}")

    return total_loss, valid_losses, valid_count


def training_step_detailed(output_dir):
    """完整训练步：包含所有状态检查"""
    print("\n" + "="*70)
    print("实验 1: 完整训练步分析")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型（使用小模型，如果显存不足）
    # SmolLM3-3B 需要约 12GB 显存（FP32）或 6GB（FP16）
    model_name = "HuggingFaceTB/SmolLM2-360M"  # 先用 360M 测试，验证通过后换 3B
    print(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # FP32 便于数值验证
        device_map=device
    )
    model.train()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # 准备数据
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used for typing practice.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    ]

    batch = prepare_batch(tokenizer, texts, max_length=128)

    # 打印样本信息
    print("\n样本内容（前 50 token）:")
    for i, text in enumerate(texts):
        ids = batch["input_ids"][i][:50].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"  样本 {i}: {decoded[:100]}...")
        print(f"    input_ids[:10]:  {ids[:10]}")
        print(f"    labels[:10]:     {batch['labels'][i][:10].tolist()}")
        print(f"    有效 token 数:   {(batch['labels'][i] != -100).sum().item()}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 训练一步
    print("\n执行训练步...")
    optimizer.zero_grad()

    loss, per_token_losses, valid_count = compute_loss_breakdown(model, batch, device)

    print(f"\nLoss 统计:")
    print(f"  总 loss:        {loss.item():.6f}")
    print(f"  有效 token 数:  {valid_count}")
    print(f"  逐 token loss (前10): {per_token_losses[:10].tolist()}")
    print(f"  最大 loss:      {per_token_losses.max().item():.6f}")
    print(f"  最小 loss:      {per_token_losses.min().item():.6f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    print(f"\n梯度统计:")
    print(f"  所有参数梯度范数均值: {sum(grad_norms) / len(grad_norms):.6f}")
    print(f"  最大梯度范数:          {max(grad_norms):.6f}")
    print(f"  最小梯度范数:          {min(grad_norms):.6f}")

    # 查看第一层权重的梯度
    first_param = next(model.parameters())
    print(f"  第一层参数形状:        {first_param.shape}")
    print(f"  第一层梯度 (前5):      {first_param.grad.flatten()[:5].tolist()}")

    # 参数更新
    param_before = {name: param.clone().detach() for name, param in model.named_parameters()}
    optimizer.step()

    # 检查参数变化
    param_changes = []
    for name, param in model.named_parameters():
        change = (param - param_before[name]).abs().max().item()
        param_changes.append(change)

    print(f"\n参数更新统计:")
    print(f"  最大参数变化:   {max(param_changes):.9f}")
    print(f"  平均参数变化:   {sum(param_changes) / len(param_changes):.9f}")
    print(f"  第一层参数变化: {(first_param - param_before[next(iter(param_before.keys()))]).flatten()[:5].tolist()}")

    # 检查优化器状态
    print(f"\nAdamW 优化器状态:")
    state = optimizer.state[first_param]
    print(f"  step:         {state['step']}")
    print(f"  exp_avg (m):  形状 {state['exp_avg'].shape}, 前5个值 {state['exp_avg'].flatten()[:5].tolist()}")
    print(f"  exp_avg_sq (v): 形状 {state['exp_avg_sq'].shape}, 前5个值 {state['exp_avg_sq'].flatten()[:5].tolist()}")

    # 保存 checkpoint
    ckpt_path = Path(output_dir) / "checkpoint_step1.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保 RNG state 是 uint8 类型且在 CPU 上
    rng_state = torch.get_rng_state().cpu()
    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)

    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if cuda_rng_state is not None:
        cuda_rng_state = [s.cpu().to(torch.uint8) if s.dtype != torch.uint8 else s.cpu() for s in cuda_rng_state]

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': rng_state,
        'cuda_rng_state': cuda_rng_state,
        'step': 1,
        'loss': loss.item(),
    }

    torch.save(checkpoint, ckpt_path)
    ckpt_size = ckpt_path.stat().st_size / 1024 / 1024
    print(f"\nCheckpoint 已保存:")
    print(f"  路径: {ckpt_path}")
    print(f"  大小: {ckpt_size:.1f} MB")
    print(f"  包含键: {list(checkpoint.keys())}")

    # 验证恢复
    print("\n验证 checkpoint 恢复...")
    model_recovered = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model_recovered.train()
    optimizer_recovered = torch.optim.AdamW(model_recovered.parameters(), lr=1e-4, weight_decay=0.01)

    # 加载 checkpoint（先加载到 CPU 避免 RNG state 被移到 GPU）
    checkpoint_loaded = torch.load(ckpt_path, map_location='cpu')

    # 先加载模型权重并移到 GPU
    model_recovered.load_state_dict(checkpoint_loaded['model_state_dict'])
    model_recovered = model_recovered.to(device)

    # 重新创建优化器（关联到 GPU 上的参数）
    optimizer_recovered = torch.optim.AdamW(model_recovered.parameters(), lr=1e-4, weight_decay=0.01)
    # 然后加载优化器状态
    optimizer_recovered.load_state_dict(checkpoint_loaded['optimizer_state_dict'])

    # RNG state 必须在 CPU 上且是 uint8
    rng_state = checkpoint_loaded['rng_state']
    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)
    torch.set_rng_state(rng_state)

    if checkpoint_loaded['cuda_rng_state'] is not None:
        cuda_rng_state = checkpoint_loaded['cuda_rng_state']
        if not isinstance(cuda_rng_state, list):
            cuda_rng_state = [cuda_rng_state]
        # 确保每个 state 都是 uint8
        cuda_rng_state = [s.to(torch.uint8) if s.dtype != torch.uint8 else s for s in cuda_rng_state]
        torch.cuda.set_rng_state_all(cuda_rng_state)

    # 验证：恢复后的模型在相同输入上应产生相同的 loss
    # 注意：我们保存的是 step 1 之后的模型，所以这里验证的是恢复后第二步的 loss
    optimizer_recovered.zero_grad()
    loss_step2, _, _ = compute_loss_breakdown(model_recovered, batch, device)
    loss_step2.backward()
    optimizer_recovered.step()

    # 同时用原始模型跑第二步作为对照
    optimizer.zero_grad()
    loss_step2_original, _, _ = compute_loss_breakdown(model, batch, device)
    loss_step2_original.backward()
    optimizer.step()

    print(f"\n恢复验证:")
    print(f"  Step 1 后 loss:          {loss.item():.6f}")
    print(f"  恢复模型 Step 2 loss:    {loss_step2.item():.6f}")
    print(f"  原始模型 Step 2 loss:    {loss_step2_original.item():.6f}")
    print(f"  两者差异:                {abs(loss_step2.item() - loss_step2_original.item()):.9f}")

    assert abs(loss_step2.item() - loss_step2_original.item()) < 1e-5, "恢复后训练轨迹不一致"
    print("  ✓ 恢复验证通过")

    return {
        'loss': loss.item(),
        'valid_tokens': valid_count,
        'grad_norm_mean': sum(grad_norms) / len(grad_norms),
        'param_change_max': max(param_changes),
        'checkpoint_size_mb': ckpt_size,
    }


def gradient_accumulation_comparison(output_dir):
    """对比梯度累积 vs 直接大 batch"""
    print("\n" + "="*70)
    print("实验 2: 梯度累积 vs 直接大 batch")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolLM2-360M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 两个独立模型
    model_accum = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model_direct = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )

    # 确保初始权重相同
    model_direct.load_state_dict(model_accum.state_dict())

    model_accum.train()
    model_direct.train()

    optimizer_accum = torch.optim.AdamW(model_accum.parameters(), lr=1e-4)
    optimizer_direct = torch.optim.AdamW(model_direct.parameters(), lr=1e-4)

    # 准备数据：4 个 micro_batch
    texts_all = [
        "The first example sentence for gradient accumulation test.",
        "The second example with different content and length.",
        "Third sentence is about machine learning and deep learning.",
        "Fourth and final sentence concludes the micro batch set.",
    ]

    micro_batches = [
        prepare_batch(tokenizer, [text], max_length=64)
        for text in texts_all
    ]

    # 路径 1: 梯度累积
    print("\n梯度累积路径 (4 个 micro_batch)...")
    optimizer_accum.zero_grad()
    loss_accum_total = 0.0

    for i, micro_batch in enumerate(micro_batches):
        input_ids = micro_batch["input_ids"].to(device)
        labels = micro_batch["labels"].to(device)
        attention_mask = micro_batch["attention_mask"].to(device)

        outputs = model_accum(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / 4  # 除以累积步数
        loss.backward()
        loss_accum_total += loss.item()

        print(f"  Micro-batch {i}: loss = {loss.item() * 4:.6f} (归一化前)")

    optimizer_accum.step()

    # 路径 2: 直接大 batch
    print("\n直接大 batch 路径 (batch_size=4)...")
    big_batch = prepare_batch(tokenizer, texts_all, max_length=64)

    optimizer_direct.zero_grad()
    input_ids_big = big_batch["input_ids"].to(device)
    labels_big = big_batch["labels"].to(device)
    attention_mask_big = big_batch["attention_mask"].to(device)

    outputs_direct = model_direct(input_ids=input_ids_big, attention_mask=attention_mask_big, labels=labels_big)
    loss_direct = outputs_direct.loss
    loss_direct.backward()
    optimizer_direct.step()

    print(f"  Direct batch loss: {loss_direct.item():.6f}")

    # 比较
    print(f"\n结果对比:")
    print(f"  梯度累积总 loss (×4): {loss_accum_total * 4:.6f}")
    print(f"  直接 batch loss:      {loss_direct.item():.6f}")
    print(f"  Loss 差异:            {abs(loss_accum_total * 4 - loss_direct.item()):.6f}")

    # 比较最终参数
    params_accum = torch.cat([p.data.flatten() for p in model_accum.parameters()])
    params_direct = torch.cat([p.data.flatten() for p in model_direct.parameters()])
    param_diff = (params_accum - params_direct).abs().max().item()

    print(f"  参数最大差异:         {param_diff:.9f}")

    # 注意：由于有效 token 数不同，loss 可能略有差异
    # 但参数更新应该在数值误差范围内一致
    if param_diff < 1e-5:
        print("  ✓ 梯度累积与直接 batch 等价")
    else:
        print(f"  ⚠ 参数差异较大，可能由于归一化或有效 token 统计不同")

    return {
        'loss_accumulated': loss_accum_total * 4,
        'loss_direct': loss_direct.item(),
        'param_diff': param_diff,
    }


def main():
    output_dir = "results/crater/7.0b"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SmolLM3 训练步实验")
    print("="*70)

    results = {}

    # 实验 1
    results['detailed_step'] = training_step_detailed(output_dir)

    # 实验 2
    results['gradient_accumulation'] = gradient_accumulation_comparison(output_dir)

    # 保存结果
    results_path = Path(output_dir) / "training_step_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {results_path}")
    print("\n" + "="*70)
    print("✓ 所有实验完成")
    print("="*70)


if __name__ == "__main__":
    main()
