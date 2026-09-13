# L7 训练部分模型选型记录

日期：2026-09-12

## 选型决定

**主力模型：SmolLM3 3B** (HuggingFaceTB/SmolLM3-3B-Base)

## 理由

### 1. 前沿性与代表性
- **2026年发布**，使用现代架构：RoPE、Grouped Query Attention、RMSNorm、SwiGLU
- 支持长上下文（NoPE + YaRN 至 128k）
- 比 GPT-2 (2019) 更能代表当前训练栈

### 2. 完整透明度
- ✅ 开源训练配置（nanotron YAML）
- ✅ 11T tokens 完整训练数据配方
- ✅ 论文详细记录（arxiv/2502.02737）
- ✅ 多阶段训练策略、消融实验
- ✅ 官方 checkpoint 可直接使用

### 3. 教学友好
**不需要训练完整模型**：
- 官方已训练好的 checkpoint 可加载验证
- 用小数据集（1000-10000 样本）跑几个 step 验证机制
- 可在几小时内完成训练步验证
- 有多个训练阶段 checkpoint 可复用

### 4. 可借用结论
SmolLM2 论文（arxiv/2502.02737）提供：
- 训练曲线与 loss 趋势
- 数据混合比例的消融实验
- 多阶段训练的性能对比
- 不同优化器配置的影响

**我们专注于"怎么实现训练系统"，而非"训练出最好模型"**

### 5. 资源可行性
- **3B 参数**：单卡可运行部分实验
- **crater (RTX 5090 D, 32GB)**：可运行短时训练验证
- **worldvln (5× L40S, 48GB/卡)**：可验证多卡并行

### 6. 官方训练配置（可直接复用）
```yaml
# 来自 huggingface/smollm text/pretraining/smollm3/stage1_8T.yaml
model:
  model_type: llama
  num_hidden_layers: 32
  num_attention_heads: 32
  num_key_value_heads: 8  # GQA
  hidden_size: 3072
  intermediate_size: 8192
  max_position_embeddings: 4096

optimizer:
  learning_rate_scheduler:
    learning_rate: 2e-4
    lr_warmup_steps: 2000
    lr_warmup_style: linear
    lr_decay_style: cosine
    min_decay_lr: 2e-5
  
  optimizer_factory:
    name: adamW
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1e-8
    weight_decay: 0.1

tokens:
  batch_accumulation_per_replica: 1
  micro_batch_size: 8
  sequence_length: 4096
  train_steps: 4718000  # ~8T tokens
```

## 对比方案

### 备选：TinyLlama 1.1B
- **优势**：更小更快，8000+ GitHub stars，基于 lit-gpt
- **劣势**：2024年发布，比 SmolLM3 旧一代
- **用途**：如果 3B 资源受限，可降级到 1.1B

### 教学对照：GPT-2 124M (nanoGPT)
- **保留价值**：作为"经典 baseline"
- **使用场景**：
  - 7.0 autograd：tiny model（已完成）
  - 7.0b 对照：GPT-2 快速演示（1小时）
  - 7.1-7.6：SmolLM3 真实系统实验
- **形成梯度**：教学 toy → 经典 baseline → 前沿实践

## 实施策略

### 不需要完整训练

**原则：验证机制，不追求收敛**

#### 1. 加载官方 checkpoint（7.0b/7.1）
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
```

#### 2. 短时验证训练（几小时内）
- 用 1000-10000 样本
- 跑 10-100 个 step
- 验证：loss 下降、梯度流、内存曲线、checkpoint 恢复
- **不需要训练到任务质量指标**

#### 3. 借用官方数据（7.2-7.6）
- SmolLM2 论文有详细训练曲线、消融实验
- 引用数据并注明来源
- 我们专注"实现"，借用"结果"

#### 4. 关键机制的 mini 实现
- 梯度累积：自己写小例子对拍
- 混合精度：验证 loss scaling
- FSDP2 分片：两卡验证正确性
- Checkpoint 恢复：实际测中断恢复

## 各章节模型使用

| 章节 | 模型 | 用途 | 训练范围 |
|------|------|------|----------|
| 7.0 autograd | 自写 tiny model | 机制演示 | 已完成，CPU |
| 7.0b 完整训练步 | **SmolLM3 3B** | 真实数据、labels、loss、更新 | 10-100 steps，验证正确性 |
| 7.1 训练循环系统 | **SmolLM3 3B** | 内存曲线、checkpointing | 短时运行，profiling |
| 7.2 训练并行 | **SmolLM3 3B** | FSDP2 实测 | 两卡验证分片正确性 |
| 7.3 训练框架 | nanotron vs FSDP2 vs DeepSpeed | 架构对比 | 源码解析 + 配置对照 |
| 7.4 数据与恢复 | **SmolLM3 3B** | checkpoint、数据位置 | 中断恢复验证 |
| 7.5 SFT/LoRA | **SmolLM3 3B** | 微调、adapter | SmolTalk 数据集 |
| 7.6 RL infra | **SmolLM3 3B** | policy 版本、rollout | 借用 veRL 配置 |

## 资源需求估算

### 单卡验证（crater）
```
3B × 2 bytes (bf16) = 6 GB 权重
+ ~4 GB 优化器状态（AdamW）
+ ~2 GB 梯度
+ ~8 GB 激活（batch=8, seq=4096, 估算）
-------------------------------------
≈ 20 GB 峰值（32 GB 卡可运行）
```

### 多卡验证（worldvln）
- FSDP2：两卡即可验证分片
- 每卡显存需求降低到 ~12 GB
- 5 卡可运行更大 batch 或更长序列

## 数据集

### Pretraining（7.0b-7.2）
- **FineWeb-Edu**（HuggingFace）：高质量 web 文本
- **小样本验证**：1000-10000 文档，tokenize 后 ~40M tokens
- **不追求覆盖全部 11T tokens**

### Instruction Tuning（7.5）
- **SmolTalk**（官方数据集）：HuggingFaceTB/smoltalk
- **DPO**：UltraFeedback（官方使用）

### 快速实验数据
- **OpenWebText**（小版本）：~8GB，可快速下载
- **TinyStories**：极小数据集，纯语法验证

## 环境准备

### Nanotron 框架
```bash
# SmolLM3 使用的 branch
git clone https://github.com/huggingface/nanotron.git
cd nanotron
git checkout smollm3
pip install -e .
```

### Datatrove（数据处理）
```bash
git clone https://github.com/huggingface/datatrove.git
cd datatrove
git checkout nouamane/avoid-s3
pip install -e .
```

### 依赖
- PyTorch >= 2.1（FSDP2 需要）
- transformers >= 4.40
- datasets
- accelerate

## 验证计划

### Phase 1：单机单卡（7.0b/7.1）
- [ ] 加载 SmolLM3 checkpoint
- [ ] 准备小数据集（~1000 样本）
- [ ] 完整训练步：forward + backward + optimizer step
- [ ] 验证 loss 下降
- [ ] 梯度累积对拍
- [ ] 混合精度验证
- [ ] Activation checkpointing 内存对比
- [ ] Checkpoint 保存与恢复

### Phase 2：多卡并行（7.2）
- [ ] FSDP2 两卡分片
- [ ] 梯度与单卡对拍
- [ ] 通信 trace
- [ ] 内存曲线对比

### Phase 3：框架对照（7.3）
- [ ] Nanotron 配置解析
- [ ] FSDP2 原生 API
- [ ] DeepSpeed 配置映射
- [ ] TorchTitan（如果时间允许）

### Phase 4：数据与恢复（7.4）
- [ ] DataLoader state_dict
- [ ] 中断恢复验证
- [ ] RNG 状态对齐

### Phase 5：后训练（7.5/7.6）
- [ ] SmolTalk SFT
- [ ] LoRA adapter
- [ ] DPO 损失计算
- [ ] （7.6 的完整 RL 系统视资源而定）

## 参考资料

### 官方资源
- Model: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
- Paper: https://arxiv.org/abs/2502.02737
- Code: https://github.com/huggingface/smollm
- Training configs: https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3
- Nanotron: https://github.com/huggingface/nanotron

### 数据集
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
- Stack-Edu: （代码数据）

## 与 GPT-2 的对比

| 维度 | GPT-2 124M (2019) | SmolLM3 3B (2026) |
|------|-------------------|-------------------|
| 架构 | 绝对位置编码 | RoPE |
| Attention | Multi-head | Grouped Query (32h/8kv) |
| Normalization | LayerNorm | RMSNorm |
| FFN | GELU | SwiGLU |
| 训练数据 | WebText (~40GB) | 11T tokens (mixed) |
| 上下文 | 1024 | 128k (with YaRN) |
| 训练时间 | 数周（当时） | 24天×384 H100 |
| 可复现性 | 1小时×单卡（nanoGPT） | 需要集群（但可借用结果）|

**教学价值**：保留 GPT-2 作为"可以自己跑完"的 baseline，用 SmolLM3 展示"现代训练是怎么做的"。

## 风险与备选

### 风险
1. **容量不足**：3B 可能超出单卡显存
   - **缓解**：减小 batch size，使用梯度累积
   - **备选**：SmolLM2-360M 或 TinyLlama 1.1B

2. **多卡不可用**：worldvln 无法访问
   - **缓解**：FSDP2 可以在单卡上演示（分片但不真正分布）
   - **备选**：使用 Gloo backend 在本地多进程模拟

3. **Nanotron 兼容性**：smollm3 branch 可能有依赖冲突
   - **缓解**：使用 PyTorch FSDP2 原生 API
   - **备选**：lit-gpt 的 TinyLlama 路线

### 降级路径
如果 SmolLM3 3B 资源受限：
1. **SmolLM2-360M**：更小，1T tokens 训练
2. **TinyLlama 1.1B**：中等大小，有完整 lit-gpt 实现
3. **GPT-2 124M**：最小，nanoGPT 可直接运行

**当前计划**：先尝试 SmolLM3 3B，遇到阻塞再降级。

## 总结

- ✅ **主力模型**：SmolLM3 3B
- ✅ **策略**：验证机制，不追求训练完整模型
- ✅ **资源**：crater 单卡 + worldvln 多卡
- ✅ **数据**：小样本 + 借用官方结果
- ✅ **对照**：保留 GPT-2 作为教学 baseline
- ✅ **风险可控**：有明确降级路径

**下一步**：开始实施 7.0b - 完整训练步
