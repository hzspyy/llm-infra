# 训练材料与模型选型依据

本文件只记录模型和公开材料的选型依据；任务、依赖、交付与验收统一写入[已完成部分的修订计划](plans/completed.md)与[未完成部分的详细计划](plans/pending.md#training-workflow)。正文进度和实测证据以 [STATUS](../STATUS.md) 为准。

## 模型与运行范围

训练体系使用多个基础模型，不设覆盖全部模态的唯一语言模型基座。文本、视觉表示、ASR、codec/TTS、扩散/flow、视频/世界模型、Omni、VLA 分别学习自己的数据、目标、优化和导出流程；微调、蒸馏、QAT/QAD、RL 再与这些流程结合。

- 数值和状态验证复用 TinyLM、线性/projector/flow 小例及已测 SmolLM2-360M。一次更新、短恢复和少量两 rank 检查用于回答确定的机制问题。
- 文本、视觉适配和小型图像生成分别完成预算内的教学训练。文本从随机权重预训练到 SFT 和部署，LoRA/DPO/RL/蒸馏从共同起点分别开展；视觉适配明确复用的预训练模块，像素域图像生成具有独立基础目标。
- Puro-2B 与 SmolLM3-3B-Base 用作真实规模配方、阶段权重、公开日志和系统设计参照，不要求重做其全量预训练。
- 资料研读、本机机制小验证、本机完整教学项目与作者规模结果分别标注。FSDP2、DeepSpeed、Megatron、TorchTitan 均完成源码/状态/接口比较，实际教学项目只选择预算内适用路径。

## 低资源项目与规模参照

| 对象 | 教学用途与来源 | 选型与成本边界 |
|---|---|---|
| [MiniMind](https://github.com/jingyaogong/minimind) | 可读的模型、数据、预训练、SFT、LoRA、偏好/RL 与蒸馏实现；文本贯穿项目优先候选 | 当前主线约 64M，旧版本含 26M 配置；结构、tokenizer 和权重必须与选定版本对应。README 的“2 小时、3 元”注明为单张 3090 上 SFT 一轮，不能作为全部训练阶段预算 |
| [MiniMind-V](https://github.com/jingyaogong/minimind-v) | 图像→视觉特征→projector→语言模型的对齐与联合适配 | 当前实现复用语言模型和冻结视觉编码器；所谓从零构建 VLM 不等于所有参数从随机初始化训练。全模型参数、可训练参数和显存分别计算；版本变更后重查冻结策略 |
| [Diffusers 基础训练](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) | 随机初始化 UNet、加噪目标、训练、保存和采样；使用小图像集与小网络 | 官方示例是流程来源，本课程缩小数据/分辨率/模型后自行测算成本。像素域主例不依赖预训练 VAE 或语言模型；flow 参照和扩展按 10.1 任务执行 |
| [Puro-2B](https://arxiv.org/html/2608.27370) / [Puro-Megatron](https://github.com/thu-pacman/Puro-Megatron) | 消费级 GPU 上的规模预训练、FP8、MuonH、阶段数据与成本分析 | 报告的正式阶段用 24/96 张 RTX 5090，最佳模型约 22,514 活跃训练 GPU 小时、约 6,900 美元的归一化预训练计算成本；不含数据准备、后训练等全部研发支出，不能理解成单卡几小时项目 |

这些项目是实现载体，最终版本按目标、数据、可读性、显存和预算选择；替换时保持完整阶段及质量/恢复/部署验收。具体数据规模、运行上限、停止条件和责任章节统一见[贯穿任务](plans/pending.md#training-workflow)，本文件不另设执行计划。

## 文本案例的核实信息

信息依据：[SmolLM2-360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json)、[SmolLM3-3B-Base config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base/blob/main/config.json)、[SmolLM3 官方介绍](https://huggingface.co/blog/smollm3)、[Nanotron 配置](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3)。版本和用途必须与具体配置对应。

| 项目 | SmolLM2-360M | SmolLM3-3B-Base |
|---|---|---|
| 层数 / hidden size | 32 / 960 | 36 / 2048 |
| Query / KV heads | 15 / 5，GQA | 16 / 4，GQA |
| intermediate size | 2560 | 11008 |
| vocab size | 49152 | 128256 |
| 位置与上下文配置 | RoPE，config max_position_embeddings=8192 | 每 4 层无 RoPE；Base config max_position_embeddings=65536、rope_theta=5000000、rope_scaling=null |
| 本课程用途 | 复用既有训练步、累积、恢复和混合精度工件 | 研究真实架构、数据混合、多阶段训练、checkpoint、评测及部署产物 |
| 可比边界 | 模型结构和工作集与 3B 不同 | 不能以小模型结果替代 3B 训练吞吐、显存峰值或收敛结论 |

SmolLM3 于 2025 年发布。长上下文、YaRN 和 128k 能力需按 Base/Instruct 的实际 checkpoint/config 与官方说明分别分析，不能把宣传能力直接写成每份 config 的默认设置。

官方 stage1 YAML 的结构字段是 36 层、hidden_size=2048、16 query/4 KV heads；学习率采用 warmup 后稳定、末段线性衰减的 WSD 配方。示例包含原训练的内部路径和恢复位置，学习时需要区分参数含义、公开可用数据与原环境地址；不将摘录改写成未经核实的“官方完整可运行配置”。

## 可借用的完整流程材料

| 对象 | 可用材料 | 不能由这些材料推出的结论 |
|---|---|---|
| SmolLM3 | [预训练配置与日志入口](https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md)、[mid/SFT/APO](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3)、[中间权重](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)、[评测](https://github.com/huggingface/smollm/tree/main/text/evaluation/smollm3) | 中间 Transformers 权重不自动包含原训练 optimizer、RNG、数据游标；配置公开不代表所有内部数据路径可直接访问 |
| Tülu 3 | [SFT/DPO/RLVR 的训练、数据与模型阶段](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) | 作者的大规模结果不能作为本课程短小验证的质量结论 |
| 视觉与 VLM | [OpenCLIP](https://github.com/mlfoundations/open_clip) 的视觉/图文基础训练；[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune) 的适配 | OpenCLIP 配方不等于 Qwen 视觉塔原始训练；微调代码不等于完整预训练公开 |
| ASR | [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer) 基础训练；[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) 微调 | 两种模型的 tokenizer、目标、采样率和流式支持不能相互套用 |
| Codec 与 TTS | [DAC](https://github.com/descriptinc/descript-audio-codec)、[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)、[F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train)、[ZipVoice](https://github.com/k2-fsa/ZipVoice) | 下游 TTS SFT 不等于训练 codec/vocoder；Qwen 单说话人示例不等于完整多说话人预训练 |
| 草稿、量化与蒸馏 | [SpecForge](https://github.com/sgl-project/SpecForge)、[ModelOpt QAT/QAD](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat)、[TRL](https://github.com/huggingface/trl)、[LCM](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)、[DMD2](https://github.com/tianweiy/DMD2) | 不同 teacher 信号/学生目标不能统称同一种 KD；runtime checkpoint 不一定能直接服务 |
| 图像、视频与世界模型 | [flow_matching](https://github.com/facebookresearch/flow_matching)、[DiffSynth Wan](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training)、[Cosmos3-Edge SFT](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/docs/training.md) | Wan 微调实现和 Cosmos 生成分支 SFT 不证明原始全量预训练/所有蒸馏阶段均已公开 |
| Omni / VLA | [Qwen3-Omni 报告](https://arxiv.org/html/2509.17765)、[ms-swift](https://github.com/modelscope/ms-swift/tree/main/examples/models/qwen3_omni)、[openpi DROID](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md) | 接受多模态输入不等于训练所有输出模块；离线动作训练不等于在线 RL 或真实闭环成功 |
| 非文本 RL | [Flow-GRPO](https://github.com/yifan123/flow_grpo)、[DanceGRPO](https://github.com/XueZeyue/DanceGRPO)、[CosyVoice2 GRPO](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2) | 连续转移/codec token 的概率和奖励不能直接套成文本 CE；参考流程不代表所有目标型号都有官方 RL 配方 |

## 训练状态与容量依据

令参数量为 P，具体容量以实测 tensor/optimizer state dtype 为准：

| 状态假设 | 不含激活、临时 buffer、通信与框架开销的字节 |
|---|---:|
| BF16 参数＋BF16 梯度＋FP32 Adam m/v，无额外 master | 2P＋2P＋8P＝12P |
| 上述状态再保留 FP32 master weight | 16P |
| FP32 参数＋FP32 梯度＋FP32 Adam m/v，autocast 计算 | 16P；低精度缓存另算 |
| LoRA/冻结模块 | 分开计算冻结 base、可训练 adapter 的梯度/optimizer、需要保存的激活；不是按总参数统一乘一个系数 |

以 P≈3B 估算，12P 已约 36 GB，16P 约 48 GB，尚未计入激活和临时状态。GB 与 GiB 必须区分；不能据“BF16 权重约 6 GB”断言 3B AdamW 全参更新能装进 32 GB 显存。某些 PyTorch 模式下 optimizer 状态会跟随参数 dtype，是否保持 FP32 必须逐 recipe 核对。

FSDP2 在单 rank 下没有跨 rank 分片收益；梯度累积和 activation checkpointing 不减少全部常驻训练状态。CPU/Gloo 的小通信参照只验证算法与协议，不能替代 GPU FSDP/NCCL 的支持性和性能证明。

混合精度的基础任务在 7.1；完整参数/计算/梯度/归约/optimizer 精度、GradScaler、FP8/MXFP8/NVFP4、缩放与数值诊断见 [7.9](plans/pending.md#c-7-9)。环境与存储遵循 [ENVIRONMENTS](../ENVIRONMENTS.md) 和[实验规范](experiment-guidelines.md)。
