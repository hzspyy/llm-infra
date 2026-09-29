# 已实现章节的技术审阅笔记

本文记录早期审阅发现的技术问题，部分内容已有后续修订。当前任务与进度以[修订计划](../plans/completed.md)、[未完成部分任务书](../plans/pending.md)和[STATUS](../../STATUS.md)为准。

## 技术问题

### R01 · P1 · 7.3 的“SmolLM3 官方配方”与指定对象不符

位置：`src/L7/7.3-training-frameworks.md:116`；`src/L7/7.3-training-frameworks.md:337`；`labs/L7/recipe_inspector.py:470`。

正文写 Stage 1 为 3T、64 卡纯 FSDP、GBS=512，并将 50/30/10/10 数据混合及后续阶段写成官方材料。脚本又指定 vocab=49152、hidden=3072、Q/KV heads=24/8。固定官方 YAML 实际为 vocab=128256、hidden=2048、heads=16/4、DP=192、TP=2、PP=1、ZeRO stage=0、microbatch=3、accumulation=1；GBS=576，LR=2e-4。官方博客把首阶段描述为 0→8T，总训练 11.2T、384 H100；不同 YAML 的结束步/文件名要独立解释，不能拼接成新配方。

这会同时误导参数量、显存、训练步数、框架选择和阶段接续。检查器只是验证其硬编码对象自洽，不能证明对象来自官方。`smollm3_recipe_manifest.json` 也是脚本构造的二次材料，不是官方原始配置。必须按真实 YAML 做规范化映射，合成用例另命名。

依据：[官方 stage1 YAML](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml)、[官方训练说明](https://huggingface.co/blog/smollm3)。对应 7.3-A/C/F、7.0b-D。

### R02 · P1 · 7.3 把 FSDP2 参数描述成一维 flatten

位置：`src/L7/7.3-training-frameworks.md:135`；`src/L7/7.3-training-frameworks.md:183`；`src/L7/7.3-training-frameworks.md:419`。

FSDP2 是逐参数分片，默认沿 dim-0；DTensor 保持全局参数 shape，不是 FSDP1 的 FlatParameter 方案。正文的“一维平铺 DTensor”及自测答案会让读者误解参数身份和 checkpoint 布局。应分别列全局 shape、local shard shape、placement 和 optimizer 引用。

依据：[FSDP2 API](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)、[官方 FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。对应 7.2-A、7.3-B。

### R03 · P1 · 7.7 对固定 TRL commit 的源码解释是另一种训练算法

位置：`src/L7/7.7-distillation-training-systems.md:152`。

正文标出 commit `cd2c5287`，给出的却是固定 inputs 上 `(1-alpha)*CE + alpha*KL`。该 commit 的 DistillationTrainer 由学生在线生成 completion，通过 hidden states 分块投影词表计算 forward/reverse KL 或 generalized JSD，beta 控制方向，使用生成 completion 的有效位置归一化。源码没有正文那段 `self.args.alpha` 复合损失，所示路径也没有统一乘 T² 的操作。必须把通用 KD 示例与实际 Trainer 分开，按真实生成、mask、beta、chunk 和 reduction 重新解析。

依据：[指定 commit 原文件](https://github.com/huggingface/trl/blob/cd2c52876d99b00baa5660328310e940b8360c6c/trl/trainer/distillation_trainer.py)，关键位置 `_chunk:105`、分歧计算 `:140`、Trainer 说明 `:296`、`compute_loss:1801`、`_compute_loss:1828`。对应 7.7-B/C。

### R04 · P1 · 7.7 将 EAGLE3 写成隐藏层 MSE 回归

位置：`src/L7/7.7-distillation-training-systems.md:136`。

EAGLE3 融合多层教师特征作为条件，但移除了旧 EAGLE 的 feature prediction constraint，直接训练 token prediction。正文将其归入隐藏层 MSE，并据此宣称避开全词表计算，混淆了输入特征与监督目标。简单 feature KD 可以作为独立参照，不能代替 EAGLE3 objective。计划 7.7-A/E 明确要求分开。

依据：[EAGLE-3 原论文](https://arxiv.org/abs/2503.01840)、已固定的 SpecForge 训练材料。

### R05 · P1 · 7.9 混淆 autocast、显式 BF16 与 FSDP mixed precision

位置：`src/L7/7.9-optimizer-mixed-precision.md:146`；`src/L7/7.9-optimizer-mixed-precision.md:201`；`labs/L7/training_precision_ledger.py:157`。

autocast 不会自动将持久参数变为 BF16。CPU 小反例中，FP32 模型加 BF16 autocast 的参数/梯度/m/v 仍为 FP32，只有线性层输出为 BF16；显式 BF16 参数加原生 AdamW 时 m/v 跟随为 BF16。正文“m/v 必须 FP32”“BF16 一律 12P”均不成立。FSDP2 `param_dtype` 控制展开参数与计算/all-gather dtype，optimizer 用原始 dtype 的分片参数，不能解释为分片常驻 BF16。

需要按实际 recipe 输出 tensor/state dtype，再推字节账；FP32 m/v 与保存低精度参数舍入残差也不是同一保证。依据：[AMP](https://docs.pytorch.org/docs/stable/amp)、[MixedPrecisionPolicy](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)、归档的本机源码及 `actual_dtype_ledger` 反例。对应 7.9-B/C。

### R06 · P1 · 7.9 的 GradScaler 操作建议会使 scheduler 判定错误

位置：`src/L7/7.9-optimizer-mixed-precision.md:352`；`labs/L7/loss_scaling_reference.py:76`。

官方 `GradScaler.step()` 返回 optimizer 的返回值，不是成功布尔值。反例中 SGD 成功把参数从 1 改为 0.8，返回仍为 None；溢出跳步也返回 None。照文中做法可能永久不推进 scheduler。应显式观测有效更新，或在明确单 optimizer 等前提下使用合适的 scale/状态判定。官方 scale 可降到 1 以下，反例从 1 降为 0.5；mini 硬钳在 1 会改变恢复行为，必须修正或标明差异。

依据：归档 `torch/amp/grad_scaler.py:375`、`:484`，`scaler_return_and_scale` 反例；[官方 AMP 文档](https://docs.pytorch.org/docs/stable/amp)。对应 7.9-D。

### R07 · P1 · 7.9 的 ULP 读数与工件/浮点运算不符

位置：`src/L7/7.9-optimizer-mixed-precision.md:286`；`src/L7/7.9-optimizer-mixed-precision.md:179`。

正文称 `1-0.0039` 在 BF16 不变、`1-0.00048` 在 FP16 不变、FP32 在 1e-8 仍更新。反例分别得到 0.99609375、0.99951171875、1.0，三项均相反。1 是二进制指数边界，向下邻点间距与向上 ULP 不同；round-to-nearest 还涉及半间距及 ties。这里应称更新被舍入吸收，不能等同指数范围下溢。原 `precision_bytes_ledger.json` 必须只读，依据实际读数重写解释。7.3 的自测答案与 7.9 还把 FP16 最小正规数当成归零阈值；CPU 转换中 1e-5 与 1e-7 均得到非零次正规数，具体 GPU 算子的 flush 行为须另核对，不能从存储格式直接推断全部梯度归零。对应 7.9-C。

### R08 · P1 · 7.6 把 PPO-Clip 当作策略比率硬约束

位置：`src/L7/7.6-rl-infra-rollout.md:140`。

`min(r*A, clip(r)*A)` 不是把所有有效比率截进区间，更不保证新策略留在信任域。反例 `A=-1,r=10` 时 loss=10，对 r 的梯度为 1，完全没有被限制到 1.2；参数共享和优化器更新也可使其它动作的比率越界。原“策略滞后安全边界”不能成立。必须分别解释 surrogate clipping、KL 监控、rollout correction 与过期轨迹处理。

依据：[PPO 官方说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)、`ppo_negative_advantage` 反例。对应 7.6-A/E。

### R09 · P1 · 7.2 的通信负载和带宽口径错位

位置：`labs/L7/distributed_collectives.py:69`；`labs/L7/distributed_collectives.py:120`；`src/L7/7.2-training-parallelism.md:88`；`src/L7/7.2-training-parallelism.md:102`。

脚本 all-gather 的 `tensor_mb` 是单 rank 输入，reduce-scatter 是单 rank 输出，但带宽公式将它当全局总张量 S，少乘 world_size。4 rank 时两者带宽需乘 4；256 MiB 档约为 9.06 与 8.41 GB/s，而非 2.26 与 2.10。all-reduce 的 256 MiB 输入与 all-gather 的 1 GiB 总输出也不是等量任务。所谓“四倍差距来自 all-gather 无流水空间”没有成立依据。

`2(N-1)/N*S/time` 是派生 bus bandwidth，不能反过来证明 NCCL 选择了 ring；需要算法日志/trace。reduce-scatter 示例 4 个长度 4 输入应每 rank 得到长度 1 的 `[28] [32] [36] [40]`，原文给出了 8 个数。另将 PCIe Gen4 x16 单向理论值写为约 16 GB/s（应约 31.5 GB/s）并将 1024 卡均当作单一 NVSwitch 域，也不能支持“大模型必须 NVLink、PCIe 只能实验”的结论。

依据：[NCCL tests 带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md?plain=1)、实际脚本与原始 JSON。对应 7.2-A/C；本轮仅重算口径，未重跑 GPU。

### R10 · P1 · 7.0b 的全局 loss 示例不可运行，且缺 DDP 平均补偿

位置：`src/L7/7.0b-one-training-step.md:495`；`src/L7/7.0b-one-training-step.md:477`。

`dist.all_reduce()` 原地写 tensor、同步调用返回 None，因而两项返回值相除会报错。不能直接对 loss 用普通原地 collective 替代可微归约。在 DDP 默认平均梯度的条件下，等 token 权重目标通常应先归约全局有效计数 N，再让本 rank 的可微 loss 为 `world_size * local_loss_sum / N`，另做 detached 全局日志统计；还须处理 N=0。size-one Gloo 反例已核对 API 返回值；多 rank 更新对拍仍需计划规定的小验证。

同章 `:470` 把同步 `dcp.save()` 写成异步保存也应改为真实 async API 并解释 staging/upload 完成边界。对应 7.0b-B/C、7.2-B。

### R11 · P1 · 7.1 用无有效更新保证的 FP16 时间作为训练加速

位置：`src/L7/7.1-training-loop-systems.md:158`；`labs/L7/training_loop_systems.py:350`。

FP16 工件梯度范数 inf、scale 回退，脚本调用 scaler.step 却没有记录 optimizer 是否实际执行、状态是否已分配。因此 6.2 ms optimizer 段、46.9 ms 总时间与较低内存不能归因于相同训练工作的加速；应核对逐张量 finite、有效更新和 optimizer state，不能只报范数。当前脚本输入为 4×256，正文写 8×512，测量配置与归档脚本关系也需补 manifest。两个预热阶段执行更新，数值对照须核对是否从等价参数和状态出发。

本轮没有 GPU 重测，不确定首次非有限值产生位置；这一限制不能被 BF16 的一次有限 loss 填补。对应 7.1-A/C、7.9-H。

### R12 · P1 · 7.10 的联合训练壳没有实现所声称的 VLM/flow 更新

位置：`labs/L7/multimodal_train_contract.py:75`；`labs/L7/multimodal_train_contract.py:84`；`src/L7/7.10-multimodal-training-orchestration.md:230`。

图文只拼接后逐位置 Linear，文本位置无法依赖图像；视觉位置还被随机词表 labels 监督。flow 分支忽略 noisy_latent，没有时间输入，只由文本均值预测。验证函数仅 backward，没有 optimizer.step。反例改变图像后文本 logits 差值为 0，改变 latent 后 flow 输出差值也为 0。这最多验证独立张量分支求梯度，不能验收图文监督、条件 flow 或“一次联合更新”。需满足至少三种目标、正确 mask/分母、输入依赖和参数前后对拍，才满足 7.10-B/C。

### R13 · P1 · 7.10 从 loss 数值比例直接推导梯度比例

位置：`src/L7/7.10-multimodal-training-orchestration.md:129`；`src/L7/7.10-multimodal-training-orchestration.md:321`。

CE=3.5、MSE=0.015 不能推出梯度相差 200 倍，更不能推出图像分支无法更新。加一个常数能改变 loss 数值而不改变梯度；反例中大 loss 梯度 0.001，小 loss 梯度 10。应在共享参数和各自参数组分别测量梯度范数、方向与更新量，再讨论权重选择；独立 head 不会仅因另一项 loss 数值大就失去梯度。当前 JSON 的 233.3 比值是常数相除，没有测量所宣称的梯度后果。对应 7.10-B/I。

### R14 · P1 · 7.10 混淆 optimizer 组间重复与组内重复

位置：`src/L7/7.10-multimodal-training-orchestration.md:304`；`labs/L7/multimodal_train_contract.py:162`。

PyTorch optimizer 在不同参数组包含相同 Parameter 时直接 ValueError，不能写成每步静默更新两次。某些版本同一组内重复可能警告并重复更新，两种情形必须分别复现。现脚本仅自己数 id 后填入 consequence。反例与归档 `Optimizer.add_param_group` 已确认组间报错。对应 7.10-C；与 2.0 的组内重复实例不能混用。

### R15 · P1 · 7.11 把数据结构模拟写成子进程故障测试

位置：`labs/L7/training_job_simulator.py:139`；`labs/L7/training_job_simulator.py:146`；`src/L7/7.11-training-job-orchestration.md:298`。

所谓 spawn 是生成 RankHeartbeat 字典，terminate 是 clear；没有启动进程、发送 SIGTERM、检查 PID 或依据 heartbeat timeout 检测超时。日志里的这些动词只是文字。每个成功步立即在内存登记 checkpoint，失败步不进入 consumed 列表，不能证明跨进程 ACK、未提交 checkpoint 或真实清理。应登记为单进程离散状态模拟，7.11-F 所要求的子进程验证仍缺。

### R16 · P1 · 7.11 的 MFU 数值来自合成常量，不是 64 卡测量

位置：`labs/L7/training_trace_analysis.py:162`；`src/L7/7.11-training-job-orchestration.md:267`；`src/L7/7.11-training-job-orchestration.md:151`。

1285 ms 由写死的阶段时长相加得到，46.41%/61.88% 是公式输出，没有读取原始 trace 或作者训练日志；只构造了 4 个代表 rank，再除以 64 卡峰值。可作为明确假设的预算练习，不能写“实测稳态 MFU”。6P/8P 是忽略部分算子的近似口径，HFU 不能直接称为实际硬件计数。MFU 也不是有效学习进度或投资回报的唯一指标，不能替代质量、有效更新、成本与 goodput。对应 7.11-C/D/G/H。

### R17 · P1 · 多章以“源码解析”标题承载未标注的简化/错误实现

位置：`src/L7/7.3-training-frameworks.md:151`；`src/L7/7.4-data-ckpt-fault.md:170`；`src/L7/7.10-multimodal-training-orchestration.md:152`；`src/L7/7.11-training-job-orchestration.md:163`。

这些章节没有明确版本的源码链接与 file:line。与本次固定源码比较，Qwen3-VL `train_qwen.py` 是训练装配和冻结控制，不含正文给出的 `forward_multimodal`；TorchTitan 的实际 trainer/checkpointer/profiler 也不是所展示的函数体。简化伪代码本身可以保留，但必须明确标签，紧邻真实原文和调用位置，不能声称是已固定源码摘录。四框架比较亦不能由 TorchTitan、DeepSpeed 两段骨架代替。

可核对文件与 commit 全部归档于 source manifest。对应章节规范第 1/3 项、7.3-B/E、7.4-C、7.10-F/G、7.11-D。

### R18 · P1 · 7.10 的模型架构分类与计划指定对象冲突

位置：`src/L7/7.10-multimodal-training-orchestration.md:177`；`src/L7/7.10-multimodal-training-orchestration.md:291`。

Chameleon 被列为 Flamingo 式 cross-attention，实际是 early-fusion token-based 模型；Qwen3-Omni 被概括为单个自回归 Transformer，掩盖了 Thinker/Talker 及语音输出路径。必须分别核对模块与 loss reach，不能从统一多模态接口推断联合训练所有模块。

依据：[Chameleon 原论文](https://arxiv.org/abs/2405.09818)、[Qwen3-Omni 技术报告](https://arxiv.org/abs/2509.17765)。对应 7.10-A/G。

### R19 · P2 · 7.4 把自定义提交协议的保证归给 DCP

位置：`src/L7/7.4-data-ckpt-fault.md:193`；`src/L7/7.4-data-ckpt-fault.md:249`。

自有模拟器使用 JSON 分片与 latest 标记，不是 DCP 实测。当前 PyTorch FileSystemWriter 将元数据临时文件改名为 `.metadata`；其存储索引不等于正文宣称的全局分片内容校验和，也不是整套 checkpoint staging 目录统一改名。正文 mini 的 rank0 直接 rename 前没有等待其它 rank 提交，不能作为真正多 rank 原子提交范例。必须将自有协议与固定 DCP 的 staging/save/metadata 边界分别解释。对应 7.4-B/C/E。

### R20 · P2 · 7.4 的 Adam 恢复因果推理过度确定

位置：`src/L7/7.4-data-ckpt-fault.md:131`；`src/L7/7.3-training-frameworks.md:402`。

Adam 首步 bias correction 同时消去 m/v 中的 `(1-beta)`，更新约为 `lr*g/(abs(g)+eps)`，不能仅看分母初始为零或 `1-beta1` 就断言步长数倍放大、必然剧烈 loss spike。缺 optimizer 会改变轨迹，但幅度和方向取决于旧状态与新梯度。恢复也不是任何状态遗漏都必然立即改变下一步 loss，例如更新前的 forward 与 optimizer state 无直接关系。应按实际反例解释哪个不变量何时先改变。

### R21 · P2 · 7.8 的去重数量与原始工件不一致

位置：`src/L7/7.8-training-data-engineering.md:288`。

工件为精确重复 3、近重复 1、过滤 4、污染 3，保留 19；正文写近重复 4。原工件正确计数为 `30-3-1-4-3=19`。且近重复植入样本有漏检，不能据保留数称所有保留项“纯净”。计划要求误删/漏删分析，不能以合成样本检测通过替代真实基准污染证明。对应 7.8-B/H。

### R22 · P2 · 7.8 packing 没有封闭跨文档的 loss 边界

位置：`labs/L7/training_data_contract.py:362`；`src/L7/7.8-training-data-engineering.md:247`。

collator 只是拼接 labels；在常见 CausalLM 整段 shift 下，第二文档首 label 会成为第一文档末 logits 的 target。提供 `[1,2]`、`[3,4]` 反例后边界 target 仍是 3。block-diagonal attention 不会自动屏蔽这个 target；需在对应 label 边界设 ignore_index，或明确实现逐文档 shift。达到容量后直接 break 也没有交付未消费尾部，需记录残留/拒绝。对应 7.8-D/G。

### R23 · P2 · 7.5 将 DPO 长度风险与 APO 的目标混为一谈

位置：`labs/L7/preference_loss_reference.py:106`；`src/L7/7.5-rl-infra.md:407`。

实验只比较两项 policy 的序列 logprob 和写死 consequence，没有把 reference logprob 和更新纳入长度实验。标准 DPO 使用序列概率比；若 policy=reference，任意长度两项 log-ratio 都为零。不能仅由长序列概率较低推导模型必然短答坍缩。长度归一化会改变 objective，应按 DPO/IPO/APO 具体配方分析，不能作为统一必需修复。

同章前沿把 APO 与 SimPO 一起解释为“目标 margin 加长度归一化”，也与固定 TRL 的 APO 分支不符：`apo_zero` 使用 chosen/rejected log-ratio 的两个 sigmoid，`apo_down` 使用 chosen 项与相对差；这里没有所称的统一长度除法或目标 margin。不能因此宣称彻底根除了长度偏置。

依据：[TRL DPO 说明](https://huggingface.co/docs/trl/dpo_trainer)、[固定 DPOTrainer 的 APO 分支](https://github.com/huggingface/trl/blob/cd2c52876d99b00baa5660328310e940b8360c6c/trl/trainer/dpo_trainer.py#L1552)及已核对的梯度公式。对应 7.5-E/F。

### R24 · P2 · 7.0 mini 重复 backward 的内部梯度不清理

位置：`labs/L7/mini_autograd.py:21`；`src/L7/7.0-autograd-anatomy.md:496`。

构造 x=2、y=x²、z=y²，首轮梯度为 32，再次 backward 得到累计 96，而保留图并正常叶子累积应为 64。内部节点残余梯度被再次传播。应支持调用级 InputBuffer/中间梯度重置，或明确拒绝重复 backward；不能让其与 PyTorch retain_graph 语义混淆。对应 7.0-A/B。

### R25 · P2 · 7.7 高温 KD 近似遗漏 logit 均值

位置：`src/L7/7.7-distillation-training-systems.md:118`。

正确一阶项是 `[(zs_i-mean(zs))-(zt_i-mean(zt))]/(V*T)`。正文省略中心化却没声明零均值条件；两组 logits 相差常数时真实分布与 KL 梯度相同，原公式却非零。T² 补偿不是任意温度下梯度严格相等；工件中的 T=1 与 T=4 范数也不相同。对应 7.7-B。

### R26 · P2 · 可运行入口缺失以及把未测能力写入状态

位置：`src/L7/7.2-training-parallelism.md:207`；`src/L7/7.0-autograd-anatomy.md:866`；`src/L7/7.0b-one-training-step.md:398`。

此外，7.9 的“4 卡一致跳步”只有 Python `all()`，7.7 没有 JSD 及四种解析梯度对拍，7.10 没有冻结输入梯度、不同学习率和 bucket 验证，7.11 没有子进程测试。STATUS 不应将这些记作已验证。现有 linkcheck 不检查代码围栏/普通文本中的这些入口，因此零断链不能排除问题。

状态文件复核还发现 1.6 的 `ppl_jetson.json` 和 `edge_llm_platform_check.json` 不在所列本地位置，不能据对应登记宣称证据已完整；1.3 的花括号路径是 shell 模式而非实际链接。STATUS 已标记前者缺证并将后者展开为四个真实文件入口。

## 非 L7 的定点发现

### R27 · P2 · 4.0 将超过存储带宽当成“未读数据”的充分条件

位置：`src/L4/4.0-checkpoint-format.md:231`。

page cache 命中后，真实遍历文件内容也可超过底层磁盘带宽；只建 mmap 与真实读 RAM 中已缓存页是不同路径。现有读数支持区分建视图、首次触页和传输，不能推广为该充分条件。应结合 cold/warm cache、page faults、RSS 和实际 IO 字节判定。依据：[Linux page cache](https://www.kernel.org/doc./html/next/mm/page_cache.html)。对应 4.0-C。

### R28 · P2 · 2.0 对 functionalization 的保证过强

位置：`src/L2/2.0-tensor-and-framework.md:1177`。

默认 `functionalize(remove='mutations')` 不会去除所有 view/alias；去除 view 需另一模式，输入写回、全局状态和 autograd 等限制仍存在。不能写成“图里没有别名，反向一定能算”，也不能将版本计数检测称为零开销。依据：[官方 functionalize 语义与限制](https://docs.pytorch.org/docs/main/generated/torch.func.functionalize.html)。对应 2.0-C。

### R29 · P2 · 1.2 将整卡功耗差唯一归因于显存

位置：`src/L1/1.2-tensor-core-lineage.md:602`。

两个不同程序的 SM 时钟与 board power 同时变化，不能分离 GDDR 搬运、活跃 SM/warp、指令发射率和执行单元利用率的贡献。应保留负载/功耗/时钟观测，将“减少访存必然换来更高频率”标为需受控验证的解释。没有独立显存功耗或等算力访问消融，不应以此解释 GEMM 差异。

### R30 · P2 · 3.3 的派生带宽不能独立证明差值来源

位置：`src/L3/3.3-decode-attention.md:159`。

表中 GB/s 本来由估计 KV 字节除以实测时间得到；再用它解释字节比与速度比的差值，只是代数重述。趋势与访存受限模型相容，但“kernel、块寻址效率导致轻微波动”需要实际执行路径和流量证据。应区分容量模型与归因实测，未取证项继续 UNVERIFIED。

## 续审新增：量化、内存与页面交付

### R31 · P1 · 4.3 的重建目标漏掉跨组项，两列补偿式也有符号和分母错误

位置：`src/L4/4.3-quantization.md:65`；`src/L4/4.3-quantization.md:95`。

设 ΔW=Ŵ−W，完整目标是 `tr(ΔW H ΔWᵀ)`，其中 `H=XᵀX`。按输入列分组后需要对 g、h 两个索引求和；正文只保留 g=h，把块对角近似写成了恒等式。最小反例 X=[1,1]、ΔW=[1,1]，真实误差平方为 4，正文分组式为 2。量化的 scale 分组不会自动消除激活之间的相关项。

两列、先固定第 0 列时，令 e=w₀−q₀，另一列的最优补偿是 `−e·Hinv[0,1]/Hinv[0,0] = +e·H[0,1]/H[1,1]`；正文写成了 `−e·H[0,1]/H[0,0]`。非等对角 H=[[4,1],[1,2]] 的反例得到 +0.00714286，而正文式给 −0.00357143。原对拍只是把同一个 inverse 表达式计算两次，没有核对正文所写的式子。对应 4.3-A；证据键 `cross_group_terms`、`two_column_compensation`。

### R32 · P1 · 4.3 的 GPTQ mini 没有更新剩余 Hessian

位置：`labs/L4/quantize_reference.py:89`；`labs/L4/quantize_reference.py:108`；`src/L4/4.3-quantization.md:97`。

`gptq_quant` 计算一次完整逆矩阵后始终使用其中的后续行；固定一列后，剩余自由变量的逆 Hessian 需要更新。官方 GPTQ 使用逆 Hessian 的上三角 Cholesky 因子实现这一过程，并向后续 block 传播补偿；当前 mini 既没有消元更新，也没有取该因子，还把 scale group 当成互不相干的 Hessian block。

保持相同的对称量化器、scale、列顺序和 damping=0，W=[1,0.31,0.346]、H=[[4,1,1],[1,2,1],[1,1,2]] 时，mini 输出 [1,2/7,2/7]；每步重算剩余 Hessian 及上三角实现均输出 [1,2/7,3/7]，最大差 1/7。两种独立参照在 1e-12 容差内一致。这个 CPU 小例检查补偿算法，不冒充官方完整量化器的 GPU 对拍。

依据：[固定 GPTQ 源码](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L99)、反例键 `gptq_compensation`。因此原表中的方法标签及“顺序无关”等解释必须重新核对，不能以第一列的自洽检查验收任务 A。

### R33 · P1 · 4.3 的 clipping 对照换了量化轴，评测分布与排序也标错

位置：`labs/L4/quantize_reference.py:65`；`labs/L4/quantize_reference.py:225`；`src/L4/4.3-quantization.md:109`。

RTN 沿 W 的输入列分组；`clip_quant` 先做 `W.t()`，随后沿输出维分组。即使只允许 ratio=1、不发生裁剪，两条路径也不是同一个量化器；2×4 小矩阵的最大元素差为 0.0857143。把两者的输出误差差异归因于 clipping 不满足控制变量要求。需要保持轴、group、scale/zero 约定一致，再比较裁剪比例。

校准 X 含每 16 通道 ×10 的离群值；`X_same` 是未放大通道的普通高斯，另一组 `outlier` 是每 8 通道 ×12，均不能标成“与校准一致”。标为相关分布的 `corr` 又直接复用了校准 X，不能作为独立评测集。表中普通高斯一行 clipping=0.09480 还低于 RTN=0.09646，正文却称 RTN 最优。数据分离原则本身正确，当前实验不能按这些标签支撑其具体解释。对应 4.3-A/E；证据键 `unclipped_control`。

### R34 · P2 · 4.3 把一次格式与旋转对照写成单调规律

位置：`src/L4/4.3-quantization.md:129`；`src/L4/4.3-quantization.md:487`。

“group 变小一定更准”不成立。调用本章 RTN，W=[1,0,6/7,4/7]，4-bit group=4 可精确重建，group=2 的 L2 权重误差反而为 0.0408163。absmax 改变 scale 后，舍入格点不是原格点的嵌套细化；同样也不能保证层输出或任务质量单调改善。证据键 `smaller_group_not_monotonic`。

两个合成输入上的 Hadamard 结果只能说明那两种配置的误差变化，不能解释 QuaRot/MR-GPTQ 在整类量化粒度上的普遍收益。应分别核对真实方法、旋转位置、权重/激活量化与完整成本。E2M1 被称为“3 个幅度档”也与正文列出的 7 个正幅度值不符；scale 的向上取整不等于所有还原值都有固定方向或倍数的偏差。对应 4.3-B/C。

### R35 · P1 · 2.0c 查询不存在的 snapshot 状态，导致碎片恒为零

位置：`labs/L2/alloc_trace.py:63`；`labs/L2/alloc_graph_combo.py:56`；`src/L2/2.0c-runtime-memory.md:330`。

固定 PyTorch v2.13.0 的 snapshot block state 只有 `active_allocated`、`active_awaiting_free`、`inactive`。`snapshot_summary()` 和 `snap()` 查询的 `active_split`/`inactive_split` 不是 state，`.get(...,0)` 因而把缺字段静默写成了零。`inactive_split_bytes` 属于 `memory_stats` 的统计口径，不能与 snapshot 的状态名混用；不同 backend/expandable 配置的计数适用性也需单列。

用官方 schema 构造一个含 4 MiB 活跃、8 MiB 段内空闲、4 MiB 待完成块的合成快照，两个提取器仍给出 inactive_split=0；`free_blocks` 还把待完成块算进可用块，返回 2 而不是 1。该反例验证解析逻辑，不是 GPU 实测。原轨迹没有保存足以恢复该统计的逐步完整 snapshot，因此其真实碎片量仍未知，“全部是整块可复用、没有碎片”的解释不能成立。

依据：[固定 snapshot schema](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/cuda/memory.py#L1057)、反例键 `snapshot_schema`。对应 2.0c-A/C。

### R36 · P1 · 2.0c 将同流复用写成必须等待 GPU，并把 record_stream 当作独立同步修法

位置：`src/L2/2.0c-runtime-memory.md:113`；`src/L2/2.0c-runtime-memory.md:188`；`src/L2/2.0c-runtime-memory.md:739`。

同一创建流上的块可在 CPU 侧先归还缓存并重新分配，后续 GPU 操作依靠流内顺序安全执行；不需要每次释放都插入 event 或等待 GPU 完成。固定 allocator 在没有额外 stream uses 时直接 `free_block`。跨流才需要另外管理执行依赖与复用顺序。正文开头“del 后总是 pending、GPU 完成才 free”的模型会误导 allocator 的关键设计。

“解决方法 2”的代码仅调用 `y.record_stream(s2)`，没有等待 s1 产生 y，仍可能读到旧值；它只能告知 allocator 跨流使用，不能代替 `s2.wait_stream(s1)`。完整生命周期受调用方管理时，也可通过让创建流等待使用流来避免 record_stream，并非该 API 永远必需。另 `memory_allocated()` 在尚未初始化时经 `memory_stats_as_nested_dict()` 返回空统计，并不触发正文所称的 CUDA 初始化。

依据：[record_stream 官方说明](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.record_stream.html)、[固定 free 路径](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L2444)。对应 2.0c-A/B。

### R37 · P2 · 2.0c 的 graph-first 没有保持图存活，不能据残留显存判断图池生命周期

位置：`labs/L2/alloc_graph_combo.py:104`；`labs/L2/alloc_graph_combo.py:134`；`src/L2/2.0c-runtime-memory.md:380`。

`graph_phase()` 在返回之前已经 `del g, static_in, static_out`；graph-first 的请求轨迹随后才运行。因此两种顺序实际比较的是曾经初始化过图与相关运行库的进程，不能称为“活跃图池与变长轨迹同时驻留”的对照。剩余 allocated/reserved 没有按 pool、活跃对象和 workspace 分解，不能据其非零就说图池与 Python 图对象无关、删图后必定不归还。

“图池不参与 expandable”也不能由多出一个 segment 推出。固定开源 v2.13.0 `alloc_block` 的 expandable 分支明确处理 CUDA graph 的 PrivatePool；普通图私有池与自定义 allocator 的 user pool 不是同一条件。需要保存每个 segment 的 pool 身份、expandable 标记及图释放前后快照，并在轨迹期间保持图对象存活。现有捕获成功和占用读数可以保留，归因保持未验证。

依据：[固定 allocator expandable 分支](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L3896)、本章组合脚本与 `combo.json`；本轮没有重跑 GPU。对应 2.0c-C。

### R38 · P2 · L7 的 SVG 出现在代码围栏中，结构扫描误判为已有图

位置：`src/L7/7.3-training-frameworks.md:17`；`src/L7/7.0b-one-training-step.md:22`；`site/L7/7.3-training-frameworks.html:91`。

现有站点的 article DOM 中，7.0b 和 7.3–7.11 共 10 页各有一个含 `<svg>` 的代码块，实际 SVG 元素为 0；7.1/7.2 则没有 SVG。原因是正文将图放在 `xml`/`svg` 围栏中，构建器按代码高亮显示。字符串扫描看到 `<svg` 不能说明“心智模型配内联 SVG”已交付。唯一有实际 SVG 的 L7 章节是 7.0。

证据：[现有 HTML 结构检查](../../results/local/review/20260914-chapters-continuation/site-figures.json)。这次仅核对 HTML 元素和源围栏，没有做完整浏览器视觉验收。修复应在正文中使用真正内联 SVG，再按章节规范核对离线呈现、深浅色与小屏。对应章节规范固定结构第 2 项。
