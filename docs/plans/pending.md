# 未完成部分的详细执行计划

本文件收录 48 个模块的编写、完善与验收任务；当前正文是否存在和实际完成情况见 STATUS。每章指定问题、模型或实现、源码入口、实验矩阵、交付文件与验收条件。已有内容的修订任务见 [已完成部分](completed.md)。任务中列出的文件先按 [STATUS](../../STATUS.md) 判断已有实现与待实现部分，不能仅凭计划文字当作实验已完成。

执行前按全局执行计划（`~/.claude/plans/eventual-painting-peacock.md`）完成输入、源码/模型 revision 和环境固定。主模型、比较问题、实验轴与判据按下列定义执行；兼容性检查确定适用版本和合法配置，不将核心研究设计留到临场决定。核心任务受限时保留未完成状态，同时推进无该依赖的任务。各章同样遵守 [章节规范](../chapter-guidelines.md)、[实验规范](../experiment-guidelines.md) 和 [环境记录](../../ENVIRONMENTS.md)。

<a id="beginner-depth"></a>
## 从入门到前沿的共同任务

**目标与依赖**：面向具备基础 Python、尚未学习 AI infra 的读者，系统建立模型、硬件、算子、训练、推理、分布式与生产系统的知识，并能阅读前沿论文和实现。所需矩阵乘、概率、导数、复杂度、二进制与操作系统概念在首次使用处给出小例和前置入口；不默认读者熟悉缩写、GPU 编程或分布式系统。以下任务适用于两份计划的全部章节，与各章任务表一起验收。

| 任务 | 执行与交付 | 验收 |
|---|---|---|
| 概念与术语 | 各章梳理必需前置，核心术语给中文解释、英文名称、所属层、具体对象/单位、小例和易混概念；基础定义接 0.0c 索引，深入解释放在责任章节 | 正文先解释再使用；batch、step、scheduler、cache、checkpoint 等多义词明确当前语义；读者不需靠外部搜索补齐关键定义 |
| 问题与技术演进 | 从朴素实现及一个实际失败/瓶颈推导方案；每次变化说明旧限制、关键设计、保留条件、新增代价及后续问题；历史时间与教学顺序分开 | 原始论文/官方实现支撑关键节点；并行发展的计算、存储、调度路线分别说明交汇关系，不按发布时间拼成单一替代链 |
| 原理与实现细节 | 同一小输入贯穿手算、数据结构/张量/时间线、最小实现、真实源码和实验；基础推导在主线，长代码与原始材料可展开。修订时同步检查开篇、图表、实验解释、陷阱与自测，实验失败按不变量组织，过程记录留在 STATUS | 至少一道预测结果题、一次实现或修改练习、一个失败诊断；答案有推导和证据位置，不能用术语复述代替理解。同章各处对已测模型、未测路径和因果强度的表述一致；源码宏、命令和结果入口指向实际文件 |
| 当代实现与前沿 | 各章选一个主要实现深入分析，比较同层替代方案，再研读直接针对剩余瓶颈的代表性新机制；M3 提供检索与判断方法 | 每项前沿交付问题、机制、源码位置、相对基线的变化、适用硬件/负载、质量/成本和未解决问题；可运行路径做有区分力的小实验，硬件受限路径完成推导与源码分析 |
| 综合迁移 | 复用模型、数据和最小系统，改变一个条件，要求读者预测并验证影响；各层交付接到下列贯穿产物 | 能独立定位瓶颈/错误、修改核心机制、解释对照结果，并说明结论在哪些条件下失效；正文存在和术语覆盖不代替这些能力 |

### 各层的概念、演进与贯穿产物

下表给出具体责任范围；术语在对应章展开，0.0c 负责跨章索引与阅读导航。前沿材料沿各章已有官方来源和 M3 检索补齐，执行时核对现行实现。

| 层与责任章节 | 入门概念与演进问题 | 细节、贯穿产物与前沿接口 |
|---|---|---|
| L0：0.0/0.0b/0.0c、0.1–0.5 | tensor、参数、activation、token/logit、loss/梯度、训练/推理；HTTP 请求到模型计算，FLOP/字节/延迟/吞吐及单位 | 一个小模型从文本→张量→生成/更新→权重保存；手算 shape、概率和资源账，能判断问题落在哪一层 |
| L1：1.1–1.7 | 进程/线程、虚拟内存、cache、SIMT/warp、SM、HBM、PCIe、NUMA、DMA、NIC/RDMA；从 CPU 到 GPU 与端侧异构 | 用内存访问、传输和网络路径解释硬件瓶颈；对照 GPU 代际与端侧约束，区分消费级和数据中心能力 |
| L2：2.0–2.8 | storage/stride/view、dispatch、kernel、stream/event、异步、tile、融合、编译图；从 eager 小算子到融合/图执行 | 同一算子经过 Python/ATen/设备 kernel；朴素→分块→流水的实现与测量，接 CuTe-DSL、TileLang、编译器动态形状和硬件专用路径 |
| L3：3.1–3.4 | Q/K/V、attention mask、head、MHA/MQA/GQA、KV、softmax；显式注意力→分块、decode 并行与状态压缩 | 手算、分页地址、线程布局及误差；FA1–FA4、FlashInfer、MLA、稀疏/线性/混合注意力分别按计算与状态问题比较 |
| L4：4.0–4.11 | 权重文件/加载、精度/量化、MoE router/expert、视觉 encoder/projector、ASR/codec/TTS/Omni | 同一样本逐层张量和资源账；稠密→专家、文本→图文/音频/音画的独立机制，接低精度、视觉适配与多模态服务 |
| L5：5.1–5.13 | prefill/decode、静态/动态/continuous batching、KV block/page、prefix cache、TTFT/TPOT、抢占/背压 | 从单请求循环到 nanoserve：按轮调度、分页、共享、分块、图执行、投机和失败回收；解释这些机制的组合条件 |
| L6：6.0–6.5 | rank/group、collective、布局、DP/TP/PP/EP/CP、带宽/时延、同步/异步；从复制到分片与分离 | 手算数据归属和通信顺序；两 rank 实现接多卡成本、MoE dispatch、PD 分离、KV 传输和长上下文，跨机条件单列 |
| L7：7.0–7.11 | sample/batch/microbatch/epoch/step、optimizer、AMP、checkpoint、预训练/SFT/LoRA/DPO/KD/RL | 完成下述教学训练项目，解释数据、目标、状态、学习曲线和部署；向 FSDP/ZeRO、FP8/FP4、异步 RL、分离式 teacher/learner 延伸 |
| L8：8.1–8.8 | 服务/副本、队列/路由、容器/镜像、编排、配额、SLO、goodput、p50/p95/p99、冷启动 | 同一服务从单进程到多副本、压测、故障和成本；接 KV 层级存储、缓存路由、混部与隔离，保留真实负载和失败请求 |
| L9：9.1–9.8 | task/request、DAG、tool call/MCP、session、持久状态、retry/idempotency、sandbox、RAG | 可恢复的最小 Agent 服务，跨模型与 CPU 工具追踪完整任务；接会话 KV、预算、分支与任务 SLO，不能以框架 API 替代系统解释 |
| L10：10.1–10.6 | 数据/噪声/时间、score/velocity、训练与采样、solver/NFE、latent/条件、动作/闭环 | 小型图像生成模型从零训练到采样；再接 DiT、少步蒸馏、视频/世界模型及 VLA，区分离线误差、生成质量和闭环成功 |
| M：M0–M3 | 假设/对照、相关/因果、数值误差、实验单位、源码版本、架构取舍 | 复用训练、服务、kernel 三类工件；完成一个陌生机制的推导、源码定位、最小复现和边界分析，培养持续跟进前沿的能力 |

<a id="c-0-0c"></a>
## 0.0c AI infra 全景、核心术语与技术演进

**依赖**：基础 Python；领域地图可作为课程入口。模型计算细节接 0.0/0.0b，完整案例随着 5.7/7.1/8.3 回填，不要求先学完这些章节。

**问题**：AI infra 解决哪些问题；术语之间有什么关系；怎样从基础机制逐步读懂现代系统与前沿研究。

**对象与源码**：复用本课程模型/训练/服务材料；[PyTorch 基础](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)、[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[FlashAttention](https://arxiv.org/abs/2205.14135)及各责任章的现行实现。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用训练一个小模型、部署一个生成服务、处理一次带工具的任务三个场景建立领域地图；标出模型算法、框架、kernel、硬件、数据/存储、通信、调度/编排、观测的职责 | 每个场景给输入、计算、状态、资源、输出和错误定位入口；读者能说明 infra 与模型研究、应用开发的交界 |
| B | 按上一表组织术语索引：中文/英文、白话定义、具体例子、单位/形状、易混项和深入章节；给 CPU/进程/内存、矩阵/概率/导数的最小前置练习 | batch/step/scheduler/cache/checkpoint 分语境解释；索引可跳到正文解释，正文可返回索引；所有必需缩写在使用前展开 |
| C | 分别重建 attention 计算、KV 状态管理、请求调度、训练并行/数值四条演进路线；用瓶颈、设计、代价和组合条件连接代表方案 | 交付有原始来源的关系图和时间节点；能解释 continuous batching 与分页、FlashAttention 与模型结构之间的关系 |
| D | 按“领域地图→模型基础→首次训练/服务→底层机制→分布式生产→前沿”给阅读路线；对每个里程碑列前置、实践产物和自测 | 初学者可沿主线渐进学习；硬件扩展和深入推导有明确入口；不需要先看懂整本教程才能完成第一个实践 |
| E | 用一个陌生故障和一篇新论文练习分类：先定位层次、前置和原瓶颈，再找到应读源码及验证路径 | 能给出调查顺序、应观察的字段和可能推翻解释的对照；引用现有章工件，不以术语背诵或论文名单验收 |

**交付**：`src/L0/0.0c-infra-map-and-vocabulary.md`、内嵌领域/演进关系图、带章节链接的术语索引、阅读路线与自测答案。正文与导航在后续编写阶段实现。

**反例与边界**：教学路线不是所有技术的历史先后关系；章节出现某个前沿名词不等于已经解释其机制。

<a id="training-workflow"></a>
## 训练、微调、蒸馏与强化学习后训练的共同任务

**范围**：覆盖从原始数据到可部署模型的全流程知识，以及支撑这些阶段的数据、计算、存储、通信和作业系统。文本语言模型、视觉编码器、ASR、音频 codec、TTS、图像扩散/flow、视频/世界模型、Omni 和动作模型分别建立自己的基础训练流程。草稿训练、量化感知训练、蒸馏和 RL 是作用于这些模型的训练方法，不能全部归约成语言模型 SFT。

**学习与运行范围**：机制小验证、低资源完整教学训练、真实规模公开流程三类任务分别验收。小张量和一至两次更新用于定位数值/状态问题；下述贯穿项目必须完成数据到训练、独立评测、恢复与部署，保留实际学习曲线；Puro-2B、SmolLM3 和大型多模态模型用于解释规模变化及生产设计。小模型的质量和性能只对本次配置成立，作者规模结果注明来源；不要求重训全部大模型或开展四框架规模训练竞赛。

### 低资源完整训练的贯穿项目

模型是教学载体。优先使用下列开放实现，执行前按源码可读性、数据可用性、质量目标、显存和运行时间确定版本；替换时保留任务、目标、产物与评测，不把某个模型名作为全课程硬约束。

| 项目及责任任务 | 初始化、数据与运行流程 | 交付与验收 |
|---|---|---|
| 文本：7.8-I → 7.1-F → 7.5-I，恢复接 7.4-H、服务接 5.7-E | 以 [MiniMind](https://github.com/jingyaogong/minimind) 的约 26M–64M 配置为候选，版本与结构对应；从随机模型权重开始，复用明确来源的 tokenizer，小语料清洗/去重/拆分→tokenization/packing→预训练→独立 SFT→导出。起始设计为上下文 256/512、预训练 20M–100M 有效 token、SFT 2k–10k 条；先测算再选一套主配置 | 数据处理说明、各阶段配置/权重、训练/验证曲线、固定留出任务结果、恢复轨迹和加载后的推理样例；预训练相对随机初始化的留出 loss、SFT 相对 base 的任务效果和遗忘分别评价。给读者从准备到使用的完整命令，不将减小数据规模称作复现作者最终能力 |
| 后训练分支：7.5-J、7.6-J、7.7-J | 从同一个已完成 SFT 的教学模型分别开展 LoRA 适配、DPO、可验证奖励 RL 和蒸馏；每种选一个明确的小任务及独立验证集，连续训练到预先规定的预算/停止条件；RL 先测基础任务成功率，蒸馏计入教师生成/评分成本 | 每个分支独立交付学习曲线、参数更新对象、任务质量/退化、训练与部署成本、导出产物；对照共同 SFT 起点或等预算直接训练。基础模型能力不足时选择更简单任务或预算内的公开小基座并注明初始化变化；不把四种方法强行串成必经流水线 |
| 视觉适配：4.5/4.6 → 7.10-J，评测接 4.8 | 以 [MiniMind-V](https://github.com/jingyaogong/minimind-v) 为候选，复用匹配的教学 LLM 与冻结视觉编码器；图文数据独立拆分→随机 projector 对齐→选择模块联合 SFT→评测→导出。起始使用 2k–10k 对图文，先完成单图/固定分辨率 | 模块×阶段更新表、对齐/SFT 曲线、留出图像问答/计数结果、文字能力变化和部署样例；打乱/移除图像对照检验视觉信息是否被使用。明确视觉塔和 LLM 的已有训练来源，不声称所有参数从零训练 |
| 图像生成：10.1-G，恢复/部署接 7.4/10.3 | 以 [Diffusers 从零训练](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)及官方 unconditional 脚本为起点，使用 MNIST/Fashion-MNIST 等小图像集、28/32 分辨率与小 UNet；随机初始化→数据/噪声/时间采样→DDPM 训练→独立评估→保存恢复→采样。flow 先完成解析参照，再作同数据小训练扩展 | 学习曲线、固定噪声的阶段样本、留出目标误差、样本多样性/记忆检查、训练时间与采样 NFE/质量；像素域主例独立于预训练 LLM/VAE，证明掌握不同基础目标；单张好图或训练 loss 不能替代质量分析 |

**资源与停止条件**：上述数值是课程设计范围，不是已测成本或质量保证。默认目标为单张不超过 24 GiB 显存；先用 100–300 step 或最多 15 分钟测出有效处理速度、峰值、评测/保存成本，再确定数据量、训练步数与停止条件。规划上限为文本预训练＋SFT 12 GPU 小时、视觉适配 8 GPU 小时、图像生成 8 GPU 小时、每个后训练分支 4 GPU 小时；教师/奖励服务时间计入对应分支，整体目标不超过 44 GPU 小时，首次短测计入预算。各项目按阶段执行，不同时驻留所有模型；数据缓存目标每项目不超过 20 GiB，checkpoint 空间另按模型/optimizer 实际状态预估，保留 latest、best 与必要阶段产物。

超出预算时先缩小模型、分辨率或数据规模，并保留全部阶段和评测；不静默退回单步演示或自动扩大预算。运行前定义随机/未适配基线、验证频率、主质量指标及最低可解释改善标准，预算耗尽但指标未满足时如实记录未通过的质量项和原因；完整流程执行与质量达标分别记录。学习曲线不能保证任意小模型获得通用能力；某次负结果可用于诊断，不能代替所有贯穿项目的质量验收。

**规模桥接**：7.3/7.9/7.11 研读 [Puro-2B 报告](https://arxiv.org/html/2608.27370)、[Puro-Megatron](https://github.com/thu-pacman/Puro-Megatron)和 [SmolLM3](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3)，把数据阶段、优化器、FP8、并行和成本映射回教学项目。Puro 是低成本规模预训练案例，不能按单卡几小时项目安排；MiniMind README 的两小时口径按具体版本/训练阶段解释。大规模预训练、语音/视频/世界模型的全量训练以公开流程为主，相关源码、数学与非文本目标仍为必学。

**执行入口交付**：在对应责任任务中实现 `labs/L7/run_teaching_lm.sh`、`run_teaching_posttrain.sh`、`run_teaching_vlm.sh` 与 `labs/L10/run_teaching_diffusion.sh`，配套项目相对路径的配置与数据准备/评测命令。入口复用选定上游训练实现，关键机制用现有 mini 对拍；配置覆盖模型版本、数据/拆分、阶段衔接、预算、保存/恢复、评测与导出。上述入口均为待实现产物，不能在实现与运行验证前写成已有命令。

完整训练各阶段复用下表的机制与验收；训练工程、成本、独立质量和失败分析共同构成交付。

| 阶段 | 必须覆盖的知识与系统问题 | 交付与验收 |
|---|---|---|
| 任务、模型与预算 | 从任务定义输入、预测对象与输出；区分从零训练、继续预训练、领域适配、SFT、偏好优化、蒸馏和 RL；确定基座、结构初始化、tokenizer/codec/VAE、上下文/时长、训练预算及数据规模关系 | 每类模型都有阶段依赖图；能解释为什么选这些目标和模块、增加数据/参数/分辨率会改变什么；规模规律和作者预算不冒充本机测量 |
| 数据获取与构建 | 数据许可与出处、标注/合成/教师生成、语言与领域混合、文本清洗去重、音视频切段与对齐、质量筛选、训练/验证/测试隔离、评测污染、偏好配对、轨迹与奖励字段 | 追踪一条原始样本到训练样本；数据版本、过滤理由、混合权重、样本单位、拆分规则和丢弃量明确；媒体按说话人/视频/episode 等独立单元隔离 |
| 表示、batch 与监督 | tokenizer/词表、图像增强、音频特征、VAE/codec、动作归一化；padding、packing、bucket、position、attention mask、loss mask、teacher forcing、噪声与时间采样、条件 dropout；有效 batch 的统计单位 | 输出实际字段、shape、mask 和分母；能还原一次 batch 的监督范围；跨文档 attention、跨模态 loss、每 token/帧/样本平均不混用 |
| 目标与优化 | CE/CTC/对比学习、重建/KL/感知/对抗损失、扩散噪声/velocity/flow 目标；辅助路由损失、SFT/DPO、各类 KD、PPO/GRPO；初始化、冻结/解冻、LoRA、AdamW、学习率、warmup/decay、clip、EMA | 每项给方程、梯度流、可训练参数和官方 loss 实现；说明损失权重、归一化、优化器参数组与阶段切换；用小参照检验关键梯度 |
| 训练计算与数值 | autograd、保存值、重算、梯度累积、optimizer step、融合/编译、FP32/BF16/FP16、scaler；FP8/FP4 格式、缩放/amax、累加、master weight、优化器/通信 dtype、溢出/下溢与跳步 | 按参数、前向、反向、归约、更新列 dtype 和状态字节；能定位首个非有限值；区分混合精度训练、PTQ、QAT/QAD、QLoRA 与低精度推理 |
| 并行、数据供应与调度 | DDP/FSDP2/ZeRO、TP/PP/SP/CP/EP；参数展开与通信重叠、不同 rank 有效样本数、动态形状、异构 batch、数据缓存/预取、straggler、teacher/learner/rollout/reward 的资源分配 | 张量/状态归属、collective 顺序、关键路径与峰值内存可检查；源码比较覆盖 FSDP2、DeepSpeed、Megatron、TorchTitan，运行验证共用少量参照 |
| 保存、恢复与阶段迁移 | 参数/optimizer/scheduler/scaler/RNG/数据游标/EMA/量化状态；分布式 checkpoint、异步 staging、原子提交、重启、world-size 改变、adapter/teacher 引用、阶段转换 | 区分仅权重 warm start 与继续同一次训练；列出何种状态缺失会破坏哪项保证；恢复和导出分别验收 |
| 评估与诊断 | 训练/验证 loss、学习曲线、梯度/更新范数、吞吐与有效更新、数据饥饿、显存时间线、MFU/HFU；任务质量、遗忘/过拟合、鲁棒性、奖励投机、蒸馏容量与误差、消融和早停 | 每条性能/质量结论连接原始材料与条件；评测与校准/训练隔离；对多模型/多阶段使用适合各自任务的指标，缺少公开日志的项目明确留空 |
| 导出与部署反馈 | 完整模型、adapter、merged 权重、量化 packed weight/scale、draft 结构/词表映射、EMA、processor、codec/VAE、normalizer、scheduler；训练与推理计算/精度/模板差异 | 从训练保存入口追到目标 loader；静态核对字段并复用小型 round-trip；检查缺配置、错基座、错模态处理与导出后误差，部署质量反馈可追到训练选择 |

### 逐模型的公开流程任务

每行建立一份“原始样本→batch→目标→梯度/更新→checkpoint→评估→导出”材料；不把框架 README 的“支持模型”作为完整训练流程。每份材料包括：阶段输入输出表、至少一份真实配置及字段解释、关键源码的版本链接、file/symbol、公开数据与权重入口、日志/曲线的出处、可训练模块、资源开销、相关失败案例，以及未公开环节。下列入口是必读起点；正文中保留必要原文，知识点必须落实到后面的章节任务。

| 模型或训练路线 | 选定的开放材料与阅读顺序 | 所属任务及公开范围 |
|---|---|---|
| 文本基座与后训练 | [SmolLM3 pretraining](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3) → [alignment-handbook 的 mid/SFT/APO](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3) → [阶段权重](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints) → [评测](https://github.com/huggingface/smollm/tree/main/text/evaluation/smollm3)；RL 用 [Tülu 3](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) 的数据、SFT/DPO/RLVR 与模型阶段 | 7.3/7.5/7.6/7.8。SmolLM3 发布的阶段模型为 Transformers 权重，不能默认含完整 optimizer/RNG；原配方中的内部 S3 数据位置需要映射到公开数据来源 |
| 视觉编码器与 VLM | [OpenCLIP](https://github.com/mlfoundations/open_clip) 的数据→ClipLoss/SigLipLoss→训练→零样本/检索评测；[Qwen3-VL 微调](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune) 的数据、冻结策略、位置/mask、保存 | 4.5–4.8、7.10。OpenCLIP 提供基础表示训练流程；Qwen3-VL 微调代码不等于公开了原模型全部预训练数据和阶段 |
| ASR | [SpeechBrain LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer) 的准备、tokenizer、CTC/attention、搜索和 WER；[Qwen3-ASR SFT](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) 的 JSONL→collator→Trainer→恢复 | 4.9、7.8/7.10。前者提供独立 ASR 基础训练，后者提供现代模型微调；不同目标/词表/采样率单列 |
| Codec、AR TTS 与 flow TTS | [DAC](https://github.com/descriptinc/descript-audio-codec) 的 RVQ/重建/对抗训练；[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) 的 audio_codes→Talker/SubTalker→speaker 导出；[F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train) 的特征/时长→flow→EMA→vocoder | 4.10、7.10。Qwen 官方当前例子是 12Hz Base 单说话人微调；codec 与 vocoder 自身的训练不能由 TTS SFT 代替 |
| 草稿模型 | [SpecForge](https://github.com/sgl-project/SpecForge/blob/3d64e7a61f5fcc7f7d78ba6164c881f831943947/docs/sections/basic_usage/training.md) 的样本/teacher 特征→EAGLE3/DFlash 目标→分离式训练→export→SGLang；部署主例仍用 5.5 固定 target/draft | 5.5、7.7。分别解释草稿从零初始化/继续训练、在线/离线 teacher、词表映射和不同 loss；不声称通用配方就是某个发布权重的原始训练记录 |
| 量化与低精度模型 | [torchao QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html) 的 prepare→fake quant→convert；[ModelOpt QAT/QAD](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat) 的 quantize→train→export | 4.3、7.5/7.7/7.9。PTQ 校准、QAT、QAD、QLoRA、FP8/FP4 训练分别建立状态表；这是一条跨章节路线，量化不单列特殊计划 |
| 图像生成基座与蒸馏 | [视觉 VAE](https://github.com/CompVis/latent-diffusion/tree/main/configs/autoencoder) → [flow_matching](https://github.com/facebookresearch/flow_matching/tree/main/examples/image) / Diffusers DDPM → [SDXL/FLUX LoRA](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) → [LCM](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation) / [DMD2](https://github.com/tianweiy/DMD2) | 10.1–10.3、7.7/7.10。VAE、条件编码器、denoiser 与蒸馏学生是不同训练对象；公开微调脚本不补造基础模型原始训练集 |
| 视频与世界模型 | [Wan2.2-TI2V-5B 的 full/LoRA](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training)；[Cosmos3-Edge SFT](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/docs/training.md) 的 Bridge JSONL→DCP→生成分支/Reasoner 配方→导出→Diffusers | 10.4、7.10。Wan 的公开训练实现与原作者完整预训练配方分开；Cosmos 的公开 SFT 及 DMD2 源码不证明全部 Edge 预训练/蒸馏数据和日志已公开 |
| Omni | [Qwen3-Omni 技术报告](https://arxiv.org/html/2509.17765) 的模块预训练和联合阶段，核对 [ms-swift 的实际 Qwen3-Omni 配方](https://github.com/modelscope/ms-swift/tree/0673cf75dca7d0b9b608b4a76632fb508ead5076/examples/models/qwen3_omni) 与模型/template 代码 | 4.11、7.10。训练的输入含音视频不自动意味着训练了语音生成；Thinker、Talker、Code2Wav 逐项核对 loss 与更新路径，缺失的原始全模态流程以报告说明 |
| VLA/动作基座 | [openpi DROID](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md) 的 episode/动作定义→过滤/normalizer→pi0-FAST 或 pi05 flow→checkpoint→serve_policy；Cosmos action 配方另作对应 | 10.6、7.10。openpi 的 pi05 当前开放实现以 flow head 为准；离线数据训练、RL/在线交互、机器人闭环是不同阶段 |
| 跨模态 RL 与语音蒸馏 | [Flow-GRPO](https://github.com/yifan123/flow_grpo) 的 SD3.5/FLUX 连续轨迹 RL；[DanceGRPO](https://github.com/XueZeyue/DanceGRPO) 的视频配方；[CosyVoice2 GRPO](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2) 的 codec rollout→token2wav/ASR reward→veRL→模型转换；[ZipVoice](https://github.com/k2-fsa/ZipVoice/blob/master/egs/zipvoice/run_emilia.sh) 的基础训练→两阶段蒸馏→平均权重→ONNX | 7.6/7.7 与 4.10/10.2/10.4。分别核对 action/transition/logprob/奖励；这些资料不能转述成 Qwen3-TTS、Cosmos3-Edge 或 Omni 官方同款训练流程 |

### 知识与基础设施的执行顺序

1. 0.0c 建立领域地图；复用 7.0/7.0b 的图与更新参照，7.8-A/B/D/I 与 7.9-A/B 提供数据、优化器及精度基础。
2. 先完成 7.1-F 的单卡预训练和 7.5-I 的 SFT，使用 7.0b-C 的基础保存/恢复，不等待多卡框架；7.4-H 再复用阶段工件验证实际中断恢复，7.11-I 汇总成本。7.2/7.3 在通信前置具备后深化并行与生产架构。
3. 从 SFT 起点分别推进 7.5-J、7.7-J、7.6-J 的后训练分支；7.6 的单卡算法实践先于多卡运行时综合实验，复用数据、精度和恢复语义。
4. 4.5/4.6 的基础任务后运行 7.10-J 视觉适配；10.1-G 独立完成图像生成训练。其他语音/视频/世界模型/Omni/VLA 分支完成自身目标与公开流程，再接 7.10 的复杂编排；已有权重的推理实验可独立推进。
5. 分别记录机制小验证、完整教学项目的运行/质量/恢复/部署与公开规模分析。责任章的贯穿任务未完成时保持未验收；不要求每个机制章独立重训，不用公开曲线抵消本课程要求的实际训练。

<a id="c-4-5"></a>
## 4.5 图像与视频预处理

**依赖**：0.4、4.1。 训练相关综合任务接 7.8/7.10，不依赖完整训练运行。

**问题**：像素如何变成 patch；resize/采帧如何改变 token 与信息；不同模型 processor 的约束如何验证。

**对象与源码**：主模型 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，结构对照 `google/gemma-3-4b-it`；[Qwen3-VL 官方代码](https://github.com/QwenLM/Qwen3-VL)、Transformers image/video processor、qwen-vl-utils。 训练侧研读 Qwen3-VL 的 qwen-vl-finetune/qwenvl/data/data_processor.py、OpenCLIP 的 src/open_clip_train/data.py；本章补齐[训练共同任务](#training-workflow)中的原始数据→训练输入。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 RGB/归一化/resize 规则，从 config 提取 patch、temporal patch 与 merge 参数；手写 normalize、patchify、grid 计算 | 与官方 processor 逐元素对拍；原图、处理张量、grid_thw、shape/stride 和 token 数可检查 |
| B | 12 张自有或许可明确图像覆盖 320×240、640×480、1280×720 与竖图；视频采帧=1/2/4 fps、帧数=4/8/16，另设非法边界 | 输出采帧时间戳与实际舍入/padding；不能将文件大小当视觉 token 成本 |
| C | 在相同 OCR/计数/时间定位小任务上改变视觉 token 预算 128/512/2048，对比 Qwen3-VL 与 Gemma processor 的 crop/merge 结构 | 保留任务答案、信息损失、CPU 时间和后续 GPU 输入规模；跨模型质量差异单列 |
| D | 比较训练随机 crop/resize/augmentation、视频分段/采帧/时间戳与推理固定 processor；追踪原始图文/视频标注→处理张量→grid/position→监督标签；解释分辨率和帧数的训练混合 | 复用 12 图/短视频样本，打印原数据、增强参数、训练/推理张量差异；数据划分按图源/视频独立单位，缓存必须包含增强和采帧身份 |
| E | 检查预计算视觉特征/latent 的版本、冻结模块、随机增强和读取成本；为多分辨率/帧数建立 bucket 与有效 token 预算 | 提交训练数据契约及一个缓存失效反例；只做预处理与小数据检查，大型视觉基座预训练按公开流程学习，教学 VLM 适配复用 7.10-J 的实际图文数据 |

**交付**：新增 `labs/L4/vision_preprocess.py`，输入 manifest、patch/grid 原始数组、数值对拍与预算曲线。

**反例与边界**：颜色通道、像素范围、视频时间基错误可能不改变 shape；token 预算设置值与实际 token 数分别记录。

<a id="c-4-6"></a>
## 4.6 视觉编码器与 connector

**依赖**：4.1、4.5。 训练相关综合任务接 7.5/7.10，不依赖完整训练运行。

**问题**：视觉 token 经哪些层进入语言模型；连接器如何改变长度和维度；中间层特征融合带来什么状态与成本。

**对象与源码**：Qwen3-VL-4B-Instruct 的 ViT、patch merger、DeepStack；Gemma-3-4b-it 的 vision tower/projector；[Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl)。 基础训练材料使用 [OpenCLIP](https://github.com/mlfoundations/open_clip) 的 src/open_clip_train/main.py、train.py、data.py、src/open_clip/loss.py；适配材料用 [Qwen3-VL 官方训练](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 逐层采 patch embedding、attention/MLP、merger 与送入 LLM 的特征；手写小型 patch merger/projector | 输入/输出长度、维度和 norm 位置与官方实现对拍，连接器不概括成统一 linear |
| B | 固定图片，扫描分辨率与 batch=1/2/4/8，分别计 processor、ViT、merger、LLM prefill 和峰值 | 完整请求成本与每阶段资源闭合；不同阶段的最优 batch 分别记录 |
| C | 解析 Qwen3-VL DeepStack 的多层视觉特征注入，保存各注入点张量；做去除/只保留末层的分析性消融 | 对照原模型任务质量与成本，不把修改后模型称为官方配置；为 4.8 定义完整特征缓存载荷 |
| D | 重建独立视觉编码器基础训练：图文配对/清洗→双编码器→特征归一化→CLIP 对比 CE 或 SigLipLoss→logit scale/bias→optimizer→checkpoint→零样本/检索评测；分析跨 rank 特征 gather 与梯度 | 用 3×3 相似度矩阵对拍两种目标和错误正负配对；明确 OpenCLIP 参考流程与 Qwen 实际视觉塔原始预训练公开范围，不能假称同一模型训练 |
| E | 沿 Qwen3-VL train_qwen.py 核对 tune_mm_vision、tune_mm_mlp、tune_mm_llm、不同参数组 LR、LoRA 注入和保存；解释先训 connector、联合解冻、DeepStack 多层梯度的条件 | 交付三模块的可训练/冻结/梯度/optimizer 状态矩阵；用冻结 encoder＋小 projector 的一次更新验证，大型真实模型读配方与公开权重；教学 VLM 的连续训练接 7.10-J |
| F | 建立 encoder/connector 微调、视觉蒸馏和独立表征训练的质量检查：跨模态 alignment、遗忘、分辨率迁移、OCR/计数/检索；连接训练特征与部署缓存 | 每类说明监督和输出产物、可复用评测及三个失败模式；不将 ViT forward 正确当作掌握其训练 |

**交付**：新增 `labs/L4/vision_connector.py`、逐层特征清单、阶段时间线和 connector mini。

**反例与边界**：最终视觉 embedding 可能不足以表示所有注入状态；connector 长度压缩不保证视觉信息无损。

<a id="c-4-7"></a>
## 4.7 多模态序列、位置与 mask

**依赖**：4.6、0.4、3.1。 训练相关综合任务接 7.5/7.10，不依赖完整训练运行。

**问题**：占位 token 如何替换成特征；多图/视频怎样分配位置；文本时间戳与多维 RoPE 如何共同表达时空。

**对象与源码**：Qwen3-VL-4B-Instruct 的 `get_rope_index`、Interleaved-MRoPE、DeepStack 与视频文本时间戳；用 Qwen2.5-VL 的位置规则作结构对照，不混用两个模型的 processor。 训练侧固定 Qwen3-VL 的 data_processor.py、packing/RoPE 处理与实际模型 loss；监督分母和阶段编排接 7.5/7.10。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手工构造“文本—图1—文本—图2”和 8 帧视频输入，打印模板、占位符、grid、embedding 替换、position_ids 与 mask | 从原始输入独立算出小例的位置数组并与官方实现逐元素对拍 |
| B | 改变图像顺序、帧时间戳、padding、视频 fps 与 cache_position；逐步比较 prefill 和增量 decode | 空间/时间坐标与有效长度匹配；占位符数量错误、丢帧和错误 RoPE section 有可定位反例 |
| C | 解析 interleaved 频率分配与文本时间戳的不同职责，在时间定位任务中分别干预二者 | 记录真实 token、特征和任务答案；仅同一模型内受控干预用于因果解释 |
| D | 把多图/视频训练样本转换为 input_ids、labels、image/video grid、时间信息、position 和 attention/loss mask；检查 assistant-only、padding、截断、packing 后文档边界与增量推理位置 | 一个包含两条样本的手工例逐元素对拍；标签被 mask 不代表阻断跨样本 attention；训练/推理模板错配与位置漂移有定位路径 |
| E | 分析文本监督、视觉 grounding 标注、视频时间定位对 token/坐标的不同要求，核对数据增强后的 bbox/time label 是否同步转换 | 保留原标注和变换后监督字段，连接独立验证任务；不需运行整个 VLM 微调 |

**交付**：新增 `labs/L4/multimodal_positions.py`，原始序列、位置数组、mask 可视化与失败输入。

**反例与边界**：三维坐标不等于将一维位置复制三遍；共享像素但时间或位置不同不自动拥有相同 KV。

<a id="c-4-8"></a>
## 4.8 多模态 batching、缓存与视频流

**依赖**：4.7、5.2、5.3、5.8。 训练相关综合任务接 7.8/7.10，不依赖完整训练运行。

**问题**：processor/encoder/KV 三层缓存如何区分；视觉工作如何进入调度预算；取消和复用如何保持特征身份。

**对象与源码**：Qwen3-VL-4B-Instruct；vLLM multimodal processor/cache、encoder scheduling，SGLang multimodal input与 feature cache；复用 4.6 的 DeepStack 特征载荷。 多模态训练 batching 与特征缓存以 Qwen3-VL 训练 collator、Cosmos PackingDataLoader 和 7.8/7.10 小实现为对照。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现带模型、processor、图像内容、分辨率、帧采样和位置身份的 feature cache；分别记录三层命中 | 同图重复请求、不同 crop、同帧不同时间、同特征不同文本上下文逐项对照，不能混淆缓存层 |
| B | 先查两引擎对应版本的模型实现、注册与实际安装/启动条件；对纯文本、单图、多图、4/8/16 帧视频重放突发与泊松到达，比较 batching/encoder 准入 | 采视觉 token、encoder 队列、TTFT、峰值与任务质量；缓存收益由省下的工作和完整时间支持；安装或启动失败写具体原因，不等同于该版本没有模型实现 |
| C | 在预处理、encoder 和 decode 阶段取消，插入慢视频输入与错误图像；比较 bounded queue 和无界队列 | 特征、KV、CPU buffers 和队列资源均回收；慢模态对文本请求的长尾影响可量化 |
| D | 比较训练 batch 与服务 batch：有效监督预算、样本权重、梯度累积、每 rank 模态混合、变长/多分辨率分桶；区别训练 feature cache 与推理 KV/prefix cache | 给出相同样本两种 batching 的有效 loss 分母与状态归属；分析冻结/解冻、增强改变、teacher revision 更新导致的缓存失效，不新增多模型训练任务 |

**交付**：新增 `labs/L4/multimodal_serving.py`、最小特征缓存、逐请求状态与缓存失效实验。

**反例与边界**：processor 缓存命中不证明省掉 vision forward；流式上传视频不保证模型支持增量编码。

<a id="c-4-9"></a>
## 4.9 语音输入与流式 ASR

**依赖**：0.4、4.1、5.1、2.0c。 训练相关综合任务接 7.5/7.8/7.10，不依赖完整训练运行。

**问题**：波形如何变成有效序列；分块输入与流式输出如何区别；边界修订和端点判断需要哪些状态。

**对象与源码**：主例 `Qwen/Qwen3-ASR-0.6B`，扩展 1.7B；结构对照 `openai/whisper-small`。阅读 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 `qwen_asr/inference/qwen3_asr.py`、`ASRStreamingState`，对照官方 vLLM backend 与 [SGLang-Omni ASR](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)。 新增 [SpeechBrain LibriSpeech Transformer](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer) 的 train.py/hparams 与 [Qwen3-ASR finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) 的 qwen3_asr_sft.py。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 16 kHz PCM 的声道、幅度、分帧到音频特征/encoder，手写长度传播与边界补齐；输出有效长度、mask、特征和音频 token | 官方样例逐层对拍；重采样、空音频、截断、静音和错误采样率有可定位反例 |
| B | 实现可重放输入分块，chunk=1/2/4 s、feed step=20/100 ms、unfixed tokens=0/5/10；逐块记录累计音频、回退文本与稳定前缀。Whisper 对照的每种 chunk 都显式送入最后不足整块的尾段，再核对最终覆盖区间 | 区分在线接收、累计重编码、模型状态复用和输出分片；不以有无 encoder KV 定义所有流式 ASR。固定窗口模型的重算成本按实际 padding/窗口推导，不能用累计长度求和直接断言实测为平方复杂度 |
| C | LibriSpeech test-clean 与 AISHELL-1 test 各固定 100 条按时长分层样本；同音频比较整段与分块、batch=1/4/16 | CER/WER、首个稳定转写、最终时延、RTF、峰值与重复/漏字一起报告，评测清单与正规化脚本固定 |
| D | 两引擎用共同整段模式对照；官方 streaming 与 SGLang 的上传后 SSE 分别记录支持语义，测试中断与慢输入 | 接口不支持持续输入时明确作为独立模式，保留音频状态回收和长尾结果 |
| E | 重建独立 ASR 基础训练：音频/转录准备→tokenizer→特征与增广→encoder/decoder→CTC＋attention 目标→优化/恢复→beam/LM→WER；解释对齐、blank、teacher forcing、label smoothing 和长度 mask | 用短音素/字符序列的 CTC 小例与 decoder CE 对照；tokenizer 类型按实际 hparams 确认；采样率、speaker split、SpecAugment 和归一化都有原始配置入口 |
| F | 沿 Qwen3-ASR JSONL audio/text→prefix/full processor→labels 的 -100 mask→训练参数组→save callback→resume；阅读 0.6B/1.7B 模型与具体脚本支持 | 固定最多 8 条样本只检查 collator/监督；列 decoder/audio tower 是否训练、LR/AMP/accumulation/clip、processor 文件和恢复状态；不声称公开了 AuT 全部预训练语料 |
| G | 比较蒸馏/领域适配的 teacher transcript、logit/feature 信号与基本 ASR 目标；设计噪声/口音/语言/长音频验证和遗忘检查 | 用公开原始评测和 7.7 监督机制补全训练知识；流式可用性需有训练 chunk/attention/状态依据，整段 SFT 不自动获得流式能力 |

**交付**：完善 `labs/L4/asr_frontend.py`、`asr_stream_bench.py`，原始音频清单、转写版本、时间线、错误和评测结果；开篇、实验、陷阱与自测统一区分 Whisper 的有限样本实测、Qwen3-ASR 未运行路径和计划中的完整评测矩阵。

**反例与边界**：最后转写正确不能证明中间字幕稳定；输入 chunk、模型计算 chunk 和网络输出 chunk 不是同一个对象。

<a id="c-4-10"></a>
## 4.10 语音生成、codec 与播放

**依赖**：4.9、5.1、10.1。 训练相关综合任务接 7.6/7.7/7.10，不依赖完整训练运行。

**问题**：多码本如何组成音频；首包和首可播放时间怎样不同；AR codec 与 flow 声学生成怎样组织状态。

**对象与源码**：共同主例 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，固定许可明确的官方参考音频与转写；文本直出对照 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`。使用 [vLLM-Omni 0.18 TTS](https://docs.vllm.ai/projects/vllm-omni/en/v0.18.0/user_guide/examples/online_serving/qwen3_tts/) 与 [SGLang-Omni TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html)；连续生成对照 [F5-TTS v1 Base](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)。 补充 [DAC 训练](https://github.com/descriptinc/descript-audio-codec)、[Qwen3-TTS 官方单说话人 SFT](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)、[F5-TTS 训练](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/train)、[ZipVoice 全流程](https://github.com/k2-fsa/ZipVoice/blob/master/egs/zipvoice/run_emilia.sh) 和 [CosyVoice2 GRPO](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 跟踪 preprocessing→tts engine→vocoder，打印码本数量、采样率、token/chunk 对应长度；手写码本打包/解包和拼接队列，分别列紧凑编码下界、实际整数张量与传输格式 | codec 边界、有效采样点和音频长度与官方实现对拍；22 字节/帧标为本章 11-bit 打包格式，不能当官方线上码率或据此解释码本大小的设计动机；静音抑制、熵编码与容器开销另列条件，Base 的参考音频不能省略 |
| B | 固定 40 条中英文短/长文本，分别比较离线与真实 chunk 输出；chunk 对应 4/8/16 个 codec frame，超出接口限制的点不运行 | 采网络首包、首可播放、播放开始、underrun、RTF、峰值、边界爆音与 ASR 回转 CER/WER |
| C | 两运行时分别对 stage batching、graph on/off 和异步 chunk 传输做消融；batch=1/4/8，加入慢消费者 | 时序、资源和输出状态一致；区分图捕获启动成本与稳态，不能跨不兼容 Transformers 栈直接混装 |
| D | F5TTS_v1_Base 使用固定参考音频、文本、初始噪声与目标时长，扫描 NFE=8/16/32；追踪 flow solver→vocoder | 与 AR 路线按内容/可懂度/时长分别报告质量—成本，不能仅比较音频 token/s |
| E | 用 DAC 的 scripts/train.py、conf/final 配置追踪音频采样/切段→encoder/RVQ/decoder→mel/STFT 重建、commitment/codebook、discriminator/feature matching→双 optimizer→checkpoint→码率/重建评测 | 列清 codec 基座自身的训练和下游 TTS 条件生成；用小码本/损失项检查梯度与有效长度，音频数据配方里的内部路径不视为完整公开数据 |
| F | 对 Qwen3-TTS-12Hz-0.6B-Base 研读 audio/text/ref_audio→prepare_data.py 的 audio_codes→dataset.py→sft_12hz.py；核对 Talker loss＋0.3×SubTalker loss、speaker embedding detach、模型参数组和保存 | 交付码本/样本/监督/更新矩阵；解释输出 config 的 CustomVoice/speaker 映射、speaker embedding 写入和 speaker_encoder 移除；官方示例的 Base 单说话人范围不外推多说话人完整训练 |
| G | 从 F5TTS_v1_Base.yaml、prepare 数据脚本、cfm.py、trainer.py 解释 mel、时长/文本 padding、随机 span mask、噪声/时间、conditional dropout、flow loss、EMA 和 vocoder；对照 ZipVoice 基础训练→两阶段蒸馏→平均权重→ONNX | 只运行小 flow/loss 或长度参照；区分训练声学模型、训练 vocoder、少步蒸馏与简单改 NFE；EMA 对短程微调的滞后有具体解释 |
| H | 沿 CosyVoice2 GRPO 完整流程标出 speech-token learner、token2wav 和 ASR reward；分析文本/说话人条件、有效时长、奖励服务、转换和独立评测 | 复用 7.6 的 RL 材料；CER/WER、说话人相似、自然度、韵律与实时性分别评价；公开流程不冒充 Qwen3-TTS 官方 RL 配方，无需实际训练语音模型 |

**交付**：完善 `labs/L4/tts_codec_probe.py`、`tts_stream_bench.py`，原始波形、码本与播放事件；可重复听检和自动指标均保留。VITS 的整段生成时序、Qwen3-TTS 的配置分析及未运行的真实流式路径在开篇、表格和自测中使用一致边界；无运行依据的 AR RTF 不写成模型固有值。

**反例与边界**：输出分块不保证持续可播放；ASR 回转指标不能代表音质、自然度或声线相似度。

<a id="c-4-11"></a>
## 4.11 原生 Omni、跨阶段调度与打断

**依赖**：4.8、4.9、4.10、5.8、5.11；多卡部署先完成 6.2 的 TP 基础。 训练相关综合任务接 7.6/7.7/7.10，不依赖完整训练运行。

**问题**：音视频怎样共享时间轴；阶段队列如何传播压力；取消如何同时终止计算、传输和旧音频播放。

**对象与源码**：原生主模型 `Qwen/Qwen3-Omni-30B-A3B-Instruct`；级联参照 Qwen3-ASR-0.6B→Qwen3-VL-4B-Instruct→Qwen3-TTS-12Hz-0.6B。阅读 [vLLM-Omni Qwen3-Omni](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/examples/online_serving/qwen3_omni.md)、[SGLang-Omni 模型配置](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_omni.html)与 [pipeline](https://sgl-project.github.io/sglang-omni/developer_reference/pipeline.html)。 训练材料使用 [Qwen3-Omni 报告](https://arxiv.org/html/2509.17765) 和 [ms-swift 固定配方](https://github.com/modelscope/ms-swift/tree/0673cf75dca7d0b9b608b4a76632fb508ead5076/examples/models/qwen3_omni)，同时检查 swift/model/models/qwen.py 和 swift/template/templates/qwen.py 的实际 forward/labels。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 Thinker/Talker/Code2Wav 输入输出、codec/hidden 状态与队列；从模型配置核算完整权重和峰值。初始部署为 worldvln Thinker TP=2、Talker 一卡、Code2Wav 一卡 | 仅在整除、版本与显存预算满足后部署；逐 stage 的 readiness、设备与传输方式明确，不能按 3B 激活参数估计总容量 |
| B | 用 20 段许可明确的音画片段，包含字幕时间标记、说话暂停与视觉事件；比较级联和原生的输入张量、输出任务与时间轴 | 两条真实链路各自验收，级联结果不替代原生能力；共同任务的理解质量与语音可懂度都有记录 |
| C | 两运行时分别扫描并发=1/2/4、队列上限=1/4/8、消费者延迟=0/100/500 ms；记录 stage 工作、跨阶段 chunk、首可播放和持续输出 | 背压由队列/事件支持；SGLang Omni 的阶段并发与其不支持的 AR overlap loop 分开，不套用基座开关 |
| D | 在 Thinker、Talker、Code2Wav、播放四处发起打断；用 session epoch 丢弃旧包，注入断连、失败与重试 | 测打断至静音、停止计算与资源回收，禁止旧响应重新播放；同会话下一请求正常执行 |
| E | 在共同 BF16 配置对照两运行时，再单独评估支持的低精度/分离部署；保存完整任务×模态×流式方式×精度×并行矩阵 | 支持声明须落实到对应源码和运行；资源不足时原生任务保持未完成，不能以 thinker-only 交付冒充全部 Omni |
| F | 重建 audio/vision encoder、Thinker、Talker、codec/Code2Wav 的基础训练、模态对齐和联合/后训练阶段；逐项列数据、目标、冻结/更新模块、损失权重、时间对齐和输出产物 | 按论文、代码、数据、权重、日志分别标公开范围；不能把所有模块都概括成 LLM SFT，也不能用独立 ASR＋TTS 级联代替原生 Omni 训练解释 |
| G | 分析 ms-swift Qwen3-Omni 数据混合、模板/label、freeze_vit/freeze_aligner、LoRA target、ZeRO/Megatron 和保存；从 loss 反查 Thinker/Talker/Code2Wav 是否实际得到梯度 | 提交每个模块的证据表，明确当前配方训练的真实范围；输入包含音频不证明更新语音生成模块；只做样本/配置/小冻结分支检查，不部署整套训练 |
| H | 比较模态不均衡、缺模态 batch、说话输出 teacher forcing、联合损失尺度、音画对齐、遗忘以及 RL/蒸馏需要的反馈；评测分别覆盖文本理解、音频理解与生成、视听一致性 | 连接 7.6/7.7/7.10 的通用系统和当前公开材料；原模型未公开的全模态 RL/蒸馏 recipe 标为未知，不拼接社区案例后称完整复现 |

**交付**：新增 `labs/L4/omni_session_runtime.py`、`omni_stream_bench.py`，级联与原生配置、原始音画输出、统一时间轴与取消回收记录。

**反例与边界**：一个 stage 输出成功不表示全链路成功；首响应时间不能替代首可播放和无断流的持续交付。

<a id="c-6-0"></a>
## 6.0 进程组与通信语义

**依赖**：1.3、2.0c。

**问题**：rank 怎样持有数据；collective 如何匹配次序；API 返回、stream 完成和 buffer 可复用有何区别。

**对象与源码**：先用本地 Gloo 两进程，再用 worldvln 两张 L40S/NCCL；复用 `labs/L6/distributed_basics.py`；PyTorch `distributed_c10d.py`、ProcessGroupNCCL/Gloo、Work 与 CUDA stream/event。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现带 rank/step/op/shape/dtype 的序列记录器，运行 all_reduce、all_gather、reduce_scatter 和 send/recv；逐步列出全局和局部值 | 小整数与单进程参照完全一致，进程组成员与张量归属可重建 |
| B | 分别注入 op 次序错配、shape 错配、单 rank 提前退出；用超时和独立子进程保证失败有界结束 | 原始堆栈、最后匹配操作、退出码和回收记录齐备；不是只演示“程序挂住” |
| C | 通信流写入、计算流读取同一 buffer；对比正确 event、过早覆盖、只等 CPU 返回和正确等待 | 数值和完成时间都验证；Work.wait 的具体语义按所选 backend 源码解释 |

**交付**：新增 `labs/L6/collective_contracts.py`、`comm_sequence.py`，CPU/GPU 两种完成语义与失败案例。

**反例与边界**：同一个 op 名不能匹配不同进程组；非阻塞返回不保证 buffer 可立即读取或重用。

<a id="c-6-0b"></a>
## 6.0b DeviceMesh 与 DTensor

**依赖**：6.0、7.0。

**问题**：全局张量怎样表示为局部片段；Shard/Replicate/Partial 的数值语义如何不同；布局传播怎样触发通信。

**对象与源码**：PyTorch DeviceMesh、DTensor placement、redistribute、sharding propagation 与 FSDP2；先 2 ranks，再 2×2 mesh；不强行使用五卡构造不可整除 mesh。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写三类 placement 的全局值重建与转换；张量用 8×6 和不均匀 7×5，记录每 rank shape/stride/offset | 与 DTensor full_tensor 对拍；Partial 在归约前不能当完整副本 |
| B | 对 linear、matmul、sum、reshape 预测输出 placement，再查实际传播规则与 collective；测试 shard 维度转换 | 输出布局、全局数值和通信序列一致；不支持的布局明确报错，不在参照中静默变换 |
| C | 运行包含 Partial 的前后向，比较单机梯度与分布式梯度；记录 redistribute 的临时 buffer 与释放 | 数值、梯度、通信字节闭合；为 6.2/7.2 提供可重复的布局实验 |

**交付**：新增 `labs/L6/mini_dtensor.py`、`dtensor_probe.py`，mesh 映射、布局转换和梯度对拍。

**反例与边界**：局部 shape 相同不保证全局语义相同；布局变化可能隐含通信，不能只计显式 collective 调用。

<a id="c-6-1"></a>
## 6.1 集合通信与成本模型

**依赖**：6.0。

**问题**：算法和协议何时切换；带宽口径怎样统一；拓扑与消息尺寸怎样决定成本。

**对象与源码**：worldvln 2/4 张 L40S；[NCCL](https://github.com/NVIDIA/nccl)、[nccl-tests](https://github.com/NVIDIA/nccl-tests)；复用现有 `labs/L7/distributed_collectives.py` 的有效采集部分，并纠正 7.2 的章节归属。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 明确定义 M 是每 rank 输入还是全局输出字节，分别推导 ring all-reduce、all-gather、reduce-scatter 与 tree 的启动/传输成本 | 时间、算法带宽和总线带宽各给公式；rank 数变化与每 rank 字节不混用 |
| B | 消息从 4 KiB 到 256 MiB 按 4 倍扫描；记录各 rank 时间、max-rank 完成、实际算法/协议/通道、近/远拓扑 | auto 与合法强制算法作对照；从日志/实现确认选择，不由套公式算出相近带宽“证明 ring” |
| C | 对同尺寸 GEMM 比较通信单独、计算单独和带正确依赖的重叠；记录竞争的 SM、带宽与 buffer | 完整调用收益与理论可重叠上限分开；相同数据量的不同 collective 不预设性能排序 |
| D | 对 NVLS、NCCL symmetric memory 和跨机 transport 阅读触发条件；已有无 NVLink 平台只运行其支持路径。对千兆 TCP 的饱和大消息、延迟主导的小消息和 CPU/proxy 成本分别解释 | 形成硬件/驱动/权限/算法矩阵；NCCL 接近裸 TCP 的结果限于本次消息尺寸与链路，不能迁移为所有 transport 零开销，也不能据带宽饱和断言 GIN 在所有负载下无收益；跨机准备依 ENVIRONMENTS，不修改共享驱动 |

**交付**：新增 `labs/L6/collective_costs.py`、拓扑与算法选择记录、参数扫描及成本模型。

**反例与边界**：PCIe Gen4×16 的编码后理论带宽、物理双向和测得有效载荷分开；单机 NVLink 不能代入千卡跨节点总线。

<a id="c-6-2"></a>
## 6.2 推理并行

**依赖**：6.0b、6.1、4.4、5.3。

**问题**：TP/PP/EP/CP 怎样切分工作和状态；聚合发生在哪里；什么负载足以覆盖通信成本。

**对象与源码**：Qwen3-1.7B/8B 为 TP/PP/CP 主例，OLMoE-1B-7B 为 EP；vLLM 与 SGLang 并行组、column/row parallel linear、pipeline runner、attention 与 expert mapper。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写两 rank column/row parallel MLP 与两阶段 PP；列每 rank 权重、激活、KV、通信与聚合公式 | 与单卡输出逐元素对拍，明确 vocab/head/expert 的整除条件 |
| B | 在 worldvln 固定 BF16，对 Qwen3-8B 比较 TP=1/2/4 与 PP=2，先 B=1/8、S=128/2048/8192 | 采完整 TTFT/TPOT、collective、峰值分片与临时 buffer；保留小模型多卡减速 |
| C | 对 OLMoE 运行 EP=2/4 的合法路径；为 CP 接 6.5 的分片 attention，分别验证 token、expert 和位置归属 | 每种并行方式有真实机制及状态检查；不将 TP 的结果泛化为 EP/CP |
| D | 给 Qwen3-Omni Thinker 提供 TP=2 部署验证，检查与后续 stage 的传输/取消；合法组合按总峰值容量选择 | 解锁 4.11 的原生链路；不以各卡显存简单相加宣称可加载 |

**交付**：新增 `labs/L6/parallel_inference.py`、`mini_tp_pp.py`，rank 归属表、真实部署配置与单卡对拍。

**反例与边界**：PP 阶段输出慢可能拖住所有后续阶段；并行速度需在同任务/质量/SLO 下比较。

<a id="c-6-3"></a>
## 6.3 MoE 通信与负载均衡

**依赖**：6.2、4.4。

**问题**：pack/dispatch/combine 怎样实现路由；通信和 grouped GEMM 如何耦合；不同通信库的职责与硬件条件如何区分。

**对象与源码**：OLMoE 与 4.4 保存的真实路由；vLLM/SGLang MoE backend、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[UCCL-EP](https://github.com/uccl-project/uccl/tree/main/ep)、NVSHMEM；分别固定 DeepEP V1 与 V2 的实现入口。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 token pack、send/recv counts、expert offsets、反排列与 weighted combine；以带唯一 token ID 的小整数验证分发 | 无丢失、重复和错位；零 token rank、空专家和倾斜分布均覆盖 |
| B | 在 worldvln 使用可运行的 NCCL all-gather/reduce-scatter 或 all-to-all 路线，tokens=1/32/256/4096、top-k=2/8、倾斜=0/20%/50% | 分阶段与完整时延、通信字节、padding、workspace、实际负载齐备；与单机专家参考对拍 |
| C | 对照 V1 的 NVSHMEM 路径、V2 ElasticBuffer/NCCL 接口和 UCCL 的 transport/proxy；核对 V2 的 NCCL 版本、NVLink/RDMA 要求 | 按实际设备决定能运行的集合；L40S 无 NVLink 不冒充 DeepEP 目标平台，仍完成协议和源码分析 |
| D | 静态专家布局与 EPLB 对照，改变访问分布和重平衡周期；记录权重迁移、同步、cold/hot experts 和请求长尾 | 同一完整 workload 下检验均衡收益是否抵消迁移；不以 token 数变均匀直接证明更快 |

**交付**：新增 `labs/L6/moe_dispatch.py`、`moe_placement_bench.py`，通信协议 mini、支持矩阵和路由/迁移时间线。

**反例与边界**：NCCL、NVSHMEM、DeepEP、UCCL 不是同一抽象层的四个可无条件互换库。

<a id="c-6-4"></a>
## 6.4 Prefill–decode 分离

**依赖**：6.2、5.2、5.8、1.5。

**问题**：KV 交接的所有权怎样转移；传输和两侧排队怎样共同影响收益；取消和重复交接怎样回收资源。

**对象与源码**：Qwen3-1.7B/8B，vLLM NIXLConnector 与 SGLang PD；[Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/v-0-8-1/design-docs/disaggregated-serving)、[NIXL](https://github.com/ai-dynamo/nixl)、Mooncake/LMCache 的具体 connector。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 定义 request/session/model revision、token/position、dtype/scale、TP layout、block IDs、传输完成与释放 ACK；写 GPU/CPU 搬运参照 | 交接后首 token、后续 logits 与并置 baseline 对拍；错误身份/布局明确拒绝 |
| B | worldvln 先做同机 1P1D，再比较 1P2D；扫描输入=128/2048/8192、输出=32/256/1024 和到达率。分列 P 侧计算、交接、D 侧排队和首输出；已有阶段实测相加只作为模型预测 | 固定相同总 GPU 预算，对比并置与分离 goodput、TTFT、TPOT、峰值及传输成本。正文与自测将单请求增加传输成本限定于相同计算路径、无排队和无其他收益的比较，不写成所有 PD 部署必然更慢 |
| C | 在注册、发送中、接收后、decode 中取消或中断；注入重复 ACK 与超时，核对两侧 block/refcount | 失败不会遗留 staging/KV 或重复提交 token；恢复后同实例可继续处理合法请求 |
| D | 分别追踪 NIXL/Mooncake/LMCache 的真实 transport 与完成接口，跨机条件具备后单独验证；异构 TP 仅在支持的布局转换下运行 | 建立布局转换与传输成本，不将单纯 memcpy 带宽当服务收益；上游论文作为设计参照而非本机测量 |

**交付**：新增 `labs/L6/kv_handoff.py`、`pd_serving_bench.py`，双方事件、状态对拍和故障回收。

**反例与边界**：服务发现和路由完成不等于 KV 可用；相同模型名但不同 revision 或量化布局不能直接交接。

<a id="c-6-5"></a>
## 6.5 长上下文分布式

**依赖**：6.0b、6.1、3.2、3.4。

**问题**：序列/上下文分片怎样改变 attention；归一化统计量怎样跨 rank 合并；通信、重算与状态何时主导。

**对象与源码**：Qwen3-8B 的合法长上下文配置；PyTorch DTensor、Megatron CP/SP、DeepSpeed Ulysses 与 ring attention 实现；kernel 使用设备支持的 FlashAttention 路径。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 在两 rank 写分片 Q/K/V、causal mask 与 online m/l/O 归并；对比 ring 传 KV 与 Ulysses all-to-all 换维。从实际通信 buffer 的 shape、dtype 和发送目标统计字节，分别列发送/接收、含/不含自身 rank；检查 ring 在传输前的 GQA 扩展，以及 Ulysses 的 Q/K/V 三次换维与 O 换回 | 小尺寸完整 attention 的值和梯度对拍；非整除长度、GQA、窗口边界有用例。`context_parallel_attention.py` 与 `long_context_bench.py` 的账一致；理论 G 个 KV head 与当前实际 H 个 head 分开，逻辑两阶段与四次 collective 分开；不沿用未经对拍的 16 倍通信结论 |
| B | worldvln 2/4 卡，S=2048/8192/16384/32768，分别测训练激活与推理 KV；固定模型原生有效上下文，比较显式分数矩阵与设备支持的 FlashAttention 路径 | 分列单层实测、36 层容量外推和真实全模型结果；平方激活只适用于保存完整 attention 矩阵的实现。ring 通信量明确随 S 及 (CP−1)/CP 缩放，不描述为与 S 无关；记录每 rank 峰值、通信轮次、重算与完整时间 |
| C | 比较 CP/SP 与 TP 的合法组合，加入负载不齐、不同 mask 和小 batch；保存实际 collective 顺序，对照 padding 后的 Ulysses 与变长 ring | 全局位置、mask 和数值一致；解释重叠的依赖与实际通信/计算转换区间。当前 mini 拒绝非整除输入不等于 Ulysses 算法无法处理，不写成 ring 是唯一选择 |

**交付**：新增 `labs/L6/context_parallel_attention.py`、`long_context_bench.py`，分片布局、梯度、容量与时间曲线。

**反例与边界**：改配置扩大上下文不等于模型质量保持；视频和混合递推模型的并行条件在各自章节单列。

<a id="c-7-3"></a>
## 7.3 训练框架架构

**依赖**：7.2；数据与精度分别连接 7.8/7.9；按[共同任务](#training-workflow)完成公开流程材料。

**问题**：一个真实训练配方如何落到模型、数据、并行和优化器；框架维护哪些状态；扩展模型或训练方法必须改哪些接口。

**对象与源码**：PyTorch FSDP2、DeepSpeed、Megatron Core/Megatron-LM、[TorchTitan](https://github.com/pytorch/torchtitan) 四路线均做源码比较；FSDP2 是并行 API，TorchTitan 是使用这些 API 的训练应用。贯穿配方为 [SmolLM3 Nanotron](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3)＋[后训练](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3)。TorchTitan 读取 `torchtitan/train.py`、`distributed/`、`components/checkpointer/`、模型 `config_registry.py`，不能沿用已迁移的旧目录。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 SmolLM3 stage1/stage2/stage3、长上下文、中训练、SFT、APO 配置画阶段图；逐项解释数据权重、token 预算、batch、scheduler、冻结/参数组、checkpoint 接续和最终模型选择 | 配方原文与解释逐字段对应；给出数据→训练→评估→阶段产物的真实文件入口；文件名、博客阶段边界和 YAML 有差异时分别按来源核对，不拼凑新配方 |
| B | 为四路线追踪 init/process group→model build→parallelize/wrap→make_batch→forward/loss→backward→optimizer→save；记录状态所有者、hook、配置验证、错误传播及扩展点 | 一张共同接口表加四份真实调用链；关键边界给出对应版本的源码链接、file/symbol；不能用四段启动命令代替架构比较 |
| C | 完善只读配方检查器，输入实际 YAML/TOML/Python config 的规范化导出；先检查 microbatch、accumulation、DP 的类型与正值，再检查 global batch、分片整除、dtype、学习率单位、样本路径、checkpoint 和资源预算 | 完善 `labs/L7/recipe_inspector.py`；零值、负值、非整数因子、loss 分母错、resume 缺状态、adapter target 不存在和预算遗漏均有反例。解析成功、静态检查通过与完整框架合法性分别说明，不执行未知配置代码 |
| D | 复用 7.2 的两 rank 小例作为更新参照；比较各框架原生 debug/tiny 模型对冻结 projector、LoRA、不同 loss 的接口要求，完成一个最小适配接口说明 | 仅一条路径实现必要小验证；其余源码/官方测试材料明确来源。不再移植 SmolLM3 到四框架，也不要求跨框架大模型数值和吞吐竞赛 |
| E | 研读训练框架如何接 AMP/FP8、selective activation checkpointing、compile、通信重叠、FSDP/ZeRO optimizer 与 DCP；追踪一项融合/重排对保存值和更新边界的影响 | 说明兼容性条件、源码差异、性能收益所需 shape/设备；公开规模结果写清模型、硬件与计量单位，接 7.9/7.11 |
| F | 用 SmolLM3 公开阶段权重索引和评测配置重建“哪个 checkpoint 评哪组任务”；核对 Transformers 导出与 Nanotron 原训练状态的区别 | 交付各阶段权重、公开训练日志和评测的入口；没有下载或没有公开的 optimizer/RNG 不标已验证恢复；本任务不重训 SmolLM3，教学训练使用共同项目 |
| G | 以 MiniMind 教学 loop 为起点，映射到 PyTorch/FSDP2、DeepSpeed、Megatron、TorchTitan 的配置、数据、并行、更新和保存接口；深入 Puro-Megatron 的阶段切换、MuonH 参数路由、FP8 与恢复源码，和 SmolLM3 比较 | 交付教学 loop→生产接口映射及同一问题的设计对照；解释小模型暂未暴露的供数/通信/数值瓶颈；只选择一条预算内框架路径接实际教学训练，不要求四框架复训 |

**交付**：阶段配方材料、四框架接口/状态/调用对照、只读配置检查器及小验证复用说明。

**反例与边界**：原生支持模型族不保证同一结构和 loss 语义；框架参数默认值、优化器实现和数据顺序的不同会改变训练。


<a id="c-7-4"></a>
## 7.4 数据、checkpoint 与容错

**依赖**：7.0b、7.2、1.5；数据内容与缓存由 7.8 定义；作业级重启接 7.11。

**问题**：训练在什么逻辑时刻形成一致快照；恢复需要重建哪些状态；数据与并行变化怎样影响继续训练的语义。

**对象与源码**：PyTorch Distributed Checkpoint、[StatefulDataLoader](https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html)、TorchTitan `components/checkpointer/dcp.py` 与[checkpoint 文档](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md)；适配器、SpecForge 和 Cosmos 的保存/导出结构作对照。运行限小模型和轻量状态。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 列出 model、optimizer、master weight、scheduler、scaler、RNG、EMA、FP8 amax/scale、sampler/worker 游标、global step、消费数据版本、adapter/teacher 引用；区分必须保存、可重建和外部不可变依赖 | 形成按预训练/SFT/KD/RL/flow 分列的状态 schema；每个字段注明 owner、保存时刻、恢复入口和缺失后果；权重文件与训练 checkpoint 分开 |
| B | 构建 12 条带 ID 的 map/iterable 样本，测试 worker=0/2、预取与未消费队列；分析跨 rank 数据分配及尾 batch | 保存实际消费序列和已处理/已提交游标；不会把 worker 聚合当跨 rank 数据状态恢复，或把预取位置当已完成更新位置 |
| C | 小模型一次更新后执行同步保存，再对 async staging/save 注入保存前、staging 后、文件写完前、manifest 提交前四种失败；沿版本实际的 Future/完成信号解释边界 | 沿真实 DCP staging、save、metadata 与完成信号解释流程，并与自定义提交协议分开；完善 `labs/L7/checkpoint_failure_injection.py`，未完成 checkpoint 不被 latest 选中，参数和 optimizer 属于同一更新 |
| D | 恢复后对相同下一批执行一次更新，与不中断参照比较样本、loss、梯度、参数、LR 和 RNG；逐项移除关键状态制造反例 | 新增 `labs/L7/resumable_training.py`；冻结 world size 的严格恢复和只加载权重的新训练分别验收；单步反例用于定位错误，实际教学训练的恢复由 H 验收 |
| E | 研读 DCP 重新分片与 world-size 改变，分析 batch/顺序/RNG/optimizer 的条件；用小型 shard 元数据演示 2→4 的布局映射 | 完整给出分片区间交集与 2→4 映射推导，说明布局恢复与训练轨迹一致的不同条件；本任务通过小型元数据演示布局变化，同时说明 SpecForge 的同 trainer world-size 限制 |
| F | 比较 adapter 保存、merged 模型、teacher 引用、EMA 权重、quantizer 状态与分片 optimizer；结合 Cosmos 的 DCP→HF→Diffusers 和 SpecForge runtime→export 说明阶段产物 | 交付可检查的轻量文件清单/索引与转换依赖；不为学习转换而下载/复制全部大 checkpoint，真实大模型 round-trip 保持独立实测条件 |
| G | 用已公开或已有 I/O 材料计算保存频率、staging 峰值、带宽、恢复时间和故障损失之间的关系；区分本地临时保存、共享盘持久化和异地容灾 | 给出含单位和假设的预算小程序；checkpoint 越频繁不一定 goodput 越高；规模结论标为推演或外部测量 |
| H | 在文本/图像贯穿项目的中间阶段计划一次中断，恢复后继续到阶段终点；先对齐相同下一批与数次更新，再比较预定后续窗口的数据、LR、loss 和最终评测；验证导出与训练态保存的差别 | 交付可运行 resume 命令、完整状态字段、消费样本连续性、恢复前后对照与恢复成本；声明确定性/容差条件，best 与 latest 各按用途加载；复用项目运行，不额外重训整套模型 |

**交付**：完整的状态与分片推导、DCP 源码解析、设计取舍和可运行命令；配套状态 schema、短恢复轨迹、提交/失败注入实现与导出材料。

**反例与边界**：async API 返回、staging 完成、文件写完和持久提交不是同一事件；保存权重不等于可恢复训练。


<a id="c-7-5"></a>
## 7.5 SFT、LoRA/QLoRA 与 DPO

**依赖**：7.0b、7.1 与 7.8 数据基础；SFT/LoRA/DPO 教学实践可先执行，QLoRA 部分接 4.3，蒸馏系统接 7.7，训练精度接 7.9。

**问题**：从预训练模型到适配模型需要哪些数据与目标；参数高效方法究竟更新什么；偏好优化与导出如何改变模型行为。

**对象与源码**：[PEFT](https://github.com/huggingface/peft)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)、bitsandbytes；读 [SmolLM3 mid/SFT/APO](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3) 的完整阶段。Qwen3-1.7B 作为熟悉的模板/模块例，数值检查用小网络；UltraChat/UltraFeedback 小样本用于解释 schema；教学 SFT、LoRA、DPO 使用共同项目中独立划分的有限数据。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对照从零训练、继续预训练、领域适配、SFT 和偏好优化的输入、监督、初始化与评测；从公开配方抽 16 条会话/偏好对，解释指令构建、拒答/工具结果、thinking 模式、EOS 与长度分布 | 给出阶段选择表及训练/验证拆分理由；保留原始数据出处、固定样本 ID 和模板；不把所有训练都称为 instruction tuning |
| B | 重建 chat template→tokenize→packing→attention/loss mask→CE；检查 prompt、assistant、多轮历史、tool response、padding、截断后的有效监督和跨样本 attention | 手写逐 token loss 与 TRL 同一小 batch 对拍；展示 answer 被截空、pad=EOS 被错误屏蔽、只设置 loss mask 却跨样本 attention 的反例 |
| C | 由 W＋sBA 推导 LoRA 参数量、梯度、rank/alpha、初始化、dropout、target_modules、modules_to_save、adapter bias；比较 full/LoRA 的 optimizer 和 activation 状态 | 小线性层 rank=2/4 做一次更新，验证冻结 base 和初始零增量；实际模型 rank=8/16/32 仅做参数/内存账，不开展质量网格 |
| D | 解释 NF4、double quant、存储/反量化/compute dtype、paged optimizer、prepare_model_for_kbit_training；对照 BF16 LoRA、NF4 QLoRA、QAT/QAD 与 FP8/FP4 训练 | 交付五种方法的更新对象、状态和部署产物表；一层 fake/real quant 路径只验证数值与梯度流；不把 QLoRA 等同“训练一个原生 INT4 GEMM 模型” |
| E | 从 preference 数据推导 Bradley–Terry/RM 的作用与 DPO 的 policy/reference log-ratio，手算 chosen/rejected sequence logprob、beta 和梯度；对照 SmolLM3 APO 的选用原因与配置 | 新增 `labs/L7/preference_loss_reference.py`；区分 DPO、RL reward model、行为策略；长度归一化、错误 pairing、reference 更新、模板不一致都有反例 |
| F | 完整分析 SFT/LoRA/DPO 的数据装载、参数组、optimizer/scheduler、AMP、累积、activation checkpoint、eval/save/resume；对应一份可读的官方配置 | 复用 7.0b/7.4/7.9 验证数值和状态；I/J 完成预算内 SFT、LoRA 与 DPO 的连续训练；报告公开学习曲线、任务评测、遗忘/过拟合分析的来源，单步 loss 不作为质量证明 |
| G | 从 PEFT 保存追到 adapter_config、adapter 权重、base revision、rank/scale、tokenizer/template 和额外训练模块；比较 merged/unmerged、量化 base 合并与重新量化 | 一层小模型 round-trip 对拍；5.10 的质量用可核对的公开已训练 adapter 及其匹配 base，或已存在的合法工件；不要求先训练新 adapter 才能完成服务学习 |
| H | 把同一适配契约映射到 ViT/connector、音频编码器、DiT、TTS Talker、VLA action head；指出文本 token loss、flow loss、codec loss、动作 mask 的差别 | 对应 4.6/4.9/4.10/10.3/10.6 真实训练入口与 7.10 参数组表；不能将 LLM q_proj/v_proj 的注入列表直接复制到所有模型 |
| I | 承接 7.1-F 的预训练权重，在 7.8 的独立会话数据上完成 SFT；逐 token 检查模板/labels/mask，再运行训练、验证、模型选择、导出和部署；保持预训练 base 为对照 | 交付 base/SFT 的学习曲线、留出问答与格式任务、遗忘/过拟合样例和推理命令；训练/验证/最终测试隔离，预先定义质量标准，结果不足时保留未通过项 |
| J | 由同一 SFT 起点分别运行 LoRA 领域适配和 DPO 偏好优化；选一个 rank 与一套 beta 主配置，用小 pilot 确定预算内步数；比较原 SFT、适配模型及等预算对照 | 两个分支各交付完整训练/评测/导出命令、曲线、任务质量与通用能力变化；DPO 同时记录偏好 log-ratio 与独立任务指标，不能把训练偏好准确率当全部收益；merged/unmerged 推理对拍 |

**交付**：SFT→偏好优化完整配方材料、监督/mask 与低秩数值参照、方法/状态对照、adapter 部署契约；待实现 `labs/L7/sft_adapter_contract.py`。

**反例与边界**：full/LoRA/QLoRA 的小实验不能给出一般质量排名；微调 checkpoint 与可独立部署模型可能需要不同文件。


<a id="c-7-6"></a>
## 7.6 RL 运行时与策略同步

**依赖**：7.5、5.1；单卡算法与 J 可先执行，分布式 rollout/learner 运行时接 6.2；恢复、精度、作业资源分别接 7.4/7.9/7.11。非文本 RL 的目标与输入分别依 4.10/10.1/10.4 基础任务，不阻塞文本/小系统分析。

**问题**：奖励怎样成为一次合法策略更新；rollout、learner、reference、critic/reward 如何交换状态；不同模型的动作和概率怎样进入 RL 系统。

**对象与源码**：[Tülu 3 全流程](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)、veRL 的 `verl/trainer/main_ppo.py`、`ppo/ray_trainer.py`、`ppo/core_algos.py`；[slime](https://github.com/THUDM/slime)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 作架构对照；[Flow-GRPO](https://github.com/yifan123/flow_grpo)、[DanceGRPO](https://github.com/XueZeyue/DanceGRPO)、[CosyVoice2 GRPO](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2) 提供非文本实际流程。教学 RL 分支按 J 完整运行；生产规模流程与三框架差异用公开材料和源码研究。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 MDP/trajectory、state/action、policy、return、baseline、advantage 讲到 policy gradient、PPO ratio/clip、value loss、GAE、entropy、KL；对照 GRPO 组内基线与 DPO 离线目标 | 用两组各 4 条合成轨迹逐项手算并和源码核对；区分 old/behavior/current/reference、token/sequence 平均、零方差组及截断边界；不是只解释 rollout 速度 |
| B | 重建 prompt/偏好数据→SFT→RM 或可验证奖励→RLVR 的链；研读 Tülu 数据混合与模型阶段、reward model 的 pairwise 训练/评估、规则奖励的答案提取；说明 RLHF/RLAIF/RLVR 的差别 | 输出奖励组成、范围、长度/格式影响、无效答案处理和测试隔离；RM 不以偏好训练集正确率代替独立质量；公开未给出的奖励标注细节不补造 |
| C | 实现 CPU 小策略的 rollout→reward→advantage→update→weight sync，记录 prompt/trajectory IDs、逐 token/transition logprob、mask、奖励和策略版本；仅一至两次更新 | 新增 `labs/L7/mini_rl_iteration.py` 与 `rl_objective_reference.py`；能重算一次 loss/梯度并发现错分母、错 mask、过期版本；不以 toy reward 变化判断真实任务收益 |
| D | 研读 veRL actor/rollout/ref/critic/reward 的 worker、Ray placement、FSDP/Megatron learner、采样引擎、参数同步和 checkpoint；比较 colocated/disaggregated、sleep/offload 与 KV 回收 | 给出进程/设备/状态归属和完整迭代关键路径；时间计入 rollout、评分、logprob、更新、同步及等待；不把生成 tok/s 当训练 goodput |
| E | 对同一序列区分采样温度/top-p 后的行为分布与 learner 原始 logprob；分析 rollout correction、importance sampling、异步 lag、partial rollout、逐 token 行为版本与缓存身份 | 对现有或小模型单批 logprob 检查；CPU 队列注入 lag=0/1/2、慢 reward 和重复返回；不能用统一 episode version 掩盖中途权重切换 |
| F | 对 veRL、slime、OpenRLHF 比较 rollout 插件、reward 接口、参数发布协议、checkpoint/restart 与资源切换；追踪一次超时/worker 失败如何影响样本去重和下一次更新 | 交付三份源码时序及小型协议模拟，取消默认多框架真实 RL 对跑；发行版本、未实现接口和可复用作者日志逐项标明 |
| G | 对 Flow-GRPO 的 `scripts/train_sd3.py`、`config/grpo.py`、`sd3_sde_with_logprob.py` 重建 ODE→SDE 探索、transition logprob、去噪轨迹、组内奖励和 KL；用 DanceGRPO 的视频配方补时空/视频奖励 | 连续动作的小高斯转移例验证概率/梯度；确定性 ODE 路径不能直接套 token CE；说明采样步数、训练步窗口和部署 solver 的不同含义 |
| H | 对 CosyVoice2 的 `run.sh`、`prepare_data.py`、`reward_tts.py` 追踪 speech token rollout→token2wav→SenseVoice→拼音错误奖励→veRL→FSDP 合并→原格式；核对去掉 lm_head bias 的转换 | 给出音频奖励服务成本、样本与时长、reward 代理误差和独立语音评测；该配方只更新指定 speech-token 模型，不宣称训练了整个声学/vocoder 管线 |
| I | 为 VLM/Omni/世界模型/VLA 分别列状态、动作、奖励来源、可训练模块和公开 RL 覆盖；结合独立评测分析 reward hacking、长度偏置、过优化、KL 漂移及灾难性遗忘 | 学习材料包含作者曲线/失败样本/检查点的确切来源；缺少官方完整 RL 配方的型号标明缺口，使用已公开同类流程说明机制；不把离线动作回归或视频评分 SFT 写成 RL |
| J | 在教学 SFT 模型上选择可验证的小算术/格式任务，先测基础成功率，再运行一条 GRPO 或 PPO 路线的 rollout→reward→update→权重同步→评测→保存；对照无 RL 起点，记录组内零方差、无效输出和策略版本 | 在共同预算内完成多轮连续训练及导出，交付 reward/独立成功率/KL/长度曲线、每轮成本与奖励投机反例；模板、精度或同步错误能定位；训练奖励上升不替代留出任务验收 |

**交付**：完整后训练阶段材料、算法数值参照、worker/权重/数据协议图、异步与恢复小模拟、图像/视频/语音 RL 流程及质量诊断表。

**反例与边界**：优化奖励不保证任务质量提高；PPO ratio、相对 reference 的 KL 和 rollout correction 各有不同分母；策略陈旧度也包含精度、模板和后端变化。


<a id="c-7-7"></a>
## 7.7 蒸馏训练系统：教师信号、学生模型与产物

**依赖**：7.0b、7.5；草稿部署、量化和连续生成分别衔接 5.5、4.3、10.1/10.2，语音衔接 4.10。

**问题**：教师究竟提供什么监督；在线/离线蒸馏如何组织数据与计算；学生产物怎样保持与目标任务和推理实现一致。

**对象与源码**：TRL 所链接版本的 [DistillationTrainer](https://github.com/huggingface/trl/blob/cd2c52876d99b00baa5660328310e940b8360c6c/trl/trainer/distillation_trainer.py) 与 `trl/experimental/gkd/`；SpecForge 对应版本的 `scripts/prepare_hidden_states.py`、`specforge/training/`、`specforge/export/`；[ModelOpt QAD](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat)、[LCM](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)、[DMD2](https://github.com/tianweiy/DMD2)、[ZipVoice 两阶段蒸馏](https://github.com/k2-fsa/ZipVoice/blob/master/egs/zipvoice/run_emilia.sh)。教学学生在 J 训练，生产级草稿/图像/语音学生以真实公开流程与机制参照学习。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 建立 hard-label/sequence KD、logit KL/JSD、feature 对齐、draft 训练、consistency/progressive distillation、distribution matching、QAD 的比较；逐类注明 teacher/student 初始化、输入分布、监督位置、冻结模块和部署目标 | 一张方法表加每类梯度图；相同“蒸馏”名称不能掩盖不同训练问题；EAGLE3/DFlash 的 teacher 特征条件与简单隐藏层 MSE 蒸馏分开解释 |
| B | 小词表手写带温度的 forward/reverse KL、JSD 和 CE，固定 support/mask/reduction；比较真实标签、teacher 生成序列、student 自生成序列三个数据来源。对教学缓存单独验证完整词表、保留原概率加尾桶、top-k 内重归一化三种目标，教师和学生使用同一温度 | 完善 `labs/L7/distillation_objectives.py` 与 `teaching_distill.py` 的数值对拍；teacher stop-gradient、归约与 T² 缩放明确。尾桶应保留真实剩余质量；相同分布、支持划分和归约下，合并类别的 KL 不得大于完整 KL（容差内），k=V 应还原完整目标 |
| C | 研读 TRL 的 Qwen2.5-1.5B-Instruct teacher→Qwen2.5-0.5B-Instruct student 与 prompt 数据；追踪生成、teacher scoring、chunked vocabulary projection、loss、optimizer、save | 只抽数据/配置和一层等价数值；核对 tokenizer/词表语义和 logits 索引，跨词表不能直接逐位置 KL；解释 DistillationTrainer 与 experimental GKD 的 API/采样策略差异及 lm_head adapter 限制 |
| D | 设计 teacher feature/logit store，记录 sample ID、模板/token IDs、teacher revision、提取层、mask、dtype、量化与输入处理；概率缓存另外固定温度、原始归一化和尾部质量，改变温度时重新生成或使用足以还原目标的记录；对比离线预生成、在线共置、分离 producer/consumer | 完善 `labs/L7/teacher_store_contract.py` 的小记录缓存；检查特征过期、样本重放、未 ACK 队尾和恢复。完整与压缩字节使用同一位置数、词表和 dtype，包含索引、offset、读写与教师生产成本；不下载整套特征库 |
| E | 从 SpecForge 真实 EAGLE3/DFlash 配方追踪 dataset→prepare_hidden_states→draft config→训练策略→验证→checkpoint；分析三层 teacher feature、training-time test、词表裁剪/映射、DFlash block mask 与逐位置 loss | 使用其 Qwen3-8B 配方作训练材料，5.5 的 Qwen3-4B/DFlash 已发布权重作部署材料并注明不是同一次训练；交付两种 objective 的小张量/mask 参照和接受长度评估条件 |
| F | 研读当前 SpecForge online 的 SGLang capture server、Mooncake feature refs 和 data-parallel consumer；核对 runtime resume、训练状态与 serving export | `specforge export --to sglang` 只按当前 EAGLE3 key 契约分析，DFlash 等使用 `--to hf`；检查 fc/norm/lm_head/t2d/d2t、目标 embedding 引用、训练前缀残留；不沿用已删除的旧 train_dflash.py 接口 |
| G | 对 SDXL LCM 跟踪 teacher DDIM step、student LoRA、边界条件、guidance embedding/采样、teacher 冻结和 loss；对 DMD2 跟踪 generator、real/fake score、critic/对抗分支与交替更新 | 一维/二维解析场验证一次目标与梯度流；输出 teacher/student/EMA/critic 的状态账；区分减少采样 NFE、模型压缩和仅更换 solver；不运行完整蒸馏 |
| H | 对 ZipVoice 研读 prepare→base train→checkpoint averaging→distill first/second→ONNX→语音评测；对 ModelOpt 比较 QAT 与 teacher 指导的 QAD | 解释时间采样、teacher checkpoint 接续、可训练模块、平均权重和部署配置；语音蒸馏不能只写“F5-TTS 少跑几步”，量化恢复也不能与领域适配质量混为一谈 |
| I | 设计学生评估：语言质量/遗忘、草稿接受长度与完整推理时间、量化任务退化、图像条件一致性、语音可懂度/说话人相似；分开 teacher 生产成本、learner 成本和部署收益 | 用公开曲线/模型卡/样本建立证据表；模型/数据/设置不一致的结果不排名；缺少日志或原始数据的流程标出实际公开程度，不能用 toy loss 代替实模结论 |
| J | 选择相同词表的小教师/学生或明确的硬标签蒸馏分支，完成教师监督生产、缓存、学生连续训练、独立评测和导出；对照等预算直接监督训练，复用教学数据的隔离规则。缓存分支先通过 B 的目标检查，再在新目录重复同一子集的在线/缓存对照 | 交付教师信号样例、teacher/student 配置、曲线、任务质量、学生推理成本，以及含教师开销的总训练成本；已有错误目标下的曲线保留并按实际目标命名，不能用其 63.6% KL 差或约 0.27 nats CE 差解释纯尾部合并代价。未做预算/容量消融时不确定归因学生质量缺口 |

**交付**：九类监督/状态比较、四类实际 teacher/student 系统材料、KD 数值参照、feature store 协议、训练→导出→部署契约和评估规则。

**反例与边界**：teacher 更大不保证对学生更适合；top-k logits 截断改变目标；特征缓存的模型、模板和位置身份必须一致；公开权重不证明完整蒸馏训练记录公开。


<a id="c-7-8"></a>
## 7.8 训练数据工程与异构 batch

**依赖**：0.4、0.0b、1.5；图像/音频/动作细节连接 4.5/4.9/10.6，精确恢复接 7.4。

**问题**：原始数据怎样成为合法监督；数据组织怎样限制训练吞吐与重现性；不同模态和目标怎样共享数据基础设施。

**对象与源码**：[Datatrove](https://github.com/huggingface/datatrove)、[SmolLM3 数据与训练配置](https://github.com/huggingface/smollm/tree/main/text/pretraining)、[Tülu 3 数据流程](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)、Qwen3-VL/ASR/TTS 的 collator、[Cosmos JSONL](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/docs/dataset_jsonl.md)、openpi DROID；Arrow/Parquet、WebDataset/tar shard、tokenized bin/mmap 作为存储对照。概念验证使用小样本/元数据；I 准备贯穿项目预算内的实际训练数据，生产全库只读配置与统计。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从任务定义原始样本、监督来源、许可/版本、质量规则和独立拆分单元；覆盖文本 document、图文对、音频 utterance/speaker、video、trajectory/episode、偏好对和 teacher 生成记录 | 交付统一 manifest schema 及七类示例；明确哪些字段供模型、哪些供数据筛选、拆分和恢复，不能把所有输入强制改成纯文本 |
| B | 文本流程复原读取→语言/质量过滤→精确/近似去重→污染排查→tokenization→sharding→混合；研读 Datatrove 的 MinHash 多阶段与 SmolLM3 数据权重 | 用 30 条自制短文植入重复、近重复、乱码和跨 split 相似样本，保留过滤前后 ID；说明误删/漏删与验证集污染，公开数据比例不等于完整原始清洗流程 |
| C | 对图文/视频复原 caption、resize/crop、时间戳/帧采样和 latent/feature 缓存；对音频复原重采样、分段、转录/说话人、codec；对 DROID 复原动作坐标、idle filter 和 normalization statistics | 每类抽最多 8 条样本或轻量 metadata；记录增强随机性和缓存身份；训练集统计不能包含验证/测试，跨说话人/episode 泄漏有可检查反例 |
| D | 比较 padding、packing、length/aspect/duration bucket、token/frame budget 与动态 batch；实现 document boundary、cu_seqlens/position 和 response/action mask 的小型 collator | 新增 `labs/L7/training_data_contract.py`；输出字段/shape/有效元素数/样本归属；目标不变才比较处理量，跨文档 attention 与 loss mask 分开检查 |
| E | 解释混合权重按文档/样本/token/时长计的差别，有放回采样、重复 epoch、课程/阶段混合和 distributed sampler 尾部；分析 global batch 与 token 预算 | 用固定 RNG 对 3 个小数据源生成采样序列，核对期望与实际占比；展示相同“50%”配置因单位不同产生的偏差 |
| F | 对 Parquet/Arrow、tar shard、预分词 mmap 比较随机读/顺序读、解压、seek、远程读取、cache hit、worker/prefetch/pinned memory、供数背压与 straggler | 用轻量文件/注入读取延迟建立供数队列参照，连接 7.1 时间线；不能只测已全驻留缓存却归因存储带宽；不制造大数据文件 |
| G | 处理坏样本、长尾/超长样本、加载失败、重复样本、worker 退出、恢复后重读；记录 consumed/acknowledged、过滤规则版本和可重放样本清单 | 与 7.4 的 checkpoint 游标关联；每次跳过均有原因和计数，失败样本不会无声改变每 rank 有效 batch 或造成 collective 次序不一致 |
| H | 数据评估覆盖覆盖率、域/语言/长度分布、标签/偏好一致性、teacher 数据多样性、奖励可计算性、视觉/音频对齐与 train-test 污染；建立版本变更影响表 | 交付“改一条数据规则影响哪些 loss/评测/cache/checkpoint”的练习；开放数据缺少内容或许可细节时记录事实边界，不复制大规模生产语料 |
| I | 为文本、视觉适配和图像生成项目准备实际数据：来源/许可→清洗/去重→按文档/图源等独立单元拆分→tokenizer/processor→分片/collator→可恢复消费；文本分别核算预训练有效 token 与 SFT 有效回答 token | 交付项目可运行数据准备命令、划分规则、原始到监督的样例、数据量/长度/丢弃统计与磁盘预算；训练规则和归一化仅由训练集决定，留出集不参与配方选择；小样本探针不能代替实际数据产物 |

**交付**：全流程数据图、七类样本 schema、数据处理/采样/collator 小实现、版本与泄漏检查、供数性能材料。

**反例与边界**：高 tokens/s 可能来自重复/易样本或错误截断；固定 seed 不保证跨 worker/world size 样本顺序相同；多模态预计算缓存会改变可用数据增强。


<a id="c-7-9"></a>
## 7.9 优化器、混合精度与训练数值稳定性

**依赖**：7.0b、4.2、2.4；分片 optimizer/global norm 与 7.2 联动，量化感知训练与 4.3/7.5/7.7 联动。

**问题**：一次更新的数值和状态字节如何形成；混合精度的各个精度分别由谁决定；数值错误怎样传播并被定位。

**对象与源码**：PyTorch AdamW、autocast/GradScaler、FSDP2 MixedPrecisionPolicy；[torchao 训练](https://docs.pytorch.org/ao/stable/workflows/training.html) 的前向/反向三类 GEMM；[Transformer Engine FP8/FP4](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)；TorchTitan `torchtitan/quantization/float8.py` / `float8.md`；SmolLM3 optimizer YAML、F5-TTS EMA 和 Cosmos 配置。CPU 参照与既有 FP16/BF16 工件用于机制；I 使用教学项目的训练窗口分析稳定性，硬件专属低精度路径按支持条件选择。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 推导 SGD/momentum/Adam/AdamW 的状态、bias correction、eps、decoupled weight decay；解释参数组、embedding/norm 的衰减策略、clip、warmup、cosine/WSD、按 step/token 调度及 EMA | 一维/二维 FP64 两次更新与 PyTorch 对拍，打印 m/v、step、LR、更新量；学习率调度不是扩散采样 scheduler；不要凭调小 loss 选择优化器 |
| B | 为 FP32、autocast BF16、FP16＋scaler、显式 BF16 参数分别列 master/compute weight、activation、saved tensor、grad、reduction、m/v、optimizer update 与通信 dtype | 新增 `labs/L7/training_precision_ledger.py`；用实际 tensor.dtype 和 optimizer state 核对字节。FP32 master 与 FP32 m/v 按具体 recipe 检查，不能当所有 PyTorch BF16 模式的默认事实 |
| C | 分解 autocast 的算子选择、累加精度、TF32 matmul 与 FP16/BF16 表示范围；手算更新小于参数 ULP、softmax/归一化/CE 的误差来源 | FP64 参照含小更新、极值、长归约和近零值；建立前向、dX、dW、参数更新四处误差表；TF32 不称为一种权重存储格式 |
| D | 分解 GradScaler scale→backward→unscale→finite 检查→clip→step/skip→scale update，分析梯度累积和多 rank 的一致跳步；关联已有 FP16 inf | 新增 `labs/L7/loss_scaling_reference.py` 或扩展原 lab；错误 clip 顺序、每 microbatch 更新 scale、scheduler 在 skip 时前进等反例均可定位；BF16 通常不用 scaler 不代表不会 overflow/NaN |
| E | 从线性层 Y、dX、dW 推导 FP8 E4M3/E5M2、tensor/row/block scaling、amax、current/delayed scaling、量化/转置/scale 传输及高精度累加；分析 FSDP FP8 all-gather 和梯度归约的区别 | CPU 实现格式/scale 参照并记录饱和、下溢和转置方向；读 TE/torchao/TorchTitan 对应源码；仅在已有支持环境且有未解决机制问题时补一个小 GEMM，不开启整模型 FP8 训练 |
| F | 研读 MXFP8 的 32 元素块/E8M0 scale、NVFP4 的 16 元素块/E4M3 scale＋全 tensor FP32 scale；追踪 TE 权重 16×16 缩放、梯度随机舍入、dW 路径的 Hadamard 变换及末层高精度保留 | 交付“格式→缩放粒度→三类 GEMM→累加→额外状态→支持条件”表；逐 recipe 核查架构/对齐/版本，不因 GPU 有 FP4 推理能力就推断支持任意 FP4 训练路径 |
| G | 对全参、LoRA、QAT、QAD、teacher/student、flow/codec 多损失建立数值风险表；解释哪些层/损失保留高精度、teacher logits 的存储、VAE/音频重建极值和 EMA 更新 | 每类都连接真实配置和 failure counterexample；不把 quantization-aware fake quant 与加速反向 GEMM 的混合精度混为一谈 |
| H | 设计逐层 finite/activation norm/grad norm/update-to-weight ratio/scale/skip-step/有效 batch 记录，逐步缩小到数据、前向、反向、归约或 optimizer | 用一个可重放小样本注入 NaN/Inf/下溢；保存首个坏张量而非只报最后 loss；长期 loss spike、收敛与精度退化用公开日志/消融学习，不以两步正确性替代 |
| I | 沿教学训练曲线记录 grad/update norm、有效 token、finite/skip、LR 和峰值；在同一 checkpoint 后的限定窗口比较一种精度或优化器改动；从 Puro-Megatron 研读 MuonH/AdamW 路由、FP8 GEMM 与持久高精度状态 | 交付教学模型的诊断案例与 Puro 参数组/精度/状态映射；区别局部数值、完整学习质量和规模吞吐；窗口对拍不宣称复现 Puro 收敛，硬件不支持时保留源码和数值参照 |

**交付**：优化器与 dtype 字节账、小数值实现、FP8/FP4 recipe 源码材料、混合精度诊断流程和按模型的风险/高精度保留表。

**反例与边界**：低精度格式、权重存储、乘法输入、累加、梯度通信和 optimizer 精度是不同维度；激活重算/累积/FSDP 单 rank 不能凭空消除常驻优化器状态。


<a id="c-7-10"></a>
## 7.10 多模态与生成模型训练编排

**依赖**：7.5、4.6；视觉适配 J 先执行，连续生成分支接 10.1，其余分支按 4.9/4.10/4.11/10.4/10.6 的输入与目标逐项接入；不要求所有大模型同时加载。

**问题**：多个基础模型怎样组成训练阶段；不同目标和模态怎样共享优化器与并行系统；冻结、缓存和阶段迁移怎样保持语义。

**对象与源码**：OpenCLIP、Qwen3-VL `qwenvl/train/train_qwen.py` / `data/data_processor.py`、Qwen3-ASR/TTS finetuning、F5-TTS `model/cfm.py` / `model/trainer.py`、Diffusers 与 DiffSynth、Cosmos `vision_sft_edge.toml` / `videophy2_sft_edge.toml`、ms-swift Qwen3-Omni、openpi。各自的预训练、微调、蒸馏、RL 资料按[流程矩阵](#training-workflow)连接。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 为视觉 encoder、音频 encoder、codec/VAE/vocoder、文本 backbone、connector、denoiser、Talker/action head 分别列初始化、独立预训练目标、联合阶段和部署依赖 | 模块×阶段矩阵含输入、目标、更新对象、公开数据/配置/产物；视觉/音频/连续生成基座有自己的完整流程，不以 LLM 流程替代 |
| B | 比较图文对比学习、VLM answer CE、ASR CTC/CE、codec 重建/对抗、TTS codec CE/flow、视频 velocity、动作 flow/AR 的 mask 与分母；分析多 loss 权重和各自有效样本单位 | 用小型统一训练壳计算至少三种不同目标，打印每个分支的有效数和梯度贡献；不同模态梯度不能仅因为拼成同 batch 就平均正确 |
| C | 实现冻结 encoder＋connector、LoRA denoiser、teacher/student 小模型，核对 requires_grad、no_grad、optimizer param groups、不同学习率、解冻和新 optimizer state | 新增 `labs/L7/multimodal_train_contract.py`，机制例先做一次更新，J 完成视觉适配训练；冻结但需传递输入梯度、共享模块被重复加到 optimizer、adapter 漏存有反例 |
| D | 对多分辨率图像、变长视频/音频、不同 action horizon 设计 bucket/packing、microbatch、梯度累积与每 rank 模态分配；跟踪未使用分支与 collective 顺序 | 复用 7.8 collator 和两 rank 小例；比较每 token/frame/sample 的目标权重，检查纯文本 rank/音频 rank 不同分支可能引起的通信问题 |
| E | 分析预计算 text embedding、image feature、VAE latent、audio code、teacher hidden state 的节约与约束；记录模型/processor/增强/crop/时间采样/随机 posterior 的缓存键 | 用小缓存失效实例检验训练增强或参数解冻后必须重算；给出 CPU/GPU/存储预算与任务链，冻结参数不是缓存可复用的充分条件 |
| F | 深入 Cosmos3-Edge 两条配方：生成 `task=vfm`、gen 相关 optimizer key、flow loss、EMA、packing、FSDP；Reasoner `task=vlm`、SigLIP2 冻结、projector/LM、评分文本 | 列出各 loss 实际 reach 的参数；官方“full SFT”按选中的生成分支解释，不能写成所有模态模块一起更新；DCP/HF/Diffusers 的产物依赖接 10.4 |
| G | 对 Qwen3-Omni 追踪样本进 Thinker、Talker、Code2Wav 的标签/梯度/优化器；对照技术报告的各阶段与 ms-swift 实际可运行 recipe | 产出三模块开放程度表。音视频输入＋文本 loss 只证明该监督路径；没有公开的 Talker/Code2Wav 全训练配方保留为资料缺口，不把支持推理视为支持联合训练 |
| H | 对 Qwen3-TTS 的 talker loss＋sub-talker loss、speaker embedding 导出、F5 EMA/flow/vocoder、openpi normalizer/action head 建立阶段迁移契约 | 解释数据/权重/processor/统计量的联动，检查重新归一化、sample rate、码本、horizon 改动对输出接口的影响；不执行整套语音/机器人训练 |
| I | 汇合模型选择、独立验证、遗忘/模态偏置、alignment drift、过拟合、蒸馏和 RL 反馈；为各模态指定质量指标与固定验证输入 | 交付一组跨阶段失败定位练习：视觉仍编码但答案 mask 错、语音 codec 对不上、flow prediction_type 错、动作 normalizer 错；知识材料和小验证均有来源 |
| J | 使用匹配的教学 LLM、冻结视觉编码器和随机 projector，运行图文对齐与联合 SFT 两阶段；逐阶段检查 loss、mask、可训练参数、恢复与导出；用 MiniMind-V 官方实现和 Qwen3-VL 对照解释结构差异 | 交付两阶段曲线与权重、独立图像任务/文字能力评测、正确/打乱/移除图像对照和部署样例；明确所有预训练组件来源及全模型显存；达到共同质量/预算标准才验收完整视觉适配项目 |

**交付**：多基座阶段矩阵、多 loss/冻结/缓存/异构 batch 小实现、三条真实复杂流程（Cosmos、Omni、语音或 VLA）的源码与产物材料。

**反例与边界**：能够处理所有模态不代表训练了所有模块；不同训练 loss 的数值不可直接作为模态任务重要性；公开组件相加不自动构成原作者完整联合训练配方。


<a id="c-7-11"></a>
## 7.11 训练作业调度、性能诊断与规模决策

**依赖**：7.2、7.3、7.4、7.9；异步后训练接 7.6/7.7，多模态接 7.10，集群通用组件连接 8.2。

**问题**：训练怎样作为可恢复作业运行；吞吐和故障如何影响有效进度；有限资源下如何选择配置与定位瓶颈。

**对象与源码**：[torchrun/Elastic](https://docs.pytorch.org/docs/stable/elastic/run.html)、TorchTitan `torchtitan/observability/profiler.py` 与[debugging](https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md)、Nanotron launch/slurm、[Kueue JobSet](https://kueue.sigs.k8s.io/docs/tasks/run/jobsets/)、veRL/slime 的资源编排。系统实验用本机小进程/离散事件，GPU 性能优先复用已有材料，不搭建真实训练集群。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从一份 job spec 列出数据/模型版本、镜像/依赖、节点/rank/world size、rendezvous、输出/缓存路径、恢复点、重试策略和最终状态；对比 torchrun、Slurm 与 Kueue/JobSet 的职责 | 新增 `labs/L7/training_job_manifest.py`；本地模拟一次待调度→运行→失败→恢复→完成的状态流，检查 rank/env 和非法配置；不提交真实大作业 |
| B | 解释 gang/admission、队列配额、抢占、拓扑、局部性、资源碎片与 elastic world size；建立 GPU-time、等待、重启损失和 goodput 的预算 | 用 2/4 卡请求与小事件表比较 FIFO/整组准入/抢占；模拟结果注明假设，不把推理 replica 扩缩容规则直接用于同步训练 |
| C | 分解 cold start、权重/数据准备、编译、steady microstep、optimizer update、eval、checkpoint、失败重跑；按有效 token/图像/音频秒/动作/更新报告进度 | 按明确的共同 step 起止与时钟条件重算时间线；活动区间并集与完整 step 时长分开，不能用最长局部事件跨度替代全局窗口。模型计算、padding、重算和 teacher/critic 工作量分别说明；MFU/HFU 的工作量、设备与时间口径一致，条件不足时不输出数值 |
| D | 对 input-bound、launch/小算子、GEMM、attention、通信、optimizer、checkpoint I/O 和 straggler 分别列观测信号、应采 trace/计数器和验证对照 | 完善 `labs/L7/training_trace_analysis.py` 读取轻量事件；以错位的 rank 活动、等待与重叠事件检验时间窗口；每类瓶颈给公开或既有案例，只有具体问题需要时才补小 trace |
| E | 分析 activation checkpoint、CPU offload、梯度累积、FSDP/ZeRO、TP/PP/CP/EP、低精度、融合 optimizer 对峰值/通信/速度的取舍 | 给可计算资源账与合法配置候选；常驻状态装不下时说明原因，不能凭梯度累积/单 rank FSDP 宣称容量解决；不以最大显存占用率作为优化目标 |
| F | 本地子进程注入慢 rank、异常退出、损坏 batch、checkpoint 未提交、reward/teacher 服务超时；记录 watchdog、重启、数据重放、样本 ACK 与清理 | 新增 `labs/L7/training_job_simulator.py`；终止仅限本次启动进程，恢复点合法且无重复计数；torchrun 重启不自动提供业务数据/checkpoint 一致性 |
| G | 分析大型公开流程预算：SmolLM3 pretrain、Cosmos Edge SFT、SpecForge teacher/consumer、CosyVoice reward 服务；对照作者机器、时长、batch、日志/权重公开情况 | 交付资源与成本表和至少两个减少浪费的配置决策；教学项目使用实测成本，Puro-2B/SmolLM3 等规模预算单列来源与计费边界，不外推单卡运行时间 |
| H | 建立阶段验收：数值正确性、有效更新、数据消耗、恢复、质量、吞吐、成本各用哪类证据；比较一项新优化前先冻结目标和评价集 | 一份可复用的训练问题定位案例，能从现象追到数据/数值/并行/作业层并说明证据缺口；作者论文曲线与本机结果不混记 |
| I | 汇总各教学项目从数据准备、加载/编译、训练、评测、checkpoint 到部署的实际时间、峰值和存储；将 pilot 预测与实际预算对照，分析一个供数/计算/保存瓶颈并验证修正 | 每项目给完整成本与有效学习进度，包含 teacher/reward 和失败重跑；质量不降才能比较速度收益；对 Puro 的 GPU 小时、并行卡数、归一化租价与未计项目分别解释 |

**交付**：作业 manifest、CPU 调度/恢复模拟、训练 trace 分析器、资源账与四类生产流程的诊断材料。

**反例与边界**：硬件利用率高不代表有效学习进度高；rendezvous 成功不代表每 rank 数据/配置一致；重启次数少不等于质量或成本更优。


<a id="c-8-1"></a>
## 8.1 路由与网关

**依赖**：5.2、5.3、5.10、5.11。

**问题**：缓存亲和与排队均衡何时冲突；缓存目录怎样保持有效；难度路由怎样约束质量。

**对象与源码**：worldvln 两个 Qwen3-1.7B 副本；SGLang [cache_aware.rs](https://github.com/sgl-project/sglang/blob/c69844f0/sgl-model-gateway/src/policies/cache_aware.rs)，[llm-d 精确缓存路由](https://llm-d.ai/docs/well-lit-paths/foundations/precise-prefix-cache-routing)、Dynamo router；难度路由对照 Qwen3-1.7B/4B。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 RR、shortest-queue、prefix-aware 三种策略；记录每请求候选 worker、队列、预测命中与最终选择 | 相同 request/arrival 清单可重放；路由决定可以由输入状态重算 |
| B | 共享前缀比例=0/25%/75%，前缀长度=128/2048/4096，加入长短请求和热 adapter；比较真实 SGLang gateway 与 mini | 分别报告实际命中、省下的 prefill、queue、TTFT/TPOT 和 goodput；不将请求数当等量负载 |
| C | 订阅 KV events 构建目录，注入延迟、重复、丢事件与 worker 重启；比较精确事件与近似前缀索引 | 检查目录重建、过期信息和错误命中的处理；“精确”指实际事件协议支持范围，不保证实时无延迟 |
| D | GSM8K 固定题集上比较固定小模型、大模型和预算路由；冻结质量/SLO 门槛后统计成本 | 路由拒绝、升级模型和错误答案均计入；不以牺牲质量获得的低成本宣称更优 |

**交付**：新增 `labs/L8/request_router.py`、`routing_bench.py`，路由轨迹、缓存事件与质量—成本曲线。

**反例与边界**：prefix 命中多的副本可能排队更久；模型名/adapter 名相同不保证缓存身份一致。

<a id="c-8-2"></a>
## 8.2 编排与服务生命周期

**依赖**：6.0、8.1。

**问题**：GPU 资源如何变成可用 worker；就绪、排空和重启如何影响请求；不同编排器的职责边界在哪里。

**对象与源码**：Ray Serve 作为项目隔离环境的首个真实用例；Kubernetes + llm-d/vLLM production-stack、Slurm 作部署对照；模型 Qwen3-1.7B，服务加载与 readiness 复用 8.7。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写 worker 状态机：allocated→loading→warming→ready→draining→stopped；关联 GPU lease、进程树和请求归属 | readiness 由模型和依赖实际可用触发；进程存活不等于可服务 |
| B | 在独立 Ray 环境部署两副本，做扩缩容、滚动更新、worker 崩溃与排空；持续重放请求 | 原始请求不静默丢失，重试按幂等规则处理；记录恢复时间与空闲/碎片资源 |
| C | 在已有独立 K8s 测试环境映射 resource request、pod placement、readiness、termination grace 和 gateway；Slurm 映射 allocation/srun/step | 三套资源到进程映射均有源码/配置；真实运行需要相应隔离环境，无环境时保留该路径未完成 |
| D | 以 TP=2 worker 说明 gang scheduling 与部分资源分配失败，对比模型预热成本与扩容速度 | 不能把获得两块 GPU 的时间当模型就绪时间；不修改共享集群调度与系统包 |

**交付**：新增 `labs/L8/worker_lifecycle.py`、`orchestration/` 的独立配置与事件解析；真实编排与状态机模拟分别标明。

**反例与边界**：Ray/K8s/Slurm 不能仅凭启动参数作性能排名；MIG/MPS 等资源能力另按硬件和权限核对。

<a id="c-8-3"></a>
## 8.3 可观测性与正确压测

**依赖**：0.1、0.2；真实引擎事件接 5.11，音画事件接 4.11。

**问题**：如何定义完整时延；开环/闭环怎样影响排队观测；质量、SLO、错误与样本量如何共同决定 goodput。

**对象与源码**：Qwen3-1.7B、vLLM/SGLang bench 与 metrics；Python asyncio/httpx 客户端、服务端请求事件、CUDA profiler；基础发生器先于其它真实压测完成。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现可重放 open-loop/closed-loop 发生器，逐请求保存计划到达、实际发送、首字节、首有效输出、每个 token、完成/错误 | 用可控延迟的 toy server 检查 coordinated omission、客户端排队和缺失事件；所有请求进入分母 |
| B | 基线固定模型、输入和输出长度，扫描泊松到达率为可持续基线的 0.3/0.6/0.9/1.1 倍，加入突发与长短混合 | 每档 3 个至少 120 秒窗口，记录真实样本数；p99 样本不足时给区间而不称稳定 |
| C | 固定 TTFT=0.5/1/2 s、TPOT=25/50/100 ms 的教学 SLO 曲线，连同正确率/有效率统计 goodput | SLO 在策略比较前冻结；拒绝、取消、超时、截断都保留，不能仅统计成功请求 |
| D | 将同一事件 schema 接到语音首可播放、视频首有效帧与跨实例 KV 传输；校准时钟并核对时间段是否重叠 | 逐请求关键路径闭合；正常计时和 profiler 分开，端到端分位数从原始请求计算 |

**交付**：新增 `labs/L8/load_generator.py`、`request_metrics.py`，共同 case schema、重放器与统计器，为其它章节复用。

**反例与边界**：客户端接收时间包含服务器计算；不同事件的 p99 不能相加；QPS 不等于满足任务质量与 SLO 的 goodput。

<a id="c-8-4"></a>
## 8.4 混合负载与容量

**依赖**：8.1、8.3、9.1；音画与视频综合分别依赖 4.11、10.3。

**问题**：哪些资源被不同任务争用；单任务最优配置为何会失效；准入与隔离怎样形成容量边界。

**对象与源码**：第一组 Qwen3-1.7B + Qwen3-Embedding-0.6B；第二组文本 + Qwen3-TTS-0.6B；第三组文本 + Wan2.2-TI2V-5B，原生 Omni 在 worldvln 独立扩展。采用相应真实 runtime 的 scheduler/worker/queue。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 为每种任务单独建立 CPU、GPU、状态、传输与 SLO 基线，再核算并置的完整峰值 | 一次只比较能放入预算的组合；不把全部模型按参数量简单相加后启动 |
| B | 固定到达列表，混合比例=25/50/75%，比较独占、并置、时间分片与不同准入上限 | 每类任务分别报告质量、TTFT/RTF、尾延迟和拒绝，不能用总 token/s 掩盖某类退化 |
| C | 在确实出现退化的同一份额、到达清单和缓存状态下，逐项关闭 tokenizer 竞争、encoder batching、KV 驱逐或大 GEMM 干扰；采按 PID 区分的显存、引擎状态与同步时间线 | 份额 0.25 的异常不能用份额 0.5 的对照排除；整卡显存高水位不能直接归因两引擎预留或自身增长。未复现的 p99 变化保留为观察，正文和自测不确定归因 |
| D | 实现按任务预算的 admission controller，分列 offered 到达率、桶预算、接受率、完成 goodput 与任务质量；在固定清单比较准入后，再沿实际 offered 率扫描饱和边界 | 输出满足多类 SLO 的可复算容量边界；4/8 QPS offered 下的 4/16 QPS 桶配置不称已服务 4/16 QPS。HTTP 成功、SLO 达标与语义质量分别评分；时间分片保留相同到达请求并计入等待，不以删除一半负载换取隔离收益 |

**交付**：新增 `labs/L8/mixed_workload.py`、`admission_budget.py`，逐类资源与 SLO 曲线及准入策略。

**反例与边界**：相同均值长度可以有完全不同的长尾；并置节省空闲不保证总服务成本更低。

<a id="c-8-5"></a>
## 8.5 成本、能耗与硬件选择

**依赖**：8.3、0.2、1.3。

**问题**：单位有效输出成本如何定义；功率与能量怎样测量；容量、利用率与质量如何影响选型。

**对象与源码**：Qwen3-1.7B 的 crater/worldvln/spark 基线，端侧由 1.6 接入；NVML/tegrastats 与服务原始事件，价格来自执行时保存的供应商报价或明确假设。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 建立 capex 摊销、设备/主机功耗、利用率、吞吐和 SLO 的账本；每个价格给币种、时间、来源与计费单位 | 读数与价格假设分开；只测 GPU 时不能声称整机能耗 |
| B | 同模型同 token/质量，持续至少 60 秒采功率时间序列并积分，分别记录 idle 与 workload；扫描 batch=1/8/32 和到达率 | 总能量、增量能量、有效 token/任务、尾延迟与峰值内存可重算，采样分辨率写清楚 |
| C | 比较同 SLO 下每千有效输出成本与容量边界，添加成本/电价/利用率敏感性分析 | 推荐由真实负载和约束支持；量化或缩短输出后必须重新检查任务质量 |

**交付**：新增 `labs/L8/energy_cost_ledger.py`、功率原始采样、价格输入、公式与可复算选型表。

**反例与边界**：峰值 TFLOP/s 和 token/s 不足以决定成本；跨机能耗比较须对齐测量范围。

<a id="c-8-6"></a>
## 8.6 KV 存储层级

**依赖**：5.2、1.5、6.4。

**问题**：GPU/CPU/存储层各持有什么状态；取回与重算的交叉点在哪里；精确复用与近似保留怎样区分。

**对象与源码**：Qwen3-1.7B/8B；[LMCache](https://github.com/LMCache/LMCache)、[Mooncake](https://github.com/kvcache-ai/Mooncake)、[Dynamo LMCache 集成](https://docs.nvidia.com/dynamo/v-0-9-1/integrations/lm-cache)；H2O/SnapKV 作为语义不同的近似路线。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 GPU/CPU 两级 store，明确 model/adapter/position/quant/layout 身份、引用、传输完成和驱逐状态 | 取回后的 KV 与直接重算逐元素/逐 token 对齐；错误布局和旧 revision 拒绝 |
| B | prefix=128/2048/8192/32768、reuse gap=0/1/10/60 s、并发=1/8，分别测写入、取回、重算和完整请求 | 内存占用与时延同报，交叉点必须计入注册、传输、等待和维护成本 |
| C | 扩展到可用 NVMe 或跨实例 store，对取消、驱逐中取回、重复请求和服务重启注入错误 | 无 double-free、旧数据或失联句柄；故障后请求和缓存状态可恢复 |
| D | 与 H2O/SnapKV 的 token 选择比较，若研究 CacheBlend 则明确其重算/近似语义；固定 RULER/NFCorpus 任务评质量 | 精确存储复用与近似上下文压缩分别验收；任意上下文片段不能直接拼 KV 冒充等价 |

**交付**：新增 `labs/L8/tiered_kv_store.py`、`kv_retrieve_vs_recompute.py`，缓存状态、取回/重算曲线和质量结果。

**反例与边界**：更高命中率可能因搬运更慢而降低 goodput；磁盘存在数据不证明其版本和 layout 可用于当前请求。

<a id="c-8-7"></a>
## 8.7 冷启动与权重分发

**依赖**：1.5、4.0、8.3。

**问题**：从进程到首个有效请求经过哪些阶段；文件/JIT/图缓存怎样改变成本；扩容滞后如何进入容量预算。

**对象与源码**：Qwen3-1.7B/8B，vLLM/SGLang loader、model init、compile、graph capture/readiness；safetensors mmap 与 checkpoint shards。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 采进程启动、import、文件读取、反序列化、H2D、编译、捕获、ready 与首请求；记录每阶段峰值 | 所有必要工作包含在服务就绪账本；首请求触发的编译不能漏出计时 |
| B | 区分冷进程/热文件缓存、冷 JIT/热 JIT 与热图；用读取/缺页和缓存路径验证条件，比较分片数与并行加载 | 不在共享机器 drop_caches；换文件名不自动代表冷数据；无法证明冷态时如实标记 |
| C | 用 8.3 到达轨迹模拟或运行扩容，比较预热副本、按需加载与不同阈值；连接 8.2 readiness | 排队、失败、空闲资源和扩容延迟共同形成预算；权重下载与本地加载分列 |

**交付**：新增 `labs/L8/startup_timeline.py`、`scale_out_budget.py`，加载阶段原始事件、缓存状态与扩容曲线。

**反例与边界**：新进程不代表全冷启动；worker 已注册不代表模型已可用。

<a id="c-8-8"></a>
## 8.8 多租户与隔离

**依赖**：5.3、5.8、8.2。

**问题**：租户身份如何贯穿缓存和 adapter；共享资源怎样造成干扰；隔离机制各自提供哪些边界。

**对象与源码**：两个合成租户、Qwen3-1.7B 与 7.5 的版本化 LoRA；vLLM/SGLang admission、priority、preemption、adapter/cache manager；可用的进程/GPU 隔离机制。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 tenant/session/adapter revision 与预算，构造同名 adapter、相同文本不同私有前缀和跨租户取消；对每次权重函数变化分别检查常驻 adapter 权重和旧 KV 的失效，复用 5.10 的关缓存/换槽对照 | 输出、KV、adapter 和资源归属可检查；所有输入使用合成数据。热修补只要改变权重也受相同失效约束，不将 `load_inplace` 无条件推荐给同名热修补；按新版本参照检查 logits 和运行中请求 |
| B | 租户 A 固定交互负载，B 逐步增加长 prompt、并发或 adapter 换入；比较共享、配额、优先级和进程隔离 | 分租户统计质量、p95/p99、拒绝、抢占与资源；不能用平均吞吐掩盖交互服务退化 |
| C | 在 worker 崩溃、预算超限和服务更新后恢复，检查缓存身份、幂等请求与资源清理 | 合法租户在恢复后继续可用；缓存身份和权限控制在对应层验证 |
| D | 说明 MIG/MPS/进程隔离与应用配额的不同能力，实际支持的独立环境才做对照 | 硬件或权限不支持的方案只记录机制和条件，不通过修改共享驱动验证 |

**交付**：新增 `labs/L8/tenant_scheduler.py`、`tenant_interference.py`，隔离/干扰曲线与状态回收验证。

**反例与边界**：prompt 约束不是资源或数据隔离；独立进程也可能争用同一 GPU 带宽和显存。

<a id="agent-systems"></a>
## Agent 系统的共同任务与执行顺序

L9 以一项有状态任务为对象，贯穿控制面中的任务图、持久状态、预算与取消，以及数据面中的模型请求、工具执行、检索和 KV。每章都要让读者能修改一种机制，并用任务完成时间、成功率、资源占用和失败恢复判断收益。框架只作为源码载体；ReAct、planner、multi-agent 等组织方式用于产生串行、分支、汇合与循环负载，不单独写成框架用法或提示词教程。

| 章节 | 核心系统对象 | 与已有层的分工 |
|---|---|---|
| 9.1 任务图与压测 | 动态 DAG、因果事件、到达过程与任务评分 | 复用 8.3 的请求计时和发生器，补任务级依赖与闭环 |
| 9.2 工具协议 | parser、MCP 会话、传输、能力与结果回传 | 5.6 负责约束解码，5.11 负责模型 API；本章接到工具服务 |
| 9.3 上下文与 KV | 消息、token、分支状态、驻留与失效 | 5.2 负责引擎块管理，8.6 负责存储传输；本章决定跨轮状态的生命周期 |
| 9.4 推理预算 | reasoning/final 预算、候选分支与验证器 | 5.5 负责 token 级投机；本章比较完整候选任务的资源与质量 |
| 9.5 持久执行 | 任务状态、checkpoint、租约、提交与取消 | 5.8 负责引擎请求失败；本章负责多步任务和外部副作用的恢复 |
| 9.6 检索与长期记忆 | 文档、向量、索引、版本、可见性与删除 | 5.12 负责 embedding/rerank 服务；本章负责数据生命周期和完整流水 |
| 9.7 工作流调度 | 任务优先级、关键路径、联合准入与 SLO | 5.3 负责引擎 iteration 调度，8.1 负责副本路由；本章协调任务的全部阶段 |
| 9.8 工具执行环境 | 进程树、容器、工作区、快照与有界池 | 8.2/8.7 负责模型 worker 与权重；本章负责 CPU 工具环境的准备、隔离和回收 |

**执行顺序与依赖**：先完成 9.1-A/B 的事件和重放基础，再推进 9.2 与 9.3；9.2 后推进 9.5，9.5 后推进 9.8。9.4 复用 9.1 的计时与任务分母，9.6 在 5.12 后建立数据路径。9.7-A/B 在 9.1/9.5 基础上先完成调度反例和单引擎接入，9.7-C/D/E 随缓存、预算、检索与沙箱接口就绪逐项集成。CPU/远端 KV、跨实例、音画和 KVM 路线按各自前置回填，基础任务不等待整条扩展路线。

**共同交付与验收**：逐章完善同一个可观察的最小 Agent 服务，复用现有 `labs/L9/` 脚本和已能支持结论的工件。固定任务、工具输入、模型/模板与评分；根任务按绝对到达时间进入，后续节点由真实依赖触发。系统策略比较用同一任务清单与到达过程，模型推理或上下文策略改变轨迹时另跑闭环质量。任务 JCT、SLO 内成功任务数/秒、最长等待、资源高水位和每成功任务成本是主指标，请求 TTFT、缓存命中和 token/s 用来解释它们。流式 chunk、模型 token、可见答案和工具参数的时间定义分别说明。

**源码与实验边界**：下列来源分别承担机制任务，不能用列框架名或转述作者加速比代替实现分析。论文按对应算法版本阅读；SDK、协议与源码使用相容版本、普通版本链接和真实文件/符号定位。现行引擎基线与研究 fork 分开安装和测量；模拟只验证策略与不变量，完整性能必须来自实际模型与工具执行。工具、索引、框架依赖均在项目隔离环境准备；本地临时文件与缓存遵循外置盘约定。各章结束按章节规范完成推导、源码、mini、实验、原始材料、设计取舍和反例。

<a id="c-9-1"></a>
## 9.1 任务图、请求轨迹与 Agent 压测

**依赖**：5.1、5.2、5.3、5.11；真实到达率实验先完成 8.3 的发生器与计时基础，音画扩展接 4.11。

**问题**：怎样从模型与工具事件恢复动态任务图和关键路径；固定轨迹重放与闭环任务执行各能回答什么；请求指标怎样对应任务成功、完成时间与服务容量。

**对象与源码**：复用 `agent_tasks.py`、`agent_trace_collect.py`、`trace_replay.py` 和 `mini_prefix_accounting.py`；Qwen3-4B、vLLM 0.29.0，SGLang 0.5.19 做同模型基础对照。负载保留三类各 100 个教学任务，并接 [BFCL 多轮任务与评分器](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md) 的 `multi_turn_base` 固定 50 题（`bfcl-eval==2025.12.17`）；事件映射到 [OpenTelemetry GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，该规范仍为 Development。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 把 session、task、node、parent、attempt、tool_call 与 engine request ID 串起来；记录逻辑到达、准入、入队、首 reasoning/final/tool delta、模型结束、工具开始/结束、任务终止。读取原始 SSE 并核对版本字段；只有请求级时间时不把 TTFT 当 prefill，也不把 e2e−TTFT 当 GPU decode。 | 给出一条完整串行任务和一条 fork/join 任务的事件图；并行区间不能简单求和，取消、超时、缺少 delta 的任务均进入分母；采集与重放使用相同的首输出定义。 |
| B | 实现依赖驱动的重放：根任务按绝对时间到达，后继必须等父节点与工具完成；工具时延按原轨迹或明确的注入分布重放。分别保留请求多重集合不变的次序扰动、整会话 bootstrap 和独立抽长度三种对照；为到达器提供独立于客户端并发槽的时钟。 | 对拍节点数、依赖、输入 token 与工具结果；bootstrap 另报工作量，不能声称与原清单边际完全相同。注入慢工具/队列拥塞后仍能记录所有逻辑到达；请求级重放不能评分为新策略下的闭环成功率。 |
| C | 采教学负载与 BFCL 多轮固定子集的轮次、fan-out、工具时延、输入/输出、前缀共享和终止原因联合分布；原评分器与任务环境状态对拍；先核对同一题在两引擎的模板、支持模式与实际请求。 | 交付实际模型与工具轨迹、公开题目 ID、原评分和完整失败样本。教学代码修复、BFCL 的模拟工具环境和生产流量分别命名；源码支持与本机运行分开。 |
| D | 按 ARRIVAL 协议在固定任务到达率下重新执行任务，同时做受控重放；测任务 JCT、SLO 内成功任务数/秒、每任务请求数、队列/工具/模型时间、峰值 KV 与 CPU 内存。音画事件另接 4.11 的真实会话。 | 区分模型服务 goodput 与任务 goodput；无引擎阶段事件时瓶颈归因保留待验证。音画模拟只检验事件逻辑，真实音画与跨实例结论各自验收。 |

**交付**：完善现有采集与重放脚本，交付任务图、原始 SSE/JSONL、请求与工具结果、两种执行模式的对照；待实现 `labs/L9/agent_trace_spans.py`。这些事件与任务输入供 9.2–9.8 共用。

**反例与边界**：TTFT 缺失的工具调用不能从样本中消失；相同 session 标签不保证数据依赖或 KV 相同。长度、缓存命中与阶段耗时的相关性不替代因果对照；未定位的前缀预测偏差保持未解释。

<a id="c-9-2"></a>
## 9.2 工具协议与调用数据路径

**依赖**：5.6、5.11、9.1；约束与草稿的综合对照接 5.5。

**问题**：工具调用从模型 token 到服务端执行经过哪些边界；能力、schema 与传输状态怎样改变调用语义和成本；断连、取消和重试何时会造成重复执行。

**对象与源码**：复用 `tool_pipeline.py`、`incremental_tool_parser.py`；vLLM 0.29.0 的 `tool_parsers/hermes_tool_parser.py` 与 chat completion/structured-output 路径，SGLang 0.5.19 对应 parser。协议固定为 [MCP 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)，源码用 [python-sdk](https://github.com/modelcontextprotocol/python-sdk) 的 `src/mcp/client/session.py`、`client/streamable_http.py`、`server/streamable_http.py`；任务扩展只按该版本的实验性能力比较。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 逐模式追踪 none/auto/required/named 的模板、parser、约束选择、mask 与真实工具调用；对两引擎保存支持与拒绝路径。工具数=1/4/16/64，分别测 schema 渲染、tokenization、未命中 prefill 和命中前缀；构造完整工具集与按需发现的对照。 | 语法有效、参数符合 schema、工具选择和工具执行成功分别评分；不能由返回 JSON 推断启用了 grammar，也不能把重复 schema 的输入 token 全算作每轮重算。按需发现同时报告漏选与额外往返。 |
| B | 保留增量解析对拍，覆盖多调用交错、转义、Unicode、截断与超长参数；payload=256 B/4 KiB/64 KiB，固定分片后测累计扫描量、CPU 时间与缓冲峰值。仅在参数闭合、schema 与调用授权完成后提交工具。 | delta 拼接与完整解析一致，已取消/不完整的参数不会执行；区分扫描复杂度与具体语言的墙钟。约束/草稿只作该段优化，DFlash 接受率不能代替完整工具调用收益。 |
| C | 实现同一个计算/检索工具的直接调用、stdio MCP、Streamable HTTP MCP 三条路径；跟踪 initialize/能力协商、tools/list、tools/call、JSON 与 SSE、连接复用、背压和 result 序列化。并发=1/8/32，额外注入的工具延迟=0/10/1000 ms，先单轴扫描。 | 用相同参数与返回值对拍；报告客户端排队、传输、服务执行、序列化和下一轮 prefill 的完整时间，定位 CPU/网络瓶颈；冷连接和已复用连接分开。 |
| D | 对照 [取消规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation) 与 [Tasks 扩展](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)：注入断连、重复回包、晚到结果、schema 变更与过期会话；检查 HTTP 授权上下文与工具 allowlist。对任务扩展先协商 capability，再测轮询、结果取得和 tasks/cancel。 | 断连、流恢复与重新执行分清；MCP session ID、JSON-RPC request ID 和业务幂等键各有归属。普通取消通知不保证副作用回滚，Tasks 为实验性且必须报告 SDK 支持范围；持久提交与补偿交给 9.5。 |

**交付**：完善现有两个脚本；待实现 `labs/L9/mcp_tool_path.py` 和 `mcp_fault_cases.py`，交付请求/响应片段、能力协商、schema、传输与完整调用时间线。

**反例与边界**：MCP 传输协议、模型工具格式与业务执行协议不是同一层；工具注解或合法 JSON 不授予执行权限。客户端重连和 SSE 重放都不能自动提供业务 exactly-once。

<a id="c-9-3"></a>
## 9.3 上下文与会话 KV 生命周期

**依赖**：5.2、5.10、9.1；真实多副本接 8.1，CPU/远端存取接 8.6；单卡生命周期实验可先完成。

**问题**：消息历史、上下文变换与引擎 KV 分别由谁持有；工具等待时保留、换出或重算的盈亏条件是什么；分支和迁移怎样同时保持语义、缓存身份与资源边界。

**对象与源码**：复用 `session_kv_bench.py`、`mini_prefix_accounting.py`；vLLM `v1/core/kv_cache_manager.py`、`block_pool.py` 与 SGLang radix cache；[InferCept](https://proceedings.mlr.press/v235/abhyankar24a.html)/[源码](https://github.com/WukLab/InferCept) 的 `vllm/core/scheduler.py`，以及 [Continuum](https://arxiv.org/html/2511.02230)/[源码](https://github.com/Hanchenli/vllm-continuum) 的 `vllm/v1/core/sched/scheduler.py`；远端取回复用 8.6 的 LMCache 路径。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 建立消息、token、位置、模型/adapter 版本、KV 块的状态表；复用 2/4/8 轮的追加/中段插入/thinking 裁剪，逐轮定位共同前缀。补分支 fork、写时复制和分支结束后的引用计数；用重复运行区分浮点波动与错误缓存复用。 | 记录真正输入引擎的 token 与已重算区间；中段改写保留改变点之前的前缀。注册 ID 与显示名称分别核对，不能从两份同权重 adapter 推断同名热更新安全；恢复消息状态不依赖 KV 仍在。 |
| B | 推导保留的字节×等待时间、换出/取回字节与时间、重算时间及对其他任务排队的影响。以工具间隔=0/1/10/60 s、上下文约 512/4096/8192 token 与背景压力为轴，比较默认容量驱逐、重算和有界驻留；从实测工具延迟分布选择 TTL。 | 给出 InferCept 的 discard/preserve/swap 成本及 Continuum 的 TTL/重入队取舍；预测工具时间注入 ±50% 偏差和长尾后仍有容量上界。TTL 在此指保留/钉住的期限，不等同于有效 KV 必须按时删除；高命中率未必改善任务 JCT。 |
| C | 比较完整历史、保留最近轮次、模型摘要与工具结果外置后按需读取；固定检索/代码任务和评分器，计入摘要调用、取回 I/O、失去的前缀及下一轮重算。 | 上下文压缩作为会改变模型输入的近似策略单独评分；输出正确性、任务成功和省下的资源共同报告，不能只比 prompt 长度。长期数据与检索索引的写入/删除归 9.6。 |
| D | 用两台真实 worker 对照固定副本、迁移后重算、CPU/远端取回；覆盖同副本恢复、新副本预热、第三轮迁移后在新副本继续，以及 adapter/位置/精度不兼容。 | 每轮目标 worker、缓存所在层、读取字节与输出有记录；迁移不必永久失去后续复用。清空同一引擎只作为空缓存对照，不能验收跨实例迁移；缺少存储接口的档位保持 UNVERIFIED。 |

**交付**：完善会话 KV 脚本；待实现 `labs/L9/session_residency_policy.py` 与 `context_compaction.py`，交付状态归属、生命周期事件、成本模型、真实取回与质量对照。

**反例与边界**：精确前缀复用的数学条件与浮点实现的逐 token 一致性分开验证；等待时间本身、缓存压力和保留策略分别控制。

<a id="c-9-4"></a>
## 9.4 推理预算与分支执行成本

**依赖**：5.1、3.3、5.13、9.1；分支 KV 共用接 9.3，能耗口径接 8.5。

**问题**：总输出、reasoning 与最终答案预算怎样转换成资源约束；多候选与验证器何时比单次长生成更合算；预算耗尽和早停怎样影响共享资源与任务成功。

**对象与源码**：复用 `reasoning_budget.py` 与 GSM8K/MATH-500 各 128 题，Qwen3-4B 与 Qwen3.5-4B；vLLM/SGLang reasoning parser、停止规则、采样与调度源码。多候选用同一目标模型，先比较多数投票，再接可检查的算术/代码测试验证器；不把判分器的正确答案暴露给预算策略。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 核对原始 SSE 的 reasoning/final 字段与 usage token；分开模型首输出、首可见答案、完成时间和服务端排队/执行时间。重算评分，分别列出截断、无答案、错误答案和请求失败；批次都完成 64 题时逐题核对正确数的差别。 | 客户端分片数不当 token 数；没有完整 token 时间戳时只报告 chunk 间隔。max_tokens 明确为总生成上限，题目、模板和采样设置的变化分别记录；阶段瓶颈需引擎事件支持。 |
| B | 保留 128/512/2048/8192 总预算与 thinking 开关曲线；实现有独立答案余量的预算控制器。区分引擎原生 reasoning budget、停止后追加引导并续写、直接截断三种实际支持的方式，计入额外请求和重复 prefill。 | 分别统计期限内给出 final、超时和无答案的任务；不支持的字段不得当成生效。预算控制、完成质量、状态增长和停止后的资源回收均可检查，阈值只用独立校准题选择。 |
| C | 在同一组题和总资源上限下比较 1 次长生成与 2/4 个候选，分别串行和有界并行；记录共享前缀、候选等待、验证器调用、选中答案、被取消分支与释放时间。 | 给出实际 token/GPU 时间而不只给预算上限；投票、验证和重试成本全部计入。早停后未选分支的队列/KV 能回收，失败样本保留；结构上的并行不保证任务更快。 |
| D | 沿上下文长度与并发单轴复用状态扫描，在纯 attention 与混合架构中读取真实层/状态大小；对长短预算混合任务测任务完成时间和每成功任务成本，给 9.7 提供预算策略输入。 | 区分批次墙钟/正确数与单任务延迟，固定题目分母并报告独立题目不确定性；功耗积分有完整时间窗口。架构、kernel、预热和调度差异不合并为单一因果结论。 |

**交付**：完善 `labs/L9/reasoning_budget.py`；待实现 `labs/L9/reasoning_budget_controller.py` 与 `candidate_executor.py`，交付预算状态、候选 DAG、取消时间线与质量/完整成本曲线。

**反例与边界**：增加候选、延长思考或提高吞吐均不自动提高准确率；token 预算相同也不保证实际工作量相同。单次有限样本的质量差不能直接归因于题目难度或并发。

<a id="c-9-5"></a>
## 9.5 持久执行、恢复与取消

**依赖**：5.8、5.11、9.1、9.2；音画打断与播放 epoch 接 4.11。

**问题**：哪些状态必须在进程故障后保留，哪些可以重算；重放、重试、幂等和补偿分别保证什么；取消与 deadline 怎样跨模型、工具和分支传播。

**对象与源码**：完善现有 `agent_runtime.py`、`agent_failure_matrix.py`，扩展 9.1 harness。研读 [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)、[Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api) 及 [源码](https://github.com/langchain-ai/langgraph)：`libs/langgraph/langgraph/pregel/_loop.py`、`_retry.py`、`libs/checkpoint/langgraph/checkpoint/base/__init__.py` 和 SQLite saver；[Temporal 的 Workflow/Activity 边界](https://docs.temporal.io/develop/python/integrations/langgraph) 作为不同持久执行模型的对照，veRL AgentLoop 仅在 7.6 对照训练 rollout 生命周期。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 把 session/turn/tool 状态机扩成持久任务图，记录稳定 IDs、输入输出、版本和取消 epoch；完善 `durable_task_store.py`。租约读取、token 递增与取得放在同一事务或条件更新中，提交同时检查 worker、token 和 expires_at；依赖与重复完成检查进入提交边界 | 停止进程后恢复同一待执行集合；覆盖两个 worker 同时竞争、租约到期但尚无接管者、旧 worker 延迟提交和显式续租。成功接管的 token 严格递增；随机模型调用保留结果或明确重推理，日志写出不等于可恢复 |
| B | 用项目内事务账本实现带唯一幂等键的工具，将副作用与去重结果放进同一提交边界；另用独立工具服务演示提交后 ACK 丢失。业务键包含租户/会话/节点的必要作用域并在重试间稳定，不复用全局 `key-fetch` 等常量；同键不同请求内容应拒绝 | 同会话重放返回原结果，不同会话的同名节点各自执行；副作用完成但 runtime 未提交的故障可恢复。区分库内副作用与外部动作：SQLite 事务不能撤销已经发出的外部副作用，补偿不是通用回滚 |
| C | 实现任务总 deadline、子调用剩余预算、有界重试和退避；将取消传给在途引擎请求、工具进程与 fork/join 子任务，并用 epoch 拒绝晚到回包。中断等待和用户恢复作为状态迁移处理。 | 在排队、流式生成、工具运行、工具已提交未 ACK 四处取消；检查终态、未释放资源与恢复结果。客户端断开不等于执行停止，每轮重新分配全额 timeout 不算总 deadline。 |
| D | 以新输出目录运行进程级故障矩阵，覆盖 kill/restart、租约过期、checkpoint 失败、重复/乱序完成、版本不兼容；从退出码及结构化结果判定，不依靠 stdout 最后一行。对比 LangGraph 的节点重放/待提交写入与自写运行时。 | 正常任务与失败恢复共享同一验收断言；事件可恢复不等于副作用只执行一次。完成结果、合法后继、账本唯一性、取消不复活和资源回收各自有证据；音画场景按 4.11 另验收。 |

**交付**：完善现有运行时、`labs/L9/durable_task_store.py`、`idempotent_tool_service.py` 与故障矩阵，交付状态迁移、事务/租约时序、恢复前后结果与故障证据；正文和自测准确区分 `os._exit(137)` 主动退出与外部 SIGKILL 注入，退出码相同不代表故障方式相同。

**反例与边界**：checkpoint 保存的是应用状态，不承诺 GPU KV 或外部工具快照持久化；本章负责消息历史与工具调用的提交和恢复，9.6 负责长期数据，9.8 负责执行环境快照。

<a id="c-9-6"></a>
## 9.6 检索与长期记忆的数据系统

**依赖**：5.12、9.1；上下文/KV 语义接 9.3，跨实例 KV 存取接 8.6。

**问题**：文本、向量、索引与 KV 怎样保持可解释的一致性；固定召回和生成质量后索引与流水如何选择；长期记忆的更新、过期和删除怎样影响任务结果。

**对象与源码**：完善 `rag_pipeline.py`、`rag_kv_probe.py`；Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B、Qwen3-4B；[FAISS](https://github.com/facebookresearch/faiss) 的 `IndexFlat`、`IndexHNSW`、`IndexIVF` 与 [度量约定](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)，[BEIR 示例](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_faiss_dense.py)。NFCorpus 评检索；HotpotQA distractor/validation 固定 200 题评带答案和支持事实的生成；[CacheBlend](https://arxiv.org/html/2405.16444) 与 [LMCache blending](https://docs.lmcache.ai/kv_cache_optimizations/blending.html) 只作为近似片段 KV 路线。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现文档/片段 ID、语料与 embedding 版本、chunk/query 指令、归一化和距离函数的映射；用 FlatIP 作精确参照，逐查询核对向量 top-k、qrels 和 rerank 结果。NFCorpus 与 HotpotQA 的语料、评分分开。 | 检索相关性、ANN 相对精确 top-k 的 Recall@k 和生成答案质量分别计分；HotpotQA 小语料子集不冒充全库检索。脚本先通过编译/小输入和异步调用验证，再接 GPU 服务。 |
| B | 对 HNSW efSearch=16/32/64/128、IVF 合法 nlist/nprobe 测固定 Recall@k≥0.95 下的构建、查询和更新；统一 inner product/归一化条件。10 万/100 万合成向量只测规模；查询集与训练/索引样本分开。 | 给出序列化索引大小和实际 RSS，向量字节公式不代替图结构/倒排表内存；新写入、删除和过滤后的精确参照可重算，不用较低召回换无条件的加速结论。 |
| C | 打通 query embedding→检索→rerank→上下文构造→模型首输出/完成，候选 top-k=5/20/100，固定最终上下文 token 上限；为各段建立有界队列与 batch 接口，单轴扫描并发和 rerank 批大小。 | 同时记录在线各阶段、完整任务时间、失败、EM/F1 和支持事实/引用有效性；离线建库和模型加载单列。仅生成 gen_ms 不作为 RAG 端到端时间，rerank 与排队不能遗漏。 |
| D | 实现带 namespace、version、有效时间和 tombstone 的长期记忆存储；用项目内合成事实做新增、纠正、删除、并发读写与重建，定义 read-after-write 或 eventual consistency；同步推进文本存储、索引与结果缓存失效。 | 在约定可见性边界之后的新请求不能检索/引用已删除或越权的测试事实，进行中请求记录读取版本，延迟更新有明确窗口；记忆系统的 TTL 是业务有效期，区别于 9.3 的 KV 驻留期限。不能把向量库检索等同于持久执行 checkpoint。 |
| E | 以原始上下文→变更上下文的成对请求测试重复、不同前缀、位置/顺序变化；先做无复用与精确前缀参照，再在真实接口下做片段 KV 拼接和选择性重算。 | 精确路径核对 token/位置/状态与数值容差，近似路径另测质量和完整成本；每个变体独自 reset 后重复自身不能证明跨变体复用。CPU/远端取回与 CacheBlend 各按实际接口验收。 |

**交付**：修复并完善已有 RAG 脚本；待实现 `labs/L9/ann_quality_bench.py` 与 `memory_store_lifecycle.py`，交付语料/索引版本配置、质量与成本曲线、更新删除事件及 KV 对照。

**反例与边界**：高检索召回不保证答案正确；向量、文本、检索结果与 KV 的身份和失效条件各自明确。访问控制和实际版本检查在服务层完成，不能靠提示词隔离数据。

<a id="c-9-7"></a>
## 9.7 工作流调度与任务级 SLO

**依赖**：5.3、8.3、9.1、9.5；缓存与预算接 9.3/9.4，检索和沙箱接 9.6/9.8，多副本、混部与隔离接 8.1/8.4/8.6/8.8。

**问题**：为什么单次请求更快仍可能使整个任务更慢；动态分支未知时怎样用已观测进度调度并防止饥饿；模型、工具和缓存资源怎样联合准入而不互相堵塞。

**对象与源码**：[Parrot（OSDI 2024）](https://www.microsoft.com/en-us/research/publication/parrot-efficient-serving-of-llm-based-applications-with-semantic-variable/)/[ParrotServe](https://github.com/microsoft/ParrotServe) 的 `parrot/serve/graph/semantic_variable.py`、`parrot/serve/scheduler/global_scheduler.py`；[Autellix](https://arxiv.org/html/2502.13965) 的 PLAS、ATLAS、process table 与 MLFQ/防饥饿算法；[Continuum](https://arxiv.org/html/2511.02230) 的 program-FCFS 与保留策略。真实基座复用 vLLM 0.29.0 `v1/core/sched/scheduler.py`、SGLang 0.5.19 scheduler 与 8.1 网关；Parrot 为未持续维护的研究原型，Autellix 论文实现基于 vLLM 0.6.1，不能当现行引擎开箱即用能力。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用串行链、宽度 2/4 的 fork/join、长短任务混合和数据依赖循环构造可手算反例；比较按请求 FCFS/轮转与 program-FCFS、PLAS 和 ATLAS 规则。任务图逐步揭示，记录已获服务、最大已观测路径、等待、队列级别与资源占用。 | CPU 事件模拟对拍每次调度和最终完成次序；解释请求平均时延、任务 JCT、makespan 与 slowdown 的不同。未来总轮数和真实剩余时间只作离线参照，不可泄露给在线策略。 |
| B | 实现 process table、ready 队列、任务级优先级、aging、deadline 与有界准入；接当前引擎测服务入口排队，并在固定版本可支持的 scheduler 扩展点验证执行中抢占与恢复。 | 请求准入排序和 token/iteration 级抢占分别交付；无引擎插桩的网关排序不称为完整 Autellix 复现。保留相同的请求级调度基线，记录抢占、KV 重算/换出及策略自身 CPU 开销。 |
| C | 把 GPU 活跃 token/KV 预算、工具 worker/沙箱槽、检索连接池与候选分支预算放进准入；比较仅 GPU 队列限流与分阶段配额/联合背压，工具时间含慢返回与突发；用已观测状态做路由和局部保留策略。 | 不能持有大量无用 KV 等工具槽而使其他任务饿死；已取消分支释放配额，重试不绕过任务预算。输出 CPU/GPU/队列时间线与各资源高水位，验证策略在哪个瓶颈下无收益。 |
| D | 共享 9.1 根任务到达清单与质量评分，在 0.3/0.6/0.9/1.1 倍任务基线到达率下按 ARRIVAL 运行；固定模型/缓存/硬件，每档 3 个至少 120 秒窗口，策略交错运行；长窗口与短正确性测试分开。 | 主表报告 SLO 内成功任务数/秒、JCT p50/p95、最长等待、成功率、超时/拒绝和每成功任务成本；同时保留请求 TTFT/吞吐，不能把作者加速比或 CPU 模拟当本机结果。少于 10000 样本不宣称 p99 稳定。 |
| E | 两真实 worker 上比较任务亲和、最短队列与按缓存/队列联合路由；注入工具长尾、服务时间估计 ±50% 偏差、worker 失效和租户突发；接 9.5 恢复与 9.8 沙箱回收。 | 既有任务能恢复且不会重复提交副作用；亲和增加排队、保留缓存挤占新任务等负结果也须解释。按各集成接口实际可用范围验收，单卡结论不外推到跨机集群。 |

**交付**：待实现 `labs/L9/workflow_scheduler.py`、`workflow_loadgen.py` 与 `workflow_serving_bench.py`；交付调度反例、真实引擎接入、策略决策与任务完成时间线、资源竞争和故障结果。

**反例与边界**：多 agent 在本章表现为任务图分支与共享资源需求，不用角色数量代表系统能力；推理策略改变任务图时，固定轨迹系统实验与重新执行的质量实验分开。

<a id="c-9-8"></a>
## 9.8 工具沙箱与执行环境池

**依赖**：1.4、1.5、9.2、9.5；节点生命周期、镜像分发和租户资源接 8.2/8.7/8.8。

**问题**：目录、进程、容器、gVisor 与 microVM 分别隔离什么；冷启动、预热池与快照如何影响启动时间和常驻成本；取消、崩溃与复用后怎样证明进程和任务状态已回收。

**对象与源码**：[OpenHands DockerWorkspace](https://docs.openhands.dev/sdk/guides/agent-server/docker-sandbox) 及 [software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) 的 `openhands-workspace/openhands/workspace/docker/workspace.py`；[gVisor Sentry 架构](https://gvisor.dev/docs/architecture_guide/intro/)、[checkpoint/restore](https://gvisor.dev/docs/user_guide/checkpoint_restore/) 与 [Firecracker snapshot](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md)；Linux namespaces/cgroup v2、进程组、文件描述符与 mount 生命周期。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 沿工具请求→执行服务→进程/容器→退出结果追踪 PID/进程组、文件、网络与身份；用同一无害命令和受控代码任务比较子进程与容器。剖析 gVisor syscall 路径、microVM 边界及实现成本。 | 给出各层状态归属和隔离能力表；任务目录不等于代码沙箱，容器共享宿主内核也不等于 microVM。实际运行与仅源码分析分别标注，不能假定当前 GPU 容器支持嵌套 Docker 或 KVM。 |
| B | 实现 NEW→STARTING→READY→BUSY→DRAINING→DESTROYED 状态机与有界池；预热槽=0/1/4、并发=1/4/16，先单轴测镜像已存在时的进程启动，再单列镜像取得、依赖装载和工作区准备。 | 计时从资源申请到首个工具可执行结果；给出启动/排队 p50/p95、闲置 RSS/磁盘、命中/溢出与任务完成时间。池中 READY 必须经过工具健康探测，进程存活不代表可执行。 |
| C | 在自有隔离环境给工具设置 CPU、内存、进程数、输出缓冲与 wall-time 上限；注入受限的大输出、内存超额、子进程残留和工具卡住，取消时传播到整个进程树；回收 workspace、挂载、句柄与租约。 | 同池其他任务仍能完成；记录退出原因与资源释放证据，stdout 截断不等于进程终止；并发任务不得看到彼此工作区和测试用凭据。资源限制和访问边界都用受控用例验证。 |
| D | 用工作区快照/恢复对照全新创建，分析 gVisor checkpoint 与 Firecracker 的 guest state、内存文件和 MAP_PRIVATE 写时复制；可运行平台才做对应系统实验。复用环境时执行状态重置，再与 9.5 的业务恢复结果连接。 | 文件恢复、进程恢复、外部副作用恢复分别验收；快照恢复测首次触页和完整工具时间，不只量启动 API。缺 KVM/runsc 权限时完成源码与生命周期小验证，对原生快照性能保持 UNVERIFIED。 |

**交付**：待实现 `labs/L9/tool_sandbox_pool.py`、`sandbox_lifecycle_bench.py` 与 `sandbox_failure_cases.py`；交付环境配置、状态机、工具可用时间、资源/回收记录和隔离边界。

**反例与边界**：本章的 CPU 工具环境与 L8 的 GPU 模型 worker 分开计量；不把子进程执行或项目目录副本称为安全沙箱。预热节省等待但占用资源，快照也不能撤销外部已发生的副作用。

<a id="c-10-1"></a>
## 10.1 DDPM、score 与 rectified flow

**依赖**：4.2、7.0b；数据/数值机制接 7.8/7.9；本章建立连续生成模型自己的基础训练流程。

**问题**：数据与噪声怎样定义训练目标；预测参数化怎样进入采样；VAE/条件编码器/denoiser 分别怎样训练。

**对象与源码**：[DDPM](https://arxiv.org/abs/2006.11239)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)；[flow_matching 的 image 与二维例](https://github.com/facebookresearch/flow_matching/tree/main/examples)、[Diffusers unconditional training](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)、[LDM VAE 配置](https://github.com/CompVis/latent-diffusion/tree/main/configs/autoencoder)。数值主例为解析 Gaussian/二维 Gaussian mixture；G 完成预算内随机初始化小 UNet 的图像训练，解析参照与学习模型分别验收。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从数据/噪声、时间方向、插值路径推导 DDPM 条件分布、score、epsilon/x0/v 与 flow velocity；列出系数、边界和各参数化的转换 | 新增 `labs/L10/prediction_parameterization.py`；FP64 小例对拍，包括 t 接近端点、零噪声与符号反转；不同文献的 v 按定义区分 |
| B | 解析 Gaussian/GMM 的 score 或条件 velocity，固定初始噪声与时间网格，采样 8/32/128 步；输出向量场、逐步轨迹、NFE 和分布指标 | 新增 `labs/L10/toy_diffusion_flow.py`；解析场用于检查求解误差，不冒充已训练模型；条件插值直线不推出边缘向量场轨迹必直 |
| C | 用同一批小数据构造 denoising/flow 的训练 pair，比较 time/noise sampling、target、loss weighting、条件 dropout、有效维度归一化；小 MLP 最多一次更新以检查梯度 | 保存 x、noise、t、condition、target、mask 和逐项 loss；学习“训练一步”与“采样一步”的差别，不用单步误差或两个不同目标的 loss 值排名 |
| D | 研读完整基础训练：数据/增强→pixel 或 VAE latent→时间/噪声→UNet/DiT→目标与加权→optimizer/AMP/EMA→checkpoint→固定采样验证；将 DDPM 和 flow image 官方入口逐项对应 | 交付两份真实配置/代码材料，明确训练时间分布、SNR、sigma/flow shift、prediction_type、CFG 条件丢弃与推理配置；小型图像集与完整教学训练由 G 交付，大型基座配方按公开范围研读 |
| E | 分开分析 VAE 的 reconstruction/KL/perceptual/adversarial、条件编码器的表示训练、denoiser 的生成目标；解释 latent scale/shift、冻结模块、posterior 采样和预计算缓存 | 用 LDM `autoencoder_kl_32x32x4.yaml` 与 `ldm/models/autoencoder.py` 追到 loss/双优化器；架构/损失知识可迁移，不能把旧 VAE 配方说成 Wan/Cosmos 原始训练过程 |
| F | 从公开学习曲线/消融解释数据量、模型容量、噪声时间采样、EMA、数值精度和训练预算对质量的影响；建立过拟合、模式遗漏、数值不稳定与错配 scheduler 的诊断 | 公开证据、解析参照和本机单步分别记录；用独立分布/感知/条件指标评估，不以 FM 名字断言更少步或更好质量 |
| G | 按共同图像项目完成小图像集划分、小 UNet 随机初始化、DDPM 训练与 EMA、验证、checkpoint/恢复、导出和固定噪声采样；分析早期/中期/最终模型、推理步数和错配 prediction_type | 交付从数据到生成的完整命令、曲线、独立误差与多样性/记忆检查、阶段样本、NFE/延迟及资源成本；训练图像不能进入最终评测。flow 的同数据训练作为后续扩展，DDPM 完整项目为必做 |

**交付**：参数化与解析场小实现、完整 DDPM/flow 基础训练材料、VAE/条件编码器/denoiser 状态矩阵、学习曲线和失败案例的来源。

**反例与边界**：将 scheduler 换名不会改变模型学得的预测目标；训练时间采样、推理时间网格和 optimizer LR scheduler 是不同对象。


<a id="c-10-2"></a>
## 10.2 Solver、步数与蒸馏

**依赖**：10.1。 训练相关综合任务接 7.7/7.9，不依赖完整训练运行。

**问题**：一步包含多少模型求值；时间网格与预测类型怎样限制 solver；少步模型的改进来自什么变化。

**对象与源码**：Diffusers Euler/Heun/DPM-Solver++/LCMScheduler；真实基线 `stabilityai/stable-diffusion-xl-base-1.0`，少步对照 `latent-consistency/lcm-sdxl`，按 [LCM 官方用法](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm) 同时替换对应权重与 scheduler。 训练过程增加 Diffusers 的 [SDXL LCM distillation](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation) 与 [DMD2](https://github.com/tianweiy/DMD2)，按 7.7 共享的 teacher/student 系统材料执行。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写 Euler/Heun 与多步状态缓存，在解析 ODE 和 10.1 网络上对拍；打印时间/sigma、模型输入缩放、预测转换与更新 | 误差随步长的变化符合相应条件；历史状态、起步阶段与末步处理均可检查 |
| B | SDXL 固定 32 个中英文 prompt、每题 3 个初始 latent，合法 solver 下扫描 10/20/40 步、CFG=1/5/7.5 | 记录实际 denoiser 次数、CFG batch、编码/VAE 固定成本、输出和完整时间；区分步数与 NFE |
| C | LCM 以其匹配配置扫描 2/4/8 步，与 SDXL 基线分别报告质量/时间；对故意错配 prediction_type 与 scheduler 保存失败样本 | 蒸馏权重变化与 solver 变化分开；不能只换调度器就宣称得到蒸馏模型能力 |
| D | 对输出保留感知距离、条件一致性和固定顺序盲评，数值轨迹误差与语义质量分别解释 | 相同 seed 不足时直接复用 latent/每步噪声；所有配置和生成样本可重放 |
| E | 重建 LCM 的 teacher solver 目标、停止梯度分支、边界条件、guidance、full/LoRA 学生、保存与匹配 scheduler；对照 progressive/consistency 与 DMD2 的 real/fake score、critic 和交替更新 | 源码中的损失、时间采样、更新对象和导出产物逐项对应；用解析场/小张量检查一次目标，不执行蒸馏收敛；发布 checkpoint 与具体脚本范围匹配 |
| F | 以 Flow-GRPO 和 ZipVoice 为例说明 RL 后训练、少步蒸馏、CFG 蒸馏和 solver 数值加速的目标差异；把预训练/适配/蒸馏的成本与推理节约分开核算 | 复用 7.6/7.7 的公开流程；质量退化、reward 偏置和误差累积有具体反例，不能把“少步”统一解释成同一种训练 |

**交付**：新增 `labs/L10/solver_reference.py`、`solver_quality_bench.py`，scheduler 配置、求值轨迹、原始样本与成本曲线。

**反例与边界**：更高阶方法可能有更多 NFE；不能把高步数输出当客观真值或用图像相似度替代全部质量。

<a id="c-10-3"></a>
## 10.3 DiT pipeline 与扩散服务

**依赖**：10.2、5.3、5.8、3.1、2.0c。 训练相关综合任务接 7.5/7.9/7.10，不依赖完整训练运行。

**问题**：完整请求由哪些计算和状态组成；请求级与步级 batching 怎样不同；不同请求如何隔离 solver、latent 和 RNG。

**对象与源码**：共同主模型 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)，Diffusers `ZImagePipeline`/transformer、vLLM-Omni DiffusionEngine、[SGLang-Diffusion Z-Image](https://docs.sglang.io/cookbook/diffusion/Z-Image/Z-Image-Turbo)。该 Turbo checkpoint 的少步/无 CFG 路线与常规有 CFG 模型分开。 完整训练/微调参照读取 Diffusers [FLUX DreamBooth/LoRA](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md) 及 train_dreambooth_lora_flux.py；与本章部署的 Z-Image-Turbo 模型卡公开范围分开。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 跟踪 tokenizer/text encoder、latent 初始化、DiT、scheduler、VAE 与编码输出；每阶段记录 shape/stride、dtype、权重和临时状态 | 先复现官方推荐采样参数并记录实际 NFE，不能把参数中的 step 数直接当求值数 |
| B | 实现可单步推进的 runner，持有每请求 latent、sigma index、solver history、RNG、condition 与取消标记 | 串行、交错和独立运行在允许数值容差内对拍；请求不能共享可变 scheduler 状态 |
| C | 固定 prompt/latent，合法分辨率 512²/768²/1024²、batch=1/2/4、到达率按基线扫描；比较整请求串行、请求级 batching、步级插入 | 给出兼容键、等待、队头阻塞、峰值和完整完成时间；各引擎实际支持粒度逐项记录，外层 async 不冒充步调度 |
| D | 两运行时对照同一 checkpoint/精度/配置，加入取消、不同分辨率和可支持 LoRA 的身份检查；逐步接入 mini 的调度策略 | 输出有效性、任务质量、状态隔离和资源回收先验收；不支持功能保持明确未完成 |
| E | 重建图文数据→bucket/增强→VAE latent＋text embedding→时间/噪声/flow target→DiT LoRA 或全参→optimizer/AMP/EMA/checkpoint→验证图→pipeline/adapter 导出；列 frozen VAE/text encoder 与可训练模块 | 交付 FLUX.1-dev 的真实配方字段表和一次小 loss/梯度例；不把它称为 Z-Image-Turbo 原始训练，Turbo 原始蒸馏未公开的部分明确列出 |
| F | 比较只训 DiT、训练 text encoder/connector、LoRA、量化感知与 teacher 蒸馏的 activation/optimizer/缓存代价；检查服务端 adapter rank/命名、base revision 和 scheduler/latent scale | 复用 7.5/7.9/7.10 的契约与公开产物，解释训练决定了哪些部署状态；不要求每条路线都做模型训练 |

**交付**：新增 `labs/L10/diffusion_step_runner.py`、`diffusion_serving_bench.py`，两运行时配置、完整阶段与请求时间线。

**反例与边界**：DiT 不必在所有形状下算力受限；Turbo 模型不适合直接套用常规 CFG/高步数扫描。

<a id="c-10-4"></a>
## 10.4 视频生成与世界模型

**依赖**：10.3、3.2、3.4；分布式扩展依赖 6.5。 训练相关综合任务接 7.6/7.7/7.10，不依赖完整训练运行。

**问题**：视频怎样变成时空 token；世界模型的理解/生成通路共享什么；帧数、求值、缓存与并行怎样共同决定成本。

**对象与源码**：视频基础参照 [Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)，前沿主例明确使用 [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)。阅读 [8 月更新说明](https://huggingface.co/nvidia/Cosmos3-Edge/discussions/62)、[官方 cookbook](https://github.com/nvidia/cosmos)、Diffusers `Cosmos3OmniPipeline`/`Cosmos3OmniTransformer`/`UniPCMultistepScheduler`/`AutoencoderKLWan` 和 [vLLM-Omni pipeline_cosmos3](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3/)。 训练材料固定 [NVIDIA/cosmos-framework 2b6c9a7](https://github.com/NVIDIA/cosmos-framework/tree/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce)：docs/training.md、docs/dataset_jsonl.md、examples/launch_sft_vision_edge.sh、examples/toml/sft_config/vision_sft_edge.toml、videophy2_sft_edge.toml、model/generator/algorithm/loss/flow_matching.py；另读 [DiffSynth Wan2.2 full/LoRA](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 首先核对 ENVIRONMENTS 已缓存的 `6f58f6b4c91288838e60b6bcb2cc45d997e961de`；新版目标固定 `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`，分别核对代码兼容和配置。旧共享快照只读，新版需要时另存学习目录 | model_index/transformer/vae/scheduler/processor、权重 hash 和源码 pin 齐备；不能把旧权重配新版示例并视为同一实验 |
| B | 新版首先复现 I2V：832×480、121 帧、24 fps、BF16、20 denoising steps、guidance=6、flow_shift=12、use_karras_sigmas=False、seed=0，使用官方 example_i2v_input 与 JSON prompt/negative prompt | 实际 scheduler 参数和 NFE 写进工件。配置文件中的 flow_shift=1/use_karras_sigmas=True 必须按目标配方显式覆盖；不从旧文生图/文生视频示例推断新版 Edge 支持范围 |
| C | 从真实配置重建 48-channel latent、空间压缩 16、时间压缩 4、latent patch=2 与时空位置；分解一次理解侧计算、缓存和每步生成侧计算 | 打印实际 grid、有效帧数、padding/crop、KV/条件/latent/workspace；以真实 Edge backbone 配置为准，不套用其它 Cosmos 子型号结构 |
| D | 官方样例通过后，用 12 组许可明确的桌面操作/物体运动/室内场景，先固定 121 帧扫描 10/20/30 步，再固定 20 步测试经版本确认合法的 61/121/149 帧；分辨率扩展单列合法配置 | 生成文件帧数/fps/尺寸正确，保存阶段时间、峰值、NFE、条件遵循与物体/运动一致性；未见拐点也如实记录，不将合成超长 shape 当真实整模型 |
| E | 同 checkpoint/输入/latent 比较 Diffusers 与 vLLM-Omni 的完整 I2V，检查理解侧 KV/条件缓存是否按身份复用；Wan2.2 仅作架构与 pipeline 对照 | 本机结果与官方 H100 吞吐分开；跨模型比较同时报告任务与质量，不据参数量归因速度 |
| F | 接入模型实际支持的 action-conditioned forward/inverse dynamics 样例，读取具身维度、归一化和 mask；支持的并行路线在 6.5 后测 | 世界预测、动作条件生成、reasoner 和独立 Policy-DROID 各自标明能力；生成视频不作为闭环控制成功证据 |
| G | 为 Cosmos3-Edge 完整重建 BridgeData2-Subset-Synthetic-Captions（数据 revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01）→caption_json/视频 JSONL→Wan2.2 VAE→convert_model_to_dcp→vision_sft_edge；解释 task=vfm、BF16、packing token 上限 45056、梯度累积 2、clip 0.1、EMA 和 FSDP | 研读官方 500 iteration 配方但不运行；按 optimizer.keys_to_select 列 moe_gen/time_embedder/vae2llm/llm2vae/k_norm_und_for_gen 的真实更新范围；核对训练 base revision 与 A 的部署快照，不能默认互换 |
| H | 比较生成 SFT 和 VideoPhy-2 Reasoner SFT：输入/标签、SigLIP2/connector/LM、冻结策略、flow loss 与物理评分文本 loss；从 DCP 依次追 export_model 的 HF safetensors/vision tower/processor，再 convert_model_to_diffusers | 交付两条阶段/梯度/产物流程和 config 字段解释；Diffusers 转换输入是导出的 HF 目录，不能直接把原始 DCP 当 pipeline；具备代码与 SFT 数据不等于全部 Edge 预训练公开 |
| I | 对 Wan2.2-TI2V-5B 官方公开权重及 DiffSynth full/LoRA 配方追 metadata.csv、832×480/49 帧样例、input_image、flow 时间/目标、dit 参数组、LoRA q/k/v/o/ffn.0/ffn.2、remove_prefix 与加载 | 只做样本/配置/小张量检查；样例 resolution/帧数与本章 Cosmos I2V 配方分开；5B 与 A14B 高低噪专家子型号不混用，第三方训练实现不冒充 Wan 原始全量训练日志 |
| J | 读取 Cosmos DMD2RF 的 teacher/fake-score/student 与独立 optimizer/checkpointer，结合 Flow-GRPO/DanceGRPO 的公开视频 RL 说明轨迹、奖励与导出；核查 Edge 原始预训练/蒸馏/RL 材料的实际公开范围 | 列论文、代码、配方、数据、权重、日志六项证据；源码支持某方法不代表发布 Edge 权重由该公开配方训练；小生成/动作验证沿 A–F，不额外启动完整训练 |

**交付**：新增 `labs/L10/cosmos_edge_anatomy.py`、`cosmos_edge_i2v_bench.py`、`video_shape_ledger.py`，固定版本配置、官方样例复现、原始视频与时空/资源账。

**反例与边界**：sound_tokenizer 为空的配置不据 Cosmos 家族介绍宣称支持音频；动作内部 padding 维度不等于具身实际控制维度；帧数和尺寸必须经过所选版本合法性检查。

<a id="c-10-5"></a>
## 10.5 扩散缓存与近似复用

**依赖**：10.2、10.3、5.2；Cosmos 具体实验依赖 10.4。 训练相关综合任务接 7.8/7.10，不依赖完整训练运行。

**问题**：相同条件计算与相似跨步特征如何区分；复用误差怎样传播；维护、回退和质量何时抵消加速。

**对象与源码**：主例 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 TeaCache/Cache-DiT，另检查 Z-Image-Turbo 的支持路径；[SGLang 缓存文档](https://docs.sglang.io/docs/sglang-diffusion/caching-acceleration)、[Cache-DiT](https://github.com/vipshop/cache-dit)、[TeaCache](https://arxiv.org/abs/2411.14324)。Cosmos3-Edge 用于条件/KV 与跨步近似的结构比较。 训练缓存的对照复用 7.8/7.10 的 latent/encoder/teacher feature store 与真实训练配置。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写带 cache key、更新、失效、阈值和回退的 feature cache；保存每层每步输入/输出；区分确定条件缓存与近似残差复用 | 无缓存参照与精确复用对拍；近似误差不能按精确等价验收 |
| B | 固定初始 latent、solver、每步随机噪声与 prompt，阈值=0/0.05/0.1/0.2；8 题调参、24 题独立评测，比较 step/block 缓存与 TaylorSeer 类预测 | 记录真实 skipped layers/NFE、维护/搬运/回退、峰值、完整时间、质量和退化样例，不只报 hit rate |
| C | 查真实支持和 no-op：当前文档中 Wan2.2 TeaCache 系数未校准可能不生效，不能沿用 Wan2.1 结果；检查 CFG 双分支、FSDP 与 batching 兼容 | 每个开关都对应执行 trace 和改变的工作量；配置被接受不算机制运行 |
| D | 在 Cosmos3-Edge 中分离固定理解侧缓存与变化 latent 的跨步近似，按同一 I2V 配方测全请求收益与质量 | 解释误差对后续轨迹的累积；不同模型阈值不相互移植，组件误差小不保证世界预测正确 |
| E | 区分推理跨步近似缓存与训练预计算缓存：latent/条件特征的随机性、增强、冻结/解冻、teacher revision、梯度需求和数据重放；分析把推理近似引入训练会改变的目标/梯度 | 以小缓存失效和 detach 例检查边界；准确复用、近似前向、近似梯度分别说明，不因推理质量可接受就宣称训练无损 |

**交付**：新增 `labs/L10/feature_cache_reference.py`、`cache_quality_bench.py`，缓存事件、逐步误差、独立生成样本及质量—时间—内存曲线。

**反例与边界**：相似不是相同；复用率高也可能因维护/搬运更慢；少量样本不能给普遍安全阈值。

<a id="c-10-6"></a>
## 10.6 VLA 与动作生成

**依赖**：10.1、10.2、1.7。 训练相关综合任务接 7.6/7.10，不依赖完整训练运行。

**问题**：模型动作张量对应哪个具身和坐标；flow/AR 生成的系统成本怎样不同；采样和执行频率怎样影响质量与闭环。

**对象与源码**：主例 [openpi DROID](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md) 的 `pi05_droid`（checkpoint `gs://openpi-assets/checkpoints/pi05_droid`）；`pi0_fast_droid` 作 AR 动作 tokenizer 对照；前沿扩展 [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)，区别于通用世界模型 checkpoint。 训练来源增加 [openpi DROID 全流程](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)、src/openpi/training/config.py、scripts/compute_norm_stats.py、scripts/train.py、src/openpi/models/pi0.py。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 DROID 记录的图像、state、指令和动作，解析 normalizer、坐标、单位、horizon、padding 与 action mask；实现官方物理动作转换 | 单条观测逐张量对拍，模型内部维度与具身维度分开；各模型必须先映射到可比动作定义 |
| B | flow 头用原生设置为基线，合法配置扫描 NFE=4/8/16；AR 路线追踪 FAST tokenizer、动作 token 数与解码，固定观测和有效 horizon | 报完整视觉/语言编码、动作头、解码与动作误差；不只测 denoiser kernel |
| C | 保留模型原生输出 horizon，消费者每次执行前 1/4/8 个动作，频率=5/10/20 Hz；在 1.7 harness 注入延迟与旧输入 | 记录动作年龄、deadline、连续性、离线误差和有效动作率；消费 chunk 与模型生成 shape 不混用 |
| D | 为 Policy-DROID 运行独立合法动作样例，比较通用世界生成与策略 checkpoint 的训练目标/输出接口；具备对应机器人或可信任务仿真后再做闭环 | 离线结果、仿真结果和真实机器人结果分开；没有闭环条件不推断成功率 |
| E | 重建 pi05_full_droid_finetune 的 RLDS droid/1.0.1、idle filter、归一化、观测/action chunk、flow 目标、weight loader、optimizer/EMA、checkpoint 与 serve_policy；对比 pi05_droid_finetune 的自定义 LeRobot 数据 | 只读取小 episode/公开统计与配置，不下载 1.8 TB 全库或运行 8×H100 训练；说明两种数据路径和动作空间差异，过滤索引/统计量必须匹配数据版本 |
| F | 比较 pi0-FAST 的 token/AR 监督与 pi05 flow action head，解释 LoRA/全参、冻结感知模块、action padding/mask、horizon、部署 normalizer；连接 Cosmos 动作条件生成和 Policy-DROID 的不同训练目标 | 一条小动作序列对拍归一化、mask 和 loss；公开 pi05 实现的 flow head 范围单列；不能将机器人离线模仿学习或世界视频预测写成在线 RL |
| G | 分析机器人数据覆盖、行为克隆分布漂移、offline/online RL 的状态/动作/奖励与仿真需求、训练后量化/蒸馏的动作误差；追踪部署时延与训练目标的关系 | 交付独立离线/仿真/真实机器人评价方案和公开范围表，复用 7.6/7.7 机制；没有交互环境不声称完成策略 RL 或闭环成功 |

**交付**：新增 `labs/L10/action_head_probe.py`、`action_chunk_runtime.py`，具身定义、动作轨迹、误差、持续时延和条件具备后的闭环日志。

**反例与边界**：世界模型能生成动作条件视频不意味着可直接替代控制策略；减少 NFE 或 chunk 不能忽略动作质量。

<a id="c-M0"></a>
## M0 架构设计与工程判断

**依赖**：2.0b、2.7、5.2、5.7。

**问题**：不变量怎样跨越模块边界；替代设计怎样改变扩展和故障定位；哪些取舍有上游证据。

**对象与源码**：nanoserve、PyTorch Dispatcher、FSDP2、vLLM/SGLang cache manager；以实际源码、FSDP2 设计文档和对应 PR/RFC 为依据，不复述泛化的“工程品味”准则。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 为三例分别列状态所有者、不变量、公开接口和错误发现位置：dispatcher key、KV 引用、FSDP parameter group | 源码和实际运行能够验证每个边界；不要只画模块框图 |
| B | 在 nanoserve 用“scheduler 直接管理 block”与“独立 cache manager”实现同一 prefix+取消扩展 | 比较实际 diff、测试、状态传递、错误传播和运行成本；新增抽象有具体需求依据 |
| C | 查找原始 RFC/PR，区分上游陈述与作者提出的替代方案；把一个失败案例从现象追到边界设计 | 形成可核对的设计分析，不能凭个人解释声称上游否决某方案 |

**交付**：新增 `labs/M/architecture_variants/`，两种实现、同一任务对照、上游原文入口与正文设计分析。

**反例与边界**：接口多不代表解耦好；代码短不代表维护成本低；设计理由不能由实现形状唯一反推。

<a id="c-M3"></a>
## M3 前沿检索与复现

**依赖**：M1、M2；完整案例分别使用 5.5 与 10.4 的工件。 训练相关综合任务接 7.3/7.6/7.7/7.10，不依赖完整训练运行。

**问题**：论文主张如何映射到代码与实验；模型/代码更新如何改变可比性；复现怎样产生支持、限制或负面结果。

**对象与源码**：固定案例为 DFlash 的并行草稿与 Cosmos3-Edge 的 I2V 更新配方；原始论文、作者 repo、模型卡、变更说明与实际 engine 实现。搜索使用 Exa MCP 或命令行，不使用 subagent 或内置 websearch。 前沿来源核对增加 SmolLM3 全流程、SpecForge 训练/export、Cosmos3-Edge SFT、Flow-GRPO/CosyVoice 等训练案例，研究证据标准与推理案例相同。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对每个案例检索原始主张、前置方法、代码、模型 revision 与支持矩阵；记录哪些数字来自作者、哪些可由本机验证 | 生成主张→机制→所需证据表；摘要或模型名字不计作机制解释 |
| B | DFlash 对齐 target/draft、mask、验证与回滚；Cosmos 对齐旧/新权重、scheduler、帧数与精度，先做最小正确性实验 | 引用 5.5/10.4 的既有工件，不重复采同一结果；新版本必须另存配置和输出 |
| C | 预先写出可能推翻主张的对照：普通 decode 完整成本、未融合草稿、旧配方/新配方、阶段成本与任务质量 | 保留失败、减速和未解释趋势；不能以组件速度或作者 H100 数字替代本机端到端结果 |
| D | 由原始工件一键重算表图，说明测量不确定性、硬件边界与尚未验证的实验；依据新版本变化决定是否需要重新实验 | 读者能独立复现限定范围，并指出结论在哪个条件下不成立；不以引用数量或最新模型数量衡量深度 |
| E | 逐例核对原始数据→目标→参数/优化器→训练系统→阶段 checkpoint→评测→导出；分别登记论文、源码、配置、数据、权重、日志和本机小验证，检查检索摘要与现行仓库路径/API 的差异 | 发布承诺、已有代码、可运行配方、作者结果和本机结果分开；每类模型至少一份完整流程材料；不以未进行全量训练作为学习未完成，也不把材料研读称为完整实验复现 |
| F | 从 L1–L10 的各层选择代表性前沿阅读入口，按问题/前置/主要实现组织；每层至少完成一项机制剖析，跨算子、服务、训练各选择一个可运行实验，其余硬件专属路线说明可迁移和不可验证部分 | 交付进入正文的前沿解释和练习；与责任章共用源码/实验，避免重复项目；读者能分析一篇未讲过的论文，判断新机制改变了什么、需要什么条件及怎样设计有效对照 |

**交付**：两个固定案例的必要配置、运行命令与表图重算入口；正文保留具体机制和实验设计，不写检索过程流水账。

**反例与边界**：代码 main 和模型 main 可能独立变化；名称相同或参数开关存在不代表复现实验相同。
