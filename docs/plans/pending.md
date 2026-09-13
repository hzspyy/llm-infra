# 未完成部分的详细执行计划

本文件收录当前尚无正文的 42 个模块。每章指定问题、模型或实现、源码入口、实验矩阵、交付文件与验收条件。已有内容的修订任务见 [已完成部分](completed.md)。这里列出的新文件与命令接口是待实现交付，不表示现有脚本或实测；实际进度和证据只写入 [STATUS](../../STATUS.md)。

执行前按活计划完成输入、源码/模型 revision 和环境固定。主模型、比较问题、实验轴与判据按下列定义执行；兼容性核查负责确定准确提交和合法配置，不将核心研究设计留到临场决定。核心任务受限时保留未完成状态，同时推进无该依赖的任务。各章同样遵守 [章节规范](../chapter-guidelines.md)、[实验规范](../experiment-guidelines.md) 和 [环境记录](../../ENVIRONMENTS.md)。

<a id="c-1-6"></a>
## 1.6 端侧推理栈

**依赖**：1.1、4.3、5.1。

**问题**：统一内存和独立显存的成本如何不同；预编译部署受哪些 ABI/算子限制；持续运行怎样改变最优配置。

**对象与源码**：Jetson Orin 与 crater；共同模型 `Qwen/Qwen3-1.7B`，由固定 [llama.cpp](https://github.com/ggml-org/llama.cpp) 转换成 F16/Q4_K_M GGUF；前沿对照 [TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html) 的 ONNX→engine→C++ runtime，按[支持矩阵](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)核对 SDK。Orin 路线限定 FP16/INT8/INT4，不安排 FP8/FP4 engine 实验。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 tokenizer/template/权重 revision；解析 GGUF tensor 与量化块，追踪 llama.cpp graph、KV 和 CUDA backend；核对 Edge-LLM 导出、builder 与 enqueue 边界 | F16 转换与 HF 参照对拍；预编译 wheel/运行库/engine 与目标平台匹配，不能在 Jetson 临时编译依赖 |
| B | 同输入扫描 prompt=128/2048/8192、输出=128、并发=1/2/4；F16 与 Q4_K_M 分开测质量、内存和完整延迟 | 保存实际 backend 与 fallback；Edge-LLM 只有兼容目标的预制产物可用才运行，不能从支持“Jetson”推断支持 Orin |
| C | 每个配置持续 30 分钟，同步记录 tegrastats、温度、功耗、频率和逐请求延迟；对照已有合法功耗模式 | 冷机与热稳态分别报告，给能量/有效输出与 deadline miss；不修改共享系统配置 |

**交付**：新增 `labs/L1/edge_runtime_bench.py`、模型转换清单、持续运行记录和资源曲线；正文落入 outline 的 1.6 文件。

**反例与边界**：架构名和格式名不等于运行后端；没有兼容预编译环境时保留相应路线 UNVERIFIED，不能回退为本机编译。

<a id="c-1-7"></a>
## 1.7 实时闭环预算

**依赖**：1.4、1.6、5.1；动作头综合依赖 10.6。

**问题**：输入年龄与处理时延有什么区别；排队和抖动如何造成 deadline miss；action chunk 怎样与控制频率协调。

**对象与源码**：Qwen3-1.7B 端侧推理为系统负载，真实动作扩展用 10.6 的 DROID 策略；Python/C++ 单调时钟、进程队列、runtime 和传感/执行 harness。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现固定时间戳输入重放器和周期消费者，记录采样、排队、开始推理、结束、动作取用/丢弃；注入 5/20/50 ms 抖动 | 同一时钟域能重建输入年龄与端到端关键路径；跨设备时钟先校准 |
| B | 频率=5/10/20 Hz，队列容量=1/4/16；比较 FIFO、latest-only、丢弃过期输入/动作与 backpressure | 记录 deadline miss、有效动作率、输入年龄、p95/p99 与丢弃原因，保持同一任务输入 |
| C | 在热稳态下接真实策略 action chunk=1/4/8，测试计算更快但消费旧动作的反例 | 离线动作误差、系统时延和闭环任务结果分别验收；无机器人时只完成重放与时间预算 |

**交付**：新增 `labs/L1/closed_loop_budget.py`、统一事件 JSONL、时间轴与过期策略对照。

**反例与边界**：独立测量的各阶段均值之和不是闭环分位数；推理时延下降不能替代任务成功率。

<a id="c-4-5"></a>
## 4.5 图像与视频预处理

**依赖**：0.4、4.1。

**问题**：像素如何变成 patch；resize/采帧如何改变 token 与信息；不同模型 processor 的约束如何验证。

**对象与源码**：主模型 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，结构对照 `google/gemma-3-4b-it`；[Qwen3-VL 官方代码](https://github.com/QwenLM/Qwen3-VL)、Transformers image/video processor、qwen-vl-utils。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 RGB/归一化/resize 规则，从 config 提取 patch、temporal patch 与 merge 参数；手写 normalize、patchify、grid 计算 | 与官方 processor 逐元素对拍；原图、处理张量、grid_thw、shape/stride 和 token 数可检查 |
| B | 12 张自有或许可明确图像覆盖 320×240、640×480、1280×720 与竖图；视频采帧=1/2/4 fps、帧数=4/8/16，另设非法边界 | 输出采帧时间戳与实际舍入/padding；不能将文件大小当视觉 token 成本 |
| C | 在相同 OCR/计数/时间定位小任务上改变视觉 token 预算 128/512/2048，对比 Qwen3-VL 与 Gemma processor 的 crop/merge 结构 | 保留任务答案、信息损失、CPU 时间和后续 GPU 输入规模；跨模型质量差异单列 |

**交付**：新增 `labs/L4/vision_preprocess.py`，输入 manifest、patch/grid 原始数组、数值对拍与预算曲线。

**反例与边界**：颜色通道、像素范围、视频时间基错误可能不改变 shape；token 预算设置值与实际 token 数分别记录。

<a id="c-4-6"></a>
## 4.6 视觉编码器与 connector

**依赖**：4.1、4.5。

**问题**：视觉 token 经哪些层进入语言模型；连接器如何改变长度和维度；中间层特征融合带来什么状态与成本。

**对象与源码**：Qwen3-VL-4B-Instruct 的 ViT、patch merger、DeepStack；Gemma-3-4b-it 的 vision tower/projector；[Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 逐层采 patch embedding、attention/MLP、merger 与送入 LLM 的特征；手写小型 patch merger/projector | 输入/输出长度、维度和 norm 位置与官方实现对拍，连接器不概括成统一 linear |
| B | 固定图片，扫描分辨率与 batch=1/2/4/8，分别计 processor、ViT、merger、LLM prefill 和峰值 | 完整请求成本与每阶段资源闭合；不同阶段的最优 batch 分别记录 |
| C | 解析 Qwen3-VL DeepStack 的多层视觉特征注入，保存各注入点张量；做去除/只保留末层的分析性消融 | 对照原模型任务质量与成本，不把修改后模型称为官方配置；为 4.8 定义完整特征缓存载荷 |

**交付**：新增 `labs/L4/vision_connector.py`、逐层特征清单、阶段时间线和 connector mini。

**反例与边界**：最终视觉 embedding 可能不足以表示所有注入状态；connector 长度压缩不保证视觉信息无损。

<a id="c-4-7"></a>
## 4.7 多模态序列、位置与 mask

**依赖**：4.6、0.4、3.1。

**问题**：占位 token 如何替换成特征；多图/视频怎样分配位置；文本时间戳与多维 RoPE 如何共同表达时空。

**对象与源码**：Qwen3-VL-4B-Instruct 的 `get_rope_index`、Interleaved-MRoPE、DeepStack 与视频文本时间戳；用 Qwen2.5-VL 的位置规则作结构对照，不混用两个模型的 processor。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手工构造“文本—图1—文本—图2”和 8 帧视频输入，打印模板、占位符、grid、embedding 替换、position_ids 与 mask | 从原始输入独立算出小例的位置数组并与官方实现逐元素对拍 |
| B | 改变图像顺序、帧时间戳、padding、视频 fps 与 cache_position；逐步比较 prefill 和增量 decode | 空间/时间坐标与有效长度匹配；占位符数量错误、丢帧和错误 RoPE section 有可定位反例 |
| C | 解析 interleaved 频率分配与文本时间戳的不同职责，在时间定位任务中分别干预二者 | 记录真实 token、特征和任务答案；仅同一模型内受控干预用于因果解释 |

**交付**：新增 `labs/L4/multimodal_positions.py`，原始序列、位置数组、mask 可视化与失败输入。

**反例与边界**：三维坐标不等于将一维位置复制三遍；共享像素但时间或位置不同不自动拥有相同 KV。

<a id="c-4-8"></a>
## 4.8 多模态 batching、缓存与视频流

**依赖**：4.7、5.2、5.3、5.8。

**问题**：processor/encoder/KV 三层缓存如何区分；视觉工作如何进入调度预算；取消和复用如何保持特征身份。

**对象与源码**：Qwen3-VL-4B-Instruct；vLLM multimodal processor/cache、encoder scheduling，SGLang multimodal input与 feature cache；复用 4.6 的 DeepStack 特征载荷。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现带模型、processor、图像内容、分辨率、帧采样和位置身份的 feature cache；分别记录三层命中 | 同图重复请求、不同 crop、同帧不同时间、同特征不同文本上下文逐项对照，不能混淆缓存层 |
| B | 请求组合为纯文本、单图、多图、4/8/16 帧视频，分别重放突发与泊松到达；对比两引擎 batching/encoder 准入 | 采实际视觉 token、encoder 队列、TTFT、峰值与任务质量；缓存收益由省下的工作和完整时间支持 |
| C | 在预处理、encoder 和 decode 阶段取消，插入慢视频输入与错误图像；比较 bounded queue 和无界队列 | 特征、KV、CPU buffers 和队列资源均回收；慢模态对文本请求的长尾影响可量化 |

**交付**：新增 `labs/L4/multimodal_serving.py`、最小特征缓存、逐请求状态与缓存失效实验。

**反例与边界**：processor 缓存命中不证明省掉 vision forward；流式上传视频不保证模型支持增量编码。

<a id="c-4-9"></a>
## 4.9 语音输入与流式 ASR

**依赖**：0.4、4.1、5.1、2.0c。

**问题**：波形如何变成有效序列；分块输入与流式输出如何区别；边界修订和端点判断需要哪些状态。

**对象与源码**：主例 `Qwen/Qwen3-ASR-0.6B`，扩展 1.7B；结构对照 `openai/whisper-small`。阅读 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 `qwen_asr/inference/qwen3_asr.py`、`ASRStreamingState`，对照官方 vLLM backend 与 [SGLang-Omni ASR](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 16 kHz PCM 的声道、幅度、分帧到音频特征/encoder，手写长度传播与边界补齐；输出有效长度、mask、特征和音频 token | 官方样例逐层对拍；重采样、空音频、截断、静音和错误采样率有可定位反例 |
| B | 实现可重放输入分块，chunk=1/2/4 s、feed step=20/100 ms、unfixed tokens=0/5/10；逐块记录累计音频、回退文本与稳定前缀 | 区分重新编码累计音频、缓存增量和伪流式 transcript；不因 SSE 逐 token 返回就宣称支持实时输入 |
| C | LibriSpeech test-clean 与 AISHELL-1 test 各固定 100 条按时长分层样本；同音频比较整段与分块、batch=1/4/16 | CER/WER、首个稳定转写、最终时延、RTF、峰值与重复/漏字一起报告，评测清单与正规化脚本固定 |
| D | 两引擎用共同整段模式对照；官方 streaming 与 SGLang 的上传后 SSE 分别记录支持语义，测试中断与慢输入 | 接口不支持持续输入时明确作为独立模式，保留音频状态回收和长尾结果 |

**交付**：新增 `labs/L4/asr_frontend.py`、`asr_stream_bench.py`，原始音频清单、转写版本、时间线、错误和评测结果。

**反例与边界**：最后转写正确不能证明中间字幕稳定；输入 chunk、模型计算 chunk 和网络输出 chunk 不是同一个对象。

<a id="c-4-10"></a>
## 4.10 语音生成、codec 与播放

**依赖**：4.9、5.1、10.1。

**问题**：多码本如何组成音频；首包和首可播放时间怎样不同；AR codec 与 flow 声学生成怎样组织状态。

**对象与源码**：共同主例 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，固定许可明确的官方参考音频与转写；文本直出对照 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`。使用 [vLLM-Omni 0.18 TTS](https://docs.vllm.ai/projects/vllm-omni/en/v0.18.0/user_guide/examples/online_serving/qwen3_tts/) 与 [SGLang-Omni TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html)；连续生成对照 [F5-TTS v1 Base](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 跟踪 preprocessing→tts engine→vocoder，打印码本数量、采样率、token/chunk 对应长度；手写码本打包/解包和拼接队列 | codec 边界、有效采样点和音频长度与官方实现对拍；Base 的参考音频条件不能省略 |
| B | 固定 40 条中英文短/长文本，分别比较离线与真实 chunk 输出；chunk 对应 4/8/16 个 codec frame，超出接口限制的点不运行 | 采网络首包、首可播放、播放开始、underrun、RTF、峰值、边界爆音与 ASR 回转 CER/WER |
| C | 两运行时分别对 stage batching、graph on/off 和异步 chunk 传输做消融；batch=1/4/8，加入慢消费者 | 时序、资源和输出状态一致；区分图捕获启动成本与稳态，不能跨不兼容 Transformers 栈直接混装 |
| D | F5TTS_v1_Base 使用固定参考音频、文本、初始噪声与目标时长，扫描 NFE=8/16/32；追踪 flow solver→vocoder | 与 AR 路线按内容/可懂度/时长分别报告质量—成本，不能仅比较音频 token/s |

**交付**：新增 `labs/L4/tts_codec_probe.py`、`tts_stream_bench.py`，原始波形、码本与播放事件；可重复听检和自动指标均保留。

**反例与边界**：输出分块不保证持续可播放；ASR 回转指标不能代表音质、自然度或声线相似度。

<a id="c-4-11"></a>
## 4.11 原生 Omni、跨阶段调度与打断

**依赖**：4.8、4.9、4.10、5.8、5.11；多卡部署先完成 6.2 的 TP 基础。

**问题**：音视频怎样共享时间轴；阶段队列如何传播压力；取消如何同时终止计算、传输和旧音频播放。

**对象与源码**：原生主模型 `Qwen/Qwen3-Omni-30B-A3B-Instruct`；级联参照 Qwen3-ASR-0.6B→Qwen3-VL-4B-Instruct→Qwen3-TTS-12Hz-0.6B。阅读 [vLLM-Omni Qwen3-Omni](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/examples/online_serving/qwen3_omni.md)、[SGLang-Omni 模型配置](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_omni.html)与 [pipeline](https://sgl-project.github.io/sglang-omni/developer_reference/pipeline.html)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 Thinker/Talker/Code2Wav 输入输出、codec/hidden 状态与队列；从模型配置核算完整权重和峰值。初始部署为 worldvln Thinker TP=2、Talker 一卡、Code2Wav 一卡 | 仅在整除、版本与显存预算满足后部署；逐 stage 的 readiness、设备与传输方式明确，不能按 3B 激活参数估计总容量 |
| B | 用 20 段许可明确的音画片段，包含字幕时间标记、说话暂停与视觉事件；比较级联和原生的输入张量、输出任务与时间轴 | 两条真实链路各自验收，级联结果不替代原生能力；共同任务的理解质量与语音可懂度都有记录 |
| C | 两运行时分别扫描并发=1/2/4、队列上限=1/4/8、消费者延迟=0/100/500 ms；记录 stage 工作、跨阶段 chunk、首可播放和持续输出 | 背压由队列/事件支持；SGLang Omni 的阶段并发与其不支持的 AR overlap loop 分开，不套用基座开关 |
| D | 在 Thinker、Talker、Code2Wav、播放四处发起打断；用 session epoch 丢弃旧包，注入断连、失败与重试 | 测打断至静音、停止计算与资源回收，禁止旧响应重新播放；同会话下一请求正常执行 |
| E | 在共同 BF16 配置对照两运行时，再单独评估支持的低精度/分离部署；保存完整任务×模态×流式方式×精度×并行矩阵 | 支持声明须落实到对应源码和运行；资源不足时原生任务保持未完成，不能以 thinker-only 交付冒充全部 Omni |

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
| D | 对 NVLS、NCCL symmetric memory 和跨机 transport 阅读触发条件；已有无 NVLink 平台只运行其支持路径 | 形成硬件/驱动/权限/算法矩阵；跨机准备依 ENVIRONMENTS，不能修改共享驱动验证特性 |

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
| B | worldvln 先做同机 1P1D，再比较 1P2D；扫描输入=128/2048/8192、输出=32/256/1024 和到达率 | 固定相同总 GPU 预算，对比并置与分离 goodput、TTFT、TPOT、峰值及传输成本 |
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
| A | 在两 rank 写分片 Q/K/V、causal mask 与 online m/l/O 归并；对比 ring 传 KV 与 Ulysses all-to-all 换维 | 小尺寸完整 attention 的值和梯度对拍；非整除长度、GQA、窗口边界有用例 |
| B | worldvln 2/4 卡，S=2048/8192/16384/32768，分别测训练激活与推理 KV；固定模型原生有效上下文 | 记录每 rank 峰值、通信轮次、重算与完整时间；不能仅以全局 token 数估计显存 |
| C | 比较 CP/SP 与 TP 的合法组合，加入负载不齐、不同 mask 和小 batch；保存实际 collective 顺序 | 全局位置、mask 和数值一致；解释重叠的依赖与实际通信/计算转换区间 |

**交付**：新增 `labs/L6/context_parallel_attention.py`、`long_context_bench.py`，分片布局、梯度、容量与时间曲线。

**反例与边界**：改配置扩大上下文不等于模型质量保持；视频和混合递推模型的并行条件在各自章节单列。

<a id="c-7-3"></a>
## 7.3 训练框架架构

**依赖**：7.2。

**问题**：模型、并行、优化器和数据的职责如何划分；一个训练步怎样触发通信；新增模块需要修改哪些边界。

**对象与源码**：同一 SmolLM3-3B-Base，基础对拍先用 SmolLM2-360M；PyTorch FSDP2、DeepSpeed ZeRO、Megatron-LM、[TorchTitan](https://github.com/pytorch/torchtitan)，参考 [FSDP2 设计](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) 和 SmolLM3 官方 nanotron 配置。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 为四框架分别列 model construction、parallelize/wrap、train_step、optimizer、checkpoint 的调用与状态归属 | 固定源码进入 hooks、通信 group 和状态转换；不能以四段启动命令代替架构分析 |
| B | 实现共同适配接口 load_weights/make_batch/train_step/save/restore；SmolLM3 的 NoPE/GQA 按实际配置移植，不假定原生支持 | 一次前向、梯度和参数更新与 HF 参照对拍；不支持时交付适配代码与限制，不能悄悄换成不同模型 |
| C | 固定 WikiText-2 样本 ID、S=512/2048、有效全局 batch、AdamW 超参数与 dropout，比较 2/4 张 L40S 的 FSDP/ZeRO/TP/PP 合法配置 | 模型、激活、优化器、通信 buffer 峰值和正常训练时间齐备；先语义一致再比较性能 |
| D | 添加冻结 vision tower、小 projector 与文本 loss 的小型多模态训练用例，比较 requires_grad、mixed precision、checkpoint 接口的改动 | 记录扩展面与出错位置；冻结层不产生 optimizer 状态，保存与恢复保持一致 |

**交付**：新增 `labs/L7/framework_adapters/`、`framework_compare.py`，四套固定配置、数值对拍、训练时间线与扩展实例。

**反例与边界**：框架默认 loss 归一化、Adam eps/betas、梯度缩放或重算不同会使性能对照失去可比性。

<a id="c-7-4"></a>
## 7.4 数据、checkpoint 与容错

**依赖**：7.2、7.0b、1.5。

**问题**：样本顺序怎样跨 worker/rank 保存；异步保存何时形成一致快照；恢复怎样验证后续更新连续。

**对象与源码**：SmolLM2-360M 的精确恢复基线与 SmolLM3 FSDP2；PyTorch Distributed Checkpoint 的 staging/save/load、[StatefulDataLoader](https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html)、distributed sampler 和 optimizer state。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 构建带样本 ID 的 map/iterable 数据集，workers=0/2/4、prefetch=1/2；保存每 rank 的 sampler、worker RNG、数据游标和未消费队列语义 | 不把 DataLoader 的 worker 聚合当成跨 rank 状态保存；重启后逐样本序列可比较 |
| B | 比较同步保存与 async staging/save，分别在 staging 前后、optimizer 后和落盘中失败；使用独立训练子进程注入 | checkpoint 中参数、optimizer、scheduler、RNG 与 step 来自同一逻辑时刻；不能保存混合版本 |
| C | 第 3/7/11 步中断，恢复后继续 5 步，与不中断运行对照样本、loss、梯度和参数；将缺失状态逐项作为反例 | 固定 world size 的恢复按容差对齐；失败 checkpoint 不冒充完成文件，manifest/提交点明确 |
| D | 在后端支持时从 2→4 ranks 恢复，检查分片与全局样本归属；测 staging 峰值、I/O、训练停顿和恢复时间 | 弹性 world size 的等价性条件独立说明；大 checkpoint 留在学习存储，只归档必要元数据和验证结果 |

**交付**：新增 `labs/L7/resumable_training.py`、`checkpoint_failure_injection.py`，数据状态与 checkpoint manifest、恢复轨迹。

**反例与边界**：只恢复权重不能恢复训练；异步接口返回不代表数据已安全落盘或 staging 已完成。

<a id="c-7-5"></a>
## 7.5 SFT、LoRA/QLoRA 与 DPO

**依赖**：7.0b、7.1、4.3。

**问题**：训练损失作用于哪些 token；低秩与量化如何改变可训练状态；偏好目标的策略与参考如何对应。

**对象与源码**：主模型 Qwen3-1.7B；PEFT LoRA、bitsandbytes NF4、TRL SFT/DPO；数据用 `HuggingFaceH4/ultrachat_200k` 的固定 1000 条训练/200 条验证样本，以及 `HuggingFaceH4/ultrafeedback_binarized` 固定 256 对训练/64 对验证。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 chat template、thinking 开关、prompt/response mask、packing 和截断；手写 SFT cross-entropy 与 DPO 的 chosen/rejected/reference logprob 计算 | 与 TRL 同批 loss/梯度对拍；长度归一化和 sum logprob 的不同目标分别说明 |
| B | q_proj/v_proj 上 rank=8/16/32、alpha/rank 固定；比较全参、LoRA 与 NF4 QLoRA 的权重存储、compute dtype、梯度和 optimizer 状态 | 冻结参数不更新；记录双重量化/反量化路径，不能把 NF4 存储称为原生 INT4 训练 GEMM |
| C | 固定有效 token、优化器和训练步，分别测完整时间、峰值与 held-out loss/指令有效性；DPO 检查 reference 冻结和样本配对 | 小规模实验只支持机制与任务有效性范围；不把单 batch loss 下降当泛化提升 |
| D | 导出有 base revision、target modules、rank、alpha、训练数据哈希的 adapter，比较 merged/unmerged 数值并交给 5.10 | 两种加载路径在同输入上对齐；量化 base 的合并与重数量化需要单独误差检查 |

**交付**：新增 `labs/L7/sft_lora_dpo.py`、`preference_loss_reference.py`，训练数据 manifest、adapter 与服务质量对照。

**反例与边界**：DPO 的 reference 不是行为采样策略；冻结 base 与不保存 base 的梯度是必须检查的实际状态。

<a id="c-7-6"></a>
## 7.6 RL 运行时与策略同步

**依赖**：7.5、6.2、5.1、5.10。

**问题**：样本与行为/训练/参考策略怎样对应；异步与 partial rollout 如何产生陈旧度；权重同步怎样影响吞吐和目标语义。

**对象与源码**：Qwen3-1.7B，GSM8K 固定 256 题训练/128 题评测，规则化答案 reward；veRL 为主实现，slime/SGLang 与 OpenRLHF 为结构对照。阅读 [veRL fully async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)、[rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)、[OpenRLHF async](https://openrlhf.readthedocs.io/en/latest/async_training.html)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现一次同步 rollout→reward→advantage→update→weight sync；每题采 4 条，固定 token 上限与模板，保存样本 ID、token logprob、mask 和各策略版本 | 逐项对拍 GRPO/PPO 所选目标；行为概率、旧训练策略与 reference 的职责不混用，公式按具体算法固定 |
| B | 同一 token 序列分别在 rollout engine 与 learner teacher-forcing 中取 logprob，控制 batch、精度与处理顺序；记录训练/推理分布差异 | importance correction 与 PPO ratio 分开计算；注入错误版本分母能被检查发现 |
| C | worldvln 分离 learner 与 rollout，队列深度=1/2/4，允许 lag=0/1/2；注入长输出与慢 reward，比较同步/异步 | 报完整迭代时间、GPU 利用、等待、样本年龄、reward 和 KL；不以局部 rollout tok/s 代替训练吞吐 |
| D | partial rollout 中热更新，保存每个 token 的行为版本、缓存身份与恢复事件；检查框架是否重算前缀或保留旧状态 | 一条样本混合版本时不能只记一个 episode version；旧行为策略不自动判错，但必须有对应 logprob 和算法处理 |
| E | 复现 veRL 的主流程后，在 slime 与 OpenRLHF 对齐同模型小任务，比较状态交接、资源切换与失败恢复；依兼容版本分别建环境 | 三框架源码/支持边界齐备，实际运行与未运行路线分列；少量更新不声称收敛质量等价 |

**交付**：新增 `labs/L7/mini_rl_iteration.py`、`rl_policy_version_probe.py`、框架配置、token 级版本与权重传输时间线。

**反例与边界**：策略陈旧度不只是 step 差；低精度后端和部分更新也可能改变实际行为分布。

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
| C | 逐项关闭 tokenizer 竞争、encoder batching、KV 驱逐或大 GEMM 干扰，采同步资源时间线 | 用干预验证瓶颈；仅时间重合不视为争用因果证据 |
| D | 实现按任务预算的 admission controller，重放同一负载并扫描资源阈值 | 输出满足多类 SLO 的可复算容量边界；超载时资源回收与拒绝行为正确 |

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
| A | 实现 GPU/CPU 两级 store，明确 model/adapter/position/quant/layout 身份、引用、传输完成和驱逐状态；加入校验和 | 取回后的 KV 与直接重算逐元素/逐 token 对齐；错误布局和旧 revision 拒绝 |
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
| A | 实现 tenant/session/adapter revision 与预算，构造同名 adapter、相同文本不同私有前缀和跨租户取消 | 输出、KV、adapter 和资源归属可检查；所有输入使用合成数据 |
| B | 租户 A 固定交互负载，B 逐步增加长 prompt、并发或 adapter 换入；比较共享、配额、优先级和进程隔离 | 分租户统计质量、p95/p99、拒绝、抢占与资源；不能用平均吞吐掩盖交互服务退化 |
| C | 在 worker 崩溃、预算超限和服务更新后恢复，检查缓存身份、幂等请求与资源清理 | 合法租户在恢复后继续可用；缓存身份和权限控制在对应层验证 |
| D | 说明 MIG/MPS/进程隔离与应用配额的不同能力，实际支持的独立环境才做对照 | 硬件或权限不支持的方案只记录机制和条件，不通过修改共享驱动验证 |

**交付**：新增 `labs/L8/tenant_scheduler.py`、`tenant_interference.py`，隔离/干扰曲线与状态回收验证。

**反例与边界**：prompt 约束不是资源或数据隔离；独立进程也可能争用同一 GPU 带宽和显存。

<a id="c-9-1"></a>
## 9.1 Agent 负载画像

**依赖**：5.1、5.2、5.3；音画会话扩展依赖 4.11。

**问题**：真实执行产生怎样的轮次和长度分布；工具等待怎样改变并发与缓存；合成分布能检验什么假设。

**对象与源码**：Qwen3-4B 的固定 tool/reasoning 配置；三类各 100 个任务：本地计算工具、多轮 NFCorpus 检索、固定小代码仓库的读取与修复。执行工具限定项目隔离环境，记录真实模型调用与工具返回。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现任务 harness，记录 session/turn/tool IDs、完整输入长度、输出长度、等待、缓存命中、重试和任务得分 | 采集实际 agent 执行；合成任务与生产流量区分，失败会话不能从统计中删去 |
| B | 生成轮次、长度、工具延迟、重复前缀、thinking token 的联合分布及分位数；比较均值匹配和分布匹配的合成负载 | 原始请求可重放；明确跨轮相关性，不能独立随机抽长度后称保留真实工作负载 |
| C | 在 5.3/8.1 的调度与路由上重放同一轨迹；另加入 4.11 的音画 chunk/打断会话 | 质量约束下的服务结论可迁移；文本与音画事件采用共同时间定义 |

**交付**：新增 `labs/L9/agent_trace_collect.py`、`trace_replay.py`，任务清单、原始事件、画像与重放一致性检查。

**反例与边界**：一次会话不能代表总体；工具 schema 重复不保证工具参数 token 的接受率高。

<a id="c-9-2"></a>
## 9.2 工具调用的系统路径

**依赖**：5.6、5.5、9.1。

**问题**：tool choice 在哪里约束输出；增量 parser 怎样维护多调用状态；约束与投机如何共同影响成本。

**对象与源码**：Qwen3-4B，vLLM/SGLang 的 Qwen tool parser、structured output adapter、DFlash/普通解码；固定计算、单位转换和文档检索工具，接口使用本地可核对实现。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对 none/auto/required/named tool choice 追踪 template、grammar、token、delta 和最终参数；工具数=1/4/16/64 | 真实支持模式逐项列出；语法有效、参数有效、工具选择和任务成功分开评分 |
| B | 实现有状态流式 parser，覆盖并行调用、转义字符串、跨 chunk Unicode、取消与不完整 JSON；payload=256/1024/4096/16384 字节 | 和完整解析参照对拍；记录累计扫描字节，验证重复全串解析是否产生二次成本 |
| C | 对结构固定/参数自由/长数值/长文本四种内容分别测约束、草稿、验证与 parser；保持 target 模型和任务一致 | 接受长度、无效输出、完整时间和任务质量共同报告；不能用模板可预测性推断参数也可预测 |

**交付**：新增 `labs/L9/tool_pipeline.py`、`incremental_tool_parser.py`，schema、bitmask、增量片段与真实工具返回。

**反例与边界**：所有 tool calling 并不都由 grammar 实现；JSON 能解析不保证执行的是正确工具。

<a id="c-9-3"></a>
## 9.3 多轮会话的 KV 生命周期

**依赖**：5.2、9.1、8.6。

**问题**：跨轮哪些 token 真正保持共同前缀；驻留、驱逐、取回和重算怎样选择；会话迁移怎样影响收益。

**对象与源码**：Qwen3-4B、vLLM/SGLang prefix cache、8.6 的 LMCache 两级存储、8.1 的 session-aware router；使用 9.1 真实执行轨迹。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 构造 2/4/8 轮会话，保存每轮模板与 token IDs，定位工具结果插入、thinking 历史裁剪、前缀改变的位置 | 预测命中块与实际命中一致；session ID 相同不作为 KV 等价依据 |
| B | 轮间等待=0/1/10/60 s，注入其他会话施压；比较常驻、LRU 驱逐、CPU 取回与重算 | 每轮 TTFT、搬运/重算字节、占用与任务质量齐备；给出随间隔和容量变化的选择条件 |
| C | 同一会话固定副本与随机迁移对照，改变 adapter revision/量化配置/位置规则并验证失效 | 迁移和恢复后输出与相应参考一致；不将旧版本缓存误接到新模型 |

**交付**：新增 `labs/L9/session_kv_bench.py`、会话驻留策略与逐轮状态/时延账。

**反例与边界**：保留缓存有机会成本；提高命中率可能挤占更有价值的新请求。

<a id="c-9-4"></a>
## 9.4 Reasoning 与长输出预算

**依赖**：5.1、3.3、5.13、9.1。

**问题**：thinking 预算怎样改变质量和资源；不同状态架构怎样随输出增长；短时最优配置能否迁移到长输出。

**对象与源码**：Qwen3-4B 的 thinking/no-thinking，Qwen3.5-4B 混合架构对照；GSM8K 与 MATH-500 各固定 128 题；vLLM/SGLang reasoning parser、预算/停止规则和状态池。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定题目、模板和评分，预算=128/512/2048/8192 token；保存 reasoning/final、停止原因、有效答案和总输出 | 未给最终答案、截断和超时都计入任务结果；不能只比较生成长度 |
| B | 并发=1/4/16，逐阶段测 prefill/decode、KV/递推状态、workspace 和长尾 | 状态增长由真实层列表和输出步数解释，不预设所有模型 KV 都相同增长 |
| C | 在相同总 token 或 GPU 时间预算下比较一次长推理与多次短采样，固定答案聚合规则 | 报每正确答案的时间、能量和队列影响；质量曲线按独立题目重采样，不外推到未测任务 |

**交付**：新增 `labs/L9/reasoning_budget.py`，题目/答案清单、预算控制器与质量—成本—状态曲线。

**反例与边界**：思考更长不保证更准；减少 token 的速度收益必须同时报告任务退化。

<a id="c-9-5"></a>
## 9.5 Agent 运行时

**依赖**：5.8、5.11；音画扩展依赖 4.11。

**问题**：多轮任务如何持有状态；取消怎样跨工具和模型传播；重试怎样与幂等和持久恢复配合。

**对象与源码**：扩展 nanoserve 与 9.1 harness；模型 Qwen3-4B；工具为计算、检索和带幂等键的项目内任务账本；参考 veRL AgentLoop 的异步任务组织。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 session/turn/tool 状态机、deadline、取消 token、事件日志和结果缓存；每次调用有唯一 ID | 正常路径可从日志重放；重复返回与乱序完成不会推进错误轮次 |
| B | 在模型等待、工具运行、工具完成但未 ACK、下一轮 prefill 四点注入超时/崩溃，比较重试策略 | 任务账本幂等、KV/队列回收正确；外部无幂等接口不能宣称通用 exactly-once |
| C | 恢复中断会话，检查模型/adapter/tool schema 版本和上下文；加入 4.11 播放 epoch 防止旧包复活 | 下一轮结果与合法恢复策略一致，取消后的旧音频和工具结果不泄漏到新轮 |

**交付**：新增 `labs/L9/agent_runtime.py`、`agent_failure_matrix.py`，持久事件、重试与恢复案例。

**反例与边界**：断开客户端不等于工具停止；进程重启后盲目重放所有工具会重复产生副作用。

<a id="c-9-6"></a>
## 9.6 RAG、索引与 memory

**依赖**：5.12、8.6、9.1。

**问题**：检索到生成的完整成本在哪里；ANN 在固定召回下怎样取舍；片段 KV 何时可复用。

**对象与源码**：Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B、Qwen3-4B；FAISS FlatIP/HNSW/IVF、[BEIR FAISS 示例](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_faiss_dense.py)。NFCorpus 用于检索评测，HotpotQA 固定 200 题用于带答案/支持事实的生成评测。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定语料 revision、chunk、query 指令、embedding 归一化与相似度；FlatIP 生成精确 top-k 参照 | 检索 ID/分数与 qrels 可核对；NFCorpus 的相关性标注不冒充生成答案标注 |
| B | HNSW efSearch=16/32/64/128、IVF nlist/nprobe 按训练数据量选择，固定 Recall@k 阈值再比较时间/内存；过滤前后分别评分 | 索引构建、增量更新和查询成本分列；10万/100万合成向量的规模实验与真实任务分开 |
| C | 对 top-k=5/20/100 接 rerank 和 prompt 注入，测各阶段与端到端；固定 HotpotQA 得分及引用有效性 | 较低 recall 或更短上下文带来的加速不能直接称系统改进 |
| D | 只对满足相同上下文/位置/模型身份的前缀预热 KV；给同片段不同前缀构造错误拼接反例，再研究 CacheBlend 等近似方案 | 精确复用和近似重算分别对拍/评质量；跨实例取回依 8.6 的真实接口 |

**交付**：新增 `labs/L9/rag_pipeline.py`、`ann_quality_bench.py`、`rag_kv_probe.py`，语料/索引 manifest、质量与完整成本。

**反例与边界**：高检索召回不保证生成正确；向量、文本片段和 KV 的缓存身份不能混为一种。

<a id="c-10-1"></a>
## 10.1 DDPM、score 与 rectified flow

**依赖**：4.2、7.0b。

**问题**：训练目标与采样向量场怎样对应；插值路径和实际生成轨迹有何区别；预测参数化怎样进入数值求解。

**对象与源码**：二维八峰 Gaussian mixture 为可检查主例，同规模 MLP；[DDPM](https://arxiv.org/abs/2006.11239)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003) 与 [flow_matching](https://github.com/facebookresearch/flow_matching) 官方实现。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定噪声分布与数据分布，推导 q(x_t\|x_0)、epsilon/x0/v 参数化、score 与 velocity；实现相互转换和一步更新 | FP64 小例对拍，明确时间方向、噪声系数与边界；不能混用不同 v 的定义 |
| B | 两种目标使用相同 4 层宽 128 MLP、batch=1024、固定 10000 更新和 seed=0/1/2；保存训练/验证样本和 loss | 参数量与训练预算一致，loss 数值不直接跨目标比较；记录各自生成分布 |
| C | 固定初始噪声，输出时间网格、逐步场值与轨迹；采样步数=8/16/32/64/128，评 sliced Wasserstein、模式覆盖与 NFE | 用共同分布指标比较；直线条件插值不能推出学得向量场轨迹必直或必然少步 |

**交付**：新增 `labs/L10/toy_diffusion_flow.py`、`prediction_parameterization.py`，训练配置、模型、逐步轨迹和分布指标。

**反例与边界**：训练目标名称不能直接决定 solver；相同 seed 在不同随机过程下不保证相同轨迹。

<a id="c-10-2"></a>
## 10.2 Solver、步数与蒸馏

**依赖**：10.1。

**问题**：一步包含多少模型求值；时间网格与预测类型怎样限制 solver；少步模型的改进来自什么变化。

**对象与源码**：Diffusers Euler/Heun/DPM-Solver++/LCMScheduler；真实基线 `stabilityai/stable-diffusion-xl-base-1.0`，少步对照 `latent-consistency/lcm-sdxl`，按 [LCM 官方用法](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm) 同时替换对应权重与 scheduler。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写 Euler/Heun 与多步状态缓存，在解析 ODE 和 10.1 网络上对拍；打印时间/sigma、模型输入缩放、预测转换与更新 | 误差随步长的变化符合相应条件；历史状态、起步阶段与末步处理均可检查 |
| B | SDXL 固定 32 个中英文 prompt、每题 3 个初始 latent，合法 solver 下扫描 10/20/40 步、CFG=1/5/7.5 | 记录实际 denoiser 次数、CFG batch、编码/VAE 固定成本、输出和完整时间；区分步数与 NFE |
| C | LCM 以其匹配配置扫描 2/4/8 步，与 SDXL 基线分别报告质量/时间；对故意错配 prediction_type 与 scheduler 保存失败样本 | 蒸馏权重变化与 solver 变化分开；不能只换调度器就宣称得到蒸馏模型能力 |
| D | 对输出保留感知距离、条件一致性和固定顺序盲评，数值轨迹误差与语义质量分别解释 | 相同 seed 不足时直接复用 latent/每步噪声；所有配置和生成样本可重放 |

**交付**：新增 `labs/L10/solver_reference.py`、`solver_quality_bench.py`，scheduler 配置、求值轨迹、原始样本与成本曲线。

**反例与边界**：更高阶方法可能有更多 NFE；不能把高步数输出当客观真值或用图像相似度替代全部质量。

<a id="c-10-3"></a>
## 10.3 DiT pipeline 与扩散服务

**依赖**：10.2、5.3、5.8、3.1、2.0c。

**问题**：完整请求由哪些计算和状态组成；请求级与步级 batching 怎样不同；不同请求如何隔离 solver、latent 和 RNG。

**对象与源码**：共同主模型 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)，Diffusers `ZImagePipeline`/transformer、vLLM-Omni DiffusionEngine、[SGLang-Diffusion Z-Image](https://docs.sglang.io/cookbook/diffusion/Z-Image/Z-Image-Turbo)。该 Turbo checkpoint 的少步/无 CFG 路线与常规有 CFG 模型分开。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 跟踪 tokenizer/text encoder、latent 初始化、DiT、scheduler、VAE 与编码输出；每阶段记录 shape/stride、dtype、权重和临时状态 | 先复现官方推荐采样参数并记录实际 NFE，不能把参数中的 step 数直接当求值数 |
| B | 实现可单步推进的 runner，持有每请求 latent、sigma index、solver history、RNG、condition 与取消标记 | 串行、交错和独立运行在允许数值容差内对拍；请求不能共享可变 scheduler 状态 |
| C | 固定 prompt/latent，合法分辨率 512²/768²/1024²、batch=1/2/4、到达率按基线扫描；比较整请求串行、请求级 batching、步级插入 | 给出兼容键、等待、队头阻塞、峰值和完整完成时间；各引擎实际支持粒度逐项记录，外层 async 不冒充步调度 |
| D | 两运行时对照同一 checkpoint/精度/配置，加入取消、不同分辨率和可支持 LoRA 的身份检查；逐步接入 mini 的调度策略 | 输出有效性、任务质量、状态隔离和资源回收先验收；不支持功能保持明确未完成 |

**交付**：新增 `labs/L10/diffusion_step_runner.py`、`diffusion_serving_bench.py`，两运行时配置、完整阶段与请求时间线。

**反例与边界**：DiT 不必在所有形状下算力受限；Turbo 模型不适合直接套用常规 CFG/高步数扫描。

<a id="c-10-4"></a>
## 10.4 视频生成与世界模型

**依赖**：10.3、3.2、3.4；分布式扩展依赖 6.5。

**问题**：视频怎样变成时空 token；世界模型的理解/生成通路共享什么；帧数、求值、缓存与并行怎样共同决定成本。

**对象与源码**：视频基础参照 [Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)，前沿主例明确使用 [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)。阅读 [8 月更新说明](https://huggingface.co/nvidia/Cosmos3-Edge/discussions/62)、[官方 cookbook](https://github.com/nvidia/cosmos)、Diffusers `Cosmos3OmniPipeline`/`Cosmos3OmniTransformer`/`UniPCMultistepScheduler`/`AutoencoderKLWan` 和 [vLLM-Omni pipeline_cosmos3](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3/)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 首先核对 ENVIRONMENTS 已缓存的 `6f58f6b4c91288838e60b6bcb2cc45d997e961de`；新版目标固定 `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`，分别核对代码兼容和配置。旧共享快照只读，新版需要时另存学习目录 | model_index/transformer/vae/scheduler/processor、权重 hash 和源码 pin 齐备；不能把旧权重配新版示例并视为同一实验 |
| B | 新版首先复现 I2V：832×480、121 帧、24 fps、BF16、20 denoising steps、guidance=6、flow_shift=12、use_karras_sigmas=False、seed=0，使用官方 example_i2v_input 与 JSON prompt/negative prompt | 实际 scheduler 参数和 NFE 写进工件。配置文件中的 flow_shift=1/use_karras_sigmas=True 必须按目标配方显式覆盖；不从旧文生图/文生视频示例推断新版 Edge 支持范围 |
| C | 从真实配置重建 48-channel latent、空间压缩 16、时间压缩 4、latent patch=2 与时空位置；分解一次理解侧计算、缓存和每步生成侧计算 | 打印实际 grid、有效帧数、padding/crop、KV/条件/latent/workspace；以真实 Edge backbone 配置为准，不套用其它 Cosmos 子型号结构 |
| D | 官方样例通过后，用 12 组许可明确的桌面操作/物体运动/室内场景，先固定 121 帧扫描 10/20/30 步，再固定 20 步测试经版本确认合法的 61/121/149 帧；分辨率扩展单列合法配置 | 生成文件帧数/fps/尺寸正确，保存阶段时间、峰值、NFE、条件遵循与物体/运动一致性；未见拐点也如实记录，不将合成超长 shape 当真实整模型 |
| E | 同 checkpoint/输入/latent 比较 Diffusers 与 vLLM-Omni 的完整 I2V，检查理解侧 KV/条件缓存是否按身份复用；Wan2.2 仅作架构与 pipeline 对照 | 本机结果与官方 H100 吞吐分开；跨模型比较同时报告任务与质量，不据参数量归因速度 |
| F | 接入模型实际支持的 action-conditioned forward/inverse dynamics 样例，读取具身维度、归一化和 mask；支持的并行路线在 6.5 后测 | 世界预测、动作条件生成、reasoner 和独立 Policy-DROID 各自标明能力；生成视频不作为闭环控制成功证据 |

**交付**：新增 `labs/L10/cosmos_edge_anatomy.py`、`cosmos_edge_i2v_bench.py`、`video_shape_ledger.py`，固定版本配置、官方样例复现、原始视频与时空/资源账。

**反例与边界**：sound_tokenizer 为空的配置不据 Cosmos 家族介绍宣称支持音频；动作内部 padding 维度不等于具身实际控制维度；帧数和尺寸必须经过所选版本合法性检查。

<a id="c-10-5"></a>
## 10.5 扩散缓存与近似复用

**依赖**：10.2、10.3、5.2；Cosmos 具体实验依赖 10.4。

**问题**：相同条件计算与相似跨步特征如何区分；复用误差怎样传播；维护、回退和质量何时抵消加速。

**对象与源码**：主例 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 TeaCache/Cache-DiT，另检查 Z-Image-Turbo 的支持路径；[SGLang 缓存文档](https://docs.sglang.io/docs/sglang-diffusion/caching-acceleration)、[Cache-DiT](https://github.com/vipshop/cache-dit)、[TeaCache](https://arxiv.org/abs/2411.14324)。Cosmos3-Edge 用于条件/KV 与跨步近似的结构比较。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写带 cache key、更新、失效、阈值和回退的 feature cache；保存每层每步输入/输出；区分确定条件缓存与近似残差复用 | 无缓存参照与精确复用对拍；近似误差不能按精确等价验收 |
| B | 固定初始 latent、solver、每步随机噪声与 prompt，阈值=0/0.05/0.1/0.2；8 题调参、24 题独立评测，比较 step/block 缓存与 TaylorSeer 类预测 | 记录真实 skipped layers/NFE、维护/搬运/回退、峰值、完整时间、质量和退化样例，不只报 hit rate |
| C | 查真实支持和 no-op：当前文档中 Wan2.2 TeaCache 系数未校准可能不生效，不能沿用 Wan2.1 结果；检查 CFG 双分支、FSDP 与 batching 兼容 | 每个开关都对应执行 trace 和改变的工作量；配置被接受不算机制运行 |
| D | 在 Cosmos3-Edge 中分离固定理解侧缓存与变化 latent 的跨步近似，按同一 I2V 配方测全请求收益与质量 | 解释误差对后续轨迹的累积；不同模型阈值不相互移植，组件误差小不保证世界预测正确 |

**交付**：新增 `labs/L10/feature_cache_reference.py`、`cache_quality_bench.py`，缓存事件、逐步误差、独立生成样本及质量—时间—内存曲线。

**反例与边界**：相似不是相同；复用率高也可能因维护/搬运更慢；少量样本不能给普遍安全阈值。

<a id="c-10-6"></a>
## 10.6 VLA 与动作生成

**依赖**：10.1、10.2、1.7。

**问题**：模型动作张量对应哪个具身和坐标；flow/AR 生成的系统成本怎样不同；采样和执行频率怎样影响质量与闭环。

**对象与源码**：主例 [openpi DROID](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md) 的 `pi05_droid`（checkpoint `gs://openpi-assets/checkpoints/pi05_droid`）；`pi0_fast_droid` 作 AR 动作 tokenizer 对照；前沿扩展 [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)，区别于通用世界模型 checkpoint。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 DROID 记录的图像、state、指令和动作，解析 normalizer、坐标、单位、horizon、padding 与 action mask；实现官方物理动作转换 | 单条观测逐张量对拍，模型内部维度与具身维度分开；各模型必须先映射到可比动作定义 |
| B | flow 头用原生设置为基线，合法配置扫描 NFE=4/8/16；AR 路线追踪 FAST tokenizer、动作 token 数与解码，固定观测和有效 horizon | 报完整视觉/语言编码、动作头、解码与动作误差；不只测 denoiser kernel |
| C | 保留模型原生输出 horizon，消费者每次执行前 1/4/8 个动作，频率=5/10/20 Hz；在 1.7 harness 注入延迟与旧输入 | 记录动作年龄、deadline、连续性、离线误差和有效动作率；消费 chunk 与模型生成 shape 不混用 |
| D | 为 Policy-DROID 运行独立合法动作样例，比较通用世界生成与策略 checkpoint 的训练目标/输出接口；具备对应机器人或可信任务仿真后再做闭环 | 离线结果、仿真结果和真实机器人结果分开；没有闭环条件不推断成功率 |

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

**依赖**：M1、M2；完整案例分别使用 5.5 与 10.4 的工件。

**问题**：论文主张如何映射到代码与实验；模型/代码更新如何改变可比性；复现怎样产生支持、限制或负面结果。

**对象与源码**：固定案例为 DFlash 的并行草稿与 Cosmos3-Edge 的 I2V 更新配方；原始论文、作者 repo、模型卡、变更说明与实际 engine 实现。搜索使用 Exa MCP 或命令行，不使用 subagent 或内置 websearch。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对每个案例检索原始主张、前置方法、代码、模型 revision 与支持矩阵；记录哪些数字来自作者、哪些可由本机验证 | 生成主张→机制→所需证据表；摘要或模型名字不计作机制解释 |
| B | DFlash 对齐 target/draft、mask、验证与回滚；Cosmos 对齐旧/新权重、scheduler、帧数与精度，先做最小正确性实验 | 引用 5.5/10.4 的既有工件，不重复采同一结果；新版本必须另存配置和输出 |
| C | 预先写出可能推翻主张的对照：普通 decode 完整成本、未融合草稿、旧配方/新配方、阶段成本与任务质量 | 保留失败、减速和未解释趋势；不能以组件速度或作者 H100 数字替代本机端到端结果 |
| D | 由原始工件一键重算表图，说明测量不确定性、硬件边界与尚未验证的实验；依据新版本变化决定是否需要重新实验 | 读者能独立复现限定范围，并指出结论在哪个条件下不成立；不以引用数量或最新模型数量衡量深度 |

**交付**：新增 `labs/M/reproduction_manifest.py` 与两个固定案例的配置/重算入口；正文保留具体机制和实验设计，不写检索过程流水账。

**反例与边界**：代码 main 和模型 main 可能独立变化；名称相同或参数开关存在不代表复现实验相同。
