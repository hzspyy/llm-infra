# 已完成部分的修订计划

本文件收录当前已有正文的 52 个模块，并逐章规定修订、深入实现和补测任务。“已完成部分”用于区分已有内容与尚未编写内容，不表示整章已验收。实际进度、工件和未解释结果只记录在 [STATUS](../../STATUS.md)；章节范围与依赖以 [outline](../../outline.json) 为准。尚无正文的模块见 [未完成部分的执行计划](pending.md)。

各章统一采用“问题、对象与源码、执行步骤、交付与验收、反例”的粒度。表中 A/B/C 等按顺序执行；后续综合任务注明前置。先复用能支持同一结论的现有工件，补采缺少的状态与对照，不重复制造相同结果。原始 results 只读。源码入口在执行前固定 commit 和真实行号；新脚本列为待实现交付，不作为现有命令使用。通用采集协议、资源约束与批次见活计划，写作和测量分别遵循 [章节规范](../chapter-guidelines.md) 与 [实验规范](../experiment-guidelines.md)。

<a id="c-0-0"></a>
## 0.0 最小完整模型（上）

**依赖**：Python 与矩阵乘基础。

**问题**：参数对象怎样组成模型；一次前向如何产生条件分布；KV 增量计算如何与完整前缀对齐。

**对象与源码**：复用 `labs/L0/tiny_lm.py`、`walk_generate.py`；真实对照固定 `Qwen/Qwen3-1.7B` 的 Transformers `modeling_qwen3.py`，沿 embedding、RMSNorm、Q/K/V、MLP、lm_head 定位。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用 B=2、S=5、d=16、H=4 的模型逐算子打印 shape/stride；手算一个 attention 行，说明缩放、因果 mask 和残差。加入独立参数表与 tied weight 对照 | 修订正文的张量流，参数计数与实际对象一致；手算概率与 FP64 参照对齐 |
| B | 实现完整前缀与逐 token KV 两种前向；扫描 S=1/2/7/16，分别去掉位置和缩放，比较 logits、最大误差与 argmax | mini 不靠生成句子判断正确性；同一模型两条路径逐步对拍，保留失败输入 |
| C | 把同一检查映射到 Qwen3 的 GQA、SwiGLU、RoPE，输出“玩具步骤→真实符号→新增状态”表；进阶实现交给 4.1 | 初学者能解释结构差异而不必先读整个推理引擎；原始配置和真实张量片段可定位 |

**反例与边界**：softmax 输出是给定前缀的分布；随机初始化模型输出流畅与否不是机制正确性判据。

<a id="c-0-0b"></a>
## 0.0b 最小完整模型（下）

**依赖**：0.0。

**问题**：标量 loss 如何产生参数梯度；保存值和共享参数如何影响反向；梯度累积何时等价。

**对象与源码**：复用 `labs/L0/walk_train.py` 与 `tiny_lm.py`；对照 PyTorch `autograd`、`CrossEntropyLoss`、AdamW；真实训练接口接 7.0b。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手推 softmax-cross-entropy、linear、残差和一个共享权重的梯度；在 FP64 下比较手算、autograd 和中心差分 | 每个梯度能追溯到本轮前向输入；逐 token loss 与有效 token 归一化一致 |
| B | 用长度 3/7 的两条样本对比直接 batch、按样本平均和按 token 累积；记录更新前后的参数、梯度与 Adam 状态 | 展示错误归一化的反例并修正；不把 microbatch 数相同当有效 batch 相同 |
| C | 注入 labels 未移位、detach、原地修改和重复 backward；把相应发现位置映射到 7.0 的 SavedVariable/version counter | 保留实际报错和梯度断点；自测要求解释原因并指出改动位置 |

**反例与边界**：单样本 loss 下降只验证优化路径；泛化和完整训练性能由后续章节承担。

<a id="c-0-1"></a>
## 0.1 从 HTTP 到 SASS

**依赖**：0.0；引擎插桩在 5.11 完成后回填。

**问题**：请求跨过哪些进程与状态边界；Python 算子如何对应 GPU 工作；不同观察层如何关联同一请求。

**对象与源码**：复用 `labs/L0/map_stack.py`、`kernel_trace.py`；Qwen3-1.7B，vLLM `entrypoints/openai`、`v1/engine`、`v1/core/sched`、`v1/worker/gpu_model_runner`；SGLang `TokenizerManager`、`Scheduler`、`DetokenizerManager`。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 保留静态调用地图，新增 request_id、进程、线程、队列事件与一次真实请求的对应；分别记录 prefill 和两个 decode step | 调用关系与实际执行分别给证据；同一 ID 能从 HTTP 追到输出 |
| B | 选择一次 QKV linear 和一次 attention，关联 Python/ATen、runtime API、kernel 与反汇编；记录异步提交和设备完成的区别 | 不要求跨层一对一；正文解释融合、分解及无法直接关联的节点 |
| C | 对比两引擎的进程边界与取消传播，增加“替换 sampler/attention 后端分别改哪里”的具体练习 | 交付事件表和扩展点表；读者能在未知请求故障中选择正确观察层 |

**反例与边界**：hook 未出现某算子不代表没有执行；一张静态调用图不能证明某次请求实际经过它。

<a id="c-0-2"></a>
## 0.2 资源账本与 roofline

**依赖**：0.0。

**问题**：权重、状态和计算如何从配置推导；prefill/decode 的账为何不同；模型预测在哪些条件下失效。

**对象与源码**：复用 `labs/L0/ledger.py`；Qwen3-1.7B 的配置、权重 header；对照 vLLM KV 容量计算与 SGLang memory pool；混合状态扩展使用 5.13 的 RecurrentGemma-2B 与 Qwen3.5。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 逐层计算 attention/MLP/embedding 的参数、FLOP 与 KV；加入 GQA、tied weights、量化元数据和张量并行分片 | 计算表由配置生成并与实际 tensor bytes 对齐；文件字节与设备分配分列 |
| B | 对 B=1/8/32、S=128/2048/8192 给出预测与实测点，分别计算权重、KV、激活、workspace；将 max(compute, memory) 的假设写明 | 交付可重算账本和误差分解；不能用峰值带宽直接宣称真实瓶颈 |
| C | 给 RecurrentGemma 与 Qwen3.5 按层类型增加固定状态项，和全 attention 的长度增长对照；用 4.4 扩展 MoE 的总参数与激活参数 | 学生修改模型类型后仍能产生正确资源项；规模差异与架构差异分别解释 |

**反例与边界**：L2 容量、缓存命中、调度和算子效率可破坏简单 roofline 估计；不把多个独立微基准相加当请求时延。

<a id="c-0-3"></a>
## 0.3 异构 GPU 对照

**依赖**：0.2；诊断复用 1.3、2.6。

**问题**：同一工作在两种 GPU 上为何不同；哪些差异来自硬件而非软件；怎样给出可迁移的选型依据。

**对象与源码**：复用 `labs/L0/bench_phases.py`、`probe_hw.py`；Qwen3-1.7B，crater RTX 5090 D 与 worldvln L40S；逐机固定实际 engine、attention 和 GEMM 后端。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 统一模型 revision、token 输入、精度、输出长度与计时边界；列出驱动、时钟、功耗、软件和 kernel 的不同 | 两机共同配置可复现；无法一致的条件明确列出，不伪装成纯硬件实验 |
| B | 固定 S=2048 扫 B=1/2/4/8/16/32，再固定 B=1/8 扫 S=128/2048/8192；分开 prefill、decode、完整调用和初始化 | 原始样本、阶段时间与资源账共同支撑交叉点；保留不符合峰值比例的点 |
| C | 选一个差异最大的共同 shape，比较实际 kernel、工作集与 CPU 提交；以相同 SLO 下的 goodput 连接 8.5 | 结论限于相同任务；跨机计数器不能替代目标机的瓶颈证据 |

**反例与边界**：跨卡差值不能全部归因带宽；显存容量与短时速度分别参与选型。

<a id="c-0-4"></a>
## 0.4 文本输入

**依赖**：0.0。

**问题**：字节和 token 如何转换；模板为何改变模型输入；padding 与位置如何保持一致。

**对象与源码**：复用 `labs/L0/bpe_from_scratch.py`、`tokenizer_tour.py`；Qwen3-1.7B 的 tokenizer.json、chat template 与 Hugging Face tokenizers BPE/added vocabulary；Unigram 用 `google/mt5-small` tokenizer 作结构对照。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用中英文、组合音标、emoji、空白与特殊 token 的固定 30 条文本实现 BPE 训练/编码；打印 UTF-8、预切分、merge 顺序和 ID | 与真实 tokenizer 对照差异来自哪层；不用小语料 merges 冒充完整词表 |
| B | 比较 plain/chat/tool 模板，左/右 padding，重复添加 BOS/EOS；对同一有效文本核对 input_ids、mask、position_ids 和 logits | 模板原文、有效长度和错误输入齐备；token 数变化能逐字节解释 |
| C | 分析 tokenizers 的批处理和预切分入口，比较 BPE 与 Unigram 的搜索目标；在 5.11 中复用长度/批量扫描 | 算法差异通过一条歧义切分示例展示；tokenization 性能与模型性能分开 |

**反例与边界**：字符数不是 token 数；Unicode 归一化策略与训练词表不一致可能改变模型输入。

<a id="c-0-5"></a>
## 0.5 文本输出

**依赖**：0.4。

**问题**：过滤顺序如何改变概率；token 如何增量成为 UTF-8；停止条件如何影响最终输出。

**对象与源码**：复用 `labs/L0/sampling_walk.py`；Qwen3-1.7B；vLLM sampler、incremental detokenizer 与 SGLang `DetokenizerManager`，GPU 细节连接 5.9。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用四分类概率和含负值 logits 演算 temperature、top-k/top-p、min-p、惩罚；实现每一步的向量输出 | 给出顺序交换和 ties 的反例；greedy 与随机采样分别定义正确性 |
| B | 构造 UTF-8 在 token 边界断开的中文/emoji，stop string 横跨多个 token，EOS 和 max_tokens 同时满足 | 实现带缓冲的增量 detokenizer；输出字节、停止位置与引擎对拍 |
| C | 增加 tool-call 与 reasoning 分段的真实输出样本，追踪 parser 和 detokenizer 的不同职责；与 5.11 的 SSE 分帧对照 | token、文本 delta、SSE event 三种计数可分别重建；不把空 delta 判为生成失败 |

**反例与边界**：一次 token 不必立即形成可显示字符；固定随机 seed 不保证不同执行后端逐 token 相同。

<a id="c-1-1"></a>
## 1.1 GPU 微架构与内存层次

**依赖**：0.2。

**问题**：延迟与吞吐如何分别测量；容量和并发如何改变访存；硬件约束如何进入 kernel 设计。

**对象与源码**：复用 `labs/L1/mem_hierarchy.cu`、`uvm_probe.cu`；crater sm_120、worldvln sm_89、spark sm_121；[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)、设备属性和实际 PTX/SASS。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 区分 pointer chase 的依赖延迟与独立 load 的吞吐；按 0.25/0.5/1/2/4 倍实测 L2 扫工作集，随机/连续/跨页访问分别运行 | 修订内存层级解释；容量、有效字节、缓存条件与计时方法一一对应 |
| B | 固定总字节扫描 warp 数、blocks/SM 和 stride，加入 shared-memory bank 冲突对照；用 spark 可用计数器复核本机实验 | 输出延迟隐藏与带宽饱和曲线；不可用 metric 不填估计值；跨机证据不代替 crater 归因 |
| C | 将一次 Qwen3-1.7B 的 GEMM/attention 映射到寄存器、shared memory、L2 和 DRAM；以 CUTLASS tile 配置计算资源上限 | 明确 resident blocks 的限制来自哪项资源；自测能预测改变 tile 后的容量变化 |

**反例与边界**：工作集能装进 L2 不等于每次命中；带宽微基准和模型调用的访问模式不同。

<a id="c-1-2"></a>
## 1.2 Tensor Core 指令与架构差异

**依赖**：1.1。

**问题**：mma/wgmma/tcgen05 的操作数归属如何变化；格式、shape 和累加精度如何约束吞吐；消费级与数据中心 Blackwell 有何差别。

**对象与源码**：复用 `labs/L1/tensor_core.cu`、`tcgen05_probe.cu`、`verify_gemm_ceiling.py`；[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 与 CUTLASS sm_89/sm_100/sm_120 实现。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对 FP16/BF16/TF32/FP8 逐项列出输入、累加、tile、lane fragment 和指令；用寄存器片段的已知矩阵验证排列 | 每条指令的 FLOP 计数与实际 SASS 对齐；不能由数据类型推断实际 Tensor Core 路径 |
| B | 独立依赖链数量 1/2/4/8/16、warp 数 4/8/16 扫吞吐；同时记录时钟和编译器优化，区分 latency 与 issue throughput | 校正周期计量与 FMA 计数；解释既有 FP8 发射率与 GEMM 差异所需的具体证据 |
| C | 解析 block-scaled MMA 的 scale 布局，并读 FA4/CuTe 的 TMEM 和 2-CTA 示例；crater 运行其实际支持的 sm_120 路线 | 产出硬件×指令×实现矩阵；sm_100 的 TMEM 机制完成源码分析，未有设备时性能保持 UNVERIFIED |

**反例与边界**：Blackwell 名称不能证明支持同一 PTX 指令；两个输入字节减少一半不能自动解释端到端加速未翻倍。

<a id="c-1-3"></a>
## 1.3 主机、NUMA、PCIe 与 DMA

**依赖**：1.1。

**问题**：数据实际经过哪些链路；pinned memory 与异步拷贝改变什么；NUMA 和重叠何时主导整体时间。

**对象与源码**：复用 `labs/L1/host_transfer.py`、`numa_bandwidth.py`、`p2p_matrix.py`；PyTorch pin-memory worker、CUDA memcpy/stream/event；worldvln 的实际拓扑。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 绘出 CPU node、内存、PCIe root complex 与 GPU 对应；同线程分别绑定近/远 NUMA 内存，核对实际页归属 | 保存拓扑、绑定和页面记录；把链路理论上限与实测搬运分开 |
| B | pageable/pinned、H2D/D2H/P2P，尺寸 4 KiB 至 256 MiB 按 4 倍递增；比较分配在计时内外与首轮/热态 | 输出启动成本、稳态吞吐和 pinning 成本；不把 API 返回当设备完成 |
| C | 实现双缓冲输入流水，与串行 copy+GEMM 比较，chunk=1/4/16/64 MiB；固定总工作量，记录依赖 event | 计算与传输重叠由同次 trace 支撑；复用 buffer 前等待正确事件，连接 7.1 的训练数据流水 |

**反例与边界**：P2P capability 为真不保证走 NVLink；PCIe 代际、lane 数、单向和双向带宽分别计量。

<a id="c-1-4"></a>
## 1.4 驱动与操作系统

**依赖**：1.3。

**问题**：Python、runtime、driver 和设备边界在哪里；异步错误在哪暴露；系统开销如何进入服务时延。

**对象与源码**：复用 `labs/L1/driver_path.py`、`syscall_ladder.sh`、`syscall_worker.py`；PyTorch CUDA 初始化、CUDA driver/runtime API、vLLM worker 初始化与 CUDA Graph replay。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 分别采 import、context 初始化、分配、首次 kernel、热 kernel 与同步；将 syscall、runtime 和 GPU 事件对应 | 启动与稳态成本分列；不用 syscall 次数直接估计 GPU 执行时间 |
| B | 独立子进程中注入非法访问、错误 launch、版本不兼容；比较同步前后报错点和退出码 | 有界结束并回收资源；解释触发点、发现点和可恢复边界 |
| C | 对同一组微小 kernel 比较逐次提交、批量提交、CUDA Graph；记录 CPU 调度间隙与 GPU 空闲 | 将 Python/GIL/线程调度、driver 与设备执行成本分开；连接 5.4 的真实图路径 |

**反例与边界**：`stime=0` 不证明没有驱动开销；不修改共享驱动设置或通过重启机器制造实验条件。

<a id="c-1-5"></a>
## 1.5 网络与存储底座

**依赖**：1.3；跨实例综合依赖 6.4。

**问题**：权重加载和 KV 传输的路径如何不同；缓存和介质怎样影响带宽；RDMA/GDS 是否真的参与执行。

**对象与源码**：复用 `labs/L1/storage_path.py`；safetensors mmap、vLLM model loader；[NIXL](https://github.com/ai-dynamo/nixl)、[Mooncake](https://github.com/kvcache-ai/Mooncake) 的 registration、transfer、completion 接口。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 将 checkpoint read/mmap、page fault、CPU 解码、H2D 分开；使用独立数据文件控制冷热，保留读取字节和介质信息 | 不通过共享机器全局 drop_caches 清缓存；“冷态”必须有实际证据 |
| B | 在已允许的项目端点上比较 TCP 与可用传输后端，块大小 64 KiB/1/16/64 MiB，并发 1/4/16；先验证收发内容和完成语义 | 实际选中的 transport、链路与 buffer 注册可追踪；库存在不当作 RDMA 已启用 |
| C | 为 6.4/8.6 提供注册、传输、等待、释放的小型适配器；注入接收端中断和重复完成事件 | 内容哈希、句柄生命周期与失败清理一致；报告注册成本是否被摊销 |

**反例与边界**：磁盘吞吐、网络吞吐和端到端加载速度不是同一测量；未具备 RDMA/GDS 条件时保留源码及本地协议对照。

<a id="c-2-0"></a>
## 2.0 张量、对象与存储

**依赖**：0.0。

**问题**：逻辑张量怎样寻址实际字节；Module/Parameter/buffer 如何拥有状态；alias 与版本计数怎样影响计算。

**对象与源码**：复用 `labs/L2/tensor_anatomy.py`、`mini_tensor.py`；PyTorch `c10/core/TensorImpl.h`、`StorageImpl.h`、`torch/nn/modules/module.py`；Qwen3-1.7B 的 tied/untied 参数配置。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 shape/stride/offset 寻址，覆盖 transpose、slice、expand、reshape 和重叠 view；打印对象 ID、data_ptr、storage_ptr、版本计数 | 区分对象相同、存储共享和数值相同；已知索引逐字节对拍 |
| B | 在共享 Parameter、普通 tensor 属性和 register_buffer 三组模型中比较 parameters/state_dict/to(device)/optimizer 行为 | 解释状态为何被保存、迁移或更新；mini 复现注册与递归遍历 |
| C | 用同一别名/原地修改程序比较 eager、functionalization 和 compile，连接 2.7 的副作用处理 | 提供 mutation 前后状态对拍与失效反例；不把所有 reshape 都描述成零拷贝 |

**反例与边界**：释放一个 Python 引用不等于设备内存可重用；生命周期由 2.0c 继续验证。

<a id="c-2-0b"></a>
## 2.0b 算子分发

**依赖**：2.0。

**问题**：Python 调用如何到达 kernel；dispatch key 与 dtype 分支有何区别；redispatch 如何避免重复包装。

**对象与源码**：复用 `labs/L2/dispatch_path_demo.py`、`mini_dispatcher.py`、`trace_linear_dispatch.py`；PyTorch `native_functions.yaml`、torchgen、`c10/core/Dispatcher.h`、`ATen/native/Linear.cpp`。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 `torch.nn.functional.linear` 找到 schema、生成绑定、注册表、native 分支和 CPU/CUDA kernel；对 add 重复同一定位步骤 | 交付两条有真实 file:line 的调用链；不能凭 hook 缺项否定 aten.linear 的存在 |
| B | 扩展 mini 分发表，加入 Autograd、Autocast 和 Backend 包装、线程局部排除集合与 redispatch；用递归错误作反例 | 输出 key 选择与包装顺序；CPU/CUDA、grad/no_grad、autocast 各路径与框架对应 |
| C | 对同一程序并置 TorchDispatchMode、FX/export、profiler 的观测，说明分解发生的位置；接入 2.8 的自定义算子 | 图边、注册事实与实际执行分开；学生能定位新算子的扩展入口和错误层次 |

**反例与边界**：算子分解、kernel 融合和 Python 包装是不同机制；不能靠 op 名称数量判断性能。

<a id="c-2-0c"></a>
## 2.0c 内存、流与异步生命周期

**依赖**：1.4、2.0b、2.1。

**问题**：引用释放、allocator 可复用与 GPU 完成如何区分；wait_stream 与 record_stream 分别保证什么；碎片与图池如何影响容量。

**对象与源码**：复用 `labs/L2/memory_lifecycle_basic.py`、`memory_cross_stream.py`、`memory_fragmentation.py`、`memory_cuda_graph.py`、`mini_block_allocator.py`；PyTorch `CUDACachingAllocator.cpp`、CUDA event 与 graph pool。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 让最小块池记录 owner stream、pending event、split/merge 和复用；输出三个生命周期时刻，并对照真实 allocator snapshot | allocated、reserved、active、inactive split、pending 的定义不混用；每次复用有完成依据 |
| B | 两条流使用确定性依赖与延迟 kernel，分别删去 wait_stream 或 record_stream；保留读写值、事件和地址复用 | 分别展示执行顺序错误和存储过早复用，不能以“偶尔没有报错”证明安全 |
| C | 固定 200 个请求的长度序列，比较规则/交错尺寸、native allocator/expandable_segments、graph on/off | 交付真实分配轨迹、峰值与碎片定义；先验证同等活跃数据量再比较，不将 reserved-allocated 全称为碎片 |

**反例与边界**：模拟 KV 分配不能代替真实引擎变长轨迹；后者在 5.2/5.4 的共同负载中回填。

<a id="c-2-1"></a>
## 2.1 CUDA 执行与 CUDA Graph

**依赖**：1.1、2.0b。

**问题**：grid/warp/lane 如何分工；同步和可见性如何影响结果；graph capture/replay 固定了哪些条件。

**对象与源码**：复用 `labs/L2/exec_model.cu`、`cudagraph_modes.py`；CUDA programming guide、PyTorch CUDA Graph、vLLM `CUDAGraphMode` 的真实 dispatch。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 打印 thread/block/SM 映射；比较 31/32/33、127/128/129 个元素和 SM 整数倍两侧的 block 数 | 所有边界元素结果对齐；解释 warp 分歧、尾块与 wave 数而非只给吞吐表 |
| B | 为 shared-memory producer/consumer 构造缺 barrier、错误 mask、跨流无依赖的失败小例 | 保存 sanitizer/数值错误与修复后的同输入结果；说明 block barrier 的作用域 |
| C | 对固定指针与动态尺寸程序捕获/replay，分别改输入值、地址、shape 和分配；测首次捕获、热 replay 与显存 | 解释哪些变化合法、哪些需要重捕获或退回；真实引擎图桶综合由 5.4 完成 |

**反例与边界**：graph replay 计数不是 kernel 计数；观察 launch 减少不能证明 GPU 计算量减少。

<a id="c-2-2"></a>
## 2.2 编译工具链

**依赖**：2.1、1.2。

**问题**：各编译阶段做什么；PTX 与 SASS 如何对应；源码优化怎样改变寄存器、spill 和真实指令。

**对象与源码**：复用 `labs/L2/toolchain_tour.sh`、`parse_ptxas.py`；nvcc/ptxas/cuobjdump，Triton compiler 与 NVIDIA CUTLASS；工具版本使用环境记录中的匹配组合。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对同一 SAXPY/归约/GEMM 保留预处理、PTX、cubin、SASS 与编译日志；分别改变优化级别、目标架构和 fast-math | 每个制品有生成命令与哈希；源文件语句对应到实际指令或被优化删除的位置 |
| B | 扫 unroll 和寄存器限制，保留寄存器、spill、local memory、occupancy 与时间 | 用访存/指令证据区分性能下降原因；不按寄存器数单独排序优劣 |
| C | 导出一个 Triton 与一个 CuTe-DSL kernel 的中间表示，和 CUDA 实现对比 lowering；构造 PTX/driver 不兼容用例 | 编译与加载错误各有原文；解释 JIT/cache 命中对首次调用的影响 |

**反例与边界**：看到 PTX `mma` 不足以确认最终指令、布局或运行路径；需对应实际加载的二进制。

<a id="c-2-3"></a>
## 2.3 Memory-bound 算子

**依赖**：2.1、1.1。

**问题**：归约的同步与流量如何计算；向量化和融合减少了什么；数值顺序怎样约束优化。

**对象与源码**：复用 `labs/L2/reduce_ladder.cu`、`torch_reduce_ref.py`；PyTorch reduce、Triton RMSNorm/softmax；真实 shape 从 Qwen3-1.7B 的 RMSNorm 输入采集。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从单线程、atomic、block reduction 到 warp shuffle 和两阶段归约，每一步只改变一种机制；推导读写量与同步次数 | 完整 CUDA mini 与 FP64 参照对拍，奇数长度和极端抵消输入都有结果 |
| B | 行宽 128/1024/4096/14336，行数 1/8/128/2048；工作集覆盖 L2 两侧，比较向量化、累加精度和单/两阶段 | 报真实字节、时间、误差；保留 launch 受限与带宽受限两类结果 |
| C | 把 add+RMSNorm 或 softmax 融合成一个 kernel，比较分开实现、Triton 和引擎实际 op | 输入、输出及 alias 约束一致；解释少落一次中间张量的收益与扩展限制 |

**反例与边界**：不同归约树通常不能要求浮点逐位相同；超过 DRAM 理论带宽先检查计量和缓存。

<a id="c-2-4"></a>
## 2.4 GEMM 优化阶梯

**依赖**：2.2、2.3、1.2。

**问题**：tile 如何分配给线程和寄存器；加载与计算怎样重叠；不同 M/N/K 下哪项资源限制吞吐。

**对象与源码**：复用 `labs/L2/gemm_ladder.cu`；cuBLAS/CUTLASS 为性能参照；[CuTe-DSL GEMM 教程](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py)与 [async pipeline](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/notebooks/async_pipeline.ipynb) 为流水分析入口。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 保留 FP32 与 BF16 两条阶梯；逐步加入 shared tile、register tile、swizzle、向量化，手画线程到矩阵片段映射 | 每步有同输入对拍、资源计数和单因素变化；覆盖不整除 tile 的 M/N/K |
| B | 在支持的 `mma.sync` 路线实现 `cp.async` 双缓冲，扩展 stage=2/3/4；写出 producer acquire/commit、consumer wait/release 与 buffer phase | 交付完整可读 kernel、流水事件和缺 wait/提前覆盖的反例；不能只把双缓冲写成建议 |
| C | 扫 M=1/8/32/128/512/2048，K/N 取 Qwen3-1.7B 的 q_proj 和 down_proj 实际维度，再加 4096 方阵；采用驻留与大于 L2 的轮转权重 | cuBLAS、CUDA mini 与 CuTe 对比均含布局转换和必要 epilogue；用实际 kernel 解释形状拐点 |
| D | 解析 sm_100 的 TMA/TMEM/2-CTA 与 sm_120 对应路径，比较 persistent scheduling、warp specialization 和寄存器分配 | 在本机支持路径运行；异构路径完成源码/资源模型，性能条件单列；形成接入 3.2/4.3/4.4 的接口 |

**反例与边界**：不能把数据中心 Blackwell 教程直接视为 RTX 5090 可运行代码；高 TFLOP/s 不能代替端到端收益。

<a id="c-2-5"></a>
## 2.5 CUDA 与 kernel DSL 对照

**依赖**：2.4。

**问题**：各 DSL 提供哪些控制边界；抽象怎样落到实际指令；开发与运行成本如何公平比较。

**对象与源码**：复用 `labs/L2/dsl_shootout.py`；CUDA、[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)、[CuTe-DSL](https://github.com/NVIDIA/cutlass)、[TileLang](https://github.com/tile-ai/tilelang)、[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 GEMM、RMSNorm 和分段 gather/reduce 三类算子；为五种实现写相同输入、输出、误差和 epilogue 契约 | 每条路线有最小实现与正常/边界用例，不能只比较一段 SAXPY |
| B | 复用 2.3/2.4 的代表形状；同时记录冷编译、缓存命中、稳态、代码行数、手工资源选择与生成 IR/SASS | 不把编译开销混入稳态；不统一强制同一 tile 而掩盖各实现可用优化，另提供受控 tile 对照 |
| C | 修改一个布局和一个融合需求，记录需要改变的 host/device 代码、barrier 与 autotune 配置；定位各自编译失败信息 | 给出以扩展任务为依据的设计比较；架构支持不足的路线说明具体限制，不用 Python 模拟填性能格 |

**反例与边界**：语法短不能证明生成代码高效；单一 GPU 和 shape 的排名不推广到全部 DSL。

<a id="c-2-6"></a>
## 2.6 GPU 性能分析

**依赖**：2.2、2.3。

**问题**：不同观测工具测量什么；怎样区分多个性能解释；profiling 扰动怎样控制。

**对象与源码**：复用 `labs/L2/profiling_tour.py`、`mini_profiler.cu`、`ncu_ladder.sh`；PyTorch profiler、nsys、ncu；目标案例取 2.4 GEMM 与 5.4 CUDA Graph。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 在固定 workload 中制造 CPU 提交、带宽、同步、occupancy 四种瓶颈；先给竞争解释，再规定区分所需事件 | 一份墙钟、runtime 和 GPU 时间线能关联同一次运行；父子事件不重复累加 |
| B | 用 nsys 定位空隙、ncu 检查支持的计数器，mini profiler 采 block 起止；对照未插桩时间 | 每个因果结论至少有单因素对照；spark 上的计数器只解释 spark 对应运行 |
| C | 从一个异常点生成最小复现：例如 tile 边界两侧或图桶切换；保存输入、kernel 名、资源及频率，关闭无关变化 | 产出“发现→竞争假设→区分实验→支持范围”案例；没有解释完的趋势保留待验证 |

**反例与边界**：活动 warp 多不必更快；GPU event 累计时间不等于请求墙钟占比；对无权限 metric 不给伪造读数。

<a id="c-2-6b"></a>
## 2.6b 算子图与编译产物

**依赖**：2.0b、2.3、7.0。

**问题**：不同阶段图表示了什么；前后向如何分图；生成 kernel 如何对应原程序的状态和副作用。

**对象与源码**：复用 `labs/L2/graph_anatomy.py`；PyTorch Dynamo、AOTAutograd、Inductor；vLLM piecewise compile 的 splitting ops 与编译 wrapper。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 使用含 view、原地 add、linear、归约和共享参数的同一程序，导出 eager/FX、Dynamo、functionalized、joint、forward/backward 图 | 每阶段记录 shape、alias、mutation、saved tensors；不能靠节点总数代替语义映射 |
| B | 对照 eager 与 compile 的输出、输入修改、梯度和 RNG；追踪一个被分解 op 和一个跨图保存值 | 图转换前后状态一致，差异可定位到具体 pass；生成代码能对应 kernel trace |
| C | 把 attention 标为 splitting op/opaque 边界，对照一次实际 vLLM 模型图；区分完整图、分段图和 CUDA Graph capture | 交付原始图、代码与调用关联；学生可解释每种“图”保存的内容与生命周期 |

**反例与边界**：export 图不等于设备执行图；编译图数量、graph replay 数和 kernel 数分开统计。

<a id="c-2-7"></a>
## 2.7 编译器内部与部署

**依赖**：2.6b、2.0c。

**问题**：guard 如何决定复用；图怎样 lower 成循环和内存计划；静态/动态及 AOT 部署分别限制什么。

**对象与源码**：复用 `labs/L2/compiler_stack.py`；PyTorch `_dynamo/guards.py`、ShapeEnv、functionalization、AOTAutograd、`_inductor/ir.py`、`scheduler.py`、codegen；[torch.compile AOT](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_compile.html)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 以 2.6b 共同程序跟踪一个 guard、一次 mutation 消除、一次 reduction lowering 和 buffer 复用；实现一个有合法性检查的 FX 变换 | 输出变化前后 IR 与状态对拍；源码分析进入选择规则，不能止于打印生成代码 |
| B | 重做形状序列 2→大尺寸、固定大尺寸、交错尺寸，记录每次编译与缓存命中、size hints、grid 和耗时；比较默认与 autotune 候选实际选择 | 将已有极端退化限制到可复现触发条件；区分缓存上限、guard 失败和低效 kernel，保留未解释结果 |
| C | 用 Qwen3-1.7B 的一个 block 比较 eager、compile、export+AOTInductor；在支持接口的固定版本增加 `aot_compile()` 保存/加载对照 | 分开编译、加载和稳态；说明 AOTInductor 的非 Python 部署与 compile AOT 的 Python callable 差异 |
| D | 注入 data-dependent 控制流、别名、副作用和 opaque op；运行 AOT 制品的有效/越界 shape，追踪约束发现位置 | 交付制品、约束与失败输入；新 API 不假定存在于既有 PyTorch 2.13，独立环境记录版本 |

**反例与边界**：max-autotune 负结果不能泛化到所有形状；“归约必然阻止后续融合”需按具体 lowering 与调度判断。

<a id="c-2-8"></a>
## 2.8 自定义算子接入

**依赖**：2.0b、2.2、2.7、7.0。

**问题**：裸 kernel 缺少哪些框架语义；fake/meta 与 autograd 如何注册；opaque 边界如何影响融合和部署。

**对象与源码**：复用 `labs/L2/custom_op.py`；PyTorch `torch.library`、custom_op、register_fake、register_autograd、opcheck；以 fused add+RMSNorm 为具体算子。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 定义 schema、alias/mutation、CPU/CUDA 实现、fake shape 函数；输入覆盖连续、转置、空 tensor 与非法 dtype | 对照 PyTorch 分解参照，opcheck 与手工状态检查均通过；错误在契约规定位置暴露 |
| B | 推导 backward，FP64 gradcheck/gradgradcheck；比较 eager、compile、CUDA Graph、export 路径 | 梯度、输入副作用和数值容差一致；明确是否支持二阶梯度 |
| C | 比较 opaque kernel 与可分解实现的前后算子融合，加入动态 batch 和 graph capture；为 Qwen3 block 接入一次替换 | 报完整 block 的时间与资源；新增后端的注册、布局、梯度和部署改动可逐文件列出 |

**反例与边界**：能在 CUDA 上返回正确值不代表支持 compile/autograd；meta 实现不得读取真实数据。

<a id="c-3-1"></a>
## 3.1 Attention 数学与内存

**依赖**：0.0、2.0、2.3；反向验证接 7.0。

**问题**：online softmax 为什么等价；状态怎样跨块合并；前后向怎样避免物化完整分数矩阵。

**对象与源码**：复用 `labs/L3/online_softmax.py`、`attention_memory.py`；PyTorch SDPA、[FlashAttention](https://github.com/Dao-AILab/flash-attention) 的 reference 与 CUDA/Triton 实现。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 m/l/O 三个统计量推导分块更新、重缩放和合并；用块长 1/3/16 验证顺序与不同分块，加入极值和全 mask 行 | FP64 参照及逐块状态可检查；说明有限精度下的容差和全 mask 输出约定 |
| B | 实现真正分块的 GPU attention，和显式 QK→softmax→PV、SDPA math/flash 对照；S=128/1024/4096/16384、D=64/128 | 保存实际 kernel、临时内存与流量模型；Python 中间张量模拟只用于语义，不能替代融合性能 |
| C | 推导 softmax backward 与重算策略，在小矩阵上对拍梯度；用 Qwen3-1.7B 的真实 Q/K/V 比较 causal/GQA 布局 | 交付正确性、峰值和完整调用时间；机制说明能支撑 3.2 的线程和流水分析 |

**反例与边界**：数学等价不保证逐位相同；没有 S² 中间矩阵不代表 attention 在所有形状下都是算力受限。

<a id="c-3-2"></a>
## 3.2 FlashAttention 代际演进

**依赖**：3.1、2.4、2.6。

**问题**：每代改变了哪项资源瓶颈；线程、warp 与 CTA 如何分工；新近似计算如何保持所需精度。

**对象与源码**：复用 `labs/L3/fa_generations.py`、`exp_throughput.cu`；FlashAttention 的 FA2/FA3/FA4，重点阅读 [FA4 论文](https://arxiv.org/html/2603.05451)；消费级低精度对照用 [SageAttention3](https://papers.neurips.cc/paper_files/paper/2025/file/4db397e0f760cc573c681e81a01a3dba-Paper-Conference.pdf)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 重建 FA1/FA2 的 Q/KV 循环、split-Q/split-K 与 warp 共享状态；改变 B×H 和 query tile，使用实际 launch grid 验证并行来源 | 输出算法/线程布局与真实 kernel 对应；吞吐变化只作为辅助，不能反推唯一实现 |
| B | 为 FA3 画 producer/consumer、TMA、wgmma 和 ping-pong 时间线；为 FA4 推导条件 rescale、软件 exp、TMEM 与 2-CTA 数据流 | 从固定源码标出 barrier、phase 和寄存器归属；用小数值程序验证近似和 rescale 条件 |
| C | 在可运行设备上对共同 S/D/B/causal 形状比较实际后端；crater 运行 FA2 与 SageAttention3 的支持路径，FA3/FA4 性能需要对应支持设备 | 保存 dispatch 结果和资源矩阵；SageAttention 的量化误差、模型质量与速度一起报告 |
| D | 将优化接到同一 Qwen3 或 Wan attention shape，检查总体时间中 attention 的比例；核对真实模型允许的 mask/layout | 组件提升和模型完整调用提升分别成立才给端到端结论；没有目标设备不省略机制分析 |

**反例与边界**：sm_120 不具有 sm_100 的全部能力；消费级 SFU 测量不能证实 B200 的流水瓶颈。

<a id="c-3-3"></a>
## 3.3 Decode attention、分页与 MLA

**依赖**：3.1、2.4。

**问题**：GQA 怎样改变字节和复用；分页如何进入 kernel 寻址；MLA 的吸收与重建为何具有不同成本。

**对象与源码**：复用 `labs/L3/decode_attention.py`；[FlashInfer](https://docs.flashinfer.ai/) 的 paged decode 与 MLA wrapper、vLLM attention backend；普通模型 Qwen3-1.7B，MLA 结构参照 `deepseek-ai/DeepSeek-V2-Lite`。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 推导 MHA/GQA/MQA 的读写和算术强度；实现 split-K 的局部 m/l/O 与最终归并，打印 grid 与 workspace | 连续小张量对拍；改变 split 数不会改变有效序列边界和 mask |
| B | 用真实 paged kernel 直接读随机 block table，page=16/32/64，S=page−1/page/page+1/2048/8192/32768；与连续 reference 对拍 | 包含重排、尾页、不同请求长度和重复引用；保留 Python gather 作为额外搬运对照，不能替代分页结果 |
| C | 从 DeepSeek-V2-Lite 单层权重提取低秩 latent、RoPE 分支与矩阵吸收公式；实现重建 K/V 和吸收两种路径，再对拍可用 MLA backend | 逐张量 shape/字节与输出可核对；完整 BF16 模型优先按 worldvln 48 GB 单卡预算，kernel 支持另核对 |
| D | 在同输入下扫描 heads、KV heads、batch 和上下文，记录 split、缓存布局和真实临时内存；为 4.3 的 KV 量化提供已验证接口 | 形成分页/连续/MLA 的成本模型和适用范围；未运行的硬件专属路径单列 |

**反例与边界**：固定工作量下的权重、KV 与调度成本分别核算；不能由 dtype 减半推出完整 decode 时间减半。

<a id="c-3-4"></a>
## 3.4 稀疏、线性与混合架构

**依赖**：3.1、4.1、4.2。

**问题**：窗口/稀疏规则保留哪些依赖；递推如何并行计算；不同状态容量如何影响真实检索和长上下文任务。

**对象与源码**：复用 `labs/L3/sparse_linear.py`；[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 的 `gated_delta_rule`、[Qwen3.5 模型实现](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md)；真实主例 `Qwen/Qwen3.5-4B`，对照已有 RecurrentGemma-2B 与全 attention Qwen3-1.7B。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 保留 ELU+1 核作为特定例子；增加 gated delta rule 的逐 token 更新、门控与 chunkwise 等价推导；用 FP64 顺序参照对拍 chunk=16/32/64/128 | 同一种算子的不同实现数值对齐；不要求不同架构和不同训练模型输出相同 |
| B | 分别检验 sliding-window 的窗口边界、sink 保留策略、RoPE 插值/频率缩放；实际输出 mask、position 与状态 | 对 S=W−1/W/W+1 的值和状态进行检查；sink 的观察与导致 sink 的机制分开 |
| C | 对 Qwen3.5 确认真实 linear/full 层列表、conv 与 recurrent state；扫描 S=512/2048/8192/16384、B=1/4，记录 chunk kernel 与 fallback | 状态增长与运行路径逐层对应；可选依赖缺失引起的 fallback 必须可见 |
| D | 固定 RULER 检索任务生成器、样本 ID、长度和距离；比较各模型原生配置及受控窗口，报告正确率、吞吐、峰值状态 | 单个合成检索反例不泛化为所有线性架构；质量差异不能仅归因于某一个算子 |

**反例与边界**：全量 cumsum 物化 S 份状态不能代表高效递推 kernel；混合架构不能按全部层都持有 KV 估算。

<a id="c-4-0"></a>
## 4.0 Checkpoint 格式与加载

**依赖**：0.0、2.0。

**问题**：文件字节如何变成参数；命名与分片如何对应模型；量化和绑定权重怎样改变加载路径。

**对象与源码**：复用 `labs/L4/checkpoint_formats.py`、`inspect_safetensors.py`；[safetensors](https://github.com/huggingface/safetensors)、Transformers model loading、vLLM weight loader；Qwen3-1.7B 与既有 Qwen2.5-1.5B AWQ/GPTQ。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写长度/header/offset 解析器，验证 byte range、dtype、shape 和 index；随机抽取 embedding、QKV、MLP 的头尾元素 | 与 safe_open 对拍；截断、重叠 offset、缺分片和错 dtype 都有明确报错 |
| B | 跟踪 checkpoint 名称→模型参数→融合 QKV/分片→设备 tensor，检查 tied weights 的指针与文件重复存储 | 文件字节、CPU 常驻、GPU 分配分别测量；参数身份与值一致 |
| C | 解析 AWQ/GPTQ 的 qweight/qzeros/scales/g_idx 与 compressed-tensors 元数据，再追到加载后 repack；为 8.7 采加载阶段事件 | 说明磁盘布局与执行布局差异；完成一次 round-trip 或可信加载器的逐元素对照 |

**反例与边界**：config 声明绑定不保证文件只存一份；mmap 不意味着数据已在 GPU，也不意味着没有 page fault。

<a id="c-4-1"></a>
## 4.1 Transformer 层实现

**依赖**：4.0、3.1。

**问题**：每个算子的数学与 layout 如何对应；逐层误差从何处出现；架构变体需要修改哪些计算和状态。

**对象与源码**：复用 `labs/L4/qwen3_from_scratch.py`；Qwen3-1.7B 的 `modeling_qwen3.py`、vLLM Qwen3 model 与 SGLang 对应模型；扩展对照 SmolLM3-3B-Base 的 NoPE 层配置。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 手写 RMSNorm、RoPE、GQA、SwiGLU、residual 与 lm_head；输出每层输入、归一化、Q/K/V、attention、MLP 的 shape/stride | 与 HF 固定输入逐层比较，首个差异可定位；不能只比较最终生成字符串 |
| B | 比较 full prefill 与 cached decode，输入含左 padding、多长度和位置偏移；扫描 B=1/4、S=1/17/128/2048 | logits、KV 有效区间和状态更新一致；尾部 padding 不参与有效输出 |
| C | 接入一个架构差异：SmolLM3 按真实 config 在部分层禁用 RoPE；与 3.4 的 Qwen3.5 区分位置设计与状态类型 | 给出新增模型所需配置、层、cache 接口与测试；不以相同参数量证明行为等价 |

**反例与边界**：weight transpose、RMSNorm 的 eps/累加精度和 RoPE pair 排列都可能保持 shape 正确而数值错误。

<a id="c-4-2"></a>
## 4.2 数值系统与确定性

**依赖**：0.0b、2.0、2.3。

**问题**：表示误差怎样传播；累加与归约顺序怎样改变结果；哪些确定性保证受后端与 batch 限制。

**对象与源码**：复用 `labs/L4/numerics.py`；PyTorch dtype、autocast、deterministic algorithms；Qwen3-1.7B 的 eager/compile 与 vLLM sampler，格式文档参照 [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 FP16/BF16/FP8 的可表示数枚举、舍入、溢出和 subnormal 示例；区分输入格式、乘法与累加精度 | 极值、抵消、长归约与 FP64 参照对齐；平均误差、最大误差和误差比值定义清楚 |
| B | 固定 200 条输入，在 batch=1/2/8/32、不同排列与 padding 下保存 logits、top-2 margin、argmax 翻转和生成差异 | 配对样本与后端选择可复现；不用一个余弦相似度代替决策稳定性 |
| C | 对照 eager/compile、TF32 开关、确定性设置和相同后端复跑；将第一处差异定位到某层或归约 | 报位级、容差级、分布级保证的实际范围；与 5.5/7.6 的验证及 logprob 对拍相连 |

**反例与边界**：小均方误差也可能改变接近并列的 argmax；开启确定性选项不保证跨设备或跨版本逐位一致。

<a id="c-4-3"></a>
## 4.3 量化工程

**依赖**：4.0、4.1、4.2、2.4；KV 实验依赖 3.3，服务实验依赖 5.1/5.2。

**问题**：校准与误差优化怎样改变模型；格式怎样约束真实 kernel；在哪些质量和负载条件下压缩有收益。

**对象与源码**：复用 `labs/L4/quantization.py` 与既有 Qwen2.5-1.5B 三份 checkpoint；新主例固定 Qwen3-4B，扩展 Qwen3-8B。阅读 GPTQ/AWQ、SmoothQuant/QuaRot 的原始目标，沿 vLLM quantization config→loader/repack→linear/MoE method→kernel；前沿使用 [MR-GPTQ](https://arxiv.org/html/2509.23202)、[FP-Quant](https://github.com/IST-DASLab/FP-Quant)、[QuTLASS](https://github.com/IST-DASLab/qutlass) 与 [TurboQuant](https://arxiv.org/html/2504.19874)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 推导层输出重建目标及 Hessian；在 X=[256,128]、W=[64,128] 上实现 RTN/clipping、GPTQ 逐列补偿、AWQ 通道等价缩放，比较校准分布改变后的误差 | 新增 `labs/L4/quantize_reference.py`；FP64 对拍中间状态。比较 AWQ 与原始参数前核对相邻层缩放，不能直接把参数坐标变化称为量化噪声 |
| B | 对 INT4、FP8、MXFP4、NVFP4 实现 pack/unpack 与 scale 参照，扫描 group=32/64/128 的合法配置；解释 scale 自身量化、zero、padding 和布局 | 已知整数向量、官方反量化与加载后权重逐元素对齐；格式、校准算法、容器和计算后端分开，实际 bits/weight 可复算 |
| C | 在 Qwen3 q_proj/down_proj 实际 shape 上比较 BF16、显式反量化、融合 W4A16、W8A8/W4A4；M=1/8/32/128/512/2048，驻留与大于 L2 两组；给 MR-GPTQ 做 identity/旋转、融合/非融合消融 | 新增 `labs/L4/quantized_linear_bench.py`；记录真实 kernel、在线旋转、scale 重排、转换与 workspace 的完整成本；sm_100/sm_120 分开核对支持 |
| D | 通过 3.3 的已验证分页接口运行真实 FP8 KV；研究 K/V 的不同误差传播，再用支持的 INT4 或 TurboQuant 路线作对照；S=2048/8192/32768，B=1/8 | 保存 scale、状态字节、attention 输出与长上下文质量；旋转/MSE 或内积保证不能外推为任务无损 |
| E | WikiText-2 train 固定 seed 抽 128×2048 token 作校准，test 全量按冻结分块评 PPL；C-Eval validation 固定分层 200 题作独立任务。服务扫描采用共同协议 | 新增 `labs/L4/quantization_eval.py`；校准/调参/测试分离。文件大小、实际显存、prefill、decode、完整调用及质量—成本曲线齐备，减速结果同样保留 |

**反例与边界**：无 clipping 且同一坐标系下的均匀 RTN 误差界不能套到所有量化过程；单文本 PPL、配置枚举和框架 dtype 转换均不能代替完整验证。

<a id="c-4-4"></a>
## 4.4 MoE 的路由与执行

**依赖**：4.1、2.4、3.3；跨 GPU 综合依赖 6.3。

**问题**：路由怎样变成具体工作；不均衡通过哪种调度影响时延；总参数、激活参数与实际读取量如何区分。

**对象与源码**：复用 `labs/L4/moe.py`；`allenai/OLMoE-1B-7B-0924-Instruct` 为可运行基线，Qwen3-30B-A3B 为容量允许时的结构扩展；vLLM fused_moe、SGLang MoE runner、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 与 [EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 实现 top-k、权重归一化、token pack、专家计算与 weighted combine；打印 permutation/inverse、expert offsets 和 padding | 小矩阵与逐 token dense expert 参照对拍；空专家、重复专家、非法路由与尾 tile 均覆盖 |
| B | 用 vLLM/Triton 的真实 grouped kernel 替换 Python 循环，比较均匀、单专家 20%/50% 倾斜和真实 OLMoE 路由；tokens=1/32/256/4096 | 分别测路由、排序、pack、GEMM、combine 与完整调用；不能把 max/mean 直接当真实 GPU 效率上界 |
| C | 采逐层、逐步专家访问集合与字节，比较 prefill/decode、batch 与量化；对 DeepGEMM 固定实际支持架构，追踪 scale 与 grouped layout | 真实流量/任务分工与简化专家覆盖公式分别报告；未运行的 Hopper/数据中心路线保留源码分析 |
| D | 在 6.3 对照静态 placement 与 EPLB，记录统计窗口、迁移字节、冷专家和路由漂移；在同一负载下改变重平衡周期 | 解释迁移成本能否被后续收益抵消；不能只以负载均匀度判断端到端加速 |

**反例与边界**：一个专家不等于一个独占 SM；专家可按 tile 切分，动态路由和缓存会改变实际权重读取量。

<a id="c-5-1"></a>
## 5.1 Prefill 与 decode

**依赖**：0.2、3.3、4.1。

**问题**：两阶段实际执行了什么；长度和 batch 怎样改变瓶颈；请求级时间如何与设备工作闭合。

**对象与源码**：复用 `labs/L5/prefill_decode.py`；Qwen3-1.7B 与 Qwen3-4B，vLLM/SGLang scheduler、model runner 和 attention metadata。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从 schedule 的 computed tokens 和输入 positions 判断 prefill/decode/mixed batch，记录首个输出由哪次 forward 产生 | 定义首次 forward、HTTP TTFT、逐 token 间隔与完整调用，不用 generate 差值冒充精确阶段时间 |
| B | 用共同 SERVE 协议扫描 prompt/batch/output，加入相同 token 总量的等长与长短混批；固定缓存条件 | 输出逐请求和逐 step 原始事件、kernel、权重/KV/激活账；用实际产出计吞吐 |
| C | 比较两引擎的混批与 chunked prefill，在同一模型上关联 GEMM M 维、attention 长度与临时内存 | 明确支持同一结论的对照与仅有相关性的趋势；后续负载由 9.1 重放验证 |

**反例与边界**：prefill 不必总是算力受限，decode 不必总是 KV 带宽受限；判断依赖实际 shape、batch 和后端。

<a id="c-5-2"></a>
## 5.2 KV cache 与前缀复用

**依赖**：3.3、4.1、2.0c。

**问题**：逻辑 token 如何对应物理块；引用计数与 copy-on-write 如何保证正确；前缀身份怎样决定复用。

**对象与源码**：复用 `labs/L5/mini_block_pool.py`、`prefix_hits_per_request.py`、`prefix_probe.py`；vLLM `BlockPool`/KV manager/hash，SGLang `RadixCache` 与请求 token pool；Qwen3-1.7B。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 扩展块池支持 allocate/append/fork/free/evict/COW；逐步打印 block table、refcount、有效长度与内容校验 | 单请求、共享前缀和分叉后写入与无缓存参考对拍；不只验块计数 |
| B | page 边界前后、前缀长度 0/15/16/17/127/128/129，中间 token 改变、相同 suffix 不同 prefix、LRU 压力分别测试 | 逐请求记录命中块、重算 token、搬运和完整时间；复用已有命中工件，只补缺少的状态 |
| C | 固定 input_ids，改变模型/adapter revision、位置规则与多模态预处理身份；分析哪些字段进入真实缓存键 | 显式展示正确失效及不被系统自动识别的更新；adapter 热更新接 5.10，视觉身份接 4.8 |
| D | 对比 vLLM 块哈希与 SGLang radix 的查找、驱逐、锁定和释放路径，改变共享前缀比例与活跃会话数 | 数值和资源先通过再比较速度；形成 8.6/9.3 的缓存接口与取回基线 |

**反例与边界**：命中率不是节省时间；相同文本片段在不同前缀/位置下不自动具有相同 KV。

<a id="c-5-3"></a>
## 5.3 调度与 chunked prefill

**依赖**：5.1、5.2；负载发生器使用 8.3 的基础任务。

**问题**：预算怎样转换为实际工作；吞吐、公平和长尾怎样冲突；预测误差和过载怎样影响调度。

**对象与源码**：复用 `labs/L5/scheduling.py`；vLLM `v1/core/sched/scheduler.py`、SGLang scheduler/prefill adder/overlap loop；Qwen3-1.7B。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 将预算 mini 扩展成离散事件调度器，持有 arrival、computed/output tokens、KV、deadline 与取消；实现 FCFS、decode-priority、轮转 | 三种策略使用相同请求轨迹；检查饥饿、资源预算和请求状态不变量 |
| B | 长 prompt=8192 插入 8 条 decode，token budget=128/256/512/2048/8192；采每请求 TTFT/TPOT、scheduled tokens 与重算 | 修正固定观察窗口和预算估计吞吐的口径；每档相同工作量且保存重复样本 |
| C | 用 ARRIVAL 协议比较两引擎与 mini 预测，加入历史长度预测的 step-time 预算；对预测误差注入 ±25%/50% 偏差 | goodput、p95/p99、最长等待与拒绝一并报告；预测式策略通过同任务可控对照验收 |

**反例与边界**：token budget 相同不代表各 step 时长相同；仅记录 step 慢不能恢复每条请求的 TPOT。

<a id="c-5-4"></a>
## 5.4 图执行与流水重叠

**依赖**：2.1、2.0c、2.7、5.1。

**问题**：图模式怎样选择；捕获桶与 padding 有何成本；CPU 调度、H2D、模型与采样可以怎样重叠。

**对象与源码**：复用 `labs/L5/execution_layer.py`、`labs/L2/cudagraph_modes.py`；vLLM graph dispatch/runner、SGLang overlap scheduler；Qwen3-1.7B。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 从当前版本枚举实际 graph modes，采捕获尺寸、指针、图池、fallback 及 graph 内 kernel；测试桶边界两侧 | 交付模式→调用→device work 映射，区分分段编译与 CUDA Graph |
| B | 同请求比较 eager、partial/full graph 与不同捕获桶，记录启动、捕获、padding、稳态、峰值与重复分布 | 不能只报吞吐比；已有非单调点由实际分派和资源解释，无法解释则保留 |
| C | 对照 CPU prepare、copy、forward、sampling 顺序与 overlap，关闭单个重叠阶段并记录事件依赖 | 输出 buffer 生命周期正确；正常计时与插桩结果分开，计算重叠不以两个单项时间相减推断 |

**反例与边界**：graph replay 更快不保证整请求更快；增加捕获尺寸会增加初始化与显存成本。

<a id="c-5-5"></a>
## 5.5 投机解码

**依赖**：3.3、4.2、5.1、5.4；混合状态实验在 5.13 基础任务后执行。

**问题**：多 token 验证如何保持分布；草稿来源怎样改变成本；验证和回滚怎样与调度及状态集成。

**对象与源码**：复用 `labs/L5/speculative.py` 的 ngram 基线；读 vLLM rejection sampler/spec decode 与 [SGLang speculative](https://docs.sglang.io/docs/advanced_features/speculative_decoding)。前沿主例为 `Qwen/Qwen3-4B` + [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)；EAGLE3 对照 `Qwen/Qwen3-8B` + [thoughtworks/Qwen3-8B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-8B-Eagle3)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 在小词表 Markov 模型实现完整多步 proposal、accept、residual sampling、EOS 和回滚；枚举短序列验证目标分布 | 新增 `labs/L5/speculative_reference.py`；不仅验证单位置边际，还验证条件前缀和最终序列分布 |
| B | 为引擎单次安装统计 hook，预热后清零；逐轮保存各位置提案、条件概率、接受长度、额外 token、KV 有效区间和时间 | 草稿/验证/采样/回滚完整计时；聚合接受率不冒充逐位置条件概率 |
| C | 分别在各自目标模型内比较普通 decode、ngram、EAGLE3 或 DFlash；k=1/2/4/8/15，batch=1/4/16，任务取 GSM8K 固定 128 题、代码补全 64 题与重复文本 | 记录任务质量、接受长度、完整延迟和容量；不同 target 的结果不能直接作为草稿方法排名。DFlash 示例中的 FA3 不直接用于 sm_120，先固定可支持 backend |
| D | 以 `Qwen/Qwen3.5-4B` 检查实际 MTP 权重与 NEXTN/对应配置，读 [Qwen3.5 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)；通过加载与数值门后测 MTP 与 recurrent rollback | 模型族声明不代替具体 checkpoint 支持；状态快照、提交与回滚与 5.13 共用同一用例；未有 MTP 权重保持未完成 |

**反例与边界**：高接受率也可能不加速；相同 seed 的 token 一致不是随机分布保证；草稿模型与目标 revision 必须匹配。

<a id="c-5-6"></a>
## 5.6 结构化输出

**依赖**：0.5、5.3、5.4。

**问题**：语法状态怎样限制 token；mask 与接受器如何保持一致；约束编译、填充和应用的成本何时主导。

**对象与源码**：复用 `labs/L5/mini_grammar.py`、`grammar_libs_compare.py`、`structured_output_audit.py`、`structural_tag_probe.py`；XGrammar、llguidance、Outlines 与 vLLM/SGLang adapters。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 给同一 tokenizer 构造 JSON、递归数组、可选字段、Unicode 字符串与 tool schema，逐步对照三库 bitmask、accept 与 EOS | 修订 mini 以覆盖 tokenizer 边界；保留 mask 允许而接受器拒绝的原始输入，追踪实际错误层 |
| B | schema 字段数 4/16/64/256、请求 batch=1/8/32，分开编译缓存、每步 fill、mask apply、采样和 parser | 库级时间与引擎内插桩分别测；不以库级差值完全解释端到端差值 |
| C | 在 SGLang 实际服务中比较 jump-forward on/off，在 vLLM 中比较 structural tag/整段约束；固定输出任务与 schema 有效性 | 报输出质量、有效率、接受长度和完整时间；检查 speculative 与 grammar 状态回滚的共同边界 |

**反例与边界**：语法正确不等于内容正确；参数支持与枚举存在不代表某模型/tokenizer/后端组合已经运行成功。

<a id="c-5-7"></a>
## 5.7 引擎架构与 nanoserve

**依赖**：5.1、5.2、5.3、5.4、4.1。

**问题**：最小引擎需要哪些状态边界；可替换后端怎样定义接口；不同引擎的架构代价如何实际比较。

**对象与源码**：复用现有 nanoserve 六阶段实现与 Qwen3-1.7B 手写前向；vLLM、SGLang、TensorRT-LLM、llama.cpp 固定源码；真实分页接口复用 3.3。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 固定 Request、Scheduler、BlockPool、ModelRunner、Sampler、OutputProcessor 的输入输出和所有权；为每个状态转换补不变量 | 用逐请求 trace 重建一次生命周期；源码表进入核心结构，不停在启动参数 |
| B | 将 gather+SDPA 执行器替换为真实 paged attention，保持其余调度与采样不变；比较 B=1/8、长短混批、共享前缀、取消 | 两执行器逐 token/logit 对拍，KV 内容与资源回收一致；性能包含 metadata 和必要布局处理 |
| C | 对四引擎运行共同支持的固定 Qwen3 配置；llama.cpp 的 GGUF 与 TensorRT-LLM engine 单独记录转换和精度 | 比较同一任务的初始化、稳态、峰值和扩展面；不可同精度的配置只作独立路线，不给无条件排名 |
| D | 新增一种调度策略或采样后端，记录跨模块改动和故障定位；将这个具体 diff 交给 M0 分析 | 抽象边界通过实际修改验证；不以接口层数或代码量判断架构优劣 |

**反例与边界**：分页寻址的 mini 不等于高效分页 kernel；引擎相同输出不说明内部机制相同。

<a id="c-5-8"></a>
## 5.8 失败与资源回收

**依赖**：5.1、5.2、5.3。

**问题**：失败如何跨层传播；哪些资源必须立即或延迟回收；背压和抢占如何改变正常请求的质量与长尾。

**对象与源码**：复用 `labs/L5/vllm_failure_paths.py` 与 nanoserve failure harness；vLLM abort/preempt/finish、SGLang abort 与 session state。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 在排队、prefill、decode、stream write 四个阶段取消；注入超时、断连、重复取消与 worker 子进程失败 | 请求/队列/KV/output 三处状态及 GPU event 生命周期有完整记录；不以连接断开代替回收 |
| B | 对重算与可用换出路径运行相同长度分布，记录牺牲请求、重算 token、复制字节、请求结果和恢复时延 | 先确认目标版本存在换出实现；不存在时在 nanoserve 验证协议并明确真实引擎限制 |
| C | 用 ARRIVAL 的过载与突发比较无界排队、有界准入和抢占，改变 oldest/youngest 策略 | 统计全部请求的 p95/p99、拒绝、超时与最长等待；恢复后合法请求在同一服务继续成功 |

**反例与边界**：捕获异常不代表恢复；独立进程重启成功与同进程继续服务分别验收。

<a id="c-5-9"></a>
## 5.9 采样引擎

**依赖**：0.5、4.2、5.4。

**问题**：过滤顺序和数值边界如何决定候选；返回 logprobs 为什么改变路径；采样开销如何影响完整服务。

**对象与源码**：复用 `labs/L5/mini_sampler.py`、`sampling_audit.py`、`sampling_contract_probe.py` 及已保存 mismatch；vLLM topk_topp Triton/native、SGLang sampler、FlashInfer sampling。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 保持已验证的处理顺序与 logprobs 契约；用原始边界 logits 定位 batch 64/256 的候选差异，控制 ties、累计精度、pivot 停止与阈值比较 | 最小失败输入和逐阶段候选齐备；根因未定位前性能结果不标等价性通过 |
| B | B=1/7/8/64/256，词表 32000/151936，构造平坦、尖峰、并列和长尾分布；分别测过滤、随机数、logsumexp、gather 和返回整理 | 记录真实后端与完整调用成本；clone、临时内存、输入是否驻留 L2 分列 |
| C | 在两引擎真实服务中扫描 logprobs=None/0/1/5/20，增加 beam width=1/2/4/8 的支持路线 | 对齐返回字段、长度惩罚与质量；beam 的 KV 分叉、重排和显存成本有实测，不能预设其“消失” |

**反例与边界**：固定 seed 不等于跨算法同样本；要返回一个 token 的 logprob 仍可能需要整行归一化。

<a id="c-5-10"></a>
## 5.10 LoRA 多租户服务

**依赖**：4.1、5.3、5.4；真实训练 adapter 的对照接 7.5。

**问题**：混合 adapter 的 token 怎样分段；slot 容量怎样改变执行波次；热更新如何影响缓存身份与恢复。

**对象与源码**：复用 `labs/L5/lora_scaling_audit.py`、`sglang_lora_kernel_probe.py`、`lora_recovery_test.py`、`lora_kv_invalidation_test.py`；vLLM Punica/LoRA manager 与 SGLang LoRABatchInfo、内存池、分段 kernel。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对照 mini、逐行 merged dense 与真实 shrink/expand，记录 token→adapter、rank、scale、segment 和 base 哨兵 | 数值、未适配行与 TP slice 对齐；保留已核对工件，补漏项不重做全部扫描 |
| B | slot=2/4/8，adapter 数覆盖每个上限的前一档、相等、后一档及两倍；rank=8/16/32，均匀/倾斜访问 | 验证拐点随 slot 迁移；分开 CPU cache 命中、H2D、分轮与 base GEMM 摊销，解释不够的趋势不归因 |
| C | 在 HTTP 服务测试非法 rank 后合法请求、同名不同版本更新、请求执行中卸载/替换；与独立引擎重启对照 | cache key、slot、请求版本和恢复结果一致；不能把重启恢复写成同进程恢复 |
| D | 在两引擎对齐真实完整路径，比较 eager/graph、驻留/换入；使用 7.5 训练并固定 revision 的 adapter 检查任务质量 | 随机 adapter 的调度证据与训练 adapter 的质量证据分开；新增后端的改动范围可定位 |

**反例与边界**：adapter 总数不等于同时活跃数；同名热更新不保证引擎自动使旧 KV 失效。

<a id="c-5-11"></a>
## 5.11 API、序列化与流协议

**依赖**：0.1、5.3、5.8。

**问题**：客户端事件与引擎事件如何对应；序列化和分词怎样阻塞前端；分帧、取消与代理怎样影响流式输出。

**对象与源码**：复用 `labs/L5/api_layer_audit.py`、`mini_sse.py`、`sse_head_of_line.py`；vLLM API middleware/StreamingResponse、SGLang TokenizerManager/DetokenizerManager 与动态 tokenizer。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对齐客户端发出、前端到达、入队、执行、生成 delta 和消费的事件，保留 HTTP chunked、SSE、token 三种边界 | 时间分解逐请求闭合；网络接收期间包含服务器工作，不能整体归为客户端开销 |
| B | token 长度 32/512/4096/16384，batch=1/8/32，比较逐条、批量、thread offload 与动态 tokenizer；在 SGLang 起真实服务 | 相同输入和模板，报告吞吐、排队和尖峰；默认关闭的功能显式记录开关 |
| C | 在独立本地代理场景测试 buffer on/off、keep-alive、慢读者与断连；HTTP/1.1 为基线，支持时补 HTTP/2 | 保存原始流和取消传播；并发 32 的波动需交错重复、事件循环与连接重用记录解释 |

**反例与边界**：首包、首 token、首个有效文本不同；不同机器时钟需校准，跨进程同一时钟也要核对时间定义。

<a id="c-5-12"></a>
## 5.12 Embedding、rerank 与非生成服务

**依赖**：5.1、3.1、4.1、5.11。

**问题**：服务任务与模型架构怎样区分；pooling/打分语义如何对齐；批处理与数据搬运的成本在哪里。

**对象与源码**：复用 `labs/L5/embedding_serving_audit.py`、`rerank_serving_audit.py` 与 pooling 对照；`BAAI/bge-small-en-v1.5`、`Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Reranker-0.6B`；[Qwen3 Embedding 官方说明](https://qwenlm.github.io/blog/qwen3-embedding/)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对 embedding 的 mask、取 last/mean/CLS、归一化与维度截断逐项对拍；对 reranker 保存完整模板、token IDs、yes/no logits 与最终分数公式 | HF 与引擎先对齐输入及计算链，再比较分数/排序；保留现有不等价链的失败输入 |
| B | 在同一 checkpoint 内比较首轮/固定长度 decode/池化任务，按参数、激活、workspace、KV 分阶段采内存 | 不将不同大小 encoder/decoder 的容量比解释为架构收益；缓存长度固定且每轮可核查 |
| C | 相同文本内容和长度分布比较 padding、packing、独立/合批；两引擎服务采客户端与服务端事件 | 逐请求分解闭合，禁止以不同样本中位数相减或以接收耗时冒充网络开销 |
| D | 固定 BEIR NFCorpus 的语料/查询/qrels，接 embedding→候选→rerank；比较 Recall@k、nDCG@10 与完整时延 | 将模型质量、batch 收益与服务开销联系起来，为 9.6 提供可复用数据和计分器 |

**反例与边界**：decoder 架构可提供一次性 pooling 服务；不能据服务接口名称推断一定持有跨请求 KV。

<a id="c-5-13"></a>
## 5.13 混合架构状态管理

**依赖**：3.4、5.2、5.5、2.0c；基础状态任务可先于投机综合完成。

**问题**：每类层持有什么状态；prefix cache 怎样表示递推快照；拒绝草稿后恢复什么值和字节。

**对象与源码**：复用 `labs/L5/hybrid_state_audit.py`；RecurrentGemma-2B 与 `Qwen/Qwen3.5-4B`；vLLM hybrid cache manager/state pool、SGLang recurrent cache，模型的 convolution/gated delta 状态。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 逐层打印 KV、conv、recurrent state 的 shape/stride/dtype 与生命周期；实现混合 pool 的 allocate/snapshot/commit/rollback | 短序列逐步状态与不缓存参照对拍；窗口层与全 attention 层分列 |
| B | 共享前缀后分叉、前缀边界前后、LRU 驱逐与重算分别测试，记录实际状态值和命中 | 不能以 prefill 更快证明状态复用正确；更换层配置和模型版本必须检查身份 |
| C | 强制在草稿第 1/2/末位拒绝，记录回滚前后值、copy kernel 和字节；与不投机输出及完整调用比较 | 直接测量回滚成本，不用理论状态字节代替耗时；快照成本计入完整调用 |
| D | 两引擎扫描 S=512/2048/4096/8192/16384，B=1/4/8，并测试取消回收 | 状态增长曲线与逐层账闭合；旧 2048 点之外的结论需要新增数据，架构间质量不假定相同 |

**反例与边界**：递推状态被更新后通常不能靠缩短长度恢复；需要合法快照、重算或算法特定回滚机制。

<a id="c-7-0"></a>
## 7.0 Autograd 引擎

**依赖**：0.0b、2.0b、2.0c。

**问题**：反向图何时构建；依赖计数怎样调度节点；保存值、流与梯度模式如何决定正确性和内存。

**对象与源码**：复用 `labs/L7/mini_autograd.py`、`trace_grad_fn.py`、`trace_backward.py`；PyTorch `torch/csrc/autograd/engine.cpp`、`function.h`、`saved_variable.cpp`、AccumulateGrad。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 将有分支、共享 Parameter、多个输出的前向映射到 Node/Edge/GraphTask；记录依赖计数、ready queue、InputBuffer 与梯度汇合 | mini 与 autograd 的执行依赖和梯度对拍；节点执行次数由依赖而非简单链式次序解释 |
| B | 分别测试 retain_graph、create_graph、no_grad、inference_mode、saved tensor hooks 与原地修改；记录保存值释放和 version counter | 数值、二阶梯度和错误原文齐备；解释推理模式与仅关闭梯度的状态差异 |
| C | 在 crater 用两个 CUDA stream 验证前向/反向和梯度消费的依赖；对 checkpoint 重算记录保存张量与峰值 | CPU 机制与 CUDA 完成语义分别验收；把生命周期结果连接 7.1 的重算策略 |

**反例与边界**：图可遍历不代表实际引擎按 DFS 执行；释放 Python 节点不证明相关 GPU 工作结束。

<a id="c-7-0b"></a>
## 7.0b 完整训练步

**依赖**：7.0、4.0、4.1、4.2。

**问题**：样本如何决定有效 loss；累积和大 batch 怎样等价；更新后恢复需要哪些状态。

**对象与源码**：复用 `labs/L7/training_step_smollm.py`、`gradient_accumulation.py`；已测 SmolLM2-360M；代表性扩展用 [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) 与 [官方训练配置](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3)。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 对齐样本 ID、labels、loss mask、packing 与有效 token；手写 cross-entropy 和 AdamW 一步，对拍框架逐参数更新 | 本文模型名称、配置与原始工件一致；不能把 SmolLM2 结果写成 SmolLM3 测量 |
| B | 固定同一组变长样本，microbatch=1/2/4，对比直接 batch、累积及错误按 microbatch 均值归一化；控制 dropout/RNG/clip 顺序 | 输出 loss、梯度和参数差异，定位现有累积不一致；容差由精度和归约确定 |
| C | 在第 1/3/7 步保存模型、optimizer、scheduler、scaler、RNG 和数据游标，再恢复执行两步，与不中断参考对照 | 样本序列、状态和更新可重建；缺 optimizer/RNG/游标分别构造失败例 |
| D | 迁移同一语义到 SmolLM3，先按真实状态 dtype 核算容量；不足时接 7.2 的 FSDP2，不擅自将全参数训练改为 LoRA | 原生 NoPE/GQA、有效 token 和优化器语义保持；规模实验与基础正确性分别报告 |

**反例与边界**：一次 loss 相近不足以证明恢复一致；仅权重恢复与完整训练恢复是不同任务。

<a id="c-7-1"></a>
## 7.1 训练循环的系统视角

**依赖**：7.0b、2.0c、2.6。

**问题**：样本到更新的关键路径是什么；重算和混合精度改变哪些状态；数据等待怎样与设备工作重叠。

**对象与源码**：复用 `labs/L7/training_loop_systems.py`；PyTorch DataLoader/pin_memory、autocast/GradScaler、checkpoint/optimizer；SmolLM2-360M 基线与 SmolLM3-3B 扩展。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 记录 sample read、collate、pin、H2D、forward、backward、clip、optimizer；分别比较纯预加载数据与固定读取延迟的控制组 | 同一有效训练更新下建立完整时间线；CPU 提交和 GPU 执行重叠不重复相加 |
| B | workers=0/2/4、prefetch=1/2/4、pin on/off；activation checkpoint 按层/每两层/on-off，固定 batch 与 tokens | saved tensors、重算调用、峰值和吞吐共同支持解释；数据等待与算子速度分开 |
| C | 对 FP32/BF16/FP16+scaler 保存参数/梯度/Adam 状态 dtype、scale、finite 检查和跳过更新次数 | 定位 FP16 inf 的首个发生层/步骤；比较真实更新吞吐，不能把跳过 optimizer 当加速 |

**反例与边界**：优化器时间可能不随前向 dtype 同比下降；短热循环不代表完整数据流水。

<a id="c-7-2"></a>
## 7.2 分片训练与流水并行

**依赖**：7.1、6.0b、6.1。

**问题**：各并行方法分片了什么；通信为何在特定 hook 发生；流水与重叠怎样改变内存和有效更新速度。

**对象与源码**：SmolLM2-360M 数值基线、SmolLM3-3B-Base 代表性模型；DDP reducer、[FSDP2 fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.md)、DeepSpeed ZeRO、[PyTorch pipeline schedules](https://docs.pytorch.org/docs/stable/distributed.pipelining.md)、Megatron parallel layers。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 将通信原语材料归入 6.0/6.1 的引用范围，正文按 DDP、ZeRO-1/2/3、FSDP2 重建参数/梯度/优化器/激活归属表 | 章节内容与 outline 一致；不以通信带宽实验代替分片训练 |
| B | 两 rank 运行同一训练步，对齐不同有效 token 数的 loss、梯度和更新；追踪 FSDP2 wrap 单元、all-gather/reduce-scatter、reshard 与预取 | 与单机全局 batch 对拍；按 hook 和真实 group 边界解释通信，不能沿用 DDP bucket 参数假设 |
| C | 在 2/4 张 L40S 上对照 DDP/FSDP2/ZeRO 的完整峰值和时间，扫描 microbatch/累积，记录参数暂时展开与通信 buffer | 数值条件一致；模型/optimizer dtype 与全局 batch 冻结；五卡仅用于合法并行配置 |
| D | 四阶段小模型在两卡上比较 GPipe、1F1B、interleaved 与 ScheduleInterleavedZeroBubble/ZBV，microbatch=4/8/16/32 | 导出 F、backward-input、backward-weight 依赖 DAG 与真实时间线；空泡、激活峰值和通信一起测，不能凭名字宣称零空泡 |
| E | 给 TP/SP/CP/EP 与上述训练步增加分片/聚合说明，选择满足整除条件的 2D 组合做实测；其余组合按 6.2/6.5 的接口分任务验证 | 全球梯度重建与成本模型一致；未运行组合不作为吞吐结论 |

**反例与边界**：用 ring 通信量公式除以耗时不能证明 NCCL 选择了 ring；跨节点不能使用单机 NVLink 带宽直接估算。

<a id="c-M1"></a>
## M1 阅读大型代码库

**依赖**：0.0；完整案例复用 2.0b、5.7。

**问题**：怎样从用户入口找到真实实现；如何验证某条路径确实运行；怎样界定扩展点和上游设计证据。

**对象与源码**：复用 `labs/M/trace_dispatch.py`、`version_manifest.py`；PyTorch linear/SDPA 与 vLLM/SGLang 请求路径；前沿迁移练习使用 Cosmos3-Edge 的 reasoner/generator 分派。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 演示 `rg` 找入口、schema/registry、调用者与实现；维护 commit、file:line、源码片段和观察层次 | 读者可独立复现定位过程；不会把注册表或注释当运行证据 |
| B | 对未知输入配置先预测分支，再用最小 probe、日志和 trace 验证；构造一个 Python hook 看不到但仍存在的分解算子 | 静态关系、运行事实与推断分开，预测错误保留具体反例 |
| C | 以 Cosmos3-Edge 的 `Cosmos3OmniPipeline` 到 scheduler/transformer 为迁移练习；挑一个扩展点，查实现与原始 PR/RFC | 完成入口、状态、约束和改动位置表；自行提出的替代设计不能写成上游否决理由 |

**反例与边界**：源码阅读不等于全文搜索名称；一个类名不能证明对应后端已经被选中。

<a id="c-M2"></a>
## M2 可信性能测量

**依赖**：0.2；GPU 实现依赖 2.1。

**问题**：计时范围如何定义；缓存/预热/同步怎样改变结论；怎样量化不确定性并检验因果解释。

**对象与源码**：复用 `labs/M/measure_warmup.py`；新增 `labs/M/measurement_protocol.py`，验证对象取 2.3 归约、5.1 请求和 7.1 训练步，依赖 PyTorch benchmark/profiler 与 CUDA event。

| 任务 | 执行步骤 | 交付与验收 |
|---|---|---|
| A | 用同一算子展示 CPU 未同步、event、墙钟、单次同步、批量提交的区别；拆开首次加载、编译、缓存、allocator 预热 | 每个计时结果能说明包含和遗漏的工作；现有 CPU 工件只支持 CPU 结论 |
| B | 工作集按实测 L2 的 0.5/1/2/4 倍扫描，交错运行基线和候选，保留至少 5 轮原始样本 | 明确在 L2 与轮转条件；中位数、区间和异常点均可重算，不选最佳样本代表总体 |
| C | 请求按独立任务/时间窗重采样，生成尾延迟与差值区间；用单因素干预区分频率、缓存和实现切换 | 输出统计脚本及一个结论被对照推翻的完整案例；观测、公式预测和解释分别表述 |

**反例与边界**：确定性评测重复同一输入不能增加独立样本量；父子事件相加与不同运行中位数相减不能构成时间分解。
