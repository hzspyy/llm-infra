# 章节修订后的技术审阅笔记

本文记录审阅时发现的知识、示例与实验解释问题。当前任务与进度统一见[修订计划](../plans/completed.md)、[未完成部分任务书](../plans/pending.md)和[STATUS](../../STATUS.md)。

## L7 确定问题

### F01 · P1 · 7.4 在公式中途截断，源码分析与实现说明整段缺失

位置：[7.4:49](../../src/L7/7.4-data-ckpt-fault.md)。

原文为“若存储行区间为 $`ENVIRONMENTS.md`设置。”，下一行直接跳到三条 lab 命令。当前全文没有“工程实现：源码解析”“设计取舍”和“动手 lab”部分，亦没有实际 DCP 源码宏。自测仍要求解释 2→4 区间映射，但正文没有该推导。生成 HTML 原样显示这个断句，侧栏也缺对应章节。

这不是需要补 GPU 实验的问题，而是读者无法获得已经声称交付的解释。应恢复完整的分片映射、真实 DCP staging/save/metadata 调用分析、自定义协议的边界及命令前提，再核对页面正文。上轮 R17 的 7.4 源码交付以及页面验收不能据旧检查结果关闭。对应 7.4-C/E、章节规范第 1/4/5 项。

### F02 · P1 · 7.0 的保存值片段不可运行，紧邻的矩阵梯度公式也错

位置：[7.0:437](../../src/L7/7.0-autograd-anatomy.md)、[7.0:441](../../src/L7/7.0-autograd-anatomy.md)。

`h=x@W.t()` 的内建 `MmBackward0` 没有 `saved_tensors` 属性；实际调用抛 AttributeError。本机保存字段是 `_saved_self=[2,3]`、`_saved_mat2=[3,4]`，或可用 saved tensor hooks 观测。`saved_tensors` 不能从自定义 Function 的接口套到所有内建 Node。

同段 `dW=dh@x.t()` 的维度为 `[2,4]@[3,2]`，无法相乘；正确是 `dh.t()@x=[4,3]`。应同时修正公式与可运行片段，不能只修属性名。依据本轮 `builtin_saved_tensors`、`matmul_backward_formula`，以及 [PyTorch saved tensors 文档](https://docs.pytorch.org/docs/main/notes/autograd.html#saved-tensors)。对应 7.0-A/B。

### F03 · P2 · 7.0 将叶子原地写入禁令误解释为 backward 保存值校验

位置：[7.0:573](../../src/L7/7.0-autograd-anatomy.md)、[7.0:578](../../src/L7/7.0-autograd-anatomy.md)。

实际在 `x.add_(1)` 当场报“a leaf Variable that requires grad…”，没有执行到 backward。`y=x*2` 求 x 的梯度只需常数 2，不需保存 x 的原值；把写入放在 `no_grad()` 后 backward 正常得到 `[2,2]`。应分别演示叶子写入限制与真正的 saved-value/version mismatch，例如保存 x 的平方运算。

同章 :603 把保存值释放写成整个反向图必然消失也过强：不需要保存中间 tensor 的加法图可以重复 backward。应解释哪些保存值被释放以及何时再次访问会失败。依据 `inplace_failure`、`actual_saved_value_mutation`、`no_saved_tensor_repeated_backward`。对应 7.0-B。

### F04 · P2 · 7.0 的 checkpoint 示例没有产生所宣称的重算与激活节省

位置：[7.0:460](../../src/L7/7.0-autograd-anatomy.md)。

`heavy_function(x)=x*2` 连用三次，未开 checkpoint 时 autograd 不保存 y1/y2/y3 以求导。对原函数加 saved-tensor hooks 与调用计数，普通路径保存列表为空；non-reentrant checkpoint 路径保存了输入，而 backward 的额外函数调用仍为 0。正文却打印“保存了所有中间值”“前向不保存，反向重算”“节省内存，增加计算”。这些是写死的解释，不是观测。

应换成确实需要保存激活的非线性/矩阵计算，并打印保存值与重算调用数；非重入 checkpoint 还可能 early-stop。当前小例不支持内存/计算取舍结论。依据 `checkpoint_multiply_by_constant`；对应 7.0-C、7.1-B。

### F05 · P2 · 7.0 的“递归打印”代码与展示输出不匹配

位置：[7.0:168](../../src/L7/7.0-autograd-anatomy.md)、[7.0:203](../../src/L7/7.0-autograd-anatomy.md)。

`print_grad_graph` 在非叶分支仅 print 类名，没有递归调用。逐字执行该代码只打印 SumBackward0→ReluBackward0，不会到 MmBackward0、TBackward0 或 AccumulateGrad。把它标为“教学片段”不能使不可能的对应输出成立。

应实际遍历 Node/Edge 并保留去重与共享边的表达，或准确标出仅打印一层及其真实输出。依据 `literal_graph_snippet`（含执行代码和 stdout）。对应 7.0-A。

### F06 · P2 · 7.11 分析器以局部事件跨度充当全局 step 时间，可高估 MFU

位置：[training_trace_analysis.py:239](../../labs/L7/training_trace_analysis.py)、[同文件:253](../../labs/L7/training_trace_analysis.py)；正文 [7.11:60](../../src/L7/7.11-training-job-orchestration.md)、[7.11:96](../../src/L7/7.11-training-job-orchestration.md)。

各 rank 先用本 rank 首末事件求 span，再取 `max(span)` 作 MFU 分母；输入却没有强制提供共同 step 起止与时钟对齐信息。共同时间轴上 rank0 compute=[0,10]、rank1=[90,100] ms 的 100 ms step，被算成 10 ms，1e9 FLOP/2×1 TFLOP/s 的 MFU 从 0.5% 变成 5%。各 rank 的等待也被当作“unattributed=0”。

正文已提醒多 rank 必须对齐时钟和统一 step 边界，但分析器没有落实该约束。保留 category union 是对的，还须单列活动包络与完整 step，接受明确的窗口/时钟契约；不足时 MFU=null。此问题不推翻当前单 rank、MFU=null 的 134.8905 ms 工件，只影响分析器处理分布式窗口的能力。依据 `trace_step_boundary`；对应 7.11-C/D。

### F20 · P2 · 7.3 的配方检查器放过零值或负值 batch 因子

位置：[recipe_inspector.py:73](../../labs/L7/recipe_inspector.py)、[同文件:76](../../labs/L7/recipe_inspector.py)；正文 [7.3:9](../../src/L7/7.3-training-frameworks.md)、[7.3:126](../../src/L7/7.3-training-frameworks.md)。

`validate()` 只检查 `global_batch_size == micro_batch_size * accumulation * dp`，未检查因子为正整数；Nanotron 规范化时又用同一乘式生成 global batch，因此乘式相等不能排除非法输入。用已归档的真实 SmolLM3 stage1 YAML 保持其余字段不变，分别令 microbatch=0、microbatch=-1、accumulation=0 或 DP=0，四种输入均返回 `validation_errors=[]`，得到的 GBS 分别为 0、-192、0、0。未修改的 YAML 仍得到 GBS=576、无错误。

这并不要求检查器证明所有框架参数合法；问题就在正文声明会检查的 batch 因子范围内。“能够解析配置只证明静态约束成立”也把解析与校验混在了一起。应分别检查类型/正值和乘积一致性，再呈现解析成功、检查项通过与完整框架合法性的不同边界。依据[原始 YAML 变体与输出](../../results/local/review/20260914-postfix-closeout/recipe-boundaries.json)；对应 7.3-C。本次只运行安全解析与静态函数，没有启动 Nanotron 或训练。

## 非 L7 确定问题

### F07 · P2 · 0.0b 错把版本计数器挂在 Storage 上

位置：[0.0b:425](../../src/L0/0.0b-minimal-model-backward.md)。

版本计数由 TensorImpl 的 VariableVersion 管理，通常 view 共享；共享 Storage 不必共享 version counter。用 `b.set_(a.untyped_storage(),...)` 构造同地址别名，修改 b 后 a 的值改变，但版本从 `[0,1]` 变成 `[0,2]`。所以“每个 tensor 的存储上挂一个计数器，任何原地写入都会 bump”会误导别名与错误检测的保证。

依据 `same_storage_different_version` 与正文所引 TensorImpl 源码。应分清存储、TensorImpl、view 版本共享和绕过正常 view 关系的别名。对应 0.0b-C、2.0-A。

### F08 · P2 · 0.0b 将报错里的张量生产者当成保存它的算子

位置：[0.0b:444](../../src/L0/0.0b-minimal-model-backward.md)。

`x→sin→square` 中 PowBackward 保存 sin 输出；原地修改后错误写的是“output 0 of Sin”，不是保存消费者 PowBackward。正文的 exp 保存自身输出恰好让二者相同，不能泛化为“报错算子名就是保存该值的算子”。应区分产生该 tensor 的 grad_fn 与访问 SavedVariable 的 backward 节点，必要时用 anomaly detection 定位。

依据 `saved_value_producer_vs_consumer`。对应 0.0b-C、7.0-B。

### F09 · P1 · 0.3 从两点 GEMM 吞吐确定推断功耗墙与温控降频

位置：[0.3:201](../../src/L0/0.3-heterogeneous-baseline.md)、[0.3:211](../../src/L0/0.3-heterogeneous-baseline.md)、[0.3:523](../../src/L0/0.3-heterogeneous-baseline.md)。

L40S 的 4096→8192 BF16 吞吐 256→199.9 TFLOP/s 是真实记录；但[硬件工件](../../results/worldvln/hw_worldvln_l40s.json)只有最大时钟与 power.limit，没有运行期 power.draw、温度、throttle reason 或该时间窗实际时钟。[probe_gemm](../../labs/L0/probe_hw.py)也只采耗时，且改了矩阵尺寸。

不能据此确认“瞬时功耗触及 350W”“从 Boost 跌落”“正是散热节流”。不同 shape/kernel、缓存和时钟等尚未分离；应保留下降事实，将原因降为待验证假设。对应 0.3-B、实验规范事实/推论边界。

### F10 · P1 · 0.3 用 eager 差分时间判定 L40S 在物理上无法满足 10 ms TPOT

位置：[0.3:549](../../src/L0/0.3-heterogeneous-baseline.md)、[0.3:556](../../src/L0/0.3-heterogeneous-baseline.md)。

11.127 ms 来自指定 eager 离线调用的差分均值，含软件路径；正文自己的估计下界为 5.48 ms，并未超过 10 ms。因此“任何 batch 都无法满足”“硬件物理不达标”“必须 1500+ GB/s”都不成立。批均值也不能证明整个实时 SLA；题目约 2000 token 的上下文与所引 decode 1024 条件不同。

应限于该配置/测法下没有达到目标，区分优化机会、请求级时延分布与真正容量限制；不把观测上界倒置为不可突破的硬件下界。对应 0.3-C、M2、5.4。

### F11 · P2 · 2.0 从 TorchDispatch 分解结果推断 aten::linear 不存在

位置：[2.0:670](../../src/L2/2.0-tensor-and-framework.md)。

实际 CPU 2.14.0 中 `torch.ops.aten.linear.default._schema` 为 `aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor`，dispatch 表存在 CompositeImplicitAutograd 等注册。观察层只见 t/addmm，说明该路径在该观察点前已经分解，不表示 ATen 算子不存在。它还与本项目 M1 的真实 linear 注册表分析矛盾。

依据 `aten_linear_exists` 的 schema/完整 dispatch 输出；对应 2.0/2.0b、M1 的层次观察要求。

### F12 · P1 · 4.8 错误声称 SGLang 0.5.19 没有 Qwen3-VL 实现

位置：[4.8:299](../../src/L4/4.8-vlm-serving.md)、[4.8:314](../../src/L4/4.8-vlm-serving.md)，STATUS 的 4.8 行。

官方 v0.5.19 解析到 commit `0bcd822377da7b5718e674eaf9c870d349424dd1`；[固定文件](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/qwen3_vl.py#L1267)包含 `Qwen3VLForConditionalGeneration`，:1658 注册 `EntryClass`。此事实已重新下载归档，并非仅依赖搜索摘要或 main 分支。

这不证明当前远程安装/设备上可以成功启动，但足以否定版本级“没有实现”。若本地缺文件、import 失败或 registry 加载失败，必须保存具体安装源码、异常与选择路径；否则状态只能写“对照未执行，安装/启动原因未查明”。当前 manifest 中同一句描述也只是手写判断，不能自证。对应 4.8-B、章节实现广度要求。

### F13 · P1 · 5.10 的文本输出不能证明同名热更新后 KV 是否失效

位置：[5.10:166](../../src/L5/5.10-lora-multitenant.md)、[5.10:171](../../src/L5/5.10-lora-multitenant.md)，[lab:99](../../labs/L5/lora_kv_invalidation_test.py)。

实验依次使用 ID=1/1/2/3 及不同路径，不是同一已加载 ID 的原位权重替换；只记录生成字符串，没有 `num_cached_tokens`、KV 值、hash 命中/失效事件，也没有 cache-off 的 v2 独立基线。v1 两次相同既兼容正确重算，也兼容复用旧 v1 缓存；v2 文本不同也不能排除复用了错误的前缀、只在后续位置使用 v2。

正文声称“没有继续使用旧 KV”，复审开始时 STATUS 则称“KV 未失效行为已验证”，两个相反结论都超出[工件](../../results/crater/lora-recovery-20260912-1027/kv/kv_invalidation.json)能力。STATUS 已更正为未验证，正文和 lab 仍待修订。固定源码以 `lora_name` 做 key，名称不变本身更不能推出 cache miss。应保留输出差异事实，重新设计版本/命中/基线验证。对应 5.10-C，不能以此关闭热更新契约。

### F14 · P1 · 5.13 的“真实 recurrent state”实际是脚本手工估算，遗漏独立状态

位置：[5.13:150](../../src/L5/5.13-hybrid-architecture-state.md)、[hybrid_state_audit.py:49](../../labs/L5/hybrid_state_audit.py)、[hybrid_rollback_cost.py:39](../../labs/L5/hybrid_rollback_cost.py)。

`inspect_layer_states` 没有读任何运行期 state tensor，只用 config 和默认 `conv_width=4` 拼出 `[10,4,256]`，统一假设 BF16；随后回滚 lab 把它固化为“真实层配置”。这样可以测一组真 tensor 的 copy 时间，但不能称为 RecurrentGemma 完整状态或任务 A/C 完成。

重新读取的 [Transformers v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L474) 明确有 `conv1d_state` 与独立 FP32 `rg_lru.recurrent_states`。仅调用原文 `_setup_cache`，给定 hidden=lru=2560、conv1d_width=4、B=1，得到 `[1,2560,3]` BF16（15360 B）及 `[1,2560]` FP32（10240 B），并非正文的一份 20480 B tensor。

这个小调用证明参考实现的状态种类与公式，**没有加载受限模型权重，也不冒充旧 vLLM 运行期布局**；原 checkpoint config 联网获取返回 401，已保留。正确修订需固定实际执行实现并打印其全部 conv/recurrent/KV 状态。现有 copy 读数保留为合成池基准，完整模型回滚与字节账重新验收。对应 5.13-A/C。

### F15 · P2 · 5.13 容量比例的方向、极限及自测答案互相矛盾

位置：[5.13:196](../../src/L5/5.13-hybrid-architecture-state.md)、[5.13:603](../../src/L5/5.13-hybrid-architecture-state.md)、[5.13:621](../../src/L5/5.13-hybrid-architecture-state.md)。

即使暂且采用章内旧假设，比例 `(368640+8192*S)/(114688*S)` 也随 S **下降**，从 S=256 的 8.3984% 到 S=2048 的 7.2998%，极限为 1/14；不会“随后续上下文线性上升”，也不趋于简单的 8/26 层数比，因为两模型的每层 KV 宽度不同。旧小结仍给 3.6%，自测仍把 prefill 时间次线性归因于定长状态，和正文已改的解释冲突。

应统一正文、图/表、自测及小结，并按 F14 重建真实状态口径；窗口 attention 超过窗口后的缓存策略还应独立分析，不能把上述简化线性公式无限外推。补充反例只检验章内算式，不能作为修正后的实模容量数据。对应 5.13-A/D。

### F16 · P2 · 5.13 错称 recurrent 层不能跨 batch 摊薄权重读取

位置：[5.13:569](../../src/L5/5.13-hybrid-architecture-state.md)、[5.13:629](../../src/L5/5.13-hybrid-architecture-state.md)。

recurrent state 是每序列独有，但线性投影权重仍在 batch 内共享。固定 RecurrentGemma 实现的 linear_x、linear_y、linear_out 都对批量输入用同一组矩阵；不能推出“只有 attention 层的 KV 拼接能带来 batch 收益”。KV 拼接本身也不是权重摊薄的来源。

应区分每序列状态流量、共享权重 GEMM 与序列方向的递推依赖，具体 batch 收益再用同路径测量。源码事实不支持当前反向的一般规律。对应 5.13-A/D。

### F17 · P2 · 4.4 将有放回的独立抽样期望当成 top-k 专家数上界

位置：[4.4:164](../../src/L4/4.4-moe.md)。

`E(1-(1-1/E)^(Tk))` 是 Tk 次独立有放回抽样的期望。每 token top-k 选择不同专家时，在跨 token 独立均匀假设下应为 `E(1-(1-k/E)^T)`；前者不是普适上界。最小反例 E=2,k=2,T=1：实际一定选 2 个，旧公式只有 1.5。紧邻文字把均匀份额 1.56% 称“理论上限”也不成立，集中路由可以超过它。

应标明概率模型与独立性条件，使用期望而非样本上界措辞；吞吐/DRAM 读取更不能直接由路由并集计数确定。依据补充反例 `moe_counterexample`；对应 4.4-C。

### F18 · P2 · 0.3 的 decode KV 字节公式多乘因子，数值又少了一个数量级

位置：[0.3:152](../../src/L0/0.3-heterogeneous-baseline.md)。

Qwen3-1.7B 的 28 层、8 KV heads、head_dim=128、BF16、S=1024，应为 `2*28*8*128*2*1024=117440512 B`，约 117.44 MB。原式多了一个独立的 2，结果却写 11.7 MB；代入后用于算术强度与硬件归因。0.2/5.13 引用的同 checkpoint 账本已有 `114688 byte/token`，可以直接交叉验证。

应由同一组配置重算公式和单位，区分参数中哪些矩阵在该步实际读取；`0.997≈1.03` 也不是该算式精度下的正确计算。对应 0.3-A/C、0.2。

### F19 · P2 · 3.3 仍将等长输入下的分页加速归因于省掉 padding 扫描

位置：[3.3:339](../../src/L3/3.3-decode-attention.md)、[3.3:687](../../src/L3/3.3-decode-attention.md)。

正文前部已限定派生带宽不能证明原因，但此处及自测仍把 1.33× 归因为“形状自由度”“不用扫没用槽位”。实际 [section_D](../../labs/L3/paged_decode.py) 是 `lengths=[8192]*4`，连续输入同为 8192，没有变长 padding 可省；自测又写成 8×8192，混入 section_C 的配置。两种后端/kernel 也不同。

因此 0.0207/0.0276 ms 可以是该路径的比较，不能证明所述 padding 机制。应同步修正所有解释与 shape，保留 kernel/调度/缓存等待验证归因；若要检验变长收益，另做同数据同有效工作量对照。对应 3.3-B；上轮 R30 的因果约束没有贯彻全章。
