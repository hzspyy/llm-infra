from pathlib import Path
import json, re
ROOT = Path(__file__).resolve().parents[1]

def apply(ident, edits):
    p = next((ROOT/'src').glob(f'*/{ident}-*.md'))
    t = p.read_text()
    for a,b in edits:
        if a not in t: raise ValueError((ident,a[:80]))
        t=t.replace(a,b)
    p.write_text(t)

def section(ident,start,end,body):
    p=next((ROOT/'src').glob(f'*/{ident}-*.md'));t=p.read_text();a=t.index(start);b=t.index(end,a)
    p.write_text(t[:a]+body.rstrip()+'\n\n'+t[b:])

apply('5.5', [('在这组配置中收益随并发减小','在这组配置中，并发增加时相对收益下降')])
apply('5.1', [
('## 三 · TTFT 与 TPOT','## 三 · 首 token 调用时间与后续步耗时估计'),
('| prompt 长度 | 生成 | TTFT ms | TPOT ms | 总时长 ms | TTFT 占比 |','| prompt 长度 | 额外生成步数 | 首 token 调用 ms | 差分 ms/步 | 65-token 调用 ms | 首 token 调用占比 |'),
('**TTFT 随 prompt 长度线性增长**','**首 token 调用时间随 prompt 长度增加**'),
('**prefill 加 batch 几乎没有收益**（本章 1.24×）。为 prefill 凑批是白等。','**本组 prefill 加 batch 的收益为 1.24×。** 短 prompt 或其他形状不能沿用这一结论。'),
])
section('5.1','关键在**权重字节那一列是常数**','---','''上表由 `labs/L5/prefill_decode.py:96–112` 的公式生成，不是硬件流量实测。它把主要参数都近似为每位置参与矩阵乘的权重，并假设一批共享一次读取，得到 $I\\approx BS$。实际 embedding 查表、只计算末位置 lm_head、KV 和中间张量都需要另算，较细的账见 L0.2。

按这组参照，bf16 线性层的权重模型平衡点为 $232000/1608.6\\approx144.2$ FLOP/byte。普通 decode 的 S=1，因此简化模型对应 batch 约 144，而不是旧脚本打印的 72。**这不是已测得的瓶颈切换点。**

长 prefill 具有更多权重复用机会，小 batch decode 的权重成本较难分摊；但请求整体可能同时受 KV、提交、同步等因素影响。''')
p=next((ROOT/'src').glob('*/5.1-*.md'));t=p.read_text();marker='## 三 · 首 token 调用时间与后续步耗时估计\n'
t=t.replace(marker,marker+'\n脚本分别计时只生成 1 个与生成 65 个 token 的离线 `LLM.generate()` 调用，以两者差值除以 64 估算后续步耗时。首个调用包含请求处理、prefill 和采样，不是 HTTP 客户端 TTFT；原日志的“生成 64”列实际表示额外 64 步。下表保留原始读数并改正列名。\n');p.write_text(t)
section('5.1','**这也说明闭环测试','---','''本实验是一次性提交后等待整批结束的批量测试。它不等同于维持固定并发、完成一条再补一条的闭环压测，也不是按独立到达过程持续发请求的开环压测。两种压测都可用于分析连续批处理，但分别回答固定并发与给定到达率下的问题。''')
section('5.1','<summary>3. 一次提交 16 条','</details>','''<summary>3. 一次提交 16 条与 64 条，为什么吞吐可能不同？</summary>

更多请求使引擎拥有更大的可调度集合，也改变了运行中 batch 的分布。但本次没有记录完整调度轨迹，不能直接断言 64 条时“批始终满载”。

应保存每轮活跃请求数、准入与完成时间，区分初始并发、更长的饱和区间和尾部排空效应。持续负载还需用开环或固定并发闭环测试。''')
apply('4.2', [
('动态范围**完全相同**','正规指数范围接近'),
('### 5.1 bf16 为什么赢了 fp16','### 5.1 更宽指数范围减少了缩放管理需求'),
('这不是 bug，也不是 `use_deterministic_algorithms` 能修的。','这种差异可能来自合法的浮点求值变化，但仍需排除实现错误；确定性开关不保证跨 batch 不变性。'),
('于是 fp16 训练必须配 loss scaling：把 loss 乘一个大数，反向之后再除回去，','fp16 训练常用 loss scaling 缓解小梯度下溢：把 loss 乘缩放因子，反向之后再除回去，'),
('bf16 直接放弃 3 位尾数换 3 位指数，那套机制全部不需要。','bf16 用较少尾数换取更宽指数范围，通常不需要同样的梯度缩放；但不免除非有限值检查。'),
('分辨率的损失可以靠"梯度累加用 fp32"补回来（局部代价），','fp32 累加能减少后续累加误差，但不能恢复已经在低精度输入中丢失的信息；'),
('bf16 的指数位和 fp32 一样（8 位），动态范围相同，这套机制全不需要。\n损失的分辨率可以靠"累加用 fp32"局部补回来。','bf16 与 fp32 的正规指数范围接近，通常可减少缩放需求。fp32 累加只能降低进一步的误差，不能恢复低精度存储时丢失的信息。'),
])
section('3.1','**分块实现**','### 实测','''**分块实现**不把完整 $S\\times S$ 中间矩阵写回显存，但不代表 Q/K/V/O 必然各读写一次。片上容量有限，不同 query tile 可能重复加载 K/V；真实流量取决于 tile、循环和缓存。

### 简化流量模型下的算术强度

若沿用旧脚本的六遍中间矩阵流量假设，朴素实现为

$$I_{naive}=\\frac{4S^2D}{12S^2+6SD}.$$

若进一步假设 Q/K/V/O 各过一遍，非因果 attention 的理想模型为

$$I_{ideal}=\\frac{4S^2D}{8SD}=S/2.$$

前者是特定操作分解的估计，后者是理想化的最少流量参照，都不是硬件计数器读数。因果计算和实际分块的流量需另算。分块的核心收益是避免完整打分矩阵落显存，而不是保证实际算术强度总为 S/2。''')
section('3.1','<summary>1. 为什么朴素 attention','</details>','''<summary>1. 为什么不能把分块 attention 的算术强度一律写成 S/2？</summary>

S/2 来自非因果 FLOP 为 $4S^2D$、Q/K/V/O 各读写一次共 $8SD$ 字节的理想假设。实际 kernel 的片上容量有限，query tile 可能重复读取 K/V，因果 mask 也改变工作量。因此需要注明模型，实际 DRAM 流量需计数器验证。

下表的两列强度是旧脚本公式值，不是直接测量；计时与内存记录仍保留。''')
section('3.2','3.1 已经证明分块 attention','1. FA1','''L3.1 推导了 online softmax 的等价更新，并区分理想流量模型与实际 kernel。分块避免完整打分矩阵落显存后，仍可从并行度、异步执行和非矩阵乘操作中寻找优化机会。

''')
section('4.3','**拿到 1.50×，不是 2×。**','精度：','''实测加速为 1.50×，未达到指令微基准的约 2×。仅凭输入字节量和理论发射率不能定位差额，且 1.2 中更宽 FP8 指令族的吞吐仍未核实。

应对照实际 kernel、tile、输出 dtype、缩放与转换、运行频率和缓存。bf16 输入 64 MiB、FP8 输入 32 MiB 都在本机 L2 容量范围内，是否命中仍需流量记录。**这条趋势未解释，不要引用**为“FP8 GEMM 必然只能获得 1.5×”的依据。''')
# Reader-facing summaries; preserve IDs, order, slugs, existing progress and specs.
briefs={
'0.0':'用两层、四个注意力头的字符模型，逐步检查参数、张量形状、注意力和采样。',
'0.0b':'沿同一个模型检查标签对齐、交叉熵、反向传播和参数更新，比较手算梯度与 autograd。',
'0.1':'沿文本请求梳理 API、调度器、模型执行器与输出处理，区分静态源码入口和运行轨迹。',
'0.2':'从模型配置计算权重、KV 与主要 FLOP，用 roofline 建立有明确假设的性能参照。',
'0.3':'固定模型与输入条件，对比 RTX 5090 D 和 L40S 的 prefill 调用及后续生成时间。',
'0.4':'从 Unicode、字节级 BPE 到 chat template、padding 与 position，检查文本如何成为模型输入。',
'0.5':'检查 logits 过滤、采样、增量解码和停止条件，区分 token 边界与文本边界。',
'1.1':'用微基准观察寄存器、共享内存、缓存与显存，分析延迟、带宽和并行访问需求。',
'1.2':'对照 Tensor Core 指令接口和编译目标，用指令微基准区分发射吞吐与 GEMM 性能。',
'1.3':'比较 pageable 与 pinned 内存、PCIe 链路、NUMA 绑定和双向传输，检查实际数据路径。',
'1.4':'梳理 CUDA 驱动、运行时与工具链，测量初始化、提交、同步及系统调用。',
'1.5':'沿权重加载路径比较存储读取、主机暂存和设备传输，区分带宽观测与路径推断。',
'2.0':'从 storage、offset 和 stride 推导张量寻址，进一步检查 dispatcher、分配器和跨流生命周期。',
'2.1':'通过分支、驻留资源、多流和线程块扫描，理解 CUDA 执行与并行度约束。',
'2.2':'保留编译命令、PTX 与 SASS，检查寄存器分配、指令选择和目标架构兼容性。',
'2.3':'逐步实现连续归约，比较原子操作、分块、向量化与数值累加的代价。',
'2.4':'从朴素 GEMM 到分块和 Tensor Core，结合源码、布局与计时分析优化取舍。',
'2.5':'对照 CUDA、Triton 与其他 kernel 编写栈，区分已测结果、编程接口和未验证路线。',
'2.6':'结合时间线、硬件计数器与自制探针定位开销，区分工具扰动和正常运行。',
'2.6b':'检查 FX 图、AOTAutograd、Inductor 产物与 CUDA Graph，区分捕获、分图、融合和重放。',
'2.7':'检查编译守卫、重编译、动态形状与 autotune，比较具体后端的行为和适用边界。',
'2.8':'为自定义 kernel 注册 schema、fake 与 autograd，检查梯度、图捕获及融合边界。',
'3.1':'推导 online softmax 和分块 attention，比较中间存储、数值误差与实际执行成本。',
'3.2':'围绕并行度、因果分块与异步执行比较 FlashAttention 代际设计，注明硬件验证边界。',
'3.3':'分析 decode 的 KV 读取、头共享与 split-K，并区分分页模拟和真实分页 kernel。',
'3.4':'比较滑动窗口、线性状态与 RoPE 缩放，区分容量模型、合成检索和模型质量证据。',
'4.0':'检查 safetensors 的头部、张量偏移、共享权重与量化清单，连接磁盘格式和加载路径。',
'4.1':'从权重实现 Qwen3 前向，通过中间张量与 HF 对照检查归一化、位置编码和注意力。',
'4.2':'比较浮点格式、舍入与累加顺序，区分同配置确定性和跨 batch 不变性。',
'4.3':'检查 4-bit 打包、缩放和反量化，分别衡量文件容量、执行速度与质量边界。',
'4.4':'从总参数、逐层路由与分组执行分析 MoE，区分结构估算、模拟开销和真实效率。',
'5.1':'比较 prefill 与 decode 的工作量、批量收益和计时口径，分析缓存与传输成本。',
'5.2':'连接 block table、引用计数和前缀身份，比较 vLLM 块哈希与 SGLang 基数树。',
'5.3':'用逐 step 记录检查 token 预算和长 prompt 干扰，明确调度统计能支持的结论。',
'5.4':'比较 eager 与图执行，检查启动、捕获尺寸、padding 和端到端收益的证据边界。',
'5.5':'推导接受长度与总迭代成本，解释拒绝采样和缓存回退，并分析 ngram 实验的统计限制。',
'5.6':'比较语法库的 mask、状态推进、终止与 schema 支持，检查约束如何接入真实引擎。',
'5.7':'并置引擎状态组织方式，逐步实现 nanoserve 的请求推进、分页、前缀复用和输出通道。',
'5.8':'注入取消、超时、断连、背压与块池耗尽，用状态和引用计数检查资源回收。',
'5.9':'比较 vLLM 与 SGLang 的过滤顺序、并列候选和 logprobs，保留数值差异及性能记录。',
}
p=ROOT/'outline.json'; outline=json.loads(p.read_text())
for layer in outline['layers']:
    for mod in layer['modules']:
        if mod['id'] in briefs: mod['brief']=briefs[mod['id']]
p.write_text(json.dumps(outline,ensure_ascii=False,indent=1)+'\n')
print('Final corrections and',len(briefs),'reader summaries updated')
