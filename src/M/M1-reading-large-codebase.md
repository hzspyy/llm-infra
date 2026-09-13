---
machine: 本地源码阅读；动态轨迹按章节硬件
measured: 2026-09-12
deps: 0.0（最小完整模型）
---

## 本章回答三个问题

大型框架包含多种语言、生成代码和动态注册，API 名称往往不能直接对应到执行文件。本章以 PyTorch 为例，介绍如何从 API 查到分发规则，再用运行轨迹核对实际调用。

1. 如何从一个 API 定位生成代码、运行路径和真实 kernel？
2. 静态源码与不同层次的 trace 各能证明什么？
3. 如何维护版本、调用边界和原始文件入口？

---

## 心智模型：五个观察层次

API、算子分发、计算图、kernel 调用和机器指令反映了不同层次的信息。定位问题时，先选择能直接观察目标行为的层次。

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" class="figure">
  <style>
    .lbl { font: 12px ui-monospace, monospace; fill: var(--fg-dim); }
    .lbl-b { font: 600 13px ui-monospace, monospace; fill: var(--fg); }
    .tiny { font: 10px ui-monospace, monospace; fill: var(--fg-faint); }
    .box { fill: var(--bg-soft); stroke: var(--rule); }
    .arr { stroke: var(--accent); stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="var(--accent)" />
    </marker>
  </defs>

  <text x="16" y="20" class="lbl-b">从一个 API 到真实执行的五层证据链</text>

  <!-- Layer 1: Python API -->
  <rect x="16" y="36" width="668" height="56" class="box"/>
  <text x="26" y="54" class="lbl-b">1. Python 调用层</text>
  <text x="26" y="72" class="lbl">model.forward() / engine.generate()</text>
  <text x="26" y="86" class="tiny">证据：API 文档、函数签名、torch.jit.trace 输出</text>

  <!-- Layer 2: Framework dispatch -->
  <rect x="16" y="104" width="668" height="56" class="box"/>
  <text x="26" y="122" class="lbl-b">2. 框架分发层</text>
  <text x="26" y="140" class="lbl">aten::linear / dispatcher key / native 实现</text>
  <text x="26" y="154" class="tiny">证据：native_functions.yaml、生成代码、__torch_dispatch__ hook</text>

  <!-- Layer 3: Compiled graph -->
  <rect x="16" y="172" width="668" height="56" class="box"/>
  <text x="26" y="190" class="lbl-b">3. 编译图层</text>
  <text x="26" y="208" class="lbl">FX 节点 / Inductor IR / graph break</text>
  <text x="26" y="222" class="tiny">证据：TORCH_LOGS="+dynamo,+aot" / export 图 / 生成 .py</text>

  <!-- Layer 4: Runtime call -->
  <rect x="16" y="240" width="668" height="56" class="box"/>
  <text x="26" y="258" class="lbl-b">4. 运行时调用层</text>
  <text x="26" y="276" class="lbl">cudaLaunchKernel / stream / event</text>
  <text x="26" y="290" class="tiny">证据：nsys trace、CUDA_LAUNCH_BLOCKING=1、allocator snapshot</text>

  <!-- Layer 5: GPU kernel -->
  <rect x="16" y="308" width="668" height="56" class="box"/>
  <text x="26" y="326" class="lbl-b">5. GPU kernel 层</text>
  <text x="26" y="344" class="lbl">SASS 指令 / 寄存器 / shared memory</text>
  <text x="26" y="358" class="tiny">证据：cuobjdump、ncu --set full、PTX/SASS dump</text>

  <path d="M 350,92 L 350,104" class="arr"/>
  <path d="M 350,160 L 350,172" class="arr"/>
  <path d="M 350,228 L 350,240" class="arr"/>
  <path d="M 350,296 L 350,308" class="arr"/>
</svg>

**关键原则**：你在第 N 层看到的证据，只能证明第 N 层发生了什么。从第 2 层的注册推断不出第 4 层实际用了哪个 kernel；从第 3 层的图节点推断不出第 5 层的指令序列。

---

## 机制：先定义计算，再预测路径

`linear` 的数学语义是 $Y=XW^T+b$，其中最后一维是输入特征，前面的维度可合并成行。二维例子的 `X` 为 `[2,3]`，`W` 为 `[4,3]`，输出为 `[2,4]`。逐元素计算为：

$$Y_{i,o}=\sum_{k=0}^{2}X_{i,k}W_{o,k}+b_o.$$

没有 bias 时省略最后一项。这个定义没有规定必须调用 `mm`、`addmm` 或哪一个 BLAS kernel；实现可以根据布局与设备选择路径，只要保持相同语义。`W.t()` 对本例是交换 shape/stride 的视图，看到 `aten.t` 不能据此增加一次 GPU launch。

预测分支时至少记录维数、shape、stride、dtype、设备、bias 和执行模式。一个简单的静态预测规则是：二维且有 bias 时调用 `addmm`；无 bias 时进入 `matmul`；连续三维且有一维 bias 时先展平再 `addmm`；本例非连续三维输入进入 `matmul` 后再加 bias。规则对应下面的固定源码，不外推到稀疏、MKLDNN、MPS 或编译模式。

## 工程实现：入口、注册与真实源码

本例运行在 PyTorch **2.14.0 CPU**，构建记录中的提交为 `08187d9e0fba026dc8217405802ab5381dc88d90`。源码按这个提交获取并保存 SHA256；版本号、实际 Python 解释器及输入配置见原始现场。源码文件不是运行 trace，注册表也只说明已安装构建的分发规则。

| 查找对象 | 固定提交内的位置 | 能回答的问题 |
|---|---|---|
| 模块调用者 | `torch/nn/modules/linear.py:130` | `Linear.forward` 在什么位置调用 `F.linear` |
| Python binding | `torch/nn/functional.py:2382` | `F.linear` 绑定到哪个 C++ 入口 |
| schema 与注册声明 | `aten/src/ATen/native/native_functions.yaml:3337` | 参数契约及 `CompositeImplicitAutograd: linear` |
| native 实现 | `aten/src/ATen/native/Linear.cpp:85` | 维数、bias、连续性如何决定下一步 |

从项目根目录复现检索；`SOURCE` 指向项目内保存的源码，读者也可换成相同提交的完整 checkout：

```bash
SOURCE=results/local/M1/20260913-source-pin
rg -n 'return F.linear' "$SOURCE/torch/nn/modules/linear.py"
rg -n 'linear =|torch._C._nn.linear' "$SOURCE/torch/nn/functional.py"
rg -n 'func: linear\(|CompositeImplicitAutograd: linear' "$SOURCE/aten/src/ATen/native/native_functions.yaml"
rg -n 'Tensor linear\(|input_dim == 2|at::matmul|at::addmm' "$SOURCE/aten/src/ATen/native/Linear.cpp"
```

`CompositeImplicitAutograd` 表示这一注册通过其他 ATen 算子组合实现，自动微分可以沿内部算子传播。它不表示最终矩阵乘没有设备专用实现。运行时的 `_dispatch_dump_table('aten::linear')` 补充了已安装构建的注册情况；表内的构建机器路径不是读者机器上必然存在的文件。

固定源码中二维分支原文为：

```cpp
if (input_dim == 2 && bias->defined()) {
  // Fused op is marginally faster.
  return at::addmm(*bias, input, weight.t());
}
```

这段条件足以反驳“每次 linear 都分解成 addmm”。注释中的性能判断仍需在目标 shape 与硬件上测量，不能直接当成本实验的性能结论。继续读到 `at::matmul(input, weight.t())`，才能解释没有 bias 的路径。

完整文件保留真实行号，可在源码页检索以上符号：

{{srcfold:results/local/M1/20260913-source-pin/torch/nn/modules/linear.py}}

{{srcfold:results/local/M1/20260913-source-pin/torch/nn/functional.py}}

{{srcfold:results/local/M1/20260913-source-pin/aten/src/ATen/native/native_functions.yaml}}

{{srcfold:results/local/M1/20260913-source-pin/aten/src/ATen/native/Linear.cpp}}

### 设计取舍

**入口与状态。** `nn.Linear` 持有参数，functional API 接受张量，native 层根据张量属性选择实现；追踪器持有自己的事件列表。给模块安装 hook 只观察模块调用边界，内部的 `t` 与 `mm` 并不是它的子模块。

**不变量。** 替换实现必须保持最后一维的线性映射、bias 广播、dtype/device 与自动微分语义。本例只验证前向 CPU FP64；新增后端还需单独检查反向、异常尺寸、流和内存生命周期。

**扩展范围。** 只想理解某个分支时，在调用者外加 `TorchDispatchMode` 足够；改变 `linear` 的所有布局选择则涉及 native 实现；只优化某个设备的 GEMM，应继续追踪 `mm/addmm` 的设备注册。全局替换 Python API 会漏掉直接从 C++ 或 ATen 入口发起的调用。

**观察代价。** Python hook 轻便但边界较粗；dispatcher probe 可见内部算子，但会引入 Python 回调；profiler 能在另一轮运行中记录 ATen 事件。本实验对照有无 probe 的输出，不把带观察器的调用时间作为性能数据，也不声称 observer 没有改变分发成本。

## 工程实现：先冻结预测，再记录实际执行

`dispatch_evidence.py` 在任何调用前写入 `cases.json`。每个 case 包含 seed、布局、bias、源码推导的 GEMM 分支，以及故意保留的错误假设“总是 addmm”。随后逐例保存全部输入值和 stride、模块 hook、dispatcher 序列及另一轮无 dispatcher probe 的 CPU profiler 事件名。

数值参照逐行逐元素累加，不复用 `linear` 或矩阵乘。空输入不调用最大值归约；极值用有限的 $10^4$ 倍输入，避免把溢出混入路径验证。CPU FP64 使用 `atol=1e-10, rtol=1e-12`，覆盖三项乘积的归约舍入；整数 shape 与算子名称按精确值检查。该容差不适用于后续 BF16/GPU 扩展。

核心观察器只有一个转发点：

```python
class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))
```

调用 `func` 让当前算子继续执行；不要在 hook 内重新调用顶层 `F.linear` 来模拟转发。完整实现还包含逐标量参照、先落盘后断言的失败保留与禁止覆盖输出目录。

{{srcfold:labs/M/dispatch_evidence.py}}

{{srcfold:labs/M/version_manifest.py}}

## 动手 lab：三个观察窗口对照

```bash
python labs/M/dispatch_evidence.py --output results/local/M1/my-new-run
```

输出目录必须尚不存在。不要把命令重定向到已有原始工件。实验固定 CPU 单线程、seed=0/1/2，无预热和性能计时；每组只核对一次数值与状态。

36 组用例覆盖二维、连续三维、非连续三维、空 batch、零输入和大数输入，分别启用/关闭 bias。**36/36** 数值检查与静态分支预测通过，最大绝对误差为 **3.637978807091713e-12**，有无 dispatcher probe 的输出逐元素相等。“总是 addmm”的假设在 **21/36** 组中被实际事件否定。

| seed=0 的输入 | module hook | dispatcher 记录中的主要计算 |
|---|---|---|
| `[2,3]`，无 bias | `Linear` | `t → mm` |
| `[2,3]`，有 bias | `Linear` | `t → addmm` |
| `[2,3,3]` 连续，有 bias | `Linear` | `view → t → addmm → view` |
| `[3,2,3]` 非连续，有 bias | `Linear` | `t → clone → _unsafe_view → mm → _unsafe_view → add` |

最后一行还说明布局转换的存在：`transpose(0,1)` 得到 stride `[3,9,1]` 的输入，后续展平路径出现 `clone`。本例启用了 `no_grad`，不要把这里观察到的 `aten.add.Tensor` 强行改写成源码另一分支中的 `add_`；张量类型与变换模式也会影响可见的分解。

练习时先复制 case 定义，再预测一维输入、不同 bias 形状与非连续 weight 的分支，最后运行验证。若预测失败，保留对应输入、输出及事件，沿源码重新定位条件；不要先把预期改成观察结果再宣布预测正确。

## 原始现场

输入、预测、实测事件与参照都保存在以下文件。profiler 文件中的事件名来自 CPU ATen 层，未采集 CUDA launch、SASS 或 GPU 时延。

{{srcfold:results/local/M1/20260913-dispatch-ab/manifest.json}}

{{srcfold:results/local/M1/20260913-dispatch-ab/cases.json}}

{{srcfold:results/local/M1/20260913-dispatch-ab/observations.json}}

{{srcfold:results/local/M1/20260913-dispatch-ab/registration.json}}

{{srcfold:results/local/M1/20260913-dispatch-ab/summary.json}}

{{srcfold:results/local/M1/20260913-source-pin/manifest.json}}

源码可由 manifest 内的 commit URL 重新获取并比对 SHA256。Python 环境取自 `sys.executable` 和当前进程的版本，不用 PATH 中另一个 `python3` 代替实际运行解释器。

## 前沿：把定位方法迁移到引擎和多模态 pipeline

### SDPA：同一 API 的成功分支与拒绝分支

SDPA 的分支预测固定为 `B=1, H=2, S=7, D=8`、causal、dropout=0，seed=0/1/2。运行使用 RTX 5090 D、PyTorch `2.13.0+cu130`，构建提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`；它与前文 CPU linear 的构建不同，分别保存源码与 manifest。

入口在 `attention.cpp:715`，CUDA 分支选择在 `sdp_utils.cpp:1049`。选择函数遍历 `priority_order`，逐一检查能力和用户开关；没有可用路线时报告错误。低精度检查在 `sdp_utils.cpp:803`，当前目标架构下允许 Half/BFloat16，不能把 FP32 输入强行交给这一 Flash 路线。

| 冻结的配置 | 预测 | 三个 seed 的实际结果 |
|---|---|---|
| FP32，只启用 MATH | math 分解 | dispatcher 有两次 `bmm` 及 softmax；CPU FP64 参照均通过 |
| BF16，默认选择 | flash | `aten._scaled_dot_product_flash_attention.default`，参照均通过 |
| BF16，只启用 FLASH_ATTENTION | flash | 同上，参照均通过 |
| FP32，只启用 FLASH_ATTENTION | 拒绝 | 三次均报 `No available kernel. Aborting execution.`，warning 保留 dtype 约束 |

12/12 分支预测符合实际；9 个成功输出通过数值检查，另外 3 个是预期拒绝，不能算成数值通过。BF16 参照使用**已经转成 BF16 的输入再转 CPU FP64**计算，避免将输入量化误差误算成 kernel 误差；本组成功输出最大绝对误差为 `0.003904484136931563`。容差为 FP32 `atol=rtol=1e-5`、BF16 `atol=0.03, rtol=0.02`，只约束这个小输入、前向计算和无 dropout 配置，不代表模型级质量保证。

每个成功 case 在无 dispatcher probe 的另一轮 profiler 中保存 CUDA kernel 事件及 Chrome trace。BF16 的真实事件包含 `pytorch_flash::flash_fwd_kernel`，完整模板参数在原始观察文件；有无 probe 的输出相等。这里把**配置、分发记录和 GPU 事件**对应起来，仍不声称已经检查 SASS 或所有可能后端，也不把 profiler 时间当性能数据。

```bash
python labs/M/sdpa_branch_probe.py --output results/crater/M1/my-sdpa-run
```

命令需要 CUDA 环境。工作集只有三个 `[1,2,7,8]` 输入和小矩阵参照，每组预热 1 次、采集 1 次轨迹，无带宽或缓存结论；不适用 M2 的性能统计口径。

{{srcfold:labs/M/sdpa_branch_probe.py}}

{{srcfold:results/crater/M1/20260913-sdpa/data/manifest.json}}

{{srcfold:results/crater/M1/20260913-sdpa/data/observations.json}}

{{srcfold:results/crater/M1/20260913-sdpa/data/summary.json}}

{{srcfold:results/crater/M1/20260913-sdpa-source/attention.cpp:715-808}}

{{srcfold:results/crater/M1/20260913-sdpa-source/sdp_utils.cpp:1049-1113}}

{{srcfold:results/crater/M1/20260913-sdpa-source/manifest.json}}

### 引擎迁移：同一个请求应在哪里找状态

以下复用 5.11 保存的 vLLM 0.29.0、SGLang 0.5.19 **安装源码**，本轮逐文件计算 SHA256；这是静态关系对照，不新增请求时延或跨进程 trace 结论。

| 问题 | vLLM | SGLang |
|---|---|---|
| 前端在哪里登记请求？ | `AsyncLLM._add_request` 先向 `output_processor` 登记，再 await `engine_core.add_request_async` | HTTP `/generate` 迭代 `tokenizer_manager.generate_request` 返回的异步生成器 |
| 谁组织流式响应？ | 前端请求输出队列向上层消费者返回结果，core 输出经独立协程分派 | `stream_results` 包装 SSE data，`StreamingResponse` 持有生成器 |
| 断连清理在哪里定位？ | 沿路由取消装饰器到前端 abort，再查 core 的请求 ID | `StreamingResponse(background=...)` 关联 `create_abort_task`，再追 tokenizer manager 的取消传播 |
| detokenize 的状态在哪里？ | 当前 V1 前端的输出处理路径 | 独立 `DetokenizerManager` 接收 scheduler IPC 消息，构造时分别初始化 tokenizer 与请求分发器 |

两种方案都必须保持请求 ID 与输出队列的对应。研究取消时，应同时记录前端收到断连、abort 提交和 worker 不再执行三个边界；从 `create_abort_task` 名称只能定位入口，不能证明 GPU 已立即停止。更换 sampler 要继续进入执行侧；修改 SSE 编码则在响应包装层，二者的改动范围不同。

{{srcfold:results/crater/api/20260912-0325/source/vllm/v1/engine/async_llm.py:494-514}}

{{srcfold:results/crater/api/20260912-0325/source/sglang/srt/entrypoints/http_server.py:908-945}}

{{srcfold:results/crater/api/20260912-0325/source/sglang/srt/managers/detokenizer_manager.py:102-126}}

{{srcfold:results/local/M1/20260913-engine-source-index/manifest.json}}

### 迁移案例：Cosmos3-Edge 的生成入口与状态边界

这里固定 Diffusers 提交 `c419dac0152186060246c93a095bc1bfaea342b3`，模型配置固定 `nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`。两者分别标识**实现**和**模型配置**，不能用 model card 的库版本字段替代代码提交。本节是源码分析与独立输入契约实验；权重未加载，完整视频生成与任务质量为 **UNVERIFIED**，留给 10.4 的固定配方实验。

先读 `model_index.json`：`_class_name` 是 `Cosmos3OmniPipeline`，scheduler 为 `UniPCMultistepScheduler`，transformer 为 `Cosmos3OmniTransformer`，VAE 为 `AutoencoderKLWan`。配置中的 `use_native_flow_schedule=true` 和 `default_use_system_prompt=false` 会影响运行路径；文件只是组件配置，不证明已成功加载这些组件。

**第一条边界：生成 pipeline 不等于顶层 reasoner/generator 路由。** 本文件的 `__call__` 接收文本、图像/视频和可选动作条件，准备 latent 后执行去噪。`tokenize_prompt` 将 token ID 交给联合 transformer；它没有先运行独立 text encoder 再把 embedding 传入生成器。不要由 `Omni` 类名推出一个尚未定位的 reasoner 服务入口。

**第二条边界：transformer 内的 `und/gen` 是 token 路径。** `transformer_cosmos3.py:790` 按 `und_len` 分开理解前缀与生成 token。attention 中，`und` 只在理解前缀上做 causal attention；`gen` 的 query 读取拼接的 `und + gen` K/V，使用非因果 attention。两条路径有各自投影和 MLP，最后恢复到联合序列位置。这是一次生成前向内部的结构，不是“先完成独立推理模型，再启动生成模型”的证据。

| 入口或状态 | 固定源码位置 | 约束及改动位置 |
|---|---|---|
| 组件注册与默认配置 | `pipeline_cosmos3_omni.py:414` | `register_modules` 持有 scheduler、transformer、VAE；构造参数和 checkpoint 配置共同决定默认值 |
| 输入检查 | `pipeline_cosmos3_omni.py:1000` | 尺寸按 VAE 空间缩放对齐；图像和视频条件互斥；声音要求 tokenizer 与模型能力同时具备 |
| 文本模板与 token ID | `pipeline_cosmos3_omni.py:1127` | `use_system_prompt=None` 才回退到保存的默认配置；替换 tokenizer 须重新核对特殊 token 和模板 |
| 时间步与 solver 历史 | `pipeline_cosmos3_omni.py:1671`；`scheduling_unipc_multistep.py:1153` | native flow 分支显式传入 sigmas；`step` 修改 `model_outputs`、`last_sample`、`_step_index`，不能把 scheduler 当无状态函数 |
| 多模态 scheduler | `pipeline_cosmos3_omni.py:1683` | 声音/动作 latent 存在时各复制一份 scheduler，避免不同模态轮流推进同一 solver 历史 |
| CFG 两次前向 | `pipeline_cosmos3_omni.py:1713`、`:1753` | 当前 `guidance_scale != 1.0` 时执行无条件分支，不应套用其他 pipeline 的 `>1` 判据 |
| transformer 分流与合并 | `transformer_cosmos3.py:790`、`:818` | 保持 `und_len`、位置 ID 和模态索引一致；更换 attention 后端必须同时保持两种 mask 语义 |
| step-end callback | `pipeline_cosmos3_omni.py:1828` | 在 scheduler 更新后调用；当前只允许暴露 `latents`，返回字典可替换下一步 latent，不直接暴露声音或 solver 历史 |

从项目根目录执行：

```bash
SOURCE=results/local/M1/20260913-cosmos-source
rg -n 'register_modules|check_inputs|self.scheduler|self.transformer|callback_outputs' "$SOURCE/pipeline_cosmos3_omni.py"
rg -n 'und_seq =|gen_seq =|is_causal=|torch.cat' "$SOURCE/transformer_cosmos3.py"
rg -n 'def step|model_outputs|last_sample|_step_index' "$SOURCE/scheduling_unipc_multistep.py"
```

### 扩展点：更换时间步策略，保留模型输出语义

选择 scheduler 作为扩展点，先检查谁决定时间步、谁推进 solver、谁持有历史。当前 pipeline 的 native flow 分支构造从 $1-1/N$ 到 0 的线性 sigma 序列并去掉末项，其中 $N$ 为 scheduler 的训练时间步数，然后调用 `set_timesteps(..., sigmas=sigmas)`。这只是传入 scheduler 的序列；scheduler 内的变换仍须继续分析，不能直接把它写成最终步长。

原始 [PR #14181：Cosmos3 edge support](https://github.com/huggingface/diffusers/pull/14181) 已合并，merge commit 为 `db44fe6638a5c462f3ea521fb065aaa42bee17ce`。它的实际 diff 添加了 `default_use_system_prompt`、`use_native_flow_schedule` 及相应分支。这支持“Edge 的默认模板和时间步策略有显式实现”的源码结论；PR 正文只有简短支持说明，不能据此编造某种 solver 更快、质量更好或上游否决替代方案的理由。完整 PR 响应和 diff 见原始文件。

**设计分析：** 单纯替换 `pipe.scheduler` 很方便，但还必须保持 `prediction_type`、sigma/时间步约定、输入 batch 维和重置协议一致。复制配置只解决参数传递，不证明 solver 数学等价。需要改 callback 来复用 latent 时，应检查每个去噪步开始时的状态是否仍符合 solver 历史；只复制 `latents` 而漏掉多步历史不能保证精确恢复。以上是根据代码提出的检查要求，不是上游给出的否决意见。

### 独立执行输入契约：能检验什么

`cosmos_contract_probe.py` 通过 AST 取出固定文件中的**原始 `check_inputs` 方法**，不改其语句；使用合成的 VAE/transformer 配置运行 12 组输入。它不实例化 pipeline、不导入整个 Diffusers、不加载权重。图像和视频使用 sentinel 表示“条件存在”，不会进行图像解码，因此通过输入检查不等于输入可被完整 pipeline 消费。

```bash
python labs/M/cosmos_contract_probe.py \
  --source results/local/M1/20260913-cosmos-source/pipeline_cosmos3_omni.py \
  --output results/local/M1/my-cosmos-contract
```

**12/12** 组实际结果与执行前保存的预测一致。接受的两例是尺寸对齐的图像条件输入，以及最后一个合法 latent 时间索引；其余 10 例在原方法中抛出 `ValueError`。例如，空间缩放设为 16 时，宽度 831 被拒绝；121 帧、时间缩放 4 对应 `(121-1)//4+1=31` 个 latent 时间位置，索引 30 被接受，31 被拒绝。

未提供 sound tokenizer 时，错误为：

```text
`enable_sound=True` requires a sound-capable checkpoint with a `sound_tokenizer`.
```

请求 `sound_latents` callback 字段也被拒绝。这说明能力检查和 callback 字段契约是两个不同边界：即使未来更换成支持声音的模型，也不能跳过 callback 的字段限制。完整运行结果记录了每个输入和未经改写的错误信息。

{{srcfold:labs/M/cosmos_contract_probe.py}}

{{srcfold:results/local/M1/20260913-cosmos-contract-v2/manifest.json}}

{{srcfold:results/local/M1/20260913-cosmos-contract-v2/observations.json}}

{{srcfold:results/local/M1/20260913-cosmos-source/model_index.json}}

{{srcfold:results/local/M1/20260913-cosmos-source/manifest.json}}

{{srcfold:results/local/M1/20260913-cosmos-source/pipeline_cosmos3_omni.py}}

{{srcfold:results/local/M1/20260913-cosmos-source/transformer_cosmos3.py}}

{{srcfold:results/local/M1/20260913-cosmos-source/scheduling_unipc_multistep.py}}

{{srcfold:results/local/M1/20260913-cosmos-source/pr-14181.json}}

{{srcfold:results/local/M1/20260913-cosmos-source/pr-14181-files.json}}

## 陷阱

- **把注册当执行。** 注册表列出 CUDA math entry 不证明本机调用过 CUDA；本例的环境明确为 CPU。
- **把分解当融合。** `t + mm` 是算子序列，不能由此断言两个 GPU kernel，也不能把一个 launch 自动解释成融合。
- **只看模块名称。** `Linear` hook 看不到内部 `mm`，但 dispatcher 与独立运行的 profiler 可以记录它。
- **拿 kernel 名猜算法。** 名称、设备指令和后端选择要关联同次配置；出现 Tensor Core 指令并不能单独证明使用了 FlashAttention。
- **使用示例行号。** 先保存文件与哈希，再引用真实位置；升级之后重新定位，不把旧行号贴到新版本。

## 自测题

1. 二维输入无 bias 时，为何不能沿 `addmm` 分支解释这次执行？
2. 模块 hook 没记录到 `mm`，什么材料能证明内部仍然执行了矩阵乘？
3. 当前注册表出现 CUDA 条目，为什么不能据此报告 GPU kernel？
4. 如果连续三维输入和非连续输入输出一致，是否能认为转换成本也相同？

::: fold 答案

1. 固定源码的二维 `addmm` 条件要求 `bias->defined()`；本例无 bias 进入 `matmul`，dispatcher 实际记录 `t → mm`。
2. 在相同 case 下检查 dispatcher 的算子序列、另一轮 CPU profiler 的 ATen 事件，并与逐元素数值参照对照。模块 hook 仅标记模块边界。
3. 注册是静态可分派信息；环境和输入都在 CPU，未记录任何 CUDA launch。设备性能需在目标硬件重新采集。
4. 不能。数值一致只核对计算语义；非连续例出现 `clone`，成本需另做无 observer 的计时实验。

:::
