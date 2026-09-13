---
machine: 本地；计时与计数器按实际权限分开
measured: 2026-09-12
deps: 0.2（资源账本）
---

## 本章回答三个问题

性能数字取决于计时范围、预热状态、输入和统计方法。本章介绍如何控制这些条件，并区分测量结果、公式估计与原因解释。

1. 计时包含哪些工作，预热、同步和缓存怎样影响结果？
2. 重复、轮转与独立样本怎样构成可信统计？
3. 如何区分观测、公式预测和机制解释？

---

## 心智模型：测量的五层代价

一次调用可能涉及以下阶段，计时方法决定其中哪些阶段被计入：

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" class="figure">
  <style>
    .lbl { font: 12px ui-monospace, monospace; fill: var(--fg-dim); }
    .lbl-b { font: 600 13px ui-monospace, monospace; fill: var(--fg); }
    .tiny { font: 10px ui-monospace, monospace; fill: var(--fg-dim); }
    .bar { fill: var(--accent); opacity: 0.7; }
    .bar-ghost { fill: var(--fg-faint); opacity: 0.2; }
    .arr { stroke: var(--fg-faint); stroke-width: 1; fill: none; }
  </style>

  <text x="16" y="20" class="lbl-b">观察边界示意（不按比例）</text>

  <!-- Timeline bars -->
  <rect x="80" y="40" width="40" height="24" class="bar-ghost"/>
  <text x="16" y="57" class="tiny">context</text>
  <text x="125" y="57" class="tiny">初始化</text>

  <rect x="80" y="72" width="60" height="24" class="bar-ghost"/>
  <text x="16" y="89" class="tiny">compile</text>
  <text x="145" y="89" class="tiny">JIT 编译</text>

  <rect x="80" y="104" width="45" height="24" class="bar-ghost"/>
  <text x="16" y="121" class="tiny">inputs</text>
  <text x="130" y="121" class="tiny">缓存冷启动</text>

  <rect x="80" y="136" width="30" height="24" class="bar"/>
  <text x="16" y="153" class="tiny">steady</text>
  <text x="115" y="153" class="lbl-b">稳态样本</text>

  <!-- Components breakdown -->
  <text x="16" y="190" class="lbl-b">以下区间有重叠，不能按条长求和：</text>
  
  <rect x="80" y="200" width="200" height="20" class="bar"/>
  <text x="290" y="215" class="tiny">设备 Event 区间</text>

  <rect x="80" y="230" width="60" height="20" class="bar" opacity="0.5"/>
  <text x="150" y="245" class="tiny">提交区间</text>

  <rect x="140" y="260" width="140" height="20" class="bar" opacity="0.3"/>
  <text x="290" y="275" class="tiny">等待完成</text>

  <!-- Pitfalls -->
  <g transform="translate(0, 310)">
    <text x="16" y="0" class="lbl-b">常见陷阱</text>
    
    <text x="16" y="20" class="lbl">❌ 只跑一次 → 包含初始化</text>
    <text x="16" y="40" class="lbl">❌ 未等完成 → 不能当完整调用时间</text>
    <text x="16" y="60" class="lbl">❌ 工作集在 L2 → 宣称 DRAM 带宽</text>
    <text x="16" y="80" class="lbl">✓ 预热 + 同步 + 重复 + 标注 L2</text>
  </g>
</svg>

计时包含初始化时，应报告整个调用的时间；测量 DRAM 带宽时，还需排除工作集主要命中 L2 的情况。

---

## 机制：明确开始与结束事件

一条 CUDA 调用通常先由 CPU 提交，再由 GPU 执行。主机函数返回与设备完成不是同一时刻；主机也可能因分配、排队或资源不足而阻塞，所以“没有同步”不能一律解释成纯 launch 开销。

设主机开始提交的时间为 $h_0$，最后一次提交返回为 $h_1$，等待设备事件完成后为 $h_2$；设备流中的首尾事件为 $e_0,e_1$。三种区间为：

$$T_{submit}=h_1-h_0,\quad T_{complete}=h_2-h_0,\quad T_{event}=e_1-e_0.$$

`T_complete` 包含主机提交、调度与等待；Event 区间反映两个事件之间的设备时间，其中也可能包含等待后续主机提交造成的空隙。它不是把 profiler 中所有 kernel duration 相加得到的量。CPU 与 GPU 会并行推进，不能把这三种时间相加；不同批次中位数的差也不是可验证的阶段分解。

一次同步一条请求，与批量提交后同步一次，回答的是不同问题。前者包含每次等待带来的提交间隙；后者更接近持续提交的吞吐条件。将 20 次批量时间除以 20 可以报告平均每次成本，但不能用一个均值计算逐请求 p95，更不能称有 20 个独立实验。

## 工程实现：CUDA Event 的真实边界

本批源码取自实际运行的 PyTorch `2.13.0+cu130`，构建提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`。安装源码 `torch/cuda/streams.py:199` 的 `Event.record` 默认选当前 stream；`:234` 的 `elapsed_time` 返回毫秒；`:245` 的 `synchronize` 等待事件捕获的工作完成并阻塞主机。

{{srcfold:results/crater/M2/20260913-protocol-a/torch_cuda_streams.py:199-258}}

这里的 wrapper 解释 API 边界，不能仅凭 Python 文件推断设备上实际运行哪些 kernel。因此实验还独立保存了一次 profiler trace：本例有 reduce、memset 和加法 kernel，`sum` 不能直接称为“纯读取指令”。正常计时不启用 profiler。

### 设计取舍

**事件归属。** 计时事件与被测工作必须处在预期 stream；跨流的依赖应显式建立。事件包装对象与 CUDA handle 都在正常采样前准备，避免把首次创建混入每个样本。设备 Event 的创建仍不等于被测算法的工作。

**输入与状态。** 正常计时排除输入生成及 H2D，保留归约输出分配和完整 Python 调用；这定义的是设备驻留输入下的调用成本。allocator 已预热，但输出不是显式预分配，因此不能称为完全不含 allocator 的 kernel 时间。

**对照选择。** 基线直接 `x.sum()`；候选把输入分成两半，各自 sum 后相加。二者读入相同数据、计算同一个和，候选多出归约与加法调用。本例用它说明增加分块未必有收益，不将它包装成上游提出的优化。浮点归约顺序改变时必须先做数值检查。

**同步策略。** 单次同步适合回答独立调用完成时间；批量末尾同步适合回答持续提交成本。把快的一种计时方式挑给候选、慢的一种给基线，会把测量协议差异误报成实现收益。

## 工程实现：可复用的归约测量协议

`measurement_protocol.py` 先写 `manifest.json` 和 `cases.json`，再运行。输入主扫描使用 FP32 全 1 张量，规模由目标设备返回的 `L2_cache_size` 确定；小型数值检查另用 seed=0/1/2 的有符号随机值、零值和非连续输入。18 组小例使用 CPU FP64 参照，容差为 `atol=1e-4, rtol=1e-5`；大例的和已知，按精确值核对。

每个配置预热 10 次，运行 5 轮；每轮基线/候选交替顺序，每种方法保留 20 个 Event 样本及整个批次的主机提交、完成时间。这里有 100 个计时样本，但只有 5 个运行轮次；轮次之间是否独立仍受时钟、后台负载和缓存历史影响。

{{srcfold:labs/M/measurement_protocol.py}}

```bash
python labs/M/measurement_protocol.py \
  --compile-probe --output results/crater/M2/my-new-run
```

命令需要 CUDA 环境；输出目录必须不存在。首次编译实验应在独立、已解释的编译缓存目录运行，目录设置与依赖由环境配置提供。本次 runner 保存了实际缓存设置，不修改共享系统包。程序保存的是第一、二、三次编译函数调用的完整时间，不能从第一项直接抽出“纯编译时间”。

## 动手 lab：实测计时边界

运行硬件为 RTX 5090 D。运行前已有一个占用 916 MiB 的引擎进程，GPU 总占用前后均为 926 MiB；各轮保存利用率、时钟与功率快照。以下是这次共享设备现场的测量，不作为独占环境的峰值性能结论，也不能由采样时利用率为零证明整段没有其他工作。

### 初始化与首次调用分开记录

| 实际采集区间 | 原始秒数 | 包含的工作 |
|---|---:|---|
| import torch | 0.542604012414813 | 当前 Python 进程的 torch 导入 |
| CUDA init + synchronize | 0.10622920002788305 | 此前未初始化的 CUDA 上下文与同步 |
| 首次 96 MiB empty 分配 + 同步 | 0.00021554064005613327 | allocator/驱动分配及等待 |
| 同尺寸释放后再次分配 + 同步 | 0.00003137998282909393 | 预留块可能被复用，reserved 仍为 96 MiB |
| compiled sum 首次调用 | 1.2459423393011093 | 图捕获、代码生成/缓存检查及执行 |
| compiled sum 第二次调用 | 0.00009794998914003372 | 同一 shape 再调用 |
| compiled sum 第三次调用 | 0.000037960708141326904 | 同一 shape 再调用 |

前几次调用的不同不能按固定序号分别归因“驱动、编译、缓存”。实验先显式初始化上下文，再独立计时分配与编译调用；编译内部阶段和缓存命中细节尚未采集，保持 **UNVERIFIED**。

### 同一 48 MiB sum 的三个计时口径

下表是 5 轮中位数，单位 µs；20 次调用的行是**整个批次**，未除以 20。

| 提交方式 | 主机提交区间 | 主机完成区间 | Event 区间 |
|---|---:|---:|---:|
| 1 次调用，末尾同步 | 12.1 | 22.83 | 16.831999644637108 |
| 20 次调用，末尾同步 | 108.401 | 256.471 | 250.20799040794373 |
| 20 次调用，每次同步 | 348.382 | 351.202 | 344.86401081085205 |

如果把第二行 108.401 µs 当成 20 次计算已经完成，会遗漏后续等待；如果把第二、三行差异都归因于 sum kernel 变快，会混入主机提交方式的变化。每次同步的 Event 区间也包含两端事件之间的提交空隙，不能叫“纯 GPU 算术时间”。

## 工作集与缓存：地址复用对照

设备返回 L2 容量 **100663296 bytes，即 96 MiB**。每个输入分别为其 0.5/1/2/4 倍，单个输入加输出及内部临时状态才构成完整工作集。这里将输入大小、整个输入池大小分列，避免把“每次读取字节”与“驻留容量”混为一谈。

| 输入 | 输入池 | 输入能否装入 L2（仅容量判断） |
|---|---|---|
| 48 MiB | 8 × 48 = 384 MiB | 能；不代表已确认命中 |
| 96 MiB | 4 × 96 = 384 MiB | 输入恰好等于容量，完整工作集更大 |
| 192 MiB | 2 × 192 = 384 MiB | 不能 |
| 384 MiB | 2 × 384 = 768 MiB | 不能 |

以下均为 Event 中位数，单位 µs；两种方法的完整驻留/轮转原始结果均已保存。

| 输入 MiB | native 驻留 | native 轮转 | split 驻留 |
|---:|---:|---:|---:|
| 48 | 15.632 | 34.464 | 27.264 |
| 96 | 21.84 | 65.184 | 32.096 |
| 192 | 122.768 | 122.528 | 130.72 |
| 384 | 243.36 | 243.328 | 249.392 |

驻留反复使用同一地址，轮转依次使用池中不同地址；两者输入值都为 1。48/96 MiB 的时间随地址复用条件明显变化，192/384 MiB 的两个中位数接近。**观测**是地址复用条件改变了时间；**机制解释**是缓存复用可能参与其中，但没有 DRAM/L2 计数器，不能断言具体命中率或把全部差值归给 L2。

一个可复查的反例是：若用 48 MiB 输入的字节数除以驻留时间，得到的是这次调用的“有效输入处理速率”；换成轮转地址，时间就从 15.632 µs 变为 34.464 µs。它推翻了“重复读取同一小输入所得速率就是固定 DRAM 带宽”的说法。读写流量、缓存命中和归约临时存储还需计数器核对，不能把有效速率当 DRAM 实测流量。

## 统计：按独立任务或时间窗重采样

服务评测的一条请求记录应包含方法、独立任务/时间窗 ID、成功/拒绝/超时状态以及统一单位的完成时间。不要因为同一请求拆成很多 token，就把每个 token 当成独立任务。新建一个随机输入也不会自动消除设备时钟与排队造成的相关性。

`cluster_bootstrap.py` 按 cluster 抽取整组记录，基线与候选使用同一组 cluster 索引；抽中的 cluster 内部样本全部保留，再计算两种方法的分位数之差。**前提是两种方法的 cluster 真正对应。** 同名“第 1 轮”如果在不同配置块中分别运行，不能只靠编号把它们当同期配对。

本次只对同一 access 配置内交错的 native/split 计算配对区间。48 MiB 驻留配置有 5 轮、每种方法 100 个 Event 样本；候选减基线的中位数差为 **11.63200056180358 µs**，2000 次按整轮重采样的 95% percentile 区间为 **[3.9040008559823036, 16.88000001013279] µs**。轮次少且相邻，区间可靠性有限，不据此声称普遍性能界限。驻留/轮转来自不同配置块，上表只作描述性对照，不给它们套配对区间。

```bash
python labs/M/cluster_bootstrap.py \
  results/local/M2/20260913-statistics/paired-resident-0.5.json \
  --output results/local/M2/my-paired-summary.json \
  --baseline native --candidate split --unit us
```

工具支持 `--q 0.95` 等分位数；样本少时能计算数值，不代表尾部已稳定。失败记录单列计数，成功请求的时延区间是条件分布，不能单独用于排名服务质量。某 cluster 全部失败时，工具拒绝计算条件时延，应改用拒绝率、完成率或 SLO 指标。

合成检查验证了常数平移的区间、重复样本不增加 cluster 数、超时计数保留，以及不匹配配对/只有一个 cluster 时拒绝。它们只验证统计实现，不是服务测量。

{{srcfold:labs/M/cluster_bootstrap.py}}

真实请求/时间窗重采样、5.1 完整请求与 7.1 训练步的协议接入仍为 **UNVERIFIED**，需在 8.3 的发生器与事件定义完成后复用；当前归约样本不能代替请求尾延迟验收。频率与实际后端切换的单因素干预也尚待实施，不能用现有时钟快照证明因果。

## 原始现场

正常采样、启动阶段与 profiler 分开保存；统计输出可由输入 JSON 重算，历史原始工件不覆盖。

{{srcfold:results/crater/M2/20260913-protocol-a/manifest.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/cases.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/startup.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/boundaries.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/samples.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/kernel_names.json}}

{{srcfold:results/crater/M2/20260913-protocol-a/source.json}}

{{srcfold:results/local/M2/20260913-statistics/paired-resident-0.5-summary.json}}

{{srcfold:results/local/M2/20260913-statistics/checks.json}}

## 前沿与陷阱

真实服务中，应先冻结到达轨迹、计时事件和 SLO，再比较策略；请求拒绝、取消、超时也属于系统结果。编译、CUDA Graph 和多流流水会进一步改变提交与完成边界，分别衔接 2.6b、2.7 和 8.3。

- **挑最快样本。** 它通常低估总体成本；保留所有样本、异常点与预先声明的聚合规则。
- **先看结果再丢弃预热。** 容易选择性删掉负面结果；先定义预热条件，同时保留初始化与编译记录。
- **工作集大就宣布 DRAM 饱和。** 容量条件只能排除某些解释，饱和与流量需要进一步证据。
- **零时钟变化就认定无干扰。** 离散遥测不是连续监测，也不能排除其他进程在采样间隙运行。
- **给重复输入虚增置信度。** 增加相同确定性样本只增加条数，不增加独立任务数。

## 自测题

1. 为什么 CUDA Event 区间可能包含设备空闲时间？
2. 用单次同步的基线与批量末尾同步的候选比较，有什么问题？
3. 把同一轮 20 条记录复制十次，能把样本量写成 200 个独立实验吗？
4. 驻留/轮转中位数不同，下一步应采什么来检验缓存解释？

::: fold 答案

1. 事件之间可以等待 CPU 继续提交或等待依赖；事件测量流中两点的时间间隔，不等于所有算术 kernel 时间之和。
2. 改变了同步和提交方式，差值混入协议成本；应统一口径后再比较实现。
3. 不能。cluster 仍为原来的轮次；统计应保留组内相关结构，并说明轮次独立性的假设。
4. 在有权限的同一目标条件下采实际 L2/DRAM 计数器，并核对 kernel、时钟和工作集；其他硬件的计数器不能直接替代目标机归因。

:::
