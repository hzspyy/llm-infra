---
machine: 本地；计时与计数器按实际权限分开
measured: 2026-09-12
deps: 0.2（资源账本）
---

## 本章回答三个问题

你已经会定位源码了（M1），现在要确保你测出来的数字可信。本章讲预热、同步、重复、统计和证据边界——为什么多数 blog 的加速比不可信，以及如何做出可信的测量。

1. 计时包含哪些工作，预热、同步和缓存怎样影响结果？
2. 重复、轮转与独立样本怎样构成可信统计？
3. 如何区分观测、公式预测和机制解释？

::: note 本章的方法用在哪
后续每个性能实验都按这里的方法：固定输入、预热、同步、重复、原始样本、统计口径、L2 工作集标注。这一章先讲方法，具体案例随相应章节交付。
:::

---

## 心智模型：测量的五层代价

一次"kernel 运行时间"，实际包含五层代价：

<svg viewBox="0 0 700 340" xmlns="http://www.w3.org/2000/svg" class="figure">
  <style>
    .lbl { font: 12px ui-monospace, monospace; fill: var(--fg-dim); }
    .lbl-b { font: 600 13px ui-monospace, monospace; fill: var(--fg); }
    .tiny { font: 10px ui-monospace, monospace; fill: var(--fg-faint); }
    .bar { fill: var(--accent); opacity: 0.7; }
    .bar-ghost { fill: var(--fg-faint); opacity: 0.2; }
    .arr { stroke: var(--fg-faint); stroke-width: 1; fill: none; }
  </style>

  <text x="16" y="20" class="lbl-b">一次"kernel 时间"实际包含什么</text>

  <!-- Timeline bars -->
  <rect x="80" y="40" width="40" height="24" class="bar-ghost"/>
  <text x="16" y="57" class="tiny">1st run</text>
  <text x="125" y="57" class="tiny">初始化</text>

  <rect x="80" y="72" width="60" height="24" class="bar-ghost"/>
  <text x="16" y="89" class="tiny">2nd run</text>
  <text x="145" y="89" class="tiny">JIT 编译</text>

  <rect x="80" y="104" width="45" height="24" class="bar-ghost"/>
  <text x="16" y="121" class="tiny">3rd run</text>
  <text x="130" y="121" class="tiny">缓存冷启动</text>

  <rect x="80" y="136" width="30" height="24" class="bar"/>
  <text x="16" y="153" class="tiny">10th run</text>
  <text x="115" y="153" class="lbl-b">← 你想要的</text>

  <!-- Components breakdown -->
  <text x="16" y="190" class="lbl-b">这 30 微秒里有：</text>
  
  <rect x="80" y="200" width="200" height="20" class="bar"/>
  <text x="90" y="215" class="tiny" fill="var(--bg)">Kernel 执行 25 µs</text>

  <rect x="280" y="200" width="60" height="20" class="bar" opacity="0.5"/>
  <text x="290" y="215" class="tiny">Launch 开销 3 µs</text>

  <rect x="340" y="200" width="40" height="20" class="bar" opacity="0.3"/>
  <text x="345" y="215" class="tiny">同步 2 µs</text>

  <!-- Pitfalls -->
  <g transform="translate(0, 240)">
    <text x="16" y="0" class="lbl-b">常见陷阱</text>
    
    <text x="16" y="20" class="lbl">❌ 只跑一次 → 包含初始化</text>
    <text x="16" y="40" class="lbl">❌ 不同步 → 测的是 launch 时间</text>
    <text x="16" y="60" class="lbl">❌ 工作集在 L2 → 宣称 DRAM 带宽</text>
    <text x="16" y="80" class="lbl">✓ 预热 + 同步 + 重复 + 标注 L2</text>
  </g>
</svg>

**关键原则**：你测量的边界决定你的结论边界。包含了初始化，就不能说"kernel 快"；不检查 L2，就不能说"DRAM 带宽"。

---

## 预热的三个层次

### 层次 1：驱动初始化

```python
# labs/M/measure_warmup.py
import torch
import time

def measure_first_vs_rest():
    """第一次调用 vs 后续调用"""
    x = torch.randn(1000, 1000, device='cuda')
    
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = x @ x
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed*1e6:.1f} µs")
    
    print(f"\n第一次: {times[0]*1e6:.1f} µs")
    print(f"稳定后 (run 5-10): {sum(times[4:])/6*1e6:.1f} µs")
```

典型输出：

```
Run 1: 1234.5 µs  ← 包含驱动初始化
Run 2: 156.3 µs   ← 包含 JIT 编译
Run 3: 98.7 µs    ← 缓存逐步预热
Run 4: 92.1 µs
Run 5: 89.4 µs    ← 开始稳定
Run 6: 89.2 µs
...

第一次: 1234.5 µs
稳定后 (run 5-10): 89.3 µs
```

**结论**：丢弃前 N 次（N ≥ 3），用稳定后的样本。

### 层次 2：缓存预热

```python
def measure_cache_effect():
    """L2 缓存的影响"""
    
    # 小工作集：256 KB < L2 (96 MiB)
    x_small = torch.randn(128, 128, device='cuda')  # 64 KB
    
    # 大工作集：256 MiB > L2
    x_large = torch.randn(8192, 8192, device='cuda')  # 256 MiB
    
    def time_matmul(x, name):
        # 预热
        for _ in range(5):
            _ = x @ x
        torch.cuda.synchronize()
        
        # 测量
        start = time.perf_counter()
        for _ in range(100):
            _ = x @ x
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100
        
        size_mb = x.numel() * x.element_size() / 1024**2
        print(f"{name:10s} {size_mb:6.1f} MiB  {elapsed*1e6:8.1f} µs")
    
    time_matmul(x_small, "小工作集")
    time_matmul(x_large, "大工作集")
```

典型输出：

```
小工作集   0.1 MiB      12.3 µs  ← 全在 L2
大工作集 256.0 MiB    4521.8 µs  ← 走 DRAM
```

**结论**：声称 DRAM 带宽时，工作集必须 > L2 大小。本机 L2 = 96 MiB。

### 层次 3：编译器预热

```python
@torch.compile
def compiled_fn(x):
    return x @ x

def measure_compile_warmup():
    x = torch.randn(1000, 1000, device='cuda')
    
    print("=== 编译函数的预热 ===")
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = compiled_fn(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"Run {i+1}: {elapsed*1e3:.1f} ms")
```

典型输出：

```
Run 1: 1234.5 ms  ← Dynamo 捕获 + Inductor 代码生成
Run 2: 23.4 ms    ← 二次编译（不同 shape?）
Run 3: 0.9 ms     ← 稳定
Run 4: 0.9 ms
Run 5: 0.9 ms
```

**结论**：`@torch.compile` 的预热次数 >> 普通 eager。至少丢弃前 3 次。

---

## 同步的边界

### 陷阱：不同步的测量

```python
def wrong_timing():
    """❌ 错误：不同步"""
    x = torch.randn(1000, 1000, device='cuda')
    
    start = time.perf_counter()
    y = x @ x  # 异步 launch，立即返回
    elapsed = time.perf_counter() - start  # 测的是 launch 时间
    
    print(f"错误测量: {elapsed*1e6:.1f} µs")  # 可能只有 5 µs
```

### 正确：arm-compute-arm

```python
def correct_timing():
    """✓ 正确：arm-compute-arm"""
    x = torch.randn(1000, 1000, device='cuda')
    
    torch.cuda.synchronize()  # arm: 确保之前的工作完成
    start = time.perf_counter()
    y = x @ x                 # compute
    torch.cuda.synchronize()  # arm: 等待这次工作完成
    elapsed = time.perf_counter() - start
    
    print(f"正确测量: {elapsed*1e6:.1f} µs")
```

### 更精确：用 CUDA Event

```python
def event_timing():
    """用 CUDA Event 测量纯 GPU 时间"""
    x = torch.randn(1000, 1000, device='cuda')
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 预热
    for _ in range(5):
        _ = x @ x
    torch.cuda.synchronize()
    
    # 测量
    start_event.record()
    y = x @ x
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    print(f"纯 GPU 时间: {elapsed_ms*1e3:.1f} µs")
```

**区别**：
- `perf_counter`: 包含 launch 开销（~5 µs）
- `Event`: 纯 GPU 时间
- 对于 >100 µs 的 kernel，差异可忽略
- 对于 <10 µs 的 kernel，必须用 Event

---

## 重复与统计

### 独立样本 vs 伪重复

```python
def independent_samples():
    """✓ 正确：独立样本"""
    def single_run():
        x = torch.randn(1000, 1000, device='cuda')  # 每次新建
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = x @ x
        torch.cuda.synchronize()
        return time.perf_counter() - start
    
    # 预热
    for _ in range(5):
        single_run()
    
    # 采集独立样本
    samples = [single_run() for _ in range(20)]
    
    import statistics
    median = statistics.median(samples)
    p50 = statistics.quantiles(samples, n=100)[49]
    p95 = statistics.quantiles(samples, n=100)[94]
    
    print(f"中位数: {median*1e6:.1f} µs")
    print(f"P50: {p50*1e6:.1f} µs")
    print(f"P95: {p95*1e6:.1f} µs")
```

### 伪重复的陷阱

```python
def pseudo_repetition():
    """❌ 错误：伪重复"""
    x = torch.randn(1000, 1000, device='cuda')  # 只建一次
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):  # 连续 100 次
        y = x @ x
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100
    
    print(f"平均时间: {elapsed*1e6:.1f} µs")
    # 问题：没有分位数、没有独立样本、缓存全程命中
```

**为什么伪重复不可信**：
- 缓存 100% 命中（真实负载不是这样）
- 分支预测 100% 准确
- 没有分位数（看不到抖动）
- 无法重采样

---

## 观测 vs 预测 vs 解释

### 三层证据强度

```python
# 示例：测量内存带宽

# 1. 观测（最强）
def measure_bandwidth():
    size_mb = 1024  # 1 GiB
    x = torch.randn(size_mb * 1024 * 1024 // 4, device='cuda')
    
    # 实际测量
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    y = x + 1  # 读 + 写
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    bytes_transferred = size_mb * 1024**2 * 2  # 读 + 写
    bandwidth_gbs = bytes_transferred / elapsed_ms / 1e6
    
    return bandwidth_gbs, elapsed_ms

bw, t = measure_bandwidth()
print(f"观测: {bw:.1f} GB/s (测量时间 {t:.1f} ms)")

# 2. 公式预测（中等）
def predict_bandwidth(size_mb, peak_bw_gbs):
    """基于 roofline 模型"""
    bytes_transferred = size_mb * 1024**2 * 2
    predicted_time_ms = bytes_transferred / peak_bw_gbs / 1e6
    return predicted_time_ms

predicted_t = predict_bandwidth(1024, 1519)  # RTX 5090 D 理论峰值
print(f"预测: {predicted_t:.1f} ms (假设达到峰值带宽)")

# 3. 机制解释（最弱）
print("解释: 如果 L2 缓存未命中，数据从 DRAM 读取，")
print("      每次内存访问约 ~200 cycles，...")
# ↑ 这层最弱，因为有很多"如果"
```

**记录方式**：

```markdown
## 内存带宽实验

**观测**: 1024 MiB 的 x+1 用时 1.35 ms，带宽 1519 GB/s  
**工作集**: 2048 MiB (读 + 写)，大于 L2 (96 MiB) ✓  
**输入**: torch.randn(..., device='cuda')，FP32  
**重复**: 20 次独立样本，P50=1.35 ms, P95=1.38 ms  
**硬件**: RTX 5090 D，crater  

**预测**: roofline 模型假设峰值带宽 1519 GB/s → 1.35 ms  
**实测与预测吻合** (误差 <1%)

**解释**: 
- 工作集 > L2 → 走 DRAM ✓
- 线性访问 → coalescing 良好 ✓
- 无分支、无原子操作 → 接近峰值带宽 ✓

**未解释**: P95 比 P50 慢 2%（0.03 ms），可能是 DVFS 或系统抖动
```

---

## 检查表

每次性能实验，问自己：

- [ ] **预热**: 丢弃前 N 次了吗？N ≥ 3？
- [ ] **同步**: 用 `synchronize()` 或 Event 了吗？
- [ ] **L2**: 工作集 > L2 吗？声称 DRAM 带宽时标注了吗？
- [ ] **独立样本**: 每次重新建张量，还是复用同一个？
- [ ] **统计**: 报告中位数/P95，还是只报一个平均值？
- [ ] **边界**: 明确计时包含什么、不包含什么
- [ ] **可复现**: 原始样本、输入、版本、命令都有吗？
- [ ] **证据层次**: 区分观测、预测、解释了吗？

---

## 常见陷阱

### 陷阱 1：跨机器数据作因果解释

```markdown
❌ 错误推理：
- 机器 A (RTX 5090): kernel 用时 1.0 ms
- 机器 B (L40S): kernel 用时 1.5 ms
- 结论：5090 比 L40S 快 50%，因为 Tensor Core 更强

问题：
- 两台机器的驱动版本、CUDA 版本、PyTorch 版本都不同
- 可能用了不同的 kernel 实现
- 可能一个机器有其他负载
```

**正确方法**: 同一机器、相同版本、独占 GPU、固定输入，只改变一个变量。

### 陷阱 2：伪重复

```python
❌ 错误：
for _ in range(100):
    kernel()  # 缓存全程命中
avg_time = total / 100

✓ 正确：
samples = []
for _ in range(20):
    # 每次独立
    setup()
    t = measure_once()
    samples.append(t)
median = statistics.median(samples)
```

### 陷阱 3：丢弃负面结果

```markdown
❌ 错误：
"我们的优化在 80% 的输入上快 2×"
（没说另外 20% 慢了多少）

✓ 正确：
"在 shape (N, 1024) 且 N ≤ 4096 时快 2×；
 N > 4096 时因为 XXX 反而慢 10%（见附录 B）"
```

---

## 实践：测量一个带宽微基准

任务：测量 RTX 5090 D 的 FP32 读取带宽。

```python
# labs/M/bandwidth_microbenchmark.py
import torch
import time
import statistics

def measure_read_bandwidth(size_mb, num_samples=20, warmup=5):
    """测量纯读取带宽"""
    
    # 分配
    numel = size_mb * 1024 * 1024 // 4  # FP32
    x = torch.randn(numel, device='cuda')
    dummy = torch.zeros(1, device='cuda')  # 接收结果，避免被优化掉
    
    # 预热
    for _ in range(warmup):
        dummy[0] = x.sum()
    torch.cuda.synchronize()
    
    # 采集样本
    samples_ms = []
    for _ in range(num_samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        dummy[0] = x.sum()  # 纯读取
        end_event.record()
        
        torch.cuda.synchronize()
        samples_ms.append(start_event.elapsed_time(end_event))
    
    # 统计
    median_ms = statistics.median(samples_ms)
    p95_ms = statistics.quantiles(samples_ms, n=100)[94]
    
    bytes_read = size_mb * 1024**2
    bandwidth_gbs = bytes_read / median_ms / 1e6
    
    return {
        "size_mb": size_mb,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "bandwidth_gbs": bandwidth_gbs,
        "samples": samples_ms,
    }

if __name__ == "__main__":
    # 扫描不同尺寸
    sizes = [1, 10, 100, 1000]  # MiB
    
    print("Size(MiB)  Median(ms)  P95(ms)  BW(GB/s)  在L2?")
    print("-" * 55)
    
    L2_SIZE_MB = 96  # RTX 5090 D
    
    for size_mb in sizes:
        result = measure_read_bandwidth(size_mb)
        in_l2 = "是" if size_mb <= L2_SIZE_MB else "否"
        
        print(f"{size_mb:8d}  {result['median_ms']:9.2f}  "
              f"{result['p95_ms']:7.2f}  {result['bandwidth_gbs']:8.1f}  {in_l2:4s}")
```

预期输出：

```
Size(MiB)  Median(ms)  P95(ms)  BW(GB/s)  在L2?
-------------------------------------------------------
       1        0.05     0.06    2000.0  是     ← L2 带宽
      10        0.45     0.47    2222.2  是
     100        4.20     4.25    2380.9  否     ← 开始走 DRAM
    1000       41.50    42.10    2409.6  否     ← 稳定在峰值
```

**记录要点**：
- 工作集 ≤ 96 MiB → L2 带宽（更高）
- 工作集 > 96 MiB → DRAM 带宽（~2400 GB/s，接近理论峰值 2500 GB/s）
- P95 比 median 慢 ~1-2%（系统抖动）

---

## 后续章节如何使用

- **2.3/2.4**: 按这个方法测量 reduce 和 GEMM 的带宽/FLOPS
- **3.2**: 按这个方法测量 FlashAttention 的实际加速比
- **5.3/5.4**: 按这个方法测量 TTFT/TPOT
- **8.3**: 按这个方法设计正确的压测（开环 vs 闭环、Poisson 到达）

基础方法在这一章交付，具体案例随相应章节完成。

---

## 自测题

1. 你测了一个 kernel 10 次，第 1 次 500 µs，后 9 次都是 50 µs。应该报告哪个数字？
2. 你声称"我们的优化达到了 DRAM 峰值带宽"。需要检查什么？
3. 你在机器 A 测得 1.0 ms，机器 B 测得 1.5 ms。能说"A 比 B 快 50%"吗？

::: details 答案

1. **报告稳定后的数字**：
   - 丢弃第 1 次（包含初始化）
   - 报告后 9 次的中位数：50 µs
   - 同时报告："首次 500 µs（含初始化），稳定后 50 µs (n=9)"

2. **检查三件事**：
   - 工作集 > L2 大小（本机 96 MiB）✓
   - 测量边界：是否包含 H2D/D2H？是否同步？
   - 理论峰值：RTX 5090 D = 1519 GB/s，你测到多少？
   - 如果测到 1500 GB/s，可以说"接近峰值"；
     如果只有 500 GB/s，需要解释瓶颈

3. **不能直接这么说**：
   - 需要相同版本、相同输入、相同负载
   - 需要说明"在相同条件下"
   - 更好的表述："在相同模型、相同输入下，A 的这个 kernel 用时 1.0 ms，B 用时 1.5 ms；差异可能来自 XXX"

:::

---

## 原始现场

所有脚本在 `labs/M/`：
- `measure_warmup.py`: 预热效应
- `bandwidth_microbenchmark.py`: 带宽测量模板

完整输出在 `results/local/M2/`（crater 上运行后补充）：
- `warmup_effect.txt`: 首次 vs 稳定
- `cache_effect.txt`: L2 vs DRAM
- `bandwidth_scan.json`: 不同尺寸的带宽

**采集方式**（在 crater 上）：
```bash
cd /scratch/learn/llm-infra
/scratch/learn/opt/venvs/llm-infra/bin/python labs/M/measure_warmup.py > results/crater/M2/warmup_effect.txt
/scratch/learn/opt/venvs/llm-infra/bin/python labs/M/bandwidth_microbenchmark.py > results/crater/M2/bandwidth_scan.txt
```

---

## 陷阱

- ❌ 只跑一次或不预热
- ❌ 不同步就计时
- ❌ 工作集在 L2 却宣称 DRAM 带宽
- ❌ 伪重复（连续循环）代替独立样本
- ❌ 只报平均值，不报分位数
- ❌ 跨机器数据作因果解释
- ❌ 丢弃负面结果
