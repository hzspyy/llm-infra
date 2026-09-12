---
machine: 本地源码阅读；动态轨迹按章节硬件
measured: 2026-09-12
deps: 0.0（最小完整模型）
---

## 本章回答三个问题

读完 L0.0 的 150 行玩具模型后，你要开始接触真实代码库：vLLM、PyTorch、SGLang。它们有十万行以上的混合语言代码、生成文件、注册系统、多级抽象。本章用可复现的方法，让你从一个 API 找到真实执行路径。

1. 如何从一个 API 定位生成代码、运行路径和真实 kernel？
2. 静态源码与不同层次的 trace 各能证明什么？
3. 如何维护版本、调用边界和原始文件入口？

::: note 本章的方法用在哪
后续每个核心章节的源码走读都按这里的方法：固定版本、入口检索、调用证据、观察层次标注、原始行号。这一章先讲方法，具体案例随相应章节交付。
:::

---

## 心智模型：五层观察窗口

你要找的不是"代码在哪"，而是"这个行为在五层观察窗口中的哪一层可以看到证据"。

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

## 入口定位的三步法

### 第一步：固定版本与环境

```python
# labs/M/version_manifest.py
import torch
import subprocess
import json
from pathlib import Path

def capture_environment():
    """采集完整的可复现环境快照"""
    manifest = {
        "pytorch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "python": {
            "version": subprocess.check_output(
                ["python3", "--version"], text=True
            ).strip(),
            "executable": subprocess.check_output(
                ["which", "python3"], text=True
            ).strip(),
        },
        "source_locations": {
            "torch_includes": str(Path(torch.__file__).parent / "include"),
            "torch_lib": str(Path(torch.__file__).parent / "lib"),
        },
    }
    
    # 尝试获取 git 提交（如果是从源码安装）
    try:
        torch_path = Path(torch.__file__).parent.parent
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=torch_path,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        manifest["pytorch"]["source_commit"] = git_hash
    except:
        manifest["pytorch"]["source_commit"] = "N/A (wheel install)"
    
    return manifest

if __name__ == "__main__":
    env = capture_environment()
    print(json.dumps(env, indent=2))
```

**为什么这一步不能跳过**：行号会变，文件会移动，实现会重构。没有版本锚点，你读的代码和实际运行的代码可能完全不是一个东西。

### 第二步：从 API 到注册点

用一个具体例子：`torch.nn.functional.linear(input, weight, bias)` 到底调了什么？

```python
# labs/M/trace_dispatch.py
import torch
import torch.nn.functional as F

# 方法 A：最直接 —— 看 Python 绑定
print("=== Method A: Python binding ===")
print(f"F.linear 实际是: {F.linear}")
print(f"文档字符串前 200 字符:\n{F.linear.__doc__[:200]}")

# 方法 B：Hook dispatcher
class DispatchTracer(torch.utils._python_dispatch._TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"  dispatch: {func}")
        return func(*args, **(kwargs or {}))

print("\n=== Method B: Dispatch trace ===")
x = torch.randn(2, 3)
w = torch.randn(4, 3)
with DispatchTracer():
    out = F.linear(x, w)

# 方法 C：看生成的注册表（需要源码）
print("\n=== Method C: Registration lookup ===")
print("在 PyTorch 源码中查找：")
print("  1. aten/src/ATen/native/native_functions.yaml")
print("  2. 搜索 'func: linear'")
print("  3. 找到 dispatch 条目")
```

运行输出：

```
=== Method A: Python binding ===
F.linear 实际是: <built-in method linear of type object at 0x...>
文档字符串前 200 字符:
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

=== Method B: Dispatch trace ===
  dispatch: aten::t.default
  dispatch: aten::addmm.default

=== Method C: Registration lookup ===
在 PyTorch 源码中查找：
  1. aten/src/ATen/native/native_functions.yaml
  2. 搜索 'func: linear'
  3. 找到 dispatch 条目
```

**观察到什么**：
- Python 层看到 `F.linear`
- Dispatcher 层看到它分解成 `t` 和 `addmm`
- 真实 kernel 在 `addmm` 里

这就是"观察层次"：你在不同层看到不同的名字和粒度。

### 第三步：从注册到实现文件

拿到 `aten::addmm` 后，如何找到实际实现？

```bash
# 在 PyTorch 源码目录（v2.13.0 tag）
$ cd pytorch
$ git checkout v2.13.0

# 方法 A：直接搜源码
$ grep -r "TORCH_IMPL.*addmm" aten/src/ATen/native/
aten/src/ATen/native/LinearAlgebra.cpp:  TORCH_IMPL_FUNC(addmm_out)(
...

# 方法 B：看生成的 dispatcher 表
$ python tools/codegen/gen.py --help
# (代码生成脚本，运行后检查 build/aten/src/ATen/RegisterCUDA.cpp)

# 方法 C：用 nm 看符号表（wheel 安装的情况）
$ nm -C $(python3 -c 'import torch; print(torch.__file__.replace("__init__.py","lib/libtorch_cpu.so"))') | grep addmm
# 会看到 at::native::addmm_out_cuda_impl 等符号
```

找到文件后，**记录三件事**：

1. **文件路径与行号**：`aten/src/ATen/native/LinearAlgebra.cpp:1234`
2. **版本锚点**：`pytorch/pytorch @ v2.13.0 (commit abc123)`
3. **观察层次**：这是"框架分发层"的 native 实现，不是 kernel

---

## 动态轨迹的采集

静态源码告诉你"可以走哪些路径"，动态轨迹告诉你"实际走了哪条路径"。

### 运行时层：nsys + 可视化

```bash
# 采集一次 linear 调用的完整轨迹
nsys profile -o linear_trace \
    --capture-range=cudaProfilerApi \
    --trace=cuda,nvtx,osrt \
    python3 trace_linear.py
```

```python
# trace_linear.py
import torch
import torch.cuda.profiler as profiler

x = torch.randn(1024, 512, device='cuda')
w = torch.randn(256, 512, device='cuda')

# 只采集这一段
profiler.start()
torch.cuda.nvtx.range_push("linear_call")
out = torch.nn.functional.linear(x, w)
torch.cuda.nvtx.range_pop()
profiler.stop()
```

打开 `linear_trace.nsys-rep`，你会看到：

```
Timeline:
  [Python] torch.nn.functional.linear
    [CUDA Runtime] cudaLaunchKernel (addmm_kernel)
      [GPU] addmm_kernel<<<128, 256>>>
```

**这一层证明**：实际调用了 `addmm_kernel`，占用 X 微秒，启动参数是 128 blocks × 256 threads。

### Kernel 层：cuobjdump + PTX

```bash
# 找到实际的 kernel 二进制
$ python3 -c 'import torch; print(torch.__file__.replace("__init__.py","lib"))'
/path/to/python3.X/site-packages/torch/lib

$ cd /path/to/python3.X/site-packages/torch/lib
$ cuobjdump -sass libtorch_cuda.so > sass_dump.txt
$ grep -A 20 "addmm" sass_dump.txt
```

你会看到真实的 SASS 指令序列。**这一层证明**：实际用了哪些 Tensor Core 指令（如 `HMMA.16816`）、寄存器分配、内存访问模式。

---

## 维护原始文件入口的方法

每次引用源码，记录：

```markdown
## 源码走读：addmm 的 CUDA 实现

**版本**: pytorch/pytorch @ v2.13.0 (commit `a1b2c3d4`)  
**文件**: `aten/src/ATen/native/cuda/Blas.cpp:456`  
**观察层次**: 框架分发层 → CUDA native 实现

```cpp
// aten/src/ATen/native/cuda/Blas.cpp:456
Tensor& addmm_out_cuda_impl(
    Tensor& result, 
    const Tensor& self,
    const Tensor& mat1, 
    const Tensor& mat2,
    const Scalar& beta, 
    const Scalar& alpha
) {
  // ... 实际代码
}
```

**这段代码做什么**: 检查形状、选择 GEMM kernel、调用 cuBLAS 或 cutlass。

**不能从这段推断**: 具体用了 cuBLAS 的哪个函数、实际 kernel 启动参数。需要第 4 层的 trace。
```

---

## 常见陷阱与识别方法

### 陷阱 1：从 hook 推断算子不存在

**错误推理**：
```python
hooks_seen = []
def hook(module, input, output):
    hooks_seen.append(module.__class__.__name__)

model.register_forward_hook(hook)
model(x)
print(hooks_seen)  # 没看到 LayerNorm
# 结论：模型没用 LayerNorm ❌
```

**为什么错**：Hook 在 `nn.Module` 层，LayerNorm 可能被编译器融合或者用 functional API 调用，绕过了 module hook。

**正确方法**：用 dispatcher hook（第 2 层）或者 `torch.fx.symbolic_trace`（第 3 层）。

### 陷阱 2：把算子分解等同于 kernel 融合

**错误推理**：
```python
# 看到 FX 图里 linear 变成了 t + addmm
# 结论：PyTorch 把两个 kernel 融合成一个 ❌
```

**为什么错**：分解发生在第 2/3 层（Python/图），融合发生在第 3/4 层（Inductor codegen / 实际 launch）。分解后的算子仍然可能分别调用两个 kernel。

**正确方法**：用 nsys 数实际的 launch 次数（第 4 层）。

### 陷阱 3：行号与版本不一致

**错误推理**：
```markdown
源码位置：`aten/src/ATen/native/Linear.cpp:123`
版本：我装的是最新的 PyTorch
```

**为什么错**："最新"是哪个 commit？三个月后"最新"变了，行号也变了。

**正确方法**：
```markdown
源码位置：`aten/src/ATen/native/Linear.cpp:123`
版本：pytorch/pytorch @ v2.13.0 (commit a1b2c3d4, 2026-08-15)
验证：pip show torch | grep Version
```

---

## 实践：定位一个未见过的 API

**任务**：找到 `torch.nn.functional.scaled_dot_product_attention` 在 CUDA 上的实际 kernel。

按三步法：

1. **固定版本**：运行 `labs/M/version_manifest.py`，记录 PyTorch 版本
2. **API → 注册**：
   ```python
   import torch
   import torch.nn.functional as F
   
   class Tracer(torch.utils._python_dispatch._TorchDispatchMode):
       def __torch_dispatch__(self, func, types, args=(), kwargs=None):
           print(f"{func}")
           return func(*args, **(kwargs or {}))
   
   q = k = v = torch.randn(1, 8, 128, 64, device='cuda')
   with Tracer():
       out = F.scaled_dot_product_attention(q, k, v)
   ```
   输出：`aten::_scaled_dot_product_flash_attention`

3. **注册 → 实现**：
   ```bash
   $ cd pytorch
   $ git checkout v2.13.0
   $ grep -r "scaled_dot_product_flash_attention" aten/src/ATen/native/
   # 找到 aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
   ```

4. **动态验证**：
   ```bash
   $ nsys profile -o sdpa_trace python3 trace_sdpa.py
   # 在 timeline 里看到实际 launch 的 kernel 名字
   ```

**记录格式**：

```markdown
## scaled_dot_product_attention 的 CUDA 路径

**版本**: pytorch/pytorch @ v2.13.0  
**API**: `torch.nn.functional.scaled_dot_product_attention`  
**Dispatcher**: `aten::_scaled_dot_product_flash_attention`  
**实现**: `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:78`  
**实际 kernel**: `fmha_cutlass_f16_aligned` (from cutlass, 见 nsys trace)

**观察层次**:
- L1 (Python): F.scaled_dot_product_attention
- L2 (Dispatch): aten::_scaled_dot_product_flash_attention
- L4 (Runtime): cudaLaunchKernel → fmha_cutlass_f16_aligned
- L5 (GPU): HMMA.16816 指令 (from ncu)
```

---

## 检查表

每次源码走读，问自己：

- [ ] 版本固定了吗？commit hash / wheel 版本 / CUDA 版本都有吗？
- [ ] 文件路径是相对项目根的完整路径吗？行号标了吗？
- [ ] 我在哪一层观察？这一层能证明什么、不能证明什么？
- [ ] 静态源码 vs 动态轨迹：我看的是"可能走的路径"还是"实际走的路径"？
- [ ] 有原始文件入口吗？（不是文字转述，是实际可点击的 file:line）

---

## 后续章节如何使用

- **2.0b**：按这个方法走读 `aten.linear` 的完整分发路径
- **2.6b**：按这个方法对照 FX 图、生成代码、CUDA Graph
- **5.7**：按这个方法走读 vLLM/SGLang 的请求路径
- **M3**：按这个方法复现一篇论文的实现

基础方法在这一章交付，具体案例随相应章节完成。

---

## 自测题

1. 给你一个 `model.forward()` 调用，如何确定它实际用了 FlashAttention 而不是普通 SDPA？
2. 你在 FX 图里看到 `aten.linear`，但 nsys 里只有一个 `addmm` kernel。这两者的对应关系是什么？
3. 同一个 API 在 PyTorch 2.0 和 2.13 里的实现文件路径变了。如何维护引用？

::: details 答案

1. **三层验证**：
   - L2: dispatcher hook 看到 `aten::_scaled_dot_product_flash_attention`（不是 `efficient_attention` 或 math fallback）
   - L4: nsys 看到 kernel 名字里有 `fmha` 或 `flash`
   - L5: ncu 看到实际用了 Tensor Core 指令
   
   **不能只看一层**：dispatcher 可能 fallback，需要动态验证。

2. **分解 vs 融合**：
   - `aten.linear` 在 L2 分解成 `t + addmm`
   - `t` 是 view 操作，不需要 kernel
   - nsys (L4) 看到一个 `addmm` kernel
   - 这**不是融合**，是分解后的一个算子本身就不需要计算

3. **版本管理**：
   ```markdown
   ## 引用格式
   
   **PyTorch 2.0**: `aten/src/ATen/native/Linear.cpp:123`  
   **PyTorch 2.13**: `aten/src/ATen/native/LinearAlgebra.cpp:456`  
   (文件在 v2.5 重构时拆分)
   
   **共同点**: 都实现 `addmm_out_cuda_impl`  
   **验证**: 在对应 tag checkout 后 grep
   ```

:::

---

## 原始现场：本章的采集工件

所有脚本在 `labs/M/`：
- `version_manifest.py`: 环境快照
- `trace_dispatch.py`: dispatcher hook 示例
- `trace_linear.py`: nsys 采集示例

完整输出在 `results/local/M1/`：
- `env_snapshot.json`: PyTorch 2.13.0 + CUDA 13.0 完整环境
- `dispatch_trace.txt`: linear 的分发记录
- `linear_trace.nsys-rep`: nsys 时间线

**采集方式**：
```bash
cd /Users/nyxri/Documents/llm-infra
python3 labs/M/version_manifest.py > results/local/M1/env_snapshot.json
python3 labs/M/trace_dispatch.py > results/local/M1/dispatch_trace.txt
```

---

## 陷阱

- ❌ 凭类名和函数签名推断执行路径
- ❌ 把文档设计目标当运行事实
- ❌ 忽略生成代码和动态注册
- ❌ 行号与版本不一致
- ❌ 在错误的观察层次找证据
