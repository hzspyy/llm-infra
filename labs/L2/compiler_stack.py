#!/usr/bin/env python3
"""L2.7 —— 编译器栈：守卫、重编译、融合边界、autotune。

2.6b 已经把 Dynamo / AOTAutograd / Inductor 三段的**产物**打印过了。
这一章问的是另一组问题：编译好的代码什么时候**不能用**了？
Dynamo 每次调用到底检查什么？Inductor 在哪里停下来不再融合？

用法：
    python compiler_stack.py            # 全跑
    python compiler_stack.py B D
"""

import os
import sys
import time

import torch
import torch._dynamo as dynamo
import torch._dynamo.utils as dutils
from torch._inductor.utils import run_and_get_code

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def n_kernels(code_list):
    """一次编译产出了几个 triton kernel。"""
    return sum(c.count("@triton.jit") for c in code_list)


def recompiles():
    return dutils.counters["stats"].get("unique_graphs", 0)


def timeit(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 守卫：编译结果凭什么可以复用")

    def f(x, alpha):
        return (x * alpha).relu().sum()

    dynamo.reset()
    exp = dynamo.explain(f)(torch.randn(8, 8, device=DEV), 2.0)
    print(f"  图数量        {exp.graph_count}")
    print(f"  图断裂次数    {exp.graph_break_count}")
    print(f"  被捕获的算子  {exp.op_count}")
    if exp.break_reasons:
        for r in exp.break_reasons:
            print("  断裂原因:", r)

    sub("Dynamo 实际装了哪些守卫（TORCH_LOGS=guards 的内容）")
    dynamo.reset()
    cf = torch.compile(f)
    cf(torch.randn(8, 8, device=DEV), 2.0)
    # 从编译产物里把守卫取出来
    got = []
    for code, entries in dynamo.eval_frame._debug_get_cache_entry_list.__doc__ and [] or []:
        pass
    try:
        entries = dynamo.eval_frame._debug_get_cache_entry_list(f.__code__)
        for ent in entries:
            gm = getattr(ent, "guard_manager", None) or getattr(ent, "check_fn", None)
            txt = str(gm)
            got = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    except Exception as exc:                                  # noqa: BLE001
        print("  取守卫失败:", exc)
    if got:
        print(f"  共 {len(got)} 行守卫，节选：")
        for ln in got[:24]:
            print("   ", ln[:110])
    print("\n  守卫是「这份编译结果什么时候还算数」的完整条件。")
    print("  每次调用都要把它跑一遍 —— 所以守卫本身也是运行时开销。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 重编译：什么会让编译结果失效")

    def f(x):
        return (x * 2).relu().sum()

    cases = [
        ("基准 (8,8) f32 cuda", lambda: torch.randn(8, 8, device=DEV)),
        ("同形状同 dtype 再来一次", lambda: torch.randn(8, 8, device=DEV)),
        ("换形状 (16,16)", lambda: torch.randn(16, 16, device=DEV)),
        ("再换形状 (32,32)", lambda: torch.randn(32, 32, device=DEV)),
        ("换 dtype bf16", lambda: torch.randn(8, 8, device=DEV, dtype=torch.bfloat16)),
        ("换维数 (8,8,8)", lambda: torch.randn(8, 8, 8, device=DEV)),
        ("换 device cpu", lambda: torch.randn(8, 8)),
        ("非连续 (转置)", lambda: torch.randn(8, 8, device=DEV).t()),
    ]
    dynamo.reset()
    dutils.counters.clear()
    cf = torch.compile(f)
    prev = 0
    print(f"  {'调用':<28} {'累计编译次数':>12} {'本次新增':>8}")
    for name, mk in cases:
        cf(mk())
        cur = recompiles()
        print(f"  {name:<28} {cur:>12} {cur - prev:>8}")
        prev = cur

    sub("cache_size_limit：编译太多次会发生什么")
    print(f"  cache_size_limit = {dynamo.config.cache_size_limit}"
          f"   accumulated = {dynamo.config.accumulated_cache_size_limit}")

    print("  先看一个**不会**炸的：python int 参数。")

    def g(x, k):
        return x * k

    dynamo.reset()
    dutils.counters.clear()
    x = torch.randn(4, device=DEV)
    cg = torch.compile(g)
    seq = []
    for i in range(12):
        cg(x, i)
        seq.append(recompiles())
    print(f"  k = 0..11 时累计编译次数: {seq}")
    print("  只编译了 2 次。第一次把 k 特化成常量 0，第二次发现 k 变了，")
    print("  于是把 k 转成动态标量，之后所有 k 都复用同一份代码。")
    print("  **这和形状的 automatic dynamic 是同一个机制**，不要以为标量一定会特化。")

    print("\n  再看一个真的会炸的：维数（rank）。rank 没法做成动态。")

    def h(x):
        return (x * 2).relu().sum()

    dynamo.reset()
    dutils.counters.clear()
    ch = torch.compile(h)
    lim = dynamo.config.cache_size_limit
    print(f"  {'维数':>5} {'累计编译':>10}  说明")
    for r in range(1, 13):
        ch(torch.randn(*([2] * r), device=DEV))
        note = ""
        if recompiles() == lim and r >= lim:
            note = "<- 卡在 cache_size_limit"
        print(f"  {r:>5} {recompiles():>10}  {note}")

    plateau = recompiles()
    print(f"\n  累计编译停在 {plateau}，等于 cache_size_limit = {lim}。")
    print("  之后每个新维数都不再编译，直接退回 eager 执行 —— 不报错，只是变慢。")

    sub("撑爆缓存本身贵不贵")
    big = torch.randn(1 << 22, device=DEV)

    def probe(t):
        return (t * 2).relu().sum()

    def bust_then_time(bust_ranks, label):
        dynamo.reset(); dutils.counters.clear()
        cp = torch.compile(probe)
        cp(big)                                    # rank 1 大张量先编一份
        for r in bust_ranks:
            cp(torch.randn(*([2] * r), device=DEV))
        for _ in range(5):
            cp(big)
        t = timeit(lambda: cp(big), n=20)
        print(f"  {label:<40} {t:>9.3f} ms   编译数={recompiles()}")
        return t

    t_keep = bust_then_time(range(2, 14), "撑爆，但 big 那条缓存还在（rank 2..13）")
    t_lost = bust_then_time(range(1, 13), "撑爆，且 big 那条被挤掉（rank 1..12）")
    t_eager = timeit(lambda: probe(big), n=20)
    print(f"  {'纯 eager 参照':<40} {t_eager:>9.3f} ms")
    print()
    print("  第一行说明：**撑爆缓存本身不贵**。只要还有一条守卫能对上，")
    print("  它照样用编译好的代码，Dynamo 也不会重新进编译器（frames 增量为 0）。")
    print(f"  第二行那 {t_lost:.0f} ms 不是「退回 eager」造成的 —— eager 只要 "
          f"{t_eager:.3f} ms。")
    print("  真正的原因是：rank 1 被一个 2 元素的张量顶掉之后触发了 automatic dynamic，")
    print("  而**动态形状的归约 kernel** 在大输入上会塌掉。这是 [C] 节的主题。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 动态形状：让一份代码吃所有长度")

    def f(x):
        return (x * 2).relu().sum()

    sub("1. 默认：先特化，被打脸一次之后自动转动态")
    dynamo.reset(); dutils.counters.clear()
    cf = torch.compile(f)
    for n in (8, 16, 32, 64, 128):
        cf(torch.randn(n, device=DEV))
        print(f"  shape=({n},)   累计编译 {recompiles()}")
    print("  第 1 次特化成 8，第 2 次发现形状变了 -> 重编译成动态，之后不再涨。")

    sub("2. 一开始就声明动态")
    dynamo.reset(); dutils.counters.clear()
    cf2 = torch.compile(f, dynamic=True)
    for n in (8, 16, 32, 64, 128):
        cf2(torch.randn(n, device=DEV))
        print(f"  shape=({n},)   累计编译 {recompiles()}")

    sub("3. 动态形状的代价：归约会塌，逐元素不会")
    if DEV != "cuda":
        print("  需要 CUDA"); return
    import re
    x = torch.randn(1 << 22, device="cuda")          # 4M 元素 = 16 MiB

    def bench(fn, dyn):
        dynamo.reset()
        cf = torch.compile(fn, dynamic=dyn) if dyn is not None else torch.compile(fn)
        _, code = run_and_get_code(cf, x)
        hints = re.findall(r"size_hints=\{[^}]*\}", "\n".join(code))
        return timeit(lambda: cf(x), n=20), hints

    cases = [("归约 (t*2).relu().sum()", lambda t: (t * 2).relu().sum()),
             ("逐元素 (t*2).relu()*1.5", lambda t: (t * 2).relu() * 1.5)]
    print(f"  {'表达式':<26} {'eager':>9} {'静态':>9} {'动态':>10} {'动态/静态':>10}")
    for name, fn in cases:
        te = timeit(lambda: fn(x), n=20)
        ts, hs = bench(fn, None)
        td, hd = bench(fn, True)
        print(f"  {name:<26} {te:>9.3f} {ts:>9.3f} {td:>10.3f} {td / ts:>9.1f}×")
        print(f"  {'':26}   静态 size_hints {hs}")
        print(f"  {'':26}   动态 size_hints {hd}")

    print("\n  逐元素几乎不受影响：kernel 本来就是一维网格，元素个数当运行时参数传进去就行。")
    print("  归约塌了：静态版拿到 `{'x': 512, 'r0_': 8192}`，是 512 个 block 的两段式归约；")
    print("  动态版只拿到 `{'x': 1}` —— 形状未知时 Inductor 无法决定怎么切分归约维，")
    print("  于是退化成**一个 block 顺序扫完整个张量**。")
    print("  这不是常量折叠少了一点，是并行度从 512 掉到 1。")
    del x
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 融合边界：Inductor 在哪里停下来")
    if DEV != "cuda":
        print("需要 CUDA"); return

    N = 1 << 24                                   # 16M 元素 = 64 MiB fp32
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")

    cases = [
        ("5 个逐元素算子",
         lambda a, b: ((a * 2 + b).relu() * 1.5 - b).sigmoid()),
        ("逐元素 + 一个归约",
         lambda a, b: ((a * 2 + b).relu() * 1.5).sum()),
        ("归约之后再逐元素",
         lambda a, b: ((a * 2 + b).sum() * 1.5).relu()),
        ("两个独立归约",
         lambda a, b: (a * 2).sum() + (b * 3).sum()),
        ("中间要写出去（两次用到不同形状）",
         lambda a, b: (a * 2).relu().reshape(4096, 4096).t().contiguous().sum()),
    ]
    print(f"  {'表达式':<34} {'triton kernel 数':>16} {'eager ms':>10} {'compiled ms':>12} {'加速':>7}")
    for name, fn in cases:
        dynamo.reset()
        cf = torch.compile(fn)
        _, code = run_and_get_code(cf, x, y)
        k = n_kernels(code)
        te = timeit(lambda: fn(x, y))
        tc = timeit(lambda: cf(x, y))
        print(f"  {name:<34} {k:>16} {te:>10.3f} {tc:>12.3f} {te / tc:>6.2f}×")

    sub("图断裂：.item() 把一张图切成两张")
    def with_item(a, b):
        s = (a * 2 + b).sum()
        if s.item() > 0:                 # 需要把值搬回 host -> 必须断
            return a * 3
        return a * 4

    dynamo.reset()
    exp = dynamo.explain(with_item)(x, y)
    print(f"  图数量 {exp.graph_count}  断裂 {exp.graph_break_count}")
    for r in exp.break_reasons[:3]:
        print("  原因:", str(r)[:150])
    del x, y
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- E
def section_E():
    title("[E] autotune：编译时间换运行时间")
    if DEV != "cuda":
        print("需要 CUDA"); return

    print("  autotune 只在「cuBLAS 的启发式选得不好」的形状上才有东西可赢。")
    print("  所以要同时测一个方方正正的 GEMM 和一个 decode 形状的瘦长 GEMM。")

    def mm(x, y):
        return (x @ y).relu()

    shapes = [
        ("方阵 4096×4096×4096（prefill 形状）", 4096, 4096, 4096),
        ("瘦长 8×4096×4096（decode 形状）", 8, 4096, 4096),
        ("瘦长 1×4096×11008（decode + FFN）", 1, 4096, 11008),
    ]
    for label, M, K, N in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        flops = 2 * M * K * N
        bytes_moved = (M * K + K * N + M * N) * 2
        te = timeit(lambda: mm(a, b), n=30)
        print(f"\n  {label}")
        print(f"  {'mode':<18} {'编译 s':>9} {'运行 ms':>10} {'TFLOP/s':>9} "
              f"{'GB/s':>9} {'相对 eager':>11}")
        print(f"  {'eager (cuBLAS)':<18} {'-':>9} {te:>10.4f} "
              f"{flops / te / 1e9:>9.1f} {bytes_moved / te / 1e6:>9.1f} {1.0:>10.2f}×")
        for mode in [None, "max-autotune"]:
            dynamo.reset()
            t0 = time.perf_counter()
            cf = torch.compile(mm) if mode is None else torch.compile(mm, mode=mode)
            cf(a, b)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            t = timeit(lambda: cf(a, b), n=30)
            print(f"  {mode or 'default':<18} {compile_s:>9.1f} {t:>10.4f} "
                  f"{flops / t / 1e9:>9.1f} {bytes_moved / t / 1e6:>9.1f} "
                  f"{te / t:>10.2f}×")
        del a, b
        torch.cuda.empty_cache()
    print("\n  方阵上 cuBLAS 已经接近实测算力上限（L1.1 测得 232 TFLOP/s），没什么可赢的。")
    print("  瘦长形状是 memory-bound 的，要看 GB/s 而不是 TFLOP/s。")


# ---------------------------------------------------------------- F
def section_F():
    title("[F] 广度：同一个模块，几条不同的编译路线")
    if DEV != "cuda":
        print("需要 CUDA"); return

    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d=1024):
            super().__init__()
            self.n = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.n(x))))

    class Stack(nn.Module):
        def __init__(self, depth=8, d=1024):
            super().__init__()
            self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    m = Stack().cuda().to(torch.bfloat16).eval()
    x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)   # decode 形状
    print("  8 层 Block、batch=1 —— 这是 launch 开销占比最高的区间（见 L1.4）。")
    print("  两种计时都报：紧凑循环（CPU 提交可以和 GPU 重叠）")
    print("  与每次同步（服务里逐 token 返回时看到的那种）。")

    def timed_sync(fn, n=50, warmup=10):
        for _ in range(warmup):
            fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    rows = []
    with torch.no_grad():
        rows.append(("eager", timeit(lambda: m(x)), timed_sync(lambda: m(x))))
        for backend in ["aot_eager", "inductor"]:
            dynamo.reset()
            try:
                cm = torch.compile(m, backend=backend)
                cm(x)
                rows.append((f"compile/{backend}", timeit(lambda: cm(x)),
                             timed_sync(lambda: cm(x))))
            except Exception as exc:                          # noqa: BLE001
                print(f"  {backend} 失败: {str(exc)[:120]}")
        for mode in ["reduce-overhead", "max-autotune"]:
            dynamo.reset()
            try:
                cm = torch.compile(m, mode=mode)
                cm(x); cm(x); cm(x)
                rows.append((f"compile/{mode}", timeit(lambda: cm(x)),
                             timed_sync(lambda: cm(x))))
            except Exception as exc:                          # noqa: BLE001
                print(f"  {mode} 失败: {str(exc)[:120]}")

    b_loop, b_sync = rows[0][1], rows[0][2]
    print(f"\n  {'路线':<26} {'紧凑循环 ms':>12} {'加速':>7} {'每次同步 ms':>13} {'加速':>7}")
    for name, tl, ts in rows:
        print(f"  {name:<26} {tl:>12.4f} {b_loop / tl:>6.2f}× "
              f"{ts:>13.4f} {b_sync / ts:>6.2f}×")

    sub("torch.export：把图固化下来，不带 python")
    try:
        with torch.no_grad():
            ep = torch.export.export(m, (x,))
        g = ep.graph_module.graph
        ops = [n.target for n in g.nodes if n.op == "call_function"]
        print(f"  导出成功：{len(list(g.nodes))} 个节点，{len(ops)} 个 call_function")
        from collections import Counter
        for op, c in Counter(str(o) for o in ops).most_common(8):
            print(f"    {op:<48} {c}")
        print("\n  export 的图不含 python 控制流，可以脱离 python 运行时部署。")
        print("  这是与 torch.compile 最大的区别：compile 仍然活在 python 进程里。")
    except Exception as exc:                                  # noqa: BLE001
        print("  export 失败:", str(exc)[:200])

    print("\n  未装 TensorRT / ONNX Runtime，这两条路线本轮无法对照（见正文「待补」）。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E, "F": section_F}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device {DEV}")
    if DEV == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
