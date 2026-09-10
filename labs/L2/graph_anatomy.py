#!/usr/bin/env python3
"""L2.6b lab · 「算子图」到底长什么样：把每一跳都打印出来。

"torch.compile 会把图捕获下来做融合" —— 这句话里的"图"是什么？
本实验把 torch.compile 的四个阶段的产物**原样落盘**：

  1. Dynamo 捕获的 FX 图（Python 字节码 → 图）
  2. AOTAutograd 的前向/反向联合图（含反向算子）
  3. Inductor 生成的 Triton 源码（融合后的真实 kernel）
  4. 编译产物的目录结构

外加两个对照：
  5. 一个真实 Transformer 层的图（不是 toy）
  6. 图断裂（graph break）：什么会让 Dynamo 放弃

用法：python graph_anatomy.py --out-dir results/graphs/
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch


def dump(out: Path, name: str, text: str) -> None:
    (out / name).write_text(text, encoding="utf-8")
    n = len(text.splitlines())
    print(f"    写出 {name:38s} {n:>5d} 行")


def capture_fx(fn, *args) -> tuple[str, list]:
    """用 Dynamo 的 explain 拿到图与断裂信息。"""
    graphs: list = []

    def backend(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(fn, backend=backend, dynamic=False)
    compiled(*args)
    return ("\n\n".join(g.print_readable(print_output=False) for g in graphs), graphs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="graphs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== torch {torch.__version__}   {torch.cuda.get_device_name(0)}\n")

    # ---------------------------------------------------------------
    # 1. 一个最小的融合例子：RMSNorm 风格的逐元素链
    # ---------------------------------------------------------------
    print("[1] Dynamo 捕获的 FX 图（最小例子）")

    def rmsnorm_like(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        v = x.float()
        s = v.pow(2).mean(-1, keepdim=True)
        return (v * torch.rsqrt(s + 1e-6)).to(x.dtype) * w

    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    fx_text, graphs = capture_fx(rmsnorm_like, x, w)
    dump(out, "10_dynamo_fx_graph.txt", fx_text)
    if graphs:
        nodes = list(graphs[0].graph.nodes)
        print(f"    图里有 {len(nodes)} 个节点，op 类型分布：")
        from collections import Counter
        for k, v in Counter(n.op for n in nodes).items():
            print(f"      {k:18s} {v}")
        dump(out, "11_fx_nodes.txt",
             "\n".join(f"{n.op:16s} {str(n.target)[:60]:62s} args={str(n.args)[:70]}"
                       for n in nodes))

    # ---------------------------------------------------------------
    # 2. Inductor 生成的代码：把环境变量打开，让它落盘
    # ---------------------------------------------------------------
    print("\n[2] Inductor 生成的 Triton 源码")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((out / "inductor_cache").absolute())
    torch._dynamo.reset()
    from torch._inductor import config as ind_cfg
    ind_cfg.debug = False

    compiled = torch.compile(rmsnorm_like, dynamic=False)
    compiled(x, w)
    torch.cuda.synchronize()

    cache = out / "inductor_cache"
    py_files = sorted(cache.rglob("*.py"), key=lambda p: -p.stat().st_size)
    kernels = [p for p in py_files if "triton" in p.read_text(errors="replace")[:4000]]
    if kernels:
        src = kernels[0].read_text(errors="replace")
        dump(out, "20_inductor_output.py", src)
        n_tri = src.count("@triton.jit")
        n_call = src.count(".run(")
        print(f"    生成了 {n_tri} 个 Triton kernel，调用 {n_call} 次")
        print(f"    对照：不编译时这段代码有 pow/mean/add/rsqrt/mul/to/mul ≈ 7 个算子")
    else:
        print(f"    （缓存目录 {cache} 里没找到 Triton 源码；可能被缓存到别处）")

    # ---------------------------------------------------------------
    # 3. 融合效果：算子数与访存量
    # ---------------------------------------------------------------
    print("\n[3] 融合前后的 kernel 数量与耗时")
    import statistics
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    def count_kernels(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        return sum(e.count for e in p.key_averages()
                   if e.device_type == DeviceType.CUDA and (e.device_time_total or 0) > 0)

    def timeit(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    n_eager = count_kernels(lambda: rmsnorm_like(x, w))
    n_comp = count_kernels(lambda: compiled(x, w))
    t_eager = timeit(lambda: rmsnorm_like(x, w))
    t_comp = timeit(lambda: compiled(x, w))
    nbytes = x.numel() * 2 * 2          # 读 x + 写 out，bf16
    print(f"    {'':10s} {'kernel 数':>10s} {'耗时ms':>10s} {'有效带宽 GB/s':>16s}")
    print(f"    {'eager':10s} {n_eager:>10d} {t_eager:>10.4f} "
          f"{nbytes/(t_eager*1e-3)/1e9:>16.1f}")
    print(f"    {'compiled':10s} {n_comp:>10d} {t_comp:>10.4f} "
          f"{nbytes/(t_comp*1e-3)/1e9:>16.1f}")
    print(f"    加速 {t_eager/t_comp:.2f}×   （L1.1 实测显存上限 1674 GB/s）")

    # ---------------------------------------------------------------
    # 4. 反向图：AOTAutograd 的联合图
    # ---------------------------------------------------------------
    print("\n[4] 反向图（AOTAutograd 联合图）")
    xg = torch.randn(1024, 512, device="cuda", requires_grad=True)
    wg = torch.randn(512, device="cuda", requires_grad=True)

    joint: list = []

    def joint_backend(gm, example_inputs):
        joint.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        from functorch.compile import aot_module_simplified

        def fwd_bwd(x_, w_):
            v = x_.float()
            s = v.pow(2).mean(-1, keepdim=True)
            return ((v * torch.rsqrt(s + 1e-6)) * w_).sum()

        def backend(gm, inputs):
            def fw(g, i):
                joint.append(("forward", g))
                return g.forward

            def bw(g, i):
                joint.append(("backward", g))
                return g.forward
            return aot_module_simplified(gm, inputs, fw_compiler=fw, bw_compiler=bw)

        c = torch.compile(fwd_bwd, backend=backend, dynamic=False)
        c(xg, wg).backward()
        parts = []
        for kind, g in joint:
            parts.append(f"########## {kind} 图\n" + g.print_readable(print_output=False))
        dump(out, "30_aot_fwd_bwd_graphs.txt", "\n\n".join(parts))
        for kind, g in joint:
            print(f"    {kind:10s} 图有 {len(list(g.graph.nodes))} 个节点")
    except Exception as exc:                                   # noqa: BLE001
        print(f"    [跳过] {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------
    # 5. 图断裂：什么会让 Dynamo 放弃
    # ---------------------------------------------------------------
    print("\n[5] 图断裂（graph break）")

    def has_break(x_):
        y = x_ * 2
        if y.sum().item() > 0:          # ← .item() 强制同步，Dynamo 必须断图
            y = y + 1
        return y.relu()

    torch._dynamo.reset()
    expl = torch._dynamo.explain(has_break)(torch.randn(64, device="cuda"))
    print(f"    图数量        {expl.graph_count}")
    print(f"    断裂次数      {expl.graph_break_count}")
    print(f"    断裂原因      {[str(r)[:70] for r in expl.break_reasons][:3]}")
    dump(out, "40_graph_break.txt", str(expl))

    print(f"\n所有产物在 {out.absolute()}")


if __name__ == "__main__":
    main()
