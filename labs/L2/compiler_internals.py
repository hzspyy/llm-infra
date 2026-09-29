#!/usr/bin/env python3
"""L2.7 lab · 编译器内部：守卫、pass、形状序列与约束发现。

对应 completed.md 的 2.7-A/B/D：
  A  guard       每次调用检查什么（原文）＋标量/形状/rank 变化各自触发什么
  A  passes      用 trace 目录里的 pre/post fusion IR 看一次 mutation 消除、
                 一次 reduction lowering 与 buffer 复用
  A  fx_transform 写一个有合法性检查的 FX 变换，前后 IR 与数值对拍
  B  shapes      形状序列 2→大尺寸 / 固定大尺寸 / 交错尺寸：编译、缓存命中、
                 size hints、grid、耗时；默认 vs max-autotune 的实际选择
  D  controlflow data-dependent 控制流、别名、副作用、opaque op 的注入与报错
  D  aoti        AOTI 制品：约束、有效/越界 shape、失败原文

每个 mode 独立可跑：

    python labs/L2/compiler_internals.py --mode guard      --out-dir <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run_with_logs(script: str, env_extra: dict, timeout: int = 600) -> tuple[int, str]:
    """在子进程里跑一段脚本并抓 TORCH_LOGS 输出。

    torch._logging 的 handler 直接写真实 stderr，`redirect_stderr` 抓不到；
    子进程 + 环境变量是最可靠的方式（也是读者自己复现时用的方式）。
    """
    import subprocess
    import tempfile
    env = dict(os.environ)
    env.update(env_extra)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    proc = subprocess.run([sys.executable, path], env=env, capture_output=True, text=True,
                          timeout=timeout)
    os.unlink(path)
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- A
def mode_guard(out: pathlib.Path) -> dict:
    res: dict = {}

    def f(x, scale):
        return (x * scale).relu().sum()

    x = torch.randn(8, D, device=DEV)
    fn = torch.compile(f, dynamic=False)

    cases = [
        ("首次调用", lambda: fn(x, 2.0)),
        ("同样输入再调一次", lambda: fn(x, 2.0)),
        ("标量参数换值（同类型）", lambda: fn(x, 3.0)),
        ("标量参数从 float 换成 int", lambda: fn(x, 2)),
        ("batch 改变（8→9）", lambda: fn(torch.randn(9, D), 2.0)),
        ("rank 改变（2D→3D）", lambda: fn(torch.randn(2, 4, D), 2.0)),
        ("dtype 改变（float32→float64）", lambda: fn(x.double(), 2.0)),
    ]
    rows = []
    for name, call in cases:
        script = f"""
import torch, contextlib, io
torch.manual_seed(0)
def f(x, scale):
    return (x * scale).relu().sum()
fn = torch.compile(f, dynamic=False)
x = torch.randn(8, {D})
call = {{
  "首次调用": lambda: fn(x, 2.0),
  "同样输入再调一次": lambda: fn(x, 2.0),
  "标量参数换值（同类型）": lambda: fn(x, 3.0),
  "标量参数从 float 换成 int": lambda: fn(x, 2),
  "batch 改变（8→9）": lambda: fn(torch.randn(9, {D}), 2.0),
  "rank 改变（2D→3D）": lambda: fn(torch.randn(2, 4, {D}), 2.0),
  "dtype 改变（float32→float64）": lambda: fn(x.double(), 2.0),
}}["{name}"]
call()
"""
        rc, log = run_with_logs(script, {"TORCH_LOGS": "guards,recompiles"})
        def tail(ln):
            return ln.split("] ", 1)[-1].strip() if "] " in ln else ln.strip()
        guards = [tail(ln) for ln in log.splitlines()
                  if any(k in ln for k in ("___check", ".size()[", "L['", "G['", "T['"))]
        recomp = [tail(ln) for ln in log.splitlines()
                  if "recompile" in ln.lower() or "Recompiling" in ln]
        rows.append({"case": name, "guard_lines": guards[:6], "guards_shown": len(guards),
                     "recompile_lines": recomp[:3], "log_lines": len(log.splitlines())})
    res["guard_cases"] = rows

    # 编译次数（Dynamo 计数器）
    try:
        from torch._dynamo.utils import counters
        res["counters"] = {k: dict(v) if isinstance(v, dict) else v
                           for k, v in list(counters.items())[:6]}
    except Exception as exc:                                   # noqa: BLE001
        res["counters_error"] = str(exc)[:120]

    print("    == 守卫：同一函数的七种输入 ==")
    for r in rows:
        print(f"    {r['case']:26s} 守卫行 {r['guards_shown']:2d}  "
              f"{('重编译: ' + r['recompile_lines'][0][:60]) if r['recompile_lines'] else ''}")
        for g in r["guard_lines"][:2]:
            print(f"        {g[:110]}")
    return res


def mode_passes(out: pathlib.Path) -> dict:
    """打开 inductor trace，看 pre/post fusion IR 里的一次 mutation 消除与 buffer 复用。"""
    import torch._inductor.config as icfg
    icfg.force_disable_caches = True       # 磁盘缓存命中会跳过 Inductor，也就没有 trace
    trace_dir = out / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    icfg.trace.enabled = True
    icfg.trace.debug_dir = str(trace_dir)
    res: dict = {}

    def f(x, residual, w):
        h = F.linear(x, w)
        h = F.relu(h)
        h = h + residual
        residual.add_(h)          # 原地写：会被函数式化改写
        return h.sum()

    x = torch.randn(64, D, device=DEV)
    residual = torch.randn(64, D, device=DEV)
    w = torch.randn(D, D, device=DEV)
    fn = torch.compile(f, dynamic=False)
    fn(x.clone(), residual.clone(), w)

    files = sorted(str(p.relative_to(trace_dir)) for p in trace_dir.glob("**/*") if p.is_file())
    res["trace_files"] = files[:40]
    for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt"):
        hits = list(trace_dir.glob(f"**/{name}"))
        if not hits:
            continue
        lines = hits[0].read_text(errors="replace").splitlines()
        res[name] = {"path": str(hits[0].relative_to(trace_dir)), "lines": len(lines),
                     "mutation": [ln.strip()[:120] for ln in lines
                                  if "Mutation" in ln or "mutat" in ln][:5],
                     "reuse": [ln.strip()[:120] for ln in lines
                               if "Reuse" in ln or "reuse" in ln or "Reinterpret" in ln][:5],
                     "reduction": [ln.strip()[:120] for ln in lines
                                   if "Reduction" in ln or "red" in ln.lower()][:5]}
    print("    == Inductor trace 目录 ==")
    print("    " + ", ".join(files[:14]))
    for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt"):
        if name in res:
            print(f"    {res[name]['path']}: {res[name]['lines']} 行；"
                  f"mutation 命中 {len(res[name]['mutation'])}、reuse 命中 {len(res[name]['reuse'])}、"
                  f"reduction 命中 {len(res[name]['reduction'])}")
    icfg.trace.enabled = False
    return res


def mode_fx_transform(out: pathlib.Path) -> dict:
    """一个有合法性检查的 FX 变换：把 relu(mul(x, s)) 换成 clamp_min(mul(x, s), 0)。

    变换本身很小（语义等价），重点是**合法性检查**：
      0 结构检查 `graph.lint()`；
      1 形状推导 `ShapeProp`；
      2 数值对拍（正数、负数、零、极大值四个探针）；
      3 只改声明要改的模式（计数替换点）。
    """
    import operator
    import torch.fx as fx
    from torch.fx.passes.shape_prop import ShapeProp

    class M(nn.Module):
        def forward(self, x):
            a = x * 2.0
            b = a.relu()
            c = b + 1.0
            return c.relu()

    m = M().eval()
    x = torch.randn(16, D)
    gm = fx.symbolic_trace(m)
    before = gm(x)
    before_nodes = len(list(gm.graph.nodes))

    replaced = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_method" and node.target == "relu":
            src = node.args[0]
            is_mul = (isinstance(src, fx.Node) and src.op == "call_function"
                      and src.target in (torch.mul, operator.mul))
            if not is_mul:
                continue
            with gm.graph.inserting_after(src):
                new = gm.graph.call_function(torch.clamp_min, (src, 0.0))
                new.meta["fused"] = "relu(mul(x,s)) → clamp_min"
            node.replace_all_uses_with(new)
            gm.graph.erase_node(node)
            replaced += 1

    gm.graph.lint()                       # 检查 0：结构
    gm.recompile()
    ShapeProp(gm).propagate(x)            # 检查 1：形状推导
    shapes_ok = all("tensor_meta" in n.meta for n in gm.graph.nodes
                    if n.op not in ("output", "placeholder"))
    probes = [x, -x, torch.zeros_like(x), x * 1e4]
    max_diff = max(float((gm(p) - M()(p)).abs().max()) for p in probes)   # 检查 2：数值
    after = gm(x)

    payload = {
        "nodes_before": before_nodes,
        "nodes_after": len(list(gm.graph.nodes)),
        "replaced": replaced,
        "lint_ok": True,
        "shape_prop_ok": shapes_ok,
        "max_abs_diff": max_diff,
        "output_bitwise_equal": bool(torch.equal(before, after)),
        "before_code": fx.symbolic_trace(m).code,
        "after_code": gm.code,
    }
    print("    == FX 变换：relu(mul(x,s)) → clamp_min(mul(x,s), 0) ==")
    print(f"    节点 {payload['nodes_before']} → {payload['nodes_after']}，替换 {replaced} 处")
    print(f"    lint 通过，形状推导通过 {shapes_ok}；四个探针最大绝对差 {max_diff:.3e}；"
          f"输出逐位相同 {payload['output_bitwise_equal']}")
    (out / "fx_transformed.py").write_text(payload["after_code"], encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- B
def mode_shapes(out: pathlib.Path) -> dict:
    res: dict = {"sequences": {}}

    def f(a, b):
        return (a @ b).relu().sum(dim=-1)

    import torch._inductor.config as icfg
    icfg.force_disable_caches = True       # 否则测到的是磁盘缓存命中，不是编译时间

    def run_sequence(name, shapes, mode="default"):
        torch._dynamo.reset()
        fn = torch.compile(f, mode=mode, dynamic=None)
        rows = []
        for i, s in enumerate(shapes):
            a = torch.randn(s, D, device=DEV)
            b = torch.randn(D, D, device=DEV)
            t0 = time.perf_counter()
            fn(a, b)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.perf_counter() - t0
            rows.append({"call": i, "shape": s, "compile_or_cache_s": round(dt, 4)})
        return rows

    seqs = {
        "2→大尺寸": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "固定大尺寸": [1024] * 8,
        "交错尺寸": [8, 128, 16, 256, 32, 512, 64, 1024, 8, 128],
    }
    for name, shapes in seqs.items():
        modes = ["default"] if name != "固定大尺寸" else ["default", "max-autotune-no-cudagraphs"]
        for mode in modes:
            rows = run_sequence(name, shapes, mode)
            res["sequences"][f"{name}|{mode}"] = rows
            first = rows[0]["compile_or_cache_s"]
            rest = sorted(r["compile_or_cache_s"] for r in rows[1:]) or [0.0]
            med = rest[len(rest) // 2]
            print(f"    {name:12s} {mode:30s} 首次 {first*1000:8.1f} ms  后续中位 {med*1000:7.2f} ms")

    # autotune 的实际选择：抓 TORCH_LOGS=autotuning 的输出
    try:
        script = f"""
import torch
torch._inductor.config.force_disable_caches = True
g = torch.compile(lambda a, b: a @ b, mode="max-autotune-no-cudagraphs")
g(torch.randn(512, {D}, device="cuda"), torch.randn({D}, {D}, device="cuda"))
torch.cuda.synchronize()
"""
        _, log = run_with_logs(script, {"TORCH_LOGS": "autotuning"}, timeout=900)
        picks = [ln.strip() for ln in log.splitlines()
                 if "BLOCK" in ln or "num_warps" in ln or "triton" in ln.lower()
                 or "ACC_TYPE" in ln]
        res["autotune_log"] = picks[:20]
        print(f"    == max-autotune 的候选记录 {len(picks)} 行（节选）==")
        for p in picks[:5]:
            print(f"        {p[:120]}")
    except Exception as exc:                                   # noqa: BLE001
        res["autotune_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return res


# --------------------------------------------------------------------------- D
def mode_controlflow(out: pathlib.Path) -> dict:
    res: dict = {}

    def data_dependent(x):
        if x.sum().item() > 0:          # 数据相关控制流
            return x * 2
        return x - 2

    def alias_mutate(x):
        x.add_(1.0)                      # 别名/副作用
        return x.sum()

    counter = {"n": 0}

    def side_effect(x):
        counter["n"] += 1                # Python 副作用
        return x * 2

    @torch.library.custom_op("l27::opaque", mutates_args=())
    def opaque(x: torch.Tensor) -> torch.Tensor:
        return x * 3

    @opaque.register_fake
    def _(x):
        return torch.empty_like(x)

    def with_opaque(x):
        return opaque(x) + 1.0

    cases = [("data_dependent", data_dependent, torch.randn(16, D, device=DEV)),
             ("alias_mutate", alias_mutate, torch.randn(16, D, device=DEV)),
             ("side_effect", side_effect, torch.randn(16, D, device=DEV)),
             ("opaque_op", with_opaque, torch.randn(16, D, device=DEV))]
    for name, fn, x in cases:
        try:
            cp = torch.compile(fn, dynamic=False)
            y = cp(x.clone())
            expl = torch._dynamo.explain(fn)(x.clone())
            res[name] = {"ok": True, "graph_count": int(expl.graph_count),
                         "graph_break_count": int(expl.graph_break_count),
                         "break_reasons": [str(r)[:140] for r in expl.break_reasons][:3]}
            print(f"    {name:16s} 图 {res[name]['graph_count']} 段，"
                  f"断裂 {res[name]['graph_break_count']} 次")
        except Exception as exc:                               # noqa: BLE001
            res[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"    {name:16s} [失败] {res[name]['error'][:110]}")
        # 副作用计数：编译路径与 eager 各跑一次，看 Python 状态被改了几次
    counter["n"] = 0
    for _ in range(2):
        side_effect(torch.randn(16, D, device=DEV))
    eager_calls = counter["n"]
    counter["n"] = 0
    cp2 = torch.compile(side_effect, dynamic=False)
    for _ in range(2):
        cp2(torch.randn(16, D, device=DEV))
    compiled_calls = counter["n"]
    res["side_effect"] = {"python_counter_eager_2_calls": eager_calls,
                          "python_counter_compiled_2_calls": compiled_calls}
    print(f"    Python 副作用计数：eager 两次={eager_calls}，编译路径两次={compiled_calls}"
          f"（后者只在跟踪期执行）")
    return res


def mode_aoti(out: pathlib.Path) -> dict:
    """AOTI 制品：编译 / 保存 / 加载 / 稳态，以及 static 与 dynamic 制品的形状约束。

    需要在 CUDA_HOME 的头文件与库可见（本项目用 CPATH/LIBRARY_PATH 指向 wheel 里的 cu13）；
    否则 C++ 后端报 `cuda.h: No such file or directory`。
    """
    res: dict = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(D, D) / D ** 0.5)

        def forward(self, x):
            return F.relu(x @ self.w).sum(dim=-1)

    m = Blk().eval().to(dev)
    x = torch.randn(16, D, device=dev)
    res["apis"] = {n: hasattr(torch._inductor, n)
                   for n in ("aoti_compile_and_package", "aoti_load_package")}
    res["apis"].update({f"export.{n}": hasattr(torch._export, n)
                        for n in ("aot_compile", "aot_load")})
    print(f"    可用 API：{res['apis']}")

    def build(dynamic: bool, name: str):
        pkg = out / f"aoti_{name}.pt2"
        t0 = time.perf_counter()
        if dynamic:
            from torch.export import Dim
            ep = torch.export.export(m, (x,), dynamic_shapes={"x": {0: Dim("batch")}})
        else:
            ep = torch.export.export(m, (x,))
        package = torch._inductor.aoti_compile_and_package(ep, package_path=str(pkg))
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn = torch._inductor.aoti_load_package(package)
        load_s = time.perf_counter() - t0
        return {"package": str(package), "compile_s": compile_s, "load_s": load_s,
                "size_mb": pkg.stat().st_size / 2**20 if pkg.exists() else None, "fn": fn}

    try:
        for name, dyn in (("static", False), ("dynamic", True)):
            try:
                art = build(dyn, name)
            except Exception as exc:                           # noqa: BLE001
                res[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"    {name} 构建失败：{res[name]['error'][:120]}")
                continue
            fn = art.pop("fn")
            rows = []
            for shape in ((16, D), (17, D), (1024, D), (2, 8, D)):
                xx = torch.randn(*shape, device=dev)
                ref = m(xx)
                try:
                    y = fn(xx)
                    same = tuple(y.shape) == tuple(ref.shape)
                    rows.append({"shape": str(shape), "out": str(tuple(y.shape)),
                                 "ref": str(tuple(ref.shape)), "same_shape": same,
                                 "max_abs_diff": float((y - ref).abs().max()) if same else None})
                except Exception as exc:                       # noqa: BLE001
                    rows.append({"shape": str(shape), "error": f"{type(exc).__name__}: {exc}"[:120]})
            res[name] = {**art, "shape_probe": rows}
            print(f"    {name}: 编译 {art['compile_s']:.2f} s，加载 {art['load_s']:.4f} s，"
                  f"制品 {art['size_mb']:.2f} MB")
            for r in rows:
                if "error" in r:
                    print(f"        {r['shape']:12s} 报错 {r['error'][:70]}")
                else:
                    print(f"        {r['shape']:12s} 输出 {r['out']:10s} 参照 {r['ref']:10s} "
                          f"形状一致 {r['same_shape']}"
                          + (f" max|d|={r['max_abs_diff']:.2e}" if r['same_shape'] else ""))
    except Exception as exc:                                   # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
        print(f"    [AOTI 失败] {res['error'][:200]}")
    return res


MODES = {"guard": mode_guard, "passes": mode_passes, "fx_transform": mode_fx_transform,
         "shapes": mode_shapes, "controlflow": mode_controlflow, "aoti": mode_aoti}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload: dict = {"torch": torch.__version__,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "modes": {}}
    todo = list(MODES) if args.mode == "all" else [args.mode]
    for name in todo:
        print(f"\n=== {name} ===")
        try:
            payload["modes"][name] = MODES[name](out)
        except Exception as exc:                               # noqa: BLE001
            payload["modes"][name] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"    [mode 失败] {payload['modes'][name]['error'][:160]}")

    (out / "compiler_internals.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out / 'compiler_internals.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
