#!/usr/bin/env python3
"""L2.6b lab · 同一程序的六个图阶段：语义表、对拍与保存值追踪。

2.6b 的正文讲了 Dynamo→AOTAutograd→Inductor 这条链，但"图"在不同阶段
表示的东西不同：节点数、别名、原地写、保存值都会变。本脚本用**同一份程序**
把六个阶段的产物逐阶段对齐：

    阶段            产物                                怎么看
    eager          真实执行的 aten 序列（dispatch mode 记录）  op 名 + shape
    fx             torch.fx.symbolic_trace 的模块级图          模块边界还在不在
    dynamo         torch._dynamo.export 的图                   分解成了什么
    functionalized make_fx + functionalize 的图              原地写变成了什么
    joint          aot_function 的前向/反向子图               保存值从哪来
    inductor       torch.compile 生成的代码与 kernel 数       算子落到几个 kernel

每个阶段都记录同一组语义：节点数、shape、别名/原地写、保存张量；
再做 eager vs compile 的契约对拍（输出、输入副作用、梯度、RNG），
以及"一个被分解的算子"和"一个跨图保存值"的追踪。

    python labs/L2/graph_stages.py --out-dir out/2.6b/stages
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
B = 8


class Block(nn.Module):
    """含共享参数、view、原地写、linear、归约的小模块。"""

    def __init__(self, d=D):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d) / d ** 0.5)
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x, residual):
        h = F.linear(x, self.w, self.b)        # linear → 会分解成 t + addmm
        h = F.relu(h)
        h = h.reshape(h.shape[0], 2, -1)       # view（前向可见）
        h = h.reshape(h.shape[0], -1)          # 再 view 回来
        h = h + residual
        h = h * h.mean(dim=-1, keepdim=True)   # 归约
        residual.add_(h.detach())              # 原地写：改的是输入（内容不进梯度路径）
        return h.sum()


def op_hist(gm) -> dict:
    c = collections.Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            c[str(n.target).replace("torch.ops.", "")] += 1
        elif n.op != "output":
            c[n.op] += 1
    return dict(c.most_common())


def shape_of(x) -> str:
    if isinstance(x, torch.Tensor):
        return "×".join(str(s) for s in x.shape)
    return type(x).__name__


def graph_table(gm, label: str) -> dict:
    rows = []
    for n in gm.graph.nodes:
        if n.op == "output":
            continue
        args = []
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                args.append(a.name)
            elif isinstance(a, (int, float, bool, str)) or a is None:
                args.append(repr(a))
            else:
                args.append(type(a).__name__)
        rows.append({"node": n.name, "opus": n.op,
                     "target": str(n.target).replace("torch.ops.", ""),
                     "args": args[:4]})
    mutating = [r["target"] for r in rows if r["target"].endswith("_") or r["target"] in
                ("aten.copy_", "aten.add_", "aten.mul_", "aten.zero_")]
    views = [r["target"] for r in rows if "view" in r["target"] or "permute" in r["target"]
             or r["target"] in ("aten.reshape", "aten.t", "aten.transpose.int", "aten.expand")]
    return {"label": label, "nodes": len(rows), "hist": op_hist(gm),
            "mutating": sorted(set(mutating)), "views": sorted(set(views)), "table": rows}


# --------------------------------------------------------------------------- stages
def stage_eager(block, x, residual):
    ops = []
    from torch.utils._python_dispatch import TorchDispatchMode

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = str(func).split(".")[-1] if hasattr(func, "__name__") else str(func)
            shapes = [shape_of(a) for a in args if isinstance(a, torch.Tensor)]
            if isinstance(out, torch.Tensor):
                shapes.append(f"->{shape_of(out)}")
            ops.append({"op": f"aten::{getattr(func, '__name__', name)}", "shapes": shapes})
            return out

    with Rec():
        block(x, residual)
    return {"label": "eager", "nodes": len(ops), "ops": ops[:60],
            "hist": dict(collections.Counter(o["op"] for o in ops).most_common(15))}


def stage_fx(block, x, residual):
    try:
        gm = torch.fx.symbolic_trace(block)
        return graph_table(gm, "fx")
    except Exception as exc:                                   # noqa: BLE001
        return {"label": "fx", "error": f"{type(exc).__name__}: {exc}"[:200]}


def stage_dynamo(block, x, residual):
    try:
        gm, _guards = torch._dynamo.export(block, x, residual, aten_graph=False)
        t = graph_table(gm, "dynamo")
        try:
            gm_aten, _ = torch._dynamo.export(block, x, residual, aten_graph=True)
            t["aten_hist"] = op_hist(gm_aten)
        except Exception as exc:                               # noqa: BLE001
            t["aten_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return t
    except Exception as exc:                                   # noqa: BLE001
        return {"label": "dynamo", "error": f"{type(exc).__name__}: {exc}"[:200]}


def stage_functionalized(block, x, residual):
    """functionalize + make_fx：原地写被改写成了什么。"""
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._subclasses.functional_tensor import FunctionalTensorMode

    try:
        with FunctionalTensorMode():
            gm = make_fx(block, tracing_mode="fake", _allow_non_fake_inputs=True)(x, residual)
        t = graph_table(gm, "functionalized")
        t["note"] = "FunctionalTensorMode + make_fx（fake tensor 追踪）"
        return t
    except Exception as exc:                                   # noqa: BLE001
        try:
            gm = make_fx(block, tracing_mode="fake", _allow_non_fake_inputs=True)(x, residual)
            t = graph_table(gm, "functionalized")
            t["note"] = f"FunctionalTensorMode 失败，退回 make_fx：{type(exc).__name__}"
            return t
        except Exception as exc2:                              # noqa: BLE001
            return {"label": "functionalized",
                    "error": f"{type(exc2).__name__}: {exc2}"[:200]}


def stage_export(block, x, residual):
    try:
        ep = torch.export.export(block, (x, residual))
        gm = ep.graph_module
        t = graph_table(gm, "export")
        t["graph_signature"] = {
            "inputs": [str(s) for s in getattr(ep.graph_signature, "user_inputs", [])],
            "outputs": [str(s) for s in getattr(ep.graph_signature, "user_outputs", [])],
            "mutated": [str(s) for s in getattr(ep.graph_signature, "input_mutations", [])],
        }
        return t
    except Exception as exc:                                   # noqa: BLE001
        return {"label": "export", "error": f"{type(exc).__name__}: {exc}"[:200]}


def stage_joint(block, x, residual):
    """aot_module_simplified：拿到前向/反向子图与保存值。"""
    out = {"label": "joint"}
    fw_graphs, bw_graphs = [], []
    try:
        from torch._functorch.aot_autograd import aot_module_simplified

        def fw(gm, inputs):
            fw_graphs.append(gm)
            return gm.forward

        def bw(gm, inputs):
            bw_graphs.append(gm)
            return gm.forward

        class TupleOut(nn.Module):
            """aot_module_simplified 要求图输出是单个张量，包一层元组。"""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, a, b):
                return (self.inner(a, b),)

        f = aot_module_simplified(TupleOut(block), (x, residual), fw, bw)
        y = f(x.clone(), residual.clone())
        y[0].sum().backward()
        out["forward"] = graph_table(fw_graphs[0], "joint.forward") if fw_graphs else None
        out["backward"] = graph_table(bw_graphs[0], "joint.backward") if bw_graphs else None
        if bw_graphs:
            bw_nodes = [n for n in bw_graphs[0].graph.nodes if n.op == "placeholder"]
            out["backward_inputs"] = len(bw_nodes)
        tot = sum(p.numel() for p in block.parameters())
        out["param_numel"] = tot
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def saved_tensors_of(fn, *args):
    seen = []

    def pack(t):
        seen.append({"shape": shape_of(t), "dtype": str(t.dtype),
                     "requires_grad": bool(t.requires_grad)})
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = fn(*args)
        y.sum().backward()
    return seen


def contract_compare(state, x, residual):
    """eager vs compile：输出、输入副作用、梯度、RNG。

    每次都从同一份 state_dict 造新模块：export / functionalize 会留在模块上的
    状态不能带进下一条路径，否则测的不是"两条路径的差别"。
    """
    res = {}

    def run(fn, seed=0):
        torch.manual_seed(seed)
        b = Block().double()
        xa, ra = x.clone().double(), residual.clone().double()
        b.load_state_dict({k: v.double() for k, v in state.items()})
        before = ra.clone()
        y = fn(b, xa, ra)
        y.sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in b.named_parameters()}
        return y.detach().clone(), ra.clone(), before, grads

    eager_y, eager_r, before, eager_g = run(lambda b, a, r: b(a, r))
    comp = torch.compile(lambda b, a, r: b(a, r), backend="aot_eager", dynamic=False)
    comp_y, comp_r, _, comp_g = run(lambda b, a, r: comp(b, a, r))

    res["output_equal"] = bool(torch.equal(eager_y, comp_y))
    res["output_max_abs_diff"] = float((eager_y - comp_y).abs().max())
    res["residual_mutated_eager"] = bool(not torch.equal(eager_r, before))
    res["residual_mutated_compiled"] = bool(not torch.equal(comp_r, before))
    res["residual_same_across_paths"] = bool(torch.equal(eager_r, comp_r))
    res["grads_match"] = {k: bool(torch.allclose(eager_g[k], comp_g[k], atol=0, rtol=0))
                          for k in eager_g}
    res["grad_max_abs_diff"] = {k: float((eager_g[k] - comp_g[k]).abs().max())
                                for k in eager_g}

    # RNG：单独一个含随机数的函数，看两条路径消耗的随机数是否一致
    def noisy(t):
        return t * torch.rand_like(t)

    torch.manual_seed(1234)
    a = torch.randn(4, 4)
    torch.manual_seed(7); ea = noisy(a)
    s_eager = torch.get_rng_state().clone()
    torch.manual_seed(7); ca = torch.compile(noisy, backend="aot_eager", dynamic=False)(a)
    s_comp = torch.get_rng_state().clone()
    res["rng_output_equal"] = bool(torch.equal(ea, ca))
    res["rng_state_equal"] = bool(torch.equal(s_eager, s_comp))
    return res


def mutation_ablation(x, residual, state):
    """原地写的四种写法 × eager/aot_eager：定位"梯度不一致"发生在哪一步。

    程序里唯一的变量是"原地写谁、写什么"：
      none          不写
      scratch       写到不参与损失的张量
      input_detach  写到输入，内容是 detach 过的
      input_nodedet 写到输入，内容在梯度路径上
    """
    rows = []

    def make(variant):
        def f(xx, rr, ss, W, B_):
            h = F.linear(xx, W, B_)
            h = F.relu(h)
            h = h + rr
            h = h * h.mean(dim=-1, keepdim=True)
            if variant == "scratch":
                ss.add_(h.detach())
            elif variant == "input_detach":
                rr.add_(h.detach())
            elif variant == "input_nodedet":
                rr.add_(h)
            return h.sum()
        return f

    for variant in ("none", "scratch", "input_detach", "input_nodedet"):
        for backend in (None, "aot_eager"):
            label = f"{variant}/{'eager' if backend is None else backend}"
            try:
                W = nn.Parameter(state["w"].clone().double())
                B_ = nn.Parameter(state["b"].clone().double())
                fn = make(variant)
                if backend:
                    fn = torch.compile(fn, backend=backend, dynamic=False)
                y = fn(x.clone(), residual.clone(), torch.zeros_like(residual), W, B_)
                y.sum().backward()
                rows.append({"label": label, "y": float(y), "gw_norm": float(W.grad.norm()),
                             "gw": W.grad.detach().clone()})
            except Exception as exc:                           # noqa: BLE001
                rows.append({"label": label, "error": f"{type(exc).__name__}: {exc}"[:140]})
    out = []
    base = next((r for r in rows if r["label"] == "none/eager"), None)
    for r in rows:
        item = {k: v for k, v in r.items() if k != "gw"}
        g = r.get("gw")
        if g is not None and base and "gw" in base:
            item["dgw_vs_none_eager"] = float((g - base["gw"]).abs().max())
        out.append(item)
    return out


def opaque_boundary(block, x, residual):
    """完整图 vs 分段图：把"attention"标成 opaque 边界。

    vLLM 的 piecewise 图把 attention 标成 splitting op，编译器就在那里把图切开。
    这里用两种等价的本地手法复现：
      - `.item()`：强制同步，Dynamo 必然断裂；
      - `torch.compiler.disable`：把一段代码声明为不编译（= opaque 边界）。
    """
    res = {}

    @torch.compiler.disable
    def opaque_attention(h):
        return h * (h.sum() * 0.0 + 1.0)

    def full(a, b):
        return block(a, b)

    def break_item(a, b):
        h = F.linear(a, block.w, block.b)
        h = F.relu(h)
        scale = float(h.abs().max().item())     # 强制同步 → 图断裂
        h = h + b
        h = h * h.mean(dim=-1, keepdim=True)
        return h.sum() * 0.0 + scale * 0.0

    def opaque(a, b):
        h = F.linear(a, block.w, block.b)
        h = F.relu(h)
        h = opaque_attention(h)                # opaque 边界
        h = h + b
        h = h * h.mean(dim=-1, keepdim=True)
        return h.sum()

    for name, fn in (("full", full), ("break_item", break_item), ("opaque", opaque)):
        try:
            expl = torch._dynamo.explain(fn)(x, residual)
            res[name] = {"graph_count": int(expl.graph_count),
                         "graph_break_count": int(expl.graph_break_count),
                         "break_reasons": [str(r)[:120] for r in expl.break_reasons][:4]}
        except Exception as exc:                               # noqa: BLE001
            res[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    state = Block().state_dict()
    x = torch.randn(B, D)
    residual = torch.randn(B, D)

    def fresh():
        torch.manual_seed(0)
        b = Block()
        b.load_state_dict({k: v.clone() for k, v in state.items()})
        return b

    payload: dict = {"torch": torch.__version__,
                     "cuda": torch.cuda.is_available(),
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "program": "linear → relu → reshape×2 → add → mean*mul → residual.add_ → sum"}

    print(f"=== torch {torch.__version__} · {payload['device']} ===")

    payload["stages"] = [stage_eager(fresh(), x, residual), stage_fx(fresh(), x, residual),
                         stage_dynamo(fresh(), x, residual), stage_export(fresh(), x, residual),
                         stage_functionalized(fresh(), x, residual)]

    print(f"\n    {'阶段':16s} {'节点数':>6s}  关键特征")
    for s in payload["stages"]:
        if "error" in s:
            print(f"    {s['label']:16s}   [失败] {s['error'][:90]}")
            continue
        feat = []
        if s.get("mutating"):
            feat.append("原地写: " + ",".join(s["mutating"][:3]))
        if s.get("views"):
            feat.append("view: " + ",".join(s["views"][:3]))
        print(f"    {s['label']:16s} {s['nodes']:6d}  " + "; ".join(feat)[:100])

    print("\n=== 前向/反向子图与保存值 ===")
    j = stage_joint(fresh(), x, residual)
    payload["joint"] = j
    if "error" in j:
        print(f"    [失败] {j['error'][:160]}")
    else:
        print(f"    前向子图 {j['forward']['nodes']} 节点，反向子图 {j['backward']['nodes']} 节点，"
              f"反向输入 {j.get('backward_inputs')} 个")
    _blk = fresh()
    saved = saved_tensors_of(lambda a, b: _blk(a, b), x.clone(), residual.clone())
    payload["saved_tensors"] = saved
    print(f"    保存张量 {len(saved)} 个：" +
          ", ".join(f"{s['shape']}({s['dtype']})" for s in saved[:6]))

    print("\n=== eager vs compile 契约对拍 ===")
    cc = contract_compare(state, x, residual)
    payload["contract"] = cc
    print(f"    输出逐位相同 {cc['output_equal']}（最大绝对差 {cc['output_max_abs_diff']:.3e}）")
    print(f"    输入被原地修改：eager {cc['residual_mutated_eager']} / "
          f"compiled {cc['residual_mutated_compiled']}；两条路径结果一致 "
          f"{cc['residual_same_across_paths']}")
    print(f"    梯度逐位相同 {cc['grads_match']}")
    print(f"    RNG 输出相同 {cc['rng_output_equal']}，RNG 状态相同 {cc['rng_state_equal']}")

    print("\n=== 原地写的四种写法（eager vs aot_eager）===")
    ma = mutation_ablation(x.double(), residual.double(), state)
    payload["mutation_ablation"] = ma
    for r in ma:
        if "error" in r:
            print(f"    {r['label']:26s} [失败] {r['error'][:80]}")
        else:
            d = r.get("dgw_vs_none_eager")
            print(f"    {r['label']:26s} y={r['y']:10.4f} |gw|={r['gw_norm']:9.4f}"
                  + (f"  dgw={d:.3e}" if d is not None else ""))

    print("\n=== opaque 边界：完整图 vs 分段图 ===")
    ob = opaque_boundary(fresh(), x, residual)
    payload["opaque_boundary"] = ob
    for k, v in ob.items():
        if "error" in v:
            print(f"    {k:6s} [失败] {v['error'][:90]}")
        else:
            print(f"    {k:6s} 图 {v['graph_count']} 段，断裂 {v['graph_break_count']} 次  "
                  f"{v['break_reasons'][:1]}")

    (out / "graph_stages.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    for s in payload["stages"]:
        if "table" in s:
            (out / f"nodes_{s['label']}.txt").write_text(
                "\n".join(f"{r['node']:12s} {r['opus']:16s} {r['target']:28s} {r['args']}"
                          for r in s["table"]), encoding="utf-8")
    print(f"\nJSON -> {out / 'graph_stages.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
