#!/usr/bin/env python3
"""追踪一次 backward 的执行顺序、节点执行次数与梯度汇合。

PyTorch 不暴露引擎内部的 ready queue，但每个 Node 可以通过
grad_fn.register_hook 观测“它被执行了”。把观测结果与按图结构算出的
依赖计数放在一起，就能验证：依赖计数归零才入队、每个节点只执行一次、
同一输出被多次消费时多份梯度先在 InputBuffer 相加。

Usage:
    python labs/L7/trace_backward.py > "$RUN_DIR/backward.txt"
"""
from __future__ import annotations

import sys

import torch


def collect_nodes(root):
    nodes: list = []
    seen: set = set()

    def walk(node):
        if node is None or node in seen:
            return
        seen.add(node)
        nodes.append(node)
        for fn, _ in node.next_functions:
            walk(fn)

    walk(root)
    return nodes


def dependency_counts(nodes) -> dict:
    counts = {node: 0 for node in nodes}
    for node in nodes:
        for fn, _ in node.next_functions:
            if fn is not None:
                counts[fn] = counts.get(fn, 0) + 1
    return counts


def label(node) -> str:
    if type(node).__name__ == "AccumulateGrad":
        v = getattr(node, "variable", None)
        return f"AccumulateGrad({tuple(v.shape) if v is not None else '?'})"
    return type(node).__name__


def build_case():
    """h 被两条支路消费；split 节点有两个输出。"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    h = x @ W.t()
    y = torch.relu(h)
    left, right = torch.split(y, 1, dim=0)
    a = (left * 3.0).sum()
    b = (right * 5.0).sum()
    # h 再被消费一次：同一个中间张量进入两个下游算子
    c = (h * 0.5).sum()
    loss = a + b + c
    return {"x": x, "W": W, "h": h, "y": y, "loss": loss}


def main() -> int:
    case = build_case()
    loss = case["loss"]
    nodes = collect_nodes(loss.grad_fn)
    counts = dependency_counts(nodes)

    print("=" * 72)
    print("依赖计数与执行观测")
    print("=" * 72)
    print(f"可达节点 {len(nodes)} 个，边 {sum(len(n.next_functions) for n in nodes)} 条")
    print("按图结构计算的依赖计数：")
    for i, node in enumerate(nodes):
        print(f"  [{i:2d}] {label(node):<34} 依赖计数={counts.get(node, 0)}")
    print()

    index_of = {node: i for i, node in enumerate(nodes)}
    order: list[int] = []
    exec_count: dict[int, int] = {}

    def make_hook(index):
        def hook(*_args, **_kwargs):
            order.append(index)
            exec_count[index] = exec_count.get(index, 0) + 1

        return hook

    for node in nodes:
        node.register_hook(make_hook(index_of[node]))

    # 记录每个节点第一次执行时，上游已经算完了几个节点。
    loss.backward()

    print("register_hook 观测到的执行顺序（括号内为节点编号）：")
    for i, idx in enumerate(order):
        print(f"  {i:2d}. [{idx:2d}] {label(nodes[idx])}")
    print()
    print("每个节点的依赖计数与实际执行（hook 触发）次数：")
    for i, node in enumerate(nodes):
        print(f"  [{i:2d}] {label(node):<34} 依赖计数={counts.get(node, 0)}  hook 触发={exec_count.get(i, 0)}")
    print()
    shared = [(i, n, counts.get(n, 0)) for i, n in enumerate(nodes) if counts.get(n, 0) > 1]
    print("共享节点（依赖计数 > 1）:")
    for i, node, cnt in shared:
        print(f"  [{i:2d}] {label(node)}: 入边 {cnt} 条，hook 触发 {exec_count.get(i, 0)} 次")
    print()

    print("梯度汇合观测（h 被 relu 与 mul 两条路径消费）:")
    print(f"  中间张量 is_leaf={case['h'].is_leaf}，默认不保留 .grad；需要时显式 retain_grad()")
    case2 = build_case()
    case2["h"].retain_grad()
    case2["loss"].backward()
    print(f"  重新跑一次并 retain_grad 后 h.grad =\n{case2['h'].grad}")
    print()

    print("zero_grad(set_to_none=True/False) 对叶子梯度的差别:")
    p = case["x"]
    print(f"  backward 后 x.grad is None: {p.grad is None}")
    p2 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    (p2 * 2).sum().backward()
    ptr_before = p2.grad.data_ptr()
    p2.grad.zero_()
    print(f"  zero_() 保留同一 storage: 前后 data_ptr 相同 = {ptr_before == p2.grad.data_ptr()}，值={p2.grad.tolist()}")
    p3 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    (p3 * 3).sum().backward()
    ptr_first = p3.grad.data_ptr()
    p3.grad = None
    (p3 * 4).sum().backward()
    print(f"  置 None 后再次 backward 用新张量: data_ptr 改变 = {ptr_first != p3.grad.data_ptr()}，值={p3.grad.tolist()}")

    print()
    print("错误阶段：图被释放后第二次 backward")
    x4 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss4 = (x4 * 2).sum()
    loss4.backward()
    try:
        loss4.backward()
    except RuntimeError as exc:
        print(f"  {str(exc).splitlines()[0]}")
    x5 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss5 = (x5 * 2).sum()
    loss5.backward(retain_graph=True)
    loss5.backward()
    print(f"  retain_graph=True 后第二次 backward 成功，x5.grad={x5.grad.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
