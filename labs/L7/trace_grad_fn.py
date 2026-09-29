#!/usr/bin/env python3
"""打印 PyTorch 前向构建的反向图：Node、Edge、共享边与保存值。

前向只用固定输入，保证输出可重复；图结构由 loss.grad_fn 遍历得到，
保存值通过 saved_tensors_hooks 的 pack/unpack 事件观测。

Usage:
    python labs/L7/trace_grad_fn.py > "$RUN_DIR/grad-fn.txt"
"""
from __future__ import annotations

import sys

import torch

torch.manual_seed(0)


def build_graph():
    """x 只作为叶子输入；h 被两条支路共享；split 产生一个两输出节点。"""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, requires_grad=True)
    W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    h = x @ W.t()                            # MmBackward0
    y = torch.relu(h)                        # ReluBackward0
    left, right = torch.split(y, 1, dim=0)   # SplitWithSizesBackward0：一个节点两个输出
    branch_a = (left * 3.0).sum()            # MulBackward0 → SumBackward0
    branch_b = (right * 5.0).sum()
    loss = branch_a + branch_b               # AddBackward0
    return {"x": x, "W": W, "h": h, "y": y, "left": left, "right": right,
            "branch_a": branch_a, "branch_b": branch_b, "loss": loss}


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


def incoming_counts(nodes) -> dict:
    """统计每个节点从可达子图收到的边数，即引擎的依赖计数。"""
    counts = {node: 0 for node in nodes}
    for node in nodes:
        for fn, _ in node.next_functions:
            if fn is not None:
                counts[fn] = counts.get(fn, 0) + 1
    return counts


def node_label(node) -> str:
    if node is None:
        return "None"
    if type(node).__name__ == "AccumulateGrad":
        v = getattr(node, "variable", None)
        return f"AccumulateGrad({tuple(v.shape) if v is not None else '?'})"
    return type(node).__name__


def describe(nodes, counts) -> None:
    print("节点总览（依赖计数 = 该节点还会收到几份上游梯度）")
    for i, node in enumerate(nodes):
        n_in = counts.get(node, 0)
        marker = "  ← 共享边" if n_in > 1 else ""
        print(f"  [{i:2d}] {node_label(node):<34} 依赖计数={n_in}{marker}  "
              f"next_functions={len(node.next_functions)}")
    print()

    print("完整 Node/Edge 遍历（缩进表示边方向：loss → 叶子）")
    visited: set = set()

    def walk(node, depth):
        indent = "  " * depth
        if node in visited:
            print(f"{indent}{node_label(node)}  <已展开，共享节点，入边 {counts.get(node, 0)} 条>")
            return
        visited.add(node)
        print(f"{indent}{node_label(node)}  依赖计数={counts.get(node, 0)}")
        for idx, (fn, input_nr) in enumerate(node.next_functions):
            if fn is None:
                print(f"{indent}  edge[{idx}] input_nr={input_nr} -> (无梯度)")
                continue
            print(f"{indent}  edge[{idx}] input_nr={input_nr} -> {node_label(fn)}")
            walk(fn, depth + 2)

    walk(nodes[0], 0)
    print()


def main() -> int:
    packed: list[str] = []
    unpacked: list[str] = []

    def pack_hook(t):
        packed.append(f"{tuple(t.shape)} {t.dtype}")
        return t

    def unpack_hook(t):
        unpacked.append(f"{tuple(t.shape)} {t.dtype}")
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        tensors = build_graph()

    loss = tensors["loss"]
    nodes = collect_nodes(loss.grad_fn)
    counts = incoming_counts(nodes)

    print("=" * 72)
    print(f"反向图：{len(nodes)} 个节点，{sum(len(n.next_functions) for n in nodes)} 条边")
    print("=" * 72)
    print()
    describe(nodes, counts)

    print("前向期间保存值 pack 事件:")
    for i, item in enumerate(packed):
        print(f"  pack[{i}] {item}")
    print(f"  共 {len(packed)} 个")
    print()

    print("内建 Node 的保存值实际字段（反向执行前）:")
    for node in nodes:
        fields = [name for name in ("_saved_self", "_saved_other") if hasattr(node, name)]
        if not fields:
            continue
        parts = []
        for name in fields:
            value = getattr(node, name)
            if value is None:
                shapes = []
            elif isinstance(value, torch.Tensor):
                shapes = [tuple(value.shape)]
            else:
                shapes = [tuple(t.shape) for t in value]
            parts.append(f"{name}={shapes}")
        print(f"  {node_label(node)}: " + "  ".join(parts))
    print()

    loss.backward()
    print("反向执行期间保存值 unpack 事件（顺序即节点读取保存值的顺序）:")
    for i, item in enumerate(unpacked):
        print(f"  unpack[{i}] {item}")
    print(f"  共 {len(unpacked)} 个")
    print()

    print("叶子梯度:")
    for name in ("x", "W"):
        print(f"  {name}.grad shape={tuple(tensors[name].grad.shape)}")
    print()
    print("保存值释放语义（自定义 Function 的 saved_tensors 在 backward 后报错）见 autograd_modes.py 的 [E] 段。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
