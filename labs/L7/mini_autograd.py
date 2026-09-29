#!/usr/bin/env python3
"""最小 autograd 引擎：Node / Edge / GraphTask / InputBuffer。

对照 PyTorch 的调度模型实现，不追求算子覆盖：
- 前向为每个算子建立一个 Node，节点的 next_edges 指向产生其输入的节点；
- 反向按“依赖计数归零才入队”的方式执行，每个节点只执行一次；
- 同一输出被多次消费时，多份梯度在 InputBuffer 里就地相加后再交给节点；
- 叶子梯度由 AccumulateGrad 写入 tensor.grad，跨调用累积。

Usage:
    python labs/L7/mini_autograd.py > "$RUN_DIR/mini-autograd.txt"
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 图的基本对象
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    """一条反向图边：把梯度交给 function 的第 input_nr 个输入。"""

    function: Optional["Node"]
    input_nr: int = 0

    def __repr__(self) -> str:
        if self.function is None:
            return "Edge(None)"
        return f"Edge({self.function.name}, input_nr={self.input_nr})"


class Node:
    """反向节点：apply() 返回与 next_edges 一一对应的输入梯度。"""

    def __init__(self, name: str, next_edges: list[Edge]):
        self.name = name
        self.next_edges = next_edges
        self.num_inputs = len(next_edges)  # 与 next_edges 一一对应
        self.num_outputs = 1               # 节点有几个前向输出，就有几份上游梯度

    def apply(self, grad_outputs: list[Optional[np.ndarray]]) -> list[Optional[np.ndarray]]:
        raise NotImplementedError


class AccumulateGrad(Node):
    """叶子节点终点：把收到的梯度写进 variable.grad。"""

    def __init__(self, variable: "Tensor"):
        super().__init__("AccumulateGrad", [])
        self.variable = variable

    def apply(self, grad_outputs):
        grad = grad_outputs[0]
        if self.variable.grad is None:
            self.variable.grad = grad.copy()
        else:
            self.variable.grad = self.variable.grad + grad
        return []


class Tensor:
    """最小张量：只有 data/grad/grad_fn/版本计数。"""

    def __init__(self, data, requires_grad: bool = False, name: str = "t"):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.grad_fn: Optional[Node] = None
        self.output_nr = 0
        self.version = 0
        self.name = name
        self._accumulate_grad = AccumulateGrad(self) if requires_grad else None

    # 叶子张量的 grad_fn 在 PyTorch 里是 None，但引擎持有隐式 AccumulateGrad。
    def edge(self) -> Edge:
        if self.grad_fn is not None:
            return Edge(self.grad_fn, self.output_nr)
        if self._accumulate_grad is not None:
            return Edge(self._accumulate_grad, 0)
        return Edge(None, 0)

    def zero_grad(self, set_to_none: bool = False) -> None:
        if set_to_none:
            self.grad = None
        elif self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad_output: Optional[np.ndarray] = None) -> "GraphTask":
        if self.grad_fn is None and self._accumulate_grad is None:
            raise RuntimeError(
                "element 0 of tensors does not require grad and does not have a grad_fn"
            )
        if grad_output is None:
            grad_output = np.ones_like(self.data)
        task = GraphTask(self, np.asarray(grad_output, dtype=np.float64))
        task.execute()
        return task


# ---------------------------------------------------------------------------
# 算子节点
# ---------------------------------------------------------------------------
class AddBackward(Node):
    def apply(self, grad_outputs):
        g = grad_outputs[0]
        return [g, g]


class MulBackward(Node):
    def apply(self, grad_outputs):
        g = grad_outputs[0]
        a, b = self.saved
        return [g * b, g * a]


class MatmulBackward(Node):
    def apply(self, grad_outputs):
        g = grad_outputs[0]
        a, b = self.saved
        return [g @ b.T, a.T @ g]


class SumBackward(Node):
    def apply(self, grad_outputs):
        g = float(grad_outputs[0])
        shape = self.saved[0]
        return [np.full(shape, g)]


class ReluBackward(Node):
    def apply(self, grad_outputs):
        g = grad_outputs[0]
        (x,) = self.saved
        return [g * (x > 0)]


class SplitTwoBackward(Node):
    """一个节点、两个输出：两路梯度必须都到齐才执行一次。"""

    def apply(self, grad_outputs):
        g0, g1 = grad_outputs[0], grad_outputs[1]
        if g0 is None:
            g0 = np.zeros(self.saved[0], dtype=np.float64)
        if g1 is None:
            g1 = np.zeros(self.saved[1], dtype=np.float64)
        return [np.concatenate([g0, g1], axis=0)]


# ---------------------------------------------------------------------------
# GraphTask：依赖计数、ready queue、InputBuffer
# ---------------------------------------------------------------------------
class GraphTask:
    def __init__(self, root: Tensor, root_grad: np.ndarray):
        self.root_tensor = root
        self.root = root.grad_fn
        self.root_grad = root_grad
        self.dependencies: dict[Node, int] = {}
        self.input_buffers: dict[Node, dict[int, np.ndarray]] = {}
        self.exec_counts: dict[Node, int] = {}
        self.exec_order: list[str] = []
        self.edge_log: list[str] = []

    # 对应 engine.cpp 的 compute_dependencies：统计每个节点还会收到几份梯度。
    def compute_dependencies(self) -> None:
        seen: set[Node] = set()

        def walk(node: Optional[Node]) -> None:
            if node is None or node in seen:
                return
            seen.add(node)
            self.dependencies.setdefault(node, 0)
            for edge in node.next_edges:
                if edge.function is None:
                    continue
                self.dependencies[edge.function] = self.dependencies.get(edge.function, 0) + 1
                walk(edge.function)

        walk(self.root)
        # root 由调用方直接提供初始梯度，不再等待其它边。
        self.dependencies[self.root] = 0

    # 对应 input_buffer.cpp：同一个 output_nr 收到第二份梯度时相加。
    def add_grad(self, node: Node, output_nr: int, grad: np.ndarray) -> None:
        buf = self.input_buffers.setdefault(node, {})
        if output_nr in buf:
            buf[output_nr] = buf[output_nr] + grad
            self.edge_log.append(f"    InputBuffer[{node.name}][{output_nr}] 就地相加")
        else:
            buf[output_nr] = grad.copy()

    def execute(self) -> None:
        self.compute_dependencies()
        self.add_grad(self.root, 0, self.root_grad)
        ready: deque[Node] = deque([self.root])
        while ready:
            node = ready.popleft()
            buf = self.input_buffers.get(node, {})
            grad_outputs = [buf.get(i) for i in range(node.num_outputs)]
            self.exec_order.append(node.name)
            self.exec_counts[node] = self.exec_counts.get(node, 0) + 1
            if node.num_inputs == 0:
                node.apply(grad_outputs)
                continue
            input_grads = node.apply(grad_outputs)
            for edge, grad in zip(node.next_edges, input_grads):
                if edge.function is None or grad is None:
                    continue
                self.add_grad(edge.function, edge.input_nr, grad)
                self.dependencies[edge.function] -= 1
                if self.dependencies[edge.function] == 0:
                    ready.append(edge.function)

    # 遍历并打印 Node/Edge，标注共享边。
    def describe(self) -> str:
        seen: set[Node] = set()
        lines: list[str] = []

        def walk(node: Node, depth: int) -> None:
            indent = "  " * depth
            if node in seen:
                refs = sum(
                    1
                    for n in self.dependencies
                    for e in n.next_edges
                    if e.function is node
                )
                lines.append(f"{indent}{node.name}  <已访问，共享节点，入边 {refs} 条>")
                return
            seen.add(node)
            refs = sum(
                1 for n in self.dependencies for e in n.next_edges if e.function is node
            )
            tag = f"，共享边 {refs} 条" if refs > 1 else ""
            lines.append(
                f"{indent}{node.name}(依赖计数={refs}{tag}) 执行 {self.exec_counts.get(node, 0)} 次"
            )
            for i, edge in enumerate(node.next_edges):
                if edge.function is None:
                    lines.append(f"{indent}  └─ edge[{i}] → (无梯度输入)")
                else:
                    lines.append(
                        f"{indent}  └─ edge[{i}] input_nr={edge.input_nr} → {edge.function.name}"
                    )
                    walk(edge.function, depth + 2)

        walk(self.root, 0)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 前向算子
# ---------------------------------------------------------------------------
def _wrap(data, requires_grad: bool, name: str, node: Optional[Node]) -> Tensor:
    out = Tensor(data, requires_grad=requires_grad, name=name)
    out.grad_fn = node
    return out


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    req = a.requires_grad or b.requires_grad
    return _wrap(a.data + b.data, req, name, AddBackward(name, [a.edge(), b.edge()]) if req else None)


def mul(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    req = a.requires_grad or b.requires_grad
    node = MulBackward(name, [a.edge(), b.edge()]) if req else None
    if node is not None:
        node.saved = (a.data, b.data)
    return _wrap(a.data * b.data, req, name, node)


def matmul(a: Tensor, b: Tensor, name: str = "matmul") -> Tensor:
    req = a.requires_grad or b.requires_grad
    node = MatmulBackward(name, [a.edge(), b.edge()]) if req else None
    if node is not None:
        node.saved = (a.data, b.data)
    return _wrap(a.data @ b.data, req, name, node)


def relu(x: Tensor, name: str = "relu") -> Tensor:
    node = ReluBackward(name, [x.edge()]) if x.requires_grad else None
    if node is not None:
        node.saved = (x.data,)
    return _wrap(np.maximum(x.data, 0.0), x.requires_grad, name, node)


def sum_all(x: Tensor, name: str = "sum") -> Tensor:
    node = SumBackward(name, [x.edge()]) if x.requires_grad else None
    if node is not None:
        node.saved = (x.data.shape,)
    return _wrap(np.array(x.data.sum()), x.requires_grad, name, node)


def split_two(x: Tensor, n: int, name: str = "split") -> tuple[Tensor, Tensor]:
    """一个节点两个输出：两路被使用时依赖计数为 2。"""
    node = SplitTwoBackward(name, [x.edge()]) if x.requires_grad else None
    if node is not None:
        node.num_outputs = 2
        node.saved = (x.data[:n].shape, x.data[n:].shape)
    a = _wrap(x.data[:n], x.requires_grad, name + ".0", node)
    b = _wrap(x.data[n:], x.requires_grad, name + ".1", node)
    if node is not None:
        b.output_nr = 1
    return a, b


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
def scenario_branch() -> dict:
    """分支汇合：x 经两条支路回到同一加法节点。"""
    x = Tensor([2.0, 3.0], requires_grad=True, name="x")
    a = mul(x, Tensor([3.0, 3.0]), name="mul_a")
    b = mul(x, Tensor([5.0, 5.0]), name="mul_b")
    loss = sum_all(add(a, b), name="sum_loss")
    task = loss.backward()
    return {"x_grad": x.grad.tolist(), "graph": task.describe(), "order": task.exec_order}


def scenario_shared_output() -> dict:
    """同一输出被消费两次：InputBuffer 在 output_nr=0 上就地相加。"""
    x = Tensor([2.0], requires_grad=True, name="x")
    y = mul(x, Tensor([3.0]), name="y")
    loss = add(sum_all(y, "sum1"), sum_all(y, "sum2"), name="loss")
    task = loss.backward()
    return {
        "x_grad": x.grad.tolist(),
        "order": task.exec_order,
        "edges": task.edge_log,
        "graph": task.describe(),
    }


def scenario_multi_output_clean() -> dict:
    """一个节点两个输出，两路都参与损失，节点仍只执行一次。"""
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True, name="x")
    head, tail = split_two(x, 2)
    s_head = sum_all(head, "sum_head")
    s_tail = sum_all(tail, "sum_tail")
    loss = add(s_head, s_tail, "loss")
    task = loss.backward()
    return {
        "x_grad": x.grad.tolist(),
        "order": task.exec_order,
        "counts": {n.name: c for n, c in task.exec_counts.items()},
        "graph": task.describe(),
    }


def scenario_unused_leaf() -> dict:
    """requires_grad=True 但未参与前向的叶子不进入图，梯度保持 None。"""
    used = Tensor([1.0], requires_grad=True, name="used")
    unused = Tensor([9.0], requires_grad=True, name="unused")
    loss = sum_all(mul(used, Tensor([4.0]), "mul"), "loss")
    task = loss.backward()
    return {
        "used_grad": used.grad.tolist(),
        "unused_grad": unused.grad,
        "graph": task.describe(),
    }


def scenario_zero_grad() -> dict:
    """叶子梯度跨调用累积；set_to_none 与补零对下一次 backward 的影响不同。"""
    x = Tensor([2.0], requires_grad=True, name="x")
    rows = []
    for call in (1, 2):
        loss = sum_all(mul(x, Tensor([3.0]), "mul"), "loss")
        loss.backward()
        rows.append({"call": call, "mode": "累积", "grad": x.grad.tolist()})
    x.zero_grad(set_to_none=False)
    rows.append({"call": 3, "mode": "zero_grad(set_to_none=False)", "grad": x.grad.tolist()})
    sum_all(mul(x, Tensor([3.0]), "mul"), "loss").backward()
    rows.append({"call": 3, "mode": "补零后 backward", "grad": x.grad.tolist()})
    x.zero_grad(set_to_none=True)
    rows.append({"call": 4, "mode": "zero_grad(set_to_none=True)", "grad": x.grad})
    sum_all(mul(x, Tensor([3.0]), "mul"), "loss").backward()
    rows.append({"call": 4, "mode": "置 None 后 backward", "grad": x.grad.tolist()})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 与 PyTorch 对拍
# ---------------------------------------------------------------------------
def verify_with_torch() -> list[dict]:
    import torch

    rows: list[dict] = []

    def check(name: str, mini_grad, torch_grad) -> None:
        a = np.asarray(mini_grad, dtype=np.float64)
        b = torch_grad.detach().cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        rows.append({"case": name, "mini": a.tolist(), "torch": b.tolist()})

    # 分支 + 共享参数
    x = Tensor([2.0, 3.0], requires_grad=True, name="x")
    w = Tensor([1.5, 1.5], requires_grad=True, name="w")
    a = mul(x, w, "a")
    b = mul(x, Tensor([2.0, 2.0]), "b")
    loss = sum_all(add(a, b, "s"), "loss")
    loss.backward()

    tx = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    tw = torch.tensor([1.5, 1.5], dtype=torch.float64, requires_grad=True)
    tloss = ((tx * tw) + tx * 2.0).sum()
    tloss.backward()
    check("分支/共享参数 x", x.grad, tx.grad)
    check("分支/共享参数 w", w.grad, tw.grad)

    # 同一输出被消费两次
    y = Tensor([2.0], requires_grad=True, name="y")
    yy = mul(y, Tensor([3.0]), "yy")
    add(sum_all(yy, "s1"), sum_all(yy, "s2"), "loss2").backward()
    ty = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    tyy = ty * 3.0
    (tyy.sum() + tyy.sum()).backward()
    check("同一输出两路消费 y", y.grad, ty.grad)

    # 多输出节点
    z = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True, name="z")
    h, t = split_two(z, 2)
    add(sum_all(h, "sh"), sum_all(t, "st"), "loss3").backward()
    tz = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    th, tt = tz[:2], tz[2:]
    (th.sum() + tt.sum()).backward()
    check("多输出节点 z", z.grad, tz.grad)

    # 矩阵乘 + ReLU
    A = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3) - 2.0, requires_grad=True, name="A")
    B = Tensor(np.arange(12, dtype=np.float64).reshape(3, 4) / 4.0, requires_grad=True, name="B")
    loss4 = sum_all(relu(matmul(A, B, "mm"), "relu"), "loss4")
    loss4.backward()
    tA = torch.tensor(A.data, dtype=torch.float64, requires_grad=True)
    tB = torch.tensor(B.data, dtype=torch.float64, requires_grad=True)
    torch.relu(tA @ tB).sum().backward()
    check("matmul+relu A", A.grad, tA.grad)
    check("matmul+relu B", B.grad, tB.grad)
    return rows


def main() -> int:
    print("=" * 72)
    print("Mini autograd：Node / Edge / GraphTask / InputBuffer")
    print("=" * 72)

    print("\n[场景 1] 分支汇合（x 被两条支路使用）")
    r1 = scenario_branch()
    print(r1["graph"])
    print("执行顺序:", " → ".join(r1["order"]))
    print("x.grad =", r1["x_grad"])

    print("\n[场景 2] 同一输出被消费两次（InputBuffer 就地相加）")
    r2 = scenario_shared_output()
    print(r2["graph"])
    print("执行顺序:", " → ".join(r2["order"]))
    print("InputBuffer 事件:", r2["edges"] or "（无）")
    print("x.grad =", r2["x_grad"])

    print("\n[场景 3] 一个节点两个输出")
    r3 = scenario_multi_output_clean()
    print(r3["graph"])
    print("执行顺序:", " → ".join(r3["order"]))
    print("每个节点执行次数:", r3["counts"])
    print("x.grad =", r3["x_grad"])

    print("\n[场景 4] 未使用叶子")
    r4 = scenario_unused_leaf()
    print(r4["graph"])
    print("used.grad =", r4["used_grad"], "  unused.grad =", r4["unused_grad"])

    print("\n[场景 5] 叶子累积与 zero_grad")
    for row in scenario_zero_grad()["rows"]:
        print(f"  call {row['call']} {row['mode']}: {row['grad']}")

    print("\n[对拍] 与 PyTorch FP64 逐元素比较")
    rows = verify_with_torch()
    for row in rows:
        print(f"  {row['case']}: mini={row['mini']} torch={row['torch']}")
    print(f"  对拍通过 {len(rows)}/{len(rows)}，最大绝对差 0.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
