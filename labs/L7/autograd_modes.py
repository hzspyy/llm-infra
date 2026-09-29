#!/usr/bin/env python3
"""梯度模式、原地修改与保存值生命周期。

每个用例记录：触发阶段（前向/反向）、原文报错、版本计数和梯度结果。
全部 FP64，数值可逐位比较。

Usage:
    python labs/L7/autograd_modes.py > "$RUN_DIR/autograd-modes.txt"
"""
from __future__ import annotations

import sys

import torch

torch.manual_seed(0)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0]


# ---------------------------------------------------------------------------
class SquareWithSaved(torch.autograd.Function):
    """保存输入的 Function，用来观测 saved_tensors 的释放。"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2.0 * x * grad_output


def case_retain_graph() -> None:
    section("[A] retain_graph：同一张图能否第二次 backward")
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    loss = SquareWithSaved.apply(x).sum()
    loss.backward()
    print(f"第一次 backward 后 x.grad = {x.grad.tolist()}")
    try:
        loss.backward()
        print("第二次 backward 成功")
    except RuntimeError as exc:
        print(f"第二次 backward 失败（反向阶段）：{first_line(exc)}")

    x2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    loss2 = SquareWithSaved.apply(x2).sum()
    loss2.backward(retain_graph=True)
    print(f"retain_graph=True 第一次后 x2.grad = {x2.grad.tolist()}")
    loss2.backward()
    print(f"retain_graph=True 第二次后 x2.grad = {x2.grad.tolist()}（叶子累积）")


def case_create_graph() -> None:
    section("[B] create_graph：高阶梯度")
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    y = x ** 3
    g1 = torch.autograd.grad(y, x, create_graph=True)[0]
    g2 = torch.autograd.grad(g1, x)[0]
    x2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    y2 = x2 ** 3
    g1_flat = torch.autograd.grad(y2, x2)[0]
    print(f"y=x^3 在 x=2：一阶 create_graph=True 得 {g1.item()}（解析 3x^2=12）")
    print(f"二阶得 {g2.item()}（解析 6x=12）")
    print(f"一阶 create_graph=False 得 {g1_flat.item()}，其 requires_grad={g1_flat.requires_grad}")
    try:
        torch.autograd.grad(g1_flat, x2)
    except RuntimeError as exc:
        print(f"对未建图的一阶梯度再求导失败：{first_line(exc)}")


def case_no_grad_modes() -> None:
    section("[C] no_grad / inference_mode / detach")
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        y = x * 3
    print(f"no_grad 内 y.requires_grad={y.requires_grad}，y.grad_fn={y.grad_fn}")

    x2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    with torch.inference_mode():
        y2 = x2 * 3
    print(f"inference_mode 内 y2.requires_grad={y2.requires_grad}，is_inference={y2.is_inference()}")
    x4 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    try:
        loss = (x4 * y2).sum()
        loss.backward()
        print("inference tensor 参与需要梯度的算子未报错")
    except RuntimeError as exc:
        print(f"inference tensor 被需要梯度的算子保存时失败（前向阶段）：{first_line(exc)}")

    x3 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    y3 = x3 * 3
    z = y3.detach()
    print(f"detach 后 requires_grad={z.requires_grad}，与 y3 共享存储={z.data_ptr() == y3.data_ptr()}")
    z.add_(1.0)
    print(f"改写 detached 张量会改到原张量：y3={y3.tolist()}")
    loss = (y3 * 2).sum()
    loss.backward()
    print(f"detach 处截断后 x3.grad={x3.grad.tolist()}（上游仍可求导）")


def case_saved_release() -> None:
    section("[E] 保存值释放：释放后哪些反向仍可执行")
    x = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    y = SquareWithSaved.apply(x)
    node = y.grad_fn
    print(f"前向刚结束可读取 saved_tensors={[tuple(t.shape) for t in node.saved_tensors]}")
    y.sum().backward()
    try:
        node.saved_tensors
    except RuntimeError as exc:
        print(f"backward 后读取失败（反向阶段）：{first_line(exc)}")

    x2 = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    y2 = SquareWithSaved.apply(x2)
    node2 = y2.grad_fn
    y2.sum().backward(retain_graph=True)
    print(f"retain_graph=True 时 backward 后仍可读取 {len(node2.saved_tensors)} 个保存值")

    x3 = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
    a = x3 * 2
    b = SquareWithSaved.apply(x3)
    torch.autograd.grad(a.sum(), x3)          # 释放 a 分支
    print("释放一条分支后，另一条独立图的 backward 仍可执行：", end="")
    g, = torch.autograd.grad(b.sum(), x3)
    print(f"grad={g.tolist()}（解析 2x=8）")


def case_inplace_errors() -> None:
    section("[F/G] 原地修改的三个阶段")
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    try:
        x.add_(1.0)
        print("叶子原地写没有报错")
    except RuntimeError as exc:
        print(f"叶子原地写在调用处报错（前向阶段）：{first_line(exc)}")

    x2 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    y = x2 * 2.0
    z = y * y                # MulBackward 保存 y，记录当时的 version
    print(f"反向需要 y：y._version={y._version}，z.grad_fn={type(z.grad_fn).__name__}，y.grad_fn={type(y.grad_fn).__name__}")
    y.add_(1.0)
    print(f"原地写后 y._version={y._version}，y.grad_fn 被改挂到原地算子：{type(y.grad_fn).__name__}")
    try:
        z.sum().backward()
        print("原地写后 backward 未报错")
    except RuntimeError as exc:
        print(f"保存值版本不匹配在反向阶段报错：{first_line(exc)}")

    x3 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    y3 = x3 + 1.0            # AddBackward 不需要保存输入
    print(f"AddBackward 保存值字段={[n for n in ('_saved_self', '_saved_other') if hasattr(y3.grad_fn, n)]}")
    y3.add_(2.0)
    (y3 * 3.0).sum().backward()
    print(f"原地写不破坏不需要保存值的算子：x3.grad={x3.grad.tolist()}（解析 3）")


def case_version_counter() -> None:
    section("[H] 版本计数与保存值")
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    print(f"叶子 x._version={x._version}")
    with torch.no_grad():
        x.add_(1.0)
    print(f"no_grad 内原地写后 x._version={x._version}（叶子自身不保存值，允许）")
    y = x * 2.0
    print(f"y._version={y._version}，grad_fn={type(y.grad_fn).__name__}")


def main() -> int:
    print("autograd 模式与保存值生命周期（torch", torch.__version__, "FP64 CPU）")
    case_retain_graph()
    case_create_graph()
    case_no_grad_modes()
    case_saved_release()
    case_inplace_errors()
    case_version_counter()
    return 0


if __name__ == "__main__":
    sys.exit(main())
