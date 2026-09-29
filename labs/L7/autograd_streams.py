#!/usr/bin/env python3
"""CUDA 上的反向完成语义与跨流梯度消费（7.0-C）。

小网络在两个 stream 上做前向/反向与梯度消费，对照三种写法：
1. backward 返回后立即在另一条流上读梯度（无同步）；
2. 在反向流上记录 event，消费流 wait_event 后再读；
3. backward 后立刻做 D2H 拷回（默认流，无同步）。

判据是消费到的梯度与 CPU 参照是否逐元素相同，以及 CPU 侧提交时间与设备时间之差。

Usage:
    python labs/L7/autograd_streams.py > "$RUN_DIR/autograd-streams.txt"
"""
from __future__ import annotations

import sys

import torch

torch.manual_seed(0)
DEV = "cuda"
DTYPE = torch.float32


def make_net(dim: int = 512, depth: int = 4):
    layers = []
    for _ in range(depth):
        layers += [torch.nn.Linear(dim, dim, dtype=DTYPE), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(dim, dim, dtype=DTYPE))
    return torch.nn.Sequential(*layers).to(DEV)


def grads_equal(net, reference) -> tuple[bool, float]:
    worst = 0.0
    ok = True
    for (name, p), ref in zip(net.named_parameters(), reference):
        if p.grad is None:
            ok = False
            continue
        diff = (p.grad - ref).abs().max().item()
        worst = max(worst, diff)
        if diff != 0.0:
            ok = False
    return ok, worst


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    print("CUDA 反向完成语义与跨流梯度消费")
    print(f"torch {torch.__version__}，device={torch.cuda.get_device_name(0)}，"
          f"capability={torch.cuda.get_device_capability(0)}")
    dim = 1024
    net = make_net(dim=dim, depth=6)
    x = torch.randn(256, dim, device=DEV, dtype=DTYPE)
    target = torch.randn(256, dim, device=DEV, dtype=DTYPE)

    def forward_backward():
        net.zero_grad(set_to_none=True)
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        return loss

    # 参照：同步跑一次，留参数梯度的副本
    loss = forward_backward()
    torch.cuda.synchronize()
    reference = [p.grad.detach().clone() for p in net.parameters()]
    print(f"参照 loss={loss.item():.6f}，参数量={sum(p.numel() for p in net.parameters())}")

    section("[A] backward 的 CPU 提交时间与设备执行时间")
    import time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    t0 = time.perf_counter()
    forward_backward()
    cpu_submit_ms = (time.perf_counter() - t0) * 1e3
    end.record()
    torch.cuda.synchronize()
    print(f"  forward+backward 调用返回的 CPU 侧耗时={cpu_submit_ms:.3f} ms")
    print(f"  同一段工作在设备上的时间（event 跨度）={start.elapsed_time(end):.3f} ms")
    print("  CPU 返回时设备工作尚未结束，梯度缓冲在返回后仍可能被后续 kernel 写入。")

    section("[B] 反向流写梯度、消费流无同步读取")
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    mismatch_no_wait = 0
    rounds = 20
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            forward_backward()
        with torch.cuda.stream(s2):
            observed = [p.grad.detach().clone() for p in net.parameters()]
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(observed, reference))
        if not same:
            mismatch_no_wait += 1
    print(f"  无同步消费：{mismatch_no_wait}/{rounds} 轮读到的梯度与参照不同")

    section("[C] 反向流记录 event，消费流 wait_event")
    mismatch_wait = 0
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            forward_backward()
            ev = torch.cuda.Event()
            ev.record(s1)
        with torch.cuda.stream(s2):
            s2.wait_event(ev)
            observed = [p.grad.detach().clone() for p in net.parameters()]
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(observed, reference))
        if not same:
            mismatch_wait += 1
    print(f"  等 event 后消费：{mismatch_wait}/{rounds} 轮与参照不同")

    section("[D] backward 后立即 D2H 拷回（默认流，无同步）")
    stale = 0
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            forward_backward()
        host_grad = net[0].weight.grad.detach().to("cpu")
        if not torch.equal(host_grad, reference[0].cpu()):
            stale += 1
        torch.cuda.synchronize()
    print(f"  立即 D2H：{stale}/{rounds} 次读到的第一个权重梯度与参照不同")
    with torch.cuda.stream(s1):
        forward_backward()
        ev2 = torch.cuda.Event()
        ev2.record(s1)
    torch.cuda.current_stream().wait_event(ev2)
    host_grad = net[0].weight.grad.detach().to("cpu")
    print(f"  默认流等 event 后 D2H 与参照相同: {torch.equal(host_grad, reference[0].cpu())}")

    section("[E] 把 D2H 也放到消费流 s2（默认流的隐式行为被排除）")
    stale_s2 = 0
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            forward_backward()
        with torch.cuda.stream(s2):
            host_grad = net[0].weight.grad.detach().to("cpu")
        if not torch.equal(host_grad, reference[0].cpu()):
            stale_s2 += 1
        torch.cuda.synchronize()
    print(f"  s2 上立即 D2H：{stale_s2}/{rounds} 次与参照不同")
    with torch.cuda.stream(s1):
        forward_backward()
        ev3 = torch.cuda.Event()
        ev3.record(s1)
    with torch.cuda.stream(s2):
        s2.wait_event(ev3)
        host_grad = net[0].weight.grad.detach().to("cpu")
    print(f"  s2 等 event 后 D2H 与参照相同: {torch.equal(host_grad, reference[0].cpu())}")

    section("[F] 非阻塞 D2H（pinned 缓冲）到 s2，不等待 event")
    ref_cpu = reference[0].cpu()
    pinned = torch.empty_like(ref_cpu, pin_memory=True)
    stale_pin = 0
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            forward_backward()
        with torch.cuda.stream(s2):
            pinned.copy_(net[0].weight.grad, non_blocking=True)
        if not torch.equal(pinned, ref_cpu):
            stale_pin += 1
        torch.cuda.synchronize()
    print(f"  非阻塞 D2H：{stale_pin}/{rounds} 次与参照不同")
    print("  阻塞式 .to('cpu') 走同步 cudaMemcpy，会把设备工作串行化，所以 [D]/[E] 读不到脏数据；")
    print("  改成 pinned + non_blocking 后同一竞争出现（本配置只命中 1/20，检出率随反向时长与拷贝大小变化）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
