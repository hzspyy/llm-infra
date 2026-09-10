#!/usr/bin/env python3
"""L1.4 证据补齐 · 供 strace 统计的最小负载。

设计要点：**这个脚本只负责跑 N 次固定动作然后立刻退出**，
不打印、不写文件、不做任何会自己产生系统调用的事。
真正的测量是外面 `syscall_ladder.sh` 做的**差分**：
    每次 launch 的系统调用数 = (N2 的总数 - N1 的总数) / (N2 - N1)
这样 import torch、建 context、加载 cubin 这些一次性开销全部抵消掉。

模式：
  launch  N 次 kernel 下发，最后同步一次
  sync    N 次 (kernel + cudaDeviceSynchronize)
  graph   把 100 次 launch 录成一张 CUDA Graph，replay N/100 次
  d2h     N 次 1 元素的 device->host 拷贝（.item()），用来对照"必须回 host"
  idle    什么都不做，只做初始化 —— 差分的基线
"""

import sys


def main() -> None:
    mode = sys.argv[1]
    n = int(sys.argv[2])

    import os as _os
    _mark_on = _os.environ.get("LADDER_MARK") == "1"

    def mark(tag: str) -> None:
        """写一条标记到 fd 2。在 strace 里它是一次 write，
        用来把 sched_yield 归到具体阶段（提交 / 同步）。"""
        if _mark_on:
            _os.write(2, f"@@{tag}\n".encode())

    import torch
    assert torch.cuda.is_available()
    x = torch.zeros(1, device="cuda")

    # 充分预热：让 cubin 加载、内存池建立、autotune 全部发生在计数之前
    for _ in range(2000):
        x.add_(1.0)
    torch.cuda.synchronize()

    if mode == "idle":
        pass

    elif mode == "launch":
        for _ in range(n):
            x.add_(1.0)
        torch.cuda.synchronize()

    elif mode == "sync":
        for _ in range(n):
            x.add_(1.0)
            torch.cuda.synchronize()

    elif mode.startswith("graph"):  # graph:k / graphsync:k
        # graph / graph:k —— k 是一张图里录多少个 kernel。
        # 用来检验 sched_yield 是不是"提交比 GPU 消费快"造成的排队回压：
        # k 越大，同样的 CPU 工作量下压进去的 kernel 越多。
        k = int(mode.split(":")[1]) if ":" in mode else 100
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(k):
                x.add_(1.0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(k):
                x.add_(1.0)
        torch.cuda.synchronize()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        # graphsync:k —— 每次 replay 后立刻同步，队列里最多一个未完成的图。
        # 如果 sched_yield 是"提交队列满了在等"，这个模式应该把它压到 0。
        per_replay_sync = mode.startswith("graphsync")
        mark("submit_begin")
        for _ in range(n // k):
            g.replay()
            if per_replay_sync:
                torch.cuda.synchronize()
        mark("submit_end_sync_begin")
        torch.cuda.synchronize()
        mark("sync_end")

    elif mode == "d2h":
        for _ in range(n):
            x.add_(1.0)
            x.item()

    else:
        raise SystemExit(f"unknown mode {mode}")

    # 立刻退出，不要走 atexit 里的清理（那会产生额外的 munmap/ioctl）
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
