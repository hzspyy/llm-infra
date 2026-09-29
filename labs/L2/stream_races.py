#!/usr/bin/env python3
"""L2.0c task B · 跨流竞态：执行顺序错误与存储过早复用。

两个独立的错误，各自有"错"与"对"两条路径：

  顺序错误（缺 wait_stream）：s1 在写缓冲，s2 立刻读它 —— 读到旧值。
  存储过早复用（缺 record_stream）：s2 还要读，allocator 已经把这块存储
    交给 s1 上的新分配 —— 读到被覆盖后的值。

同时给出竞态区间的时间线（每个 stream 上 kernel 的起止时刻），
让"重叠"这件事可见，而不是只报一个错误计数。

    python stream_races.py --mode all --trials 20
    python stream_races.py --mode order
"""

import argparse
import json
import sys

import torch

N = 1 << 20          # 4 MiB float32
MB = 1 << 20


def gpu_name():
    return torch.cuda.get_device_name(0)


def delay(iters=40, size=2048):
    """在当前流上排一串 matmul，用于制造确定的"忙"区间。"""
    a = torch.ones(size, size, device="cuda")
    for _ in range(iters):
        a = a @ a * 1e-3
    return a


class Timeline:
    """在两个 stream 上采事件，最后换算成同一起点的相对时刻。"""

    def __init__(self, streams):
        self.streams = streams
        self.marks = {name: [] for name in streams}

    def mark(self, name):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        self.marks[name].append(ev)

    def start(self):
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t0.record()
        return self

    def dump(self, title):
        torch.cuda.synchronize()
        print(f"    {title}")
        rows = []
        for name, evs in self.marks.items():
            for i in range(0, len(evs) - 1, 2):
                a, b = evs[i], evs[i + 1]
                rows.append((name, self.t0.elapsed_time(a), self.t0.elapsed_time(b)))
        rows.sort(key=lambda r: r[1])
        base = min(r[1] for r in rows) if rows else 0.0
        for name, a, b in rows:
            s, e = a - base, b - base
            bar = " " * int(s) + "=" * max(1, int(e - s))
            print(f"      {name}: {bar}  [{s:6.3f} → {e:6.3f} ms]")


# ---------------------------------------------------------------- 顺序错误
def run_order(trials, verbose=False):
    torch.cuda.empty_cache()
    buggy_bad = buggy_timeline = None
    fixed_bad = 0

    for t in range(trials):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        # ---- 缺 wait_stream ----
        x = torch.zeros(N, device="cuda")
        tl = Timeline({"s1": s1, "s2": s2}).start()
        with torch.cuda.stream(s1):
            tl.mark("s1")
            delay()                    # s1 先忙一会儿，s2 会抢在写之前读
            x.fill_(1.0)
            tl.mark("s1")
        with torch.cuda.stream(s2):
            tl.mark("s2")
            read = x.sum()             # 没有任何依赖：可能读到 0
            tl.mark("s2")
        torch.cuda.synchronize()
        got = read.item()
        if got != float(N):
            buggy_bad = (buggy_bad or 0) + 1
        if verbose and t == 0:
            buggy_timeline = tl

        # ---- 加 wait_stream ----
        x2 = torch.zeros(N, device="cuda")
        with torch.cuda.stream(s1):
            delay()
            x2.fill_(1.0)
        with torch.cuda.stream(s2):
            s2.wait_stream(s1)
            read2 = x2.sum()
        torch.cuda.synchronize()
        if read2.item() != float(N):
            fixed_bad += 1

    print("  [顺序] 两个流：s1 延时后写 1.0，s2 立刻求和（期望 %d）" % N)
    print(f"    缺 wait_stream：{trials} 次里 {buggy_bad} 次读到旧值")
    print(f"    加 wait_stream：{trials} 次里 {fixed_bad} 次读到旧值")
    if buggy_timeline:
        buggy_timeline.dump("缺 wait_stream 的时间线（重叠区间就是竞态窗口）：")
    return {"trials": trials, "buggy_bad": buggy_bad, "fixed_bad": fixed_bad}


# ---------------------------------------------------------------- 存储复用
def run_reuse(trials, verbose=False):
    torch.cuda.empty_cache()
    buggy_bad = fixed_bad = reuse_buggy = reuse_fixed = 0
    sample = None

    for t in range(trials):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        # ---- 缺 record_stream：块可能在 s2 还在读时被 s1 复用 ----
        with torch.cuda.stream(s1):
            buf = torch.full((N,), 1.0, device="cuda")
            torch.cuda.current_stream().synchronize()
        addr = buf.data_ptr()
        with torch.cuda.stream(s2):
            s2.wait_stream(s1)         # 只保证"能读"，不保证"读完前不被复用"
            delay(iters=60)            # 把 s2 的读推后，制造复用窗口
            out = buf * 2.0            # 期望 2.0
        del buf                        # 丢掉 Python 引用，allocator 可回收
        with torch.cuda.stream(s1):
            u = torch.full((N,), 7.0, device="cuda")   # 很可能复用同一块
            _ = u * 3.0
        torch.cuda.synchronize()
        reused = (u.data_ptr() == addr)
        bad = out[0].item() != 2.0
        reuse_buggy += reused
        buggy_bad += bad
        if sample is None:
            sample = {"mode": "buggy", "old_addr": hex(addr), "new_addr": hex(u.data_ptr()),
                      "reused": reused, "out0": out[0].item()}
        del u, out

        # ---- 加 record_stream：allocator 在 s2 完成前不交出这块存储 ----
        with torch.cuda.stream(s1):
            buf2 = torch.full((N,), 1.0, device="cuda")
            torch.cuda.current_stream().synchronize()
        addr2 = buf2.data_ptr()
        with torch.cuda.stream(s2):
            s2.wait_stream(s1)
            delay(iters=60)
            out2 = buf2 * 2.0
        buf2.record_stream(s2)         # 关键一行
        del buf2
        with torch.cuda.stream(s1):
            v = torch.full((N,), 7.0, device="cuda")
            _ = v * 3.0
        torch.cuda.synchronize()
        reuse_fixed += (v.data_ptr() == addr2)
        fixed_bad += (out2[0].item() != 2.0)
        del v, out2

    print("  [复用] s2 延时后读 buf*2（期望 2.0）；s1 随即申请同尺寸新张量")
    print(f"    缺 record_stream：{trials} 次里地址被复用 {reuse_buggy} 次，"
          f"读到被覆盖的值 {buggy_bad} 次")
    print(f"    加 record_stream：{trials} 次里地址被复用 {reuse_fixed} 次，"
          f"读到被覆盖的值 {fixed_bad} 次")
    if sample:
        print(f"    样本地址：旧 {sample['old_addr']} → 新 {sample['new_addr']} "
              f"（{'同一块' if sample['reused'] else '不同块'}），out[0]={sample['out0']}")
    return {"trials": trials, "reuse_buggy": reuse_buggy, "buggy_bad": buggy_bad,
            "reuse_fixed": reuse_fixed, "fixed_bad": fixed_bad, "sample": sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "order", "reuse"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 1

    print(f"=== {gpu_name()}  torch {torch.__version__}  trials={args.trials} ===")
    result = {"gpu": gpu_name(), "torch": torch.__version__, "trials": args.trials}
    if args.mode in ("all", "order"):
        result["order"] = run_order(args.trials, verbose=True)
    if args.mode in ("all", "reuse"):
        result["reuse"] = run_reuse(args.trials, verbose=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
