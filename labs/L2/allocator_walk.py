#!/usr/bin/env python3
"""L2.0 后半 —— caching allocator、碎片与 stream 生命周期。

`del x` 之后显存还给驱动了吗？一次 OOM 里 reserved 和 allocated 差在哪？
这些问题只能用 memory_snapshot 和真实的 OOM 报文回答。

注意：PYTORCH_CUDA_ALLOC_CONF 必须在 CUDA 初始化前设置，所以碎片 A/B
用两个子进程各跑一遍（见 --frag-child）。

用法：
    python allocator_walk.py            # 全跑
    python allocator_walk.py G H
"""

import os
import subprocess
import sys

import torch

MB = 1024 * 1024


def title(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 72 - len(s)))


def mem(tag=""):
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    print(f"  {tag:<34} allocated={a / MB:9.3f} MiB  reserved={r / MB:9.3f} MiB"
          f"  驱动可见空闲={free / MB:8.1f} MiB")


def dump_segments(limit=8):
    snap = torch.cuda.memory_snapshot()
    print(f"  共 {len(snap)} 个 segment")
    print(f"  {'address':>16} {'type':<6} {'total':>10} {'alloc':>10} "
          f"{'stream':>7}  blocks(size/state)")
    for s in snap[:limit]:
        blocks = " ".join(
            f"{b['size'] / 1024:.0f}K/{b['state'].replace('active_', '')}"
            for b in s["blocks"][:6])
        more = " ..." if len(s["blocks"]) > 6 else ""
        print(f"  {s['address']:>#16x} {s['segment_type']:<6}"
              f" {s['total_size'] / 1024:9.0f}K {s['allocated_size'] / 1024:9.0f}K"
              f" {s['stream']:>7}  {blocks}{more}")
    if len(snap) > limit:
        print(f"  ... 还有 {len(snap) - limit} 个")


# ---------------------------------------------------------------- G
def section_G():
    title("[G] caching allocator：del 之后显存还给驱动了吗")

    sub("0. CUDA context 本身要多少显存")
    # 注意：torch.cuda.mem_get_info() 自己就会建 context，用它测不出 context 的开销。
    # 必须从进程外面看 —— nvidia-smi 读的是驱动侧的账。
    def smi_used():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split("\n")[0].strip()
        return int(out)

    u0 = smi_used()
    print(f"  1) torch.cuda 尚未初始化        nvidia-smi 已用 = {u0:>5} MiB")
    torch.cuda.init()
    u1 = smi_used()
    print(f"  2) torch.cuda.init() 之后       nvidia-smi 已用 = {u1:>5} MiB   (+{u1 - u0})")
    torch.cuda.synchronize()          # 真正逼出 context，但不经过 allocator
    u2 = smi_used()
    print(f"  3) cudaDeviceSynchronize 之后   nvidia-smi 已用 = {u2:>5} MiB   (+{u2 - u1})")
    _ = torch.empty(1, device="cuda")
    u3 = smi_used()
    print(f"  4) 第一次 torch.empty(1) 之后   nvidia-smi 已用 = {u3:>5} MiB   (+{u3 - u2})")
    print(f"\n  -> torch.cuda.init() 是惰性的，它 **没有** 建 context（+{u1 - u0} MiB）。")
    print(f"  -> context 在第一次真正碰 GPU 时才建，代价 ≈ {u2 - u1} MiB。")
    print(f"  -> 之后 allocator 的第一个 segment 只占 {u3 - u2} MiB。")
    print(f"  -> 也就是说，一个什么都没干的 torch 进程已经吃掉 {u3 - u0} MiB 显存。")
    free, total = torch.cuda.mem_get_info()
    print(f"  torch 自己看到的: free={free / MB:.1f} / {total / MB:.1f} MiB")
    del _
    torch.cuda.empty_cache()

    sub("1. 分配 4 字节，看看真的动了多少")
    torch.cuda.reset_peak_memory_stats()
    mem("起点")
    a = torch.empty(1, dtype=torch.float32, device="cuda")
    mem("torch.empty(1)  = 4 字节")
    print(f"  -> allocated 是 {torch.cuda.memory_allocated()} 字节，不是 4。"
          f"分配粒度被向上取整到 512。")
    print(f"  -> reserved 是 {torch.cuda.memory_reserved() / MB:.0f} MiB，"
          f"因为 small pool 一次向驱动要 2 MiB。")
    dump_segments()

    sub("2. 再要 100 个小块：全部从同一个 2 MiB segment 里切")
    small = [torch.empty(256, dtype=torch.float32, device="cuda") for _ in range(100)]
    mem("100 × 1 KiB")
    dump_segments()

    sub("3. 一个 3 MiB 的大块：走 large pool，另开 segment")
    big = torch.empty(3 * MB // 4, dtype=torch.float32, device="cuda")
    mem("+ 3 MiB")
    dump_segments()

    sub("4. del 大块 —— allocated 掉了，reserved 没掉")
    del big
    mem("del big")
    print("  显存一个字节都没还给驱动。它被挂回 allocator 的空闲链表，等下一次复用。")
    dump_segments()

    sub("5. 同样大小的请求直接命中缓存（没有 cudaMalloc）")
    s0 = torch.cuda.memory_stats()
    big2 = torch.empty(3 * MB // 4, dtype=torch.float32, device="cuda")
    s1 = torch.cuda.memory_stats()
    print(f"  num_alloc_retries      {s0.get('num_alloc_retries', 0)} -> "
          f"{s1.get('num_alloc_retries', 0)}")
    print(f"  segment.all.allocated  {s0['segment.all.allocated']} -> "
          f"{s1['segment.all.allocated']}   (没有新增 = 没有调 cudaMalloc)")
    print(f"  reserved_bytes.all.current 不变: "
          f"{s0['reserved_bytes.all.current'] == s1['reserved_bytes.all.current']}")

    sub("6. empty_cache() 才真的还给驱动")
    del big2, small, a
    mem("del 全部")
    torch.cuda.empty_cache()
    mem("empty_cache()")
    print("  驱动可见空闲回涨了 —— 这才是真的 cudaFree。")

    sub("7. memory_stats 里值得记住的几行")
    st = torch.cuda.memory_stats()
    for k in ["allocated_bytes.all.peak", "reserved_bytes.all.peak",
              "active_bytes.all.peak", "allocation.all.allocated",
              "segment.all.allocated", "num_alloc_retries", "num_ooms"]:
        v = st.get(k, "n/a")
        if isinstance(v, int) and k.endswith("bytes.all.peak"):
            print(f"  {k:<34} {v:>14,}  ({v / MB:.1f} MiB)")
        else:
            print(f"  {k:<34} {v:>14,}")


# ---------------------------------------------------------------- frag child
def frag_child():
    """在一个干净进程里做碎片实验。由 section_FRAG 启子进程调用。

    结论先说：碎片在 PyTorch 里**通常不表现为崩溃**，而表现为一次隐藏的停顿 ——
    allocator 内部先 OOM，再把整个缓存 cudaFree 掉重试。所以这里测的是
    「那一次分配花了多久」和 num_alloc_retries，而不是等一个异常。
    """
    import gc
    import time
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(默认)")
    print(f"PYTORCH_CUDA_ALLOC_CONF = {conf}")
    chunk = 256 * MB
    hold = []
    # 1. 把显存吃满
    while True:
        try:
            hold.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
        except torch.cuda.OutOfMemoryError:
            break
    print(f"吃满：{len(hold)} × 256 MiB")
    mem("满载")
    # 2. 隔一个放一个 -> 留下一堆 256 MiB 的洞
    freed = 0
    for i in range(0, len(hold), 2):
        hold[i] = None
        freed += 1
    gc.collect()
    print(f"释放了 {freed} 块（隔一释一），理论空闲 {freed * 256} MiB，"
          f"但全是 256 MiB 的洞，且分属 {freed} 个不同的 segment")
    mem("释放后")

    s0 = torch.cuda.memory_stats()
    r0 = torch.cuda.memory_reserved()
    # 3. 要一个 512 MiB 的连续块 —— 没有任何一个洞装得下
    print("\n请求一块 512 MiB（比任何一个洞都大）：")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        big = torch.empty(512 * MB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  >>> 成功，耗时 {dt:.1f} ms")
    except torch.cuda.OutOfMemoryError as exc:
        dt = (time.perf_counter() - t0) * 1000
        print(f"  >>> OutOfMemoryError（{dt:.1f} ms）")
        for line in str(exc).splitlines():
            print("      " + line)
        big = None
    s1 = torch.cuda.memory_stats()
    r1 = torch.cuda.memory_reserved()
    print(f"  num_alloc_retries {s0.get('num_alloc_retries')} -> "
          f"{s1.get('num_alloc_retries')}   "
          f"num_ooms {s0.get('num_ooms')} -> {s1.get('num_ooms')}")
    print(f"  reserved {r0 / MB:.0f} -> {r1 / MB:.0f} MiB   "
          f"(掉了 {(r0 - r1) / MB:.0f} MiB = 被 cudaFree 还给驱动的缓存)")
    print(f"  segment 数 = {len(torch.cuda.memory_snapshot())}")

    # 4. 对照：同样的分配，在没有碎片的干净堆上要多久
    del hold, big
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    clean = torch.empty(512 * MB, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    print(f"\n  对照：干净堆上同样要 512 MiB，耗时 "
          f"{(time.perf_counter() - t0) * 1000:.1f} ms")
    del clean

    # 5. 真正的 OOM 长什么样：要一块比总显存还大的
    print("\n真正的 OOM（请求 40 GiB，卡只有 32 GiB）：")
    try:
        torch.empty(40 * 1024 * MB, dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError as exc:
        for line in str(exc).splitlines():
            print("      " + line)


def churn_child():
    """变长负载：反复分配不同大小的 activation，留一部分活着。

    这才是 expandable_segments 针对的场景 —— 缓存里攒下一堆「尺寸不对」的块，
    reserved 一路涨而 allocated 不涨。指标是 reserved/allocated，不是单次延迟。
    """
    import gc
    import random
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(默认)")
    print(f"PYTORCH_CUDA_ALLOC_CONF = {conf}")
    rng = random.Random(20260910)
    live = []
    budget = 8 * 1024 * MB
    for _ in range(600):
        n = rng.randint(1, 96) * 4 * MB          # 4 .. 384 MiB，尺寸各不相同
        try:
            live.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
        except torch.cuda.OutOfMemoryError:
            live.clear()
            gc.collect()
            continue
        while sum(t.numel() for t in live) > budget:
            live.pop(rng.randrange(len(live)))
    gc.collect()
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    st = torch.cuda.memory_stats()
    print(f"  600 次变长分配后")
    print(f"  allocated      {a / MB:9.0f} MiB")
    print(f"  reserved       {r / MB:9.0f} MiB")
    print(f"  reserved/allocated = {r / max(a, 1):.2f}×   "
          f"<-- 这个比值就是碎片")
    print(f"  浪费           {(r - a) / MB:9.0f} MiB")
    print(f"  peak reserved  {st['reserved_bytes.all.peak'] / MB:9.0f} MiB")
    print(f"  peak allocated {st['allocated_bytes.all.peak'] / MB:9.0f} MiB")
    print(f"  num_alloc_retries = {st.get('num_alloc_retries')}   "
          f"num_ooms = {st.get('num_ooms')}")
    print(f"  segment 数 = {len(torch.cuda.memory_snapshot())}")


def _run_child(flag: str, conf: str) -> None:
    here = os.path.abspath(__file__)
    env = dict(os.environ)
    if conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = conf
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    r = subprocess.run([sys.executable, here, flag],
                       env=env, capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr[-2000:])


def section_FRAG():
    title("[G-2] 碎片其一：满载 + 等大的洞 —— 结果不是 OOM，是一次停顿")
    for conf in ["", "expandable_segments:True"]:
        sub(f"PYTORCH_CUDA_ALLOC_CONF = {conf or '(默认，cudaMalloc 分段)'}")
        _run_child("--frag-child", conf)

    title("[G-3] 碎片其二：变长负载 —— reserved 相对 allocated 漂多远")
    for conf in ["", "expandable_segments:True"]:
        sub(f"PYTORCH_CUDA_ALLOC_CONF = {conf or '(默认，cudaMalloc 分段)'}")
        _run_child("--churn-child", conf)


# ---------------------------------------------------------------- H
def section_H():
    title("[H] stream：allocator 记的是「哪条流」，不是「哪个时刻」")

    sub("1. 块上记着分配它的 stream")
    s = torch.cuda.Stream()
    torch.cuda.empty_cache()
    a = torch.empty(4 * MB, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(s):
        b = torch.empty(4 * MB, dtype=torch.uint8, device="cuda")
    print(f"  default stream = {torch.cuda.current_stream().cuda_stream:#x}")
    print(f"  side    stream = {s.cuda_stream:#x}")
    dump_segments()
    print("  segment 的 stream 字段就是分配时的 current_stream。"
          "块只在这条流上是安全的。")
    del a, b
    torch.cuda.empty_cache()

    sub("2. 跨流不同步 —— 真的会读到旧值吗")
    n = 4096
    trials = 6
    bad_nosync = 0
    bad_sync = 0
    for _ in range(trials):
        m1 = torch.randn(n, n, device="cuda")
        m2 = torch.randn(n, n, device="cuda")
        s2 = torch.cuda.Stream()
        x = torch.zeros(n, n, device="cuda")
        # 生产者：默认流上一串很慢的 matmul，最后写进 x
        for _ in range(20):
            x = m1 @ m2
        with torch.cuda.stream(s2):
            got = x.sum().item()          # 没有 wait_stream
        torch.cuda.synchronize()
        want = x.sum().item()
        if abs(got - want) > 1e-3 * max(1.0, abs(want)):
            bad_nosync += 1

        x = torch.zeros(n, n, device="cuda")
        for _ in range(20):
            x = m1 @ m2
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            got = x.sum().item()
        torch.cuda.synchronize()
        want = x.sum().item()
        if abs(got - want) > 1e-3 * max(1.0, abs(want)):
            bad_sync += 1
        del m1, m2, x
    print(f"  不加 wait_stream：{trials} 次里 {bad_nosync} 次读到不一致的值")
    print(f"  加了 wait_stream：{trials} 次里 {bad_sync} 次读到不一致的值")
    print("  注：.item() 自带一次同步，所以这个竞态常常被掩盖 —— "
          "掩盖不等于不存在，见正文。")
    torch.cuda.empty_cache()

    sub("3. record_stream：让 allocator 知道别的流还在用这块内存")
    torch.cuda.empty_cache()
    s3 = torch.cuda.Stream()
    t = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
    print(f"  t 在默认流上分配，data_ptr={t.data_ptr():#x}")
    with torch.cuda.stream(s3):
        t.record_stream(s3)
    print("  t.record_stream(s3) 之后，allocator 会等 s3 上的事件完成才复用这块。")
    print("  没有这一句，t 一被 del，同一块地址可能立刻发给默认流上的下一次分配，")
    print("  而 s3 上的 kernel 还在读它 —— 典型的 use-after-free，且不会报错。")
    del t
    torch.cuda.empty_cache()


SECTIONS = {"G": section_G, "FRAG": section_FRAG, "H": section_H}

if __name__ == "__main__":
    if "--frag-child" in sys.argv:
        frag_child()
        sys.exit(0)
    if "--churn-child" in sys.argv:
        churn_child()
        sys.exit(0)
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"gpu   {torch.cuda.get_device_name(0)}")
    print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(未设置)')}")
    for s in want:
        SECTIONS[s]()
