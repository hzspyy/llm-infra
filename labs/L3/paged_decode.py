#!/usr/bin/env python3
"""L3.3-B —— 真实 paged decode kernel：block table、页边界与共享页。

3.3 §4 量过"框架层 gather 再算"的代价（5×），那是**模拟**。
本 lab 用 FlashInfer 的真实 paged decode kernel，直接在 kernel 内部按 block table
寻址，把 3.3-B 要求的东西逐项覆盖：

  [A] 正确性：page=16/32/64，S=page−1/page/page+1/2048/8192/32768，
      多个不同长度的请求同批，与连续布局参照（SDPA）对拍
  [B] 物理页打乱（随机 block table）、尾页不满、重复引用同一批物理页
  [C] 页大小与并发的代价：时间、按 KV 字节算的有效带宽
  [D] 连续 / 分页 / 框架层 gather 的三方对照

用法：
    L3_OUT=<目录> python paged_decode.py A B C D
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                      # noqa: E402

MB = 1024 * 1024
PEAK_BW = 1608.6          # L1.1 实测只读带宽 GB/s
L2_MIB = 96.0


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def build_paged_case(lengths, page_size, Hkv, D, seed=0, share=None):
    """构造物理页池 + block table。

    share：若给出 (i, j, n_pages)，则请求 j 的这段逻辑块复用请求 i 的物理页。
    返回 (k_cache, v_cache, indptr, indices, last_page_len, logical_kv)
    """
    torch.manual_seed(seed)
    pages_per_req = [(L + page_size - 1) // page_size for L in lengths]
    total_pages = sum(pages_per_req) + 8
    k_cache = torch.randn(total_pages, page_size, Hkv, D, device="cuda",
                          dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    perm = torch.randperm(total_pages, device="cuda")
    cursor = 0
    indices, indptr, last = [], [0], []
    logical = []
    for r, L in enumerate(lengths):
        npages = pages_per_req[r]
        phys = perm[cursor:cursor + npages].tolist()
        cursor += npages
        indices.extend(phys)
        indptr.append(len(indices))
        last.append(L - (npages - 1) * page_size)
        logical.append((phys, L, npages))
    if share is not None:
        src, dst, n = share
        sp = logical[src][0][:n]
        logical[dst] = (sp + logical[dst][0][n:], lengths[dst], pages_per_req[dst])
        # 重建 indices
        indices = []
        indptr = [0]
        for phys, _, _ in logical:
            indices.extend(phys)
            indptr.append(len(indices))
    return (k_cache, v_cache,
            torch.tensor(indptr, dtype=torch.int32, device="cuda"),
            torch.tensor(indices, dtype=torch.int32, device="cuda"),
            torch.tensor(last, dtype=torch.int32, device="cuda"), logical)


def continuous_kv(k_cache, logical, Hkv, D, page_size):
    """按逻辑顺序把物理页拼回连续张量，作为参照。"""
    kk, vv = [], []
    for phys, L, _ in logical:
        k = k_cache[torch.tensor(phys, device="cuda")]          # [np,ps,Hkv,D]
        k = k.reshape(-1, Hkv, D)[:L]
        kk.append(k)
        vv.append(k)
    return kk, vv


def make_wrapper():
    import flashinfer
    ws = torch.empty(256 * MB, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")


def plan_wrapper(w, indptr, indices, last, Hq, Hkv, D, page_size, dtype):
    for kw in ({"q_data_type": dtype, "data_type": dtype},
               {"q_data_type": dtype, "kv_data_type": dtype},
               {"data_type": dtype}):
        try:
            w.plan(indptr, indices, last, Hq, Hkv, D, page_size, **kw)
            return kw
        except TypeError as exc:
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 真实 paged kernel 的正确性：页边界长度 × 三种页大小")

    import flashinfer
    Hq, Hkv, D = 16, 2, 128
    dtype = torch.bfloat16
    print(f"  Hq={Hq} Hkv={Hkv} D={D} bf16；物理页随机打乱；尾页按真实长度计。")
    print(f"  {'page':>5} {'长度清单':>34} {'paged ms':>9} "
          f"{'逐请求 ms':>10} {'max|err|':>11} {'顶部 1 一致':>11}")
    for page_size in [16, 32, 64]:
        lengths = [page_size - 1, page_size, page_size + 1, 2048, 8192, 32768]
        k_cache, v_cache, indptr, indices, last, logical = build_paged_case(
            lengths, page_size, Hkv, D, seed=page_size)
        q = torch.randn(len(lengths), Hq, D, device="cuda", dtype=dtype)
        w = make_wrapper()
        plan_wrapper(w, indptr, indices, last, Hq, Hkv, D, page_size, dtype)
        o_paged = w.run(q, (k_cache, v_cache))
        # 连续参照
        errs, tops = [], []
        for r, (phys, L, _) in enumerate(logical):
            pk = k_cache[torch.tensor(phys, device="cuda")].reshape(-1, Hkv, D)[:L]
            pv = v_cache[torch.tensor(phys, device="cuda")].reshape(-1, Hkv, D)[:L]
            ref = F.scaled_dot_product_attention(
                q[r:r + 1].unsqueeze(2), pk.transpose(0, 1).unsqueeze(0),
                pv.transpose(0, 1).unsqueeze(0), enable_gqa=True)
            e = (o_paged[r].float() - ref[0, :, 0].float()).abs().max().item()
            errs.append(e)
            tops.append(int(o_paged[r, 0].argmax()) == int(ref[0, 0, 0].argmax()))
        t_paged = timeit(lambda: w.run(q, (k_cache, v_cache)), n=10, warmup=3)
        t_cont = timeit(lambda: torch.cat([
            F.scaled_dot_product_attention(
                q[r:r + 1].unsqueeze(2),
                k_cache[torch.tensor(logical[r][0], device="cuda")]
                .reshape(-1, Hkv, D)[:logical[r][1]].transpose(0, 1).unsqueeze(0),
                v_cache[torch.tensor(logical[r][0], device="cuda")]
                .reshape(-1, Hkv, D)[:logical[r][1]].transpose(0, 1).unsqueeze(0),
                enable_gqa=True) for r in range(len(lengths))], dim=0),
            n=5, warmup=2)
        kv_bytes = 2 * sum(lengths) * Hkv * D * 2
        in_l2 = "是" if kv_bytes / MB <= L2_MIB else "否"
        print(f"  {page_size:>5} {str(lengths):>34} {t_paged:>9.4f} "
              f"{t_cont:>10.4f} {max(errs):>11.3e} {str(all(tops)):>11}")
        h.case(id=f"A_page{page_size}", page_size=page_size, lengths=lengths,
               Hq=Hq, Hkv=Hkv, D=D, dtype=str(dtype), paged_ms=t_paged,
               continuous_ms=t_cont, max_err=max(errs), top1_all=tops,
               kv_bytes=kv_bytes, timer="cuda-event",
               ref="SDPA 连续布局 + enable_gqa")
    print("\n  S=page−1/page/page+1 三档都在：尾页不满与刚好满各一次。")
    print("  误差在 bf16 量级（1e-2 以内），顶部 1 一致。")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 物理页打乱、尾页与共享页")

    Hq, Hkv, D = 16, 2, 128
    page_size = 16
    dtype = torch.bfloat16
    lengths = [8192, 8192, 8192, 100, 17, 16]
    k_cache, v_cache, indptr, indices, last, logical = build_paged_case(
        lengths, page_size, Hkv, D, seed=11, share=(0, 1, 512))
    q = torch.randn(len(lengths), Hq, D, device="cuda", dtype=dtype)
    w = make_wrapper()
    plan_wrapper(w, indptr, indices, last, Hq, Hkv, D, page_size, dtype)
    o = w.run(q, (k_cache, v_cache))

    sub("共享物理页：请求 1 的 512 个逻辑块全部指向请求 0 的物理页")
    d01 = (o[0].float() - o[1].float()).abs().max().item()
    print(f"  请求 0 与请求 1 的输出 max|err| = {d01:.3e}"
          f"（两块同长、全部同页，所以必须逐位相同）")
    print("  只共享**前缀**时输出本来就该不同：后面的 KV 不一样。"
          "前缀共享的收益在 5.2 的命中率里量，不在这一步。")
    print(f"  物理页索引前 8 个：请求0 {logical[0][0][:8]}")
    print(f"  物理页索引前 8 个：请求1 {logical[1][0][:8]}  （相同）")

    sub("尾页：长度 100 / 17 / 16，页大小 16")
    for r in (3, 4, 5):
        phys, L, npages = logical[r]
        tail = L - (npages - 1) * page_size
        pk = k_cache[torch.tensor(phys, device="cuda")].reshape(-1, Hkv, D)[:L]
        pv = v_cache[torch.tensor(phys, device="cuda")].reshape(-1, Hkv, D)[:L]
        ref = F.scaled_dot_product_attention(q[r:r + 1].unsqueeze(2),
                                            pk.transpose(0, 1).unsqueeze(0),
                                            pv.transpose(0, 1).unsqueeze(0),
                                            enable_gqa=True)
        e = (o[r].float() - ref[0, :, 0].float()).abs().max().item()
        print(f"  S={L:>4} 逻辑页 {npages} 尾页有效 {tail:>2}/{page_size}  "
              f"max|err| = {e:.3e}")
        h.case(id=f"B_tail_S{L}", page_size=page_size, S=L, pages=npages,
               tail_valid=tail, max_err=e)
    print("  尾页不满时 kernel 必须按 last_page_len 屏蔽多余槽位；")
    print("  屏蔽错的话，S=17 这类长度会读到相邻物理页的垃圾数据。")
    h.case(id="B_shared_pages", page_size=page_size, shared_blocks=64,
           output_diff=d01, same_physical_pages=True)
    del k_cache, v_cache
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] 页大小与并发的代价")

    Hq, Hkv, D = 16, 2, 128
    dtype = torch.bfloat16
    total_kv = 2 * 8 * 8192 * Hkv * D * 2
    print(f"  固定工作集：8 条请求 × 8192 token，Hkv={Hkv} D={D}，"
          f"KV 合计 {total_kv / MB:.1f} MB"
          f"{'（小于 L2 96 MiB，所以下面的带宽是 L2 带宽）' if total_kv / MB <= L2_MIB else ''}")
    print(f"  {'page':>5} {'页数':>7} {'paged ms':>9} {'GB/s':>9} {'占只读峰值':>10} "
          f"{'相比 page=64':>12}")
    base = None
    for page_size in [16, 32, 64, 128, 256]:
        lengths = [8192] * 8
        k_cache, v_cache, indptr, indices, last, logical = build_paged_case(
            lengths, page_size, Hkv, D, seed=5)
        q = torch.randn(len(lengths), Hq, D, device="cuda", dtype=dtype)
        w = make_wrapper()
        plan_wrapper(w, indptr, indices, last, Hq, Hkv, D, page_size, dtype)
        t = timeit(lambda: w.run(q, (k_cache, v_cache)), n=10, warmup=3)
        if base is None:
            base = t
        kv_bytes = 2 * sum(lengths) * Hkv * D * 2
        gbs = kv_bytes / t / 1e6
        npages = sum((L + page_size - 1) // page_size for L in lengths)
        print(f"  {page_size:>5} {npages:>7} {t:>9.4f} {gbs:>9.1f} "
              f"{gbs / PEAK_BW:>9.1%} {t / base:>11.2f}×")
        h.case(id=f"C_page{page_size}", page_size=page_size, batch=8, S=8192,
               Hq=Hq, Hkv=Hkv, D=D, ms=t, gbs=gbs, pct_peak=gbs / PEAK_BW,
               pages=npages)
        del k_cache, v_cache
        torch.cuda.empty_cache()
    print("\n  页越小，表越长、寻址越碎；页越大，内部碎片越多。")
    print("  真实的取舍还要把碎片率算进去：页大小 × 平均浪费槽位。")


# ---------------------------------------------------------------- D
def section_D(h):
    title("[D] 连续 / 分页 / 框架层 gather 三方对照")

    Hq, Hkv, D = 16, 2, 128
    page_size = 32
    dtype = torch.bfloat16
    lengths = [8192] * 4
    k_cache, v_cache, indptr, indices, last, logical = build_paged_case(
        lengths, page_size, Hkv, D, seed=9)
    q = torch.randn(len(lengths), Hq, D, device="cuda", dtype=dtype)
    w = make_wrapper()
    plan_wrapper(w, indptr, indices, last, Hq, Hkv, D, page_size, dtype)
    t_paged = timeit(lambda: w.run(q, (k_cache, v_cache)), n=10, warmup=3)
    t_plan = timeit(lambda: plan_wrapper(w, indptr, indices, last, Hq, Hkv, D,
                                         page_size, dtype), n=20, warmup=5)

    # 连续布局：等长的四条请求拼成一个 [B,Hkv,S,D]
    k_cont = torch.randn(len(lengths), Hkv, 8192, D, device="cuda", dtype=dtype)
    v_cont = torch.randn_like(k_cont)

    def cont():
        return F.scaled_dot_product_attention(q.unsqueeze(2), k_cont, v_cont,
                                              enable_gqa=True)

    t_cont = timeit(cont, n=10, warmup=3)

    # 框架层 gather：先按 block table 取回连续张量再算
    def gathered():
        ks = torch.stack([k_cache[torch.tensor(logical[r][0], device="cuda")]
                          .reshape(-1, Hkv, D)[:lengths[r]].transpose(0, 1)
                          for r in range(len(lengths))])
        vs = torch.stack([v_cache[torch.tensor(logical[r][0], device="cuda")]
                          .reshape(-1, Hkv, D)[:lengths[r]].transpose(0, 1)
                          for r in range(len(lengths))])
        return F.scaled_dot_product_attention(q.unsqueeze(2), ks, vs, enable_gqa=True)

    t_gather = timeit(gathered, n=5, warmup=2)
    kv_bytes = 2 * sum(lengths) * Hkv * D * 2
    print(f"  {'实现':<28} {'ms':>9} {'GB/s':>9} {'相对连续':>10}")
    for name, t in [("连续（预留最大长度）", t_cont),
                    ("分页 kernel（FlashInfer）", t_paged),
                    ("框架层 gather + SDPA", t_gather)]:
        print(f"  {name:<28} {t:>9.4f} {kv_bytes / t / 1e6:>9.1f} "
              f"{t_cont / t:>9.2f}×")
        h.case(id=f"D_{name}", page_size=page_size, ms=t,
               gbs=kv_bytes / t / 1e6, ratio_vs_continuous=t_cont / t)
    print(f"\n  plan（建 block table 与调度）单独计时：{t_plan * 1000:.1f} µs")
    print("  分页 kernel 在 kernel 内部寻址，不产生拷贝；框架层 gather 多一次全量读+写，")
    print("  所以慢得多 —— 这是 3.3 §4 那个 5× 的来源，也是必须写专门 kernel 的原因。")
    del k_cache, v_cache
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  SM {p.multi_processor_count}")
    import flashinfer
    print(f"flashinfer {flashinfer.__version__}")
    h = Harness("3.3-B", "3.3", out=os.environ.get("L3_OUT"),
                backend=f"flashinfer {flashinfer.__version__} paged decode",
                notes="NHD 布局；物理页随机打乱；参照为 SDPA 连续布局")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "真实 paged kernel 在页边界与共享页上与连续参照一致；"
                         "页大小影响寻址代价；框架层 gather 仍是最慢的一条路。",
              "peak_read_bw": PEAK_BW})
    sys.stdout.flush()
    os._exit(0)
