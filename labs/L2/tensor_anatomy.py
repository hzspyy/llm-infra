#!/usr/bin/env python3
"""L2.0 —— tensor 不是数组：storage / offset / stride / view / dispatcher。

把 PyTorch 的 tensor 拆到字节一级，再跟一次 `a + b` 走完分发。
每一节都只打印，不做任何"结论"，结论留给正文。

用法：
    python tensor_anatomy.py            # 全跑
    python tensor_anatomy.py A C F      # 只跑指定节
"""

import ctypes
import itertools
import os
import sys

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def title(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 72 - len(s)))


def raw_bytes(storage):
    """直接按地址读进程内存 —— storage 就是一段裸字节。"""
    n = storage.nbytes()
    buf = (ctypes.c_ubyte * n).from_address(storage.data_ptr())
    return bytes(buf)


def hexdump(b, per_line=16, base=0):
    for i in range(0, len(b), per_line):
        chunk = b[i:i + per_line]
        hexs = " ".join(f"{c:02x}" for c in chunk)
        print(f"  {base + i:04x}  {hexs}")


def view_row(name, t, base_storage_ptr):
    same = "是" if t.untyped_storage().data_ptr() == base_storage_ptr else "否"
    return (
        f"{name:<30} {str(tuple(t.shape)):<14} {str(t.stride()):<14}"
        f" {t.storage_offset():>4} {t.data_ptr():>#16x} {same:^6}"
        f" {'是' if t.is_contiguous() else '否':^6}"
    )


def view_table(pairs, base):
    bp = base.untyped_storage().data_ptr()
    print(
        f"{'表达式':<28} {'shape':<14} {'stride':<14} {'off':>4}"
        f" {'data_ptr':>16} {'同源':^5} {'连续':^5}"
    )
    print("-" * 100)
    for name, t in pairs:
        print(view_row(name, t, bp))


def byte_map(t, name, max_rows=24):
    """逐个逻辑索引 -> storage 元素号 -> 字节区间 -> 实际值。"""
    es = t.element_size()
    off = t.storage_offset()
    st = t.stride()
    storage = t.untyped_storage()
    raw = raw_bytes(storage)
    print(f"{name}   shape={tuple(t.shape)} stride={st} offset={off} "
          f"itemsize={es}B storage={storage.nbytes()}B")
    print(f"  {'逻辑索引':<12} {'元素号':>6} {'字节区间':>12}  {'原始字节':<12} {'值'}")
    n = 0
    for idx in itertools.product(*[range(s) for s in t.shape]):
        elem = off + sum(i * s for i, s in zip(idx, st))
        b0 = elem * es
        chunk = raw[b0:b0 + es]
        val = t[idx].item()
        print(f"  {str(list(idx)):<12} {elem:>6} {b0:>5}..{b0 + es - 1:<6}"
              f" {' '.join(f'{c:02x}' for c in chunk):<12} {val}")
        n += 1
        if n >= max_rows:
            print(f"  ... (共 {t.numel()} 个)")
            break


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 一块 storage，很多个 view")

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x =")
    print(x)
    print()
    print(f"x.untyped_storage().nbytes() = {x.untyped_storage().nbytes()}  "
          f"(12 个 float32 = 48 字节)")
    print(f"x.untyped_storage().data_ptr() = {x.untyped_storage().data_ptr():#x}")
    print()
    print("storage 的 48 字节（小端 float32）：")
    hexdump(raw_bytes(x.untyped_storage()))

    sub("同一块 storage 上的 8 个 view")
    pairs = [
        ("x", x),
        ("x.t()", x.t()),
        ("x[1]", x[1]),
        ("x[:, 1]", x[:, 1]),
        ("x[1:, 2:]", x[1:, 2:]),
        ("x.view(2, 6)", x.view(2, 6)),
        ("x[:, :1].expand(3, 4)", x[:, :1].expand(3, 4)),
        ("x.flip(0)   <-- 不是 view", x.flip(0)),
    ]
    view_table(pairs, x)
    print()
    print("注意 x[1] 的 data_ptr 比 x 大 16 字节 = storage_offset(4) × itemsize(4)。")
    print("flip 的 storage 指针不同 —— 它真的拷贝了。")

    sub("x.t() 到底读了哪些字节")
    byte_map(x.t(), "x.t()")

    sub("x[1:, 2:] 到底读了哪些字节")
    byte_map(x[1:, 2:], "x[1:, 2:]")

    sub("改一个 view，其它 view 全变")
    y = x.t()
    print(f"改之前  x[0,0]={x[0,0].item()}  y[0,0]={y[0,0].item()}")
    y[0, 0] = 999.0
    print(f"y[0,0]=999 之后  x[0,0]={x[0,0].item()}")
    print("storage 前 4 字节：", " ".join(f"{c:02x}" for c in raw_bytes(x.untyped_storage())[:4]))
    y[0, 0] = 0.0


# ---------------------------------------------------------------- B
def section_B():
    title("[B] stride = 0：广播不是复制")

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    e = x[:, :1].expand(3, 4)
    print("x[:, :1] 是一个 3×1 的列：")
    print(x[:, :1])
    print("\n.expand(3,4) 之后逻辑上是 3×4：")
    print(e)
    print(f"\nstride = {e.stride()}   <-- 第 1 维步长为 0")
    print(f"storage 还是 {e.untyped_storage().nbytes()} 字节，一个字节都没有多。")

    sub("同一个字节被读了 4 次")
    byte_map(e, "x[:, :1].expand(3, 4)", max_rows=12)

    sub("expand vs repeat")
    r = x[:, :1].repeat(1, 4)
    print(f"expand: storage={e.untyped_storage().nbytes():>4}B  stride={e.stride()}  "
          f"contiguous={e.is_contiguous()}")
    print(f"repeat: storage={r.untyped_storage().nbytes():>4}B  stride={r.stride()}  "
          f"contiguous={r.is_contiguous()}")
    print("值完全一样：", torch.equal(e, r))

    sub("a + b 的广播就是给 b 装了个 stride=0")
    a = torch.randn(4, 3, device=DEV)
    b = torch.randn(1, 3, device=DEV)
    bb = b.expand(4, 3)
    print(f"b  shape={tuple(b.shape)} stride={b.stride()}")
    print(f"b.expand(4,3) stride={bb.stride()}  (广播时 TensorIterator 内部就是这么干的)")
    print("a+b 与 a+b.expand(4,3) 相等：", torch.equal(a + b, a + bb))


# ---------------------------------------------------------------- C
def section_C():
    title("[C] as_strided：手工造一个 view")

    v = torch.arange(10, dtype=torch.float32)
    print("v =", v.tolist())
    print(f"storage = {v.untyped_storage().nbytes()} 字节")

    sub("滑动窗口：size=(8,3) stride=(1,1)，零拷贝")
    w = torch.as_strided(v, size=(8, 3), stride=(1, 1))
    print(w)
    print(f"\nstorage 指针相同：{w.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()}")
    print(f"逻辑上 8×3=24 个元素，实际只有 {v.numel()} 个 —— 每个元素平均被读 {24 / v.numel():.1f} 次")
    byte_map(w, "sliding window", max_rows=9)

    sub("拿它算 3 点滑动平均（一次 matmul，没有 im2col 拷贝）")
    k = torch.full((3,), 1 / 3)
    print((w @ k).tolist())

    sub("as_strided 会越过自己的边界")
    head = v[:4]
    print(f"head = v[:4] = {head.tolist()}   （它的 shape 只有 4）")
    over = torch.as_strided(head, size=(10,), stride=(1,))
    print(f"as_strided(head, (10,), (1,)) = {over.tolist()}")
    print("后面 6 个值不属于 head，但属于同一块 storage —— as_strided 只认 storage，不认 shape。")
    print("这就是 as_strided 危险的地方：它绕过了所有边界检查。")


# ---------------------------------------------------------------- D
def section_D():
    title("[D] contiguous：什么时候真的拷贝")

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    sub("已经连续时 contiguous() 是空操作")
    print(f"x.is_contiguous() = {x.is_contiguous()}")
    print(f"x.contiguous() is x  ->  {x.contiguous() is x}")
    print(f"data_ptr 相同        ->  {x.contiguous().data_ptr() == x.data_ptr()}")

    sub("不连续时真的分配 + 拷贝")
    t = x.t()
    tc = t.contiguous()
    print(f"x.t().is_contiguous()      = {t.is_contiguous()}")
    print(f"x.t().data_ptr()           = {t.data_ptr():#x}")
    print(f"x.t().contiguous().data_ptr() = {tc.data_ptr():#x}   <-- 变了")
    print(f"新 storage 大小 = {tc.untyped_storage().nbytes()} 字节")
    print("新 storage 的字节序（转置后的顺序）：")
    hexdump(raw_bytes(tc.untyped_storage()))

    sub("view 会拒绝，reshape 会偷偷拷贝")
    try:
        t.view(-1)
    except RuntimeError as exc:
        print("x.t().view(-1) ->")
        for line in str(exc).splitlines():
            print("   ", line)
    rs = t.reshape(-1)
    print(f"\nx.t().reshape(-1) 成功，data_ptr 变了：{rs.data_ptr() != t.data_ptr()}")
    print("reshape 在能做 view 时做 view，不能时 **静默拷贝**。这是隐藏显存与带宽开销的常见来源。")

    sub("memory_format：同样的 shape，不同的字节顺序")
    a = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    c = a.to(memory_format=torch.channels_last)
    print(f"contiguous_format  shape={tuple(a.shape)} stride={a.stride()}")
    print(f"channels_last      shape={tuple(c.shape)} stride={c.stride()}")
    print(f"值相等：{torch.equal(a, c)}   a.is_contiguous()={a.is_contiguous()}   "
          f"c.is_contiguous()={c.is_contiguous()}")
    print(f"c.is_contiguous(memory_format=channels_last) = "
          f"{c.is_contiguous(memory_format=torch.channels_last)}")
    print("\n前 12 个 float 在 storage 里的实际顺序：")
    print("  NCHW :", [f"{v:.0f}" for v in
                       torch.frombuffer(raw_bytes(a.untyped_storage()), dtype=torch.float32)[:12].tolist()])
    print("  NHWC :", [f"{v:.0f}" for v in
                       torch.frombuffer(raw_bytes(c.untyped_storage()), dtype=torch.float32)[:12].tolist()])
    print("同一个 tensor，同一批数值，字节顺序完全不同。'contiguous' 是相对于某个顺序而言的。")

    if DEV != "cuda":
        return
    sub("转置拷贝有多贵（8192×8192 bf16，crater）")
    g = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    nbytes = g.numel() * g.element_size()

    def timeit(fn, n=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    t_copy = timeit(lambda: g.contiguous().clone())
    t_tcopy = timeit(lambda: g.t().contiguous())
    print(f"  连续拷贝  g.clone()          {t_copy:7.3f} ms  "
          f"{2 * nbytes / t_copy / 1e6:8.1f} GB/s")
    print(f"  转置拷贝  g.t().contiguous() {t_tcopy:7.3f} ms  "
          f"{2 * nbytes / t_tcopy / 1e6:8.1f} GB/s   ({t_tcopy / t_copy:.2f}×)")

    sub("但逐元素算子对转置几乎免疫（TensorIterator 会重排维度）")
    t_mul = timeit(lambda: g * 2)
    t_tmul = timeit(lambda: g.t() * 2)
    print(f"  g * 2        {t_mul:7.3f} ms  {2 * nbytes / t_mul / 1e6:8.1f} GB/s")
    print(f"  g.t() * 2    {t_tmul:7.3f} ms  {2 * nbytes / t_tmul / 1e6:8.1f} GB/s   "
          f"({t_tmul / t_mul:.2f}×)")
    print(f"  g.t()*2 的输出 stride = {(g.t() * 2).stride()}   "
          f"（输出继承了输入的物理布局，不是行主序）")
    del g
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- E
def section_E():
    title("[E] 失败现场：把非 contiguous 张量交给一个自写 kernel")
    if DEV != "cuda":
        print("需要 CUDA，跳过。")
        return

    from torch.utils.cpp_extension import load_inline

    src = r"""
// 假设内存连续 —— 绝大多数教程里的写法
__global__ void naive_scale(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// 认 stride 的版本：把线性 id 拆回二维逻辑索引，再用 stride 算元素号
__global__ void strided_scale(const float* __restrict__ in,
                              float* __restrict__ out,
                              int rows, int cols, long s0, long s1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    out[i] = in[r * s0 + c * s1] * 2.0f;
}

torch::Tensor naive(torch::Tensor x) {
    auto out = torch::empty({x.size(0), x.size(1)}, x.options());
    int n = x.numel();
    naive_scale<<<(n + 255) / 256, 256>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

torch::Tensor strided(torch::Tensor x) {
    auto out = torch::empty({x.size(0), x.size(1)}, x.options());
    int n = x.numel();
    strided_scale<<<(n + 255) / 256, 256>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        x.size(0), x.size(1), x.stride(0), x.stride(1));
    return out;
}
"""
    cpp = ("torch::Tensor naive(torch::Tensor x);\n"
           "torch::Tensor strided(torch::Tensor x);\n")
    build = os.environ.get("LEARN_ROOT", "/tmp") + "/.cache/torchext/l20"
    os.makedirs(build, exist_ok=True)
    mod = load_inline(
        name="l20_scale", cpp_sources=cpp, cuda_sources=src,
        functions=["naive", "strided"], build_directory=build, verbose=False,
    )

    x = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    xt = x.t()
    want = xt * 2

    sub("输入连续时：对的")
    print("naive(x) =\n", mod.naive(x))
    print("max |err| =", (mod.naive(x) - x * 2).abs().max().item())

    sub("输入是 x.t() 时：错的，而且不报错")
    got = mod.naive(xt)
    print("期望 xt*2 =\n", want)
    print("kernel 给出 =\n", got)
    print("max |err| =", (got - want).abs().max().item())
    print("\nkernel 按 storage 顺序读了 0,1,2,...,11，再按行主序写出去 ——")
    print("它拿到的是 x*2 被塞进 4×3 的形状，转置那一步完全没发生。")
    print(f"xt.data_ptr()==x.data_ptr(): {xt.data_ptr() == x.data_ptr()}   "
          f"kernel 只拿到一个指针，它看不见 stride={xt.stride()}。")

    sub("两种修法")
    fix1 = mod.naive(xt.contiguous())
    fix2 = mod.strided(xt)
    print(f"1) .contiguous() 后再喂：max|err| = {(fix1 - want).abs().max().item()}   （多一次全量拷贝）")
    print(f"2) kernel 自己认 stride：max|err| = {(fix2 - want).abs().max().item()}   （零拷贝，但访存不合并）")

    sub("两种修法的代价 —— 两个尺寸，一个在 L2 以内，一个在 L2 以外")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    print(f"  本卡 L2 = {l2 / 1024 / 1024:.0f} MiB，实测 DRAM 上限 1519 GB/s（见 L1.1）")

    def timeit(fn, n=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    for n in (4096, 8192):
        big = torch.randn(n, n, device="cuda")
        bt = big.t()
        nb = big.numel() * 4
        print(f"\n  == {n}×{n} float32 = {nb / 1024 / 1024:.0f} MiB "
              f"({'放得进' if nb <= l2 else '放不进'} L2) ==")
        t0 = timeit(lambda: mod.naive(big))
        t1 = timeit(lambda: mod.naive(bt.contiguous()))
        t2 = timeit(lambda: mod.strided(bt))
        t3 = timeit(lambda: bt * 2)
        print(f"  naive(连续输入)          {t0:7.3f} ms   "
              f"{2 * nb / t0 / 1e6:8.1f} GB/s")
        print(f"  contiguous() + naive     {t1:7.3f} ms   ({t1 / t0:.2f}× 基准)  多读写一遍全量")
        print(f"  strided kernel           {t2:7.3f} ms   ({t2 / t0:.2f}× 基准)  读不合并")
        print(f"  PyTorch 自己的 bt * 2    {t3:7.3f} ms   ({t3 / t0:.2f}× 基准)")
        del big, bt
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- F
def section_F():
    title("[F] dispatcher：a + b 到底经过几跳")

    sub("每个 tensor 自带一个 DispatchKeySet")
    cases = [
        ("torch.randn(3)                 ", torch.randn(3)),
        ("torch.randn(3, device='cuda')  ", torch.randn(3, device=DEV)),
        ("... .requires_grad_(True)      ", torch.randn(3, device=DEV).requires_grad_(True)),
        ("torch.randn(3, device='meta')  ", torch.randn(3, device="meta")),
        ("torch.randn(3).to_sparse()     ", torch.randn(3).to_sparse()),
        ("torch.randn(3, dtype=torch.bfloat16)", torch.randn(3, dtype=torch.bfloat16)),
    ]
    for name, t in cases:
        print(f"{name}  {torch._C._dispatch_key_set(t)}")
    print()
    print("注意 requires_grad 并没有改变 key set —— AutogradCUDA 一直都在。")
    print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
    with torch.no_grad():
        print(f"no_grad 内: is_grad_enabled={torch.is_grad_enabled()}")
        print(f"no_grad 内 exclude set = {torch._C._dispatch_tls_local_exclude_set()}")
    print(f"no_grad 外 exclude set = {torch._C._dispatch_tls_local_exclude_set()}")
    print("no_grad 翻的是 GradMode 这个 TLS 布尔，不是 dispatcher 的 exclude set。")

    sub("aten::add.Tensor 的分发表（节选）")
    table = torch._C._dispatch_dump_table("aten::add.Tensor")
    keep = ("CPU:", "CUDA:", "Meta:", "Autograd:", "ADInplaceOrView:",
            "AutocastCUDA:", "Functionalize:", "SparseCPU:", "SparseCUDA:",
            "NestedTensorCUDA:", "BackendSelect:", "Python:", "FuncTorchBatched:")
    for line in table.splitlines():
        if line.startswith(keep):
            print("  " + line)
    print(f"\n完整表共 {len(table.splitlines())} 行（所有后端 × 所有功能键）。")

    sub("同一个算子的注册来源")
    print(torch._C._dispatch_dump("aten::add.Tensor"))

    sub("每个 key 上注册了多少个算子")
    for k in ["CPU", "CUDA", "Meta", "SparseCUDA", "QuantizedCUDA",
              "NestedTensorCUDA", "CompositeImplicitAutograd",
              "CompositeExplicitAutograd", "Autograd", "ADInplaceOrView"]:
        try:
            n = len(torch._C._dispatch_get_registrations_for_dispatch_key(k))
            print(f"  {k:<28} {n:>6}")
        except Exception as exc:
            print(f"  {k:<28}  ERR {exc}")

    sub("两层 mode：python 层看到什么 / aten 层看到什么")
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class FnLog(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__qualname__", str(func))
            print(f"    [torch_function] {name}")
            return func(*args, **(kwargs or {}))

    class DispLog(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            shapes = [tuple(a.shape) if isinstance(a, torch.Tensor) else a
                      for a in args]
            print(f"        [torch_dispatch] {func}  {shapes}")
            return func(*args, **(kwargs or {}))

    a = torch.randn(2, 3, device=DEV)
    w = torch.randn(4, 3, device=DEV)
    b = torch.randn(4, device=DEV)

    for label, fn in [
        ("F.linear(a, w, b)", lambda: torch.nn.functional.linear(a, w, b)),
        ("a + a", lambda: a + a),
        ("F.softmax(a, -1)", lambda: torch.nn.functional.softmax(a, -1)),
        ("F.layer_norm(a, (3,))", lambda: torch.nn.functional.layer_norm(a, (3,))),
        ("F.gelu(a)", lambda: torch.nn.functional.gelu(a)),
        ("a.t().contiguous()", lambda: a.t().contiguous()),
    ]:
        print(f"\n  {label}")
        with FnLog(), DispLog():
            fn()

    sub("反向也在同一套分发上")
    w2 = w.clone().requires_grad_(True)
    print("  F.linear(a, w2).sum().backward()")
    with DispLog():
        torch.nn.functional.linear(a, w2).sum().backward()

    sub("grad_fn 链")
    w3 = w.clone().requires_grad_(True)
    out = torch.nn.functional.linear(a, w3).sum()
    node = out.grad_fn
    depth = 0
    while node is not None and depth < 8:
        print("  " * (depth + 1) + str(node))
        nxt = [n for n, _ in node.next_functions if n is not None]
        node = nxt[0] if nxt else None
        depth += 1


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E, "F": section_F}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {DEV}")
    if DEV == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}")
    for s in want:
        SECTIONS[s]()
