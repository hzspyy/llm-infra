#!/usr/bin/env python3
"""L2.8 —— 把一个自写 CUDA kernel 接进 PyTorch。

算子：out[i] = sum_j relu(x[i,j] * w[j])
反向：dx[i,j] = g[i] * w[j] * 1[x*w>0]
      dw[j]   = sum_i g[i] * x[i,j] * 1[x*w>0]

一路走完：裸函数 -> torch.library 注册 -> fake（meta）-> autograd -> torch.compile。
每一步都打印 dispatch 表 / 图 / gradcheck 结果，看清楚少了哪一步会怎样。

用法：
    python custom_op.py            # 全跑
    python custom_op.py C E
"""

import os
import sys

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIB = "l28"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


# ---------------------------------------------------------------- kernel
CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

template <typename scalar_t>
__global__ void fss_fwd(const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ w,
                        scalar_t* __restrict__ out, int R, int C) {
    int row = blockIdx.x;
    if (row >= R) return;
    // 一个 block 归约一行；这是 L2.3 那个两段式归约的最小版本
    extern __shared__ unsigned char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t acc = 0;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        scalar_t v = x[row * C + j] * w[j];
        acc += v > scalar_t(0) ? v : scalar_t(0);
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = smem[0];
}

template <typename scalar_t>
__global__ void fss_bwd(const scalar_t* __restrict__ g,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ w,
                        scalar_t* __restrict__ gx,
                        scalar_t* __restrict__ gw, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    int i = idx / C, j = idx % C;
    scalar_t v = x[idx] * w[j];
    scalar_t mask = v > scalar_t(0) ? scalar_t(1) : scalar_t(0);
    gx[idx] = g[i] * w[j] * mask;
    atomicAdd(&gw[j], g[i] * x[idx] * mask);
}

at::Tensor fss_forward(at::Tensor x, at::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "需要 CUDA 张量");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
                "这个 kernel 假设连续内存 —— 见 L2.0 [E] 节");
    int R = x.size(0), C = x.size(1);
    auto out = at::empty({R}, x.options());
    int threads = 256;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fss_fwd", [&] {
        fss_fwd<scalar_t><<<R, threads, threads * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), R, C);
    });
    return out;
}

std::vector<at::Tensor> fss_backward(at::Tensor g, at::Tensor x, at::Tensor w) {
    int R = x.size(0), C = x.size(1);
    auto gx = at::empty_like(x);
    auto gw = at::zeros_like(w);
    int n = R * C, threads = 256;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fss_bwd", [&] {
        fss_bwd<scalar_t><<<(n + threads - 1) / threads, threads>>>(
            g.contiguous().data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(), gx.data_ptr<scalar_t>(),
            gw.data_ptr<scalar_t>(), R, C);
    });
    return {gx, gw};
}
"""

CPP_SRC = ("at::Tensor fss_forward(at::Tensor x, at::Tensor w);\n"
           "std::vector<at::Tensor> fss_backward(at::Tensor g, at::Tensor x, at::Tensor w);\n")

_mod = None


def kernel():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        build = os.environ.get("LEARN_ROOT", "/tmp") + "/.cache/torchext/l28"
        os.makedirs(build, exist_ok=True)
        _mod = load_inline(name="l28_fss", cpp_sources=CPP_SRC,
                           cuda_sources=CUDA_SRC,
                           functions=["fss_forward", "fss_backward"],
                           build_directory=build, verbose=False)
    return _mod


def ref(x, w):
    """纯 PyTorch 参照实现，用来对答案。"""
    return torch.relu(x * w).sum(dim=-1)


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 起点：一个能跑、但 PyTorch 完全不认识的 kernel")
    m = kernel()
    x = torch.randn(64, 512, device=DEV, dtype=torch.float64)
    w = torch.randn(512, device=DEV, dtype=torch.float64)
    got, want = m.fss_forward(x, w), ref(x, w)
    print(f"  前向对不对: max|err| = {(got - want).abs().max().item():.3e}")
    print(f"  返回类型: {type(got)}  shape={tuple(got.shape)}")

    sub("它在 dispatcher 里存在吗")
    # 注意：_dispatch_dump 对不存在的算子返回空串，不抛异常
    for name in ["l28::fused_scale_sum", "aten::add.Tensor"]:
        print(f"  {name:<28} "
              f"{'存在' if torch._C._dispatch_dump(name) else '**不存在**'}")
    print("  它只是一个 pybind 出来的 python 可调用对象，dispatcher 里没有它。")

    sub("requires_grad 会怎样")
    xg = x.detach().clone().requires_grad_(True)
    out = m.fss_forward(xg, w)
    print(f"  out.requires_grad = {out.requires_grad}   grad_fn = {out.grad_fn}")
    print("  没有反向。autograd 根本不知道发生过这次计算。")

    sub("torch.compile 遇到它会怎样 —— 这里有个陷阱")
    import torch._dynamo as dynamo
    def f(a, b):
        return m.fss_forward(a, b) * 2

    dynamo.reset()
    exp = dynamo.explain(f)(x, w)
    print(f"  explain(): 图数量 {exp.graph_count}   图断裂 {exp.graph_break_count}"
          f"   捕获算子 {exp.op_count}")
    print("  看起来一切正常。但把图打印出来：")

    gms = []
    def cb(gm, ex):
        gms.append(gm); return gm.forward
    dynamo.reset()
    torch.compile(f, backend=cb)(x, w)
    for g in gms:
        for line in g.code.strip().splitlines():
            print("   ", line)
    print("\n  **kernel 调用根本不在图里。** 图的入参 L_stack0_ 是"
          "已经算好的结果，")
    print("  只有后面那个 `* 2` 被捕获了。Dynamo 在图外 eager 执行了这次调用。")

    print("\n  用 fullgraph=True 才会说实话：")
    dynamo.reset()
    try:
        torch.compile(f, fullgraph=True)(x, w)
        print("  竟然成功了")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    print("\n  教训：explain() 报 0 次断裂**不等于**全部captured。")
    print("  想确认有没有东西被漏在图外，用 fullgraph=True，或者把图打出来看。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 注册成一个真正的算子")
    m = kernel()

    @torch.library.custom_op(f"{LIB}::fused_scale_sum", mutates_args=())
    def fused_scale_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return m.fss_forward(x, w)

    print("  注册完成。现在它在 dispatcher 里：")
    dump = torch._C._dispatch_dump(f"{LIB}::fused_scale_sum")
    for line in dump.splitlines()[:12]:
        print("   ", line)

    sub("它出现在哪些 dispatch key 上")
    for k in ["CUDA", "CPU", "Meta", "Autograd", "CompositeExplicitAutograd"]:
        has = torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{LIB}::fused_scale_sum", k)
        print(f"  {k:<28} {has}")
    print("\n  注意 CUDA 和 CPU 都是 False，而 CompositeExplicitAutograd 是 True：")
    print("  `torch.library.custom_op` 默认注册的是**与后端无关**的实现，")
    print("  由 CompositeExplicitAutograd 这个别名键覆盖所有后端（2.0 §3.3 那张表）。")
    print("  想按设备分别注册，用 `custom_op(..., device_types='cuda')`"
          " 或 `register_kernel`。")
    print("\n  Meta 和 Autograd 这两行虽然是 True，但注册的是**占位实现**：")
    print("  它们会在被调用时报错，提示你还没写 fake / autograd。")
    print("  这正是接下来两节要补的两件事。")
    return fused_scale_sum


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 少了 fake（meta）函数会怎样")
    import torch._dynamo as dynamo
    m = kernel()
    name = f"{LIB}::no_fake"

    @torch.library.custom_op(name, mutates_args=())
    def no_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return m.fss_forward(x, w)

    x = torch.randn(64, 512, device=DEV)
    w = torch.randn(512, device=DEV)
    print("  eager 调用没问题:", no_fake(x, w).shape)

    sub("但 torch.compile 需要在不真的执行的情况下推出形状")
    dynamo.reset()
    try:
        cf = torch.compile(lambda a, b: no_fake(a, b) * 2, fullgraph=True)
        cf(x, w)
        print("  竟然成功了")
    except Exception as exc:                                  # noqa: BLE001
        print("  失败:", type(exc).__name__)
        for line in str(exc).splitlines()[:8]:
            print("   ", line[:150])

    sub("补上 fake 之后")
    @no_fake.register_fake
    def _(x, w):
        # 只描述形状与 dtype，不做任何真实计算 —— 这就是 meta 的全部工作
        torch._check(x.dim() == 2, lambda: "x 必须是二维")
        torch._check(w.dim() == 1, lambda: "w 必须是一维")
        return x.new_empty(x.shape[0])

    dynamo.reset()
    cf = torch.compile(lambda a, b: no_fake(a, b) * 2, fullgraph=True)
    out = cf(x, w)
    print(f"  成功，shape={tuple(out.shape)}")
    print(f"  Meta key 现在有 kernel 了: "
          f"{torch._C._dispatch_has_kernel_for_dispatch_key(name, 'Meta')}")

    sub("编译后的图里，这个算子长什么样")
    gms = []
    def collect(gm, ex):
        gms.append(gm)
        return gm.forward
    dynamo.reset()
    torch.compile(lambda a, b: no_fake(a, b) * 2, backend=collect)(x, w)
    if gms:
        print(gms[0].code.strip())
    print("\n  它是图里的一个**不透明节点**：Inductor 知道它的形状，")
    print("  知道要在这里调用它，但不知道它内部在算什么。")


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 接上反向，并用 gradcheck 验一遍")
    m = kernel()
    name = f"{LIB}::with_grad"

    @torch.library.custom_op(name, mutates_args=())
    def with_grad(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return m.fss_forward(x, w)

    @with_grad.register_fake
    def _(x, w):
        return x.new_empty(x.shape[0])

    def setup_context(ctx, inputs, output):
        x, w = inputs
        ctx.save_for_backward(x, w)

    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        gx, gw = m.fss_backward(grad, x, w)
        return gx, gw

    with_grad.register_autograd(backward, setup_context=setup_context)
    print(f"  Autograd key 现在有 kernel 了: "
          f"{torch._C._dispatch_has_kernel_for_dispatch_key(name, 'Autograd')}")

    sub("gradcheck：用有限差分逐个元素验梯度")
    torch.manual_seed(0)
    x = torch.randn(8, 16, device=DEV, dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, device=DEV, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(with_grad, (x, w), eps=1e-6,
                                  atol=1e-8, rtol=1e-5)
    print(f"  gradcheck 通过: {ok}")

    sub("和参照实现对拍")
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    with_grad(x, w).sum().backward()
    ref(xr, wr).sum().backward()
    print(f"  dx max|err| = {(x.grad - xr.grad).abs().max().item():.3e}")
    print(f"  dw max|err| = {(w.grad - wr.grad).abs().max().item():.3e}")

    sub("失败现场：反向写错但前向是对的")
    name2 = f"{LIB}::bad_grad"

    @torch.library.custom_op(name2, mutates_args=())
    def bad_grad(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return m.fss_forward(x, w)

    @bad_grad.register_fake
    def _(x, w):
        return x.new_empty(x.shape[0])

    def bad_backward(ctx, grad):
        x, w = ctx.saved_tensors
        # 忘了 relu 的掩码 —— 前向完全不受影响
        gx = grad.unsqueeze(1) * w.unsqueeze(0)
        gw = (grad.unsqueeze(1) * x).sum(0)
        return gx, gw

    bad_grad.register_autograd(bad_backward, setup_context=setup_context)

    x2 = torch.randn(8, 16, device=DEV, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(16, device=DEV, dtype=torch.float64, requires_grad=True)
    print(f"  前向仍然正确: max|err| = "
          f"{(bad_grad(x2, w2) - ref(x2, w2)).abs().max().item():.3e}")
    try:
        torch.autograd.gradcheck(bad_grad, (x2, w2))
        print("  gradcheck 竟然通过了（不应该）")
    except Exception as exc:                                  # noqa: BLE001
        print("  gradcheck 抓到了:")
        for line in str(exc).splitlines()[:10]:
            print("   ", line[:130])


# ---------------------------------------------------------------- E
def section_E():
    title("[E] 注册之后它能被融合吗")
    if DEV != "cuda":
        print("需要 CUDA"); return
    import re
    import time
    import torch._dynamo as dynamo
    from torch._inductor.utils import run_and_get_code

    m = kernel()
    name = f"{LIB}::fusible_probe"

    @torch.library.custom_op(name, mutates_args=())
    def op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return m.fss_forward(x, w)

    @op.register_fake
    def _(x, w):
        return x.new_empty(x.shape[0])

    x = torch.randn(4096, 4096, device="cuda")
    w = torch.randn(4096, device="cuda")

    def with_custom(a, b):
        return (op(a, b) * 2.0).relu() + 1.0

    def pure_torch(a, b):
        return (torch.relu(a * b).sum(dim=-1) * 2.0).relu() + 1.0

    print(f"  两者数值一致: "
          f"{torch.allclose(with_custom(x, w), pure_torch(x, w), atol=1e-3)}")

    for label, fn in [("自定义算子 + 后续逐元素", with_custom),
                      ("等价的纯 torch 写法", pure_torch)]:
        dynamo.reset()
        cf = torch.compile(fn)
        _, code = run_and_get_code(cf, x, w)
        src = "\n".join(code)
        k = src.count("@triton.jit")
        calls = re.findall(r"torch\.ops\.\w+\.\w+", src)
        print(f"\n  {label}")
        print(f"    Inductor 生成 triton kernel: {k}")
        print(f"    图里对外部算子的调用: {sorted(set(calls))}")

    def timeit(fn, n=30):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    dynamo.reset(); c1 = torch.compile(with_custom); c1(x, w)
    dynamo.reset(); c2 = torch.compile(pure_torch); c2(x, w)
    print(f"\n  自定义算子（编译后）  {timeit(lambda: c1(x, w)):7.3f} ms")
    print(f"  纯 torch（编译后）    {timeit(lambda: c2(x, w)):7.3f} ms")
    print(f"  我的 kernel（裸调）    {timeit(lambda: m.fss_forward(x, w)):7.3f} ms")
    print(f"  eager 纯 torch        {timeit(lambda: pure_torch(x, w)):7.3f} ms")
    print("\n  自定义算子是一个**不透明的边界**：Inductor 不能把它前后的算子")
    print("  融进它，也不能把它拆开重排。写得比库快，才值得付这个边界的代价。")
    del x, w
    torch.cuda.empty_cache()


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device {DEV}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
