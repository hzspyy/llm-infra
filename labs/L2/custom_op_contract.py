#!/usr/bin/env python3
"""L2.8 lab · 自定义算子的三段契约：注册与边界、梯度与多路径、融合与真实接入。

对应 completed.md 的 2.8-A/B/C。算子是同一族的两条：

    masked_sum(x, w)[i] = sum_j relu(x[i, j] * w[j])        （A/B 主用例）
    silu_mul(gate, up)  = silu(gate) * up                   （C 的真实接入用例）

每个 mode 独立可跑：

    python labs/L2/custom_op_contract.py --mode a --out-dir <dir>
    python labs/L2/custom_op_contract.py --mode b ...
    python labs/L2/custom_op_contract.py --mode c ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIB = "l28c"

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_sum_fwd_k(const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ out, int rows, int cols) {
    int i = blockIdx.x;
    if (i >= rows) return;
    float acc = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = x[(size_t)i * cols + j] * w[j];
        acc += v > 0.f ? v : 0.f;
    }
    __shared__ float s[256];
    s[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[i] = s[0];
}

__global__ void masked_sum_bwd_k(const float* __restrict__ g, const float* __restrict__ x,
                                 const float* __restrict__ w, float* __restrict__ dx,
                                 float* __restrict__ dw, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.f;
    for (int i = 0; i < rows; ++i) {
        float v = x[(size_t)i * cols + j] * w[j];
        float m = v > 0.f ? 1.f : 0.f;
        dx[(size_t)i * cols + j] = g[i] * w[j] * m;
        s += g[i] * x[(size_t)i * cols + j] * m;
    }
    dw[j] = s;
}

__global__ void silu_mul_fwd_k(const float* __restrict__ gate, const float* __restrict__ up,
                               float* __restrict__ out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.f + expf(-g))) * up[i];
    }
}

__global__ void silu_mul_bwd_k(const float* __restrict__ go, const float* __restrict__ gate,
                               const float* __restrict__ up, float* __restrict__ dg,
                               float* __restrict__ du, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = 1.f / (1.f + expf(-g));
    float ds = s * (1.f + g * (1.f - s));
    dg[i] = go[i] * ds * up[i];
    du[i] = go[i] * s;
}

torch::Tensor masked_sum_fwd(torch::Tensor x, torch::Tensor w) {
    auto xc = x.contiguous(), wc = w.contiguous();
    int rows = xc.size(0), cols = xc.size(1);
    auto out = torch::empty({rows}, xc.options());
    masked_sum_fwd_k<<<rows, 256>>>(xc.data_ptr<float>(), wc.data_ptr<float>(),
                                    out.data_ptr<float>(), rows, cols);
    return out;
}

std::vector<torch::Tensor> masked_sum_bwd(torch::Tensor g, torch::Tensor x, torch::Tensor w) {
    auto xc = x.contiguous(), wc = w.contiguous();
    int rows = xc.size(0), cols = xc.size(1);
    auto dx = torch::empty_like(xc), dw = torch::empty_like(wc);
    int threads = 256, blocks = (cols + threads - 1) / threads;
    masked_sum_bwd_k<<<blocks, threads>>>(g.contiguous().data_ptr<float>(),
                                          xc.data_ptr<float>(), wc.data_ptr<float>(),
                                          dx.data_ptr<float>(), dw.data_ptr<float>(), rows, cols);
    return {dx, dw};
}

torch::Tensor silu_mul_fwd(torch::Tensor gate, torch::Tensor up) {
    auto gc = gate.contiguous(), uc = up.contiguous();
    auto out = torch::empty_like(gc);
    long n = gc.numel();
    silu_mul_fwd_k<<<(n + 255) / 256, 256>>>(gc.data_ptr<float>(), uc.data_ptr<float>(),
                                             out.data_ptr<float>(), n);
    return out;
}

std::vector<torch::Tensor> silu_mul_bwd(torch::Tensor go, torch::Tensor gate, torch::Tensor up) {
    auto gc = gate.contiguous();
    auto dg = torch::empty_like(gc), du = torch::empty_like(gc);
    long n = gc.numel();
    silu_mul_bwd_k<<<(n + 255) / 256, 256>>>(go.contiguous().data_ptr<float>(),
                                             gc.data_ptr<float>(), up.contiguous().data_ptr<float>(),
                                             dg.data_ptr<float>(), du.data_ptr<float>(), n);
    return {dg, du};
}
"""

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        build = pathlib.Path("/scratch/learn/.cache/torchext/l28c")
        build.mkdir(parents=True, exist_ok=True)
        cpp = ("torch::Tensor masked_sum_fwd(torch::Tensor, torch::Tensor);"
               "std::vector<torch::Tensor> masked_sum_bwd(torch::Tensor, torch::Tensor, torch::Tensor);"
               "torch::Tensor silu_mul_fwd(torch::Tensor, torch::Tensor);"
               "std::vector<torch::Tensor> silu_mul_bwd(torch::Tensor, torch::Tensor, torch::Tensor);")
        _EXT = load_inline(name="l28c_ext", cpp_sources=cpp, cuda_sources=CUDA_SRC,
                           functions=["masked_sum_fwd", "masked_sum_bwd",
                                      "silu_mul_fwd", "silu_mul_bwd"],
                           build_directory=str(build), verbose=False)
    return _EXT


def ref_masked_sum(x, w):
    return (x * w).relu().sum(-1)


def build_ops():
    """注册两个自定义算子：schema + CUDA 实现 + fake + autograd。"""
    if hasattr(build_ops, "done"):
        return
    build_ops.done = True

    def _check(x, w):
        if not x.is_floating_point():
            raise RuntimeError(f"x 必须是浮点张量，实际 {x.dtype}")
        if x.dim() != 2:
            raise RuntimeError(f"x 必须是 2D，实际 {x.dim()}D")
        if w.dim() != 1 or w.numel() != x.shape[1]:
            raise RuntimeError(
                f"w 必须是长度 {x.shape[1]} 的 1D 张量，实际 shape={tuple(w.shape)}")

    @torch.library.custom_op(f"{LIB}::masked_sum", mutates_args=())
    def masked_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        _check(x, w)                       # 形状/dtype 契约在包装层显式检查
        return ext().masked_sum_fwd(x, w)

    @masked_sum.register_fake
    def _(x, w):
        return x.new_empty((x.shape[0],))

    # 反向也做成一个 custom op：它有自己的 fake 实现，所以 AOTAutograd
    # 追踪反向时不会拿 fake tensor 去调真实 kernel。
    @torch.library.custom_op(f"{LIB}::masked_sum_backward", mutates_args=())
    def masked_sum_backward(g: torch.Tensor, x: torch.Tensor,
                            w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx, dw = ext().masked_sum_bwd(g, x, w)
        return dx, dw

    @masked_sum_backward.register_fake
    def _(g, x, w):
        return x.new_empty(x.shape), w.new_empty(w.shape)

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1])

    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return torch.ops.l28c.masked_sum_backward.default(g, x, w)

    torch.library.register_autograd(f"{LIB}::masked_sum", backward, setup_context=setup)

    @torch.library.custom_op(f"{LIB}::silu_mul", mutates_args=())
    def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return ext().silu_mul_fwd(gate, up)

    @silu_mul.register_fake
    def _(gate, up):
        return torch.empty_like(gate)

    def setup2(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1])

    def backward2(ctx, go):
        gate, up = ctx.saved_tensors
        dg, du = ext().silu_mul_bwd(go, gate, up)
        return dg, du

    torch.library.register_autograd(f"{LIB}::silu_mul", backward2, setup_context=setup2)


# --------------------------------------------------------------------------- A
def mode_a(out: pathlib.Path) -> dict:
    build_ops()
    op = torch.ops.l28c.masked_sum.default
    res: dict = {}

    x = torch.randn(4, 8, device=DEV)
    w = torch.randn(8, device=DEV)
    y = op(x, w)
    ref = ref_masked_sum(x, w)
    res["basic"] = {"max_abs_diff": float((y - ref).abs().max()),
                    "out_shape": list(y.shape), "out_dtype": str(y.dtype),
                    "ref_shape": list(ref.shape)}
    print(f"    基本用例：out {tuple(y.shape)} vs 参照 {tuple(ref.shape)}，"
          f"max|d|={res['basic']['max_abs_diff']:.3e}")

    # 边界输入：连续 / 转置 / 空 / 非法 dtype / rank / w 形状
    cases = []
    xt = torch.randn(8, 4, device=DEV).t()                    # 转置视图（非连续）
    cases.append(("转置（非连续）", xt, torch.randn(8, device=DEV)))
    cases.append(("非法 dtype int32", torch.randint(0, 5, (4, 8), device=DEV), w))
    cases.append(("rank 1", torch.randn(8, device=DEV), w))
    cases.append(("w 更长（16 vs cols=8）", x, torch.randn(16, device=DEV)))
    cases.append(("w 更短（4 vs cols=8）", x, torch.randn(4, device=DEV)))
    for name, xx, ww in cases:
        try:
            yy = op(xx, ww)
            ok, why = False, None
            try:
                ok = bool(torch.allclose(yy, ref_masked_sum(xx.float(), ww.float()), atol=1e-4))
            except Exception as exc:                           # noqa: BLE001
                ok, why = None, f"{type(exc).__name__}: {exc}"[:120]
            cases_row = {"case": name, "ok": True, "out_shape": list(yy.shape),
                         "matches_ref": ok, "ref_error": why}
            print(f"    {name:22s} 通过，输出 {tuple(yy.shape)} 与参照一致 {ok}"
                  + (f"（参照不可算：{why[:50]}）" if why else ""))
        except Exception as exc:                               # noqa: BLE001
            cases_row = {"case": name, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"    {name:16s} 报错 {type(exc).__name__}: {str(exc)[:90]}")
        res.setdefault("boundary", []).append(cases_row)

    # 手工状态检查：输入不被修改、版本计数不变
    xv, wv = x.clone(), w.clone()
    v0 = (xv._version, wv._version)
    op(xv, wv)
    res["no_input_mutation"] = {"versions_before": list(v0),
                                 "versions_after": [xv._version, wv._version],
                                 "values_unchanged": bool(torch.equal(xv, x) and torch.equal(wv, w))}
    print(f"    输入不被修改：版本计数 {v0} → {(xv._version, wv._version)}，"
          f"数值不变 {res['no_input_mutation']['values_unchanged']}")

    # opcheck
    try:
        r = torch.library.opcheck(op, (x, w))
        res["opcheck"] = {k: str(v)[:120] for k, v in r.items()}
        print("    opcheck：" + "，".join(f"{k}={str(v)[:40]}" for k, v in r.items()))
    except Exception as exc:                                   # noqa: BLE001
        res["opcheck"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"    opcheck 失败：{res['opcheck']['error'][:140]}")

        # 空 tensor 的真实失败模式：包装层不特判 0 行，kernel 以 grid=0 发射。
    # 这个错误是**异步**的，会污染后续 CUDA 调用，所以单独起进程测。
    import subprocess
    code = f"""
import sys, torch
sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parent)!r})
from custom_op_contract import build_ops
build_ops()
e = torch.empty(0, 8, device="cuda")
w = torch.randn(8, device="cuda")
y = torch.ops.l28c.masked_sum.default(e, w)
print("OUT", tuple(y.shape))
torch.cuda.synchronize()
print("SYNC_OK")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**__import__("os").environ})
    out_lines = (proc.stdout + proc.stderr).strip().splitlines()
    res["empty_tensor_isolated"] = {
        "returncode": proc.returncode,
        "stdout": [ln for ln in out_lines if ln.startswith(("OUT", "SYNC_OK"))],
        "error_line": next((ln for ln in out_lines if "Error" in ln or "error" in ln), None),
    }
    print(f"    空 tensor（隔离进程）：退出码 {proc.returncode}，"
          f"{res['empty_tensor_isolated']['stdout']} "
          f"{res['empty_tensor_isolated']['error_line']}")

    return res


# --------------------------------------------------------------------------- B
def mode_b(out: pathlib.Path) -> dict:
    build_ops()
    op = torch.ops.l28c.masked_sum.default
    res: dict = {}

    # gradcheck/gradgradcheck 验的是**公式**（fp64 的 torch 版），
    # 不是 fp32 的 kernel —— 这一点要在结论里说清。
    def formula(x, w):
        return (x * w).relu().sum(-1)

    xd = torch.randn(3, 4, dtype=torch.float64, device=DEV, requires_grad=True)
    wd = torch.randn(4, dtype=torch.float64, device=DEV, requires_grad=True)
    try:
        res["gradcheck_fp64"] = bool(torch.autograd.gradcheck(formula, (xd, wd), eps=1e-6, atol=1e-6))
    except Exception as exc:                                   # noqa: BLE001
        res["gradcheck_fp64"] = f"{type(exc).__name__}: {exc}"[:200]
    print(f"    gradcheck(fp64 公式) = {res['gradcheck_fp64']}")
    try:
        res["gradgradcheck_fp64"] = bool(
            torch.autograd.gradgradcheck(formula, (xd, wd), eps=1e-6, atol=1e-6))
    except Exception as exc:                                   # noqa: BLE001
        res["gradgradcheck_fp64"] = f"{type(exc).__name__}: {exc}"[:200]
    print(f"    gradgradcheck(fp64 公式) = {res['gradgradcheck_fp64']}")

    # 反向在 kernel 上的对拍：与公式的梯度比
    x = torch.randn(4, 8, device=DEV, requires_grad=True)
    w = torch.randn(8, device=DEV, requires_grad=True)
    y = op(x, w)
    y.sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    formula(x2, w2).sum().backward()
    res["grad_vs_formula"] = {
        "dx_max_abs_diff": float((gx - x2.grad).abs().max()),
        "dw_max_abs_diff": float((gw - w2.grad).abs().max())}
    print(f"    kernel 反向 vs 公式：max|Δdx|={res['grad_vs_formula']['dx_max_abs_diff']:.3e} "
          f"max|Δdw|={res['grad_vs_formula']['dw_max_abs_diff']:.3e}")

    # 四条路径：eager / compile / reduce-overhead（CUDA Graph）/ export
    def run_eager(a, b):
        return op(a, b).sum()

    paths: dict = {}
    for name, fn in (("eager", run_eager),):
        x3 = x.detach().clone().requires_grad_(True)
        w3 = w.detach().clone().requires_grad_(True)
        t0 = time.perf_counter(); loss = fn(x3, w3); loss.backward()
        torch.cuda.synchronize() if DEV == "cuda" else None
        paths[name] = {"loss": float(loss), "dx_norm": float(x3.grad.norm()),
                       "dw_norm": float(w3.grad.norm()),
                       "first_call_s": time.perf_counter() - t0}
    for name, mode in (("compile", None), ("reduce-overhead", "reduce-overhead")):
        try:
            torch._dynamo.reset()
            cfn = torch.compile(run_eager, mode=mode, dynamic=False) if mode else \
                torch.compile(run_eager, dynamic=False)
            x3 = x.detach().clone().requires_grad_(True)
            w3 = w.detach().clone().requires_grad_(True)
            t0 = time.perf_counter(); loss = cfn(x3, w3); loss.backward()
            torch.cuda.synchronize() if DEV == "cuda" else None
            paths[name] = {"loss": float(loss), "dx_norm": float(x3.grad.norm()),
                           "dw_norm": float(w3.grad.norm()),
                           "first_call_s": time.perf_counter() - t0,
                           "loss_matches_eager": bool(abs(float(loss) - paths["eager"]["loss"]) < 1e-3)}
            print(f"    {name:16s} loss={float(loss):.6f} 首次 {paths[name]['first_call_s']:.2f} s "
                  f"与 eager 一致 {paths[name]['loss_matches_eager']}")
        except Exception as exc:                               # noqa: BLE001
            paths[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"    {name:16s} [失败] {paths[name]['error'][:110]}")
    try:
        class M(torch.nn.Module):
            def forward(self, a, b):
                return op(a, b).sum()

        ep = torch.export.export(M().eval(), (x.detach(), w.detach()))
        paths["export"] = {"nodes": len(list(ep.graph_module.graph.nodes)),
                           "targets": sorted({str(n.target) for n in ep.graph_module.graph.nodes
                                              if n.op == "call_function"})[:6]}
        print(f"    export           图 {paths['export']['nodes']} 节点，"
              f"含 {paths['export']['targets']}")
    except Exception as exc:                                   # noqa: BLE001
        paths["export"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"    export           [失败] {paths['export']['error'][:110]}")
    res["paths"] = paths
    return res


# --------------------------------------------------------------------------- C
def mode_c(out: pathlib.Path) -> dict:
    build_ops()
    silent = torch.ops.l28c.silu_mul.default
    res: dict = {}

    def count_kernels(fn, reps=5):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            if DEV == "cuda":
                torch.cuda.synchronize()
        ks = {e.key: e.count for e in prof.key_averages()
              if e.self_device_time_total > 0 or e.count}
        return len(ks), ks

    # ---- C1: opaque 边界 vs 可分解实现 ----
    x = torch.randn(64, 512, device=DEV)
    w = torch.randn(512, device=DEV)

    def via_op(a, b):
        return torch.ops.l28c.masked_sum.default(a, b).sum()

    def via_decomp(a, b):
        return (a * b).relu().sum(-1).sum()

    c1 = {}
    for name, fn in (("opaque", via_op), ("decomposable", via_decomp)):
        t0 = time.perf_counter()
        try:
            cf = torch.compile(fn, dynamic=False)
            cf(x, w)
            if DEV == "cuda":
                torch.cuda.synchronize()
            n_kernels, _ = count_kernels(lambda: cf(x, w))
            ts = []
            for _ in range(10):
                a_ = torch.cuda.Event(True); b_ = torch.cuda.Event(True)
                a_.record(); cf(x, w); b_.record(); torch.cuda.synchronize()
                ts.append(a_.elapsed_time(b_))
            c1[name] = {"first_call_s": time.perf_counter() - t0,
                        "kernel_kinds": n_kernels, "median_ms": statistics.median(ts)}
        except Exception as exc:                               # noqa: BLE001
            c1[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"    {name:14s} {c1[name]}")
    res["opaque_vs_decomposable"] = c1

    # ---- C2: 动态 batch + CUDA Graph 捕获 ----
    c2 = {}
    try:
        cf = torch.compile(lambda a, b: torch.ops.l28c.masked_sum.default(a, b).sum(), dynamic=True)
        outs = []
        for batch in (8, 32, 128):
            outs.append(float(cf(torch.randn(batch, 512, device=DEV), w)))
        c2["dynamic_batch"] = {"ok": True, "losses": outs,
                                "unique_shapes": len({round(o, 3) for o in outs})}
        print(f"    动态 batch：三种 batch 都跑通，loss={['%.4f' % o for o in outs]}")
    except Exception as exc:                                   # noqa: BLE001
        c2["dynamic_batch"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"    动态 batch [失败] {c2['dynamic_batch']['error'][:110]}")

    if DEV == "cuda":
        try:
            a = torch.randn(16, 512, device=DEV)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    torch.ops.l28c.masked_sum.default(a, w)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                out_g = torch.ops.l28c.masked_sum.default(a, w)
            g.replay(); torch.cuda.synchronize()
            ref = torch.ops.l28c.masked_sum.default(a, w)
            c2["cuda_graph"] = {"ok": True,
                                 "replay_matches": bool(torch.allclose(out_g, ref, atol=1e-5))}
            print(f"    CUDA Graph 捕获：成功，replay 与直接调用一致 "
                  f"{c2['cuda_graph']['replay_matches']}")
        except Exception as exc:                               # noqa: BLE001
            c2["cuda_graph"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"    CUDA Graph 捕获 [失败] {c2['cuda_graph']['error'][:120]}")
    res["dynamic_and_graph"] = c2

    # ---- C3: 把 Qwen3 的 MLP 激活换成自定义算子 ----
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "q", str(pathlib.Path(__file__).resolve().parent / "qwen3_block_aoti.py"))
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        from transformers import AutoConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

        snap = q.find_snapshot()
        cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
        sd, keys, shards = q.load_layer0(snap)
        mlp_sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
        mlp = Qwen3MLP(cfg).eval().to(DEV, torch.bfloat16)
        mlp.load_state_dict(mlp_sd, strict=False)
        hid = torch.randn(4, 512, cfg.hidden_size, device=DEV, dtype=torch.bfloat16)

        def mlp_orig(h):
            return mlp(h)

        def mlp_custom(h):
            gate = mlp.act_fn(mlp.gate_proj(h))
            up = mlp.up_proj(h)
            return mlp.down_proj(
                torch.ops.l28c.silu_mul.default(gate.float(), up.float()).to(h.dtype))

        def timeit(fn, reps=10):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                a_ = torch.cuda.Event(True); b_ = torch.cuda.Event(True)
                a_.record(); fn(); b_.record(); torch.cuda.synchronize()
                ts.append(a_.elapsed_time(b_))
            return statistics.median(ts)

        c3 = {"weights": {"tensors": len(mlp_sd), "shards": shards},
              "eager_orig_ms": timeit(lambda: mlp_orig(hid)),
              "eager_custom_ms": timeit(lambda: mlp_custom(hid))}
        k_o, _ = count_kernels(lambda: mlp_orig(hid))
        k_c, _ = count_kernels(lambda: mlp_custom(hid))
        c3["kernels_eager_orig"] = k_o
        c3["kernels_eager_custom"] = k_c
        try:
            cf = torch.compile(mlp_orig, dynamic=False)
            cf(hid); torch.cuda.synchronize()
            c3["compiled_orig_ms"] = timeit(lambda: cf(hid))
            k_co, _ = count_kernels(lambda: cf(hid))
            c3["kernels_compiled_orig"] = k_co
        except Exception as exc:                               # noqa: BLE001
            c3["compiled_orig_error"] = f"{type(exc).__name__}: {exc}"[:160]
        res["qwen3_mlp_replacement"] = c3
        print(f"    Qwen3 MLP：eager 原版 {c3['eager_orig_ms']:.3f} ms / "
              f"换成自定义算子 {c3['eager_custom_ms']:.3f} ms；"
              f"kernel 数 {k_o} → {k_c}"
              + (f"；compiled 原版 {c3.get('compiled_orig_ms', float('nan')):.3f} ms"
                 if "compiled_orig_ms" in c3 else ""))
    except Exception as exc:                                   # noqa: BLE001
        res["qwen3_mlp_replacement"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"    Qwen3 MLP 替换 [失败] {res['qwen3_mlp_replacement']['error'][:140]}")
    return res


MODES = {"a": mode_a, "b": mode_b, "c": mode_c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    payload: dict = {"torch": torch.__version__, "device": DEV, "modes": {}}
    todo = list(MODES) if args.mode == "all" else [args.mode]
    for name in todo:
        print(f"\n=== {name} ===")
        try:
            payload["modes"][name] = MODES[name](out)
        except Exception as exc:                               # noqa: BLE001
            payload["modes"][name] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"    [mode 失败] {payload['modes'][name]['error'][:160]}")
    (out / "custom_op_contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out / 'custom_op_contract.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
