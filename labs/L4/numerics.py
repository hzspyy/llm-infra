#!/usr/bin/env python3
"""L4.2 —— 数值系统与确定性。

  [A] 几种浮点格式的解剖：位怎么分、能表示什么、分辨率多少
  [B] 累加顺序：同一批数，加法顺序不同结果不同
  [C] batch 不变性：同一条 prompt，batch 大小不同，logits 一样吗
  [D] 确定性开关：哪些能让结果可复现，代价是什么

用法：
    python numerics.py
    python numerics.py C
"""

import os
import struct
import sys

import torch

MB = 1024 * 1024


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 浮点格式解剖")

    fmts = [
        ("float32", torch.float32, 1, 8, 23),
        ("bfloat16", torch.bfloat16, 1, 8, 7),
        ("float16", torch.float16, 1, 5, 10),
        ("float8_e4m3fn", torch.float8_e4m3fn, 1, 4, 3),
        ("float8_e5m2", torch.float8_e5m2, 1, 5, 2),
    ]
    print(f"  {'格式':<16} {'位':>4} {'符号':>4} {'指数':>4} {'尾数':>4} "
          f"{'最大值':>12} {'最小正规':>12} {'相对分辨率':>12}")
    for name, dt, s_, e_, m_ in fmts:
        bits = s_ + e_ + m_
        try:
            fi = torch.finfo(dt)
            mx, mn, eps = fi.max, fi.tiny, fi.eps
            print(f"  {name:<16} {bits:>4} {s_:>4} {e_:>4} {m_:>4} "
                  f"{mx:>12.3e} {mn:>12.3e} {eps:>12.3e}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {name:<16} finfo 失败 {exc}")

    print("\n  两个独立的东西：")
    print("    指数位决定**动态范围**（能表示多大/多小）")
    print("    尾数位决定**相对分辨率**（相邻两个数差多少）")
    print("  bf16 和 fp32 的指数位都是 8 -> **动态范围完全相同**，")
    print("  bf16 只是分辨率差 2^16 倍。这就是它比 fp16 更适合训练的原因：")
    print("  fp16 只有 5 位指数，最大 65504，梯度一大就溢出，要靠 loss scaling 兜。")

    sub("同一个数在各格式里变成什么")
    for val in [1.0, 0.1, 3.14159265, 1e-8, 65504.0, 1e5, 448.0]:
        row = [f"  {val:>14.8g}"]
        for name, dt, *_ in fmts:
            t = torch.tensor([val], dtype=torch.float32).to(dt).float().item()
            row.append(f"{t:>14.6g}")
        print(" ".join(row))
    print(f"  {'':>14} " + " ".join(f"{n.replace('float','f'):>14}" for n, *_ in fmts))
    print("\n  65504 是 fp16 的最大值；448 是 fp8_e4m3 的最大值。")
    print("  超过就变成 inf 或被截断 —— 这是量化里必须先做缩放的原因。")

    sub("bf16 的位模式：直接看字节")
    for val in [1.0, 1.5, 2.0, 3.14159265]:
        f32 = struct.unpack("<I", struct.pack("<f", val))[0]
        b16 = torch.tensor([val]).bfloat16().view(torch.int16).item() & 0xFFFF
        print(f"  {val:>10.6f}  fp32 = {f32:032b}")
        print(f"  {'':>10}  bf16 = {b16:016b}   <- 就是 fp32 的高 16 位")
    print("  bf16 = 截断 fp32 的低 16 位。所以 fp32<->bf16 转换极其便宜，")
    print("  也解释了为什么它们的动态范围一模一样。")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 累加顺序：加法不满足结合律")

    torch.manual_seed(0)
    n = 1 << 20
    x = torch.randn(n, dtype=torch.float32)
    exact = x.double().sum().item()
    print(f"  {n} 个 fp32 随机数，float64 参照和 = {exact:.10f}")

    orders = {
        "顺序累加 (float32)": lambda t: t.sum().item(),
        "逆序累加": lambda t: t.flip(0).sum().item(),
        "随机打乱": lambda t: t[torch.randperm(t.numel())].sum().item(),
        "两两分治 (torch 默认在 CPU 上就是这个)": None,
        "float64 累加": lambda t: t.double().sum().item(),
    }
    for name, fn in orders.items():
        if fn is None:
            continue
        v = fn(x)
        print(f"  {name:<40} {v:>18.10f}  误差 {v - exact:>+12.3e}")

    sub("同一个 GPU 归约，跑 10 次")
    if torch.cuda.is_available():
        g = x.cuda()
        vals = {g.sum().item() for _ in range(10)}
        print(f"  10 次结果的不同取值个数: {len(vals)}")
        print(f"  取值: {sorted(vals)}")
        print("  同一个 kernel、同一份输入，GPU 上通常是确定的 ——")
        print("  因为归约的切分方式由 grid 决定，而 grid 只取决于形状。")

    sub("换一个形状（触发不同的归约切分）就可能变")
    if torch.cuda.is_available():
        print(f"  {'长度':>10} {'sum':>20} {'与 float64 的差':>18}")
        for m in [n, n - 1, n // 2, n // 2 - 1, n + 1]:
            t = x[:m].cuda() if m <= n else torch.cat([x, x[:1]]).cuda()
            ex = t.double().sum().item()
            print(f"  {m:>10} {t.sum().item():>20.8f} {t.sum().item() - ex:>+18.3e}")
        print("  形状变了，归约的切分与累加顺序就变了，末位随之变化。")
        print("  **这不是 bug，是浮点加法不满足结合律的必然结果**（2.4 讨论过）。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] batch 不变性：同一条 prompt，batch 大小不同，结果一样吗")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = os.environ.get("L42_MODEL", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16).cuda().eval()

    text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    S = ids.shape[1]
    print(f"  prompt {S} 个 token，模型 {repo}")
    print("  做法：把同一条 prompt 放在 batch 的第 0 位，")
    print("  后面填不同数量的**其它**序列，只看第 0 条的 logits。")
    print("  数学上第 0 条的结果与 batch 里有没有别人**无关**（causal + 无 padding 交互）。")

    torch.manual_seed(0)
    filler = torch.randint(1000, 5000, (64, S), device="cuda")
    base = None
    print(f"\n  {'batch':>6} {'与 batch=1 的 max|diff|':>24} {'argmax 相同':>12} "
          f"{'逐位相同':>10}")
    for B in [1, 2, 4, 8, 16, 32, 64]:
        inp = torch.cat([ids, filler[:B - 1]], dim=0) if B > 1 else ids
        with torch.no_grad():
            out = model(inp).logits[0, -1].float()
        if base is None:
            base = out.clone()
            print(f"  {B:>6} {'(基准)':>24} {'-':>12} {'-':>10}")
            continue
        d = (out - base).abs().max().item()
        print(f"  {B:>6} {d:>24.6e} "
              f"{str(out.argmax().item() == base.argmax().item()):>12} "
              f"{str(torch.equal(out, base)):>10}")

    print("\n  若「逐位相同」不是全 True，说明这个引擎**不是 batch 不变的**：")
    print("  同一条请求的输出会随同批次里的其它请求而变。")
    print("  在温度 0 的场景下，这意味着**同样的输入可能给出不同的答案**。")

    sub("原因")
    print("  batch 变了 -> GEMM 的形状变了 -> cuBLAS 可能选不同的 kernel")
    print("  -> 归约的切分与累加顺序变了 -> 末位不同 -> 经过 28 层放大")
    print("  -> 偶尔翻转某个 token 的 argmax。")
    print("  [B] 节那张「换形状就变」的表就是这条链的第一环。")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 确定性开关")

    print(f"  torch.backends.cuda.matmul.allow_tf32 = "
          f"{torch.backends.cuda.matmul.allow_tf32}")
    print(f"  torch.backends.cudnn.allow_tf32       = "
          f"{torch.backends.cudnn.allow_tf32}")
    print(f"  torch.backends.cudnn.deterministic    = "
          f"{torch.backends.cudnn.deterministic}")
    print(f"  torch.backends.cudnn.benchmark        = "
          f"{torch.backends.cudnn.benchmark}")
    fp32p = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    print(f"  matmul.fp32_precision                 = {fp32p}")

    sub("TF32 改变的是什么")
    print("  TF32：19 位（1 符号 + 8 指数 + 10 尾数），在 tensor core 上跑 fp32 矩阵乘。")
    print("  动态范围同 fp32，分辨率退到接近 fp16。默认在 Ampere+ 上**可能是开的**。")
    if torch.cuda.is_available():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        ref = (a.double() @ b.double()).float()
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            r = a @ b
            print(f"    allow_tf32={tf32!s:<6} max|diff| vs float64 = "
                  f"{(r - ref).abs().max().item():.6e}")
        torch.backends.cuda.matmul.allow_tf32 = False

    sub("torch.use_deterministic_algorithms 管什么")
    print("  它让**有多种实现的算子**选确定的那个（例如 scatter_add、index_put）。")
    print("  它**不能**让 GEMM 在不同 batch 下给出相同结果 —— ")
    print("  那不是「非确定」，那是「不同的输入形状走了不同的 kernel」。")
    print("  [C] 节的 batch 不变性问题**不是这个开关能解决的**。")
    if torch.cuda.is_available():
        x = torch.randn(1000, device="cuda")
        idx = torch.randint(0, 10, (1000,), device="cuda")
        for det in (False, True):
            torch.use_deterministic_algorithms(det, warn_only=True)
            vals = set()
            for _ in range(5):
                o = torch.zeros(10, device="cuda")
                o.scatter_add_(0, idx, x)
                vals.add(tuple(o.tolist()))
            print(f"    use_deterministic_algorithms({det!s:<5}) "
                  f"scatter_add 5 次的不同取值: {len(vals)}")
        torch.use_deterministic_algorithms(False)


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"gpu {torch.cuda.get_device_name(0)}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
