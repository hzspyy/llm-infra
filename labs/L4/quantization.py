#!/usr/bin/env python3
"""L4.3 —— 量化工程：把打包的整数拆开，看清楚它到底损失了什么。

  [A] 手工解包 AWQ 的 qweight，还原成 fp16，和原始权重比
  [B] 量化误差的分布：哪些权重损失最大
  [C] 质量：真实困惑度对照（bf16 vs AWQ vs GPTQ）
  [D] 速度：fp8 GEMM vs bf16 GEMM（这张卡支持 fp8）
  [E] KV cache 量化：decode 带宽直接减半

用法：
    python quantization.py
    python quantization.py A C
"""

import glob
import json
import os
import struct
import sys

import torch

# 模型都已在本地缓存；强制离线，避免 vLLM/transformers 每次去连 Hub
# （连不上时会直接抛 httpx.ConnectError，即使文件就在本地）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MB = 1024 * 1024
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def snap(repo):
    d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots"
    g = sorted(glob.glob(d + "/*"))
    return g[0] if g else None


def timeit(fn, n=30, warmup=10):
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


# ---------------------------------------------------------------- A
def unpack_awq(qweight, qzeros, scales, bits=4, group_size=128):
    """AWQ 的打包：每个 int32 装 8 个 4-bit，但顺序是交错的。

    qweight: [in_features, out_features // 8]  int32
    qzeros : [in_features // group_size, out_features // 8]  int32
    scales : [in_features // group_size, out_features]  fp16
    返回 [in_features, out_features] 的 fp16
    """
    # AWQ 的 GEMM 版把 8 个 nibble 按这个顺序放：0,4,1,5,2,6,3,7
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    shifts = (order * bits).to(qweight.device)

    def unpack(x):
        # x: [..., n] int32 -> [..., n*8]
        out = (x.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
        return out.reshape(*x.shape[:-1], -1)

    w = unpack(qweight)                       # [in, out]
    z = unpack(qzeros)                        # [in/gs, out]
    gs = group_size
    z = z.repeat_interleave(gs, dim=0)[: w.shape[0]]
    sc = scales.repeat_interleave(gs, dim=0)[: w.shape[0]]
    return ((w.float() - z.float()) * sc.float()).half()


def section_A():
    title("[A] 手工解包 AWQ，还原成 fp16")

    d = snap("Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    if not d:
        print("  没有 AWQ 模型，跳过"); return
    from safetensors import safe_open
    cfg = json.load(open(d + "/config.json"))
    qc = cfg["quantization_config"]
    gs, bits = qc["group_size"], qc["bits"]
    print(f"  quantization_config: bits={bits} group_size={gs} "
          f"zero_point={qc.get('zero_point')} version={qc.get('version')}")

    name = "model.layers.0.self_attn.q_proj"
    ts = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.startswith(name + "."):
                    ts[k.split(".")[-1]] = sf.get_tensor(k)
    for k, v in sorted(ts.items()):
        print(f"    {k:<10} {str(tuple(v.shape)):<16} {v.dtype}")

    qw, qz, sc = ts["qweight"], ts["qzeros"], ts["scales"]
    print(f"\n  打包关系：")
    print(f"    qweight {tuple(qw.shape)} int32 -> 每个 int32 装 {32 // bits} 个 "
          f"{bits}-bit -> 展开后 {qw.shape[0]}×{qw.shape[1] * (32 // bits)}")
    print(f"    scales  {tuple(sc.shape)} = [in/{gs}, out]，每 {gs} 行共享一个缩放")
    print(f"    qzeros  {tuple(qz.shape)} int32，同样打包，每 {gs} 行共享一个零点")

    w = unpack_awq(qw, qz, sc, bits, gs)
    print(f"\n  还原出来的权重 {tuple(w.shape)} {w.dtype}")
    print(f"    前 8 个值: {[round(v, 5) for v in w[0, :8].float().tolist()]}")
    print(f"    整体 min={w.float().min().item():.5f} "
          f"max={w.float().max().item():.5f} std={w.float().std().item():.5f}")

    sub("每个权重只有 16 个可能取值（4 bit）")
    g0 = w[:gs, 0].float()
    uniq = torch.unique(g0)
    print(f"    取第 0 列的前 {gs} 行（同一组，共享一个 scale/zero）")
    print(f"    不同取值个数: {len(uniq)}   （上限是 2^{bits} = {2 ** bits}）")
    print(f"    取值: {[round(v, 5) for v in uniq.tolist()]}")
    print(f"    相邻间隔（= scale）: "
          f"{round((uniq[1] - uniq[0]).item(), 6) if len(uniq) > 1 else 'n/a'}")
    print(f"    scales[0,0] = {sc[0, 0].float().item():.6f}  <- 应当相等")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 量化误差有多大，落在哪里")

    d_q = snap("Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    d_f = snap("Qwen/Qwen2.5-1.5B-Instruct")
    if not d_q:
        print("  缺 AWQ 模型"); return
    from safetensors import safe_open
    cfg = json.load(open(d_q + "/config.json"))
    gs, bits = cfg["quantization_config"]["group_size"], cfg["quantization_config"]["bits"]

    name = "model.layers.0.self_attn.q_proj"
    ts = {}
    for f in sorted(glob.glob(d_q + "/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.startswith(name + "."):
                    ts[k.split(".")[-1]] = sf.get_tensor(k)
    w = unpack_awq(ts["qweight"], ts["qzeros"], ts["scales"], bits, gs).float()
    sc = ts["scales"].float()

    # 未量化的同一个权重
    ref = None
    if d_f:
        for f in sorted(glob.glob(d_f + "/*.safetensors")):
            with safe_open(f, framework="pt", device="cpu") as sf:
                if name + ".weight" in sf.keys():
                    ref = sf.get_tensor(name + ".weight").float().T  # [in, out]
    if ref is not None:
        err = (w - ref).abs()
        print(f"  与未量化权重对照 {tuple(ref.shape)}")
        print(f"    max|err|  = {err.max().item():.6f}")
        print(f"    mean|err| = {err.mean().item():.6f}")
        print(f"    权重本身 mean|w| = {ref.abs().mean().item():.6f}")
        print(f"    平均相对误差 = {(err.mean() / ref.abs().mean()).item():.2%}")
        print(f"    余弦相似度 = "
              f"{torch.nn.functional.cosine_similarity(w.flatten(), ref.flatten(), dim=0).item():.6f}")
        sub("误差最大的权重是不是绝对值最大的那些")
        big = ref.abs() > ref.abs().quantile(0.999)
        print(f"    最大的 0.1% 权重：平均 |err| = {err[big].mean().item():.6f}"
              f"，占其自身的 {(err[big].mean() / ref[big].abs().mean()).item():.2%}")
        small = ref.abs() < ref.abs().quantile(0.5)
        print(f"    最小的 50% 权重：平均 |err| = {err[small].mean().item():.6f}"
              f"，占其自身的 {(err[small].mean() / ref[small].abs().mean()).item():.2%}")
        print("    量化误差的绝对值对大小权重是**同一量级**（同组共享 scale），")
        print("    所以**相对误差**在小权重上大得多。")
        print("    AWQ 的核心思想正是：按激活的重要性挑出关键通道，")
        print("    在缩放上给它们更好的待遇。")
    print(f"\n  量化后的权重 {tuple(w.shape)}")
    print(f"  每组的 scale 就是这一组的量化步长。scale 的分布：")
    print(f"    min={sc.min().item():.6f}  中位数={sc.median().item():.6f}  "
          f"max={sc.max().item():.6f}  max/min={sc.max().item()/sc.min().item():.1f}×")
    print(f"\n  **量化误差的上界是 scale/2**（四舍五入到最近的格点）。")
    print(f"    最好的组: ±{sc.min().item()/2:.6f}")
    print(f"    最差的组: ±{sc.max().item()/2:.6f}")
    print(f"  组与组之间差 {sc.max().item()/sc.min().item():.0f} 倍 —— ")
    print(f"  这就是 group_size 存在的意义：一个 scale 管全矩阵的话，")
    print(f"  所有权重都要按最大的那个 outlier 缩放，小权重会被压成 0。")

    sub("group_size 的取舍（按定义算，不是实测）")
    out_f = w.shape[1]
    for g in [32, 64, 128, 256, 1024]:
        per_group = g * bits + 16 + bits
        print(f"    group_size={g:>5}  等效 {per_group/g:>5.2f} bit/权重  "
              f"scales+zeros 开销 {(16+bits)/(g*bits):>6.1%}")
    print("  组越小，量化越准（每组的动态范围更窄），但元数据开销越大。")
    print("  128 是当前的常见折中。")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] 质量与速度：用 vLLM 跑真实的量化 kernel")

    # 用 vLLM 而不是 transformers：AWQ/GPTQ 在 transformers 里要额外装
    # gptqmodel / optimum，而 vLLM 自带这两种格式的 kernel，
    # 而且这才是这些 checkpoint 实际被使用的路径。
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    import math

    text = ("Machine learning models are trained on large corpora of text. "
            "The transformer architecture, introduced in 2017, uses self-attention "
            "to model dependencies between tokens regardless of their distance. "
            "Modern language models scale this architecture to billions of parameters, "
            "requiring careful engineering of memory bandwidth, numerical precision, "
            "and parallel execution across many accelerators. Quantization reduces "
            "the number of bits used to store each weight, trading some accuracy for "
            "lower memory footprint and higher throughput during inference. ") * 4

    repos = [("bf16 原版", "Qwen/Qwen2.5-1.5B-Instruct"),
             ("AWQ 4bit", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
             ("GPTQ 4bit", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4")]
    print("  用同一段文本算困惑度（越低越好）+ 量一次 decode 吞吐。")
    print("  ⚠ **一段文本不构成模型质量评测**，只用来看量化有没有量级上的退化。")
    print("  真正的评测要跑标准数据集，本轮没有做。")
    print(f"\n  {'模型':<14} {'困惑度':>10} {'相对 bf16':>10} "
          f"{'权重显存 MB':>13} {'decode tok/s':>13}")
    base_ppl = None
    for label, repo in repos:
        if not snap(repo):
            print(f"  {label:<14} 未下载，跳过"); continue
        try:
            llm = LLM(model=repo, gpu_memory_utilization=0.45,
                      max_model_len=2048, enforce_eager=True,
                      disable_log_stats=True)
            out = llm.generate([text],
                               SamplingParams(max_tokens=1, temperature=0.0,
                                              prompt_logprobs=0),
                               use_tqdm=False)[0]
            lps = [d[t].logprob for d, t in
                   zip(out.prompt_logprobs[1:], out.prompt_token_ids[1:])]
            ppl = math.exp(-sum(lps) / len(lps))
            # decode 吞吐：128 条各生成 64 token
            import time
            sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
            prompts = [text[:200]] * 64
            llm.generate(prompts, sp, use_tqdm=False)       # 预热
            t0 = time.perf_counter()
            o2 = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntok = sum(len(x.outputs[0].token_ids) for x in o2)
            if base_ppl is None:
                base_ppl = ppl
            szmb = sum(os.path.getsize(x)
                       for x in glob.glob(snap(repo) + "/*.safetensors")) / MB
            print(f"  {label:<14} {ppl:>10.4f} {ppl / base_ppl:>9.3f}× "
                  f"{szmb:>13.1f} {ntok / dt:>13.1f}")
            # 同进程建第二个引擎前必须显式 shutdown，否则显存不还
            # （这是 B1 就记下的坑，见 plan 的踩坑清单第 9 条）
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:                                 # noqa: BLE001
                pass
            del llm
            import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:                              # noqa: BLE001
            print(f"  {label:<14} 失败: {type(exc).__name__}: {str(exc)[:90]}")
            torch.cuda.empty_cache()

    sub("权重占用（按文件大小，不含 KV 与激活）")
    for label, repo in repos:
        d = snap(repo)
        if not d:
            continue
        sz = sum(os.path.getsize(x) for x in glob.glob(d + "/*.safetensors"))
        print(f"  {label:<14} {sz / MB:>10.1f} MB")


# ---------------------------------------------------------------- D
def section_D():
    title("[D] fp8 GEMM vs bf16 GEMM（这张卡支持 fp8）")

    if not torch.cuda.is_available():
        print("  需要 CUDA"); return
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name} sm_{p.major}{p.minor}")
    print("  L1.2 实测：bf16 511 FLOP/clk/SM，fp8 1022 —— 纯发射率正好 2×。")
    print("  这里看端到端的 GEMM 能拿到多少。")

    n = 4096
    a16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    flops = 2 * n ** 3
    t16 = timeit(lambda: a16 @ b16)
    print(f"\n  {'路径':<28} {'ms':>9} {'TFLOP/s':>10} {'相对 bf16':>10}")
    print(f"  {'bf16 @ bf16':<28} {t16:>9.4f} {flops / t16 / 1e9:>10.1f} {1.0:>9.2f}×")

    try:
        a8 = a16.to(torch.float8_e4m3fn)
        b8 = b16.to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device="cuda")
        def fp8mm():
            return torch._scaled_mm(a8, b8.t().contiguous().t(),
                                    scale_a=sa, scale_b=sa,
                                    out_dtype=torch.bfloat16)
        fp8mm()
        t8 = timeit(fp8mm)
        print(f"  {'fp8_e4m3 @ fp8_e4m3':<28} {t8:>9.4f} "
              f"{flops / t8 / 1e9:>10.1f} {t16 / t8:>9.2f}×")
        ref = (a16.float() @ b16.float())
        got = fp8mm().float()
        rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
        print(f"\n  fp8 结果与 fp32 参照的平均相对误差: {rel:.4%}")
        print(f"  （权重值域 ~N(0,1)，fp8_e4m3 上限 448，这里没有溢出）")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  fp8 路径失败: {type(exc).__name__}: {str(exc)[:150]}")

    sub("为什么端到端拿不到 2×")
    print("  纯发射率是 2×，但 GEMM 还要搬数据。4096³ 的 bf16 输入是 64 MB，")
    print("  fp8 是 32 MB —— 算力翻倍而访存只减半，")
    print("  在这个尺寸上很可能已经不是纯算力受限了。")
    del a16, b16
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- E
def section_E():
    title("[E] KV cache 量化：decode 带宽直接减半")

    import torch.nn.functional as F
    B, H, HKV, D = 8, 32, 8, 128
    print(f"  B={B} query头={H} KV头={HKV} D={D}")
    print("  3.3 的结论：decode 时间 = KV 字节 / 带宽。dtype 减半，时间就该减半。")
    print(f"\n  {'kv_len':>8} {'dtype':>10} {'KV 大小':>10} {'ms':>9} {'相对 bf16':>10}")
    for S in [8192, 32768]:
        base = None
        for dt, nb in [(torch.bfloat16, 2), (torch.float8_e4m3fn, 1)]:
            q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
            kv = 2 * B * HKV * S * D * nb
            if dt == torch.bfloat16:
                t = timeit(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
                base = t
            else:
                k8, v8 = k.to(dt), v.to(dt)
                try:
                    # SDPA 不吃 fp8；模拟真实做法：kernel 内反量化再算
                    def run():
                        return F.scaled_dot_product_attention(
                            q, k8.to(torch.bfloat16), v8.to(torch.bfloat16),
                            enable_gqa=True)
                    t = timeit(run)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  {S:>8} {'fp8':>10}  失败 {str(exc)[:50]}")
                    continue
            print(f"  {S:>8} {str(dt).replace('torch.', ''):>10} {kv / MB:>8.1f}MB "
                  f"{t:>9.4f} {base / t:>9.2f}×")
            del q, k, v
            torch.cuda.empty_cache()
    print("\n  ⚠ 上面 fp8 那行是**在框架层显式反量化**再走普通 attention，")
    print("  多了一次读 fp8 + 写 bf16，所以它反而更慢 —— 这和 3.3 §四 分页那次")
    print("  是同一类错误：框架层模拟量不出 kernel 内部做这件事的代价。")
    print("  真实的 fp8 KV attention 在 kernel 内部反量化，只读 fp8 那一份字节。")
    print("  **本 lab 没有真实 fp8 KV kernel 可用，所以只给出字节账：**")
    for S in [8192, 32768, 131072]:
        kv16 = 2 * B * HKV * S * D * 2
        print(f"    kv_len={S:>7}  bf16 {kv16 / MB:>8.1f}MB  "
              f"fp8 {kv16 / 2 / MB:>8.1f}MB  int4 {kv16 / 4 / MB:>8.1f}MB")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
