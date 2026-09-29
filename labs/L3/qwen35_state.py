#!/usr/bin/env python3
"""L3.4-C —— Qwen3.5 的线性/全注意力层列表、卷积与递推状态，以及 S/B 扫描。

混合架构的账必须逐层算，不能按"所有层都持有 KV"估。本 lab：

  [A] 从 config 读出真实的层类型列表，按层算三类状态：
      全注意力层的 KV、线性层的卷积状态、线性层的递推状态
  [B] 实例化模型，核对模块名与状态张量的实际 shape；
      跑一次带 cache 的前向，检查缓存对象里到底存了什么
  [C] S=512/2048/8192/16384 × B=1/4 的 prefill 扫描与 decode 单步，
      记录时间、峰值与**实际走的 kernel**（fla 加速 还是 torch 参照回退）
  [D] 结论：状态增长与运行路径逐层对应

用法：
    L3_OUT=<目录> python qwen35_state.py A B C D
    FLA_SRC=<fla 源码目录> python qwen35_state.py C     # 走加速路径
"""

import logging
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                       # noqa: E402

MODEL = os.environ.get("L35_MODEL", "Qwen/Qwen3.5-4B")
FLA_SRC = os.environ.get(
    "FLA_SRC",
    "/scratch/learn/opt/fla-src/516143e31fce09925e6c39ac37148444bad176c4")
# 必须排在 transformers 之前导入：kernel 的选择发生在建模文件**导入时**，
# 之后再把 fla 加进 sys.path 已经晚了（回退路径已经被固定下来）。
if os.path.isdir(FLA_SRC):
    sys.path.insert(0, FLA_SRC)
MB = 1024 * 1024


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


class Capture(logging.Handler):
    """抓 transformers 的 warning_once（回退提示走的就是它）。"""

    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def timeit(fn, n=5, warmup=2):
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


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    peak = torch.cuda.max_memory_allocated()
    del out
    torch.cuda.empty_cache()
    return (peak - base) / MB


def kernel_names(fn, limit=6):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out, seen = [], set()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.key:
            if e.key not in seen:
                seen.add(e.key)
                out.append(e.key)
    return out[:limit]


def load_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 层列表与三类状态的逐层账")

    cfg = load_config()
    types = list(cfg.layer_types)
    n_full = types.count("full_attention")
    n_lin = types.count("linear_attention")
    print(f"  {MODEL}: {cfg.num_hidden_layers} 层，"
          f"full_attention {n_full} 层 / linear_attention {n_lin} 层")
    print(f"  full_attention_interval = {cfg.full_attention_interval}；"
          f"层列表 = {types[:8]} …")
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    Dh = cfg.head_dim
    kd = cfg.linear_key_head_dim
    vd = cfg.linear_value_head_dim
    Hk = cfg.linear_num_key_heads
    Hv = cfg.linear_num_value_heads
    ker = cfg.linear_conv_kernel_dim
    print(f"  全注意力：Hq={Hq} Hkv={Hkv} head_dim={Dh}")
    print(f"  线性层：key 头 {Hk}×{kd}，value 头 {Hv}×{vd}，"
          f"卷积核 {ker}，状态 dtype {cfg.mamba_ssm_dtype}")

    kv_per_tok = 2 * Hkv * Dh * 2                       # K+V, bf16
    conv_dim = 2 * Hk * kd + Hv * vd                    # q,k,v 通道
    conv_state = conv_dim * (ker - 1) * 4               # fp32 卷积状态
    rec_state = Hv * kd * vd * 4                        # fp32 递推状态
    print(f"\n  {'层类型':>18} {'每请求状态（字节）':>20} {'说明':>28}")
    print(f"  {'full_attention':>18} {kv_per_tok:>20} "
          f"{'KV 随上下文线性增长':>28}")
    print(f"  {'linear_attention':>18} {conv_state + rec_state:>20} "
          f"{'conv %d + recurrent %d':>28}" % (conv_state, rec_state))
    per_tok = n_full * kv_per_tok
    fixed = n_lin * (conv_state + rec_state)
    print(f"\n  每 token 新增 = {n_full} × {kv_per_tok} = {per_tok} B/token")
    print(f"  固定部分   = {n_lin} × {conv_state + rec_state} = {fixed} B "
          f"（不随上下文增长）")
    print(f"  {'上下文':>9} {'KV 部分':>14} {'固定部分':>12} {'合计':>12}")
    for ctx in [4096, 32768, 131072, 262144]:
        kv = per_tok * ctx
        print(f"  {ctx:>9} {kv / MB:>12.1f}MB {fixed / MB:>10.2f}MB "
              f"{(kv + fixed) / MB:>10.1f}MB")
        h.case(id=f"A_ctx{ctx}", ctx=ctx, kv_bytes=kv, fixed_bytes=fixed,
               total_bytes=kv + fixed, full_layers=n_full, linear_layers=n_lin)
    h.case(id="A_layer_types", layer_types=types, full_attention_interval=
           cfg.full_attention_interval, kv_per_token=kv_per_tok,
           conv_state_bytes=conv_state, recurrent_state_bytes=rec_state,
           conv_dim=conv_dim, Hq=Hq, Hkv=Hkv, Dh=Dh, Hk=Hk, Hv=Hv, kd=kd, vd=vd,
           kernel=ker)
    print("\n  如果按'所有层都持有 KV'估，会把 "
          f"{n_lin} 层也算进去：那会是 {cfg.num_hidden_layers * kv_per_tok} B/token，"
          f"比真实值高 {cfg.num_hidden_layers / n_full:.1f}×。")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] 实例化模型：模块名、状态张量与 cache 内容")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cap = Capture()
    lg = logging.getLogger("transformers")
    lg.addHandler(cap)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        tok = AutoTokenizer.from_pretrained(MODEL)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda")
            cls = type(model).__name__
        except Exception as exc:                                   # noqa: BLE001
            print(f"  AutoModelForCausalLM 不可用（{str(exc)[:60]}），"
                  f"尝试条件生成类")
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda")
            cls = type(model).__name__
    print(f"  载入 {MODEL} → {cls}")
    checks = {"fla": False, "causal_conv1d": False}
    for pkg in checks:
        try:
            __import__(pkg)
            checks[pkg] = True
        except Exception:                                          # noqa: BLE001
            checks[pkg] = False
    print(f"  可选依赖：fla={checks['fla']}  causal_conv1d={checks['causal_conv1d']}")
    for m in cap.msgs:
        if "falling back" in m or "fla" in m or "causal_conv1d" in m:
            print(f"  [transformers 警告] {m[:160]}")
    for w in wlist[:6]:
        txt = str(w.message)
        if any(k in txt for k in ("fla", "conv1d", "fallback", "slow")):
            print(f"  [python 警告] {txt[:160]}")
    has_fallback = any("falling back" in m for m in cap.msgs)
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        fla_ok = True
    except Exception:                                              # noqa: BLE001
        fla_ok = False
    print(f"  fla 可用（含 ops）={fla_ok}；本次出现回退提示={has_fallback}"
          "（提示只在建模文件导入时判断一次）")

    layers = model.model.layers if hasattr(model, "model") else model.layers
    for idx, kind in [(0, "linear"), (3, "full")]:
        lay = layers[idx]
        sub_names = [n for n, _ in lay.named_children()]
        print(f"\n  第 {idx} 层（{kind}）：子模块 {sub_names}")
        for n, p in lay.named_parameters():
            if any(k in n for k in ("conv", "A_log", "dt_bias", "in_proj",
                                    "q_proj", "k_proj", "v_proj", "o_proj",
                                    "norm")):
                print(f"    {n:<52} {tuple(p.shape)}  {p.dtype}")
        h.case(id=f"B_layer{idx}_{kind}", submodules=sub_names,
               params={n: list(p.shape) for n, p in lay.named_parameters()
                       if any(k in n for k in ("conv", "A_log", "dt_bias",
                                               "in_proj", "q_proj", "k_proj",
                                               "v_proj", "o_proj", "norm"))})

    sub("带 cache 的前向：缓存对象里存了什么")
    text = "混合架构里，线性层不产生 KV，而是固定大小的卷积与递推状态。" * 4
    ids = tok(text, return_tensors="pt").input_ids[:, :96].cuda()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    print(f"  past_key_values 类型：{type(pkv).__name__}")
    if hasattr(pkv, "layers"):
        print(f"  缓存层数：{len(pkv.layers)}")
        for i in [0, 1, 3]:
            c = pkv.layers[i] if not isinstance(pkv.layers, dict) else pkv.layers[i]
            fields = {}
            for name in ("keys", "values", "conv_state", "recurrent_state",
                         "key_cache", "value_cache", "state"):
                if hasattr(c, name):
                    v = getattr(c, name)
                    if v is not None:
                        fields[name] = (list(v.shape) if hasattr(v, "shape")
                                        else type(v).__name__)
            if not fields and isinstance(c, (tuple, list)):
                fields = {f"tuple[{j}]": list(x.shape) for j, x in enumerate(c)
                          if hasattr(x, "shape")}
            print(f"    第 {i} 层缓存：{fields}")
            h.case(id=f"B_cache_layer{i}", fields={k: str(v) for k, v in fields.items()})
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] S/B 扫描：prefill、decode 与运行的 kernel")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cap = Capture()
    logging.getLogger("transformers").addHandler(cap)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    text = "混合架构里，线性层用固定大小的状态，全注意力层才持有 KV cache。" * 400
    ids_full = tok(text, return_tensors="pt").input_ids

    fast = False
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        fast = True
    except Exception:                                              # noqa: BLE001
        fast = False
    print(f"  本次运行路径：{'fla 加速' if fast else 'torch 参照回退'}；"
          f"FLA_SRC={FLA_SRC}")
    for m in cap.msgs:
        if "falling back" in m:
            print(f"  [回退提示] {m[:150]}")

    print(f"\n  {'B':>2} {'S':>7} {'prefill ms':>11} {'峰值 MB':>9} "
          f"{'decode 单步 ms':>14}")
    for B in [1, 4]:
        for S in [512, 2048, 8192, 16384]:
            n = ids_full.shape[1]
            rep = (S + n - 1) // n
            ids = ids_full.repeat(B, rep)[:, :S].cuda()
            try:
                def fwd():
                    try:
                        return model(ids, num_logits_to_keep=1)
                    except TypeError:
                        return model(ids)
                with torch.no_grad():
                    t = timeit(fwd, n=3, warmup=1)
                    p = peak_mb(fwd)
                    out = model(ids, use_cache=True, num_logits_to_keep=1)
                    cache = out.past_key_values
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    t_dec = timeit(lambda: model(nxt, past_key_values=cache,
                                                 use_cache=True), n=5, warmup=2)
            except torch.cuda.OutOfMemoryError:
                print(f"  {B:>2} {S:>7}  显存不足")
                torch.cuda.empty_cache()
                continue
            print(f"  {B:>2} {S:>7} {t:>11.3f} {p:>9.1f} {t_dec:>14.4f}")
            h.case(id=f"C_B{B}_S{S}", B=B, S=S, prefill_ms=t, peak_mb=p,
                   decode_step_ms=t_dec, path="fla" if fast else "torch-fallback",
                   dtype="bfloat16")
            del ids, out, cache
            torch.cuda.empty_cache()

    sub("实际 kernel（一次 S=2048 prefill）")
    ids = ids_full[:, :2048].cuda()
    with torch.no_grad():
        kn = kernel_names(lambda: model(ids))
    for k in kn:
        print(f"    {k[:90]}")
    h.case(id="C_kernels", S=2048, kernels=kn, path="fla" if fast else "torch-fallback")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D(h):
    title("[D] 逐层对应：状态增长与运行路径")

    cfg = load_config()
    types = list(cfg.layer_types)
    print("  层号  类型              状态")
    for i, t in enumerate(types[:12]):
        if t == "full_attention":
            st = f"KV {2 * cfg.num_key_value_heads * cfg.head_dim * 2} B/token（随上下文增长）"
        else:
            cd = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim \
                + cfg.linear_num_value_heads * cfg.linear_value_head_dim
            st = (f"conv {cd}×{cfg.linear_conv_kernel_dim - 1} + "
                  f"recurrent {cfg.linear_num_value_heads}×{cfg.linear_key_head_dim}×"
                  f"{cfg.linear_value_head_dim}（固定）")
        print(f"  {i:>4}  {t:<16} {st}")
    print("  …")
    print("  只有 full_attention 层的状态随上下文增长；线性层是常数。")
    print("  这解释了扫描里同 S 下混合模型比全注意力模型省多少显存，")
    print("  也说明'每 token 状态字节'必须逐层加，不能取平均。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} sm_{p.major}{p.minor}")
    h = Harness("3.4-C", "3.4", out=os.environ.get("L3_OUT"),
                backend="transformers qwen3_5（fla 或 torch 回退）",
                notes=f"model={MODEL}; FLA_SRC={FLA_SRC}")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "层列表与三类状态逐层对应；S/B 扫描给出时间与峰值；"
                         "回退路径由 transformers 的提示可见。",
              "model": MODEL, "fla_src": FLA_SRC})
    sys.stdout.flush()
    os._exit(0)
