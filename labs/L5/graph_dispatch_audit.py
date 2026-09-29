#!/usr/bin/env python3
"""L5.4 任务 A/B（分派事实）—— 图模式到底怎么被选中：模式、桶、指针、图池、fallback。

5.4 的正文此前只量了「开图 vs 不开图」的总时间，没有回答计划里的第一个问题：
**当前版本实际有哪些 graph mode，运行时按什么规则选中其中一个？**

本脚本把分派链路上的事实原样取出来（全部来自运行中的引擎对象与真实源码行号）：

  [A] 模式与捕获清单：configured mode -> 候选描述符 -> 实际捕获的图
  [B] 运行时分派轨迹：真实工作负载下每一步选中的 mode 与 padding
  [C] 指针与图池：静态输入缓冲区的地址稳定性、图内存、KV 容量
  [D] 由 run_graph_dispatch_audit.sh 驱动：nsys --cuda-graph-trace=node 数图内 kernel

`--tamper-pointer` 是单独一个进程里的负向实验：把运行时输入缓冲区换成一个
新分配的地址，看 FULL 图重放读到的是谁的数据。这是「图绑定地址」的直接证据。

用法：
    python graph_dispatch_audit.py                 # A+B+C
    python graph_dispatch_audit.py A B             # 只跑指定节
    python graph_dispatch_audit.py --tamper-pointer # 指针负向实验
    python graph_dispatch_audit.py D --mode FULL_AND_PIECEWISE --bs 4
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L54_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)), flush=True)


def src_loc(module_name, symbol):
    """返回 module.symbol 的真实 file:line，取不到就返回 None。"""
    import importlib
    try:
        parts = module_name.split(".")
        obj = None
        rest: list[str] = []
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
                rest = parts[i:] + symbol.split(".")
                break
            except ImportError:
                continue
        if obj is None:
            return f"<{module_name} 不可导入>"
        for part in rest:
            obj = getattr(obj, part)
        path = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
        return f"{path}:{line}"
    except Exception as exc:                                     # noqa: BLE001
        return f"<{module_name}.{symbol} 定位失败: {exc}>"


def safe_util(reserve_gib=6.0, cap=0.55, floor=0.22):
    """按当前空闲显存留出 reserve_gib 余量，再折算 gpu_memory_utilization。

    crater 上可能同时有别的 lab 在跑（本 lab 不挤占其它任务），所以预算按
    「当前空闲 - 余量」算；floor 保证权重(3.2 GiB)+激活+KV 至少装得下。
    """
    import torch
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    util = (free / gib - reserve_gib) / (total / gib)
    print(f"  [mem] free={free / gib:.2f} GiB total={total / gib:.2f} GiB "
          f"-> gpu_memory_utilization={max(min(cap, util), floor):.4f}", flush=True)
    return max(min(cap, util), floor)


# ------------------------------------------------------------------ 分派打点
DISPATCH_LOG: list[dict] = []
_ADDR_LOG: dict = {}


def install_hooks():
    """给 CudaGraphManager.dispatch 打点；记录候选表与最终选择。"""
    from vllm.v1.worker.gpu import cudagraph_utils as cu

    orig = cu.CudaGraphManager.dispatch

    def patched(self, num_reqs, num_tokens, uniform_token_count,
                num_active_loras, max_query_len=None):
        desc = orig(self, num_reqs, num_tokens, uniform_token_count,
                    num_active_loras, max_query_len)
        key = (num_tokens, self._resolve_effective_loras(num_active_loras))
        cands = self._candidates.get(key, [])
        DISPATCH_LOG.append({
            "num_reqs": num_reqs,
            "num_tokens": num_tokens,
            "uniform_token_count": uniform_token_count,
            "max_query_len": max_query_len,
            "num_active_loras": num_active_loras,
            "candidates": [
                {"mode": c.cg_mode.name, "num_tokens": c.num_tokens,
                 "num_reqs": c.num_reqs,
                 "uniform_token_count": c.uniform_token_count,
                 "max_query_len": c.max_query_len}
                for c in cands[:8]
            ],
            "chosen_mode": desc.cg_mode.name,
            "chosen_num_tokens": desc.num_tokens,
            "chosen_num_reqs": desc.num_reqs,
            "chosen_uniform": desc.uniform_token_count,
            "graphs_captured": self._graphs_captured,
        })
        return desc

    cu.CudaGraphManager.dispatch = patched
    return cu


def runner_of(llm):
    return (llm.llm_engine.engine_core.engine_core
            .model_executor.driver_worker.worker.model_runner)


def make_llm(**kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(),
             max_model_len=4096, enable_prefix_caching=False,
             disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                            # noqa: BLE001
        pass
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def rand_ids(n, rng):
    return [rng.randint(1000, 60000) for _ in range(n)]


# ------------------------------------------------------------------------ A
def section_A():
    title("[A] 模式与捕获清单：当前版本实际有哪些图")
    import torch
    from vllm.config import CUDAGraphMode

    print("源码入口（运行时取得，非文档抄录）：")
    for mod, sym in [
        ("vllm.config.compilation", "CUDAGraphMode"),
        ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager"),
        ("vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager", "dispatch"),
        ("vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager", "_init_candidates"),
        ("vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager", "run_fullgraph"),
        ("vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager", "run_pw_graph"),
        ("vllm.v1.worker.gpu.cudagraph_utils", "ModelCudaGraphManager"),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
    ]:
        print(f"  {mod}.{sym}\n      {src_loc(mod, sym)}")

    print("\nCUDAGraphMode 的取值与语义（枚举自带的方法决定 mode 的组合方式）：")
    for m in CUDAGraphMode:
        extra = ""
        if m.separate_routine():
            extra = (f"  separate_routine=True  decode_mode={m.decode_mode().name}"
                     f"  mixed_mode={m.mixed_mode().name}"
                     f"  max={m.max_cudagraph_mode().name}")
        print(f"  {m.name:<20} value={str(m.value):<18}"
              f" piecewise={m.has_piecewise_cudagraphs()}{extra}")

    llm = make_llm()
    cfg = llm.llm_engine.vllm_config
    cc = cfg.compilation_config
    mr = runner_of(llm)
    cgm = mr.cudagraph_manager

    sub("配置层")
    print(f"  compilation mode        : {cc.mode}")
    print(f"  configured cudagraph_mode: {cc.cudagraph_mode}")
    print(f"  cudagraph_capture_sizes : {len(cc.cudagraph_capture_sizes)} 个 "
          f"{cc.cudagraph_capture_sizes}")
    print(f"  max_cudagraph_capture_size: {cc.max_cudagraph_capture_size}")
    print(f"  compile_sizes           : {cc.compile_sizes}  "
          f"(空 => 不为特定尺寸单独编译)")
    print(f"  compile_ranges_endpoints: {cc.compile_ranges_endpoints}")
    print(f"  cudagraph_num_of_warmups: {cc.cudagraph_num_of_warmups}")
    print(f"  splitting_ops           : {len(cc.splitting_ops)} 个")
    for op in cc.splitting_ops:
        print(f"      {op}")
    print("  ↑ 这些算子把 torch.compile 的图切开：分段的边界就是 attention/状态更新，")
    print("    PIECEWISE 图在边界处停下，边界算子自己跑（图外）。")

    sub("运行时管理器（capture 之后的真实状态）")
    print(f"  runner                  : {type(mr).__module__}.{type(mr).__name__}")
    print(f"  cudagraph_manager       : {type(cgm).__name__}")
    print(f"  resolved cudagraph_mode : {cgm.cudagraph_mode}")
    print(f"  decode_query_len        : {cgm.decode_query_len}")
    print(f"  varlen_decode           : {cgm.varlen_decode}")
    print(f"  use_breakable_cg        : {cgm.use_breakable_cg}")
    print(f"  max_num_reqs (max_num_seqs): {cgm.max_num_reqs}")
    print(f"  async_scheduling        : {cfg.scheduler_config.async_scheduling}")
    print(f"  实际捕获的 FULL 图      : {sum(1 for d in cgm.graphs if d.cg_mode.name == 'FULL')}")
    print(f"  实际捕获的 PIECEWISE 图 : {sum(1 for d in cgm.graphs if d.cg_mode.name == 'PIECEWISE')}")

    sub("候选描述符（capture 前构好的表，dispatch 只查这张表）")
    for mode, descs in cgm._capture_descs.items():
        print(f"\n  {mode.name}: {len(descs)} 个描述符")
        print(f"    {'num_tokens':>11} {'num_reqs':>9} {'uniform':>8} {'max_q':>6}")
        for d in descs[:6]:
            print(f"    {d.num_tokens:>11} {str(d.num_reqs):>9} "
                  f"{str(d.uniform_token_count):>8} {str(d.max_query_len):>6}")
        if len(descs) > 6:
            print(f"    ... 共 {len(descs)} 行，最大 {descs[0].num_tokens}，"
                  f"最小 {descs[-1].num_tokens}")

    sub("padding 表（num_tokens -> 会被 pad 到的捕获尺寸）")
    print(f"  {'实际 token':>10} {'pad 到':>8} {'浪费':>7}   {'候选数':>7}")
    for n in [1, 16, 17, 23, 24, 25, 32, 33, 255, 256, 257, 511, 512, 513, 2048]:
        key = (n, 0)
        cands = cgm._candidates.get(key, [])
        best = None
        for c in cands:
            if c.num_tokens >= n and (best is None or c.num_tokens < best.num_tokens):
                best = c
        if best is None:
            print(f"  {n:>10} {'—':>8} {'—':>7}   {len(cands):>7}   -> NONE（无可用图）")
        else:
            print(f"  {n:>10} {best.num_tokens:>8} {best.num_tokens / n:>6.2f}×   "
                  f"{len(cands):>7}   -> {best.cg_mode.name}")
    shutdown(llm)


# ------------------------------------------------------------------------ B
def section_B():
    title("[B] 运行时分派轨迹：真实负载下每一步选中了什么")

    llm = make_llm()
    from vllm import SamplingParams, TokensPrompt

    sub("B1 纯 decode（B=8，prompt 64，gen 24）")
    DISPATCH_LOG.clear()
    rng = random.Random(11)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(64, rng)) for _ in range(8)]
    sp = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    llm.generate(ps, sp, use_tqdm=False)
    _dump_dispatch("B1", limit=6)

    sub("B2 长短混批（1 条 3000-token prefill 与 8 条 decode 同批）")
    DISPATCH_LOG.clear()
    rng = random.Random(12)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(3000, rng))]
    ps += [TokensPrompt(prompt_token_ids=rand_ids(64, rng)) for _ in range(8)]
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    llm.generate(ps, sp, use_tqdm=False)
    _dump_dispatch("B2", limit=10)

    sub("B3 桶边界两侧：batch = 15/16/17/23/24/25/33/40")
    DISPATCH_LOG.clear()
    rows = []
    for B in [15, 16, 17, 23, 24, 25, 33, 40]:
        rng = random.Random(100 + B)
        ps = [TokensPrompt(prompt_token_ids=rand_ids(48, rng)) for _ in range(B)]
        sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        before = len(DISPATCH_LOG)
        llm.generate(ps, sp, use_tqdm=False)
        calls = DISPATCH_LOG[before:]
        # 只看纯 decode 的步（uniform_token_count == 1）
        dec = [c for c in calls if c["uniform_token_count"] == 1]
        last = dec[-1] if dec else (calls[-1] if calls else None)
        if last:
            rows.append((B, last["num_tokens"], last["chosen_mode"],
                         last["chosen_num_tokens"], last["chosen_num_reqs"]))
    print(f"  {'请求数':>6} {'本步 token':>10} {'选中 mode':>12} "
          f"{'图 token':>9} {'图 reqs':>8}")
    for B, nt, mode, cgt, cgr in rows:
        print(f"  {B:>6} {nt:>10} {mode:>12} {cgt:>9} {str(cgr):>8}")

    sub("B4 超出最大捕获尺寸：单条 3000-token prefill")
    DISPATCH_LOG.clear()
    rng = random.Random(13)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(3000, rng))]
    sp = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    llm.generate(ps, sp, use_tqdm=False)
    _dump_dispatch("B4", limit=4)

    print("\n  分派规则来自 CudaGraphManager.dispatch：")
    print("    1) num_tokens > max_cudagraph_capture_size 或表里没这个 key -> NONE")
    print("    2) 否则按候选优先级取第一个 _is_compatible 的描述符")
    print("    3) FULL 要求 uniform_token_count 与 num_reqs 都匹配，")
    print("       PIECEWISE 的 num_reqs=None / uniform_token_count=None 可以吃任意请求数")
    print("  所以 3000-token prefill 落在 NONE（超出 512），而不是被 pad 到 512。")
    shutdown(llm)


def _dump_dispatch(tag, limit):
    print(f"  {'num_tokens':>10} {'uniform':>8} {'chosen':>10} "
          f"{'图 token':>9} {'候选数':>7}")
    shown = 0
    for c in DISPATCH_LOG:
        if shown >= limit:
            break
        print(f"  {c['num_tokens']:>10} {str(c['uniform_token_count']):>8} "
              f"{c['chosen_mode']:>10} {c['chosen_num_tokens']:>9} "
              f"{len(c['candidates']):>7}")
        shown += 1
    modes = {}
    for c in DISPATCH_LOG:
        modes[c["chosen_mode"]] = modes.get(c["chosen_mode"], 0) + 1
    print(f"  [{tag}] 共 {len(DISPATCH_LOG)} 次 dispatch 调用，"
          f"mode 分布 {modes}")


# ------------------------------------------------------------------------ C
def section_C(tamper=False):
    title("[C] 指针与图池：图绑定了什么，代价是多少")

    if tamper:
        _tamper_pointer()
        return

    llm = make_llm()
    from vllm import SamplingParams, TokensPrompt
    cfg = llm.llm_engine.vllm_config
    mr = runner_of(llm)
    cgm = mr.cudagraph_manager
    bufs = mr.input_buffers

    sub("C1 静态输入缓冲区的地址在多次请求之间是否变化")
    print(f"  input_buffers.input_ids  shape={tuple(bufs.input_ids.shape)} "
          f"dtype={bufs.input_ids.dtype}")
    rng = random.Random(21)
    seen = []
    for i in range(3):
        ps = [TokensPrompt(prompt_token_ids=rand_ids(64, rng)) for _ in range(16)]
        sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        llm.generate(ps, sp, use_tqdm=False)
        seen.append((bufs.input_ids.data_ptr(), bufs.positions.data_ptr(),
                     bufs.is_padding.data_ptr()))
        print(f"  第 {i + 1} 轮请求后 input_ids.ptr={seen[-1][0]:#x} "
              f"positions.ptr={seen[-1][1]:#x} is_padding.ptr={seen[-1][2]:#x}")
    same = len(set(seen)) == 1
    print(f"  => 三轮之间地址{'完全一致' if same else '发生了变化'}")
    print("  含义：图捕获时记录的是 input_buffers.input_ids[:num_tokens] 的地址，")
    print("        运行时把新 token 写进同一块缓冲区，所以重放不用改指针。")

    sub("C2 每步的输入地址与图内约定是否一致")
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    print(f"  capture 侧绑定: {src_loc('vllm.v1.worker.gpu.cudagraph_utils.ModelCudaGraphManager', 'capture')}")
    print(f"  dispatch 侧选图: {src_loc('vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager', 'dispatch')}")
    print(f"  replay 侧: {src_loc('vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager', 'run_fullgraph')}")
    print("  负向对照见 --tamper-pointer：把运行时缓冲区换成新地址，重放仍读旧地址。")

    sub("C3 图池代价：捕获数量、图内存、KV 容量")
    n_full = sum(1 for d in cgm.graphs if d.cg_mode.name == "FULL")
    n_pw = sum(1 for d in cgm.graphs if d.cg_mode.name == "PIECEWISE")
    kv = mr.kv_cache_config
    nb = getattr(kv, "num_blocks", None)
    kv_tokens = getattr(kv, "num_tokens", None)
    print(f"  捕获的图        : PIECEWISE {n_pw} + FULL {n_full} = {n_pw + n_full}")
    print(f"  graph pool      : {cgm.pool}")
    print(f"  KV cache blocks : {nb}")
    if kv_tokens:
        print(f"  KV cache tokens : {kv_tokens}")
    print(f"  权重/non-torch  : 见引擎启动日志 'Graph capturing finished' 与")
    print(f"                    'CUDA graph pool memory'（由 run 脚本 grep 落盘）")
    shutdown(llm)

    sub("C4 对照：同一负载下 eager 与开图的启动时间/KV 容量")
    rows = []
    for label, kw in [("eager", dict(enforce_eager=True)),
                      ("graph(FULL_AND_PIECEWISE)", {})]:
        torch_free0 = None
        import torch
        torch.cuda.empty_cache()
        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        l2 = make_llm(**kw)
        t_init = time.perf_counter() - t0
        m2 = runner_of(l2)
        kv2 = m2.kv_cache_config
        cgm2 = m2.cudagraph_manager
        rows.append((label, t_init, getattr(kv2, "num_blocks", None),
                     getattr(kv2, "num_tokens", None),
                     len(cgm2.graphs) if cgm2 is not None else 0))
        print(f"  {label:<26} 启动 {t_init:>6.2f}s  KV blocks "
              f"{getattr(kv2, 'num_blocks', None)}  KV tokens "
              f"{getattr(kv2, 'num_tokens', None)}  捕获图 "
              f"{len(cgm2.graphs) if cgm2 is not None else 0}")
        shutdown(l2)
    if len(rows) == 2 and rows[0][3] and rows[1][3]:
        print(f"\n  开图让可服务的 KV token 从 {rows[0][3]:,} 变成 {rows[1][3]:,}"
              f"（{(rows[1][3] - rows[0][3]) / rows[0][3]:+.2%}），")
        print(f"  启动时间从 {rows[0][1]:.2f}s 变成 {rows[1][1]:.2f}s。")
        print("  这两项才是图的真实代价：显存不是『消失了』，是从 KV 池里扣走的。")


def _tamper_pointer():
    """负向实验：运行时换掉静态输入缓冲区，看 FULL 图重放读谁。"""
    sub("C5 负向实验：把运行时输入缓冲区换成新地址")
    llm = make_llm()
    import torch
    from vllm import SamplingParams, TokensPrompt
    mr = runner_of(llm)
    bufs = mr.input_buffers
    cgm = mr.cudagraph_manager

    rng = random.Random(31)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(64, rng)) for _ in range(16)]
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    base = llm.generate(ps, sp, use_tqdm=False)
    base_tokens = [o.outputs[0].token_ids for o in base]

    used = [d for d in cgm.graphs if d.cg_mode.name == "FULL"]
    print(f"  该负载用了 {len(used)} 张 FULL 图；最常用的描述符 num_tokens="
          f"{sorted(d.num_tokens for d in used)[:5]} ...")

    old_ptr = bufs.input_ids.data_ptr()
    old_input = bufs.input_ids.clone()
    new_buf = torch.zeros_like(bufs.input_ids)
    bufs.input_ids = new_buf
    print(f"  旧 input_ids.ptr={old_ptr:#x}  ->  新 ptr={new_buf.data_ptr():#x}")
    try:
        out2 = llm.generate(ps, sp, use_tqdm=False)
        got = [o.outputs[0].token_ids for o in out2]
        same = got == base_tokens
        print(f"  换地址后输出与换之前{'一致' if same else '不一致'}")
        if not same:
            for i, (a, b) in enumerate(zip(base_tokens, got)):
                if a != b:
                    print(f"    请求 {i}: 原 {a[:6]}...  现在 {b[:6]}...")
                    break
        print("  解释：FULL 图重放时执行的是捕获期记录的命令流，读的是捕获时那块")
        print("        缓冲区的地址；运行时往新缓冲区写 token 不会改变图读到的东西。")
    except Exception as exc:                                     # noqa: BLE001
        print(f"  换地址后请求失败：{type(exc).__name__}: {exc}")
    finally:
        bufs.input_ids = old_input
    shutdown(llm)


# ------------------------------------------------------------------------ D
def section_D(mode, bs, gen, plen):
    """nsys 采样窗口 + 无插桩差分计时，供任务 G 把每步时间拆成 CPU 与 GPU。

    这一步同时给出两组数：
      * 无插桩的墙钟与进程 CPU 时间（差分到纯 decode），
      * cudaProfilerApi 窗口内的 nsys 记录（kernel device 时间可靠；
        注意 `--cuda-graph-trace=node` 会把 `cudaGraphLaunch` 的 CPU 时间
        从几十微秒放大到毫秒级，所以图模式的 CPU 提交时间不能取 API 时长，
        只取提交次数与 GPU 时间）。

    gen=0 表示只跑 prefill 窗口（max_tokens=1）。两个窗口相减得到纯 decode，
    和计时用的差分口径一致。

    四个对照点，用来把「torch.compile 分段」与「CUDA Graph」分开：
      NONE               enforce_eager=True，既不编译也不捕获
      COMPILE_ONLY       VLLM_COMPILE 编译，但 cudagraph_mode=NONE（只有分段）
      PIECEWISE          mixed_mode 段图，切分算子留在图外
      FULL_AND_PIECEWISE prefill 段图 + decode 整步一张图
    """
    import torch
    from vllm import SamplingParams, TokensPrompt

    if mode == "NONE":
        kw = dict(enforce_eager=True)
    elif mode == "COMPILE_ONLY":
        kw = dict(compilation_config={"cudagraph_mode": "NONE"})
    else:
        kw = dict(compilation_config={"cudagraph_mode": mode})
    llm = make_llm(**kw)
    rng = random.Random(41)
    ps = [TokensPrompt(prompt_token_ids=rand_ids(plen, rng)) for _ in range(bs)]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spG = SamplingParams(max_tokens=gen + 1, temperature=0.0, ignore_eos=True)
    llm.generate(ps, spG, use_tqdm=False)          # 预热 + 触发捕获
    llm.generate(ps, sp1, use_tqdm=False)          # A 窗口也要预热：
    # 第一次 max_tokens=1 会带上冷启动（autotune、工作区分配、图分派首次命中），
    # 不预热就会把这份开销算进 A 窗口，差分出负的 decode GPU 时间。

    # 无插桩差分：prefill 窗口与 prefill+decode 窗口分别计墙钟与进程 CPU 时间
    t0, c0 = time.perf_counter(), time.process_time()
    llm.generate(ps, sp1, use_tqdm=False)
    wall_a = time.perf_counter() - t0
    cpu_a = time.process_time() - c0
    if gen > 0:
        t0, c0 = time.perf_counter(), time.process_time()
        llm.generate(ps, spG, use_tqdm=False)
        wall_b = time.perf_counter() - t0
        cpu_b = time.process_time() - c0
        wall_ms = (wall_b - wall_a) / gen * 1000
        cpu_ms = (cpu_b - cpu_a) / gen * 1000
    else:
        wall_ms, cpu_ms = wall_a * 1000, cpu_a * 1000

    # 数窗口内真实发生了多少次模型 forward：GPU/CPU 时间要靠它折算到每步。
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    steps = {"n": 0, "tokens": 0}
    orig_exec = GPUModelRunner.execute_model

    def counting_exec(self, scheduler_output, *a, **kw):
        steps["n"] += 1
        steps["tokens"] += int(getattr(scheduler_output,
                                       "total_num_scheduled_tokens", 0) or 0)
        return orig_exec(self, scheduler_output, *a, **kw)

    profile_sp = spG if gen > 0 else sp1
    GPUModelRunner.execute_model = counting_exec
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    llm.generate(ps, profile_sp, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    GPUModelRunner.execute_model = orig_exec

    mr = runner_of(llm)
    cgm = mr.cudagraph_manager
    n = len(cgm.graphs) if cgm is not None else 0
    cc = llm.llm_engine.vllm_config.compilation_config
    print(f"MODE={mode} compile_mode={cc.mode} cudagraph_mode={cc.cudagraph_mode} "
          f"bs={bs} gen={gen} plen={plen} graphs={n} "
          f"STEPS={steps['n']} TOKENS={steps['tokens']} "
          f"WALL_MS={wall_ms:.4f} CPU_MS={cpu_ms:.4f}", flush=True)
    return llm


SECTIONS = {"A": section_A, "B": section_B, "C": section_C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=None)
    ap.add_argument("--tamper-pointer", action="store_true")
    ap.add_argument("--mode", default="FULL_AND_PIECEWISE")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--plen", type=int, default=128)
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}  model {MODEL}", flush=True)
    if args.tamper_pointer:
        install_hooks()
        _tamper_pointer()
        sys.stdout.flush()
        os._exit(0)

    want = [s.upper() for s in (args.sections or [])] or ["A", "B", "C"]
    if "D" in want:
        llm = section_D(args.mode, args.bs, args.gen, args.plen)
        shutdown(llm)
        sys.stdout.flush()
        os._exit(0)
    if "B" in want:
        # 只在需要分派轨迹时打点：打点会替换 dispatch，A 节要报它的原始行号。
        install_hooks()
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
