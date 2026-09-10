#!/usr/bin/env python3
"""L5.6 —— 结构化输出：在采样之前把非法 token 的 logit 打成 -inf。

约束解码不是"提示词写得好"，是**在 logits 上加一个掩码**：
用一台自动机跟踪当前状态，每一步算出"接下来哪些 token 合法"，
把其余全部置 -inf。所以它对格式是**硬保证**，不是概率上的倾向。

  [A] 不约束时 JSON 的有效率是多少（这个机制存在的理由）
  [B] 掩码长什么样：逐步打印词表里有多少 token 合法  <- 本章最重要的一节
  [C] 三个后端的吞吐代价：xgrammar / guidance / outlines
  [D] 语法编译的时间去哪了：首次 vs 命中缓存
  [E] 语法复杂度的影响：松 schema vs 紧 schema

用法：
    python structured_output.py          # 全部
    python structured_output.py B C      # 只跑某几节
"""

import json
import os
import statistics
import sys
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L56_MODEL", "Qwen/Qwen3-1.7B")

# 一个"松"的 schema：三个字段，字符串没有格式限制
SCHEMA_LOOSE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "city": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "city", "age"],
}

# 一个"紧"的 schema：枚举 + 正则 + 数值范围，合法 token 少得多
SCHEMA_TIGHT = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[A-Z][a-z]{2,9}$"},
        "city": {"type": "string", "enum": ["Beijing", "Shanghai", "Shenzhen"]},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
    },
    "required": ["name", "city", "age"],
    "additionalProperties": False,
}

PROMPT = ("Extract the person's info as JSON with keys name, city, age.\n"
          "Text: Zhang works in Shenzhen and is 31 years old.\n"
          "JSON: ")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def popcount(mask):
    """bitmask 是 int32 打包的位图：每 32 个 token 一个 int32。
    数 1 的个数 = 当前这一步合法的 token 数。"""
    return int(sum(bin(x & 0xFFFFFFFF).count("1")
                   for x in mask.flatten().tolist()))


def safe_util(reserve_gib=4.0, cap=0.55):
    """共享机器：只吃掉「空闲 - 预留」的那部分，且不超过 cap。"""
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def make_llm(**kw):
    from vllm import LLM
    d = dict(model=MODEL, gpu_memory_utilization=safe_util(),
             max_model_len=4096, enforce_eager=True,
             enable_prefix_caching=False, disable_log_stats=True)
    d.update(kw)
    return LLM(**d)


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- A
def classify(txt):
    """把一次无约束输出归到四类之一。

    第一版只用 json.loads 判 True/False，结果 0/64 —— 那个数字**会骗人**：
    模型其实写对了 JSON，只是后面还接着重复一遍、或者加了几句解说。
    「不会写 JSON」和「不会只写 JSON」是完全不同的两种失败，
    而约束解码主要解决的是后者。所以这里必须分开数。
    """
    txt = txt.strip()
    try:                                  # 整段就是一个 JSON
        obj = json.loads(txt)
        return ("整段合法", obj)
    except Exception:
        pass
    try:                                  # 开头是合法 JSON，后面有尾巴
        obj, end = json.JSONDecoder().raw_decode(txt)
        return ("前缀合法+尾巴", obj)
    except Exception:
        return ("根本不是 JSON", None)


def schema_ok(obj):
    return (isinstance(obj, dict)
            and {"name", "city", "age"} <= set(obj)
            and isinstance(obj.get("age"), int)
            and not isinstance(obj.get("age"), bool))


def part_a():
    """不加约束时，模型自己能吐出合法 JSON 吗？——要分清是哪一种失败。"""
    title("[A] 不约束时 JSON 的有效率")
    from vllm import SamplingParams
    llm = make_llm()
    n = 64
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=96)
    outs = llm.generate([PROMPT] * n, sp)

    kinds = {"整段合法": 0, "前缀合法+尾巴": 0, "根本不是 JSON": 0}
    strict_ok = 0          # 整段合法 且 满足 schema  <- 唯一能直接用的
    schema_bad = 0         # 能解析出对象但 schema 不对（例如 age 是字符串）
    examples = {}
    for o in outs:
        txt = o.outputs[0].text.strip()
        kind, obj = classify(txt)
        kinds[kind] += 1
        if obj is not None and not schema_ok(obj):
            schema_bad += 1
        if kind == "整段合法" and schema_ok(obj):
            strict_ok += 1
        examples.setdefault(kind, txt[:150])

    print(f"\n  样本数 {n}，同一个 prompt，temperature=0.8\n")
    print(f"  {'分类':<16} {'个数':>5} {'占比':>8}")
    for k, v in kinds.items():
        print(f"  {k:<16} {v:>5} {100*v/n:>7.1f}%")
    print(f"\n  能解析出对象但 schema 不对（如 age 是字符串）: {schema_bad}")
    print(f"  **整段合法 且 满足 schema（唯一能直接用的）: "
          f"{strict_ok}/{n} = {100*strict_ok/n:.1f}%**")
    sub("每类各一条原文（未加工）")
    for k, t in examples.items():
        print(f"  [{k}] {t!r}")
    shutdown(llm)


# ---------------------------------------------------------------- B
def part_b():
    """掩码到底长什么样——逐步数一数词表里有多少 token 合法。

    这一节不需要 GPU，也不需要 vLLM：直接用 xgrammar 的编译器 + matcher。
    """
    title("[B] 掩码的形状：每一步有多少 token 合法")
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = len(tok)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=V)
    compiler = xgr.GrammarCompiler(info)

    for label, schema in (("松 schema", SCHEMA_LOOSE), ("紧 schema", SCHEMA_TIGHT)):
        sub(f"{label}：走一遍一段合法输出，逐步统计")
        cg = compiler.compile_json_schema(json.dumps(schema))
        m = xgr.GrammarMatcher(cg)
        mask = xgr.allocate_token_bitmask(1, info.vocab_size)

        target = json.dumps({"name": "Zhang", "city": "Shenzhen", "age": 31},
                            separators=(",", ":"))
        ids = tok.encode(target, add_special_tokens=False)
        print(f"  目标串: {target}")
        print(f"  词表大小 V = {V}\n")
        print(f"  {'步':>3} {'合法 token 数':>13} {'占词表':>9}  {'唯一?':>5}  "
              f"{'接受的 token':<16}")
        rows = []
        for i, tid in enumerate(ids):
            m.fill_next_token_bitmask(mask, 0)
            allowed = popcount(mask)
            piece = tok.decode([tid])
            forced = "是" if allowed == 1 else ""
            rows.append((i, allowed, allowed / V, forced, piece))
            print(f"  {i:>3} {allowed:>13} {100*allowed/V:>8.2f}%  {forced:>5}  "
                  f"{piece!r:<16}")
            ok = m.accept_token(tid)
            if not ok:
                print(f"      !! matcher 拒绝了 token {tid} {piece!r}")
                break
        n_forced = sum(1 for r in rows if r[1] == 1)
        avg = statistics.mean(r[1] for r in rows)
        print(f"\n  共 {len(rows)} 步；**只有 1 个合法 token（被完全强制）的步数 "
              f"= {n_forced}**（{100*n_forced/len(rows):.0f}%）")
        print(f"  平均合法 token 数 = {avg:.0f} / {V}  "
              f"（{100*avg/V:.2f}%）")

    sub("对照：掩码本身占多少字节")
    print(f"  词表 {V} -> bitmask int32 数 = {(V + 31) // 32}"
          f" = {((V + 31) // 32) * 4} B / 请求 / 步")
    print(f"  而一份 logits 是 {V} × 4 B = {V * 4 / 1024:.0f} KB")
    print("  掩码比 logits 小 32 倍——所以传掩码而不是传下标列表。")


# ---------------------------------------------------------------- C
def part_c():
    """三个后端 + 无约束基线，同一个 schema 的吞吐代价。"""
    title("[C] 三个后端的代价")
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    n = 64
    results = []
    for backend in [None, "xgrammar", "guidance", "outlines"]:
        label = backend or "无约束"
        kw = {} if backend is None else {
            "structured_outputs_config": {"backend": backend}}
        try:
            llm = make_llm(**kw)
        except Exception as e:
            print(f"  {label:<10} 起不来: {type(e).__name__}: {str(e)[:90]}")
            continue
        sp = SamplingParams(temperature=0.0, max_tokens=64)
        if backend is not None:
            sp.structured_outputs = StructuredOutputsParams(
                json=SCHEMA_LOOSE)
        llm.generate([PROMPT], sp)                      # 预热 + 编译语法
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT] * n, sp)
        dt = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        valid = 0
        for o in outs:
            try:
                json.loads(o.outputs[0].text.strip())
                valid += 1
            except Exception:
                pass
        results.append((label, dt, toks / dt, valid, n))
        print(f"  {label:<10} {dt*1000:>8.1f} ms  {toks/dt:>9.1f} tok/s  "
              f"合法 {valid}/{n}")
        shutdown(llm)

    if results:
        base = next((r for r in results if r[0] == "无约束"), None)
        sub("相对无约束")
        for label, dt, tps, valid, n_ in results:
            rel = f"{tps/base[2]:.2f}×" if base else "-"
            print(f"  {label:<10} {tps:>9.1f} tok/s  {rel:>7}  "
                  f"合法率 {100*valid/n_:.1f}%")


# ---------------------------------------------------------------- D
def part_d():
    """语法编译的时间：第一次 vs 之后。"""
    title("[D] 语法编译的时间去哪了")
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = len(tok)

    t0 = time.perf_counter()
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=V)
    t_info = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiler = xgr.GrammarCompiler(info)
    t_comp = time.perf_counter() - t0

    print(f"\n  TokenizerInfo 构建   {t_info*1000:>9.2f} ms   （每个模型一次）")
    print(f"  GrammarCompiler 构建 {t_comp*1000:>9.2f} ms   （每个模型一次）")

    for label, schema in (("松 schema", SCHEMA_LOOSE), ("紧 schema", SCHEMA_TIGHT)):
        s = json.dumps(schema)
        ts = []
        for i in range(5):
            t0 = time.perf_counter()
            compiler.compile_json_schema(s)
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"\n  {label} compile_json_schema：")
        print(f"    第 1 次 {ts[0]:>8.2f} ms")
        print(f"    第 2-5 次 {statistics.mean(ts[1:]):>6.2f} ms "
              f"(min {min(ts[1:]):.2f}, max {max(ts[1:]):.2f})")
        print(f"    命中缓存后快 {ts[0]/max(statistics.mean(ts[1:]), 1e-9):.0f}×")

    sub("一个没见过的 schema 每次都要重编译")
    ts = []
    for i in range(5):
        s = json.dumps({"type": "object",
                        "properties": {f"f{i}_{j}": {"type": "string"}
                                       for j in range(6)}})
        t0 = time.perf_counter()
        compiler.compile_json_schema(s)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"  五个互不相同的 schema: "
          f"{', '.join(f'{t:.2f}' for t in ts)} ms  "
          f"(均值 {statistics.mean(ts):.2f})")


# ---------------------------------------------------------------- E
def part_e():
    """schema 紧一点，掩码算得更贵还是更便宜？"""
    title("[E] 语法复杂度的影响")
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = len(tok)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=V)
    compiler = xgr.GrammarCompiler(info)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)

    print(f"\n  {'schema':<12} {'编译 ms':>9} {'填掩码 µs/步':>14} "
          f"{'平均合法 token':>14}")
    for label, schema in (("松", SCHEMA_LOOSE), ("紧", SCHEMA_TIGHT)):
        s = json.dumps(schema)
        t0 = time.perf_counter()
        cg = compiler.compile_json_schema(s)
        t_c = (time.perf_counter() - t0) * 1000

        m = xgr.GrammarMatcher(cg)
        target = json.dumps({"name": "Zhang", "city": "Shenzhen", "age": 31},
                            separators=(",", ":"))
        ids = tok.encode(target, add_special_tokens=False)
        per_step, allowed_counts = [], []
        for tid in ids:
            t0 = time.perf_counter()
            m.fill_next_token_bitmask(mask, 0)
            per_step.append((time.perf_counter() - t0) * 1e6)
            allowed_counts.append(popcount(mask))
            if not m.accept_token(tid):
                break
        print(f"  {label:<12} {t_c:>9.2f} {statistics.median(per_step):>14.1f} "
              f"{statistics.mean(allowed_counts):>14.0f}")

    print("\n  注：填掩码在 CPU 上做，和 GPU 的前向可以重叠——"
          "所以它是否进入关键路径，取决于引擎有没有把它挪到别的线程。")


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}

if __name__ == "__main__":
    want = [a.upper() for a in sys.argv[1:]] or list(PARTS)
    print(f"model = {MODEL}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU 空闲 {free/1024**3:.1f} / {total/1024**3:.1f} GiB "
          f"-> gpu_memory_utilization = {safe_util():.3f}")
    for k in want:
        if k in PARTS:
            PARTS[k]()
        else:
            print(f"没有这一节: {k}")
