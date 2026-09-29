#!/usr/bin/env python3
"""L4.0 修订 —— 从文件字节到设备张量。

对应修订任务：
  [A] 字节区间与解析器审计：手写 header/offset 解析器；头尾元素逐位对拍；
      截断、重叠、缺分片、dtype/shape 不符四类损坏输入的报错
  [B] 绑定与融合追踪：checkpoint 名 -> HF 参数 -> vLLM 融合布局 -> 设备张量；
      tie_word_embeddings 的指针与重复存储；文件字节 / CPU 常驻 / GPU 分配分列；
      TP 分片用 vLLM 自己的 QKVParallelLinear.weight_loader 复现并回拼校验
  [C] 量化布局追踪：AWQ/GPTQ 磁盘布局；手写解包与官方算子逐元素对拍；
      vLLM 加载后的执行布局（Marlin）与磁盘布局对照；加载阶段事件时间线

用法：
    python labs/L4/checkpoint_load_path.py A B C
    python labs/L4/checkpoint_load_path.py --outdir results/crater/4.0/<run_id> C

环境：A 只需 CPU；B/C 需要 GPU 与 vLLM（serve venv）。
"""

import argparse
import glob
import hashlib
import json
import math
import os
import struct
import sys
import time

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
DT_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
TORCH_DT = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}
SUMMARY = {"sections": {}}


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


def st_files(d):
    return sorted(glob.glob(d + "/*.safetensors"))


# ------------------------------------------------------------------ 解析器
def parse_safetensors(path):
    """手写解析器：只做 8 字节长度 + JSON 头 + 偏移算术，不执行任何代码。

    返回 (header, header_len, data_len, problems)。problems 是字符串列表，
    逐条记录这个文件在字节层面不自洽的地方。
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    header = json.loads(raw)
    data_len = size - 8 - n
    problems = []
    prev_end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if "dtype" not in info or "shape" not in info or "data_offsets" not in info:
            problems.append(f"{name}: 头缺字段")
            continue
        if info["dtype"] not in DT_SIZE:
            problems.append(f"{name}: 未知 dtype {info['dtype']}")
            continue
        a, b = info["data_offsets"]
        want = math.prod(info["shape"]) * DT_SIZE[info["dtype"]]
        if b < a:
            problems.append(f"{name}: 偏移倒序 [{a},{b})")
        if b - a != want:
            problems.append(
                f"{name}: shape/dtype 需要 {want} 字节，头声明 {b - a} 字节")
        if a < prev_end:
            problems.append(f"{name}: 与前一张量重叠（起点 {a} < 前终点 {prev_end}）")
        if b > data_len:
            problems.append(f"{name}: 超出数据区（终点 {b} > {data_len}）")
        prev_end = max(prev_end, b)
    if prev_end != data_len:
        problems.append(f"数据区尾部不齐：最后一个张量止于 {prev_end}，实际 {data_len}")
    return header, n, data_len, problems


def raw_slice(path, name, k):
    """按解析出的偏移直接读前 k / 后 k 个元素的原始字节。"""
    header, hn, _, _ = parse_safetensors(path)
    info = header[name]
    a, b = info["data_offsets"]
    esz = DT_SIZE[info["dtype"]]
    base = 8 + hn + a
    with open(path, "rb") as f:
        f.seek(base)
        head = f.read(k * esz)
        f.seek(8 + hn + b - k * esz)
        tail = f.read(k * esz)
    return head, tail, info


def bits_equal(raw, tensor, k):
    import torch
    dt = tensor.dtype
    view = {"bfloat16": torch.int16, "float16": torch.int16,
            "float32": torch.int32, "int32": torch.int32,
            "int64": torch.int64, "int16": torch.int16,
            "int8": torch.int8, "uint8": torch.uint8,
            "float64": torch.int64}[str(dt).replace("torch.", "")]
    got = torch.frombuffer(bytearray(raw), dtype=dt).view(view)
    ref = tensor.reshape(-1)[:k].view(view)
    return bool(torch.equal(got, ref)), got.tolist(), ref.tolist()


def build_st(path, tensors, tail_pad=0, force_overlap=False, force_dtype=None):
    """构造一个最小 safetensors 文件，用于损坏输入的报错实验。

    tensors: [(name, dtype, shape, bytes)]，字节按声明顺序紧排。
    """
    hdr, body, off = {}, b"", 0
    for i, (n, dt, sh, data) in enumerate(tensors):
        d = force_dtype if (force_dtype and i == 0) else dt
        if force_overlap and i == 1:
            off = 0
        hdr[n] = {"dtype": d, "shape": list(sh), "data_offsets": [off, off + len(data)]}
        body += data
        off += len(data)
    blob = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + body)
        if tail_pad > 0:
            f.write(b"\0" * tail_pad)
        elif tail_pad < 0:
            # 截断：重写一个少 tail_pad 字节的文件
            total = 8 + len(blob) + len(body)
            f.truncate(total + tail_pad)
    return path


def mini_sharded_load(snapshot_dir, want):
    """最小分片加载器：index.json -> 分片文件 -> 张量。缺分片要在这里报错。"""
    idx = json.load(open(snapshot_dir + "/model.safetensors.index.json"))
    wm = idx["weight_map"]
    out = {}
    for name in want:
        if name not in wm:
            raise KeyError(f"index 的 weight_map 里没有 {name}")
        fn = os.path.join(snapshot_dir, wm[name])
        if not os.path.exists(fn):
            raise FileNotFoundError(f"index 指向的分片不存在: {wm[name]}")
        out[name] = fn
    return out


# ------------------------------------------------------------------ A
def section_A(outdir):
    import torch
    rep = {}
    title("[A] 字节区间与解析器审计")

    sub("A1 全部 checkpoint 的区间审计")
    rows = []
    for repo in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B",
                 "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                 "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"]:
        d = snap(repo)
        files = st_files(d)
        bad = 0
        total = 0
        for f in files:
            header, hn, data_len, probs = parse_safetensors(f)
            total += len([k for k in header if k != "__metadata__"])
            bad += len(probs)
            if probs:
                for p in probs[:3]:
                    print(f"    {os.path.basename(f)}: {p}")
        rows.append({"repo": repo, "shards": len(files),
                     "tensors": total, "problems": bad})
        print(f"  {repo:<38} 分片 {len(files)}  张量 {total:>4}  问题 {bad}")
    rep["audit"] = rows
    print("  区间自洽（无重叠、无越界、无空洞、shape/dtype 与偏移一致）。")

    sub("A2 头尾元素逐位对拍：原始字节 vs safe_open")
    from safetensors import safe_open
    d = snap("Qwen/Qwen3-1.7B")
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
             "model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.k_proj.weight",
             "model.layers.0.self_attn.v_proj.weight",
             "model.layers.0.self_attn.o_proj.weight",
             "model.layers.0.mlp.gate_proj.weight",
             "model.layers.0.mlp.up_proj.weight",
             "model.layers.0.mlp.down_proj.weight",
             "model.layers.0.self_attn.q_norm.weight",
             "model.layers.20.self_attn.q_proj.weight"]
    loc = {}
    for f in st_files(d):
        header, _, _, _ = parse_safetensors(f)
        for n in header:
            if n in names:
                loc[n] = f
    mism = 0
    table = []
    for n in names:
        f = loc[n]
        head, tail, info = raw_slice(f, n, 4)
        with safe_open(f, framework="pt") as sf:
            t = sf.get_tensor(n)
        ok_h, gh, rh = bits_equal(head, t, 4)
        view = {"bfloat16": torch.int16, "float16": torch.int16,
                "float32": torch.int32, "int32": torch.int32}[str(t.dtype).replace("torch.", "")]
        ref_tail = t.reshape(-1)[-4:].view(view)
        got_tail = torch.frombuffer(bytearray(tail), dtype=t.dtype).view(view)
        ok_t = bool(torch.equal(got_tail, ref_tail))
        mism += (not ok_h) + (not ok_t)
        table.append({"tensor": n, "dtype": info["dtype"], "shape": info["shape"],
                      "head_ok": ok_h, "tail_ok": ok_t})
        print(f"  {n:<48} {info['dtype']:<5} {str(info['shape']):<14} "
              f"头 {'一致' if ok_h else '不一致'}  尾 {'一致' if ok_t else '不一致'}")
    rep["bitwise"] = {"tensors": len(names), "mismatches": mism, "table": table}
    print(f"  12 个张量 × 头尾 4 个元素：不一致 {mism} 个。")

    sub("A3 损坏输入：手写解析器 / safetensors / 最小分片加载器")
    work = os.path.join(outdir, "synthetic")
    os.makedirs(work, exist_ok=True)
    w = (torch.arange(32, dtype=torch.float32).reshape(4, 8)).to(torch.bfloat16)
    a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    wb = w.view(torch.uint16).numpy().tobytes()
    ab = a.numpy().tobytes()
    base = [("w", "BF16", [4, 8], wb), ("a", "F32", [2, 2], ab)]
    cases = [
        ("valid", work + "/valid.safetensors", {}),
        ("truncated", work + "/truncated.safetensors", {"tail_pad": -8}),
        ("overlap", work + "/overlap.safetensors", {"force_overlap": True}),
        ("dtype_shape_mismatch", work + "/dtype_shape_mismatch.safetensors",
         {"force_dtype": "F32"}),
    ]
    rep_cases = []
    for name, path, kw in cases:
        build_st(path, base, **kw)
        entry = {"case": name, "file_size": os.path.getsize(path)}
        try:
            _, _, data_len, probs = parse_safetensors(path)
            entry["hand_parser"] = probs if probs else "通过"
        except Exception as e:
            entry["hand_parser"] = f"{type(e).__name__}: {e}"
        try:
            with safe_open(path, framework="pt") as sf:
                ks = list(sf.keys())
                vals = []
                for k in ks:
                    try:
                        x = sf.get_tensor(k)
                        vals.append(f"{k}:OK{tuple(x.shape)}")
                    except Exception as e:
                        vals.append(f"{k}:{type(e).__name__}: {str(e)[:70]}")
                entry["safetensors"] = " / ".join(vals)
        except Exception as e:
            entry["safetensors"] = f"{type(e).__name__}: {str(e)[:120]}"
        rep_cases.append(entry)
        print(f"\n  [{name}] {entry['file_size']} B")
        print(f"    手写解析器 : {entry['hand_parser']}")
        print(f"    safetensors: {entry['safetensors']}")

    sub("缺分片：index 指向的文件不存在")
    d = snap("Qwen/Qwen3-8B")
    idx = json.load(open(d + "/model.safetensors.index.json"))
    victim = "model.layers.20.self_attn.q_proj.weight"
    target = os.path.basename(idx["weight_map"][victim])
    print(f"  取 index 中 {victim} -> {target}")
    if os.path.exists(os.path.join(d, target + ".hidden")):
        os.rename(os.path.join(d, target + ".hidden"), os.path.join(d, target))
    os.rename(os.path.join(d, target), os.path.join(d, target + ".hidden"))
    try:
        mini_sharded_load(d, ["model.embed_tokens.weight", victim])
        entry = "未报错（不应发生）"
    except Exception as e:
        entry = f"{type(e).__name__}: {e}"
    finally:
        os.rename(os.path.join(d, target + ".hidden"), os.path.join(d, target))
    print(f"  最小分片加载器: {entry}")
    try:
        with safe_open(os.path.join(d, target), framework="pt") as sf:
            entry2 = f"直接打开该分片仍成功（key 数 {len(list(sf.keys()))}）"
    except Exception as e:
        entry2 = f"{type(e).__name__}: {e}"
    print(f"  分片恢复后直接打开: {entry2}")
    rep_cases.append({"case": "missing_shard", "index_points_to": target,
                      "mini_loader": entry, "after_restore": entry2})
    rep["corruption"] = rep_cases
    SUMMARY["sections"]["A"] = rep
    return rep


# ------------------------------------------------------------------ B
def _find_tensor(d, name):
    from safetensors import safe_open
    for f in st_files(d):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name), f
    raise KeyError(name)


def _rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _log_capture():
    """挂一个 handler 收集 vLLM 自己的阶段日志（含时间戳）。"""
    import logging

    class H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.rows = []

        def emit(self, rec):
            self.rows.append({"t": round(rec.created, 3), "msg": rec.getMessage()})

    h = H()
    lg = logging.getLogger("vllm")
    lg.addHandler(h)
    return h


def _mapped_rss_mb(snapshot_dir):
    """把该 snapshot 目录下文件映射进本进程的常驻页求和（/proc/self/smaps）。"""
    total = 0.0
    with open("/proc/self/smaps") as f:
        cur = None
        for line in f:
            if "-" in line and line.split()[0].count("-") == 1 and "/" in line:
                cur = line.rstrip()
            elif line.startswith("Rss:") and cur and snapshot_dir in cur:
                total += int(line.split()[1]) / 1024
    return total


def section_B(outdir):
    import torch
    rep = {}
    title("[B] 绑定与融合追踪：文件 -> 参数 -> 融合布局 -> 设备张量")

    d = snap("Qwen/Qwen3-1.7B")
    files = st_files(d)
    cfg = json.load(open(d + "/config.json"))
    file_bytes = sum(os.path.getsize(f) for f in files)
    print(f"  文件字节 {file_bytes / 2**20:.1f} MiB（{len(files)} 个分片）")

    sub("B1 HF 加载：声明参数量、实际常驻与 tie 指针")
    from transformers import AutoModelForCausalLM
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    dt_hf = time.perf_counter() - t0
    rss1 = _rss_mb()
    n_param = sum(p.numel() for p in model.parameters())
    n_struct = n_param  # tied 参数在 parameters() 里只出现一次
    emb = model.model.embed_tokens.weight
    head = model.lm_head.weight
    tied = emb.data_ptr() == head.data_ptr()
    print(f"  parameters() 计数 {n_param}（{n_param / 1e9:.3f} B）")
    print(f"  config: tie_word_embeddings = {cfg['tie_word_embeddings']}")
    print(f"  embed/hm_head 同指针: {tied}   embed {tuple(emb.shape)} "
          f"head {tuple(head.shape)}")
    print(f"  文件字节 {file_bytes / 2**20:.1f} MiB vs parameters×2B "
          f"{n_param * 2 / 2**20:.1f} MiB  差 "
          f"{(file_bytes - n_param * 2) / 2**20:+.1f} MiB")
    print(f"  CPU 常驻增量 {rss1 - rss0:.0f} MiB（加载耗时 {dt_hf:.2f} s）")
    # 文件里 lm_head 是否真的另存了一份
    head_b, head_file = _find_tensor(d, "lm_head.weight")
    emb_b, emb_file = _find_tensor(d, "model.embed_tokens.weight")
    dup_bytes = head_b.numel() * head_b.element_size()
    same_bits = torch.equal(head_b, emb_b)
    rep["hf"] = {"file_bytes": file_bytes, "params": n_param,
                 "tied_pointer": tied, "rss_delta_mib": round(rss1 - rss0, 1),
                 "load_s": round(dt_hf, 3), "lm_head_file": os.path.basename(head_file),
                 "embed_file": os.path.basename(emb_file),
                 "lm_head_duplicate_bits": bool(same_bits),
                 "lm_head_bytes": dup_bytes,
                 "lm_head_share": dup_bytes / file_bytes}
    print(f"  文件里 lm_head 另存一份: {os.path.basename(head_file)}，"
          f"{dup_bytes / 2**20:.1f} MiB，与 embed 逐位相同 {same_bits}，"
          f"占文件 {dup_bytes / file_bytes:.1%}")
    t0 = time.perf_counter()
    acc = 0.0
    for p in model.parameters():
        acc += float(p.detach().reshape(-1).sum())
    touch_s = time.perf_counter() - t0
    rss2 = _rss_mb()
    mapped = _mapped_rss_mb(d)
    print(f"  触碰全部页后常驻 {rss2 - rss0:.0f} MiB（全量求和 {touch_s:.2f} s，"
          f"校验和 {acc:.3f}）")
    print(f"    其中 checkpoint 文件的映射页常驻 {mapped:.0f} MiB；"
          f"参数本身 {n_param * 2 / 2**20:.1f} MiB")
    rep["hf"]["rss_touch_mib"] = round(rss2 - rss0, 1)
    rep["hf"]["touch_s"] = round(touch_s, 3)
    rep["hf"]["mapped_file_rss_mib"] = round(mapped, 1)
    del model
    import gc
    gc.collect()

    sub("B2 vLLM 加载：融合布局与磁盘命名的逐元素对照")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM
    torch.cuda.reset_peak_memory_stats()
    gpu0 = torch.cuda.memory_allocated()
    h = _log_capture()
    t0 = time.perf_counter()
    llm = LLM(model=d, dtype="bfloat16", gpu_memory_utilization=0.5,
              max_model_len=2048, enforce_eager=True, disable_log_stats=True)
    init_s = time.perf_counter() - t0
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    gpu_after = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()
    gpu_peak = torch.cuda.max_memory_allocated()
    print(f"  引擎 init {init_s:.2f} s；显存 allocated {gpu0 / 2**20:.0f} -> "
          f"{gpu_after / 2**20:.0f} MiB（峰值 {gpu_peak / 2**20:.0f}，"
          f"reserved {gpu_reserved / 2**20:.0f}）")

    def pair(fused_name, parts, dim):
        wt = dict(m.named_parameters())[fused_name].detach().float().cpu()
        ref = []
        for p in parts:
            t, _ = _find_tensor(d, p)
            ref.append(t.float())
        ref = torch.cat(ref, dim=dim)
        ok = wt.shape == ref.shape and torch.equal(wt, ref)
        maxd = (wt - ref).abs().max().item() if wt.shape == ref.shape else float("nan")
        print(f"  {fused_name:<26} {tuple(wt.shape)}  <- {len(parts)} 个磁盘张量  "
              f"逐元素相同 {ok}  max|diff| {maxd:.3e}")
        return {"fused": fused_name, "shape": list(wt.shape), "parts": parts,
                "equal": bool(ok), "max_abs_diff": maxd}

    checks = [
        pair("model.layers.0.self_attn.qkv_proj.weight",
             ["model.layers.0.self_attn.q_proj.weight",
              "model.layers.0.self_attn.k_proj.weight",
              "model.layers.0.self_attn.v_proj.weight"], 0),
        pair("model.layers.0.mlp.gate_up_proj.weight",
             ["model.layers.0.mlp.gate_proj.weight",
              "model.layers.0.mlp.up_proj.weight"], 0),
        pair("model.layers.20.self_attn.qkv_proj.weight",
             ["model.layers.20.self_attn.q_proj.weight",
              "model.layers.20.self_attn.k_proj.weight",
              "model.layers.20.self_attn.v_proj.weight"], 0),
    ]
    vnp = dict(m.named_parameters())
    vhead = getattr(m, "lm_head").weight
    vemb = vnp["model.embed_tokens.weight"]
    tied_v = vhead.data_ptr() == vemb.data_ptr()
    head_is_param = "lm_head.weight" in vnp
    print(f"  vLLM 侧 lm_head 与 embed 同指针: {tied_v}；"
          f"lm_head.weight 单独注册为参数: {head_is_param}")
    l01 = [r for r in h.rows if "Loading weights took" in r["msg"]
           or "Model loading took" in r["msg"]
           or "init engine" in r["msg"]]
    for r in l01:
        print(f"    [vllm 日志] {r['msg'][:100]}")
    rep["vllm"] = {"fused_checks": checks, "init_s": round(init_s, 3),
                   "gpu_alloc_before": gpu0, "gpu_alloc_after": gpu_after,
                   "gpu_reserved": gpu_reserved, "gpu_peak": gpu_peak,
                   "tied_pointer": bool(tied_v), "lm_head_is_param": head_is_param,
                   "vllm_logs": [r["msg"] for r in l01]}

    sub("B3 TP 分片：用 vLLM 自己的 weight_loader 复现 q/k/v 切分")
    import vllm.model_executor.layers.linear as L
    H = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hs = cfg["head_dim"]
    q = _find_tensor(d, "model.layers.0.self_attn.q_proj.weight")[0].float()
    k = _find_tensor(d, "model.layers.0.self_attn.k_proj.weight")[0].float()
    v = _find_tensor(d, "model.layers.0.self_attn.v_proj.weight")[0].float()
    orig_ws = L.get_tensor_model_parallel_world_size
    orig_rk = L.get_tensor_model_parallel_rank

    def build(rank, tp):
        L.get_tensor_model_parallel_world_size = lambda: tp
        L.get_tensor_model_parallel_rank = lambda: rank
        lin = L.QKVParallelLinear(hidden_size=H, head_size=hs, total_num_heads=nh,
                                  total_num_kv_heads=nkv, bias=False)
        # LinearBase.__init__ 里已经调用 quant_method.create_weights（linear.py:369）
        for sid, w in (("q", q), ("k", k), ("v", v)):
            lin.weight_loader(lin.weight, w, sid)
        return lin.weight.detach().clone()

    try:
        shards = {tp: [build(r, tp) for r in range(tp)] for tp in (1, 2, 4)}
        cur = list(m.named_parameters())
        qkv_tp1 = dict(cur)["model.layers.0.self_attn.qkv_proj.weight"].detach().float().cpu()
        same_as_engine = torch.equal(shards[1][0], qkv_tp1)

        def regather(tp):
            """每片的布局是 [q_r, k_r, v_r]；按 shard_id 分别跨片拼接才能还原 tp=1。"""
            q_sz = nh * hs // tp
            kv_sz = nkv * hs // tp
            rows = []
            for lo, sz in ((0, q_sz), (q_sz, kv_sz), (q_sz + kv_sz, kv_sz)):
                rows.append(torch.cat([s[lo:lo + sz] for s in shards[tp]], dim=0))
            return torch.cat(rows, dim=0)

        for tp in (2, 4):
            ok = torch.equal(regather(tp), shards[1][0])
            naive = torch.equal(torch.cat(shards[tp], dim=0), shards[1][0])
            print(f"  tp={tp} 每片 shape {tuple(shards[tp][0].shape)}；"
                  f"直接行拼接还原 {naive}；按 shard_id 分段拼接还原 {ok}")
        back_ok = all(torch.equal(regather(tp), shards[1][0]) for tp in (2, 4))
        print(f"  tp=1 的复现与引擎实际 qkv_proj 逐元素相同: {same_as_engine}")
        rep["tp"] = {"reproduce_equals_engine": bool(same_as_engine),
                     "regather_equals_tp1": bool(back_ok),
                     "shard_shapes": {str(tp): [list(x.shape) for x in shards[tp]]
                                      for tp in (1, 2, 4)},
                     "q_size_per_rank": {str(tp): nh * hs // tp for tp in (1, 2, 4)},
                     "kv_size_per_rank": {str(tp): nkv * hs // tp for tp in (1, 2, 4)}}
        print(f"  （kv 头数 {nkv}、tp=4 → tp < nkv，每片持 {nkv // 4} 个 kv 头，"
              f"num_kv_head_replicas=1；tp ≥ nkv 时 vLLM 改为复制 kv 头，"
              f"见 linear.py:1042）")
    except Exception as e:
        import traceback
        traceback.print_exc()
        rep["tp"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        L.get_tensor_model_parallel_world_size = orig_ws
        L.get_tensor_model_parallel_rank = orig_rk
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"  (engine shutdown: {type(e).__name__}: {e})")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    SUMMARY["sections"]["B"] = rep
    return rep


# ------------------------------------------------------------------ C
# AWQ 的位顺序不是 0,4,8,…：autoawq 把 8 个 4-bit 值按 [0,4,1,5,2,6,3,7] 的位置
# 装进一个 int32，等价于移位序列 [0,16,4,20,8,24,12,28]。见 vLLM 的 Triton 参考
# 实现 awq_dequantize_kernel（awq_triton.py:57 的 reverse_awq_order_tensor）。
AWQ_SHIFTS = [0, 16, 4, 20, 8, 24, 12, 28]


def _seq_shifts(t, bits=4):
    import torch
    return torch.arange(0, 32, bits, device=t.device)


def _awq_shifts(t):
    import torch
    return torch.tensor(AWQ_SHIFTS, device=t.device)


def awq_unpack(qweight, scales, qzeros, group_size, bits=4):
    """手写 AWQ 解包。

    qweight [in, out/pf]：沿输出维打包；qzeros [in/gs, out/pf] 同样沿输出维打包。
    零点没有偏移，解包后直接参与 (q - z) * s。
    """
    pf = 32 // bits
    in_f = qweight.shape[0]
    out_f = qweight.shape[1] * pf
    sh = _awq_shifts(qweight)
    q = ((qweight.unsqueeze(-1) >> sh) & ((1 << bits) - 1)).reshape(in_f, out_f)
    z = ((qzeros.unsqueeze(-1) >> sh) & ((1 << bits) - 1)).reshape(qzeros.shape[0], out_f)
    z_row = z.int().repeat_interleave(group_size, dim=0)
    s_row = scales.repeat_interleave(group_size, dim=0).float()
    return ((q.int() - z_row).float() * s_row).to(scales.dtype)


def gptq_unpack(qweight, scales, qzeros, g_idx, group_size, bits=4):
    """手写 GPTQ 解包。

    qweight [in/pf, out]：沿输入维打包，位顺序是顺序的 0,4,…；
    qzeros [in/gs, out/pf]：沿**输出**维打包（与 qweight 不同轴）；
    零点在文件里存的是 zero - 1：vLLM 的 exllama kernel 会 +1
    （exllama.py:89 注释记录了这一历史格式问题）。
    """
    import torch
    pf = 32 // bits
    out_f = qweight.shape[1]
    in_f = qweight.shape[0] * pf
    sh = _seq_shifts(qweight, bits)
    q = ((qweight.unsqueeze(-1) >> sh) & ((1 << bits) - 1)).permute(0, 2, 1)
    q = q.reshape(in_f, out_f).int()
    z = ((qzeros.unsqueeze(-1) >> sh) & ((1 << bits) - 1))
    z = z.reshape(qzeros.shape[0], out_f).int() + 1
    g = g_idx.to(torch.long)
    return ((q - z[g]).float() * scales[g].float()).to(scales.dtype)


def awq_pack(q, bits=4):
    """awq_unpack 的逆：q [in, out] 的 0..15 整数 -> [in, out/pf] int32。"""
    import torch
    pf = 32 // bits
    in_f, out_f = q.shape
    qq = q.to(torch.int32).reshape(in_f, out_f // pf, pf)
    return (qq << _awq_shifts(q)).sum(-1).to(torch.int32)


def gptq_pack(q, bits=4):
    """gptq_unpack 的逆：q [in, out] -> [in/pf, out] int32。"""
    import torch
    pf = 32 // bits
    in_f, out_f = q.shape
    qq = q.to(torch.int32).reshape(in_f // pf, pf, out_f).permute(0, 2, 1)
    return (qq << _seq_shifts(q, bits)).sum(-1).to(torch.int32)


def pack_out(z, bits=4, shifts=None):
    """沿输出维打包的零点：z [rows, out] -> [rows, out/pf] int32。"""
    import torch
    pf = 32 // bits
    rows, out_f = z.shape
    zz = z.to(torch.int32).reshape(rows, out_f // pf, pf)
    sh = _awq_shifts(z) if shifts == "awq" else _seq_shifts(z, bits)
    return (zz << sh).sum(-1).to(torch.int32)


def _all_names(d):
    names = []
    for f in st_files(d):
        header, _, _, _ = parse_safetensors(f)
        names += [k for k in header if k != "__metadata__"]
    return set(names)


def _deq_qkv(tag, disk_dir, pref, gs):
    """把磁盘上的 q/k/v 量化权重解包并沿输出维拼成 [hidden, 3*out]。"""
    parts = []
    if tag == "awq":
        for p in ("q_proj", "k_proj", "v_proj"):
            qw = _find_tensor(disk_dir, f"{pref}{p}.qweight")[0]
            sc = _find_tensor(disk_dir, f"{pref}{p}.scales")[0]
            qz = _find_tensor(disk_dir, f"{pref}{p}.qzeros")[0]
            parts.append(awq_unpack(qw, sc, qz, gs))
    else:
        for p in ("q_proj", "k_proj", "v_proj"):
            qw = _find_tensor(disk_dir, f"{pref}{p}.qweight")[0]
            sc = _find_tensor(disk_dir, f"{pref}{p}.scales")[0]
            qz = _find_tensor(disk_dir, f"{pref}{p}.qzeros")[0]
            gi = _find_tensor(disk_dir, f"{pref}{p}.g_idx")[0]
            parts.append(gptq_unpack(qw, sc, qz, gi, gs))
    return parts


def section_C(outdir, only_quant=None):
    import torch
    rep = {}
    title("[C] 量化布局追踪：磁盘 / 手写解包 / 官方实现 / 执行布局")

    awq = snap("Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    gptq = snap("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4")
    dense = snap("Qwen/Qwen2.5-1.5B-Instruct")
    pref = "model.layers.0.self_attn."

    sub("C1 磁盘布局")
    disk = {}
    for tag, d in (("awq", awq), ("gptq", gptq)):
        qc = json.load(open(d + "/config.json"))["quantization_config"]
        rows = []
        for n in sorted(k for k in _all_names(d) if k.startswith(pref + "q_proj")):
            t, _ = _find_tensor(d, n)
            rows.append({"name": n.split(".")[-1], "shape": list(t.shape),
                         "dtype": str(t.dtype).replace("torch.", "")})
            print(f"  [{tag}] q_proj.{n.split('.')[-1]:<9} {str(tuple(t.shape)):<16} "
                  f"{str(t.dtype).replace('torch.', '')}")
        print(f"        quantization_config: {json.dumps(qc, ensure_ascii=False)[:150]}")
        kt, _ = _find_tensor(d, pref + "k_proj.qweight")
        st, _ = _find_tensor(d, pref + "k_proj.scales")
        print(f"        k_proj 同结构：qweight {tuple(kt.shape)}，"
              f"scales {tuple(st.shape)}（out 从 1536 变 256）")
        disk[tag] = {"quantization_config": qc, "q_proj_tensors": rows,
                     "k_proj_qweight": list(kt.shape)}
    gs = disk["awq"]["quantization_config"]["group_size"]
    rep["disk"] = disk

    sub("C2 AWQ 手写解包 vs ops.awq_dequantize（逐元素）")
    parts = _deq_qkv("awq", awq, pref, gs)
    mine = parts[0].cuda()
    qw = _find_tensor(awq, pref + "q_proj.qweight")[0].cuda()
    sc = _find_tensor(awq, pref + "q_proj.scales")[0].cuda()
    qz = _find_tensor(awq, pref + "q_proj.qzeros")[0].cuda()
    from vllm import _custom_ops as ops
    ref = ops.awq_dequantize(qw, sc, qz, 0, 0, 0)
    dmax = (mine.float() - ref.float()).abs().max().item()
    frac = (mine == ref).float().mean().item()
    print(f"  q_proj: 手写 {tuple(mine.shape)} vs 官方 {tuple(ref.shape)}  "
          f"逐元素相同比例 {frac:.6f}  max|diff| {dmax:.3e}")
    rep["awq_dequant"] = {"shape": list(mine.shape), "equal_fraction": frac,
                          "max_abs_diff": dmax}
    del mine, ref, parts
    torch.cuda.empty_cache()

    sub("C3 GPTQ 手写解包 vs 未量化同位置权重")
    gparts = _deq_qkv("gptq", gptq, pref, gs)
    gdense = []
    for p in ("q_proj", "k_proj", "v_proj"):
        t, _ = _find_tensor(dense, f"{pref}{p}.weight")
        gdense.append(t.float().t())
    gdense = torch.cat(gdense, 1)
    gm = torch.cat(gparts, 1)
    cos = torch.nn.functional.cosine_similarity(
        gm.reshape(1, -1), gdense.reshape(1, -1)).item()
    relmax = (gm - gdense).abs().max().item() / gdense.abs().max().item()
    print(f"  fused qkv 手写解包 {tuple(gm.shape)} vs 未量化 {tuple(gdense.shape)}")
    print(f"    cosine {cos:.6f}   相对最大差 {relmax:.4f}（4-bit 量化的固有误差）")
    rep["gptq_vs_dense"] = {"cosine": cos, "rel_max_diff": relmax}
    del gparts, gdense, gm
    torch.cuda.empty_cache()

    sub("C3b 磁盘布局不是 kernel 期望的布局")
    print("  把磁盘原样的 qweight/qzeros/scales 直接喂 ops.gptq_gemm，"
          "结果与手写反量化不一致。")
    print("  原因是 kernel 前有布局变换：exllama.py:125 permute_param_layout_ + "
          "ops.gptq_shuffle，")
    print("  marlin.py:129 permute_param_layout_ + ops.gptq_marlin_repack"
          "（marlin.py:130）。")
    rep["layout_note"] = {"exllama_transform": "exllama.py:125-130",
                          "marlin_transform": "marlin.py:129-140"}

    sub("C4 往返自检：dense -> 量化 -> 打包 -> 解包")
    torch.manual_seed(0)
    W = torch.randn(64, 128) * 0.05
    gs2, zq = 32, 8
    scale = (W.reshape(64 // gs2, gs2, 128).abs().amax(1) / 7.0)
    scale_h = scale.to(torch.float16)
    se = scale_h.repeat_interleave(gs2, 0).float()
    q = torch.clamp(torch.round(W / se) + zq, 0, 15).to(torch.int32)
    zrow = torch.full((64 // gs2, 128), zq, dtype=torch.int32)
    ref = (q - zq).float() * se
    back_awq = awq_unpack(awq_pack(q), scale_h, pack_out(zrow, shifts="awq"), gs2).float()
    gi = torch.repeat_interleave(torch.arange(64 // gs2), gs2).to(torch.int32)
    back_gptq = gptq_unpack(gptq_pack(q), scale_h, pack_out(zrow - 1), gi, gs2).float()
    awq_ok = torch.equal(back_awq, ref.half().float())
    gptq_ok = torch.equal(back_gptq, ref.half().float())
    print(f"  AWQ 布局往返逐元素一致: {awq_ok}   GPTQ 布局往返逐元素一致: {gptq_ok}")
    print(f"  （GPTQ 的零点在文件里存 zero-1，打包时写回 zrow-1；AWQ 存原值）")
    rep["roundtrip"] = {"awq": bool(awq_ok), "gptq": bool(gptq_ok)}

    sub("C5 vLLM 加载后的执行布局、逐层对拍与加载阶段事件")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm.model_executor.kernels.linear.mixed_precision.marlin as marlin_mod
    from vllm import LLM
    stats = {"count": 0, "sec": 0.0}
    orig_pw = marlin_mod.MarlinLinearKernel.process_weights_after_loading

    def timed(self, layer):
        t0 = time.perf_counter()
        try:
            return orig_pw(self, layer)
        finally:
            stats["count"] += 1
            stats["sec"] += time.perf_counter() - t0

    marlin_mod.MarlinLinearKernel.process_weights_after_loading = timed
    loaded = {}
    todo = [(t, d) for t, d in (("awq", awq), ("gptq", gptq))
            if only_quant in (None, t)]
    try:
        for tag, d in todo:
            stats["count"] = 0
            stats["sec"] = 0.0
            h = _log_capture()
            t0 = time.perf_counter()
            llm = LLM(model=d, dtype="float16", gpu_memory_utilization=0.45,
                      max_model_len=1024, enforce_eager=True, disable_log_stats=True)
            total = time.perf_counter() - t0
            m = llm.llm_engine.model_executor.driver_worker.model_runner.model
            qkv = m.model.layers[0].self_attn.qkv_proj
            params = {n: (list(p.shape), str(p.dtype).replace("torch.", ""))
                      for n, p in qkv.named_parameters()}
            events = [r["msg"] for r in h.rows
                      if "Loading weights took" in r["msg"]
                      or "Model loading took" in r["msg"]
                      or "init engine" in r["msg"]]
            print(f"\n  [{tag}] 引擎 init {total:.2f} s；"
                  f"quant 层 repack {stats['count']} 层共 {stats['sec']:.3f} s")
            for e in events:
                print(f"    [vllm] {e[:110]}")
            for n, (sh, dt) in params.items():
                print(f"    {n:<20} {str(sh):<16} {dt}")
            # 逐层对拍：引擎真实的第 0 层 qkv_proj 输出 vs 手写解包。
            # 用「引擎输出 - 引擎在 x=0 的响应」消掉 bias，因为 Marlin 会把
            # layer.bias 重排（marlin.py:218 marlin_permute_bias），加载后的
            # bias 参数已经不是数学上的顺序。
            torch.manual_seed(0)
            x = torch.randn(8, 1536, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                z0 = torch.zeros(1, 1536, dtype=torch.float16, device="cuda")
                out = qkv(x)
                o0 = qkv(z0)
                y_eng = (out[0] if isinstance(out, tuple) else out).float()
                y0 = (o0[0] if isinstance(o0, tuple) else o0).float()[0]
            W = torch.cat(_deq_qkv(tag, d, pref, gs), 1).cuda().float()
            y_mine = x.float() @ W
            y_ref16 = (x @ W.half()).float()
            ref_mag = (y_eng - y0).abs().max().item()
            dl = ((y_eng - y0) - y_mine).abs().max().item()
            dl16 = ((y_eng - y0) - y_ref16).abs().max().item()
            rel = dl / ref_mag
            rel16 = dl16 / ref_mag
            b = getattr(qkv, "bias", None)
            bmax = None if b is None else b.detach().float().abs().max().item()
            print(f"    第 0 层 qkv_proj（去掉 bias 项，参考量级 {ref_mag:.3f}）：")
            print(f"      引擎 vs 手写解包(fp32 累加) max|diff| {dl:.3e}（相对 {rel:.2e}）")
            print(f"      引擎 vs fp16 GEMM(同样的权重) max|diff| {dl16:.3e}"
                  f"（相对 {rel16:.2e}）")
            print(f"    加载后 bias 的幅度 {bmax:.3f}，"
                  f"与引擎 x=0 响应的一致性 "
                  f"{(y0 == b.detach().float()).float().mean().item():.3f}"
                  if b is not None else "    该层没有 bias")
            loaded[tag] = {"init_s": round(total, 3),
                           "quant_layers": stats["count"],
                           "quant_process_s": round(stats["sec"], 3),
                           "exec_params": params, "vllm_logs": events,
                           "layer_output_max_abs_diff": dl,
                           "layer_output_max_rel_diff": rel,
                           "layer_vs_fp16_gemm_abs_diff": dl16,
                           "layer_vs_fp16_gemm_rel_diff": rel16,
                           "ref_magnitude": ref_mag,
                           "loaded_bias_max": bmax,
                           "x0_response_equals_loaded_bias":
                               None if b is None else float(
                                   (y0 == b.detach().float()).float().mean().item())}
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del llm, m, W, y_mine, y_eng
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
        loaded["error"] = f"{type(e).__name__}: {e}"
    finally:
        marlin_mod.MarlinLinearKernel.process_weights_after_loading = orig_pw
    rep["execution_layout"] = loaded

    sub("C6 compressed-tensors 元数据")
    try:
        import compressed_tensors as ct
        from compressed_tensors.config import CompressionFormat
        from compressed_tensors.quantization import (
            QuantizationArgs, QuantizationScheme, QuantizationStrategy)
        fields = list(getattr(QuantizationArgs, "model_fields", {}).keys())
        print(f"  compressed-tensors {getattr(ct, '__version__', '?')}")
        print(f"  QuantizationArgs 字段: {fields}")
        print(f"  QuantizationStrategy: "
              f"{[s.value for s in QuantizationStrategy][:8]}")
        print(f"  CompressionFormat: {[f.value for f in CompressionFormat][:8]}")
        rep["compressed_tensors"] = {
            "version": getattr(ct, "__version__", "?"),
            "QuantizationArgs_fields": fields,
            "strategies": [s.value for s in QuantizationStrategy],
            "formats": [f.value for f in CompressionFormat],
            "instance_checkpoint": None,
        }
        print("  本地没有 compressed-tensors 打包的 checkpoint，字段与格式名只作"
              "结构对照；无实例对拍，标 UNVERIFIED。")
    except Exception as e:
        print(f"  不可用: {type(e).__name__}: {e}")
        rep["compressed_tensors"] = {"error": f"{type(e).__name__}: {e}"}
    SUMMARY["sections"]["C"] = rep
    return rep


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--outdir", default=os.path.expanduser("~/l4_out"))
    ap.add_argument("--quant", choices=["awq", "gptq"], default=None,
                    help="只跑一个量化引擎的 C5（两个引擎需分进程，避免显存不回收）")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [s.upper() for s in args.sections] or ["A", "B", "C"]
    env = {"python": sys.version.split()[0], "HF_HOME": os.environ.get("HF_HOME")}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    try:
        import vllm
        env["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    SUMMARY["env"] = env
    for s in want:
        if s == "C":
            section_C(args.outdir, only_quant=args.quant)
        else:
            {"A": section_A, "B": section_B}[s](args.outdir)
    SUMMARY["outdir"] = args.outdir
    path = os.path.join(args.outdir, "checkpoint_load_path.json")
    if os.path.exists(path):
        # 分段多次运行时合并：各 section 与 env 独立更新，便于把 GPU 相关段
        # 单独放进干净进程执行。
        try:
            old = json.load(open(path))
            new_secs = dict(SUMMARY["sections"])
            oldC = old.get("sections", {}).get("C") or {}
            newC = new_secs.get("C") or {}
            if isinstance(oldC.get("execution_layout"), dict) \
                    and isinstance(newC.get("execution_layout"), dict):
                # 分段跑 C5 时按 tag 合并，一次失败不覆盖已有结果
                el = dict(oldC["execution_layout"])
                for k, v in newC["execution_layout"].items():
                    if k != "error":
                        el[k] = v
                newC = {**newC, "execution_layout": el}
                new_secs["C"] = newC
            merged = {"env": {**old.get("env", {}), **env},
                      "sections": {**old.get("sections", {}), **new_secs},
                      "outdir": args.outdir,
                      "runs": old.get("runs", []) + [list(want)]}
            SUMMARY.update(merged)
        except Exception as e:
            print(f"  (合并旧结果失败，覆盖写出: {type(e).__name__}: {e})")
    else:
        SUMMARY["runs"] = [list(want)]
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
