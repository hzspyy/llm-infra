#!/usr/bin/env python3
"""labs/L8/kv_retrieve_vs_recompute.py - 8.6 的实测驱动 (真实 Qwen3-1.7B + 真实 KV).

四个子命令:

  exact   A 段: 把真实 KV 存进三级 store 再取回, 逐元素对拍; 用取回的 KV 续解码并与
          重新计算的结果逐 token 比对; 校验物理身份漂移被拒、换 adapter 是 miss;
          构造"抓两个上下文的 K 和 V 拼起来"的反例。
  scan    B 段: prefix 长度 128/2048/8192/32768 × 并发 1/8, 分别测写入、取回(三级)、
          重算、完整请求四项时间与内存占用; 再扫 reuse gap=0/1/10/60 s, 给出
          "取回 vs 重算"的交叉点和 TTL 之下的行为。
  errors  C 段: NVMe 一级 + 故障注入 (取回中取消、持引用驱逐、驱逐后取回、引用下溢、
          重复写入、文件损坏、服务重启恢复)。
  approx  D 段: 固定长上下文检索任务 (NFCorpus 文档拼接 + 固定 code needle), 比较
          精确复用、sink+window、heavy-hitter 三种 token 选择; 并给出"直接拼接 KV
          片段"的反例, 说明近似保留必须重算而不是拼 KV。

所有时间用墙钟与 device-event 分别记录; KV 是字节搬运, 数值对拍用逐元素精确相等。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labs.L8.tiered_kv_store import (  # noqa: E402
    KVIdentity,
    KVIdentityError,
    KVStateError,
    TieredKVStore,
    read_identity,
)

SNAP = "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
NFCORPUS = "/scratch/learn/models/hf/hub/datasets--BeIR--nfcorpus/snapshots/b5026a0e96e8a7ac4f95f482a596389289d46269/corpus"
GSM8K = "/scratch/learn/models/hf/hub/datasets--openai--gsm8k/snapshots"


# --------------------------------------------------------------------------
# 公共工具
# --------------------------------------------------------------------------
def load_model(snap: str = SNAP):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(
        snap, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    return tok, model


def head_dim_of(model) -> int:
    return int(getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads))


def read_parquet_texts(path: Path, columns: List[str], limit: int) -> List[str]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns)
    out: List[str] = []
    for i in range(t.num_rows):
        parts = [str(t.column(c)[i].as_py()) for c in columns]
        out.append(" ".join(p for p in parts if p and p != "None"))
        if len(out) >= limit:
            break
    return out


def real_text(n_chars: int, source: str = "auto") -> str:
    """取真实文本; NFCorpus 优先, GSM8K 兜底, 都拿不到时用内置英文说明。"""
    parts: List[str] = []
    if source in ("auto", "nfcorpus"):
        root = Path(NFCORPUS)
        for f in sorted(root.rglob("*.parquet")):
            try:
                parts.extend(read_parquet_texts(f, ["title", "text"], 4000))
            except Exception:  # noqa: BLE001
                continue
    if len(" ".join(parts)) < n_chars and source in ("auto", "gsm8k"):
        for f in sorted(Path(GSM8K).rglob("*.parquet")):
            try:
                parts.extend(read_parquet_texts(f, ["question"], 4000))
            except Exception:  # noqa: BLE001
                continue
            if len(" ".join(parts)) > n_chars:
                break
    if not parts:
        parts = ["A language model reads tokens and writes tokens. " * 50]
    s = "\n".join(parts)
    while len(s) < n_chars:
        s += "\n" + s[: max(1, len(s) // 2)]
    return s[:n_chars]


def build_ids(tok, length: int) -> torch.Tensor:
    ids = tok(real_text(length * 6), return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < length:
        ids = ids.repeat((length // max(1, ids.numel())) + 1)
    return ids[:length].unsqueeze(0).to("cuda")


def kv_to_list(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(cache.layers[i].keys, cache.layers[i].values) for i in range(len(cache))]


def list_to_cache(kv: List[Tuple[torch.Tensor, torch.Tensor]]):
    """零拷贝构造 DynamicCache: 直接引用给定张量。

    transformers 5.x 的 `DynamicCache.update()` 对每个 layer 各做一次 `torch.cat`,
    28 层会把 224 MiB 的 KV 复制成 448 MiB, 并把复制时间算进取回时延。KV 取回本来
    就是"把已有字节挂回 block 池", 不该再复制一遍, 因此这里直接建 layer。
    """
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer
    c = DynamicCache()
    layers = []
    for k, v in kv:
        lay = DynamicLayer()
        lay.keys = k
        lay.values = v
        lay.is_initialized = True
        layers.append(lay)
    c.layers = layers
    return c


def kv_bytes(kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)


def kv_equal(a: List[Tuple[torch.Tensor, torch.Tensor]],
             b: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[bool, float]:
    exact = True
    max_abs = 0.0
    for (k0, v0), (k1, v1) in zip(a, b):
        for x, y in ((k0, k1), (v0, v1)):
            if x.shape != y.shape or x.dtype != y.dtype:
                exact = False
                continue
            max_abs = max(max_abs, (x.float() - y.float()).abs().max().item())
            if not torch.equal(x, y):
                exact = False
    return exact, max_abs


def prefill(model, ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
    """返回 (last_logits, kv_list, 墙钟秒, device 秒)。"""
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    t0 = time.monotonic()
    s0.record()
    with torch.no_grad():
        out = model(ids, position_ids=position_ids, use_cache=True)
    s1.record()
    torch.cuda.synchronize()
    return out.logits[:, -1, :].float(), kv_to_list(out.past_key_values), \
        time.monotonic() - t0, s0.elapsed_time(s1) / 1000.0


def decode(model, cache, first_tokens: torch.Tensor, steps: int,
           start_pos: Optional[int] = None) -> Tuple[List[int], float, float]:
    """从给定 cache 继续贪心解码 steps 个 token; 返回 (tokens, 墙钟, device 秒)。"""
    ids = first_tokens
    gen: List[int] = []
    pos = start_pos
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    t0 = time.monotonic()
    s0.record()
    with torch.no_grad():
        for _ in range(steps):
            pids = None
            if pos is not None:
                pids = torch.tensor([[pos]], device=ids.device)
                pos += 1
            out = model(ids, past_key_values=cache, position_ids=pids, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(int(nxt.item()))
            ids = nxt
            cache = out.past_key_values
    s1.record()
    torch.cuda.synchronize()
    return gen, time.monotonic() - t0, s0.elapsed_time(s1) / 1000.0


def sync_timed(fn):
    """先同步再计时: non_blocking 的 H2D 若不等待完成, 量到的是入队时间。"""
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    return out, time.monotonic() - t0


def rss_mib() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def identity_for(model, token_len: int, adapter: str = "", layout: str = "BHSD",
                 dtype: str = "bfloat16", quant: str = "none",
                 model_revision: str = "70d244cc") -> KVIdentity:
    return KVIdentity(model_revision=model_revision, adapter_revision=adapter, dtype=dtype,
                      layout=layout, num_layers=model.config.num_hidden_layers,
                      num_kv_heads=model.config.num_key_value_heads,
                      head_dim=head_dim_of(model), token_len=token_len, quant=quant)


# --------------------------------------------------------------------------
# A 段
# --------------------------------------------------------------------------
def cmd_exact(args) -> Dict[str, Any]:
    tok, model = load_model(args.snap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TieredKVStore(out_dir / "nvme", ttl_s=0.0, device="cuda:0",
                          gpu_budget_bytes=2 ** 30)
    L = args.exact_len
    ids = build_ids(tok, L)
    tokens = tuple(ids[0].tolist())
    logits_ref, kv_ref, prefill_wall, prefill_dev = prefill(model, ids)
    ident = identity_for(model, L)
    res: Dict[str, Any] = {"L": L, "kv_bytes": kv_bytes(kv_ref),
                           "prefill_wall_s": prefill_wall, "prefill_device_s": prefill_dev,
                           "vram_after_prefill_MiB": torch.cuda.memory_allocated() / 2 ** 20}

    e, res["put_s"] = sync_timed(lambda: store.put(tokens, ident, kv_ref))
    res["nvme_write_bytes"] = e.nvme_path.stat().st_size if e.nvme_path else 0
    res["put_timing"] = dict(store.timing)

    def stats_delta(before: Dict[str, int]) -> Dict[str, int]:
        return {k: store.stats[k] - before.get(k, 0) for k in store.stats}

    # 三级分别取回并逐元素对拍
    tiers: Dict[str, Any] = {}
    for tier in ("gpu", "cpu", "nvme"):
        b = dict(store.stats)
        got, dt = sync_timed(lambda t=tier: store.get(tokens, ident, force_tier=t))
        store.release(tokens, ident)
        exact, max_abs = kv_equal(kv_ref, got)
        tiers[tier] = {"seconds": dt, "exact": exact, "max_abs_diff": max_abs,
                       "stats": stats_delta(b)}
        del got
        torch.cuda.empty_cache()
    res["tiers"] = tiers

    # 用取回的 KV 续解码 vs 重新计算的 KV 续解码
    got_nvme, _ = sync_timed(lambda: store.get(tokens, ident, force_tier="nvme"))
    store.release(tokens, ident)
    first = logits_ref.argmax(dim=-1, keepdim=True)
    gen_ref, _, _ = decode(model, list_to_cache(kv_ref), first, args.decode_steps, start_pos=L)
    gen_cached, _, _ = decode(model, list_to_cache(got_nvme), first, args.decode_steps, start_pos=L)
    res["decode_from_recomputed_kv"] = gen_ref
    res["decode_from_retrieved_kv"] = gen_cached
    res["decode_tokens_identical"] = gen_ref == gen_cached
    del got_nvme
    torch.cuda.empty_cache()

    # 物理身份漂移: 键相同, 但 layout/dtype/长度/量化变了 -> 必须显式拒绝
    bad_cases = {
        "wrong_layout": identity_for(model, L, layout="BSHD"),
        "wrong_dtype": identity_for(model, L, dtype="float16"),
        "wrong_len": identity_for(model, L - 7),
        "wrong_quant": identity_for(model, L, quant="fp8"),
    }
    rejected: Dict[str, str] = {}
    for name, bad in bad_cases.items():
        try:
            store.get(tokens, bad)
            rejected[name] = "NOT_REJECTED"
        except KVIdentityError as ex:
            rejected[name] = f"KVIdentityError: {str(ex)[:120]}"
        except Exception as ex:  # noqa: BLE001
            rejected[name] = f"other: {type(ex).__name__}: {ex}"
    res["physical_identity_rejections"] = rejected
    res["all_physical_drifts_rejected"] = all(
        v.startswith("KVIdentityError") for v in rejected.values())

    # 换 adapter: 缓存身份本身不同 -> miss, 而不是命中别人的 KV
    got_other = store.get(tokens, identity_for(model, L, adapter="lora-rev2"))
    res["other_adapter_lookup"] = "HIT(bug)" if got_other is not None else "miss (None)"
    res["other_adapter_is_miss"] = got_other is None
    # 换模型 revision: 键不同 -> miss (旧 revision 的盘上文件不会被当前请求命中)
    got_old = store.get(tokens, identity_for(model, L, model_revision="other-rev"))
    res["old_revision_is_miss"] = got_old is None

    # 反例: 把两个上下文的 K 和 V 拼起来
    ids2 = ids.clone()
    ids2[0, L // 2:] = build_ids(tok, L - L // 2)[0]
    _, kv2, _, _ = prefill(model, ids2)
    spliced = [(kv_ref[i][0], kv2[i][1]) for i in range(len(kv_ref))]
    gen_spliced, _, _ = decode(model, list_to_cache(spliced), first, args.decode_steps, start_pos=L)
    res["spliced_decode_tokens"] = gen_spliced
    res["spliced_matches_reference"] = gen_spliced == gen_ref
    res["kv_diff_half_context_swapped"] = max(
        (kv_ref[i][0].float() - kv2[i][0].float()).abs().max().item() for i in range(len(kv_ref)))
    del kv2, spliced
    torch.cuda.empty_cache()

    store.snapshot(out_dir / "store_snapshot.json")
    res["store_stats"] = store.stats
    res["store_timing"] = dict(store.timing)
    res["bytes_report"] = store.bytes_report()
    res["nvme_identity_trailer_ok"] = read_identity(e.nvme_path).to_dict() == ident.to_dict()
    (out_dir / "exact.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps({k: res[k] for k in (
        "L", "kv_bytes", "gpu_roundtrip_exact" if "gpu_roundtrip_exact" in res else "tiers",
        "decode_tokens_identical", "all_physical_drifts_rejected",
        "other_adapter_is_miss", "old_revision_is_miss",
        "spliced_matches_reference", "kv_diff_half_context_swapped")}, ensure_ascii=False))
    return res


# --------------------------------------------------------------------------
# B 段
# --------------------------------------------------------------------------
def cmd_scan(args) -> Dict[str, Any]:
    tok, model = load_model(args.snap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # GPU 一级给 6 GiB 预算: 四个长度合计 4.3 GiB, 全部能驻留, 因此
    # retrieve_gpu 量的是"已经在显存里"的命中 (无拷贝), retrieve_cpu 才是真实 H2D。
    store = TieredKVStore(out_dir / "nvme_scan", ttl_s=0.0, device="cuda:0",
                          gpu_budget_bytes=6 * 2 ** 30)
    rows: List[Dict[str, Any]] = []
    for L in [int(x) for x in args.lengths.split(",")]:
        if L > args.max_len:
            continue
        ids = build_ids(tok, L)
        tokens = tuple(ids[0].tolist())
        ident = identity_for(model, L)
        torch.cuda.reset_peak_memory_stats()
        logits, kv, prefill_wall, prefill_dev = prefill(model, ids)
        peak_prefill = torch.cuda.max_memory_allocated() / 2 ** 20
        nbytes = kv_bytes(kv)

        e, put_s = sync_timed(lambda: store.put(tokens, ident, kv))
        row: Dict[str, Any] = {
            "prefix_len": L, "kv_bytes": nbytes, "kv_MiB": round(nbytes / 2 ** 20, 2),
            "prefill_wall_s": round(prefill_wall, 4), "prefill_device_s": round(prefill_dev, 4),
            "write_s": round(put_s, 4),
            "nvme_write_bytes": e.nvme_path.stat().st_size if e.nvme_path else 0,
            "peak_gpu_MiB": round(peak_prefill, 1), "host_rss_MiB": round(rss_mib(), 1),
        }
        # 三级取回
        for tier in ("gpu", "cpu", "nvme"):
            got, dt = sync_timed(lambda t=tier: store.get(tokens, ident, force_tier=t))
            exact, max_abs = kv_equal(kv, got)
            row[f"retrieve_{tier}_s"] = round(dt, 4)
            row[f"retrieve_{tier}_exact"] = exact
            store.release(tokens, ident)
            del got
            torch.cuda.empty_cache()

        # 完整请求: (取回 + 解码) vs (重算 + 解码)
        first = logits.argmax(dim=-1, keepdim=True)
        gen_re, dec_recompute, _ = decode(model, list_to_cache(kv), first,
                                          args.decode_steps, start_pos=L)

        def _full_cached():
            g = store.get(tokens, ident, force_tier="cpu")
            out = decode(model, list_to_cache(g), first, args.decode_steps, start_pos=L)
            store.release(tokens, ident)
            return out, g

        ((gen_ca, dec_cached, _), got3), _full_wall = sync_timed(_full_cached)
        del got3
        torch.cuda.empty_cache()
        row["full_request_recompute_s"] = round(prefill_wall + dec_recompute, 4)
        row["full_request_retrieve_s"] = round(row["retrieve_cpu_s"] + dec_cached, 4)
        row["full_request_retrieve_measured_s"] = round(_full_wall, 4)
        row["full_request_tokens_identical"] = gen_re == gen_ca
        row["decode_s_from_recompute_kv"] = round(dec_recompute, 4)
        row["decode_s_from_retrieved_kv"] = round(dec_cached, 4)
        row["retrieve_vs_recompute"] = {
            "retrieve_s": row["retrieve_cpu_s"], "recompute_s": round(prefill_wall, 4),
            "retrieve_wins": row["retrieve_cpu_s"] < prefill_wall,
            "speedup": round(prefill_wall / row["retrieve_cpu_s"], 2) if row["retrieve_cpu_s"] else None,
        }

        # 并发取回 (CPU 一级, 每个线程一份真实 H2D 副本)
        conc_res: Dict[str, Any] = {}
        for c in [int(x) for x in args.concurrency.split(",")]:
            need_gib = nbytes * c / 2 ** 30
            if need_gib > args.conc_budget_gib:
                conc_res[str(c)] = {"skipped": f"KV 副本 {need_gib:.1f} GiB > 预算 "
                                              f"{args.conc_budget_gib} GiB"}
                continue

            def one(_i: int):
                g, dt = sync_timed(lambda: store.get(tokens, ident, force_tier="cpu"))
                row_exact, _ = kv_equal(kv, g)
                store.release(tokens, ident)
                del g
                return dt, row_exact

            t0 = time.monotonic()
            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                outs = list(ex.map(one, range(c)))
            total = time.monotonic() - t0
            conc_res[str(c)] = {
                "total_s": round(total, 4),
                "per_op_s": [round(o[0], 4) for o in outs],
                "aggregate_ops_per_s": round(c / total, 3) if total else None,
                "all_exact": all(o[1] for o in outs),
                "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            }
            torch.cuda.empty_cache()
        row["concurrent_retrieve_from_cpu"] = conc_res

        # reuse gap: 同一前缀隔一段时间后再来一次 (可限定只在部分长度上做)
        gap_res: Dict[str, Any] = {}
        gap_lengths = [int(x) for x in args.gap_lengths.split(",") if x]
        if gap_lengths and L not in gap_lengths:
            row["reuse_gap"] = {"skipped": f"L={L} 不在 --gap-lengths {gap_lengths} 内"}
            gap_res = row["reuse_gap"]
        for ttl in ([] if isinstance(gap_res, dict) and "skipped" in gap_res
                    else [float(x) for x in args.ttls.split(",")]):
            st = TieredKVStore(out_dir / f"nvme_gap_ttl{ttl}", ttl_s=ttl, device="cuda:0")
            st.put(tokens, ident, kv)
            st.drop_tiers(tokens, ident)
            per_gap: Dict[str, Any] = {}
            for gap in [float(x) for x in args.gaps.split(",")]:
                # 每个 gap 独立: 重新写一份并只保留 NVMe, 以隔离上一次的驱逐
                st.put(tokens, ident, kv)
                st.drop_tiers(tokens, ident)
                time.sleep(gap)
                st.evict_expired()
                b = dict(st.stats)
                got, dt = sync_timed(lambda: st.get(tokens, ident, force_tier="nvme"))
                hit = got is not None
                if hit:
                    st.release(tokens, ident)
                    del got
                torch.cuda.empty_cache()
                delta = {k: st.stats[k] - b.get(k, 0) for k in st.stats}
                per_gap[str(gap)] = {
                    "hit": hit, "retrieve_s": round(dt, 4),
                    "full_request_s": round(dt + dec_recompute, 4) if hit
                    else round(prefill_wall + dec_recompute, 4),
                    "stats": {k: v for k, v in delta.items() if v},
                }
            st.stop_maintenance()
            gap_res[str(ttl)] = per_gap
            del st
            gc.collect()
        row["reuse_gap"] = gap_res

        del kv, logits
        gc.collect()
        torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False)[:500], flush=True)

    # 维护成本: 一次驱逐扫描的耗时随条目数增长
    maint = []
    mstore = TieredKVStore(out_dir / "nvme_maint", ttl_s=1e-6, device="cuda:0")
    small = [(torch.zeros(1, 8, 64, 128, dtype=torch.bfloat16, device="cuda"),
              torch.zeros(1, 8, 64, 128, dtype=torch.bfloat16, device="cuda"))] * 2
    for n in [1, 8, 64]:
        mstore.entries.clear()
        for i in range(n):
            mstore.put((i, i + 1, i + 2), identity_for(model, 64, adapter=f"a{i}"), small)
        time.sleep(0.01)
        t0 = time.monotonic()
        evicted = mstore.evict_expired()
        maint.append({"entries": n, "evicted": evicted, "sweep_s": round(time.monotonic() - t0, 6)})
    mstore.stop_maintenance()
    del small, mstore
    torch.cuda.empty_cache()

    (out_dir / "scan.json").write_text(json.dumps({
        "note": "取回含注册/传输/等待/维护四段成本; 传输与等待在 retrieve_*_s 内, 注册是 write_s, "
                "维护成本单列在 maintenance 与 store_timing.maintenance_s",
        "rows": rows, "maintenance_sweep": maint, "store_timing": dict(store.timing),
        "store_stats": store.stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": rows, "maintenance_sweep": maint}


# --------------------------------------------------------------------------
# C 段
# --------------------------------------------------------------------------
def cmd_errors(args) -> Dict[str, Any]:
    tok, model = load_model(args.snap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvme = out_dir / "nvme_err"
    store = TieredKVStore(nvme, ttl_s=args.ttl, device="cuda:0")
    L = args.err_len
    ids = build_ids(tok, L)
    tokens = tuple(ids[0].tolist())
    _, kv, _, _ = prefill(model, ids)
    ident = identity_for(model, L)
    res: Dict[str, Any] = {"L": L, "kv_bytes": kv_bytes(kv), "ttl_s": args.ttl}

    # 1) 重复写入: 第二次命中而不是新条目
    e1 = store.put(tokens, ident, kv)
    e2 = store.put(tokens, ident, kv)
    res["duplicate_put_same_entry"] = e1.key == e2.key
    res["duplicate_put_hits"] = store.stats["put_hit"]

    # 2) 取回中取消: 取回抛错且引用回到 0
    store.inject_cancel(tokens, ident)
    try:
        store.get(tokens, ident)
        res["cancel_raises"] = False
    except KVStateError as ex:
        res["cancel_raises"] = True
        res["cancel_message"] = str(ex)
    res["refcount_after_cancel"] = store.entries[e1.key].refcount
    res["state_after_cancel"] = store.entries[e1.key].state
    store.resume(tokens, ident)
    # 取消后可再次正常取回
    got, dt = sync_timed(lambda: store.get(tokens, ident, force_tier="nvme"))
    exact, max_abs = kv_equal(kv, got)
    res["resume_get_exact"] = exact
    res["resume_get_max_abs_diff"] = max_abs
    store.release(tokens, ident)
    del got
    torch.cuda.empty_cache()

    # 3) 持引用时不可驱逐; 归还后可驱逐
    got = store.get(tokens, ident, force_tier="cpu")
    store.evict_expired(now=time.monotonic() + 10 * args.ttl + 1)
    res["state_while_referenced"] = store.entries[e1.key].state
    res["deferred_eviction_count"] = store.stats["deferred_eviction"]
    exact_ref, _ = kv_equal(kv, got)
    res["data_valid_while_referenced"] = exact_ref
    del got
    torch.cuda.empty_cache()
    store.release(tokens, ident)
    # 引用下溢必须报错
    try:
        store.release(tokens, ident)
        res["double_release_raises"] = False
    except KVStateError as ex:
        res["double_release_raises"] = True
        res["double_release_message"] = str(ex)
    store.evict_expired(now=time.monotonic() + 10 * args.ttl + 2)
    res["state_after_release_and_expire"] = store.entries[e1.key].state
    res["nvme_file_removed"] = not (store.entries[e1.key].nvme_path and
                                    store.entries[e1.key].nvme_path.exists())
    after = store.get(tokens, ident)
    res["get_after_evict"] = "RETURNED_DATA(bug)" if after is not None else "miss (None)"
    res["get_after_evict_is_miss"] = after is None

    # 4) 重新写入 -> 服务重启恢复
    store.put(tokens, ident, kv)
    store.drop_tiers(tokens, ident)
    res["bytes_before_restart"] = store.bytes_report()
    store.snapshot(out_dir / "store_err_snapshot.json")
    files = sorted(nvme.glob("kv_*.bin"))
    res["nvme_files_before_restart"] = len(files)
    res["nvme_bytes_before_restart"] = sum(f.stat().st_size for f in files)
    restored_ident = read_identity(files[0]).to_dict()
    res["restart_identity_from_file"] = restored_ident
    res["restart_identity_matches"] = restored_ident == ident.to_dict()

    store2 = TieredKVStore(nvme, ttl_s=0.0, device="cuda:0")
    res["restart_open"] = store2.open_existing()
    got2, dt2 = sync_timed(lambda: store2.get(tokens, ident, force_tier="nvme"))
    exact2, max2 = kv_equal(kv, got2)
    res["restart_get_exact"] = exact2
    res["restart_get_max_abs_diff"] = max2
    res["restart_get_s"] = dt2
    store2.release(tokens, ident)
    del got2
    torch.cuda.empty_cache()

    # 5) 损坏文件不应被当作命中: 截断 / 追加垃圾 / 身份 JSON 损坏
    corrupt: Dict[str, Any] = {}
    f0 = files[0]
    for name, mutate in (
        ("truncated", lambda p: p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])),
        ("appended", lambda p: p.write_bytes(p.read_bytes() + b"\x00" * 4096)),
        ("bad_trailer", lambda p: p.write_bytes(p.read_bytes()[:-8] + b"\xff" * 8)),
    ):
        tmp = out_dir / f"corrupt_{name}"
        tmp.mkdir(exist_ok=True)
        dst = tmp / f0.name
        dst.write_bytes(f0.read_bytes())
        mutate(dst)
        st = TieredKVStore(tmp, ttl_s=0.0, device="cuda:0")
        r = st.open_existing()
        corrupt[name] = {"open_result": r,
                         "get": "miss" if st.get(tokens, ident) is None else "HIT(bug)"}
        st.stop_maintenance()
    res["corrupt_files"] = corrupt

    store.stop_maintenance()
    store2.stop_maintenance()
    res["stats"] = store.stats
    res["stats_after_restart"] = store2.stats
    (out_dir / "errors.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:1800])
    return res


# --------------------------------------------------------------------------
# D 段
# --------------------------------------------------------------------------
def approx_keep_indices(kv: List[Tuple[torch.Tensor, torch.Tensor]], budget: int,
                        method: str, sink: int = 4) -> torch.Tensor:
    """返回要保留的 token 位置 (升序)。method ∈ {sink_window, heavy_hitter}。"""
    L = kv[0][0].shape[2]
    keep = set(range(min(sink, L)))
    if method == "sink_window":
        n_window = max(0, budget - len(keep))
        keep.update(range(max(0, L - n_window), L))
    elif method == "heavy_hitter":
        # 用各层 K 的 L2 范数当"被关注程度"的廉价代理 (真实实现用累计注意力权重)
        scores = torch.zeros(L, device=kv[0][0].device)
        for k, _ in kv:
            scores += k.float().pow(2).sum(dim=(0, 1, 3))
        order = torch.argsort(scores, descending=True).tolist()
        for idx in order:
            if len(keep) >= budget:
                break
            keep.add(idx)
    keep.add(L - 1)   # 最后一个位置必须保留, 否则续解码缺当前位置
    return torch.tensor(sorted(keep), device=kv[0][0].device)


def crop_kv(kv: List[Tuple[torch.Tensor, torch.Tensor]], idx: torch.Tensor):
    return [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in kv]


def build_needle_task(tok, ctx_len: int, pos_frac: float, code: str):
    """用 NFCorpus 文档拼出固定长上下文, 在指定位置插入固定格式的 needle。"""
    text = real_text(ctx_len * 6, source="nfcorpus")
    doc = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    L = min(ctx_len, doc.shape[1])
    base = doc[:, :L]
    pos = max(4, int(L * pos_frac))
    needle = tok(f"\nImportant: the access code for vault UMBRA is {code}.\n",
                 add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([base[0, :pos], needle, base[0, pos:]]).unsqueeze(0).to("cuda")
    q = tok("\nQuestion: what is the access code for vault UMBRA? Answer:",
            add_special_tokens=False).input_ids
    q_ids = torch.tensor([q], device="cuda")
    return ids, q_ids, pos, int(ids.shape[1])


def answer_after_context(model, context_kv, q_ids, start_pos: int, steps: int):
    """把问题接到给定 cache 之后做贪心解码; 用于"直接拼接 KV 片段"的反例。"""
    cache = list_to_cache(context_kv)
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    t0 = time.monotonic()
    s0.record()
    with torch.no_grad():
        o = model(q_ids, past_key_values=cache, position_ids=torch.arange(
            start_pos, start_pos + q_ids.shape[1], device=q_ids.device).unsqueeze(0),
            use_cache=True)
        out = [int(o.logits[:, -1, :].argmax(dim=-1).item())]
        c = o.past_key_values
        for i in range(steps - 1):
            o = model(torch.tensor([[out[-1]]], device=q_ids.device), past_key_values=c,
                      position_ids=torch.tensor([[start_pos + q_ids.shape[1] + i]],
                                                device=q_ids.device), use_cache=True)
            c = o.past_key_values
            out.append(int(o.logits[:, -1, :].argmax(dim=-1).item()))
    s1.record()
    torch.cuda.synchronize()
    return out, time.monotonic() - t0, s0.elapsed_time(s1) / 1000.0


def greedy_single_pass(model, ctx_tokens: torch.Tensor, ctx_pos: torch.Tensor,
                       q_ids: torch.Tensor, steps: int, cont_pos: int):
    """把 (上下文 token, 问题) 放在一次前向里再续解码, 返回 (tokens, 墙钟, device)。

    完整上下文与压缩上下文共用这一段: 两者的差别只有"保留了哪些 token、它们的
    位置是什么", 不会把 cache_position 与 position_ids 不一致这类实现细节读成
    压缩本身的代价。
    """
    seq = torch.cat([ctx_tokens, q_ids[0]]).unsqueeze(0)
    fake = torch.zeros(q_ids.shape[1], dtype=ctx_pos.dtype, device=ctx_pos.device)
    pos = torch.cat([ctx_pos, fake + torch.arange(cont_pos, cont_pos + q_ids.shape[1],
                                                  device=ctx_pos.device)]).unsqueeze(0)
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    t0 = time.monotonic()
    s0.record()
    with torch.no_grad():
        o = model(seq, position_ids=pos, use_cache=True)
        out = [int(o.logits[:, -1, :].argmax(dim=-1).item())]
        c = o.past_key_values
        for i in range(steps - 1):
            o = model(torch.tensor([[out[-1]]], device=q_ids.device), past_key_values=c,
                      position_ids=torch.tensor([[cont_pos + q_ids.shape[1] + i]],
                                                device=q_ids.device), use_cache=True)
            c = o.past_key_values
            out.append(int(o.logits[:, -1, :].argmax(dim=-1).item()))
    s1.record()
    torch.cuda.synchronize()
    return out, time.monotonic() - t0, s0.elapsed_time(s1) / 1000.0


def score_target(model, ctx_tokens: torch.Tensor, ctx_pos: torch.Tensor, q_ids: torch.Tensor,
                 target_ids: torch.Tensor, cont_pos: int) -> Tuple[float, int]:
    """teacher-forcing 打分: 给定压缩后的上下文与问题, 目标答案 token 的平均 logprob。

    逐 token 贪心是否命中是一个二值信号, 对 1-2 个 token 的偏差不敏感; 平均 logprob
    给出连续的分级, 用来区分"完全丢失"和"接近但差一点"。
    """
    n = int(target_ids.numel())
    seq = torch.cat([ctx_tokens, q_ids[0], target_ids[:-1]]).unsqueeze(0)
    pos = torch.arange(cont_pos, cont_pos + q_ids.shape[1] + n - 1,
                       device=ctx_pos.device)
    pos = torch.cat([ctx_pos, pos]).unsqueeze(0)
    with torch.no_grad():
        o = model(seq, position_ids=pos, use_cache=False)
    logits = o.logits[0, -n:, :].float()
    lp = torch.log_softmax(logits, dim=-1)
    vals = [float(lp[i, int(target_ids[i])]) for i in range(n)]
    matches = sum(int(logits[i].argmax()) == int(target_ids[i]) for i in range(n))
    return sum(vals) / max(1, n), matches


def cmd_approx(args) -> Dict[str, Any]:
    tok, model = load_model(args.snap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for pos_frac in [float(x) for x in args.needle_positions.split(",")]:
        code = f"ZQ{args.seed}{int(pos_frac * 100):02d}"
        ids, q_ids, pos, total = build_needle_task(tok, args.ctx_len, pos_frac, code)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        kv_full = kv_to_list(out.past_key_values)
        full_len = int(ids.shape[1])
        all_pos = torch.arange(full_len, device="cuda")
        row: Dict[str, Any] = {"needle_pos_frac": pos_frac, "needle_pos": pos, "code": code,
                               "ctx_tokens": full_len,
                               "kv_bytes": kv_bytes(kv_full)}
        # 参照: 完整上下文 (等价于一次普通 prefill)
        ans_ids, wall, dev = greedy_single_pass(model, ids[0], all_pos, q_ids,
                                                args.answer_steps, full_len)
        full_answer = tok.decode(ans_ids, skip_special_tokens=True)
        target_ids = torch.tensor(ans_ids[:args.target_len], device="cuda")
        row["exact_full"] = {"hit": code in full_answer, "answer": full_answer.strip()[:80],
                             "wall_s": round(wall, 4), "device_s": round(dev, 4),
                             "target_tokens": ans_ids[:args.target_len]}
        # 参照本身的 teacher-forced logprob 给出"上限", 也是各方法比较的基线
        base_lp, base_match = score_target(model, ids[0], all_pos, q_ids, target_ids, full_len)
        row["exact_full"]["target_mean_logprob"] = round(base_lp, 4)
        row["exact_full"]["target_token_match"] = base_match
        for method in ("sink_window", "heavy_hitter"):
            for ratio in [float(x) for x in args.budgets.split(",")]:
                budget = max(8, int(full_len * ratio))
                idx = approx_keep_indices(kv_full, budget, method)
                kept = len(idx.tolist())
                kept_tokens = ids[0].index_select(0, idx)
                # 正确的近似语义: 保留 token 与**原始位置**, 重新前向, 而不是拼 KV 片段
                a_ids, wall_a, _ = greedy_single_pass(model, kept_tokens, idx, q_ids,
                                                      args.answer_steps, full_len)
                ans = tok.decode(a_ids, skip_special_tokens=True)
                lp, match = score_target(model, kept_tokens, idx, q_ids, target_ids, full_len)
                key = f"{method}@{ratio}"
                row[key] = {"budget_tokens": budget, "kept_tokens": kept,
                            "hit": code in ans, "answer": ans.strip()[:80],
                            "wall_s": round(wall_a, 4),
                            "target_mean_logprob": round(lp, 4),
                            "target_token_match": match}
                # 反例: 直接把保留下来的 KV 片段接起来 (不重算)
                spliced = crop_kv(kv_full, idx)
                s_ids, _, _ = answer_after_context(model, spliced, q_ids, full_len,
                                                   args.answer_steps)
                sans = tok.decode(s_ids, skip_special_tokens=True)
                row[f"{key}_spliced_kv"] = {"hit": code in sans, "answer": sans.strip()[:80]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False)[:400], flush=True)
        del kv_full
        gc.collect()
        torch.cuda.empty_cache()
    (out_dir / "approx.json").write_text(json.dumps({
        "note": "近似的正确语义是保留位置后**重算**, 不是直接拼接 KV 片段; "
                "spliced_kv 列给出后者的反例。精确复用的判据见 exact.json",
        "task": "NFCorpus 文档拼接 + 固定 code needle, 位置 10/50/90%",
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "scan", "errors", "approx"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snap", default=SNAP)
    ap.add_argument("--exact-len", type=int, default=2048)
    ap.add_argument("--err-len", type=int, default=512)
    ap.add_argument("--ttl", type=float, default=0.5)
    ap.add_argument("--lengths", default="128,2048,8192,32768")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--concurrency", default="1,8")
    ap.add_argument("--gaps", default="0,1,10,60")
    ap.add_argument("--ttls", default="0,5")
    ap.add_argument("--gap-lengths", default="",
                    help="只在这些前缀长度上做 reuse gap 扫描; 空表示全部")
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--needle-positions", default="0.1,0.5,0.9")
    ap.add_argument("--budgets", default="0.1,0.25")
    ap.add_argument("--answer-steps", type=int, default=12)
    ap.add_argument("--target-len", type=int, default=6,
                    help="teacher-forcing 打分用的目标答案 token 数 (取自完整上下文贪心结果)")
    ap.add_argument("--conc-budget-gib", type=float, default=12.0,
                    help="并发取回允许占用的 KV 副本总字节上限; 超了跳过并记录")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    {"exact": cmd_exact, "scan": cmd_scan, "errors": cmd_errors,
     "approx": cmd_approx}[args.mode](args)


if __name__ == "__main__":
    main()
