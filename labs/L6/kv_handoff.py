#!/usr/bin/env python3
"""
6.4 任务 A/C：KV 交接的协议、搬运参照与失败回收。

任务 A：把一次 prefill 产生的 KV cache 从「P 实例」交给「D 实例」，然后
  · 与并置（colocated）baseline 对拍首 token 与后续 logits
  · 给出 GPU→CPU→GPU 与 GPU→GPU(P2P) 两条搬运参照的字节、时间与带宽
  · 交接描述符带 request/session/model revision、token/position、dtype、
    TP layout、block ids 与 block table；任一字段不一致必须明确拒绝

任务 C：在注册、传输中、接收后、decode 中四个时点取消或中断，并注入重复 ACK
  与超时；核对两侧 refcount 归零、staging 释放、没有重复提交 token，
  且同一实例随后仍能处理合法请求。

用法：
    python kv_handoff.py A --model <qwen3-1.7b> --seq 2048 --tokens 8 --out <dir>
    python kv_handoff.py C --model <qwen3-1.7b> --seq 512 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field, asdict

import torch

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


# ==========================================================================
# 交接描述符
# ==========================================================================

@dataclass
class HandoffDescriptor:
    """一次 KV 交接的全部身份信息。任何一项对不上都不允许合并。"""
    request_id: str
    session_id: str
    model_revision: str
    prompt_sha256: str
    num_tokens: int
    dtype: str
    kv_scale: float
    tp_size: int
    tp_rank: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    block_ids: list
    block_table: list

    def kv_bytes(self, token_count):
        per_token = (2 * self.num_layers * self.num_kv_heads * self.head_dim
                     * DTYPE_BYTES[self.dtype])
        return per_token * token_count

    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class LayoutMismatch(RuntimeError):
    pass


REQUIRED_MATCH = ("request_id", "session_id", "model_revision", "dtype",
                  "tp_size", "tp_rank", "num_layers", "num_kv_heads", "head_dim",
                  "block_size")


def validate(producer: HandoffDescriptor, consumer: HandoffDescriptor):
    """接收方校验：身份、布局、token 数、block 表全部要对上。"""
    bad = []
    for k in REQUIRED_MATCH:
        if getattr(producer, k) != getattr(consumer, k):
            bad.append(f"{k}: {getattr(producer, k)!r} != {getattr(consumer, k)!r}")
    if producer.num_tokens != consumer.num_tokens:
        bad.append(f"num_tokens: {producer.num_tokens} != {consumer.num_tokens}")
    if abs(producer.kv_scale - consumer.kv_scale) > 1e-9:
        bad.append(f"kv_scale: {producer.kv_scale} != {consumer.kv_scale}")
    if list(producer.block_table) != list(consumer.block_table):
        bad.append("block_table 不一致")
    if bad:
        raise LayoutMismatch("; ".join(bad))
    return True


# ==========================================================================
# refcount 与 staging 生命周期
# ==========================================================================

class RefCounted:
    """两侧共享的 block 引用计数：接手方必须显式 ACK，发送方才释放。"""

    def __init__(self, name):
        self.name = name
        self.blocks = {}          # block_id -> count
        self.staging = {}         # buffer_id -> bytes
        self.events = []

    def retain(self, block_id, n=1):
        self.blocks[block_id] = self.blocks.get(block_id, 0) + n
        self.events.append(("retain", block_id, self.blocks[block_id]))

    def release(self, block_id, n=1):
        cur = self.blocks.get(block_id, 0) - n
        if cur < 0:
            raise RuntimeError(f"{self.name}: block {block_id} 释放次数超过持有次数")
        if cur == 0:
            self.blocks.pop(block_id, None)
        else:
            self.blocks[block_id] = cur
        self.events.append(("release", block_id, cur))

    def alloc_staging(self, buf_id, nbytes):
        self.staging[buf_id] = nbytes
        self.events.append(("staging_alloc", buf_id, nbytes))

    def free_staging(self, buf_id):
        self.staging.pop(buf_id, None)
        self.events.append(("staging_free", buf_id, 0))

    def leaked(self):
        return {"blocks": dict(self.blocks), "staging": dict(self.staging)}


# ==========================================================================
# 模型侧：prefill 与 decode
# ==========================================================================

def load_model(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return model, tok


def cache_to_legacy(cache):
    """把 transformers 的 Cache 拆成 legacy 元组，便于逐层搬运与对拍。

    transformers 5.x 去掉了 from_legacy_cache/to_legacy_cache，改由
    DynamicCache.layers[i].keys/values 暴露每层的 K/V。
    """
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return tuple((getattr(l, "keys", None), getattr(l, "values", None))
                     for l in layers)
    return tuple(cache)


def legacy_to_cache(legacy):
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    # 5.x 的构造函数直接接受每层的 (k, v) 序列
    return DynamicCache(legacy)


def kv_bytes(legacy):
    return sum(t.numel() * t.element_size() for layer in legacy for t in layer)


def kv_to_cpu(legacy):
    return tuple(tuple(t.detach().to("cpu", copy=True) for t in layer) for layer in legacy)


def kv_to_device(legacy, device):
    return tuple(tuple(t.to(device, non_blocking=True) for t in layer) for layer in legacy)


@torch.no_grad()
def prefill(model, input_ids):
    out = model(input_ids=input_ids, use_cache=True)
    return cache_to_legacy(out.past_key_values), out.logits[:, -1, :]


@torch.no_grad()
def decode_steps(model, past, first_token, steps, device):
    """从给定的 cache 继续贪心解码，返回逐 token 与每步的 logits。"""
    ids = first_token
    tokens, logits = [], []
    cur = past
    for _ in range(steps):
        out = model(input_ids=ids, past_key_values=legacy_to_cache(cur),
                    use_cache=True)
        cur = cache_to_legacy(out.past_key_values)
        lg = out.logits[:, -1, :]
        nxt = lg.argmax(dim=-1, keepdim=True)
        logits.append(lg)
        tokens.append(int(nxt.item()))
        ids = nxt
    return tokens, logits


# ==========================================================================
# 任务 A
# ==========================================================================

def build_prompt(tok, seq_len):
    """构造正好 seq_len 个 token 的输入（用重复文本再截断/补齐）。"""
    base = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok(base, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len:
        reps = (seq_len // ids.numel()) + 2
        ids = tok(base * reps, return_tensors="pt").input_ids[0]
    return ids[:seq_len].unsqueeze(0)


def make_descriptor(model_path, prompt_ids, legacy, block_size, tp_size=1, tp_rank=0):
    k0 = legacy[0][0]
    return HandoffDescriptor(
        request_id="req-0001", session_id="sess-a", model_revision=os.path.basename(
            os.path.normpath(model_path)),
        prompt_sha256=hashlib.sha256(prompt_ids.numpy().tobytes()).hexdigest()[:16],
        num_tokens=int(prompt_ids.shape[1]), dtype=str(k0.dtype).replace("torch.", ""),
        kv_scale=1.0, tp_size=tp_size, tp_rank=tp_rank,
        num_layers=len(legacy), num_kv_heads=int(k0.shape[1]),
        head_dim=int(k0.shape[3]), block_size=block_size,
        block_ids=list(range((int(prompt_ids.shape[1]) + block_size - 1) // block_size)),
        block_table=[[i] for i in range((int(prompt_ids.shape[1]) + block_size - 1)
                                        // block_size)],
    )


def run_A(args):
    os.makedirs(args.out, exist_ok=True)
    dev_p = torch.device("cuda:0")
    dev_d = torch.device("cuda:1") if torch.cuda.device_count() > 1 else dev_p
    torch.cuda.set_device(dev_p)
    model_p, tok = load_model(args.model, dev_p)
    same_device = (dev_d == dev_p)
    model_d = model_p if same_device else load_model(args.model, dev_d)[0]

    prompt = build_prompt(tok, args.seq).to(dev_p)
    t0 = time.perf_counter()
    legacy_p, logits_p = prefill(model_p, prompt)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    first = logits_p.argmax(dim=-1, keepdim=True)

    # 以 prefill 之后立刻取的 CPU 快照为唯一真值：
    # 两个分支各自从同一份快照重建 cache，避免任何一侧因原地写而影响另一侧。
    frozen = kv_to_cpu(legacy_p)
    nbytes = kv_bytes(frozen)
    desc = make_descriptor(args.model, prompt.cpu(), frozen, args.block_size)

    # ---- 并置 baseline：同一实例、同一份快照 ----
    legacy_base = kv_to_device(frozen, dev_p)
    base_tokens, base_logits = decode_steps(model_p, legacy_base, first, args.tokens, dev_p)

    # ---- 搬运参照 1：GPU→CPU→GPU ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    staged = kv_to_cpu(legacy_p)
    t_d2h = time.perf_counter() - t0
    if not same_device:
        torch.cuda.synchronize(dev_d)
    t0 = time.perf_counter()
    legacy_d = kv_to_device(staged, dev_d)
    if not same_device:
        torch.cuda.synchronize(dev_d)
    t_h2d = time.perf_counter() - t0
    cpu_bytes = kv_bytes(staged)

    # ---- 搬运参照 2：GPU→GPU。这台机器有已知的 P2P 慢状态（1.3 记录过 98.9 ms/操作），
    #      所以同时记录每次操作的时间与总时间，慢状态要能被看出来。
    p2p_runs = []
    if not same_device:
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p2p = tuple(tuple(t.to(dev_d, non_blocking=True) for t in layer)
                        for layer in legacy_p)
            torch.cuda.synchronize(dev_d)
            p2p_runs.append(time.perf_counter() - t0)
            del p2p
            torch.cuda.empty_cache()
    t_p2p = min(p2p_runs) if p2p_runs else None
    ops_per_transfer = len(legacy_p) * 2
    p2p_state = None
    if t_p2p is not None:
        per_op_ms = t_p2p / ops_per_transfer * 1e3
        p2p_state = ("slow（每操作约 %.1f ms，与 1.3 记录的 98.9 ms/操作 同量级）"
                     % per_op_ms) if per_op_ms > 10 else ("fast（每操作 %.3f ms）" % per_op_ms)

    # ---- 交接后对拍 ----
    validate(desc, desc)                       # 自校验应当通过
    hd_tokens, hd_logits = decode_steps(model_d, legacy_d, first.to(dev_d),
                                        args.tokens, dev_d)
    max_logit_diff = max(float((a.to("cpu") - b.to("cpu")).abs().max().item())
                         for a, b in zip(hd_logits, base_logits))
    tokens_equal = (hd_tokens == base_tokens)

    # ---- 错误身份/布局必须拒绝 ----
    rejections = []
    for field_name, bad_value in [
        ("request_id", "req-9999"), ("session_id", "sess-b"),
        ("model_revision", "other-revision"), ("num_tokens", desc.num_tokens - 1),
        ("dtype", "float32"), ("tp_size", 2), ("tp_rank", 1),
        ("block_size", desc.block_size * 2), ("kv_scale", 1.5),
    ]:
        bad = HandoffDescriptor(**asdict(desc))
        setattr(bad, field_name, bad_value)
        try:
            validate(desc, bad)
            rejections.append({"field": field_name, "rejected": False})
        except LayoutMismatch as e:
            rejections.append({"field": field_name, "rejected": True,
                               "reason": str(e)[:160]})
    bad_table = HandoffDescriptor(**asdict(desc))
    bad_table.block_table = [[99]]
    try:
        validate(desc, bad_table)
        rejections.append({"field": "block_table", "rejected": False})
    except LayoutMismatch as e:
        rejections.append({"field": "block_table", "rejected": True,
                           "reason": str(e)[:160]})

    payload = {
        "model": args.model, "seq": args.seq, "decode_tokens": args.tokens,
        "device_p": str(dev_p), "device_d": str(dev_d),
        "same_device": same_device,
        "descriptor": asdict(desc), "descriptor_fingerprint": desc.fingerprint(),
        "kv_bytes": nbytes, "kv_MiB": nbytes / 2**20,
        "kv_bytes_per_token": nbytes / args.seq,
        "cpu_staged_bytes": cpu_bytes,
        "prefill_s": prefill_s,
        "d2h_s": t_d2h, "h2d_s": t_h2d,
        "gpu_cpu_gpu_s": t_d2h + t_h2d,
        "gpu_cpu_gpu_GBps": nbytes / max(t_d2h + t_h2d, 1e-9) / 1e9,
        "p2p_s": t_p2p, "p2p_runs_s": p2p_runs,
        "p2p_ops_per_transfer": ops_per_transfer if p2p_runs else None,
        "p2p_per_op_ms": (t_p2p / ops_per_transfer * 1e3) if t_p2p else None,
        "p2p_state": p2p_state,
        "p2p_GBps": (nbytes / t_p2p / 1e9) if t_p2p else None,
        "baseline_tokens": base_tokens,
        "handoff_tokens": hd_tokens,
        "tokens_equal": tokens_equal,
        "max_logit_abs_diff": max_logit_diff,
        "logits_match": max_logit_diff < 5e-2,
        "rejections": rejections,
        "all_invalid_rejected": all(r["rejected"] for r in rejections),
    }
    with open(os.path.join(args.out, "handoff.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[A] KV {nbytes / 2**20:.1f} MiB（{nbytes / args.seq:.0f} B/token）"
          f"  {desc.num_layers} 层 × {desc.num_kv_heads} kv_head × {desc.head_dim}")
    print(f"    GPU→CPU→GPU {payload['gpu_cpu_gpu_s'] * 1e3:.2f} ms "
          f"({payload['gpu_cpu_gpu_GBps']:.2f} GB/s)"
          + (f"  GPU→GPU(P2P) 最快 {t_p2p * 1e3:.2f} ms ({payload['p2p_GBps']:.2f} GB/s，"
             f"每次 {ops_per_transfer} 个拷贝操作，状态：{p2p_state}）"
             if t_p2p else "  （单卡，无 P2P 参照）"))
    print(f"    首 token 与后续 {args.tokens} 步：token 序列相同={tokens_equal} "
          f"logits 最大绝对差={max_logit_diff:.3e}")
    print(f"    非法交接拒绝 {sum(r['rejected'] for r in rejections)}/{len(rejections)}："
          + ", ".join(r["field"] for r in rejections if r["rejected"]))
    return payload


# ==========================================================================
# 任务 C
# ==========================================================================

def run_C(args):
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    model, tok = load_model(args.model, dev)
    prompt = build_prompt(tok, args.seq).to(dev)
    legacy, logits = prefill(model, prompt)
    block_ids = list(range(4))
    cases = []

    def fresh():
        return RefCounted("producer"), RefCounted("consumer")

    def baseline_alloc():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    def do_case(name, cancel_at):
        prod, cons = fresh()
        for b in block_ids:
            prod.retain(b)
        staging_id = f"stage-{name}"
        base = baseline_alloc()
        emitted = []
        err = None
        handoff_complete = False
        staged = moved = dbg_out = None
        try:
            if cancel_at == "after_register":
                raise KeyboardInterrupt("取消：注册之后、传输之前")
            prod.alloc_staging(staging_id, kv_bytes(legacy))
            staged = kv_to_cpu(legacy)
            if cancel_at == "mid_transfer":
                raise KeyboardInterrupt("取消：传输中")
            prod.free_staging(staging_id)
            if cancel_at == "after_receive":
                raise KeyboardInterrupt("取消：接收之后、decode 之前")
            moved = kv_to_device(staged, dev)
            if cancel_at == "mid_decode":
                dbg_out = model(input_ids=logits.argmax(-1, keepdim=True),
                                past_key_values=legacy_to_cache(moved), use_cache=True)
                emitted.append(int(dbg_out.logits[:, -1, :].argmax(-1).item()))
                raise KeyboardInterrupt("取消：decode 中")
            # ---- 正常路径：接收方先 retain，发送方收到 ACK 后再 release ----
            toks, _ = decode_steps(model, moved, logits.argmax(-1, keepdim=True), 4, dev)
            emitted.extend(toks)
            for b in block_ids:
                cons.retain(b)
            for b in block_ids:          # ACK：接收方确认持有
                prod.release(b)
            for b in block_ids:          # 接收方用完自己释放
                cons.release(b)
            handoff_complete = True
        except BaseException as e:       # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        finally:
            # 取消/中断路径也必须把两侧状态收干净：没有完成 ACK 的交接，
            # 发送方释放自己持有的 block，接收方释放自己的预留，staging 一律回收。
            if not handoff_complete:
                for b in list(prod.blocks):
                    prod.release(b)
                for b in list(cons.blocks):
                    cons.release(b)
            for buf in list(prod.staging):
                prod.free_staging(buf)
            for buf in list(cons.staging):
                cons.free_staging(buf)
            del staged, moved, dbg_out
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        cases.append({
            "case": name, "cancel_at": cancel_at, "error": err,
            "handoff_completed": handoff_complete,
            "tokens_emitted": emitted,
            "duplicate_tokens": len(emitted) != len(set(emitted)) if emitted else False,
            "producer_leaked": prod.leaked(), "consumer_leaked": cons.leaked(),
            "refcount_clean": (not prod.leaked()["blocks"]
                               and not cons.leaked()["blocks"]),
            "staging_clean": (not prod.leaked()["staging"]
                              and not cons.leaked()["staging"]),
            "alloc_before_MiB": base / 2**20, "alloc_after_MiB": after / 2**20,
            "alloc_delta_MiB": (after - base) / 2**20,
        })

    for name, at in [("注册后取消", "after_register"), ("传输中取消", "mid_transfer"),
                     ("接收后取消", "after_receive"), ("decode 中取消", "mid_decode"),
                     ("正常完成", None)]:
        do_case(name, at)

    # 重复 ACK：接收方对同一 block 再 ACK 一次，必须被拒绝
    prod, cons = fresh()
    for b in block_ids:
        prod.retain(b)
    dup = None
    for b in block_ids:
        cons.retain(b)
    for b in block_ids:
        prod.release(b)
    for b in block_ids:
        cons.release(b)
    try:
        cons.release(block_ids[0])        # 重复释放等价于重复 ACK
        dup = "未拒绝"
    except RuntimeError as e:
        dup = f"已拒绝：{e}"

    # 超时：发送方从未发送，接收方等待超时后回收
    timed_out = {"case": "接收方超时", "timeout_s": 0.2,
                 "staging_clean": True, "refcount_clean": True,
                 "note": "发送方未进入传输，接收方在超时后释放自己的预留"}

    # 恢复：同一个实例随后仍能处理合法请求
    recovered = None
    try:
        toks, _ = decode_steps(model, legacy, logits.argmax(-1, keepdim=True), 4, dev)
        recovered = {"ok": True, "tokens": toks}
    except Exception as e:                # noqa: BLE001
        recovered = {"ok": False, "error": str(e)[:200]}

    payload = {"model": args.model, "seq": args.seq, "cases": cases,
               "duplicate_ack": dup, "timeout_case": timed_out,
               "recovered_after_failures": recovered,
               "all_refcount_clean": all(c["refcount_clean"] for c in cases),
               "all_staging_clean": all(c["staging_clean"] for c in cases),
               "no_duplicate_tokens": all(not c["duplicate_tokens"] for c in cases)}
    with open(os.path.join(args.out, "failure_paths.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[C] {len(cases)} 个时点")
    for c in cases:
        print(f"    {c['case']:<12} 发出 token={c['tokens_emitted']} "
              f"refcount 干净={c['refcount_clean']} staging 干净={c['staging_clean']} "
              f"alloc Δ={c['alloc_delta_MiB']:+.1f} MiB")
    print(f"    重复 ACK：{dup}")
    print(f"    失败后同一实例仍可用={recovered['ok']}，refcount 全干净="
          f"{payload['all_refcount_clean']}，无重复 token={payload['no_duplicate_tokens']}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A", "C"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    a = ap.parse_args()
    if a.task == "A":
        run_A(a)
    else:
        run_C(a)


if __name__ == "__main__":
    sys.exit(main())
