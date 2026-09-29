#!/usr/bin/env python3
"""
6.4 任务 B：并置与分离的同机对照。

两条腿：

  colocated  —— vLLM 离线引擎，扫输入 128/2048/8192 × 输出 32/256/1024，
                给出 TTFT、TPOT、goodput 与峰值显存。这是并置基线。

  1p1d       —— 同机两个模型实例构成真正的 P/D 分离：P 实例（GPU0）只做 prefill
                并产出 KV，按 kv_handoff 的协议把 KV 交给 D 实例（GPU1），
                D 实例只做 decode。TTFT 因此 = prefill + 搬运，搬运这一段是分离
                相对并置**新增**的成本，单独记；解码段的 TPOT 与并置自己的解码段比。

注意这条 1P1D 是「两个 HF 实例 + 显式 KV 交接」，不是生产 serving 栈：本机没有
NIXL / Mooncake / LMCache，也没有装 SGLang，vLLM 自带的 ExampleConnector 是磁盘路径，
没有接入。所以它测的是「分离这条路的成本结构」，不是某个引擎的 PD 性能。

用法：
    python pd_serving_bench.py colocated --model <qwen3-1.7b> --out <dir>
    python pd_serving_bench.py 1p1d --model <qwen3-1.7b> --out <dir> [--decode-instances 1|2]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_handoff as kh  # noqa: E402

COMBOS = [(128, 32), (128, 256), (128, 1024),
          (2048, 32), (2048, 256), (2048, 1024),
          (8192, 32), (8192, 256), (8192, 1024)]
REDUCED = [(128, 32), (2048, 256), (8192, 1024)]


# ==========================================================================
# 并置基线（vLLM）
# ==========================================================================

def run_colocated(args):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    combos = COMBOS if args.full else REDUCED
    max_len = max(s for s, _ in combos) + max(o for _, o in combos) + 8
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              disable_log_stats=True, enable_prefix_caching=False,
              max_model_len=max_len, gpu_memory_utilization=args.gpu_util,
              trust_remote_code=True)
    engine = llm.llm_engine
    rows = []
    for seq, out_len in combos:
        sp = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
        # 预热
        engine.add_request("warm", {"prompt_token_ids":
                                    [(1000 + i) % 5000 for i in range(64)]}, sp)
        while engine.has_unfinished_requests():
            engine.step()
        rid = f"c{seq}x{out_len}"
        t_add = time.perf_counter()
        engine.add_request(rid, {"prompt_token_ids":
                                 [(1000 + i) % 5000 for i in range(seq)]}, sp)
        t_first = t_last = None
        while engine.has_unfinished_requests():
            outs = engine.step()
            now = time.perf_counter()
            for o in outs:
                if o.request_id == rid:
                    if t_first is None and len(o.outputs[0].token_ids) > 0:
                        t_first = now
                    if o.finished:
                        t_last = now
        ttft = t_first - t_add
        tpot = (t_last - t_first) / max(1, out_len - 1)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"mode": "colocated", "seq": seq, "out": out_len,
                     "ttft_s": ttft, "tpot_s": tpot,
                     "total_s": t_last - t_add,
                     "goodput_tok_s": out_len / (t_last - t_add),
                     "peak_GiB": peak})
        print(f"  并置 S={seq:<5} out={out_len:<5} TTFT={ttft * 1e3:8.2f} ms "
              f"TPOT={tpot * 1e3:7.3f} ms 总={t_last - t_add:7.3f} s "
              f"goodput={out_len / (t_last - t_add):7.1f} tok/s 峰值={peak:.2f} GiB",
              flush=True)
    try:
        engine.engine_core.shutdown()
    except Exception:
        pass
    payload = {"mode": "colocated", "model": args.model, "rows": rows}
    _save(args.out, "colocated.json", payload)
    return payload


# ==========================================================================
# 1P1D（两个 HF 实例 + 显式 KV 交接）
# ==========================================================================

@torch.no_grad()
def decode_timed(model, past, first_token, steps, dev, on_first=None):
    ids = first_token
    cur = past
    tokens = []
    t_first = None
    t0 = time.perf_counter()
    for i in range(steps):
        out = model(input_ids=ids, past_key_values=kh.legacy_to_cache(cur),
                    use_cache=True)
        cur = kh.cache_to_legacy(out.past_key_values)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(int(nxt.item()))
        if i == 0:
            t_first = time.perf_counter()
        ids = nxt
    t_end = time.perf_counter()
    return tokens, (t_first - t0), ((t_end - t_first) / max(1, steps - 1)), (t_end - t0)


def run_1p1d(args):
    """P 实例在 GPU0、D 实例在 GPU1（可选再加 GPU2 做 1P2D 轮转）。"""
    dev_p = torch.device("cuda:0")
    torch.cuda.set_device(dev_p)
    model_p, tok = kh.load_model(args.model, dev_p)
    d_devs = [torch.device(f"cuda:{i + 1}") for i in range(args.decode_instances)]
    models_d = [kh.load_model(args.model, d)[0] for d in d_devs]
    combos = COMBOS if args.full else REDUCED
    # 预热：HF 的首次前向包含图/JIT 开销，不预热会把第一个组合的 prefill
    # 抬高一个数量级（实测 S=128 首次 751 ms，预热后同规模只要几十毫秒）。
    def warmup(model, dev):
        w = kh.build_prompt(tok, 64).to(dev)
        leg, lg = kh.prefill(model, w)
        cur = kh.kv_to_device(kh.kv_to_cpu(leg), dev)
        kh.decode_steps(model, cur, lg.argmax(-1, keepdim=True), 2, dev)
        torch.cuda.synchronize(dev if dev.type == "cuda" else None)

    warmup(model_p, dev_p)
    for m, d in zip(models_d, d_devs):
        warmup(m, d)

    rows = []
    for idx, (seq, out_len) in enumerate(combos):
        prompt = kh.build_prompt(tok, seq).to(dev_p)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        legacy, logits = kh.prefill(model_p, prompt)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        first = logits.argmax(dim=-1, keepdim=True)
        frozen = kh.kv_to_cpu(legacy)
        nbytes = kh.kv_bytes(frozen)

        d = d_devs[idx % len(d_devs)]                    # 1P2D 时轮转
        mi = idx % len(models_d)
        t0 = time.perf_counter()
        moved = kh.kv_to_device(frozen, d)
        torch.cuda.synchronize(d)
        transfer_s = time.perf_counter() - t0

        tokens, ttft_d, tpot, decode_s = decode_timed(
            models_d[mi], moved, first.to(d), out_len, d)
        total = prefill_s + transfer_s + decode_s
        rows.append({
            "mode": f"1p{args.decode_instances}d", "seq": seq, "out": out_len,
            "prefill_s": prefill_s, "transfer_s": transfer_s,
            "decode_s": decode_s, "total_s": total,
            "kv_MiB": nbytes / 2**20,
            "kv_transfer_GBps": nbytes / max(transfer_s, 1e-9) / 1e9,
            "ttft_s": prefill_s + transfer_s,
            "ttft_decode_part_s": ttft_d,
            "tpot_s": tpot,
            "goodput_tok_s": out_len / total,
            "transfer_share": transfer_s / total,
            "decode_device": str(d),
        })
        print(f"  1P{args.decode_instances}D S={seq:<5} out={out_len:<5} "
              f"prefill={prefill_s * 1e3:8.2f} ms 搬运={transfer_s * 1e3:7.2f} ms "
              f"({nbytes / max(transfer_s, 1e-9) / 1e9:5.2f} GB/s) "
              f"decode={decode_s:6.3f} s TTFT={prefill_s + transfer_s:6.3f} s "
              f"TPOT={tpot * 1e3:7.3f} ms 总={total:7.3f} s "
              f"goodput={out_len / total:6.1f} tok/s 搬运占比={transfer_s / total * 100:4.1f}%",
              flush=True)
    payload = {"mode": f"1p{args.decode_instances}d", "model": args.model,
               "kv_bytes_per_token": kh.kv_bytes(kh.kv_to_cpu(
                   kh.cache_to_legacy(kh.prefill(model_p, kh.build_prompt(tok, 128).to(dev_p))[0]))) / 128,
               "rows": rows}
    _save(args.out, f"1p{args.decode_instances}d.json", payload)
    return payload


def _save(out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["colocated", "1p1d"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.60)
    ap.add_argument("--decode-instances", type=int, default=1)
    a = ap.parse_args()
    if a.task == "colocated":
        run_colocated(a)
    else:
        run_1p1d(a)


if __name__ == "__main__":
    sys.exit(main())
