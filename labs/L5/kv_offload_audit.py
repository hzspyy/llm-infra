#!/usr/bin/env python3
"""L5.8 —— 真实引擎里的"可用换出路径"：vLLM 的 SimpleCPUOffloadConnector。

前面用 nanoserve 实现了换出协议；真实 vLLM 0.29.0 的抢占路径不做换出
（`scheduler.py:1405`），唯一带 CPU 块的机制是 `v1/simple_kv_offload/`
这个 **KV connector**（`SimpleCPUOffloadConnector`，注册在
`kv_transfer/kv_connector/factory.py:234`）。它依赖 prefix caching，
通过 `--kv-transfer-config` 的 `kv_connector_extra_config` 配置容量。

这个实验把它接到真实服务上，问一个可判定的问题：
**GPU KV 被挤掉之后，再请求同一段前缀，是从 CPU 换回来还是重算？**

设计（两种配置各跑一遍同样的序列）：
  1. 用较小的 GPU KV 预算 + 较长 prompt，让 cache 一定装不下全部前缀；
  2. 第一轮把 P1…Pk 全部服务一遍（写入缓存）；
  3. 中间用一批不同的 prompt 把 GPU 上的 P1 挤掉；
  4. 第二轮再请求 P1，记录它的 **TTFT**（首 token 延迟）。
     - 重算：TTFT ≈ 整个 prefill 的耗时；
     - 换回：TTFT ≈ 一次 CPU→GPU 拷贝 + 一步 decode。

判据是"第二轮 P1 的 TTFT 相对第一轮 P1 的 TTFT 少了多少"，不是文本是否相同。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL = "Qwen/Qwen3-1.7B"


def make_prompts(n_fill, fill_reps, anchor_reps):
    word = "lighthouse keeper ocean storm beam rope harbor"
    anchor = ("Explain in detail how " +
              " ".join([word] * anchor_reps) + " works, step by step.")
    fillers = [(f"[{i}] " + " ".join([word] * fill_reps) + f" case {i}.")
               for i in range(n_fill)]
    return anchor, fillers


def build(kv_transfer: dict | None, util: float, attempts: int = 2):
    """构造引擎；同一进程里连续建引擎时，上一个引擎释放显存可能落进下一个的
    profiling 窗口，触发 `init_free_memory >= free_gpu_memory` 断言。等一会儿重试。"""
    import time as _t
    from vllm import LLM
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=util, enforce_eager=True,
              disable_log_stats=False, enable_prefix_caching=True)
    if kv_transfer is not None:
        from vllm.config import KVTransferConfig
        kw["kv_transfer_config"] = KVTransferConfig(**kv_transfer)
    last = None
    for k in range(attempts):
        if k:
            _t.sleep(10)
        try:
            return LLM(**kw)
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"  [build] 第 {k + 1} 次失败：{str(e)[:120]}", flush=True)
    raise last


def ttft_ms(out):
    """直接用引擎给的 first_token_latency。

    注意不能拿 first_token_ts 减 arrival_time：前者是 engine core 的单调时钟，
    后者是前端墙钟，跨进程跨时钟相减没有意义（5.11 的边界之一）。
    """
    m = out.metrics
    if m is None:
        return None
    lat = getattr(m, "first_token_latency", None)
    return round(lat * 1000, 3) if lat else None


def run_sequence(llm, anchor, fillers, tag):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    rec = {"tag": tag, "phases": {}}

    t0 = time.perf_counter()
    a1 = llm.generate([anchor], sp, use_tqdm=False)[0]
    rec["phases"]["anchor_first"] = dict(ttft_ms=ttft_ms(a1),
                                         wall_s=round(time.perf_counter() - t0, 3))

    # 第二次：完全在 GPU 缓存里命中，作为"不重算"的参照点（同时消掉预热）
    t0 = time.perf_counter()
    a_hit = llm.generate([anchor], sp, use_tqdm=False)[0]
    rec["phases"]["anchor_gpu_hit"] = dict(ttft_ms=ttft_ms(a_hit),
                                           wall_s=round(time.perf_counter() - t0, 3))

    t0 = time.perf_counter()
    f = llm.generate(fillers, sp, use_tqdm=False)
    rec["phases"]["fillers"] = dict(n=len(fillers),
                                    wall_s=round(time.perf_counter() - t0, 3),
                                    ttft_median_ms=round(
                                        sorted(x for x in (ttft_ms(o) for o in f)
                                               if x is not None)[len(f) // 2], 3))

    # 反复"挤掉—再请求"三轮，让 3.1 ms 这种量级的差有重复支撑
    cycles = []
    for c in range(3):
        llm.generate(fillers, sp, use_tqdm=False)
        t0 = time.perf_counter()
        a2 = llm.generate([anchor], sp, use_tqdm=False)[0]
        cycles.append(dict(ttft_ms=ttft_ms(a2),
                           wall_s=round(time.perf_counter() - t0, 3)))
    rec["phases"]["anchor_second"] = cycles[0]
    rec["phases"]["anchor_after_eviction_cycles"] = cycles
    import statistics as _st
    rec["anchor_after_eviction_median_ms"] = round(
        _st.median([c["ttft_ms"] for c in cycles]), 3)
    rec["anchor_tokens_same"] = (list(a1.outputs[0].token_ids) ==
                                 list(a2.outputs[0].token_ids))
    try:
        rec["metrics"] = {getattr(m, "name", ""): getattr(m, "value", None)
                          for m in llm.get_metrics()
                          if "prefix_cache" in getattr(m, "name", "")}
    except Exception as e:
        rec["metrics"] = {"error": str(e)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--util", type=float, default=0.22)
    ap.add_argument("--anchor-reps", type=int, default=120)
    ap.add_argument("--n-fill", type=int, default=12)
    ap.add_argument("--fill-reps", type=int, default=220)
    ap.add_argument("--cpu-gb", type=float, default=2.0)
    ap.add_argument("--lazy", action="store_true",
                    help="lazy_offload=True：写在请求结束/腾位时才做，而不是每步 eager")
    ap.add_argument("--capacity-sweep", action="store_true",
                    help="扫 cpu_gb 容量（0.25/0.5/1/2/4 GiB），固定 lazy=False")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    anchor, fillers = make_prompts(args.n_fill, args.fill_reps, args.anchor_reps)
    report = dict(model=MODEL, util=args.util, n_fill=args.n_fill,
                  fill_reps=args.fill_reps, anchor_reps=args.anchor_reps,
                  cpu_bytes_to_use=int(args.cpu_gb * 1024 ** 3))

    # A：不开 offload
    llm = build(None, args.util)
    report["offload_off"] = run_sequence(llm, anchor, fillers, "offload_off")
    del llm
    gc.collect()
    _free()

    # B：开 offload（cpu 后端，容量 cpu_gb）
    def offload_cfg(cpu_bytes, lazy):
        return dict(kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
                    kv_connector_extra_config={
                        "cpu_bytes_to_use": cpu_bytes,
                        "kv_offload_backend": "cpu", "lazy_offload": lazy})

    cfg = offload_cfg(report["cpu_bytes_to_use"], args.lazy)
    llm = build(cfg, args.util)
    report["offload_on"] = run_sequence(llm, anchor, fillers, "offload_on")
    report["offload_config"] = cfg
    del llm
    gc.collect()
    _free()

    # C：容量横扫（固定 lazy=False，只改 cpu 容量）
    if args.capacity_sweep:
        report["capacity_sweep"] = []
        for gb in (0.25, 0.5, 1.0, 2.0, 4.0):
            c = offload_cfg(int(gb * 1024 ** 3), False)
            llm = build(c, args.util)
            r = run_sequence(llm, anchor, fillers, f"cap_{gb}gb")
            r["cpu_gb"] = gb
            report["capacity_sweep"].append(r)
            del llm
            gc.collect()
            _free()

    (args.out / "kv_offload.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for key in ("offload_off", "offload_on"):
        p = report[key]["phases"]
        print(f"{key:<12} 首轮 anchor TTFT {p['anchor_first']['ttft_ms']} ms  "
              f"GPU 命中 {p['anchor_gpu_hit']['ttft_ms']} ms  "
              f"（{p['anchor_first']['wall_s']} s）  填充分钟 "
              f"{p['fillers']['wall_s']} s  第二轮 anchor TTFT "
              f"{p['anchor_second']['ttft_ms']} ms（{p['anchor_second']['wall_s']} s）"
              f"  输出一致 {report[key]['anchor_tokens_same']}")
    off = report["offload_off"].get("anchor_after_eviction_median_ms")
    on = report["offload_on"].get("anchor_after_eviction_median_ms")
    hit_off = report["offload_off"]["phases"]["anchor_gpu_hit"]["ttft_ms"]
    if off and on:
        print(f"\n被挤掉后 anchor 的 TTFT 中位（3 轮）：不开 {off} ms → 开 {on} ms，"
              f"差 {off - on:.3f} ms；完全命中参照 {hit_off} ms")
    print("判读：若开 offload 后第二轮的 TTFT 明显下降（接近“不重算”），"
          "说明 KV 是从 CPU 换回来的；若两者接近，说明仍然重算。")


def _free():
    try:
        import torch
        torch.cuda.synchronize()
        time.sleep(2)
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
