#!/usr/bin/env python3
"""L5.2 补测 · 前缀缓存与驱逐的作用域（实例本地 vs 跨实例）。

5.2-D 要给出 8.6/9.3 的缓存接口基线：**驱逐发生在哪个作用域**。本探针在同一
模型、同一段固定 token 前缀上依次做四件事，并把每步的真实命中量记下来：

  1. `ask(P)`            —— 首次，期望命中 0（建缓存）；
  2. `ask(P)`            —— 紧接着，期望命中整段前缀；
  3. `reset_prefix_cache()` 后再 `ask(P)` —— 显式驱逐路径，期望命中回到 0；
  4. 重新建缓存后 `ask(P)` —— 确认可重建。

把同一段前缀在两台机器（两个引擎实例）上分别跑一次，就能区分「实例本地驱逐」
与「跨实例共享」：B 的首次请求若命中 0，说明缓存不跨实例；A 上第 3 步的 0 说明
驱逐只作用于本实例。

用法（在各自机器上运行）：
    python prefix_scope_probe.py --tag crater --out <dir> --prefix-len 512
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L52_MODEL", "Qwen/Qwen3-1.7B")


def build(block_size=16, max_model_len=4096):
    import torch
    from vllm import LLM
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    util = min(0.55, max(free / gib - 4.0, 1.0) / (total / gib))
    return LLM(model=MODEL, max_model_len=max_model_len, disable_log_stats=False,
               enable_prefix_caching=True, enforce_eager=True,
               gpu_memory_utilization=util, block_size=block_size)


def install():
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    box: list[int] = []
    if not getattr(KVCacheManager, "_l52_scope", False):
        orig = KVCacheManager.get_computed_blocks

        def get_computed_blocks(self, request):
            blocks, n, boundary = orig(self, request)
            install.box.append(int(n))
            return blocks, n, boundary

        KVCacheManager.get_computed_blocks = get_computed_blocks
        KVCacheManager._l52_scope = True
    install.box = box
    return box


install.box: list[int] = []


def ask(llm, ids, out_len=1):
    from vllm import SamplingParams, TokensPrompt
    install.box.clear()
    t0 = time.perf_counter()
    llm.generate([TokensPrompt(prompt_token_ids=ids)],
                 SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True),
                 use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    return (install.box[-1] if install.box else 0), wall


def reset_prefix_cache(llm):
    """0.29 的显式驱逐入口；不同小版本挂在 LLM 或 engine 上，逐个尝试。"""
    for path in ("reset_prefix_cache", "llm_engine.reset_prefix_cache",
                 "llm_engine.engine_core.reset_prefix_cache"):
        obj = llm
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and callable(obj):
            try:
                obj()
                return path
            except Exception as exc:                               # noqa: BLE001
                return f"{path} 调用失败：{exc}"
    return "未找到 reset_prefix_cache 入口"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-len", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = [1000 + (i * 13) % 60000 for i in range(args.prefix_len)]
    box = install()
    out, rep = [], {}
    out.append(f"L5.2 补测 · 缓存与驱逐的作用域 · {args.tag} · {MODEL} · "
               f"prompt {args.prefix_len} token")

    llm = build()
    try:
        h1, w1 = ask(llm, ids)
        h2, w2 = ask(llm, ids)
        entry = reset_prefix_cache(llm)
        h3, w3 = ask(llm, ids)
        h4, w4 = ask(llm, ids)
        rep = dict(tag=args.tag, prefix_len=args.prefix_len, reset_entry=entry,
                   first_hit=h1, second_hit=h2, after_reset_hit=h3, rebuilt_hit=h4,
                   wall_ms=[round(x, 2) for x in (w1, w2, w3, w4)])
        out.append(f"  首次命中 {h1} token（{w1:.1f} ms）")
        out.append(f"  紧接再来 {h2} token（{w2:.1f} ms）")
        out.append(f"  显式驱逐（{entry}）后再来 {h3} token（{w3:.1f} ms）")
        out.append(f"  重建后再来 {h4} token（{w4:.1f} ms）")
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:                                          # noqa: BLE001
            pass
        del llm
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, f"scope_{args.tag}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, f"scope_{args.tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
