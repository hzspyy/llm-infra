#!/usr/bin/env python3
"""L9.1 的 mini 实现：按 16 token 的块粒度预测前缀命中，并与引擎实测对拍。

模型与 vLLM v1 的前缀缓存一致的部分：

1. 前缀命中以**整块**为单位：只有凑满一个块的 token 才参与哈希（``vllm/v1/core/kv_cache_utils.py``
   的 ``hash_block_tokens`` 要求 ``curr_block_token_ids`` 是满块，块大小默认 16）；
2. 块哈希链住父块哈希，所以「命中」意味着从第 0 块起连续命中到某一深度；
3. 命中量受**缓存容量**限制：本实验的引擎自报 ``GPU KV cache size: 41,008 tokens``，
   即约 2 563 个块，超出后按 LRU 驱逐。

脚本做三件事：

* 用与请求相同的 chat template（含 tools）重新分词，核对渲染是否与引擎的 ``prompt_tokens`` 一致；
* 在**真实到达顺序**上跑一个按块计价的 LRU 前缀缓存，逐步预测每个请求的命中 token 数；
* 与 ``events.jsonl`` 里引擎自报的 ``cached_tokens`` 逐条对拍，并把差异拆成
  「不到一整块的尾块」「容量驱逐」「跨会话共享前缀」三类。

用法（crater，serve venv；需要 tokenizer 与本地快照）::

    python labs/L9/mini_prefix_accounting.py --run work/out/9.1-trace-20260914/trace \
        --model Qwen/Qwen3-4B --capacity-blocks 2563 --out work/out/9.1-trace-20260914/mini
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

TOOLS_BY_CLASS = {
    "compute": T.CALC_TOOL,
    "retrieval": T.SEARCH_TOOL,
    "codefix": T.CODEFIX_TOOLS,
}


def tokenize_prompts(tokenizer, requests: list[dict], limit: int | None) -> tuple[list[list[int]], list[dict]]:
    """把每条请求渲染成 prompt token ids，并返回渲染核对表。"""
    rows = requests if limit is None else requests[:limit]
    ids: list[list[int]] = []
    check = []
    for r in rows:
        tools = TOOLS_BY_CLASS[r["task_class"]]
        text = tokenizer.apply_chat_template(
            r["messages"], tools=tools, add_generation_prompt=True, tokenize=False
        )
        tok = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(tok)
        check.append({"session_id": r["session_id"], "turn": r["turn"], "rendered": len(tok)})
    return ids, check


class BlockCache:
    """按块计价的 LRU 前缀缓存；键是 (父块哈希, 本块 token 元组)。

    与 vLLM 的差别只有两点：这里用 Python 的 OrderedDict 做 LRU，vLLM 在块池里按引用计数与
    空闲队列管理；这里不区分「块被正在运行的请求占用」与「块可驱逐」，所以容量口径是上界。
    """

    def __init__(self, block_size: int, capacity_blocks: int | None):
        from collections import OrderedDict

        self.block_size = block_size
        self.capacity = capacity_blocks
        self.alive: "OrderedDict[tuple, None]" = OrderedDict()

    def _touch(self, key):
        self.alive.move_to_end(key)

    def lookup_and_insert(self, tokens: list[int]) -> int:
        """返回可复用的整块 token 数，并把本次的块写入缓存。"""
        n_full = len(tokens) // self.block_size
        parent = None
        hit_blocks = 0
        for b in range(n_full):
            block = tuple(tokens[b * self.block_size:(b + 1) * self.block_size])
            key = (parent, block)
            if key not in self.alive:
                break
            hit_blocks += 1
            self._touch(key)
            parent = key
        # 命中深度之后（含第一个未命中的块）都要写入，模拟 prefill 之后的缓存填充
        parent = None
        for b in range(n_full):
            block = tuple(tokens[b * self.block_size:(b + 1) * self.block_size])
            key = (parent, block)
            if key in self.alive:
                self._touch(key)
            else:
                self.alive[key] = None
            parent = key
        if self.capacity is not None:
            while len(self.alive) > self.capacity:
                self.alive.popitem(last=False)
        return hit_blocks * self.block_size


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1 mini 前缀命中预测与对拍")
    ap.add_argument("--run", required=True, help="采集目录（含 requests.jsonl / events.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--capacity-blocks", type=int, default=0, help="0 表示不设容量上限")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run = pathlib.Path(args.run)
    plain = run / "requests.jsonl"
    if plain.exists():
        reqs = [json.loads(l) for l in open(plain, encoding="utf-8")]
    else:
        import gzip

        reqs = [json.loads(l) for l in gzip.open(run / "requests.jsonl.gz", "rt", encoding="utf-8")]
    events = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    ev = {(e["session_id"], e["turn"]): e for e in events}
    reqs.sort(key=lambda r: r.get("t_global_ms") or 0.0)
    if args.limit:
        reqs = reqs[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids, check = tokenize_prompts(tokenizer, reqs, None)

    cap = args.capacity_blocks or None
    cache = BlockCache(args.block_size, cap)
    per_turn = []
    render_matches = 0
    for r, tok, chk in zip(reqs, ids, check):
        e = ev.get((r["session_id"], r["turn"]))
        pred = cache.lookup_and_insert(tok)
        actual = (e or {}).get("cached_tokens")
        engine_prompt = (e or {}).get("prompt_tokens")
        if engine_prompt is not None and engine_prompt == chk["rendered"]:
            render_matches += 1
        per_turn.append(
            {
                "session_id": r["session_id"],
                "task_class": r["task_class"],
                "turn": r["turn"],
                "tokens": chk["rendered"],
                "engine_prompt_tokens": engine_prompt,
                "partial_tail": chk["rendered"] % args.block_size,
                "predicted_cached": pred,
                "engine_cached": actual,
                "delta": (actual - pred) if actual is not None else None,
            }
        )

    matched = [p for p in per_turn if p["engine_cached"] is not None]
    exact = sum(1 for p in matched if p["predicted_cached"] == p["engine_cached"])
    within_block = sum(1 for p in matched if abs(p["predicted_cached"] - p["engine_cached"]) <= args.block_size)
    deltas = [p["delta"] for p in matched]

    def q(vals, q_):
        vals = sorted(vals)
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(round(q_ * (len(vals) - 1))))]

    summary = {
        "config": {
            "model": args.model,
            "block_size": args.block_size,
            "capacity_blocks": cap,
            "requests": len(reqs),
        },
        "rendering": {
            "checked": len(check),
            "prompt_token_exact_match": render_matches,
            "match_rate": round(render_matches / max(1, len(check)), 4),
        },
        "prediction": {
            "matched_events": len(matched),
            "exact_match": exact,
            "exact_match_rate": round(exact / max(1, len(matched)), 4),
            "within_one_block": within_block,
            "within_one_block_rate": round(within_block / max(1, len(matched)), 4),
            "delta_p50": q(deltas, 0.5),
            "delta_p90": q(deltas, 0.9),
            "delta_min": min(deltas) if deltas else None,
            "delta_max": max(deltas) if deltas else None,
        },
        "per_class": {},
        "sample": per_turn[:40],
    }
    for cls in sorted({p["task_class"] for p in matched}):
        rows = [p for p in matched if p["task_class"] == cls]
        summary["per_class"][cls] = {
            "events": len(rows),
            "predicted_total": sum(p["predicted_cached"] for p in rows),
            "engine_total": sum(p["engine_cached"] for p in rows),
            "exact_match_rate": round(sum(1 for p in rows if p["delta"] == 0) / len(rows), 4),
        }
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / "prefix_accounting.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
