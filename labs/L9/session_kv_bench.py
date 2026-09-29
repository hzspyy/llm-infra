#!/usr/bin/env python3
"""L9.3：多轮会话的 KV 生命周期。

三个模式：

``structure``（任务 A）—— 构造 2/4/8 轮会话，三种历史组织方式：
  * ``append``：每轮把工具结果追加到历史末尾（前缀保留）；
  * ``mid_insert``：把新检索到的片段插进 system 段（前缀在插入点断裂）；
  * ``trim_thinking``：下一轮的历史里去掉上一轮的 thinking（前缀在裁剪点断裂）。
  逐轮记录引擎自报的 ``prompt_tokens`` / ``cached_tokens``，并用 16 token 块模型预测命中，
  两者一致即说明「预测命中块 == 实际命中块」。

``residency``（任务 B）—— 轮间等待 0/1/10/60 s × 缓存压力（无压力 / 并发长前缀请求），
  记录每轮 TTFT、命中量、需要重算的 token；另在关闭前缀缓存的引擎上取「全部重算」参照。

``migration``（任务 C）—— 同一会话固定副本 vs 每轮迁移（迁移用 ``/reset_prefix_cache``
  模拟"下一轮落到没有该前缀的副本"），对比逐轮输出（贪心解码应逐 token 相同）与 TTFT；
  再对照不同 LoRA adapter 下的命中量，检查缓存身份是否把 adapter 算进去。

用法::

    python labs/L9/session_kv_bench.py structure --base-url ... --out DIR --turns 2 4 8
    python labs/L9/session_kv_bench.py residency --base-url ... --out DIR
    python labs/L9/session_kv_bench.py migration --base-url ... --out DIR [--lora a,b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402  （复用固定工具 schema，让模板与 9.1 一致）
from mini_prefix_accounting import BlockCache  # noqa: E402

BLOCK_SIZE = 16

SYSTEM = (
    "你是一个严谨的研究助理。你可以使用 search_corpus 工具检索 NFCorpus 医学文献库。"
    "每一轮先说明你需要什么，再调用工具；拿到结果后用一句话概括发现。"
)
TOOL = T.SEARCH_TOOL
TASK = "请检索关于 statins 与心血管风险的研究，并逐轮补充新的证据。"

TOOL_RESULTS = [
    "MED-2427 (score 12.4): Statins reduce major adverse cardiac events in high-risk patients.",
    "MED-10 (score 9.8): Cholesterol lowering and cardiovascular mortality: a cohort study.",
    "MED-3655 (score 8.1): Primary prevention with statins: number needed to treat varies widely.",
    "MED-2991 (score 7.6): Adverse effects of statin therapy in low-risk populations.",
    "MED-1379 (score 7.1): Long-term adherence to statin therapy and outcomes.",
    "MED-88 (score 6.4): Statin intensity and LDL-C reduction targets.",
    "MED-4102 (score 6.0): Statins in the elderly: benefits and risks.",
    "MED-777 (score 5.5): Biomarkers for statin response.",
]


def build_session(turns: int, mode: str, thinking: bool):
    """返回每轮的 messages 列表（按顺序，第 k 项是第 k+1 轮的输入）。"""
    plans = []
    history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK},
    ]
    for turn in range(1, turns + 1):
        plans.append([dict(m) for m in history])
        result = TOOL_RESULTS[(turn - 1) % len(TOOL_RESULTS)]
        assistant = {
            "role": "assistant",
            "content": f"第 {turn} 轮：我先检索一批文献。",
            "tool_calls": [{
                "id": f"call_{turn}",
                "type": "function",
                "function": {"name": "search_corpus",
                             "arguments": json.dumps({"query": f"statins round {turn}", "k": 5})},
            }],
        }
        tool_msg = {"role": "tool", "tool_call_id": f"call_{turn}", "content": result}
        if mode == "append":
            history = history + [assistant, tool_msg]
        elif mode == "mid_insert":
            # 把新证据插进 system 段：前缀在 system 处就变了
            new_system = dict(history[0])
            new_system["content"] = SYSTEM + f"\n（已知证据 {turn}）" + result
            history = [new_system] + [m for m in history[1:] if m["role"] != "system"] + [assistant, tool_msg]
        elif mode == "trim_thinking":
            # 模拟"下一轮把历史里的 thinking 裁掉"：assistant 段被替换成占位文本，
            # 于是共同前缀在**第一条 assistant 消息**处就断裂（不是只在末尾不同）
            trimmed = []
            for m in history:
                if m["role"] == "assistant":
                    m = dict(m)
                    m["content"] = "（思考已裁剪）"
                trimmed.append(m)
            history = trimmed + [assistant, tool_msg]
        else:
            raise ValueError(mode)
    return plans


async def reset_cache(client) -> bool:
    """清空引擎前缀缓存；用裸 HTTP 调用（openai 客户端的 post 需要 cast_to 参数）。"""
    import httpx

    base = str(client.base_url).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        async with httpx.AsyncClient(timeout=30) as hc:
            resp = await hc.post(f"{base}/reset_prefix_cache")
        ok = resp.status_code == 200 and bool(resp.json().get("success", True))
        if not ok:
            print(f"[warn] reset_prefix_cache status={resp.status_code} body={resp.text[:120]}", flush=True)
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] reset_prefix_cache failed: {exc}", flush=True)
        return False


async def send(client, model, messages, *, max_tokens, thinking, adapter=None, greedy=True):
    """一次性非流式请求；返回 usage 与输出文本。"""
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    kwargs = {}
    if adapter:
        kwargs["model"] = adapter
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=kwargs.get("model", model),
        messages=messages,
        tools=TOOL,
        tool_choice="auto",
        temperature=0.0 if greedy else 0.7,
        max_tokens=max_tokens,
        extra_body=extra,
    )
    wall = (time.perf_counter() - t0) * 1000.0
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None)
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
        "ttft_ms": None,
        "e2e_ms": round(wall, 3),
        "text": (resp.choices[0].message.content or "")[:200],
        "tool_calls": [tc.function.name for tc in (resp.choices[0].message.tool_calls or [])],
    }


def tokenizer_for(model: str):
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def tok_len(tok, messages, thinking: bool) -> list[int]:
    text = tok.apply_chat_template(messages, tools=TOOL, add_generation_prompt=True,
                                   tokenize=False,
                                   chat_template_kwargs={"enable_thinking": thinking})
    return tok(text, add_special_tokens=False)["input_ids"]


async def cmd_structure(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    tok = tokenizer_for(args.model)
    rows = []
    for mode in ("append", "mid_insert", "trim_thinking"):
        for turns in args.turns:
            for thinking in (False, True):
                cache = BlockCache(BLOCK_SIZE, args.capacity_blocks or None)
                plans = build_session(turns, mode, thinking)
                # 每种配置开局都清一次引擎缓存，让预测器与引擎从同一空状态出发
                await reset_cache(client)
                for i, messages in enumerate(plans, start=1):
                    ids = tok_len(tok, messages, thinking)
                    predicted = cache.lookup_and_insert(ids)
                    obs = await send(client, args.model, messages, max_tokens=args.max_tokens,
                                     thinking=thinking)
                    cached = obs["cached_tokens"]
                    rows.append({
                        "mode": mode, "turns": turns, "turn": i, "thinking": thinking,
                        "prompt_tokens_engine": obs["prompt_tokens"],
                        "prompt_tokens_local": len(ids),
                        "partial_tail": len(ids) % BLOCK_SIZE,
                        "predicted_cached": predicted,
                        "engine_cached": cached,
                        "delta": (cached - predicted) if cached is not None else None,
                        "recomputed_tokens": (obs["prompt_tokens"] - (cached or 0)),
                        "e2e_ms": obs["e2e_ms"],
                    })
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    await client.close()
    matched = [r for r in rows if r["engine_cached"] is not None]
    summary = {
        "config": {"model": args.model, "turns": args.turns, "block_size": BLOCK_SIZE,
                   "capacity_blocks": args.capacity_blocks},
        "rows": rows,
        "by_mode": {},
    }
    for mode in ("append", "mid_insert", "trim_thinking"):
        sub = [r for r in matched if r["mode"] == mode]
        if not sub:
            continue
        summary["by_mode"][mode] = {
            "turns_observed": len(sub),
            "exact_match": sum(1 for r in sub if r["delta"] == 0),
            "within_one_block": sum(1 for r in sub if abs(r["delta"] or 0) <= BLOCK_SIZE),
            "hit_ratio_mean": round(statistics.fmean(
                (r["engine_cached"] or 0) / max(1, r["prompt_tokens_engine"]) for r in sub), 4),
            "recomputed_tokens_mean": round(statistics.fmean(r["recomputed_tokens"] for r in sub), 1),
        }
    (out / "structure.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["by_mode"], ensure_ascii=False, indent=1))
    return 0


async def _pressure(client, model, n, tokens_target):
    """注入压力：n 条互不相同的长前缀请求，把缓存挤出去。"""
    filler = "".join(f"filler-{i}-" for i in range(tokens_target // 3))
    tasks = []
    for i in range(n):
        tasks.append(client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"[{i}] {filler} 请只回答 OK。"}],
            temperature=0.0, max_tokens=4,
        ))
    await asyncio.gather(*tasks, return_exceptions=True)


async def cmd_residency(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    rows = []
    for gap in args.gaps:
        for pressure in args.pressure:
            plans = build_session(args.session_turns, "append", False)
            for i, messages in enumerate(plans, start=1):
                if pressure > 0:
                    await _pressure(client, args.model, pressure, args.pressure_tokens)
                if gap > 0:
                    await asyncio.sleep(gap)
                obs = await send(client, args.model, messages, max_tokens=args.max_tokens,
                                 thinking=False)
                rows.append({
                    "gap_s": gap, "pressure_requests": pressure, "turn": i,
                    "prompt_tokens": obs["prompt_tokens"], "cached_tokens": obs["cached_tokens"],
                    "recomputed_tokens": obs["prompt_tokens"] - (obs["cached_tokens"] or 0),
                    "hit_ratio": round((obs["cached_tokens"] or 0) / max(1, obs["prompt_tokens"]), 4),
                    "e2e_ms": obs["e2e_ms"], "prefix_caching": args.prefix_caching,
                })
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    await client.close()
    agg = {}
    for r in rows:
        key = f"gap{r['gap_s']}|pressure{r['pressure_requests']}"
        a = agg.setdefault(key, {"turns": 0, "hit_ratio": [], "recomputed": [], "e2e_ms": []})
        a["turns"] += 1
        a["hit_ratio"].append(r["hit_ratio"])
        a["recomputed"].append(r["recomputed_tokens"])
        a["e2e_ms"].append(r["e2e_ms"])
    table = {k: {"turns": v["turns"],
                 "hit_ratio_mean": round(statistics.fmean(v["hit_ratio"]), 4),
                 "recomputed_tokens_mean": round(statistics.fmean(v["recomputed"]), 1),
                 "e2e_ms_mean": round(statistics.fmean(v["e2e_ms"]), 2)}
             for k, v in sorted(agg.items())}
    (out / "residency.json").write_text(
        json.dumps({"config": {"gaps": args.gaps, "pressure": args.pressure,
                               "prefix_caching": args.prefix_caching,
                               "session_turns": args.session_turns, "model": args.model},
                    "rows": rows, "table": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=1))
    return 0


async def cmd_migration(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    plans = build_session(args.session_turns, "append", False)
    rows = []

    async def run_policy(policy: str):
        await reset_cache(client)
        for i, messages in enumerate(plans, start=1):
            if policy == "migrate" and i > 1:
                # 迁移：下一轮落到没有该前缀的副本（本机用清空缓存模拟）
                await reset_cache(client)
            obs = await send(client, args.model, messages, max_tokens=args.max_tokens, thinking=False)
            rows.append({"policy": policy, "turn": i, "cached_tokens": obs["cached_tokens"],
                         "prompt_tokens": obs["prompt_tokens"],
                         "hit_ratio": round((obs["cached_tokens"] or 0) / max(1, obs["prompt_tokens"]), 4),
                         "e2e_ms": obs["e2e_ms"], "text": obs["text"][:80]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    await run_policy("pinned")
    await run_policy("migrate")

    # 输出一致性：贪心解码下两种策略应对同样输入给出同样输出
    pinned = {r["turn"]: r["text"] for r in rows if r["policy"] == "pinned"}
    migrated = {r["turn"]: r["text"] for r in rows if r["policy"] == "migrate"}
    same = sum(1 for t in pinned if pinned[t] == migrated.get(t))

    adapter_rows = []
    if args.lora:
        names = [n for n in args.lora.split(",") if n]
        for name in names + [None]:
            await reset_cache(client)
            messages = plans[0]
            # 同一段文本先在该 adapter 下预热一次，再测第二次的命中量
            first = await send(client, args.model, messages, max_tokens=8, thinking=False, adapter=name)
            second = await send(client, args.model, messages, max_tokens=8, thinking=False, adapter=name)
            adapter_rows.append({
                "adapter": name or "(base)",
                "first_cached": first["cached_tokens"], "first_prompt": first["prompt_tokens"],
                "second_cached": second["cached_tokens"], "second_prompt": second["prompt_tokens"],
                "second_hit_ratio": round((second["cached_tokens"] or 0) / max(1, second["prompt_tokens"]), 4),
            })
            print(json.dumps(adapter_rows[-1], ensure_ascii=False), flush=True)

    await client.close()
    summary = {
        "config": {"session_turns": args.session_turns, "model": args.model, "lora": args.lora},
        "pinned_vs_migrate": {
            "turns": len(pinned),
            "outputs_identical": same,
            "pinned_hit_ratio_mean": round(statistics.fmean(
                r["hit_ratio"] for r in rows if r["policy"] == "pinned"), 4),
            "migrate_hit_ratio_mean": round(statistics.fmean(
                r["hit_ratio"] for r in rows if r["policy"] == "migrate"), 4),
        },
        "adapter_identity": adapter_rows,
        "rows": rows,
    }
    (out / "migration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


async def cmd_adapter_identity(args) -> int:
    """adapter 是否是缓存身份的一部分：先在一个 adapter 下预热，再换 adapter 发同一段文本。"""
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    names = [n for n in args.lora.split(",") if n]
    plans = build_session(2, "append", False)
    messages = plans[0]
    rows = []
    await reset_cache(client)
    order = [names[0], names[1], names[0], names[1], None] if len(names) >= 2 else names + [None]
    for step, name in enumerate(order):
        obs = await send(client, args.model, messages, max_tokens=8, thinking=False, adapter=name)
        rows.append({
            "step": step, "adapter": name or "(base)",
            "prompt_tokens": obs["prompt_tokens"], "cached_tokens": obs["cached_tokens"],
            "hit_ratio": round((obs["cached_tokens"] or 0) / max(1, obs["prompt_tokens"]), 4),
            "text": obs["text"][:40],
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    await client.close()
    summary = {
        "config": {"model": args.model, "adapters": names, "sequence": order},
        "rows": rows,
        "note": "同一个 prompt 在同一引擎上依次换 adapter 发送；只有预热过的那一个应当命中",
    }
    (out / "adapter_identity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.3 会话 KV 生命周期")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("structure", cmd_structure), ("residency", cmd_residency),
                     ("migration", cmd_migration), ("adapter-identity", cmd_adapter_identity)):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8014/v1")
        p.add_argument("--model", default="Qwen/Qwen3-4B")
        p.add_argument("--out", required=True)
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--timeout", type=float, default=600.0)
        p.add_argument("--session-turns", type=int, default=4)
        if name == "structure":
            p.add_argument("--turns", type=int, nargs="+", default=[2, 4, 8])
            p.add_argument("--capacity-blocks", type=int, default=0)
        if name == "residency":
            p.add_argument("--gaps", type=float, nargs="+", default=[0, 1, 10, 60])
            p.add_argument("--pressure", type=int, nargs="+", default=[0, 24])
            p.add_argument("--pressure-tokens", type=int, default=1500)
            p.add_argument("--prefix-caching", action="store_true")
        if name in ("migration", "adapter-identity"):
            p.add_argument("--lora", default="")
        p.set_defaults(func=lambda a, f=fn: asyncio.run(f(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
