#!/usr/bin/env python3
"""L9.1 任务 B/C：轨迹画像、合成负载对照、原样重放与音画会话。

子命令：

* ``profile``  —— 从 ``events.jsonl`` / ``sessions.jsonl`` 出轮次、长度、工具等待、前缀
  命中的分位数与跨轮相关性；均值匹配与分布匹配的差异在这里被量化。
* ``synth``    —— 用真实请求重建两份合成负载：``independent``（同一批轮次请求打散重排，
  保留各轮边际、破坏会话内相关与长前缀）与 ``session``（整会话有放回重采样，保留联合结构）。
* ``replay``   —— 把一份计划原样送回引擎，记录每请求 TTFT/时间/命中，并与原始轨迹对比（重放
  一致性检查）；支持固定并发与泊松到达两种施加方式。
* ``router``   —— 在真实轨迹上离线计算 round-robin 与会话亲和两种路由的前缀命中差异
  （基于实测 ``cached_tokens``，属分析而非在线路由实测）。
* ``audio``    —— 按 4.11 的 chunk/epoch/打断定义生成音画会话事件，与文本事件共用同一套时间
  字段（``t_arrive_ms`` / ``t_start_ms`` / ``t_end_ms`` / ``epoch``）。

用法示例::

    python labs/L9/trace_replay.py profile --run "$RUN/trace" --out "$OUT/profile"
    python labs/L9/trace_replay.py synth   --run "$RUN/trace" --out "$OUT/synth"
    python labs/L9/trace_replay.py replay  --run "$RUN/trace" --plan original \
        --base-url http://127.0.0.1:8011/v1 --out "$OUT/replay-original-c16"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

TOOLS_BY_CLASS = {
    "compute": T.CALC_TOOL,
    "retrieval": T.SEARCH_TOOL,
    "codefix": T.CODEFIX_TOOLS,
}

TEXT_START = "t_arrive_ms"


# --------------------------------------------------------------------------------------
# 通用统计工具
# --------------------------------------------------------------------------------------

def quantiles(values: list[float], qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    out = {}
    for q in qs:
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        out[f"p{int(q * 100)}"] = round(vals[idx], 3)
    out["mean"] = round(statistics.fmean(vals), 3)
    out["n"] = len(vals)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = statistics.fmean(p[0] for p in pairs)
    my = statistics.fmean(p[1] for p in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def load_run(run: pathlib.Path) -> tuple[list[dict], list[dict]]:
    events = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    sessions = [json.loads(l) for l in open(run / "sessions.jsonl", encoding="utf-8")]
    return events, sessions


# --------------------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------------------

def cmd_profile(args) -> int:
    run = pathlib.Path(args.run)
    events, sessions = load_run(run)
    by_class_events: dict[str, list[dict]] = {}
    for e in events:
        by_class_events.setdefault(e["task_class"], []).append(e)
    by_class_sessions: dict[str, list[dict]] = {}
    for s in sessions:
        by_class_sessions.setdefault(s["task_class"], []).append(s)

    out: dict = {"run": str(run), "sessions": len(sessions), "model_calls": len(events), "classes": {}}
    for cls in sorted(by_class_events):
        ev = by_class_events[cls]
        ss = by_class_sessions.get(cls, [])
        per_session_turns = [s["turns"] for s in ss]
        # 每次调用一条事件；只有带 tool_wait_ms 的是工具轮
        tool_ev = [e for e in ev if e.get("tool_wait_ms") is not None]
        hit_ratio = [
            (e["cached_tokens"] or 0) / e["prompt_tokens"]
            for e in ev
            if e.get("prompt_tokens")
        ]
        # 跨轮相关性：会话内的输出长度自相关、轮次-输出关系、prompt-输出关系
        lag1 = []
        turn_vs_out = []
        for s in ss:
            rows = sorted([e for e in ev if e["session_id"] == s["session_id"]], key=lambda e: e["turn"])
            outs = [e["completion_tokens"] or 0 for e in rows]
            for i in range(len(outs) - 1):
                lag1.append((outs[i], outs[i + 1]))
            for e in rows:
                turn_vs_out.append((e["turn"], e["completion_tokens"] or 0))
        out["classes"][cls] = {
            "sessions": len(ss),
            "task_success": round(sum(1 for s in ss if s.get("score") == 1.0) / max(1, len(ss)), 4),
            "error_sessions": sum(1 for s in ss if s.get("error")),
            "turns": quantiles(per_session_turns),
            "turns_histogram": {str(t): sum(1 for x in per_session_turns if x == t) for t in sorted(set(per_session_turns))},
            "model_calls": len(ev),
            "tool_calls": sum(s["tool_calls"] for s in ss),
            "prompt_tokens_per_turn": quantiles([e["prompt_tokens"] for e in ev]),
            "completion_tokens_per_turn": quantiles([e["completion_tokens"] for e in ev]),
            "reasoning_tokens_per_turn": quantiles([e["reasoning_tokens"] for e in ev]),
            "cached_tokens_per_turn": quantiles([e["cached_tokens"] or 0 for e in ev]),
            "cache_hit_ratio": quantiles(hit_ratio),
            "ttft_ms": quantiles([e["ttft_ms"] for e in ev]),
            "e2e_ms": quantiles([e["e2e_ms"] for e in ev]),
            "tool_wait_ms": quantiles([e["tool_wait_ms"] for e in tool_ev]),
            "tool_calls_per_turn": quantiles([len(e["tool_names"]) for e in ev]),
            "common_prefix_chars": quantiles([e["common_prefix_chars"] for e in ev]),
            "prefix_share": quantiles(
                [e["common_prefix_chars"] / e["prompt_chars"] for e in ev if e.get("prompt_chars")]
            ),
            "corr_turn_vs_completion": pearson([x for x, _ in turn_vs_out], [y for _, y in turn_vs_out]),
            "corr_prompt_vs_completion": pearson(
                [e["prompt_tokens"] for e in ev], [e["completion_tokens"] for e in ev]
            ),
            "corr_cached_vs_prompt": pearson(
                [e["prompt_tokens"] for e in ev], [e["cached_tokens"] or 0 for e in ev]
            ),
            "autocorr_completion_lag1": pearson([a for a, _ in lag1], [b for _, b in lag1]),
            "first_turn_completion_vs_turns": pearson(
                [(next(e for e in ev if e["session_id"] == s["session_id"] and e["turn"] == 1)["completion_tokens"] or 0)
                 for s in ss],
                [s["turns"] for s in ss],
            ),
            "thinking": sorted({e["thinking"] for e in ev}),
        }
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / "profile.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
    return 0


# --------------------------------------------------------------------------------------
# synth
# --------------------------------------------------------------------------------------

def load_requests(run: pathlib.Path) -> list[dict]:
    """读取逐轮请求；仓库里归档为 ``requests.jsonl.gz``，现场目录是未压缩的。"""
    plain = run / "requests.jsonl"
    if plain.exists():
        fh = open(plain, encoding="utf-8")
    else:
        import gzip

        fh = gzip.open(run / "requests.jsonl.gz", "rt", encoding="utf-8")
    with fh:
        return [json.loads(l) for l in fh]


def build_plans(run: pathlib.Path, seed: int = 0) -> dict:
    """返回两份重放计划与它们的离线前缀结构。"""
    reqs = load_requests(run)
    by_session: dict[str, list[dict]] = {}
    for r in reqs:
        by_session.setdefault(r["session_id"], []).append(r)
    for rows in by_session.values():
        rows.sort(key=lambda r: r["turn"])
    session_ids = sorted(by_session)
    rng = random.Random(seed)

    # session：整会话有放回重采样，保留联合结构与长前缀
    session_plan = []
    for new_id, sid in enumerate(rng.choices(session_ids, k=len(session_ids))):
        for r in by_session[sid]:
            session_plan.append(dict(r, session_id=f"s{new_id}", source_session=sid))

    # independent：按轮次分别打散，保留每轮边际，破坏会话内相关与跨轮前缀
    by_turn: dict[int, list[dict]] = {}
    for r in reqs:
        by_turn.setdefault(r["turn"], []).append(r)
    pools = {t: list(rows) for t, rows in by_turn.items()}
    for rows in pools.values():
        rng.shuffle(rows)
    max_turn = max(pools)
    independent_plan = []
    order = list(range(len(session_ids)))
    rng.shuffle(order)
    for new_id in order:
        for t in range(1, max_turn + 1):
            pool = pools.get(t)
            if not pool:
                continue
            r = pool.pop()
            independent_plan.append(dict(r, session_id=f"i{new_id}", source_session=r["session_id"]))

    # multiset：请求多重集合不变、次序整体扰动，既无会话结构也无依赖
    multiset_plan = [dict(r) for r in reqs]
    rng.shuffle(multiset_plan)

    def prefix_stats(plan: list[dict]) -> dict:
        per_session: dict[str, list[str]] = {}
        for r in plan:
            per_session.setdefault(r["session_id"], []).append(
                "\n".join(f"<{m['role']}>{m.get('content') or ''}" for m in r["messages"])
            )
        shares, firsts = [], []
        for rows in per_session.values():
            prev = ""
            for text in rows:
                common = 0
                for a, b in zip(prev, text):
                    if a != b:
                        break
                    common += 1
                shares.append(common / max(1, len(text)))
                if prev:
                    firsts.append(common)
                prev = text
        return {
            "requests": len(plan),
            "sessions": len(per_session),
            "prefix_share": quantiles(shares),
            "consecutive_common_chars": quantiles(firsts),
            "total_chars": sum(len(t) for rows in per_session.values() for t in rows),
        }

    return {
        "session_plan": session_plan,
        "independent_plan": independent_plan,
        "multiset_plan": multiset_plan,
        "session_stats": prefix_stats(session_plan),
        "independent_stats": prefix_stats(independent_plan),
        "multiset_stats": prefix_stats(multiset_plan),
    }


def cmd_synth(args) -> int:
    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = build_plans(run, args.seed)
    for name in ("session", "independent", "multiset"):
        rows = plans[f"{name}_plan"]
        with open(out / f"{name}_plan.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "session": plans["session_stats"],
        "independent": plans["independent_stats"],
        "multiset": plans["multiset_stats"],
        "note": "三份计划的请求多重集合相同（同一批真实请求），差别只在会话内联合结构与次序",
    }
    (out / "synth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------

async def _send_one(client, model, req, max_tokens, temperature, sem, records, arrival=None):
    async with sem:
        if arrival is not None:
            await asyncio.sleep(arrival)
        t0 = time.perf_counter()
        ttft = None
        text_parts: list[str] = []
        prompt_tokens = completion_tokens = cached = None
        error = None
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=req["messages"],
                tools=TOOLS_BY_CLASS.get(req["task_class"]),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs": {"enable_thinking": req.get("thinking", False)}},
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    d = getattr(chunk.usage, "prompt_tokens_details", None)
                    if d is not None:
                        cached = getattr(d, "cached_tokens", None)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                piece = (getattr(delta, "content", None)
                         or getattr(delta, "reasoning_content", None)
                         or getattr(delta, "reasoning", None))
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    text_parts.append(piece)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        e2e = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "session_id": req["session_id"],
                "task_id": req["task_id"],
                "task_class": req["task_class"],
                "turn": req["turn"],
                "source_session": req.get("source_session"),
                "t_arrive_ms": round((arrival or 0.0) * 1000.0, 3),
                "t_start_ms": round((time.perf_counter() - t0) * 1000.0 - (0 if error else 0), 3),
                "t_end_ms": round(e2e, 3),
                "ttft_ms": round(ttft, 3) if ttft is not None else None,
                "e2e_ms": round(e2e, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached,
                "chars": sum(len(t) for t in text_parts),
                "error": error,
            }
        )


async def cmd_replay_async(args) -> int:
    from openai import AsyncOpenAI

    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plan == "original":
        plan = load_requests(run)
    else:
        plans = build_plans(run, args.seed)
        plan = plans[f"{args.plan}_plan"]
    if args.limit:
        plan = plan[: args.limit]
    if args.shuffle:
        random.Random(args.seed).shuffle(plan)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    records: list[dict] = []
    arrivals = None
    if args.arrival_rate and args.arrival_rate > 0:
        rng = random.Random(args.seed)
        arrivals = [rng.expovariate(args.arrival_rate) for _ in plan]
    t0 = time.perf_counter()
    tasks = [
        _send_one(client, args.model, req, args.max_tokens, args.temperature, sem, records,
                  arrivals[i] if arrivals else None)
        for i, req in enumerate(plan)
    ]
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    await client.close()

    with open(out / "replay_requests.jsonl", "w", encoding="utf-8") as fh:
        for r in sorted(records, key=lambda r: (r["session_id"], r["turn"])):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in records if not r["error"]]
    summary = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "plan": args.plan,
            "requests": len(plan),
            "concurrency": args.concurrency,
            "arrival_rate_rps": args.arrival_rate,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "wall_s": round(wall, 3),
        "requests": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "throughput_req_s": round(len(records) / wall, 3),
        "ttft_ms": quantiles([r["ttft_ms"] for r in ok]),
        "e2e_ms": quantiles([r["e2e_ms"] for r in ok]),
        "prompt_tokens": quantiles([r["prompt_tokens"] for r in ok]),
        "completion_tokens": quantiles([r["completion_tokens"] for r in ok]),
        "cache_hit_ratio": quantiles(
            [(r["cached_tokens"] or 0) / r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
        ),
        "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "total_cached_tokens": sum(r["cached_tokens"] or 0 for r in ok),
    }
    (out / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def cmd_replay(args) -> int:
    return asyncio.run(cmd_replay_async(args))


# --------------------------------------------------------------------------------------
# 重放一致性：原始轨迹 vs 重放
# --------------------------------------------------------------------------------------

def cmd_consistency(args) -> int:
    run = pathlib.Path(args.run)
    events, _sessions = load_run(run)
    rep = [json.loads(l) for l in open(pathlib.Path(args.replay) / "replay_requests.jsonl", encoding="utf-8")]
    orig = {(e["session_id"], e["turn"]): e for e in events}
    pairs = []
    for r in rep:
        key = (r.get("source_session") or r["session_id"], r["turn"])
        o = orig.get(key)
        if o is None or r["error"]:
            continue
        pairs.append(
            {
                "key": f"{key[0]}#{key[1]}",
                "orig_prompt": o["prompt_tokens"],
                "rep_prompt": r["prompt_tokens"],
                "prompt_equal": int(o["prompt_tokens"] == r["prompt_tokens"]),
                "orig_ttft": o["ttft_ms"],
                "rep_ttft": r["ttft_ms"],
                "orig_completion": o["completion_tokens"],
                "rep_completion": r["completion_tokens"],
                "orig_cached": o["cached_tokens"],
                "rep_cached": r["cached_tokens"],
            }
        )
    matched = [p for p in pairs if p["orig_prompt"] is not None]
    summary = {
        "matched_requests": len(pairs),
        "prompt_token_match_rate": round(
            sum(p["prompt_equal"] for p in matched) / max(1, len(matched)), 4
        ),
        "prompt_tokens_max_abs_diff": max(
            (abs((p["orig_prompt"] or 0) - (p["rep_prompt"] or 0)) for p in matched), default=None
        ),
        "ttft_ms_original": quantiles([p["orig_ttft"] for p in matched]),
        "ttft_ms_replay": quantiles([p["rep_ttft"] for p in matched]),
        "completion_tokens_original": quantiles([p["orig_completion"] for p in matched]),
        "completion_tokens_replay": quantiles([p["rep_completion"] for p in matched]),
        "cached_tokens_original": quantiles([p["orig_cached"] or 0 for p in matched]),
        "cached_tokens_replay": quantiles([p["rep_cached"] or 0 for p in matched]),
        "note": "重放使用贪心解码，长度不保证逐条相同；一致性按分布与 prompt token 数核对",
    }
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / "consistency.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs[:500]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# router：离线分析会话亲和 vs round-robin 的前缀命中差
# --------------------------------------------------------------------------------------

def cmd_router(args) -> int:
    """在真实轨迹上离线比较两种路由策略能吃到的前缀命中。

    口径：把每个请求按会话到达顺序排成一条全局序列。``round_robin`` 把请求轮流发到各副本，
    于是同一会话的相邻两轮常常落在不同副本上，只有前一轮恰好在同一副本时才可能命中；
    ``affinity`` 把整个会话固定在一个副本上（副本由当前累计服务时间最小者决定），此时逐轮
    命中量与单引擎实测的 ``cached_tokens`` 相同。两者都用实测命中 token 数计价，属离线分析。
    """
    run = pathlib.Path(args.run)
    events, _sessions = load_run(run)
    by_session: dict[str, list[dict]] = {}
    for e in events:
        by_session.setdefault(e["session_id"], []).append(e)
    for rows in by_session.values():
        rows.sort(key=lambda e: e["turn"])
    ordered = sorted(
        by_session,
        key=lambda sid: min((e.get("t_global_ms") if e.get("t_global_ms") is not None else 0.0) for e in by_session[sid]),
    )

    replicas = args.replicas
    results: dict[str, dict] = {}
    for policy in ("round_robin", "least_loaded", "affinity"):
        load = [0.0] * replicas
        assignment: dict[str, int] = {}
        cached = prompt = 0
        reuse_turns = total_turns = 0
        if policy == "affinity":
            for sid in ordered:
                target = min(range(replicas), key=lambda i: load[i])
                assignment[sid] = target
                load[target] += sum(e["e2e_ms"] for e in by_session[sid]) / 1000.0
        req_idx = 0
        last_replica: dict[str, int] = {}
        for sid in ordered:
            for i, e in enumerate(by_session[sid]):
                if policy == "affinity":
                    r = assignment[sid]
                elif policy == "round_robin":
                    r = req_idx % replicas
                else:  # least_loaded：逐请求选当前累计服务时间最小的副本，不看 session
                    r = min(range(replicas), key=lambda k: load[k])
                    load[r] += e["e2e_ms"] / 1000.0
                req_idx += 1
                prompt += e["prompt_tokens"] or 0
                total_turns += 1
                if i > 0 and last_replica.get(sid) == r:
                    cached += e["cached_tokens"] or 0
                    reuse_turns += 1
                last_replica[sid] = r
        results[policy] = {
            "cached_tokens": cached,
            "prompt_tokens": prompt,
            "cache_hit_ratio": round(cached / max(1, prompt), 4),
            "reusable_turns": reuse_turns,
            "turns": total_turns,
            "load_s": [round(x, 3) for x in load],
        }
    summary = {
        "replicas": replicas,
        "sessions": len(by_session),
        "measured_single_engine_cached_tokens": sum(e["cached_tokens"] or 0 for e in events),
        "measured_single_engine_cache_hit_ratio": round(
            sum(e["cached_tokens"] or 0 for e in events) / max(1, sum(e["prompt_tokens"] or 0 for e in events)), 4
        ),
        **results,
        "note": (
            "离线归属分析：命中量按实测逐轮 cached_tokens 计价，假设只有同副本上的相邻轮次能复用；"
            "副本容量为虚设，负载用会话内 e2e 之和近似，不代表真实 8.1 路由实测"
        ),
    }
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / "router.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# audio：4.11 的 chunk/epoch 会话事件（与文本事件共用时间字段）
# --------------------------------------------------------------------------------------

def cmd_audio(args) -> int:
    """按 4.11 的参数生成音画会话：talker 每帧 85 ms、播放 80 ms、有界队列、epoch 打断。"""
    rng = random.Random(args.seed)
    gen_ms = args.gen_ms
    play_ms = args.play_ms
    events = []
    sessions_summary = []
    for s in range(args.sessions):
        session_id = f"audio-{s:03d}"
        epoch = 0
        queue: list[int] = []
        t = 0.0
        played = dropped_full = dropped_epoch = underrun = 0
        first_playable = None
        interrupt_at = rng.choice([None, 2, 5, 12, 20])
        for chunk in range(args.chunks):
            t_arrive = t
            t_gen_start = t
            t_gen_end = t + gen_ms
            t = t_gen_end
            if interrupt_at is not None and chunk == interrupt_at:
                epoch += 1
                dropped_epoch += len(queue)
                queue.clear()
                events.append(
                    {
                        "session_id": session_id,
                        "turn": chunk,
                        "chunk_idx": chunk,
                        "epoch": epoch,
                        "event": "interrupt",
                        "t_arrive_ms": round(t_arrive, 3),
                        "t_start_ms": round(t_gen_start, 3),
                        "t_end_ms": round(t_gen_end, 3),
                        "queue_len": len(queue),
                    }
                )
                continue
            if len(queue) >= args.queue_bound:
                dropped_full += 1
                events.append(
                    {
                        "session_id": session_id,
                        "turn": chunk,
                        "chunk_idx": chunk,
                        "epoch": epoch,
                        "event": "dropped_full",
                        "t_arrive_ms": round(t_arrive, 3),
                        "t_start_ms": round(t_gen_start, 3),
                        "t_end_ms": round(t_gen_end, 3),
                        "queue_len": len(queue),
                    }
                )
                continue
            queue.append(chunk)
            # 播放端以 play_ms 消费一帧
            if t - t_gen_end >= play_ms or len(queue) > 1:
                play_start = max(t_gen_end, t_gen_end)
                play_end = play_start + play_ms
                queue.pop(0)
                played += 1
                if first_playable is None:
                    first_playable = play_end - t_gen_start
                events.append(
                    {
                        "session_id": session_id,
                        "turn": chunk,
                        "chunk_idx": chunk,
                        "epoch": epoch,
                        "event": "played",
                        "t_arrive_ms": round(t_arrive, 3),
                        "t_start_ms": round(play_start, 3),
                        "t_end_ms": round(play_end, 3),
                        "queue_len": len(queue),
                    }
                )
            else:
                underrun += 1
        sessions_summary.append(
            {
                "session_id": session_id,
                "chunks": args.chunks,
                "played": played,
                "dropped_full": dropped_full,
                "dropped_epoch": dropped_epoch,
                "underrun": underrun,
                "epochs": epoch + 1,
                "first_playable_ms": round(first_playable, 3) if first_playable is not None else None,
                "gen_ms_per_chunk": gen_ms,
                "play_ms_per_chunk": play_ms,
            }
        )
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "audio_events.jsonl", "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {
        "config": {
            "sessions": args.sessions,
            "chunks_per_session": args.chunks,
            "gen_ms_per_chunk": gen_ms,
            "play_ms_per_chunk": play_ms,
            "queue_bound": args.queue_bound,
            "seed": args.seed,
        },
        "events": len(events),
        "sessions": sessions_summary,
        "time_fields": ["t_arrive_ms", "t_start_ms", "t_end_ms", "epoch", "session_id", "turn"],
        "note": "时间字段与文本重放事件共用，可直接与 trace 事件放在同一时间轴上比较",
    }
    (out / "audio_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "sessions"}, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# dagreplay：依赖驱动的重放（根任务绝对到达，后继等父节点与工具完成）
# --------------------------------------------------------------------------------------

def load_tool_waits(run: pathlib.Path) -> dict:
    """每个节点（session,turn）的工具占用时间。

    优先用 spans.jsonl 里真实的 tool_start/tool_end 跨度；没有 spans 时退回
    events.jsonl 的 ``tool_wait_ms``。返回 ``{(session_id, node_id): ms}``。
    """
    waits: dict[tuple[str, str], float] = {}
    spans_path = run / "spans.jsonl"
    if spans_path.exists():
        starts: dict[tuple[str, str], list[float]] = {}
        ends: dict[tuple[str, str], list[float]] = {}
        for line in open(spans_path, encoding="utf-8"):
            r = json.loads(line)
            key = (r["session_id"], r.get("node_id") or "")
            if r["span"] == "tool_start":
                starts.setdefault(key, []).append(r["t_ms"])
            elif r["span"] == "tool_end":
                ends.setdefault(key, []).append(r["t_ms"])
        for key, ss in starts.items():
            ee = ends.get(key, [])
            if ee:
                waits[key] = max(ee) - min(ss)
    if not waits:
        ev = run / "events.jsonl"
        if ev.exists():
            for line in open(ev, encoding="utf-8"):
                r = json.loads(line)
                if r.get("tool_wait_ms") is not None and r.get("node"):
                    waits[(r["session_id"], r["node"])] = float(r["tool_wait_ms"])
    return waits


def session_arrival_offsets(run: pathlib.Path) -> dict[str, float]:
    """每个会话相对最早到达的绝对到达偏移（秒）。

    优先用 spans.jsonl 的 ``task_arrive``（统一事件模型）；退回 events.jsonl 的 ``t_global_ms``。
    """
    arrivals: dict[str, float] = {}
    spans_path = run / "spans.jsonl"
    if spans_path.exists():
        for line in open(spans_path, encoding="utf-8"):
            r = json.loads(line)
            if r["span"] == "task_arrive":
                sid = r["session_id"]
                arrivals[sid] = min(arrivals.get(sid, float("inf")), r["t_ms"])
    if not arrivals:
        ev = run / "events.jsonl"
        if ev.exists():
            for line in open(ev, encoding="utf-8"):
                r = json.loads(line)
                sid = r["session_id"]
                t = r.get("t_global_ms")
                if t is not None:
                    arrivals[sid] = min(arrivals.get(sid, float("inf")), float(t))
    if not arrivals:
        return {}
    base = min(arrivals.values())
    return {sid: (t - base) / 1000.0 for sid, t in arrivals.items()}


async def cmd_dagreplay_async(args) -> int:
    """依赖驱动重放：每个会话是一条串行链，后继必须等父节点与工具完成。

    到达器与客户端并发槽解耦：到达时刻由一个独立循环按绝对时钟打点并立即落盘，
    会话协程随后才去竞争信号量。因此注入慢工具或队列拥塞后，所有逻辑到达仍在记录里。
    """
    from openai import AsyncOpenAI

    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs = load_requests(run)
    waits = load_tool_waits(run)
    by_session: dict[str, list[dict]] = {}
    for r in reqs:
        by_session.setdefault(r["session_id"], []).append(r)
    for rows in by_session.values():
        rows.sort(key=lambda r: r["turn"])
    sessions = sorted(by_session)
    rng = random.Random(args.seed)
    if args.plan == "session":
        # bootstrap：整会话有放回重采样，工作量另报（请求数与原清单不保证相同）
        sessions = [dict(sid=rng.choice(sessions))["sid"] for _ in sessions]
    elif args.plan == "independent":
        # 独立抽轮次：每轮从该轮池子里独立抽样，破坏会话内依赖
        by_turn: dict[int, list[dict]] = {}
        for r in reqs:
            by_turn.setdefault(r["turn"], []).append(r)
        new_plan: dict[str, list[dict]] = {}
        for i in range(len(sessions)):
            rows = []
            for t in sorted(by_turn):
                rows.append(dict(rng.choice(by_turn[t]), session_id=f"i{i}"))
            new_plan[f"i{i}"] = rows
        by_session = new_plan
        sessions = sorted(new_plan)

    t0 = time.perf_counter()
    records: list[dict] = []
    arrivals: list[dict] = []
    admit_delays: list[float] = []
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    injected = args.inject_tool_ms

    async def run_session(sid: str, rows: list[dict], arrival_ms: float) -> None:
        # 逻辑到达在到达器里已打点；这里只记录会话级起点并竞争并发槽。
        async with sem:
            admit_ms = (time.perf_counter() - t0) * 1000.0
            admit_delays.append(admit_ms - arrival_ms)
            for i, req in enumerate(rows):
                parent_ready = None
                if i > 0:
                    parent_ready = (time.perf_counter() - t0) * 1000.0
                    wait_ms = injected if injected is not None else waits.get(
                        (req.get("source_session") or sid, f"{req['task_id']}#t{req['turn'] - 1}"), 0.0
                    )
                    if wait_ms > 0:
                        await asyncio.sleep(wait_ms / 1000.0)
                dispatch_ms = (time.perf_counter() - t0) * 1000.0
                first_ms = None
                prompt_tokens = completion_tokens = cached = None
                error = None
                try:
                    stream = await client.chat.completions.create(
                        model=args.model, messages=req["messages"],
                        tools=TOOLS_BY_CLASS.get(req["task_class"]),
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        stream=True, stream_options={"include_usage": True},
                        extra_body={"chat_template_kwargs": {"enable_thinking": req.get("thinking", False)}},
                    )
                    async for chunk in stream:
                        if chunk.usage is not None:
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens
                            d = getattr(chunk.usage, "prompt_tokens_details", None)
                            if d is not None:
                                cached = getattr(d, "cached_tokens", None)
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is None:
                            continue
                        if (getattr(delta, "content", None)
                                or getattr(delta, "reasoning_content", None)
                                or getattr(delta, "reasoning", None)
                                or getattr(delta, "tool_calls", None)):
                            if first_ms is None:
                                first_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                node_end = (time.perf_counter() - t0) * 1000.0
                records.append({
                    "session_id": sid, "source_session": req.get("source_session"),
                    "task_id": req["task_id"], "task_class": req["task_class"], "turn": req["turn"],
                    "t_arrive_ms": round(arrival_ms, 3),
                    "t_admit_ms": round(admit_ms, 3),
                    "t_parent_ready_ms": round(parent_ready, 3) if parent_ready is not None else None,
                    "t_dispatch_ms": round(dispatch_ms, 3),
                    "t_first_output_ms": round(first_ms, 3) if first_ms is not None else None,
                    "t_end_ms": round(node_end, 3),
                    "parent_wait_ms": round(dispatch_ms - parent_ready, 3) if parent_ready is not None else 0.0,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cached_tokens": cached, "error": error,
                })

    # 到达器：独立于并发槽的绝对时钟。三种口径——原轨迹绝对到达 / 泊松 / 全量突发。
    if args.arrival_mode == "trace":
        offsets = session_arrival_offsets(run)
        sessions = sorted(sessions, key=lambda s: offsets.get(s, 0.0))
        gaps = None
    elif args.arrival_mode == "rate":
        if not (args.arrival_rate and args.arrival_rate > 0):
            raise SystemExit("--arrival-mode rate 需要 --arrival-rate > 0")
        gaps = [rng.expovariate(args.arrival_rate) for _ in sessions]
    else:
        gaps = [0.0] * len(sessions)
    t_wall0 = loop.time()
    t_next = t_wall0
    tasks = []
    for i, sid in enumerate(sessions):
        if args.arrival_mode == "trace":
            target = t_wall0 + offsets.get(sid, 0.0)
        elif args.arrival_mode == "rate":
            t_next += gaps[i]
            target = t_next
        else:
            target = loop.time()
        wait = target - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        arrival_ms = (time.perf_counter() - t0) * 1000.0
        arrivals.append({"session_id": sid, "t_arrive_ms": round(arrival_ms, 3),
                         "scheduled_offset_s": round(target - t_wall0, 4)})
        tasks.append(asyncio.create_task(run_session(sid, by_session[sid], arrival_ms)))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    await client.close()

    with open(out / "dag_replay_nodes.jsonl", "w", encoding="utf-8") as fh:
        for r in sorted(records, key=lambda r: r["t_dispatch_ms"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "dag_replay_arrivals.jsonl", "w", encoding="utf-8") as fh:
        for r in arrivals:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in records if not r["error"]]
    # 依赖违例：某节点在父节点结束之前就被派发（串行链下应恒为 0）
    by_key = {(r["session_id"], r["turn"]): r for r in records}
    violations = 0
    for r in records:
        prev = by_key.get((r["session_id"], r["turn"] - 1))
        if prev is None:
            continue
        if r["t_dispatch_ms"] < prev["t_end_ms"] - 1e-6:
            violations += 1
    summary = {
        "config": {
            "base_url": args.base_url, "model": args.model, "plan": args.plan,
            "concurrency": args.concurrency, "arrival_rate_rps": args.arrival_rate,
            "inject_tool_ms": injected, "arrival_mode": args.arrival_mode,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens, "seed": args.seed,
        },
        "mode": "dependency-driven",
        "wall_s": round(wall, 3),
        "sessions": len(sessions),
        "nodes": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "dependency_violations": violations,
        "logical_arrivals_recorded": len(arrivals),
        "arrivals_expected": len(sessions),
        "all_arrivals_recorded": len(arrivals) == len(sessions),
        "admission_delay_ms": quantiles(admit_delays),
        "parent_wait_ms": quantiles([r["parent_wait_ms"] for r in records if r["turn"] > 1]),
        "node_latency_ms": quantiles([r["t_end_ms"] - r["t_dispatch_ms"] for r in ok]),
        "first_output_ms": quantiles([r["t_first_output_ms"] - r["t_dispatch_ms"] for r in ok
                                      if r["t_first_output_ms"] is not None]),
        "prompt_tokens": quantiles([r["prompt_tokens"] for r in ok]),
        "completion_tokens": quantiles([r["completion_tokens"] for r in ok]),
        "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "note": (
            "依赖驱动重放按真实 DAG 发请求：后继等父节点与工具完成，故只能用于固定轨迹下的"
            "系统变量对照；它不执行新策略下的闭环任务，因此不能评分为新策略的闭环成功率"
        ),
    }
    (out / "dag_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def cmd_dagreplay(args) -> int:
    return asyncio.run(cmd_dagreplay_async(args))


def cmd_dagcompare(args) -> int:
    """对拍：原始轨迹 vs 依赖驱动重放 vs 三种请求级对照的结构与工作量。"""
    run = pathlib.Path(args.run)
    reqs = load_requests(run)
    waits = load_tool_waits(run)
    # 统一事件模型用 spans.jsonl；旧格式用 events.jsonl 兜底。
    orig_prompt_by_key: dict[tuple[str, int], float] = {}
    orig_prompt_total = 0.0
    if (run / "spans.jsonl").exists():
        for line in open(run / "spans.jsonl", encoding="utf-8"):
            s = json.loads(line)
            if s["span"] != "model_end":
                continue
            turn = int(s["node_id"].split("#t")[-1])
            orig_prompt_by_key[(s["session_id"], turn)] = s.get("prompt_tokens") or 0
            orig_prompt_total += s.get("prompt_tokens") or 0
    else:
        events, _ = load_run(run)
        for e in events:
            orig_prompt_by_key[(e["session_id"], e["turn"])] = e.get("prompt_tokens") or 0
            orig_prompt_total += e.get("prompt_tokens") or 0
    orig_nodes = len(reqs)
    orig_sessions = len({r["session_id"] for r in reqs})
    orig_edges = sum(1 for r in reqs if r["turn"] > 1)
    orig_prompt = orig_prompt_total

    report: dict = {
        "original": {
            "sessions": orig_sessions, "nodes": orig_nodes, "dependency_edges": orig_edges,
            "prompt_tokens": orig_prompt, "tool_nodes": len(waits),
            "tool_wait_ms_total": round(sum(waits.values()), 3),
        },
        "modes": {},
    }

    dag_nodes_path = pathlib.Path(args.dag) / "dag_replay_nodes.jsonl" if args.dag else None
    if dag_nodes_path and dag_nodes_path.exists():
        dag = [json.loads(l) for l in open(dag_nodes_path, encoding="utf-8")]
        orig_by_key = {}
        for r in reqs:
            orig_by_key[(r["session_id"], r["turn"])] = r
        pairs = [(orig_by_key.get((d["session_id"], d["turn"])), d) for d in dag]
        matched = [(o, d) for o, d in pairs if o is not None and not d["error"]]
        prompt_equal = 0
        max_diff = 0
        for o, d in matched:
            ot = orig_prompt_by_key.get((o["session_id"], o["turn"]))
            if ot is not None and d["prompt_tokens"] is not None:
                if ot == d["prompt_tokens"]:
                    prompt_equal += 1
                max_diff = max(max_diff, abs(ot - d["prompt_tokens"]))
        report["modes"]["dag"] = {
            "sessions": len({d["session_id"] for d in dag}),
            "nodes": len(dag),
            "dependency_edges": sum(1 for d in dag if d["turn"] > 1),
            "matched_to_original": len(matched),
            "prompt_tokens": sum(d["prompt_tokens"] or 0 for d in dag if not d["error"]),
            "prompt_token_equal_rate": round(prompt_equal / max(1, len(matched)), 4),
            "prompt_tokens_max_abs_diff": max_diff,
            "tool_latency_replayed_nodes": sum(1 for d in dag if d["turn"] > 1),
            "source": str(dag_nodes_path),
        }

    for name, path in (args.mode or []):
        p = pathlib.Path(path) / "replay_requests.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        ok = [r for r in rows if not r["error"]]
        report["modes"][name] = {
            "requests": len(rows),
            "sessions": len({r["session_id"] for r in rows}),
            "dependency_edges": None,
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "note": "请求级重放：会话结构被打散或重采样，无依赖边",
            "source": str(p),
        }

    report["bootstrap_workload_note"] = (
        "session/bootstrap 计划是有放回重采样，请求数与 prompt token 总量不必等于原清单；"
        "此处逐项另报，不声称边际完全相同"
    )
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / "dag_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1 画像/合成/重放/路由/音画")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("synth")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("replay")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan", choices=["original", "session", "independent", "multiset"], default="original")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--arrival-rate", type=float, default=0.0, help=">0 时按泊松到达发请求")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("consistency")
    p.add_argument("--run", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("router")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--replicas", type=int, default=2)
    p.set_defaults(func=cmd_router)

    p = sub.add_parser("audio")
    p.add_argument("--out", required=True)
    p.add_argument("--sessions", type=int, default=40)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--gen-ms", type=float, default=85.0)
    p.add_argument("--play-ms", type=float, default=80.0)
    p.add_argument("--queue-bound", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_audio)

    p = sub.add_parser("dagreplay")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan", choices=["dag", "session", "independent"], default="dag")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--arrival-rate", type=float, default=0.0, help=">0 时根任务按泊松绝对时刻到达")
    p.add_argument("--arrival-mode", choices=["trace", "rate", "burst"], default="trace",
                   help="根任务到达口径：trace=按原轨迹绝对到达 / rate=泊松 / burst=全量突发")
    p.add_argument("--inject-tool-ms", type=float, default=None,
                   help="不按原轨迹重放工具时延，而是注入固定延迟（毫秒）")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_dagreplay)

    p = sub.add_parser("dagcompare")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dag", default=None, help="dagreplay 的输出目录")
    p.add_argument("--mode", action="append", nargs=2, metavar=("NAME", "DIR"),
                   help="请求级对照：名字 + replay 输出目录，可重复")
    p.set_defaults(func=cmd_dagcompare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
