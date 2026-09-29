#!/usr/bin/env python3
"""L9.4 任务 A/C 的补齐：口径核对（原始 SSE + 引擎阶段直方图）与早停后的取消/KV 回收。

两个子命令各自回答一个此前留空的问题：

* ``caliber``：把**原始 SSE** 与引擎的**每请求阶段直方图**放在一起核对。
  客户端只能看到「请求发出 → 首个 delta」与「流结束」，而引擎侧把每个请求拆成
  queue / prefill / decode 三段（`vllm:request_*_time_seconds_sum|count`）。
  两者对齐之后才能回答"TTFT 里有多少是排队、多少是 prefill"——这正是 9.1 与 9.4 都缺的那一步。
  同时核对 reasoning 与 final 两条流的字段、分离点与 `reasoning_tokens` 计数。

* ``early-stop``：k 个候选分支并发跑，验证器接受一个答案后**取消其余分支**，并采样
  `num_requests_running` / `num_requests_waiting` / `kv_cache_usage_perc` 直到回到基线，
  给出「取消后多久回收、回收了多少 KV」。对照组是全部跑完（不早停）。

用法::

    python labs/L9/candidate_executor.py caliber --out out/9.4/caliber --base-url http://127.0.0.1:8061/v1
    python labs/L9/candidate_executor.py early-stop --out out/9.4/early --base-url http://127.0.0.1:8061/v1 --k 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402

PHASES = ("queue", "prefill", "decode", "inference")


def metrics_root(base_url: str) -> str:
    return (base_url[:-3] if base_url.endswith("/v1") else base_url).rstrip("/")


async def read_phases(base_url: str) -> dict:
    """读引擎的每请求阶段直方图（sum/count），用于窗口内均值。"""
    import httpx2

    out = {"raw": {}}
    async with httpx2.AsyncClient(timeout=10.0) as cli:
        text = (await cli.get(metrics_root(base_url) + "/metrics")).text
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for phase in PHASES:
            for kind in ("sum", "count"):
                key = f"vllm:request_{phase}_time_seconds_{kind}"
                if line.startswith(key + "{"):
                    out["raw"][f"{phase}_{kind}"] = float(line.rsplit(" ", 1)[1])
        for gauge in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            if line.startswith(f"vllm:{gauge}{{"):
                out[gauge] = float(line.rsplit(" ", 1)[1])
    return out


def phase_delta(before: dict, after: dict) -> dict:
    out = {}
    for phase in PHASES:
        c0 = before["raw"].get(f"{phase}_count")
        c1 = after["raw"].get(f"{phase}_count")
        s0 = before["raw"].get(f"{phase}_sum")
        s1 = after["raw"].get(f"{phase}_sum")
        if None in (c0, c1, s0, s1):
            continue
        dc = c1 - c0
        ds = s1 - s0
        out[phase] = {"requests": dc, "total_s": round(ds, 4),
                      "mean_ms": round(ds / dc * 1000.0, 3) if dc else None}
    return out


# --------------------------------------------------------------------------------------
# caliber：原始 SSE 字段 + 引擎阶段
# --------------------------------------------------------------------------------------

async def raw_sse_probe(base_url: str, model: str, prompt: str, max_tokens: int,
                        thinking: bool, out_file: pathlib.Path) -> dict:
    """不经过 SDK 直连引擎，把原始字节与逐 chunk 字段落盘。"""
    import httpx2

    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    raw = bytearray()
    t0 = time.perf_counter()
    first_reasoning = first_content = first_any = None
    reasoning_chars = content_chars = 0
    usage = None
    n_chunks = 0
    async with httpx2.AsyncClient(timeout=300.0) as cli:
        async with cli.stream("POST", metrics_root(base_url) + "/v1/chat/completions",
                              json=body) as resp:
            async for piece in resp.aiter_bytes():
                raw.extend(piece)
                for block in piece.decode("utf-8", errors="replace").split("\n\n"):
                    payload = [l[5:].lstrip() for l in block.splitlines() if l.startswith("data:")]
                    body_txt = "\n".join(payload).strip()
                    if not body_txt or body_txt == "[DONE]":
                        continue
                    try:
                        obj = json.loads(body_txt)
                    except json.JSONDecodeError:
                        continue
                    n_chunks += 1
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices") or []:
                        d = ch.get("delta") or {}
                        r = d.get("reasoning") or d.get("reasoning_content")
                        c = d.get("content")
                        now = (time.perf_counter() - t0) * 1000.0
                        if r:
                            reasoning_chars += len(r)
                            if first_reasoning is None:
                                first_reasoning = now
                            if first_any is None:
                                first_any = now
                        if c:
                            content_chars += len(c)
                            if first_content is None:
                                first_content = now
                            if first_any is None:
                                first_any = now
    out_file.write_bytes(bytes(raw))
    return {"chunks": n_chunks, "bytes": len(raw),
            "thinking": thinking,
            "reasoning_chars": reasoning_chars, "content_chars": content_chars,
            "first_reasoning_ms": round(first_reasoning, 3) if first_reasoning else None,
            "first_content_ms": round(first_content, 3) if first_content else None,
            "first_any_ms": round(first_any, 3) if first_any else None,
            "usage": usage,
            "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "raw_file": str(out_file)}


async def cmd_caliber_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task = T.build_tasks("compute", 1, args.seed)[0]
    prompt = task["prompt"]

    before = await read_phases(args.base_url)
    t0 = time.perf_counter()
    think = await raw_sse_probe(args.base_url, args.model, prompt, args.max_tokens, True,
                               out / "sse_thinking.txt")
    # 同一请求再发一次（非 thinking）作为对照
    plain = await raw_sse_probe(args.base_url, args.model, prompt, args.max_tokens, False,
                                out / "sse_plain.txt")
    after = await read_phases(args.base_url)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    phases = phase_delta(before, after)

    queue_ms = (phases.get("queue") or {}).get("mean_ms")
    prefill_ms = (phases.get("prefill") or {}).get("mean_ms")
    decode_ms = (phases.get("decode") or {}).get("mean_ms")
    checks = [
        {"name": "reasoning_and_content_are_separate_fields",
         "expected": "thinking 档的思考文本在 reasoning 字段、最终答案在 content 字段",
         "got": {"reasoning_chars": think["reasoning_chars"], "content_chars": think["content_chars"]},
         "match": think["reasoning_chars"] > 0},
        {"name": "plain_mode_has_no_reasoning",
         "expected": "关闭 thinking 时 reasoning 字段为空",
         "got": plain["reasoning_chars"],
         "match": plain["reasoning_chars"] == 0},
        {"name": "reasoning_tokens_reported_by_engine",
         "expected": "usage 里给出 reasoning_tokens",
         "got": think["reasoning_tokens"],
         "match": (think["reasoning_tokens"] or 0) > 0},
        {"name": "engine_reports_phase_times",
         "expected": "引擎给出 queue/prefill/decode 的阶段均值",
         "got": {"queue_ms": queue_ms, "prefill_ms": prefill_ms, "decode_ms": decode_ms},
         "match": all(v is not None for v in (queue_ms, prefill_ms, decode_ms))},
        {"name": "ttft_is_not_prefill_only",
         "expected": "客户端首输出时间 ≥ 引擎 prefill 均值（含排队与首 token 解码）",
         "got": {"client_first_any_ms": think["first_any_ms"], "engine_prefill_ms": prefill_ms},
         "match": (think["first_any_ms"] or 0) >= (prefill_ms or 0)},
    ]
    report = {"config": {"base_url": args.base_url, "model": args.model,
                         "max_tokens": args.max_tokens,
                         "task": {"task_id": task["task_id"],
                                  "ground_truth": task.get("ground_truth")}},
              "thinking": think, "plain": plain,
              "engine_phases_delta": phases, "wall_ms": round(wall_ms, 3),
              "checks": checks, "all_match": all(c["match"] for c in checks),
              "note": ("客户端口径只有「首输出/结束」两个点；阶段归因必须用引擎直方图。"
                       "本档 window 内只跑了 2 个请求，阶段均值即这两个请求的均值，不构成分布")}
    (out / "caliber.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {json.dumps(c['got'], ensure_ascii=False)[:200]}")
    print("all_match:", report["all_match"])
    return 0


# --------------------------------------------------------------------------------------
# early-stop：候选分支的早停、取消与回收
# --------------------------------------------------------------------------------------

async def one_candidate(base_url: str, model: str, prompt: str, max_tokens: int,
                        temperature: float, cancel_evt: asyncio.Event, results: dict,
                        idx: int) -> None:
    """一个候选分支；被取消时立刻停止读取（关闭连接）。"""
    import httpx2

    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    buf: list[str] = []
    usage = None
    cancelled = False
    t0 = time.perf_counter()
    try:
        async with httpx2.AsyncClient(timeout=300.0) as cli:
            async with cli.stream("POST", metrics_root(base_url) + "/v1/chat/completions",
                                  json=body) as resp:
                async for piece in resp.aiter_bytes():
                    if cancel_evt.is_set():
                        cancelled = True
                        break
                    for block in piece.decode("utf-8", errors="replace").split("\n\n"):
                        payload = [l[5:].lstrip() for l in block.splitlines()
                                   if l.startswith("data:")]
                        txt = "\n".join(payload).strip()
                        if not txt or txt == "[DONE]":
                            continue
                        try:
                            obj = json.loads(txt)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for ch in obj.get("choices") or []:
                            c = (ch.get("delta") or {}).get("content")
                            if c:
                                buf.append(c)
    except Exception as exc:  # noqa: BLE001
        results[idx] = {"idx": idx, "error": f"{type(exc).__name__}: {exc}"[:200],
                        "cancelled": cancelled, "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
        return
    text = "".join(buf)
    m = re.findall(r"FINAL:\s*(-?\d+)", text)
    results[idx] = {"idx": idx, "temperature": temperature, "cancelled": cancelled,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "chars": len(text), "final_found": m[-1] if m else None,
                    "completion_tokens": (usage or {}).get("completion_tokens"),
                    "reasoning_tokens": ((usage or {}).get("completion_tokens_details")
                                         or {}).get("reasoning_tokens")}


async def sample_until_idle(base_url: str, out: list[dict], stop: asyncio.Event,
                            t0: float, interval: float = 0.2) -> None:
    while not stop.is_set():
        m = await read_phases(base_url)
        out.append({"t_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "running": m.get("num_requests_running"),
                    "waiting": m.get("num_requests_waiting"),
                    "kv": m.get("kv_cache_usage_perc")})
        await asyncio.sleep(interval)


async def cmd_early_stop_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task = T.build_tasks("compute", 1, args.seed)[0]
    # ground_truth 由 make_env 挂到环境上（ComputeEnv.truth），任务字典里没有这个键
    truth = str(task["ground_truth"])
    prompt = task["prompt"]
    for mode in ("early_stop", "run_all"):
        results: dict[int, dict] = {}
        cancel = asyncio.Event()
        samples: list[dict] = []
        stop = asyncio.Event()
        t0 = time.perf_counter()
        mon = asyncio.create_task(sample_until_idle(args.base_url, samples, stop, t0))
        before = await read_phases(args.base_url)
        temps = [0.0, 0.3, 0.7, 1.0, 1.3, 1.6][: args.k]
        tasks = [asyncio.create_task(one_candidate(args.base_url, args.model, prompt,
                                                   args.max_tokens, t, cancel, results, i))
                 for i, t in enumerate(temps)]

        accepted_idx = None
        if mode == "early_stop":
            # 验证器接受第一个正确候选；随后取消其余分支
            while True:
                done = [i for i, t in enumerate(tasks) if t.done()]
                for i in done:
                    r = results.get(i) or {}
                    if r.get("final_found") == truth:
                        accepted_idx = i
                        break
                if accepted_idx is not None or all(t.done() for t in tasks):
                    break
                await asyncio.sleep(0.05)
            cancel.set()
            # 被取消的分支也要回收，等它们退出
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t0
        # 采样到空闲，记录回收时间
        drain_start = time.perf_counter()
        for _ in range(args.drain_ticks):
            await asyncio.sleep(0.2)
        after = await read_phases(args.base_url)
        stop.set()
        await asyncio.gather(mon, return_exceptions=True)
        phases = phase_delta(before, after)
        # 相对回收时间：从发出第一批请求算起，第一次看到 running=waiting=0 的时刻
        idle_at = None
        for s in samples:
            if s["running"] == 0 and (s["waiting"] or 0) == 0 and s["t_ms"] > 100.0:
                idle_at = s["t_ms"]
                break
        cancel_at = None
        if mode == "early_stop":
            cancel_at = next((s["t_ms"] for s in samples if s.get("cancelled_marker")), None)
        ok_count = sum(1 for r in results.values() if r.get("final_found") == truth)
        report_row = {
            "mode": mode, "k": args.k, "wall_s": round(wall, 3),
            "accepted_candidate": accepted_idx,
            "correct_candidates": ok_count,
            "cancelled_candidates": sum(1 for r in results.values() if r.get("cancelled")),
            "completion_tokens_total": sum(r.get("completion_tokens") or 0
                                           for r in results.values()),
            "candidates": [results.get(i) for i in range(len(temps))],
            "engine_phases": phases,
            "kv_peak": max((s["kv"] or 0) for s in samples) if samples else None,
            "running_peak": max((s["running"] or 0) for s in samples) if samples else None,
            "idle_reached_at_ms": idle_at,
            "idle_after_cancel_ms": (round(idle_at, 1) if idle_at is not None else None),
            "kv_peak_after_idle": min((s["kv"] or 0) for s in samples[-3:]) if samples else None,
            "drain_ticks": args.drain_ticks,
        }
        (out / f"{mode}.json").write_text(json.dumps({**report_row, "samples": samples},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{mode}] wall={report_row['wall_s']}s accepted={accepted_idx} "
              f"cancelled={report_row['cancelled_candidates']} correct={ok_count} "
              f"tokens={report_row['completion_tokens_total']} kv_peak={report_row['kv_peak']} "
              f"running_peak={report_row['running_peak']}", flush=True)
    early = json.loads((out / "early_stop.json").read_text(encoding="utf-8"))
    allr = json.loads((out / "run_all.json").read_text(encoding="utf-8"))
    checks = [
        {"name": "early_stop_cancels_remaining_branches",
         "expected": "早停档有候选被取消（cancelled > 0）",
         "got": early["cancelled_candidates"],
         "match": early["cancelled_candidates"] > 0},
        {"name": "early_stop_saves_tokens",
         "expected": "早停档总输出 token 少于全部跑完档",
         "got": {"early": early["completion_tokens_total"], "run_all": allr["completion_tokens_total"]},
         "match": early["completion_tokens_total"] < allr["completion_tokens_total"]},
        {"name": "early_stop_finishes_sooner",
         "expected": "早停档墙钟更短",
         "got": {"early": early["wall_s"], "run_all": allr["wall_s"]},
         "match": early["wall_s"] < allr["wall_s"]},
        {"name": "engine_reports_kv_usage",
         "expected": "采样到 KV 使用率非空（用于回收判据）",
         "got": {"early_kv_peak": early["kv_peak"], "run_all_kv_peak": allr["kv_peak"]},
         "match": (early["kv_peak"] is not None and allr["kv_peak"] is not None)},
        {"name": "cancelled_branches_release_engine_slots",
         "expected": "整个实验在 10 秒内回到 running=0（含取消后的回收）",
         "got": {"idle_reached_at_ms": early["idle_reached_at_ms"],
                 "kv_tail": early.get("kv_peak_after_idle")},
         "match": early["idle_reached_at_ms"] is not None and early["idle_reached_at_ms"] < 10000},
    ]
    report = {"config": {"base_url": args.base_url, "model": args.model, "k": args.k,
                         "max_tokens": args.max_tokens, "truth": truth},
              "early_stop": {k: v for k, v in early.items() if k != "samples"},
              "run_all": {k: v for k, v in allr.items() if k != "samples"},
              "checks": checks, "all_match": all(c["match"] for c in checks),
              "note": ("验证器只做「答案是否等于本地算出的整数」这一件事；它不看思考过程。"
                       "KV 使用率是引擎自报的瞬时值，采样间隔 0.2 s，只能给回收的量级")}
    (out / "candidate_executor.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {json.dumps(c['got'], ensure_ascii=False)[:200]}")
    print("all_match:", report["all_match"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.4 候选执行、早停与阶段口径")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("caliber")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_caliber_async(a)))

    p = sub.add_parser("early-stop")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--drain-ticks", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: asyncio.run(cmd_early_stop_async(a)))
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
