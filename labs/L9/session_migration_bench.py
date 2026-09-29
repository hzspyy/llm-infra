#!/usr/bin/env python3
"""L9.3 任务 D：两台真实 worker 上的会话迁移、重算与跨实例取回。

三种投递方式作用在同一条 4 轮会话上：

* ``fixed_replica``：4 轮都在 A 上（固定副本基线）；
* ``migrate_recompute``：1–2 轮在 A，3–4 轮在 B（迁移后 B 只能重算）；
* ``migrate_fetch``：1–2 轮在 A，把该会话前缀的 KV 载荷从 A 传到 B，再在 B 上跑 3–4 轮。

第三种是本章要回答的核心：**搬运本身能不能替代重算**。vLLM 0.29.0 没有把外部 KV 块注入
前缀缓存的公开接口，所以本实验把「搬运」和「重算」分开测：搬运用真实 TCP 传输同一份字节
（`kv_bytes_per_token × 前缀 token`），重算用 B 上第 3 轮的真实耗时。两者一起给出成本对照，
而「注入后命中率是否恢复」这一项明确保持 `UNVERIFIED`——不是猜，是把缺的接口写成可检查的缺口。

另外两个身份不兼容的对照：

* ``template_mismatch``：同一批消息切换 ``enable_thinking``，渲染出的 prompt 不同，缓存身份随之不同；
* adapter / 精度身份的对照沿用本章前一轮的 `adapter-identity` 工件，本脚本只复核位置/模板这一层。

每轮记录：目标 worker、prompt/命中/未命中 token、TTFT、输出哈希、缓存所在层（GPU 前缀缓存或无）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import socket
import statistics
import sys
import threading
import time

KV_BYTES_PER_TOKEN = 147_456      # Qwen3-4B（9.4 已核对）


def _now() -> float:
    return time.perf_counter()


def build_session(n_rounds: int, prefix_tokens: int, marker: str = "") -> list[list[dict]]:
    """构造一条逐轮追加的会话；每轮的消息列表是下一轮的前缀。

    ``marker`` 让每个模式使用互不相同的前缀：否则后一个模式会命中前一个模式在 B 上留下的
    缓存，把「搬运没有恢复命中」误判成「搬运恢复了命中」。
    """
    unit = ("系统提示：你是一个只回答事实的助手，回答尽量短。 "
            f"以下是背景资料（标记 {marker}），请记住它。 ")
    filler = unit * max(1, prefix_tokens // 16)
    messages: list[dict] = [{"role": "system", "content": filler},
                            {"role": "user", "content": f"第 1 问（{marker}）：背景资料的第一句是什么？"}]
    rounds = [list(messages)]
    for i in range(2, n_rounds + 1):
        messages = messages + [{"role": "assistant", "content": f"（第 {i-1} 轮回答占位）"},
                               {"role": "user", "content": f"第 {i} 问：再复述一次第一句。"}]
        rounds.append(list(messages))
    return rounds


async def one_round(base_url: str, model: str, messages: list[dict], *, thinking: bool,
                    max_tokens: int, timeout: float) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
    t0 = _now()
    ttft = None
    text: list[str] = []
    prompt_tokens = cached = completion = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
            stream=True, stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion = chunk.usage.completion_tokens
                d = getattr(chunk.usage, "prompt_tokens_details", None)
                if d is not None:
                    cached = getattr(d, "cached_tokens", None)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            piece = (getattr(delta, "content", None) or getattr(delta, "reasoning", None))
            if piece:
                if ttft is None:
                    ttft = (_now() - t0) * 1000.0
                text.append(piece)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await client.close()
    out = "".join(text)
    return {"prompt_tokens": prompt_tokens, "cached_tokens": cached,
            "completion_tokens": completion,
            "ttft_ms": round(ttft, 3) if ttft is not None else None,
            "e2e_ms": round((_now() - t0) * 1000.0, 3),
            "output_sha1": hashlib.sha1(out.encode()).hexdigest()[:12],
            "output_chars": len(out), "error": error}


async def run_mode(mode: str, rounds: list[list[dict]], *, url_a: str, url_b: str,
                   model: str, transfer: dict | None, thinking: bool,
                   max_tokens: int, timeout: float) -> dict:
    rows = []
    for i, messages in enumerate(rounds, start=1):
        if mode == "fixed_replica":
            target, url = "A", url_a
        elif mode == "migrate_recompute":
            target, url = ("A" if i <= 2 else "B"), (url_a if i <= 2 else url_b)
        elif mode == "migrate_fetch":
            target, url = ("A" if i <= 2 else "B"), (url_a if i <= 2 else url_b)
            if i == 3 and transfer:
                # 迁移时搬运该会话前缀的 KV 载荷（真实 TCP），搬运时间记进本轮
                rows.append({"round": i, "phase": "kv_transfer", **transfer})
        else:
            raise ValueError(mode)
        res = await one_round(url, model, messages, thinking=thinking,
                              max_tokens=max_tokens, timeout=timeout)
        rows.append({"round": i, "target_worker": target, "phase": "model",
                     "cache_layer": ("gpu_prefix_cache" if (res["cached_tokens"] or 0) > 0 else "none"),
                     "messages": len(messages), **res})
    return {"mode": mode, "rows": rows}


async def run_template_mismatch(rounds: list[list[dict]], url: str, model: str,
                               max_tokens: int, timeout: float) -> dict:
    """同一批消息切换 thinking：渲染不同 → 缓存身份不同。"""
    msgs = rounds[0]
    off = await one_round(url, model, msgs, thinking=False, max_tokens=max_tokens, timeout=timeout)
    on = await one_round(url, model, msgs, thinking=True, max_tokens=max_tokens, timeout=timeout)
    off2 = await one_round(url, model, msgs, thinking=False, max_tokens=max_tokens, timeout=timeout)
    return {"case": "template_mismatch",
            "thinking_off": {"prompt_tokens": off["prompt_tokens"], "cached": off["cached_tokens"]},
            "thinking_on": {"prompt_tokens": on["prompt_tokens"], "cached": on["cached_tokens"]},
            "thinking_off_again": {"prompt_tokens": off2["prompt_tokens"],
                                   "cached": off2["cached_tokens"]},
            "note": "同一 worker 上模板不同即不同身份；不同 worker 之间还要额外核对模型/精度/adapter"}


# --------------------------------------------------------------------------------------
# 跨实例传输：接收端在 B 上，发送端在 A 上
# --------------------------------------------------------------------------------------

def cmd_serve_recv(args) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    srv.settimeout(args.timeout)
    print(f"[recv] listening on {args.port}", flush=True)
    try:
        conn, addr = srv.accept()
    except socket.timeout:
        print("[recv] timeout", flush=True)
        return 1
    conn.settimeout(args.timeout)
    received = 0
    t0 = time.perf_counter()
    # 读到对端关闭为止（发送端发完 kv_bytes 就关连接）；expect_bytes 只作安全上界，
    # 不作为停止条件——否则实际载荷比预估大时会提前停止读取，发送端会收到 RST。
    with conn:
        while received < args.expect_bytes:
            chunk = conn.recv(min(1 << 20, args.expect_bytes - received))
            if not chunk:
                break
            received += len(chunk)
    elapsed = time.perf_counter() - t0
    result = {"role": "recv", "bytes": received, "expected": args.expect_bytes,
              "elapsed_s": round(elapsed, 4),
              "mbps": round(received / max(1e-9, elapsed) / 1e6, 2),
              "peer": addr[0]}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    srv.close()
    return 0


def transfer_kv(host: str, port: int, nbytes: int, timeout: float) -> dict:
    """把 nbytes 的 KV 载荷从本机发到接收端，返回实测带宽与耗时。"""
    payload = b"k" * (1 << 20)
    t0 = time.perf_counter()
    sent = 0
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        while sent < nbytes:
            n = sock.send(payload[: min(len(payload), nbytes - sent)])
            sent += n
    elapsed = time.perf_counter() - t0
    return {"role": "send", "bytes": sent, "elapsed_s": round(elapsed, 4),
            "mbps": round(sent / max(1e-9, elapsed) / 1e6, 2),
            "to": f"{host}:{port}"}


async def cmd_bench_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 每次运行用互不相同的前缀：worker 上的前缀缓存会跨实验保留，
    # 复用上一次的标记会让"B 首次见到该前缀"这个前提不成立。
    tag = args.run_tag or str(int(time.time()))
    rounds_by_mode = {m: build_session(args.rounds, args.prefix_tokens,
                                       marker=f"{m}-{tag}")
                      for m in ("fixed_replica", "migrate_recompute", "migrate_fetch")}
    rounds = rounds_by_mode["fixed_replica"]
    results: dict[str, dict] = {}

    # 先量一次基线，拿到真实 prompt token 数（基线本身也是该前缀在 A 上的首次写入）
    base = await one_round(args.worker_a, args.model, rounds[0], thinking=False,
                           max_tokens=args.max_tokens, timeout=args.timeout)
    prefix_tokens = base["prompt_tokens"] or args.prefix_tokens
    kv_bytes = KV_BYTES_PER_TOKEN * prefix_tokens
    print(f"[baseline] prompt_tokens={prefix_tokens} kv_bytes={kv_bytes} "
          f"({kv_bytes / 1e6:.1f} MB)", flush=True)

    # 跨实例搬运：真实 TCP，接收端在 B 上
    transfer = None
    if args.receiver:
        host, port = args.receiver.split(":")
        transfer = transfer_kv(host, int(port), kv_bytes, args.timeout)
        print(f"[transfer] {transfer['bytes']} bytes in {transfer['elapsed_s']}s "
              f"= {transfer['mbps']} MB/s", flush=True)

    for mode in ("migrate_fetch", "fixed_replica", "migrate_recompute"):
        res = await run_mode(mode, rounds_by_mode[mode], url_a=args.worker_a, url_b=args.worker_b,
                             model=args.model, transfer=transfer, thinking=False,
                             max_tokens=args.max_tokens, timeout=args.timeout)
        results[mode] = res
        model_rows = [r for r in res["rows"] if r["phase"] == "model"]
        print(f"[{mode}] " + " | ".join(
            f"r{r['round']}@{r['target_worker']} prompt={r['prompt_tokens']} "
            f"cached={r['cached_tokens']} ttft={r['ttft_ms']}" for r in model_rows), flush=True)

    results["template_mismatch"] = await run_template_mismatch(
        build_session(1, args.prefix_tokens, marker=f"template-{tag}"), args.worker_b, args.model,
        args.max_tokens, args.timeout)

    def model_rows(mode: str) -> list[dict]:
        return [r for r in results[mode]["rows"] if r["phase"] == "model"]

    fixed = model_rows("fixed_replica")
    recompute = model_rows("migrate_recompute")
    fetch = model_rows("migrate_fetch")
    b_recompute_ttft = next((r["ttft_ms"] for r in recompute
                             if r["round"] == 3 and r["target_worker"] == "B"), None)
    def ratio(row: dict) -> float:
        """复用率 = cached / prompt。block 粒度下永远有 1 个 16-token 的公共块会命中，
        所以判据必须用比例，不能用"cached 是否为 0"。"""
        pt = row.get("prompt_tokens") or 0
        return (row.get("cached_tokens") or 0) / pt if pt else 0.0

    inject_gap = {
        "engine_has_public_kv_import_api": False,
        "observed_cached_tokens_after_transfer": next(
            (r["cached_tokens"] for r in fetch if r["round"] == 3), None),
        "observed_reuse_ratio_after_transfer": next(
            (round(ratio(r), 4) for r in fetch if r["round"] == 3), None),
        "note": ("搬运完成了真实字节传输，但 vLLM 0.29.0 没有把外部块注入前缀缓存的公开接口，"
                 "因此 B 上第 3 轮仍然全部重算；「搬运 + 注入后命中率恢复多少」保持 UNVERIFIED"),
    }
    checks = [
        {"name": "fixed_replica_reuses_prefix",
         "expected": "固定副本时第 2-4 轮复用率 > 0.95",
         "got": [round(ratio(r), 4) for r in fixed[1:]],
         "match": all(ratio(r) > 0.95 for r in fixed[1:])},
        {"name": "migration_loses_prefix_reuse_on_target",
         "expected": "迁到 B 的第 3 轮复用率 < 0.02（只剩模板公共块）",
         "got": round(ratio(recompute[2]), 4),
         "match": ratio(recompute[2]) < 0.02},
        {"name": "second_round_on_target_reuses_new_cache",
         "expected": "B 上第 4 轮复用率 > 0.95（B 自己建立的前缀）",
         "got": round(ratio(recompute[3]), 4),
         "match": ratio(recompute[3]) > 0.95},
        {"name": "transfer_moves_real_bytes",
         "expected": "跨实例传输真实 KV 载荷并测得带宽",
         "got": transfer,
         "match": bool(transfer) and transfer["bytes"] >= kv_bytes and transfer["mbps"] > 0},
        {"name": "transfer_alone_does_not_restore_hits",
         "expected": "只搬运不注入时 B 上第 3 轮复用率仍 < 0.02（接口缺口，UNVERIFIED）",
         "got": round(ratio(fetch[2]), 4),
         "match": ratio(fetch[2]) < 0.02},
        {"name": "recompute_cost_of_first_touch_on_target",
         "expected": "B 首次处理该前缀的 TTFT 明显高于 A 的命中态（记录比值，不做阈值判定）",
         "got": {"B_first_ttft_ms": recompute[2]["ttft_ms"],
                 "A_warm_ttft_ms": fixed[1]["ttft_ms"],
                 "ratio": round((recompute[2]["ttft_ms"] or 0)
                                / max(1e-9, fixed[1]["ttft_ms"] or 1e-9), 3)},
         "match": (recompute[2]["ttft_ms"] or 0) > (fixed[1]["ttft_ms"] or 0)},
        {"name": "template_switch_keeps_common_prefix",
         "expected": "切换 thinking 改变渲染（prompt token 不同），公共前缀仍可复用",
         "got": {"off_tokens": results["template_mismatch"]["thinking_off"]["prompt_tokens"],
                 "on_tokens": results["template_mismatch"]["thinking_on"]["prompt_tokens"],
                 "on_cached": results["template_mismatch"]["thinking_on"]["cached"],
                 "off_again_cached": results["template_mismatch"]["thinking_off_again"]["cached"]},
         "match": (results["template_mismatch"]["thinking_on"]["prompt_tokens"]
                   != results["template_mismatch"]["thinking_off"]["prompt_tokens"]
                   and (results["template_mismatch"]["thinking_off_again"]["cached"] or 0) > 0)},
    ]
    report = {
        "config": {"run_tag": tag, "worker_a": args.worker_a, "worker_b": args.worker_b,
                   "model": args.model, "rounds": args.rounds,
                   "prefix_tokens": prefix_tokens, "kv_bytes": kv_bytes,
                   "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                   "receiver": args.receiver, "max_tokens": args.max_tokens},
        "baseline": base,
        "transfer": transfer,
        "results": results,
        "recompute_vs_transfer": {
            "b_round3_recompute_ttft_ms": b_recompute_ttft,
            "transfer_s": transfer["elapsed_s"] if transfer else None,
            "note": ("重算是 B 上真实测量；搬运是真实 TCP 传输时间。两者单位不同，"
                     "只能分别报告，不能相加成「取回总成本」——因为缺注入接口"),
        },
        "injection_gap": inject_gap,
        "checks": checks,
        "all_match": all(c["match"] for c in checks),
    }
    (out / "session_migration.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: "
              f"{json.dumps(c['got'], ensure_ascii=False)[:200]}")
    print("all_match:", report["all_match"])
    return 0


def cmd_bench(args) -> int:
    return asyncio.run(cmd_bench_async(args))


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.3 两 worker 会话迁移与跨实例取回")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve-recv")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--expect-bytes", type=int, default=4_000_000_000,
                   help="安全上界；接收端读到对端关闭为止")
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(func=cmd_serve_recv)

    p = sub.add_parser("bench")
    p.add_argument("--out", required=True)
    p.add_argument("--worker-a", default="http://127.0.0.1:8041/v1")
    p.add_argument("--worker-b", required=True)
    p.add_argument("--receiver", default=None, help="B 上接收端的 host:port")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--prefix-tokens", type=int, default=2048)
    p.add_argument("--run-tag", default=None,
                   help="本次运行的前缀标记；默认取当前时间戳，保证与历史运行不共享前缀")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
