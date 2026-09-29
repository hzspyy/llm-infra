#!/usr/bin/env python3
"""L9.5 任务 C 的引擎侧一档：客户端断开后，真实引擎到底停不停、KV 什么时候还回来。

9.5 前几轮用 stub 流验证了取消在四个位置的状态机语义。任务书还要求「真实引擎侧的流式
取消」，因为**客户端侧停止读取不等于服务端停止执行**，而这一点只能在真引擎上用引擎自己
的指标读出来。本脚本对 vLLM 0.29.0（或 SGLang 0.5.19）做三件事：

1. ``control_full``：一次跑到结束的请求，量出「一次成功请求」在指标上的增量，作为对照；
2. ``disconnect``：流式请求读若干 chunk 后**关闭连接**，随后按时间采样
   ``vllm:num_requests_running`` 与 ``vllm:kv_cache_usage_perc``，记录引擎回到 0 的耗时、
   被放弃前已收到的 token 数，以及断开之后引擎**又生成了多少 token**；
3. ``abort_endpoint``（SGLang）：用引擎自己的 ``/abort_request`` 主动中止，和断开路径对照。

判定依据是引擎指标、客户端收到的 chunk 数与退出码；不使用客户端是否收到终止帧来推断服务端
状态。

用法::

    /scratch/learn/envs/serve/bin/python labs/L9/engine_stream_cancel.py run \\
        --engine vllm --base-url http://127.0.0.1:8061/v1 --model Qwen/Qwen3-4B \\
        --out out/9.5/engine-stream-cancel/vllm.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

PROMPT = ("Count from 1 to 3000, one number per line, and do not stop early. "
          "Then repeat the whole list once more.")
WATCH = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
         "vllm:generation_tokens_total", "vllm:request_success_total",
         "vllm:num_preemptions_total", "sglang:num_running_reqs", "sglang:num_queue_reqs",
         "sglang:token_usage", "sglang:generation_tokens_total", "sglang:num_requests_total")

# 两引擎的指标名不同：同一件事要在各自的名字上读，不能只读一家再让另一家返回 null。
METRIC_NAMES = {
    "running": {"vllm": "vllm:num_requests_running", "sglang": "sglang:num_running_reqs"},
    "waiting": {"vllm": "vllm:num_requests_waiting", "sglang": "sglang:num_queue_reqs"},
    "kv": {"vllm": "vllm:kv_cache_usage_perc", "sglang": "sglang:token_usage"},
    "gen": {"vllm": "vllm:generation_tokens_total", "sglang": "sglang:generation_tokens_total"},
    "success": {"vllm": "vllm:request_success_total", "sglang": "sglang:num_requests_total"},
}


def _root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def wait_metrics(base_url: str, engine: str, timeout_s: float = 15.0) -> dict[str, float]:
    """等引擎的指标面板可用（SGLang 在首个请求前后才发布 gauge）。"""
    t0 = time.perf_counter()
    last: dict[str, float] = {}
    while time.perf_counter() - t0 < timeout_s:
        last = read_metrics(base_url)
        if _agg(last, METRIC_NAMES["running"][engine]) is not None:
            return last
        time.sleep(0.5)
    return last


def read_metrics(base_url: str) -> dict[str, float]:
    """把 Prometheus 文本里关心的序列读成 {序列名+标签: 值}。"""
    import httpx

    try:
        text = httpx.get(_root(base_url) + "/metrics", timeout=30.0).text
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if not any(name.startswith(w) for w in WATCH):
            continue
        try:
            out[line.rsplit(" ", 1)[0].strip()] = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return out


def _agg(metrics: dict[str, float], name: str) -> float | None:
    vals = [v for k, v in metrics.items() if k.split("{", 1)[0] == name]
    return round(sum(vals), 4) if vals else None


def _series(metrics: dict[str, float], name: str) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k.split("{", 1)[0] == name}


def stream_and_abandon(base_url: str, model: str, read_chunks: int, hold_s: float) -> dict:
    """流式请求：读若干 chunk 后直接断开连接（关闭 HTTP 流）。"""
    import httpx

    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1024, "temperature": 0.0, "stream": True,
            "ignore_eos": True}
    chunks = 0
    text_len = 0
    finish_reason = None
    usage = None
    t0 = time.perf_counter()
    try:
        with httpx.stream("POST", url, json=body, timeout=httpx.Timeout(60.0)) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                try:
                    d = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # 思考文本的字段名两引擎不同：vLLM 是 `reasoning`，SGLang 是
                    # `reasoning_content`。只数 `content` 会把「收到 0 个 chunk」误当成
                    # 「引擎什么也没生成」，进而让断开发生在流结束之后。
                    piece = ((delta.get("content") or "")
                             + (delta.get("reasoning") or "")
                             + (delta.get("reasoning_content") or ""))
                    if piece:
                        chunks += 1
                        text_len += len(piece)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                if chunks >= read_chunks or time.perf_counter() - t0 >= hold_s:
                    break
    except Exception as exc:  # noqa: BLE001
        return {"stream_error": f"{type(exc).__name__}: {exc}", "chunks": chunks,
                "content_chars": text_len, "finish_reason": finish_reason,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}
    return {"chunks": chunks, "content_chars": text_len, "finish_reason": finish_reason,
            "usage": usage, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}


def drain_watch(base_url: str, engine: str, mark: float, max_s: float = 20.0,
                step_s: float = 0.1) -> dict:
    """断开之后按时间采样引擎状态，直到 running 回到 0。

    同时记录 generation token 计数器最后一次变化的时间——用它区分「引擎停了」与
    「引擎还在算，只是没有客户端在读」。
    """
    t0 = time.perf_counter()
    samples: list[dict] = []
    zero_at = None
    last_gen_change: float | None = None
    prev_gen = None
    while time.perf_counter() - t0 < max_s:
        m = read_metrics(base_url)
        running = _agg(m, METRIC_NAMES["running"][engine])
        kv = _agg(m, METRIC_NAMES["kv"][engine])
        gen = _agg(m, METRIC_NAMES["gen"][engine])
        t_ms = round((time.perf_counter() - t0) * 1000, 1)
        samples.append({"t_ms": t_ms, "running": running, "kv_usage": kv,
                        "generation_tokens": gen})
        if gen is not None and prev_gen is not None and gen != prev_gen:
            last_gen_change = t_ms
        if gen is not None:
            prev_gen = gen
        if running == 0 and zero_at is None and len(samples) > 2:
            zero_at = t_ms
            break
        time.sleep(step_s)
    final_gen = samples[-1]["generation_tokens"] if samples else None
    return {"samples": samples, "running_zero_at_ms": zero_at,
            "gen_tokens_last_change_ms": last_gen_change,
            "generated_tokens_total_delta": None if (final_gen is None or mark is None)
            else round(final_gen - mark, 2)}


def full_generation(base_url: str, model: str, max_tokens: int = 64) -> dict:
    import httpx

    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.perf_counter()
    try:
        r = httpx.post(url, json=body, timeout=120.0)
        d = r.json()
        return {"status": r.status_code, "finish_reason": d["choices"][0]["finish_reason"],
                "usage": d.get("usage"),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def abort_endpoint(base_url: str, model: str, hold_s: float) -> dict:
    """SGLang 的 /abort_request：先起一个流，再让引擎自己中止。"""
    import httpx

    root = _root(base_url)
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1024, "temperature": 0.0, "stream": True, "ignore_eos": True}
    out: dict = {}
    t0 = time.perf_counter()
    try:
        with httpx.stream("POST", url, json=body, timeout=httpx.Timeout(120.0)) as r:
            chunks = 0
            for line in r.iter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    chunks += 1
                if time.perf_counter() - t0 >= hold_s:
                    break
            out["chunks_before_abort"] = chunks
            try:
                ar = httpx.post(root + "/abort_request", json={"abort_all": True}, timeout=30.0)
                out["abort_endpoint_status"] = ar.status_code
                out["abort_endpoint_body"] = ar.text[:200]
            except Exception as exc:  # noqa: BLE001
                out["abort_endpoint_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["stream_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 引擎侧流式取消")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--read-chunks", type=int, default=8)
    args = ap.parse_args()

    base_url, model = args.base_url, args.model
    report: dict = {"config": {"engine": args.engine, "base_url": base_url, "model": model,
                              "repeat": args.repeat, "read_chunks": args.read_chunks}}
    m0 = wait_metrics(base_url, args.engine)
    report["metrics_baseline"] = {
        "running": _agg(m0, METRIC_NAMES["running"][args.engine]),
        "kv_usage": _agg(m0, METRIC_NAMES["kv"][args.engine]),
        "generation_tokens": _agg(m0, METRIC_NAMES["gen"][args.engine]),
        "success_total": _series(m0, METRIC_NAMES["success"][args.engine]),
        "raw_error": m0.get("_error")}
    gen_before = _agg(m0, METRIC_NAMES["gen"][args.engine])

    # 对照：一次跑完的请求
    ctrl = full_generation(base_url, model)
    m1 = read_metrics(base_url)
    gen_after_ctrl = _agg(m1, METRIC_NAMES["gen"][args.engine])
    ctrl["metrics_after"] = {
        "generation_tokens_delta": None if (gen_before is None or gen_after_ctrl is None)
        else round(gen_after_ctrl - gen_before, 2),
        "success_total": _series(m1, METRIC_NAMES["success"][args.engine])}
    report["control_full"] = ctrl

    # 断开：重复若干次，逐次观察引擎回收
    runs = []
    for i in range(args.repeat):
        before = read_metrics(base_url)
        gen_mark = _agg(before, METRIC_NAMES["gen"][args.engine])
        s = stream_and_abandon(base_url, model, args.read_chunks, hold_s=0.4)
        time.sleep(0.05)
        after_close = read_metrics(base_url)
        watch = drain_watch(base_url, args.engine, gen_mark)
        final = watch["samples"][-1] if watch["samples"] else {}
        runs.append({
            "iteration": i,
            "stream": s,
            "running_immediately_after_close": _agg(after_close,
                                                    METRIC_NAMES["running"][args.engine]),
            "kv_usage_immediately_after_close": _agg(after_close,
                                                     METRIC_NAMES["kv"][args.engine]),
            "running_zero_at_ms": watch["running_zero_at_ms"],
            "gen_tokens_last_change_ms": watch["gen_tokens_last_change_ms"],
            "kv_final": final.get("kv_usage"),
            "generation_tokens_at_disconnect": gen_mark,
            "generation_tokens_when_drained": final.get("generation_tokens"),
            "generation_tokens_delta_total": watch["generated_tokens_total_delta"],
            "client_received_chunks": s.get("chunks"),
            "samples": watch["samples"],
        })
    report["disconnect"] = runs

    # 引擎侧主动中止（SGLang 有 /abort_request；vLLM 没有该端点，记录跳过原因）
    if args.engine == "sglang":
        before = read_metrics(base_url)
        ab = abort_endpoint(base_url, model, hold_s=0.4)
        watch = drain_watch(base_url, args.engine,
                            _agg(before, METRIC_NAMES["gen"][args.engine]))
        ab["running_zero_at_ms"] = watch["running_zero_at_ms"]
        ab["samples"] = watch["samples"]
        report["abort_endpoint"] = ab
    else:
        report["abort_endpoint"] = {"skipped": "vLLM 0.29.0 的 OpenAI 服务端没有中止端点，"
                                               "只保留客户端断开路径"}

    m2 = read_metrics(base_url)
    report["metrics_final"] = {"running": _agg(m2, METRIC_NAMES["running"][args.engine]),
                              "kv_usage": _agg(m2, METRIC_NAMES["kv"][args.engine]),
                              "success_total": _series(m2, METRIC_NAMES["success"][args.engine])}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"config": report["config"], "control_full": ctrl,
                      "disconnect": [{k: v for k, v in r.items() if k != "samples"}
                                     for r in runs],
                      "abort_endpoint": report["abort_endpoint"],
                      "metrics_baseline": report["metrics_baseline"],
                      "metrics_final": report["metrics_final"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
