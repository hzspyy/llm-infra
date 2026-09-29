#!/usr/bin/env python3
"""L5.13 超长上下文扫描：RecurrentGemma-2B 的 prefill 与 decode 随上下文的变化。"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import requests

CONTEXTS = [512, 2048, 4096, 8100]   # 8100+1 恰好不超过 max_model_len=8192
REPEATS = 3


def one(base, n_tokens, max_tokens=1):
    # prompt 长度必须留出 max_tokens 的位置，否则 8192 这一档会被直接拒绝
    prompt = "Token " * n_tokens
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/completions",
                      json={"model": "rg", "prompt": prompt,
                            "max_tokens": max_tokens, "temperature": 0.0},
                      timeout=600)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dict(target=n_tokens, error=r.text[:200], status=r.status_code)
    d = r.json()
    return dict(target=n_tokens, prompt_tokens=d["usage"]["prompt_tokens"],
                completion_tokens=d["usage"]["completion_tokens"],
                elapsed_ms=round(dt * 1000, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8143")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.contexts:
        got = []
        for rep in range(REPEATS):
            r = one(args.base, n, 1)
            r.update(rep=rep)
            got.append(r)
            print(f"  prefill target={n:>5} rep={rep}  "
                  f"{r.get('prompt_tokens')} token  {r.get('elapsed_ms')} ms"
                  f"{'  ' + str(r.get('error')) if 'error' in r else ''}", flush=True)
        ok = [g["elapsed_ms"] for g in got if "elapsed_ms" in g]
        rows.append(dict(kind="prefill", target=n, reps=got,
                         median_ms=round(statistics.median(ok), 2) if ok else None,
                         actual_tokens=got[0].get("prompt_tokens")))

    # 固定 4096 上下文下再量一次 decode：prefill 一次 + 生成 16 个 token
    for n in [4096]:
        r1 = one(args.base, n, 1)
        r16 = one(args.base, n, 16)
        if "elapsed_ms" in r1 and "elapsed_ms" in r16:
            per_tok = (r16["elapsed_ms"] - r1["elapsed_ms"]) / 15
            rows.append(dict(kind="decode", target=n,
                             ttft_ms=r1["elapsed_ms"], total16_ms=r16["elapsed_ms"],
                             per_token_ms=round(per_tok, 3),
                             prompt_tokens=r16["prompt_tokens"]))
            print(f"  decode @{n}: TTFT {r1['elapsed_ms']} ms，"
                  f"16 token 总 {r16['elapsed_ms']} ms → 每 token {per_tok:.2f} ms")

    (args.out / "longctx_scan.json").write_text(
        json.dumps(dict(rows=rows,
                        note="prompt 用 'Token ' 重复构造，实际 token 数由 usage.prompt_tokens 给出；"
                             "decode 每 token 时间由 16 与 1 两次调用差分，扣掉 prefill"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读法：prefill 时间随上下文次线性（大矩阵乘摊薄权重读取）；")
    print("      decode 每 token 时间应当近似常数——定长递推状态不随上下文增长，")
    print("      只有 8 层 attention 的 KV 随上下文线性增长。")


if __name__ == "__main__":
    main()
