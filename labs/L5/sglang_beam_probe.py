#!/usr/bin/env python3
"""L5.9 —— SGLang beam search 的响应结构探针与逐 beam 提取。

上一轮发现 `beam_width=4, n=4` 能跑通（`completion_tokens=128`），
但逐 beam 的 token 序列没取到：beam 请求的响应结构和普通请求不同。
这个脚本只做三件事：把**原始响应**落盘、打印键树、按实际结构提取逐 beam 输出。

用法（先起好 SGLang）：
    python sglang_beam_probe.py --base http://127.0.0.1:8148 --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import requests

PROMPTS = [
    "The prefix cache mechanism lets multiple requests",
    "A paged KV cache stores key and value tensors",
]
MAX_NEW = 24
BEAM_WIDTH = 4


def key_tree(o, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 3:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            extra = f" len={len(v)}" if isinstance(v, (list, str)) else ""
            out.append(f"{'  ' * depth}{prefix}{k}: {t}{extra}")
            key_tree(v, "", depth + 1, out)
    elif isinstance(o, list) and o:
        key_tree(o[0], "[0].", depth + 1, out)
    return out


def extract_beams(body):
    """逐 beam 输出在 `meta_info.beam_results` 里，顶层 `output_ids` 只是最好的一条。

    这是从原始响应里读出来的结构（`beam-raw-*.json`）：
      body.output_ids                  最好 beam 的 token id（24 个）
      body.meta_info.completion_tokens  4 条之和（96）
      body.meta_info.sequence_score     该请求的 beam 分数
      body.meta_info.beam_results[i]    {output_ids, text, meta_info}
    """
    meta = body.get("meta_info", {})
    res = meta.get("beam_results") or []
    return [dict(token_ids=c.get("output_ids") or [],
                 text=(c.get("text") or "")[:120],
                 finish_reason=(c.get("meta_info") or {}).get("finish_reason"))
            for c in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8148")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    report = dict(beam_width=BEAM_WIDTH, max_new_tokens=MAX_NEW, prompts=[])
    for i, text in enumerate(PROMPTS):
        ids = tok.encode(text, add_special_tokens=False)
        payload = {"input_ids": ids,
                   "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_NEW,
                                       "beam_width": BEAM_WIDTH, "n": BEAM_WIDTH}}
        t0 = time.perf_counter()
        r = requests.post(args.base + "/generate", json=payload, timeout=300)
        dt = (time.perf_counter() - t0) * 1000
        (args.out / f"beam-raw-{i}.json").write_text(
            json.dumps(dict(status=r.status_code, elapsed_ms=dt,
                            body=(r.json() if r.status_code == 200 else r.text[:400])),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        if r.status_code != 200:
            print(f"  prompt{i}: HTTP {r.status_code} {r.text[:120]}")
            report["prompts"].append(dict(index=i, status=r.status_code))
            continue
        body = r.json()
        beams = extract_beams(body)
        uniq = {tuple(b["token_ids"]) for b in beams}
        # 与贪心对照：直接用顶层 output_ids，不需要 logprob
        # （`return_logprob` 不是 /generate 的顶层字段，放在顶层会 TypeError）
        gr = requests.post(args.base + "/generate",
                           json={"input_ids": ids,
                                 "sampling_params": {"temperature": 0.0,
                                                     "max_new_tokens": MAX_NEW}},
                           timeout=300)
        g_ids = gr.json().get("output_ids", []) if gr.status_code == 200 else []
        report["prompts"].append(dict(
            index=i, status=200, elapsed_ms=round(dt, 1),
            n_beams=len(beams), n_unique=len(uniq),
            completion_tokens=body["meta_info"].get("completion_tokens"),
            sequence_score=body["meta_info"].get("sequence_score"),
            beams=[dict(n=len(b["token_ids"]), finish_reason=b["finish_reason"],
                        first8=b["token_ids"][:8]) for b in beams],
            greedy_ids=g_ids, beam_matches_greedy=[b["token_ids"] == g_ids for b in beams],
            raw_keys=key_tree(body)))
        print(f"  prompt{i}: {len(beams)} 条 beam（completion_tokens="
              f"{body['meta_info'].get('completion_tokens')}），去重后 {len(uniq)} 条；"
              f"与贪心相同 {report['prompts'][-1]['beam_matches_greedy']}；{dt:.0f} ms")
        print("    顶层键：" + "; ".join(report["prompts"][-1]["raw_keys"][:8]))

    (args.out / "beam_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
