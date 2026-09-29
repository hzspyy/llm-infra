#!/usr/bin/env python3
"""L5.11 任务 B 的 SGLang 端：前端分词与动态批分词器的实测。

计划要求「token 长度 32/512/4096/16384，batch=1/8/32，比较逐条、批量、
thread offload 与动态 tokenizer；在 SGLang 起真实服务」。

这里用一条不靠减不同样本中位数的分解：
**同一份 token 序列，分别以 text 和 input_ids 两种方式提交**。
两者在服务端走的是同一段推理，差别只有「前端要不要分词 / 后端要不要反分词」。
同一批请求、同一服务、同一长度下两者的差就是前端成本。

动态批分词器（`--enable-dynamic-batch-tokenizer`）开关各跑一遍，
看它把这份成本压下去多少。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LENGTHS = [32, 512, 4096]      # 可用 --lengths 覆盖（高并发小请求要用 8/32）
BATCHES = [1, 8, 32]           # 可用 --batches 覆盖（攒批收益要看 128+）
REPEATS = 3


def local_ids(text):
    """本地分词。不依赖服务的 /tokenize——0.5.19 上它的请求体约定不同，
    返回 400；而且这里要的只是"同一份 token 内容"，本地算更可控。"""
    return _TOK.encode(text, add_special_tokens=False)


def one_request(base, payload, timeout=600):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/generate", json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dt, r.status_code, r.text[:120]
    return dt, 200, None


def run_cell(base, text, ids, mode, batch, gen=1):
    """mode='text' 提交字符串，mode='ids' 提交 input_ids（跳过前端分词）。

    两种模式承载完全相同的 token 内容：ids 由即将提交的那段文本在本地编码而来。
    """
    if mode == "text":
        payloads = [{"text": text, "sampling_params": {"max_new_tokens": gen, "temperature": 0.0}}] * batch
    else:
        payloads = [{"input_ids": ids, "sampling_params": {"max_new_tokens": gen, "temperature": 0.0}}] * batch
    t0 = time.perf_counter()
    if batch == 1:
        dts = [one_request(base, payloads[0])[0]]
    else:
        with ThreadPoolExecutor(max_workers=batch) as ex:
            dts = [f.result()[0] for f in [ex.submit(one_request, base, p) for p in payloads]]
    wall = time.perf_counter() - t0
    return dict(mode=mode, batch=batch, wall_s=round(wall, 4),
                median_ms=round(statistics.median(dts) * 1000, 3),
                max_ms=round(max(dts) * 1000, 3),
                per_req_ms=[round(d * 1000, 3) for d in dts])


_TOK = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8145")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    rows = []
    for L in args.lengths:
        text = "Token " * L
        ids = local_ids(text)
        for B in args.batches:
            for mode in ("text", "ids"):
                got = []
                for rep in range(args.repeats):
                    r = run_cell(args.base, text, ids, mode, B)
                    r.update(rep=rep, target_len=L, actual_tokens=len(ids))
                    got.append(r)
                med = statistics.median(g["median_ms"] for g in got)
                row = dict(target_len=L, actual_tokens=len(ids), batch=B, mode=mode,
                           median_of_median_ms=round(med, 3),
                           wall_median_s=round(statistics.median(g["wall_s"] for g in got), 4),
                           reps=got)
                rows.append(row)
                print(f"  L={L:>5} B={B:>2} {mode:<4} 每请求中位 {med:>9.3f} ms  "
                      f"批墙钟 {row['wall_median_s']:>7.3f} s", flush=True)

    pairs = []
    for L in args.lengths:
        for B in args.batches:
            t = next(r for r in rows if r["target_len"] == L and r["batch"] == B and r["mode"] == "text")
            i = next(r for r in rows if r["target_len"] == L and r["batch"] == B and r["mode"] == "ids")
            pairs.append(dict(target_len=L, batch=B,
                              text_ms=t["median_of_median_ms"],
                              ids_ms=i["median_of_median_ms"],
                              frontend_ms=round(t["median_of_median_ms"] - i["median_of_median_ms"], 3)))
    (args.out / "sglang_frontend.json").write_text(
        json.dumps(dict(model=args.model, rows=rows, pairs=pairs,
                        method="同一批 token 序列分别以 text 与 input_ids 提交；"
                               "同一服务同一长度下两者之差即前端分词/反分词成本"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n前端成本（text 中位 − ids 中位，单位 ms）：")
    print(f"  {'长度':>6}" + "".join(f"{'B=' + str(b):>10}" for b in args.batches))
    for L in args.lengths:
        row = f"  {L:>6}"
        for B in args.batches:
            p = next(x for x in pairs if x["target_len"] == L and x["batch"] == B)
            row += f"{p['frontend_ms']:>10.3f}"
        print(row)


if __name__ == "__main__":
    main()
