#!/usr/bin/env python3
"""L5.5 任务 C/F · 草稿路线在各自目标模型内的对照（vLLM 0.29.0）。

同一目标模型内比较四种路径：

  * `none`   —— 普通 decode（基线，也是 token 一致性的参照）
  * `ngram`  —— 从上下文里抄，不需要额外权重
  * `dflash` —— `z-lab/Qwen3-4B-DFlash-b16`（块级并行草稿，目标 Qwen3-4B）
  * `eagle3` —— `thoughtworks/Qwen3-8B-Eagle3`（特征条件草稿，目标 Qwen3-8B）

每个 (方法, k) 起一次引擎，内部扫 batch 与任务，避免反复重建引擎。输出：

  * 每轮接受统计：`accepted/drafts + 1` = 平均前进长度，取自 scheduler 的
    `SpecDecodingStats`，不用总时间反推；
  * 目标任务上的输出吞吐与完整时间；
  * 与 `none` 基线的 **token 级一致性**（同 prompt、贪心、ignore_eos）：每条输出前
    8 个 token 与整条输出的 sha1 前缀；
  * GSM8K 任务的最终数字评分（同一判据，用于确认对照没有改变质量）。

不同 target 的数字不能互相排名（草稿与目标都不同）；同一 target 内
`none/ngram/草稿模型` 才是可比的对照。

用法：
    python spec_draft_compare.py --target Qwen/Qwen3-4B \
        --pairs none,ngram,dflash --ks 1,2,4,8 --batches 1,4,16 \
        --task copy --out <dir>
    python spec_draft_compare.py --target Qwen/Qwen3-8B \
        --pairs none,ngram,eagle3 --ks 1,2,4 --batches 1,4 \
        --task gsm8k --questions 32 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch                                                        # noqa: E402

import speculative as S                                             # noqa: E402

MTP_METHOD = {"Qwen/Qwen3.5-4B": "qwen3_5_mtp"}

DRAFT = {
    "dflash": dict(model="z-lab/Qwen3-4B-DFlash-b16", method="dflash",
                   target="Qwen/Qwen3-4B"),
    "eagle3": dict(model="thoughtworks/Qwen3-8B-Eagle3", method="eagle3",
                   target="Qwen/Qwen3-8B"),
}

GSM_TMPL = ("Question: {q}\n\n"
            "Solve it and end your answer with a line of the form 'The answer is N'.\n")
GSM_CACHE: list[tuple[str, str]] = []


def make(target, spec, util, max_model_len, enforce_eager=True):
    from vllm import LLM
    kw = dict(model=target, gpu_memory_utilization=util,
              max_model_len=max_model_len, enforce_eager=enforce_eager,
              enable_prefix_caching=False, disable_log_stats=(spec is None))
    if spec:
        kw["speculative_config"] = spec
    return LLM(**kw)


def gen(llm, prompts, n_out):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)          # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = (time.perf_counter() - t0) * 1000
    toks = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    return dt, toks, texts


def load_gsm8k(n: int) -> list[tuple[str, str]]:
    if GSM_CACHE:
        return GSM_CACHE
    import glob
    hub = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
    snaps = sorted(glob.glob(f"{hub}/datasets--openai--gsm8k/snapshots/*"))
    path = None
    for s in snaps:
        for cand in ("main/test-00000-of-00001.parquet", "main/test.jsonl"):
            p = os.path.join(s, cand)
            if os.path.exists(p):
                path = p
                break
        if path:
            break
    if not path:
        raise SystemExit("没有找到 gsm8k 测试集")
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()[:n]
    else:
        rows = [json.loads(x) for x in open(path)][:n]
    GSM_CACHE.extend((r["question"].strip(),
                      r["answer"].split("####")[-1].strip().replace(",", ""))
                     for r in rows)
    return GSM_CACHE


def build_prompts(args, B, rng_ids):
    if args.task == "gsm8k":
        qs = load_gsm8k(args.questions)
        picked = [qs[i % len(qs)] for i in range(B)]
        return [GSM_TMPL.format(q=q) for q, _ in picked], [g for _, g in picked]
    if args.task == "copy":
        doc = ("The memory bandwidth of a GPU determines how fast weights and KV "
               "cache can be read. Decode is memory bound because it processes one "
               "token at a time. " * 6)
        p = ("Repeat the following text exactly, word for word:\n\n" + doc
             + "\n\nRepeat it now:\n" + doc)
        return [p] * B, None
    return [" ".join(str(x) for x in rng_ids)] * B, None


def score_gsm8k(texts, golds):
    if not golds:
        return None
    ok = 0
    for t, g in zip(texts, golds):
        m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
        if not m:
            continue
        try:
            if abs(float(m[-1]) - float(g)) < 1e-6:
                ok += 1
        except ValueError:
            pass
    return f"{ok}/{len(golds)}"


def digest_of(toks) -> str:
    return hashlib.sha1(" ".join(map(str, toks)).encode()).hexdigest()[:12]


def run_pair(target, method, k, util, max_model_len, batches, args, rng_ids):
    spec = None
    if method == "ngram":
        spec = dict(method="ngram", num_speculative_tokens=k,
                    prompt_lookup_max=k, prompt_lookup_min=1)
    elif method == "mtp":
        # 配套 MTP 权重在目标 checkpoint 内，只给方法名与草稿长度。
        spec = dict(method=MTP_METHOD.get(target, "mtp"),
                    num_speculative_tokens=k)
    elif method in DRAFT:
        d = DRAFT[method]
        if d["target"] != target:
            return None
        spec = dict(model=d["model"], method=d["method"],
                    num_speculative_tokens=k)
    llm = make(target, spec, util, max_model_len)
    if spec is not None:
        S.hook_spec_stats(llm)

    rows = []
    for B in batches:
        prompts, golds = build_prompts(args, B, rng_ids)
        dt, toks, texts = gen(llm, prompts, args.out_len)
        ntok = sum(len(t) for t in toks)
        m = S.spec_metrics(llm) if spec is not None else None
        adv = None
        if m and m.get("drafts"):
            adv = round(m["accepted"] / m["drafts"] + 1.0, 3)
        rows.append(dict(batch=B, ms=round(dt, 1), tokens=ntok,
                         tok_s=round(ntok / (dt / 1000), 1),
                         accept_advance=adv,
                         accepted=(m or {}).get("accepted"),
                         drafts=(m or {}).get("drafts"),
                         first_tokens=toks[0][:8],
                         digest=digest_of(toks[0]),
                         score=score_gsm8k(texts, golds)))
    S.shutdown(llm)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--pairs", default="none,ngram")
    ap.add_argument("--ks", default="4")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--task", default="copy", choices=["copy", "random", "gsm8k"])
    ap.add_argument("--questions", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--util", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    util = args.util or S.safe_util()
    rng_ids = [1000 + (i * 7) % 50000 for i in range(512)]
    out, rep = [], {}
    out.append(f"L5.5 草稿路线对照 · target {args.target} · 任务 {args.task} · "
               f"输出 {args.out_len} token · gpu_memory_utilization {util:.3f}")
    methods = [m for m in args.pairs.split(",") if m]
    ks = [int(x) for x in args.ks.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    for method in methods:
        for k in ([0] if method == "none" else ks):
            tag = method + (f"_k{k}" if k else "")
            t0 = time.time()
            try:
                rows = run_pair(args.target, method, k, util, args.max_model_len,
                                batches, args, rng_ids)
            except Exception as exc:                               # noqa: BLE001
                out.append(f"\n[{tag}] 启动或运行失败：{type(exc).__name__}: {exc}")
                rep[tag] = dict(error=f"{type(exc).__name__}: {exc}")
                continue
            if rows is None:
                out.append(f"\n[{tag}] 跳过：草稿模型与目标模型不匹配")
                continue
            rep[tag] = dict(method=method, k=k, rows=rows,
                            elapsed_s=round(time.time() - t0, 1))
            out.append(f"\n[{tag}] 用时 {rep[tag]['elapsed_s']} s")
            out.append(f"  {'batch':>6}{'完整 ms':>10}{'token':>7}{'tok/s':>9}"
                       f"{'平均前进':>10}{'接受/草稿轮':>14}{'得分':>8}"
                       f"{'输出sha1':>14}  首 token")
            for r in rows:
                acc = (f"{r['accepted']}/{r['drafts']}"
                       if r["accepted"] is not None else "-")
                out.append(f"  {r['batch']:>6}{r['ms']:>10.1f}{r['tokens']:>7}"
                           f"{r['tok_s']:>9.1f}"
                           f"{str(r['accept_advance']):>10}{acc:>14}"
                           f"{str(r['score'] or '-'):>8}{r['digest']:>14}"
                           f"  {r['first_tokens']}")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "spec_draft_compare.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "spec_draft_compare.json"), "w") as f:
        json.dump(dict(args=vars(args), pairs=rep), f, indent=1, default=str)
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
