#!/usr/bin/env python3
"""L5.12 补测 · rerank 的等价链：把判定搬到引擎的对数概率路径上。

上一轮的结论是"不等价"，原因已经定位清楚：Qwen3-Reranker-0.6B 的 config 声明
`architectures=["Qwen3ForCausalLM"]`，官方打分口径是**末位置 lm_head 的 yes/no 两行**
做 softmax；而引擎侧用 `--runner pooling --convert embed` 加载时走的是
`pooler_for_embed`（LAST-token 池化 + 激活），量的是隐状态分数，不是 LM head。
两个量不是同一个东西，所以 `UNVERIFIED`。

"换加载方式"就是把引擎当成**生成模型**加载，用 `/v1/completions` 取
第一个生成位置的 `logprobs`，再算两标签 softmax：

    sigmoid(lp(yes) − lp(no)) = softmax([logit_no, logit_yes])[1]

词表归一化在两者相减时抵消，所以这个量与官方口径**代数等价**。
本脚本负责引擎侧；HF 参照由 `rerank_serving_audit.py reference` 产出，两者按
逐文档分数、排序与 Kendall τ 对照。

用法（先起好生成式 vLLM，见 run_rerank_logprob.sh）：
    python rerank_logprob_chain.py --base http://127.0.0.1:8152 \
        --hf-reference <dir>/reference.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-Reranker-0.6B"
QUERY = "Which mechanism lets multiple requests reuse an identical prompt prefix?"
DOCS = [
    "Prefix caching reuses computed key and value blocks for matching prompt prefixes.",
    "Temperature scales logits before sampling the next token.",
    "A paged cache stores key and value tensors in fixed-size blocks.",
    "Gradient accumulation combines gradients before updating model parameters.",
    "A prefix cache lookup depends on the preceding token sequence and cache identity.",
    "The restaurant serves soup at noon.",
]


def post(base, endpoint, payload, timeout=180):
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=raw,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        return e.code, {"error": body}, (time.perf_counter() - t0) * 1000


def rank_order(v):
    return sorted(range(len(v)), key=lambda i: -v[i])


def kendall_tau(a, b):
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8152")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--hf-reference", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tok.apply_chat_template([{"role": "query", "content": QUERY},
                                        {"role": "document", "content": d}],
                                       tokenize=False) for d in DOCS]
    label_ids = {t: tok.convert_tokens_to_ids(t) for t in ("no", "yes")}
    id2label = {v: k for k, v in label_ids.items()}

    rows, scores = [], []
    for i, p in enumerate(prompts):
        status, body, ms = post(args.base, "/v1/completions", {
            "model": args.model,
            "prompt": p,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": args.topn,
        })
        entry = dict(doc=i, status=status, client_ms=round(ms, 1))
        if status == 200:
            lp = (body["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}]
            top = lp[0] if lp else {}
            # vLLM 0.29 的 completions 里 top_logprobs 是 {token 文本: logprob}。
            # 不能按 "yes" 这样的文本匹配：词表里 'yes'/'Yes'/'YES' 是**不同的 token**，
            # 匹配错了会读到完全不同的量。这里把文本映射回 id，只认参考用的那两个 id。
            got = {}
            for k, v in top.items():
                tid = tok.convert_tokens_to_ids(str(k))
                lab = id2label.get(tid)
                if lab is not None:
                    got[lab] = v
            entry.update(found=sorted(got), logprob_no=got.get("no"),
                         logprob_yes=got.get("yes"),
                         n_candidates=len(top),
                         top_keys=[str(k).strip() for k in list(top)[:6]])
            if "no" in got and "yes" in got:
                s = 1.0 / (1.0 + math.exp(-(got["yes"] - got["no"])))
                entry["prob_yes_two_label"] = s
                scores.append(s)
            else:
                entry["prob_yes_two_label"] = None
                scores.append(float("nan"))
        else:
            entry["error"] = body
            scores.append(float("nan"))
        rows.append(entry)
        print(f"  doc{i}: HTTP {status}  找到标签 {entry.get('found')}  "
              f"P(yes) {entry.get('prob_yes_two_label')}", flush=True)

    report = dict(model=args.model, base=args.base, query=QUERY, documents=DOCS,
                  label_token_ids=label_ids, topn=args.topn,
                  engine_prob_yes_two_label=scores, rows=rows)

    if args.hf_reference:
        with open(args.hf_reference) as f:
            ref = json.load(f)
        ref_p = ref["probabilities"]
        ok = [i for i, s in enumerate(scores) if s == s]
        diffs = [abs(scores[i] - ref_p[i]) for i in ok]
        report["comparison"] = dict(
            reference=args.hf_reference,
            hf_probabilities=ref_p,
            n_compared=len(ok),
            max_abs_diff=max(diffs) if diffs else None,
            engine_order=rank_order([s if s == s else -9 for s in scores]),
            hf_order=rank_order(ref_p),
            kendall_tau=kendall_tau(
                [s if s == s else -9 for s in scores], ref_p),
            top1_same=(rank_order([s if s == s else -9 for s in scores])[0]
                       == rank_order(ref_p)[0]))
        print(f"\n对照：n={len(ok)}  max|Δ| {report['comparison']['max_abs_diff']}  "
              f"τ {report['comparison']['kendall_tau']}  "
              f"top1 相同 {report['comparison']['top1_same']}")

    with open(os.path.join(args.out, "rerank_logprob_chain.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {args.out}/rerank_logprob_chain.json")


if __name__ == "__main__":
    main()
