#!/usr/bin/env python3
"""L5.9 —— 采样契约的端到端对照：同一批 token 输入喂两个引擎。

既有对照都在库层（`apply_top_k_top_p` vs SGLang 的 torch 实现）。
这一层往上走一步：**同一份 input_ids** 分别提交 vLLM `/v1/completions`
与 SGLang `/generate`，比三件事：

  1. 贪心输出是否逐 token 相同（同模型、同输入、同确定性路径）；
  2. 逐 token logprob 是否在容差内一致（这比 token 相同更强）；
  3. 采样（temperature/top_k/top_p + 固定 seed）在两端各自的**有效率**与重复性。

beam search 只在 SGLang 侧测：`beam_width` 是它的原生参数，
而 vLLM 0.29.0 的 v1 已经没有 beam search（源码里 `use_beam_search` 不存在）。

用法：
  python sampling_engine_e2e.py --engine vllm   --base http://127.0.0.1:8146 --out DIR
  python sampling_engine_e2e.py --engine sglang --base http://127.0.0.1:8147 --out DIR
  python sampling_engine_e2e.py --compare DIR_VLLM DIR_SGLANG
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import requests

PROMPTS = [
    "The prefix cache mechanism lets multiple requests",
    "A paged KV cache stores key and value tensors",
    "Prefix caching reuses computed blocks for matching",
]
MAX_TOKENS = 32


def post(base, path, payload, timeout=300):
    t0 = time.perf_counter()
    r = requests.post(base + path, json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dict(status=r.status_code, error=r.text[:200], elapsed_ms=dt * 1000)
    return dict(status=200, elapsed_ms=dt * 1000, body=r.json())


def vllm_case(base, ids, kind):
    if kind == "greedy":
        sp = {"temperature": 0.0, "max_tokens": MAX_TOKENS, "logprobs": 1,
              # vLLM 默认不回 token id，必须显式要
              "return_token_ids": True}
    elif kind == "sampled":
        sp = {"temperature": 0.8, "top_k": 50, "top_p": 0.9,
              "max_tokens": MAX_TOKENS, "seed": 1234,
              "return_token_ids": True}
    else:
        return None
    r = post(base, "/v1/completions",
             {"model": "m", "prompt": ids, "max_tokens": MAX_TOKENS, **sp})
    if r.get("status") != 200:
        return r
    ch = r["body"]["choices"][0]
    lp = ch.get("logprobs") or {}
    return dict(status=200, elapsed_ms=r["elapsed_ms"],
                token_ids=ch.get("token_ids"),
                text=ch.get("text"), finish_reason=ch.get("finish_reason"),
                logprobs=lp.get("token_logprobs"),
                prompt_tokens=r["body"]["usage"]["prompt_tokens"],
                completion_tokens=r["body"]["usage"]["completion_tokens"])


def sglang_case(base, ids, kind):
    if kind == "greedy":
        sp = {"temperature": 0.0, "max_new_tokens": MAX_TOKENS}
        extra = {"return_logprob": True, "logprob_start_len": 0}
    elif kind == "sampled":
        # SGLang 0.5.19 的参数名是 sampling_seed；用 seed 会 500
        # （TypeError: Unexpected keyword argument 'seed'）
        sp = {"temperature": 0.8, "top_k": 50, "top_p": 0.9,
              "max_new_tokens": MAX_TOKENS, "sampling_seed": 1234}
        # 要 token id 就得打开 logprob（SGLang 的 output_ids 不总在 meta_info 里）
        extra = {"return_logprob": True, "logprob_start_len": 0}
    elif kind == "beam":
        sp = {"temperature": 0.0, "max_new_tokens": MAX_TOKENS,
              "beam_width": 4, "n": 4}
        extra = {}
    else:
        return None
    r = post(base, "/generate", {"input_ids": ids, "sampling_params": sp, **extra})
    if r.get("status") != 200:
        return r
    b = r["body"]
    meta = b.get("meta_info", {})
    out = dict(status=200, elapsed_ms=r["elapsed_ms"],
               token_ids=meta.get("output_ids"),
               finish_reason=meta.get("finish_reason"),
               prompt_tokens=meta.get("prompt_tokens"),
               completion_tokens=meta.get("completion_tokens"))
    # 只取**生成**位置的 logprob：vLLM 的 token_logprobs 也只覆盖生成的位置，
    # 把 prompt 的 input_token_logprobs 拼进来会整体错位（实测能差到 10 以上）。
    if meta.get("output_token_logprobs"):
        out["logprobs"] = [x[0] for x in meta["output_token_logprobs"]]
        if not out.get("token_ids"):
            out["token_ids"] = [x[1] for x in meta["output_token_logprobs"]]
    if kind == "beam":
        out["beam_outputs"] = [
            dict(token_ids=c.get("token_ids"), finish_reason=c.get("finish_reason"))
            for c in (b.get("outputs") or [])]
    return out


def run_engine(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = ["greedy", "sampled"] + (["beam"] if args.engine == "sglang" else [])
    results = {}
    for i, text in enumerate(PROMPTS):
        ids = tok.encode(text, add_special_tokens=False)
        for kind in cases:
            fn = vllm_case if args.engine == "vllm" else sglang_case
            r = fn(args.base, ids, kind)
            results[f"{i}:{kind}"] = r
            r = r or {}
            print(f"  [{args.engine}] prompt{i} {kind:<8} "
                  f"{r.get('completion_tokens')} token  "
                  f"{r.get('finish_reason')}  {r.get('elapsed_ms', 0):.0f} ms"
                  f"{'  ERR ' + str(r.get('error')) if r.get('error') else ''}",
                  flush=True)
    # 采样重复性：同一 seed 跑三次，看输出是否逐 token 相同
    ids = tok.encode(PROMPTS[0], add_special_tokens=False)
    reps = [sglang_case(args.base, ids, "sampled") if args.engine == "sglang"
            else vllm_case(args.base, ids, "sampled") for _ in range(3)]
    ok_reps = [r for r in reps if r.get("token_ids")]
    same = (len(ok_reps) == len(reps)
            and len({tuple(r["token_ids"]) for r in ok_reps}) == 1)
    (args.out / "e2e.json").write_text(
        json.dumps(dict(engine=args.engine, model=args.model, results=results,
                        sampled_repeat_identical=same,
                        sampled_repeat_ok=len(ok_reps),
                        sampled_repeat_errors=[r.get("error") for r in reps
                                               if not r.get("token_ids")],
                        sampled_repeat_token_ids=[r.get("token_ids") for r in reps],
                        note="输入是本地 tokenizer 编出的 input_ids，"
                             "两端都不再走自己的分词，比较的是引擎本身"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  采样同 seed 三次逐 token 相同：{same}")


def compare(dir_v, dir_s):
    a = json.loads((dir_v / "e2e.json").read_text())
    b = json.loads((dir_s / "e2e.json").read_text())
    print("键                    vLLM token / SGLang token   贪心相同  logprob 最大差")
    for i in range(len(PROMPTS)):
        for kind in ("greedy", "sampled"):
            k = f"{i}:{kind}"
            ra, rb = a["results"].get(k, {}), b["results"].get(k, {})
            ta, tb = ra.get("token_ids") or [], rb.get("token_ids") or []
            same = bool(ta) and ta == tb
            first_diff = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            diff = None
            diff_before = None
            la, lb = ra.get("logprobs"), rb.get("logprobs")
            if la and lb:
                # 两端的对齐方式不同：vLLM 对 prompt id 输入会在首位给 null，
                # SGLang 的 input_token_logprobs 末位是占位。只比双方都有的位置。
                la2 = [x for x in la if x is not None]
                lb2 = [x for x in lb if x is not None]
                n = min(len(la2), len(lb2))
                diff = max(abs(x - y) for x, y in zip(la2[:n], lb2[:n])) if n else None
                # 首个不同 token 之前的最大 logprob 差：区分"同轨迹只是数值差"
                # 与"从某一步起走岔了"
                cut = first_diff if first_diff else n
                diff_before = (max(abs(x - y) for x, y in zip(la2[:cut], lb2[:cut]))
                               if cut else None)
            print(f"  {k:<18} {len(ta):>3} / {len(tb):>3} token  相同 {str(same):<5} "
                  f"首个不同位置 {str(first_diff):<5} 全长最大差 "
                  f"{('%.3e' % diff) if diff is not None else 'n/a':<10} 分岔前最大差 "
                  f"{('%.3e' % diff_before) if diff_before is not None else 'n/a'}")
    print(f"\nSGLang beam 输出（vLLM 0.29.0 无 beam search）：")
    beams = b["results"].get("0:beam", {}).get("beam_outputs") or []
    for j, c in enumerate(beams):
        print(f"  beam {j}: {len(c.get('token_ids') or [])} token  {c.get('finish_reason')}")
    print(f"\n采样同 seed 重复性：vLLM {a['sampled_repeat_identical']}，"
          f"SGLang {b['sampled_repeat_identical']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"])
    ap.add_argument("--base")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--compare", nargs=2, type=pathlib.Path)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        assert args.engine and args.base and args.out
        run_engine(args)


if __name__ == "__main__":
    main()
