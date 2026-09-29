#!/usr/bin/env python3
"""L5.10 补测 · SGLang 侧 top-k logprob 到底放在哪、怎么取。

两引擎的 logprob 返回形状完全不同，跨引擎对照必须先读对字段：

  * vLLM（OpenAI 兼容）：`choices[0].logprobs.top_logprobs` 是
    **`[{token 文本: logprob}, …]`**，逐位置一个 dict；分词文本里可能带前导空格，
    而 `yes`/`Yes`/`YES` 是不同 token，按文本匹配会读错（5.12 的 rerank 对照踩过）。
  * SGLang（`/generate`）：`return_logprob=True` 与 `top_logprobs_num=k` 是
    **`GenerateReqInput` 的顶层字段**（`srt/managers/io_struct.py:226/231`），
    放进 `sampling_params` 会被拒；返回在 `meta_info` 下，键名形如
    `output_token_logprobs` / `input_token_logprobs`，每个元素是
    `[logprob, token_id, [top-k 列表]]` 的**嵌套数组**。

本脚本对着真实 SGLang 发一次请求，把 `meta_info` 的**完整键树**与逐位置结构打印出来，
并核对三件事：位置数是否等于输出 token 数、top-k 的 k 是否等于请求值、
top-1 是否等于该位置实际采样到的 token。vLLM 侧的键名与文本匹配坑在 5.12 已记录，
这里只给出 SGLang 侧的取法与该引擎的字段清单。

用法（先起好 SGLang，见 run_sglang_logprob.sh）：
    python sglang_topk_logprob.py --base http://127.0.0.1:8195 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

PROMPT = "用一句话说明前缀缓存的作用。"


def key_tree(o, prefix="", depth=0, out=None):
    out = [] if out is None else out
    if depth > 2:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            n = f" len={len(v)}" if isinstance(v, (list, str, dict)) else ""
            out.append(f"{'  ' * depth}{prefix}{k}: {t}{n}")
            key_tree(v, "", depth + 1, out)
    elif isinstance(o, list) and o:
        key_tree(o[0], "[0].", depth + 1, out)
    return out


def post(base, endpoint, payload, timeout=300):
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=raw,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        return e.code, {"error": body}, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8195")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # return_logprob / top_logprobs_num 是 GenerateReqInput 的**顶层字段**
    # （io_struct.py:226/231），放进 sampling_params 会被服务端拒绝（HTTP 500）。
    st, body, ms = post(args.base, "/generate", {
        "text": PROMPT,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": args.max_new},
        "return_logprob": True,
        "top_logprobs_num": args.topk,
    })
    report = dict(base=args.base, status=st, client_ms=round(ms, 1),
                  prompt=PROMPT, requested_topk=args.topk,
                  max_new_tokens=args.max_new)
    if st != 200:
        report["error"] = body
        print(f"HTTP {st}: {body}")
    else:
        meta = body.get("meta_info", {})
        report["meta_keys"] = sorted(meta)
        report["key_tree"] = key_tree(meta)
        out_lp = meta.get("output_token_logprobs")
        out_topk = meta.get("output_top_logprobs")
        report["has_output_token_logprobs"] = out_lp is not None
        report["has_output_top_logprobs"] = out_topk is not None
        if out_lp:
            first = out_lp[0]
            report["first_element_shape"] = (
                f"len={len(first)}: [logprob={first[0]!r}, token_id={first[1]!r}, "
                f"third={first[2] if len(first) > 2 else None!r}]")
            # 逐 beam/top-k 列表在**另一个键** output_top_logprobs 里，
            # 每个位置一个列表，元素是 [logprob, token_id]。
            report["topk_sizes_seen"] = sorted(
                {len(x) for x in (out_topk or []) if x})
            top1_match = 0
            for pos, elem in enumerate(out_lp):
                tk = (out_topk or [None] * len(out_lp))[pos] if out_topk else None
                if tk and tk[0][1] == elem[1]:
                    top1_match += 1
            report["n_positions"] = len(out_lp)
            report["n_output_tokens"] = len(body.get("output_ids") or [])
            report["top1_equals_sampled"] = top1_match
            report["first_three_positions"] = [
                dict(logprob=e[0], token_id=e[1],
                     topk=[[t[0], t[1]] for t in
                           ((out_topk or [])[pos] if out_topk else [])][:3])
                for pos, e in enumerate(out_lp[:3])]
            report["top_logprobs_len"] = (len(out_topk) if out_topk else None)
        report["meta_info_sample"] = {
            k: meta[k] for k in ("prompt_tokens", "completion_tokens",
                                 "finish_reason") if k in meta}
        with open(os.path.join(args.out, "sglang_topk_logprob.json"), "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("meta_info 键：" + ", ".join(report["meta_keys"]))
        print(f"  output_token_logprobs: {report.get('n_positions')} 个位置，"
              f"输出 token {report.get('n_output_tokens')} 个；"
              f"top-k 尺寸 {report.get('topk_sizes_seen')}；"
              f"top-1 与实际 token 相同的位置 {report.get('top1_equals_sampled')}")
        print(f"  首元素：{report.get('first_element_shape')}")
        for row in report.get("first_three_positions", []):
            ids = [t[1] for t in row["topk"]]
            lp = row["logprob"]
            print(f"    位置 token={row['token_id']}  logprob="
                  f"{lp if lp is None else round(lp, 4)}  topk_ids={ids}")
    with open(os.path.join(args.out, "sglang_topk_logprob.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {args.out}/sglang_topk_logprob.json")


if __name__ == "__main__":
    main()
