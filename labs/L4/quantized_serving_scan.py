#!/usr/bin/env python3
"""L4.3 修订（任务 E 的剩余部分）—— 同一协议下的精度服务扫描。

三个 checkpoint 走同一套协议（同一引擎、同一提示、同一采样参数）：
  bf16 / AWQ / GPTQ-Int4 的 Qwen2.5-1.5B-Instruct

每个模型测：
  [E-a] 首 token 延迟（max_tokens=1）与 32 token 解码吞吐，S=1024、B=1/8
  [E-b] 引擎自报的 KV 容量与权重显存
  [E-c] 与 bf16 的首 token 一致性（贪心首 token、top-1 logprob 差）

每个模型必须单独一个进程（vLLM 在进程内不归还显存）。

用法：
    python labs/L4/quantized_serving_scan.py --model bf16 --outdir out/4.3/scan
"""

import argparse
import glob
import json
import os
import sys
import time

MODELS = {
    "bf16": "Qwen/Qwen2.5-1.5B-Instruct",
    "awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "gptq": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
}
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(repo)
    return got[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/l43scan"))
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--s-len", type=int, default=1024)
    ap.add_argument("--batches", default="1,8")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    repo = MODELS[args.model]
    print("=" * 78 + f"\n[E] 精度服务扫描：{args.model}（{repo}）\n" + "=" * 78, flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(snap(repo))
    rep = {"model": args.model, "repo": repo,
           "revision": os.path.basename(snap(repo))}
    t0 = time.perf_counter()
    llm = LLM(model=snap(repo), gpu_memory_utilization=args.util,
              max_model_len=4096, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)
    rep["init_s"] = time.perf_counter() - t0
    text = ("The history of computing spans many decades. " * 200)
    ids = tok(text)["input_ids"][:args.s_len]
    # 预热：排除首次走量化 kernel 的编译/装载成本（否则第一格会被严重污染）
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=8, temperature=0.0))
    rep["warmup_s"] = time.perf_counter() - t0
    print(f"  预热一次（8 token）：{rep['warmup_s']:.2f} s")
    rows = []
    for B in [int(x) for x in args.batches.split(",") if x]:
        prompts = [{"prompt_token_ids": ids}] * B
        # 首 token 延迟
        t0 = time.perf_counter()
        r1 = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0,
                                                  logprobs=5))
        ttft = time.perf_counter() - t0
        lp = r1[0].outputs[0].logprobs[0]
        top = max(lp.items(), key=lambda kv: kv[1].logprob)
        # 32 token 解码
        t0 = time.perf_counter()
        r2 = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0))
        dt = time.perf_counter() - t0
        toks = r2[0].outputs[0].token_ids
        rows.append({"batch": B, "s_len": args.s_len, "ttft_s": ttft,
                     "first_top1": top[0], "first_logprob": top[1].logprob,
                     "decode32_s": dt, "decode_tok_s_total": B * len(toks) / dt,
                     "decode_tok_s_per_req": len(toks) / dt,
                     "token_ids": toks})
        print(f"  B={B} S={args.s_len}：首 token {ttft*1e3:8.1f} ms  "
              f"top-1 {top[0]}（logprob {top[1].logprob:+.4f}）  "
              f"32 token 解码 {dt*1e3:8.1f} ms（总 {B*len(toks)/dt:6.1f} tok/s，"
              f"每请求 {len(toks)/dt:5.1f}）")
    rep["rows"] = rows
    print("  （显存与 KV 预算从引擎日志解析，见分析脚本）")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    path = os.path.join(args.outdir, "quantized_serving_scan.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev[args.model] = rep
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
