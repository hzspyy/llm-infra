#!/usr/bin/env python3
"""L1.6 lab · GGUF 与 HF 参照对拍：转换是否正确，用同一段文本贪心解码验证。

转换脚本把 bf16 权重写成 f16 GGUF，中间经过一次精度变化与一次张量布局重排。
「模型能跑出通顺的话」不能证明转换正确，必须给出**可判定的等价性证据**：

  1. 词表与模板：同一段文本在 HF tokenizer 与 GGUF 词表下的 token 数一致
  2. 贪心解码：temperature=0、固定 32 个新 token，逐字符比较续写文本
  3. 首次分歧位置：不一致时保存两边的原文（而不是只报「不一样」）

用法（crater 上比对 F16 GGUF 与 HF 原始权重）：
    python gguf_hf_crosscheck.py \
        --model-dir <Qwen3-1.7B snapshot> \
        --gguf Qwen3-1.7B-F16.gguf \
        --server-bin llama.cpp/build/bin/llama-server
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_runtime_bench import Server  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "1, 1, 2, 3, 5, 8, 13,",
    "def add(a, b):\n    return",
    "在推理系统里，KV cache 的作用是",
]


class HFRef:
    """HF 参照：权重只加载一次，四条 prompt 复用同一个模型。"""

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                              dtype=torch.float32)
        except TypeError:                      # 旧版 transformers 用 torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.float32)
        self.model.eval()

    def greedy(self, prompt: str, n: int) -> dict:
        torch = self.torch
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=n, do_sample=False,
                                      num_beams=1, use_cache=True,
                                      pad_token_id=self.tok.eos_token_id)
        new_ids = out[0][ids.shape[1]:].tolist()
        return {"prompt_tokens": int(ids.shape[1]), "new_ids": new_ids,
                "text": self.tok.decode(new_ids, skip_special_tokens=True),
                "model_dtype": str(self.model.dtype)}


def gguf_greedy(srv: Server, prompt: str, n: int) -> dict:
    """非流式取 content：流式拼接会引入边界差异，比对要用一次完整返回。"""
    payload = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0,
                          "stream": False, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.loads(r.read().decode())
    return {"prompt_tokens": body.get("timings", {}).get("prompt_n"),
            "text": body.get("content", ""), "timings": body.get("timings", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server-bin", default="llama-server")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--log", default="/tmp/llama_xcheck.log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {"measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "model_dir": args.model_dir, "gguf": args.gguf,
           "n_new_tokens": args.n, "cases": []}
    print(f"=== GGUF ↔ HF 贪心对拍（{args.n} 个新 token，temperature=0）")
    hf = HFRef(args.model_dir)
    print(f"    HF 参照 dtype={hf.model.dtype}")
    with Server(args.server_bin, args.gguf, args.ctx, 1, args.ngl, 4,
                log=Path(args.log)) as srv:
        for prompt in PROMPTS:
            h = hf.greedy(prompt, args.n)
            g = gguf_greedy(srv, prompt, args.n)
            i = 0
            while i < min(len(h["text"]), len(g["text"])) and h["text"][i] == g["text"][i]:
                i += 1
            same = h["text"] == g["text"]
            res["cases"].append({
                "prompt": prompt, "hf_text": h["text"], "gguf_text": g["text"],
                "hf_new_ids": h["new_ids"],
                "prompt_tokens_hf": h["prompt_tokens"],
                "prompt_tokens_gguf": g["prompt_tokens"],
                "prompt_tokens_match": h["prompt_tokens"] == g["prompt_tokens"],
                "text_match": same, "first_diff_char": i,
                "gguf_timings": g["timings"]})
            print(f"    {prompt[:28]!r:<32} tokens "
                  f"{h['prompt_tokens']}/{g['prompt_tokens']}  "
                  f"{'一致' if same else f'分歧@{i}'}")
            if not same:
                print(f"        HF  : {h['text']!r}")
                print(f"        GGUF: {g['text']!r}")

    n_same = sum(1 for c in res["cases"] if c["text_match"])
    n_tok = sum(1 for c in res["cases"] if c["prompt_tokens_match"])
    res["summary"] = {"cases": len(res["cases"]), "text_match": n_same,
                      "prompt_token_match": n_tok}
    print(f"\n    文本一致 {n_same}/{len(res['cases'])}，"
          f"prompt token 数一致 {n_tok}/{len(res['cases'])}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
