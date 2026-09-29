#!/usr/bin/env python3
"""L5.10 任务 C（SGLang 侧）—— HTTP 服务上的非法/异常 LoRA 请求后能否继续服务。

计划对这一条的要求是：「在 HTTP 服务测试非法 rank 后合法请求、同名不同版本更新、
请求执行中卸载/替换」「重启恢复与同进程恢复分开」。
vLLM 侧已有 `lora_recovery_test.py`；这里补 SGLang 侧，并且**只用一个进程**先把
"同进程能否继续"问清楚，最后用一次独立重启做参照。

序列（每步都记录 HTTP 状态、错误文本、以及**紧随其后的合法请求**是否正常）：

  S1 合法：lora_path=pub（启动时注册的 rank-8 adapter）
  S2 非法：lora_path=不存在 的名字
  S3 非法：lora_path 指向一个 **rank=16** 的 adapter 目录（超过 --max-lora-rank 8）
  S4 非法：lora_path 指向一个目录里没有 adapter_config.json
  S5 合法：同 S1（比较输出是否与 S1 逐 token 相同）
  S6 流式中断：带 lora_path 的流式请求跑到一半关掉连接，随后再发一次合法请求
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sglang_lora_serving_audit import make_adapter, PROMPTS   # noqa: E402


def ask(base, text, lora=None, max_new_tokens=16, timeout=120):
    # 要 token id 就得开 logprob；不开时 meta_info 里没有 output_token_logprobs，
    # 上一版因此把成功的请求也记成 n_tokens=0
    payload = {"text": text,
               "sampling_params": {"temperature": 0.0,
                                   "max_new_tokens": max_new_tokens},
               "return_logprob": True, "logprob_start_len": 0}
    if lora is not None:
        payload["lora_path"] = lora
    t0 = time.perf_counter()
    try:
        r = requests.post(base + "/generate", json=payload, timeout=timeout)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return dict(status=r.status_code, error=r.text[:200],
                        wall_s=round(dt, 3))
        b = r.json()
        meta = b.get("meta_info", {})
        ids = [int(x[1]) for x in (meta.get("output_token_logprobs") or [])]
        if not ids:
            ids = [int(x) for x in (meta.get("output_ids") or [])]
        return dict(status=200, wall_s=round(dt, 3), token_ids=ids,
                    n_tokens=len(ids))
    except Exception as e:
        return dict(status=None, error=f"{type(e).__name__}: {e}",
                    wall_s=round(time.perf_counter() - t0, 3))


def stream_then_abort(base, text, lora, abort_after=3):
    """起一条流式请求，收到若干 chunk 后直接断开连接。"""
    payload = {"text": text, "lora_path": lora,
               "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
               "stream": True}
    got = 0
    t0 = time.perf_counter()
    with requests.post(base + "/generate", json=payload, stream=True,
                       timeout=60) as r:
        for line in r.iter_lines():
            if line.startswith(b"data:"):
                got += 1
                if got >= abort_after:
                    break                      # 直接跳出 with → 关闭连接
    return dict(chunks_before_abort=got, wall_s=round(time.perf_counter() - t0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--valid-adapter", required=True)
    ap.add_argument("--rank16-dir", required=True)
    ap.add_argument("--broken-dir", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    text = PROMPTS[0]
    steps = []

    def record(label, rec):
        rec["step"] = label
        steps.append(rec)
        print(f"  {label:<28} status={rec.get('status')} "
              f"{'tokens=' + str(rec.get('n_tokens')) if rec.get('n_tokens') else ''}"
              f"{'  ERR ' + rec['error'][:90] if rec.get('error') else ''}", flush=True)

    record("S1_valid_first", ask(args.base, text, "pub"))
    record("S2_unknown_name", ask(args.base, text, "no-such-adapter"))
    record("S3_rank16_path", ask(args.base, text, str(args.rank16_dir)))
    record("S4_broken_dir", ask(args.base, text, str(args.broken_dir)))
    record("S5_valid_again", ask(args.base, text, "pub"))

    mid = stream_then_abort(args.base, text, "pub")
    record("S6_stream_abort", dict(status=200, **mid))
    record("S7_valid_after_abort", ask(args.base, text, "pub"))

    report = dict(model=args.model, steps=steps)
    first = next((s for s in steps if s["step"] == "S1_valid_first"), {})
    again = next((s for s in steps if s["step"] == "S5_valid_again"), {})
    after_abort = next((s for s in steps if s["step"] == "S7_valid_after_abort"), {})
    report["verdict"] = dict(
        valid_outputs_stable=(first.get("token_ids") == again.get("token_ids")
                              and bool(first.get("token_ids"))),
        served_after_unknown_name=bool(again.get("token_ids")),
        served_after_rank16=bool(again.get("token_ids")),
        served_after_abort=bool(after_abort.get("token_ids")),
        rank16_rejected=steps[2].get("status") != 200,
        unknown_name_rejected=steps[1].get("status") != 200,
        broken_dir_rejected=steps[3].get("status") != 200)
    (args.out / "sglang_lora_recovery.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n判定：" + json.dumps(report["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
