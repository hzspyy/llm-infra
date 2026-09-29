#!/usr/bin/env python3
"""L5.10 补测 · SGLang 的 unload 到底是"移除"还是"驱逐"。

源码先把事实摆清楚（`sglang 0.5.19`）：

  * `/unload_lora_adapter`（`srt/entrypoints/http_server.py:1574`）要求请求体带
    `lora_name`（`srt/managers/tokenizer_control_mixin.py:768-771` 断言），
    最终由 `LoRAManager._unload_lora_adapter`（`srt/lora/lora_manager.py:332`）
    从内存池与 `configs/loras/lora_refs` 里按 `lora_id` 删除。
  * 但每个带 `lora_path` 的请求都会先走 `_resolve_lora_path`
    （`srt/managers/tokenizer_manager.py:3344`）：它向注册表问"哪些路径已经不在注册表里"，
    然后**隐式重新加载**它们（日志 `Reloading evicted adapter`，`:3364-3378`）。

所以卸载是**驱逐**，不是"持久移除"：只要之后再有请求点到同一个 `lora_path`，
适配器会被自动装回来。这个脚本用四个动作把这条语义跑出来：

  A 装载 → 请求（拿到输出 A）
  B 卸载 → 看返回
  C 再请求同一路径 → 是否报错？输出是否与 A 相同？服务端日志里有没有 Reloading
  D 请求一个从未装载过的路径 → 报错原文
  E 卸载一个不存在/已卸载的名字 → 报错原文

用法（先起好 SGLang，见 run_sglang_unload.sh）：
    python sglang_lora_unload_semantics.py --base http://127.0.0.1:8190 \
        --lora-path <adapter dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-4B"
PROMPT = "用一句话说明前缀缓存的作用。"


def post(base, endpoint, payload, timeout=300):
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=raw,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except Exception:                                       # noqa: BLE001
            body = {"raw": body[:400]}
        return e.code, body, (time.perf_counter() - t0) * 1000


def generate(base, model, lora_path=None, max_tokens=16):
    sp = {"temperature": 0.0, "max_new_tokens": max_tokens}
    payload = {"text": PROMPT, "sampling_params": sp}
    if lora_path:
        payload["lora_path"] = lora_path
    return post(base, "/generate", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8190")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora-name", default="pub")
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    steps = []
    base_out = generate(args.base, args.model)
    steps.append(dict(step="base(无 adapter)", status=base_out[0],
                      ms=round(base_out[2], 1),
                      text=(base_out[1].get("text") if base_out[0] == 200 else base_out[1])))

    st, body, ms = post(args.base, "/load_lora_adapter",
                        {"lora_name": args.lora_name, "lora_path": args.lora_path})
    steps.append(dict(step="A 装载", status=st, ms=round(ms, 1), body=body))

    st, body, ms = generate(args.base, args.model, args.lora_name)
    a_text = body.get("text") if st == 200 else None
    a_ids = body.get("output_ids") if st == 200 else None
    steps.append(dict(step="A 用 adapter 生成", status=st, ms=round(ms, 1),
                      text=a_text, output_ids=a_ids))

    st, body, ms = post(args.base, "/unload_lora_adapter",
                        {"lora_name": args.lora_name})
    steps.append(dict(step="B 卸载", status=st, ms=round(ms, 1), body=body))

    st, body, ms = generate(args.base, args.model, args.lora_name)
    c_ids = body.get("output_ids") if st == 200 else None
    steps.append(dict(step="C 再请求同一路径", status=st, ms=round(ms, 1),
                      text=body.get("text") if st == 200 else body,
                      output_ids=c_ids,
                      same_as_A=(c_ids == a_ids if c_ids and a_ids else None),
                      error=body.get("error") if st != 200 else None))

    st, body, ms = generate(args.base, args.model, "/tmp/never-loaded-adapter")
    steps.append(dict(step="D 从未装载的路径", status=st, ms=round(ms, 1),
                      body=body))

    st, body, ms = post(args.base, "/unload_lora_adapter",
                        {"lora_name": "no-such-name"})
    steps.append(dict(step="E 卸载不存在的名字", status=st, ms=round(ms, 1),
                      body=body))

    report = dict(model=args.model, lora_name=args.lora_name,
                  lora_path=args.lora_path, prompt=PROMPT, steps=steps)
    with open(os.path.join(args.out, "lora_unload_semantics.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for s in steps:
        print(f"  {s['step']:<18} HTTP {s['status']}  "
              f"{json.dumps({k: v for k, v in s.items() if k not in ('step','status','ms')}, ensure_ascii=False)[:160]}")
    print(f"\n写入 {args.out}/lora_unload_semantics.json")


if __name__ == "__main__":
    main()
