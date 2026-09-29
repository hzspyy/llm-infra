#!/usr/bin/env python3
"""L5.10 —— 把上一轮对 SGLang unload 的误读纠正回来。

上一轮观察到：`/unload_lora_adapter {"lora_name":"pub"}` 报成功、`loaded_adapters`
里也没有 `pub` 了，但紧接着带 `lora_path=pub` 的请求仍然 200、输出还是 C1 的权重，
并据此写成"卸载未生效/账本与实际不一致"。**这个解读是错的。**

读源码可以看到真正发生了什么：`tokenizer_manager._resolve_lora_path()`
（`srt/managers/tokenizer_manager.py:3344`）在请求进来时先查
`lora_registry.get_unregistered_loras(...)`，对已被卸载的名字会**自动从
`lora_ref_cache` 隐式重载**，并打日志 `Reloading evicted adapter: <name>`；
只有它不在 cache 里才报
`Got LoRA adapter that has never been loaded`。

这个探针用两个只差一步的顺序把它们分开：

  A  unload → **不请求** → 显式 load 同名     ⇒ 若 200，说明注册表真的空了
  B  unload → **请求一次**（触发隐式重载）→ 显式 load 同名 ⇒ 若 400 "already loaded"，
     说明 400 是那次请求把名字装回来的结果，而不是账本自相矛盾
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sglang_lora_serving_audit import PROMPTS       # noqa: E402


def post(base, path, payload):
    r = requests.post(base + path, json=payload, timeout=300)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:200]


def gen(base, text, lora, max_new_tokens=12):
    r = requests.post(base + "/generate", json={
        "text": text, "lora_path": lora,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        "return_logprob": True, "logprob_start_len": 0}, timeout=180)
    if r.status_code != 200:
        return dict(status=r.status_code, error=r.text[:160])
    meta = r.json().get("meta_info", {})
    ids = [int(x[1]) for x in (meta.get("output_token_logprobs") or [])]
    return dict(status=200, token_ids=ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--adapter-c1", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    text = PROMPTS[0]
    rep = {"base": args.base}

    rep["ref_before"] = gen(args.base, text, "pub")
    print("参照输出:", rep["ref_before"].get("token_ids", [])[:6])

    # ---- A：unload → 不请求 → 显式 load ----
    rep["A_unload"] = post(args.base, "/unload_lora_adapter", {"lora_name": "pub"})[0:2]
    rep["A_reload_without_request"] = post(
        args.base, "/load_lora_adapter",
        {"lora_name": "pub", "lora_path": str(args.adapter_c1)})
    print(f"A unload={rep['A_unload'][0]} 之后不请求直接 load → "
          f"{rep['A_reload_without_request'][0]} "
          f"{rep['A_reload_without_request'][1].get('error_message', '')[:60]}")

    # ---- B：unload → 请求（隐式重载）→ 显式 load ----
    rep["B_unload"] = post(args.base, "/unload_lora_adapter", {"lora_name": "pub"})[0:2]
    rep["B_request_after_unload"] = gen(args.base, text, "pub")
    rep["B_reload_after_request"] = post(
        args.base, "/load_lora_adapter",
        {"lora_name": "pub", "lora_path": str(args.adapter_c1)})
    print(f"B unload={rep['B_unload'][0]} → 请求 {rep['B_request_after_unload'].get('status')} "
          f"→ 显式 load {rep['B_reload_after_request'][0]} "
          f"{rep['B_reload_after_request'][1].get('error_message', '')[:60]}")

    rep["verdict"] = dict(
        unload_really_empties_registry=(
            rep["A_reload_without_request"][0] == 200),
        request_implicitly_reloads=(
            rep["B_request_after_unload"].get("status") == 200),
        explicit_reload_then_fails_as_expected=(
            rep["B_reload_after_request"][0] != 200
            and "already loaded" in json.dumps(rep["B_reload_after_request"][1])),
        implicit_reload_same_weights=(
            rep["B_request_after_unload"].get("token_ids")
            == rep["ref_before"].get("token_ids")))
    (args.out / "sglang_lora_unload_probe.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("判定：" + json.dumps(rep["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
