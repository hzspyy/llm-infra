#!/usr/bin/env python3
"""L5.10 任务 C（SGLang 侧）—— 同名权重替换与运行中卸载/替换。

计划要求：「分开同一已加载 ID 的权重替换与更换 adapter ID」
「请求执行中卸载/替换」「记录缓存命中或实际 KV 状态」。

SGLang 提供两个管理接口（`srt/entrypoints/http_server.py:1551/1574`）：
`POST /load_lora_adapter {lora_name, lora_path, pinned}` 与
`POST /unload_lora_adapter {lora_name}`。

判据设计（不依赖第二个服务）：
  1. 启动时注册 `pub` = 权重 C1，请求得到 T1 —— 这就是 C1 的参照；
  2. 把**不同权重** C2 用**同一个名字 `pub`** 重新 load，再请求得到 T2；
  3. 另外把 C2 用**另一个名字 `pub2`** load 一次，请求得到 T_ref2 ——
     这是同一台引擎上 C2 权重的干净参照；
  4. 于是：T2 == T_ref2 ⇒ 同名更新**真的换了权重**；
     T2 == T1 ⇒ 同名更新被忽略（老权重还在用）；
     两者都不是 ⇒ 需要进一步查（例如 KV 复用）。
  5. 再测 unload：卸载 `pub` 后请求应当被拒；重新 load 后应当恢复成 T1。
  6. 最后测**生成中途卸载**：流式请求跑到一半 unload，记录客户端看到什么、
     以及之后引擎还能不能服务。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sglang_lora_serving_audit import make_adapter, PROMPTS      # noqa: E402


def gen(base, text, lora=None, max_new_tokens=16):
    payload = {"text": text,
               "sampling_params": {"temperature": 0.0,
                                   "max_new_tokens": max_new_tokens},
               "return_logprob": True, "logprob_start_len": 0}
    if lora:
        payload["lora_path"] = lora
    r = requests.post(base + "/generate", json=payload, timeout=180)
    if r.status_code != 200:
        return dict(status=r.status_code, error=r.text[:180])
    meta = r.json().get("meta_info", {})
    ids = [int(x[1]) for x in (meta.get("output_token_logprobs") or [])]
    if not ids:
        ids = [int(x) for x in (meta.get("output_ids") or [])]
    return dict(status=200, token_ids=ids, n_tokens=len(ids))


def admin(base, path, payload):
    t0 = time.perf_counter()
    r = requests.post(base + path, json=payload, timeout=300)
    return dict(endpoint=path, payload=payload, status=r.status_code,
                body=r.text[:200], wall_s=round(time.perf_counter() - t0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="/scratch/learn/models/hf/hub/"
                                       "models--Qwen--Qwen3-1.7B/snapshots/"
                                       "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--adapter-c1", required=True)
    ap.add_argument("--adapter-c2", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    text = PROMPTS[0]
    steps, admins = [], []

    def step(label, rec):
        rec["step"] = label
        steps.append(rec)
        print(f"  {label:<30} status={rec.get('status')} "
              f"tokens={rec.get('n_tokens')} "
              f"{'ERR ' + rec['error'][:70] if rec.get('error') else ''}", flush=True)

    def admin_call(label, path, payload):
        rec = admin(args.base, path, payload)
        rec["step"] = label
        admins.append(rec)
        print(f"  {label:<30} {path} → {rec['status']} {rec['body'][:80]}", flush=True)
        return rec

    # 0) base 参照：必须采集，否则"卸载后请求仍成功"无法区分
    #    "adapter 还在生效" 与 "静默降级成 base"
    step("S0_base_reference", gen(args.base, text, None))
    # 1) C1 参照
    step("S1_pub_C1", gen(args.base, text, "pub"))
    # 2) C2 用另一个名字：干净参照
    admin_call("A1_load_pub2_C2", "/load_lora_adapter",
               {"lora_name": "pub2", "lora_path": str(args.adapter_c2)})
    step("S2_pub2_C2_reference", gen(args.base, text, "pub2"))
    # 3) 同名 pub 换成 C2
    admin_call("A2_reload_pub_as_C2", "/load_lora_adapter",
               {"lora_name": "pub", "lora_path": str(args.adapter_c2)})
    step("S3_pub_after_same_name_reload", gen(args.base, text, "pub"))
    # 4) 卸载 pub → 请求应被拒
    admin_call("A3_unload_pub", "/unload_lora_adapter", {"lora_name": "pub"})
    step("S4_pub_after_unload", gen(args.base, text, "pub"))
    # 5) 按路径重新装载 C1 → 应恢复
    admin_call("A4_reload_pub_C1", "/load_lora_adapter",
               {"lora_name": "pub", "lora_path": str(args.adapter_c1)})
    step("S5_pub_after_reload_C1", gen(args.base, text, "pub"))
    # 6) 生成中途卸载
    mid = {}
    payload = {"text": text, "lora_path": "pub",
               "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
               "stream": True}
    try:
        with requests.post(args.base + "/generate", json=payload, stream=True,
                           timeout=60) as r:
            got = 0
            for line in r.iter_lines():
                if line.startswith(b"data:"):
                    got += 1
                    if got == 3:
                        mid["unload"] = admin(args.base, "/unload_lora_adapter",
                                              {"lora_name": "pub"})
                        break
        mid["chunks_before_unload"] = got
    except Exception as e:
        mid["error"] = f"{type(e).__name__}: {e}"
    step("S6_after_midstream_unload", gen(args.base, text, "pub"))
    mid["subsequent_request"] = steps[-1]

    report = dict(model=args.model, steps=steps, admin_calls=admins,
                  midstream_unload=mid)
    s = {x["step"]: x for x in steps}
    t1 = s["S1_pub_C1"].get("token_ids")
    t2 = s["S2_pub2_C2_reference"].get("token_ids")
    s3 = s["S3_pub_after_same_name_reload"].get("token_ids")
    s5 = s["S5_pub_after_reload_C1"].get("token_ids")
    s0 = s["S0_base_reference"].get("token_ids")
    report["verdict"] = dict(
        base_differs_from_C1=(s0 != t1),
        base_differs_from_C2=(s0 != t2),
        C1_and_C2_differ=t1 != t2,
        after_unload_matches_C1=(s["S4_pub_after_unload"].get("token_ids") == t1),
        after_unload_falls_back_to_base=(s["S4_pub_after_unload"].get("token_ids") == s0),
        same_name_reload_took_effect=(s3 == t2),
        same_name_reload_ignored=(s3 == t1),
        unload_rejects_request=s["S4_pub_after_unload"].get("status") != 200,
        reload_restores_C1=(s5 == t1),
        served_after_midstream_unload=steps[-1].get("status") == 200)
    (args.out / "sglang_lora_admin.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n判定：" + json.dumps(report["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
