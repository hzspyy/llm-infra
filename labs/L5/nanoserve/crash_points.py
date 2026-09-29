#!/usr/bin/env python3
"""L5.8 —— 崩溃点矩阵：哪些"半写状态"能重跑恢复，哪些不能。

`crash_recovery.py` 只在 `step()` 之前注入异常，那是最干净的一种崩溃。
这一轮把注入点挪进真正的写路径，回答"半写状态要不要重测"这个问题：

  P0  step 之前            —— 引擎状态没动（对照）
  P1  **KV 写入中途**       —— 借 `PagedModelExec.decode_forward` 每层一次调用
                             `paged_decode_attention` 的事实，在第 k 次调用时抛异常；
                             此时第 0..k-1 层已经写好了这一步的 KV，k 层之后没写
  P2  `_emit` 之后         —— 采样出来的 token 已经 append 进 output_ids
  P3  `_free` 之内         —— 释放块的中途（最容易留下泄漏）
  P4  `_schedule` 之内     —— 块表已扩、请求还没进 running

每一种都做三件事：
  1. 崩溃后**不处理**，直接看引擎还能不能服务；
  2. 在一个**新引擎**上重跑同一条请求，作为输出参照；
  3. 在**同一个引擎**上重启循环后重跑，比较输出是否与参照逐 token 相同、
     以及块池泄漏与运行/等待队列的残留。

判据写死在脚本里：
  * 若同引擎恢复后的输出与参照不同 → 说明半写状态被继续使用，是**静默错误**；
  * 若泄漏块 > 0 或 running 残留 → 说明需要重建而不是重跑。
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from engine import Request, State                              # noqa: E402
from model import PagedModel                                   # noqa: E402
from paged_engine import PagedEngine, PagedModelExec           # noqa: E402
import paged_engine                                            # noqa: E402

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCKS = 128
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = PagedModelExec(REPO, HUB, num_blocks=BLOCKS, block_size=16)
    return _MODEL


def engine():
    return PagedEngine(get_model(), block_size=16, max_batched_tokens=256,
                       max_num_seqs=4, enable_prefix_cache=False, eos_ids=[])


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


def run_request(eng, ids, max_tokens, crash=None):
    """跑一条请求，可选在指定点注入一次异常。返回结果与崩溃信息。"""
    req = Request("r0", ids, max_tokens=max_tokens)
    eng.add(req)
    info = dict(crashed=None, steps=0)
    guard = install(eng, crash, info) if crash else None
    try:
        eng.run_until_idle(max_steps=4000)
    except Exception as e:
        info["crashed"] = f"{type(e).__name__}: {e}"
        if guard is not None:
            guard["armed"] = False          # 只炸一次
    if guard is not None:
        uninstall(guard)
    return req, info


def install(eng, crash, info):
    """crash: (kind, n)。返回可卸载的 guard。"""
    kind, n = crash
    guard = dict(kind=kind, n=n, armed=True, count=0)
    patch = {}

    if kind == "kv_write":
        import paged_attn as _pa

        def wrapped(*a, **kw):
            if guard["armed"]:
                guard["count"] += 1
                if guard["count"] == n:
                    raise RuntimeError("injected crash: mid-KV-write "
                                       f"(layer {n})")
            return _pa.paged_decode_attention(*a, **kw)

        patch = (paged_engine, "paged_decode_attention",
                 paged_engine.paged_decode_attention, wrapped)
    elif kind == "emit":
        orig = eng._emit

        def wrapped_emit(req, logits, trace):
            if guard["armed"]:
                guard["count"] += 1
                if guard["count"] == n:
                    req.output_ids.append(0)      # 先留下一个 token 再炸
                    raise RuntimeError("injected crash: after emit")
            return orig(req, logits, trace)

        patch = (eng, "_emit", orig, wrapped_emit)
    elif kind == "free":
        orig = eng._free

        def wrapped_free(req):
            if guard["armed"]:
                guard["count"] += 1
                if guard["count"] == n:
                    if req.block_table:
                        eng.pool.release(req.block_table[0])   # 只还第一个
                    raise RuntimeError("injected crash: inside _free")
            return orig(req)

        patch = (eng, "_free", orig, wrapped_free)
    elif kind == "schedule":
        orig = eng._grow_block_table

        def wrapped_grow(req, upto):
            if guard["armed"]:
                guard["count"] += 1
                if guard["count"] == n:
                    orig(req, upto)                    # 块表已扩
                    raise RuntimeError("injected crash: after grow")
            return orig(req, upto)

        patch = (eng, "_grow_block_table", orig, wrapped_grow)
    elif kind == "pre_step":
        orig = eng.step

        def wrapped_step():
            if guard["armed"]:
                guard["count"] += 1
                if guard["count"] == n:
                    raise RuntimeError("injected crash: before step")
            return orig()

        patch = (eng, "step", orig, wrapped_step)
    else:
        raise ValueError(kind)

    obj, name, orig, new = patch
    setattr(obj, name, new)
    guard["patch"] = (obj, name, orig)
    info["crash_kind"] = kind
    info["crash_at"] = n
    return guard


def uninstall(guard):
    obj, name, orig = guard["patch"]
    setattr(obj, name, orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    ids = prompt_ids(tok, "Continue this story about a lighthouse keeper. " * 2)

    # 干净参照
    eng0 = engine()
    ref_req, _ = run_request(eng0, ids, args.max_tokens)
    reference = list(ref_req.output_ids)
    del eng0
    gc.collect()

    points = [("P0_pre_step", ("pre_step", 8)),
              ("P1_kv_write_mid", ("kv_write", 8)),
              ("P2_after_emit", ("emit", 3)),
              ("P3_inside_free", ("free", 1)),
              ("P4_after_grow", ("schedule", 2))]
    rows = []
    for label, crash in points:
        eng = engine()
        victim, info = run_request(eng, ids, args.max_tokens, crash=crash)
        after = dict(running=len(eng.running), waiting=len(eng.waiting),
                     leak=eng.leak_check()["leaked"],
                     victim_tokens=len(victim.output_ids),
                     victim_state=victim.state.value)

        # ---- 判据 1：崩溃后**继续跑同一个引擎**，让受害者自己跑完。
        # 这才是"半写状态会不会被继续使用"的检验：victim 仍留在 running 里，
        # 引擎会从它当前状态接着算。若半写的 KV 被沿用，输出会与干净参照不同。
        resume_err = None
        try:
            eng.run_until_idle(max_steps=4000)
        except Exception as e:
            resume_err = f"{type(e).__name__}: {e}"
        victim_after_resume = list(victim.output_ids)
        # 判据 1 只在受害者真的跑完时才有意义
        resume_matches = (victim_after_resume == reference) if not resume_err else None

        # ---- 判据 2：把残留的受害者 abort 掉，块池应当回到干净状态
        cleanup_ok = None
        if victim.state is not State.FINISHED:
            try:
                eng.abort(victim.req_id, "test-cleanup")
                cleanup_ok = eng.leak_check()["leaked"] == 0
            except Exception as e:
                cleanup_ok = f"abort失败 {type(e).__name__}: {e}"
        after_cleanup = dict(running=len(eng.running), waiting=len(eng.waiting),
                             leak=eng.leak_check()["leaked"])

        # ---- 判据 3：清干净之后再发一条同样的请求，与干净参照比
        fresh_matches = None
        try:
            req3, _ = run_request(eng, ids, args.max_tokens)
            fresh_matches = list(req3.output_ids) == reference
        except Exception as e:
            fresh_matches = f"失败 {type(e).__name__}: {e}"

        rows.append(dict(point=label, crash_kind=crash[0], crash_at=crash[1],
                         crashed=info["crashed"],
                         state_after_crash=after,
                         resume_error=resume_err,
                         victim_tokens_after_resume=len(victim_after_resume),
                         resume_output_matches_reference=resume_matches,
                         victim_final_state=victim.state.value,
                         abort_cleanup_ok=cleanup_ok,
                         state_after_cleanup=after_cleanup,
                         fresh_request_matches_reference=fresh_matches,
                         reference_tokens=len(reference)))
        del eng
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  {label:<16} 崩溃={bool(info['crashed'])} "
              f"崩溃后 running/waiting={after['running']}/{after['waiting']} "
              f"泄漏={after['leak']}；续跑后 victim {len(victim_after_resume)} token "
              f"与参照一致={resume_matches}；abort 后泄漏="
              f"{after_cleanup['leak']}；新请求一致={fresh_matches}", flush=True)

    report = dict(blocks=BLOCKS, max_tokens=args.max_tokens,
                  reference_tokens=len(reference), rows=rows)
    (args.out / "crash_points.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n判读：同引擎输出与参照不一致 = 半写状态被继续使用（静默错误）；"
          "泄漏>0 或 running 残留 = 只能重建，不能重跑。")


if __name__ == "__main__":
    main()
