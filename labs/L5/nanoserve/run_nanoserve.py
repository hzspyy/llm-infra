#!/usr/bin/env python3
"""L5.7 nanoserve 六次迭代。每个阶段单独可跑，输出是 JSONL + 人读摘要。

    python run_nanoserve.py --out NEW_DIRECTORY [--stages 1 2 3 4 5 6]

阶段：
  1 单请求闭环，并与 transformers 的贪心解码逐 token 对拍
  2 连续批处理：错峰到达 vs 静态批
  3 分页 KV：block table、slot mapping、泄漏检查
  4 前缀复用：共享 system prompt 时省下的 prefill token
  5 chunked prefill：长 prompt 对同批 decode 的影响，扫 token 预算
  6 取消：跑到一半 abort，块必须全部回收
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from engine import Engine, Request, State, OutOfBlocks
from model import PagedModel

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
OUT: Path
LOG = []


def emit(kind, **fields):
    row = {"kind": kind, **fields}
    LOG.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def free_gib():
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def build(num_blocks, **engine_kw):
    model = PagedModel(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return model, Engine(model, block_size=BLOCK_SIZE, **engine_kw)


# ------------------------------------------------------------------ 阶段 1
def stage1(tok, eos):
    """一条请求走完全程。状态迁移和 block table 在这里第一次出现。"""
    model, eng = build(256, max_batched_tokens=512, max_num_seqs=4,
                       enable_prefix_cache=False, eos_ids=eos)
    emit("model", repo=REPO, snapshot=model.snapshot,
         layers=model.cfg.n_layers, n_q=model.cfg.n_q, n_kv=model.cfg.n_kv,
         head_dim=model.cfg.head_dim, vocab=model.cfg.vocab,
         kv_bytes_per_token=model.kv_bytes_per_token(),
         kv_pool_bytes=model.cache_bytes(), block_size=BLOCK_SIZE, num_blocks=256)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Name three prime numbers."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    req = Request("r0", ids, max_tokens=24)
    eng.add(req)
    emit("s1_admit", req="r0", state=req.state.value, prompt_tokens=len(ids))
    seen = [req.state.value]
    while req.state in (State.WAITING, State.PREFILL, State.DECODE):
        t = eng.step()
        if req.state.value != seen[-1]:
            seen.append(req.state.value)
            emit("s1_transition", step=t.step, state=req.state.value,
                 block_table=list(req.block_table), kv_len=req.kv_len)
        if t.step < 3:
            emit("s1_step", step=t.step, prefill=t.prefill, decode=t.decode,
                 prefill_tokens=t.prefill_tokens, decode_tokens=t.decode_tokens,
                 pool_free=t.pool.get("free"), seconds=round(t.seconds, 5))
    text = tok.decode(req.output_ids)
    emit("s1_done", states=seen, finish_reason=req.finish_reason,
         output_tokens=len(req.output_ids), text=text,
         leak=eng.leak_check())

    # 同一条请求，把预算压到 8 个 token，让 PREFILL 这个中间态显形。
    eng2 = Engine(model, block_size=BLOCK_SIZE, max_batched_tokens=8,
                  max_num_seqs=4, enable_prefix_cache=False, eos_ids=eos)
    r2 = Request("r1", ids, max_tokens=3)
    eng2.add(r2)
    traj = [{"step": -1, "state": r2.state.value, "num_computed": r2.num_computed,
             "blocks": [], "kv_len": r2.kv_len}]
    while r2.state in (State.WAITING, State.PREFILL, State.DECODE):
        t = eng2.step()
        traj.append({"step": t.step, "state": r2.state.value,
                     "num_computed": r2.num_computed,
                     "blocks": list(r2.block_table), "kv_len": r2.kv_len,
                     "prefill_tokens": t.prefill_tokens,
                     "decode_tokens": t.decode_tokens})
    emit("s1_trajectory", budget=8, prompt_tokens=len(ids),
         prefill_chunks=r2.prefill_chunks, steps=traj, leak=eng2.leak_check())

    # 对拍：同一 prompt、同样贪心，和 transformers 逐 token 比。
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        REPO, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    with torch.inference_mode():
        out = ref.generate(torch.tensor([ids], device="cuda"),
                           max_new_tokens=24, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    ref_ids = out[0, len(ids):].tolist()
    n = min(len(ref_ids), len(req.output_ids))
    same = [a == b for a, b in zip(ref_ids[:n], req.output_ids[:n])]
    first_div = same.index(False) if not all(same) else None
    emit("s1_crosscheck", reference="transformers.generate greedy",
         ref_tokens=ref_ids, mine=req.output_ids, compared=n,
         identical=all(same) and len(ref_ids) == len(req.output_ids),
         first_divergence=first_div, ref_text=tok.decode(ref_ids))
    del ref
    torch.cuda.empty_cache()
    return model


# ------------------------------------------------------------------ 阶段 2
def stage2(tok, eos):
    """连续批处理 vs 静态批。

    两种模式**用同一份错峰到达序列**，唯一的差别是新请求能不能中途上车：
      continuous —— 到了就进队列，下一步就能被调度
      static     —— 攒着，等引擎彻底跑空才把攒下的一起放进去
    负载用「同时在跑的请求数」来扫：请求越多，静态批让人干等的时间越长。
    """
    base = "Write one sentence about the number {}."
    ARRIVE_EVERY = 2
    MAX_TOKENS = 24

    def encode(i):
        return tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": base.format(i)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)

    def run(continuous: bool, n: int, encoded):
        _, eng = build(1024, max_batched_tokens=1024, max_num_seqs=n,
                       enable_prefix_cache=False, eos_ids=eos)
        pending = list(range(n))
        buffered: list[Request] = []
        created: list[Request] = []
        t0 = time.perf_counter()
        step = 0
        while pending or buffered or eng.waiting or eng.running:
            if pending and step % ARRIVE_EVERY == 0:
                i = pending.pop(0)
                r = Request(f"q{i}", encoded[i], max_tokens=MAX_TOKENS)
                created.append(r)
                (eng.add if continuous else buffered.append)(r)
            if not continuous and buffered and not eng.running and not eng.waiting:
                for r in buffered:
                    eng.add(r)
                buffered.clear()
            eng.step()
            step += 1
        wall = time.perf_counter() - t0
        lat = sorted((r.finished_at - r.arrival) * 1000 for r in created)
        out_tokens = sum(len(r.output_ids) for r in created)
        return eng, {"wall_s": round(wall, 4), "steps": eng.step_index,
                     "mean_ms": round(sum(lat) / len(lat), 1),
                     "p50_ms": round(lat[len(lat) // 2], 1),
                     "max_ms": round(lat[-1], 1),
                     "min_ms": round(lat[0], 1),
                     "output_tokens": out_tokens,
                     "tok_per_s": round(out_tokens / wall, 1),
                     "leaked": eng.leak_check()["leaked"]}

    REPEATS = 3
    for n in (4, 8, 16):
        encoded = [encode(i) for i in range(n)]
        reps = {"continuous": [], "static": []}
        for rep in range(REPEATS):
            for mode in (True, False):
                eng, stats = run(mode, n, encoded)
                reps["continuous" if mode else "static"].append(stats)
                if n == 8 and mode and rep == 0:
                    for t in eng.traces[:12]:
                        emit("s2_step", n=n, continuous=mode, step=t.step,
                             prefill=t.prefill, decode=t.decode, finished=t.finished)
        def med(key, mode):
            xs = sorted(r[key] for r in reps[mode])
            return xs[len(xs) // 2]
        summary = {m: {k: med(k, m) for k in
                       ("wall_s", "steps", "mean_ms", "p50_ms", "max_ms", "tok_per_s")}
                   for m in ("continuous", "static")}
        emit("s2_load", n_requests=n, arrive_every_steps=ARRIVE_EVERY,
             max_tokens=MAX_TOKENS, repeats=REPEATS, median=summary,
             raw=reps,
             mean_ratio=round(summary["static"]["mean_ms"]
                              / summary["continuous"]["mean_ms"], 3),
             throughput_ratio=round(summary["continuous"]["tok_per_s"]
                                    / summary["static"]["tok_per_s"], 3))


# ------------------------------------------------------------------ 阶段 3
def stage3(tok, eos):
    """分页：两条序列的 block table 在物理上交错，逻辑上各自连续。"""
    _, eng = build(64, max_batched_tokens=64, max_num_seqs=4,
                   enable_prefix_cache=False, eos_ids=eos)
    ids_a = tok.encode("A" * 1, add_special_tokens=False) * 40
    ids_b = tok.encode("B" * 1, add_special_tokens=False) * 40
    a, b = Request("A", ids_a, max_tokens=8), Request("B", ids_b, max_tokens=8)
    eng.add(a)
    eng.add(b)
    for _ in range(6):
        eng.step()
        emit("s3_tables", step=eng.step_index - 1,
             A={"state": a.state.value, "blocks": list(a.block_table),
                "kv": a.kv_len},
             B={"state": b.state.value, "blocks": list(b.block_table),
                "kv": b.kv_len},
             pool_free=eng.pool.num_free)
    # 逻辑位置 -> 物理槽位
    mapping = [{"pos": p, "block": a.block_table[p // BLOCK_SIZE],
                "slot": a.block_table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE}
               for p in (0, 15, 16, 31, 32)]
    emit("s3_slot_mapping", req="A", entries=mapping, block_size=BLOCK_SIZE)
    eng.run_until_idle()
    emit("s3_leak", **eng.leak_check())


# ------------------------------------------------------------------ 阶段 4
def stage4(tok, eos):
    """前缀复用：同一段 system prompt 的第 2..N 条请求少算多少 token。"""
    system = ("You are a careful assistant. Answer in one short sentence. "
              "Do not add explanations, disclaimers or code fences. ") * 4
    questions = ["What is 2+2?", "What is the capital of France?",
                 "Name one prime.", "What color is the sky?"]
    encoded = [tok.encode(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False),
        add_special_tokens=False) for q in questions]
    shared = 0
    while all(e[shared] == encoded[0][shared] for e in encoded):
        shared += 1
    emit("s4_prompts", lengths=[len(e) for e in encoded],
         shared_prefix_tokens=shared, block_size=BLOCK_SIZE,
         predicted_reusable_blocks=shared // BLOCK_SIZE)

    for enable in (True, False):
        _, eng = build(512, max_batched_tokens=256, max_num_seqs=1,
                       enable_prefix_cache=enable, eos_ids=eos)
        rows = []
        for i, ids in enumerate(encoded):
            r = Request(f"p{i}", ids, max_tokens=8)
            eng.add(r)
            eng.run_until_idle()          # 一条一条跑，命中的只能是上一条留下的块
            rows.append({"req": r.req_id, "prompt_tokens": len(ids),
                         "cached_blocks": r.cached_prefix_blocks,
                         "computed_prefill_tokens": len(ids) - r.cached_prefix_blocks * BLOCK_SIZE,
                         "ttft_ms": round((r.first_token_at - r.arrival) * 1000, 2)})
        emit("s4_result", prefix_cache=enable, rows=rows,
             pool=eng.pool.snapshot(), leak=eng.leak_check())


# ------------------------------------------------------------------ 阶段 5
def stage5(tok, eos):
    """chunked prefill：长 prompt 插队时，同批 decode 的每步耗时。"""
    long_prompt = tok.encode(("The quick brown fox jumps over the lazy dog. " * 120),
                             add_special_tokens=False)
    short = [tok.encode(f"Count to {i}.", add_special_tokens=False) for i in range(3)]
    for budget in (64, 256, 1024):
        _, eng = build(512, max_batched_tokens=budget, max_num_seqs=8,
                       enable_prefix_cache=False, eos_ids=eos)
        for i, ids in enumerate(short):
            eng.add(Request(f"d{i}", ids, max_tokens=40))
        for _ in range(6):                       # 先让 decode 跑起来
            eng.step()
        base = [t.seconds for t in eng.traces[-3:]]
        big = Request("BIG", long_prompt, max_tokens=4)
        eng.add(big)
        during = []
        while big.state is not State.DECODE and big.state is not State.FINISHED:
            t = eng.step()
            during.append({"step": t.step, "prefill_tokens": t.prefill_tokens,
                           "decode_tokens": t.decode_tokens,
                           "seconds": round(t.seconds, 5)})
        emit("s5_budget", max_batched_tokens=budget,
             long_prompt_tokens=len(long_prompt),
             prefill_chunks=big.prefill_chunks,
             steady_step_seconds=[round(x, 5) for x in base],
             steps_during_prefill=during,
             max_step_seconds=round(max(d["seconds"] for d in during), 5))
        for r in list(eng.running):
            eng.abort(r.req_id, "stage-end")
        emit("s5_leak", max_batched_tokens=budget, **eng.leak_check())


# ------------------------------------------------------------------ 阶段 6
def stage6(tok, eos):
    """取消：跑到一半 abort，块必须一块不剩地回收。"""
    _, eng = build(128, max_batched_tokens=256, max_num_seqs=4,
                   enable_prefix_cache=False, eos_ids=eos)
    ids = tok.encode("Tell me a long story about a robot. " * 8,
                     add_special_tokens=False)
    reqs = [Request(f"c{i}", ids, max_tokens=64) for i in range(3)]
    for r in reqs:
        eng.add(r)
    for _ in range(5):
        eng.step()
    before = eng.pool.snapshot()
    victim = reqs[1]
    held = list(victim.block_table)
    ok = eng.abort(victim.req_id, "client-disconnect")
    after = eng.pool.snapshot()
    emit("s6_abort", req=victim.req_id, aborted=ok, state=victim.state.value,
         finish_reason=victim.finish_reason, held_blocks=held,
         pool_before=before, pool_after=after,
         freed=after["free"] - before["free"])
    eng.run_until_idle()
    emit("s6_leak", **eng.leak_check(),
         done={r.req_id: (r.state.value, r.finish_reason) for r in reqs})


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    OUT = args.out
    src = {f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
           for f in ("engine.py", "model.py", "run_nanoserve.py")}
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    gen = GenerationConfig.from_pretrained(REPO, local_files_only=True)
    eos = gen.eos_token_id
    eos = (eos,) if isinstance(eos, int) else tuple(eos)
    free, total = free_gib()
    emit("environment", model=REPO, torch=torch.__version__,
         gpu=torch.cuda.get_device_name(0), free_gib=round(free, 2),
         total_gib=round(total, 2), eos_ids=list(eos), sources=src)
    for s in args.stages:
        emit("stage_begin", stage=s)
        t0 = time.perf_counter()
        STAGES[s](tok, eos)
        emit("stage_end", stage=s, seconds=round(time.perf_counter() - t0, 3))
    (args.out / "log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in LOG) + "\n")
