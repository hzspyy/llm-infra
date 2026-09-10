#!/usr/bin/env python3
"""L5.8 失败路径实验。

    python run_failures.py --out NEW_DIRECTORY [--cases 1 2 3 4 5]

  1 三种状态下取消：waiting / prefill / decode，各自的清理路径与块账
  2 超时：队列里超时 vs 运行中超时
  3 抢占：块池不够时踢谁、丢多少已算的 token；对照「不抢占直接抛异常」
  4 背压：等待队列设不设上限，排队延迟 vs 拒绝率
  5 泄漏：把 abort 写错（忘记还块），看块池怎么被抽干、泄漏检查怎么抓到
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
from engine import Request, State, OutOfBlocks
from failure import ResilientEngine, attach_deadline
from model import PagedModel

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
LOG = []
MODEL = None


def emit(kind, **fields):
    row = {"kind": kind, **fields}
    LOG.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def get_model(num_blocks):
    """块池大小要变时才重新建模型；其余情况复用，省显存也省时间。"""
    global MODEL
    if MODEL is None or MODEL.num_blocks != num_blocks:
        MODEL = None
        torch.cuda.empty_cache()
        MODEL = PagedModel(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return MODEL


def engine(num_blocks, **kw):
    return ResilientEngine(get_model(num_blocks), block_size=BLOCK_SIZE, **kw)


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False),
        add_special_tokens=False)


# ------------------------------------------------------------------ 1 取消
def case1(tok, eos):
    """同一个 abort() 调用，在三种状态下走的清理路径不同。"""
    ids = prompt_ids(tok, "Tell me a long story about a robot. " * 6)
    for target_state in ("waiting", "prefill", "decode"):
        eng = engine(48, max_batched_tokens=32, max_num_seqs=2,
                     enable_prefix_cache=False, eos_ids=eos)
        keep = Request("keep", ids, max_tokens=32)
        victim = Request("victim", ids, max_tokens=32)
        eng.add(keep)
        eng.add(victim)
        # 走到目标状态
        guard = 0
        while victim.state.value != target_state and guard < 40:
            eng.step()
            guard += 1
        before = eng.pool.snapshot()
        blocks = list(victim.block_table)
        in_waiting = any(r.req_id == "victim" for r in eng.waiting)
        in_running = any(r.req_id == "victim" for r in eng.running)
        ok = eng.abort("victim", "client-disconnect")
        after = eng.pool.snapshot()
        emit("c1_abort", target_state=target_state, reached=victim.state.value,
             was_in_waiting=in_waiting, was_in_running=in_running,
             aborted=ok, held_blocks=blocks,
             free_before=before["free"], free_after=after["free"],
             returned=after["free"] - before["free"],
             keep_blocks=list(keep.block_table),
             keep_refcounts_unchanged=all(
                 before["refcounts"].get(str(b)) == after["refcounts"].get(str(b))
                 for b in keep.block_table),
             events=eng.stats.events[-1:])
        eng.run_until_idle()
        emit("c1_leak", target_state=target_state, **eng.leak_check(),
             keep_finish=keep.finish_reason, victim_finish=victim.finish_reason)


# ------------------------------------------------------------------ 2 超时
def case2(tok, eos):
    """deadline 到点就 abort。区别在于它当时在队列里还是在跑。"""
    ids = prompt_ids(tok, "Write a very long essay about the sea. " * 4)
    eng = engine(64, max_batched_tokens=64, max_num_seqs=1,
                 enable_prefix_cache=False, eos_ids=eos)
    slow = Request("slow", ids, max_tokens=200)          # 跑很久，占住唯一的 seq
    queued = Request("queued", ids, max_tokens=8)
    attach_deadline(slow, 0.25)
    attach_deadline(queued, 0.05)                        # 还没上车就会超时
    eng.add(slow)
    eng.add(queued)
    fired = []
    t0 = time.perf_counter()
    while (eng.waiting or eng.running) and time.perf_counter() - t0 < 5:
        fired += eng.check_deadlines()
        eng.step()
    eng.check_deadlines()
    emit("c2_timeout", fired=fired,
         slow={"state": slow.state.value, "reason": slow.finish_reason,
               "output_tokens": len(slow.output_ids)},
         queued={"state": queued.state.value, "reason": queued.finish_reason,
                 "output_tokens": len(queued.output_ids)},
         events=[e for e in eng.stats.events if e["kind"] in ("timeout", "abort")],
         leak=eng.leak_check())


# ------------------------------------------------------------------ 3 抢占
def case3(tok, eos):
    """块池故意开小，让并发请求把它撑爆。"""
    ids = prompt_ids(tok, "Continue this story about a lighthouse keeper. " * 3)
    n, blocks, max_tokens = 6, 24, 64
    for preempt in (True, False):
        eng = engine(blocks, max_batched_tokens=256, max_num_seqs=n,
                     enable_prefix_cache=False, eos_ids=eos)
        if not preempt:
            eng._preempt_one = lambda: None          # 关掉抢占，看它怎么死
        reqs = [Request(f"x{i}", ids, max_tokens=max_tokens) for i in range(n)]
        for r in reqs:
            eng.add(r)
        crashed = None
        t0 = time.perf_counter()
        try:
            eng.run_until_idle(max_steps=4000)
        except OutOfBlocks as e:
            crashed = str(e)
        wall = time.perf_counter() - t0
        emit("c3_run", preempt_enabled=preempt, num_blocks=blocks, requests=n,
             prompt_tokens=len(ids), max_tokens=max_tokens, crashed=crashed,
             wall_s=round(wall, 4), steps=eng.step_index,
             preempted=eng.stats.preempted,
             recomputed_kv_tokens=eng.stats.recomputed_tokens,
             finished=sum(1 for r in reqs if r.state is State.FINISHED),
             produced_tokens=sum(len(r.output_ids) for r in reqs),
             leak=eng.leak_check(),
             preempt_events=[e for e in eng.stats.events if e["kind"] == "preempt"][:8])


# ------------------------------------------------------------------ 4 背压
def case4(tok, eos):
    """等待队列有没有上限，是「都慢」和「一部分被拒」之间的选择。"""
    ids = prompt_ids(tok, "Summarize the theory of relativity in one line.")
    burst = 24
    for limit in (None, 12, 4):
        eng = engine(256, max_batched_tokens=256, max_num_seqs=4,
                     enable_prefix_cache=False, eos_ids=eos, max_waiting=limit)
        reqs = [Request(f"b{i}", ids, max_tokens=16) for i in range(burst)]
        accepted = [r for r in reqs if eng.try_add(r)]
        eng.run_until_idle(max_steps=4000)
        lat = sorted((r.finished_at - r.arrival) * 1000 for r in accepted
                     if r.finished_at)
        emit("c4_backpressure", max_waiting=limit, burst=burst,
             accepted=len(accepted), rejected=eng.stats.rejected,
             p50_ms=round(lat[len(lat) // 2], 1) if lat else None,
             max_ms=round(lat[-1], 1) if lat else None,
             steps=eng.step_index, leak=eng.leak_check())


# ------------------------------------------------------------------ 5 泄漏
def case5(tok, eos):
    """把 abort 写错：请求摘掉了，块没还。看池子怎么一点点被抽干。"""
    ids = prompt_ids(tok, "Say hello. " * 4)
    for leak in (True, False):
        eng = engine(32, max_batched_tokens=64, max_num_seqs=2,
                     enable_prefix_cache=False, eos_ids=eos, leak_on_abort=leak)
        rows, crashed = [], None
        try:
            for i in range(20):
                r = Request(f"L{i}", ids, max_tokens=32)
                eng.add(r)
                for _ in range(3):
                    eng.step()
                served = len(r.output_ids)
                eng.abort(r.req_id, "client-disconnect")
                rows.append({"i": i, "free": eng.pool.num_free,
                             "tokens_produced": served})
        except (OutOfBlocks, RuntimeError) as e:
            crashed = f"{type(e).__name__}: {e}"
        first_starved = next((r["i"] for r in rows if r["tokens_produced"] == 0), None)
        emit("c5_leak", leak_on_abort=leak, per_request=rows,
             first_request_that_produced_nothing=first_starved,
             crashed=crashed, pool=eng.pool.snapshot(), check=eng.leak_check())


CASES = {1: case1, 2: case2, 3: case3, 4: case4, 5: case5}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    gen = GenerationConfig.from_pretrained(REPO, local_files_only=True)
    eos = gen.eos_token_id
    eos = (eos,) if isinstance(eos, int) else tuple(eos)
    free, total = torch.cuda.mem_get_info()
    emit("environment", model=REPO, torch=torch.__version__,
         gpu=torch.cuda.get_device_name(0),
         free_gib=round(free / 2**30, 2), total_gib=round(total / 2**30, 2),
         block_size=BLOCK_SIZE, eos_ids=list(eos),
         sources={f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
                  for f in ("engine.py", "failure.py", "model.py", "run_failures.py")})
    for c in args.cases:
        emit("case_begin", case=c)
        t0 = time.perf_counter()
        CASES[c](tok, eos)
        emit("case_end", case=c, seconds=round(time.perf_counter() - t0, 3))
    (args.out / "log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in LOG) + "\n")
