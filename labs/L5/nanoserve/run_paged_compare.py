#!/usr/bin/env python3
"""L5.7 任务 B 的对照运行：gather+SDPA 执行器 vs 真实分页 kernel。

同一条请求轨迹分别在两个引擎上跑：
  base   —— 5.7 原有的 `Engine` + `PagedModel`（gather 后 SDPA）
  paged  —— `PagedEngine` + `PagedModelExec`（decode 走 `paged_decode_attention`）

比较三件事：
  1. **逐 token / 逐 logits 对拍**：同一步上两条读路径的 logits 最大绝对差，
     以及两端到端贪心输出是否逐 token 相同；
  2. **资源回收一致**：两边跑完后的 `leak_check` 与块池快照相同；
  3. **性能（含 metadata）**：B=1/8、长短混批、共享前缀三种负载的总墙钟与步数。

取消路径单独测：跑到一半 abort，两边都必须把块还回池子。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from engine import Engine, Request, State
from model import PagedModel
from paged_engine import PagedEngine, PagedModelExec

REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCK_SIZE = 16
MODEL = None


def get_model(num_blocks):
    """两个引擎共用同一份权重：`PagedModelExec` 继承自 `PagedModel`，
    基类引擎调 `forward`（gather+SDPA），分页引擎调 `decode_forward`。
    共用一个实例既省一次权重加载，也保证两边读的是同一份 KV。"""
    global MODEL
    if MODEL is None or MODEL.num_blocks != num_blocks:
        MODEL = None
        torch.cuda.empty_cache()
        MODEL = PagedModelExec(REPO, HUB, num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return MODEL


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


def run(eng_cls, tok, prompts, max_tokens, blocks=256, max_seqs=8,
        prefix_cache=False, abort_at=None):
    model = get_model(blocks)
    eng = eng_cls(model, block_size=BLOCK_SIZE, max_batched_tokens=256,
                  max_num_seqs=max_seqs, enable_prefix_cache=prefix_cache,
                  eos_ids=[])
    reqs = [Request(f"r{i}", prompt_ids(tok, p), max_tokens=max_tokens)
            for i, p in enumerate(prompts)]
    for r in reqs:
        eng.add(r)
    t0 = time.perf_counter()
    if abort_at is None:
        eng.run_until_idle(max_steps=20000)
    else:
        for step in range(abort_at):
            if not eng.waiting and not eng.running:
                break
            eng.step()
        if eng.running:
            eng.abort(eng.running[0].req_id, "test-cancel")
        eng.run_until_idle(max_steps=20000)
    wall = time.perf_counter() - t0
    return dict(wall_s=round(wall, 4), steps=eng.step_index,
                paged_tokens=getattr(eng, "paged_tokens", 0),
                produced={r.req_id: list(r.output_ids) for r in reqs},
                finished=sum(1 for r in reqs if r.state is State.FINISHED),
                aborted=sum(1 for r in reqs if r.state is State.ABORTED),
                leak=eng.leak_check(), pool=eng.pool.snapshot(),
                paged_steps=getattr(eng, "paged_steps", 0)), eng


def logit_probe(tok, prompt, max_tokens, blocks=64):
    """同一次 prefill 之后，用两条读路径各算一次 decode 步的 logits。"""
    model = get_model(blocks)
    eng = Engine(model, block_size=BLOCK_SIZE, max_batched_tokens=256,
                 max_num_seqs=1, enable_prefix_cache=False, eos_ids=[])
    ids = prompt_ids(tok, prompt)
    r = Request("p", ids, max_tokens=max_tokens)
    eng.add(r)
    for _ in range(2):
        eng.step()
    # 到这里 r 在 decode，KV 已写好。构造同一组下标，两条路径各读一次。
    start = r.kv_len
    # decode 分支在跑之前会先把块表长到 kv_len+1；探针必须做同样的事，
    # 否则 _slots(start) 会越出当前块表。
    eng._grow_block_table(r, start + 1)
    slot = eng._slots(r, start, 1)[0]
    ctx = start + 1
    gather = eng._gather(r, ctx, ctx)
    dev = model.device
    input_ids = torch.tensor([[r.all_ids[start]]], dtype=torch.long, device=dev)
    positions = torch.tensor([[start]], dtype=torch.long, device=dev)
    slot_map = torch.tensor([[slot]], dtype=torch.long, device=dev)
    gi = torch.tensor([gather], dtype=torch.long, device=dev)
    ctx_lens = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.inference_mode():
        # 基类路径需要模型是 PagedModelExec 才能同时提供 decode_forward
        base_logits = PagedModel.forward(model, input_ids, positions, slot_map, gi, ctx_lens)
        bt = torch.tensor([r.block_table], dtype=torch.int32, device=dev)
        sl = torch.tensor([ctx], dtype=torch.int32, device=dev)
        paged_logits = PagedModelExec.decode_forward(
            model, input_ids, positions, slot_map, bt, sl)
    d = (base_logits.float() - paged_logits.float()).abs()
    return dict(context=ctx, max_abs_diff=float(d.max()),
                mean_abs_diff=float(d.mean()),
                argmax_equal=bool(base_logits.argmax(-1).equal(paged_logits.argmax(-1))),
                base_top1=int(base_logits.argmax(-1)), paged_top1=int(paged_logits.argmax(-1)),
                base_logit_range=[float(base_logits.max()), float(base_logits.min())])


def first_difference(a, b):
    """两条路径的端到端输出首个不同的位置；返回 (位置, 细节)。"""
    for req in a:
        x, y = a[req], b.get(req, [])
        for i, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return i, dict(req=req, base=[u], paged=[v])
        if len(x) != len(y):
            return min(len(x), len(y)), dict(req=req, base=[len(x)], paged=[len(y)])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    filler = "Continue this story about a lighthouse keeper. "
    rows = []

    probe = logit_probe(tok, filler * 2, 8)
    print(f"logits 对拍：上下文 {probe['context']} token，"
          f"max|Δ| {probe['max_abs_diff']:.3e}，mean|Δ| {probe['mean_abs_diff']:.3e}，"
          f"argmax 相同 {probe['argmax_equal']}")

    cases = [("B=1", [filler * 2], 32, 1),
             ("B=8", [filler * 2] * 8, 32, 8),
             ("长短混批", [filler * 2] + [filler * 8] * 3 + [filler * 2] * 4, 32, 8)]
    for name, prompts, mt, seqs in cases:
        res = {}
        for tag, cls in [("base", Engine), ("base_rep", Engine),
                         ("paged", PagedEngine)]:
            r, _ = run(cls, tok, prompts, mt, max_seqs=seqs)
            res[tag] = r
        same = res["base"]["produced"] == res["paged"]["produced"]
        leak_same = res["base"]["leak"] == res["paged"]["leak"]
        # 对照：同一条路径跑两遍必须逐 token 相同，否则差异不能归因到读路径
        rep_same = res["base"]["produced"] == res["base_rep"]["produced"]
        first_diff, detail = first_difference(res["base"]["produced"],
                                             res["paged"]["produced"])
        rows.append(dict(case=name, base=res["base"], base_rep=res["base_rep"],
                         paged=res["paged"], outputs_identical=same,
                         base_repeat_identical=rep_same,
                         first_diff_position=first_diff, first_diff_detail=detail,
                         leak_identical=leak_same))
        print(f"{name:<8} base {res['base']['wall_s']:>6.3f}s/{res['base']['steps']:>4}步  "
              f"paged {res['paged']['wall_s']:>6.3f}s/{res['paged']['steps']:>4}步"
              f"（分页步 {res['paged']['paged_steps']}，分页 token "
              f"{res['paged'].get('paged_tokens')}）  输出相同 {same}  "
              f"（base 重跑一致 {rep_same}）  回收一致 {leak_same}")
        if first_diff is not None:
            print(f"         首个不同：请求 {detail['req']} 第 {first_diff} 个 token"
                  f"（base {detail['base']} / paged {detail['paged']}）")

    # 共享前缀
    prompts = [filler * 2] * 2
    res = {}
    for tag, cls in [("base", Engine), ("paged", PagedEngine)]:
        r, _ = run(cls, tok, prompts, 16, prefix_cache=True, max_seqs=2)
        res[tag] = r
    same = res["base"]["produced"] == res["paged"]["produced"]
    rows.append(dict(case="共享前缀", base=res["base"], paged=res["paged"],
                     outputs_identical=same,
                     leak_identical=res["base"]["leak"] == res["paged"]["leak"]))
    print(f"{'共享前缀':<8} base {res['base']['wall_s']:>6.3f}s  "
          f"paged {res['paged']['wall_s']:>6.3f}s  输出相同 {same}")

    # 取消
    res = {}
    for tag, cls in [("base", Engine), ("paged", PagedEngine)]:
        r, _ = run(cls, tok, [filler * 4] * 4, 64, max_seqs=4, abort_at=6)
        res[tag] = r
    rows.append(dict(case="取消", base=res["base"], paged=res["paged"]))
    print(f"{'取消':<8} base 泄漏 {res['base']['leak']['leaked']}  "
          f"paged 泄漏 {res['paged']['leak']['leaked']}  "
          f"两边 abort 数 {res['base']['aborted']}/{res['paged']['aborted']}")

    (args.out / "paged_compare.json").write_text(
        json.dumps(dict(probe=probe, rows=rows,
                        note="base=gather+SDPA，paged=真实分页 kernel；"
                             "分页路径只在纯 decode 组启用，prefill 仍走基类"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读法：两条读路径共用同一份 KV 与缓存池，logits 差只来自归约顺序；")
    print("      输出与块池回收必须完全一致，性能差才可归因到读路径本身。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
