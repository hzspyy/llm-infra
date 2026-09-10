#!/usr/bin/env python3
"""L5.8：在真实 vLLM 上触发抢占与取消，抓原文日志和统计。

    L58_GPU_MEMORY_UTILIZATION=0.12 python vllm_failure_paths.py --out NEW_DIRECTORY

A 抢占：把 KV 预算压到很小，让并发长输出撑爆块池，抓 vLLM 自己打的日志。
B 取消：generate 跑到一半 abort，检查引擎侧的请求数与显存是否回到基线。
"""
import argparse, io, json, logging, os, re, time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
MODEL = os.environ.get("L58_MODEL", "Qwen/Qwen3-1.7B")

p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
p.add_argument("--n", type=int, default=48)
p.add_argument("--max-tokens", type=int, default=512)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)

if "L58_GPU_MEMORY_UTILIZATION" not in os.environ:
    raise ValueError("显式设置 L58_GPU_MEMORY_UTILIZATION，本脚本不给默认值")
util = float(os.environ["L58_GPU_MEMORY_UTILIZATION"])

import torch
from vllm import LLM, SamplingParams

def emit(kind, **f):
    print(json.dumps({"kind": kind, **f}, ensure_ascii=False), flush=True)

free, total = torch.cuda.mem_get_info()
emit("gpu_before", free_bytes=free, total_bytes=total, budget_fraction=util)

buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setLevel(logging.DEBUG)
logging.getLogger("vllm").addHandler(handler)
logging.getLogger("vllm").setLevel(logging.DEBUG)

config = dict(model=MODEL, gpu_memory_utilization=util, max_model_len=1024,
              max_num_seqs=64, enforce_eager=True, enable_prefix_caching=False,
              disable_log_stats=False, seed=7)
llm = LLM(**config)
# vLLM 只在 stats 里记抢占，不打日志。直接包一层 _preempt_request 数出来。
from vllm.v1.core.sched.scheduler import Scheduler
PREEMPTIONS = []
_orig = Scheduler._preempt_request
def _counted(self, request, timestamp, drop_stale_output=False):
    PREEMPTIONS.append({"req": request.request_id,
                        "num_computed_tokens_before": request.num_computed_tokens,
                        "num_running": len(self.running),
                        "num_waiting": len(self.waiting)})
    return _orig(self, request, timestamp, drop_stale_output)
Scheduler._preempt_request = _counted
try:
    kv = llm.llm_engine.vllm_config.cache_config
    emit("cache_config", num_gpu_blocks=kv.num_gpu_blocks,
         block_size=kv.block_size,
         kv_tokens=(kv.num_gpu_blocks or 0) * kv.block_size)
    prompts = [f"Write a very long, detailed story number {i} about a lighthouse. "
               "Keep going and do not stop." for i in range(args.n)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    text = buf.getvalue()
    preempt_lines = [l for l in text.splitlines()
                     if re.search(r"preempt|Preempt|recompute|swap", l)]
    emit("A_preemption", requests=args.n, max_tokens=args.max_tokens,
         wall_seconds=round(wall, 3),
         output_tokens=sum(len(o.outputs[0].token_ids) for o in outs),
         finish_reasons=sorted({o.outputs[0].finish_reason for o in outs}),
         preempt_log_lines=preempt_lines[:20],
         preempt_line_count=len(preempt_lines),
         preemptions=len(PREEMPTIONS),
         discarded_computed_tokens=sum(p["num_computed_tokens_before"]
                                       for p in PREEMPTIONS),
         preempted_distinct_requests=len({p["req"] for p in PREEMPTIONS}),
         first_preemptions=PREEMPTIONS[:8])
    (args.out / "preemptions.json").write_text(
        json.dumps(PREEMPTIONS, ensure_ascii=False, indent=2))
    (args.out / "engine.log").write_text(text)
    save = {"config": config, "n": args.n, "max_tokens": args.max_tokens}
    (args.out / "config.json").write_text(json.dumps(save, ensure_ascii=False, indent=2))

    # B 取消：直接调 engine core 的 abort，绕过 LLM 的同步封装。
    core = llm.llm_engine.engine_core
    from vllm.inputs import TokensPrompt
    tok = llm.get_tokenizer()
    ids = tok.encode(prompts[0])
    llm.llm_engine.add_request("cancel-me", TokensPrompt(prompt_token_ids=ids), sp)
    for _ in range(6):
        llm.llm_engine.step()
    sched = core.engine_core.scheduler if hasattr(core, "engine_core") else None
    before = {"unfinished": llm.llm_engine.get_num_unfinished_requests(),
              "free_blocks": (sched.kv_cache_manager.block_pool.get_num_free_blocks()
                              if sched else None)}
    llm.llm_engine.abort_request(["cancel-me"])
    llm.llm_engine.step()
    after = {"unfinished": llm.llm_engine.get_num_unfinished_requests(),
             "free_blocks": (sched.kv_cache_manager.block_pool.get_num_free_blocks()
                             if sched else None)}
    emit("B_abort", before=before, after=after,
         has_unfinished=llm.llm_engine.has_unfinished_requests())
except Exception as e:                       # 记录失败本身也是结果
    emit("error", type=type(e).__name__, message=str(e))
    (args.out / "engine.log").write_text(buf.getvalue())
    raise
finally:
    free2, _ = torch.cuda.mem_get_info()
    emit("gpu_after", free_bytes=free2)
    llm.llm_engine.engine_core.shutdown()
