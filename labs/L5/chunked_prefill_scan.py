#!/usr/bin/env python3
"""L5.3 任务 B / L5.1 任务 C · chunked prefill 预算扫描（逐 step + 逐请求事件）。

修订计划要求（5.3-B）：

    长 prompt=8192 插入 8 条 decode，token 预算=128/256/512/2048/8192；
    采每请求 TTFT/TPOT、scheduled tokens 与重算；
    修正固定观察窗口与「按预算估计吞吐」的口径；每档相同工作量并保存重复样本。

所以这里不再数「80 步窗口里发生了什么」，而是：

  * 每档用完全相同的请求清单与生成长度（工作量相同）；
  * 手动 add_request + step，一直跑到所有请求结束（不是固定窗口）；
  * 逐 step 记录：本步 token 数、每请求的 computed 前后与阶段（prefill /
    prefill-chunk / prefill-last / decode）、running/waiting、KV 占用、
    墙钟、device 事件、分配器字节；
  * 逐请求记录：引擎自己的 queued/scheduled/first_token/last_token 时间戳，
    以及从 step 事件恢复的逐 token 间隔（用来量长 prefill 插进来时 decode
    被卡住多久）；
  * 每档重复 3 次，报中位数与极差。

用法（在 crater 上）：
    python chunked_prefill_scan.py --out /scratch/learn/work/out/5.3/budget-<id>
"""

import argparse
import json
import os
import random
import statistics
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L53_MODEL", "Qwen/Qwen3-1.7B")
LONG_LEN = 8192
N_DECODE = 8
DECODE_PROMPT = 64
DECODE_OUT = 64
LONG_OUT = 8
WARM_DECODE_TOKENS = 8          # 让 8 条 decode 先进入稳态再插入长 prompt
REPEATS = 3
BUDGETS = [128, 256, 512, 2048, 8192]


def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    usable = max(free / gib - reserve_gib, 1.0)
    return min(cap, usable / (total / gib))


def _blobs(outputs):
    """EngineCore.step_fn 的返回值在不同调用路径下是 list / dict / 单个对象。"""
    if outputs is None:
        return []
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if hasattr(outputs, "outputs") or hasattr(outputs, "scheduler_stats"):
        return [outputs]
    return list(outputs.values())


def _f(x):
    return float(x) if x is not None else float("nan")


class Mon:
    """逐 step 与逐请求事件的采集器（与 serve_protocol.py 同口径）。"""

    def __init__(self, core):
        self.core = core
        self.sched = core.scheduler
        self.steps = []
        self._sched = None
        self._ev = []
        self.hist = {}
        self.req_stats = {}
        self._original_schedule = None

    def patch(self):
        import vllm.v1.core.sched.scheduler as vsch

        mon = self
        self._original_schedule = vsch.Scheduler.schedule

        def schedule(self_s, *a, **k):
            out = self._original_schedule(self_s, *a, **k)
            mon._sched = (self_s, out)
            return out

        vsch.Scheduler.schedule = schedule

        orig_fn = self.core.step_fn

        def step_fn():
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            mem0 = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            ev0.record()
            outputs, executed = orig_fn()
            ev1.record()
            wall = (time.perf_counter() - t0) * 1000
            mem1 = torch.cuda.memory_allocated()
            mon._record(outputs, executed, wall, ev0, ev1, mem0, mem1)
            return outputs, executed

        self.core.step_fn = step_fn

    def reset(self):
        self.steps = []
        self._sched = None
        self._ev = []
        self.hist = {}
        self.req_stats = {}
        torch.cuda.synchronize()

    def _record(self, outputs, executed, wall_ms, ev0, ev1, mem0, mem1):
        sched, so = self._sched if self._sched else (self.sched, None)
        idx = len(self.steps)
        reqs = {}
        if so is not None:
            for req_id, n_sched in so.num_scheduled_tokens.items():
                req = sched.requests.get(req_id)
                if req is None:
                    continue
                after = req.num_computed_tokens
                before = after - n_sched
                plen = req.num_prompt_tokens
                if after < plen and before == 0:
                    phase = "prefill"
                elif after < plen:
                    phase = "prefill-chunk"
                elif before < plen <= after:
                    phase = "prefill-last"
                else:
                    phase = "decode"
                reqs[req_id] = dict(before=before, after=after, sched=n_sched,
                                    prompt=plen, phase=phase)
                self.hist.setdefault(req_id, []).append((idx, phase, after))

        produced = {}
        stats = {}
        for blob in _blobs(outputs):
            for eco in getattr(blob, "outputs", []) or []:
                if getattr(eco, "new_token_ids", None):
                    produced[eco.request_id] = list(eco.new_token_ids)
            sstats = getattr(blob, "scheduler_stats", None)
            for rs in (getattr(sstats, "req_stats", None) or []):
                stats[rs.request_id] = dict(
                    num_generation_tokens=rs.num_generation_tokens,
                    queued_ts=rs.queued_ts, scheduled_ts=rs.scheduled_ts,
                    first_token_ts=rs.first_token_ts, last_token_ts=rs.last_token_ts,
                )

        n_pref = sum(1 for r in reqs.values() if r["phase"].startswith("prefill"))
        n_dec = sum(1 for r in reqs.values() if r["phase"] == "decode")
        self.steps.append(dict(
            step=idx, wall_ms=wall_ms,
            tokens=int(so.total_num_scheduled_tokens) if so is not None else 0,
            reqs=reqs, produced=produced, n_prefill=n_pref, n_decode=n_dec,
            mixed=bool(n_pref and n_dec),
            running=len(sched.running), waiting=len(sched.waiting),
            kv_usage=sched.kv_cache_manager.usage,
            kv_tokens=sum(r["after"] for r in reqs.values()),
            mem_alloc0=mem0, mem_alloc1=mem1,
        ))
        self._ev.append((ev0, ev1))
        for rid, s in stats.items():
            cur = self.req_stats.setdefault(rid, {})
            for k, v in s.items():
                if v:
                    cur[k] = v

    def device_ms(self):
        torch.cuda.synchronize()
        out = []
        for ev0, ev1 in self._ev:
            try:
                out.append(ev0.elapsed_time(ev1))
            except Exception:                                   # noqa: BLE001
                out.append(float("nan"))
        return out


def make_engine(budget, eager, util=None):
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    # util 每档必须相同：逐档重算会随残留显存漂移，档间就不可比了
    util = util if util is not None else safe_util()
    return LLMEngine.from_engine_args(EngineArgs(
        model=MODEL, gpu_memory_utilization=util, max_model_len=16384,
        enforce_eager=eager, enable_prefix_caching=False,
        disable_log_stats=False, max_num_batched_tokens=budget,
        # 引擎要求 max_num_seqs <= max_num_batched_tokens；本组只有 9 条请求
        max_num_seqs=min(256, budget)))


def shutdown(eng):
    try:
        eng.engine_core.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    del eng
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def snapshot_stats(eng, acc):
    """逐 step 抓引擎前端的逐请求时间戳。

    这些时间戳挂在 `output_processor.request_states[rid].stats` 上，
    请求一结束条目就被删掉，所以必须在跑的过程中抓，不能等结束后再读。
    """
    try:
        states = eng.output_processor.request_states
    except AttributeError:
        return
    for rid, stobj in states.items():
        s = getattr(stobj, "stats", None)
        if s is None:
            continue
        fr = rid.rsplit("-", 1)[0]
        d = acc.setdefault(fr, {})
        for k in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts",
                  "arrival_time", "first_token_latency"):
            v = getattr(s, k, 0) or 0
            if v:
                d[k] = v


def run_once(budget, prompts, eager=False, util=None):
    from vllm import SamplingParams, TokensPrompt

    eng = make_engine(budget, eager, util)
    core = getattr(eng.engine_core, "engine_core", eng.engine_core)
    mon = Mon(core)
    mon.patch()
    mon.reset()
    # 引擎已把 KV 池等一次性分配做完；从这里开始量运行期峰值与逐步增量
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stats_acc = {}

    sp_dec = SamplingParams(max_tokens=DECODE_OUT, temperature=0.0, ignore_eos=True)
    for i in range(N_DECODE):
        eng.add_request(f"d{i}", TokensPrompt(prompt_token_ids=prompts[f"d{i}"]), sp_dec)

    # 先推进到稳态：每条 decode 已经出过 WARM_DECODE_TOKENS 个 token
    emitted = {f"d{i}": 0 for i in range(N_DECODE)}
    warm_steps = 0
    while any(v < WARM_DECODE_TOKENS for v in emitted.values()) and warm_steps < 500:
        eng.step()
        snapshot_stats(eng, stats_acc)
        warm_steps += 1
        for rid, ids in mon.steps[-1]["produced"].items():
            fr = rid.rsplit("-", 1)[0]
            if fr in emitted:
                emitted[fr] += len(ids)
    warm_wall = statistics.median([s["wall_ms"] for s in mon.steps[-10:]])

    # 插入长 prompt，从这一步开始逐 step 记录，一直跑到全部结束
    mark = len(mon.steps)
    eng.add_request("LONG", TokensPrompt(prompt_token_ids=prompts["long"]),
                    SamplingParams(max_tokens=LONG_OUT, temperature=0.0, ignore_eos=True))
    guard = 0
    while eng.has_unfinished_requests() and guard < 20000:
        eng.step()
        snapshot_stats(eng, stats_acc)
        guard += 1
    dev = mon.device_ms()

    # 逐请求：从 step 事件恢复发出 token 的位置与间隔
    first_emit, emit_steps, sched_of = {}, {}, {}
    for rec in mon.steps:
        for rid, r in rec["reqs"].items():
            fr = rid.rsplit("-", 1)[0]
            sched_of.setdefault(fr, []).append((rec["step"], r["phase"], r["sched"]))
        for rid, ids in rec["produced"].items():
            fr = rid.rsplit("-", 1)[0]
            first_emit.setdefault(fr, rec["step"])
            emit_steps.setdefault(fr, []).append(rec["step"])

    reqs = {}
    for fr in list(sched_of):
        steps_of = sorted(emit_steps.get(fr, []))
        gaps = [sum(mon.steps[s]["wall_ms"]
                    for s in range(steps_of[i] + 1, steps_of[i + 1] + 1))
                for i in range(len(steps_of) - 1)]
        seq = sched_of[fr]
        pf = [s for s in seq if s[1].startswith("prefill")]
        st = {**stats_acc.get(fr, {}), **mon.req_stats.get(fr, {})}
        ttft = ((st.get("first_token_ts", 0) - st.get("queued_ts", 0)) * 1000
                if st.get("first_token_ts") and st.get("queued_ts") else None)
        fe = first_emit.get(fr)
        # 引擎时间戳缺失时的回退：从插入点累加到首次产出那一步的墙钟
        ttft_fb = (sum(mon.steps[s]["wall_ms"] for s in range(mark, fe + 1))
                   if fe is not None and fe >= mark else None)
        reqs[fr] = dict(
            scheduled_tokens=sum(x[2] for x in seq),
            prefill_steps=len(pf),
            prefill_chunk_sizes=[x[2] for x in pf],
            prefill_done_step=(pf[-1][0] if pf else None),
            first_emit_step=fe,
            emit_count=len(steps_of),
            tpot_gaps_ms=gaps,
            tpot_median_ms=(statistics.median(gaps) if gaps else None),
            tpot_max_ms=(max(gaps) if gaps else None),
            eng_queued_ts=st.get("queued_ts"), eng_scheduled_ts=st.get("scheduled_ts"),
            eng_first_token_ts=st.get("first_token_ts"),
            eng_last_token_ts=st.get("last_token_ts"),
            eng_ttft_ms=ttft if ttft is not None else ttft_fb,
            eng_ttft_source="engine_ts" if ttft is not None else "step_wall_fallback",
            eng_prefill_ms=((st["first_token_ts"] - st["scheduled_ts"]) * 1000
                            if st.get("first_token_ts") and st.get("scheduled_ts") else None),
            eng_decode_ms=((st["last_token_ts"] - st["first_token_ts"]) * 1000
                           if st.get("last_token_ts") and st.get("first_token_ts") else None),
        )

    after = mon.steps[mark:]
    out = dict(
        budget=budget, enforce_eager=eager, warm_wall_ms=warm_wall,
        warmup_steps=warm_steps,
        steps_after_insert=len(after),
        wall_after_insert_ms=sum(s["wall_ms"] for s in after),
        device_after_insert_ms=sum(x for x in dev[mark:] if x == x),
        total_device_ms=sum(x for x in dev if x == x),
        slowest_step_ms=max((s["wall_ms"] for s in after), default=None),
        peak_waiting=max((s["waiting"] for s in after), default=0),
        peak_running=max((s["running"] for s in after), default=0),
        mixed_steps=sum(1 for s in after if s["mixed"]),
        total_scheduled_tokens=sum(s["tokens"] for s in after),
        mem_base_mib=base_alloc / (1024 ** 2),
        mem_peak_mib=torch.cuda.max_memory_allocated() / (1024 ** 2),
        mem_step_delta_max_mib=max((s["mem_alloc1"] - s["mem_alloc0"]
                                    for s in after), default=0) / (1024 ** 2),
        requests=reqs,
        steps=after,
    )
    import vllm.v1.core.sched.scheduler as vsch
    vsch.Scheduler.schedule = mon._original_schedule      # 还原全局补丁
    shutdown(eng)
    return out


def summarize(reps):
    """把每档的重复样本压成中位数 + 极差。"""
    lines = []
    lines.append(f"模型 {MODEL}；8 条 decode（prompt {DECODE_PROMPT}，生成 {DECODE_OUT}）"
                 f" + 1 条长 prompt {LONG_LEN}（生成 {LONG_OUT}）；每档 {REPEATS} 次重复")
    lines.append("")
    hdr = (f"{'预算':>6}{'长TTFT中位 ms':>15}{'长TTFT极差':>22}"
           f"{'长完成中位 ms':>15}{'长分块数':>9}{'decode TPOT中位 ms':>20}"
           f"{'decode 最大间隔 ms':>20}{'总device ms':>13}{'最慢step ms':>13}"
           f"{'峰等待':>7}{'总调度token':>12}")
    lines.append(hdr)
    table = {}
    for budget, rs in reps.items():
        long_ttft = [r["requests"]["LONG"]["eng_ttft_ms"] for r in rs
                     if r["requests"]["LONG"]["eng_ttft_ms"]]
        long_done = [r["wall_after_insert_ms"] for r in rs]
        chunks = [r["requests"]["LONG"]["prefill_steps"] for r in rs]
        dec_tpot = [q["tpot_median_ms"] for r in rs for k, q in r["requests"].items()
                    if k != "LONG" and q["tpot_median_ms"]]
        dec_max = [q["tpot_max_ms"] for r in rs for k, q in r["requests"].items()
                   if k != "LONG" and q["tpot_max_ms"]]
        dev = [r["device_after_insert_ms"] for r in rs]
        slow = [r["slowest_step_ms"] for r in rs]
        tbl = dict(
            long_ttft_ms=long_ttft, long_wall_ms=long_done, long_chunks=chunks,
            decode_tpot_ms=dec_tpot, decode_max_gap_ms=dec_max,
            device_ms=dev, slowest_step_ms=slow,
            peak_waiting=[r["peak_waiting"] for r in rs],
            total_sched_tokens=[r["total_scheduled_tokens"] for r in rs],
            mixed_steps=[r["mixed_steps"] for r in rs],
        )
        table[budget] = tbl
        spread = (f"[{min(long_ttft):.1f}, {max(long_ttft):.1f}]" if long_ttft else "—")
        lines.append(
            f"{budget:>6}{(statistics.median(long_ttft) if long_ttft else float('nan')):>15.1f}"
            f"{spread:>22}{statistics.median(long_done):>15.1f}"
            f"{statistics.median(chunks):>9.0f}"
            f"{(statistics.median(dec_tpot) if dec_tpot else float('nan')):>20.2f}"
            f"{(max(dec_max) if dec_max else float('nan')):>20.1f}"
            f"{statistics.median(dev):>13.1f}"
            f"{max(slow):>13.1f}{max(tbl['peak_waiting']):>7.0f}"
            f"{statistics.median(tbl['total_sched_tokens']):>12.0f}")
    lines.append("")
    lines.append("长 prompt 的实际分块（每档取一次重复，列出 prefill 各步的调度 token 数）：")
    for budget, rs in reps.items():
        ch = rs[0]["requests"]["LONG"]["prefill_chunk_sizes"]
        lines.append(f"  预算 {budget:>5}：{len(ch)} 步 -> {ch[:12]}"
                     f"{' ...' if len(ch) > 12 else ''}")
    lines.append("")
    lines.append("逐请求明细（第 1 次重复）：")
    for budget, rs in reps.items():
        r = rs[0]
        L = r["requests"]["LONG"]
        lines.append(f"  预算 {budget:>5}  LONG 引擎TTFT {_f(L['eng_ttft_ms']):.1f} ms"
                     f"（来源 {L['eng_ttft_source']}），"
                     f"prefill {L['prefill_steps']} 步，"
                     f"首输出在第 {L['first_emit_step']} 步（prefill 完成于第 "
                     f"{L['prefill_done_step']} 步）")
        ds = [(k, q) for k, q in r["requests"].items() if k != "LONG"]
        med = statistics.median(q["tpot_median_ms"] for _, q in ds)
        mx = max(q["tpot_max_ms"] for _, q in ds)
        lines.append(f"          8 条 decode：TPOT 中位 {med:.2f} ms，"
                     f"最大单次间隔 {mx:.1f} ms（稳态 {r['warm_wall_ms']:.2f} ms/步）")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--util", type=float, default=None,
                    help="整轮固定的 gpu_memory_utilization；不传则按空闲显存算一次")
    ap.add_argument("--single", default=None,
                    help="只跑一档：<budget>:<repeat>，结果写 run_b*_r*.json")
    ap.add_argument("--merge", action="store_true",
                    help="把目录里的 run_b*_r*.json 合并汇总（不启动引擎）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    budgets = [int(x) for x in args.budgets.split(",")]
    rng = random.Random(20260921)
    prompts = {f"d{i}": [rng.randint(1000, 60000) for _ in range(DECODE_PROMPT)]
               for i in range(N_DECODE)}
    prompts["long"] = [rng.randint(1000, 60000) for _ in range(LONG_LEN)]

    torch.cuda.init()
    util = args.util if args.util is not None else safe_util()
    meta = dict(model=MODEL, budgets=budgets, repeats=args.repeats,
                enforce_eager=args.eager, long_len=LONG_LEN,
                n_decode=N_DECODE, decode_prompt=DECODE_PROMPT,
                decode_out=DECODE_OUT, long_out=LONG_OUT,
                gpu_memory_utilization=util,
                torch=torch.__version__, vllm=__import__("vllm").__version__,
                gpu=torch.cuda.get_device_name(0))
    print(json.dumps(meta, indent=2), flush=True)

    # 每个 (预算, 重复) 用一个**新进程**跑：一个进程里反复建/销毁引擎会
    # 留下显存残留，档与档之间就不可比了。--single/--merge 由 run_*.sh 驱动。
    if args.single:
        budget, rep = (int(x) for x in args.single.split(":"))
        r = run_once(budget, prompts, eager=args.eager, util=util)
        r["repeat"] = rep
        with open(os.path.join(args.out, f"run_b{budget}_r{rep}.json"), "w") as f:
            json.dump(dict(meta=meta, run=r), f, indent=1)
        L = r["requests"]["LONG"]
        print(f"  budget {budget:>5} rep {rep}: LONG ttft {_f(L['eng_ttft_ms']):.1f} ms  "
              f"prefill steps {L['prefill_steps']}  wall {r['wall_after_insert_ms']:.1f} "
              f"slowest {r['slowest_step_ms']:.1f}  peak_wait {r['peak_waiting']}",
              flush=True)
        return

    if args.merge:
        import glob
        reps = {}
        for path in sorted(glob.glob(os.path.join(args.out, "run_b*_r*.json"))):
            with open(path) as f:
                blob = json.load(f)
            reps.setdefault(blob["run"]["budget"], []).append(blob["run"])
        for v in reps.values():
            v.sort(key=lambda x: x["repeat"])
        text = summarize(reps)
        with open(os.path.join(args.out, "chunked_budget.json"), "w") as f:
            json.dump(dict(meta=meta, reps=reps), f, indent=1)
        with open(os.path.join(args.out, "chunked_budget.txt"), "w") as f:
            f.write(text + "\n")
        print("\n" + text, flush=True)
        return

    reps = {}
    for budget in budgets:
        reps[budget] = []
        for rep in range(args.repeats):
            r = run_once(budget, prompts, eager=args.eager, util=util)
            r["repeat"] = rep
            reps[budget].append(r)
            L = r["requests"]["LONG"]
            print(f"  budget {budget:>5} rep {rep}: LONG ttft {_f(L['eng_ttft_ms']):.1f} ms  "
                  f"prefill steps {L['prefill_steps']}  wall {r['wall_after_insert_ms']:.1f} "
                  f"slowest {r['slowest_step_ms']:.1f}  peak_wait {r['peak_waiting']}",
                  flush=True)

    text = summarize(reps)
    with open(os.path.join(args.out, "chunked_budget.json"), "w") as f:
        json.dump(dict(meta=meta, reps=reps), f, indent=1)
    with open(os.path.join(args.out, "chunked_budget.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
