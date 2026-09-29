#!/usr/bin/env python3
"""L5 —— 公共 SERVE 协议的带插桩实现（5.1 / 5.3 / 5.12 复用）。

协议（见 docs/plans/eventual-painting-peacock.md 的「测量与实验设计」表）：

  同一模型/token 输入，固定输出 128；
  先 S=2048 扫 B=1/2/4/8/16/32/64，再 B=1/8 扫 S=128/2048/8192；
  预热 2 次，5 轮交错运行。

本脚本把「逐请求时间」与「逐 step 事件」分开采集，并显式关联
「首个输出 token 由哪一次 forward 产生」：

  * 逐 step：墙钟、device 事件时间、本步新增 token、每请求的
    computed-before/after 与阶段（prefill / prefill-last / prefill-chunk / decode）、
    正在跑/在排队的请求数、KV 占用。
  * 逐请求：引擎自己的 queued/scheduled/first_token/last_token 时间戳
    （`disable_log_stats` 必须为 False，否则这些事件不会记录），
    以及从 step 事件恢复的逐 token 间隔。
  * 阶段判定不依赖 `generate` 的两次调用差值：题目要求的
    「首个输出由哪次 forward 产生」是从 step 输出来归属的，
    并附带一致性断言（该步的 computed-after 必须已经覆盖整段 prompt）。

用法：
    python serve_protocol.py --scan b|s|mix --out <dir> [--rounds 5]
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MB = 1024 * 1024
MODEL = os.environ.get("L5_MODEL", "Qwen/Qwen3-1.7B")
OUT_LEN = 128
WARMUP = 2
ROUNDS = 5


# ----------------------------------------------------------------- 插桩
class StepMonitor:
    """记录每个引擎 step 的调度内容、时间与输出归属。"""

    def __init__(self):
        self.steps = []
        self._sched = None
        self._ev = []          # (ev0, ev1)
        self._patched = False
        self.hist = {}         # 内部 request id -> [(step, phase, after)]

    def patch(self, llm):
        """挂上调度器与引擎 step 两个钩子。

        `EngineCore.__init__` 会把 `self.step` 绑成实例属性 `step_fn`，
        类级别的替换对已构造的引擎无效，所以这里直接rebind 实例的 `step_fn`。
        """
        import vllm.v1.core.sched.scheduler as vsch

        mon = self
        orig_schedule = vsch.Scheduler.schedule

        def schedule(self_s, throttle_prefills=False):
            out = orig_schedule(self_s, throttle_prefills)
            mon._sched = (self_s, out)
            return out

        vsch.Scheduler.schedule = schedule

        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", client)
        orig_fn = core.step_fn

        def step_fn():
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            ev0.record()
            outputs, executed = orig_fn()
            ev1.record()
            wall = (time.perf_counter() - t0) * 1000
            mon._record(outputs, executed, wall, ev0, ev1)
            return outputs, executed

        core.step_fn = step_fn
        self.core = core
        self._patched = True

    def reset(self):
        self.steps = []
        self._sched = None
        self._ev = []
        self.hist = {}
        torch.cuda.synchronize()

    def _record(self, outputs, executed, wall_ms, ev0, ev1):
        sched, so = self._sched if self._sched else (None, None)
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
        outs = outputs or {}
        for eco in outs.values():
            if eco is None:
                continue
            for o in getattr(eco, "outputs", []) or []:
                if getattr(o, "new_token_ids", None):
                    produced[o.request_id] = list(o.new_token_ids)

        n_pref = sum(1 for r in reqs.values()
                     if r["phase"].startswith("prefill"))
        n_dec = sum(1 for r in reqs.values() if r["phase"] == "decode")

        rec = dict(
            step=idx,
            wall_ms=wall_ms,
            tokens=int(so.total_num_scheduled_tokens) if so is not None else 0,
            reqs=reqs,
            produced=produced,
            n_prefill=n_pref,
            n_decode=n_dec,
            mixed=bool(n_pref and n_dec),
            running=len(sched.running) if sched is not None else None,
            waiting=(len(sched.waiting) + len(sched.skipped_waiting)
                     if sched is not None else None),
            kv_usage=(sched.kv_cache_manager.usage
                      if sched is not None else None),
            kv_tokens=sum(r["after"] for r in reqs.values()),
        )
        self.steps.append(rec)
        self._ev.append((ev0, ev1))

    def device_ms(self):
        torch.cuda.synchronize()
        out = []
        for ev0, ev1 in self._ev:
            try:
                out.append(ev0.elapsed_time(ev1))
            except Exception:                                  # noqa: BLE001
                out.append(float("nan"))
        return out


# ----------------------------------------------------------------- 工具
def safe_util(reserve_gib=4.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    usable = max(free / gib - reserve_gib, 1.0)
    return min(cap, usable / (total / gib))


def model_facts():
    import glob
    hub = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
    snaps = sorted(glob.glob(f"{hub}/models--{MODEL.replace('/', '--')}/snapshots/*"))
    cfg = json.load(open(snaps[-1] + "/config.json"))
    H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", H // nq)
    V = cfg["vocab_size"]
    params = L * (H * nq * hd + 2 * H * nkv * hd + nq * hd * H + 3 * H * I) + V * H
    return dict(path=snaps[-1], hidden=H, inter=I, layers=L, heads=nq,
                kv_heads=nkv, head_dim=hd, vocab=V, params=params,
                weight_bytes=params * 2, kv_bytes_per_token=2 * L * nkv * hd * 2)


def make_llm(facts):
    from vllm import LLM
    return LLM(
        model=MODEL,
        gpu_memory_utilization=safe_util(),
        max_model_len=16384,
        disable_log_stats=False,        # 逐请求 times 必须靠它才会被记录
        enable_prefix_caching=False,    # prefix cache 显式关闭
        enforce_eager=False,            # 保留引擎默认的图执行
    )


def shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                          # noqa: BLE001
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def prompt_pool(rng, length, count):
    return [[rng.randint(1000, 60000) for _ in range(length)] for _ in range(count)]


def configs_for(scan):
    if scan == "b":
        return [dict(tag=f"S2048_B{B}", lens=[2048] * B) for B in (1, 2, 4, 8, 16, 32, 64)]
    if scan == "s":
        return [dict(tag=f"S{S}_B{B}", lens=[S] * B) for B in (1, 8)
                for S in (128, 2048, 8192)]
    if scan == "mix":
        return [
            dict(tag="equal_8x1024", lens=[1024] * 8),
            dict(tag="mixed_6144_2x1024", lens=[6144, 1024, 1024]),
            dict(tag="equal_8x2048", lens=[2048] * 8),
            dict(tag="mixed_8192_4x2048", lens=[8192, 2048, 2048, 2048, 2048]),
        ]
    raise SystemExit(f"unknown scan {scan}")


# ----------------------------------------------------------------- 单次运行
def run_once(llm, mon, facts, cfg, rng, prompt_cache):
    from vllm import SamplingParams, TokensPrompt

    prompts = []
    for i, ln in enumerate(cfg["lens"]):
        pool = prompt_cache.setdefault(ln, prompt_pool(rng, ln, 64))
        prompts.append(TokensPrompt(prompt_token_ids=pool[i % len(pool)]))
    sp = SamplingParams(max_tokens=OUT_LEN, temperature=0.0, ignore_eos=True)

    mon.reset()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    dev = mon.device_ms()

    steps = []
    for rec in mon.steps:
        d = dict(rec)
        d["device_ms"] = dev[rec["step"]] if rec["step"] < len(dev) else None
        steps.append(d)

    # 引擎内部 request id = 前端 id + "-" + uuid（input_processor.py:279），
    # 逐 step 事件用内部 id，RequestOutput 用前端 id，这里做归并。
    def front(rid):
        return rid.rsplit("-", 1)[0]

    # 首个输出归属：前端 request_id -> step（被发出的那一步）
    first_emit = {}
    token_steps = {}
    for rec in steps:
        for rid, ids in rec["produced"].items():
            fr = front(rid)
            first_emit.setdefault(fr, rec["step"])
            token_steps.setdefault(fr, []).extend([rec["step"]] * len(ids))

    # 该请求的 prefill 段落：首次跑完整段 prompt 的 step、prefill 段数（>1 说明被抢占重算）
    hist_front = {}
    for rid, seq in mon.hist.items():
        hist_front.setdefault(front(rid), []).extend(seq)
    for v in hist_front.values():
        v.sort()

    def prefill_info(rid):
        seq = hist_front.get(rid, [])
        pf = [s for s in seq if s[1].startswith("prefill")]
        if not pf:
            return dict(prefill_done_step=None, prefill_done_phase=None,
                        prefill_steps=0, prefill_episodes=0, recomputed=False)
        done = next((s for s in pf if s[2] >= _prompt_of(rid)), None)
        eps = 0
        prev = None
        for s in seq:
            if s[1].startswith("prefill"):
                if prev is None or not prev[1].startswith("prefill"):
                    eps += 1
            prev = s
        return dict(prefill_done_step=done[0] if done else None,
                    prefill_done_phase=done[1] if done else None,
                    prefill_steps=len(pf), prefill_episodes=eps, recomputed=eps > 1)

    prompt_of = {rid: ln for rid, ln in zip([o.request_id for o in outs], cfg["lens"])}

    def _prompt_of(rid):
        return prompt_of.get(rid, 0)

    reqs = []
    for o, ln in zip(outs, cfg["lens"]):
        rid = o.request_id
        m = o.metrics
        pi = prefill_info(rid)
        st = sorted(token_steps.get(rid, []))
        gaps = [steps[st[i + 1]]["wall_ms"] for i in range(len(st) - 1)]
        fs = first_emit.get(rid)
        fstep = steps[fs] if fs is not None and fs < len(steps) else None
        ds = pi["prefill_done_step"]
        reqs.append(dict(
            request_id=rid,
            prompt_tokens=ln,
            out_tokens=len(o.outputs[0].token_ids),
            out_chars=len(o.outputs[0].text),
            prefill_done_step=ds,
            prefill_done_phase=pi["prefill_done_phase"],
            prefill_steps=pi["prefill_steps"],
            prefill_episodes=pi["prefill_episodes"],
            recomputed=pi["recomputed"],
            first_emit_step=fs,
            first_emit_device_ms=fstep["device_ms"] if fstep else None,
            step_lag=((fs - ds) if ds is not None and fs is not None else None),
            emitted_steps=len(st),
            tpot_wall_ms=(statistics.median(gaps) if gaps else None),
            tpot_wall_p90=(sorted(gaps)[int(len(gaps) * 0.9) - 1]
                           if len(gaps) >= 10 else None),
            eng_queued_ms=_ms(m, "scheduled_ts", "queued_ts"),
            eng_prefill_ms=_ms(m, "first_token_ts", "scheduled_ts"),
            eng_decode_ms=_ms(m, "last_token_ts", "first_token_ts"),
            eng_inference_ms=_ms(m, "last_token_ts", "scheduled_ts"),
            eng_full_ms=_ms(m, "last_token_ts", "queued_ts"),
            eng_first_token_latency_ms=(m.first_token_latency * 1000
                                        if m and m.first_token_latency else None),
        ))

    return dict(
        config=cfg,
        wall_ms=wall,
        device_ms=sum(x for x in dev if x and x == x),
        n_steps=len(steps),
        total_sched_tokens=sum(s["tokens"] for s in steps),
        mixed_steps=sum(1 for s in steps if s["mixed"]),
        peak_running=max((s["running"] or 0) for s in steps) if steps else 0,
        peak_waiting=max((s["waiting"] or 0) for s in steps) if steps else 0,
        peak_kv_usage=max((s["kv_usage"] or 0) for s in steps) if steps else 0,
        requests=reqs,
        steps=steps,
        ledger=dict(
            weight_bytes_per_step=facts["weight_bytes"],
            kv_bytes_per_token=facts["kv_bytes_per_token"],
            kv_peak_tokens=max((s["kv_tokens"] for s in steps), default=0),
            kv_peak_bytes=(max((s["kv_tokens"] for s in steps), default=0)
                           * facts["kv_bytes_per_token"]),
            max_seq_len=max(max((r["after"] for r in s["reqs"].values()), default=0)
                            for s in steps) if steps else 0,
        ),
    )


def _ms(m, a, b):
    if m is None:
        return None
    va, vb = getattr(m, a, 0.0), getattr(m, b, 0.0)
    if not va or not vb:
        return None
    return (va - vb) * 1000


# ----------------------------------------------------------------- 汇总
def summarize(runs, facts):
    lines = []
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(r["config"]["tag"], []).append(r)

    lines.append(f"model {MODEL}  params {facts['params'] / 1e9:.2f}B  "
                 f"weights {facts['weight_bytes'] / MB / 1024:.2f} GiB  "
                 f"KV {facts['kv_bytes_per_token']} B/token")
    lines.append("")
    lines.append(f"{'配置':<18}{'prompt tok':>11}{'B':>4}{'steps':>7}{'mixed':>7}"
                 f"{'排队 ms':>9}{'prefill ms':>11}{'TPOT ms':>9}"
                 f"{'完成 ms':>10}{'墙钟 ms':>10}{'输出 tok/s':>11}"
                 f"{'峰值run':>8}{'峰值排队':>9}{'重算条':>7}")
    for tag, rs in by_cfg.items():
        queued = [r["eng_queued_ms"] for x in rs for r in x["requests"]
                  if r.get("eng_queued_ms") is not None]
        ttft = [r["eng_prefill_ms"] for x in rs for r in x["requests"]
                if r.get("eng_prefill_ms")]
        tpot = [r["eng_decode_ms"] / max(1, r["out_tokens"] - 1)
                for x in rs for r in x["requests"] if r.get("eng_decode_ms")]
        full = [r["eng_full_ms"] for x in rs for r in x["requests"]
                if r.get("eng_full_ms")]
        wall = statistics.median(x["wall_ms"] for x in rs)
        dev = statistics.median(x["device_ms"] for x in rs)
        steps = statistics.median(x["n_steps"] for x in rs)
        mixed = statistics.median(x["mixed_steps"] for x in rs)
        pw = statistics.median(x["peak_waiting"] for x in rs)
        pr = statistics.median(x["peak_running"] for x in rs)
        rc = sum(1 for x in rs for q in x["requests"] if q.get("recomputed")) / len(rs)
        out_tokens = len(rs[0]["config"]["lens"]) * OUT_LEN
        lines.append(
            f"{tag:<18}{sum(rs[0]['config']['lens']):>11}"
            f"{len(rs[0]['config']['lens']):>4}{steps:>7.0f}{mixed:>7.0f}"
            f"{(statistics.median(queued) if queued else float('nan')):>9.2f}"
            f"{(statistics.median(ttft) if ttft else float('nan')):>11.2f}"
            f"{(statistics.median(tpot) if tpot else float('nan')):>9.3f}"
            f"{(statistics.median(full) if full else float('nan')):>10.2f}"
            f"{wall:>10.1f}{out_tokens / (wall / 1000):>11.1f}"
            f"{pr:>7.0f}{pw:>8.0f}{rc:>7.0f}")
    lines.append("")

    # 首输出归属的一致性核对
    lags, bad = [], []
    n_recomp = 0
    for r in runs:
        for req in r["requests"]:
            ph = req.get("prefill_done_phase")
            if req.get("recomputed"):
                n_recomp += 1
            if req["first_emit_step"] is None:
                bad.append((r["config"]["tag"], req["request_id"], "no-emit-step"))
            elif ph not in ("prefill", "prefill-last"):
                bad.append((r["config"]["tag"], req["request_id"], f"phase={ph}"))
            elif req["step_lag"] is not None:
                lags.append(req["step_lag"])
    from collections import Counter
    lines.append(f"首输出归属核对：{len(runs)} 次运行 / "
                 f"{sum(len(r['requests']) for r in runs)} 条请求，异常 {len(bad)} 条；")
    lines.append(f"  首次覆盖整段 prompt 的那次 forward 与被发出的 step 相差 "
                 f"{dict(Counter(lags))}（step）")
    lines.append(f"  其间发生 prefill 重算（prefill 段数 > 1）的请求：{n_recomp} 条")
    for b in bad[:10]:
        lines.append(f"    {b}")
    lines.append("")

    # mixed step 的组成
    lines.append("含混合批的 step（同一步里既有 prefill 又有 decode）：")
    any_mixed = False
    for tag, rs in by_cfg.items():
        tot = sum(x["mixed_steps"] for x in rs)
        if not tot:
            continue
        any_mixed = True
        r = next(x for x in rs if x["mixed_steps"])
        ex = next(s for s in r["steps"] if s["mixed"])
        comp = ", ".join(
            f"{k.split('-')[-1]}:{v['phase']}x{v['sched']}"
            for k, v in list(ex["reqs"].items())[:8])
        lines.append(f"  {tag:<18} mixed step 数 中位 {tot / len(rs):.0f}，"
                     f"示例 step{ex['step']} tokens={ex['tokens']} "
                     f"running={ex['running']} waiting={ex['waiting']}：{comp}")
    if not any_mixed:
        lines.append("  （本扫描未出现混合批）")
    return "\n".join(lines)


def walk(llm, mon, facts, prompt_len=64, steps=4):
    """打印一个请求从 prefill 到前几步 decode 的逐 step 形状与读写账。

    Q/K/V 形状由 config 与该 step 实际调度的 token 数推出（每层相同）；
    历史长度来自 step 事件里的 computed-after。
    """
    from vllm import SamplingParams, TokensPrompt

    ids = [1000 + (i * 7) % 50000 for i in range(prompt_len)]
    mon.reset()
    llm.generate([TokensPrompt(prompt_token_ids=ids)],
                 SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True),
                 use_tqdm=False)
    dev = mon.device_ms()

    H, hd, nq, nkv, L = (facts["hidden"], facts["head_dim"], facts["heads"],
                         facts["kv_heads"], facts["layers"])
    rows = []
    for rec in mon.steps:
        for rid, r in rec["reqs"].items():
            n = r["sched"]
            hist = r["after"]
            q = f"[1,{nq},{n},{hd}]"
            kv = f"[1,{nkv},{hist},{hd}]"
            flops = 2 * facts["params"] * n
            wbytes = facts["weight_bytes"]
            kvi = hist * facts["kv_bytes_per_token"]
            inten = flops / (wbytes + kvi)
            rows.append((rec["step"], r["phase"], r["before"], r["after"], n, q, kv,
                         wbytes, kvi, n * facts["kv_bytes_per_token"], flops,
                         inten, rec["wall_ms"], dev[rec["step"]]))
    print(f"\n逐 step 形状与读写账（prompt={prompt_len} token，层数 {L}，"
          f"每层 Q/K/V 形状相同）")
    hdr = (f"{'step':>4}{'阶段':>14}{'computed前':>10}{'computed后':>10}"
           f"{'本步token':>9}{'Q(每层)':>18}{'K/V读取(每层)':>20}"
           f"{'权重读 B':>10}{'KV读 B':>10}{'KV写 B':>9}{'FLOP':>12}"
           f"{'强度':>8}{'墙钟ms':>9}{'device ms':>10}")
    print(hdr)
    for (st, ph, b, a, n, q, kv, wb, kvi, kvw, fl, it, w, d) in rows:
        print(f"{st:>4}{ph:>14}{b:>10}{a:>10}{n:>9}{q:>18}{kv:>20}"
              f"{wb:>10}{kvi:>10}{kvw:>9}{fl:>12.3e}{it:>8.2f}{w:>9.3f}{d:>10.3f}")
    out = os.path.join(os.environ.get("WALK_OUT", "."), "walk.txt")
    with open(out, "w") as f:
        f.write(hdr + "\n")
        for row in rows:
            f.write(" ".join(str(x) for x in row) + "\n")
    print(f"\n  写入 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="b", choices=["b", "s", "mix"])
    ap.add_argument("--out", default=None,
                    help="输出目录；与 --from-json 二选一")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--walk", action="store_true",
                    help="只跑一个请求并打印逐 step 形状/读写账")
    ap.add_argument("--walk-len", type=int, default=64)
    ap.add_argument("--walk-steps", type=int, default=4)
    ap.add_argument("--configs", default=None,
                    help="逗号分隔的配置 tag 子集，便于补测单个配置")
    ap.add_argument("--from-json", default=None,
                    help="只对已有 serve_*.json 重新汇总（不启动引擎）")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            blob = json.load(f)
        text = summarize(blob["runs"], blob["meta"]["facts"])
        out_txt = os.path.splitext(args.from_json)[0] + ".txt"
        with open(out_txt, "w") as f:
            f.write(text + "\n")
        print(text)
        print(f"\n重新汇总写入 {out_txt}")
        return

    if not args.out:
        raise SystemExit("需要 --out <dir>（或 --from-json <json>）")
    os.makedirs(args.out, exist_ok=True)
    facts = model_facts()
    cfgs = configs_for(args.scan)
    if args.configs:
        want = set(args.configs.split(","))
        cfgs = [c for c in cfgs if c["tag"] in want]

    mon = StepMonitor()
    print(f"torch {torch.__version__}  model {MODEL}", flush=True)
    llm = make_llm(facts)
    mon.patch(llm)

    if args.walk:
        os.environ["WALK_OUT"] = args.out
        walk(llm, mon, facts, args.walk_len, args.walk_steps)
        shutdown(llm)
        sys.stdout.flush()
        os._exit(0)

    sc = llm.llm_engine.vllm_config.scheduler_config
    meta = dict(model=MODEL, facts={k: v for k, v in facts.items()},
                scan=args.scan, max_num_batched_tokens=sc.max_num_batched_tokens,
                max_num_seqs=sc.max_num_seqs, out_len=OUT_LEN,
                warmup=args.warmup, rounds=args.rounds,
                enable_prefix_caching=False, enforce_eager=False,
                torch=torch.__version__,
                vllm=__import__("vllm").__version__,
                gpu=torch.cuda.get_device_name(0))
    print(json.dumps(meta, indent=2), flush=True)

    rng = random.Random(0)
    prompt_cache = {}
    runs = []
    for cfg in cfgs:
        for _ in range(args.warmup):
            run_once(llm, mon, facts, cfg, rng, prompt_cache)
    for rd in range(args.rounds):
        for cfg in cfgs:
            r = run_once(llm, mon, facts, cfg, rng, prompt_cache)
            r["round"] = rd
            runs.append(r)
            print(f"  round {rd} {cfg['tag']:<20} wall {r['wall_ms']:.1f} ms  "
                  f"steps {r['n_steps']}  mixed {r['mixed_steps']}", flush=True)

    text = summarize(runs, facts)
    with open(os.path.join(args.out, f"serve_{args.scan}.json"), "w") as f:
        json.dump(dict(meta=meta, runs=runs), f, indent=1)
    with open(os.path.join(args.out, f"serve_{args.scan}.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)
    shutdown(llm)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()