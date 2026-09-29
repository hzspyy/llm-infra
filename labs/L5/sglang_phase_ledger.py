#!/usr/bin/env python3
"""L5.1 补测 · SGLang 侧的逐 step 阶段归属。

vLLM 侧用 `serve_protocol.py` 在进程内 rebind `EngineCore.step_fn` 拿到逐 step
事件；SGLang 是另一套结构：`sgl.Engine` 强制 `spawn` 启动 scheduler 子进程，
父进程的 monkeypatch 不会被继承。这里用 spawn 的导入语义把它补上——子进程会
以 `__mp_main__` 重新导入本脚本，于是放在模块顶层（main 保护之外）的 patch
在调度进程里同样生效；记录直接追加到 `SGL_PHASE_TRACE` 指向的 JSONL。

每个 step 记录 `ScheduleBatch` 的成员与三个计数：

  * `extend` = 本步调度的 token 数（`batch.extend_lens`）
  * `before` = 本步之前的长度（`batch.prefix_lens`）
  * `after`  = 本步之后的长度（`batch.seq_lens_cpu`）
  * `prompt` = `len(req.origin_input_ids)`

阶段判定与 vLLM 侧完全同一套判据（`mini_phase_ledger.classify`），因此两引擎
的逐 step 归属可以直接并排比较。radix cache 关闭，与 vLLM 侧口径一致。

用法：
    SGL_PHASE_TRACE=/path/trace.jsonl python sglang_phase_ledger.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SGLANG_DISABLE_OUTLINES_DISK_CACHE", "1")

TRACE = os.environ.get("SGL_PHASE_TRACE")
SYNC = os.environ.get("SGL_PHASE_SYNC") == "1"
MODEL = os.environ.get("L5_MODEL", "Qwen/Qwen3-1.7B")


def _snapshot(batch) -> dict:
    def as_list(x):
        try:
            if hasattr(x, "tolist"):            # tensor → python 标量列表
                return list(x.tolist())
            return list(x)
        except Exception:                                          # noqa: BLE001
            return []

    ex = as_list(getattr(batch, "extend_lens", []))
    sq = as_list(getattr(batch, "seq_lens_cpu", []))
    fm = getattr(batch, "forward_mode", None)
    try:
        is_dec = bool(fm.is_decode())
    except Exception:                                              # noqa: BLE001
        is_dec = False
    mode = getattr(fm, "name", None) or str(fm)
    # 纯 decode 批不重写 extend_lens/prefix_lens（它们只在 prepare_for_extend 里赋值），
    # 所以本步 token 数按 forward mode 取：decode 每个请求 1 个，EXTEND 取 extend_lens。
    reqs = []
    for i, r in enumerate(getattr(batch, "reqs", [])):
        after = sq[i] if i < len(sq) else None
        if is_dec:
            ext = 1
        else:
            ext = ex[i] if i < len(ex) else None
        before = after - ext if (after is not None and ext is not None) else None
        reqs.append(dict(
            rid=str(getattr(r, "rid", i)),
            prompt=len(getattr(r, "origin_input_ids", []) or []),
            extend=ext, before=before, after=after,
        ))
    return dict(ts=time.time(), mode=mode, is_decode=is_dec, n_reqs=len(reqs),
                raw=dict(extend=repr(ex), seq=repr(sq)),
                reqs=reqs)


def install_patch() -> bool:
    """把 run_batch 记录挂到 Scheduler 上；在父进程与 spawn 子进程各执行一次。"""
    if not TRACE:
        return False
    try:
        import torch
        import sglang.srt.managers.scheduler as sm
    except Exception as exc:                                       # noqa: BLE001
        print(f"patch 未安装：{exc}", file=sys.stderr)
        return False
    if getattr(sm.Scheduler, "_l5_phase_probe", False):
        return True
    orig = sm.Scheduler.run_batch
    orig_pbr = sm.Scheduler.process_batch_result

    def _append(rec: dict) -> None:
        rec["pid"] = os.getpid()
        try:
            with open(TRACE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:                                   # noqa: BLE001
            print(f"记录写入失败：{exc}", file=sys.stderr)

    def run_batch(self, batch, pp_proxy_tensors=None):
        if SYNC:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rec = _snapshot(batch)
        rec["forward_iter"] = getattr(batch, "forward_iter", None)
        _append(rec)
        out = orig(self, batch, pp_proxy_tensors)
        _append(dict(ts=time.time(), kind="done", n_reqs=rec["n_reqs"],
                     forward_iter=rec["forward_iter"],
                     wall_ms=(time.perf_counter() - t0) * 1000))
        return out

    def process_batch_result(self, batch, result):
        out = orig_pbr(self, batch, result)
        reqs = []
        for i, r in enumerate(getattr(batch, "reqs", [])):
            try:
                fin = bool(r.finished())
            except Exception:                                      # noqa: BLE001
                fin = None
            reqs.append(dict(rid=str(getattr(r, "rid", i)),
                             n_out=len(getattr(r, "output_ids", []) or []),
                             finished=fin))
        _append(dict(ts=time.time(), kind="result",
                     forward_iter=getattr(batch, "forward_iter", None), reqs=reqs))
        return out

    sm.Scheduler.run_batch = run_batch
    sm.Scheduler.process_batch_result = process_batch_result
    sm.Scheduler._l5_phase_probe = True
    return True


PATCHED = install_patch()


# ------------------------------------------------------------------ 判定
def classify(before: int, after: int, prompt: int) -> str:
    if after < prompt and before == 0:
        return "prefill"
    if after < prompt:
        return "prefill-chunk"
    if before < prompt <= after:
        return "prefill-last"
    return "decode"


def load_steps(path: str) -> list[dict]:
    """把 JSONL 还原成逐 step 列表，并把 process_batch_result 的记录配到对应 step。

    记录顺序是 `step`（run_batch 前）、`done`（run_batch 后）、`result`
    （process_batch_result 之后）；overlap 调度下 `result` 可能滞后到下一步之后，
    但顺序一一对应，按出现次序配对即可。
    """
    steps, results = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "done":
                if steps:
                    steps[-1]["wall_ms"] = rec["wall_ms"]
            elif kind == "result":
                results.append(rec)
            else:
                steps.append(rec)
    for s, r in zip(steps, results):
        s["result"] = r
    return steps


def report(steps, gen_lens, prompt_len, args) -> tuple[str, dict]:
    lines = [f"L5.1 SGLang 逐 step 阶段归属 · {MODEL} · prompt={prompt_len} "
             f"batch={args.batch} 实际生成 {gen_lens}",
             f"调度进程 pid {'/'.join(sorted({str(s['pid']) for s in steps}))}，"
             f"共 {len(steps)} 个 forward"]
    hdr = (f"{'step':>4}{'forward_mode':>12}{'请求数':>7}{'本步token':>9}"
           f"{'阶段':>14}{'before':>8}{'after':>7}{'本步产出':>9}{'首输出':>7}{'墙钟ms':>9}")
    lines.append(hdr)
    per_req: dict[str, int] = {}
    bad_decode = 0
    phase_count: dict[str, int] = {}
    first_emit: dict[str, int] = {}
    for i, rec in enumerate(steps):
        n = sum(r["extend"] or 0 for r in rec["reqs"])
        before = min((r["before"] for r in rec["reqs"]), default=0)
        after = max((r["after"] for r in rec["reqs"]), default=0)
        res = rec.get("result") or {}
        n_out = {x["rid"]: x.get("n_out", 0) for x in res.get("reqs", [])}
        for rid, k in n_out.items():
            if k > 0 and rid not in first_emit:
                first_emit[rid] = i
        produced = sum(n_out.values())
        phases = set()
        for r in rec["reqs"]:
            if r["prompt"] is None or r["before"] is None or r["after"] is None:
                continue
            ph = classify(r["before"], r["after"], r["prompt"])
            phases.add(ph)
            phase_count[ph] = phase_count.get(ph, 0) + 1
            per_req[r["rid"]] = per_req.get(r["rid"], 0) + (r["extend"] or 0)
            if ph == "decode" and r["extend"] != 1:
                bad_decode += 1
        phase = phases.pop() if len(phases) == 1 else "mixed"
        lines.append(f"{i:>4}{rec['mode'][:12]:>12}{rec['n_reqs']:>7}{n:>9}"
                     f"{phase:>14}{before:>8}{after:>7}{produced:>9}"
                     f"{sum(1 for v in first_emit.values() if v == i) or '':>7}"
                     f"{rec.get('wall_ms', float('nan')):>9.3f}")
    lines.append("")
    lines.append(f"阶段计数 {phase_count}")
    lines.append(f"decode 步 extend≠1 的请求数 {bad_decode}")
    lines.append(f"首输出步分布（步号 → 请求数）："
                 f"{dict(sorted(Counter(first_emit.values()).items()))}"
                 f"，共 {len(first_emit)} 条请求")
    deltas = {}
    base = prompt_len + (max(gen_lens) if gen_lens else 0) - 1
    for rid, total in per_req.items():
        d = total - base
        deltas[d] = deltas.get(d, 0) + 1
    lines.append(f"各请求各步 extend 之和 − (prompt+实际生成−1 = {base}) 的分布：{deltas}")
    text = "\n".join(lines)
    blob = dict(model=MODEL, args=vars(args), prompt_len=prompt_len, gen_lens=gen_lens,
                n_steps=len(steps), phase_count=phase_count,
                bad_decode=bad_decode, delta_hist={str(k): v for k, v in deltas.items()},
                first_emit={str(k): v for k, v in first_emit.items()},
                steps=[dict(step=i, mode=s["mode"], n_reqs=s["n_reqs"],
                            sched=sum(r["extend"] or 0 for r in s["reqs"]),
                            wall_ms=s.get("wall_ms"), reqs=s["reqs"],
                            result=s.get("result"))
                       for i, s in enumerate(steps)])
    return text, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not TRACE:
        raise SystemExit("需要 SGL_PHASE_TRACE=<jsonl 路径>")
    open(TRACE, "w").close()
    if not PATCHED:
        raise SystemExit("Scheduler patch 未安装（检查 sglang 环境）")

    import sglang as sgl
    engine = sgl.Engine(model_path=MODEL, tp_size=1,
                        mem_fraction_static=args.mem_fraction,
                        page_size=args.page_size,
                        chunked_prefill_size=args.chunked_prefill_size,
                        disable_radix_cache=True, random_seed=0,
                        log_level="error")
    ids = [1000 + (i * 7) % 50000 for i in range(args.prompt_len)]
    prompts = [ids for _ in range(args.batch)]
    outs = engine.generate(input_ids=prompts,
                           sampling_params=dict(max_new_tokens=args.out_len,
                                                temperature=0.0, ignore_eos=True))
    if isinstance(outs, dict):
        outs = [outs]
    gen_lens = [len(o.get("output_ids") or []) for o in outs]
    time.sleep(1.0)
    steps = load_steps(TRACE)
    text, blob = report(steps, gen_lens, args.prompt_len, args)
    blob["gen_lens"] = gen_lens
    with open(os.path.join(args.out, "sglang_phase_ledger.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "sglang_phase_ledger.json"), "w") as f:
        json.dump(blob, f, indent=1)
    print(text)
    try:
        engine.shutdown()
    except Exception:                                              # noqa: BLE001
        pass
    print(f"\n写入 {args.out}/sglang_phase_ledger.txt")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
