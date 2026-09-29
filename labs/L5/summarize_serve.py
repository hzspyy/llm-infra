#!/usr/bin/env python3
"""把 serve_protocol.py 的原始结果压成可入库的小工件。

原始 JSON 的每个 step 都带完整 per-request 明细，单次扫描 4–17 MB。入库时保留：

  * 每次运行的全部标量与逐请求记录（逐请求时间、阶段、首个输出归属）；
  * 逐 step 标量的**列式**序列（时间、调度 token、运行/排队数、KV）；
  * 只在真正出现混合批的 step（每配置最多 6 个）和前 3 个 step 上保留
    per-request 明细与逐请求产出 token，避免整表膨胀。

用法：
    python summarize_serve.py <raw.json> <out_compact.json>
"""

import json
import statistics
import sys

FIELDS = ("wall_ms", "device_ms", "tokens", "n_prefill", "n_decode", "mixed",
          "running", "waiting", "kv_tokens", "kv_usage")
PER_CFG_DETAIL = 6
HEAD_STEPS = 3


def table(path):
    """按配置汇总：prefill 步耗时、纯 decode 步耗时、TTFT/TPOT 与 KV 峰值。"""
    d = json.load(open(path))
    by = {}
    for r in d["runs"]:
        by.setdefault(r["config"]["tag"], []).append(r)
    print(f"# {path}")
    print(f"{'配置':<20}{'B':>4}{'steps':>7}{'mixed':>7}{'prefill步ms':>12}"
          f"{'decode步ms':>11}{'TTFT ms':>10}{'TPOT ms':>10}{'完成 ms':>10}"
          f"{'KV峰值tok':>11}{'峰值排队':>9}{'重算条':>7}")
    for tag, rs in by.items():
        s0 = [r["step_series"]["device_ms"][0] for r in rs
              if r["step_series"]["device_ms"]]
        dec = []
        for r in rs:
            ser = r["step_series"]
            pure = [ser["device_ms"][i] for i in range(len(ser["device_ms"]))
                    if ser["n_decode"][i] and not ser["n_prefill"][i]]
            dec.append(statistics.median(pure[:20]) if pure else float("nan"))
        ttft = [q["eng_prefill_ms"] for r in rs for q in r["requests"]
                if q.get("eng_prefill_ms")]
        tpot = [q["eng_decode_ms"] / max(1, q["out_tokens"] - 1)
                for r in rs for q in r["requests"] if q.get("eng_decode_ms")]
        full = [q["eng_full_ms"] for r in rs for q in r["requests"]
                if q.get("eng_full_ms")]
        kv = [max(r["step_series"]["kv_tokens"]) for r in rs]
        pw = [max(r["step_series"]["waiting"]) for r in rs]
        rc = [sum(1 for q in r["requests"] if q.get("recomputed")) for r in rs]
        m = statistics.median
        print(f"{tag:<20}{len(rs[0]['config']['lens']):>4}"
              f"{m(r['n_steps'] for r in rs):>7.0f}"
              f"{m(r['mixed_steps'] for r in rs):>7.0f}"
              f"{m(s0):>12.3f}{m(dec):>11.3f}"
              f"{m(ttft):>10.2f}{m(tpot):>10.3f}{m(full):>10.1f}"
              f"{m(kv):>11.0f}{m(pw):>9.0f}{m(rc):>7.0f}")


def main():
    if sys.argv[1] == "--table":
        for p in sys.argv[2:]:
            table(p)
        return
    raw_path, out_path = sys.argv[1], sys.argv[2]
    d = json.load(open(raw_path))
    runs = []
    for r in d["runs"]:
        series = {k: [] for k in FIELDS}
        detail = {}
        budget = 0
        for i, s in enumerate(r["steps"]):
            for k in FIELDS:
                series[k].append(s.get(k))
            if i < HEAD_STEPS or (s["mixed"] and budget < PER_CFG_DETAIL):
                detail[str(i)] = dict(reqs=s["reqs"], produced=s["produced"])
                if s["mixed"] and i >= HEAD_STEPS:
                    budget += 1
        rr = dict((k, v) for k, v in r.items() if k != "steps")
        rr["step_series"] = series
        rr["step_detail"] = detail
        runs.append(rr)
    with open(out_path, "w") as f:
        json.dump(dict(meta=d["meta"], runs=runs), f)
    print(f"{raw_path} -> {out_path}")


if __name__ == "__main__":
    main()