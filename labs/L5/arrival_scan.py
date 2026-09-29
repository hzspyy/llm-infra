#!/usr/bin/env python3
"""L5.3 任务 C · ARRIVAL 协议：0.3/0.6/0.9/1.1 倍容量下的任务级指标。

复用 8.3 的发生器（`labs/L8/load_generator.py`），不另写一套计时：

  1. 闭环基线：固定并发，量出该负载下的可持续 QPS，作为倍率的 1.0 倍；
  2. 开环扫描：泊松到达，0.3/0.6/0.9/1.1 倍基线，每档 **3 个至少 120 秒窗口**；
  3. 汇总：每档报样本数、完成/错误/超时/拒绝、goodput（SLO 内成功请求/秒）、
     TTFT p50/p99 与 p99 的窗口间极差、TPOT 中位、端到端 p99、客户端排队。

协议要求：不足 10000 请求不宣称 p99 稳定。汇总里显式打印每档样本数与合计样本数，
结论只按同口径排序使用。

用法（服务需已在 base-url 上就绪）：
    python arrival_scan.py --engine vllm --base-url http://127.0.0.1:8100 \
        --out /scratch/learn/work/out/5.3/arrival-vllm
    python arrival_scan.py --engine sglang --from-json <dir>/arrival_scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "L8" / "load_generator.py"
PY = sys.executable
MODEL = "Qwen/Qwen3-1.7B"


def run_gen(args: list[str], out_dir: Path, tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(GEN), *args, "--out-dir", str(out_dir), "--tag", tag]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary_path = out_dir / f"summary_{tag}.json"
    if proc.returncode != 0 or not summary_path.exists():
        raise RuntimeError(f"{tag} 失败：\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    with open(summary_path) as f:
        return json.load(f)


def digest(s: dict) -> dict:
    lat = s["latency"]
    ttft_p50 = lat["true_ttft"]["p50"]
    if ttft_p50 is None:
        ttft_p50 = lat["observed_ttft"]["p50"]
    ttft_p99 = lat["true_ttft"]["p99"]
    if ttft_p99 is None:
        ttft_p99 = lat["observed_ttft"]["p99"]
    return dict(
        tag=s["tag"], window_s=s["window_s"], planned=s["planned_requests"],
        counts=s["counts"],
        qps=s["throughput"]["request_goodput_qps"],
        out_tok_qps=s["throughput"]["output_token_qps"],
        slo_2s=s["goodput"]["SLO-2s/100ms"]["ratio"],
        slo_2s_attained=s["goodput"]["SLO-2s/100ms"]["attained"],
        ttft_p50=ttft_p50, ttft_p99=ttft_p99,
        obs_ttft_p99=lat["observed_ttft"]["p99"],
        tpot_p50=lat["tpot"]["p50"],
        e2e_p99=lat["observed_e2e"]["p99"],
        client_queue_p99=s["client_queue"]["p99"],
        arrival_span=s["arrival"]["span_s"],
    )


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ------------------------------------------------------------------ 引擎侧指标
# 客户端只能看到"请求变慢了"；要归因到准入与抢占，必须读服务端自己的计数。
# 名字的第一个元素是首选（按"累计 counter"优先，gauge 只在没有 counter 时使用）：
#   vLLM 0.29.0    vllm:num_preemptions_total（Counter）
#   SGLang 0.5.19  sglang:num_retracted_reqs_total（Counter）——
#                  注意同名不带 _total 的是 Gauge，每次上报后复位，差值恒为 0。
METRIC_NAMES = {
    "vllm": dict(preempt=["vllm:num_preemptions_total", "vllm:num_preemptions"],
                 waiting=["vllm:num_requests_waiting"],
                 running=["vllm:num_requests_running"],
                 kv=["vllm:kv_cache_usage_perc"],
                 hits=["vllm:prefix_cache_hits_total"]),
    "sglang": dict(preempt=["sglang:num_retracted_reqs_total",
                            "sglang:num_retracted_reqs"],
                   waiting=["sglang:num_queue_reqs"],
                   running=["sglang:num_running_reqs"],
                   kv=["sglang:token_usage"],
                   hits=["sglang:cache_hit_rate"]),
}


def fetch_text(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode("utf-8", "replace")


def fetch_metrics(url: str, engine: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=5) as r:
        text = r.read().decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    out: dict = {}
    for field, names in METRIC_NAMES[engine].items():
        for n in names:                    # 名字按首选顺序，第一个命中的胜出
            total, found = 0.0, False
            for line in lines:
                if not line.startswith(n):
                    continue
                rest = line[len(n):]
                if rest[:1] not in (" ", "{"):
                    continue
                try:
                    total += float(line.rsplit(" ", 1)[1])
                    found = True
                except ValueError:
                    pass
            if found:
                out[field] = total
                break
    return out


class MetricPoller(threading.Thread):
    """窗口内每 2 s 抓一次 /metrics，给出增量（counter）与峰值（gauge）。"""

    def __init__(self, url: str, engine: str, interval: float = 2.0):
        super().__init__(daemon=True)
        self.url, self.engine, self.interval = url, engine, interval
        self.samples: list[dict] = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.samples.append(fetch_metrics(self.url, self.engine))
            except Exception:                                      # noqa: BLE001
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=10)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        s0, s1 = self.samples[0], self.samples[-1]

        def delta(k):
            return round(s1[k] - s0[k], 4) if k in s0 and k in s1 else None

        def peak(k):
            vals = [s[k] for s in self.samples if k in s]
            return round(max(vals), 4) if vals else None

        def total(k):
            return round(sum(s.get(k, 0.0) for s in self.samples), 4)

        # vLLM 的抢占是 Counter（取首末差）；SGLang 只导出 Gauge，且每次上报后复位，
        # 所以窗口内要按采样点累加，取首末差会恒为 0。
        preempt = total("preempt") if self.engine == "sglang" else delta("preempt")
        return dict(engine=self.engine, preempt=preempt, preempt_delta=delta("preempt"),
                    preempt_total=total("preempt"), hits_delta=delta("hits"),
                    waiting_peak=peak("waiting"), running_peak=peak("running"),
                    kv_peak=peak("kv"), samples=len(self.samples))


def agg_of(mult: float, rate: float, windows: list[dict]) -> dict:
    return dict(
        mult=mult, rate=rate,
        planned=med([w["planned"] for w in windows]),
        completed=med([w["counts"].get("SUCCESS", 0) for w in windows]),
        errors=med([w["counts"].get("ERROR", 0) for w in windows]),
        timeouts=med([w["counts"].get("TIMEOUT", 0) for w in windows]),
        rejected=med([w["counts"].get("REJECTED", 0) for w in windows]),
        goodput_qps=med([w["qps"] for w in windows]),
        out_tok_qps=med([w["out_tok_qps"] for w in windows]),
        slo_ratio=med([w["slo_2s"] for w in windows]),
        ttft_p50=med([w["ttft_p50"] for w in windows]),
        ttft_p99=med([w["ttft_p99"] for w in windows]),
        ttft_p99_range=[min(w["ttft_p99"] for w in windows),
                        max(w["ttft_p99"] for w in windows)],
        tpot_p50=med([w["tpot_p50"] for w in windows]),
        e2e_p99=med([w["e2e_p99"] for w in windows]),
        client_queue_p99=med([w["client_queue_p99"] for w in windows]),
    )


def render(engine: str, bd: dict, rows: list[dict], meta: dict) -> str:
    lines = [f"ARRIVAL 协议 · 引擎 {engine} · prompt {meta['prompt_len']} / "
             f"输出 {meta['output_len']} · 每档 {meta['windows']} 个 "
             f"{meta['duration']:.0f} s 窗口"]
    lines.append(f"闭环基线：并发 {meta['concurrency']}、{bd['planned']} 条，"
                 f"可持续 {bd['qps']:.2f} QPS（输出 {bd['out_tok_qps']:.0f} tok/s），"
                 f"TTFT p50 {bd['ttft_p50']*1000:.1f} ms，窗口 {bd['window_s']:.1f} s")
    lines.append("")
    lines.append(f"  {'倍率':>5}{'到达 QPS':>9}{'样本':>7}{'完成':>6}{'错误':>5}"
                 f"{'超时':>5}{'拒绝':>5}{'goodput QPS':>12}{'SLO 内':>8}"
                 f"{'TTFT p50 ms':>12}{'TTFT p99 ms':>12}{'p99 极差 ms':>16}"
                 f"{'TPOT p50 ms':>12}{'e2e p99 s':>10}")
    for r in rows:
        a = r["aggregate"]
        spread = (f"[{a['ttft_p99_range'][0]*1000:.0f}, "
                  f"{a['ttft_p99_range'][1]*1000:.0f}]")
        lines.append(
            f"  {a['mult']:>5.1f}{a['rate']:>9.2f}{a['planned']:>7.0f}"
            f"{a['completed']:>6.0f}{a['errors']:>5.0f}{a['timeouts']:>5.0f}"
            f"{a['rejected']:>5.0f}"
            f"{(a['goodput_qps'] or 0):>12.2f}{(a['slo_ratio'] or 0):>8.3f}"
            f"{a['ttft_p50']*1000:>12.1f}{a['ttft_p99']*1000:>12.1f}"
            f"{spread:>16}"
            f"{a['tpot_p50']*1000:>12.2f}{(a['e2e_p99'] or 0):>10.2f}")
    total = sum(r["aggregate"]["planned"] for r in rows)
    per = statistics.mean([r["aggregate"]["planned"] for r in rows])
    lines.append("")
    lines.append(f"全部窗口合计计划请求 {total:.0f} 条，单档平均 {per:.0f} 条 —— "
                 f"不足 10000 条，p99 只作同口径排序，不外推为稳定尾延迟。")

    if any(w.get("engine") for r in rows for w in r["windows"]):
        lines.append("")
        lines.append("引擎侧（窗口内事件累计 / gauge 峰值）：")
        lines.append(f"  {'倍率':>5}{'抢占或 retract':>16}{'等待峰值':>10}"
                     f"{'运行峰值':>10}{'KV 峰值':>12}{'命中增量':>12}")
        for r in rows:
            ws = [w.get("engine") or {} for w in r["windows"]]
            lines.append(
                f"  {r['mult']:>5.1f}"
                f"{str(med([w.get('preempt') for w in ws])):>16}"
                f"{str(med([w.get('waiting_peak') for w in ws])):>10}"
                f"{str(med([w.get('running_peak') for w in ws])):>10}"
                f"{str(med([w.get('kv_peak') for w in ws])):>12}"
                f"{str(med([w.get('hits_delta') for w in ws])):>12}")
        lines.append("  口径：vLLM 的抢占是 Counter，取窗口首末差；SGLang 只导出 Gauge 且"
                     "每次上报后复位，按采样点累加。")
        lines.append("  抢占/retract = 0 表示这批负载下引擎没有触发 KV 回收；"
                     "等待峰值上升而抢占为 0 说明压力停在队列而不是运行集。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-json", default=None,
                    help="只对已有 arrival_scan.json 重新汇总（不压服务）")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--rates", default="0.3,0.6,0.9,1.1")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--baseline-requests", type=int, default=600)
    ap.add_argument("--no-metrics", action="store_true",
                    help="不抓 /metrics（默认抓 base-url 的 /metrics）")
    ap.add_argument("--save-metrics", action="store_true", default=True,
                    help="每窗口保存一份原始 /metrics 文本（默认保存）")
    args = ap.parse_args()

    meta = dict(engine=args.engine, prompt_len=args.prompt_len,
                output_len=args.output_len, windows=args.windows,
                duration=args.duration, concurrency=args.concurrency)

    if args.from_json:
        with open(args.from_json) as f:
            blob = json.load(f)
        rows = blob["rows"]
        for r in rows:
            r["aggregate"] = agg_of(r["mult"], r["rate"], r["windows"])
        text = render(args.engine, blob["baseline"], rows, meta)
        out_txt = os.path.splitext(args.from_json)[0] + ".txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)
        print(f"\n重新汇总写入 {out_txt}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--backend", "openai", "--base-url", args.base_url, "--model", MODEL,
              "--prompt-len", str(args.prompt_len), "--output-len", str(args.output_len)]

    base = run_gen([*common, "--mode", "closed", "--concurrency", str(args.concurrency),
                    "--total-requests", str(args.baseline_requests)],
                   out / "baseline", f"{args.engine}_baseline")
    bd = digest(base)
    capacity = bd["qps"]

    rows = []
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    for mult in (float(x) for x in args.rates.split(",")):
        rate = capacity * mult
        windows = []
        for w in range(args.windows):
            tag = f"{args.engine}_r{mult}_w{w}"
            poller = None if args.no_metrics else MetricPoller(metrics_url, args.engine)
            if poller:
                poller.start()
            try:
                s = run_gen([*common, "--mode", "open", "--arrival", "poisson",
                             "--rate", f"{rate:.4f}", "--duration", str(args.duration),
                             "--seed", str(1000 + w)],
                            out / tag, tag)
            finally:
                if poller:
                    poller.stop()
                if poller and args.save_metrics:
                    try:
                        (out / tag / "metrics_end.txt").write_text(
                            fetch_text(metrics_url), encoding="utf-8")
                    except Exception:                              # noqa: BLE001
                        pass
            d = digest(s)
            if poller:
                d["engine"] = poller.summary()
            windows.append(d)
        rows.append(dict(mult=mult, rate=rate, windows=windows,
                         aggregate=agg_of(mult, rate, windows)))

    text = render(args.engine, bd, rows, meta)
    print(text)
    (out / "arrival_scan.txt").write_text(text + "\n", encoding="utf-8")
    (out / "arrival_scan.json").write_text(
        json.dumps(dict(meta=meta, base_url=args.base_url, baseline=bd, rows=rows),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {out}/arrival_scan.txt 与 arrival_scan.json")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
