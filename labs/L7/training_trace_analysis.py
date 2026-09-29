#!/usr/bin/env python3
"""
labs/L7/training_trace_analysis.py
显式事件时间线分析与合成预算示例。

针对分布式训练步执行时间线进行深度剖析：
1. 分解训练步关键路径分段：
   - 数据加载排队气泡 (Data Loading Bubble)
   - 前向矩阵乘与注意力 (Forward GEMM & Attention)
   - 激活重算开销 (Activation Recompute)
   - 反向梯度计算 (Backward GEMM)
   - 集合通信等待 (NCCL AllGather / ReduceScatter / Barrier)
   - 优化器更新与参数裁切 (Optimizer Step & Grad Clip)
   - 异步快照暂存 (In-Memory Checkpoint Staging)
2. 计算 MFU (Model FLOPs Utilization) 与 HFU (Hardware FLOPs Utilization):
   - MFU = (6 * Params * Tokens_per_step) / (Peak_TFLOPs * Step_Time)
   - 合成示例的执行 FLOP 占比来自 6P/8P 近似，不是硬件计数器 HFU。
3. 自动化瓶颈模式识别与优化诊断：
   - 通信受限 (Communication-Bound / Network Straggler)
   - 供数饥饿 (Input-Bound / Disk I/O Throttling)
   - 慢节点倾斜 (Straggler Node Skew)
"""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RankTraceTimeline:
    rank_id: int
    data_loading_ms: float
    forward_gemm_ms: float
    forward_attn_ms: float
    recompute_ms: float
    backward_gemm_ms: float
    backward_attn_ms: float
    comm_wait_ms: float
    optimizer_ms: float
    ckpt_staging_ms: float

    @property
    def total_step_time_ms(self) -> float:
        return (
            self.data_loading_ms
            + self.forward_gemm_ms
            + self.forward_attn_ms
            + self.recompute_ms
            + self.backward_gemm_ms
            + self.backward_attn_ms
            + self.comm_wait_ms
            + self.optimizer_ms
            + self.ckpt_staging_ms
        )


@dataclass
class ModelWorkloadSpec:
    model_name: str
    num_params: int
    tokens_per_step: int
    has_activation_recompute: bool
    recompute_ratio: float = 1.0  # 1.0 表示全量重算 Transformer Block 前向
    gpu_peak_tflops_bf16: float = 989.0  # NVIDIA H100 SXM5 峰值 BF16 Dense 算力 (无张量核稀疏)
    num_gpus: int = 64


class TrainingTraceAnalyzer:
    """训练性能剖析器"""

    def __init__(self, workload: ModelWorkloadSpec, rank_traces: List[RankTraceTimeline]):
        self.workload = workload
        self.rank_traces = rank_traces

    def compute_metrics(self) -> Dict[str, Any]:
        # 各卡总耗时
        step_times = [t.total_step_time_ms for t in self.rank_traces]
        max_step_time_ms = max(step_times)
        min_step_time_ms = min(step_times)
        avg_step_time_ms = sum(step_times) / len(step_times)

        # 慢卡倾斜度 (Straggler Skew)
        straggler_skew_pct = (max_step_time_ms - min_step_time_ms) / min_step_time_ms * 100.0

        # FLOPs 计算
        # 标准无重算模型前反向 FLOPs = 6 * P * tokens
        p = self.workload.num_params
        tokens = self.workload.tokens_per_step
        model_flops_per_step = 6.0 * p * tokens

        # 近似执行 FLOPs 推演（不是硬件计数器）：全量重算按额外 2P 估算
        if self.workload.has_activation_recompute:
            hardware_flops_per_step = (6.0 + 2.0 * self.workload.recompute_ratio) * p * tokens
        else:
            hardware_flops_per_step = model_flops_per_step

        # 集群理论算力供给 (FLOPs / ms)
        # peak_tflops * 10^12 FLOPs/s / 1000 ms/s = peak_tflops * 10^9 FLOPs/ms
        cluster_peak_flops_per_ms = self.workload.num_gpus * (self.workload.gpu_peak_tflops_bf16 * 1e9)
        cluster_peak_flops_per_step = cluster_peak_flops_per_ms * max_step_time_ms

        mfu = (model_flops_per_step / cluster_peak_flops_per_step) * 100.0
        hfu = (hardware_flops_per_step / cluster_peak_flops_per_step) * 100.0

        # 各阶段耗时占比 (以最慢 Rank 为关键路径基准)
        slowest_rank = max(self.rank_traces, key=lambda r: r.total_step_time_ms)
        breakdown_pct = {
            "data_loading": round((slowest_rank.data_loading_ms / max_step_time_ms) * 100.0, 2),
            "forward_gemm": round((slowest_rank.forward_gemm_ms / max_step_time_ms) * 100.0, 2),
            "forward_attn": round((slowest_rank.forward_attn_ms / max_step_time_ms) * 100.0, 2),
            "recompute": round((slowest_rank.recompute_ms / max_step_time_ms) * 100.0, 2),
            "backward_gemm": round((slowest_rank.backward_gemm_ms / max_step_time_ms) * 100.0, 2),
            "backward_attn": round((slowest_rank.backward_attn_ms / max_step_time_ms) * 100.0, 2),
            "comm_wait": round((slowest_rank.comm_wait_ms / max_step_time_ms) * 100.0, 2),
            "optimizer": round((slowest_rank.optimizer_ms / max_step_time_ms) * 100.0, 2),
            "ckpt_staging": round((slowest_rank.ckpt_staging_ms / max_step_time_ms) * 100.0, 2),
        }

        # 瓶颈诊断
        diagnoses = []
        if breakdown_pct["comm_wait"] > 25.0:
            diagnoses.append({
                "severity": "CRITICAL",
                "bottleneck": "COMMUNICATION_BOUND",
                "message": f"通信阻塞等待占比高达 {breakdown_pct['comm_wait']}%，可能存在跨交换机带宽不足、AllGather 通信与计算未重叠或通信拓扑未对齐",
            })
        if breakdown_pct["data_loading"] > 10.0:
            diagnoses.append({
                "severity": "WARNING",
                "bottleneck": "INPUT_BOUND",
                "message": f"数据加载排队气泡占 {breakdown_pct['data_loading']}%，需增加 DataLoader workers 或启用共享内存与 pin_memory",
            })
        if straggler_skew_pct > 15.0:
            diagnoses.append({
                "severity": "WARNING",
                "bottleneck": "STRAGGLER_SKEW",
                "message": f"卡间时延极差达 {straggler_skew_pct:.1f}%，存在慢卡拖累整体同步步频",
            })

        return {
            "workload": asdict(self.workload),
            "timing_summary_ms": {
                "max_step_time": round(max_step_time_ms, 2),
                "min_step_time": round(min_step_time_ms, 2),
                "avg_step_time": round(avg_step_time_ms, 2),
                "straggler_skew_pct": round(straggler_skew_pct, 2),
            },
            "utilization_pct": {
                "mfu": round(mfu, 2),
                "hfu": round(hfu, 2),
                "recompute_overhead_ratio": round(hfu / mfu if mfu > 0 else 1.0, 3),
            },
            "breakdown_pct": breakdown_pct,
            "diagnoses": diagnoses,
        }


def generate_scenario_traces() -> Dict[str, Tuple[ModelWorkloadSpec, List[RankTraceTimeline]]]:
    workload_3b = ModelWorkloadSpec(
        model_name="synthetic-dense-3B-budget",
        num_params=3_000_000_000,
        tokens_per_step=2_097_152,  # 512 * 4096
        has_activation_recompute=True,
        recompute_ratio=1.0,
        gpu_peak_tflops_bf16=989.0,
        num_gpus=64,
    )

    # 场景 1：高效稳态 (High Efficiency FSDP2)
    # 单步约 1250 ms，MFU 约 47.7%，计算密集，通信完美重叠，低气泡
    traces_steady = []
    for r in range(4):  # 采样 4 张代表性卡
        traces_steady.append(
            RankTraceTimeline(
                rank_id=r,
                data_loading_ms=10.0,
                forward_gemm_ms=320.0,
                forward_attn_ms=110.0,
                recompute_ms=210.0,
                backward_gemm_ms=450.0,
                backward_attn_ms=120.0,
                comm_wait_ms=15.0,  # 极低通信等待 (通信与反向充分重叠)
                optimizer_ms=45.0,
                ckpt_staging_ms=5.0,
            )
        )

    # 场景 2：通信受限与慢卡拖累 (Comm Bound + Straggler)
    # 发生跨节点网络拥塞与网卡抖动，导致通信等待剧增至 850 ms，单步拉长至 2125 ms
    traces_comm_bound = [
        RankTraceTimeline(0, 12.0, 320.0, 110.0, 210.0, 450.0, 120.0, 750.0, 45.0, 5.0),
        RankTraceTimeline(1, 10.0, 320.0, 110.0, 210.0, 450.0, 120.0, 740.0, 45.0, 5.0),
        RankTraceTimeline(2, 11.0, 320.0, 110.0, 210.0, 450.0, 120.0, 760.0, 45.0, 5.0),
        RankTraceTimeline(3, 85.0, 320.0, 110.0, 210.0, 450.0, 120.0, 920.0, 45.0, 5.0),  # 慢卡
    ]

    # 场景 3：供数受限 (Input-bound)
    # 数据加载排队气泡达 480 ms，单步拉长至 1755 ms
    traces_input_bound = [
        RankTraceTimeline(r, 480.0, 320.0, 110.0, 210.0, 450.0, 120.0, 25.0, 45.0, 5.0)
        for r in range(4)
    ]

    return {
        "healthy_steady_state": (workload_3b, traces_steady),
        "comm_bound_straggler": (workload_3b, traces_comm_bound),
        "input_bound_dataloader": (workload_3b, traces_input_bound),
    }


def interval_union(intervals):
    intervals=sorted(intervals)
    total=0.;right=None
    for start,end in intervals:
        if end<start:raise ValueError('event ends before it starts')
        if right is None or start>=right:
            total+=end-start
            right=end
        elif end>right:
            total+=end-right
            right=end
    return total


def clip_intervals(intervals, window):
    start, end = window
    out = []
    for a, b in intervals:
        a2, b2 = max(a, start), min(b, end)
        if b2 > a2:
            out.append((a2, b2))
    return out


def analyze_events(payload):
    """按明确的共同 step 窗口分析事件；没有窗口时不给 MFU 分母。

    每个 rank 的局部跨度只作诊断：同步训练的全局关键路径由共同 step 边界
    （或显式的跨 rank 观测窗口）决定，最长局部跨度会系统性低估它。
    """
    if payload.get('source_kind') not in ('measured','measured_instrumented','author_log','synthetic_fixture'):
        raise ValueError('event input requires an explicit measured/author source kind')
    events=payload['events']
    if not events:raise ValueError('empty event list')
    if any(not math.isfinite(e[k]) for e in events for k in ('start_ms','end_ms')):
        raise ValueError('event times must be finite')

    ranks={}
    for rank in sorted({e['rank'] for e in events}):
        own=[e for e in events if e['rank']==rank]
        span=max(e['end_ms'] for e in own)-min(e['start_ms'] for e in own)
        busy=interval_union([(e['start_ms'],e['end_ms']) for e in own])
        categories={key:interval_union([(e['start_ms'],e['end_ms']) for e in own if e['category']==key])
                    for key in sorted({e['category'] for e in own})}
        ranks[str(rank)]={'wall_span_ms':span,'union_covered_ms':busy,'unattributed_ms':span-busy,
                          'category_union_ms':categories,'category_sums_may_overlap':True}

    max_local_span=max(r['wall_span_ms'] for r in ranks.values())
    observed_start=min(e['start_ms'] for e in events)
    observed_end=max(e['end_ms'] for e in events)
    observed_span=observed_end-observed_start
    covered_all=interval_union([(e['start_ms'],e['end_ms']) for e in events])

    window=payload.get('step_window')
    window_ms=None
    window_union=None
    window_unattributed=None
    if window is not None:
        start,end=window['start_ms'],window['end_ms']
        if not (math.isfinite(start) and math.isfinite(end)) or end<=start:
            raise ValueError('step_window must be a finite increasing interval')
        window_ms=end-start
        if window_ms < max_local_span:
            raise ValueError('provided step_window is shorter than a rank-local activity span')
        outside=[e for e in events if e['start_ms']<start or e['end_ms']>end]
        if outside:
            raise ValueError('events fall outside the declared common step window')
        window_union=interval_union(clip_intervals([(e['start_ms'],e['end_ms']) for e in events],(start,end)))
        window_unattributed=window_ms-window_union

    flops=payload.get('model_flops_per_step_estimate')
    peak=payload.get('peak_tflops_per_device')
    devices=payload.get('num_devices')
    inputs_present=all(x is not None for x in (flops,peak,devices))
    if inputs_present:
        if flops < 0 or peak <= 0 or devices <= 0:
            raise ValueError('invalid FLOP, peak or device count')

    def mfu_for(denominator_ms):
        if not inputs_present or denominator_ms is None or denominator_ms <= 0:
            return None
        return 100.0*flops/(peak*1e12*devices*denominator_ms/1000.0)

    mfu=mfu_for(window_ms)
    if window_ms is None:
        basis='no common step window in the input: MFU denominator unavailable'
    elif not inputs_present:
        basis='common step window present but model FLOP/peak/device inputs incomplete'
    else:
        basis='model FLOP estimate over the declared common step window; not a hardware counter'
    return {'source_kind':payload['source_kind'],'source':payload.get('source'), 'ranks':ranks,
            'step_window_ms':window_ms,'step_window_clock':(window or {}).get('clock'),
            'window_union_covered_ms':window_union,'window_unattributed_ms':window_unattributed,
            'observed_span_ms':observed_span,'covered_all_ranks_ms':covered_all,
            'observed_unattributed_ms':observed_span-covered_all,
            'max_rank_local_span_ms':max_local_span,
            'max_local_span_is_not_global_window':True,
            'mfu_estimate_pct':mfu,'mfu_basis':basis,
            'mfu_wrong_basis_pct':mfu_for(max_local_span) if window_ms is not None else None,
            'hardware_flops_measured':False,'quality_measured':False,
            'note':payload.get('note')}


def cross_rank_window_fixture():
    """两 rank 局部活动错位：局部跨度低估共同窗口。

    rank0 在 [100,200] 忙，rank1 在 [190,290] 忙；同步训练里这个 step 从
    rank0 开始到 rank1 结束，窗口是 [100,290]=190 ms，而最长局部跨度只有
    100 ms。第二段在窗口内留出 rank1 到达前的 50 ms 空档（[200,250]），
    用来分别验证"跨 rank 观测跨度"和"活动区间并集"。
    """
    events=[
        {'rank':0,'category':'forward_backward','start_ms':100.0,'end_ms':200.0},
        {'rank':1,'category':'forward_backward','start_ms':190.0,'end_ms':240.0},
        {'rank':1,'category':'comm','start_ms':250.0,'end_ms':290.0},
    ]
    return {'source_kind':'synthetic_fixture',
            'source':'cross-rank window fixture: rank-local spans overlap but do not share boundaries',
            'events':events,
            'step_window':{'start_ms':100.0,'end_ms':290.0,'clock':'coordinator step boundary',
                           'note':'the step ends when the last rank finishes its comm segment'},
            'model_flops_per_step_estimate':6.0*1_000_000_000*4096,
            'peak_tflops_per_device':989.0,
            'num_devices':2,
            'note':'fixture only: durations are constructed to expose denominator mistakes'}


def write_json(outdir, filename, result, cases):
    with (outdir / filename).open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with (outdir / 'cases.json').open('x') as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Read explicit event traces, a cross-rank window fixture, or an explicitly synthetic budget example')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path)
    group.add_argument('--synthetic-demo',action='store_true')
    group.add_argument('--cross-rank-demo',action='store_true')
    parser.add_argument('--outdir',required=True,type=Path)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    if args.input:
        payload=json.loads(args.input.read_text())
        result=analyze_events(payload)
        cases={'source_kind':result['source_kind'],'source_file':str(args.input),
               'common_step_window_provided':result['step_window_ms'] is not None,
               'GPU_execution_by_this_analyzer':False,'hardware_counters':False}
    elif args.cross_rank_demo:
        payload=cross_rank_window_fixture()
        result=analyze_events(payload)
        cases={'source_kind':result['source_kind'],
               'fixture':'rank-local spans 100 ms vs common window 190 ms',
               'GPU_execution_by_this_analyzer':False,'hardware_counters':False}
        with (args.outdir / 'cross_rank_fixture.json').open('x') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    else:
        result={'source_kind':'synthetic_budget','scenarios':{}}
        for name,(workload,traces) in generate_scenario_traces().items():
            result['scenarios'][name]={'assumptions':asdict(workload),
                'simulated_rank_count':len(traces),'raw_synthetic_rank_timings':[asdict(t) for t in traces],
                'metrics':TrainingTraceAnalyzer(workload,traces).compute_metrics()}
        result['scope']='All durations and the 64-device deployment are assumptions, not H100 measurements'
        cases={'source_kind':'synthetic_budget','GPU_execution_by_this_analyzer':False,'hardware_counters':False}
    write_json(args.outdir,'training_trace_metrics.json',result,cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
