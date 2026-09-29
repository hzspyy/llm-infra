#!/usr/bin/env python3
"""labs/L8/request_metrics.py - 请求级事件记录、时延分解与 goodput 统计.

这一份模块被 8.3 以及后续 8.1/8.4/8.6 的服务实验共用。它只做三件事:

1. 定义逐请求事件 schema: 计划到达 -> 实际发送 -> 首字节 -> 首个有效 token ->
   每个 token -> 完成/错误。所有时间戳取同一进程的单调时钟 (time.monotonic)。
2. 计算时延分解: 客户端排队延迟 (协调遗漏的来源)、观测 TTFT 与真实 TTFT、
   TPOT/ITL、观测与真实端到端时延。
3. 在冻结的 SLO 门槛下统计 goodput。分母是**全部计划到达的请求**，拒绝、超时、
   取消、错误和缺失事件一律留在分母里。
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 请求终态。SUCCESS 表示服务端正常送完; 其余都计入失败分母。
STATUS_SUCCESS = "SUCCESS"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_REJECTED = "REJECTED"      # HTTP 429/503 等服务端明确拒绝
STATUS_ERROR = "ERROR"            # 连接错误/5xx/解析失败
STATUS_ABORTED = "ABORTED"        # 客户端主动取消
STATUS_TRUNCATED = "TRUNCATED"    # 连接正常结束但输出少于请求长度

ALL_STATUSES = (
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_REJECTED,
    STATUS_ERROR,
    STATUS_ABORTED,
    STATUS_TRUNCATED,
)


@dataclasses.dataclass
class RequestRecord:
    """单个请求的完整生命周期。所有时间为绝对单调时钟秒数。"""

    request_id: str
    prompt_len: int
    requested_output_len: int
    planned_arrival_s: float
    actual_send_s: Optional[float] = None
    first_byte_s: Optional[float] = None
    first_token_s: Optional[float] = None
    token_timestamps_s: List[float] = dataclasses.field(default_factory=list)
    finish_s: Optional[float] = None
    status: str = STATUS_SUCCESS
    error: Optional[str] = None
    server_reported_prompt_tokens: Optional[int] = None
    server_reported_output_tokens: Optional[int] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # ---- 事件完整性: 用于检查"缺失事件" -----------------------------------
    @property
    def missing_events(self) -> List[str]:
        gaps: List[str] = []
        if self.actual_send_s is None:
            gaps.append("actual_send")
        if self.status == STATUS_SUCCESS:
            if self.first_byte_s is None:
                gaps.append("first_byte")
            if self.first_token_s is None:
                gaps.append("first_token")
            if self.finish_s is None:
                gaps.append("finish")
            if self.metadata.get("done_missing"):
                gaps.append("done_marker")
        return gaps

    # ---- 时延分解 ---------------------------------------------------------
    @property
    def client_queue_delay_s(self) -> Optional[float]:
        """计划到达与实际发送之差; 开环下这是客户端/网络背压的直接证据。"""
        if self.actual_send_s is None:
            return None
        return max(0.0, self.actual_send_s - self.planned_arrival_s)

    @property
    def observed_ttft_s(self) -> Optional[float]:
        """常规口径 TTFT: 从实际发送到首个有效 token。"""
        if self.first_token_s is None or self.actual_send_s is None:
            return None
        return max(0.0, self.first_token_s - self.actual_send_s)

    @property
    def true_ttft_s(self) -> Optional[float]:
        """修正协调遗漏后的 TTFT: 从计划到达算起。"""
        if self.first_token_s is None:
            return None
        return max(0.0, self.first_token_s - self.planned_arrival_s)

    @property
    def observed_e2e_s(self) -> Optional[float]:
        if self.finish_s is None or self.actual_send_s is None:
            return None
        return max(0.0, self.finish_s - self.actual_send_s)

    @property
    def true_e2e_s(self) -> Optional[float]:
        if self.finish_s is None:
            return None
        return max(0.0, self.finish_s - self.planned_arrival_s)

    @property
    def num_chunks(self) -> int:
        """SSE 数据块的个数。一个块可能携带多个 token, 所以它不是 token 数。"""
        return len(self.token_timestamps_s)

    @property
    def num_tokens(self) -> int:
        """输出 token 数。优先用服务端 usage 的报告值; 没有 usage 时退回块数。

        真实引擎在高负载下会把多个 token 合并进一个 SSE 块, 用块数当 token 数会
        把完整的响错误判成截断, 所以 usage 是唯一可靠的 token 计数。
        """
        if self.server_reported_output_tokens is not None:
            return self.server_reported_output_tokens
        return len(self.token_timestamps_s)

    @property
    def tpot_s(self) -> Optional[float]:
        """首 token 之后的均摊出 token 间隔 (以 usage 的 token 数为分母)。"""
        n = self.num_tokens
        if n <= 1 or self.first_token_s is None or self.finish_s is None:
            return None
        return max(0.0, (self.finish_s - self.first_token_s) / (n - 1))

    @property
    def inter_chunk_latencies_s(self) -> List[float]:
        """相邻 SSE 块之间的间隔。它是客户端真正能观测到的流式粒度, 不是逐 token。"""
        ts = self.token_timestamps_s
        return [max(0.0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RequestRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def write_jsonl(path: str, records: Iterable[RequestRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_jsonl() + "\n")


def read_jsonl(path: str) -> List[RequestRecord]:
    out: List[RequestRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RequestRecord.from_dict(json.loads(line)))
    return out


# --------------------------------------------------------------------------
# 分位数: 用 order statistic 的最近秩定义, 而不是线性插值。压测的尾分位数
# 常被插值美化, 这里固定 rank = ceil(p/100 * n) - 1, 并在样本不足时给出标记。
# --------------------------------------------------------------------------
def percentile(values: Sequence[Optional[float]], p: float) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    if n == 1:
        return vals[0]
    rank = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
    return vals[rank]


def summarize(values: Sequence[Optional[float]], pcts: Sequence[float] = (50, 90, 95, 99)) -> Dict[str, Any]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    res: Dict[str, Any] = {"n": len(vals)}
    if not vals:
        for p in pcts:
            res[f"p{p}"] = None
        res["insufficient_tail"] = True
        return res
    for p in pcts:
        res[f"p{p}"] = percentile(vals, p)
    res["mean"] = statistics.fmean(vals)
    res["insufficient_tail"] = len(vals) < 100
    return res


def correct_coordinated_omission(
    values: Sequence[Optional[float]],
    expected_interval_s: float,
) -> List[float]:
    """闭环压测的协调遗漏修正 (Gil Tene 的 expected-interval 口径)。

    闭环客户端在一个请求完成前不会发下一个请求。若某请求耗时 L, 那么在它占用
    客户端的这段时间里, 本应按 `expected_interval_s` 的节奏发出但被压住的请求
    floor(L / expected_interval) - 1 个, 每个都至少等了一个 expected_interval。
    这里把它们补回样本, 使尾分位数不再只由"发得出去的那些请求"决定。

    expected_interval 取系统在未过载时的可持续服务间隔 (1 / 可持续 QPS)。
    """
    if expected_interval_s <= 0:
        raise ValueError("expected_interval_s must be positive")
    out: List[float] = []
    for v in values:
        if v is None or math.isnan(v):
            continue
        out.append(v)
        extra = int(v / expected_interval_s) - 1
        out.extend([expected_interval_s] * max(0, extra))
    return out


@dataclasses.dataclass
class SLOCriteria:
    """在比较任何策略之前冻结的 SLO 门槛。"""

    name: str
    max_true_ttft_s: float
    max_tpot_s: float
    max_true_e2e_s: float
    min_output_tokens: int = 1
    require_full_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# 8.3-C 冻结的三档教学 SLO 曲线。
TEACHING_SLOS: Dict[str, SLOCriteria] = {
    "SLO-0.5s/25ms": SLOCriteria("SLO-0.5s/25ms", 0.5, 0.025, 5.0, require_full_output=True),
    "SLO-1s/50ms": SLOCriteria("SLO-1s/50ms", 1.0, 0.050, 8.0, require_full_output=True),
    "SLO-2s/100ms": SLOCriteria("SLO-2s/100ms", 2.0, 0.100, 20.0, require_full_output=True),
}


def attained(r: RequestRecord, slo: SLOCriteria) -> bool:
    """一条请求是否满足 SLO。失败或缺事件一律不达标。"""
    if r.status != STATUS_SUCCESS or r.missing_events:
        return False
    if r.true_ttft_s is None or r.true_ttft_s > slo.max_true_ttft_s:
        return False
    if r.num_tokens < slo.min_output_tokens:
        return False
    if slo.require_full_output and r.num_tokens < r.requested_output_len:
        return False
    if r.tpot_s is not None and r.tpot_s > slo.max_tpot_s:
        return False
    if r.true_e2e_s is None or r.true_e2e_s > slo.max_true_e2e_s:
        return False
    return True


@dataclasses.dataclass
class RunSummary:
    tag: str
    window_s: float
    planned_requests: int
    counts: Dict[str, int]
    goodput: Dict[str, Any]
    latency: Dict[str, Any]
    throughput: Dict[str, Any]
    event_gaps: Dict[str, int]
    client_queue: Dict[str, Any]
    arrival: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate(
    records: Sequence[RequestRecord],
    tag: str,
    window_s: float,
    slos: Optional[Dict[str, SLOCriteria]] = None,
    reserved_events: Optional[Sequence[float]] = None,
) -> RunSummary:
    """对一次运行做完整统计。window_s 是计划测量窗口, 用于吞吐分母。

    reserved_events: 计划到达时间清单。传入时以计划到达数作为分母, 这样"没发出去的
    请求"也会被计入 (协调遗漏的极端情形)。不传时用记录条数。
    """
    slos = slos or TEACHING_SLOS
    counts: Dict[str, int] = {s: 0 for s in ALL_STATUSES}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1

    planned = len(reserved_events) if reserved_events is not None else len(records)
    # 计划了但没有任何记录 (客户端整体卡死) 的请求: 单列, 也进分母。
    unrecorded = max(0, planned - len(records))

    goodput: Dict[str, Any] = {}
    for name, slo in slos.items():
        ok = sum(1 for r in records if attained(r, slo))
        goodput[name] = {
            "attained": ok,
            "planned": planned,
            "ratio": ok / planned if planned else None,
            "effective_qps": ok / window_s if window_s > 0 else None,
            "unrecorded": unrecorded,
        }

    latency = {
        "observed_ttft": summarize([r.observed_ttft_s for r in records]),
        "true_ttft": summarize([r.true_ttft_s for r in records]),
        "tpot": summarize([r.tpot_s for r in records]),
        "itl": summarize([x for r in records for x in r.inter_chunk_latencies_s]),
        "observed_e2e": summarize([r.observed_e2e_s for r in records]),
        "true_e2e": summarize([r.true_e2e_s for r in records]),
    }

    out_tokens = sum(r.num_tokens for r in records)
    successes = counts[STATUS_SUCCESS]
    throughput = {
        "success_requests": successes,
        "planned_requests": planned,
        "request_goodput_qps": successes / window_s if window_s > 0 else None,
        "output_tokens": out_tokens,
        "output_token_qps": out_tokens / window_s if window_s > 0 else None,
    }

    gap_counter: Dict[str, int] = {}
    for r in records:
        for g in r.missing_events:
            gap_counter[g] = gap_counter.get(g, 0) + 1

    arrivals = [r.actual_send_s for r in records if r.actual_send_s is not None]
    arrival = {
        "first_send_s": min(arrivals) if arrivals else None,
        "last_send_s": max(arrivals) if arrivals else None,
        "span_s": (max(arrivals) - min(arrivals)) if len(arrivals) > 1 else 0.0,
    }

    return RunSummary(
        tag=tag,
        window_s=window_s,
        planned_requests=planned,
        counts=counts,
        goodput=goodput,
        latency=latency,
        throughput=throughput,
        event_gaps=gap_counter,
        client_queue=summarize([r.client_queue_delay_s for r in records]),
        arrival=arrival,
    )


if __name__ == "__main__":
    # 自检: 一条被客户端卡住 1s 的请求, 观测 TTFT 与真实 TTFT 应差 1s。
    rec = RequestRecord(
        request_id="toy-0",
        prompt_len=128,
        requested_output_len=4,
        planned_arrival_s=100.0,
        actual_send_s=101.0,
        first_byte_s=101.02,
        first_token_s=101.05,
        token_timestamps_s=[101.05, 101.07, 101.09, 101.11],
        finish_s=101.11,
    )
    assert abs(rec.observed_ttft_s - 0.05) < 1e-9, rec.observed_ttft_s
    assert abs(rec.true_ttft_s - 1.05) < 1e-9, rec.true_ttft_s
    assert abs(rec.client_queue_delay_s - 1.0) < 1e-9
    s = evaluate([rec], "selftest", window_s=10.0)
    assert s.goodput["SLO-2s/100ms"]["attained"] == 1, s.goodput
    assert s.goodput["SLO-0.5s/25ms"]["attained"] == 0, s.goodput
    print("request_metrics self-test passed")
