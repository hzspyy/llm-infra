#!/usr/bin/env python3
"""labs/L8/tenant_scheduler.py - 8.8-A/C 的多租户调度与状态归属模拟 (CPU, 确定性).

真实引擎能给出"这批请求的时延", 但很难把每一块 KV 的归属、每一次抢占与取消的引用
计数逐条打印出来。这一份 lab 把多租户服务里必须显式化的四件事做成可检查的状态机:

1. **身份**: 每个请求带 tenant / session / adapter revision; 缓存键由
   (tenant, adapter_revision, prefix 哈希) 组成。`--ignore-revision` 把 adapter
   revision 从键里去掉, 用来复现"同名不同 revision 命中别人的 KV"。
2. **预算**: 每个租户有 token 预算、并发上限与 adapter 槽位上限; 准入按策略决定
   接收或拒绝, 拒绝原因逐条记录, 不静默丢弃。
3. **抢占**: KV 块不足时按优先级抢占低优先级租户的在跑请求; 被抢占请求的块立即
   归还, 引用计数归零, 记一次 preempt, 不计入完成。
4. **持久状态**: 请求带幂等键; 完成后把结果与幂等键写成同一条提交记录。重启后重放
   同一批请求, 已提交的按记录返回, 不重复执行、不重复计费。

退出前运行审计: 驻留块 + 空闲块 = 容量 (块账闭合); 每块驻留 KV 只有一个所有者;
没有引用泄漏; 幂等重放不产生第二份结果。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BLOCK_TOKENS = 16


@dataclasses.dataclass(frozen=True)
class Tenant:
    tenant_id: str
    priority: int              # 越小越高
    token_budget: int          # 一个窗口内可用的 prefill token 预算
    max_concurrency: int
    max_adapters: int


@dataclasses.dataclass
class Request:
    req_id: str
    idem_key: str
    tenant: str
    session: str
    adapter_revision: str
    prefix: Tuple[int, ...]
    output_tokens: int
    arrival_s: float
    kind: str = "interactive"
    status: str = "pending"       # pending|done|rejected|preempted|failed
    reason: Optional[str] = None
    started_s: Optional[float] = None
    finished_s: Optional[float] = None
    blocks: int = 0
    cached_tokens: int = 0
    compute_tokens: int = 0
    end_s: Optional[float] = None      # 预计完成时刻 (并发调度用)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prefix)

    @property
    def latency_s(self) -> Optional[float]:
        if self.started_s is None or self.finished_s is None:
            return None
        return self.finished_s - self.started_s


def prefix_hash(prefix: Tuple[int, ...], limit: int = 64) -> str:
    h = hashlib.sha256()
    for t in prefix[:limit]:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()[:16]


class TenantScheduler:
    def __init__(self, policy: str, kv_blocks: int, ignore_revision: bool = False,
                 ignore_tenant: bool = False, preempt: bool = True,
                 time_per_block_s: float = 0.01, service_rate: float = 4.0,
                 max_inflight: int = 32):
        self.policy = policy                     # shared | quota | priority
        self.kv_capacity = kv_blocks
        self.ignore_revision = ignore_revision
        self.ignore_tenant = ignore_tenant
        self.preempt_enabled = preempt
        self.time_per_block_s = time_per_block_s
        self.service_rate = service_rate
        self.max_inflight = max_inflight         # 引擎侧全局并发上限
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.tenants: Dict[str, Tenant] = {}
        self.used_tokens: Dict[str, int] = defaultdict(int)
        self.running: Dict[str, List[Request]] = defaultdict(list)
        self.adapters_loaded: Dict[str, set] = defaultdict(set)
        self.inflight: List[Tuple[float, Request]] = []
        self.events: List[Dict[str, Any]] = []
        self.committed: Dict[str, Dict[str, Any]] = {}
        self.now = 0.0
        self.stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"admitted": 0, "rejected": 0, "preempted": 0, "done": 0,
                     "cache_hit_tokens": 0, "compute_tokens": 0, "idem_replay": 0,
                     "adapter_loads": 0})

    # ---- 块账 ------------------------------------------------------------
    def resident_blocks(self) -> int:
        return sum(e["blocks"] for e in self.cache.values())

    def capacity_left(self) -> int:
        return self.kv_capacity - self.resident_blocks()

    # ---- 缓存身份 --------------------------------------------------------
    def cache_key(self, r: Request) -> str:
        parts = []
        if not self.ignore_tenant:
            parts.append(r.tenant)
        if not self.ignore_revision:
            parts.append(r.adapter_revision)
        parts.append(prefix_hash(r.prefix))
        return "|".join(parts)

    def lookup(self, r: Request) -> Tuple[int, Optional[Dict[str, Any]]]:
        entry = self.cache.get(self.cache_key(r))
        if entry is None:
            return 0, None
        n = min(len(r.prefix) // BLOCK_TOKENS * BLOCK_TOKENS, entry["prefix_len"])
        return n, entry

    # ---- 准入 ------------------------------------------------------------
    def admit(self, r: Request) -> Tuple[bool, Optional[str]]:
        t = self.tenants[r.tenant]
        if sum(len(v) for v in self.running.values()) >= self.max_inflight:
            return False, "global_concurrency"
        if self.policy == "shared":
            return True, None
        if len(self.running[r.tenant]) >= t.max_concurrency:
            return False, "concurrency"
        if self.used_tokens[r.tenant] + r.prompt_tokens > t.token_budget:
            return False, "token_budget"
        return True, None

    def ensure_adapter(self, r: Request) -> None:
        t = self.tenants[r.tenant]
        loaded = self.adapters_loaded[r.tenant]
        if r.adapter_revision in loaded:
            return
        if len(loaded) >= t.max_adapters:
            victim = sorted(loaded)[0]
            loaded.discard(victim)
            self.events.append({"t": round(self.now, 4), "event": "adapter_swap_out",
                                "tenant": r.tenant, "revision": victim})
        loaded.add(r.adapter_revision)
        self.stats[r.tenant]["adapter_loads"] += 1
        self.events.append({"t": round(self.now, 4), "event": "adapter_load",
                            "tenant": r.tenant, "revision": r.adapter_revision})

    # ---- 执行 ------------------------------------------------------------
    def run(self, r: Request) -> None:
        """把一个请求放进在跑集合; 它按代价在未来的某个时刻完成。"""
        cached, entry = self.lookup(r)
        r.cached_tokens = cached
        r.compute_tokens = r.prompt_tokens - cached
        blocks_needed = max(1, (r.prompt_tokens + r.output_tokens + BLOCK_TOKENS - 1)
                            // BLOCK_TOKENS)
        if entry is not None:
            blocks_needed = max(blocks_needed, entry["blocks"])
        if blocks_needed > self.capacity_left():
            if self.preempt_enabled:
                self.preempt_for(blocks_needed, r)
            if blocks_needed > self.capacity_left():
                r.status = "rejected"
                r.reason = "kv_capacity"
                self.stats[r.tenant]["rejected"] += 1
                self.events.append({"t": round(self.now, 4), "event": "reject",
                                    "req": r.req_id, "tenant": r.tenant,
                                    "reason": "kv_capacity"})
                return
        key = self.cache_key(r)
        entry = self.cache.get(key)
        if entry is None:
            entry = {"owner": (r.tenant, r.adapter_revision),
                     "prefix_len": r.prompt_tokens, "refs": 0,
                     "blocks": blocks_needed, "tenants_seen": {r.tenant},
                     "revisions_seen": {r.adapter_revision}}
            self.cache[key] = entry
        entry["tenants_seen"].add(r.tenant)
        entry["revisions_seen"].add(r.adapter_revision)
        entry["refs"] += 1
        r.blocks = blocks_needed
        r.started_s = self.now
        self.running[r.tenant].append(r)
        self.used_tokens[r.tenant] += r.compute_tokens
        self.stats[r.tenant]["admitted"] += 1
        self.stats[r.tenant]["cache_hit_tokens"] += cached
        self.stats[r.tenant]["compute_tokens"] += r.compute_tokens
        self.events.append({"t": round(self.now, 4), "event": "admit", "req": r.req_id,
                            "tenant": r.tenant, "cached": cached,
                            "compute": r.compute_tokens, "blocks": blocks_needed})
        cost = (r.compute_tokens / BLOCK_TOKENS * self.time_per_block_s
                + r.output_tokens / self.service_rate)
        r.end_s = self.now + max(0.005, cost)
        self.inflight.append((r.end_s, r))

    def advance_to(self, t: float) -> None:
        """把在 t 时刻之前应当完成的请求按结束时间结算。"""
        self.inflight.sort(key=lambda x: x[0])
        while self.inflight and self.inflight[0][0] <= t:
            end_s, r = self.inflight.pop(0)
            self.now = max(self.now, end_s)
            self.finish(r, end_s=end_s)
        self.now = max(self.now, t)

    def _release_entry(self, r: Request, evict: bool) -> None:
        key = self.cache_key(r)
        entry = self.cache.get(key)
        if entry is None:
            return
        entry["refs"] = max(0, entry["refs"] - 1)
        if evict and entry["refs"] == 0:
            del self.cache[key]

    def preempt_for(self, blocks_needed: int, new_req: Request) -> int:
        """按优先级抢占: 只抢优先级更低 (数值更大) 的租户; 被抢的块立即释放。"""
        victims: List[Request] = []
        for tid, reqs in self.running.items():
            if self.tenants[tid].priority <= self.tenants[new_req.tenant].priority:
                continue
            victims.extend(reqs)
        victims.sort(key=lambda x: (-self.tenants[x.tenant].priority, x.started_s or 0))
        freed = 0
        for v in victims:
            if freed >= blocks_needed:
                break
            v.status = "preempted"
            v.reason = "kv_preempted"
            v.finished_s = self.now
            freed += v.blocks
            self.running[v.tenant].remove(v)
            self.used_tokens[v.tenant] = max(0, self.used_tokens[v.tenant] - v.compute_tokens)
            self.stats[v.tenant]["preempted"] += 1
            self.events.append({"t": round(self.now, 4), "event": "preempt",
                                "req": v.req_id, "tenant": v.tenant,
                                "by": new_req.req_id, "blocks": v.blocks})
            self._release_entry(v, evict=True)
        return freed

    def finish(self, r: Request, end_s: Optional[float] = None) -> None:
        r.status = "done"
        r.finished_s = end_s if end_s is not None else self.now
        if r in self.running[r.tenant]:
            self.running[r.tenant].remove(r)
        # 结果与幂等键写在同一条记录里 (事务式提交)
        self.committed[r.idem_key] = {"req_id": r.req_id, "tenant": r.tenant,
                                      "status": "done", "t": round(r.finished_s or 0, 4)}
        self.stats[r.tenant]["done"] += 1
        self.events.append({"t": round(r.finished_s or 0, 4), "event": "done",
                            "req": r.req_id, "tenant": r.tenant,
                            "latency": round(r.latency_s or 0, 4)})
        self._release_entry(r, evict=False)     # 保留 KV 供后续复用

    def cancel(self, req_id: str) -> Dict[str, Any]:
        """跨租户取消: 只能取消正在跑的请求, 释放的块只归还到它自己的租户账上。"""
        for tid, reqs in self.running.items():
            for r in list(reqs):
                if r.req_id == req_id:
                    r.status = "failed"
                    r.reason = "cancelled"
                    r.finished_s = self.now
                    reqs.remove(r)
                    self.used_tokens[tid] = max(0, self.used_tokens[tid] - r.compute_tokens)
                    self.stats[tid]["rejected"] += 1
                    self._release_entry(r, evict=True)
                    self.events.append({"t": round(self.now, 4), "event": "cancel",
                                        "req": r.req_id, "tenant": tid})
                    return {"cancelled": True, "tenant": tid}
        return {"cancelled": False}

    # ---- 重启与幂等 ------------------------------------------------------
    def snapshot(self, path: Path) -> None:
        path.write_text(json.dumps({
            "committed": self.committed,
            "cache": {k: {"owner": list(v["owner"]), "prefix_len": v["prefix_len"],
                          "blocks": v["blocks"]} for k, v in self.cache.items()},
            "used_tokens": dict(self.used_tokens),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    def replay(self, requests: List[Request], path: Path) -> List[Dict[str, Any]]:
        snap = json.loads(path.read_text())
        self.committed = snap["committed"]
        out = []
        for r in requests:
            rec = self.committed.get(r.idem_key)
            if rec is not None:
                r.status = "done"
                r.reason = "idempotent_replay"
                self.stats[r.tenant]["idem_replay"] += 1
                out.append({"req": r.req_id, "replayed": True, "record": rec})
            else:
                self.ensure_adapter(r)
                self.run(r)
                out.append({"req": r.req_id, "replayed": False, "status": r.status})
        return out

    # ---- 审计与统计 ------------------------------------------------------
    def audit(self) -> Dict[str, Any]:
        bad = []
        for key, entry in self.cache.items():
            if entry["refs"] != 0:
                bad.append({"key": key, "issue": "refs_leak", "refs": entry["refs"]})
            if len(entry["tenants_seen"]) > 1 and not self.ignore_tenant:
                bad.append({"key": key, "issue": "cross_tenant_block",
                            "tenants": sorted(entry["tenants_seen"])})
            if len(entry.get("revisions_seen", ())) > 1 and not self.ignore_revision:
                bad.append({"key": key, "issue": "cross_revision_block",
                            "revisions": sorted(entry["revisions_seen"])})
        resident = self.resident_blocks()
        return {
            "issues": bad,
            "resident_blocks": resident,
            "free_blocks": self.capacity_left(),
            "kv_capacity": self.kv_capacity,
            "block_accounting_ok": resident + self.capacity_left() == self.kv_capacity,
            "owners": {k: {"owner": list(e["owner"]), "blocks": e["blocks"]}
                       for k, e in list(self.cache.items())[:6]},
        }

    def per_tenant(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {tid: dict(s) for tid, s in self.stats.items()}
        lats: Dict[str, List[float]] = defaultdict(list)
        for e in self.events:
            if e["event"] == "done":
                lats[e["tenant"]].append(e["latency"])
        for tid, xs in lats.items():
            xs.sort()
            out[tid]["p50"] = round(xs[len(xs) // 2], 4)
            out[tid]["p95"] = round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 4)
            out[tid]["p99"] = round(xs[min(len(xs) - 1, int(0.99 * len(xs)))], 4)
            out[tid]["n_latency_samples"] = len(xs)
        return out


# --------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------
def build_scenario(seed: int = 0, n_a: int = 60, n_b: int = 40, prefix_len: int = 2048,
                   long_len: int = 8192):
    rng = random.Random(seed)
    tenants = [
        Tenant("A", priority=0, token_budget=200_000, max_concurrency=8, max_adapters=1),
        Tenant("B", priority=1, token_budget=400_000, max_concurrency=16, max_adapters=2),
    ]
    shared_prefix = tuple(rng.randrange(1000, 90000) for _ in range(prefix_len))
    reqs: List[Request] = []
    t = 0.0
    for i in range(n_a):
        t += 0.4
        reqs.append(Request(f"A{i}", f"idem-A{i}", "A", f"A-s{i % 4}", "rev-a",
                            shared_prefix, 8, t, "interactive"))
    for i in range(n_b):
        t += 0.25
        rev = "rev-b1" if i % 2 == 0 else "rev-b2"   # 同名 adapter 的两个 revision
        long = i % 3 == 0
        if i < 2:
            # 前两条 B 请求故意用与 A **完全相同**的文本: 用来检验"同一段文本、
            # 不同租户/不同 revision"不会命中同一份 KV。
            p = shared_prefix
        else:
            p = tuple(rng.randrange(1000, 90000)
                      for _ in range(long_len if long else prefix_len))
        reqs.append(Request(f"B{i}", f"idem-B{i}", "B", f"B-s{i % 8}", rev, p, 48, t,
                            "long" if long else "batch"))
    reqs.sort(key=lambda r: r.arrival_s)
    return tenants, reqs


def run_case(policy: str, kv_blocks: int, ignore_revision: bool = False,
             ignore_tenant: bool = False, preempt: bool = True,
             seed: int = 0) -> Dict[str, Any]:
    tenants, reqs = build_scenario(seed=seed)
    sch = TenantScheduler(policy, kv_blocks, ignore_revision, ignore_tenant, preempt,
                          service_rate=8.0)
    for t in tenants:
        sch.tenants[t.tenant_id] = t
    rejected_by_reason: Dict[str, int] = defaultdict(int)
    in_flight_peak = 0
    for r in reqs:
        # 时间推进到本次到达: 之前应当完成的请求先结算, 并发才真实存在
        sch.advance_to(r.arrival_s)
        ok, reason = sch.admit(r)
        if not ok:
            r.status = "rejected"
            r.reason = reason
            sch.stats[r.tenant]["rejected"] += 1
            rejected_by_reason[reason] += 1
            sch.events.append({"t": round(sch.now, 4), "event": "reject",
                               "req": r.req_id, "tenant": r.tenant, "reason": reason})
            continue
        sch.ensure_adapter(r)
        sch.run(r)
        in_flight_peak = max(in_flight_peak, sum(len(v) for v in sch.running.values()))
    sch.advance_to(float("inf"))
    return {
        "policy": policy, "kv_blocks": kv_blocks, "ignore_revision": ignore_revision,
        "ignore_tenant": ignore_tenant, "preempt": preempt,
        "rejected_by_reason": dict(rejected_by_reason),
        "in_flight_peak": in_flight_peak,
        "per_tenant": sch.per_tenant(), "audit": sch.audit(),
        "n_events": len(sch.events), "scheduler": sch, "requests": reqs,
    }


def identity_cases(sch: TenantScheduler, reqs: List[Request]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    r_b1 = next(r for r in reqs if r.tenant == "B" and r.adapter_revision == "rev-b1")
    r_b2 = next(r for r in reqs if r.tenant == "B" and r.adapter_revision == "rev-b2")
    out["same_name_diff_revision_keys_differ"] = sch.cache_key(r_b1) != sch.cache_key(r_b2)
    a = next(r for r in reqs if r.tenant == "A")
    b = next(r for r in reqs if r.tenant == "B" and r.prefix == a.prefix)
    out["same_text_cross_tenant_keys_differ"] = sch.cache_key(a) != sch.cache_key(b)
    same = [r for r in reqs if r.tenant == "A" and r.adapter_revision == "rev-a"]
    out["same_tenant_same_prefix_same_key"] = len({sch.cache_key(r) for r in same}) == 1
    out["every_block_owned_by_key_tenant"] = all(
        k.split("|")[0] == e["owner"][0] for k, e in sch.cache.items())
    return out


def cancel_case() -> Dict[str, Any]:
    tenants, scenario = build_scenario()
    sch = TenantScheduler("quota", 1200)
    for t in tenants:
        sch.tenants[t.tenant_id] = t
    a = [r for r in scenario if r.tenant == "A"][:4]
    b = [r for r in scenario if r.tenant == "B"][:4]
    for r in a:
        sch.ensure_adapter(r)
        sch.run(r)
    sch.advance_to(float("inf"))         # A 的请求先全部完成
    for r in b:
        sch.ensure_adapter(r)
    rb = b[0]
    rb.started_s = sch.now
    rb.blocks = 4
    rb.compute_tokens = rb.prompt_tokens
    sch.running["B"].append(rb)
    sch.used_tokens["B"] += rb.compute_tokens
    sch.cache[sch.cache_key(rb)] = {"owner": ("B", rb.adapter_revision),
                                    "prefix_len": rb.prompt_tokens, "refs": 1,
                                    "blocks": 4, "tenants_seen": {"B"}}
    before = {"free": sch.capacity_left(), "used_a": sch.used_tokens["A"],
              "done_a": sch.stats["A"]["done"]}
    res = sch.cancel(rb.req_id)
    after = {"free": sch.capacity_left(), "used_a": sch.used_tokens["A"],
             "done_a": sch.stats["A"]["done"]}
    return {
        "cancel_result": res,
        "tenant_a_free_blocks_restored": after["free"] == before["free"] + 4,
        "tenant_a_used_tokens_unchanged": after["used_a"] == before["used_a"],
        "tenant_a_done_unchanged": after["done_a"] == before["done_a"],
        "cancelled_req_status": rb.status,
        "b_entry_removed": sch.cache_key(rb) not in sch.cache,
        "cancel_unknown_req": sch.cancel("does-not-exist"),
        "audit_after_cancel_issues": sch.audit()["issues"],
    }


def restart_case(out: Path) -> Dict[str, Any]:
    tenants, scenario = build_scenario()
    sch = TenantScheduler("quota", 1200)
    for t in tenants:
        sch.tenants[t.tenant_id] = t
    for r in scenario[:20]:
        sch.ensure_adapter(r)
        sch.run(r)
    sch.advance_to(float("inf"))
    snap = out / "scheduler_snapshot.json"
    sch.snapshot(snap)
    committed_before = len(sch.committed)
    sch2 = TenantScheduler("quota", 1200)
    for t in tenants:
        sch2.tenants[t.tenant_id] = t
    replay = sch2.replay(scenario[:20], snap)
    return {
        "committed_before": committed_before,
        "replayed": sum(1 for x in replay if x["replayed"]),
        "recomputed": sum(1 for x in replay if not x["replayed"]),
        "committed_after": len(sch2.committed),
        "no_duplicate_results": len(sch2.committed) == committed_before,
        "replay_stats": {k: dict(v) for k, v in sch2.stats.items()},
    }


def simulate(tenants: Dict[str, Tenant], reqs: List[Request], policy: str,
             kv_capacity: int, max_inflight: int, preempt: bool = True,
             token_rate: float = 2000.0, queue_factor: int = 4) -> Dict[str, Any]:
    """带排队与共享算力的离散事件模拟。

    引擎总算力按 `token_rate` (token/s) 计, 在在跑请求之间**均分**: k 个请求同时
    在跑时每个拿到 token_rate/k。因此租户 B 的长请求会真实地拉长 A 的等待与完成
    时间, 而不是各自独立计时。准入只按租户预算与队列上限决定收不收; 收下的请求在
    队列里等 KV 与执行槽, 等待时间计入时延。
    """
    cache: Dict[str, Dict[str, Any]] = {}
    used_tokens: Dict[str, int] = defaultdict(int)
    running: List[Dict[str, Any]] = []          # {"r": req, "remaining": tokens}
    queued: List[Request] = []
    in_system: Dict[str, int] = defaultdict(int)
    events: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"admitted": 0, "rejected": 0, "preempted": 0, "done": 0,
                 "cache_hit_tokens": 0, "compute_tokens": 0, "queue_s": 0.0,
                 "queue_n": 0})
    now = 0.0
    EPS = 1e-9

    def capacity_left() -> int:
        return kv_capacity - sum(e["blocks"] for e in cache.values())

    def key_of(r: Request) -> str:
        return f"{r.tenant}|{r.adapter_revision}|{prefix_hash(r.prefix)}"

    def release(r: Request, evict: bool) -> None:
        e = cache.get(key_of(r))
        if e is None:
            return
        e["refs"] = max(0, e["refs"] - 1)
        if evict and e["refs"] == 0:
            del cache[key_of(r)]

    def complete(r: Request, t: float) -> None:
        r.status = "done"
        r.finished_s = t
        in_system[r.tenant] -= 1
        stats[r.tenant]["done"] += 1
        events.append({"t": round(t, 4), "event": "done", "req": r.req_id,
                       "tenant": r.tenant,
                       "latency": round(t - r.arrival_s, 4)})
        release(r, evict=False)

    def start(r: Request, t: float) -> None:
        e = cache.get(key_of(r))
        cached = 0
        if e is not None:
            cached = min(len(r.prefix) // BLOCK_TOKENS * BLOCK_TOKENS, e["prefix_len"])
        r.cached_tokens = cached
        r.compute_tokens = r.prompt_tokens - cached
        if e is None:
            blocks = max(1, (r.prompt_tokens + r.output_tokens + BLOCK_TOKENS - 1)
                         // BLOCK_TOKENS)
            cache[key_of(r)] = {"owner": (r.tenant, r.adapter_revision),
                                "prefix_len": r.prompt_tokens, "refs": 0,
                                "blocks": blocks, "last_used_s": t,
                                "tenants_seen": {r.tenant},
                                "revisions_seen": {r.adapter_revision}}
        else:
            cache[key_of(r)]["tenants_seen"].add(r.tenant)
            cache[key_of(r)]["revisions_seen"].add(r.adapter_revision)
        cache[key_of(r)]["refs"] += 1
        cache[key_of(r)]["last_used_s"] = t
        r.blocks = cache[key_of(r)]["blocks"]
        r.started_s = t
        r.status = "running"
        running.append({"r": r, "remaining": float(r.compute_tokens + r.output_tokens)})
        used_tokens[r.tenant] += r.compute_tokens
        stats[r.tenant]["admitted"] += 1
        stats[r.tenant]["cache_hit_tokens"] += cached
        stats[r.tenant]["compute_tokens"] += r.compute_tokens
        stats[r.tenant]["queue_s"] += t - r.arrival_s
        stats[r.tenant]["queue_n"] += 1
        events.append({"t": round(t, 4), "event": "start", "req": r.req_id,
                       "tenant": r.tenant, "cached": cached, "blocks": r.blocks,
                       "queue_s": round(t - r.arrival_s, 4)})

    def evict_lru(need: int, keep_key: Optional[str] = None) -> int:
        freed = 0
        for k, e in sorted(list(cache.items()), key=lambda kv: kv[1]["last_used_s"]):
            if freed >= need:
                break
            if e["refs"] != 0 or k == keep_key:
                continue
            del cache[k]
            freed += e["blocks"]
            events.append({"t": round(now, 4), "event": "evict_cache",
                           "blocks": e["blocks"]})
        return freed

    def preempt_for(need: int, new_req: Request, t: float) -> int:
        victims = [x for x in running
                   if tenants[x["r"].tenant].priority > tenants[new_req.tenant].priority]
        victims.sort(key=lambda x: (-tenants[x["r"].tenant].priority, x["r"].started_s or 0))
        freed = 0
        for x in victims:
            if freed >= need:
                break
            v = x["r"]
            running.remove(x)
            v.status = "preempted"
            v.reason = "kv_preempted"
            v.finished_s = t
            freed += v.blocks
            in_system[v.tenant] -= 1
            stats[v.tenant]["preempted"] += 1
            events.append({"t": round(t, 4), "event": "preempt", "req": v.req_id,
                           "tenant": v.tenant, "by": new_req.req_id,
                           "blocks": v.blocks})
            release(v, evict=True)
        return freed

    def try_start(t: float) -> None:
        order = list(queued)
        if policy == "priority":
            order.sort(key=lambda r: (tenants[r.tenant].priority, r.arrival_s))
        for r in order:
            if r not in queued:
                continue
            tnt = tenants[r.tenant]
            tenant_running = sum(1 for x in running if x["r"].tenant == r.tenant)
            if len(running) >= max_inflight:
                break
            if policy != "shared" and tenant_running >= tnt.max_concurrency:
                continue
            is_hit = cache.get(key_of(r)) is not None
            if is_hit:
                need_blocks = 0
            else:
                need_blocks = max(1, (r.prompt_tokens + r.output_tokens
                                      + BLOCK_TOKENS - 1) // BLOCK_TOKENS)
            if need_blocks > capacity_left():
                evict_lru(need_blocks - capacity_left(), keep_key=None)
            if need_blocks > capacity_left() and preempt:
                preempt_for(need_blocks - capacity_left(), r, t)
            if need_blocks > capacity_left():
                continue
            queued.remove(r)
            start(r, t)

    def advance_to(t_next: float) -> None:
        nonlocal now
        dt = max(0.0, t_next - now)
        k = len(running)
        if k and dt > 0:
            share = dt * token_rate / k
            for x in running:
                x["remaining"] -= share
        now = max(now, t_next)
        finished = [x for x in running if x["remaining"] <= EPS]
        for x in finished:
            running.remove(x)
            complete(x["r"], now)

    idx = 0
    guard = 0
    while (idx < len(reqs) or running or queued) and guard < 10_000_000:
        guard += 1
        t_arrival = reqs[idx].arrival_s if idx < len(reqs) else float("inf")
        if running:
            k = len(running)
            t_finish = now + min(x["remaining"] for x in running) * k / token_rate
        else:
            t_finish = float("inf")
        t_next = min(t_arrival, t_finish)
        if t_next == float("inf"):
            break
        advance_to(t_next)
        if t_arrival <= t_finish:
            r = reqs[idx]
            idx += 1
            tnt = tenants[r.tenant]
            if len(running) + len(queued) >= max_inflight * queue_factor:
                r.status = "rejected"
                r.reason = "global_queue"
                stats[r.tenant]["rejected"] += 1
                events.append({"t": round(now, 4), "event": "reject",
                               "req": r.req_id, "tenant": r.tenant,
                               "reason": "global_queue"})
                continue
            if policy != "shared" and in_system[r.tenant] >= tnt.max_concurrency * queue_factor:
                r.status = "rejected"
                r.reason = "tenant_queue"
                stats[r.tenant]["rejected"] += 1
                events.append({"t": round(now, 4), "event": "reject",
                               "req": r.req_id, "tenant": r.tenant,
                               "reason": "tenant_queue"})
                continue
            if policy != "shared" and used_tokens[r.tenant] + r.prompt_tokens > tnt.token_budget:
                r.status = "rejected"
                r.reason = "token_budget"
                stats[r.tenant]["rejected"] += 1
                events.append({"t": round(now, 4), "event": "reject",
                               "req": r.req_id, "tenant": r.tenant,
                               "reason": "token_budget"})
                continue
            queued.append(r)
            in_system[r.tenant] += 1
        try_start(now)

    out: Dict[str, Any] = {}
    for tid, s in stats.items():
        xs = sorted(e["latency"] for e in events
                    if e["event"] == "done" and e["tenant"] == tid)
        qs = sorted(e["queue_s"] for e in events
                    if e["event"] == "start" and e["tenant"] == tid)
        d = dict(s)
        d["queue_s"] = round(d["queue_s"], 4)
        if xs:
            d["p50"] = round(xs[len(xs) // 2], 4)
            d["p95"] = round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 4)
            d["p99"] = round(xs[min(len(xs) - 1, int(0.99 * len(xs)))], 4)
            d["max"] = round(xs[-1], 4)
        if qs:
            d["queue_p50"] = round(qs[len(qs) // 2], 4)
            d["queue_p99"] = round(qs[min(len(qs) - 1, int(0.99 * len(qs)))], 4)
        out[tid] = d
    out["_audit"] = {
        "resident_blocks": sum(e["blocks"] for e in cache.values()),
        "kv_capacity": kv_capacity,
        "block_accounting_ok": sum(e["blocks"] for e in cache.values()) <= kv_capacity,
        "cross_tenant": [k for k, e in cache.items() if len(e["tenants_seen"]) > 1],
        "cross_revision": [k for k, e in cache.items() if len(e["revisions_seen"]) > 1],
    }
    return {"per_tenant": out, "events": events}


def pressure_sweep(kv_blocks: int, seed: int = 0) -> Dict[str, Any]:
    """租户 A 固定交互负载, B 的请求份额与长 prompt 比例逐步增加。"""
    rows = []
    for policy in ("shared", "quota", "priority", "quota_no_preempt"):
        for frac_b, long_frac in ((0.0, 0.0), (0.25, 0.0), (0.5, 0.34), (0.75, 0.67),
                                  (1.0, 1.0)):
            rng = random.Random(seed)
            tenants = {
                "A": Tenant("A", priority=0, token_budget=10 ** 9,
                            max_concurrency=8, max_adapters=1),
                "B": Tenant("B", priority=1, token_budget=10 ** 9,
                            max_concurrency=16, max_adapters=2),
            }
            n_b = int(40 * frac_b)
            a_prefix = tuple(rng.randrange(1000, 90000) for _ in range(2048))
            t_a, t_b = 0.0, 0.0
            reqs: List[Request] = []
            for i in range(60):
                t_a += 0.4
                reqs.append(Request(f"A{i}", f"idem-A{i}", "A", f"A-s{i % 4}", "rev-a",
                                    a_prefix, 64, t_a, "interactive"))
            for i in range(n_b):
                t_b += 0.25
                long = rng.random() < long_frac
                p = tuple(rng.randrange(1000, 90000)
                          for _ in range(8192 if long else 2048))
                reqs.append(Request(f"B{i}", f"idem-B{i}", "B", f"B-s{i % 8}",
                                    "rev-b1" if i % 2 == 0 else "rev-b2", p, 48, t_b,
                                    "long" if long else "batch"))
            reqs.sort(key=lambda r: r.arrival_s)
            res = simulate(tenants, reqs,
                           ("shared" if policy == "shared" else
                            "priority" if policy == "priority" else "quota"),
                           kv_capacity=4 * kv_blocks, max_inflight=48, token_rate=600.0,
                           preempt=(policy != "quota_no_preempt"))
            pt = res["per_tenant"]
            keys = ("admitted", "rejected", "preempted", "done", "p50", "p95", "p99",
                    "max", "cache_hit_tokens")
            rows.append({
                "policy": policy, "b_share": frac_b, "long_frac": long_frac, "n_b": n_b,
                "A": {k: pt.get("A", {}).get(k) for k in keys},
                "B": {k: pt.get("B", {}).get(k) for k in keys},
                "audit": pt["_audit"],
            })
    return {"rows": rows}


def identity_probe(ignore_revision: bool, ignore_tenant: bool) -> Dict[str, Any]:
    """极小场景直接看"同一段文本被不同租户/不同 revision 请求"时的命中情况。

    容量与预算都放到不受限, 因此唯一的变量就是缓存键里有没有 tenant / adapter
    revision。`cached_tokens > 0` 表示这次请求拿到了别人的 KV。
    """
    tenants = [Tenant("A", priority=0, token_budget=10 ** 9, max_concurrency=8,
                      max_adapters=1),
               Tenant("B", priority=1, token_budget=10 ** 9, max_concurrency=8,
                      max_adapters=2)]
    sch = TenantScheduler("quota", kv_blocks=10_000, ignore_revision=ignore_revision,
                          ignore_tenant=ignore_tenant, max_inflight=8)
    for t in tenants:
        sch.tenants[t.tenant_id] = t
    P = tuple(range(1000, 1000 + 2048))
    reqs = [
        Request("A0", "pA0", "A", "sA", "rev-a", P, 32, 0.0),
        Request("B0", "pB0", "B", "sB", "rev-b1", P, 32, 0.1),
        Request("B1", "pB1", "B", "sB", "rev-b2", P, 32, 0.2),
    ]
    for r in reqs:
        sch.advance_to(r.arrival_s)
        sch.ensure_adapter(r)
        sch.run(r)
    sch.advance_to(float("inf"))
    per_req = {r.req_id: {"tenant": r.tenant, "revision": r.adapter_revision,
                          "cached_tokens": r.cached_tokens,
                          "compute_tokens": r.compute_tokens,
                          "key": sch.cache_key(r)} for r in reqs}
    entries = [{"key": k, "owner": list(e["owner"]),
                "tenants_seen": sorted(e["tenants_seen"]),
                "revisions_seen": sorted(e["revisions_seen"]),
                "blocks": e["blocks"], "refs": e["refs"]}
               for k, e in sch.cache.items()]
    return {
        "ignore_revision": ignore_revision, "ignore_tenant": ignore_tenant,
        "per_request": per_req,
        "entries": entries,
        "wrong_reuse_B1_after_revision_change": per_req["B1"]["cached_tokens"] > 0,
        "wrong_reuse_B0_across_tenant": per_req["B0"]["cached_tokens"] > 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-blocks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"kv_blocks": args.kv_blocks, "seed": args.seed,
                               "cases": {}}
    cases = {
        "shared": dict(policy="shared", kv_blocks=args.kv_blocks),
        "quota": dict(policy="quota", kv_blocks=args.kv_blocks),
        "quota_no_preempt": dict(policy="quota", kv_blocks=args.kv_blocks, preempt=False),
        "quota_small_kv": dict(policy="quota", kv_blocks=args.kv_blocks // 4),
        "revision_missing": dict(policy="quota", kv_blocks=args.kv_blocks,
                                 ignore_revision=True),
        "tenant_missing": dict(policy="quota", kv_blocks=args.kv_blocks,
                               ignore_tenant=True),
    }
    for name, kw in cases.items():
        r = run_case(seed=args.seed, **kw)
        sch = r.pop("scheduler")
        reqs = r.pop("requests")
        if name in ("quota", "revision_missing", "tenant_missing"):
            r["identity"] = identity_cases(sch, reqs)
        summary["cases"][name] = r
    summary["cancel"] = cancel_case()
    summary["restart_replay"] = restart_case(out)
    summary["pressure_sweep"] = pressure_sweep(args.kv_blocks, args.seed)
    summary["identity_probe"] = {
        "correct": identity_probe(False, False),
        "ignore_revision": identity_probe(True, False),
        "ignore_tenant": identity_probe(False, True),
        "ignore_both": identity_probe(True, True),
    }
    (out / "tenant_scheduler.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["per_tenant"] for k, v in summary["cases"].items()},
                     ensure_ascii=False)[:1200])
    print("identity(quota):",
          json.dumps(summary["cases"]["quota"]["identity"], ensure_ascii=False))
    print("identity(revision_missing):",
          json.dumps(summary["cases"]["revision_missing"]["identity"], ensure_ascii=False))
    print("revision_missing audit:",
          json.dumps(summary["cases"]["revision_missing"]["audit"]["issues"],
                     ensure_ascii=False)[:300])
    print("cancel:", json.dumps(summary["cancel"], ensure_ascii=False)[:400])
    print("restart:", json.dumps(summary["restart_replay"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
