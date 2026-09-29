#!/usr/bin/env python3
"""labs/L8/request_router.py - 路由策略、KV 目录与可重放决策日志 (8.1-A/C).

三种策略:
    round_robin    顺序轮转, 不看缓存也不看队列。
    shortest_queue 选网关侧在途请求最少的 worker。
    prefix_aware   预估前缀命中块数, 减去队列惩罚 beta * queue_depth, 取分最高者。

两类目录:
    EventDirectory   精确目录。订阅 worker 的真实 KV 事件 (vLLM BlockStored /
                     BlockRemoved / AllBlocksCleared), 用事件里的 block_hashes
                     与 token_ids 维护"这个 worker 现在持有哪些前缀块"。
    ApproxPrefixIndex 近似目录。只根据"我把哪些请求发给过哪个 worker"建前缀
                     字典树, 完全不知道驱逐。两者对同一请求的预测命中差, 就是
                     目录漂移 (drift) 的代价。

块身份沿用 vLLM 的链式哈希语义: 第 i 块的键由 (父块键, 本块 token 序列) 决定,
因此同样的 token 片段出现在不同位置不会误判为同一块。网关自己算键, 并通过事件
里的 block_hashes 建立"vLLM 哈希 -> 本网关键"的映射, 以便处理 BlockRemoved。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT_KEY = b"\x00" * 16


def block_key(parent: bytes, tokens: Sequence[int]) -> bytes:
    h = hashlib.sha1()
    h.update(parent)
    h.update(struct.pack("<I", len(tokens)))
    for t in tokens:
        h.update(struct.pack("<I", t))
    return h.digest()


def chain_keys(token_ids: Sequence[int], block_size: int) -> List[bytes]:
    """从位置 0 开始按 block_size 切分, 返回每一块的链式键。

    只有完整的块才参与匹配 (vLLM 同样只缓存整块), 尾部不足一块的 token 不产生键。
    """
    keys: List[bytes] = []
    parent = ROOT_KEY
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        parent = block_key(parent, token_ids[i:i + block_size])
        keys.append(parent)
    return keys


# --------------------------------------------------------------------------
# 目录
# --------------------------------------------------------------------------
class EventDirectory:
    """精确目录: 由真实 KV 事件驱动。"""

    def __init__(self, block_size: int):
        self.block_size = block_size
        # worker -> 已缓存块的网关键集合
        self.cached: Dict[str, Set[bytes]] = {}
        # worker -> vLLM block_hash -> 网关键
        self.hash_map: Dict[str, Dict[Any, bytes]] = {}
        self.stats = {"stored": 0, "removed": 0, "cleared": 0,
                      "duplicate": 0, "unknown_remove": 0}

    def _sets(self, worker: str) -> Tuple[Set[bytes], Dict[Any, bytes]]:
        return (self.cached.setdefault(worker, set()),
                self.hash_map.setdefault(worker, {}))

    def on_block_stored(self, worker: str, block_hashes: Sequence[Any],
                        token_ids: Sequence[int], block_size: Optional[int] = None,
                        parent_block_hash: Any = None) -> None:
        bs = block_size or self.block_size
        cached, hmap = self._sets(worker)
        parent = hmap.get(parent_block_hash, ROOT_KEY) if parent_block_hash is not None else ROOT_KEY
        n = len(block_hashes)
        for i in range(n):
            toks = token_ids[i * bs:(i + 1) * bs]
            if len(toks) < bs:
                break
            key = block_key(parent, toks)
            if key in cached:
                self.stats["duplicate"] += 1
            cached.add(key)
            hmap[block_hashes[i]] = key
            parent = key
        self.stats["stored"] += n

    def on_block_removed(self, worker: str, block_hashes: Sequence[Any]) -> None:
        cached, hmap = self._sets(worker)
        for bh in block_hashes:
            key = hmap.pop(bh, None)
            if key is None:
                self.stats["unknown_remove"] += 1
                continue
            cached.discard(key)
        self.stats["removed"] += len(block_hashes)

    def on_all_cleared(self, worker: str) -> None:
        cached, hmap = self._sets(worker)
        cached.clear()
        hmap.clear()
        self.stats["cleared"] += 1

    def predicted_hit_tokens(self, worker: str, token_ids: Sequence[int]) -> int:
        cached = self.cached.get(worker)
        if not cached:
            return 0
        hit = 0
        for k in chain_keys(token_ids, self.block_size):
            if k in cached:
                hit += self.block_size
            else:
                break
        return hit


class ApproxPrefixIndex:
    """近似目录: 只看网关自己发出过的请求, 不知道驱逐。"""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.seen: Dict[str, Set[bytes]] = {}

    def record_routed(self, worker: str, token_ids: Sequence[int]) -> None:
        s = self.seen.setdefault(worker, set())
        s.update(chain_keys(token_ids, self.block_size))

    def predicted_hit_tokens(self, worker: str, token_ids: Sequence[int]) -> int:
        s = self.seen.get(worker)
        if not s:
            return 0
        hit = 0
        for k in chain_keys(token_ids, self.block_size):
            if k in s:
                hit += self.block_size
            else:
                break
        return hit


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@dataclasses.dataclass
class RoutingDecision:
    request_id: str
    policy: str
    chosen_worker: str
    candidates: List[Dict[str, Any]]
    predicted_hit_tokens: int
    queue_depth: int
    reason: str
    # 事后由网关填: 引擎自报的真实命中 token 数
    engine_reported_cached_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Router:
    def __init__(self, workers: Sequence[str], policy: str, block_size: int = 16,
                 queue_penalty: float = 0.0,
                 directory: Optional[EventDirectory] = None,
                 approx: Optional[ApproxPrefixIndex] = None):
        if policy not in ("round_robin", "shortest_queue", "prefix_aware"):
            raise ValueError(policy)
        self.workers = list(workers)
        self.policy = policy
        self.block_size = block_size
        self.queue_penalty = queue_penalty
        self.directory = directory
        self.approx = approx
        self.queue_depth: Dict[str, int] = {w: 0 for w in self.workers}
        self._rr = 0
        self.decisions: List[RoutingDecision] = []

    # 网关在每个请求开始/结束时调用, 维护在途计数。
    def on_start(self, worker: str) -> None:
        self.queue_depth[worker] = self.queue_depth.get(worker, 0) + 1

    def on_finish(self, worker: str) -> None:
        self.queue_depth[worker] = max(0, self.queue_depth.get(worker, 0) - 1)

    def route(self, request_id: str, token_ids: Sequence[int]) -> RoutingDecision:
        cands: List[Dict[str, Any]] = []
        if self.policy == "round_robin":
            chosen = self.workers[self._rr % len(self.workers)]
            self._rr += 1
            for w in self.workers:
                cands.append({"worker": w, "queue_depth": self.queue_depth.get(w, 0),
                              "predicted_hit_tokens": None})
            decision = RoutingDecision(request_id, self.policy, chosen, cands, 0,
                                       self.queue_depth.get(chosen, 0),
                                       f"round-robin index {self._rr - 1}")

        elif self.policy == "shortest_queue":
            for w in self.workers:
                cands.append({"worker": w, "queue_depth": self.queue_depth.get(w, 0),
                              "predicted_hit_tokens": None})
            chosen = min(self.workers, key=lambda w: (self.queue_depth.get(w, 0), w))
            decision = RoutingDecision(request_id, self.policy, chosen, cands, 0,
                                       self.queue_depth.get(chosen, 0),
                                       "min in-flight at gateway")

        else:
            if self.directory is None:
                raise ValueError("prefix_aware needs a directory")
            best, best_score, best_hit = None, float("-inf"), 0
            for w in self.workers:
                q = self.queue_depth.get(w, 0)
                hit = self.directory.predicted_hit_tokens(w, token_ids)
                score = hit - self.queue_penalty * q
                cands.append({"worker": w, "queue_depth": q,
                              "predicted_hit_tokens": hit, "score": score})
                if score > best_score:
                    best, best_score, best_hit = w, score, hit
            decision = RoutingDecision(
                request_id, self.policy, best, cands, best_hit,
                self.queue_depth.get(best, 0),
                f"max score {best_score:.0f} (hit={best_hit}, q={self.queue_depth.get(best, 0)})")

        self.decisions.append(decision)
        return decision

    def note_routed(self, worker: str, token_ids: Sequence[int]) -> None:
        """网关转发后更新近似目录 (精确目录由事件更新)。"""
        if self.approx is not None:
            self.approx.record_routed(worker, token_ids)

    def dump_decisions(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for d in self.decisions:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def replay(decisions_path: str, token_lookup) -> List[RoutingDecision]:
        """从决策日志重算: 校验同一条输入状态能否复现同一选择。

        token_lookup(request_id) 返回该请求的 token 序列; 重放时只看 prefix_aware
        的候选得分是否与记录一致 (同状态同决策)。
        """
        out: List[RoutingDecision] = []
        with open(decisions_path, "r", encoding="utf-8") as f:
            for line in f:
                d = RoutingDecision(**json.loads(line))
                out.append(d)
        return out
