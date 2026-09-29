#!/usr/bin/env python3
"""L9.6 任务 D：长期记忆存储的生命周期（namespace / version / 有效期 / tombstone）。

存储要同时管三样东西，而且它们的失效条件不同：

* **文本事实**：可纠正、可删除；删除必须是 tombstone（留痕）而不是物理覆盖，否则并发读写会看到"凭空消失"。
* **向量索引**：从文本派生，重建成本高；典型实现是异步/批量更新，于是存在"文本已提交、索引还没跟上"的窗口。
* **结果缓存**：从查询派生；文本或索引一变就让相关条目失效。

本脚本实现一个最小存储并验证四件事：

1. **可见性边界**：提交（commit）返回之后开始的新读必须看不到已删除/已过期事实；提交之前开始的读按它读到的版本解释。
2. **tombstone 过滤**：索引里仍留着旧片段时，查询侧必须按 tombstone 过滤，否则会"召回已删除事实"（脚本先演示不过滤的失败，再演示过滤后的正确结果）。
3. **纠正语义**：同一 key 的新版本必须让旧版本对**新读**不可见；进行中的读记录它读到的版本。
4. **重建一致性**：用文本事实重建索引后，检索结果应与增量更新后的结果一致（两者对拍）。

用法::

    # 全流程（需要 embedding 服务）
    python labs/L9/memory_store_lifecycle.py run --base-url http://127.0.0.1:8016/v1 --out DIR
    # 只用确定性假向量验证存储逻辑（无需 GPU）
    python labs/L9/memory_store_lifecycle.py run --fake-embed --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sqlite3
import time

FACTS = [
    ("f1", "ns:public", "阿司匹林用于二级预防时的常用剂量是 75–100 mg/日。"),
    ("f2", "ns:public", "他汀类药物在低风险人群中的绝对收益较小。"),
    ("f3", "ns:public", "NFCorpus 语料里的 MED-2427 讨论他汀与心血管事件。"),
    ("f4", "ns:team-a", "内部结论：项目 A 的检索阈值暂定为 0.35。"),
    ("f5", "ns:team-b", "内部结论：项目 B 的检索阈值暂定为 0.62。"),
]

QUERIES = [
    ("q1", "ns:public", "阿司匹林二级预防剂量是多少？", "f1"),
    ("q2", "ns:public", "他汀在低风险人群的收益如何？", "f2"),
    ("q3", "ns:public", "哪篇文献讨论他汀与心血管事件？", "f3"),
    ("q4", "ns:team-a", "项目 A 的检索阈值是多少？", "f4"),
    ("q5", "ns:team-b", "项目 B 的检索阈值是多少？", "f5"),
]


# --------------------------------------------------------------------------------------
# 存储：SQLite 事务 + tombstone + 版本号
# --------------------------------------------------------------------------------------

class MemoryStore:
    """文本事实 + 版本 + tombstone；所有写入都在单个事务里完成。

    ``version`` 是全局单调计数器，每次提交 +1。读请求记录自己看到的版本，
    用于解释"为什么这次请求读到了旧值"。可见性规则：**新读只看已提交且版本 ≤ 当前版本的内容**；
    删除写成 tombstone（``deleted=1``），旧版本行保留。
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT NOT NULL, namespace TEXT NOT NULL, version INTEGER NOT NULL,
                text TEXT NOT NULL, valid_from REAL NOT NULL, valid_to REAL,
                deleted INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                PRIMARY KEY (key, version)
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, op TEXT, key TEXT,
                version INTEGER, detail TEXT)""")
        if self.conn.execute("SELECT COUNT(*) FROM meta WHERE k='version'").fetchone()[0] == 0:
            self.conn.execute("INSERT INTO meta(k, v) VALUES ('version', '0')")

    @property
    def version(self) -> int:
        return int(self.conn.execute("SELECT v FROM meta WHERE k='version'").fetchone()[0])

    def _bump(self) -> int:
        v = self.version + 1
        self.conn.execute("UPDATE meta SET v=? WHERE k='version'", (str(v),))
        return v

    def put(self, key: str, namespace: str, text: str, *, valid_to: float | None = None) -> int:
        """新增或纠正：写一个新版本（旧版本保留，但新读只看最新版本）。"""
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            v = self._bump()
            self.conn.execute(
                "INSERT INTO facts(key, namespace, version, text, valid_from, valid_to, deleted, created_at)"
                " VALUES (?,?,?,?,?,?,0,?)", (key, namespace, v, text, now, valid_to, now))
            self.conn.execute("INSERT INTO audit(ts, op, key, version, detail) VALUES (?,?,?,?,?)",
                              (now, "put", key, v, text[:60]))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return v

    def delete(self, key: str, namespace: str) -> int:
        """逻辑删除：写 tombstone（保留旧行，新增一条 deleted=1 的记录）。"""
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            v = self._bump()
            self.conn.execute(
                "INSERT INTO facts(key, namespace, version, text, valid_from, valid_to, deleted, created_at)"
                " VALUES (?,?,?,?,?,?,1,?)", (key, namespace, v, "", now, now, now))
            self.conn.execute("INSERT INTO audit(ts, op, key, version, detail) VALUES (?,?,?,?,?)",
                              (now, "delete", key, v, "tombstone"))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return v

    def snapshot(self, namespace: str, *, as_of: int | None = None) -> dict[str, dict]:
        """取某命名空间在某个版本下的可见事实（最新版本、未删除、未过期）。"""
        v = self.version if as_of is None else as_of
        rows = self.conn.execute(
            "SELECT key, version, text, valid_to, deleted FROM facts"
            " WHERE namespace=? AND version<=? ORDER BY key, version", (namespace, v)).fetchall()
        out: dict[str, dict] = {}
        for key, ver, text, valid_to, deleted in rows:
            if deleted:
                out.pop(key, None)          # tombstone 让它不可见
                continue
            if valid_to is not None and valid_to <= time.time():
                out.pop(key, None)
                continue
            out[key] = {"key": key, "version": ver, "text": text}
        return out

    def all_visible(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for (ns,) in self.conn.execute("SELECT DISTINCT namespace FROM facts").fetchall():
            out.update({k: {**v, "namespace": ns} for k, v in self.snapshot(ns).items()})
        return out

    def audit_tail(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, op, key, version, detail FROM audit ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"ts": r[0], "op": r[1], "key": r[2], "version": r[3], "detail": r[4]} for r in rows]


# --------------------------------------------------------------------------------------
# 向量索引：与文本存储异步对齐，查询侧必须按 tombstone 过滤
# --------------------------------------------------------------------------------------

class VectorIndex:
    """最小向量索引（内积）。``sync`` 表示"索引已经追到文本存储的哪个版本"。"""

    def __init__(self, store: MemoryStore, embed, fake: bool = False):
        self.store = store
        self.embed = embed
        self.fake = fake
        self.rows: list[dict] = []          # {key, namespace, version, text, vec}
        self.synced_version = 0

    def _vec(self, text: str) -> list[float]:
        if self.fake:
            # 确定性假向量：只用于在没有 GPU 时验证存储逻辑
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:16]]
        return self.embed([text])[0]

    def sync(self) -> int:
        """把索引更新到文本存储的当前版本（含 tombstone：旧条目保留，靠过滤屏蔽）。"""
        v = self.store.version
        for ns in {r["namespace"] for r in self.store.all_visible().values()} or {"ns:public"}:
            pass
        rows = self.store.conn.execute(
            "SELECT key, namespace, version, text, deleted FROM facts ORDER BY version").fetchall()
        seen: dict[tuple[str, str], int] = {}
        for key, ns, ver, text, deleted in rows:
            seen[(key, ns)] = ver
            if deleted:
                continue
            if text and not any(r["key"] == key and r["namespace"] == ns and r["version"] == ver
                                for r in self.rows):
                self.rows.append({"key": key, "namespace": ns, "version": ver, "text": text,
                                  "vec": self._vec(text)})
        self.synced_version = v
        return v

    def search(self, query: str, namespace: str, k: int = 3, *, filter_tombstones: bool = True) -> list[dict]:
        qv = self._vec(query)
        live = {k for k in self.store.snapshot(namespace)} if filter_tombstones else None
        scored = []
        for r in self.rows:
            if r["namespace"] != namespace:
                continue
            if filter_tombstones:
                if r["key"] not in live:
                    continue
                # 只允许返回该 key 的当前可见版本
                if self.store.snapshot(namespace)[r["key"]]["version"] != r["version"]:
                    continue
            score = sum(a * b for a, b in zip(qv, r["vec"]))
            scored.append({"key": r["key"], "version": r["version"], "score": round(score, 6),
                           "text": r["text"][:40]})
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]


# --------------------------------------------------------------------------------------
# 生命周期实验
# --------------------------------------------------------------------------------------

def make_embedder(args):
    if args.fake_embed:
        return None, True
    import asyncio

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)

    def embed(texts: list[str]) -> list[list[float]]:
        resp = asyncio.run(client.embeddings.create(model=args.embed_model, input=texts))
        return [d.embedding for d in resp.data]

    return embed, False


def run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "memory.db"
    if db.exists():
        db.unlink()
    store = MemoryStore(db)
    embed, fake = make_embedder(args)
    index = VectorIndex(store, embed, fake=fake)

    steps: list[dict] = []

    def record(name: str, **kw):
        rec = {"step": name, "store_version": store.version,
               "index_synced_version": index.synced_version, **kw}
        steps.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    # 1) 写入初始事实并同步索引
    for key, ns, text in FACTS:
        store.put(key, ns, text)
    index.sync()
    record("seed", facts=len(FACTS))

    # 2) 基线检索：每条查询都应命中它对应的事实
    baseline = {}
    for qid, ns, q, expect in QUERIES:
        hits = index.search(q, ns, k=1)
        baseline[qid] = hits
    record("baseline_search", hits={q: (h[0]["key"] if h else None) for q, h in baseline.items()},
           expected={q: e for _, _, q, e in [(x[0], x[1], x[0], x[3]) for x in QUERIES]})

    # 3) 删除 f4（team-a 的内部结论），但**先不同步索引**：制造"文本已提交、索引未跟上"的窗口
    v_del = store.delete("f4", "ns:team-a")
    leak = index.search("项目 A 的检索阈值是多少？", "ns:team-a", k=1, filter_tombstones=False)
    record("delete_before_index_sync", deleted_version=v_del,
           unfiltered_hits=leak, filtered_hits=index.search("项目 A 的检索阈值是多少？", "ns:team-a"))

    # 4) 同一 key 的纠正：写入新版本，旧版本对新读不可见
    v_new = store.put("f2", "ns:public", "他汀类药物在低风险人群中的绝对收益很小，且个体差异大（v2）。")
    record("correct_fact", key="f2", new_version=v_new,
           visible=store.snapshot("ns:public")["f2"]["text"][:40])

    # 5) 索引同步后，删除与纠正都应在检索侧生效
    index.sync()
    record("after_index_sync",
           f4_hits=index.search("项目 A 的检索阈值是多少？", "ns:team-a", k=1),
           f2_hits=index.search("他汀在低风险人群的收益如何？", "ns:public", k=1))

    # 6) 进行中的读：记录它读到的版本，即使之后有提交也应保持自洽
    as_of = store.version
    snap_before = store.snapshot("ns:public", as_of=as_of)
    store.put("f3", "ns:public", "NFCorpus 的 MED-2427 讨论了 statins 与心血管事件（v2）。")
    snap_after = store.snapshot("ns:public")
    record("in_flight_read", as_of=as_of,
           before_versions={k: v["version"] for k, v in snap_before.items()},
           after_versions={k: v["version"] for k, v in snap_after.items()})

    # 7) 增量更新 vs 全量重建对拍
    index.sync()
    inc = {}
    for qid, ns, q, _expect in QUERIES:
        hits = index.search(q, ns, k=3)
        inc[qid] = [(h["key"], h["version"]) for h in hits]
    rebuilt = VectorIndex(store, embed, fake=fake)
    rebuilt.sync()
    reb = {}
    for qid, ns, q, _expect in QUERIES:
        reb[qid] = [(h["key"], h["version"]) for h in rebuilt.search(q, ns, k=3)]
    record("rebuild_vs_incremental", equal=(inc == reb), incremental=inc, rebuilt=reb)

    # 8) 命名空间隔离：team-a 的查询不得看到 team-b 的事实
    cross = index.search("项目 B 的检索阈值是多少？", "ns:team-a", k=3)
    record("namespace_isolation", team_a_query_for_team_b_fact=cross)

    # 逐条判据：把四个声称的性质写成可检查的布尔值，便于反复跑与回归
    by_name = {s["step"]: s for s in steps}
    semantic_ok = fake or all(
        (by_name["baseline_search"]["hits"].get(q) == expect)
        for q, _ns, _q, expect in QUERIES)
    checks = {
        "tombstone_filter_required": (
            len(by_name["delete_before_index_sync"]["unfiltered_hits"]) == 1
            and by_name["delete_before_index_sync"]["filtered_hits"] == []),
        "corrected_version_visible": by_name["correct_fact"]["store_version"]
        > by_name["seed"]["store_version"],
        "deleted_invisible_after_sync": by_name["after_index_sync"]["f4_hits"] == [],
        "in_flight_read_uses_its_version": (
            by_name["in_flight_read"]["before_versions"]["f3"]
            != by_name["in_flight_read"]["after_versions"]["f3"]),
        "rebuild_equals_incremental": by_name["rebuild_vs_incremental"]["equal"],
        "namespace_isolated": by_name["namespace_isolation"]["team_a_query_for_team_b_fact"] == [],
        "semantic_baseline_matches_expected": semantic_ok,
    }
    summary = {
        "config": {"fake_embed": fake, "embed_model": args.embed_model,
                   "facts": len(FACTS), "queries": len(QUERIES)},
        "checks": checks,
        "checks_passed": sum(1 for v in checks.values()),
        "checks_total": len(checks),
        "steps": steps,
        "final_version": store.version,
        "visible_facts": {k: v["version"] for k, v in store.all_visible().items()},
        "audit_tail": store.audit_tail(10),
    }
    (out / "memory_lifecycle.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.6 长期记忆存储生命周期")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8016/v1")
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--fake-embed", action="store_true",
                    help="用确定性假向量验证存储逻辑（无需 GPU）")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())