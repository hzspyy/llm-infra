#!/usr/bin/env python3
"""L9.5 任务 A/B/D：持久任务存储、事务 checkpoint、worker 租约与幂等副作用。

与 `agent_runtime.py`（进程内状态机）的区别：这里的状态**全部落在 SQLite 的事务边界里**，
并且用真实进程故障（`os._exit`）而不是抛异常来验证恢复。

三张表与一条不变量：

* ``tasks``      —— 任务图节点：``(session, node)`` 唯一，带 ``state``/``attempt``/``result``；
* ``effects``    —— 副作用结果：``idem_key`` 唯一，与节点完成写进**同一事务**；
* ``leases``     —— worker 租约：带 ``fencing_token``，过期后新 worker 用更大的 token 接管，
  旧 worker 的迟到写入必须被拒绝。

核心不变量：**"副作用已发生"与"运行时知道它已发生"必须在同一个事务里提交**。
把它拆成两个事务就会出现"副作用完成了、但恢复时不知道"的窗口。

用法::

    python labs/L9/durable_task_store.py execute --out DIR [--crash-after NODE]
    python labs/L9/durable_task_store.py resume  --out DIR
    python labs/L9/durable_task_store.py leases  --out DIR
    python labs/L9/durable_task_store.py window  --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys
import time

GRAPH = [
    # node, 依赖, 幂等键
    {"node": "fetch", "dep": None, "idem": "key-fetch"},
    {"node": "compute", "dep": "fetch", "idem": "key-compute"},
    {"node": "commit", "dep": "compute", "idem": "key-commit"},
]


class Store:
    def __init__(self, path: pathlib.Path, worker: str, lease_s: float = 30.0):
        self.path = path
        self.worker = worker
        self.lease_s = lease_s
        self.conn = sqlite3.connect(str(path), isolation_level=None, timeout=10)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._schema()

    def _schema(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS tasks(
            session TEXT NOT NULL, node TEXT NOT NULL, dep TEXT, idem_key TEXT NOT NULL,
            state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, result TEXT,
            updated_at REAL NOT NULL, PRIMARY KEY(session, node))""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS effects(
            idem_key TEXT PRIMARY KEY, payload TEXT, result TEXT, committed_at REAL,
            worker TEXT, fencing_token INTEGER)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS leases(
            session TEXT PRIMARY KEY, worker TEXT NOT NULL, fencing_token INTEGER NOT NULL,
            expires_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, session TEXT,
            node TEXT, attempt INTEGER, detail TEXT)""")

    # -- 事件 ---------------------------------------------------------------
    def log(self, kind: str, session: str, node: str | None = None, attempt: int | None = None,
            detail: str = ""):
        self.conn.execute("INSERT INTO events(ts, kind, session, node, attempt, detail)"
                          " VALUES (?,?,?,?,?,?)", (time.time(), kind, session, node, attempt, detail))

    def events(self, n: int = 40) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, kind, session, node, attempt, detail FROM events ORDER BY id DESC LIMIT ?",
            (n,)).fetchall()
        return [{"ts": r[0], "kind": r[1], "session": r[2], "node": r[3], "attempt": r[4],
                 "detail": r[5]} for r in rows]

    # -- 任务图 -------------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, time.time()))

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def init_session(self, session: str):
        now = time.time()
        for g in GRAPH:
            self.conn.execute(
                "INSERT OR IGNORE INTO tasks(session, node, dep, idem_key, state, attempt, updated_at)"
                " VALUES (?,?,?,?,?,0,?)", (session, g["node"], g["dep"], g["idem"], "pending", now))
        self.log("session_init", session, detail=f"{len(GRAPH)} nodes")

    def state(self, session: str) -> dict[str, str]:
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT node, state FROM tasks WHERE session=? ORDER BY rowid", (session,)).fetchall()}

    def pending(self, session: str) -> list[tuple[str, str, str]]:
        """按依赖顺序返回待执行节点（只返回依赖已完成的）。"""
        st = self.state(session)
        out = []
        for g in GRAPH:
            if st.get(g["node"]) == "done":
                continue
            if g["dep"] is not None and st.get(g["dep"]) != "done":
                continue
            out.append((g["node"], g["idem"], g["dep"]))
        return out

    # -- 租约与 fencing ------------------------------------------------------
    def claim(self, session: str) -> int:
        now = time.time()
        row = self.conn.execute("SELECT worker, fencing_token, expires_at FROM leases WHERE session=?",
                                (session,)).fetchone()
        if row and row[2] > now and row[0] != self.worker:
            raise RuntimeError(f"lease held by {row[0]} until {row[2]:.1f}")
        token = (row[1] if row else 0) + 1
        self.conn.execute(
            "INSERT INTO leases(session, worker, fencing_token, expires_at, updated_at)"
            " VALUES (?,?,?,?,?) ON CONFLICT(session) DO UPDATE SET"
            " worker=excluded.worker, fencing_token=excluded.fencing_token,"
            " expires_at=excluded.expires_at, updated_at=excluded.updated_at",
            (session, self.worker, token, now + self.lease_s, now))
        self.log("lease_claim", session, detail=f"worker={self.worker} token={token}")
        return token

    def _token_valid(self, session: str, token: int) -> bool:
        row = self.conn.execute(
            "SELECT worker, fencing_token FROM leases WHERE session=?", (session,)).fetchone()
        if not row:
            return False
        return row[0] == self.worker and row[1] == token

    # -- 执行一个步骤：副作用 + 状态在同一事务 ------------------------------
    def run_node(self, session: str, node: str, idem_key: str, *, two_transaction: bool = False,
                 crash_before_commit: bool = False, fail_checkpoint: bool = False,
                 require_dep: bool = False) -> dict:
        """执行一个节点。

        ``two_transaction=True`` 时副作用先独立提交、节点状态再提交——中间崩溃就留下
        "副作用完成了，但运行时不知道"的窗口；默认的单事务模式把两者放进同一次提交。
        """
        # 依赖检查：乱序完成必须在**开始任何副作用之前**被拒绝
        st = self.state(session)
        g = next((x for x in GRAPH if x["node"] == node), None)
        if require_dep and g and g["dep"] is not None and st.get(g["dep"]) != "done":
            self.log("dep_not_ready", session, node, detail=f"dep={g['dep']} state={st.get(g['dep'])}")
            raise RuntimeError(f"dependency {g['dep']} not done; refusing to run {node}")
        if require_dep and st.get(node) == "done":
            self.log("duplicate_completion_ignored", session, node, detail="already done")
            return {"node": node, "attempt": None, "result": "already-done", "idempotent": True}
        token = self.claim(session)
        attempt = self.conn.execute("SELECT attempt FROM tasks WHERE session=? AND node=?",
                                    (session, node)).fetchone()[0] + 1
        self.log("node_start", session, node, attempt, f"token={token}")

        if two_transaction:
            result = self._execute_effect(idem_key, node, token)          # 自动提交
            if crash_before_commit:
                self.log("crash_between_transactions", session, node, attempt)
                sys.stdout.flush()
                os._exit(137)
            self._commit_node_state(session, node, attempt, token, result)
            return {"node": node, "attempt": attempt, "result": result, "two_transaction": True}

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT worker, fencing_token FROM leases WHERE session=?", (session,)).fetchone()
            if not row or row[0] != self.worker or row[1] != token:
                self.conn.execute("ROLLBACK")
                self.log("fencing_rejected", session, node, attempt,
                         f"stale token={token} current={row[1] if row else None}")
                raise PermissionError("stale fencing token: refusing to commit")
            result = self._execute_effect(idem_key, node, token, in_transaction=True)
            if fail_checkpoint:
                # checkpoint 写失败：副作用已在同一事务里插入，必须随事务一起回滚
                self.log("checkpoint_write_failed", session, node, attempt, "injected")
                raise sqlite3.OperationalError("injected checkpoint write failure")
            if crash_before_commit:
                self.log("crash_before_commit", session, node, attempt, "os._exit(137)")
                sys.stdout.flush()
                os._exit(137)                     # 未 COMMIT，整个事务回滚
            self.conn.execute(
                "UPDATE tasks SET state='done', attempt=?, result=?, updated_at=?"
                " WHERE session=? AND node=?", (attempt, result, time.time(), session, node))
            self.conn.execute("INSERT INTO events(ts, kind, session, node, attempt, detail)"
                              " VALUES (?,?,?,?,?,?)",
                              (time.time(), "node_done", session, node, attempt, result))
            self.conn.execute("COMMIT")
        except PermissionError:
            raise
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return {"node": node, "attempt": attempt, "result": result}

    def _execute_effect(self, idem_key: str, node: str, token: int,
                        *, in_transaction: bool = False) -> str:
        row = self.conn.execute("SELECT result FROM effects WHERE idem_key=?", (idem_key,)).fetchone()
        if row:
            self.log("effect_reused", "", node, None, f"{idem_key} -> {row[0]}")
            return row[0]
        result = f"effect({node})"
        self.conn.execute("INSERT INTO effects(idem_key, payload, result, committed_at, worker,"
                          " fencing_token) VALUES (?,?,?,?,?,?)",
                          (idem_key, node, result, time.time(), self.worker, token))
        if not in_transaction:
            self.log("effect_committed", "", node, None, f"{idem_key} -> {result}")
        return result

    def _commit_node_state(self, session: str, node: str, attempt: int, token: int, result: str):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT worker, fencing_token FROM leases WHERE session=?", (session,)).fetchone()
            if not row or row[0] != self.worker or row[1] != token:
                self.conn.execute("ROLLBACK")
                self.log("fencing_rejected", session, node, attempt,
                         f"stale token={token} current={row[1] if row else None}")
                raise PermissionError("stale fencing token: refusing to commit")
            self.conn.execute(
                "UPDATE tasks SET state='done', attempt=?, result=?, updated_at=?"
                " WHERE session=? AND node=?", (attempt, result, time.time(), session, node))
            self.conn.execute("COMMIT")
        except PermissionError:
            raise
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def effect_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0]

    def duplicate_effects(self) -> int:
        rows = self.conn.execute("SELECT idem_key, COUNT(*) c FROM effects GROUP BY idem_key"
                                 " HAVING c>1").fetchall()
        return len(rows)


def execute(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = args.session
    store = Store(out / "tasks.db", worker=args.worker, lease_s=args.lease)
    store.init_session(session)
    done = 0
    while True:
        todo = store.pending(session)          # 每轮重新求值：依赖完成后后继才进入待执行
        if not todo:
            break
        node, idem, _dep = todo[0]
        crash = bool(args.crash_after) and node == args.crash_after
        res = store.run_node(session, node, idem, two_transaction=args.two_transaction,
                             crash_before_commit=crash)
        done += 1
        print(json.dumps({"node": res["node"], "attempt": res["attempt"], "result": res["result"],
                          "state": store.state(session)}, ensure_ascii=False), flush=True)
    summary = {"mode": "execute", "session": session, "worker": args.worker,
               "state": store.state(session), "effects": store.effect_count(),
               "duplicates": store.duplicate_effects(), "nodes_run": done}
    (out / "last_execute.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def resume(args) -> int:
    out = pathlib.Path(args.out)
    store = Store(out / "tasks.db", worker=args.worker, lease_s=args.lease)
    session = args.session
    before = store.state(session)
    effects_before = store.effect_count()
    ran = []
    while True:
        todo = store.pending(session)
        if not todo:
            break
        node, idem, _dep = todo[0]
        res = store.run_node(session, node, idem)
        ran.append(res["node"])
    summary = {"mode": "resume", "session": session, "worker": args.worker,
               "state_before": before, "state_after": store.state(session),
               "effects_before": effects_before, "effects_after": store.effect_count(),
               "duplicates": store.duplicate_effects(), "nodes_run": ran}
    (out / "last_resume.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def leases(args) -> int:
    """租约过期与 fencing：旧 worker 的迟到提交必须被拒绝。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "leases.db"
    if db.exists():
        db.unlink()
    session = "lease-demo"
    a = Store(db, worker="worker-A", lease_s=1.0)
    a.init_session(session)
    token_a = a.claim(session)
    a._execute_effect(GRAPH[0]["idem"], GRAPH[0]["node"], token_a)
    time.sleep(1.5)                      # 等 A 的租约过期
    b = Store(db, worker="worker-B", lease_s=30.0)
    token_b = b.claim(session)
    # B 用更大的 token 接管并提交
    b._commit_node_state(session, GRAPH[0]["node"], 1, token_b, "effect(fetch)")
    # A 拿着过期 token 迟到提交 → 必须被拒绝
    rejected = None
    try:
        a._commit_node_state(session, GRAPH[0]["node"], 1, token_a, "effect(fetch)")
    except PermissionError as exc:
        rejected = str(exc)
    summary = {
        "mode": "leases", "session": session,
        "token_a": token_a, "token_b": token_b,
        "stale_commit_rejected": rejected,
        "state_after": b.state(session),
        "events": b.events(12),
    }
    (out / "leases.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def inspect(args) -> int:
    """打印某个会话的状态、副作用数与重复数（用于崩溃后取证）。"""
    store = Store(pathlib.Path(args.out) / "tasks.db", worker="inspector")
    info = {"session": args.session, "state": store.state(args.session),
            "effects": store.effect_count(), "duplicates": store.duplicate_effects(),
            "events": store.events(8)}
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


def window(args) -> int:
    """同一事务 vs 两个事务：用子进程注入真实崩溃，再恢复。

    崩溃点在**第一个节点的副作用提交之后**，两种模式的区别因此可以被观察到：
    两事务模式留下"副作用已提交、状态未提交"的窗口；单事务模式两者一起回滚。
    """
    import subprocess

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode, extra in (("two_transaction", ["--two-transaction"]), ("single_transaction", [])):
        d = out / mode
        if d.exists():
            import shutil
            shutil.rmtree(d)
        d.mkdir(parents=True)
        session = f"window-{mode}"
        py = sys.executable
        script = str(pathlib.Path(__file__))

        def cmd(sub, *extra):
            args = [py, script, sub, "--out", str(d)]
            if sub != "inspect":
                args += ["--session", session, "--worker", "w1"]
            else:
                args += ["--session", session]
            return args + list(extra)

        crash = subprocess.run(cmd("execute", "--crash-after", "fetch", *extra),
                               capture_output=True, text=True)
        after_crash = json.loads(subprocess.run(cmd("inspect"), capture_output=True, text=True).stdout)
        resume = subprocess.run(cmd("resume"), capture_output=True, text=True)
        final = json.loads(subprocess.run(cmd("inspect"), capture_output=True, text=True).stdout)
        results[mode] = {
            "crash_exit_code": crash.returncode,
            "effects_after_crash": after_crash["effects"],
            "duplicates_after_crash": after_crash["duplicates"],
            "state_after_crash": after_crash["state"],
            "resume_exit_code": resume.returncode,
            "effects_after_resume": final["effects"],
            "duplicates_after_resume": final["duplicates"],
            "state_after_resume": final["state"],
            "note": ("副作用独立提交：崩溃留下『已做但运行时不知道』的窗口，恢复靠幂等键复用"
                     if mode == "two_transaction" else
                     "副作用与节点状态同一事务：崩溃时一起回滚，恢复后不存在该窗口"),
        }
    summary = {"mode": "window", "results": results}
    (out / "window.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def checkpoint_fail(args) -> int:
    """checkpoint 写失败：副作用必须随事务回滚，重试后只提交一次。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "store.sqlite"
    st = Store(db, "w1", args.lease)
    st.init_session(args.session)
    result: dict = {"injected": None, "after_failure": {}, "after_retry": {}}
    try:
        st.run_node(args.session, "fetch", "key-fetch", fail_checkpoint=True)
        result["injected"] = "no error raised"
    except sqlite3.OperationalError as exc:
        result["injected"] = f"{type(exc).__name__}: {exc}"
    result["after_failure"] = {"effects": st.effect_count(), "state": st.state(args.session)}
    st.run_node(args.session, "fetch", "key-fetch")
    result["after_retry"] = {"effects": st.effect_count(), "state": st.state(args.session)}
    result["ok"] = (result["after_failure"]["effects"] == 0
                    and result["after_failure"]["state"]["fetch"] != "done"
                    and result["after_retry"]["effects"] == 1
                    and result["after_retry"]["state"]["fetch"] == "done")
    (out / "checkpoint_fail.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def duplicate(args) -> int:
    """重复完成：同一幂等键两次执行只产生一条副作用；换键重试则两条。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    same_db = out / "same_key.sqlite"
    st = Store(same_db, "w1", args.lease)
    st.init_session("dup-same")
    a = st.run_node("dup-same", "fetch", "key-fetch")
    b = st.run_node("dup-same", "fetch", "key-fetch")     # 同一键再来一次（模拟重复回包）
    same = {"first": a, "second": b, "effects": st.effect_count(),
            "duplicate_effects": st.duplicate_effects()}
    diff_db = out / "new_key.sqlite"
    st2 = Store(diff_db, "w1", args.lease)
    st2.init_session("dup-new")
    st2.run_node("dup-new", "fetch", "key-fetch")
    st2.run_node("dup-new", "fetch", "key-fetch-attempt2")   # 客户端换了业务键：真重复
    diff = {"effects": st2.effect_count()}
    result = {"same_key": same, "new_key_on_retry": diff,
              "ok": same["effects"] == 1 and same["duplicate_effects"] == 0 and diff["effects"] == 2}
    (out / "duplicate.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def ordering(args) -> int:
    """乱序完成：依赖未满足时必须在产生副作用之前拒绝；重复完成是幂等 no-op。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "store.sqlite"
    st = Store(db, "w1", args.lease)
    st.init_session("order")
    result: dict = {}
    result["pending_at_start"] = [p[0] for p in st.pending("order")]
    try:
        st.run_node("order", "commit", "key-commit", require_dep=True)
        result["out_of_order"] = "no error raised"
    except RuntimeError as exc:
        result["out_of_order"] = f"{type(exc).__name__}: {exc}"
    result["effects_after_refusal"] = st.effect_count()
    st.run_node("order", "fetch", "key-fetch", require_dep=True)
    result["pending_after_fetch"] = [p[0] for p in st.pending("order")]
    st.run_node("order", "compute", "key-compute", require_dep=True)
    result["pending_after_compute"] = [p[0] for p in st.pending("order")]
    st.run_node("order", "commit", "key-commit", require_dep=True)
    dup = st.run_node("order", "commit", "key-commit", require_dep=True)
    result["duplicate_completion"] = dup
    result["final_state"] = st.state("order")
    result["effects"] = st.effect_count()
    result["ok"] = (result["effects_after_refusal"] == 0
                    and "not done" in result["out_of_order"]
                    and result["final_state"] == {"fetch": "done", "compute": "done", "commit": "done"}
                    and result["effects"] == 3
                    and dup.get("idempotent") is True)
    (out / "ordering.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def versions(args) -> int:
    """版本不兼容：记录 schema 哈希，恢复时不一致必须拒绝。"""
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "store.sqlite"
    st = Store(db, "w1", args.lease)
    st.init_session("ver")
    st.set_meta("tool_schema_hash", args.recorded_hash)
    st.run_node("ver", "fetch", "key-fetch")
    recorded = st.get_meta("tool_schema_hash")
    st2 = Store(db, "w1", args.lease)
    current = args.current_hash
    compatible = recorded == current
    decision = "proceed" if compatible else "refused"
    result = {
        "recorded": recorded, "current": current, "compatible": compatible,
        "resume_decision": decision,
        "reason": None if compatible else "tool schema changed",
        "effects_before_resume": st.effect_count(),
        "ok": (not compatible and decision == "refused"),
    }
    (out / "versions.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 持久任务存储")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(out=None, session="sess-1", worker="w1", lease=30.0)

    p = sub.add_parser("execute")
    for k, v in common.items():
        if k != "out":
            p.add_argument(f"--{k}", default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--crash-after", default=None,
                   help="在这个节点「副作用已提交、状态未提交」时强制退出（os._exit 137）")
    p.add_argument("--two-transaction", action="store_true",
                   help="反例：副作用独立提交、节点状态另一次提交")
    p.set_defaults(func=execute)

    p = sub.add_parser("resume")
    for k, v in common.items():
        if k != "out":
            p.add_argument(f"--{k}", default=v)
    p.add_argument("--out", required=True)
    p.set_defaults(func=resume)

    for name, fn in (("leases", leases), ("window", window)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.set_defaults(func=fn)

    p = sub.add_parser("checkpoint-fail")
    p.add_argument("--out", required=True)
    p.add_argument("--session", default="ckpt")
    p.add_argument("--lease", type=float, default=30.0)
    p.set_defaults(func=checkpoint_fail)

    p = sub.add_parser("duplicate")
    p.add_argument("--out", required=True)
    p.add_argument("--lease", type=float, default=30.0)
    p.set_defaults(func=duplicate)

    p = sub.add_parser("ordering")
    p.add_argument("--out", required=True)
    p.add_argument("--lease", type=float, default=30.0)
    p.set_defaults(func=ordering)

    p = sub.add_parser("versions")
    p.add_argument("--out", required=True)
    p.add_argument("--lease", type=float, default=30.0)
    p.add_argument("--recorded-hash", default="aaaa1111")
    p.add_argument("--current-hash", default="bbbb2222")
    p.set_defaults(func=versions)

    p = sub.add_parser("inspect")
    p.add_argument("--out", required=True)
    p.add_argument("--session", default="sess-1")
    p.set_defaults(func=inspect)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())