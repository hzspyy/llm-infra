#!/usr/bin/env python3
"""L9.5 任务 D 的模型对照：LangGraph 的节点重放/待提交写入 vs 自写存储的事务提交。

两套模型对「节点已经产生副作用、但运行时还没记下它」这个窗口给出不同答案：

* **LangGraph**：状态由 checkpointer 持久化，节点失败后从上一个 checkpoint **重放该节点**。
  因此副作用本身必须幂等，否则重放会执行第二次。
* **自写存储**（`durable_task_store.py`）：副作用与「我知道它发生了」在同一事务提交，
  未提交就等于没发生；恢复时靠业务幂等键复用已提交结果。

本脚本用同一张三节点图跑三组对照，并把结论写成 JSON：

1. ``naive``：工具节点先写外部账本、再抛异常；从 checkpoint 恢复后账本会出现 **2** 条；
2. ``idempotent``：工具按业务键去重；恢复后账本仍是 **1** 条；
3. ``commit_then_crash``：把「写账本」与「记状态」放进同一次提交（自写模型），
   崩溃发生在提交前 → 账本 0 条。

运行需要隔离环境（本机装在 `/Volumes/data/venvs/langgraph`）::

    /Volumes/data/venvs/langgraph/bin/python labs/L9/langgraph_compare.py replay --out out/9.5/langgraph
    /Volumes/data/venvs/langgraph/bin/python labs/L9/langgraph_compare.py checkpoints --out out/9.5/langgraph
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time
from typing import TypedDict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

LEDGER_SCHEMA = """CREATE TABLE IF NOT EXISTS side_effects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key TEXT, node TEXT, created_at REAL)"""


def ledger_append(db: pathlib.Path, idem_key: str | None, node: str) -> int:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.executescript(LEDGER_SCHEMA)
        if idem_key is not None:
            row = conn.execute("SELECT COUNT(*) FROM side_effects WHERE idem_key=?",
                               (idem_key,)).fetchone()
            if row[0] > 0:
                return conn.execute("SELECT COUNT(*) FROM side_effects").fetchone()[0]
        conn.execute("INSERT INTO side_effects(idem_key, node, created_at) VALUES (?,?,?)",
                     (idem_key, node, time.time()))
        return conn.execute("SELECT COUNT(*) FROM side_effects").fetchone()[0]
    finally:
        conn.close()


def ledger_count(db: pathlib.Path) -> int:
    if not db.exists():
        return 0
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(LEDGER_SCHEMA)
        return conn.execute("SELECT COUNT(*) FROM side_effects").fetchone()[0]
    finally:
        conn.close()


class State(TypedDict):
    prompt: str
    tool_done: bool
    answer: str


def _saver(path: pathlib.Path):
    """构造一个可用的 SQLite checkpointer（`from_conn_string` 返回的是上下文管理器）。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(ledger: pathlib.Path, mode: str):
    """三节点图：plan → tool → finish；tool 是产生外部副作用的那一步。"""
    from langgraph.graph import END, START, StateGraph

    class Boom(Exception):
        pass

    def plan(state: State) -> State:
        return {**state, "prompt": state.get("prompt") or "6*7"}

    def tool(state: State) -> State:
        idem = f"tool-{mode}" if mode == "idempotent" else None
        ledger_append(ledger, idem, "tool")
        # 两档都在副作用之后失败：差别只在工具是否按业务键去重
        raise Boom("node failed after producing the side effect")

    def finish(state: State) -> State:
        return {**state, "answer": "42"}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("tool", tool)
    g.add_node("finish", finish)
    g.add_edge(START, "plan")
    g.add_edge("plan", "tool")
    g.add_edge("tool", "finish")
    g.add_edge("finish", END)
    return g


def run_once(out: pathlib.Path, mode: str) -> dict:
    """第一次运行：naive 档在副作用之后失败；其余档正常完成。"""
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"{mode}.ledger.sqlite"
    ckpt = out / f"{mode}.checkpoints.sqlite"
    for p in (ledger, ckpt):
        if p.exists():
            p.unlink()
    graph = build_graph(ledger, mode).compile(checkpointer=_saver(ckpt))
    cfg = {"configurable": {"thread_id": f"t-{mode}"}}
    error = None
    try:
        graph.invoke({"prompt": "6*7", "tool_done": False, "answer": ""}, cfg)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"mode": mode, "error": error, "effects": ledger_count(ledger),
            "ledger": str(ledger), "checkpoints": str(ckpt), "config": cfg}


def resume(out: pathlib.Path, mode: str, run: dict) -> dict:
    """从 checkpoint 恢复：LangGraph 会重放未完成的节点。"""
    graph = build_graph(pathlib.Path(run["ledger"]), mode).compile(
        checkpointer=_saver(pathlib.Path(run["checkpoints"])))
    cfg = run["config"]
    state_before = graph.get_state(cfg)
    error = None
    result_state = None
    try:
        result_state = graph.invoke(None, cfg)     # None = 从上次 checkpoint 继续
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"mode": mode, "error": error, "effects_after_resume": ledger_count(pathlib.Path(run["ledger"])),
            "state_before": {"next": list(state_before.next), "values": state_before.values},
            "state_after": result_state}


def commit_then_crash(out: pathlib.Path) -> dict:
    """自写模型：副作用与状态同事务；提交前崩溃 → 账本 0 条。"""
    import os

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    if os.environ.get("L95_CHILD") == "1":
        from durable_task_store import Store

        db = out / "single.sqlite"
        st = Store(db, "w1", 30.0)
        st.init_session("lg")
        st.run_node("lg", "fetch", "key-fetch", crash_before_commit=True)   # os._exit(137)
        return {"unreachable": True}
    import subprocess

    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "L95_CHILD": "1"}
    proc = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "commit-then-crash",
                           "--out", str(out)], env=env, capture_output=True, text=True)
    db = out / "single.sqlite"
    effects = 0
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            effects = conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
        finally:
            conn.close()
    return {"exit_code": proc.returncode, "effects_after_crash": effects,
            "note": "副作用与状态在同一事务；提交前崩溃则两者都不存在"}


def cmd_replay(args) -> int:
    out = pathlib.Path(args.out)
    results = {}
    for mode in ("naive", "idempotent"):
        run = run_once(out, mode)
        if run["error"]:
            run["resume"] = resume(out, mode, run)
        results[mode] = run
    results["commit_then_crash"] = commit_then_crash(out / "single-tx")

    naive, idem = results["naive"], results["idempotent"]
    checks = [
        {"name": "first_run_side_effect_before_failure",
         "expected": "naive 档首次运行在抛异常前已写下 1 条副作用",
         "got": naive["effects"],
         "match": naive["effects"] == 1 and bool(naive["error"])},
        {"name": "langgraph_replay_repeats_side_effect",
         "expected": "LangGraph 从 checkpoint 恢复会重放节点，账本变 2 条",
         "got": naive["resume"]["effects_after_resume"],
         "match": naive["resume"]["effects_after_resume"] == 2},
        {"name": "idempotent_tool_survives_replay",
         "expected": "同样重放一次，工具按业务键去重时账本仍是 1 条",
         "got": {"first": idem["effects"],
                 "after_resume": idem.get("resume", {}).get("effects_after_resume")},
         "match": (idem["effects"] == 1
                   and idem.get("resume", {}).get("effects_after_resume") == 1)},
        {"name": "single_transaction_has_no_effect_before_commit",
         "expected": "自写模型提交前崩溃 → 0 条副作用、退出码 137",
         "got": {"exit": results["commit_then_crash"]["exit_code"],
                 "effects": results["commit_then_crash"]["effects_after_crash"]},
         "match": (results["commit_then_crash"]["exit_code"] == 137
                   and results["commit_then_crash"]["effects_after_crash"] == 0)},
    ]
    report = {"framework": {"langgraph": _version("langgraph"),
                            "langgraph_checkpoint_sqlite": _version("langgraph-checkpoint-sqlite")},
              "results": results, "checks": checks,
              "all_match": all(c["match"] for c in checks),
              "conclusion": ("可恢复不等于副作用只执行一次：LangGraph 的恢复语义是重放未完成节点，"
                             "因此节点副作用必须自带幂等键；自写存储把副作用与状态同事务提交，"
                             "代价是提交前崩溃会丢掉已做的工作（需要重算），换来的是不会重复。"
                             "两者都需要业务幂等键才能跨进程安全重试")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "langgraph_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {c['got']}")
    print("all_match:", report["all_match"])
    return 0


def cmd_checkpoints(args) -> int:
    """把 checkpoint 历史导出成可读结构，作为模型对照的原始材料。"""
    out = pathlib.Path(args.out)
    run = run_once(out, "naive")
    if run["error"]:
        run["resume"] = resume(out, "naive", run)
    graph = build_graph(pathlib.Path(run["ledger"]), "naive").compile(
        checkpointer=_saver(pathlib.Path(run["checkpoints"])))
    cfg = run["config"]
    history = []
    for snap in graph.get_state_history(cfg):
        history.append({
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
            "step": snap.metadata.get("step"),
            "source": snap.metadata.get("source"),
            "next": list(snap.next),
            "values": snap.values,
            "tasks": [{"name": getattr(t, "name", None),
                      "error": str(t.error) if getattr(t, "error", None) else None,
                      "writes": [str(w) for w in (getattr(t, "writes", None) or [])]}
                      for t in (snap.tasks or [])],
            "interrupts": [str(i) for i in (snap.interrupts or [])],
            "created_at": getattr(snap, "created_at", None),
        })
    conn = sqlite3.connect(run["checkpoints"])
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        conn.close()
    report = {"run": run, "checkpoint_history": history,
              "checkpoint_tables": counts,
              "note": ("history 的 source/next 说明恢复点是「哪个节点待执行」；"
                       "每个 task 的 writes 是该节点待提交的写入——进程在写入与 checkpoint 提交之间死掉时，"
                       "这些写入会被重放，这正是副作用必须幂等的原因")}
    (out / "langgraph_checkpoints.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    print(json.dumps({"tables": counts, "history_len": len(history),
                      "last": history[0] if history else None}, ensure_ascii=False, indent=1)[:1200])
    return 0


def _version(pkg: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 LangGraph 重放模型对照")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("replay", cmd_replay), ("checkpoints", cmd_checkpoints),
                     ("commit-then-crash", lambda a: _commit_child(a))):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


def _commit_child(args) -> int:
    commit_then_crash(pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
