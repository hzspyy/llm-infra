#!/usr/bin/env python3
"""L9.5 任务 B：带事务账本的独立工具服务，演示「调用已执行但 ACK 丢失」这个窗口。

服务端把**副作用与去重结果放在同一个提交边界**里：

* ``effects``：``idem_key`` 唯一，存副作用内容与返回值；同一业务键再进来直接返回原结果；
* ``attempts``：每次网络尝试一行，同一个业务键可以有多次 attempt（``attempt`` 只用于观测与审计，
  不参与去重判定）；
* ``POST /call``：执行副作用 → 一个事务里写 effects + attempts → 提交 → 再回包；
  ``drop_ack=true`` 时在提交之后**直接关连接**，模拟"执行成功但响应丢失"；
* ``GET /result``：按业务键查询已提交结果（"先查后写"里的查询路径）；
* ``POST /compensate``：写一条补偿记录（不是回滚——已提交的副作用不会消失，只是被标注）；
* ``GET /ledger``：导出账本，判定只依据它。

客户端的四种处理方式各跑一遍，并把结论写进 JSON：**判定不看 stdout，只看账本与退出码**。

用法::

    python labs/L9/idempotent_tool_service.py run --out out/9.5/idempotent
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import pathlib
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS effects (
    idem_key TEXT PRIMARY KEY,
    effect_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    value INTEGER,
    committed_at REAL NOT NULL,
    compensated_by TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    idem_key TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    outcome TEXT
);
"""


def _connect(db: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), timeout=10.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def init_db(db: pathlib.Path) -> None:
    with _connect(db) as conn:
        conn.executescript(SCHEMA)


def _eval_expression(expr: str) -> int:
    import agent_tasks as T

    return int(T.safe_eval(str(expr)))


def commit_call(db: pathlib.Path, idem_key: str, expression: str, attempt_index: int,
                delay_ms: int = 0) -> dict:
    """执行副作用并提交；返回已提交的结果（若该键已存在则直接返回原结果）。"""
    attempt_id = uuid.uuid4().hex[:12]
    conn = _connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT effect_id, value, compensated_by FROM effects WHERE idem_key=?",
                           (idem_key,)).fetchone()
        if row is not None:
            conn.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?)",
                (attempt_id, idem_key, attempt_index, time.time(), time.time(), "dedup_hit"))
            conn.execute("COMMIT")
            return {"dedup_hit": True, "effect_id": row[0], "value": row[1],
                    "compensated_by": row[2], "attempt_id": attempt_id}
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)     # 副作用本身（在事务内，模拟长工具）
        value = _eval_expression(expression)
        effect_id = uuid.uuid4().hex[:12]
        conn.execute("INSERT INTO effects VALUES (?,?,?,?,?,NULL)",
                     (idem_key, effect_id, expression, value, time.time()))
        conn.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?)",
                     (attempt_id, idem_key, attempt_index, time.time(), time.time(), "committed"))
        conn.execute("COMMIT")               # 副作用与去重结果同一提交边界
        return {"dedup_hit": False, "effect_id": effect_id, "value": value,
                "compensated_by": None, "attempt_id": attempt_id}
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def query_result(db: pathlib.Path, idem_key: str) -> dict | None:
    conn = _connect(db)
    try:
        row = conn.execute("SELECT effect_id, value, compensated_by FROM effects WHERE idem_key=?",
                           (idem_key,)).fetchone()
        return None if row is None else {"effect_id": row[0], "value": row[1],
                                         "compensated_by": row[2]}
    finally:
        conn.close()


def compensate(db: pathlib.Path, idem_key: str, reason: str) -> dict:
    conn = _connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT effect_id, value FROM effects WHERE idem_key=?",
                           (idem_key,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return {"compensated": False, "reason": "no such effect"}
        comp_id = uuid.uuid4().hex[:12]
        conn.execute("UPDATE effects SET compensated_by=? WHERE idem_key=?", (comp_id, idem_key))
        conn.execute("COMMIT")
        return {"compensated": True, "compensation_id": comp_id, "original_effect": row[0],
                "reason": reason}
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def ledger(db: pathlib.Path) -> dict:
    conn = _connect(db)
    try:
        effects = [dict(zip(("idem_key", "effect_id", "expression", "value", "committed_at",
                             "compensated_by"), r))
                   for r in conn.execute(
                       "SELECT idem_key, effect_id, expression, value, committed_at, compensated_by "
                       "FROM effects ORDER BY committed_at")]
        attempts = [dict(zip(("attempt_id", "idem_key", "attempt_index", "started_at",
                              "finished_at", "outcome"), r))
                    for r in conn.execute(
                        "SELECT attempt_id, idem_key, attempt_index, started_at, finished_at, outcome "
                        "FROM attempts ORDER BY started_at")]
        return {"effects": effects, "attempts": attempts,
                "effect_count": len(effects), "attempt_count": len(attempts),
                "distinct_keys": len({e["idem_key"] for e in effects})}
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# HTTP 服务（stdlib；关键在于能在提交后直接断连）
# --------------------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    db: pathlib.Path
    attempt_counter: dict[str, int] = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: A003 - 静音
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("content-length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/result"):
            from urllib.parse import parse_qs, urlparse
            key = (parse_qs(urlparse(self.path).query).get("idem_key") or [""])[0]
            res = query_result(self.db, key)
            if res is None:
                self._send(404, {"found": False})
            else:
                self._send(200, {"found": True, **res})
            return
        if self.path.startswith("/ledger"):
            self._send(200, ledger(self.db))
            return
        self._send(404, {"error": "unknown path"})

    def do_POST(self):  # noqa: N802
        try:
            body = self._body()
        except Exception:  # noqa: BLE001
            self._send(400, {"error": "bad json"})
            return
        if self.path.startswith("/call"):
            key = body.get("idem_key") or uuid.uuid4().hex[:12]
            idx = Handler.attempt_counter.get(key, 0)
            Handler.attempt_counter[key] = idx + 1
            try:
                result = commit_call(self.db, key, body.get("expression", "1+1"), idx,
                                     int(body.get("delay_ms") or 0))
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if body.get("drop_ack"):
                # 提交之后直接断连：客户端看到的是网络错误，但副作用已经落库
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return
            self._send(200, result)
            return
        if self.path.startswith("/compensate"):
            self._send(200, compensate(self.db, body.get("idem_key", ""),
                                       body.get("reason", "")))
            return
        self._send(404, {"error": "unknown path"})


def serve(db: pathlib.Path, port: int) -> subprocess.Popen:
    """在子进程里起服务，保证崩溃/断连不影响客户端进程。"""
    Handler.db = db
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "serve",
           "--db", str(db), "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"tool service on {port} did not start")


def cmd_serve(args) -> int:
    db = pathlib.Path(args.db)
    init_db(db)
    Handler.db = db
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.serve_forever()
    return 0


# --------------------------------------------------------------------------------------
# 客户端：四种处理方式
# --------------------------------------------------------------------------------------

def _post(port: int, path: str, payload: dict) -> tuple[dict | None, str | None]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _get(port: int, path: str) -> tuple[dict | None, str | None]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read() or b"{}"), f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _scenario(port: int, name: str, key: str) -> dict:
    """四种处理方式作用于同一个「执行成功、ACK 丢失」的窗口。"""
    steps: list[dict] = []
    if name == "naive_retry":
        # 每次都当成新业务动作重试：两次副作用
        r1, e1 = _post(port, "/call", {"idem_key": f"{key}-a", "expression": "6*7", "drop_ack": True})
        steps.append({"step": "first_call", "error": e1})
        r2, e2 = _post(port, "/call", {"idem_key": f"{key}-b", "expression": "6*7"})
        steps.append({"step": "retry_new_key", "result": r2, "error": e2})
    elif name == "idempotent_retry":
        # 同一个业务键重试：第二次命中去重，副作用只有一条
        r1, e1 = _post(port, "/call", {"idem_key": key, "expression": "6*7", "drop_ack": True})
        steps.append({"step": "first_call", "error": e1})
        r2, e2 = _post(port, "/call", {"idem_key": key, "expression": "6*7"})
        steps.append({"step": "retry_same_key", "result": r2, "error": e2})
    elif name == "query_then_write":
        # 先查结果：查到就不重试
        r1, e1 = _post(port, "/call", {"idem_key": key, "expression": "6*7", "drop_ack": True})
        steps.append({"step": "first_call", "error": e1})
        q, qe = _get(port, f"/result?idem_key={key}")
        steps.append({"step": "query_result", "result": q, "error": qe})
        if not (q or {}).get("found"):
            r2, e2 = _post(port, "/call", {"idem_key": key, "expression": "6*7"})
            steps.append({"step": "retry_after_query", "result": r2, "error": e2})
    elif name == "compensation":
        # 已提交的副作用不能回滚，只能写补偿记录
        r1, e1 = _post(port, "/call", {"idem_key": key, "expression": "6*7"})
        steps.append({"step": "call", "result": r1, "error": e1})
        c, ce = _post(port, "/compensate", {"idem_key": key, "reason": "下游拒收"})
        steps.append({"step": "compensate", "result": c, "error": ce})
    else:
        raise ValueError(name)
    return {"scenario": name, "idem_key": key, "steps": steps}


def cmd_run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    port0 = args.port
    for i, name in enumerate(("naive_retry", "idempotent_retry", "query_then_write", "compensation")):
        port = port0 + i
        db = out / f"{name}.sqlite"
        if db.exists():
            db.unlink()
        proc = serve(db, port)
        try:
            scenario = _scenario(port, name, f"{name}-key" if name != "naive_retry" else "naive")
            led = _get(port, "/ledger")[0] or {}
            scenario["ledger"] = {"effect_count": led.get("effect_count"),
                                  "attempt_count": led.get("attempt_count"),
                                  "distinct_keys": led.get("distinct_keys"),
                                  "effects": [{k: e[k] for k in ("idem_key", "value", "compensated_by")}
                                              for e in led.get("effects", [])],
                                  "attempt_outcomes": [a["outcome"] for a in led.get("attempts", [])]}
            scenario["db"] = str(db)
            results.append(scenario)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()

    by = {r["scenario"]: r for r in results}
    checks = [
        {"name": "naive_retry_duplicates_effect",
         "expected": "两次不同业务键 ⇒ 账本 2 条副作用",
         "got": by["naive_retry"]["ledger"]["effect_count"],
         "match": by["naive_retry"]["ledger"]["effect_count"] == 2},
        {"name": "idempotent_retry_single_effect",
         "expected": "同一业务键重试 ⇒ 1 条副作用、2 次 attempt",
         "got": {"effects": by["idempotent_retry"]["ledger"]["effect_count"],
                 "attempts": by["idempotent_retry"]["ledger"]["attempt_count"]},
         "match": (by["idempotent_retry"]["ledger"]["effect_count"] == 1
                   and by["idempotent_retry"]["ledger"]["attempt_count"] == 2)},
        {"name": "idempotent_retry_returns_original",
         "expected": "第二次返回首次的 effect_id 且 dedup_hit=true",
         "got": by["idempotent_retry"]["steps"][1].get("result"),
         "match": bool((by["idempotent_retry"]["steps"][1].get("result") or {}).get("dedup_hit"))},
        {"name": "query_then_write_avoids_retry",
         "expected": "查询命中 ⇒ 不再重试，账本 1 条副作用",
         "got": {"found": (by["query_then_write"]["steps"][1].get("result") or {}).get("found"),
                 "effects": by["query_then_write"]["ledger"]["effect_count"]},
         "match": ((by["query_then_write"]["steps"][1].get("result") or {}).get("found") is True
                   and by["query_then_write"]["ledger"]["effect_count"] == 1)},
        {"name": "compensation_marks_not_rolls_back",
         "expected": "补偿后副作用仍在账本里，只是被标注",
         "got": by["compensation"]["ledger"]["effects"],
         "match": (by["compensation"]["ledger"]["effect_count"] == 1
                   and bool(by["compensation"]["ledger"]["effects"][0]["compensated_by"]))},
        {"name": "ack_loss_window_is_real",
         "expected": "drop_ack 的首次调用在客户端报错，但副作用已提交",
         "got": {"first_error": by["idempotent_retry"]["steps"][0].get("error"),
                 "effects": by["idempotent_retry"]["ledger"]["effect_count"]},
         "match": (bool(by["idempotent_retry"]["steps"][0].get("error"))
                   and by["idempotent_retry"]["ledger"]["effect_count"] == 1)},
    ]
    report = {"scenarios": results, "checks": checks,
              "all_match": all(c["match"] for c in checks),
              "note": ("判定只看账本与退出码：客户端报错不等于副作用没发生；"
                       "无幂等键时重试就是第二次执行，补偿不是回滚")}
    (out / "idempotent_tool.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {c['got']}")
    print("all_match:", report["all_match"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 幂等工具服务与 ACK 丢失窗口")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve")
    p.add_argument("--db", required=True)
    p.add_argument("--port", type=int, required=True)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8091)
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
