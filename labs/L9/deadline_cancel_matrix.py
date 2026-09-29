#!/usr/bin/env python3
"""L9.5 任务 C：任务总 deadline、有界重试退避，以及取消在四个位置的传播与 epoch 拒收。

被取消的对象是一条最小任务图：``plan → (fan-out: 2 个子任务) → join``。四个取消位置分别对应
一次任务真正可能被打断的时刻：

1. ``queued``    —— 还在等准入槽，没开始任何外部调用；
2. ``streaming`` —— 模型正在流式输出（用带延迟的 stub 流模拟，含一个晚到的终止帧）；
3. ``tool_running`` —— 工具子进程正在跑（真实 subprocess，取消后必须整个进程树消失）；
4. ``tool_committed_unacked`` —— 工具已提交副作用但结果还没回填（取消后副作用必须仍在，
   只能按业务幂等键取回，不能重放）。

另外两条机制单独验证：

* ``epoch``：每次重试/恢复都把 epoch 加一；晚到的回包带有旧 epoch，必须被丢弃且不改变状态；
* ``deadline``：每个子调用有自己的超时与有界退避，但**总 deadline 是任务级**的，
  每轮重新起算超时不算总 deadline。

判定依据是事件日志、账本、子进程存活状态与退出码，不看 stdout。

用法::

    python labs/L9/deadline_cancel_matrix.py --out out/9.5/deadline-cancel
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CANCEL_POINTS = ("queued", "streaming", "tool_running", "tool_committed_unacked")


class Ledger:
    """进程内账本：副作用按业务幂等键去重（与 9.5-B 的服务端同语义）。"""

    def __init__(self) -> None:
        self.effects: dict[str, dict] = {}
        self.executions = 0

    def execute(self, idem_key: str, expr: str) -> dict:
        if idem_key in self.effects:
            return {"dedup_hit": True, **self.effects[idem_key]}
        import agent_tasks as T

        self.executions += 1
        value = int(T.safe_eval(expr))
        rec = {"effect_id": f"eff-{self.executions}", "value": value}
        self.effects[idem_key] = rec
        return {"dedup_hit": False, **rec}

    def count(self) -> int:
        return len(self.effects)


class Runtime:
    """带 epoch、总 deadline 与四位置取消的最小运行时。"""

    def __init__(self, out: pathlib.Path, *, total_deadline_s: float, child_timeout_s: float,
                 max_retries: int, slots: int = 1) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.events_fh = open(out / "events.jsonl", "w", encoding="utf-8")
        self.total_deadline_s = total_deadline_s
        self.child_timeout_s = child_timeout_s
        self.max_retries = max_retries
        self.slots = asyncio.Semaphore(slots)
        self.t0 = time.perf_counter()
        self.epoch = 0
        self.cancelled = False
        self.ledger = Ledger()
        self.children: list[subprocess.Popen] = []
        self.terminal = None
        self.deadline_hit = False

    # -- 基础设施 -------------------------------------------------------------
    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def emit(self, kind: str, **kw) -> dict:
        rec = {"ts_ms": round(self.now_ms(), 3), "kind": kind, "epoch": self.epoch, **kw}
        self.events_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events_fh.flush()
        return rec

    def remaining_s(self) -> float:
        return self.total_deadline_s - (time.perf_counter() - self.t0)

    def cancel(self, reason: str) -> None:
        """取消：epoch 加一，所有在途动作后续的回包都视为过期。"""
        self.cancelled = True
        self.epoch += 1
        self.emit("cancel", reason=reason)

    def _alive_children(self) -> list[int]:
        alive = []
        for p in self.children:
            if p.poll() is None:
                alive.append(p.pid)
        return alive

    # -- 各位置的行为 ---------------------------------------------------------
    async def admit(self) -> bool:
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=max(0.01, self.remaining_s()))
            return True
        except asyncio.TimeoutError:
            return False

    async def stream_model(self, stop_after: int, late_final_ms: float) -> dict:
        """带延迟的 stub 流；``late_final_ms`` 之后才送达的终止帧属于过期 epoch。"""
        started_epoch = self.epoch
        chunks = 0
        for i in range(6):
            await asyncio.sleep(0.05)
            if self.cancelled:
                break
            chunks += 1
            if chunks >= stop_after:
                break
        # 终止帧可能晚到
        await asyncio.sleep(late_final_ms / 1000.0)
        stale = started_epoch != self.epoch
        self.emit("model_final", started_epoch=started_epoch, chunks=chunks, accepted=not stale)
        return {"chunks": chunks, "accepted": not stale, "stale": stale}

    def start_tool_process(self, seconds: float, idem_key: str) -> subprocess.Popen:
        code = (
            "import time,sys\n"
            f"time.sleep({seconds})\n"
            "print('tool-done')\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)   # 独立进程组，便于整组回收
        self.children.append(proc)
        self.emit("tool_start", pid=proc.pid, idem_key=idem_key)
        return proc

    def kill_tool_process(self, proc: subprocess.Popen) -> dict:
        """取消工具：杀整个进程组，然后确认没有残留。"""
        killed = []
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            killed.append(proc.pid)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        alive = self._alive_children()
        self.emit("tool_killed", pid=proc.pid, alive_after=alive, killpg=killed)
        return {"killed_pid": proc.pid, "alive_after": alive, "reaped": proc.poll() is not None}

    # -- 主流程 ---------------------------------------------------------------
    async def run_task(self, cancel_point: str | None) -> dict:
        self.emit("task_arrive", cancel_point=cancel_point)
        admitted = await self.admit()
        self.emit("task_admit", admitted=admitted)
        if not admitted:
            self.terminal = "deadline_before_admit"
            return self.finish()

        if cancel_point == "queued":
            self.cancel("cancel_at_queued")
            self.terminal = "cancelled"
            return self.finish()

        # fan-out 两个子任务，每个子任务=一次模型流 + 一次工具
        results = await asyncio.gather(*[self.subtask(i, cancel_point) for i in range(2)],
                                       return_exceptions=True)
        self.emit("join", results=[r if isinstance(r, dict) else str(r) for r in results])
        if self.cancelled:
            self.terminal = "cancelled"
        elif self.deadline_hit:
            self.terminal = "deadline"
        else:
            self.terminal = "ok"
        return self.finish()

    async def subtask(self, idx: int, cancel_point: str | None) -> dict:
        idem_key = f"task-{idx}"
        stream = await self.stream_model(stop_after=3, late_final_ms=200.0)
        if cancel_point == "streaming" and idx == 0:
            self.cancel("cancel_at_streaming")
        if self.cancelled:
            self.emit("subtask_abort", node=idx, stage="after_stream")
            return {"node": idx, "aborted": True}
        proc = self.start_tool_process(seconds=1.5, idem_key=idem_key)
        if cancel_point == "tool_running" and idx == 0:
            await asyncio.sleep(0.3)
            self.cancel("cancel_at_tool_running")
            info = self.kill_tool_process(proc)
            self.emit("subtask_abort", node=idx, stage="tool_running", **info)
            return {"node": idx, "aborted": True, **info}
        # 工具执行：等它有界超时；超时后按剩余预算重试（退避）
        attempt = 0
        while True:
            if self.cancelled:
                # 取消必须传播到 fork/join 的兄弟子任务：取消后不允许提交新副作用
                info = self.kill_tool_process(proc)
                self.emit("subtask_abort", node=idx, stage="cancelled_during_tool", **info)
                return {"node": idx, "aborted": True, **info}
            attempt += 1
            budget = min(self.child_timeout_s, max(0.0, self.remaining_s()))
            if budget <= 0:
                self.deadline_hit = True
                self.emit("subtask_abort", node=idx, stage="deadline_before_tool")
                return {"node": idx, "aborted": True, "reason": "deadline"}
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=budget)
                break
            except asyncio.TimeoutError:
                if attempt > self.max_retries:
                    self.kill_tool_process(proc)
                    self.deadline_hit = True
                    self.emit("subtask_abort", node=idx, stage="tool_timeout")
                    return {"node": idx, "aborted": True, "reason": "retry_exhausted"}
                backoff = 0.05 * (2 ** (attempt - 1))
                self.emit("retry_backoff", node=idx, attempt=attempt, backoff_s=round(backoff, 3))
                await asyncio.sleep(min(backoff, max(0.0, self.remaining_s())))
        if self.cancelled:
            info = self.kill_tool_process(proc)
            self.emit("subtask_abort", node=idx, stage="before_commit", **info)
            return {"node": idx, "aborted": True, **info}
        # 副作用提交（提交本身是幂等的）
        if cancel_point == "tool_committed_unacked" and idx == 0:
            res = self.ledger.execute(idem_key, "6*7")
            self.emit("tool_committed", node=idx, idem_key=idem_key, **res)
            self.cancel("cancel_at_tool_committed_unacked")
            return {"node": idx, "aborted": True, "committed": True, "idem_key": idem_key,
                    "effect_id": res.get("effect_id")}
        res = self.ledger.execute(idem_key, "6*7")
        self.emit("tool_committed", node=idx, idem_key=idem_key, **res)
        return {"node": idx, "aborted": False, **res}

    def finish(self) -> dict:
        self.emit("task_end", terminal=self.terminal, effects=self.ledger.count(),
                  alive_children=self._alive_children(), epoch=self.epoch)
        return {"terminal": self.terminal, "effects": self.ledger.count(),
                "executions": self.ledger.executions,
                "alive_children": self._alive_children(), "epoch": self.epoch}

    # -- 恢复：用业务键取回已提交结果，不重放 --------------------------------
    def recover(self) -> dict:
        recovered = {k: v for k, v in self.ledger.effects.items()}
        self.emit("recover", recovered_keys=sorted(recovered), replayed=0)
        return {"recovered": recovered, "replayed": 0}


async def run_case(out: pathlib.Path, cancel_point: str | None, **kw) -> dict:
    rt = Runtime(out, **kw)
    result = await rt.run_task(cancel_point)
    rec = rt.recover()
    result["recovered_effects"] = sorted(rec["recovered"])
    result["replay_count"] = rec["replayed"]
    # 资源回收：取消/结束后不应有存活子进程
    await asyncio.sleep(0.2)
    result["alive_children_after_settle"] = rt._alive_children()
    rt.events_fh.close()
    result["event_count"] = sum(1 for _ in open(out / "events.jsonl", encoding="utf-8"))
    return result


async def run_deadline_case(out: pathlib.Path, total_s: float, child_s: float, retries: int) -> dict:
    """总 deadline 的验收：子调用超时+退避都不允许把总时限撑开。"""
    rt = Runtime(out, total_deadline_s=total_s, child_timeout_s=child_s, max_retries=retries)
    t0 = time.perf_counter()
    # 一个必然超时的工具（sleep 10 s），子超时 0.3 s、允许 5 次重试：总时限 1.2 s 必须先到
    proc = rt.start_tool_process(seconds=10.0, idem_key="deadline-key")
    attempt = 0
    while True:
        attempt += 1
        budget = min(rt.child_timeout_s, max(0.0, rt.remaining_s()))
        if budget <= 0:
            rt.deadline_hit = True
            break
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=budget)
            break
        except asyncio.TimeoutError:
            if attempt > rt.max_retries:
                break
            await asyncio.sleep(min(0.05 * (2 ** (attempt - 1)), max(0.0, rt.remaining_s())))
    elapsed = time.perf_counter() - t0
    rt.kill_tool_process(proc)
    rt.terminal = "deadline" if rt.deadline_hit else "ok"
    res = rt.finish()
    res.update({"elapsed_s": round(elapsed, 3), "attempts": attempt,
                "total_deadline_s": total_s, "child_timeout_s": child_s,
                "max_retries": retries,
                "within_deadline": elapsed <= total_s + 0.3})
    rt.events_fh.close()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.5 取消、deadline 与 epoch 矩阵")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-deadline-s", type=float, default=6.0)
    ap.add_argument("--child-timeout-s", type=float, default=4.0)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = {}
    for point in CANCEL_POINTS:
        cases[point] = asyncio.run(run_case(out / point, point,
                                            total_deadline_s=args.total_deadline_s,
                                            child_timeout_s=args.child_timeout_s,
                                            max_retries=args.max_retries))
    cases["happy"] = asyncio.run(run_case(out / "happy", None,
                                          total_deadline_s=args.total_deadline_s,
                                          child_timeout_s=args.child_timeout_s,
                                          max_retries=args.max_retries))
    deadline = asyncio.run(run_deadline_case(out / "deadline", total_s=1.2,
                                             child_s=0.3, retries=5))

    # epoch 拒收：stale 终止帧被丢弃
    stale_events = []
    for line in open(out / "streaming" / "events.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        if rec["kind"] == "model_final":
            stale_events.append(rec)
    late_rejected = [e for e in stale_events if e["accepted"] is False]

    checks = [
        {"name": "cancel_at_queued_stops_before_any_work",
         "expected": "queued 取消后 0 副作用、无存活子进程",
         "got": {"effects": cases["queued"]["effects"],
                 "alive": cases["queued"]["alive_children_after_settle"]},
         "match": cases["queued"]["effects"] == 0 and not cases["queued"]["alive_children_after_settle"]},
        {"name": "cancel_at_streaming_drops_late_final",
         "expected": "streaming 取消后晚到终止帧被 epoch 拒收",
         "got": {"late_events": len(stale_events), "rejected": len(late_rejected)},
         "match": len(late_rejected) >= 1},
        {"name": "cancel_at_tool_running_reaps_process_tree",
         "expected": "tool_running 取消后子进程被回收",
         "got": {"alive": cases["tool_running"]["alive_children_after_settle"],
                 "effects": cases["tool_running"]["effects"]},
         "match": (not cases["tool_running"]["alive_children_after_settle"]
                   and cases["tool_running"]["effects"] == 0)},
        {"name": "cancel_at_committed_keeps_effect_and_recovers_by_key",
         "expected": "已提交副作用保留，恢复按业务键取回且不重放",
         "got": {"effects": cases["tool_committed_unacked"]["effects"],
                 "recovered": cases["tool_committed_unacked"]["recovered_effects"],
                 "replay": cases["tool_committed_unacked"]["replay_count"]},
         "match": (cases["tool_committed_unacked"]["effects"] >= 1
                   and len(cases["tool_committed_unacked"]["recovered_effects"]) >= 1
                   and cases["tool_committed_unacked"]["replay_count"] == 0)},
        {"name": "happy_path_completes_without_leaks",
         "expected": "正常路径 2 条副作用、无残留进程",
         "got": {"effects": cases["happy"]["effects"], "terminal": cases["happy"]["terminal"],
                 "alive": cases["happy"]["alive_children_after_settle"]},
         "match": (cases["happy"]["effects"] == 2 and cases["happy"]["terminal"] == "ok"
                   and not cases["happy"]["alive_children_after_settle"])},
        {"name": "total_deadline_not_reset_by_retries",
         "expected": "5 次重试 × 0.3 s 子超时不能突破 1.2 s 总时限",
         "got": {"elapsed_s": deadline["elapsed_s"], "attempts": deadline["attempts"],
                 "within": deadline["within_deadline"]},
         "match": deadline["within_deadline"] and deadline["elapsed_s"] < 2.0},
    ]
    report = {"config": vars(args), "cases": cases, "deadline": deadline,
              "checks": checks, "all_match": all(c["match"] for c in checks),
              "note": ("四个取消位置分别对应「没开始 / 在流式 / 工具在跑 / 已提交未回填」；"
                       "epoch 是拒绝晚到回包的唯一依据；总 deadline 不能被子调用的重试重新起算")}
    (out / "deadline_cancel.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    for c in checks:
        print(f"[{'OK ' if c['match'] else 'FAIL'}] {c['name']}: {c['got']}")
    print("all_match:", report["all_match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
