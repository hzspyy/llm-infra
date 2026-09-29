#!/usr/bin/env bash
# L9.5 持久任务存储：正常路径、单事务崩溃恢复、两事务反例、租约 fencing、提交窗口对照。
# 纯 CPU，可在任意机器运行（本轮在本地 macOS 上执行）。
#
#   bash labs/L9/run_durable_store.sh <out_dir>
set -u
ROOT=${1:?usage: run_durable_store.sh <out_dir>}
PY=${L95_PY:-python}
mkdir -p "$ROOT"; rm -rf "$ROOT"/*

run() { echo "--- $*"; "$PY" labs/L9/durable_task_store.py "$@"; }

run execute --out "$ROOT/happy" --session happy --worker w1
echo "$?" > "$ROOT/happy.exit"

# 单事务：崩溃时副作用与状态一起回滚
run execute --out "$ROOT/single-tx" --session s1 --worker w1 --crash-after compute
echo "$?" > "$ROOT/single-tx-crash.exit"
run resume  --out "$ROOT/single-tx" --session s1 --worker w1
echo "$?" > "$ROOT/single-tx-resume.exit"

# 两事务反例：副作用已提交、状态未提交
run execute --out "$ROOT/two-tx" --session s2 --worker w1 --two-transaction --crash-after compute
echo "$?" > "$ROOT/two-tx-crash.exit"
run resume  --out "$ROOT/two-tx" --session s2 --worker w1
echo "$?" > "$ROOT/two-tx-resume.exit"

run leases --out "$ROOT/leases"
echo "$?" > "$ROOT/leases.exit"
run window --out "$ROOT/window"
echo "$?" > "$ROOT/window.exit"

# 汇总判据：把本次实验声称的性质写成可检查的布尔值
"$PY" - "$ROOT" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])


def load(*parts):
    p = root.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


happy = load("happy", "last_execute.json")
single = load("single-tx", "last_resume.json")
two = load("two-tx", "last_resume.json")
leases = load("leases", "leases.json")
window = load("window", "window.json")


def exits(name):
    p = root / f"{name}.exit"
    return int(p.read_text().strip()) if p.exists() else None


checks = {
    "happy_graph_completes": bool(happy) and all(v == "done" for v in happy["state"].values()),
    "happy_no_duplicate_effects": bool(happy) and happy["duplicates"] == 0,
    "crash_exit_code_137": exits("single-tx-crash") == 137 and exits("two-tx-crash") == 137,
    "single_tx_rolls_back_effect": bool(window)
    and window["results"]["single_transaction"]["effects_after_crash"] == 0,
    "two_tx_leaves_committed_effect": bool(window)
    and window["results"]["two_transaction"]["effects_after_crash"] == 1,
    "window_resume_no_duplicate": bool(window)
    and window["results"]["two_transaction"]["duplicates_after_resume"] == 0
    and window["results"]["single_transaction"]["duplicates_after_resume"] == 0,
    "resume_reuses_effect_no_duplicate": bool(two) and two["duplicates"] == 0
    and two["effects_after"] == 3,
    "single_tx_resume_completes": bool(single)
    and all(v == "done" for v in single["state_after"].values()),
    "stale_lease_write_rejected": bool(leases) and leases["stale_commit_rejected"] is not None,
    "fencing_token_incremented": bool(leases) and leases["token_b"] == leases["token_a"] + 1,
}
summary = {"root": str(root), "checks": checks,
           "checks_passed": sum(checks.values()), "checks_total": len(checks),
           "happy": happy, "single_tx_resume": single, "two_tx_resume": two,
           "leases": {"token_a": leases["token_a"], "token_b": leases["token_b"],
                      "rejected": leases["stale_commit_rejected"]} if leases else None,
           "window": window["results"] if window else None}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"checks": checks, "passed": summary["checks_passed"]}, ensure_ascii=False, indent=1))
PYEOF
