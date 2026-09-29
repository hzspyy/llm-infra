#!/usr/bin/env bash
# L9.5 任务 D：进程级故障矩阵（新输出目录）。覆盖 kill/restart、租约过期、checkpoint 失败、
# 重复/乱序完成、版本不兼容；全部按退出码与结构化结果判定，不看 stdout 最后一行。
#
#   bash labs/L9/run_durable_matrix.sh <out_dir> [python]
#
# 正常任务与失败恢复共用同一组验收断言（见末尾的 summary.json）。
set -u
ROOT=${1:?usage: run_durable_matrix.sh <out_dir> [python]}
PY=${2:-python}
mkdir -p "$ROOT"; rm -rf "$ROOT"/*

run() { "$PY" labs/L9/durable_task_store.py "$@" > /dev/null 2>&1; echo "$?"; }

# --- 1) 正常路径 ------------------------------------------------------------------
run execute --out "$ROOT/happy" --session happy --worker w1 > "$ROOT/happy.exit"

# --- 2) kill/restart：单事务崩溃（副作用随事务回滚） --------------------------------
run execute --out "$ROOT/single-tx" --session s1 --worker w1 --crash-after compute \
  > "$ROOT/single-tx-crash.exit"
run resume  --out "$ROOT/single-tx" --session s1 --worker w1 > "$ROOT/single-tx-resume.exit"

# --- 3) 两事务反例：副作用已提交、状态未提交 ----------------------------------------
run execute --out "$ROOT/two-tx" --session s2 --worker w1 --two-transaction --crash-after compute \
  > "$ROOT/two-tx-crash.exit"
run resume  --out "$ROOT/two-tx" --session s2 --worker w1 > "$ROOT/two-tx-resume.exit"

# --- 4) 租约过期与 fencing --------------------------------------------------------
run leases --out "$ROOT/leases" > "$ROOT/leases.exit"
run window --out "$ROOT/window" > "$ROOT/window.exit"

# --- 5) checkpoint 写失败 ---------------------------------------------------------
run checkpoint-fail --out "$ROOT/checkpoint-fail" > "$ROOT/checkpoint-fail.exit"

# --- 6) 重复完成（同键 vs 换键） ---------------------------------------------------
run duplicate --out "$ROOT/duplicate" > "$ROOT/duplicate.exit"

# --- 7) 乱序完成与依赖拒绝 --------------------------------------------------------
run ordering --out "$ROOT/ordering" > "$ROOT/ordering.exit"

# --- 8) 版本不兼容 ----------------------------------------------------------------
run versions --out "$ROOT/versions" --recorded-hash aaaa1111 --current-hash bbbb2222 \
  > "$ROOT/versions.exit"

# --- 统一判据：正常路径与失败恢复共用同一组断言 ------------------------------------
"$PY" - "$ROOT" <<'PYEOF'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])


def load(*parts):
    p = root.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def exit_of(name):
    p = root / f"{name}.exit"
    return int(p.read_text().strip()) if p.exists() else None


happy = load("happy", "last_execute.json")
single = load("single-tx", "last_resume.json")
two = load("two-tx", "last_resume.json")
leases = load("leases", "leases.json")
window = load("window", "window.json")
ckpt = load("checkpoint-fail", "checkpoint_fail.json")
dup = load("duplicate", "duplicate.json")
order = load("ordering", "ordering.json")
vers = load("versions", "versions.json")

checks = {
    # 正常路径
    "happy_graph_completes": bool(happy) and all(v == "done" for v in happy["state"].values()),
    "happy_no_duplicate_effects": bool(happy) and happy["duplicates"] == 0,
    "happy_exit_zero": exit_of("happy") == 0,
    # kill/restart
    "crash_exit_code_137": exit_of("single-tx-crash") == 137 and exit_of("two-tx-crash") == 137,
    "single_tx_rolls_back_effect": bool(window)
    and window["results"]["single_transaction"]["effects_after_crash"] == 0,
    "two_tx_leaves_committed_effect": bool(window)
    and window["results"]["two_transaction"]["effects_after_crash"] == 1,
    "resume_no_duplicate_effects": bool(window)
    and window["results"]["two_transaction"]["duplicates_after_resume"] == 0
    and window["results"]["single_transaction"]["duplicates_after_resume"] == 0,
    "single_tx_resume_completes": bool(single)
    and all(v == "done" for v in single["state_after"].values()),
    # 租约
    "stale_lease_write_rejected": bool(leases) and leases["stale_commit_rejected"] is not None,
    "fencing_token_incremented": bool(leases) and leases["token_b"] == leases["token_a"] + 1,
    # checkpoint 失败
    "checkpoint_failure_rolls_back_effect": bool(ckpt) and ckpt["after_failure"]["effects"] == 0,
    "checkpoint_failure_then_retry_single_effect": bool(ckpt)
    and ckpt["after_retry"]["effects"] == 1 and ckpt["after_retry"]["state"]["fetch"] == "done",
    "checkpoint_fail_exit_zero": exit_of("checkpoint-fail") == 0,
    # 重复完成
    "same_key_single_effect": bool(dup) and dup["same_key"]["effects"] == 1,
    "new_key_duplicates": bool(dup) and dup["new_key_on_retry"]["effects"] == 2,
    # 乱序完成
    "out_of_order_refused_before_effect": bool(order) and order["effects_after_refusal"] == 0
    and "not done" in order["out_of_order"],
    "pending_respects_dependencies": bool(order)
    and order["pending_after_fetch"] == ["compute"] and order["pending_after_compute"] == ["commit"],
    "duplicate_completion_is_noop": bool(order) and order["duplicate_completion"].get("idempotent") is True,
    # 版本不兼容
    "version_mismatch_refused": bool(vers) and vers["resume_decision"] == "refused",
    "version_exit_nonzero": exit_of("versions") == 0,
    # 所有新注入都用退出码表达成功（0=断言通过），而不是靠 stdout 文本
    "injection_exit_codes_are_structured": all(
        exit_of(n) == 0 for n in ("duplicate", "ordering", "checkpoint-fail", "window", "leases")),
}
summary = {
    "root": str(root),
    "checks": checks,
    "checks_passed": sum(checks.values()),
    "checks_total": len(checks),
    "exits": {k: exit_of(k) for k in (
        "happy", "single-tx-crash", "single-tx-resume", "two-tx-crash", "two-tx-resume",
        "leases", "window", "checkpoint-fail", "duplicate", "ordering", "versions")},
    "note": ("kill/restart 用 os._exit(137) 注入，退出码 137 是证据的一部分；"
             "每个注入点自己给出结构化 ok 字段并据此返回退出码，汇总只读这些字段"),
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"checks_passed": summary["checks_passed"], "checks_total": summary["checks_total"],
                  "failed": [k for k, v in checks.items() if not v]}, ensure_ascii=False, indent=1))
PYEOF
