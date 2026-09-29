#!/usr/bin/env bash
# L9.8：隔离能力核验 → 沙箱池状态机/限制 → 生命周期与工具可用时间 → 失败注入与回收证据。
#
#   bash labs/L9/run_sandbox_labs.sh <out_root> [python]
#
# 前置：本机需为 Linux；能力核验会先判定哪些隔离原语可用（容器/gVisor/microVM 在本机不可用，
# 相关路线保持 UNVERIFIED，见 ENVIRONMENTS.md 的「隔离能力边界」）。
set -u
ROOT=${1:?usage: run_sandbox_labs.sh <out_root> [python]}
PY=${2:-python}
LABS=$(cd "$(dirname "$0")" && pwd)

run() { echo "--- $*"; "$PY" "$LABS/$1" "${@:2}"; }

run sandbox_capability_probe.py --out "$ROOT/capability" > "$ROOT/capability.log" 2>&1
echo "$?" > "$ROOT/capability.exit"

run tool_sandbox_pool.py --out "$ROOT/pool" --warm 1 --tasks 4 --max-slots 2 --timeout-s 8 \
  > "$ROOT/pool.log" 2>&1
echo "$?" > "$ROOT/pool.exit"

run sandbox_lifecycle_bench.py --out "$ROOT/lifecycle" --tasks 12 \
  --warms 0,1,4 --concurrency 1,4,16 --dep-mb 8 > "$ROOT/lifecycle.log" 2>&1
echo "$?" > "$ROOT/lifecycle.exit"

run sandbox_failure_cases.py --out "$ROOT/failures" > "$ROOT/failures.log" 2>&1
echo "$?" > "$ROOT/failures.exit"

"$PY" - "$ROOT" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])

def load(*parts):
    p = root.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def ex(name):
    p = root / f"{name}.exit"
    return int(p.read_text().strip()) if p.exists() else None

cap = load("capability", "sandbox_capability.json")
pool = load("pool", "pool.json")
life = load("lifecycle", "lifecycle.json")
fail = load("failures", "failures.json")

checks = {
    "capability_matrix_recorded": bool(cap) and "capability_matrix" in cap,
    "capability_lists_unverified": bool(cap) and len(cap["verdict"]["unverified_here"]) > 0,
    "pool_prewarm_ready": bool(pool) and pool["prewarm"]["ready"] == pool["prewarm"]["warm_requested"],
    "pool_returns_to_ready": bool(pool) and all(r["back_to_ready"] for r in pool["results"] if r["acquired"]),
    "pool_no_running_after_tasks": bool(pool) and all(
        r["cleanup"]["running_after"] == 0 for r in pool["results"] if r["acquired"]),
    "pool_output_cap_enforced": bool(pool) and any(
        r["result"]["output_truncated"] for r in pool["results"] if r["acquired"]),
    "pool_wall_timeout_enforced": bool(pool) and any(
        r["result"]["exit_reason"] == "wall_timeout" for r in pool["results"] if r["acquired"]),
    "lifecycle_warm_moves_startup_off_task_path": bool(life) and (
        life["cells"]["warm4-c4"]["queue_wait_ms"]["miss_n"] == 0
        and life["cells"]["warm0-c4"]["queue_wait_ms"]["miss_n"] > 0),
    "lifecycle_overflow_bounded": bool(life) and all(
        c["pool"]["slots"] <= life["config"]["max_slots"] for c in life["cells"].values()),
    "failures_all_ok": bool(fail) and fail["all_ok"] is True,
    "injection_exit_codes": all(ex(n) == 0 for n in ("capability", "pool", "lifecycle", "failures")),
}
summary = {"root": str(root), "checks": checks, "checks_passed": sum(checks.values()),
           "checks_total": len(checks),
           "exits": {n: ex(n) for n in ("capability", "pool", "lifecycle", "failures")},
           "unverified_here": (cap or {}).get("verdict", {}).get("unverified_here")}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"checks_passed": summary["checks_passed"], "checks_total": summary["checks_total"],
                  "failed": [k for k, v in checks.items() if not v]}, ensure_ascii=False, indent=1))
PYEOF
