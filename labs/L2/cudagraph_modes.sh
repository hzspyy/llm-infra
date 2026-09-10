#!/usr/bin/env bash
# L2.6b 证据补齐 · 三种 cudagraph_mode 的 launch/kernel 计数。
#
# 每种模式一个干净进程（图捕获是全局状态，同进程切换不可靠），
# 用 nsys 的 --capture-range=cudaProfilerApi 只统计 generate 那一段，
# 把模型加载与图捕获排除在外。
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
NSYS=${NSYS:-$ROOT/tools/nsight_systems-linux-x86_64-2026.3.2.313-archive/target-linux-x64/nsys}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$ROOT/results/cudagraph}
mkdir -p "$OUT"

MODES=${*:-"NONE PIECEWISE FULL_AND_PIECEWISE"}

for m in $MODES; do
    echo "=========== $m ==========="
    rm -f "$OUT/$m.nsys-rep" "$OUT/$m.sqlite"
    # --cuda-graph-trace=node 是关键：默认值 graph 会把**整张图记成一个单位**，
    # KERNEL 表里就只剩非图的 kernel，于是"图模式 GPU kernel 数暴跌"是假象。
    "$NSYS" profile -t cuda -f true --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o "$OUT/$m" \
        "$PY" "$HERE/cudagraph_modes.py" --mode "$m" \
        > "$OUT/$m.log" 2>&1
    tail -2 "$OUT/$m.log" | grep -E "MODE=|Error|error" || true
    "$NSYS" export --type sqlite --force-overwrite true \
        -o "$OUT/$m.sqlite" "$OUT/$m.nsys-rep" >/dev/null 2>&1
    [ -f "$OUT/$m.sqlite" ] || { echo "  导出 sqlite 失败，看 $OUT/$m.log"; continue; }
done

echo
echo "=========== 汇总 ==========="
"$PY" - "$OUT" $MODES <<'PYEOF'
import sqlite3, sys, pathlib
out = pathlib.Path(sys.argv[1]); modes = sys.argv[2:]

def q(c, sql, *a):
    try:
        return c.execute(sql, a).fetchone()[0] or 0
    except sqlite3.Error:
        return 0

rows = []
for m in modes:
    db = out / f"{m}.sqlite"
    if not db.exists():
        continue
    c = sqlite3.connect(str(db))
    names = {r[0] for r in c.execute(
        "select name from sqlite_master where type='table'")}
    # CUDA runtime/driver API 调用名在 StringIds 里
    api_tab = ("CUPTI_ACTIVITY_KIND_RUNTIME"
               if "CUPTI_ACTIVITY_KIND_RUNTIME" in names else None)
    launch = graph = 0
    per_api = {}
    if api_tab:
        for name, n in c.execute(f"""
            select s.value, count(*) from {api_tab} r
            join StringIds s on s.id = r.nameId
            group by s.value"""):
            per_api[name] = n
            if "LaunchKernel" in name:
                launch += n
            if "GraphLaunch" in name or "graphLaunch" in name:
                graph += n
    kern = q(c, "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL")
    memcpy = q(c, "select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY")
    gpu_ns = q(c, "select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL")
    rows.append((m, launch, graph, kern, memcpy, gpu_ns, per_api))

print(f"{'模式':<20} {'LaunchKernel':>13} {'GraphLaunch':>12} "
      f"{'GPU kernel 数':>13} {'memcpy':>8} {'GPU 忙 (ms)':>12}")
print("-" * 88)
for m, l, g, k, mc, ns, _ in rows:
    print(f"{m:<20} {l:>13,} {g:>12,} {k:>13,} {mc:>8,} {ns/1e6:>12.2f}")

if rows:
    base = rows[0]
    print()
    print(f"以 {base[0]} 为基准：")
    for m, l, g, k, mc, ns, _ in rows:
        cpu_calls = l + g
        b_calls = base[1] + base[2]
        print(f"  {m:<20} CPU 侧 launch 调用 {cpu_calls:>8,} "
              f"({cpu_calls / max(b_calls,1):.3f}×)   "
              f"GPU kernel {k:>8,} ({k / max(base[3],1):.3f}×)")

print()
print("=== 每种模式里 launch 相关 API 的明细 ===")
for m, *_rest in rows:
    per_api = _rest[-1]
    hits = {k: v for k, v in per_api.items()
            if "aunch" in k or "Graph" in k}
    print(f"\n--- {m} ---")
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<44} {v:>9,}")
    if not hits:
        print("  （没有 launch 类 API 记录）")
PYEOF
echo
echo "原始 nsys 报告在 $OUT/"
