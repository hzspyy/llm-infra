#!/usr/bin/env bash
# L5.4 任务 G 的归因采集：每个 (mode, bs) 跑两个 nsys 窗口相减得到纯 decode。
#
#   A 窗口：--gen 0  -> max_tokens=1，只有一次 prefill forward
#   B 窗口：--gen G  -> max_tokens=G+1，prefill + G 步 decode
#   decode 的 (wall, cpu, gpu, 提交数) = (B − A) / G
#
# 无插桩的 wall 与进程 CPU 时间由 lab 自己打印（WALL_MS= / CPU_MS=），
# 因为 `--cuda-graph-trace=node` 会把 cudaGraphLaunch 的 CPU 时长放大上百倍，
# 只有 kernel 的 device 时间和提交**次数**不受影响。
#
# 用法：RUN_ID=20260913-bash labs/L5/run_execution_attribution.sh
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
NSYS=${NSYS:-$ROOT/tools/nsight_systems-linux-x86_64-2026.3.2.313-archive/target-linux-x64/nsys}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/execution-attribution-$RUN_ID}
MODES=${MODES:-"NONE COMPILE_ONLY FULL_AND_PIECEWISE"}
BATCHES=${BATCHES:-"1 16 64"}
GEN=${GEN:-16}
PLEN=${PLEN:-64}
mkdir -p "$OUT/nsys"

for m in $MODES; do
    for b in $BATCHES; do
        for g in 0 "$GEN"; do
            tag="${m}_bs${b}_gen${g}"
            printf "== %-28s" "$tag"
            rep="$OUT/nsys/$tag"
            "$NSYS" profile -t cuda -f true --cuda-graph-trace=node \
                --capture-range=cudaProfilerApi --capture-range-end=stop \
                -o "$rep" "$PY" "$HERE/graph_dispatch_audit.py" D \
                --mode "$m" --bs "$b" --gen "$g" --plen "$PLEN" \
                > "$rep.log" 2>&1
            "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" \
                "$rep.nsys-rep" >/dev/null 2>&1
            grep -h "^MODE=" "$rep.log" | tail -1
        done
    done
done

echo
"$PY" "$HERE/summarize_attribution.py" "$OUT"
echo
echo "原始工件在 $OUT/"
