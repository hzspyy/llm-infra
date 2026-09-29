#!/usr/bin/env bash
# L5.4 任务 A/B/C 的采集脚本（crater）。
#
#   1) audit.txt            —— 分派事实：模式、桶、padding、指针、图池（无 profiler）
#   2) tamper_pointer.txt   —— 指针负向实验：换掉静态输入缓冲区，看 FULL 图读谁
#   3) nsys/<mode>*         —— 四个对照点的 --cuda-graph-trace=node 展开
#   4) nsys_summary.txt     —— 图内/图外 kernel、每张图的规模、提交 API 计数
#
# 用法：RUN_ID=20260913-bash labs/L5/run_graph_dispatch_audit.sh
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
NSYS=${NSYS:-$ROOT/tools/nsight_systems-linux-x86_64-2026.3.2.313-archive/target-linux-x64/nsys}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/graph-dispatch-$RUN_ID}
MODES=${MODES:-"NONE COMPILE_ONLY PIECEWISE FULL_AND_PIECEWISE"}
BS=${BS:-4}
GEN=${GEN:-16}
PLEN=${PLEN:-128}
mkdir -p "$OUT/nsys"

echo "== [1/4] 分派事实（A/B/C） =="
VLLM_LOGGING_LEVEL=WARNING "$PY" "$HERE/graph_dispatch_audit.py" A B C \
    > "$OUT/audit.txt" 2>&1
echo "   -> $OUT/audit.txt ($(wc -l < "$OUT/audit.txt") 行)"

echo "== [2/4] 指针负向实验 =="
VLLM_LOGGING_LEVEL=WARNING "$PY" "$HERE/graph_dispatch_audit.py" --tamper-pointer \
    > "$OUT/tamper_pointer.txt" 2>&1
grep -E "不一致|一致|ptr=" "$OUT/tamper_pointer.txt" | tail -4

echo "== [3/4] nsys 图展开（$MODES）=="
for m in $MODES; do
    rep="$OUT/nsys/$m"
    printf "   %-22s" "$m"
    "$NSYS" profile -t cuda -f true --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o "$rep" "$PY" "$HERE/graph_dispatch_audit.py" D \
        --mode "$m" --bs "$BS" --gen "$GEN" --plen "$PLEN" \
        > "$rep.log" 2>&1
    "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" \
        "$rep.nsys-rep" >/dev/null 2>&1
    grep -h "^MODE=" "$rep.log" | tail -1
done

echo "== [4/4] 汇总 =="
{
    echo "run_id $RUN_ID  modes: $MODES  bs=$BS gen=$GEN plen=$PLEN"
    echo "engine log lines:"
    grep -h "Graph capturing finished\|CUDA graph pool memory\|GPU KV cache size\|Available KV cache memory" \
        "$OUT"/nsys/*.log 2>/dev/null | sed 's/^/  /'
    for m in $MODES; do
        [ -f "$OUT/nsys/$m.sqlite" ] && \
            "$PY" "$HERE/summarize_graph_nsys.py" "$OUT/nsys/$m.sqlite" "$m"
    done
} > "$OUT/nsys_summary.txt" 2>&1
cat "$OUT/nsys_summary.txt"
echo
echo "原始工件在 $OUT/"
