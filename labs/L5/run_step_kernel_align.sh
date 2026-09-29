#!/usr/bin/env bash
# L5.1 补测 · 逐 step kernel 与形状对齐（crater）
#
#   1) profiler-<tag>_p<len>/   —— torch profiler 口径：每步的 aten 输入形状与 kernel 表
#   2) nsys-<tag>.sqlite        —— nsys 口径：--cuda-graph-trace=node 展开图内 kernel
#   3) nsys_step_align.txt      —— 按 NVTX range 把 kernel 归到 step
#
# 用法：RUN_ID=20260922 L5STEP_TAGS="eager graph" PLENS="64 2048" \
#           bash labs/L5/run_step_kernel_align.sh
set -u

ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$ROOT/envs/serve/bin/python}
NSYS=${NSYS:-$ROOT/tools/nsight_systems-linux-x86_64-2026.3.2.313-archive/target-linux-x64/nsys}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-$ROOT/work/out/step-kernel-align-$RUN_ID}
TAGS=${L5STEP_TAGS:-"eager graph"}
PLENS=${PLENS:-"64 2048"}
NSYS_TAGS=${NSYS_TAGS:-"eager graph"}
STEPS=${STEPS:-4}
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

for tag in $TAGS; do
    eager=""
    [ "$tag" = "eager" ] && eager="--eager"
    for plen in $PLENS; do
        echo "== profiler 口径：$tag prompt=$plen =="
        "$PY" "$HERE/step_kernel_align.py" --out "$OUT/profiler-${tag}_p${plen}" \
            --prompt-len "$plen" --steps "$STEPS" $eager 2>&1 | tail -12
    done
done

for tag in $NSYS_TAGS; do
    eager=""
    [ "$tag" = "eager" ] && eager="--eager"
    echo "== nsys 口径：$tag =="
    rm -rf "$OUT/nsys-$tag.nsys-rep" "$OUT/nsys-$tag.sqlite"
    "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true \
        -o "$OUT/nsys-$tag" \
        "$PY" "$HERE/step_kernel_align.py" --out "$OUT/nvtx-$tag" --no-profiler \
        --prompt-len 64 --steps "$STEPS" $eager > "$OUT/nsys-$tag.log" 2>&1
    "$NSYS" export --type sqlite --force-overwrite true \
        -o "$OUT/nsys-$tag.sqlite" "$OUT/nsys-$tag.nsys-rep" >/dev/null 2>&1
done

{
    for tag in $NSYS_TAGS; do
        if [ -f "$OUT/nsys-$tag.sqlite" ]; then
            "$PY" "$HERE/summarize_step_nsys.py" "$OUT/nsys-$tag.sqlite" "$tag"
        fi
    done
} > "$OUT/nsys_step_align.txt" 2>&1
cat "$OUT/nsys_step_align.txt"
echo "工件目录：$OUT"
