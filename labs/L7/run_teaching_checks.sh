#!/usr/bin/env bash
# 教学训练的机制对拍入口：7.4-H 中断恢复、7.9-I 精度/优化器限定窗口。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   bash labs/L7/run_teaching_checks.sh resume
#   bash labs/L7/run_teaching_checks.sh precision-window
#
# resume：A 连续跑 300 step；B 跑 150 秒后被 SIGKILL，再从 last.pt 恢复到 300 step。
#          两条运行的 cosine 总步数相同，所以逐 step 的 loss 应当一致；
#          compare 报告对齐步数、loss/grad 的最大偏差与参数逐张量最大差。
# precision-window：从同一 checkpoint 出发，各跑 200 step，比较 bf16/adamw、
#          bf16/fused、fp16/adamw 三种配置的 loss 轨迹、吞吐与峰值。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的目录>}
DATA=${DATA:-$RUN/data-pretrain}

common=(--data-dir "$DATA" --minimind-src "$SRC" --seq-len 512 --micro-bs 32
        --sample-every 0 --eval-every 50 --save-every 50 --max-steps 600
        --keep-snapshots 20)

case "${1:-}" in
  resume)
    rm -rf "$RUN/resume"
    mkdir -p "$RUN/resume"
    echo "== A：连续 600 step =="
    "$PY" "$LABS/L7/teaching_pretrain.py" "${common[@]}" --outdir "$RUN/resume/A" \
      2>&1 | tee "$RUN/resume/A.log"
    echo "== B：15 秒后 SIGKILL，冻结恢复点，再从该点跑到 600 step =="
    timeout -s KILL 15 "$PY" "$LABS/L7/teaching_pretrain.py" "${common[@]}" \
      --outdir "$RUN/resume/B" 2>&1 | tee "$RUN/resume/B-interrupted.log" || true
    cp "$RUN/resume/B/last.pt" "$RUN/resume/B/ckpt-before-resume.pt"
    "$PY" "$LABS/L7/teaching_pretrain.py" "${common[@]}" --outdir "$RUN/resume/B" \
      --resume "$RUN/resume/B/ckpt-before-resume.pt" 2>&1 | tee "$RUN/resume/B-resumed.log"
    echo "== C：从 B 的同一个恢复点重放，隔离'恢复机制'与'两次运行本身的差异' =="
    "$PY" "$LABS/L7/teaching_pretrain.py" "${common[@]}" --outdir "$RUN/resume/C" \
      --resume "$RUN/resume/B/ckpt-before-resume.pt" 2>&1 | tee "$RUN/resume/C-replay.log"
    for pair in "A B resume-A-vs-B" "A C replay-A-vs-C" "B C resume-B-vs-C"; do
      set -- $pair
      "$PY" "$LABS/L7/teaching_compare_runs.py" \
        --run-a "$RUN/resume/$1" --run-b "$RUN/resume/$2" \
        --ckpt-a "$RUN/resume/$1/last.pt" --ckpt-b "$RUN/resume/$2/last.pt" \
        --label "$3" --outdir "$RUN/resume/compare-$3" | tee "$RUN/resume/compare-$3.txt"
    done
    ;;
  precision-window)
    init="$RUN/resume/A/step50.pt"
    [ -f "$init" ] || { echo "先跑 resume 生成 $init" >&2; exit 2; }
    rm -rf "$RUN/ablation"
    mkdir -p "$RUN/ablation"
    run_variant () {
      local name=$1; shift
      "$PY" "$LABS/L7/teaching_pretrain.py" --data-dir "$DATA" --minimind-src "$SRC" \
        --seq-len 512 --micro-bs 32 --sample-every 0 --eval-every 100 --save-every 100 \
        --max-steps 200 --init-from "$init" --outdir "$RUN/ablation/$name" "$@" \
        2>&1 | tee "$RUN/ablation/$name.log"
    }
    run_variant bf16-adamw  --precision bf16 --optimizer adamw
    run_variant bf16-fused  --precision bf16 --optimizer fused
    run_variant fp16-adamw  --precision fp16 --optimizer adamw
    for variant in bf16-fused fp16-adamw; do
      "$PY" "$LABS/L7/teaching_compare_runs.py" \
        --run-a "$RUN/ablation/bf16-adamw" --run-b "$RUN/ablation/$variant" \
        --label "$variant-vs-bf16-adamw" --outdir "$RUN/ablation/compare-$variant" \
        | tee "$RUN/ablation/compare-$variant.txt"
    done
    ;;
  *)
    echo "usage: RUN=<dir> bash $0 {resume|precision-window}" >&2
    exit 2
    ;;
esac
