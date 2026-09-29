#!/usr/bin/env bash
# 蒸馏训练入口（7.7-J）：同一 SFT 教师下的小学生，在线软标签、纯监督对照与 top-k 教师缓存。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   bash labs/L7/run_teaching_distill.sh kd
#   bash labs/L7/run_teaching_distill.sh ce
#   bash labs/L7/run_teaching_distill.sh cache-build
#   bash labs/L7/run_teaching_distill.sh cache-online
#   bash labs/L7/run_teaching_distill.sh cache-use
#   bash labs/L7/run_teaching_distill.sh eval
#   bash labs/L7/run_teaching_distill.sh cost
#
# 教师是 7.5-I 的 SFT 权重（768/8、63.9M），学生是随机初始化的 512/8（30.0M），同词表。
# 蒸馏分支预算 4 GPU 小时；pilot 与缓存构建计入预算。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的目录>}

STUDENT_HIDDEN=${STUDENT_HIDDEN:-512}
STUDENT_LAYERS=${STUDENT_LAYERS:-8}
KD_STEPS=${KD_STEPS:-10000}
DISTILL_SECONDS=${DISTILL_SECONDS:-14400}
TEACHER="$RUN/sft/best.pt"

cmd=${1:-}
case "$cmd" in
  kd-pilot)
    # 只跑限时窗口：量每步时间、峰值与教师前向占比，再决定正式步数
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-kd-pilot" --alpha 0.5 --temperature 1.5 \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --micro-bs 16 --lr 1e-4 --max-steps "$KD_STEPS" --limit-seconds 300 \
      --eval-every 500 --log-every 50
    ;;
  kd)
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-kd" --alpha 0.5 --temperature 1.5 \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --micro-bs 16 --lr 1e-4 --max-steps "$KD_STEPS" --limit-seconds "$DISTILL_SECONDS" \
      --eval-every 1000 --save-every 2000 --log-every 100
    ;;
  ce)
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-ce" --alpha 1.0 \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --micro-bs 16 --lr 1e-4 --max-steps "$KD_STEPS" --limit-seconds "$DISTILL_SECONDS" \
      --eval-every 1000 --save-every 2000 --log-every 100
    ;;
  cache-build)
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-cache-build" --alpha 0.5 --temperature 1.5 \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --build-cache "$RUN/distill-cache" --cache-samples 512 --cache-topk 64
    ;;
  cache-online)
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-cache-online" --alpha 0.5 --temperature 1.5 --subset 512 \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --micro-bs 16 --lr 1e-4 --max-steps 2000 --eval-every 500 --save-every 500
    ;;
  cache-use)
    "$PY" "$LABS/L7/teaching_distill.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" --teacher-from "$TEACHER" \
      --outdir "$RUN/distill-cache-cached" --alpha 0.5 --temperature 1.5 \
      --teacher-cache "$RUN/distill-cache" \
      --student-hidden-size "$STUDENT_HIDDEN" --student-num-layers "$STUDENT_LAYERS" \
      --micro-bs 16 --lr 1e-4 --max-steps 2000 --eval-every 500 --save-every 500
    ;;
  eval)
    for arm in distill-kd distill-ce distill-cache-online distill-cache-cached; do
      "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
        --checkpoint "$RUN/$arm/best.pt" --hidden-size "$STUDENT_HIDDEN" \
        --num-hidden-layers "$STUDENT_LAYERS" \
        --sft-data-dir "$RUN/data-sft" --pretrain-data-dir "$RUN/data-pretrain" \
        --outdir "$RUN/eval/$arm" --label "$arm" --n-prompts 200
    done
    ;;
  cost)
    "$PY" "$LABS/L7/teaching_student_cost.py" --minimind-src "$SRC" \
      --teacher "$TEACHER" --student "$RUN/distill-kd/best.pt" \
      --outdir "$RUN/distill-cost"
    ;;
  *)
    echo "usage: RUN=<dir> bash $0 {kd-pilot|kd|ce|cache-build|cache-online|cache-use|eval|cost}" >&2
    exit 2
    ;;
esac
