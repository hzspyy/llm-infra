#!/usr/bin/env bash
# 可验证奖励 RL 入口（7.6-J）：可验证计数/算术任务上的 GRPO，规则奖励、组内基线、权重同步与评测。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   TASK=count|arith bash labs/L7/run_teaching_rl.sh sweep   # 三个难度各测基础成功率，用来选主任务
#   bash labs/L7/run_teaching_rl.sh data        # 按 DIFFICULTY 生成主任务集
#   bash labs/L7/run_teaching_rl.sh base        # 起点（SFT）在验证集上的成功率
#   bash labs/L7/run_teaching_rl.sh train       # GRPO
#   bash labs/L7/run_teaching_rl.sh eval        # 起点与 RL 权重在留出测试集上对照
#
# 任务完全可验证（出题时已知答案，判分取回复中的最终整数），不引入奖励模型。奖励含正确性、
# 格式与重复惩罚三项；GRPO 用同一 prompt 的 G 条采样做组内基线，零方差组被单独计数。
# 预算 4 GPU 小时；起点是同一 SFT 权重，对照即为该起点。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的目录>}

TASK=${TASK:-count}
DIFFICULTY=${DIFFICULTY:-medium}
RL_STEPS=${RL_STEPS:-200}
GROUP_SIZE=${GROUP_SIZE:-8}
MICRO_BS=${MICRO_BS:-16}
RL_SECONDS=${RL_SECONDS:-14400}
RL_OUT=${RL_OUT:-rl-grpo}
TEMPERATURE=${TEMPERATURE:-0.9}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-1.0}
LR=${LR:-1e-6}
TASK_DIR="$RUN/rl-data"

cmd=${1:-}
case "$cmd" in
  sweep)
    for level in easy medium hard; do
      "$PY" "$LABS/L7/teaching_grpo.py" --build-data "$RUN/rl-data-$TASK-$level" \
        --task "$TASK" --difficulty "$level" --n-train 2000 --n-val 200 --n-test 200
      "$PY" "$LABS/L7/teaching_grpo.py" --eval-only --task-dir "$RUN/rl-data-$TASK-$level" \
        --minimind-src "$SRC" --checkpoint "$RUN/sft/best.pt" --split val \
        --outdir "$RUN/rl-eval/sweep-$TASK-$level" --micro-bs 16 --max-new-tokens 192
    done
    ;;
  data)
    "$PY" "$LABS/L7/teaching_grpo.py" --build-data "$TASK_DIR" \
      --task "$TASK" --difficulty "$DIFFICULTY" --n-train 2000 --n-val 200 --n-test 200
    ;;
  base)
    "$PY" "$LABS/L7/teaching_grpo.py" --eval-only --task-dir "$TASK_DIR" \
      --minimind-src "$SRC" --checkpoint "$RUN/sft/best.pt" --split val \
      --outdir "$RUN/rl-eval/base-val" --micro-bs 16 --max-new-tokens 192
    ;;
  train)
    "$PY" "$LABS/L7/teaching_grpo.py" --task-dir "$TASK_DIR" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/$RL_OUT" \
      --group-size "$GROUP_SIZE" --micro-bs "$MICRO_BS" --max-steps "$RL_STEPS" \
      --limit-seconds "$RL_SECONDS" --lr "$LR" --temperature "$TEMPERATURE" --top-p 0.95 \
      --format-weight "$FORMAT_WEIGHT" \
      --max-new-tokens 192 --eval-every 25 --save-every 25
    ;;
  train-kl)
    "$PY" "$LABS/L7/teaching_grpo.py" --task-dir "$TASK_DIR" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/rl-grpo-kl" \
      --group-size "$GROUP_SIZE" --micro-bs "$MICRO_BS" --max-steps "$RL_STEPS" \
      --limit-seconds "$RL_SECONDS" --lr "$LR" --temperature "$TEMPERATURE" --top-p 0.95 \
      --beta-kl 0.02 --format-weight "$FORMAT_WEIGHT" \
      --max-new-tokens 192 --eval-every 25 --save-every 25
    ;;
  eval)
    "$PY" "$LABS/L7/teaching_grpo.py" --eval-only --task-dir "$TASK_DIR" \
      --minimind-src "$SRC" --pth "$RUN/$RL_OUT/best.pth" --split test \
      --outdir "$RUN/rl-eval/grpo-$RL_OUT-test" --micro-bs 16 --max-new-tokens 192
    "$PY" "$LABS/L7/teaching_grpo.py" --eval-only --task-dir "$TASK_DIR" \
      --minimind-src "$SRC" --checkpoint "$RUN/sft/best.pt" --split test \
      --outdir "$RUN/rl-eval/base-test" --micro-bs 16 --max-new-tokens 192
    ;;
  *)
    echo "usage: RUN=<dir> bash $0 {sweep|data|base|train|train-kl|eval}" >&2
    exit 2
    ;;
esac
