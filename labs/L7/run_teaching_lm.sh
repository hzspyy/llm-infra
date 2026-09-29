#!/usr/bin/env bash
# 教学语言模型项目入口（7.8-I 数据 -> 7.1-F 预训练 -> 7.5-I SFT -> 评测）。
#
# 复用 MiniMind 的模型与分词实现（固定 commit），本入口只负责数据准备、预算停止
# 条件、测量与阶段产物。所有产物写在 $RUN 下；$RUN 必须是新目录。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   bash labs/L7/run_teaching_lm.sh data
#   bash labs/L7/run_teaching_lm.sh pilot
#   bash labs/L7/run_teaching_lm.sh pretrain
#   bash labs/L7/run_teaching_lm.sh eval-pretrain
#   bash labs/L7/run_teaching_lm.sh sft
#   bash labs/L7/run_teaching_lm.sh eval-sft
#
# 预算上限（共同任务的单项目范围）：预训练 + SFT 共 12 GPU 小时；pilot 计入预算。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
DATA=${DATA:-$LEARN_ROOT/models/hf/hub/datasets--jingyaogong--minimind_dataset/snapshots/312afb4f76391145c6902f765bb51691c09a12f5}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的新目录>}

PRETRAIN_TOKENS=${PRETRAIN_TOKENS:-80000000}
VAL_TOKENS=${VAL_TOKENS:-2000000}
SEQ_LEN=${SEQ_LEN:-512}
MICRO_BS=${MICRO_BS:-32}
MAX_STEPS=${MAX_STEPS:-6000}
PRETRAIN_SECONDS=${PRETRAIN_SECONDS:-14400}      # 预训练 4 GPU 小时上限
SFT_MAX_STEPS=${SFT_MAX_STEPS:-1200}
SFT_SECONDS=${SFT_SECONDS:-5400}                 # SFT 1.5 GPU 小时上限

cmd=${1:-}
case "$cmd" in
  data)
    mkdir -p "$RUN"
    "$PY" "$LABS/L7/teaching_data.py" pretrain \
      --raw "$DATA/pretrain_t2t_mini.jsonl" --tokenizer "$SRC/model" \
      --outdir "$RUN/data-pretrain" --budget-tokens "$PRETRAIN_TOKENS" \
      --val-tokens "$VAL_TOKENS" --seq-len "$SEQ_LEN" \
      --source-revision 312afb4f76391145c6902f765bb51691c09a12f5
    "$PY" "$LABS/L7/teaching_data.py" sft \
      --raw "$DATA/sft_t2t_mini.jsonl" --tokenizer "$SRC/model" \
      --outdir "$RUN/data-sft" --max-samples 8000 --max-len 768 \
      --source-revision 312afb4f76391145c6902f765bb51691c09a12f5
    ;;
  pilot)
    # 100-300 step 或最多 15 分钟：先量有效处理速度、峰值与评测/保存成本
    "$PY" "$LABS/L7/teaching_pretrain.py" \
      --data-dir "$RUN/data-pretrain" --minimind-src "$SRC" \
      --outdir "$RUN/pilot" --seq-len "$SEQ_LEN" --micro-bs "$MICRO_BS" \
      --max-steps 200 --limit-seconds 900 --eval-every 100 --save-every 100 \
      --sample-every 100 --sample-tokens 32
    ;;
  pretrain)
    "$PY" "$LABS/L7/teaching_pretrain.py" \
      --data-dir "$RUN/data-pretrain" --minimind-src "$SRC" \
      --outdir "$RUN/pretrain" --seq-len "$SEQ_LEN" --micro-bs "$MICRO_BS" \
      --max-steps "$MAX_STEPS" --limit-seconds "$PRETRAIN_SECONDS" \
      --eval-every 250 --save-every 250 --sample-every 500
    ;;
  eval-pretrain)
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" --random-init \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/random-init" --label random-init
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
      --checkpoint "$RUN/pretrain/best.pt" --pretrain-data-dir "$RUN/data-pretrain" \
      --outdir "$RUN/eval/pretrain-best" --label pretrain-best
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
      --checkpoint "$RUN/pretrain/last.pt" --pretrain-data-dir "$RUN/data-pretrain" \
      --outdir "$RUN/eval/pretrain-last" --label pretrain-last
    ;;
  sft)
    "$PY" "$LABS/L7/teaching_sft.py" \
      --data-dir "$RUN/data-sft" --minimind-src "$SRC" \
      --init-from "$RUN/pretrain/best.pt" --outdir "$RUN/sft" \
      --max-steps "$SFT_MAX_STEPS" --limit-seconds "$SFT_SECONDS" \
      --eval-every 150 --save-every 150
    ;;
  eval-sft)
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
      --checkpoint "$RUN/pretrain/best.pt" --sft-data-dir "$RUN/data-sft" \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/base-on-sft" --label base
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
      --checkpoint "$RUN/sft/best.pt" --sft-data-dir "$RUN/data-sft" \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/sft-best" --label sft
    "$PY" "$LABS/L7/teaching_eval.py" --minimind-src "$SRC" \
      --checkpoint "$RUN/sft/last.pt" --sft-data-dir "$RUN/data-sft" \
      --pretrain-data-dir "$RUN/data-pretrain" --outdir "$RUN/eval/sft-last" --label sft-last
    ;;
  export)
    "$PY" "$LABS/L7/teaching_export.py" --checkpoint "$RUN/pretrain/best.pt" \
      --out "$SRC/out/pretrain_768.pth" --stage pretrain --outdir "$RUN/export/pretrain"
    "$PY" "$LABS/L7/teaching_export.py" --checkpoint "$RUN/sft/best.pt" \
      --out "$SRC/out/full_sft_768.pth" --stage sft --outdir "$RUN/export/sft"
    echo "上游推理入口： cd $SRC && $PY eval_llm.py --weight full_sft --hidden_size 768"
    ;;
  cost)
    "$PY" "$LABS/L7/teaching_cost_report.py" --run-dir "$RUN" --outdir "$RUN/cost"
    ;;
  *)
    echo "usage: RUN=<dir> bash $0 {data|pilot|pretrain|eval-pretrain|sft|eval-sft|export|cost}" >&2
    exit 2
    ;;
esac
