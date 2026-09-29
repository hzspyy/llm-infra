#!/usr/bin/env bash
# 后训练分支入口（7.5-J）：同一 SFT 起点上的 LoRA 领域适配与 DPO 偏好优化。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   bash labs/L7/run_teaching_posttrain.sh data
#   bash labs/L7/run_teaching_posttrain.sh lora-pilot
#   bash labs/L7/run_teaching_posttrain.sh lora
#   bash labs/L7/run_teaching_posttrain.sh lora-full-ft      # 等数据/等步数的全参对照
#   bash labs/L7/run_teaching_posttrain.sh dpo-pilot
#   bash labs/L7/run_teaching_posttrain.sh dpo
#   bash labs/L7/run_teaching_posttrain.sh eval
#
# 训练用本项目自己的循环（`teaching_lora.py` / `teaching_dpo.py`）：数据装载、损失与
# 注入方式复用上游 MiniMind 实现，DPO 启动时还会与上游 `train_dpo.py` 损失做数值对拍；
# 新增的是逐步记录、held-out 曲线、步级 checkpoint、续训与导出。
# 两个分支的预算：每个 4 GPU 小时（共同任务的后训练分支上限）。pilot 计入预算。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
DATA=${DATA:-$LEARN_ROOT/models/hf/hub/datasets--jingyaogong--minimind_dataset/snapshots/312afb4f76391145c6902f765bb51691c09a12f5}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的目录>}

LORA_STEPS=${LORA_STEPS:-1000}
DPO_STEPS=${DPO_STEPS:-2000}
LORA_SECONDS=${LORA_SECONDS:-14400}
DPO_SECONDS=${DPO_SECONDS:-14400}
PILOT_SECONDS=${PILOT_SECONDS:-300}

cmd=${1:-}
case "$cmd" in
  data)
    "$PY" "$LABS/L7/teaching_data.py" lora \
      --raw "$DATA/sft_t2t_mini.jsonl" --outdir "$RUN/data-lora" \
      --reservoir 10000 --max-samples 8000 \
      --source-revision 312afb4f76391145c6902f765bb51691c09a12f5
    "$PY" "$LABS/L7/teaching_data.py" dpo \
      --raw "$DATA/dpo.jsonl" --outdir "$RUN/data-dpo" \
      --reservoir 12000 --max-pairs 8000 \
      --source-revision 312afb4f76391145c6902f765bb51691c09a12f5
    ;;
  lora-pilot)
    # 只跑限时窗口：量每步时间、峰值与 held-out 子技能起点，再决定正式步数
    "$PY" "$LABS/L7/teaching_lora.py" \
      --data-dir "$RUN/data-lora" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/lora-pilot" \
      --max-steps "$LORA_STEPS" --limit-seconds "$PILOT_SECONDS" \
      --rank 16 --micro-bs 32 --lr 1e-4 --max-len 768 \
      --log-every 25 --eval-every 250 --save-every 250
    ;;
  lora)
    "$PY" "$LABS/L7/teaching_lora.py" \
      --data-dir "$RUN/data-lora" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/lora" \
      --max-steps "$LORA_STEPS" --limit-seconds "$LORA_SECONDS" \
      --rank 16 --micro-bs 32 --lr 1e-4 --max-len 768 \
      --log-every 25 --eval-every 250 --save-every 250
    ;;
  lora-full-ft)
    # 等数据、同步数、同起点的全参对照（上游 train_full_sft.py 的默认超参）
    cd "$SRC/trainer" && "$PY" train_full_sft.py \
      --data_path "$RUN/data-lora/train.jsonl" --save_weight full_lora_control \
      --from_weight full_sft --epochs 1 --batch_size 32 --learning_rate 1e-5 \
      --max_seq_len 768 --num_workers 4 --log_interval 50 --save_interval 1000 \
      --save_dir ../out
    ;;
  dpo-pilot)
    "$PY" "$LABS/L7/teaching_dpo.py" \
      --data-dir "$RUN/data-dpo" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/dpo-pilot" \
      --max-steps "$DPO_STEPS" --limit-seconds "$PILOT_SECONDS" \
      --beta 0.15 --micro-bs 4 --lr 4e-8 --max-len 1024 \
      --log-every 50 --eval-every 100 --save-every 500
    ;;
  dpo)
    "$PY" "$LABS/L7/teaching_dpo.py" \
      --data-dir "$RUN/data-dpo" --minimind-src "$SRC" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/dpo" \
      --max-steps "$DPO_STEPS" --limit-seconds "$DPO_SECONDS" \
      --beta 0.15 --micro-bs 4 --lr 4e-8 --max-len 1024 \
      --log-every 50 --eval-every 250 --save-every 500
    ;;
  eval)
    EVAL=(--minimind-src "$SRC" --n-prompts 120 --n-pairs 300
          --sft-data-dir "$RUN/data-sft" --pretrain-data-dir "$RUN/data-pretrain")
    "$PY" "$LABS/L7/teaching_preference_eval.py" "${EVAL[@]}" \
      --checkpoint "$RUN/sft/best.pt" --lora-data-dir "$RUN/data-lora" \
      --label base-on-lora --outdir "$RUN/eval/base-on-lora"
    "$PY" "$LABS/L7/teaching_preference_eval.py" "${EVAL[@]}" \
      --pth "$SRC/out/full_lora_control_768.pth" --lora-data-dir "$RUN/data-lora" \
      --label full-ft-on-lora --outdir "$RUN/eval/full-ft-on-lora"
    "$PY" "$LABS/L7/teaching_preference_eval.py" "${EVAL[@]}" \
      --checkpoint "$RUN/sft/best.pt" --lora "$RUN/lora/best.pth" --merge-check \
      --lora-data-dir "$RUN/data-lora" --label lora --outdir "$RUN/eval/lora"
    "$PY" "$LABS/L7/teaching_preference_eval.py" "${EVAL[@]}" \
      --checkpoint "$RUN/sft/best.pt" --reference "$RUN/sft/best.pt" \
      --dpo-data-dir "$RUN/data-dpo" --label base-on-dpo --outdir "$RUN/eval/base-on-dpo"
    "$PY" "$LABS/L7/teaching_preference_eval.py" "${EVAL[@]}" \
      --pth "$RUN/dpo/best.pth" --reference "$RUN/sft/best.pt" \
      --dpo-data-dir "$RUN/data-dpo" --label dpo --outdir "$RUN/eval/dpo"
    ;;
  *)
    echo "usage: RUN=<dir> bash $0 {data|lora-pilot|lora|lora-full-ft|dpo-pilot|dpo|eval}" >&2
    exit 2
    ;;
esac
