#!/usr/bin/env bash
# 视觉适配入口（7.10-J）：冻结 CLIP 视觉塔 + 随机 projector + 教学 LLM 的两阶段训练。
#
#   export RUN=/scratch/learn/work/out/teaching-lm-20260922
#   export CLIP=/scratch/learn/models/hf/hub/models--openai--clip-vit-base-patch32/snapshots/<rev>
#   bash labs/L7/run_teaching_vlm.sh data
#   bash labs/L7/run_teaching_vlm.sh base          # 随机 projector + SFT LLM 的未适配基线
#   bash labs/L7/run_teaching_vlm.sh align         # 只训练 projector（描述文本）
#   bash labs/L7/run_teaching_vlm.sh sft           # projector + LLM 联合训练（问答）
#   bash labs/L7/run_teaching_vlm.sh eval
#
# 数据是本地合成的形状图案问答，按图片哈希做图像级隔离；判据是留出问答的精确匹配，
# 并用换图/空图两个对照检验模型是否真的在使用图像。预算 8 GPU 小时（视觉适配分支上限）。
set -euo pipefail

LEARN_ROOT=${LEARN_ROOT:-/scratch/learn}
PY=${PY:-$LEARN_ROOT/envs/serve/bin/python}
SRC=${SRC:-$LEARN_ROOT/opt/src/minimind}
LABS=${LABS:-$LEARN_ROOT/work/labs}
RUN=${RUN:?请先 export RUN=<学习盘上的目录>}
CLIP=${CLIP:?请先 export CLIP=<CLIP ViT-B/32 快照目录>}

ALIGN_STEPS=${ALIGN_STEPS:-1500}
SFT_STEPS=${SFT_STEPS:-1500}
VLM_SECONDS=${VLM_SECONDS:-14400}
EVAL_BATCH=${EVAL_BATCH:-8}

cmd=${1:-}
common=(--data-dir "$RUN/vlm-data" --minimind-src "$SRC" --vision-path "$CLIP")
case "$cmd" in
  data)
    "$PY" "$LABS/L7/teaching_vlm.py" --mode data --outdir "$RUN/vlm-data"
    ;;
  base)
    "$PY" "$LABS/L7/teaching_vlm.py" --mode eval "${common[@]}" \
      --ckpt "$RUN/sft/best.pt" --split test \
      --outdir "$RUN/vlm-eval/base" --micro-bs "$EVAL_BATCH"
    ;;
  align)
    "$PY" "$LABS/L7/teaching_vlm.py" --mode align "${common[@]}" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/vlm-align" \
      --lr 5e-4 --micro-bs 8 --max-steps "$ALIGN_STEPS" --limit-seconds "$VLM_SECONDS" \
      --eval-every 250 --save-every 250
    ;;
  sft)
    "$PY" "$LABS/L7/teaching_vlm.py" --mode sft "${common[@]}" \
      --init-from "$RUN/sft/best.pt" --align-from "$RUN/vlm-align/best.pt" \
      --outdir "$RUN/vlm-sft" \
      --lr 1e-4 --micro-bs 8 --max-steps "$SFT_STEPS" --limit-seconds "$VLM_SECONDS" \
      --eval-every 250 --save-every 250
    ;;
  sft-long)
    # 诊断 arm：跳过对齐阶段、直接联合训练，步数与 lr 都放大，检验"不涨"是不是优化预算问题
    "$PY" "$LABS/L7/teaching_vlm.py" --mode sft "${common[@]}" \
      --init-from "$RUN/sft/best.pt" --outdir "$RUN/vlm-sft-long" \
      --lr 3e-4 --micro-bs 8 --max-steps 6000 --limit-seconds "$VLM_SECONDS" \
      --eval-every 1000 --save-every 1000
    ;;
  sft-long-stage1)
    # 同样的长预算，但 projector 用对齐阶段的产物初始化：检验对齐阶段在预算充足时是否有用
    "$PY" "$LABS/L7/teaching_vlm.py" --mode sft "${common[@]}" \
      --init-from "$RUN/sft/best.pt" --align-from "$RUN/vlm-align/best.pt" \
      --outdir "$RUN/vlm-sft-long-stage1" \
      --lr 3e-4 --micro-bs 8 --max-steps 6000 --limit-seconds "$VLM_SECONDS" \
      --eval-every 1000 --save-every 1000
    ;;
  eval)
    for arm in align sft; do
      "$PY" "$LABS/L7/teaching_vlm.py" --mode eval "${common[@]}" \
        --ckpt "$RUN/vlm-$arm/best.pt" --split test \
        --outdir "$RUN/vlm-eval/$arm" --micro-bs "$EVAL_BATCH"
    done
    ;;
  *)
    echo "usage: RUN=<dir> CLIP=<snapshot> bash $0 {data|base|align|sft|sft-long|sft-long-stage1|eval}" >&2
    exit 2
    ;;
esac
