#!/usr/bin/env bash
# 8.4 混合负载与容量: A(单任务基线) / B(混部份额扫描) / C(干预) / D(准入边界) 四段实测.
#
#   PY  venv 解释器 (默认远程 serve 环境)
#   OUT 输出目录
set -euo pipefail
PY=${PY:-/scratch/learn/envs/serve/bin/python}
OUT=${OUT:-/scratch/learn/work/out/8.4/20260921-mixed}
LAB=${LAB:-/scratch/learn/work/labs}
mkdir -p "$OUT"
cd "$LAB"

COMMON="--python $PY --gen-rate 4 --emb-rate 8 --duration-s 60 \
        --gen-prompt-len 512 --emb-prompt-len 512 --gen-output-len 64"

echo "== A: 单任务基线 ($(date -u +%FT%TZ))"
"$PY" L8/mixed_workload.py baseline --out-dir "$OUT/baseline" $COMMON 2>&1 | tee "$OUT/baseline.log"

echo "== B: 混部份额扫描 0/0.25/0.5/0.75/1.0 ($(date -u +%FT%TZ))"
"$PY" L8/mixed_workload.py mixed --out-dir "$OUT/mixed" --shares 0,0.25,0.5,0.75,1.0 \
      $COMMON 2>&1 | tee "$OUT/mixed.log"

echo "== C: 干预 (向量化串行化 / 生成关前缀缓存 / 时间分片) ($(date -u +%FT%TZ))"
"$PY" L8/mixed_workload.py intervene --out-dir "$OUT/intervene" --share-emb 0.5 \
      $COMMON 2>&1 | tee "$OUT/intervene.log"

echo "== D: 准入预算与容量边界 ($(date -u +%FT%TZ))"
"$PY" L8/admission_budget.py --out-dir "$OUT/admission" --python "$PY" \
      --gen-rate 4 --emb-rate 8 --duration-s 60 --gen-prompt-len 512 --emb-prompt-len 512 \
      --gen-output-len 64 --gen-budgets 1,2,4 --emb-budgets 4,8,16 \
      2>&1 | tee "$OUT/admission.log"

echo "ALL DONE $(date -u +%FT%TZ)"
