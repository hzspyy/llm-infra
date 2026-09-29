#!/usr/bin/env bash
# 8.6 KV 作为存储层级: A(exact) / C(errors) / D(approx) / B(scan) / E(真实引擎层) 五段实测。
#
#   PY  venv 解释器 (默认远程 serve 环境)
#   OUT 输出目录 (默认远程学习目录下的 out/8.6/<run_id>)
#   LAB lab 代码目录
set -euo pipefail
PY=${PY:-/scratch/learn/envs/serve/bin/python}
OUT=${OUT:-/scratch/learn/work/out/8.6/20260921-kvtier}
LAB=${LAB:-/scratch/learn/work/labs}
mkdir -p "$OUT"
cd "$LAB"

echo "== A: exact ($(date -u +%FT%TZ))"
"$PY" L8/kv_retrieve_vs_recompute.py exact --out-dir "$OUT/A" \
      --exact-len 2048 --decode-steps 8 2>&1 | tee "$OUT/A.log"

echo "== C: errors ($(date -u +%FT%TZ))"
"$PY" L8/kv_retrieve_vs_recompute.py errors --out-dir "$OUT/C" \
      --err-len 512 --ttl 0.5 2>&1 | tee "$OUT/C.log"

echo "== D: approx ($(date -u +%FT%TZ))"
"$PY" L8/kv_retrieve_vs_recompute.py approx --out-dir "$OUT/D" \
      --ctx-len 4096 --needle-positions 0.1,0.5,0.9 --budgets 0.1,0.25 \
      --answer-steps 20 2>&1 | tee "$OUT/D.log"

echo "== B: scan ($(date -u +%FT%TZ))"
# 逐长度分开跑: 单段失败不影响其它长度; gap 扫描只在交叉点附近的两个长度上做。
for L in 128 2048 8192 32768; do
  "$PY" L8/kv_retrieve_vs_recompute.py scan --out-dir "$OUT/B/L$L" \
        --lengths "$L" --max-len 32768 --gaps 0,1,10,60 --gap-lengths 2048,8192 \
        --ttls 0,5 --concurrency 1,8 --conc-budget-gib 12 --decode-steps 8 \
        2>&1 | tee "$OUT/B/L$L.log"
done

echo "== E: 真实引擎的 KV offloading ($(date -u +%FT%TZ))"
# none / cpu / tiering 三档, tiering 结束后重启引擎检验 fs 层恢复
"$PY" L8/kv_offload_engine.py --out-dir "$OUT/E" --modes none,cpu,tiering \
      --python "$PY" --prefix-len 4096 --num-prefixes 8 --num-flush 8 \
      --gpu-mem-util 0.30 --max-model-len 8192 --cpu-gib 8 --restart \
      2>&1 | tee "$OUT/E.log"

echo "ALL DONE $(date -u +%FT%TZ)"
