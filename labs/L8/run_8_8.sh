#!/usr/bin/env bash
# 8.8 多租户、隔离与安全: A(身份与归属) / B(干扰对照) / C(取消与重启) 三段实测。
#
#   PY  venv 解释器 (默认远程 serve 环境)
#   OUT 输出目录
set -euo pipefail
PY=${PY:-/scratch/learn/envs/serve/bin/python}
OUT=${OUT:-/scratch/learn/work/out/8.8/20260921-tenant}
LAB=${LAB:-/scratch/learn/work/labs}
mkdir -p "$OUT"
cd "$LAB"

echo "== A/C(CPU): 多租户调度、身份审计、取消与重启 ($(date -u +%FT%TZ))"
"$PY" L8/tenant_scheduler.py --out "$OUT/scheduler" 2>&1 | tee "$OUT/scheduler.log"

echo "== A(引擎): adapter revision 与缓存身份 ($(date -u +%FT%TZ))"
"$PY" L8/tenant_interference.py identity --out-dir "$OUT/identity" --python "$PY" \
      2>&1 | tee "$OUT/identity.log"

echo "== B(引擎): 共享 vs 进程隔离的干扰对照 ($(date -u +%FT%TZ))"
"$PY" L8/tenant_interference.py interference --out-dir "$OUT/interference" --python "$PY" \
      --a-count 60 --b-count 12 --interval-s 1.0 --duration-s 60 \
      2>&1 | tee "$OUT/interference.log"

echo "ALL DONE $(date -u +%FT%TZ)"
