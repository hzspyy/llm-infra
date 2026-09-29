#!/usr/bin/env bash
# L5 公共 SERVE 协议扫描（5.1 / 5.3 / 5.12 复用）。
# 用法： run_serve_protocol.sh <scan> <out_dir> [configs]
set -euo pipefail

PY=/scratch/learn/envs/serve/bin/python
LEARN=/scratch/learn
SCAN="${1:-b}"
OUT="${2:-$LEARN/work/out/5.1/serve-$SCAN}"
CONFIGS="${3:-}"

source "$LEARN/env.sh"

EXTRA=""
if [ -n "$CONFIGS" ]; then EXTRA="--configs $CONFIGS"; fi

cd "$LEARN/work/labs/L5"
"$PY" serve_protocol.py --scan "$SCAN" --out "$OUT" $EXTRA 2>&1 | tee "$OUT/run.log"