#!/usr/bin/env bash
set -u
source /scratch/learn/env.sh
cd /scratch/learn
run_root=work/out/structured-20260911-0250
export L56_GPU_MEMORY_UTILIZATION=0.60
for backend in none xgrammar guidance outlines; do
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/$backend.gpu-before.txt"
  /scratch/learn/envs/serve/bin/python -u work/labs/L5/structured_output_audit.py bench --backend "$backend" --out "$run_root/$backend" > "$run_root/$backend.log" 2>&1
  rc=$?
  echo "$rc" > "$run_root/$backend.exit"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/$backend.gpu-after.txt"
done
