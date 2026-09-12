#!/usr/bin/env bash
set -u
source /scratch/learn/env.sh
cd /scratch/learn
run_root=work/out/sampling-20260911-1045
mkdir -p "$run_root"
export L59_GPU_MEMORY_UTILIZATION=0.60
for mode in vllm sglang gpu engine; do
  python_bin=/scratch/learn/envs/serve/bin/python
  if [ "$mode" = sglang ]; then python_bin=/scratch/learn/envs/sgl/bin/python; fi
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/$mode.gpu-before.txt"
  "$python_bin" -u work/labs/L5/sampling_audit.py "$mode" --out "$run_root/$mode" > "$run_root/$mode.log" 2>&1
  echo $? > "$run_root/$mode.exit"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/$mode.gpu-after.txt"
done
