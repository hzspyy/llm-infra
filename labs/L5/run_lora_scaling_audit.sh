#!/usr/bin/env bash
set -euo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export L510_GPU_MEMORY_UTILIZATION=0.60
run_root="/scratch/learn/work/out/lora-scaling-${1:?pass unique run id}"
mkdir "$run_root"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
  printf '%s\n' 'GPU occupied: no experiment started' > "$run_root/blocked.txt"
  exit 1
fi
cp /scratch/learn/work/labs/L5/lora_scaling_audit.py "$run_root/lora_scaling_audit.py.snapshot"
cp "$0" "$run_root/run_lora_scaling_audit.sh.snapshot"
set +e
/scratch/learn/envs/serve/bin/python -u /scratch/learn/work/labs/L5/lora_scaling_audit.py --out "$run_root/data" > "$run_root/run.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_root/exit.txt"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
exit "$code"
