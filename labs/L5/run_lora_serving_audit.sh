#!/usr/bin/env bash
set -u
source /scratch/learn/env.sh
cd /scratch/learn
run_root=work/out/lora-20260911-1110-r2
mkdir -p "$run_root"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
  echo 'GPU is occupied; no experiment started' > "$run_root/blocked.txt"
  exit 1
fi
export L510_GPU_MEMORY_UTILIZATION=0.60
/scratch/learn/envs/serve/bin/python -u work/labs/L5/lora_serving_audit.py --out "$run_root/data" > "$run_root/run.log" 2>&1
echo $? > "$run_root/exit.txt"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
