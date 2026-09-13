#!/bin/bash
source /scratch/learn/env.sh
cd /scratch/learn/work/m2-protocol-20260913
export TORCHINDUCTOR_CACHE_DIR=/scratch/learn/.cache/m2-protocol-20260913/inductor
export TRITON_CACHE_DIR=/scratch/learn/.cache/m2-protocol-20260913/triton
export TMPDIR=/scratch/learn/.cache/m2-protocol-20260913/tmp
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > processes-before.txt
/scratch/learn/envs/serve/bin/python -B measurement_protocol.py --compile-probe --output /scratch/learn/work/out/M2/20260913-protocol-a > run.log 2>&1
code=$?
echo "$code" > process-exit.txt
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > gpu-after.txt
