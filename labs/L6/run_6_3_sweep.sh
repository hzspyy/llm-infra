#!/bin/bash
# 6.3 任务 B：MoE dispatch/combine 的通信扫描。
# tokens=1/32/256/4096，top-k=2/8，倾斜=0/20%/50%，两条路线，2 与 4 卡。
set -u
source /root/learn/env.sh >/dev/null 2>&1
cd /root/learn/work/labs/L6
OUT=/root/learn/work/out/6.3/20260914-moe/B
PY=/root/learn/envs/serve/bin/python
mkdir -p "$OUT"

run() {
  local n=$1 route=$2 t=$3 k=$4 s=$5 gpus=$6
  echo "=== n=$n route=$route tokens=$t topk=$k skew=$s $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpus timeout 600 $PY moe_dispatch.py B \
    --out "$OUT" --world-size "$n" --tokens "$t" --topk "$k" --skew "$s" \
    --experts 64 --hidden 256 --route "$route" 2>&1 | grep -E "^\[B:|^     " | head -4
  echo "=== rc=$? $(date -u +%H:%M:%S)"
  pkill -9 -f "VLLM::Worke[r]" 2>/dev/null
  sleep 3
}

# 主扫描：4 卡，all_to_all
for t in 1 32 256 4096; do
  for k in 2 8; do
    run 4 alltoall "$t" "$k" 0.0 0,1,2,3
  done
done
# 倾斜
for s in 0.2 0.5; do
  for k in 2 8; do
    run 4 alltoall 256 "$k" "$s" 0,1,2,3
  done
done
# 对照路线：all_gather
run 4 allgather 256 8 0.0 0,1,2,3
run 4 allgather 4096 8 0.0 0,1,2,3
# 2 卡对照
run 2 alltoall 256 8 0.0 0,1
run 2 alltoall 4096 8 0.0 0,1
run 2 allgather 256 8 0.0 0,1
run 2 allgather 4096 8 0.0 0,1
echo "ALL DONE $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
