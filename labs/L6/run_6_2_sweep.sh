#!/bin/bash
# 6.2 任务 B：Qwen3-8B 的 TP=1/2/4 与 PP=2 扫描。
# 每个配置重建一次引擎，配置内复用引擎跑完 (B,S) 组合。
set -u
source /root/learn/env.sh >/dev/null 2>&1
export HF_HUB_CACHE="$HF_HOME/hub"
cd /root/learn/work/labs/L6
MODEL=/root/learn/models/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT=/root/learn/work/out/6.2/20260914-parallel
PY=/root/learn/envs/serve/bin/python
mkdir -p "$OUT"

free_gib() {  # 参数：gpu 列表，输出最小空闲 GiB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v want="$1" 'BEGIN{split(want,a,","); for(i in a) w[a[i]]=1}
        {gsub(/ /,"",$1); if ($1 in w) {v=$2/1024; if (m=="" || v<m) m=v}}
        END{printf "%.1f", m}'
}

run() {
  local name=$1 tp=$2 pp=$3 gpus=$4; shift 4
  local free; free=$(free_gib "$gpus")
  echo "=== $name tp=$tp pp=$pp gpus=$gpus free=${free}GiB $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpus VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    timeout 2400 $PY parallel_inference.py sweep --model "$MODEL" \
      --tp "$tp" --pp "$pp" --out "$OUT/$name" --log "$OUT/$name.log" \
      --max-tokens 128 --gpu-util 0.70 "$@"
  local rc=$?
  echo "=== $name rc=$rc $(date -u +%H:%M:%S)"
  pkill -9 -f "VLLM::Worker" 2>/dev/null
  sleep 10
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
  echo
}

run B_tp1 1 1 0 --full
run B_tp2 2 1 0,1 --full
run B_tp4 4 1 0,1,2,3
run B_pp2 1 2 0,1
echo "ALL DONE $(date -u +%H:%M:%S)"
