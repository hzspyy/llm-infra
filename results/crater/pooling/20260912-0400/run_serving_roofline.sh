#!/usr/bin/env bash
# 阶段 1/2/4 重跑：每个服务端独立输出目录，避免同名文件互相覆盖。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export ENC_MODEL="${ENC_MODEL:-BAAI/bge-small-en-v1.5}"
run_root="/scratch/learn/work/out/pooling-20260912-0400"
PY=/scratch/learn/envs/serve/bin/python

start_server() {
  $PY -m vllm.entrypoints.openai.api_server --model "$2" --served-model-name "$3" \
      --port "$1" --dtype bfloat16 --gpu-memory-utilization 0.35 --no-enable-log-requests \
      $4 > "$5" 2>&1 &
  echo $!
}
wait_health() { for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
stop() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null || true; }

srv=$(start_server 8100 "$ENC_MODEL" bge-small "--runner pooling --max-model-len 512" "$run_root/server-bge.log")
if wait_health 8100; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8100 \
      --model bge-small --out "$run_root/data-bge" --mode fixed --length 128 \
      --concurrency 1 8 32 128 256 > "$run_root/audit-bge-fixed.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-bge-fixed.txt"
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8100 \
      --model bge-small --out "$run_root/data-bge" --mode variable \
      --concurrency 1 8 32 128 256 > "$run_root/audit-bge-var.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-bge-var.txt"
fi
stop "$srv"

srv=$(start_server 8101 "$LLM_MODEL" qwen-embed "--runner pooling --convert embed --max-model-len 2048" "$run_root/server-qwen-embed.log")
if wait_health 8101; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8101 \
      --model qwen-embed --out "$run_root/data-qwen" --mode fixed --length 256 \
      --concurrency 1 8 32 128 > "$run_root/audit-qwen-embed.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-qwen-embed.txt"
fi
stop "$srv"

$PY -u /scratch/learn/work/labs/L5/pooling_roofline.py --out "$run_root/data" > "$run_root/roofline.log" 2>&1
printf "%s\n" "$?" > "$run_root/exit-roofline.txt"
$PY -u /scratch/learn/work/labs/L5/pooling_batch_limit.py --out "$run_root/data" --budget-gb 30 \
    > "$run_root/batch-limit.log" 2>&1
printf "%s\n" "$?" > "$run_root/exit-batch-limit.txt"

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
echo SERVING_ROOFLINE_DONE
