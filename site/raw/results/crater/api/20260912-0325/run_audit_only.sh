#!/usr/bin/env bash
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"; export HF_DATASETS_CACHE="$HF_HOME/datasets"; export HF_XET_CACHE="$HF_HOME/xet"
run_root="/scratch/learn/work/out/api-layer-20260912-0325"
PY=/scratch/learn/envs/serve/bin/python
start_server() {
  $PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --served-model-name qwen3-1.7b \
      --port "$1" --max-model-len 4096 --gpu-memory-utilization 0.35 \
      --no-enable-prefix-caching --no-enable-log-requests $2 > "$3" 2>&1 &
  echo $!
}
wait_health() { for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
srv=$(start_server 8100 "" "$run_root/server-main.log")
if wait_health 8100; then
  $PY -u /scratch/learn/work/labs/L5/api_layer_audit.py --base-url http://127.0.0.1:8100 \
      --model qwen3-1.7b --out "$run_root/data" > "$run_root/audit.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-audit.txt"
else
  printf "%s\n" 2 > "$run_root/exit-audit.txt"
fi
kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
srv=$(start_server 8101 "--sse-keep-alive-interval 1 --max-num-seqs 1" "$run_root/server-keepalive.log")
if wait_health 8101; then
  $PY -u /scratch/learn/work/labs/L5/api_layer_audit.py --base-url http://127.0.0.1:8101 \
      --model qwen3-1.7b --out "$run_root/data-keepalive" --requests 1 --concurrency 6 \
      > "$run_root/keepalive.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-keepalive.txt"
else
  printf "%s\n" 2 > "$run_root/exit-keepalive.txt"
fi
kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
echo AUDIT_ALL_DONE
