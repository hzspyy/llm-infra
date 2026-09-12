#!/usr/bin/env bash
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"; export HF_DATASETS_CACHE="$HF_HOME/datasets"; export HF_XET_CACHE="$HF_HOME/xet"
run_root="/scratch/learn/work/out/api-layer-20260912-0325"
PY=/scratch/learn/envs/serve/bin/python
$PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --served-model-name qwen3-1.7b \
    --port 8102 --max-model-len 4096 --gpu-memory-utilization 0.35 \
    --no-enable-prefix-caching --no-enable-log-requests \
    --sse-keep-alive-interval 1 --max-num-seqs 1 > "$run_root/server-keepalive2.log" 2>&1 &
srv=$!
ok=2
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8102/health >/dev/null 2>&1 && { ok=0; break; }; sleep 1; done
if [ "$ok" = 0 ]; then
  $PY -u /scratch/learn/work/labs/L5/api_layer_audit.py --base-url http://127.0.0.1:8102 \
      --model qwen3-1.7b --out "$run_root/data-keepalive2" --requests 1 --concurrency 20 \
      > "$run_root/keepalive2.log" 2>&1
  ok=$?
fi
printf "%s\n" "$ok" > "$run_root/exit-keepalive2.txt"
kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
echo KA2_DONE
