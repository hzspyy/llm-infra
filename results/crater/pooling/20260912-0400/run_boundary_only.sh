#!/usr/bin/env bash
# 5.12 边界：pooling 请求默认不被 chunk，所以 token 预算成了序列长度的硬上限。
# 同一组参数分别用在生成模型与 pooling 模型上，看 vLLM 接不接受。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1
run_root="/scratch/learn/work/out/pooling-20260912-0400"
PY=/scratch/learn/envs/serve/bin/python
L511_BUDGET=256

try_start() {   # $1=name $2=port $3=model $4=extra
  local name=$1 port=$2 model=$3 extra=$4
  $PY -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$name" \
      --port "$port" --dtype bfloat16 --gpu-memory-utilization 0.35 --no-enable-log-requests \
      $extra > "$run_root/server-$name.log" 2>&1 &
  local pid=$!
  local ok=2
  for _ in $(seq 1 150); do
    if ! kill -0 "$pid" 2>/dev/null; then ok=1; break; fi
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=0; break; }
    sleep 1
  done
  printf "%s\n" "$ok" > "$run_root/startup-$name.txt"
  echo "$pid"
}

# ① 生成模型 + 小预算：chunked prefill 默认开，应该能起来
pid=$(try_start gen-small-budget 8110 Qwen/Qwen2.5-1.5B-Instruct "--max-model-len 2048 --max-num-batched-tokens $L511_BUDGET")
if [ "$(cat $run_root/startup-gen-small-budget.txt)" = 0 ]; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8110 \
      --model gen-small-budget --out "$run_root/data-boundary-gen" --mode fixed --length 1024 \
      --concurrency 1 --repeat 1 --timeout 60 > "$run_root/boundary-gen.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-boundary-gen.txt"
fi
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

# ② pooling 模型 + 小预算：应该被拒
pid=$(try_start pool-small-budget 8111 BAAI/bge-small-en-v1.5 "--runner pooling --max-model-len 512 --max-num-batched-tokens $L511_BUDGET")
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

# ③ pooling 模型 + 预算等于 max_model_len：可以起，且 512 token 的请求能过
pid=$(try_start pool-fit-budget 8112 BAAI/bge-small-en-v1.5 "--runner pooling --max-model-len 512 --max-num-batched-tokens 512")
if [ "$(cat $run_root/startup-pool-fit-budget.txt)" = 0 ]; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8112 \
      --model pool-fit-budget --out "$run_root/data-boundary-fit" --mode fixed --length 512 \
      --concurrency 1 --repeat 1 --timeout 60 > "$run_root/boundary-fit.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-boundary-fit.txt"
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8112 \
      --model pool-fit-budget --out "$run_root/data-boundary-fit" --mode fixed --length 128 \
      --concurrency 4 --repeat 1 --timeout 60 > "$run_root/boundary-fit-multi.log" 2>&1
  printf "%s\n" "$?" > "$run_root/exit-boundary-fit-multi.txt"
fi
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
echo BOUNDARY_DONE
