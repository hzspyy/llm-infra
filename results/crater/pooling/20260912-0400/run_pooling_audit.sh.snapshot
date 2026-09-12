#!/usr/bin/env bash
# 5.12 非生成模型的服务。产出：三点 roofline、embedding 的 batch 扫描与变长对照、
# 以及 pooling 请求不被 chunk 带来的 token 预算边界。
set -euo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
# 官方端点会间歇性 ConnectError 104（见 ENVIRONMENTS.md），权重已在本地缓存，
# 强制离线可避免启动时去列远端文件。缺文件时改用 HF_ENDPOINT=https://hf-mirror.com 补。
export HF_HUB_OFFLINE=1
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export ENC_MODEL="${ENC_MODEL:-BAAI/bge-small-en-v1.5}"

run_root="/scratch/learn/work/out/pooling-${1:?pass unique run id}"
mkdir "$run_root"
mkdir "$run_root/data"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
  printf '%s\n' 'GPU occupied: no experiment started' > "$run_root/blocked.txt"
  exit 1
fi
for f in pooling_roofline.py embedding_serving_audit.py; do
  cp "/scratch/learn/work/labs/L5/$f" "$run_root/$f.snapshot"
done
cp "$0" "$run_root/run_pooling_audit.sh.snapshot"
PY=/scratch/learn/envs/serve/bin/python
code=0

start_server() {   # $1=port $2=model $3=served name $4=extra args $5=logfile
  # shellcheck disable=SC2086
  $PY -m vllm.entrypoints.openai.api_server --model "$2" --served-model-name "$3" \
      --port "$1" --dtype bfloat16 --gpu-memory-utilization 0.35 \
      --no-enable-log-requests $4 > "$5" 2>&1 &
  echo $!
}

wait_health() {
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

stop() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null || true; }

# ---- 阶段 1：原生 encoder-only（BERT）--------------------------------------
srv=$(start_server 8100 "$ENC_MODEL" bge-small "--runner pooling --max-model-len 512" "$run_root/server-bge.log")
set +e
if wait_health 8100; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8100 \
      --model bge-small --out "$run_root/data-bge" --mode fixed --length 128 \
      --concurrency 1 8 32 128 256 > "$run_root/audit-bge-fixed.log" 2>&1
  printf '%s\n' "$?" > "$run_root/exit-bge-fixed.txt"
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8100 \
      --model bge-small --out "$run_root/data-bge" --mode variable \
      --concurrency 1 8 32 128 256 > "$run_root/audit-bge-var.log" 2>&1
  printf '%s\n' "$?" > "$run_root/exit-bge-var.txt"
else
  printf '%s\n' 'server did not become healthy' > "$run_root/server-bge-timeout.txt"
  printf '%s\n' 2 > "$run_root/exit-bge-fixed.txt"
fi
stop "$srv"
set -e

# ---- 阶段 2：同一个因果 backbone 当 embedding 用 ---------------------------
srv=$(start_server 8101 "$LLM_MODEL" qwen-embed "--runner pooling --convert embed --max-model-len 2048" "$run_root/server-qwen-embed.log")
set +e
if wait_health 8101; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8101 \
      --model qwen-embed --out "$run_root/data-qwen" --mode fixed --length 256 \
      --concurrency 1 8 32 128 > "$run_root/audit-qwen-embed.log" 2>&1
  printf '%s\n' "$?" > "$run_root/exit-qwen-embed.txt"
else
  printf '%s\n' 'server did not become healthy' > "$run_root/server-qwen-embed-timeout.txt"
  printf '%s\n' 2 > "$run_root/exit-qwen-embed.txt"
fi
stop "$srv"
set -e

# ---- 阶段 3：token 预算边界（pooling 默认不 chunk）-------------------------
srv=$(start_server 8102 "$ENC_MODEL" bge-small \
      "--runner pooling --max-model-len 512 --max-num-batched-tokens 256" "$run_root/server-budget.log")
set +e
if wait_health 8102; then
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8102 \
      --model bge-small --out "$run_root/data-boundary-fit" --mode fixed --length 64 \
      --concurrency 4 --repeat 1 --timeout 60 > "$run_root/boundary-fit.log" 2>&1
  printf '%s\n' "$?" > "$run_root/exit-boundary-fit.txt"
  $PY -u /scratch/learn/work/labs/L5/embedding_serving_audit.py --base-url http://127.0.0.1:8102 \
      --model bge-small --out "$run_root/data-boundary-over" --mode fixed --length 512 \
      --concurrency 1 --repeat 1 --timeout 60 > "$run_root/boundary-over.log" 2>&1
  printf '%s\n' "$?" > "$run_root/exit-boundary-over.txt"
else
  printf '%s\n' 'server did not become healthy' > "$run_root/server-budget-timeout.txt"
  printf '%s\n' 2 > "$run_root/exit-boundary-fit.txt"
fi
stop "$srv"
set -e

# ---- 阶段 4：三点 roofline（torch，直接量形状）-----------------------------
set +e
$PY -u /scratch/learn/work/labs/L5/pooling_roofline.py --out "$run_root/data" \
    > "$run_root/roofline.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_root/exit-roofline.txt"
set -e

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
exit "$code"
