#!/usr/bin/env bash
# 5.11 API / 协议层实验。产出：原始 SSE 字节流、层间耗时分解、tokenization 曲线、
# 并发时延、以及「tokenizer 在事件循环里」的对照。
set -euo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export NANOSERVE_MODEL="${NANOSERVE_MODEL:-Qwen/Qwen3-1.7B}"
export L511_GPU_MEMORY_UTILIZATION=0.35

run_root="/scratch/learn/work/out/api-layer-${1:?pass unique run id}"
mkdir "$run_root"
mkdir "$run_root/data"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
  printf '%s\n' 'GPU occupied: no experiment started' > "$run_root/blocked.txt"
  exit 1
fi
for f in api_layer_audit.py sse_head_of_line.py mini_sse.py; do
  cp "/scratch/learn/work/labs/L5/$f" "$run_root/$f.snapshot"
done
cp "$0" "$run_root/run_api_layer_audit.sh.snapshot"
PY=/scratch/learn/envs/serve/bin/python
code=0

start_server() {   # $1 = port, $2 = extra args, $3 = logfile
  # shellcheck disable=SC2086
  $PY -m vllm.entrypoints.openai.api_server \
      --model "$NANOSERVE_MODEL" --served-model-name qwen3-1.7b \
      --port "$1" --max-model-len 4096 \
      --gpu-memory-utilization "$L511_GPU_MEMORY_UTILIZATION" \
      --no-enable-prefix-caching --no-enable-log-requests $2 \
      > "$3" 2>&1 &
  echo $!
}

wait_health() {    # $1 = port, $2 = timeout seconds
  for _ in $(seq 1 "$2"); do
    if curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

# ---- 阶段 1：主审计（keep-alive 关闭，字节数不被注释污染）------------------
srv=$(start_server 8100 "" "$run_root/server-main.log")
set +e
if wait_health 8100 180; then
  $PY -u /scratch/learn/work/labs/L5/api_layer_audit.py \
      --base-url http://127.0.0.1:8100 --model qwen3-1.7b \
      --out "$run_root/data" > "$run_root/audit.log" 2>&1
  code=$?
else
  printf '%s\n' 'server did not become healthy' > "$run_root/server-main-timeout.txt"
  code=2
fi
kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
printf '%s\n' "$code" > "$run_root/exit-audit.txt"
set -e

# ---- 阶段 2：keep-alive 注释 -------------------------------------------------
# 间隔参数是**整秒**（vllm/entrypoints/launchers/cli_args.py:140），所以要让注释真的出现，
# 得造出「排队超过 1 秒」的流：max-num-seqs 压到 1，再并发发 6 条长输出请求。
if [ "$code" -eq 0 ]; then
  srv=$(start_server 8101 "--sse-keep-alive-interval 1 --max-num-seqs 1" "$run_root/server-keepalive.log")
  set +e
  if wait_health 8101 180; then
    $PY -u /scratch/learn/work/labs/L5/api_layer_audit.py \
        --base-url http://127.0.0.1:8101 --model qwen3-1.7b \
        --out "$run_root/data-keepalive" --requests 1 --concurrency 6 \
        > "$run_root/keepalive.log" 2>&1
    printf '%s\n' "$?" > "$run_root/exit-keepalive.txt"
  else
    printf '%s\n' 'server did not become healthy' > "$run_root/server-keepalive-timeout.txt"
    printf '%s\n' '2' > "$run_root/exit-keepalive.txt"
  fi
  kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
  set -e
fi

# ---- 阶段 3：tokenizer 在事件循环里（nanoserve）--------------------------------
# probe=tokenize 用只做编码的端点，把「事件循环被占住」和「引擎多一条请求」分开；
# probe=generate 是端到端版本，两者都留，便于对照混淆项有多大。
for cfg in "tokenize 1000" "tokenize 4" "generate 1000"; do
  probe=${cfg% *}
  reps=${cfg#* }
  for mode in inloop thread; do
    set +e
    $PY -u /scratch/learn/work/labs/L5/sse_head_of_line.py \
        --mode "$mode" --probe "$probe" --b-reps "$reps" \
        --out "$run_root/data" > "$run_root/hol-$probe-$mode-b$reps.log" 2>&1
    printf '%s\n' "$?" > "$run_root/exit-hol-$probe-$mode-b$reps.txt"
    set -e
  done
done

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$run_root/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-after.txt"
exit "$code"
