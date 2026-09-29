#!/usr/bin/env bash
# L9.4 × 9.7：预算策略接到任务级调度上的对照实验。
#
#   bash labs/L9/run_budget_scheduling.sh <host> <out_root> [n] [rate]
#
# 起一个带 priority 调度器的 vLLM，然后在同一批长短混合任务上依次跑三个预算策略：
# uniform_long / budget_aware / budget_aware_priority。运行时按进程组回收 GPU 进程。
set -u
HOST=${1:?usage: run_budget_scheduling.sh <host> <out_root> [n] [rate]}
ROOT=${2:?missing out_root}
N=${3:-64}
RATE=${4:-3.0}
PORT=${L94_PORT:-8061}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9
MODEL=Qwen/Qwen3-4B

ssh -o BatchMode=yes "$HOST" "rm -rf $ROOT; mkdir -p $ROOT; \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-before.txt; \
  $PY -c 'import importlib.metadata as m; print(\"vllm\", m.version(\"vllm\"))' > $ROOT/versions.txt 2>&1; \
  source /scratch/learn/env.sh; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
  setsid --fork bash -c 'echo \$\$ > $ROOT/vllm.pgid; exec $PY -u -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.45 --max-num-seqs 32 \
    --scheduling-policy priority --enable-prefix-caching --no-enable-log-requests' \
    </dev/null > $ROOT/vllm-server.log 2>&1; sleep 1; cat $ROOT/vllm.pgid"

for i in $(seq 1 240); do
  ssh -o BatchMode=yes "$HOST" "curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null" \
    && { echo "[engine] ready after ${i}s"; break; }
  sleep 1
done

ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PY $LABS/budget_task_scheduling.py run \
  --base-url http://127.0.0.1:$PORT/v1 --model $MODEL --out $ROOT/report.json \
  --n $N --rate $RATE > $ROOT/run.log 2>&1; echo \$? > $ROOT/run.exit"

ssh -o BatchMode=yes "$HOST" "pgid=\$(cat $ROOT/vllm.pgid 2>/dev/null); \
  if [ -n \"\$pgid\" ]; then kill -TERM -- -\$pgid 2>/dev/null; sleep 6; kill -KILL -- -\$pgid 2>/dev/null; fi; sleep 2; \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-after.txt"
echo "[done] $HOST:$ROOT"
