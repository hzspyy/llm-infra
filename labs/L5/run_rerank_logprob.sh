#!/usr/bin/env bash
# 起生成式 vLLM（Reranker checkpoint 的 config 是 Qwen3ForCausalLM），跑 logprob 链对照。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PY=/scratch/learn/envs/serve/bin/python
PORT=8152
OUT=/scratch/learn/work/out/5.12/rerank-logprob-20260921
SNAP=$(ls -d /scratch/learn/models/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/*/ | head -1)
mkdir -p "$OUT"
pkill -f "port $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --port "$PORT" --host 127.0.0.1 --max-model-len 4096 \
  --gpu-memory-utilization 0.35 --max-logprobs 20000 --dtype bfloat16 \
  > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  C=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [ "$C" = "200" ] && { echo "ready ${i}s"; break; }
  sleep 1
done
cd /scratch/learn/work/labs/L5
# 服务是用快照路径起的，请求里的 model 必须与之一致（否则 404 model does not exist）
"$PY" -u rerank_logprob_chain.py --base "http://127.0.0.1:$PORT" --out "$OUT" \
  --model "$SNAP" --hf-reference "$1" 2>&1 | tail -20
kill "$PID" 2>/dev/null; sleep 8; pkill -f "port $PORT" 2>/dev/null
echo RERANK_LOGPROB_DONE
