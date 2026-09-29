#!/usr/bin/env bash
# 5.10 补测：SGLang 的 unload 语义（crater2，sgl 环境）。
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_OFFLINE=1
PY=/scratch/learn/envs/sgl/bin/python
PORT=8190
OUT=${1:-/scratch/learn/work/out/5.10/sglang-unload-20260921}
MODEL=/scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots/$(ls /scratch/learn/models/hf/hub/models--Qwen--Qwen3-4B/snapshots | head -1)
ADAPTER=/scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots/$(ls /scratch/learn/models/hf/hub/models--trl-lib--Qwen3-4B-LoRA/snapshots | head -1)
mkdir -p "$OUT"
pkill -9 -f "[p]ort $PORT" 2>/dev/null
sleep 2
nohup "$PY" -m sglang.launch_server --model-path "$MODEL" --port $PORT \
  --host 127.0.0.1 --context-length 4096 --mem-fraction-static 0.45 \
  --page-size 16 --chunked-prefill-size 4096 --disable-radix-cache \
  --enable-lora --max-lora-rank 8 --max-loras-per-batch 4 \
  --lora-target-modules q_proj v_proj qkv_proj o_proj gate_proj up_proj down_proj \
  > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 300); do
  C=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d '{"text":"ready","sampling_params":{"max_new_tokens":1,"temperature":0}}' 2>/dev/null || echo 000)
  [ "$C" = "200" ] && { echo "ready ${i}s"; break; }
  sleep 1
done
cd /scratch/learn/work/labs/L5
"$PY" -u sglang_lora_unload_semantics.py --base "http://127.0.0.1:$PORT" \
  --lora-path "$ADAPTER" --out "$OUT" 2>&1 | tail -12
grep -a "Reloading evicted\|unloading\|loaded" "$OUT/server.log" | tail -8 > "$OUT/server_lora_lines.txt"
cat "$OUT/server_lora_lines.txt"
kill -9 "$PID" 2>/dev/null; sleep 5; pkill -9 -f "[p]ort $PORT" 2>/dev/null
echo SGLANG_UNLOAD_DONE
