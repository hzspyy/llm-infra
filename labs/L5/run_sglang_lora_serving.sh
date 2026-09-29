#!/usr/bin/env bash
# 5.10：SGLang 侧挂真实 LoRA（base vs adapter，三轮交错）。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/sglang-lora-serving-${1:?pass run id}
PORT=8164
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi
ADAPTER=$RUN/adapter
$PY -c "
import sys; sys.path.insert(0, '/scratch/learn/work/labs/L5')
from sglang_lora_serving_audit import make_adapter
import pathlib
make_adapter(pathlib.Path('$ADAPTER'), '$MODEL')
print('adapter written to $ADAPTER')
"
HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.30 \
    --enable-lora --max-lora-rank 8 --lora-paths pub=$ADAPTER \
    > "$RUN/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo 'server not healthy' > "$RUN/timeout.txt"; tail -12 "$RUN/server.log"; kill $SRV 2>/dev/null; exit 1
fi
$PY -u /scratch/learn/work/labs/L5/sglang_lora_serving_audit.py \
    --base "http://127.0.0.1:$PORT" --label triton --out "$RUN" \
    --adapter-root "$ADAPTER" > "$RUN/audit.log" 2>&1
echo "audit exit=$?"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
grep -hE "lora|LoRA" "$RUN/server.log" | head -5 > "$RUN/lora-lines.txt"
tail -12 "$RUN/audit.log"
