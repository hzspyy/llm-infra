#!/usr/bin/env bash
# 5.11：HTTP/1.1 基线与 h2c 对照。SGLang 开 --enable-http2 后两种协议都试。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/http2-${1:?pass run id}
PORT=8166
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi
HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.30 --enable-http2 \
    > "$RUN/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo 'server not healthy' > "$RUN/timeout.txt"; tail -8 "$RUN/server.log"; kill $SRV 2>/dev/null; exit 1
fi
$PY -u /scratch/learn/work/labs/L5/http2_audit.py \
    --base "http://127.0.0.1:$PORT" --out "$RUN" --n 32 > "$RUN/audit.log" 2>&1
echo "audit exit=$?"
grep -hiE "h2|http2|hypercorn" "$RUN/server.log" | head -5 > "$RUN/server-http2-lines.txt"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
cat "$RUN/audit.log" | tail -8
