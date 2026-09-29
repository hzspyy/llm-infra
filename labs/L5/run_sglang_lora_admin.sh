#!/usr/bin/env bash
# 5.10：SGLang 的同名权重替换、卸载与生成中途卸载。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/sglang-lora-admin-${1:?pass run id}
PORT=8170
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi
$PY -c "
import sys, pathlib; sys.path.insert(0, '/scratch/learn/work/labs/L5')
from sglang_lora_serving_audit import make_adapter
make_adapter(pathlib.Path('$RUN/c1'), '$MODEL', rank=8, seed=1111)
make_adapter(pathlib.Path('$RUN/c2'), '$MODEL', rank=8, seed=2222)
print('C1/C2 ready')
"
HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.30 \
    --enable-lora --max-lora-rank 8 --lora-paths pub=$RUN/c1 \
    > "$RUN/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo 'not healthy' > "$RUN/timeout.txt"; tail -10 "$RUN/server.log"; kill $SRV 2>/dev/null; exit 1
fi
$PY -u /scratch/learn/work/labs/L5/sglang_lora_admin_test.py --base "http://127.0.0.1:$PORT" \
    --out "$RUN" --adapter-c1 "$RUN/c1" --adapter-c2 "$RUN/c2" > "$RUN/test.log" 2>&1
echo "exit=$?" > "$RUN/exit.txt"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
tail -16 "$RUN/test.log"
