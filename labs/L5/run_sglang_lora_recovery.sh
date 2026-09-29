#!/usr/bin/env bash
# 5.10：SGLang 上的 LoRA 异常路径与同进程恢复（外加一次独立重启参照）。
set -uo pipefail
source /scratch/learn/env.sh
PY=/scratch/learn/envs/sgl/bin/python
MODEL=${SGL_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=/scratch/learn/work/out/sglang-lora-recovery-${1:?pass run id}
PORT=8168
rm -rf "$RUN"; mkdir -p "$RUN"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

# 三个 adapter：合法 rank8、超限 rank16、目录不完整
$PY -c "
import sys, pathlib; sys.path.insert(0, '/scratch/learn/work/labs/L5')
from sglang_lora_serving_audit import make_adapter
make_adapter(pathlib.Path('$RUN/valid'), '$MODEL', rank=8, seed=99)
make_adapter(pathlib.Path('$RUN/rank16'), '$MODEL', rank=16, seed=98)
d = pathlib.Path('$RUN/broken'); d.mkdir(parents=True, exist_ok=True)
(d/'adapter_model.safetensors').write_bytes(b'not-a-real-safetensors')
print('adapters ready')
"

start_server() {
    HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
        --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
        --attention-backend triton --mem-fraction-static 0.30 \
        --enable-lora --max-lora-rank 8 --lora-paths pub=$RUN/valid \
        > "$1" 2>&1 &
    echo $!
}
wait_health() {
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; sleep 1
    done; return 1
}

SRV=$(start_server "$RUN/server-first.log")
if ! wait_health; then echo 'not healthy' > "$RUN/timeout.txt"; kill $SRV 2>/dev/null; exit 1; fi
$PY -u /scratch/learn/work/labs/L5/sglang_lora_recovery_test.py --base "http://127.0.0.1:$PORT" \
    --out "$RUN" --valid-adapter "$RUN/valid" --rank16-dir "$RUN/rank16" --broken-dir "$RUN/broken" \
    > "$RUN/test-first.log" 2>&1
echo "same-process exit=$?" >> "$RUN/exit.txt"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3

# 独立重启参照
SRV=$(start_server "$RUN/server-second.log")
if wait_health; then
    $PY -u /scratch/learn/work/labs/L5/sglang_lora_recovery_test.py --base "http://127.0.0.1:$PORT" \
        --out "$RUN/restart" --valid-adapter "$RUN/valid" --rank16-dir "$RUN/rank16" --broken-dir "$RUN/broken" \
        > "$RUN/test-restart.log" 2>&1
    echo "restart exit=$?" >> "$RUN/exit.txt"
fi
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
# 第三配置：启动时就把 rank=16 的 adapter 注册进去，看 rank 校验发生在哪一步
HF_HUB_OFFLINE=1 $PY -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.30 \
    --enable-lora --max-lora-rank 8 --lora-paths pub=$RUN/valid big=$RUN/rank16 \
    > "$RUN/server-rank16.log" 2>&1 &
SRV=$!
if wait_health; then
    echo 'rank16 注册成功（启动未拒绝）' > "$RUN/rank16-verdict.txt"
    curl -sS -o /dev/null -w '%{http_code}\n' -X POST "http://127.0.0.1:$PORT/generate" \
        -H 'Content-Type: application/json' \
        --data-binary '{"text":"hi","lora_path":"big","sampling_params":{"max_new_tokens":4}}' \
        >> "$RUN/rank16-verdict.txt"
else
    echo 'rank16 注册被拒绝（启动即失败）' > "$RUN/rank16-verdict.txt"
fi
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
grep -iE "rank|lora" "$RUN/server-rank16.log" | tail -6 > "$RUN/rank16-lines.txt"
grep -iE "lora|rank" "$RUN/server-first.log" | tail -4 > "$RUN/lora-lines.txt"
tail -12 "$RUN/test-first.log"
