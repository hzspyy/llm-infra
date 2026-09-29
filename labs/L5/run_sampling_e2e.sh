#!/usr/bin/env bash
# L5.9 采样契约端到端：vLLM 与 SGLang 各起一次服务，喂同一批 input_ids。
set -euo pipefail
source /scratch/learn/env.sh
MODEL=${E2E_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
PY_V=/scratch/learn/envs/serve/bin/python
PY_S=/scratch/learn/envs/sgl/bin/python
run_root="/scratch/learn/work/out/sampling-e2e-${1:?pass unique run-id}"
mkdir -p "$run_root"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi

wait_health() { for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

# ---- vLLM ----
VPORT=8146
HF_HUB_OFFLINE=1 $PY_V -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name m --port $VPORT --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.35 --no-enable-log-requests \
    > "$run_root/vllm-server.log" 2>&1 &
VSRV=$!
if wait_health $VPORT; then
    $PY_V -u /scratch/learn/work/labs/L5/sampling_engine_e2e.py --engine vllm \
        --base "http://127.0.0.1:${VPORT}" --out "$run_root/vllm" \
        > "$run_root/vllm.log" 2>&1 || true
else
    echo 'vllm not healthy' > "$run_root/vllm-timeout.txt"
fi
kill $VSRV 2>/dev/null || true; wait $VSRV 2>/dev/null || true
sleep 3

# ---- SGLang ----
SPORT=8147
HF_HUB_OFFLINE=1 $PY_S -m sglang.launch_server --model-path "$MODEL" \
    --port $SPORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --attention-backend triton --mem-fraction-static 0.45 \
    > "$run_root/sglang-server.log" 2>&1 &
SSRV=$!
if wait_health $SPORT; then
    $PY_S -u /scratch/learn/work/labs/L5/sampling_engine_e2e.py --engine sglang \
        --base "http://127.0.0.1:${SPORT}" --out "$run_root/sglang" \
        > "$run_root/sglang.log" 2>&1 || true
else
    echo 'sglang not healthy' > "$run_root/sglang-timeout.txt"
fi
kill $SSRV 2>/dev/null || true; wait $SSRV 2>/dev/null || true

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-after.txt"
echo "--- vllm ---";   tail -6 "$run_root/vllm.log" 2>/dev/null
echo "--- sglang ---"; tail -8 "$run_root/sglang.log" 2>/dev/null
echo "--- compare ---"
[ -f "$run_root/vllm/e2e.json" ] && [ -f "$run_root/sglang/e2e.json" ] && \
  $PY_V -u /scratch/learn/work/labs/L5/sampling_engine_e2e.py \
      --compare "$run_root/vllm" "$run_root/sglang" || echo '（一侧缺失，跳过对照）'
