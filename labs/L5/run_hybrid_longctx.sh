#!/usr/bin/env bash
# L5.13 超长上下文（4096+）：RecurrentGemma-2B 在 vLLM 上的上下文扫描。
#
# 既有扫描只到 2048。这里把它推到 8192，并额外量一次 4096 上下文下的
# 逐 token decode 时间，用来分开"prefill 摊薄"与"decode 每步常数"两件事。
#
# 注意 RecurrentGemma 在本版本走 Transformers 回退后端
# （日志：TransformersForCausalLM has no vLLM implementation），
# 默认会连 torch.compile 与 51 个图捕获尺寸一起做，启动超过 600 s 而起不来；
# 这里显式 --enforce-eager 只保留权重加载与 compile 后端之外的路径。
# 预算也不能照小模型的习惯给：回退路径下 0.35（11 GiB）连缓存块都留不出来，
# 会以 "No available memory for the cache blocks" 失败；本机独占时用 0.6。
# 另外必须关掉前缀缓存：同一目标长度重复三次，第二次起会直接命中缓存，
# 量到的是查表而不是 prefill（首轮 ~224 ms vs 后两轮 ~15 ms 就是这个原因）。
#
# 用法：bash run_hybrid_longctx.sh <run-id>
set -euo pipefail
source /scratch/learn/env.sh
MODEL_DIR=${RG_MODEL:-/scratch/learn/models/hf/models--google--recurrentgemma-2b/snapshots/3620f4ca9c5d16ee56c00180474a3201ec7f734a}
PORT=8143
PY=/scratch/learn/envs/serve/bin/python
run_root="/scratch/learn/work/out/hybrid-longctx-${1:?pass unique run-id}"
mkdir -p "$run_root"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi
cp /scratch/learn/work/labs/L5/hybrid_longctx_scan.py "$run_root/lab.snapshot.py"

HF_HUB_OFFLINE=1 $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name rg \
    --port $PORT --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.6 --no-enable-log-requests \
    --enforce-eager --no-enable-prefix-caching \
    > "$run_root/server.log" 2>&1 &
SRV=$!
stop() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true; }
trap stop EXIT

for _ in $(seq 1 900); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    echo 'server not healthy' > "$run_root/server-timeout.txt"; exit 1; }

set +e
$PY -u /scratch/learn/work/labs/L5/hybrid_longctx_scan.py \
    --base "http://127.0.0.1:${PORT}" --out "$run_root" > "$run_root/scan.log" 2>&1
printf '%s\n' "$?" > "$run_root/exit.txt"
set -e

grep -hE "GPU KV cache size|Available KV cache memory|Resolved pooling|maximum sequence length" "$run_root/server.log" \
    > "$run_root/server-capacity.txt" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$run_root/gpu-after.txt"
tail -20 "$run_root/scan.log"
