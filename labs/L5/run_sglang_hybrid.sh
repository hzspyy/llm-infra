#!/usr/bin/env bash
# L5.13 SGLang 混合架构支持：静态核查 + 一次真实启动尝试（记录失败原因）。
set -uo pipefail
source /scratch/learn/env.sh
MODEL=${RG_MODEL:-/scratch/learn/models/hf/models--google--recurrentgemma-2b/snapshots/3620f4ca9c5d16ee56c00180474a3201ec7f734a}
PORT=8149
PY=/scratch/learn/envs/sgl/bin/python
run_root="/scratch/learn/work/out/sglang-hybrid-${1:?pass unique run-id}"
mkdir -p "$run_root"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-before.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$run_root/processes-before.txt"
if [ -s "$run_root/processes-before.txt" ]; then
    echo 'GPU occupied: no experiment started' > "$run_root/blocked.txt"; exit 1
fi

$PY -u /scratch/learn/work/labs/L5/sglang_hybrid_probe.py --out "$run_root" \
    > "$run_root/survey.log" 2>&1
echo "survey exit=$?"

# 真实启动尝试：SGLang 载入 RecurrentGemma，记录它到底在哪一步拒绝
HF_HUB_OFFLINE=1 timeout 420 $PY -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
    --mem-fraction-static 0.45 > "$run_root/launch.log" 2>&1
printf '%s\n' "$?" > "$run_root/launch-exit.txt"

grep -nE "Error|error|not support|Unsupported|KeyError|ValueError|assert" \
    "$run_root/launch.log" | head -12 > "$run_root/launch-errors.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$run_root/gpu-after.txt"

cat "$run_root/survey.log"
echo "--- 启动尝试（退出码 $(cat "$run_root/launch-exit.txt")）---"
head -12 "$run_root/launch-errors.txt"
echo "--- 日志尾部 ---"
tail -6 "$run_root/launch.log"
