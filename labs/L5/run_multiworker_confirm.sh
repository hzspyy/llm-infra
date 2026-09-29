#!/usr/bin/env bash
# 5.11 补测：vLLM --api-server-count 的进程数确认 + 前端受限负载对照。
#
# 上一轮"多前端无效"的结论不可用，因为没能确认 4 个前端进程真的起来了
# （`python -m ...api_server` 日志只打一个 APIServer pid，pgrep 在 count=1 时也数到 4）。
# 这一轮在服务运行**期间**抓三类证据：
#   1) 进程表里 VLLM::APIServer 的完整 PID 列表（vLLM 用 setproctitle 改名）；
#   2) 监听 8180 的进程列表（count=4 时靠 SO_REUSEPORT 共享同一端口）；
#   3) 两个负载形状：513-token prompt（前端不重）与 8-token prompt / 高并发（前端受限）。
set -uo pipefail
source /scratch/learn/env.sh
PYV=/scratch/learn/envs/serve/bin/python
MODEL=${MW_MODEL:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
RUN=${1:?pass run dir}
PORT=8180
mkdir -p "$RUN"

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" | tee "$RUN/gpu-free.txt"
[ "$free" -ge 12000 ] || { echo "less than 12 GiB free" > "$RUN/blocked.txt"; exit 1; }

probe_processes() {   # $1 标签
    local tag="$1"
    {
        echo "=== ps 里 VLLM::APIServer / EngineCore（$tag）"
        ps -eo pid,ppid,etime,args | grep -E "VLLM::(APIServer|EngineCore)" | grep -v grep
        echo "=== APIServer 计数"
        ps -eo args | grep -c "VLLM::APIServer" || true
        echo "=== 监听 $PORT 的进程"
        (ss -ltnp 2>/dev/null | grep ":$PORT " ) || (lsof -iTCP:$PORT -sTCP:LISTEN 2>/dev/null)
    } > "$RUN/processes-$tag.txt" 2>&1
}

run_case() {   # $1 标签 $2 api-server-count
    local tag="$1" n="$2"
    echo "--- $tag (api-server-count=$n)"
    HF_HUB_OFFLINE=1 setsid $PYV -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name m --port $PORT --dtype bfloat16 \
        --max-model-len 4096 --gpu-memory-utilization 0.34 \
        --no-enable-prefix-caching --no-enable-log-requests \
        --api-server-count "$n" > "$RUN/vllm-$tag.log" 2>&1 &
    local pgid=$!
    ok=0
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        sleep 1
    done
    if [ "$ok" != 1 ]; then echo "not healthy" > "$RUN/timeout-$tag.txt"; kill -9 -"$pgid" 2>/dev/null; return 1; fi
    sleep 5                     # 让所有前端进程都进入监听状态
    probe_processes "$tag"
    # 形状 A：513 token prompt（前端分词不重）
    $PYV -u /scratch/learn/work/labs/L5/multiworker_audit.py --api openai \
        --base "http://127.0.0.1:$PORT" --label "${tag}_long" --out "$RUN" \
        --prompt-tokens 512 --batch 32 --repeats 3 > "$RUN/audit-$tag-long.log" 2>&1
    # 形状 B：8 token prompt、并发 128（前端受限）
    $PYV -u /scratch/learn/work/labs/L5/multiworker_audit.py --api openai \
        --base "http://127.0.0.1:$PORT" --label "${tag}_short" --out "$RUN" \
        --prompt-tokens 8 --batch 128 --repeats 3 > "$RUN/audit-$tag-short.log" 2>&1
    kill -9 -"$pgid" 2>/dev/null
    pkill -9 -f "[p]ort $PORT" 2>/dev/null
    sleep 5
}

run_case vllm_api1 1
run_case vllm_api4 4

echo "=== 进程证据摘要"
for f in "$RUN"/processes-*.txt; do echo "--- $f"; grep -E "APIServer|计数|LISTEN|:8180" "$f" | head -12; done
echo MULTIWORKER_CONFIRM_DONE
