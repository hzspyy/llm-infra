#!/usr/bin/env bash
# L5.11 补测 · SGLang 的 tokenizer/detokenizer worker 数：进程结构 + 前端负载。
#
# 上一轮只比较了客户端数字，没能独立确认 worker 数真的变了。本脚本在每组服务
# 运行期间抓进程与线程结构（ps + /proc/<pid>/cmdline），再跑同一份负载。
#
# 用法：RUN_ID=20260922 bash labs/L5/run_sglang_worker_probe.sh
set -uo pipefail

ROOT=${LEARN_ROOT:-/scratch/learn}
PYS=${PYS:-$ROOT/envs/sgl/bin/python}
MODEL=${MW_MODEL:-$ROOT/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
PORT=${PORT:-8155}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)}
RUN=${RUN:-$ROOT/work/out/sglang-worker-$RUN_ID}
mkdir -p "$RUN"

source "$ROOT/env.sh"
export HF_HUB_OFFLINE=1

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU free: ${free} MiB" > "$RUN/gpu-free.txt"
if [ "$free" -lt 8000 ]; then echo "less than 8 GiB free" > "$RUN/blocked.txt"; exit 1; fi

wait_health() {
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

snapshot_processes() {  # $1 = 标签
    local tag="$1"
    {
        echo "== ps（pid/ppid/线程数/命令）=="
        ps -eo pid,ppid,nlwp,etime,cmd | grep -E "sglang|Tokenizer|Detokenizer|Scheduler" \
            | grep -v grep
        echo
        echo "== /proc/<pid>/cmdline =="
        for p in $(pgrep -f "sglang.launch_server" 2>/dev/null); do
            echo "--- pid $p"
            tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null
            echo
            echo "    线程数 $(ls /proc/$p/task 2>/dev/null | wc -l)"
        done
        echo
        echo "== 线程名分布（前 20）=="
        ps -eLo pid,tid,comm | grep -E "sglang|tok|detok|sched|Tokenizer|Detokenizer" \
            | awk '{print $3}' | sort | uniq -c | sort -rn | head -20
    } > "$RUN/processes-$tag.txt" 2>&1
}

run_one() {  # $1 标签，其余为 SGLang 参数
    local tag="$1"; shift
    pkill -f "port $PORT" 2>/dev/null
    sleep 2
    HF_HUB_OFFLINE=1 "$PYS" -m sglang.launch_server --model-path "$MODEL" \
        --port "$PORT" --host 127.0.0.1 --dtype bfloat16 --disable-radix-cache \
        --attention-backend triton --mem-fraction-static 0.30 "$@" \
        > "$RUN/sglang-$tag.log" 2>&1 &
    local srv=$!
    if wait_health; then
        sleep 5
        snapshot_processes "$tag"
        "$PYS" -u "$ROOT/work/labs/L5/multiworker_audit.py" \
            --base "http://127.0.0.1:$PORT" --label "$tag" --out "$RUN" \
            > "$RUN/audit-$tag.log" 2>&1
    else
        echo "sglang($tag) not healthy" > "$RUN/timeout-$tag.txt"
    fi
    kill "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null
    sleep 3
}

run_one tok1 --tokenizer-worker-num 1 --detokenizer-worker-num 1
run_one tok4 --tokenizer-worker-num 4 --detokenizer-worker-num 4

echo "== 进程结构摘要 =="
for tag in tok1 tok4; do
    echo "--- $tag"
    grep -c "sglang" "$RUN/processes-$tag.txt" 2>/dev/null || true
    grep -E "tokenizer-worker-num" "$RUN/processes-$tag.txt" | head -3
done
echo "工件目录：$RUN"
