#!/usr/bin/env bash
# L5.3-B / L5.1-C · chunked prefill 预算扫描（每个 (预算, 重复) 一个新进程）。
#
# 为什么不是一个大进程：反复建/销毁 vLLM 引擎会在显存里留下残留，
# 逐档重算 gpu_memory_utilization 的话档与档就不可比了，而且中途会
# 因为「没有可用 KV 显存」失败。改成每个组合一个进程，并固定 util。
#
# 用法： run_chunked_budget.sh <out_dir> [budgets] [repeats]
set -euo pipefail

PY=/scratch/learn/envs/serve/bin/python
LEARN=/scratch/learn
OUT="${1:?用法: run_chunked_budget.sh <out_dir> [budgets] [repeats]}"
BUDGETS="${2:-128,256,512,2048,8192}"
REPS="${3:-3}"
UTIL="${UTIL:-0.45}"

source "$LEARN/env.sh"
mkdir -p "$OUT"
cd "$LEARN/work/labs/L5"

for b in ${BUDGETS//,/ }; do
  for r in $(seq 0 $((REPS - 1))); do
    echo "=== budget $b rep $r ==="
    # 上一进程退出与 CUDA 上下文回收之间有窗口：vLLM 的显存 profiling 会因为
    # 「初始空闲 22.3 GiB、之后空闲 26.7 GiB」直接断言失败。等一会儿再起，
    # 失败就重试一次（同一次重复的失败输出保留在日志里）。
    sleep 8
    for attempt in 1 2; do
      if "$PY" chunked_prefill_scan.py --out "$OUT" --budgets "$b" --repeats 1 \
          --util "$UTIL" --single "$b:$r" 2>&1 \
          | grep -v "Capturing CUDA\|^INFO \|WARNING\|autotuner\|flashinfer" | tail -3; then
        break
      fi
      echo "  retry budget $b rep $r (attempt $attempt)"
      sleep 15
    done
  done
done

"$PY" chunked_prefill_scan.py --out "$OUT" --budgets "$BUDGETS" --repeats "$REPS" \
    --merge 2>&1 | tail -45
