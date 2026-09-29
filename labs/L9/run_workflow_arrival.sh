#!/usr/bin/env bash
# L9.7 任务 D：ARRIVAL 协议下的任务级 SLO 与成本。
#
#   bash labs/L9/run_workflow_arrival.sh <host> <out_root> [baseline_rate]
#
# 步骤：在 host 上起一个「priority 调度 + 工具调用」的 vLLM → 标定任务基线到达率 →
# 按 0.3/0.6/0.9/1.1 倍基线、每档 3 个 ≥120 秒窗口、策略交错跑完整任务。
#
# 两个必须记住的点：
#   1) 引擎必须带 --enable-auto-tool-choice 与 --tool-call-parser：缺了它们，
#      tool_choice=auto 的请求会被直接 400，任务全部以"模型报错"进入分母（表现为 success=0）；
#   2) 客户端要检查窗口结束后的引擎存活，否则引擎中途死掉时结果只是"全是失败"。
set -u
HOST=${1:?usage: run_workflow_arrival.sh <host> <out_root> [baseline_rate]}
ROOT=${2:?missing out_root}
BASE=${3:-}
PORT=${L97_PORT:-8061}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9
COMMON="--base-url http://127.0.0.1:$PORT/v1 --classes compute,retrieval --n-per-class 40 \
  --concurrency 16 --max-turns 4 --max-tokens 512 --context-char-budget 20000"

ssh -o BatchMode=yes "$HOST" "rm -rf $ROOT; mkdir -p $ROOT; \
  setsid --fork nohup /scratch/learn/start_workflow_worker.sh $PORT $ROOT 16 </dev/null > $ROOT/server.log 2>&1; sleep 1; echo launched"
for i in $(seq 1 120); do
  ssh -o BatchMode=yes "$HOST" "curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null" && { echo "[engine] ready after ${i}"; break; }
  sleep 2
done

if [ -z "$BASE" ]; then
  ssh -o BatchMode=yes "$HOST" "cd /scratch/learn && setsid --fork nohup $PY $LABS/workflow_loadgen.py calibrate \
    --out $ROOT/cal $COMMON --rates 1,2,4 --window-s 40 </dev/null > $ROOT/cal.log 2>&1; sleep 1; echo cal-launched"
  # 标定耗时约 2–3 分钟；此处按固定等待，随后由使用者确认 cal.json
  sleep 200
  BASE=$(ssh -o BatchMode=yes "$HOST" "$PY -c \"
import json;d=json.load(open('$ROOT/cal/calibrate.json'))
rows=[r for r in d['rows'] if r['tasks_in_flight']==0]
print(max([r['rate'] for r in rows], default=1.0))\"" 2>/dev/null | tail -1)
  echo "[calibrate] baseline_rate=$BASE"
fi

ssh -o BatchMode=yes "$HOST" "cd /scratch/learn && setsid --fork nohup $PY $LABS/workflow_loadgen.py arrival \
  --out $ROOT/run $COMMON --baseline-rate $BASE --scales 0.3,0.6,0.9,1.1 --windows 3 --window-s 120 \
  --policies engine_fcfs,plas --max-tasks-per-window 4000 --slo-ms-list 10000,20000,30000,60000 \
  </dev/null > $ROOT/run.log 2>&1; sleep 1; echo arrival-launched"
echo "日志：ssh $HOST tail -f $ROOT/run.log"
