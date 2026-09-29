#!/usr/bin/env bash
# L9.3 任务 D：两台真实 worker 上的会话迁移、重算与跨实例 KV 搬运。
#
#   bash labs/L9/run_session_migration.sh <host_a> <host_b> <out_root>
#
# 两台机器需在同一子网且能直连 TCP（本实验用 192.168.105.100 与 .101）。
# 步骤：在 A/B 各起一个 vLLM（prefix caching 开）→ 在 B 起接收端 → 在 A 跑 bench。
#
# 两条踩过的坑写在这里，避免重复：
#   1) 用 pidfile 而不是 `pkill -f vllm` 做互斥：`pkill -f <模式>` 会匹配到发起启动的
#      那条 ssh 命令自身，把 shell 一起杀掉（表现为"命令没有输出、目录没建"）；
#   2) bench 每次运行必须用不同的前缀标记（`--run-tag`）：worker 上的前缀缓存会跨实验保留，
#      复用标记会让"B 首次见到该前缀"这一前提不成立，得到的复用率会虚高。
set -u
HOST_A=${1:?usage: run_session_migration.sh <host_a> <host_b> <out_root> [prefix_tokens]}
HOST_B=${2:?missing host_b}
ROOT=${3:?missing out_root}
PREFIX=${4:-2048}
PORT_A=${L93A_PORT:-8041}
PORT_B=${L93B_PORT:-8042}
RECV_PORT=${L93RECV_PORT:-8970}
PY=/scratch/learn/envs/serve/bin/python
LABS=/scratch/learn/work/labs/L9

mkdir -p "$ROOT"

launch() {   # $1=host $2=port $3=outdir $4=log
  ssh -o BatchMode=yes "$1" "setsid --fork nohup /scratch/learn/start_worker.sh $2 $3 </dev/null > $4 2>&1; sleep 1; echo launched"
}

wait_ready() {  # $1=host $2=port
  for i in $(seq 1 120); do
    ssh -o BatchMode=yes "$1" "curl -sf http://127.0.0.1:$2/v1/models >/dev/null" && { echo "[$1] ready after ${i}s"; return 0; }
    sleep 2
  done
  echo "[$1] not ready"; return 1
}

launch "$HOST_A" "$PORT_A" "$ROOT/work-A" "$ROOT/server-A.log"
wait_ready "$HOST_A" "$PORT_A"
B_IP=$(ssh -o BatchMode=yes "$HOST_B" "hostname -I | tr ' ' '\n' | grep -m1 '^192\.'")

launch "$HOST_B" "$PORT_B" "$ROOT/work-B" "$ROOT/server-B.log"
wait_ready "$HOST_B" "$PORT_B"

# B 上的接收端：只作为一次传输的落点，读到对端关闭为止
ssh -o BatchMode=yes "$HOST_B" "rm -f $ROOT/recv.json; setsid --fork nohup $PY $LABS/session_migration_bench.py serve-recv --port $RECV_PORT --out $ROOT/recv.json </dev/null > $ROOT/recv.log 2>&1; sleep 2; tail -1 $ROOT/recv.log"

ssh -o BatchMode=yes "$HOST_A" "cd /scratch/learn && $PY work/labs/L9/session_migration_bench.py bench \
  --out $ROOT/bench --worker-a http://127.0.0.1:$PORT_A/v1 --worker-b http://$B_IP:$PORT_B/v1 \
  --receiver $B_IP:$RECV_PORT --rounds 4 --prefix-tokens $PREFIX --max-tokens 16" \
  | tee "$ROOT/bench.log"

# 回收：显式按 pidfile 与 GPU 进程列表杀，不用模式匹配
for h in "$HOST_A" "$HOST_B"; do
  ssh -o BatchMode=yes "$h" 'for pf in /scratch/learn/worker-*.pid; do [ -f "$pf" ] && kill "$(cat "$pf")" 2>/dev/null; done; for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill "$pid" 2>/dev/null; done; sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
done
