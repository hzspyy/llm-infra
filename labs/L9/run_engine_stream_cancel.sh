#!/usr/bin/env bash
# L9.5 任务 C（引擎侧一档）：真实引擎上跑流式取消，读引擎自己的指标判断服务端是否停止。
#
#   bash labs/L9/run_engine_stream_cancel.sh <host> <out_root> [repeat]
#
# 依次在 vLLM 0.29.0 与 SGLang 0.5.19 上各跑一次；两边都起真实服务并读 /metrics，
# 运行结束后立刻回收 GPU 进程。SGLang 需要显式 --enable-metrics，否则没有 /metrics。
set -u
HOST=${1:?usage: run_engine_stream_cancel.sh <host> <out_root> [repeat]}
ROOT=${2:?missing out_root}
REPEAT=${3:-3}
VPORT=${L95_V_PORT:-8061}
SPORT=${L95_S_PORT:-8062}
PY=/scratch/learn/envs/serve/bin/python
PYS=/scratch/learn/envs/sgl/bin/python
LABS=/scratch/learn/work/labs/L9
MODEL=Qwen/Qwen3-4B

ssh -o BatchMode=yes "$HOST" "rm -rf $ROOT; mkdir -p $ROOT; \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-before.txt"

wait_ready() {
  local port=$1 t=$2 i
  for i in $(seq 1 "$t"); do
    ssh -o BatchMode=yes "$HOST" "curl -sf http://127.0.0.1:$port/v1/models >/dev/null" \
      && { echo "[engine :$port] ready after ${i}s"; return 0; }
    sleep 1
  done
  echo "[engine :$port] 未就绪"; return 1
}

reap() {  # pgid_file
  # `setsid --fork bash -c` 让服务成为新会话的组长，pgid 写在文件里；只杀这个进程组，
  # 不会碰到同机上别人的进程（pkill 按名字匹配会误伤，也是本轮踩过的坑）。
  ssh -o BatchMode=yes "$HOST" "pgid=\$(cat $1 2>/dev/null); if [ -n \"\$pgid\" ]; then \
    kill -TERM -- -\$pgid 2>/dev/null; sleep 5; kill -KILL -- -\$pgid 2>/dev/null; fi; sleep 2; \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader"
}

# --- vLLM ---------------------------------------------------------------------------
ssh -o BatchMode=yes "$HOST" "source /scratch/learn/env.sh; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
  setsid --fork bash -c 'echo \$\$ > $ROOT/vllm.pgid; exec $PY -u -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name $MODEL --host 127.0.0.1 --port $VPORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.45 --max-num-seqs 16 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --enable-prefix-caching --no-enable-log-requests' \
    </dev/null > $ROOT/vllm-server.log 2>&1; sleep 1; cat $ROOT/vllm.pgid"
wait_ready "$VPORT" 240 || { ssh -o BatchMode=yes "$HOST" "tail -30 $ROOT/vllm-server.log"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PY $LABS/engine_stream_cancel.py run \
  --engine vllm --base-url http://127.0.0.1:$VPORT/v1 --model $MODEL --repeat $REPEAT \
  --out $ROOT/vllm.json > $ROOT/vllm.log 2>&1; echo \$? > $ROOT/vllm.exit"
reap "$ROOT/vllm.pgid"

# --- SGLang -------------------------------------------------------------------------
ssh -o BatchMode=yes "$HOST" "source /scratch/learn/env.sh; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
  setsid --fork bash -c 'echo \$\$ > $ROOT/sglang.pgid; exec $PYS -u -m sglang.launch_server \
    --model-path $MODEL --served-model-name $MODEL --host 127.0.0.1 --port $SPORT \
    --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.35 \
    --tool-call-parser qwen25 --reasoning-parser qwen3 --enable-metrics' \
    </dev/null > $ROOT/sglang-server.log 2>&1; sleep 1; cat $ROOT/sglang.pgid"
wait_ready "$SPORT" 300 || { ssh -o BatchMode=yes "$HOST" "tail -30 $ROOT/sglang-server.log"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PY $LABS/engine_stream_cancel.py run \
  --engine sglang --base-url http://127.0.0.1:$SPORT/v1 --model $MODEL --repeat $REPEAT \
  --out $ROOT/sglang.json > $ROOT/sglang.log 2>&1; echo \$? > $ROOT/sglang.exit"
reap "$ROOT/sglang.pgid"

ssh -o BatchMode=yes "$HOST" "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-after.txt; \
  $PY -c 'import importlib.metadata as m; print(\"vllm\", m.version(\"vllm\"))' > $ROOT/versions.txt 2>&1; \
  $PYS -c 'import sglang; print(\"sglang\", sglang.__version__)' >> $ROOT/versions.txt 2>&1"
echo "[done] $HOST:$ROOT"
