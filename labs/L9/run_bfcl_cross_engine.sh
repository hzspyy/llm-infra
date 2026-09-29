#!/usr/bin/env bash
# L9.1 任务 C：BFCL 固定 50 题 + 跨引擎模板/支持模式核对（同一批题在两个引擎上各跑一次）。
#
#   bash labs/L9/run_bfcl_cross_engine.sh <host> <out_root> [n]
#
# 步骤：起 vLLM（hermes parser）→ BFCL 50 题 → 跨引擎探针 → 收服务 → 起 SGLang
# （qwen25 parser）→ 同一批 50 题 → 探针 → 收服务。两边的启动命令、版本、GPU 用量与
# 原始输出都留在 out_root 下，便于复核「模板/支持模式」的差异来自引擎而不是题集。
#
# 注意：BFCL 的判分与工具环境来自 bfcl-eval 的同一份安装；两个引擎都用 OpenAI 兼容
# 接口，所以 run_bfcl_cross_engine.sh 只换 base_url 与解析器，不换客户端代码。
set -u
HOST=${1:?usage: run_bfcl_cross_engine.sh <host> <out_root> [n]}
ROOT=${2:?missing out_root}
N=${3:-50}
PORT=${L91_PORT:-8061}
SPORT=${L91_SGL_PORT:-8062}
PY=/scratch/learn/envs/serve/bin/python
PYB=/scratch/learn/envs/bfcl/bin/python
PYS=/scratch/learn/envs/sgl/bin/python
LABS=/scratch/learn/work/labs/L9
MODEL=Qwen/Qwen3-4B

ssh -o BatchMode=yes "$HOST" "rm -rf $ROOT; mkdir -p $ROOT/{entry,vllm,sglang}; \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-before.txt; \
  $PYB -c 'import importlib.metadata as m; print(\"bfcl-eval\", m.version(\"bfcl-eval\"))' > $ROOT/versions.txt 2>&1; \
  $PY -c 'import importlib.metadata as m; print(\"vllm\", m.version(\"vllm\")); print(\"transformers\", m.version(\"transformers\"))' >> $ROOT/versions.txt 2>&1; \
  $PYS -c 'import sglang; print(\"sglang\", sglang.__version__)' >> $ROOT/versions.txt 2>&1"

# --- 取一条真实 BFCL 题，作为两引擎共用输入 -----------------------------------------
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PYB $LABS/cross_engine_template.py dump-entry \
  --entry-id multi_turn_base_0 --out $ROOT/entry/entry.json" || exit 1

start_vllm() {
  ssh -o BatchMode=yes "$HOST" "source /scratch/learn/env.sh; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    setsid --fork bash -c 'echo \$\$ > $ROOT/vllm.pgid; exec $PY -u -m vllm.entrypoints.openai.api_server \
      --model $MODEL --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.45 --max-num-seqs 16 \
      --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
      --enable-prefix-caching --enable-prompt-tokens-details --no-enable-log-requests' \
      </dev/null > $ROOT/vllm-server.log 2>&1; sleep 1; cat $ROOT/vllm.pgid"
}

start_sglang() {
  ssh -o BatchMode=yes "$HOST" "source /scratch/learn/env.sh; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    setsid --fork bash -c 'echo \$\$ > $ROOT/sglang.pgid; exec $PYS -u -m sglang.launch_server \
      --model-path $MODEL --served-model-name $MODEL --host 127.0.0.1 --port $SPORT \
      --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.35 \
      --tool-call-parser qwen25 --reasoning-parser qwen3' \
      </dev/null > $ROOT/sglang-server.log 2>&1; sleep 1; cat $ROOT/sglang.pgid"
}

wait_ready() {  # port timeout_s
  local port=$1 t=$2 i
  for i in $(seq 1 "$t"); do
    ssh -o BatchMode=yes "$HOST" "curl -sf http://127.0.0.1:$port/v1/models >/dev/null" \
      && { echo "[engine :$port] ready after ${i}s"; return 0; }
    sleep 1
  done
  echo "[engine :$port] 未就绪"; return 1
}

kill_pgid() {  # pgid_file
  # 只回收本脚本启动的进程组；引擎的子进程（scheduler/EngineCore）不会被漏掉。
  ssh -o BatchMode=yes "$HOST" "pgid=\$(cat $1 2>/dev/null); if [ -n \"\$pgid\" ]; then \
    kill -TERM -- -\$pgid 2>/dev/null; sleep 5; kill -KILL -- -\$pgid 2>/dev/null; fi; sleep 2; \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader"
}

# --- vLLM ---------------------------------------------------------------------------
start_vllm;  wait_ready "$PORT" 180 || { ssh -o BatchMode=yes "$HOST" "tail -40 $ROOT/vllm-server.log"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PYB $LABS/bfcl_multi_turn.py run \
  --out $ROOT/vllm/bfcl --base-url http://127.0.0.1:$PORT/v1 --model $MODEL \
  --n $N --concurrency 4 --max-tokens 1024 --max-steps-per-turn 4 \
  > $ROOT/vllm/bfcl.log 2>&1; echo \$? > $ROOT/vllm/bfcl.exit"
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PY $LABS/cross_engine_template.py probe \
  --entry $ROOT/entry/entry.json --engine vllm --base-url http://127.0.0.1:$PORT/v1 \
  --model $MODEL --tokenizer $MODEL --out $ROOT/vllm/probe.json > $ROOT/vllm/probe.log 2>&1; echo \$? > $ROOT/vllm/probe.exit"

kill_pgid "$ROOT/vllm.pgid"

# --- SGLang -------------------------------------------------------------------------
start_sglang; wait_ready "$SPORT" 300 || { ssh -o BatchMode=yes "$HOST" "tail -40 $ROOT/sglang-server.log"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PYB $LABS/bfcl_multi_turn.py run \
  --out $ROOT/sglang/bfcl --base-url http://127.0.0.1:$SPORT/v1 --model $MODEL \
  --n $N --concurrency 4 --max-tokens 1024 --max-steps-per-turn 4 \
  > $ROOT/sglang/bfcl.log 2>&1; echo \$? > $ROOT/sglang/bfcl.exit"
ssh -o BatchMode=yes "$HOST" "cd /scratch/learn/work && $PY $LABS/cross_engine_template.py probe \
  --entry $ROOT/entry/entry.json --engine sglang --base-url http://127.0.0.1:$SPORT/v1 \
  --model $MODEL --tokenizer $MODEL --out $ROOT/sglang/probe.json > $ROOT/sglang/probe.log 2>&1; echo \$? > $ROOT/sglang/probe.exit"

ssh -o BatchMode=yes "$HOST" "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $ROOT/gpu-after.txt"
kill_pgid "$ROOT/sglang.pgid"

echo "[done] $HOST:$ROOT"
