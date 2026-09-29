#!/usr/bin/env bash
# L5.7 任务 C 的 llama.cpp 路线：GGUF 转换、量化与精度记录（纯 CPU，不占 GPU）。
#
# 计划要求「llama.cpp 的 GGUF 与 TensorRT-LLM engine 单独记录转换和精度」
# 「比较同一任务的初始化、稳态、峰值和扩展面」。这里做 llama.cpp 那一半：
#   build → convert_hf_to_gguf(f16) → quantize(Q4_K_M) → llama-bench → llama-perplexity
#
# 已有的产物不重做（checkout / f16 / Q4_K_M 都按存在即跳过），每次运行写新的 run 目录。
set -euo pipefail
source /scratch/learn/env.sh 2>/dev/null || true
RUN=${1:?pass a run id}
ROOT=/scratch/learn/work
LC=$ROOT/llama.cpp
GGUF=/scratch/learn/models/gguf
OUT=$ROOT/out/llamacpp-$RUN
mkdir -p "$OUT"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/gpu-before.txt"

# ---- 1. 构建（已有就复用，但把 commit 记下来）----
if [ ! -x "$LC/build/bin/llama-bench" ]; then
    rm -rf "$LC"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LC" > "$OUT/clone.log" 2>&1
fi
cd "$LC"
git rev-parse HEAD > "$OUT/llamacpp-commit.txt"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.log" 2>&1
cmake --build build --config Release -j "$(nproc)" > "$OUT/build.log" 2>&1
echo "build ok: $(cat "$OUT/llamacpp-commit.txt")"

# ---- 2. 转换 f16 ----
SNAP=$(ls -d /scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/*/ | head -1)
if [ ! -f "$GGUF/Qwen3-1.7B-f16.gguf" ]; then
    PYTHONPATH=/scratch/learn/envs/serve/lib/python3.12/site-packages \
        /scratch/learn/envs/serve/bin/python convert_hf_to_gguf.py "$SNAP" \
        --outfile "$GGUF/Qwen3-1.7B-f16.gguf" --outtype f16 > "$OUT/convert.log" 2>&1
else
    echo "reuse existing f16 gguf" > "$OUT/convert.log"
fi
echo "hf snapshot: $SNAP" >> "$OUT/convert.log"

# ---- 3. 量化 ----
if [ ! -f "$GGUF/Qwen3-1.7B-Q4_K_M.gguf" ]; then
    build/bin/llama-quantize "$GGUF/Qwen3-1.7B-f16.gguf" \
        "$GGUF/Qwen3-1.7B-Q4_K_M.gguf" Q4_K_M > "$OUT/quantize.log" 2>&1
else
    echo "reuse existing Q4_K_M gguf" > "$OUT/quantize.log"
fi
ls -l "$GGUF" > "$OUT/model-sizes.txt"

# ---- 4. 固定语料 + 基准 ----
cat README.md docs/*.md 2>/dev/null | head -c 400000 > "$ROOT/llama_corpus.txt"
wc -c "$ROOT/llama_corpus.txt" > "$OUT/corpus-size.txt"
: > "$OUT/bench.log"
for m in f16 Q4_K_M; do
    echo "--- $m ---" >> "$OUT/bench.log"
    build/bin/llama-bench -m "$GGUF/Qwen3-1.7B-$m.gguf" -p 512 -n 128 -t 8,16,32 \
        >> "$OUT/bench.log" 2>&1
done

# ---- 5. 困惑度（量化精度）----
: > "$OUT/perplexity.log"
for m in f16 Q4_K_M; do
    echo "--- $m ---" >> "$OUT/perplexity.log"
    build/bin/llama-perplexity -m "$GGUF/Qwen3-1.7B-$m.gguf" -f "$ROOT/llama_corpus.txt" \
        -t 16 -c 512 >> "$OUT/perplexity.log" 2>&1
done
grep -h "Final estimate" "$OUT/perplexity.log" > "$OUT/perplexity-final.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/gpu-after.txt"

echo "=== bench 摘要 ==="; grep -E "^\|" "$OUT/bench.log" | sed 's/|/ /g' | awk '{$1=$1;print}'
echo "=== 困惑度 ==="; cat "$OUT/perplexity-final.txt"
