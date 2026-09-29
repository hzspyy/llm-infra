#!/bin/bash
# L2.4-D · sm_120 上 CUTLASS CuTe-DSL 的 TMA + persistent GEMM 路径
#
# 运行官方 blackwell_geforce dense GEMM 例程（128×256×64 等 tile，TMA 多级流水、
# persistent tile scheduler、mma.sync 寄存器累加器），并保留：
#   - 每次运行的耗时 / TFLOPS（例程自己打印）
#   - CuTe-DSL 生成的 clean MLIR（CUTE_DSL_KEEP=ir）
#   - PTX 与 sm_120a cubin，以及 SASS 指令普查与资源用量
#
# 依赖：CUTLASS 固定 commit 147295a3 的源码树（见 ENVIRONMENTS）。
#
#   bash labs/L2/run_dsl_sm120_gemm.sh [out_dir]
set -euo pipefail

CUTLASS_SRC="${CUTLASS_SRC:-/scratch/learn/opt/src/cutlass-147295a3}"
EX="$CUTLASS_SRC/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm"
OUT="${1:-$PWD/out/2.4/dsl-sm120}"
PY="${PY:-/scratch/learn/envs/serve/bin/python}"
CUDA_HOME="${CUDA_HOME:-/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13}"

mkdir -p "$OUT/artifacts"
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-/scratch/learn/.cache/cute_dsl}"
export CUDA_TOOLKIT_PATH="$CUDA_HOME"      # 不设会让 libNVVM 找不到后端

echo "CUTLASS commit: $(git -C "$CUTLASS_SRC" rev-parse HEAD)" | tee "$OUT/dsl_sm120.txt"
{
  echo "例程: examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"
  echo "命令: dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk <TILE> --iterations 5 --warmup_iterations 2"
  echo
} >> "$OUT/dsl_sm120.txt"

for TILE in 64,64,64 128,128,64 128,256,64; do
    D="$OUT/artifacts/tile_${TILE//,/_}"
    mkdir -p "$D"
    echo "=== tile $TILE ===" | tee -a "$OUT/dsl_sm120.txt"
    ( cd "$EX" && \
      CUTE_DSL_KEEP=ir,ptx,cubin CUTE_DSL_DUMP_DIR="$D" \
      "$PY" dense_gemm.py --mnkl 4096,4096,4096,1 --tile_shape_mnk "$TILE" \
        --iterations 5 --warmup_iterations 2 2>&1 ) \
      | grep -E "Tile Shape|Execution time|ab_stage|microseconds|PASS|Error|error" \
      | tee -a "$OUT/dsl_sm120.txt"
    echo >> "$OUT/dsl_sm120.txt"
done

echo "== SASS 普查与资源用量 ==" | tee "$OUT/sass_census.txt"
"$PY" "$(dirname "$0")/sass_census.py" "$OUT"/artifacts/tile_*/*.cubin 2>&1 | tee -a "$OUT/sass_census.txt"

echo | tee -a "$OUT/sass_census.txt"
echo "== 生成的 MLIR 里各关键 op 的出现次数 ==" | tee -a "$OUT/sass_census.txt"
for F in "$OUT"/artifacts/tile_*/*clean.mlir; do
    [ "$(wc -c < "$F")" -lt 5000 ] && continue   # 跳过 host wrapper 那份小 IR
    echo "-- $(basename "$(dirname "$F")")  $(wc -c < "$F") 字节" | tee -a "$OUT/sass_census.txt"
    for OP in "cute.gemm" "mma_f16_f16_f32_16x8x16" "cute.copy" "copy_ldsm" \
              "nvvm.cp.async.bulk" "nvvm.mbarrier" "nvvm.setmaxregister" \
              "tcgen05" "tmem" "elect"; do
        printf "   %-26s %s\n" "$OP" "$(grep -o "$OP" "$F" | wc -l)" | tee -a "$OUT/sass_census.txt"
    done
done

# ---- 轻量摘录：把大件产物留在运行机，只把可引用的片段与哈希带回仓库 ----
CB="${CUOBJDUMP:-/scratch/learn/opt/cuobjdump/cuda_cuobjdump-linux-x86_64-13.0.85-archive/bin/cuobjdump}"
NV=/scratch/learn/envs/serve/lib/python3.12/site-packages/nvidia/cu13/bin
{
    echo "== 生成物体积与 SHA256（完整 cubin/PTX/MLIR 留在运行机，不入库）=="
    for f in $(find "$OUT/artifacts" -type f | sort); do
        printf "%-24s %9s  %s\n" "$(basename "$(dirname "$f")")/$(basename "$f" | cut -c1-28)" \
            "$(wc -c < "$f")" "$(sha256sum "$f" | cut -c1-16)"
    done
    echo
    echo "== PTX 里 TMA 与 mma.sync 的条数 =="
    for t in "$OUT"/artifacts/tile_*; do
        printf "%-16s cp.async.bulk %4s   mma.sync.m16n8k16 %4s   ldmatrix %3s\n" \
            "$(basename "$t")" "$(grep -c cp.async.bulk "$t"/*.ptx)" \
            "$(grep -c mma.sync.aligned.m16n8k16 "$t"/*.ptx)" "$(grep -c ldmatrix "$t"/*.ptx)"
    done
    echo
    echo "== PTX 片段：TMA 3-D tile 装载（128×256×64）=="
    grep -nE "cp\.async\.bulk\.tensor\.3d\.global|cp\.async\.bulk\.commit_group" \
        "$OUT"/artifacts/tile_128_256_64/*.ptx | head -3
    echo
    echo "== SASS 片段：TMA 与 HMMA（128×256×64）=="
    F=$(ls "$OUT"/artifacts/tile_128_256_64/*.cubin)
    PATH="$NV:$PATH" "$CB" -sass "$F" 2>/dev/null | grep -E "UTMALDG|UTMASTG|UTMACCTL|HMMA" | head -10
    echo
    echo "== SASS 片段：同一 kernel 的本地访存（spill）=="
    PATH="$NV:$PATH" "$CB" -sass "$F" 2>/dev/null | grep -E "LDL|STL" | head -4
    echo
    echo "== clean MLIR 片段 =="
    grep -nE "setmaxregister|cute\.gemm" "$OUT"/artifacts/tile_128_256_64/*clean.mlir | head -3 | cut -c1-200
} > "$OUT/artifacts_excerpt.txt"
echo "摘录 -> $OUT/artifacts_excerpt.txt"
