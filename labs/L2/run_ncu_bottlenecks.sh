#!/usr/bin/env bash
# L2.6-B · 在 spark（GB10，唯一能读硬件计数器的机器）上复核四类瓶颈的判据。
#
# 做三件事：
#   1. 不带 profiler 跑一遍，记墙钟与 GPU 忙时间（对照未插桩时间）；
#   2. 用 ncu 采一组固定计数器，看哪几类瓶颈能被计数器直接量化；
#   3. 用 nsys 采时间线，看哪几类瓶颈只能从"GPU 空闲"看出来。
#
# GB10 是统一内存：没有任何 dram__* 计数器，访存流量只能在 L2（lts__）看。
#
#   bash labs/L2/run_ncu_bottlenecks.sh [out_dir]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HOME/learn/results/2.6-bottlenecks}"
mkdir -p "$OUT"
export PATH=/usr/local/cuda-13.0/bin:$PATH
NCU="${NCU:-/usr/local/cuda-13.0/bin/ncu}"
NSYS="${NSYS:-/usr/local/bin/nsys}"
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"

echo "== 环境 ==" | tee "$OUT/ncu_bottlenecks.txt"
{
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
  "$NCU" --version 2>&1 | sed -n '3p'
  echo "RmProfilingAdminOnly = $(awk '/RmProfilingAdminOnly/{print $2}' /proc/driver/nvidia/params)"
  echo "编译目标 $ARCH"
} 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"

nvcc -O3 -arch="$ARCH" -o "$OUT/ncu_bottlenecks" "$HERE/ncu_bottlenecks.cu" 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"
echo "nvcc exit=$?" | tee -a "$OUT/ncu_bottlenecks.txt"

echo | tee -a "$OUT/ncu_bottlenecks.txt"
echo "== 1. 未插桩：墙钟与 GPU 忙时间 ==" | tee -a "$OUT/ncu_bottlenecks.txt"
"$OUT/ncu_bottlenecks" info 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"
for c in cpu_submit bandwidth sync occ96 occ8; do
    "$OUT/ncu_bottlenecks" "$c" 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"
done

echo | tee -a "$OUT/ncu_bottlenecks.txt"
echo "== 2. ncu 计数器（每个 kernel 单独抓）==" | tee -a "$OUT/ncu_bottlenecks.txt"
{
  echo "-- 可用性检查：dram__* 在 GB10 上有几条"
  "$NCU" --query-metrics 2>/dev/null | grep -c '^dram__' || true
} 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"

# 注意：续行反斜杠后面**不能留缩进**。带上前导空格的指标名 ncu 不会报错，
# 而是每条都记成 n/a（Metric Unit 为空），看起来像"这台机器没有计数器"。
METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,\
lts__t_sectors.sum,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
launch__grid_size,\
launch__block_size"

ncu_case() {   # ncu_case <case> <kernel-regex> <launch-skip> <name>
    local case="$1" kregex="$2" skip="$3" name="$4"
    local csv="$OUT/ncu_${name}.csv"
    "$NCU" --target-processes all -k "regex:$kregex" -s "$skip" -c 1 \
           --metrics "$METRICS" --csv --log-file "$csv" \
           "$OUT/ncu_bottlenecks" "$case" >/dev/null 2>&1
    echo "-- $name：$csv" | tee -a "$OUT/ncu_bottlenecks.txt"
    # CSV 前面可能夹着 ==PROF== 与应用自己的输出，从表头行开始取
    awk '/^"ID"/{f=1} f' "$csv" | head -4 | cut -c1-260 | tee -a "$OUT/ncu_bottlenecks.txt"
}

ncu_case bandwidth "add_kernel" 0 bandwidth
ncu_case occ96     "smem_kernel" 0 occ96
ncu_case occ8      "smem_kernel" 0 occ8
ncu_case sync      "add_kernel" 0 sync

echo | tee -a "$OUT/ncu_bottlenecks.txt"
echo "== 3. nsys 时间线（CPU 提交与同步看这一列）==" | tee -a "$OUT/ncu_bottlenecks.txt"
for c in cpu_submit sync bandwidth; do
    "$NSYS" profile -o "$OUT/nsys_$c" --force-overwrite true \
        "$OUT/ncu_bottlenecks" "$c" >/dev/null 2>&1
    "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
        "$OUT/nsys_$c.nsys-rep" > "$OUT/nsys_${c}_stats.csv" 2>/dev/null
    echo "-- $c：$OUT/nsys_${c}_stats.csv" | tee -a "$OUT/ncu_bottlenecks.txt"
    head -4 "$OUT/nsys_${c}_stats.csv" | cut -c1-160 | tee -a "$OUT/ncu_bottlenecks.txt"
done

echo | tee -a "$OUT/ncu_bottlenecks.txt"
echo "== 4. mini profiler（自己采 block 起止，对照未插桩时间）==" | tee -a "$OUT/ncu_bottlenecks.txt"
nvcc -O3 -arch="$ARCH" -o "$OUT/mini_profiler" "$HERE/mini_profiler.cu" 2>&1 | tee -a "$OUT/ncu_bottlenecks.txt"
"$OUT/mini_profiler" 2>&1 | tail -20 | tee -a "$OUT/ncu_bottlenecks.txt"
