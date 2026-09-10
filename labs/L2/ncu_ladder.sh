#!/usr/bin/env bash
# L2.6 lab（spark 专用）· 用真实硬件计数器审判前几章的推断。
#
# spark 是裸机，放开 NVreg_RestrictProfilingToAdminUsers 之后 ncu 可用。
# 三台容器机（crater/crater2/worldvln）跑不了——见 profile_ladder.sh 的权限诊断。
#
# 要回答的三个"以前只能推断"的问题：
#   1. L1.1 说 bank conflict 慢 10.1×  —— 冲突到底发生了几次？
#   2. L2.3 说 v4 的 float4 改善了访存合并 —— 每次请求碰几个 sector？
#   3. L2.6 说静态指令数 ≠ 动态执行次数   —— 那动态次数到底是多少？
#
# 注意 GB10 是**统一内存**架构：`ncu --query-metrics | grep ^dram__` 一条都没有，
# 访存流量只能在 L2（lts__）这一层看。
#
# 用法：bash ncu_ladder.sh [输出目录]
set -uo pipefail

OUT="${1:-$HOME/learn/results/ncu}"
LAB="$(cd "$(dirname "$0")" && pwd)"
export PATH=/usr/local/cuda-13.0/bin:$PATH
NCU="${NCU:-$(command -v ncu)}"
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
mkdir -p "$OUT"

echo "=== 环境 ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
"$NCU" --version 2>&1 | sed -n '3p'
echo "RmProfilingAdminOnly = $(awk '/RmProfilingAdminOnly/{print $2}' /proc/driver/nvidia/params)"
echo "编译目标 $ARCH"
echo

# ---------------------------------------------------------------------------
# 采集：每个 kernel 名单独跑一次 ncu（-c 1），把 CSV 拼起来。
# 直接对整个程序 --launch-count N 会按 launch 顺序抓，抓不全每一级。
# ---------------------------------------------------------------------------
collect() {   # collect <输出csv> <二进制> <metrics> <kernel名...>
    local csv="$1" bin="$2" metrics="$3"; shift 3
    : > "$csv"
    local first=1
    for k in "$@"; do
        "$NCU" --csv --kernel-name-base function -k "regex:^${k}$|^${k}<" -c 1 \
               --metrics "$metrics" "$bin" 2>>"${csv%.csv}.err" \
            | sed -n '/^"ID"/,$p' \
            | { if [ $first -eq 1 ]; then cat; first=0; else tail -n +2; fi; } >> "$csv"
        first=0
    done
    grep -c '^"' "$csv" > /dev/null 2>&1 || true
}

render() {   # render <csv> <标题>
    python3 - "$1" "$2" <<'PYEOF'
import csv, sys, collections
lines = open(sys.argv[1], errors="replace").read().splitlines()
# 多次 ncu 调用拼起来的 CSV 里会有重复表头，过滤掉
rows = [r for r in csv.DictReader(l for l in lines if l.startswith('"'))
        if r.get("Kernel Name") and r["Kernel Name"] != "Kernel Name"]
if not rows:
    print("  (无数据；看 .err)"); raise SystemExit
per = collections.OrderedDict()
order = []
for r in rows:
    k = r["Kernel Name"].split("(")[0].replace("void ", "")
    if k not in per:
        per[k] = {}; order.append(k)
    per[k][r["Metric Name"]] = (r["Metric Value"], r.get("Metric Unit", ""))
metrics = list(dict.fromkeys(r["Metric Name"] for r in rows))
SHORT = {
    "gpu__time_duration.sum": "耗时ns",
    "smsp__inst_executed.sum": "动态指令(warp)",
    "lts__t_sectors.sum": "L2 sector",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "sector/请求",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "全局读请求",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "bank冲突",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "smem读wavefront",
    "l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum": "smem读请求",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "占用率%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM吞吐%",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "访存吞吐%",
    "sm__inst_executed_pipe_tensor_subpipe_hmma.sum": "HMMA指令",
    "launch__registers_per_thread": "寄存器",
    "launch__shared_mem_per_block_static": "smem/blk",
}
W = 16
print(f"  {sys.argv[2]}")
print("  " + f"{'kernel':24s}" + "".join(f"{SHORT.get(m, m[-14:]):>{W}s}" for m in metrics))
for k in order:
    print("  " + f"{k[:24]:24s}" +
          "".join(f"{per[k].get(m, ('',''))[0]:>{W}s}" for m in metrics))
PYEOF
}

# ---------------------------------------------------------------------------
# [1] bank conflict：从"时间慢了 10 倍"到"冲突发生了 N 次"
#
#     判据不是时间，是 wavefront/request 的比值：
#       无冲突 -> 每个请求 1 个 wavefront
#       N 路冲突 -> 每个请求 N 个 wavefront（要串行发 N 次）
# ---------------------------------------------------------------------------
echo "=== [1] bank conflict：从推断到观测 ==="
nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/bank_probe" "$LAB/bank_conflict_probe.cu" 2>&1 | tail -2
"$OUT/bank_probe" | tee "$OUT/10_bank_timing.txt"
echo
BANK_M="gpu__time_duration.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
collect "$OUT/11_bank_ncu.csv" "$OUT/bank_probe" "$BANK_M" probe_stride1 probe_stride32 probe_stride33
render  "$OUT/11_bank_ncu.csv" "ncu 读出的实际冲突次数（三个 probe 依次是 stride 1/32/33）："
echo

# ---------------------------------------------------------------------------
# [2] L2.3 归约阶梯
# ---------------------------------------------------------------------------
echo "=== [2] L2.3 归约阶梯：合并质量与动态指令数 ==="
nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/reduce_ladder" "$LAB/reduce_ladder.cu" 2>&1 | tail -2
RED_M="gpu__time_duration.sum,\
smsp__inst_executed.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
lts__t_sectors.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
collect "$OUT/20_reduce_ncu.csv" "$OUT/reduce_ladder" "$RED_M" \
        red_v0 red_v1 red_v2 red_v3 red_v4 red_v5 red_v6
render  "$OUT/20_reduce_ncu.csv" "每一级的硬件计数器："
echo

# ---------------------------------------------------------------------------
# [3] L2.4 GEMM 阶梯
# ---------------------------------------------------------------------------
echo "=== [3] L2.4 GEMM 阶梯：tensor core 与共享内存 ==="
nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/gemm_ladder" "$LAB/gemm_ladder.cu" -lcublas 2>&1 | tail -2
GEMM_M="gpu__time_duration.sum,\
sm__inst_executed_pipe_tensor_subpipe_hmma.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
sm__warps_active.avg.pct_of_peak_sustained_active"
collect "$OUT/30_gemm_ncu.csv" "$OUT/gemm_ladder" "$GEMM_M" \
        gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v5 gemm_v6
render  "$OUT/30_gemm_ncu.csv" "每一级的硬件计数器："
echo

# ---------------------------------------------------------------------------
# [4] --set full：完整报告 + 源码级归因
# ---------------------------------------------------------------------------
echo "=== [4] --set full 完整报告（首尾两级）==="
"$NCU" --set full -k "regex:red_v0|red_v6" -c 2 \
       -o "$OUT/40_reduce_full" --force-overwrite \
       "$OUT/reduce_ladder" > "$OUT/40_full_run.log" 2>&1
echo "    -> 40_reduce_full.ncu-rep  $(du -h "$OUT/40_reduce_full.ncu-rep" 2>/dev/null | cut -f1)"
"$NCU" --import "$OUT/40_reduce_full.ncu-rep" --page details \
       > "$OUT/41_full_details.txt" 2>&1
echo "    -> 41_full_details.txt  ($(wc -l < "$OUT/41_full_details.txt") 行)"
"$NCU" --import "$OUT/40_reduce_full.ncu-rep" --page raw \
       > "$OUT/43_full_raw.txt" 2>&1
echo "    -> 43_full_raw.txt      ($(wc -l < "$OUT/43_full_raw.txt") 行)"
echo

# ---------------------------------------------------------------------------
# [5] --cache-control：ncu 默认每次 replay 前刷 cache，测的是冷缓存
# ---------------------------------------------------------------------------
echo "=== [5] --cache-control：ncu 默认测的是冷缓存 ==="
for cc in all none; do
    "$NCU" --csv --cache-control "$cc" --clock-control base \
           -k "regex:red_v6" -c 1 \
           --metrics gpu__time_duration.sum,lts__t_sectors.sum \
           "$OUT/reduce_ladder" 2>/dev/null | sed -n '/^"ID"/,$p' > "$OUT/50_cc_$cc.csv"
    python3 - "$OUT/50_cc_$cc.csv" "$cc" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(l for l in open(sys.argv[1], errors="replace") if l.startswith('"')))
d = {r["Metric Name"]: r["Metric Value"] for r in rows}
print(f"    --cache-control={sys.argv[2]:5s} red_v6 耗时 "
      f"{d.get('gpu__time_duration.sum','?'):>12s} ns   "
      f"L2 sector {d.get('lts__t_sectors.sum','?'):>14s}")
PYEOF
done
echo "    （对照：干净循环里 CUDA event 量到的稳态耗时见 [2] 表）"

echo
echo "所有产物在 $OUT"
