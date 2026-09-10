#!/usr/bin/env bash
# L1.4 证据补齐 · 一次 kernel launch 到底进不进内核态。
#
# 1.4 原来的论据是 `stime=0`，那只说明**内核态时间小于计时精度**，
# 不能证明系统调用次数为零（STATUS.md §2.3 已撤回该推断）。
# 这个脚本用 strace 真的去数。
#
# 方法：差分。同一模式跑两个规模 N1 < N2，
#       每次 launch 的系统调用数 = (total(N2) - total(N1)) / (N2 - N1)
# 一次性开销（import torch ≈ 数万次 openat/mmap、建 context、加载 cubin）
# 在相减时抵消，所以不需要去猜哪些系统调用属于启动。
#
# 用法：
#   bash syscall_ladder.sh            # 全部模式
#   bash syscall_ladder.sh launch     # 单个模式
set -u

PY=${PY:-/scratch/learn/envs/serve/bin/python}
STRACE=${STRACE:-/scratch/learn/bin/strace}
HERE=$(cd "$(dirname "$0")" && pwd)
WORKER=$HERE/syscall_worker.py
OUT=${OUT:-/scratch/learn/results/syscall}
mkdir -p "$OUT"

N1=${N1:-50000}
N2=${N2:-200000}
MODES=${*:-"idle launch sync graph d2h"}

echo "strace : $($STRACE --version 2>&1 | head -1)"
echo "python : $PY"
echo "规模   : N1=$N1  N2=$N2"
echo

# 跑一次，回显 strace -c 的汇总表，并把总数抓出来
run() {
    local mode=$1 n=$2
    local f="$OUT/${mode}_${n}.txt"
    $STRACE -c -f -o "$f" "$PY" "$WORKER" "$mode" "$n" >/dev/null 2>&1
    # -c 的最后一行是 total
    awk '/^-+ /{seen++} seen==2 && /total/{print $4}' "$f" | tail -1
}

printf "%-8s %12s %12s %14s   %s\n" 模式 "N=$N1" "N=$N2" "每次调用" 说明
printf "%s\n" "--------------------------------------------------------------------------"

declare -A TOT1 TOT2
for m in $MODES; do
    t1=$(run "$m" "$N1")
    t2=$(run "$m" "$N2")
    TOT1[$m]=$t1; TOT2[$m]=$t2
    per=$(awk -v a="$t1" -v b="$t2" -v n1="$N1" -v n2="$N2" \
          'BEGIN{ if (n2>n1) printf "%.6f", (b-a)/(n2-n1); else print "n/a" }')
    case $m in
      idle)   note="基线：N 不影响它，差分应当≈0" ;;
      launch) note="纯下发" ;;
      sync)   note="每次 launch 后 cudaDeviceSynchronize" ;;
      graph)  note="100 次 launch 录成一张图后 replay" ;;
      d2h)    note="每次 launch 后 .item() 回读" ;;
      *)      note="" ;;
    esac
    printf "%-8s %12s %12s %14s   %s\n" "$m" "$t1" "$t2" "$per" "$note"
done

echo
echo "=== 增量里都是哪些系统调用（N2 减 N1）==="
echo "注意：只在其中一个规模里出现的系统调用，用 join 是看不到的（第一版就漏了）。"
echo "所以这里对两边的名字取并集再补 0。"
for m in $MODES; do
    echo
    echo "--- $m ---"
    for n in "$N1" "$N2"; do
        awk '/^-+ /{f++;next} f==1&&NF>=4{print $NF" "$4}' "$OUT/${m}_${n}.txt" | sort > "/tmp/ladder_${m}_${n}.tsv"
    done
    # 名字取并集，缺的补 0
    cut -d" " -f1 "/tmp/ladder_${m}_${N1}.tsv" "/tmp/ladder_${m}_${N2}.tsv" | sort -u > "/tmp/ladder_${m}_names"
    join -a1 -e 0 -o 0,2.2 "/tmp/ladder_${m}_names" "/tmp/ladder_${m}_${N1}.tsv" | sort > "/tmp/ladder_${m}_a"
    join -a1 -e 0 -o 0,2.2 "/tmp/ladder_${m}_names" "/tmp/ladder_${m}_${N2}.tsv" | sort > "/tmp/ladder_${m}_b"
    join "/tmp/ladder_${m}_a" "/tmp/ladder_${m}_b" \
      | awk -v n1="$N1" -v n2="$N2" \
          '{d=$3-$2; if (d>10) printf "  %-18s %9d -> %9d   delta=%9d   每次 %.5f\n", $1,$2,$3,d,d/(n2-n1)}' \
      | sort -k5 -rn
    join "/tmp/ladder_${m}_a" "/tmp/ladder_${m}_b" | awk '{d=$3-$2; if (d>10) c++} END{if (!c) print "  （没有任何系统调用随 N 增长超过 10 次）"}'
done

echo
echo "原始 strace 汇总表在 $OUT/"
