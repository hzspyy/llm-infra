#!/usr/bin/env bash
# L2.6 lab · 三把尺子各自能量什么，以及为什么其中一把在这台机器上量不了。
#
#   nsys  —— 时间轴。走 CUPTI activity 接口，只需要能 attach 到自己的进程。
#   ncu   —— 单 kernel 内部。要读 SM 硬件性能计数器，那是**特权资源**。
#   自制   —— mini_profiler.cu，用 %smid/%globaltimer 重建 occupancy，零特权。
#
# 本脚本先做权限诊断（这决定了后面能跑什么），再跑能跑的部分，
# 并把 ncu 的失败原文原样留下——那是最有教学价值的一条错误信息。
#
# 用法：bash profile_ladder.sh [输出目录]
set -uo pipefail

OUT="${1:-results/prof}"
LAB="$(cd "$(dirname "$0")" && pwd)"
ROOT="${LEARN_ROOT:-/scratch/learn}"
NCU="${NCU:-$ROOT/tools/nsight-compute/ncu}"
NSYS="${NSYS:-$ROOT/tools/nsight-systems/bin/nsys}"
mkdir -p "$OUT"

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
ARCH="sm_${CC}"

# ---------------------------------------------------------------------------
# 0. 权限诊断。ERR_NVGPUCTRPERM 是 GPU profiling 最常见的一堵墙，
#    它由三件事共同决定，缺一不可地都要查。
# ---------------------------------------------------------------------------
{
echo "=== [0] 环境与 profiling 权限诊断 ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
echo "编译目标           $ARCH"
echo
echo "--- 证据 1：驱动是否限制性能计数器（2019 年起默认限制，CVE-2018-6260）---"
grep -i "RmProfilingAdminOnly" /proc/driver/nvidia/params 2>/dev/null || echo "(读不到 params)"
echo
echo "--- 证据 2：本进程有没有 CAP_SYS_ADMIN ---"
CAPEFF=$(grep CapEff /proc/self/status | awk '{print $2}')
echo "CapEff = $CAPEFF"
python3 -c "
c=int('$CAPEFF',16)
print('  CAP_SYS_ADMIN(bit 21) =', bool(c>>21&1))
print('  CAP_PERFMON  (bit 38) =', bool(c>>38&1))"
echo
echo "--- 证据 3：capability 在哪个 user namespace 里有效 ---"
UIDMAP="$(cat /proc/self/uid_map)"
echo "uid_map: $UIDMAP"
echo "  形如 '0 0 4294967295' = 就在宿主的 user namespace 里；"
echo "  形如 '0 1000000 ...'  = UID 被移位，容器内的 root 在宿主看来是普通 uid，"
echo "                          此时容器里的 CAP_SYS_ADMIN 对驱动不算数。"
ls -la /dev/nvidiactl 2>/dev/null
echo "  (属主显示成 nobody 是 UID 移位的旁证)"
echo "--- 结论 ---"
echo "  能读性能计数器 = RmProfilingAdminOnly=0  或  (有 CAP_SYS_ADMIN 且在宿主 namespace)"
echo
} 2>&1 | tee "$OUT/00_perm_diag.txt"

# ---------------------------------------------------------------------------
# 1. ncu 尝试。留下原文——不管成功还是失败。
# ---------------------------------------------------------------------------
echo "=== [1] ncu 尝试 ==="
if [ -x "$NCU" ]; then
    "$NCU" --version 2>&1 | sed -n '3p'
    cat > "$OUT/_smoke.cu" <<'EOF'
__global__ void k(float* p, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=p[i]*2.f+1.f;}
int main(){float*p; cudaMalloc(&p, 1<<26); k<<<(1<<24)/256,256>>>(p, 1<<24); cudaDeviceSynchronize(); return 0;}
EOF
    nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/_smoke" "$OUT/_smoke.cu" 2>&1 | tail -2
    "$NCU" --csv --metrics gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
           "$OUT/_smoke" > "$OUT/10_ncu_attempt.txt" 2>&1
    tail -4 "$OUT/10_ncu_attempt.txt"
else
    echo "(未安装 ncu：$NCU)" | tee "$OUT/10_ncu_attempt.txt"
fi
echo

# ---------------------------------------------------------------------------
# 2. 自制 profiler：ncu 报告里最核心的三个数字，零特权自己算
# ---------------------------------------------------------------------------
echo "=== [2] 自制 profiler（%smid + %globaltimer）==="
nvcc -O3 -arch="$ARCH" -o "$OUT/mini_profiler" "$LAB/mini_profiler.cu" 2>&1 | tail -3
"$OUT/mini_profiler" 2>&1 | tee "$OUT/20_mini_profiler.txt"
echo

# ---------------------------------------------------------------------------
# 3. nsys：时间轴。它只需要 CUPTI activity 接口，不碰性能计数器，
#    所以在同一台机器上**能跑**——这正好说明两者读的东西不同。
# ---------------------------------------------------------------------------
echo "=== [3] nsys 时间轴 ==="
if [ ! -x "$NSYS" ]; then NSYS="$ROOT/tools/nsight-systems/target-linux-x64/nsys"; fi
"$NSYS" --version 2>&1 | head -1

nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/reduce_ladder.bin" "$LAB/reduce_ladder.cu" 2>&1 | tail -2

"$NSYS" profile --force-overwrite true -o "$OUT/30_reduce" \
        --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        "$OUT/reduce_ladder.bin" > "$OUT/30_nsys_run.log" 2>&1
echo "    -> 30_reduce.nsys-rep  $(du -h "$OUT/30_reduce.nsys-rep" 2>/dev/null | cut -f1)"

# nsys stats 直接出统计表（这一步会把 .nsys-rep 转成 sqlite）
for rpt in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_trace; do
    "$NSYS" stats --report "$rpt" --format csv \
            --output "$OUT/31_${rpt}" "$OUT/30_reduce.nsys-rep" \
            > /dev/null 2>&1
done
ls "$OUT"/31_* 2>/dev/null | while read -r f; do
    echo "    -> $(basename "$f")  ($(wc -l < "$f") 行)"
done

# 直接查 sqlite：nsys 的数据库 schema 本身就值得看一眼
"$NSYS" export --type sqlite --force-overwrite true \
        -o "$OUT/32_reduce.sqlite" "$OUT/30_reduce.nsys-rep" > /dev/null 2>&1
if [ -f "$OUT/32_reduce.sqlite" ]; then
    echo "    -> 32_reduce.sqlite  $(du -h "$OUT/32_reduce.sqlite" | cut -f1)"
    python3 - "$OUT/32_reduce.sqlite" "$OUT" <<'PYEOF'
import sqlite3, sys, pathlib
db, out = sys.argv[1], pathlib.Path(sys.argv[2])
c = sqlite3.connect(db)
tabs = [r[0] for r in c.execute(
    "select name from sqlite_master where type='table' order by name")]
lines = [f"nsys sqlite 里有 {len(tabs)} 张表，和 GPU 执行直接相关的几张：", ""]
for t in tabs:
    if any(k in t.upper() for k in ("KERNEL", "RUNTIME", "MEMCPY", "NVTX")):
        n = c.execute(f"select count(*) from {t}").fetchone()[0]
        if n:
            lines.append(f"  {t:52s} {n:>7d} 行")
lines += ["", "CUPTI_ACTIVITY_KIND_KERNEL 的列（每个 kernel 记录了什么）：", ""]
cols = c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)").fetchall()
for col in cols:
    lines.append(f"  {col[1]:44s} {col[2]}")
lines += ["", "按 kernel 聚合（直接 SQL，不经过任何 GUI）：", ""]
q = """select s.value as name, count(*) n,
              sum(k.end-k.start)/1e6 tot_ms, avg(k.end-k.start)/1e3 avg_us
       from CUPTI_ACTIVITY_KIND_KERNEL k
       join StringIds s on s.id = k.demangledName
       group by s.value order by tot_ms desc"""
try:
    lines.append(f"  {'kernel':40s} {'次数':>5s} {'总ms':>9s} {'均us':>9s}")
    for name, n, tot, avg in c.execute(q):
        lines.append(f"  {str(name)[:40]:40s} {n:>5d} {tot:>9.3f} {avg:>9.1f}")
except Exception as e:
    lines.append(f"  查询失败: {e}")
txt = "\n".join(lines)
(out / "33_sqlite_tour.txt").write_text(txt, encoding="utf-8")
print(txt)
PYEOF
fi
echo

# ---------------------------------------------------------------------------
# 4. 静态分析：SASS 指令构成。零运行、零特权，却能回答不少"为什么慢"。
# ---------------------------------------------------------------------------
echo "=== [4] SASS 指令构成（静态，无需运行）==="
# cuobjdump 不在 CUDA wheel 里，但 Triton 自带一份
CUOBJDUMP="$(command -v cuobjdump || ls "$ROOT"/envs/*/lib/python3.*/site-packages/triton/backends/nvidia/bin/cuobjdump 2>/dev/null | head -1)"
echo "    cuobjdump = $CUOBJDUMP"
"$CUOBJDUMP" -sass "$OUT/reduce_ladder.bin" > "$OUT/40_reduce_sass.txt" 2>&1
python3 - "$OUT/40_reduce_sass.txt" "$OUT" <<'PYEOF'
import re, sys, pathlib, collections
src = pathlib.Path(sys.argv[1]).read_text(errors="replace")
out = pathlib.Path(sys.argv[2])
# 按 kernel 切段
blocks = re.split(r"\n\s*Function : ", src)
rows = []
# 注意助记符会带后缀：全局原子加是 REDG.E.ADD.F32...，所以按前缀归类而不是精确匹配。
INTEREST = ("LDG", "STG", "LDS", "STS", "RED", "ATOM", "SHFL", "BAR",
            "FFMA", "HMMA", "NOP")
for b in blocks[1:]:
    name = b.split("\n", 1)[0].strip()
    ops = collections.Counter()
    # 行形如：  /*0010*/   @!P0  IADD3 R2, R2, 0x1, RZ ;
    # 谓词前缀 @P0 / @!P0 可有可无，助记符是第一个全大写 token。
    for m in re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)", b):
        ops[m.group(1).split(".")[0]] += 1
    if not ops:
        continue
    rows.append((name, sum(ops.values()), ops))

def cnt(ops, key):                       # 前缀归类：REDG 算进 RED，LDGSTS 不算进 LDG
    return sum(v for k, v in ops.items()
               if k.startswith(key) and not (key == "LDG" and k.startswith("LDGSTS")))

lines = [f"{'kernel':24s} {'总指令':>6s} " + " ".join(f"{k:>5s}" for k in INTEREST)]
for name, tot, ops in rows:
    lines.append(f"{name[:24]:24s} {tot:>6d} " +
                 " ".join(f"{cnt(ops,k):>5d}" for k in INTEREST))
txt = "\n".join(lines)
(out / "41_sass_opmix.txt").write_text(txt, encoding="utf-8")
print(txt)
print()
print("LDG/STG=全局访存  LDS/STS=共享内存  RED/ATOM=原子  SHFL=warp 洗牌")
print("BAR=__syncthreads  FFMA=标量乘加  HMMA=tensor core")
PYEOF

# ---------------------------------------------------------------------------
# 5. compute-sanitizer：正确性侧的"profiler"。CUDA wheel 里自带，同样零特权。
#    很多"性能问题"其实是越界/竞态，先用它排掉再谈优化。
# ---------------------------------------------------------------------------
echo
echo "=== [5] compute-sanitizer（正确性）==="
cat > "$OUT/_racy.cu" <<'EOF'
// 故意写错：共享内存归约漏掉一次 __syncthreads()
#include <cstdio>
__global__ void racy(const float* in, float* out) {
    __shared__ float s[256];
    s[threadIdx.x] = in[blockIdx.x * 256 + threadIdx.x];
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        // ← 这里少了 __syncthreads()
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
int main() {
    float *a, *b; cudaMalloc(&a, 256*64*4); cudaMalloc(&b, 64*4);
    racy<<<64, 256>>>(a, b); cudaDeviceSynchronize();
    printf("kernel 跑完了，没报任何错\n"); return 0;
}
EOF
nvcc -O3 -arch="$ARCH" -lineinfo -o "$OUT/_racy" "$OUT/_racy.cu" 2>&1 | tail -2
echo "--- 直接跑（静默通过）---"
"$OUT/_racy"
echo "--- compute-sanitizer --tool racecheck ---"
# 注意：CUDA wheel（nvidia-cuda-sanitizer-api 之外）里的那个 compute-sanitizer
# 缺少注入库的目录布局，会报 "terminated before first instrumented API call"。
# 要用 redist 的 cuda_sanitizer_api 包，或系统 CUDA 里那份。
SAN="$(ls "$ROOT"/tools/sanitizer/bin/compute-sanitizer /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | head -1)"
SAN="${SAN:-compute-sanitizer}"
echo "    compute-sanitizer = $SAN"
"$SAN" --tool racecheck --print-limit 6 "$OUT/_racy" \
    > "$OUT/50_racecheck.txt" 2>&1
grep -E "RACECHECK SUMMARY|hazard|Error" "$OUT/50_racecheck.txt" | head -8
echo "    -> 50_racecheck.txt  ($(wc -l < "$OUT/50_racecheck.txt") 行)"

echo
echo "所有产物在 $OUT"
