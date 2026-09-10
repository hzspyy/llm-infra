#!/usr/bin/env bash
# L2.2 lab · nvcc 的每一跳，全部落到磁盘上看。
#
# 「nvcc 把 .cu 编译成可执行文件」这句话掩盖了 8 个独立的程序在接力。
# 本脚本让每一跳都留下产物，然后逐个打开看。
#
# 用法：bash toolchain_tour.sh [输出目录]
set -u
OUT="${1:-/tmp/toolchain_tour}"
mkdir -p "$OUT"
cd "$OUT"

CUOBJ="$(command -v cuobjdump || find / -name cuobjdump -type f 2>/dev/null | head -1)"
CUFILT="$(command -v cu++filt || find / -name 'cu++filt' -type f 2>/dev/null | head -1)"

cat > demo.cu <<'EOF'
#include <cuda_runtime.h>

// 一个刻意包含多种元素的 kernel：
//   - 模板（会被 name mangling 编码进符号名）
//   - 共享内存（在 PTX 里是 .shared 段）
//   - 循环 + 归约（能看到 SASS 的展开与谓词）
//   - __restrict__（影响 ptxas 的别名分析）
template <int TILE>
__global__ void reduce_tile(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float s[TILE];
    int t = threadIdx.x;
    int i = blockIdx.x * TILE + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = TILE / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

template __global__ void reduce_tile<256>(const float*, float*, int);
EOF

echo "############ 1. nvcc 到底调用了哪些程序（--dryrun）"
nvcc -arch=sm_120 -O3 -c demo.cu -o demo.o --dryrun 2>&1 \
  | grep -oE '#\$ [a-zA-Z0-9_+/.-]*(cicc|ptxas|fatbinary|cudafe\+\+|nvlink|gcc|cc)[a-zA-Z0-9_+/.-]*' \
  | sed 's/#\$ //' | awk '{n=split($0,a,"/"); print "  " NR ". " a[n]}' | head -12
echo "  （完整命令见 $OUT/dryrun.txt）"
nvcc -arch=sm_120 -O3 -c demo.cu -o demo.o --dryrun > dryrun.txt 2>&1

echo
echo "############ 2. 保留所有中间产物（--keep）"
rm -f demo.o
nvcc -arch=sm_120 -O3 -c demo.cu -o demo.o --keep --keep-dir "$OUT" -Xptxas -v 2>&1 \
  | grep -E "registers|smem|spill" | head -4
echo "  产物："
ls -la demo* 2>/dev/null | awk '{printf "    %-42s %8s\n", $9, $5}' | grep -v '^\s*$'

echo
echo "############ 3. PTX（虚拟 ISA）—— 头部与一段循环"
if [ -f demo.ptx ]; then
  head -18 demo.ptx | sed 's/^/    /'
  echo "    ..."
  grep -n "bar.sync\|ld.shared\|st.shared\|add.f32" demo.ptx | head -6 | sed 's/^/    /'
fi

echo
echo "############ 4. SASS（真实机器码）—— 同一段"
if [ -n "$CUOBJ" ] && [ -f demo.o ]; then
  "$CUOBJ" -sass demo.o 2>/dev/null | grep -E "^\s+/\*[0-9a-f]+\*/" | head -18 | sed 's/^/    /'
  echo "    ..."
  echo "    指令统计（前 10 种）："
  "$CUOBJ" -sass demo.o 2>/dev/null | grep -oE "^\s+/\*[0-9a-f]+\*/\s+@?!?P?[0-9]?\s*[A-Z][A-Z0-9_]*" \
    | awk '{print $NF}' | sort | uniq -c | sort -rn | head -10 | sed 's/^/      /'
fi

echo
echo "############ 5. name mangling：符号名里编码了什么"
if [ -n "$CUOBJ" ] && [ -f demo.o ]; then
  SYM=$("$CUOBJ" -symbols demo.o 2>/dev/null | grep -oE "_Z[A-Za-z0-9_]*reduce_tile[A-Za-z0-9_]*" | head -1)
  echo "    mangled:   $SYM"
  if [ -n "$CUFILT" ] && [ -n "$SYM" ]; then
    echo "    demangled: $("$CUFILT" "$SYM" 2>/dev/null)"
  fi
fi

echo
echo "############ 6. fatbin 里装了几份代码（sm_120 / sm_120a / sm_120f / 多架构）"
for ARCH in "sm_120" "sm_120a" "sm_120f" "compute_90,code=sm_90" ; do
  case "$ARCH" in
    compute_*) FLAG="-gencode=arch=${ARCH}" ;;
    *) FLAG="-arch=${ARCH}" ;;
  esac
  if nvcc $FLAG -O3 -c demo.cu -o t.o 2>/dev/null; then
    N=$("$CUOBJ" -lelf t.o 2>/dev/null | grep -c "cubin")
    P=$("$CUOBJ" -lptx t.o 2>/dev/null | grep -c "ptx")
    printf "    %-26s cubin=%-3s ptx=%-3s 大小=%s\n" "$ARCH" "$N" "$P" "$(stat -c%s t.o 2>/dev/null)"
  else
    printf "    %-26s 编译失败\n" "$ARCH"
  fi
done
echo "    多架构 fatbin（同时装 sm_80 / sm_89 / sm_120 + PTX）："
nvcc -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_89,code=sm_89 \
     -gencode=arch=compute_120,code=sm_120 \
     -gencode=arch=compute_120,code=compute_120 \
     -O3 -c demo.cu -o multi.o 2>/dev/null
if [ -f multi.o ]; then
  echo "      大小 $(stat -c%s multi.o) 字节；包含的架构："
  "$CUOBJ" -lelf multi.o 2>/dev/null | grep -oE "sm_[0-9]+[af]?" | sort -u | tr '\n' ' ' | sed 's/^/      /'
  echo
  "$CUOBJ" -lptx multi.o 2>/dev/null | grep -oE "compute_[0-9]+" | sort -u | tr '\n' ' ' | sed 's/^/      PTX: /'
  echo
fi

echo
echo "############ 7. -O 级别对寄存器与指令数的影响"
printf "    %-8s %-12s %-10s %s\n" "-O" "寄存器" "SASS指令" "smem"
for O in 0 1 2 3; do
  nvcc -arch=sm_120 -O$O -Xptxas -O$O -c demo.cu -o o$O.o -Xptxas -v 2>o$O.log
  REG=$(grep -oE "Used [0-9]+ registers" o$O.log | grep -oE "[0-9]+" | head -1)
  SM=$(grep -oE "[0-9]+ bytes smem" o$O.log | grep -oE "[0-9]+" | head -1)
  INS=$("$CUOBJ" -sass o$O.o 2>/dev/null | grep -cE "^\s+/\*[0-9a-f]+\*/")
  printf "    %-8s %-12s %-10s %s\n" "-O$O" "${REG:-?}" "${INS:-?}" "${SM:-?}"
done

echo
echo "############ 8. __launch_bounds__ 强制占用率的代价"
cat > lb.cu <<'EOF'
#include <cuda_runtime.h>
template <int MAXT>
__global__ void __launch_bounds__(MAXT) heavy(float* o, const float* i, int n) {
    float a[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) a[k] = i[(threadIdx.x + k) % n];
    #pragma unroll
    for (int it = 0; it < 8; ++it)
        #pragma unroll
        for (int k = 0; k < 32; ++k) a[k] = fmaf(a[k], 1.0001f, 0.5f);
    float s = 0;
    #pragma unroll
    for (int k = 0; k < 32; ++k) s += a[k];
    o[threadIdx.x] = s;
}
template __global__ void heavy<128>(float*, const float*, int);
template __global__ void heavy<512>(float*, const float*, int);
template __global__ void heavy<1024>(float*, const float*, int);
EOF
nvcc -arch=sm_120 -O3 -c lb.cu -o lb.o -Xptxas -v 2>lb.log
printf "    %-24s %-12s %s\n" "kernel" "寄存器" "spill (stores/loads)"
grep -E "Function properties for|Used [0-9]+ registers|spill" lb.log | paste - - 2>/dev/null | head -8 | sed 's/^/    /'
grep -B1 -A1 "Used .* registers" lb.log | grep -oE "(heavy[^']*|Used [0-9]+ registers|[0-9]+ bytes spill stores)" | paste - - - 2>/dev/null | sed 's/^/    /' | head -6

echo
echo "所有产物在 $OUT"
