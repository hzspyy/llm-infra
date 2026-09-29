#!/usr/bin/env bash
# L1.2: 把候选的 mma PTX 形式逐条送进 ptxas，记录「支持 / 报错原文」。
# 判断某条指令在某个架构上能不能用，只有编译器说了算——版本号和代号都不算数。
#
# 用法: bash labs/L1/mma_isa_probe.sh <output-dir> [arch]
set -uo pipefail
OUT="${1:?output dir}"
ARCH="${2:-sm_120a}"
mkdir -p "$OUT"
SRC="$OUT/_probe.cu"

cat > "$SRC" <<'EOF'
#include <cstdint>
__global__ void k(uint32_t* o) {
  uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,h0=0,h1=0,sa=0,sb=0;
  float d0=0,d1=0,d2=0,d3=0;
#ifdef I_BF16_F32
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_BF16_F16
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.bf16.bf16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};":"+r"(h0),"+r"(h1):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_F16_F16
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};":"+r"(h0),"+r"(h1):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_FP8_F32
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_FP8_F16
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};":"+r"(h0),"+r"(h1):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_FP8_K64
  asm volatile("mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_FP4_BARE
  asm volatile("mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_KIND_F8F6F4
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_KIND_F8F6F4_FP4
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
#ifdef I_MXF8_BLOCK_SCALE
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
#endif
#ifdef I_MXF4_BLOCK_SCALE
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
#endif
#ifdef I_WGMMA
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"l"((unsigned long long)0),"l"((unsigned long long)0));
#endif
#ifdef I_TCGEN05
  asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;"::"r"(a0),"l"((unsigned long long)0),"l"((unsigned long long)0),"r"(b0));
#endif
  o[0] = h0 ^ h1 ^ (uint32_t)d0 ^ a0;
}
EOF

{
echo "# mma 指令族编译探针  arch=$ARCH  $(nvcc --version | tail -1)"
echo "# 判据：ptxas 能不能生成代码。能编译不代表快，但不能编译就是真用不了。"
echo
printf "%-26s %-52s %s\n" "标签" "PTX 形式" "结果"
} > "$OUT/isa_probe.txt"

probe() {
  local tag="$1" desc="$2"
  local msg
  if nvcc -O3 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -D"$tag" -cubin \
       -o /dev/null "$SRC" 2> "$OUT/_err.txt"; then
    msg="支持"
  else
    msg="失败: $(grep -m1 -i error "$OUT/_err.txt" | sed 's/.*error *: *//' | cut -c1-60)"
  fi
  printf "%-26s %-52s %s\n" "$tag" "$desc" "$msg" >> "$OUT/isa_probe.txt"
}

probe I_BF16_F32          "m16n8k16 bf16 -> f32"
probe I_BF16_F16          "m16n8k16 bf16 -> f16"
probe I_F16_F16           "m16n8k16 f16 -> f16"
probe I_FP8_F32           "m16n8k32 e4m3 -> f32"
probe I_FP8_F16           "m16n8k32 e4m3 -> f16"
probe I_FP8_K64           "m16n8k64 e4m3 -> f32"
probe I_FP4_BARE          "m16n8k64 e2m1 -> f32（不带 kind）"
probe I_KIND_F8F6F4       "kind::f8f6f4 m16n8k32 e4m3 -> f32"
probe I_KIND_F8F6F4_FP4   "kind::f8f6f4 m16n8k32 e2m1 -> f32"
probe I_MXF8_BLOCK_SCALE  "kind::mxf8f6f4.block_scale m16n8k32 e4m3"
probe I_MXF4_BLOCK_SCALE  "kind::mxf4.block_scale m16n8k64 e2m1"
probe I_WGMMA             "wgmma.mma_async m64n8k16（Hopper）"
probe I_TCGEN05           "tcgen05.mma（数据中心 Blackwell）"

rm -f "$SRC" "$OUT/_err.txt"
cat "$OUT/isa_probe.txt"
