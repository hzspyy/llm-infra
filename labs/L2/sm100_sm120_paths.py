#!/usr/bin/env python3
"""2.4-D · 把 sm_100 与 sm_120 两条 GEMM 路径的源码事实固定成 JSON。

只在固定 commit 的 CUTLASS 源码树上做只读 grep，不改动源码树，不编译。
每条事实记录 file、行号与原始行文本，外加文件 SHA256，便于正文引用 file:line。

    python labs/L2/sm100_sm120_paths.py \
        --cutlass /scratch/learn/opt/src/cutlass-147295a3 \
        --out-dir results/crater/2.4/<run_id>/source
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

# (文件, 说明, 正则)
QUERIES: list[tuple[str, str, str]] = [
    ("include/cute/arch/config.hpp", "TMA 门控（sm_90/sm_100/sm_120 家族都开）",
     r"CUTE_ARCH_TMA_SM90_ENABLED|CUTE_ARCH_TMA_SM120_ENABLED|CUTE_ARCH_TMA_SM100_ENABLED"),
    ("include/cute/arch/config.hpp", "tcgen05 门控（只对 sm_100/101/103/107/110 家族开）",
     r"CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED|CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED|"
     r"CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED"),
    ("include/cute/arch/config.hpp", "TMEM 门控（sm_120 家族一律不在列表里）",
     r"CUTE_ARCH_TCGEN05_TMEM_ENABLED"),
    ("include/cute/arch/config.hpp", "sm_120 家族自己的 MMA 门控",
     r"CUTE_ARCH_MMA_SM120_ENABLED|CUTE_ARCH_F8F6F4_MMA_ENABLED|CUTE_ARCH_MXF8F6F4_MMA_ENABLED"),
    ("include/cutlass/arch/config.h", "架构宏的判定条件（A/F 后缀来自 __CUDA_ARCH_FEAT_*）",
     r"CUTLASS_ARCH_MMA_SM120_ENABLED|CUTLASS_ARCH_MMA_SM120A_ENABLED|"
     r"CUTLASS_ARCH_MMA_SM100_ENABLED|CUTLASS_ARCH_MMA_SM100A_ENABLED|__CUDA_ARCH_FEAT_SM120_ALL"),
    ("include/cute/arch/tmem_capacity_sm100.hpp", "TMEM 几何与容量：128 行 × 512 列 × 32 bit",
     r"Sm100TmemCapacity|Sm107TmemCapacity|TargetTmemCapacity|__CUDA_ARCH__"),
    ("include/cute/arch/mma_sm120.hpp", "sm_120 的 MMA：mma.sync kind::f8f6f4，累加器在寄存器",
     r"struct SM120_16x8x32_TN|kind::f8f6f4|using CRegisters"),
    ("include/cute/arch/mma_sm120_sparse.hpp", "sm_120 稀疏 MMA：m16n8k64",
     r"struct SM120_SPARSE_16x8x64_TN|kind::f8f6f4\.sp"),
    ("include/cute/arch/mma_sm100_umma.hpp", "sm_100 的 MMA：tcgen05.mma，累加器在 TMEM",
     r"tcgen05\.mma\.cta_group|tcgen05\.commit|tcgen05\.wait|tmem"),
    ("include/cute/arch/tmem_allocator_sm100.hpp", "TMEM 的分配与回收指令",
     r"tcgen05\.alloc|tcgen05\.dealloc|tcgen05\.relinquish"),
    ("include/cute/arch/mma_sm90_gmma.hpp", "wgmma：只有 sm_90a 有",
     r"wgmma\.mma_async"),
    ("include/cute/arch/copy_sm90_tma.hpp", "TMA：cp.async.bulk.tensor",
     r"cp\.async\.bulk\.tensor|CUTE_ARCH_TMA_SM90_ENABLED"),
    ("include/cutlass/gemm/collective/sm120_mma_tma.hpp", "sm_120 的 CUTLASS collective 用哪条 MMA/TMA",
     r"SM120_|SM80_16x8x16|Tma|tma|Pipeline|StageCount"),
    ("include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp", "sm_100 的 persistent tile scheduler",
     r"class |struct |get_tile_idx|persistent"),
    ("include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp", "sm_100 warp-specialized kernel 的 warp 角色",
     r"enum class WarpRole|WarpRole::|kNumComputeWarps|kNumLoadWarps|Mma|Sched"),
    ("include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp", "sm_100 collective 的 producer/consumer 分工",
     r"class CollectiveMma|producer|consumer|PipelineTmaUmma"),
]

FILE_FOR_HASH = [
    "include/cute/arch/config.hpp",
    "include/cutlass/arch/config.h",
    "include/cute/arch/tmem_capacity_sm100.hpp",
    "include/cute/arch/mma_sm120.hpp",
    "include/cute/arch/mma_sm100_umma.hpp",
    "include/cute/arch/tmem_allocator_sm100.hpp",
    "include/cute/arch/copy_sm90_tma.hpp",
    "include/cutlass/gemm/collective/sm120_mma_tma.hpp",
    "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp",
    "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp",
]

MAX_HITS = 40


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutlass", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.cutlass)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"],
                            capture_output=True, text=True).stdout.strip()

    result: dict = {
        "repository": remote,
        "commit": commit,
        "facts": [],
        "file_sha256": {},
    }
    for rel in FILE_FOR_HASH:
        p = root / rel
        result["file_sha256"][rel] = sha256(p) if p.exists() else None

    for rel, desc, pattern in QUERIES:
        p = root / rel
        if not p.exists():
            result["facts"].append({"file": rel, "desc": desc, "error": "文件不存在"})
            continue
        hits = []
        rx = re.compile(pattern)
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append({"line": i, "text": line.strip()[:220]})
        result["facts"].append({
            "file": rel,
            "desc": desc,
            "pattern": pattern,
            "hit_count": len(hits),
            "hits": hits[:MAX_HITS],
        })
        print(f"{rel:70s} {len(hits):4d} hits  ({desc})")

    (out / "sm100_sm120_paths.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\ncommit {commit[:12]} -> {out / 'sm100_sm120_paths.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
