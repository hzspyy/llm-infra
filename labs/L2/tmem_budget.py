#!/usr/bin/env python3
"""2.4-D · 两条路径的累加器预算算术（源码常量 + 小学生算式，不含实测）。

常量来自固定 commit 的 CUTLASS 源码（`include/cute/arch/tmem_capacity_sm100.hpp`
与 `include/cutlass/arch/config.h`）：
  - TMEM：每 SM 128 行 × 512 列 × 32 bit = 256 KB，只存在于 sm_100/101/103/107/110
  - sm_120：`TargetTmemCapacityColumns` 落在 `#else` 分支，值为 0
  - 寄存器档案：64K 个 32 bit 寄存器 = 256 KB/SM，每个线程上限 255 个
  - tcgen05 的 M 维映射到 TMEM 的 128 行，因此 tile 的 N 维决定占用多少列

输出是一张"同一块 tile 在两条路径上各占什么资源"的表。表格里的每个数字都是
由上面的常量直接算出来的，不是测量值；实测见 persistent_gemm 与
run_dsl_sm120_gemm 的工件。

    python labs/L2/tmem_budget.py [--out out/2.4/tmem_budget.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

TMEM_ROWS = 128
TMEM_COLUMNS = 512
TMEM_BYTES = TMEM_ROWS * TMEM_COLUMNS * 4          # 262144 B
REGFILE_REGS = 65536                                # 每 SM
MAX_REGS_PER_THREAD = 255

TILES = [(128, 64), (128, 128), (128, 256), (128, 512), (256, 128)]


def tmem_row(tile_m: int, tile_n: int, acc_bytes: int) -> dict:
    """fp32 累加器：N 列各占 1 列 TMEM（每列 128 行 × 32 bit）。"""
    cols = tile_n * acc_bytes // 4
    return {
        "tile": f"{tile_m}x{tile_n}",
        "acc_bytes": acc_bytes,
        "tmem_columns": cols,
        "pct_of_tmem": round(cols / TMEM_COLUMNS * 100, 1),
        "fits": cols <= TMEM_COLUMNS,
        "ctas_per_sm_by_tmem": TMEM_COLUMNS // cols if cols else 0,
        "note": "M 必须 ≤ 128（tcgen05 的 M 映射到 TMEM 的 128 行）" if tile_m > 128 else "",
    }


def reg_row(tile_m: int, tile_n: int, threads: int) -> dict:
    """寄存器累加器：每线程 tile_m*tile_n/threads 个 fp32 累加器。"""
    acc = tile_m * tile_n // threads
    return {
        "tile": f"{tile_m}x{tile_n}",
        "threads": threads,
        "acc_per_thread": acc,
        "pct_of_255": round(acc / MAX_REGS_PER_THREAD * 100, 1),
        "needs_spill": acc > MAX_REGS_PER_THREAD,
        "blocks_per_sm_by_regs_only": REGFILE_REGS // (acc * threads) if acc else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out: dict = {
        "source_constants": {
            "tmem_rows": TMEM_ROWS,
            "tmem_columns": TMEM_COLUMNS,
            "tmem_bytes_per_sm": TMEM_BYTES,
            "tmem_columns_on_sm120": 0,
            "register_file_regs_per_sm": REGFILE_REGS,
            "max_regs_per_thread": MAX_REGS_PER_THREAD,
            "source_files": [
                "include/cute/arch/tmem_capacity_sm100.hpp",
                "include/cutlass/arch/config.h",
            ],
        },
        "tmem_path_sm100": [tmem_row(m, n, 4) for m, n in TILES],
        "register_path_sm120": [],
    }

    print("TMEM 路径（sm_100/101/103/107/110）：每 SM 128 行 × 512 列 × 32 bit = 256 KB")
    print(f"  {'tile':>10s} {'累加器':>8s} {'占用列':>7s} {'占 TMEM':>8s} {'可放 CTA':>9s}")
    for r in out["tmem_path_sm100"]:
        print(f"  {r['tile']:>10s} {'fp32':>8s} {r['tmem_columns']:>7d} "
              f"{r['pct_of_tmem']:>7.1f}% {r['ctas_per_sm_by_tmem']:>9d}"
              + (f"   {r['note']}" if r["note"] else ""))
    print(f"  128×512 fp32 累加器正好占满 512 列；256 行需要 cta_group::2 跨两个 SM 的 TMEM。")

    print("\n寄存器累加器路径（sm_120）：每 SM 64K 寄存器，每线程上限 255")
    print(f"  {'tile':>10s} {'线程':>5s} {'累加器/线程':>12s} {'占 255':>8s} {'会 spill':>9s} {'按寄存器上限的 block/SM':>12s}")
    for m, n in TILES:
        if m > 128:
            continue
        for threads in (128, 256):
            r = reg_row(m, n, threads)
            if threads == 256:
                out["register_path_sm120"].append(r)
            print(f"  {r['tile']:>10s} {threads:>5d} {r['acc_per_thread']:>12d} "
                  f"{r['pct_of_255']:>7.1f}% {str(r['needs_spill']):>9s} "
                  f"{r['blocks_per_sm_by_regs_only']:>12d}")
    print("\n  注：寄存器还要放 A/B 片段、地址与循环变量，实测占用高于此表；")
    print("      sm_120 实测见 persistent_gemm（寄存器随 tile 增大）与")
    print("      run_dsl_sm120_gemm（128×256 tile 触发 1552 B stack、33% 指令是本地访存）。")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nJSON -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
