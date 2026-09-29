#!/usr/bin/env python3
"""2.4-D 架构门控探针：同一批指令对多个 target 能否通过 ptxas。

目的不是测性能，而是把 sm_100（数据中心 Blackwell）与 sm_120（GeForce
Blackwell）两条 GEMM 路径的分界固定到"哪条指令被哪个 target 接受"这一层，
并保留 ptxas 的原始报错。

用法：
    python labs/L2/arch_gate.py --ptxas <path-to-ptxas> \
        --out-dir results/<machine>/2.4/<run_id>/arch-gate

脚本只调用 ptxas，不需要 GPU。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

# 每个片段是一段完整可编译的 PTX 模块体（不含 .target / .version 行）。
# 这些片段照抄 CUTLASS 在对应路径上真正生成的指令形式，见 labs 说明与正文源码解析。
SNIPPETS: dict[str, str] = {
    # ---- sm_90+/sm_100/sm_120 共有的 TMA 路径 ----
    "tma_bulk_tensor_2d": """
.shared .align 128 .b8 smem_dst[128];
.shared .align 8 .b8 mbar[8];
.visible .entry tma_probe()
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    mov.u32 %r1, smem_dst;
    mov.u32 %r2, mbar;
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        [%r1], [%rd3, {%r3, %r4}], [%r2];
    ret;
}
""",
    # ---- sm_100 家族独有的 tcgen05 路径 ----
    "tcgen05_alloc": """
.shared .align 4 .b8 tmem_addr[4];
.visible .entry tcgen05_alloc_probe()
{
    .reg .b32 %r<4>;
    mov.u32 %r1, tmem_addr;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], 64;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
}
""",
    "tcgen05_mma_cta_group1": """
.visible .entry tcgen05_mma_probe()
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<2>;
    tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, %p1;
    ret;
}
""",
    "tcgen05_mma_cta_group2": """
.visible .entry tcgen05_mma_2cta_probe()
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<2>;
    tcgen05.mma.cta_group::2.kind::f16 [%r1], %rd1, %rd2, %r3, %p1;
    ret;
}
""",
    "tcgen05_ld_32x32b": """
.visible .entry tcgen05_ld_probe()
{
    .reg .b32 %r<40>;
    tcgen05.ld.sync.aligned.32x32b.x1.b32
        {%r0}, [%r1];
    ret;
}
""",
    # ---- sm_120 家族走的是这一条：寄存器累加器的 mma.sync ----
    "mma_sync_f8f6f4_f32": """
.visible .entry mma_f8f6f4_probe()
{
    .reg .b32 %r<12>;
    .reg .f32 %f<8>;
    mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r4, %r5}, {%f4, %f5, %f6, %f7};
    ret;
}
""",
    "mma_sync_bf16_m16n8k16": """
.visible .entry mma_bf16_probe()
{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r4, %r5}, {%f4, %f5, %f6, %f7};
    ret;
}
""",
    "wgmma_m64n16k16": """
.visible .entry wgmma_probe()
{
    .reg .b64 %rd<4>;
    .reg .f32 %f<8>;
    wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16
        {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, %rd0, %rd1, 1, 1, 1, 0, 0;
    ret;
}
""",
}

VERSION_LINE = ".version 8.8\n"
ADDRESS_SIZE = ".address_size 64\n"


def build_module(body: str, target: str) -> str:
    return VERSION_LINE + f".target {target}\n" + ADDRESS_SIZE + body


def first_error(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "error" in s.lower():
            return s
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def run_one(ptxas: str, target: str, name: str, body: str, out_dir: pathlib.Path) -> dict:
    ptx_path = out_dir / f"{name}__{target}.ptx"
    cubin_path = out_dir / f"{name}__{target}.cubin"
    ptx_path.write_text(build_module(body, target))
    proc = subprocess.run(
        [ptxas, f"-arch={target}", "-o", str(cubin_path), str(ptx_path)],
        capture_output=True, text=True,
    )
    return {
        "target": target,
        "ptxas_returncode": proc.returncode,
        "accepted": proc.returncode == 0,
        "stderr_first_error": first_error(proc.stderr),
        "ptx_sha256": __import__("hashlib").sha256(ptx_path.read_bytes()).hexdigest()[:16],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptxas", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default="sm_90a,sm_100a,sm_120,sm_120a")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    result: dict = {"ptxas": args.ptxas, "targets": targets, "instructions": {}}
    ver = subprocess.run([args.ptxas, "--version"], capture_output=True, text=True).stdout
    result["ptxas_version"] = ver.strip().splitlines()[-3:] if ver else []

    names = list(SNIPPETS)
    print(f"{'instruction':28s} " + " ".join(f"{t:>9s}" for t in targets))
    for name in names:
        row = {}
        cells = []
        for target in targets:
            info = run_one(args.ptxas, target, name, SNIPPETS[name], out_dir)
            row[target] = info
            cells.append(f"{'OK' if info['accepted'] else 'REJECT':>9s}")
        result["instructions"][name] = row
        print(f"{name:28s} " + " ".join(cells))

    (out_dir / "arch_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out_dir / 'arch_gate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
