#!/usr/bin/env python3
"""L2.2-C · 三种 lowering 路径与 PTX/driver 兼容性。

1. Triton：同一个算子沿 ttir → ttgir → llir → ptx → cubin 逐级导出，
   与 CUDA 实现的 ptx/sass 指令数对照；
2. PTX / cubin 加载：合法的、`.version` 超前的、sm_90 的、sm_120a 的四种输入，
   由 driver API 直接给出成功或失败原文；
3. JIT 缓存：PTX-only 模块首次加载（驱动 JIT）与缓存命中的时间差。

    python toolchain_lowering.py --out-dir <dir>
"""

import argparse
import glob
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

OUT = pathlib.Path(os.environ.get("OUT_DIR", "."))


def cuobjdump_path():
    p = shutil.which("cuobjdump")
    if p:
        return p
    root = os.environ.get("LEARN_ROOT", "/scratch/learn")
    hits = sorted(glob.glob(f"{root}/opt/cuobjdump/*/bin/cuobjdump"))
    return hits[0] if hits else "cuobjdump"


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def sha12(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def count_sass_instructions(path):
    code, out, _ = sh(f"{cuobjdump_path()} -sass {path}")
    if code != 0:
        return -1
    return sum(1 for line in out.splitlines() if line.strip().startswith("/*"))


def count_ptx_instructions(path):
    n = 0
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith(".") or s.endswith(":"):
            continue
        n += 1
    return n


# ------------------------------------------------------------------ 1. Triton
TRITON_SRC = '''
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(x_ptr + offs, mask=m, other=0.0)
    b = tl.load(y_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, a * 2.0 + b, mask=m)

def run():
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")
    o = torch.empty_like(x)
    add_kernel[(1024,)](x, y, o, 1 << 20, BLOCK=1024)
    torch.cuda.synchronize()
    return o

if __name__ == "__main__":
    run()
'''


def triton_section():
    print("\n[1] Triton 的 lowering 阶段")
    src = OUT / "triton_probe.py"
    src.write_text(TRITON_SRC)
    dump = OUT / "triton_dump"
    dump.mkdir(exist_ok=True)
    env = dict(os.environ, TRITON_KERNEL_DUMP="1", TRITON_DUMP_DIR=str(dump),
               TRITON_ALWAYS_COMPILE="1")
    code, out, err = sh([sys.executable, str(src)], env=env)
    if code != 0:
        print("  Triton 运行失败：", err.strip().splitlines()[-1] if err else "?")
        return {}
    files = sorted(p for p in dump.rglob("*") if p.is_file())
    stages = {}
    for p in files:
        stages.setdefault(p.suffix, []).append(p)
    rows = []
    for p in files:
        suffix = p.suffix.lstrip(".")
        if suffix in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            rows.append((suffix, str(p.relative_to(dump)), p.stat().st_size, sha12(p)))
    if not rows:            # 有些版本不落盘，退回进程内取 asm
        print("  未发现 dump 文件，尝试从编译缓存里取 asm（见下）")
    order = {"ttir": 0, "ttgir": 1, "llir": 2, "ptx": 3, "cubin": 4}
    rows.sort(key=lambda r: order.get(r[0], 9))
    print(f"  {'阶段':<8}{'文件':<44}{'字节':>10}  sha256(前12)")
    for suffix, name, size, h in rows:
        print(f"  {suffix:<8}{name:<44}{size:>10}  {h}")
    ttir = next((dump / r[1] for r in rows if r[0] == "ttir"), None)
    ttgir = next((dump / r[1] for r in rows if r[0] == "ttgir"), None)
    if ttir:
        print("\n  ttir 前几行（高层，只有 tensor 语义）：")
        for line in ttir.read_text().splitlines()[:6]:
            print("    " + line)
    if ttgir:
        print("  ttgir 里的关键线索（布局/共享内存/线程映射）：")
        for line in ttgir.read_text().splitlines():
            if any(k in line for k in ("#blocked", "shared", "convert_layout", "ttg.")):
                print("    " + line.strip()[:110])
                break
    return {"stages": rows}


# ------------------------------------------------------------------ 2. 加载兼容性
def load_section():
    print("\n[2] PTX / cubin 的加载：合法、`.version` 超前、sm_90、sm_120a")
    cu = OUT / "loadkern.cu"
    cu.write_text('''
#include <cuda_runtime.h>
extern "C" __global__ void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y,
                                        int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + 1.0f;
}
''')
    jobs = [
        ("valid_sm120.ptx", ["-arch=sm_120", "-ptx"]),
        ("compute90.ptx", ["-gencode=arch=compute_90,code=compute_90", "-ptx"]),
        ("valid_sm120.cubin", ["-arch=sm_120", "-cubin"]),
        ("sm120a.cubin", ["-arch=sm_120a", "-cubin"]),
    ]
    for name, flags in jobs:
        sh(["nvcc"] + flags + [str(cu), "-o", str(OUT / name)])
    # 伪造一个"未来 PTX ISA"的模块
    valid = OUT / "valid_sm120.ptx"
    if valid.exists():
        text = valid.read_text()
        lines = []
        for line in text.splitlines():
            if line.startswith(".version"):
                lines.append(".version 10.0")
            else:
                lines.append(line)
        (OUT / "future_version.ptx").write_text("\n".join(lines) + "\n")
        jobs.append(("future_version.ptx", None))

    loader = OUT / "ptx_load"
    code, _, err = sh(["nvcc", "-O2", "-std=c++17", "-arch=sm_120",
                       str(pathlib.Path(__file__).with_name("ptx_load.cu")),
                       "-o", str(loader),
                       "-L/usr/lib/x86_64-linux-gnu", "-l:libcuda.so.1"])
    if code != 0:
        print("  loader 编译失败：", err.strip().splitlines()[-1] if err else "?")
        return {}
    results = {}
    for name, _ in [(j[0], j[1]) for j in jobs]:
        p = OUT / name
        if not p.exists():
            continue
        code, out, _ = sh([str(loader), str(p)])
        line = out.strip().splitlines()[0] if out.strip() else "(无输出)"
        print("  " + line)
        for extra in out.strip().splitlines()[1:]:
            print("        " + extra.strip())
        results[name] = out.strip()
    return results


# ------------------------------------------------------------------ 3. JIT 缓存
def cache_section():
    print("\n[3] JIT 缓存：PTX-only 模块的首次加载 vs 缓存命中")
    ptx = OUT / "valid_sm120.ptx"
    loader = OUT / "ptx_load"
    if not ptx.exists() or not loader.exists():
        print("  缺少 PTX 或 loader，跳过")
        return {}
    cache = OUT / "jitcache"
    cache.mkdir(exist_ok=True)
    runs = {}
    for label, env_extra in [("cold(CUDA_CACHE_DISABLE=1)", {"CUDA_CACHE_DISABLE": "1"}),
                             ("warm(默认缓存)", {"CUDA_CACHE_MAXSIZE": str(1 << 28),
                                              "CUDA_CACHE_PATH": str(cache)})]:
        times = []
        for _ in range(3):
            env = dict(os.environ, **env_extra)
            code, out, _ = sh([str(loader), str(ptx)], env=env)
            for line in out.splitlines():
                if "load=" in line:
                    times.append(float(line.split("load=")[1].split(" ms")[0]))
        runs[label] = times
        print(f"  {label:<28} 3 次 load = " + ", ".join(f"{t:.3f}" for t in times) + " ms")
    return runs


def cuda_compare(tri):
    """把 Triton 的 ptx/sass 与手写 CUDA 的同算子实现并排比。"""
    print("\n[4] 同一算子的 lowering 对照：Triton vs 手写 CUDA（y = 2x + b）")
    cu = OUT / "cmp.cu"
    cu.write_text('''
#include <cuda_runtime.h>
extern "C" __global__ void add2_kernel(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = 2.0f * x[i] + y[i];
}
''')
    sh(["nvcc", "-arch=sm_120", "-O3", "-ptx", str(cu), "-o", str(OUT / "cmp.ptx")])
    sh(["nvcc", "-arch=sm_120", "-O3", "-cubin", str(cu), "-o", str(OUT / "cmp.cubin")])
    tri_ptx = next((OUT / "triton_dump").rglob("*.ptx"), None)
    tri_cubin = next((OUT / "triton_dump").rglob("*.cubin"), None)
    rows = [("手写 CUDA", OUT / "cmp.ptx", OUT / "cmp.cubin"),
            ("Triton", tri_ptx, tri_cubin)]
    print(f"  {'实现':<12}{'ptx 指令行':>12}{'sass 指令':>12}  sha256(cubin,前12)")
    for name, ptx, cubin in rows:
        pn = count_ptx_instructions(ptx) if ptx and pathlib.Path(ptx).exists() else -1
        sn = count_sass_instructions(cubin) if cubin and pathlib.Path(cubin).exists() else -1
        h = sha12(cubin) if cubin and pathlib.Path(cubin).exists() else "-"
        print(f"  {name:<12}{pn:>12}{sn:>12}  {h}")
    print("  说明：两条路径的 tile 选择由各自编译器决定，指令数只能说明 lowering 粒度，")
    print("        不等于性能差异；性能对照见 2.5。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    global OUT
    OUT = pathlib.Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== 2.2-C lowering 与加载兼容性  out={OUT} ===")
    tri = triton_section()
    load_section()
    cache_section()
    cuda_compare(tri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
