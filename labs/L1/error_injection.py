#!/usr/bin/env python3
"""L1.4 lab · 错误注入：触发点、发现点与可恢复边界。

L1.4 的正文说「异步错误在同步时才暴露」，但只给结论不给反例。
这个 runner 把四类错误各跑一遍，**每类都在独立子进程里**（因为有些错误会
把 CUDA context 永久毒化，同进程里做不了第二个实验）：

  1. oob --sync     越界写显存，同步时报 cudaErrorIllegalAddress
  2. oob --nosync   同一次越界，不同步：错误推迟到后面某次调用才暴露
  3. badlaunch      block dim 超上限：launch 当场返回错误，context 仍然可用
  4. ptx <version>  让驱动 JIT 指定 .version 的 PTX，测驱动的 PTX 版本天花板
  5. cubin sm_89    加载为别的架构编译的 cubin（本机是 sm_120）
  6. ptxas 版本     用本地 ptxas 编译 .version 9.3 的 PTX（工具链版本不兼容）

每条都记录：退出码、捕获到的错误名、错误出现在哪一步、子进程退出后显存是否回收。

用法：
    python error_injection.py --out errinj.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

MINIMAL_PTX = """.version {ver}
.target sm_90
.address_size 64
.visible .entry noop()
{{
    ret;
}}
"""

TINY_CU = """extern "C" __global__ void noop() {}
"""


def sh(cmd: list[str], timeout: int = 180) -> dict:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(cmd), "returncode": r.returncode,
            "stdout": r.stdout, "stderr": r.stderr[-2000:]}


def gpu_state() -> dict:
    out = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        out["memory"] = r.stdout.strip()
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        out["compute_apps"] = r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def arch_of() -> str:
    import torch
    maj, mnr = torch.cuda.get_device_capability(0)
    return f"sm_{maj}{mnr}"


def build(work: Path, arch: str) -> dict:
    """编译探针、生成 PTX 变体与错误架构 cubin。"""
    arts: dict = {}
    probe = work / "errinj_probe"
    arts["compile_probe"] = sh(["nvcc", "-O2", f"-arch={arch}", "-o", str(probe),
                                str(HERE / "errinj_probe.cu"), "-lcuda", "-lcudart"])
    arts["probe_exists"] = probe.exists()

    tiny = work / "tiny.cu"
    tiny.write_text(TINY_CU)
    # 为另一个架构编译 cubin（本机若是 sm_120，就编 sm_89）
    other = "sm_89" if arch != "sm_89" else "sm_90"
    wrong = work / f"wrong_arch_{other}.cubin"
    arts["compile_wrong_cubin"] = sh(["nvcc", "-cubin", f"-arch={other}",
                                      "-o", str(wrong), str(tiny)])
    arts["wrong_cubin"] = str(wrong)

    # 合法 PTX（给 cubin 对照用）与不同 .version 的 PTX
    ptx_files = {}
    for ver in ("8.0", "9.0", "9.3", "10.0"):
        p = work / f"noop_v{ver}.ptx"
        p.write_text(MINIMAL_PTX.format(ver=ver))
        ptx_files[ver] = str(p)
    arts["ptx_files"] = ptx_files

    # 本机 ptxas 对高版本 PTX 的接受度（工具链侧的同类问题）
    p93 = ptx_files["9.3"]
    arts["ptxas_9_3"] = sh(["ptxas", f"-arch={arch}", str(p93), "-o", str(work / "a.cubin")])
    return arts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None, help="编译与 ptx 产物的目录")
    args = ap.parse_args()

    import torch
    name = torch.cuda.get_device_properties(0).name
    arch = arch_of()
    work = Path(args.work) if args.work else Path(os.environ.get("LEARN_ROOT", ".")) / "work" / "errinj"
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "errinj_probe"

    print(f"=== {name}   本机架构 {arch}   工作目录 {work}")
    arts = build(work, arch)
    print(f"    探针编译退出码 {arts['compile_probe']['returncode']}，"
          f"存在={arts['probe_exists']}")
    if not arts["probe_exists"]:
        print(arts["compile_probe"]["stderr"][-1500:])
        raise SystemExit("探针没编译出来")

    cases: list[tuple[str, list[str]]] = [
        ("oob + sync", [str(probe), "oob", "--sync"]),
        ("oob + nosync", [str(probe), "oob", "--nosync"]),
        ("badlaunch", [str(probe), "badlaunch"]),
        ("cubin 错误架构", [str(probe), "cubin", arts["wrong_cubin"]]),
    ]
    for ver, path in arts["ptx_files"].items():
        cases.append((f"ptx .version {ver}", [str(probe), "ptx", path]))

    before = gpu_state()
    results = []
    print(f"\n    {'用例':>18} {'退出码':>6}  关键输出")
    for label, cmd in cases:
        r = sh(cmd)
        key = [ln for ln in r["stdout"].splitlines()
               if ("error" in ln.lower() or "CUDA_" in ln or "Unexpected" in ln
                  or "error" in r["stderr"].lower())]
        summary = key[-1] if key else (r["stdout"].splitlines() or [""])[-1]
        print(f"    {label:>18} {r['returncode']:>6}  {summary[:110]}")
        results.append({"case": label, **r})
    after = gpu_state()

    print("\n[资源回收] 全部子进程退出后：")
    print(f"    显存: {before.get('memory')}  →  {after.get('memory')}")
    print(f"    计算进程: {before.get('compute_apps')!r}  →  {after.get('compute_apps')!r}")

    # ptxas 的工具链版本检查单独说明
    pa = arts["ptxas_9_3"]
    print(f"\n[工具链] ptxas 编译 .version 9.3 的 PTX：退出码 {pa['returncode']}")
    if pa["returncode"] != 0:
        print("    " + " / ".join(pa["stderr"].strip().splitlines()[-2:]))
    print(f"[架构对照] {arch} 的机器加载 {arts['wrong_cubin']}：见上表「cubin 错误架构」")

    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "gpu": name, "arch": arch, "work_dir": str(work),
        "artifacts": {k: (v if not isinstance(v, dict) or "cmd" not in v
                          else {"cmd": v["cmd"], "returncode": v["returncode"]})
                      for k, v in arts.items()},
        "cases": results,
        "gpu_state_before": before, "gpu_state_after": after,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
