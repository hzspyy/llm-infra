#!/usr/bin/env python3
"""L1.3 lab · 主机、NUMA、PCIe root complex 与 GPU 的对应关系。

问题：一次 H2D 到底跨了几段链路？「GPU 挂在 NUMA node 0」这句话
指的是哪一级总线？只抄 `nvidia-smi topo -m` 是不够的——那张表说的是
GPU 之间的通路，不是 CPU/内存/root complex 的归属。

本脚本只读 sysfs 与 nvidia-smi，不做任何测量，输出一份可复核的拓扑：

  A. CPU：socket 数、每 socket 核数、NUMA 节点及其 CPU 列表、节点距离
  B. 内存：每个节点的容量（来自 /sys/devices/system/node/*/meminfo）
  C. GPU：PCI 地址、numa_node、从设备到 host bridge 的完整 PCIe 路径、
         当前/最大链路代数与宽度
  D. 汇总：哪些 GPU 共用一个 root complex

不依赖 numactl / lspci（crater 上两者都没有装），只用 sysfs。
CUDA 侧还需要 torch 来查 can_device_access_peer，作为可选的补充。

用法：
    python topology_map.py --out topology.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CPU_ROOT = Path("/sys/devices/system/cpu")
NODE_ROOT = Path("/sys/devices/system/node")


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def cpu_topology() -> dict:
    """从 sysfs 读 CPU 侧的 NUMA 结构。不用 lscpu，因为字段名跨版本会变。"""
    out: dict = {}
    model = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    out["model"] = model

    pkgs: dict[str, list[int]] = {}
    for cpu in CPU_ROOT.glob("cpu[0-9]*"):
        try:
            pkg = (cpu / "topology" / "physical_package_id").read_text().strip()
        except OSError:
            continue
        pkgs.setdefault(pkg, []).append(int(cpu.name[3:]))
    out["sockets"] = {f"socket{k}": {"n_cpus": len(v), "first_cpus": sorted(v)[:6]}
                      for k, v in sorted(pkgs.items())}
    out["n_sockets"] = len(pkgs)
    out["n_online_cpus"] = len(list(CPU_ROOT.glob("cpu[0-9]*")))

    nodes: dict[str, dict] = {}
    for node in sorted(NODE_ROOT.glob("node[0-9]*")):
        n = node.name
        try:
            cpus = (node / "cpulist").read_text().strip()
            mem_kb = 0
            for line in (node / "meminfo").read_text().splitlines():
                if line.startswith("Node") and "MemTotal" in line:
                    mem_kb = int(re.search(r"(\d+) kB", line).group(1))
            dist = (node / "distance").read_text().split()
        except OSError:
            continue
        nodes[n] = {"cpulist": cpus, "mem_total_gib": round(mem_kb / 1024 / 1024, 1),
                    "distance": [int(x) for x in dist]}
    out["numa_nodes"] = nodes
    out["n_numa_nodes"] = len(nodes)
    return out


def normalize_bdf(bus: str) -> str:
    """nvidia-smi 报的是 00000000:16:00.0，sysfs 用 0000:16:00.0 的短域名。

    两个坑：直接拼字符串会得到 0000:00000000:16:00.0（路径不存在）；
    而且 nvidia-smi 的十六进制大小写不统一（本机 GPU0/1 小写、GPU2/3/4 大写），
    sysfs 全是小写。两者都处理掉，否则一半的卡查不到拓扑。
    """
    parts = bus.strip().split(":")
    if len(parts) < 2:
        return bus.lower()
    return ("0000:" + ":".join(parts[-2:])).lower()


def pcie_path(bus: str) -> dict:
    """从 GPU 的 PCI 设备一路向上走到 host bridge，记录每一级。

    bus 形如 0000:41:00.0。注意 /sys/bus/pci/devices/<bdf> 是符号链接，
    必须先 resolve 再向上走，否则 parent 落在 /sys/bus/pci/devices 上，
    链子一步就断（第一次跑就是这样得到空链的）。
    """
    dev = Path(f"/sys/bus/pci/devices/{bus}")
    if not dev.exists():
        return {"error": f"{bus} 不在 sysfs 里"}
    try:
        cur = dev.resolve()
    except OSError:
        return {"error": f"{bus} 无法解析"}
    chain = []
    seen = set()
    while True:
        if cur in seen:
            break
        seen.add(cur)
        try:
            cls = (cur / "class").read_text().strip()
            vendor = (cur / "vendor").read_text().strip()
            device = (cur / "device").read_text().strip()
        except OSError:
            break
        entry = {"bdf": cur.name, "class": cls, "vendor": vendor, "device": device,
                 "kind": classify(cls)}
        for f in ("current_link_speed", "max_link_speed", "current_link_width",
                  "max_link_width"):
            p = cur / f
            if p.exists():
                try:
                    entry[f] = p.read_text().strip()
                except OSError:
                    pass
        chain.append(entry)
        parent = cur.parent
        if not (parent / "class").exists():
            break
        cur = parent
    return {"chain": chain, "root_complex": chain[-1]["bdf"] if chain else "?"}


def classify(cls: str) -> str:
    if cls.startswith("0x0300") or cls.startswith("0x0302"):
        return "VGA/3D controller"
    if cls.startswith("0x0604"):
        return "PCIe bridge"
    if cls.startswith("0x0600"):
        return "host bridge (root complex)"
    if cls.startswith("0x0c03"):
        return "USB controller"
    return f"class {cls}"


def gpu_topology() -> list[dict]:
    # 注意：`numa_node` 不是合法的 --query-gpu 字段，混进去会让整条查询失败，
    # 返回空列表（第一次跑就是这么踩的）。NUMA 归属一律从 sysfs 读。
    q = ("index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
         "pcie.link.width.current,pcie.link.width.max")
    out = []
    raw = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    if not raw:
        return out
    for line in raw.strip().splitlines():
        idx, name, bus, gc, gm, wc, wm = [x.strip() for x in line.split(",")]
        bus_full = normalize_bdf(bus)
        entry = {"index": int(idx), "name": name, "pci_bus_id": bus_full,
                 "pcie": {"gen_current": gc, "gen_max": gm,
                          "width_current": wc, "width_max": wm}}
        node_file = Path(f"/sys/bus/pci/devices/{bus_full}/numa_node")
        if node_file.exists():
            entry["numa_node_sysfs"] = node_file.read_text().strip()
        entry["pcie_path"] = pcie_path(bus_full)
        out.append(entry)
    return out


def root_complex_groups(gpus: list[dict]) -> dict:
    groups: dict[str, list[int]] = {}
    for g in gpus:
        rc = g.get("pcie_path", {}).get("root_complex", "?")
        groups.setdefault(rc, []).append(g["index"])
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--topo", action="store_true", help="同时跑 nvidia-smi topo -m")
    args = ap.parse_args()

    res = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "hostname": sh(["hostname"]).strip(),
        "cpu": cpu_topology(),
        "gpu": gpu_topology(),
    }
    res["root_complexes"] = root_complex_groups(res["gpu"])
    if args.topo:
        res["nvidia_smi_topo_m"] = sh(["nvidia-smi", "topo", "-m"])

    c = res["cpu"]
    print(f"=== {res['hostname']}   ({c['model']})")
    print(f"    socket 数 {c['n_sockets']}，在线 CPU {c['n_online_cpus']}，"
          f"NUMA 节点 {c['n_numa_nodes']}")
    for n, v in c["numa_nodes"].items():
        print(f"    {n}: cpu {v['cpulist']}  内存 {v['mem_total_gib']} GiB  "
              f"距离 {v['distance']}")

    print("\n[GPU → root complex]")
    for g in res["gpu"]:
        p = g.get("pcie_path", {})
        chain = p.get("chain", [])
        hops = " ← ".join(x["bdf"] for x in chain)
        print(f"    GPU{g['index']} {g['name']}")
        print(f"        numa_node(sysfs)={g.get('numa_node_sysfs', '?')}"
              f"   PCIe gen {g['pcie']['gen_current']}/{g['pcie']['gen_max']}"
              f"  width x{g['pcie']['width_current']}/x{g['pcie']['width_max']}")
        print(f"        {hops}")
    print("\n[共享 root complex 的 GPU]")
    for rc, idxs in res["root_complexes"].items():
        print(f"    {rc}: GPU {idxs}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
