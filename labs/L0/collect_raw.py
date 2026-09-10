#!/usr/bin/env python3
"""L0 lab · 原始现场采集器。

把「你自己动手时会看到的原样东西」抓下来，不加工、不总结：
config.json 全文、tokenizer 的真实词条、权重清单与形状、GPU 与驱动的原始输出、
Triton 编译产物目录、PTX 与 SASS 原文。

这些东西看起来琐碎，但只有见过真身，后面讲机制时那些名词才落得下来——
你会知道 `num_key_value_heads` 长什么样、词表里为什么有 `Ġ`、
权重文件里的 key 是怎么命名的、safetensors 的 index 是怎么分片的。

用法：
    python collect_raw.py --model /path/to/Qwen3-1.7B --out-dir results/raw/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[命令失败] {cmd}\n{exc}"


def write(out: Path, name: str, text: str) -> None:
    (out / name).write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"  写出 {name}  ({len(text.splitlines())} 行)")


def collect_model(model: Path, out: Path) -> None:
    print("[模型目录]")
    write(out, "10_model_dir_listing.txt",
          sh(f"ls -la {model}") + "\n\n# 各文件大小\n" + sh(f"du -sh {model}/* | sort -h"))

    for fn in ("config.json", "generation_config.json", "tokenizer_config.json",
               "model.safetensors.index.json"):
        p = model / fn
        if p.exists():
            write(out, f"11_{fn}", p.read_text(encoding="utf-8"))

    # 权重清单：名字 + 形状 + dtype。这是「模型」在磁盘上的真身。
    try:
        from safetensors import safe_open
        shards = sorted(model.glob("*.safetensors"))
        lines = [f"# {len(shards)} 个 safetensors 分片", ""]
        total = 0
        for sp in shards:
            lines.append(f"## {sp.name}  ({sp.stat().st_size / 1e9:.2f} GB)")
            with safe_open(sp, framework="pt") as f:
                keys = list(f.keys())
                lines.append(f"   {len(keys)} 个张量")
                for k in keys:
                    sl = f.get_slice(k)
                    shape = sl.get_shape()
                    n = 1
                    for d in shape:
                        n *= d
                    total += n
                    lines.append(f"   {k:<58} {str(shape):<20} {sl.get_dtype()}")
            lines.append("")
        lines.append(f"# 张量元素总数 {total:,}  (= 参数量)")
        write(out, "12_weight_manifest.txt", "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        write(out, "12_weight_manifest.txt", f"[跳过] {exc}")

    # tokenizer：真实词条长什么样
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(str(model))
        vocab = tk.get_vocab()
        inv = {v: k for k, v in vocab.items()}
        lines = [
            f"# tokenizer 类型: {type(tk).__name__}",
            f"# vocab_size = {tk.vocab_size}, len(vocab) = {len(vocab)}",
            f"# 特殊 token: bos={tk.bos_token!r}/{tk.bos_token_id}  "
            f"eos={tk.eos_token!r}/{tk.eos_token_id}  pad={tk.pad_token!r}/{tk.pad_token_id}",
            "",
            "## id 0-40（字节级 BPE 的起始区，通常是单字节）",
        ]
        for i in range(min(41, len(inv))):
            lines.append(f"  {i:>6}  {inv.get(i)!r}")
        lines += ["", "## id 5000-5020（普通词片；Ġ 表示前导空格）"]
        for i in range(5000, min(5021, len(inv))):
            lines.append(f"  {i:>6}  {inv.get(i)!r}")
        lines += ["", "## 词表末尾 20 个（特殊 token / 预留位常在这里）"]
        for i in range(max(0, len(inv) - 20), len(inv)):
            lines.append(f"  {i:>6}  {inv.get(i)!r}")
        lines += ["", "## 一句话被切成什么样"]
        for s in ["Hello world!", "推理引擎的 KV cache 分页管理",
                  "def forward(self, x):\n    return self.w2(F.silu(self.w1(x)))"]:
            ids = tk.encode(s)
            lines.append(f"  输入: {s!r}")
            lines.append(f"  ids : {ids}")
            lines.append(f"  片段: {[tk.decode([i]) for i in ids]}")
            lines.append("")
        lines += ["## chat template 原文（决定 messages 怎么变成一个字符串）", ""]
        lines.append(str(getattr(tk, "chat_template", None)))
        write(out, "13_tokenizer.txt", "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        write(out, "13_tokenizer.txt", f"[跳过] {exc}")


def collect_machine(out: Path) -> None:
    print("[机器]")
    write(out, "20_nvidia_smi.txt", "\n\n".join([
        "$ nvidia-smi", sh("nvidia-smi"),
        "$ nvidia-smi -q | head -80", sh("nvidia-smi -q | head -80"),
        "$ nvidia-smi topo -m", sh("nvidia-smi topo -m"),
    ]))
    write(out, "21_cpu_mem_numa.txt", "\n\n".join([
        "$ lscpu", sh("lscpu"),
        "$ numactl --hardware", sh("numactl --hardware"),
        "$ free -h", sh("free -h"),
    ]))
    write(out, "22_toolchain.txt", "\n\n".join([
        "$ nvcc --version", sh("nvcc --version"),
        "$ ptxas --version", sh("ptxas --version"),
        "$ which nvcc ptxas nvdisasm", sh("which nvcc ptxas nvdisasm"),
        "$ python -c 'import torch; ...'", sh(
            "python -c \"import torch;p=torch.cuda.get_device_properties(0);"
            "print(torch.__version__, torch.version.cuda);print(p)\""),
    ]))


def collect_compile_artifacts(out: Path, triton_cache: Path) -> None:
    print("[编译产物]")
    if not triton_cache.exists():
        write(out, "30_triton_cache.txt", f"[无] {triton_cache} 不存在")
        return
    listing = sh(f"find {triton_cache} -maxdepth 2 -type f | head -60")
    write(out, "30_triton_cache.txt",
          f"# Triton JIT 缓存目录 {triton_cache}\n"
          f"# 每个 kernel 一个哈希目录，里面是完整的下降链路产物\n\n{listing}")

    cubins = sorted(triton_cache.rglob("*.cubin"), key=lambda p: -p.stat().st_size)
    if not cubins:
        return
    picked = cubins[0]
    d = picked.parent
    write(out, "31_one_kernel_dir.txt",
          f"# 挑一个 kernel 看它的全部产物\n\n{sh(f'ls -la {d}')}")

    ptx = next(iter(d.glob("*.ptx")), None)
    if ptx:
        txt = ptx.read_text(encoding="utf-8", errors="replace")
        head = "\n".join(txt.splitlines()[:120])
        write(out, "32_ptx_head.txt",
              f"# {ptx}\n# 共 {len(txt.splitlines())} 行，这里是前 120 行\n"
              f"# PTX 是虚拟 ISA：.version 是 PTX ISA 版本，.target 是目标架构\n\n{head}")

    ttgir = next(iter(d.glob("*.ttgir")), None)
    if ttgir:
        txt = ttgir.read_text(encoding="utf-8", errors="replace")
        write(out, "33_ttgir_head.txt",
              f"# {ttgir}（TritonGPU IR，MLIR 方言，已带上 layout 标注）\n"
              f"# 共 {len(txt.splitlines())} 行，这里是前 60 行\n\n"
              + "\n".join(txt.splitlines()[:60]))

    sass = sh(f"nvdisasm -c {picked} | head -90", timeout=180)
    write(out, "34_sass_head.txt",
          f"# nvdisasm -c {picked.name}\n"
          f"# SASS 是真正在 sm_120 上执行的机器码\n\n{sass}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--triton-cache", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    collect_model(Path(args.model), out)
    collect_machine(out)
    import os
    collect_compile_artifacts(
        out, Path(args.triton_cache or os.environ.get("TRITON_CACHE_DIR",
                                                      Path.home() / ".triton" / "cache")))

    index = sorted(p.name for p in out.iterdir())
    (out / "00_INDEX.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\n完成 -> {out}")


if __name__ == "__main__":
    main()
