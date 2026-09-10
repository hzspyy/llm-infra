#!/usr/bin/env python3
"""L4.0 lab · safetensors 文件到底是什么：逐字节拆开。

「模型权重」这个词太抽象。一个 .safetensors 文件的格式其实简单到可以手写解析器：

    [0..8)      u64 小端       —— JSON 头部的字节长度 N
    [8..8+N)    UTF-8 JSON     —— 每个张量的 dtype / shape / data_offsets
    [8+N..)     裸字节         —— 所有张量的数据，紧挨着排，没有分隔符

就这三段。没有魔数、没有版本号、没有压缩、没有 pickle。
「safe」指的就是这个：解析它**不需要执行任何代码**，
而 PyTorch 的 .bin/.pth 是 pickle，加载等于执行任意代码。

本脚本不调用 safetensors 库去读，而是**自己按字节解析一遍**，
再用官方库读同一个张量做交叉验证。

用法：
    python inspect_safetensors.py --model /path/to/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# safetensors 的 dtype 字符串 → (每元素字节数, 说明)
DTYPES = {
    "F64": (8, "float64"), "F32": (4, "float32"), "F16": (2, "float16"),
    "BF16": (2, "bfloat16"), "F8_E4M3": (1, "float8 e4m3"), "F8_E5M2": (1, "float8 e5m2"),
    "I64": (8, "int64"), "I32": (4, "int32"), "I16": (2, "int16"), "I8": (1, "int8"),
    "U8": (1, "uint8"), "BOOL": (1, "bool"),
}


def hexdump(b: bytes, base: int = 0, width: int = 16, limit: int = 160) -> str:
    out = []
    for i in range(0, min(len(b), limit), width):
        chunk = b[i:i + width]
        hexs = " ".join(f"{c:02x}" for c in chunk)
        text = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        out.append(f"{base + i:08x}  {hexs:<{width * 3}} |{text}|")
    if len(b) > limit:
        out.append(f"... 还有 {len(b) - limit} 字节")
    return "\n".join(out)


def bf16_to_float(raw: bytes) -> float:
    """bf16 就是 fp32 的高 16 位。补上 16 个零位就能当 fp32 解释。

    这是 bf16 相对 fp16 的核心优势：**指数位和 fp32 完全一样（8 位）**，
    所以动态范围相同，转换只是截断尾数，不会溢出。
    fp16 只有 5 位指数，训练时容易上溢/下溢，才需要 loss scaling（L7.1）。
    """
    return struct.unpack("<f", b"\x00\x00" + raw)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="model.layers.0.self_attn.q_norm.weight",
                    help="挑一个小张量完整验证（默认挑 128 个元素的 q_norm）")
    args = ap.parse_args()

    model = Path(args.model)
    shards = sorted(model.glob("*.safetensors"))
    print(f"模型目录: {model}")
    print(f"分片: {[s.name for s in shards]}\n")

    # ---------- 0. 分片索引 ----------
    idx_path = model / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        print("=" * 78)
        print("【0】分片索引 model.safetensors.index.json")
        print("=" * 78)
        print(f"  metadata: {idx.get('metadata')}")
        wm = idx["weight_map"]
        print(f"  weight_map: {len(wm)} 个张量名 → 分片名")
        for k in list(wm)[:3]:
            print(f"    {k:<50} -> {wm[k]}")
        print("    ...")
        print("  作用：告诉加载器每个张量在哪个文件里。"
              "分片本身只是为了绕开单文件大小与下载断点，和模型结构无关。\n")

    shard = shards[0]
    raw = shard.open("rb")

    # ---------- 1. 头部长度 ----------
    print("=" * 78)
    print(f"【1】前 8 字节 = JSON 头部长度（u64 小端）  文件: {shard.name}")
    print("=" * 78)
    head8 = raw.read(8)
    print(hexdump(head8, 0))
    n_header = struct.unpack("<Q", head8)[0]
    print(f"\n  解析: {n_header} 字节的 JSON 头")
    print(f"  手算: " + " + ".join(f"{b}×256^{i}" for i, b in enumerate(head8) if b) +
          f" = {n_header}")

    # ---------- 2. JSON 头 ----------
    header_bytes = raw.read(n_header)
    print("\n" + "=" * 78)
    print("【2】JSON 头的原始字节（前 160 字节）")
    print("=" * 78)
    print(hexdump(header_bytes, 8))

    header = json.loads(header_bytes)
    meta = header.pop("__metadata__", None)
    print(f"\n  __metadata__ = {meta}")
    print(f"  张量条目数 = {len(header)}")
    print("\n  头部里每个张量长这样（原样打印三条）：")
    for k in list(header)[:3]:
        print(f"    {k!r}:")
        print(f"        {json.dumps(header[k], ensure_ascii=False)}")

    data_start = 8 + n_header
    print(f"\n  裸数据区从字节 {data_start} 开始（= 8 + {n_header}）")
    print(f"  文件总大小 {shard.stat().st_size:,} 字节")

    # ---------- 3. 校验：offsets 是否首尾相连、总长是否对得上 ----------
    print("\n" + "=" * 78)
    print("【3】自己验证一遍布局")
    print("=" * 78)
    entries = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    total = 0
    gaps = 0
    for name, info in entries:
        b0, b1 = info["data_offsets"]
        n_elem = 1
        for d in info["shape"]:
            n_elem *= d
        itemsize = DTYPES[info["dtype"]][0]
        expect = n_elem * itemsize
        assert b1 - b0 == expect, f"{name}: 区间 {b1-b0} != {expect}"
        if b0 != total:
            gaps += 1
        total = b1
    print(f"  {len(entries)} 个张量的 data_offsets 全部满足 "
          f"(end - start) == prod(shape) * itemsize  ✓")
    print(f"  区间之间的空隙数: {gaps}（0 表示所有张量紧挨着排，无对齐填充）")
    print(f"  数据区总长 {total:,} 字节，8 + {n_header} + {total:,} = "
          f"{data_start + total:,}，文件实际 {shard.stat().st_size:,}  "
          f"{'✓ 一致' if data_start + total == shard.stat().st_size else '✗ 不一致'}")

    # ---------- 4. 拿一个张量，自己按字节解出来 ----------
    name = args.tensor
    if name not in header:
        name = next(k for k in header if k.endswith("q_norm.weight")) \
            if any(k.endswith("q_norm.weight") for k in header) else entries[1][0]
    info = header[name]
    b0, b1 = info["data_offsets"]
    itemsize = DTYPES[info["dtype"]][0]

    print("\n" + "=" * 78)
    print(f"【4】手动解码一个张量: {name}")
    print("=" * 78)
    print(f"  头部记录: dtype={info['dtype']}({DTYPES[info['dtype']][1]}) "
          f"shape={info['shape']} data_offsets=[{b0}, {b1}]")
    print(f"  绝对文件偏移 = {data_start} + {b0} = {data_start + b0}")
    raw.seek(data_start + b0)
    blob = raw.read(min(64, b1 - b0))
    print(f"\n  该张量的前 {len(blob)} 字节：")
    print(hexdump(blob, data_start + b0))

    if info["dtype"] == "BF16":
        vals = [bf16_to_float(blob[i:i + 2]) for i in range(0, min(16, len(blob)), 2)]
        print("\n  手动把 bf16 解成 float（bf16 = fp32 的高 16 位，低 16 位补零）：")
        print("  注意字节序：文件里是**小端**，所以 `7d 40` 这两个字节代表的 u16 是 0x407d，")
        print("  补上低 16 位零后的 fp32 位模式是 0x407D0000，不是 0x7d400000。")
        for i, v in enumerate(vals[:8]):
            b = blob[i * 2:i * 2 + 2]
            u16 = struct.unpack("<H", b)[0]
            print(f"    元素[{i}]  文件字节 {b[0]:02x} {b[1]:02x} (LE)"
                  f"  →  u16 0x{u16:04X}  →  fp32 0x{u16:04X}0000  →  {v:.6f}")

        # 交叉验证
        try:
            from safetensors import safe_open
            with safe_open(str(shard), framework="pt") as f:
                t = f.get_tensor(name)
            ref = [round(float(x), 6) for x in t.flatten()[:8].float().tolist()]
            mine = [round(v, 6) for v in vals[:8]]
            print(f"\n  官方库读出来: {ref}")
            print(f"  我手算的:     {mine}")
            print(f"  一致? {'✓' if ref == mine else '✗'}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [跳过交叉验证] {exc}")

    # ---------- 5. 为什么这个格式对加载速度重要 ----------
    print("\n" + "=" * 78)
    print("【5】这个格式为什么让加载变快")
    print("=" * 78)
    print("""  1. 张量在文件里是**连续的裸字节**，布局与内存中一致（row-major）。
     所以加载 = mmap 文件 + 按 offset 切片，**零反序列化、零拷贝**。
  2. 头部是纯 JSON，读它不需要读数据区 —— 可以先看 shape/dtype 再决定读不读，
     这就是 `safe_open(...).get_slice(name)` 能只读一个张量的原因。
  3. 没有 pickle，解析不执行任何代码（对照 torch.load 的 RCE 风险）。
  4. 代价：**没有对齐填充**。张量起点可能不是 64/128 字节对齐的，
     某些需要对齐的零拷贝路径（比如 GPUDirect Storage）要额外处理。""")

    raw.close()


if __name__ == "__main__":
    main()
