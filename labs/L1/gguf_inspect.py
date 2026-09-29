#!/usr/bin/env python3
"""L1.6 lab · 自己解析 GGUF：文件头、元数据、张量表与量化块账。

为什么不用现成库：这一章要回答的是「格式名不等于执行后端」，
而 GGUF 的 dtype 字段正是「同一个 .gguf 文件里混着四种数值格式」的地方。
自己按规范解析一遍，才能把「Q4_K_M 到底是什么」落到字节上。

GGUF v3 布局（little-endian）：

    "GGUF" | version u32 | tensor_count u64 | kv_count u64
    kv_count × { key:str  value_type:u32  value }
    tensor_count × { name:str  n_dims:u32  dims:u64[n_dims]  type:u32  offset:u64 }
    padding 到 general.alignment（默认 32）
    tensor 数据区（每个张量的 offset 相对数据区起点）

K-quant 的块结构（QK_K = 256）：

    Q4_K   144 B / 256 权重 = 4.50 bpw   (d:f16 + dmin:f16 + scales:12B + qs:128B)
    Q5_K   176 B / 256      = 5.50 bpw
    Q6_K   210 B / 256      = 6.56 bpw
    Q8_0    34 B /  32      = 8.50 bpw

用法：
    python gguf_inspect.py --model Qwen3-1.7B-Q4_K_M.gguf --out gguf_q4km.json
    python gguf_inspect.py --model Qwen3-1.7B-F16.gguf --compare
"""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

# GGML 类型表：名称、块内权重数、块字节数
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}

# 元数据值类型
KV_TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
            7: "bool", 8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64"}
SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
           5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
           12: ("<d", 8)}


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"文件在 {n} 字节处提前结束")
        return b

    def scalar(self, t: int):
        fmt, size = SCALARS[t]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            et = struct.unpack("<I", self.raw(4))[0]
            n = struct.unpack("<Q", self.raw(8))[0]
            if et == 8:
                return [self.string() for _ in range(n)]
            return [self.value(et) for _ in range(n)]
        return self.scalar(t)


def inspect(path: Path) -> dict:
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path} 不是 GGUF（magic={magic!r}）")
        version = struct.unpack("<I", r.raw(4))[0]
        n_tensors = struct.unpack("<Q", r.raw(8))[0]
        n_kv = struct.unpack("<Q", r.raw(8))[0]

        kv = {}
        for _ in range(n_kv):
            key = r.string()
            t = struct.unpack("<I", r.raw(4))[0]
            kv[key] = r.value(t)

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = struct.unpack("<I", r.raw(4))[0]
            dims = list(struct.unpack(f"<{nd}Q", r.raw(8 * nd)))
            ttype = struct.unpack("<I", r.raw(4))[0]
            off = struct.unpack("<Q", r.raw(8))[0]
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
        data_start = fh.tell()

    align = int(kv.get("general.alignment", 32))
    if data_start % align:
        data_start += align - (data_start % align)

    size = path.stat().st_size
    for i, t in enumerate(tensors):
        name, blk, tsize = GGML_TYPES.get(t["type"], (f"type{t['type']}", 1, 1))
        numel = 1
        for d in t["dims"]:
            numel *= d
        t["type_name"] = name
        t["numel"] = numel
        t["blocks"] = numel // blk if blk > 1 else numel
        t["bytes"] = t["blocks"] * tsize
        t["bpw"] = round(tsize * 8 / blk, 4)
        t["data_begin"] = data_start + t["offset"]
        nxt = (data_start + tensors[i + 1]["offset"]) if i + 1 < len(tensors) else size
        t["gap_bytes"] = nxt - (t["data_begin"] + t["bytes"])
        t["sha_head16"] = None

    # 抽查一个张量的头部字节，作为「量化块真的在文件里」的原始证据
    with open(path, "rb") as fh:
        for t in tensors:
            if t["type_name"] in ("Q4_K", "Q6_K"):
                fh.seek(t["data_begin"])
                head = fh.read(16)
                t["sha_head16"] = head.hex()
                break

    hist: dict[str, dict] = {}
    for t in tensors:
        h = hist.setdefault(t["type_name"], {"tensors": 0, "bytes": 0, "weights": 0})
        h["tensors"] += 1
        h["bytes"] += t["bytes"]
        h["weights"] += t["numel"]

    total_weights = sum(t["numel"] for t in tensors)
    total_bytes = sum(t["bytes"] for t in tensors)
    return {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "file": str(path), "size_bytes": size, "version": version,
        "tensor_count": n_tensors, "kv_count": n_kv,
        "alignment": align, "data_start": data_start,
        "metadata": {k: (v if not isinstance(v, (list, bytes)) or len(v) < 12 else
                         f"<{len(v)} items>") for k, v in kv.items()},
        "type_histogram": {k: v for k, v in sorted(hist.items(), key=lambda x: -x[1]["bytes"])},
        "total_weights": total_weights, "total_tensor_bytes": total_bytes,
        "effective_bpw": round(total_bytes * 8 / total_weights, 4),
        "bytes_vs_file": {"tensor_bytes": total_bytes,
                          "file_bytes": size,
                          "overhead_bytes": size - total_bytes},
        "tensors": tensors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=12, help="打印前 N 个张量")
    ap.add_argument("--compare", action="store_true", help="同时对比同目录的 F16 版本")
    args = ap.parse_args()

    p = Path(args.model)
    res = inspect(p)
    print(f"=== {p.name}  {res['size_bytes'] / 2**30:.2f} GiB  GGUF v{res['version']}")
    print(f"    张量 {res['tensor_count']} 个，元数据 {res['kv_count']} 项，"
          f"对齐 {res['alignment']} B，数据区起点 {res['data_start']}")
    arch = res["metadata"].get("general.architecture")
    print(f"    architecture={arch}  "
          f"name={res['metadata'].get('general.name')}")
    ng = res["metadata"].get("general.file_type")
    print(f"    file_type={ng}  context_length="
          f"{res['metadata'].get(f'{arch}.context_length')}  "
          f"block_count={res['metadata'].get(f'{arch}.block_count')}")

    print("\n[类型直方图] 同一个文件里混着几种数值格式")
    print(f"    {'类型':>8} {'张量数':>6} {'字节':>12} {'权重':>12} {'bpw':>6}")
    for name, h in res["type_histogram"].items():
        blk, tsize = next((b, s) for n, b, s in GGML_TYPES.values() if n == name)
        print(f"    {name:>8} {h['tensors']:>6} {h['bytes'] / 2**20:>10.1f} MiB "
              f"{h['weights'] / 1e6:>10.2f} M {tsize * 8 / blk:>6.2f}")
    print(f"    合计 {res['total_tensor_bytes'] / 2**30:.2f} GiB，"
          f"有效 {res['effective_bpw']:.2f} bpw，"
          f"文件额外开销 {res['bytes_vs_file']['overhead_bytes'] / 2**20:.2f} MiB")

    print(f"\n[前 {args.top} 个张量] 名称 / shape / 类型 / 块数 / 字节")
    for t in res["tensors"][:args.top]:
        print(f"    {t['name']:38s} {str(t['dims']):>20s} {t['type_name']:>5} "
              f"{t['blocks']:>9} {t['bytes'] / 2**20:>8.2f} MiB")

    bad = [t for t in res["tensors"] if t["gap_bytes"] not in (0,)
           and t["name"] != res["tensors"][-1]["name"]]
    if bad:
        print(f"\n[布局核对] {len(bad)} 个张量的字节数与下一个张量的偏移对不上"
              f"（首个：{bad[0]['name']} gap={bad[0]['gap_bytes']}）")
    else:
        print("\n[布局核对] 每个张量的块数×块字节数与下一个张量的偏移完全吻合")

    q = next((t for t in res["tensors"] if t["type_name"] in ("Q4_K", "Q6_K")), None)
    if q:
        print(f"\n[量化块原始字节] {q['name']}（{q['type_name']}）数据区前 16 字节："
              f"{q['sha_head16']}")
        print(f"    每 {GGML_TYPES[q['type']][1]} 个权重一个块，块长 "
              f"{GGML_TYPES[q['type']][2]} 字节，共 {q['blocks']} 块")

    if args.compare:
        f16 = p.with_name(p.name.replace("Q4_K_M", "F16"))
        if f16.exists():
            r16 = inspect(f16)
            print(f"\n[与 F16 对照] {f16.name}")
            print(f"    {'':>10} {'文件':>12} {'张量字节':>12} {'bpw':>7}")
            print(f"    {'F16':>10} {r16['size_bytes'] / 2**30:>10.2f} GiB "
                  f"{r16['total_tensor_bytes'] / 2**30:>10.2f} GiB {r16['effective_bpw']:>7.2f}")
            print(f"    {'Q4_K_M':>10} {res['size_bytes'] / 2**30:>10.2f} GiB "
                  f"{res['total_tensor_bytes'] / 2**30:>10.2f} GiB {res['effective_bpw']:>7.2f}")
            ratio = r16["total_tensor_bytes"] / res["total_tensor_bytes"]
            names16 = sorted(t["name"] for t in r16["tensors"])
            namesq = sorted(t["name"] for t in res["tensors"])
            same_names = names16 == namesq
            same_shape = {t["name"]: (t["dims"], t["numel"]) for t in r16["tensors"]} == \
                         {t["name"]: (t["dims"], t["numel"]) for t in res["tensors"]}
            print(f"    体积比 {ratio:.2f}×；张量名集合{'一致' if same_names else '不一致'}"
                  f"，shape/元素数{'一致' if same_shape else '不一致'}")
            res["f16_compare"] = {
                "f16_bytes": r16["size_bytes"], "f16_bpw": r16["effective_bpw"],
                "ratio": round(ratio, 3)}

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
