#!/usr/bin/env python3
"""L2.0 自己写一遍 —— 在一个 bytearray 上实现 tensor 的六元组。

纯标准库，没有 numpy 没有 torch。目的是证明 view / transpose / broadcast /
contiguous 这些东西**全都只是 stride 的算术**，一行 memcpy 都不需要。

    python mini_tensor.py
"""

import struct
from itertools import product

DTYPES = {"f32": ("<f", 4), "i32": ("<i", 4), "u8": ("<B", 1)}


class Storage:
    """一段裸字节。它不知道形状，只知道自己有多长。"""

    def __init__(self, nbytes: int, data: bytearray | None = None):
        self.data = data if data is not None else bytearray(nbytes)

    def nbytes(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Storage(nbytes={self.nbytes()}, id={id(self) & 0xFFFFFF:#08x})"


class MiniTensor:
    """六元组：storage · offset · sizes · strides · dtype · (这里省掉 keyset)。

    注意 offset 和 stride 的单位都是**元素**，不是字节 —— PyTorch 也是这样。
    """

    def __init__(self, storage, sizes, strides=None, offset=0, dtype="f32"):
        self.storage = storage
        self.sizes = tuple(sizes)
        self.dtype = dtype
        self.offset = offset
        self.strides = tuple(strides) if strides is not None \
            else contiguous_strides(self.sizes)

    # ---- 基本属性 -------------------------------------------------
    @property
    def itemsize(self):
        return DTYPES[self.dtype][1]

    def numel(self):
        n = 1
        for s in self.sizes:
            n *= s
        return n

    def elem_index(self, idx):
        """六元组的核心算术：逻辑索引 -> storage 里的第几个元素。"""
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = (idx,)
        e = self.elem_index(idx)
        fmt, size = DTYPES[self.dtype]
        return struct.unpack_from(fmt, self.storage.data, e * size)[0]

    def __setitem__(self, idx, val):
        if isinstance(idx, int):
            idx = (idx,)
        e = self.elem_index(idx)
        fmt, size = DTYPES[self.dtype]
        struct.pack_into(fmt, self.storage.data, e * size, val)

    def indices(self):
        return product(*[range(s) for s in self.sizes])

    def tolist(self):
        def build(dim, prefix):
            if dim == len(self.sizes):
                return self[prefix]
            return [build(dim + 1, prefix + (i,)) for i in range(self.sizes[dim])]
        return build(0, ())

    # ---- 全是 view，没有一次拷贝 -----------------------------------
    def t(self):
        assert len(self.sizes) == 2
        return MiniTensor(self.storage, self.sizes[::-1], self.strides[::-1],
                          self.offset, self.dtype)

    def select(self, dim, i):
        """x[i] on dim —— 掉一个维度，offset 前进 i*stride[dim]。"""
        sizes = self.sizes[:dim] + self.sizes[dim + 1:]
        strides = self.strides[:dim] + self.strides[dim + 1:]
        return MiniTensor(self.storage, sizes, strides,
                          self.offset + i * self.strides[dim], self.dtype)

    def narrow(self, dim, start, length):
        sizes = list(self.sizes)
        sizes[dim] = length
        return MiniTensor(self.storage, sizes, self.strides,
                          self.offset + start * self.strides[dim], self.dtype)

    def expand(self, *sizes):
        """把长度为 1 的维度的 stride 改成 0 —— 广播的全部实现。"""
        assert len(sizes) == len(self.sizes)
        strides = []
        for want, have, st in zip(sizes, self.sizes, self.strides):
            if have == want:
                strides.append(st)
            elif have == 1:
                strides.append(0)
            else:
                raise ValueError(f"不能把 {have} 扩到 {want}")
        return MiniTensor(self.storage, sizes, strides, self.offset, self.dtype)

    def as_strided(self, sizes, strides, offset=None):
        return MiniTensor(self.storage, sizes, strides,
                          self.offset if offset is None else offset, self.dtype)

    # ---- contiguous：PyTorch 的定义，逐字照抄 -----------------------
    def is_contiguous(self) -> bool:
        """c10/core/Contiguity.h:15 _compute_contiguous 的 Python 版。"""
        if self.numel() == 0:
            return True
        expected = 1
        for d in range(len(self.sizes) - 1, -1, -1):
            if self.sizes[d] == 1:        # 长度 1 的维度 stride 是什么都无所谓
                continue
            if self.strides[d] != expected:
                return False
            expected *= self.sizes[d]
        return True

    def contiguous(self):
        if self.is_contiguous():
            return self                   # 已经连续 -> 原样返回，零成本
        out = MiniTensor(Storage(self.numel() * self.itemsize),
                         self.sizes, dtype=self.dtype)
        for idx in self.indices():        # 这就是那次"隐藏的拷贝"
            out[idx] = self[idx]
        return out

    # ---- 一个真正会跑的算子 ----------------------------------------
    def add(self, other):
        """广播加法。先 expand 成同形，再逐元素 —— 和 TensorIterator 同构。"""
        shape = broadcast_shape(self.sizes, other.sizes)
        a = self.expand_to(shape)
        b = other.expand_to(shape)
        out = MiniTensor(Storage(numel_of(shape) * self.itemsize), shape,
                         dtype=self.dtype)
        for idx in out.indices():
            out[idx] = a[idx] + b[idx]
        return out

    def expand_to(self, shape):
        pad = len(shape) - len(self.sizes)
        sizes = (1,) * pad + self.sizes
        strides = (0,) * pad + self.strides
        return MiniTensor(self.storage, sizes, strides, self.offset,
                          self.dtype).expand(*shape)

    def __repr__(self):
        return (f"MiniTensor(sizes={self.sizes}, strides={self.strides}, "
                f"offset={self.offset}, contig={self.is_contiguous()}, "
                f"storage={self.storage.nbytes()}B)")


def contiguous_strides(sizes):
    strides, acc = [], 1
    for s in reversed(sizes):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


def numel_of(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def broadcast_shape(a, b):
    """广播在**左边**补 1，不是右边。补错方向 (2,3)+(3,) 就会假报不兼容。"""
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + tuple(a)
    pb = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(pa, pb):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise ValueError(f"形状不兼容: {a} vs {b}")
    return tuple(out)


def arange(n, dtype="f32"):
    st = Storage(n * DTYPES[dtype][1])
    t = MiniTensor(st, (n,), dtype=dtype)
    for i in range(n):
        t[(i,)] = float(i) if dtype == "f32" else i
    return t


def hexdump(storage, per=16):
    b = storage.data
    for i in range(0, len(b), per):
        print(f"  {i:04x}  " + " ".join(f"{c:02x}" for c in b[i:i + per]))


def show(name, t):
    print(f"{name:<26} {t}")


if __name__ == "__main__":
    print("=" * 74)
    print("一块 48 字节的 storage，和五副不同的眼镜")
    print("=" * 74)
    x = arange(12).as_strided((3, 4), (4, 1))
    show("x", x)
    print(x.tolist())
    print("\nstorage 的 48 字节：")
    hexdump(x.storage)

    print("\n" + "-" * 74)
    views = {
        "x.t()": x.t(),
        "x.select(0, 1)  = x[1]": x.select(0, 1),
        "x.select(1, 1)  = x[:,1]": x.select(1, 1),
        "x.narrow(1,2,2) = x[:,2:]": x.narrow(1, 2, 2),
        "x[:, :1].expand(3,4)": x.narrow(1, 0, 1).expand(3, 4),
    }
    for name, v in views.items():
        show(name, v)
        print(f"{'':26} 同一块 storage: {v.storage is x.storage}")

    print("\n" + "-" * 74)
    print("x.t() 每个逻辑索引读哪个字节")
    tv = x.t()
    for idx in tv.indices():
        e = tv.elem_index(idx)
        print(f"  {str(list(idx)):<8} -> 元素 {e:>2} -> 字节 "
              f"{e * 4:>2}..{e * 4 + 3:<2} -> {tv[idx]}")

    print("\n" + "-" * 74)
    print("contiguous：什么时候真的拷贝")
    for name, v in [("x", x), ("x.t()", x.t()),
                    ("x[:, :1].expand(3,4)", x.narrow(1, 0, 1).expand(3, 4))]:
        c = v.contiguous()
        print(f"  {name:<24} contig={v.is_contiguous()!s:<5} "
              f"contiguous() 是同一个对象: {c is v!s:<5} "
              f"新 storage: {'否' if c.storage is v.storage else '是'}")

    print("\n  长度 1 的维度不影响连续性（PyTorch 同样跳过）：")
    weird = x.storage and MiniTensor(x.storage, (3, 1, 4), (4, 999, 1))
    show("  sizes=(3,1,4) strides=(4,999,1)", weird)

    print("\n" + "-" * 74)
    print("广播加法：b 的 stride 变成 0，一个字节都没多分配")
    a = arange(6).as_strided((2, 3), (3, 1))
    b = arange(3)
    print(f"  a = {a.tolist()}   {a}")
    print(f"  b = {b.tolist()}   {b}")
    print(f"  a+b = {a.add(b).tolist()}")
    print(f"  b 参与运算时的样子: {b.expand_to((2, 3))}")

    print("\n" + "-" * 74)
    print("as_strided 造滑动窗口（零拷贝）")
    v = arange(10)
    w = v.as_strided((8, 3), (1, 1))
    show("v.as_strided((8,3),(1,1))", w)
    for row in w.tolist():
        print("   ", row)
    print(f"  逻辑元素 {w.numel()} 个，storage 只有 "
          f"{w.storage.nbytes() // 4} 个 —— 每个平均被读 "
          f"{w.numel() / (w.storage.nbytes() // 4):.1f} 次")
