#!/usr/bin/env python3
"""L2.0-A 张量的身份：对象、存储、数值是三个不同的问题。

覆盖 transpose / slice / expand / reshape / 重叠 view，逐个打印
对象 id、data_ptr、storage 指针、storage_offset、版本计数，
并用一套独立写出来的索引算术对已知索引做逐字节对拍。

不做任何结论，结论留给正文。

用法：
    python tensor_identity.py            # 全跑
    python tensor_identity.py A1 A3      # 只跑指定节
"""

import ctypes
import struct
import sys

import torch

from mini_tensor import MiniTensor, Storage, contiguous_strides

# 用整数张量，字节即数值，省掉浮点表示法带来的歧义。
BASE = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
FMT = {"torch.int32": ("<i", 4), "torch.int64": ("<q", 8),
       "torch.float32": ("<f", 4), "torch.uint8": ("<B", 1)}


def title(s):
    print()
    print("=" * 96)
    print(s)
    print("=" * 96)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 84 - len(s)))


def storage_of(t):
    return t.untyped_storage()


def storage_ptr(t):
    return storage_of(t).data_ptr()


def raw(t):
    st = storage_of(t)
    return ctypes.string_at(st.data_ptr(), st.nbytes())


def ref_elem(idx, stride, offset):
    """独立实现的寻址：偏移加上每个逻辑索引乘对应步长。"""
    acc = offset
    for i, s in zip(idx, stride):
        acc += i * s
    return acc


def value_at_bytes(buf, fmt, elem, itemsize):
    return struct.unpack_from(fmt, buf, elem * itemsize)[0]


def all_indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in all_indices(shape[1:]):
            yield (i,) + rest


# ------------------------------------------------------------------ A1
def section_A1():
    title("[A1] 同一块 storage 上的一组 view：身份字段逐列打印")

    x = BASE
    views = [
        ("x", x),
        ("x.transpose(1, 2)", x.transpose(1, 2)),
        ("x.transpose(0, 2)", x.transpose(0, 2)),
        ("x[1]", x[1]),
        ("x[..., 1:3]", x[..., 1:3]),
        ("x[:, 1]", x[:, 1]),
        ("x[:, 1:3, ::2]", x[:, 1:3, ::2]),
        ("x[0, :, :1].expand(2, 3, 4)", x[0, :, :1].expand(2, 3, 4)),
        ("x.reshape(6, 4)", x.reshape(6, 4)),
        ("x.unfold(2, 3, 1)", x.unfold(2, 3, 1)),
        ("x.as_strided((4, 4), (1, 1))", x.as_strided((4, 4), (1, 1))),
        ("x.flip(2)  <-- 拷贝", x.flip(2)),
        ("x.transpose(1, 2).reshape(-1)  <-- 拷贝", x.transpose(1, 2).reshape(-1)),
        ("x.clone()  <-- 拷贝", x.clone()),
    ]

    print("基础张量 x = arange(24, int32).reshape(2, 3, 4)")
    print(f"x.untyped_storage(): ptr={storage_ptr(x):#x} "
          f"nbytes={storage_of(x).nbytes()} (24 × 4B)")
    print()
    print(f"{'表达式':<34} {'shape':<14} {'stride':<14} {'off':>4} "
          f"{'对象 id':>14} {'data_ptr':>13} {'storage_ptr':>13} {'ver':>4} "
          f"{'同源':^5} {'连续':^5} {'numel':>6} {'storage 元素':>11}")
    print("-" * 172)
    for name, v in views:
        same_store = storage_ptr(v) == storage_ptr(x)
        s_elems = storage_of(v).nbytes() // v.element_size()
        print(f"{name:<34} {str(uniform_shape(v)):<14} {str(tuple(v.stride())):<14} "
              f"{v.storage_offset():>4} {id(v):>14} {v.data_ptr():>#15x} "
              f"{storage_ptr(v):>#15x} {v._version:>4} "
              f"{'是' if same_store else '否':^5} "
              f"{'是' if v.is_contiguous() else '否':^5} "
              f"{v.numel():>6} {s_elems:>11}")
    print()
    print("说明：'同源' 指共享同一块 storage（storage_ptr 相同）；")
    print("'storage 元素' 是整块 storage 的元素数，numel 是逻辑元素数。")
    print("expand / as_strided / unfold 的 storage 一个字节都没多，", end=" ")
    print("numel 却可能是 storage 元素数的数倍 —— 逻辑大小与占用是两回事。")


def uniform_shape(t):
    return tuple(t.shape)


# ------------------------------------------------------------------ A2
def section_A2():
    title("[A2] 对象相同 / 存储共享 / 数值相同：三个独立的问题")

    x = BASE
    a = x.reshape(6, 4)              # 能 view 就 view
    b = x.transpose(1, 2).reshape(-1)            # 不能 view 就拷
    c = x.clone()
    d = x.expand(2, 2, 3, 4) if False else x[0, :, :1].expand(2, 3, 4)
    e = x[0]

    pairs = [
        ("x, x", x, x),
        ("x, x.reshape(6,4)", x, a),
        ("x, x.transpose(1, 2).reshape(-1)", x, b),
        ("x, x.clone()", x, c),
        ("x, x[0,:,:1].expand(2,3,4)", x, d),
        ("x, x[0]", x, e),
    ]
    print(f"{'比较对象':<32} {'同一对象':^8} {'同一 storage':^12} "
          f"{'同 storage 指针':^15} {'同 numel':^8} {'展平后逐元素相等':^16} {'形状相同':^8}")
    print("-" * 110)
    for name, p, q in pairs:
        if p.numel() == q.numel():
            flat_eq = "是" if torch.equal(p.reshape(-1), q.reshape(-1)) else "否"
        else:
            flat_eq = "—"
        print(f"{name:<32} {'是' if p is q else '否':^8} "
              f"{'是' if storage_of(p).data_ptr() == storage_of(q).data_ptr() else '否':^12} "
              f"{'是' if p.data_ptr() == q.data_ptr() else '否':^15} "
              f"{'是' if p.numel() == q.numel() else '否':^8} "
              f"{flat_eq:^16} "
              f"{'是' if uniform_shape(p) == uniform_shape(q) else '否':^8}")
    print()
    print("x.reshape(6,4) 与 x 不是同一对象，但共用 storage，ptr 相同 —— 它是 view。")
    print("x.transpose(1, 2).reshape(-1) 数值上等于 x 的转置展平，storage 完全不同 —— 它偷偷拷了一次。")
    print("数值相同可以由拷贝实现，也可以由 stride=0 实现，两者代价差一个量级。")


# ------------------------------------------------------------------ A3
def section_A3():
    title("[A3] 已知索引逐字节对拍：自己的索引算术 vs 张量真实读到的字节")

    x = BASE
    views = [
        ("x", x),
        ("x.transpose(1, 2)", x.transpose(1, 2)),
        ("x.transpose(0, 2)", x.transpose(0, 2)),
        ("x[1]", x[1]),
        ("x[..., 1:3]", x[..., 1:3]),
        ("x[:, 1:3, ::2]", x[:, 1:3, ::2]),
        ("x[0, :, :1].expand(2, 3, 4)", x[0, :, :1].expand(2, 3, 4)),
        ("x.unfold(2, 3, 1)", x.unfold(2, 3, 1)),
        ("x.as_strided((4, 4), (1, 1))", x.as_strided((4, 4), (1, 1))),
        ("x.flip(2)", x.flip(2)),
    ]

    fmt, itemsize = FMT[str(x.dtype)]
    base_ptr = storage_ptr(x)
    sptr_x = base_ptr
    total_checked = 0
    total_mismatch = 0

    for name, v in views:
        buf = raw(v)
        st = storage_of(v)
        off = v.storage_offset()
        stride = tuple(v.stride())
        ptr_delta = v.data_ptr() - st.data_ptr()
        # 指针偏移必须等于 offset × itemsize；view 的 stride 不参与这一步。
        ptr_ok = ptr_delta == off * itemsize
        idxs = list(all_indices(uniform_shape(v)))
        bad = 0
        shown = 0
        print()
        print(f"{name}   shape={uniform_shape(v)} stride={stride} "
              f"offset={off} storage={st.nbytes()}B")
        print(f"  data_ptr - storage_ptr = {ptr_delta} B = offset({off}) × "
              f"itemsize({itemsize}) -> {'一致' if ptr_ok else '不一致'}")
        print(f"  {'逻辑索引':<18} {'算出的元素号':>10} {'字节区间':>12} "
              f"{'原始字节':<14} {'按字节解码':>10} {'张量读数':>10}")
        for idx in idxs:
            elem = ref_elem(idx, stride, off)
            b0 = elem * itemsize
            chunk = buf[b0:b0 + itemsize]
            decoded = value_at_bytes(buf, fmt, elem, itemsize)
            actual = int(v[idx].item())
            total_checked += 1
            if decoded != actual:
                bad += 1
            if shown < 5:
                print(f"  {str(list(idx)):<18} {elem:>10} "
                      f"{b0:>5}..{b0 + itemsize - 1:<6} "
                      f"{' '.join(f'{c:02x}' for c in chunk):<14} "
                      f"{decoded:>10} {actual:>10}")
                shown += 1
        total_mismatch += bad
        print(f"  逐索引核对 {len(idxs)} 个，字节解码与张量读数不一致 {bad} 个")
    print()
    print(f"合计核对 {total_checked} 个逻辑索引，不一致 {total_mismatch} 个。")
    print("重叠 view（as_strided / unfold / expand）里同一个字节被核对多次，这是预期行为。")


# ------------------------------------------------------------------ A4
def section_A4():
    title("[A4] 用 mini_tensor 复核同一组索引：两套独立算术")

    x = BASE
    xs = x.to(torch.float32)
    mini = MiniTensor(Storage(xs.numel() * 4), (2, 3, 4), dtype="f32")
    for idx in all_indices((2, 3, 4)):
        mini[idx] = float(xs[idx].item())

    cases = [
        ("transpose(0,2)", mini.transpose(0, 2), xs.transpose(0, 2)),
        ("select(0,1)", mini.select(0, 1), xs[1]),
        ("narrow(1,1,2)", mini.narrow(1, 1, 2), xs[:, 1:3]),
        ("expand", mini.narrow(0, 0, 1).expand(2, 3, 4), xs[0:1].expand(2, 3, 4)),
        ("as_strided 重叠", mini.as_strided((4, 4), (1, 1)),
         xs.as_strided((4, 4), (1, 1))),
    ]
    print(f"{'用例':<20} {'mini shape':<12} {'mini stride':<12} "
          f"{'元素号全对':^10} {'数值全对':^8} {'mini contig':^11} {'torch contig':^12}")
    print("-" * 92)
    for name, mv, tv in cases:
        same_idx = True
        same_val = True
        for idx in all_indices(tuple(mv.sizes)):
            if mv.elem_index(idx) != ref_elem(idx, tuple(mv.strides), mv.offset):
                same_idx = False
            if mv[idx] != float(tv[idx].item()):
                same_val = False
        print(f"{name:<20} {str(tuple(mv.sizes)):<12} {str(tuple(mv.strides)):<12} "
              f"{'是' if same_idx else '否':^10} {'是' if same_val else '否':^8} "
              f"{'是' if mv.is_contiguous() else '否':^11} "
              f"{'是' if tv.is_contiguous() else '否':^12}")

    sub("mini 的 reshape：连续时走 view，不连续时拷贝")
    for name, mv in [("mini(连续)", mini), ("mini.transpose(0,2)", mini.transpose(0, 2))]:
        r = mv.reshape(4, 6) if len(mv.sizes) == 2 else mv.reshape(6, 4)
        print(f"  {name:<22} reshape 后 storage 相同: "
              f"{'是' if r.storage is mv.storage else '否'}  "
              f"stride={tuple(r.strides)}")

    sub("mini 的 contiguous 与 torch 对齐")
    for name, mv, tv in cases:
        mc, tc = mv.contiguous(), tv.contiguous()
        ok = all(mc[idx] == float(tc[idx].item())
                 for idx in all_indices(tuple(mc.sizes)))
        print(f"  {name:<20} mini: 新 storage={'是' if mc.storage is not mv.storage else '否'} "
              f"contig={mc.is_contiguous()}   数值与 torch.contiguous() 全对: "
              f"{'是' if ok else '否'}")


# ------------------------------------------------------------------ A5
def section_A5():
    title("[A5] 版本计数：view 共享它，拷贝不共享")

    x = torch.zeros(6, dtype=torch.float32)
    v = x.view(2, 3)
    c = x.clone()
    print(f"初始          x._version={x._version}  view._version={v._version}  "
          f"clone._version={c._version}")
    print(f"x 与 view 同一 storage: {storage_ptr(x) == storage_ptr(v)}；"
          f"x 与 clone 同一 storage: {storage_ptr(x) == storage_ptr(c)}")
    print(f"view 的 _base is x: {v._base is x}")

    v[0, 0] = 1.0
    print(f"改写 view 后  x._version={x._version}  view._version={v._version}  "
          f"clone._version={c._version}   （view 与 base 一起 +1）")

    c[0] = 5.0
    print(f"改写 clone 后 x._version={x._version}  clone._version={c._version}")

    sub("原地修改打断保存值：真实报错原文")
    a = torch.ones(4, requires_grad=True)
    y = a * 2
    z = y * y          # MulBackward 把 y 存下来给反向用
    loss = z.sum()
    print("  构造 y = a * 2; z = y * y; loss = z.sum()")
    print(f"  反向要用到 y，此时 y._version={y._version}，z.grad_fn={z.grad_fn}")
    y.add_(1)          # 改掉反向要用的保存值
    print(f"  原地修改后 y._version={y._version}，再 backward：")
    try:
        loss.backward()
        print("  没有报错（预期会报错）")
    except RuntimeError as exc:
        for line in str(exc).splitlines():
            print("  " + line)
    print(f"  此时 a._version={a._version}  y._version={y._version}  "
          f"a.grad={'None' if a.grad is None else a.grad.tolist()}")

    sub("对照：改一个没有被反向引用的张量不会报错")
    b = torch.ones(4, requires_grad=True)
    c = (b * 3).sum()
    unrelated = torch.zeros(4)
    unrelated.add_(7)
    c.backward()
    print(f"  unrelated._version={unrelated._version}  b.grad={b.grad.tolist()}  无报错")

    sub("reshape 的结果既不保证同对象，也不保证是 view")
    t = BASE
    r1 = t.reshape(6, 4)
    r2 = t.transpose(1, 2).reshape(-1)
    print(f"  t.reshape(6,4)        同一对象: {r1 is t}  "
          f"同 storage: {storage_ptr(r1) == storage_ptr(t)}")
    print(f"  t.transpose(1, 2).reshape(-1)     同 storage: {storage_ptr(r2) == storage_ptr(t)}  "
          f"（转置后不连续，reshape 走了拷贝）")
    before = t[0, 0, 0].item()
    r2[0] = 12345
    print(f"  改写 r2[0] 后 t[0,0,0] 从 {before} 变成 {t[0, 0, 0].item()}  —— "
          f"{'被带着改了' if t[0, 0, 0].item() != before else '没有变，确实是拷贝'}")


SECTIONS = {"A1": section_A1, "A2": section_A2, "A3": section_A3,
            "A4": section_A4, "A5": section_A5}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device=cpu  "
          f"（本机无 GPU，寻址与身份问题与设备无关）")
    for s in want:
        SECTIONS[s]()
