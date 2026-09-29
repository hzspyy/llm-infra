#!/usr/bin/env python3
"""L2.0-B mini 实现：Parameter / buffer / 子模块的注册与递归遍历。

只依赖 torch.Tensor 做数据容器，注册表、遍历顺序、去重、
state_dict 与 to() 的语义全部自己写一遍，用于和 nn.Module 对拍。

    python mini_module.py
"""

import torch


class MiniParameter:
    """参数就是“被登记在册 + 会被 optimizer 更新”的张量。"""

    def __init__(self, data, requires_grad=True):
        if not isinstance(data, torch.Tensor):
            raise TypeError("MiniParameter 只能包 torch.Tensor")
        self.data = data
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"MiniParameter(shape={tuple(self.data.shape)}, dtype={self.data.dtype})"


class MiniModule:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_non_persistent", set())
        object.__setattr__(self, "_modules", {})

    # ---- 注册 ------------------------------------------------------
    def __setattr__(self, name, value):
        if isinstance(value, MiniParameter):
            self.register_parameter(name, value)
        elif isinstance(value, MiniModule):
            self.add_module(name, value)
        elif isinstance(value, torch.Tensor):
            # 普通 tensor 属性：能参与前向，但不进注册表，PyTorch 同样如此。
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def register_parameter(self, name, param):
        if param is None:
            self._parameters.pop(name, None)
            object.__delattr__(self, name) if hasattr(self, name) else None
            return
        if not isinstance(param, MiniParameter):
            raise TypeError(f"{name} 不是 MiniParameter")
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name, tensor, persistent=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 不是 Tensor")
        self._buffers[name] = tensor
        if not persistent:
            self._non_persistent.add(name)
        object.__setattr__(self, name, tensor)

    def add_module(self, name, module):
        if not isinstance(module, MiniModule):
            raise TypeError(f"{name} 不是 MiniModule")
        self._modules[name] = module
        object.__setattr__(self, name, module)

    # ---- 递归遍历 --------------------------------------------------
    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()

        def walk(mod, pre):
            for name, p in mod._parameters.items():
                if p is None:
                    continue
                if remove_duplicate:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                yield pre + name, p
            if recurse:
                for mname, child in mod._modules.items():
                    yield from walk(child, pre + mname + ".")

        yield from walk(self, prefix)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()

        def walk(mod, pre):
            for name, b in mod._buffers.items():
                if b is None:
                    continue
                if remove_duplicate:
                    if id(b) in seen:
                        continue
                    seen.add(id(b))
                yield pre + name, b
            if recurse:
                for mname, child in mod._modules.items():
                    yield from walk(child, pre + mname + ".")

        yield from walk(self, prefix)

    def parameters(self, **kw):
        return [p for _, p in self.named_parameters(**kw)]

    def buffers(self, **kw):
        return [b for _, b in self.named_buffers(**kw)]

    def modules(self):
        yield self
        for child in self._modules.values():
            yield from child.modules()

    # ---- 序列化 ----------------------------------------------------
    def state_dict(self, prefix=""):
        """先本层参数，再本层 buffer，然后递归子模块 —— 与 PyTorch 同序。"""
        out = {}
        for name, p in self._parameters.items():
            if p is not None:
                out[prefix + name] = p.data
        for name, b in self._buffers.items():
            if b is not None and name not in self._non_persistent:
                out[prefix + name] = b
        for mname, child in self._modules.items():
            out.update(child.state_dict(prefix + mname + "."))
        return out

    def load_state_dict(self, sd, strict=True):
        local = self.state_dict()
        missing = [k for k in local if k not in sd]
        unexpected = [k for k in sd if k not in local]
        for k in local:
            if k in sd:
                local[k].copy_(sd[k])
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing={missing} "
                f"unexpected={unexpected}")
        return missing, unexpected

    # ---- 迁移 ------------------------------------------------------
    def to(self, dtype):
        """参数与 buffer 都迁，普通 tensor 属性不迁 —— 与 nn.Module._apply 一致。"""
        for p in self._parameters.values():
            if p is not None:
                p.data = p.data.to(dtype)
        for name, b in self._buffers.items():
            if b is not None:
                self._buffers[name] = b.to(dtype)
                object.__setattr__(self, name, self._buffers[name])
        for child in self._modules.values():
            child.to(dtype)
        return self


class MiniLinear(MiniModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = MiniParameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = MiniParameter(torch.zeros(out_features))

    def forward(self, x):
        out = x @ self.weight.data.t()
        if "bias" in self._parameters:
            out = out + self.bias.data
        return out


if __name__ == "__main__":
    print("=" * 78)
    print("[mini] 注册、遍历、state_dict、to() 的最小复现")
    print("=" * 78)

    class Net(MiniModule):
        def __init__(self):
            super().__init__()
            self.fc = MiniLinear(4, 3)
            self.extra = MiniParameter(torch.ones(2))   # 与 fc.bias 无关
            self.register_buffer("running", torch.zeros(3))
            self.register_buffer("temp", torch.ones(3), persistent=False)
            self.plain = torch.randn(4)                  # 普通属性

    m = Net()
    print("参数：", [(n, tuple(p.data.shape)) for n, p in m.named_parameters()])
    print("buffer：", [(n, tuple(b.shape)) for n, b in m.named_buffers()])
    print("state_dict 键：", sorted(m.state_dict().keys()))
    print("temp 不在 state_dict（persistent=False）：",
          "temp" not in m.state_dict())
    print("plain 不在任何注册表：",
          "plain" not in m.state_dict() and
          all(n != "plain" for n, _ in m.named_parameters()))

    print("\n[mini] 共享同一个 MiniParameter 时的去重")
    m2 = Net()
    m2.tied = m2.fc.weight
    print("  remove_duplicate=True :",
          [n for n, _ in m2.named_parameters()])
    print("  remove_duplicate=False:",
          [n for n, _ in m2.named_parameters(remove_duplicate=False)])

    print("\n[mini] to(float64) 只迁参数与 buffer")
    m3 = Net()
    m3.to(torch.float64)
    print("  fc.weight dtype:", m3.fc.weight.data.dtype)
    print("  running   dtype:", m3.running.dtype)
    print("  plain     dtype:", m3.plain.dtype, "  <- 没被迁")

    print("\n[mini] load_state_dict 缺键时的 strict 报错")
    target = Net()
    partial = {k: v for k, v in target.state_dict().items() if "extra" not in k}
    try:
        target.load_state_dict(partial)
    except RuntimeError as exc:
        print("  ", exc)
    missing, unexpected = target.load_state_dict(partial, strict=False)
    print("  strict=False 返回 missing=", missing, " unexpected=", unexpected)
