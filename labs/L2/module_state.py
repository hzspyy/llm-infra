#!/usr/bin/env python3
"""L2.0-B Module 的状态归属：Parameter、普通属性、buffer。

同一批张量放进三种注册方式，比较 parameters / state_dict / to /
optimizer / load_state_dict 的实际行为，再用 mini_module 复现同样的遍历。

只打印事实，结论留给正文。

用法：
    python module_state.py            # 全跑
    python module_state.py B2 B5
"""

import sys
import warnings

import torch
import torch.nn as nn

from mini_module import MiniLinear, MiniModule, MiniParameter

torch.manual_seed(0)


def title(s):
    print()
    print("=" * 96)
    print(s)
    print("=" * 96)


def sub(s):
    print()
    print("--- " + s + " " + "-" * max(0, 84 - len(s)))


# ---------------------------------------------------------------- 模型组
class SharedParam(nn.Module):
    """emb.weight 与 head.weight 是同一个 Parameter 对象。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(6, 4)
        self.head = nn.Linear(4, 6, bias=False)
        self.head.weight = self.emb.weight


class ThreeKinds(nn.Module):
    """同一个值分别以 Parameter / 普通属性 / buffer 三种方式持有。"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.p = nn.Parameter(torch.ones(3))
        self.plain = torch.ones(4)                       # 普通属性
        self.register_buffer("run", torch.zeros(3))      # 持久 buffer
        self.register_buffer("tmp", torch.ones(3), persistent=False)


# ------------------------------------------------------------------ B1
def section_B1():
    title("[B1] 三种归属：注册表里有什么，state_dict 里就有什么")

    m = ThreeKinds()
    print("self._parameters :", list(m._parameters.keys()))
    print("self._buffers    :", list(m._buffers.keys()))
    print("self._modules    :", list(m._modules.keys()))
    print()
    print("named_parameters():", [n for n, _ in m.named_parameters()])
    print("named_buffers()   :", [n for n, _ in m.named_buffers()])
    print("state_dict() 键   :", sorted(m.state_dict().keys()))
    print()
    print("普通属性 plain 与 non-persistent buffer tmp 的位置：")
    print(f"  'plain' in _parameters/_buffers/_modules : "
          f"{'plain' in m._parameters}/{'plain' in m._buffers}/"
          f"{'plain' in m._modules}")
    print(f"  'plain' in state_dict()                  : {'plain' in m.state_dict()}")
    print(f"  'plain' in named_parameters()            : "
          f"{'plain' in [n for n, _ in m.named_parameters()]}")
    print(f"  'tmp'   in state_dict()                  : {'tmp' in m.state_dict()}")
    print(f"  'tmp'   in named_buffers()               : "
          f"{'tmp' in [n for n, _ in m.named_buffers()]}")
    print()
    print("普通 tensor 属性可以参与前向，但不进注册表；")
    print("non-persistent buffer 进 named_buffers、会被迁移，但不进 state_dict。")

    sub("模块直接持有 tensor 时发生什么")
    m.plain2 = torch.ones(3)
    print(f"  m.plain2 = torch.ones(3)  -> _parameters={list(m._parameters.keys())} "
          f"_buffers={list(m._buffers.keys())}")
    try:
        m.tensor_param = nn.Parameter(torch.ones(2))
        print(f"  m.tensor_param = Parameter(...) -> _parameters 增加 "
              f"'tensor_param': {'tensor_param' in m._parameters}")
    except Exception as exc:
        print("  赋值 Parameter 报错：", exc)


# ------------------------------------------------------------------ B2
def section_B2():
    title("[B2] 共享参数：去重、state_dict 的两个键、optimizer 的双倍更新")

    m = SharedParam()
    print("参数对象判断：")
    print(f"  m.head.weight is m.emb.weight : {m.head.weight is m.emb.weight}")
    print(f"  data_ptr 相同                : "
          f"{m.head.weight.data_ptr() == m.emb.weight.data_ptr()}")
    print(f"  named_parameters()                -> {[n for n, _ in m.named_parameters()]}")
    print(f"  named_parameters(remove_duplicate=False) -> "
          f"{[n for n, _ in m.named_parameters(remove_duplicate=False)]}")

    sub("state_dict 里出现两次，是两个对象、同一块存储")
    sd = m.state_dict()
    print(f"  keys          : {sorted(sd.keys())}")
    print(f"  is 同一对象    : {sd['emb.weight'] is sd['head.weight']}")
    print(f"  data_ptr 相同 : {sd['emb.weight'].data_ptr() == sd['head.weight'].data_ptr()}")

    sub("两次 state_dict 之间是否共享")
    sd2 = m.state_dict()
    print(f"  sd['emb.weight'].data_ptr 与 sd2['emb.weight'].data_ptr 相同: "
          f"{sd['emb.weight'].data_ptr() == sd2['emb.weight'].data_ptr()}")

    sub("optimizer 收到重复参数时更新两次")
    tokens = torch.tensor([1, 2, 3])
    ref = SharedParam()
    with torch.no_grad():
        ref.emb.weight.copy_(m.emb.weight)
    loss = ref.head(ref.emb(tokens)).sum()
    loss.backward()
    grad_abs = ref.emb.weight.grad.abs().sum().item()

    dedup = SharedParam()
    with torch.no_grad():
        dedup.emb.weight.copy_(m.emb.weight)
    dedup.zero_grad()
    loss_d = dedup.head(dedup.emb(tokens)).sum()
    loss_d.backward()
    before = dedup.emb.weight.detach().clone()
    opt1 = torch.optim.SGD(dedup.parameters(), lr=0.1)
    opt1.step()
    d1 = (dedup.emb.weight.detach() - before).abs().sum().item()

    nodup = SharedParam()
    with torch.no_grad():
        nodup.emb.weight.copy_(m.emb.weight)
    nodup.zero_grad()
    loss_n = nodup.head(nodup.emb(tokens)).sum()
    loss_n.backward()
    before_n = nodup.emb.weight.detach().clone()
    params = [p for _, p in nodup.named_parameters(remove_duplicate=False)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt2 = torch.optim.SGD(params, lr=0.1)
        opt2.step()
    d2 = (nodup.emb.weight.detach() - before_n).abs().sum().item()

    print(f"  梯度绝对值和 |g|1 = {grad_abs:.6f}，lr = 0.1")
    print(f"  去重后一步的参数变化  = {d1:.6f}   （= lr × |g|1 = {0.1 * grad_abs:.6f}）")
    print(f"  不去重一步的参数变化  = {d2:.6f}   （= 2 × lr × |g|1）")
    print(f"  比值 d2 / d1          = {d2 / d1:.4f}")
    print(f"  param_groups[0]['params'] 长度 = {len(opt2.param_groups[0]['params'])}")
    for w in caught:
        print("  警告原文：", str(w.message).strip())

    sub("load_state_dict 用两个键写同一块存储")
    m3 = SharedParam()
    sd3 = {k: v.clone() for k, v in m3.state_dict().items()}
    sd3["emb.weight"] = torch.full_like(sd3["emb.weight"], 1.0)
    sd3["head.weight"] = torch.full_like(sd3["head.weight"], 2.0)
    m3.load_state_dict(sd3)
    print(f"  sd 里 emb.weight 全 1、head.weight 全 2，加载后 emb.weight 全 2: "
          f"{bool((m3.emb.weight == 2.0).all())}")
    print(f"  共享关系仍然成立: {m3.head.weight is m3.emb.weight}")


# ------------------------------------------------------------------ B3
def section_B3():
    title("[B3] to() 的遍历范围：谁被迁移，谁被漏掉")

    m = ThreeKinds()
    before = {
        "lin.weight(id)": id(m.lin.weight),
        "run(id)": id(m.run),
        "lin.weight dtype": str(m.lin.weight.dtype),
        "plain dtype": str(m.plain.dtype),
        "p dtype": str(m.p.dtype),
    }
    m.to(torch.float64)
    print("to(torch.float64) 之后：")
    print(f"  lin.weight dtype : {m.lin.weight.dtype}   （对象 id 保持: "
          f"{id(m.lin.weight) == before['lin.weight(id)']}）")
    print(f"  p (Parameter)    : {m.p.dtype}")
    print(f"  run (buffer)     : {m.run.dtype}    （对象 id 保持: "
          f"{id(m.run) == before['run(id)']}）")
    print(f"  tmp (non-persist): {m.tmp.dtype}")
    print(f"  plain (普通属性)  : {m.plain.dtype}    <- 没被迁移")
    print(f"  dtype 对照：之前 lin.weight={before['lin.weight dtype']} "
          f"plain={before['plain dtype']} p={before['p dtype']}")

    sub("把同一个模型搬到 meta，普通属性留在原设备")
    m2 = ThreeKinds().to("meta")
    print(f"  lin.weight.device : {m2.lin.weight.device}")
    print(f"  p.device          : {m2.p.device}")
    print(f"  run.device        : {m2.run.device}")
    print(f"  tmp.device        : {m2.tmp.device}")
    print(f"  plain.device      : {m2.plain.device}  dtype={m2.plain.dtype}  "
          f"<- 既没换设备也没换 dtype")
    print(f"  meta 模型的 state_dict 仍可枚举："
          f"{sorted(m2.state_dict().keys())}")

    sub("漏迁的普通属性会在前向里怎样暴露")
    m3 = ThreeKinds().to(torch.float64)
    out = m3.lin(torch.ones(1, 4, dtype=torch.float64)) + m3.plain
    print(f"  float64 输入 + float32 plain -> {out.dtype}  "
          f"（类型提升静默生效，不报错，但每一步都额外付转换成本）")
    m4 = ThreeKinds().to("meta")
    try:
        out = m4.lin(torch.ones(1, 4, device="meta")) + m4.plain
        print("  meta 输入 + cpu plain -> 没有报错")
    except RuntimeError as exc:
        for line in str(exc).splitlines()[:4]:
            print("  RuntimeError: " + line)


# ------------------------------------------------------------------ B4
def section_B4():
    title("[B4] state_dict 的严格性，以及 eval 与 autograd 无关")

    m = SharedParam()
    good = {k: v.clone() for k, v in m.state_dict().items()}

    sub("缺一个键")
    partial = {k: v for k, v in good.items() if k != "head.weight"}
    try:
        m.load_state_dict(partial)
    except RuntimeError as exc:
        for line in str(exc).splitlines():
            print("  " + line)
    missing, unexpected = m.load_state_dict(partial, strict=False)
    print(f"  strict=False 返回 missing={missing} unexpected={unexpected}")

    sub("多一个键")
    extra = dict(good)
    extra["not_a_param"] = torch.zeros(2)
    try:
        m.load_state_dict(extra)
    except RuntimeError as exc:
        for line in str(exc).splitlines():
            print("  " + line)
    missing, unexpected = m.load_state_dict(extra, strict=False)
    print(f"  strict=False 返回 missing={missing} unexpected={unexpected}")

    sub("state_dict 默认 detach：外面拿到的不是参数本身")
    sd = m.state_dict()
    print(f"  sd['emb.weight'].requires_grad = {sd['emb.weight'].requires_grad}")
    sd_vars = m.state_dict(keep_vars=True)
    print(f"  keep_vars=True 时 is 参数对象: "
          f"{sd_vars['emb.weight'] is m.emb.weight}")

    sub("eval() 只改 training 标志，不关 autograd")
    m.eval()
    out = m.emb(torch.tensor([1]))
    print(f"  m.training = {m.training}")
    print(f"  输出 requires_grad = {out.requires_grad}  grad_fn = {type(out.grad_fn).__name__}")
    with torch.no_grad():
        out2 = m.emb(torch.tensor([1]))
    print(f"  no_grad 下 requires_grad = {out2.requires_grad}  grad_fn = {out2.grad_fn}")


# ------------------------------------------------------------------ B5
def section_B5():
    title("[B5] mini_module 与 nn.Module 对拍")

    class MiniThree(MiniModule):
        def __init__(self):
            super().__init__()
            self.lin = MiniLinear(4, 4)
            self.p = MiniParameter(torch.ones(3))
            self.plain = torch.ones(3)
            self.register_buffer("run", torch.zeros(3))
            self.register_buffer("tmp", torch.ones(3), persistent=False)

    class MiniTied(MiniModule):
        def __init__(self):
            super().__init__()
            self.emb = MiniModule()
            self.emb.weight = MiniParameter(torch.randn(6, 4))
            self.head = MiniLinear(4, 6, bias=False)
            self.head.weight = self.emb.weight

    mini3, tv3 = MiniThree(), ThreeKinds()
    mini_t, tv_t = MiniTied(), SharedParam()

    rows = [
        ("named_parameters 名字集合",
         sorted(n for n, _ in mini3.named_parameters()),
         sorted(n for n, _ in tv3.named_parameters())),
        ("named_buffers 名字集合",
         sorted(n for n, _ in mini3.named_buffers()),
         sorted(n for n, _ in tv3.named_buffers())),
        ("state_dict 键集合",
         sorted(mini3.state_dict().keys()), sorted(tv3.state_dict().keys())),
        ("共享参数去重后的名字",
         sorted(n for n, _ in mini_t.named_parameters()),
         sorted(n for n, _ in tv_t.named_parameters())),
        ("不去重时的名字",
         sorted(n for n, _ in mini_t.named_parameters(remove_duplicate=False)),
         sorted(n for n, _ in tv_t.named_parameters(remove_duplicate=False))),
        ("共享参数 state_dict 双键",
         sorted(mini_t.state_dict().keys()),
         sorted(tv_t.state_dict().keys())),
    ]
    print(f"{'比较项':<30} {'一致':^6}  mini / torch")
    print("-" * 96)
    for name, a, b in rows:
        print(f"{name:<30} {'是' if a == b else '否':^6}  {a}")
        if a != b:
            print(f"{'':<30} {'':^6}  torch={b}")

    mini3.to(torch.float64)
    tv3.to(torch.float64)
    moved = [
        ("lin.weight 被迁移", str(mini3.lin.weight.data.dtype), str(tv3.lin.weight.dtype)),
        ("run 被迁移", str(mini3.run.dtype), str(tv3.run.dtype)),
        ("tmp 被迁移", str(mini3.tmp.dtype), str(tv3.tmp.dtype)),
        ("plain 不被迁移", str(mini3.plain.dtype), str(tv3.plain.dtype)),
    ]
    print()
    print(f"{'to(float64) 后的 dtype':<30} {'一致':^6}  mini / torch")
    print("-" * 96)
    for name, a, b in moved:
        print(f"{name:<30} {'是' if a == b else '否':^6}  {a} / {b}")

    mini3._parameters.pop("p")
    sub("mini 的 strict 缺键报错与 torch 对应")
    target_m, target_t = MiniThree(), ThreeKinds()
    sd_m = {k: v for k, v in target_m.state_dict().items() if k != "run"}
    sd_t = {k: v for k, v in target_t.state_dict().items() if k != "run"}
    try:
        target_m.load_state_dict(sd_m)
    except RuntimeError as exc:
        print("  mini :", exc)
    try:
        target_t.load_state_dict(sd_t)
    except RuntimeError as exc:
        print("  torch:", str(exc).replace("\n", " ").strip())


SECTIONS = {"B1": section_B1, "B2": section_B2, "B3": section_B3,
            "B4": section_B4, "B5": section_B5}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  device=cpu  seed=0")
    for s in want:
        SECTIONS[s]()
