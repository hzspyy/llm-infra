#!/usr/bin/env python3
"""四种训练循环的状态账：谁被更新、谁只前向、哪些中间结果可以缓存。

四条循环用同一套小模块搭出来，便于逐项对照：
  A 全参微调              encoder + projector + head 全部可训练
  B LoRA                  base 冻结，只训练低秩增量
  C 冻结 encoder + projector   冻结部分可以 no_grad，也可能必须建图
  D teacher / student     teacher 只前向，student 更新

每条循环记录可训练参数、optimizer 状态、为反向保存的字节、以及
"哪一段前向可以缓存、什么条件下缓存失效"。

Usage:
    python labs/L7/loop_lifecycles.py > "$RUN_DIR/lifecycles.txt"
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM, HIDDEN, VOCAB, BS, SEQ = 128, 512, 256, 8, 64
IGNORE = -100


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


class Encoder(nn.Module):
    """代表视觉/音频编码器：输入无梯度需求时它可以整段 no_grad。"""

    def __init__(self, layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(DIM, DIM) for _ in range(layers)])

    def forward(self, x):
        for blk in self.blocks:
            x = torch.tanh(blk(x))
        return x


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, HIDDEN)

    def forward(self, x):
        return F.silu(self.fc(x))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, VOCAB)

    def forward(self, x):
        return self.fc(x)


class LoRALinear(nn.Module):
    """W 冻结，只训练 s·BA。"""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)


class SavedBytes:
    def __init__(self):
        self.bytes = 0
        self.count = 0
        self._seen: set[int] = set()

    def __enter__(self):
        def pack(t):
            if t.data_ptr() not in self._seen:
                self._seen.add(t.data_ptr())
                self.count += 1
                self.bytes += t.numel() * t.element_size()
            return t

        self._h = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._h.__enter__()
        return self

    def __exit__(self, *exc):
        return self._h.__exit__(*exc)


def make_batch(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(BS, SEQ, DIM, generator=gen)
    y = torch.randint(0, VOCAB, (BS, SEQ), generator=gen)
    return x, y


def optimizer_bytes(opt) -> int:
    return sum(t.numel() * t.element_size()
               for st in opt.state.values() for t in st.values()
               if torch.is_tensor(t))


def count(params) -> int:
    return sum(p.numel() for p in params)


def run_loop(name: str, build, step_fn, note: str) -> dict:
    torch.manual_seed(0)
    modules, trainable = build()
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    x, y = make_batch()
    saved = SavedBytes()
    with saved:
        loss = step_fn(modules, x, y)
    loss.backward()
    opt.step()
    total = sum(count(m.parameters()) for m in modules.values())
    rec = {"loop": name, "total_params": total, "trainable": count(trainable),
           "saved_bytes": saved.bytes, "saved_count": saved.count,
           "opt_bytes": optimizer_bytes(opt), "loss": round(float(loss.detach()), 6),
           "note": note}
    print(f"{name:<26}{rec['total_params']:>10,}{rec['trainable']:>12,}"
          f"{rec['trainable'] / rec['total_params'] * 100:>8.1f}%"
          f"{rec['saved_bytes'] / 2 ** 20:>12.2f}{rec['opt_bytes'] / 2 ** 20:>12.2f}"
          f"  {note}")
    return rec


def section_loops() -> list[dict]:
    head("四种循环：可训练范围、保存值与 optimizer 状态")
    print(f"输入 {BS}×{SEQ}×{DIM}，词表 {VOCAB}；保存值与 optimizer 状态单位 MiB")
    print(f"{'循环':<26}{'总参数':>10}{'可训练':>12}{'占比':>9}"
          f"{'保存值':>12}{'optimizer':>12}  说明")

    def build_full():
        mods = {"encoder": Encoder(), "projector": Projector(), "head": Head()}
        return mods, [p for m in mods.values() for p in m.parameters()]

    def step_full(mods, x, y):
        h = mods["head"](mods["projector"](mods["encoder"](x)))
        return F.cross_entropy(h.reshape(-1, VOCAB), y.reshape(-1))

    def build_lora():
        enc, proj, hd = Encoder(), Projector(), Head()
        for m in (enc, proj, hd):
            for p in m.parameters():
                p.requires_grad_(False)
        proj.fc = LoRALinear(proj.fc)
        hd.fc = LoRALinear(hd.fc)
        mods = {"encoder": enc, "projector": proj, "head": hd}
        return mods, [p for m in mods.values() for p in m.parameters() if p.requires_grad]

    def build_frozen_nograd():
        mods = {"encoder": Encoder(), "projector": Projector(), "head": Head()}
        for p in mods["encoder"].parameters():
            p.requires_grad_(False)
        trainable = [p for k in ("projector", "head") for p in mods[k].parameters()]
        return mods, trainable

    def step_frozen_nograd(mods, x, y):
        with torch.no_grad():                       # 冻结段整体不建图
            feat = mods["encoder"](x)
        h = mods["head"](mods["projector"](feat))
        return F.cross_entropy(h.reshape(-1, VOCAB), y.reshape(-1))

    def step_frozen_graph(mods, x, y):
        feat = mods["encoder"](x)                   # 忘了 no_grad：图仍然建起来
        h = mods["head"](mods["projector"](feat))
        return F.cross_entropy(h.reshape(-1, VOCAB), y.reshape(-1))

    def build_kd():
        teacher = nn.Sequential(Encoder(layers=8), Projector(), Head())
        student = {"encoder": Encoder(), "projector": Projector(), "head": Head()}
        for p in teacher.parameters():
            p.requires_grad_(False)
        mods = dict(student)
        mods["teacher"] = teacher
        return mods, [p for k in ("encoder", "projector", "head")
                      for p in mods[k].parameters()]

    def step_kd(mods, x, y):
        with torch.no_grad():
            t_logits = mods["teacher"](x)
        s_logits = mods["head"](mods["projector"](mods["encoder"](x)))
        return F.kl_div(F.log_softmax(s_logits, -1),
                        F.log_softmax(t_logits, -1),
                        log_target=True, reduction="batchmean")

    rows = [
        run_loop("A 全参微调", build_full, step_full, "全部参数进 optimizer"),
        run_loop("B LoRA", build_lora, step_full, "base 冻结，只训练 s·BA"),
        run_loop("C 冻结 encoder(no_grad)", build_frozen_nograd, step_frozen_nograd,
                 "冻结段不建图"),
        run_loop("C' 冻结 encoder(无 no_grad)", build_frozen_nograd, step_frozen_graph,
                 "同样冻结，未写 no_grad"),
        run_loop("D teacher/student", build_kd, step_kd,
                 "teacher 只前向，无梯度无状态"),
    ]
    c, c2 = rows[2], rows[3]
    print(f"\nC 与 C' 的保存值相同（{c['saved_bytes'] / 2 ** 20:.2f} 与 "
          f"{c2['saved_bytes'] / 2 ** 20:.2f} MiB）：")
    print("  输入和 encoder 参数都不要求梯度时，autograd 根本不会为这段前向建图，")
    print("  显式 no_grad 在这个拓扑下省的是图节点与 Python 开销，不是激活内存。")
    print("  一旦上游出现可训练模块，情况立刻不同——见下一段。")
    print("D 的 teacher 参数计入常驻显存，但没有梯度和 optimizer 状态；")
    print("  它的前向时间要计入每步成本，不能只统计 student 的 forward/backward。")
    return rows


def section_frozen_boundary() -> None:
    head("冻结模块什么时候仍然必须建图")
    torch.manual_seed(0)
    enc, proj, hd = Encoder(), Projector(), Head()
    for p in enc.parameters():
        p.requires_grad_(False)
    x, y = make_batch()

    print("情形 1：可训练模块在冻结段之后 —— 冻结段可以 no_grad")
    with SavedBytes() as s1:
        with torch.no_grad():
            feat = enc(x)
        loss = F.cross_entropy(hd(proj(feat)).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    print(f"  保存值 {s1.bytes / 2 ** 20:.2f} MiB，"
          f"encoder 权重梯度={[p.grad for p in enc.parameters()][0]}")

    print("\n情形 2：可训练模块在冻结段之前 —— 冻结段必须建图来传梯度")
    torch.manual_seed(0)
    pre = nn.Linear(DIM, DIM)                       # 可训练的前置适配层
    with SavedBytes() as s2:
        feat = enc(pre(x))                          # 这里不能 no_grad
        loss = F.cross_entropy(hd(proj(feat)).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    print(f"  保存值 {s2.bytes / 2 ** 20:.2f} MiB，"
          f"前置层拿到梯度范数={pre.weight.grad.norm():.6f}，"
          f"encoder 权重梯度={[p.grad for p in enc.parameters()][0]}")
    print("  两种情形的可训练参数数量相同，激活开销相差 "
          f"{(s2.bytes - s1.bytes) / 2 ** 20:.2f} MiB。")
    print("  判据不是这个模块冻结了没有，而是它下游的梯度需不需要穿过它。")


def section_feature_cache() -> None:
    head("特征缓存的键：冻结不是可缓存的充分条件")
    torch.manual_seed(0)
    enc = Encoder()
    for p in enc.parameters():
        p.requires_grad_(False)
    x, _ = make_batch()

    def augment(sample, seed):
        gen = torch.Generator().manual_seed(seed)
        return sample + 0.1 * torch.randn(sample.shape, generator=gen)

    cache: dict[tuple, torch.Tensor] = {}

    def encode(sample_id, aug_seed, encoder_version):
        key = (sample_id, aug_seed, encoder_version)
        if key in cache:
            return cache[key], True
        with torch.no_grad():
            feat = enc(augment(x[sample_id], aug_seed))
        cache[key] = feat
        return feat, False

    f0, hit0 = encode(0, 1, "v1")
    f0b, hit1 = encode(0, 1, "v1")
    f0c, hit2 = encode(0, 2, "v1")
    print(f"  同一 (样本, 增强 seed, encoder 版本)：命中={hit1}，"
          f"逐元素相同={torch.equal(f0, f0b)}")
    print(f"  只换增强 seed：命中={hit2}，与原特征最大差={((f0 - f0c).abs().max()):.6f}")
    print("  如果缓存键里没有增强 seed，第二个 epoch 会拿到第一个 epoch 的增强结果，")
    print("  等于把数据增强悄悄关掉了。")

    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.01)                            # 解冻并更新一次的效果
    f_after, hit3 = encode(0, 1, "v1")
    with torch.no_grad():
        fresh = enc(augment(x[0], 1))
    print(f"\n  encoder 参数变化后仍按老键命中={hit3}，"
          f"缓存值与重算值最大差={((f_after - fresh).abs().max()):.6f}")
    print("  一旦 encoder 会被更新（哪怕只是某个阶段解冻），特征缓存就必须带上参数版本。")
    print("  能缓存的前提是：这段前向对本次训练是确定性的常量。")


def section_denominators() -> None:
    head("同一次更新的三种计数")
    x, y = make_batch()
    labels = y.clone()
    labels[:, :8] = IGNORE
    shifted = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    print(f"  样本数={BS}  输入 token={BS * SEQ}  "
          f"有效 target={int((shifted != IGNORE).sum())}  optimizer 更新=1")
    print("  三者不成比例：padding、prompt mask 和累积都会改变它们之间的关系。")
    print("  报告吞吐时必须说明分母；跳过的 step 不计入更新数，也不该计入吞吐。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU")
    section_loops()
    section_frozen_boundary()
    section_feature_cache()
    section_denominators()
