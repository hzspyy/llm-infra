#!/usr/bin/env python3
"""严格恢复：数据游标、随机状态与 optimizer 缺一不可。

三段：
  A 数据游标   12 条带 ID 的样本，map 与 iterable、worker=0/2、预取与未消费队列
  B 严格恢复   训练两步，在第一步后保存，恢复后跑第二步，与不中断的参照逐项比较
  C 反例       依次拿掉 optimizer / scheduler / RNG / 数据游标，看哪一项坏掉

checkpoint 只写在内存字典里，磁盘上的 DCP 契约在 dcp_checkpoint_contract.py。

Usage:
    python labs/L7/resumable_training.py > "$RUN_DIR/resume.txt"
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

N_SAMPLES, BATCH = 12, 2
DIM = 8


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ A 数据
class IdDataset(Dataset):
    """每条样本带一个可追踪的 ID，取到哪条一目了然。"""

    def __init__(self, n: int = N_SAMPLES):
        gen = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, DIM, generator=gen)
        self.y = torch.randn(n, DIM, generator=gen)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"id": i, "x": self.x[i], "y": self.y[i]}


class IdIterable(IterableDataset):
    """iterable 数据集必须自己处理分片，否则每个 worker 都会重放全部样本。"""

    def __init__(self, n: int = N_SAMPLES, shard_by_worker: bool = True):
        self.base = IdDataset(n)
        self.shard_by_worker = shard_by_worker

    def __iter__(self):
        info = get_worker_info()
        indices = range(len(self.base))
        if self.shard_by_worker and info is not None:
            indices = range(info.id, len(self.base), info.num_workers)
        for i in indices:
            yield self.base[i]


def consumed_ids(loader, steps: int | None = None) -> list[list[int]]:
    out = []
    for i, batch in enumerate(loader):
        if steps is not None and i >= steps:
            break
        out.append([int(v) for v in batch["id"]])
    return out


def section_a() -> None:
    head("A 取到了哪些样本：map / iterable × worker 0/2")
    ds = IdDataset()
    for workers in (0, 2):
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=workers)
        print(f"  map 数据集，worker={workers}：{consumed_ids(loader)}")
    print("  map 数据集的顺序由 sampler 决定，worker 只是并行取数，顺序不变。")

    for shard in (True, False):
        loader = DataLoader(IdIterable(shard_by_worker=shard), batch_size=BATCH,
                            num_workers=2)
        ids = consumed_ids(loader)
        flat = [i for b in ids for i in b]
        print(f"\n  iterable，worker=2，自行分片={shard}：{ids}")
        print(f"    共 {len(flat)} 条，去重后 {len(set(flat))} 条"
              + ("" if shard else " ← 每个 worker 都重放了全部样本"))

    print("\n  预取：worker>0 时 DataLoader 会提前把后面的 batch 取出来。")
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2,
                        prefetch_factor=2)
    it = iter(loader)
    first = next(it)
    print(f"    只消费了第 1 个 batch（id={[int(v) for v in first['id']]}），"
          f"但 worker 已经在准备后面的 batch。")
    print("    已读取（consumed）与已提交（acknowledged）因此是两个游标：")
    print("    崩溃时预取队列里的样本并没有参与任何一次更新，恢复点只能用后者。")
    del it, loader

    print("\n  跨 rank 与尾 batch：12 条样本、batch=2、2 个 rank")
    for drop_last in (False, True):
        for rank in (0, 1):
            idx = list(range(rank, N_SAMPLES, 2))          # 交错分配
            batches = [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]
            if drop_last and batches and len(batches[-1]) < BATCH:
                batches = batches[:-1]
            print(f"    drop_last={drop_last} rank{rank}：{batches}")
    print("    每 rank 的 batch 数必须相同，否则某个 rank 会多进入一次 collective。")
    print("    drop_last=False 且样本数不能整除时，需要补齐或让所有 rank 一起丢。")


# ------------------------------------------------------------------ B 严格恢复
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(DIM, 16)
        self.drop = nn.Dropout(0.3)          # 让 RNG 真正参与前向
        self.fc2 = nn.Linear(16, DIM)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def build():
    torch.manual_seed(0)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    return model, opt, sched


def one_step(model, opt, sched, batch) -> dict:
    model.train()
    loss = (model(batch["x"]) - batch["y"]).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()
    sched.step()
    return {"ids": [int(v) for v in batch["id"]],
            "loss": round(float(loss.detach()), 8),
            "grad_norm": round(gnorm, 8), "lr": sched.get_last_lr()[0],
            "w": model.fc1.weight.detach().clone()}


def snapshot(model, opt, sched, cursor: int) -> dict:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "scheduler": copy.deepcopy(sched.state_dict()),
        "rng": torch.get_rng_state().clone(),
        "cursor": cursor,
    }


def restore(snap: dict, drop: str | None = None):
    model, opt, sched = build()
    # 必须深拷贝：Optimizer.load_state_dict 在 dtype/device 相同时会直接持有
    # 传进来的张量，之后的 step 会原地改写它们——也就是改写你的"备份"。
    snap = copy.deepcopy(snap)
    model.load_state_dict(snap["model"])
    if drop != "optimizer":
        opt.load_state_dict(snap["optimizer"])
    if drop != "scheduler":
        sched.load_state_dict(snap["scheduler"])
    if drop != "rng":
        torch.set_rng_state(snap["rng"])
    cursor = 0 if drop == "cursor" else snap["cursor"]
    return model, opt, sched, cursor


def section_b() -> tuple[dict, dict]:
    head("B 严格恢复：保存一步，恢复后继续第二步")
    ds = IdDataset()
    batches = [ds[i * BATCH: i * BATCH + BATCH] for i in range(2)]
    batches = [{"id": torch.tensor([b["id"] for b in
                                    [ds[j] for j in range(i * BATCH, i * BATCH + BATCH)]]),
                "x": torch.stack([ds[j]["x"] for j in range(i * BATCH, i * BATCH + BATCH)]),
                "y": torch.stack([ds[j]["y"] for j in range(i * BATCH, i * BATCH + BATCH)])}
               for i in range(2)]

    torch.manual_seed(1234)
    model, opt, sched = build()
    step1 = one_step(model, opt, sched, batches[0])
    snap = snapshot(model, opt, sched, cursor=BATCH)
    step2 = one_step(model, opt, sched, batches[1])
    print(f"  不中断：step1 {step1['ids']} loss={step1['loss']} lr={step1['lr']}")
    print(f"           step2 {step2['ids']} loss={step2['loss']} "
          f"grad_norm={step2['grad_norm']} lr={step2['lr']}")

    model_r, opt_r, sched_r, cursor = restore(snap)
    resumed = one_step(model_r, opt_r, sched_r, batches[cursor // BATCH])
    diff = float((resumed["w"] - step2["w"]).abs().max())
    print(f"\n  恢复后：step2 {resumed['ids']} loss={resumed['loss']} "
          f"grad_norm={resumed['grad_norm']} lr={resumed['lr']}")
    print(f"  与参照的 fc1.weight 最大差={diff:.3e}；样本 ID 一致="
          f"{resumed['ids'] == step2['ids']}；loss 一致="
          f"{resumed['loss'] == step2['loss']}")
    print("  五项要同时对上：样本 ID、loss、梯度范数、LR、更新后的参数。")
    return snap, step2


# ------------------------------------------------------------------ C 反例
def section_c(snap: dict, reference: dict) -> None:
    head("C 逐项拿掉一个状态")
    ds = IdDataset()
    batches = [{"id": torch.tensor([j for j in range(i * BATCH, i * BATCH + BATCH)]),
                "x": torch.stack([ds[j]["x"] for j in range(i * BATCH, i * BATCH + BATCH)]),
                "y": torch.stack([ds[j]["y"] for j in range(i * BATCH, i * BATCH + BATCH)])}
               for i in range(2)]
    print(f"  {'拿掉的状态':<14}{'样本 ID':<12}{'loss':>12}{'grad_norm':>12}"
          f"{'lr':>10}{'参数最大差':>14}")
    print(f"  {'（完整恢复）':<14}{str(reference['ids']):<12}{reference['loss']:>12.8f}"
          f"{reference['grad_norm']:>12.8f}{reference['lr']:>10.5f}{0.0:>14.3e}")
    for drop in ("optimizer", "scheduler", "rng", "cursor"):
        model, opt, sched, cursor = restore(snap, drop=drop)
        got = one_step(model, opt, sched, batches[cursor // BATCH])
        diff = float((got["w"] - reference["w"]).abs().max())
        print(f"  {drop:<14}{str(got['ids']):<12}{got['loss']:>12.8f}"
              f"{got['grad_norm']:>12.8f}{got['lr']:>10.5f}{diff:>14.3e}")
    print("\n  四种缺失的表现各不相同：")
    print("    optimizer：样本与 loss 都对，但 m/v 归零、LR 也随 param_groups 回到初值")
    print("    rng：dropout 掩码不同，loss 与梯度当场就不一样")
    print("    cursor：重放了已经训练过的样本，loss 看起来更低，其实是重复训练")
    print("    scheduler：这一步看不出来——StepLR 把当前 LR 存在 optimizer 的")
    print("               param_groups 里，恢复 optimizer 就顺带恢复了它。")

    print("\n  换一种调度就不一样了——比较三种 scheduler 丢掉 last_epoch 之后的 LR：")
    makers = {
        "StepLR(gamma=0.5)": lambda o: torch.optim.lr_scheduler.StepLR(o, 1, 0.5),
        "CosineAnnealingLR(T=10)": lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, 10),
        "LambdaLR(warmup 5 步)":
            lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda e: min(1.0, (e + 1) / 5)),
    }
    for name, make in makers.items():
        opt_ref = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-2)
        sch_ref = make(opt_ref)
        for _ in range(4):                     # 先训 4 步，再保存
            opt_ref.param_groups[0]["params"][0].grad = torch.zeros(1)
            opt_ref.step()
            sch_ref.step()
        state = copy.deepcopy(sch_ref.state_dict())
        groups = copy.deepcopy(opt_ref.state_dict()["param_groups"])
        after_ref = []
        for _ in range(3):
            opt_ref.step()
            sch_ref.step()
            after_ref.append(round(sch_ref.get_last_lr()[0], 6))

        opt_bad = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-2)
        sch_bad = make(opt_bad)
        opt_bad.param_groups[0].update({k: v for k, v in groups[0].items()
                                        if k != "params"})     # 只恢复 optimizer
        after_bad = []
        for _ in range(3):
            opt_bad.step()
            sch_bad.step()
            after_bad.append(round(sch_bad.get_last_lr()[0], 6))
        same = after_ref == after_bad
        print(f"    {name:<26}完整恢复 {after_ref}")
        print(f"    {'':<26}缺 scheduler {after_bad}"
              f"{'（相同）' if same else ' ← 立刻错位'}")
    print("    StepLR 的当前 LR 就住在 optimizer 的 param_groups 里，恢复 optimizer 顺带救回它；")
    print("    按 last_epoch 计算的 cosine 与 warmup 没有这个巧合，LR 会跳回曲线的错误位置。")
    print("  只有 loss 一项相同不足以说明恢复正确；样本 ID、LR、梯度和参数要一起对。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU")
    section_a()
    snap, ref = section_b()
    section_c(snap, ref)
