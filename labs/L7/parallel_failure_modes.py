#!/usr/bin/env python3
"""分布式训练特有的失败面：分支不一致、冻结分支、collective 次序与并行配置合法性。

两部分：
  · 两 rank（gloo，CPU）复现三类真实失败，全部带超时，不会挂住
      A rank 之间执行不同分支 → DDP 的 reducer 报错
      B teacher 只前向：requires_grad 忘关会报错，绕过 DDP 前向则静默不同步
      C 两 rank 的 collective 次序不一致 → 超时
  · 单进程的并行配置合法性检查：整除条件、每 rank 状态与非法组合

Usage:
    torchrun --standalone --nproc_per_node=2 labs/L7/parallel_failure_modes.py --part dist
    python labs/L7/parallel_failure_modes.py --part config
"""
from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


class TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.shared = nn.Linear(8, 8)
        self.branch_a = nn.Linear(8, 4)
        self.branch_b = nn.Linear(8, 4)

    def forward(self, x, use_a: bool):
        h = torch.relu(self.shared(x))
        return self.branch_a(h) if use_a else self.branch_b(h)


def case_divergent_branches(rank: int) -> None:
    """rank0 走分支 A、rank1 走分支 B：另一条分支的参数没有梯度可归约。"""
    model = torch.nn.parallel.DistributedDataParallel(TwoBranch())
    x = torch.randn(4, 8)
    err = None
    try:
        for step in range(2):                    # reducer 在第二次前向才发现
            model(x, use_a=(rank == 0)).sum().backward()
            model.zero_grad(set_to_none=False)
    except Exception as exc:                     # 保留真实报错的首行
        err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"
    if rank == 0:
        print(f"  A 两 rank 走不同分支 → {err or '没有报错'}")
        print("    DDP 假定每次迭代所有参数都参与；未参与的参数没有梯度可归约，")
        print("    reducer 在下一次前向发现上一次归约没完成。")
        print("    可选项：find_unused_parameters=True（每步多一次图遍历），")
        print("    或者让两 rank 执行同一组分支。")


class KD(nn.Module):
    """student 更新，teacher 只前向。"""

    def __init__(self, freeze_teacher: bool):
        super().__init__()
        torch.manual_seed(0)
        self.student = nn.Linear(8, 4)
        self.teacher = nn.Linear(8, 4)
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            t = self.teacher(x)
        return (self.student(x) - t).pow(2).mean()


def _two_steps(model, x, direct: bool):
    for _ in range(2):
        out = model.module(x) if direct else model(x)
        out.backward()
        if _ == 0:
            model.zero_grad(set_to_none=False)


def case_frozen_teacher(rank: int) -> None:
    x = torch.randn(4, 8) + rank        # 两 rank 的数据不同，同步与否可以区分
    results = []

    for freeze, direct, label in (
            (False, False, "teacher requires_grad=True，经 DDP 前向"),
            (True, False, "teacher requires_grad=False，经 DDP 前向"),
            (True, True, "绕过 DDP 直接调用 model.module(x)")):
        model = torch.nn.parallel.DistributedDataParallel(KD(freeze))
        err = None
        try:
            _two_steps(model, x, direct)
        except Exception as exc:
            err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}"
        # 用 all_gather 比较两 rank 的 student 梯度：不同即说明没有同步
        spread = None
        if err is None:
            g = model.module.student.weight.grad.detach().clone()
            buf = [torch.zeros_like(g) for _ in range(dist.get_world_size())]
            dist.all_gather(buf, g)
            spread = float((buf[0] - buf[1]).abs().max())
        results.append((label, err, spread))
        del model
        dist.barrier()

    if rank == 0:
        for label, err, spread in results:
            status = err or "正常完成"
            print(f"  B {label}")
            print(f"     结果：{status}")
            spread_txt = "未比较（这一轮已经报错）" if spread is None else (
                f"{spread:.3e}" + ("（0 表示确实做了归约）" if spread == 0 else
                                   "（非零说明两 rank 各算各的）"))
            print(f"     两 rank 的 student 梯度最大差：{spread_txt}")
        print("     teacher 在 no_grad 下前向，永远拿不到梯度；requires_grad 忘了关，")
        print("     它就进了 DDP 的归约集合，reducer 因此等不到这部分梯度。")
        print("     更隐蔽的是第三行：绕过 DDP 的 forward 不会报错，但梯度根本没有同步，")
        print("     两 rank 从此各训各的。")


def case_order_mismatch(rank: int) -> None:
    """两 rank 的 collective 次序不一致：gloo 在超时后报错，而不是静默出错。"""
    a = torch.ones(4) * (rank + 1)
    b = torch.zeros(4)
    err = None
    try:
        if rank == 0:
            dist.all_reduce(a)
            dist.broadcast(b, src=0)
        else:
            dist.broadcast(b, src=0)
            dist.all_reduce(a)
        dist.barrier()
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    if rank == 0:
        print(f"  C 两 rank 的 collective 次序相反 → {err or '本次没有报错'}")
        print("    同名 collective 按调用顺序配对，次序不一致时可能配错，也可能一直等下去。")
        print("    条件分支里的 collective 必须保证所有 rank 都执行、且顺序一致。")


def run_dist(args) -> None:
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=args.timeout))
    if rank == 0:
        head("两 rank 的真实失败（gloo，CPU，带超时）")
    case_divergent_branches(rank)
    dist.barrier()
    case_frozen_teacher(rank)
    dist.barrier()
    if args.with_order_mismatch:
        case_order_mismatch(rank)
    elif rank == 0:
        print("  C 次序不一致的用例默认关闭，加 --with-order-mismatch 才执行"
              f"（会等满 {args.timeout} 秒超时）")
    dist.destroy_process_group()


# ----------------------------------------------------------------- 配置合法性
def check_config(cfg: dict) -> list[str]:
    """检查一组并行配置的整除条件，返回违规说明。"""
    problems = []
    world = cfg["world"]
    tp, pp, cp, ep = cfg["tp"], cfg["pp"], cfg["cp"], cfg.get("ep", 1)
    dp_total = world / (tp * pp * cp)
    if world % (tp * pp * cp):
        problems.append(f"world={world} 不能被 tp×pp×cp={tp * pp * cp} 整除")
    if cfg["heads"] % tp:
        problems.append(f"注意力头数 {cfg['heads']} 不能被 tp={tp} 整除")
    if cfg["kv_heads"] % tp:
        problems.append(f"KV 头数 {cfg['kv_heads']} 不能被 tp={tp} 整除"
                        "（GQA 下这是最常见的越界）")
    if cfg["hidden"] % tp:
        problems.append(f"hidden={cfg['hidden']} 不能被 tp={tp} 整除")
    if cfg["layers"] % pp:
        problems.append(f"层数 {cfg['layers']} 不能被 pp={pp} 整除（否则各 stage 不均）")
    if cfg["seq"] % (cp * 2) and cp > 1:
        problems.append(f"序列长度 {cfg['seq']} 不能被 2×cp={2 * cp} 整除"
                        "（因果 mask 下常按两段分配以均衡负载）")
    if ep > 1:
        if cfg.get("experts", 0) % ep:
            problems.append(f"专家数 {cfg.get('experts')} 不能被 ep={ep} 整除")
        if ep > dp_total:
            problems.append(f"ep={ep} 超过可用的数据并行宽度 {dp_total:g}")
    if cfg["global_batch"] % (dp_total * cfg["micro_bs"] or 1):
        problems.append(f"global batch {cfg['global_batch']} 不能被 "
                        f"dp×micro_bs={dp_total:g}×{cfg['micro_bs']} 整除")
    return problems


def run_config() -> None:
    head("并行配置的整除条件")
    base = {"world": 16, "tp": 2, "pp": 2, "cp": 1, "ep": 1, "heads": 32,
            "kv_heads": 8, "hidden": 4096, "layers": 32, "seq": 4096,
            "experts": 0, "global_batch": 256, "micro_bs": 2}
    cases = [
        ("合法：tp2 × pp2 × dp4", base),
        ("tp=16：KV 头数不够分", {**base, "tp": 16}),
        ("pp=5：层数不能整除", {**base, "pp": 5, "world": 20}),
        ("cp=3：序列长度不能按两段整除", {**base, "cp": 3, "world": 24, "seq": 4096}),
        ("ep=8 但只有 4 路数据并行", {**base, "ep": 8, "experts": 64}),
        ("global batch 与 dp×micro_bs 不匹配", {**base, "global_batch": 250}),
    ]
    for name, cfg in cases:
        dp = cfg["world"] / (cfg["tp"] * cfg["pp"] * cfg["cp"])
        problems = check_config(cfg)
        tag = "通过" if not problems else "拒绝"
        print(f"\n[{tag}] {name}")
        print(f"  world={cfg['world']} tp={cfg['tp']} pp={cfg['pp']} cp={cfg['cp']} "
              f"ep={cfg.get('ep', 1)} → dp={dp:g}")
        for p in problems:
            print(f"    · {p}")

    head("一份合法二维配置的状态归属")
    cfg = base
    dp = cfg["world"] // (cfg["tp"] * cfg["pp"])
    params_b = 7.0                                # 以 7B 为例，单位 10^9
    print(f"  16 卡 = tp2 × pp2 × dp4，7B 参数、BF16 权重 + FP32 master 与 m/v")
    per_tp_pp = params_b / (cfg["tp"] * cfg["pp"])
    print(f"  每张卡承担 {per_tp_pp:.2f}B 参数（tp 切宽、pp 切深，两者相乘）")
    print(f"  数据并行只复制，不减少参数；FSDP/ZeRO 才会在 dp 维再切一次：")
    for name, bytes_per in (("BF16 参数", 2), ("FP32 master", 4), ("m+v", 8)):
        total = per_tp_pp * 1e9 * bytes_per / 2 ** 30
        print(f"    {name:<12} 每卡 {total:6.2f} GiB，若再按 dp={dp} 分片 "
              f"{total / dp:6.2f} GiB")
    print("  激活另算：它跟 micro_bs、seq、cp 与重算策略有关，不随参数分片下降。")
    print("  梯度归约组是 dp（与 cp 组合后是 dp×cp），TP 组内做的是激活的 all-reduce，")
    print("  两者的通信量、频率与所在阶段都不同，见 6.2 与 6.5。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dist", "config"], default="config")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--with-order-mismatch", action="store_true")
    args = ap.parse_args()
    if args.part == "dist":
        run_dist(args)
    else:
        run_config()


if __name__ == "__main__":
    main()
