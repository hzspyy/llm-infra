#!/usr/bin/env python3
"""训练状态清单：谁拥有它、什么时候写、缺了会坏哪一条保证。

四段：
  A 实际状态   从真实对象里取 state_dict，列出字段、dtype 与字节
  B 按训练类型 预训练 / SFT / KD / RL / flow 各自必须保存什么
  C 阶段产物   训练 checkpoint、adapter、merged、EMA、量化与导出格式的区别
  D 预算       保存频率、staging 峰值、带宽、恢复时间与故障损失的关系

Usage:
    python labs/L7/checkpoint_state_schema.py > "$RUN_DIR/state-schema.txt"
"""
from __future__ import annotations

import torch
import torch.nn as nn

MiB = 2 ** 20


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def nbytes(t) -> int:
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


# ------------------------------------------------------------------ A 实际状态
def section_a() -> None:
    head("A 一次真实训练里实际存在的状态")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    x = torch.randn(8, 256)
    loss = model(x).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    sched.step()

    rows = []
    params = sum(nbytes(v) for v in model.state_dict().values())
    rows.append(("model.state_dict()", "模型", f"{len(model.state_dict())} 个张量",
                 params, "每次保存", "load_state_dict", "没有它就没有模型"))
    opt_sd = opt.state_dict()
    opt_bytes = sum(nbytes(t) for st in opt_sd["state"].values() for t in st.values())
    rows.append(("optimizer.state_dict()['state']", "optimizer",
                 f"{len(opt_sd['state'])} 组 m/v/step", opt_bytes, "每次保存",
                 "load_state_dict", "m/v 归零，下一步的更新量与中断前不同"))
    rows.append(("optimizer.state_dict()['param_groups']", "optimizer",
                 f"{len(opt_sd['param_groups'])} 组超参", 0, "每次保存",
                 "load_state_dict", "LR、weight decay、参数分组丢失"))
    rows.append(("scheduler.state_dict()", "scheduler",
                 str(sched.state_dict()), 0, "每次保存", "load_state_dict",
                 "LR 回到起点，warmup 与衰减位置错位"))
    rows.append(("scaler.state_dict()", "AMP", str(scaler.state_dict()), 0,
                 "每次保存", "load_state_dict", "scale 与增长计数复位，可能立刻溢出或长期偏小"))
    rows.append(("EMA 缓冲区", "训练脚本", f"{len(ema)} 个张量",
                 sum(nbytes(v) for v in ema.values()), "每次保存",
                 "自定义", "推理用的那份权重丢失（F5-TTS 这类配方的最终产物）"))
    rows.append(("torch RNG / CUDA RNG", "全局",
                 f"{tuple(torch.get_rng_state().shape)} 的 ByteTensor",
                 nbytes(torch.get_rng_state()), "每次保存", "set_rng_state",
                 "dropout、数据增强、采样序列全部改变"))
    rows.append(("sampler / dataloader 游标", "数据层", "已提交样本号或 shard+offset",
                 0, "更新提交后", "自定义或 StatefulDataLoader",
                 "重复或跳过样本，epoch 边界错位"))
    rows.append(("global_step / 消费 token 数", "训练脚本", "标量",
                 0, "每次保存", "自定义", "日志、调度与预算全部对不上"))
    rows.append(("FP8 amax 历史 / 量化 scale", "低精度模块", "每张量一段定长 buffer",
                 0, "每次保存", "模块自身", "delayed scaling 的缩放因子从头估计"))
    rows.append(("adapter / teacher 的引用", "配方", "base revision、adapter 路径",
                 0, "每次保存", "配置", "合并到错的 base 上，静默产出错模型"))

    print(f"{'状态':<38}{'归属':<12}{'内容':<26}{'字节':>10}")
    for name, owner, content, size, *_ in rows:
        print(f"{name:<38}{owner:<12}{content[:26]:<26}{size:>10,}")
    print(f"\n{'状态':<38}{'保存时刻':<12}{'恢复入口':<26}缺失后果")
    for name, _, _, _, when, how, cost in rows:
        print(f"{name:<38}{when:<12}{how:<26}{cost}")
    print("\n注意 scheduler 与 scaler 的 state_dict 只有几个标量，字节可以忽略，"
          "但缺了它们恢复出来的就不是同一次训练。")
    print("字节多的项和重要的项不是同一批。")


# ------------------------------------------------------------------ B 按类型
def section_b() -> None:
    head("B 按训练类型分列：必须保存 / 可重建 / 外部不可变依赖")
    columns = ["预训练", "SFT", "KD", "RL", "flow/扩散"]
    rows = [
        ("模型参数", ["必须"] * 5),
        ("optimizer m/v", ["必须"] * 5),
        ("scheduler 位置", ["必须"] * 5),
        ("AMP scale", ["必须", "必须", "必须", "必须", "必须"]),
        ("RNG", ["必须", "必须", "必须", "必须", "必须（噪声与时间采样）"]),
        ("数据游标", ["必须", "必须", "必须", "按 rollout 定义", "必须"]),
        ("EMA 权重", ["—", "—", "—", "—", "必须（常是最终产物）"]),
        ("adapter 权重", ["—", "LoRA 时必须", "学生用 adapter 时必须", "必须", "LoRA 时必须"]),
        ("base / teacher 引用", ["—", "必须记 revision", "必须记 teacher revision",
                                 "必须记 reference 模型", "必须记 VAE/文本编码器"]),
        ("reward / critic 状态", ["—", "—", "—", "必须（各自的参数与 optimizer）", "—"]),
        ("rollout 队列与策略版本", ["—", "—", "—", "必须", "—"]),
        ("teacher 特征缓存", ["—", "—", "可重建（记住缓存键即可）", "—", "—"]),
        ("量化 / FP8 scale 状态", ["有则必须", "有则必须", "有则必须", "有则必须", "有则必须"]),
    ]
    print(f"{'状态':<24}" + "".join(f"{c:<16}" for c in columns))
    for name, cells in rows:
        print(f"{name:<24}" + "".join(f"{c:<16}" for c in cells))
    print("\n三类要分开：必须保存（丢了就不是同一次训练）、可重建（有键就能再算）、")
    print("外部不可变依赖（base 权重、teacher、数据版本——它们不进 checkpoint，")
    print("但 checkpoint 必须记住指向它们的标识）。")


# ------------------------------------------------------------------ C 阶段产物
def section_c() -> None:
    head("C 阶段产物：同样叫 checkpoint，用途完全不同")
    rows = [
        ("训练 checkpoint（DCP 目录）",
         "参数 + optimizer + scheduler + RNG + 游标，按 rank 分片",
         "严格恢复这次训练", "不能直接被推理框架加载"),
        ("权重导出（safetensors / HF 目录）",
         "只有参数与 config、tokenizer",
         "冷启动、评测、部署", "不能恢复训练轨迹"),
        ("adapter（PEFT 目录）",
         "低秩增量 + adapter_config（base 名与 revision、rank、target 模块）",
         "挂在匹配的 base 上服务或继续训练", "换 base 会静默产出错模型"),
        ("merged 权重",
         "base 与 adapter 合并后的完整参数",
         "当作普通模型部署", "失去单独更新 adapter 的能力；量化 base 合并需重新量化"),
        ("EMA 权重",
         "另一份参数，不含 optimizer",
         "推理与评测（扩散/TTS 常用）", "与训练权重不同，不能互相替代"),
        ("量化产物",
         "packed weight + scale/zero-point + 量化配置",
         "低精度部署", "反量化不回原始权重；QAT 的训练态与部署态是两份"),
        ("草稿模型导出",
         "draft 结构 + 词表映射 + 与 target 的对应关系",
         "投机解码部署", "与 target 不匹配就完全失效"),
    ]
    print(f"{'产物':<30}{'内容':<42}{'用途':<22}边界")
    for name, content, use, limit in rows:
        print(f"{name:<30}{content:<42}{use:<22}{limit}")
    print("\n一条实际存在的例子：SmolLM3 公开的 133 个分支是 Transformers 权重导出，")
    print("可以冷启动、可以评测，但没有 optimizer/RNG/游标，不能严格恢复那次训练。")


# ------------------------------------------------------------------ D 预算
def section_d() -> None:
    head("D 保存频率的预算：goodput 不是越频繁越高")
    params_b = 7.0
    bytes_per_param = 2 + 4 + 8            # BF16 权重 + FP32 master + m/v
    ckpt_bytes = params_b * 1e9 * bytes_per_param
    step_seconds = 3.0
    write_bw = [0.5e9, 2e9, 8e9]           # 共享盘 / 本地 NVMe / 并行文件系统
    mtbf_hours = 6.0

    print(f"  7B 模型、BF16 权重 + FP32 master + m/v = {bytes_per_param} 字节/参数")
    print(f"  单次 checkpoint {ckpt_bytes / 1e9:.1f} GB；每步 {step_seconds} 秒；"
          f"平均故障间隔 {mtbf_hours} 小时")
    print(f"\n{'写带宽':<14}{'写一次':>10}{'每 N 步保存':>12}{'保存开销':>10}"
          f"{'期望重算':>10}{'总浪费':>10}{'最优 N':>8}")
    for bw in write_bw:
        write_s = ckpt_bytes / bw
        best = None
        for n in (50, 100, 200, 500, 1000, 2000, 5000):
            save_overhead = write_s / (n * step_seconds)          # 同步保存的时间占比
            expected_lost = (n * step_seconds / 2) / (mtbf_hours * 3600)
            total = save_overhead + expected_lost
            if best is None or total < best[1]:
                best = (n, total, save_overhead, expected_lost)
        n, total, save_o, lost = best
        print(f"{bw / 1e9:>6.1f} GB/s  {write_s:>9.1f}s{n:>12}"
              f"{save_o * 100:>9.2f}%{lost * 100:>9.2f}%{total * 100:>9.2f}%{n:>8}")
    print("\n  同步保存的开销随 N 反比下降，故障时的期望重算随 N 线性上升，两者之和有最小值。")
    print("  异步 staging 把第一项压到接近零，代价是 staging 期间的额外显存/内存峰值：")
    staging = ckpt_bytes / 1e9
    print(f"    staging 峰值 ≈ 一份完整状态 {staging:.1f} GB（分片后按 rank 均摊），")
    print("    并且在 staging 完成之前不能改写参数——所以 optimizer.step 要等它。")
    print("\n  三类存储要分开记账：")
    print("    本地临时保存  最快，节点挂了就没了，只能救进程级故障")
    print("    共享盘持久化  能救节点故障，带宽通常是瓶颈")
    print("    异地容灾      能救机房故障，延迟高，通常只保留少量里程碑")
    print("  恢复时间 = 读 checkpoint + 重建进程组 + 预热，它与保存频率无关，")
    print("  但它决定了一次故障的固定损失，必须与期望重算量一起算。")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | CPU")
    section_a()
    section_b()
    section_c()
    section_d()
