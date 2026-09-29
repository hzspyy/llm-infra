#!/usr/bin/env python3
"""LoRA 与 QLoRA 的更新对象、状态与产物。

四段：
  A 低秩更新   W + s·BA 的参数量、初始增量、冻结验证与一次真实更新
  B 规模账     真实模型尺寸下 rank 8/16/32 的参数、optimizer 与激活
  C NF4        bitsandbytes 的真实 4bit 存储、absmax 分块、double quant 与反量化误差
  D 产物       PEFT 保存的字段、merge 前后的等价性与量化 base 的边界

需要 peft 与 bitsandbytes；NF4 一段需要 CUDA。

Usage（crater，envs/serve）:
    python labs/L7/lora_qlora_contract.py --outdir "$RUN_DIR/lora"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

IN, OUT = 32, 16


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ A 低秩更新
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)     # PEFT 的默认初始化
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def delta_w(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x):
        return self.base(x) + self.scaling * nn.functional.linear(
            nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B)


def section_a() -> dict:
    head("A W + s·BA：初始增量为零，base 不动")
    torch.manual_seed(0)
    rows = []
    for rank, alpha in ((2, 4), (4, 8)):
        base = nn.Linear(IN, OUT, bias=False)
        layer = LoRALinear(base, rank, alpha)
        w0 = base.weight.detach().clone()
        delta0 = layer.delta_w().detach()
        x = torch.randn(8, IN)
        y = torch.randn(8, OUT)
        opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.1)
        loss = (layer(x) - y).pow(2).mean()
        loss.backward()
        grads = {"A": float(layer.lora_A.grad.norm()), "B": float(layer.lora_B.grad.norm()),
                 "base": layer.base.weight.grad}
        opt.step()
        delta1 = layer.delta_w().detach()
        rows.append({
            "rank": rank, "alpha": alpha, "scaling": layer.scaling,
            "trainable": int(layer.lora_A.numel() + layer.lora_B.numel()),
            "base_params": int(base.weight.numel()),
            "delta0_absmax": float(delta0.abs().max()),
            "delta1_absmax": float(delta1.abs().max()),
            "base_changed": not torch.equal(w0, base.weight.detach()),
            "grad_A": grads["A"], "grad_B": grads["B"],
            "base_grad_is_none": grads["base"] is None,
        })
        print(f"  rank={rank} alpha={alpha} s={layer.scaling}：可训练 "
              f"{rows[-1]['trainable']} / base {rows[-1]['base_params']}"
              f"（{rows[-1]['trainable'] / rows[-1]['base_params']:.1%}）")
        print(f"    初始 |ΔW|max={delta0.abs().max():.3e}（B 全零，所以模型行为与 base 完全一致）")
        print(f"    一次更新后 |ΔW|max={delta1.abs().max():.3e}；"
              f"base.weight 是否被改动={rows[-1]['base_changed']}；"
              f"base.weight.grad={grads['base']}")
        print(f"    A 的梯度范数={grads['A']:.6f}，B 的梯度范数={grads['B']:.6f}")
    print("\n  第一步之前 B=0，所以 ∂L/∂A = sBᵀ(...) 也是零——A 在第一步不动，只有 B 动。")
    print("  这正是 B 零初始化的作用：加载 adapter 的瞬间模型输出不变。")
    print("  scaling = alpha/rank：改 rank 而不改 alpha 会同时改变增量的量级。")
    return {"cases": rows}


# ------------------------------------------------------------------ B 规模账
def section_b() -> dict:
    head("B 真实尺寸下的账：SmolLM3-3B 的形状")
    hidden, layers, heads, kv_heads = 2048, 36, 16, 4
    inter = 11008
    head_dim = hidden // heads
    shapes = {
        "q_proj": (heads * head_dim, hidden), "k_proj": (kv_heads * head_dim, hidden),
        "v_proj": (kv_heads * head_dim, hidden), "o_proj": (hidden, heads * head_dim),
        "gate_proj": (inter, hidden), "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }
    total = sum(o * i for o, i in shapes.values()) * layers
    print(f"  每层 7 个 Linear，共 {layers} 层，线性层参数合计 {total / 1e9:.3f}B")
    print(f"\n  {'target':<22}{'rank':>6}{'可训练参数':>14}{'占线性层':>10}"
          f"{'optimizer(FP32)':>18}")
    rows = []
    for targets, label in (((("q_proj", "v_proj")), "q_proj,v_proj"),
                           ((tuple(shapes)), "全部 7 个 Linear")):
        for rank in (8, 16, 32):
            n = sum((shapes[t][0] + shapes[t][1]) * rank for t in targets) * layers
            rows.append({"targets": label, "rank": rank, "trainable": n,
                         "fraction": n / total, "optimizer_MiB": n * 8 / 2 ** 20})
            print(f"  {label:<22}{rank:>6}{n:>14,}{n / total:>9.2%}"
                  f"{n * 8 / 2 ** 20:>15.1f} MiB")
    print("\n  同样是 rank=16，注入 7 个 Linear 的可训练参数是只注入 q/v 的 "
          f"{rows[4]['trainable'] / rows[1]['trainable']:.1f} 倍。")
    print("  optimizer 状态按可训练参数算（m 与 v 各 4 字节），这是 LoRA 省下的主要部分。")
    print("  激活不省：反向仍然要穿过整个 base，保存值与全参微调同量级（见 7.1 的实测）。")
    return {"rows": rows, "linear_params": total}


# ------------------------------------------------------------------ C NF4
def section_c(device: str) -> dict:
    head("C NF4 的真实存储：分块 absmax 与 double quant")
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError as exc:
        print(f"  bitsandbytes 不可用：{exc}")
        return {"available": False}
    if device != "cuda":
        print("  NF4 的打包与反量化需要 CUDA，跳过")
        return {"available": False}

    torch.manual_seed(0)
    dim = 1024
    ref = nn.Linear(dim, dim, bias=False).to(torch.bfloat16)
    rows = []
    for double_quant in (False, True):
        q = Linear4bit(dim, dim, bias=False, compute_dtype=torch.bfloat16,
                       quant_type="nf4", compress_statistics=double_quant)
        q.weight = bnb.nn.Params4bit(ref.weight.data.clone(), requires_grad=False,
                                     quant_type="nf4",
                                     compress_statistics=double_quant)
        q = q.to(device)
        state = q.weight.quant_state
        packed = q.weight.data
        extra = 0
        detail = []
        if state.absmax is not None:
            extra += state.absmax.numel() * state.absmax.element_size()
            detail.append(f"absmax {tuple(state.absmax.shape)}×"
                          f"{state.absmax.element_size()}B")
        if getattr(state, "state2", None) is not None:
            extra += state.state2.absmax.numel() * state.state2.absmax.element_size()
            detail.append(f"二级 absmax {tuple(state.state2.absmax.shape)}")
        if getattr(state, "offset", None) is not None:
            extra += state.offset.numel() * state.offset.element_size()
        total_bytes = packed.numel() * packed.element_size() + extra
        deq = bnb.functional.dequantize_4bit(packed, state).to(torch.bfloat16)
        err = (deq.float() - ref.weight.to(device).float()).abs()
        rows.append({"double_quant": double_quant,
                     "packed_dtype": str(packed.dtype), "packed_bytes": packed.numel(),
                     "extra_bytes": extra, "bits_per_param": total_bytes * 8 / dim / dim,
                     "blocksize": state.blocksize,
                     "max_abs_err": float(err.max()), "rel_err": float(
                         err.mean() / ref.weight.to(device).abs().mean())})
        print(f"  double_quant={double_quant}：打包权重 dtype={packed.dtype} "
              f"shape={tuple(packed.shape)}，blocksize={state.blocksize}")
        print(f"    额外统计量：{' + '.join(detail)} = {extra} 字节")
        print(f"    合计 {total_bytes * 8 / dim / dim:.3f} bit/参数；"
              f"反量化最大误差 {err.max():.4f}，平均相对误差 "
              f"{err.mean() / ref.weight.to(device).abs().mean():.3%}")
    print(f"\n  BF16 同样的权重是 16 bit/参数，NF4 双重量化后约 "
          f"{rows[1]['bits_per_param']:.2f} bit/参数。")
    print("  存储是 4bit，计算不是：forward 先按 blocksize 反量化回 compute_dtype 再做 GEMM。")

    x = torch.randn(4, dim, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        y_q = Linear4bit(dim, dim, bias=False, compute_dtype=torch.bfloat16,
                         quant_type="nf4").to(device)
    print("  QLoRA 的可训练参数仍然是 BF16/FP32 的 adapter，base 只是被压小了。")
    return {"available": True, "rows": rows}


# ------------------------------------------------------------------ D 产物
class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(IN, IN, bias=False)
        self.v_proj = nn.Linear(IN, IN, bias=False)
        self.out = nn.Linear(IN, OUT, bias=False)

    def forward(self, x):
        return self.out(self.v_proj(torch.relu(self.q_proj(x))))


def section_d(tmpdir: Path) -> dict:
    head("D PEFT 的产物：adapter_config 里到底写了什么")
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        print(f"  peft 不可用：{exc}")
        return {"available": False}

    torch.manual_seed(0)
    base = TinyBlock()
    base_w = {k: v.detach().clone() for k, v in base.state_dict().items()}
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(base, cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  可训练参数：{trainable}")
    x = torch.randn(8, IN)
    y = torch.randn(8, OUT)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    for _ in range(2):
        loss = (model(x) - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    adapter_dir = tmpdir / "adapter"
    model.save_pretrained(str(adapter_dir))
    files = sorted(p.name for p in adapter_dir.iterdir())
    conf = json.loads((adapter_dir / "adapter_config.json").read_text())
    print(f"  保存出来的文件：{files}")
    keys = ["base_model_name_or_path", "revision", "r", "lora_alpha", "lora_dropout",
            "target_modules", "modules_to_save", "bias", "init_lora_weights",
            "use_rslora", "use_dora", "peft_type", "peft_version"]
    print("  adapter_config.json 的关键字段：")
    for k in keys:
        print(f"    {k} = {conf.get(k)}")
    print(f"  配置里共有 {len(conf)} 个字段，base 身份只由 base_model_name_or_path 与")
    print("  revision 两项描述；本例包的是裸 nn.Module，所以两项都是 null。")
    print("  接 HF 模型时前者记的是 from_pretrained 的路径，revision 默认仍是 null——")
    print("  也就是说，除非显式填写，adapter 不知道自己是对着 base 的哪个版本训练的。")

    with torch.no_grad():
        out_lora = model(x).clone()
    merged = model.merge_and_unload()
    with torch.no_grad():
        out_merged = merged(x)
    diff = float((out_lora - out_merged).abs().max())
    delta_q = float((merged.q_proj.weight.detach() - base_w["q_proj.weight"]).abs().max())
    delta_out = float((merged.out.weight.detach() - base_w["out.weight"]).abs().max())
    print(f"\n  merge 前后输出最大差={diff:.3e}（合并只是把 s·BA 加进 W）")
    print(f"  被注入的 q_proj 权重变化={delta_q:.6f}；未注入的 out 权重变化={delta_out:.1e}")

    torch.manual_seed(1)
    other = TinyBlock()
    with torch.no_grad():
        other.q_proj.weight.add_(0.01)          # 一个"差不多"的 base
    reloaded = PeftModel.from_pretrained(other, str(adapter_dir))
    with torch.no_grad():
        out_wrong = reloaded(x)
    print(f"  把同一份 adapter 挂到另一个 base 上：输出与正确组合最大差="
          f"{float((out_wrong - out_lora).abs().max()):.4f}，没有任何报错")
    print("\n  三种产物的区别：adapter（增量，需要 base）、merged（完整权重，不可再拆）、")
    print("  量化 base 上的 merge（要先反量化再合并，通常还要重新量化，误差不可逆）。")
    return {"available": True, "files": files,
            "config_keys": {k: conf.get(k) for k in keys},
            "merge_diff": diff, "wrong_base_diff": float((out_wrong - out_lora).abs().max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir")
    ap.add_argument("--workdir", default=None, help="adapter 落盘目录（默认用 outdir）")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device={device}")

    tmp = Path(args.workdir or args.outdir or "./lora-tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    report = {"device": device}
    report["lora"] = section_a()
    report["scale"] = section_b()
    report["nf4"] = section_c(device)
    report["artifacts"] = section_d(tmp)

    print(f"\n{'=' * 78}\n五种方法的更新对象与产物\n{'=' * 78}")
    table = [
        ("全参 BF16", "全部参数", "参数+梯度+m/v 全量", "完整权重"),
        ("LoRA", "s·BA", "只有 adapter 有梯度与 m/v", "adapter 或 merged 权重"),
        ("QLoRA", "s·BA", "同上；base 以 NF4 常驻", "adapter（必须配同一份量化 base）"),
        ("QAT", "全部参数（带 fake quant）", "全量 + 量化 scale", "量化权重 + scale"),
        ("FP8/FP4 训练", "全部参数", "全量 + amax/scale 历史", "高精度权重（部署再量化）"),
    ]
    print(f"{'方法':<14}{'更新对象':<22}{'训练状态':<28}部署产物")
    for row in table:
        print(f"{row[0]:<14}{row[1]:<22}{row[2]:<28}{row[3]}")
    print("QLoRA 不等于训练出一个 4bit 模型：被更新的仍然是高精度的低秩增量。")

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "lora_qlora.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化结果写入 {out}/lora_qlora.json")


if __name__ == "__main__":
    main()
